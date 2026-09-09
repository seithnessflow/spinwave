//! Convolution reverb over a two-channel impulse response.
//!
//! Uniform partitioned convolution (overlap-save FFT, partition size
//! [`PARTITION_SIZE`], FFT size `2 * PARTITION_SIZE`). The folded bus signal
//! is convolved **dual-mono**: the left IR channel processes the left lanes
//! and the right IR channel the right lanes (L→L / R→R, no cross terms, so
//! this is not a "true stereo" four-channel convolution). A mono impulse
//! response must be passed as both `left` and `right`. Both voice slots
//! carry the same folded signal, so lanes 0/1 are the authoritative L/R
//! pair and the result is broadcast back to both voices.
//!
//! **Latency**: the wet path is delayed by exactly [`LATENCY_SAMPLES`]
//! (= one partition, 2048 samples) relative to the dry path — the classic
//! input-buffering latency of uniform partitioned convolution. The dry path
//! has zero latency.
//!
//! **Cost**: every partition step multiplies-accumulates `num_partitions`
//! spectra of `NUM_BINS` bins, i.e. the per-sample cost grows linearly with
//! the IR length (a 10 s IR at 48 kHz is ~235 partitions). Non-uniform
//! partitioning (small head partitions for low latency, large tail
//! partitions for throughput) is deliberately not implemented; the MAC loop
//! runs on separate real/imaginary slices so it auto-vectorises.
//!
//! All FFT plans, spectra, delay lines and scratch buffers are allocated in
//! [`ConvolutionEngine::build`] (called by
//! [`ConvolutionReverb::set_impulse_response`] and
//! [`ConvolutionReverb::set_sample_rate`]); `process` never allocates. A
//! host that builds engines on a worker thread can hand them over with
//! [`ConvolutionReverb::swap_engine`], which crossfades over one partition
//! and returns the displaced engine for off-thread disposal.
//!
//! Three synthetic built-in IR generators are provided ([`ir_plate`],
//! [`ir_hall`], [`ir_spring`]): deterministic, seeded, exponentially decaying
//! filtered noise shaped after each family — not measurements of real spaces.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use spinwave_poly::{math, PolyF32};

use super::lanes::first_voice_mask;

/// Samples per partition (hop size of the frequency-domain delay line).
pub const PARTITION_SIZE: usize = 2048;
/// FFT length used for the overlap-save segments.
pub const FFT_SIZE: usize = 2 * PARTITION_SIZE;
const NUM_BINS: usize = FFT_SIZE / 2 + 1;
/// Wet-path latency in samples relative to the dry path.
pub const LATENCY_SAMPLES: usize = PARTITION_SIZE;
/// Impulse responses are truncated to this many seconds at the engine rate.
pub const MAX_IR_SECONDS: f32 = 10.0;
/// Upper bound for [`ConvolutionParams::predelay_seconds`].
pub const MAX_PREDELAY_SECONDS: f32 = 1.0;
/// Time constant of the pre-delay smoother (seconds): a pre-delay change
/// glides there instead of jumping per block.
pub const PREDELAY_SMOOTH_SECONDS: f32 = 0.05;

/// Block-rate convolution reverb parameters.
#[derive(Clone, Copy, Debug)]
pub struct ConvolutionParams {
    /// Dry/wet in [0, 1]; applied as an equal-power crossfade
    /// (0 = fully dry, 1 = fully wet).
    pub dry_wet: f32,
    /// Extra delay on the wet path in seconds, clamped to
    /// [0, [`MAX_PREDELAY_SECONDS`]]. Adds on top of [`LATENCY_SAMPLES`].
    /// Changes are smoothed with a [`PREDELAY_SMOOTH_SECONDS`] time
    /// constant and read with linear interpolation.
    pub predelay_seconds: f32,
    /// Gain applied to the wet path in dB, clamped to [-60, 24].
    pub ir_gain_db: f32,
}

impl Default for ConvolutionParams {
    fn default() -> ConvolutionParams {
        ConvolutionParams {
            dry_wet: 0.5,
            predelay_seconds: 0.0,
            ir_gain_db: 0.0,
        }
    }
}

/// Reasons an impulse response cannot be turned into an engine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IrError {
    /// `sample_rate_of_ir` or `engine_sample_rate` was 0.
    ZeroSampleRate,
    /// A channel contained NaN or infinite samples.
    NonFinite,
    /// The IR is empty or silent after resampling/truncation.
    Silent,
}

impl core::fmt::Display for IrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IrError::ZeroSampleRate => f.write_str("impulse response sample rate is zero"),
            IrError::NonFinite => f.write_str("impulse response contains non-finite samples"),
            IrError::Silent => f.write_str("impulse response is empty or silent"),
        }
    }
}

impl std::error::Error for IrError {}

/// Preallocated per-IR convolution state. Build it on any thread with
/// [`ConvolutionEngine::build`], then install it with
/// [`ConvolutionReverb::swap_engine`]; `tick` never allocates.
pub struct ConvolutionEngine {
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    num_partitions: usize,
    /// Per channel: one `NUM_BINS` spectrum per IR partition, split into
    /// real and imaginary planes so the MAC loop vectorises.
    ir_re: [Vec<Vec<f32>>; 2],
    ir_im: [Vec<Vec<f32>>; 2],
    /// Per channel: frequency-domain delay line of past input spectra.
    fdl_re: [Vec<Vec<f32>>; 2],
    fdl_im: [Vec<Vec<f32>>; 2],
    /// Index of the most recent entry in the delay line.
    fdl_pos: usize,
    /// Per channel: rolling `FFT_SIZE` time-domain input (overlap-save).
    history: [Vec<f32>; 2],
    /// Fill position of the current partition, in [0, PARTITION_SIZE).
    in_pos: usize,
    /// Per channel: wet output of the last computed partition.
    out_fifo: [Vec<f32>; 2],
    time_buf: Vec<f32>,
    spec_buf: Vec<Complex<f32>>,
    acc_re: Vec<f32>,
    acc_im: Vec<f32>,
    acc: Vec<Complex<f32>>,
    scratch_fwd: Vec<Complex<f32>>,
    scratch_inv: Vec<Complex<f32>>,
    sample_rate: f32,
}

impl ConvolutionEngine {
    /// Builds the convolution state for a two-channel IR (pass a mono IR
    /// twice). The IR is linearly resampled if `sample_rate_of_ir` differs
    /// from `engine_sample_rate`, truncated to [`MAX_IR_SECONDS`] at the
    /// engine rate, and energy-normalized so a unit-impulse IR is
    /// transparent-ish (unity RMS energy per channel pair). Cold path: all
    /// allocation happens here.
    pub fn build(
        left: &[f32],
        right: &[f32],
        sample_rate_of_ir: u32,
        engine_sample_rate: u32,
    ) -> Result<ConvolutionEngine, IrError> {
        if sample_rate_of_ir == 0 || engine_sample_rate == 0 {
            return Err(IrError::ZeroSampleRate);
        }
        if !left.iter().chain(right).all(|s| s.is_finite()) {
            return Err(IrError::NonFinite);
        }

        let max_len = (MAX_IR_SECONDS * engine_sample_rate as f32) as usize;
        let mut left = resample_linear(left, sample_rate_of_ir, engine_sample_rate);
        let mut right = resample_linear(right, sample_rate_of_ir, engine_sample_rate);
        left.truncate(max_len);
        right.truncate(max_len);
        let ir_len = left.len().max(right.len());
        left.resize(ir_len, 0.0);
        right.resize(ir_len, 0.0);

        let energy: f64 = left
            .iter()
            .zip(&right)
            .map(|(&l, &r)| (l as f64) * (l as f64) + (r as f64) * (r as f64))
            .sum::<f64>()
            / 2.0;
        if ir_len == 0 || energy <= 1e-24 {
            return Err(IrError::Silent);
        }
        // A unit impulse in both channels has energy 1 -> normalization 1.
        let normalization = ((1.0 / energy.sqrt()) as f32).min(1.0e4);

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FFT_SIZE);
        let c2r = planner.plan_fft_inverse(FFT_SIZE);

        let num_partitions = ir_len.div_ceil(PARTITION_SIZE).max(1);
        let mut time_buf = vec![0.0f32; FFT_SIZE];
        let mut scratch_fwd = vec![Complex::new(0.0, 0.0); r2c.get_scratch_len()];
        let scratch_inv = vec![Complex::new(0.0, 0.0); c2r.get_scratch_len()];
        let mut spectrum = vec![Complex::new(0.0, 0.0); NUM_BINS];

        let mut ir_re: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
        let mut ir_im: [Vec<Vec<f32>>; 2] = [Vec::new(), Vec::new()];
        for (channel, samples) in [&left, &right].into_iter().enumerate() {
            for partition in 0..num_partitions {
                let start = partition * PARTITION_SIZE;
                let end = (start + PARTITION_SIZE).min(ir_len);
                time_buf.fill(0.0);
                for (slot, &value) in time_buf.iter_mut().zip(&samples[start..end]) {
                    *slot = value * normalization;
                }
                let _ = r2c.process_with_scratch(&mut time_buf, &mut spectrum, &mut scratch_fwd);
                ir_re[channel].push(spectrum.iter().map(|c| c.re).collect());
                ir_im[channel].push(spectrum.iter().map(|c| c.im).collect());
            }
        }

        let planes = || (0..num_partitions).map(|_| vec![0.0f32; NUM_BINS]).collect::<Vec<_>>();

        Ok(ConvolutionEngine {
            r2c,
            c2r,
            num_partitions,
            ir_re,
            ir_im,
            fdl_re: [planes(), planes()],
            fdl_im: [planes(), planes()],
            fdl_pos: 0,
            history: [vec![0.0; FFT_SIZE], vec![0.0; FFT_SIZE]],
            in_pos: 0,
            out_fifo: [vec![0.0; PARTITION_SIZE], vec![0.0; PARTITION_SIZE]],
            time_buf,
            spec_buf: spectrum,
            acc_re: vec![0.0; NUM_BINS],
            acc_im: vec![0.0; NUM_BINS],
            acc: vec![Complex::new(0.0, 0.0); NUM_BINS],
            scratch_fwd,
            scratch_inv,
            sample_rate: engine_sample_rate as f32,
        })
    }

    /// Engine sample rate the IR was prepared for.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Number of IR partitions (cost proxy).
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Feeds one stereo input sample and returns the wet output for this
    /// position (one partition behind). Allocation-free.
    #[inline]
    fn tick(&mut self, left: f32, right: f32) -> (f32, f32) {
        let wet = (self.out_fifo[0][self.in_pos], self.out_fifo[1][self.in_pos]);
        self.history[0][PARTITION_SIZE + self.in_pos] = left;
        self.history[1][PARTITION_SIZE + self.in_pos] = right;
        self.in_pos += 1;
        if self.in_pos == PARTITION_SIZE {
            self.step();
            self.in_pos = 0;
        }
        wet
    }

    /// Runs one partition step: FFT the newest `PARTITION_SIZE` input samples
    /// (with the previous partition as overlap), multiply-accumulate against
    /// every IR partition and inverse-FFT the wet block into `out_fifo`.
    fn step(&mut self) {
        self.fdl_pos = (self.fdl_pos + self.num_partitions - 1) % self.num_partitions;
        let scale = 1.0 / FFT_SIZE as f32;

        for channel in 0..2 {
            self.time_buf.copy_from_slice(&self.history[channel]);
            let _ = self.r2c.process_with_scratch(
                &mut self.time_buf,
                &mut self.spec_buf,
                &mut self.scratch_fwd,
            );
            for ((re, im), value) in self.fdl_re[channel][self.fdl_pos]
                .iter_mut()
                .zip(self.fdl_im[channel][self.fdl_pos].iter_mut())
                .zip(&self.spec_buf)
            {
                *re = value.re;
                *im = value.im;
            }

            self.acc_re.fill(0.0);
            self.acc_im.fill(0.0);
            for partition in 0..self.num_partitions {
                let slot = (self.fdl_pos + partition) % self.num_partitions;
                mac_complex_planes(
                    &mut self.acc_re,
                    &mut self.acc_im,
                    &self.fdl_re[channel][slot],
                    &self.fdl_im[channel][slot],
                    &self.ir_re[channel][partition],
                    &self.ir_im[channel][partition],
                );
            }
            for ((acc, &re), &im) in self.acc.iter_mut().zip(&self.acc_re).zip(&self.acc_im) {
                *acc = Complex::new(re, im);
            }
            // Spectra of real signals keep these bins real; enforce it so the
            // inverse transform never sees rounding noise there.
            self.acc[0].im = 0.0;
            self.acc[NUM_BINS - 1].im = 0.0;

            let _ = self.c2r.process_with_scratch(
                &mut self.acc,
                &mut self.time_buf,
                &mut self.scratch_inv,
            );
            for (out, &value) in self.out_fifo[channel]
                .iter_mut()
                .zip(&self.time_buf[PARTITION_SIZE..])
            {
                *out = value * scale;
            }

            // The block just consumed becomes the overlap of the next one.
            self.history[channel].copy_within(PARTITION_SIZE.., 0);
        }
    }

    fn clear_runtime_state(&mut self) {
        for channel in 0..2 {
            for plane in self.fdl_re[channel].iter_mut().chain(self.fdl_im[channel].iter_mut()) {
                plane.fill(0.0);
            }
            self.history[channel].fill(0.0);
            self.out_fifo[channel].fill(0.0);
        }
        self.fdl_pos = 0;
        self.in_pos = 0;
    }
}

/// `acc += x * h` on split complex planes (slices of equal length); the
/// straight-line loop over four input slices auto-vectorises.
#[inline]
fn mac_complex_planes(
    acc_re: &mut [f32],
    acc_im: &mut [f32],
    x_re: &[f32],
    x_im: &[f32],
    h_re: &[f32],
    h_im: &[f32],
) {
    let n = acc_re.len();
    let (acc_re, acc_im) = (&mut acc_re[..n], &mut acc_im[..n]);
    let (x_re, x_im, h_re, h_im) = (&x_re[..n], &x_im[..n], &h_re[..n], &h_im[..n]);
    for i in 0..n {
        acc_re[i] += x_re[i] * h_re[i] - x_im[i] * h_im[i];
        acc_im[i] += x_re[i] * h_im[i] + x_im[i] * h_re[i];
    }
}

/// An engine change in flight: the old engine keeps running until the new
/// one has produced its first partition, then both crossfade over one
/// partition.
struct EngineSwap {
    old: ConvolutionEngine,
    /// Samples until the fade starts (the new engine's latency).
    countdown: usize,
    /// Samples into the fade.
    fade_pos: usize,
}

/// Uniform partitioned convolution reverb; see the module docs for the
/// partition scheme, channel layout and latency.
pub struct ConvolutionReverb {
    engine: Option<ConvolutionEngine>,
    swap: Option<EngineSwap>,
    retired: Option<ConvolutionEngine>,
    /// Raw IR as last given (engine-rate independent) so a sample-rate
    /// change can rebuild the engine without asking the host again.
    raw_ir: Option<(Vec<f32>, Vec<f32>, u32)>,
    sample_rate: f32,
    /// Per channel: pre-delay ring for the wet path (sized for
    /// [`MAX_PREDELAY_SECONDS`] at the engine rate).
    predelay: [Vec<f32>; 2],
    predelay_pos: usize,
    /// Smoothed pre-delay in samples (fractional).
    current_predelay: f32,
    dry: PolyF32,
    wet: PolyF32,
    gain: f32,
}

impl Default for ConvolutionReverb {
    fn default() -> ConvolutionReverb {
        ConvolutionReverb::new()
    }
}

impl ConvolutionReverb {
    /// Creates a reverb with no impulse response loaded: the wet path is
    /// silent until an engine is installed.
    pub fn new() -> ConvolutionReverb {
        ConvolutionReverb {
            engine: None,
            swap: None,
            retired: None,
            raw_ir: None,
            sample_rate: 0.0,
            predelay: [Vec::new(), Vec::new()],
            predelay_pos: 0,
            current_predelay: 0.0,
            dry: PolyF32::ZERO,
            wet: PolyF32::ZERO,
            gain: 1.0,
        }
    }

    /// Wet-path latency in samples (zero until an IR is loaded).
    pub fn latency_samples(&self) -> usize {
        if self.engine.is_some() {
            LATENCY_SAMPLES
        } else {
            0
        }
    }

    /// Whether an engine (IR) is installed.
    pub fn has_engine(&self) -> bool {
        self.engine.is_some()
    }

    /// Loads a two-channel impulse response (pass a mono IR twice), builds
    /// the convolution state and installs it — with a one-partition
    /// crossfade if another IR was already playing. The raw IR is retained
    /// so [`ConvolutionReverb::set_sample_rate`] can rebuild it later. On
    /// error the current engine is unloaded. Cold path: allocates.
    pub fn set_impulse_response(
        &mut self,
        left: &[f32],
        right: &[f32],
        sample_rate_of_ir: u32,
        engine_sample_rate: u32,
    ) -> Result<(), IrError> {
        self.ensure_predelay(engine_sample_rate);
        match ConvolutionEngine::build(left, right, sample_rate_of_ir, engine_sample_rate) {
            Ok(engine) => {
                self.raw_ir = Some((left.to_vec(), right.to_vec(), sample_rate_of_ir));
                let _ = self.swap_engine(Some(engine));
                self.retired = None;
                Ok(())
            }
            Err(error) => {
                self.raw_ir = None;
                let _ = self.swap_engine(None);
                self.retired = None;
                Err(error)
            }
        }
    }

    /// Re-resamples the retained IR for a new engine rate and installs the
    /// rebuilt engine (crossfaded). Without a retained IR it only resizes
    /// the pre-delay line. Cold path: allocates.
    pub fn set_sample_rate(&mut self, engine_sample_rate: u32) {
        self.ensure_predelay(engine_sample_rate);
        if let Some((left, right, ir_rate)) = self.raw_ir.clone() {
            let engine = ConvolutionEngine::build(&left, &right, ir_rate, engine_sample_rate).ok();
            let _ = self.swap_engine(engine);
            self.retired = None;
        }
    }

    /// Installs `new_engine` (or unloads with `None`), real-time safe: no
    /// allocation, no deallocation. When an engine is already playing, both
    /// run in parallel until the new one has produced its first partition,
    /// then they crossfade linearly over one partition. Returns the engine
    /// displaced by this call, if any: an engine that was still fading out
    /// from a previous swap (its fade is cut short). The engine that
    /// finishes fading is parked in the reverb; collect it with
    /// [`ConvolutionReverb::take_retired`]. If a fade completes while a
    /// retired engine is still parked, the older one is dropped in place
    /// (that deallocation is the only non-RT-safe path, and only happens
    /// when the caller never collects).
    pub fn swap_engine(
        &mut self,
        new_engine: Option<ConvolutionEngine>,
    ) -> Option<ConvolutionEngine> {
        let interrupted = self.swap.take().map(|swap| swap.old);
        let old = core::mem::replace(&mut self.engine, new_engine);
        match (old, self.engine.is_some()) {
            (Some(old), true) => {
                self.swap = Some(EngineSwap { old, countdown: PARTITION_SIZE, fade_pos: 0 });
            }
            (Some(old), false) => {
                // Nothing to fade into: retire the old engine right away.
                self.retired = Some(old);
            }
            (None, _) => {}
        }
        interrupted
    }

    /// Takes the engine that finished fading out after a swap, so the
    /// caller can drop it off the audio thread.
    pub fn take_retired(&mut self) -> Option<ConvolutionEngine> {
        self.retired.take()
    }

    fn ensure_predelay(&mut self, engine_sample_rate: u32) {
        let sample_rate = engine_sample_rate as f32;
        let predelay_len = ((MAX_PREDELAY_SECONDS * sample_rate) as usize).max(1) + 2;
        if self.predelay[0].len() != predelay_len || self.sample_rate != sample_rate {
            self.predelay = [vec![0.0; predelay_len], vec![0.0; predelay_len]];
            self.predelay_pos = 0;
            self.current_predelay = self.current_predelay.min((predelay_len - 2) as f32);
            self.sample_rate = sample_rate;
        }
    }

    /// Clears every delay line, spectrum history and mix ramp; keeps the
    /// loaded impulse response. An in-flight crossfade is finished
    /// immediately (the old engine is parked as retired).
    pub fn hard_reset(&mut self) {
        self.dry = PolyF32::ZERO;
        self.wet = PolyF32::ZERO;
        self.gain = 1.0;
        if let Some(engine) = &mut self.engine {
            engine.clear_runtime_state();
        }
        if let Some(swap) = self.swap.take() {
            self.retired = Some(swap.old);
        }
        for channel in &mut self.predelay {
            channel.fill(0.0);
        }
        self.predelay_pos = 0;
    }

    /// Processes one block (any length; the effect is designed for blocks of
    /// at most 1024 samples). Allocation-free.
    pub fn process(
        &mut self,
        params: &ConvolutionParams,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        let num_samples = audio_in.len();
        assert_eq!(audio_out.len(), num_samples);
        if num_samples == 0 {
            return;
        }
        let tick_increment = 1.0 / num_samples as f32;

        let mix = PolyF32::splat(params.dry_wet.clamp(0.0, 1.0));
        let mut current_wet = self.wet;
        let mut current_dry = self.dry;
        self.wet = math::equal_power_fade(mix);
        self.dry = math::equal_power_fade_inverse(mix);
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;

        let mut current_gain = self.gain;
        self.gain = 10.0f32.powf(params.ir_gain_db.clamp(-60.0, 24.0) / 20.0);
        let delta_gain = (self.gain - current_gain) * tick_increment;

        let Some(engine) = &mut self.engine else {
            // No IR: the wet path is silent, the dry path still ramps.
            for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
                let mut input = sample & first_voice_mask();
                input += input.swap_voices();
                current_dry += delta_dry;
                *out = current_dry * input;
            }
            return;
        };

        // Pre-delay: one-pole glide toward the target across the block,
        // applied linearly per sample and read with linear interpolation.
        let predelay_len = self.predelay[0].len();
        let max_predelay = (predelay_len - 2) as f32;
        let target_predelay = (params.predelay_seconds.clamp(0.0, MAX_PREDELAY_SECONDS)
            * self.sample_rate)
            .min(max_predelay);
        let smoothing = 1.0
            - (-(num_samples as f32) / (PREDELAY_SMOOTH_SECONDS * self.sample_rate.max(1.0)))
                .exp();
        let end_predelay = self.current_predelay
            + (target_predelay - self.current_predelay) * smoothing;
        let end_predelay = if (end_predelay - target_predelay).abs() < 1e-3 {
            target_predelay
        } else {
            end_predelay
        };
        let mut current_predelay = self.current_predelay;
        let delta_predelay = (end_predelay - current_predelay) * tick_increment;
        self.current_predelay = end_predelay;

        let mut swap_finished = false;
        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            let mut input = sample & first_voice_mask();
            input += input.swap_voices();
            let (left, right) = (input.lane(0), input.lane(1));

            let (mut wet_left, mut wet_right) = engine.tick(left, right);
            if let Some(swap) = &mut self.swap {
                let (old_left, old_right) = swap.old.tick(left, right);
                if swap.countdown > 0 {
                    swap.countdown -= 1;
                    wet_left = old_left;
                    wet_right = old_right;
                } else {
                    let t = swap.fade_pos as f32 / PARTITION_SIZE as f32;
                    wet_left = old_left + (wet_left - old_left) * t;
                    wet_right = old_right + (wet_right - old_right) * t;
                    swap.fade_pos += 1;
                    if swap.fade_pos >= PARTITION_SIZE {
                        swap_finished = true;
                    }
                }
            }

            self.predelay[0][self.predelay_pos] = wet_left;
            self.predelay[1][self.predelay_pos] = wet_right;
            current_predelay += delta_predelay;
            let whole = current_predelay.floor();
            let fraction = current_predelay - whole;
            let back = whole as usize;
            let read_a = (self.predelay_pos + predelay_len - back) % predelay_len;
            let read_b = (read_a + predelay_len - 1) % predelay_len;
            let delayed_left = self.predelay[0][read_a]
                + (self.predelay[0][read_b] - self.predelay[0][read_a]) * fraction;
            let delayed_right = self.predelay[1][read_a]
                + (self.predelay[1][read_b] - self.predelay[1][read_a]) * fraction;
            let delayed = PolyF32::stereo(delayed_left, delayed_right);
            self.predelay_pos = (self.predelay_pos + 1) % predelay_len;

            current_dry += delta_dry;
            current_wet += delta_wet;
            current_gain += delta_gain;
            *out = current_dry * input + current_wet * (delayed * current_gain);
        }

        if swap_finished {
            if let Some(swap) = self.swap.take() {
                self.retired = Some(swap.old);
            }
        }
    }
}

/// Linear resampling; a zero source rate yields an empty buffer instead of
/// an absurd allocation (callers validate rates before getting here).
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    if from_rate == 0 || to_rate == 0 {
        return Vec::new();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = ((input.len() as f64 / ratio).ceil() as usize).max(1);
    (0..out_len)
        .map(|n| {
            let position = n as f64 * ratio;
            let index = position as usize;
            let fraction = (position - index as f64) as f32;
            let a = input.get(index).copied().unwrap_or(0.0);
            let b = input.get(index + 1).copied().unwrap_or(0.0);
            a + (b - a) * fraction
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Built-in synthetic impulse responses
// ---------------------------------------------------------------------------

/// Deterministic xorshift32 noise source (stable across platforms).
struct NoiseSource(u32);

impl NoiseSource {
    fn new(seed: u32) -> NoiseSource {
        NoiseSource(seed.max(1))
    }

    /// Uniform noise in [-1, 1].
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn ir_length(seconds: f32, sample_rate: u32) -> (f32, usize) {
    let seconds = seconds.clamp(0.05, MAX_IR_SECONDS);
    let len = ((seconds * sample_rate as f32) as usize).max(8);
    (seconds, len)
}

/// T60-style exponential envelope: reaches -60 dB at `seconds`.
fn decay_envelope(t: f32, seconds: f32) -> f32 {
    (-6.907_755 * t / seconds).exp()
}

/// Synthetic plate impulse response: dense, bright, immediate onset.
///
/// Exponentially decaying white noise with a mild first-difference tilt for
/// the metallic sheen, decorrelated per channel. Deterministic; `seconds` is
/// the T60 and is clamped to [0.05, [`MAX_IR_SECONDS`]].
pub fn ir_plate(seconds: f32, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let (seconds, len) = ir_length(seconds, sample_rate);
    let mut channels = [Vec::with_capacity(len), Vec::with_capacity(len)];
    for (channel, seed) in channels.iter_mut().zip([0x1234_5678u32, 0x8badf00d]) {
        let mut noise = NoiseSource::new(seed);
        let mut previous = 0.0f32;
        for n in 0..len {
            let t = n as f32 / sample_rate as f32;
            let white = noise.next();
            // Brighten: mix in the first difference (gentle high tilt).
            let sample = 0.6 * white + 0.4 * (white - previous);
            previous = white;
            channel.push(sample * decay_envelope(t, seconds));
        }
    }
    let [left, right] = channels;
    (left, right)
}

/// Synthetic hall impulse response: slow onset, tail darkening over time.
///
/// Exponentially decaying noise with a ~20 ms build-up and a one-pole
/// low-pass whose cutoff falls along the tail. Deterministic; `seconds` is
/// the T60 and is clamped to [0.05, [`MAX_IR_SECONDS`]].
pub fn ir_hall(seconds: f32, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let (seconds, len) = ir_length(seconds, sample_rate);
    let mut channels = [Vec::with_capacity(len), Vec::with_capacity(len)];
    for (channel, seed) in channels.iter_mut().zip([0xdead_beefu32, 0xcafe_babe]) {
        let mut noise = NoiseSource::new(seed);
        let mut low_pass = 0.0f32;
        for n in 0..len {
            let t = n as f32 / sample_rate as f32;
            let onset = (t / 0.02).min(1.0);
            let progress = (t / seconds).min(1.0);
            // Darken the tail: smoothing coefficient falls from 0.6 to 0.06.
            let coefficient = 0.6 - 0.54 * progress;
            low_pass += coefficient * (noise.next() - low_pass);
            channel.push(low_pass * onset * decay_envelope(t, seconds) * 1.6);
        }
    }
    let [left, right] = channels;
    (left, right)
}

/// Synthetic spring impulse response: decaying train of dispersive chirps.
///
/// Repeating downward chirps (the classic "boing") over a low-level noise
/// bed, with slightly different echo periods per channel for stereo width.
/// Deterministic; `seconds` is the T60 and is clamped to
/// [0.05, [`MAX_IR_SECONDS`]].
pub fn ir_spring(seconds: f32, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let (seconds, len) = ir_length(seconds, sample_rate);
    let sr = sample_rate as f32;
    let chirp_len = ((0.04 * sr) as usize).max(4);
    let mut channels = [vec![0.0f32; len], vec![0.0f32; len]];
    for (channel, (period, seed)) in channels
        .iter_mut()
        .zip([(0.056f32, 0x0bad_cafeu32), (0.061, 0x5eed_5eed)])
    {
        // Chirp train.
        let mut echo = 0usize;
        loop {
            let start = (echo as f32 * period * sr) as usize;
            if start >= len {
                break;
            }
            let amplitude = decay_envelope(echo as f32 * period, seconds);
            let mut phase = 0.0f32;
            for (offset, slot) in channel[start..(start + chirp_len).min(len)]
                .iter_mut()
                .enumerate()
            {
                let tau = offset as f32 / sr;
                // Downward chirp from ~3.2 kHz to ~400 Hz.
                let frequency = 400.0 + 2800.0 * (-tau * 90.0).exp();
                phase += frequency / sr;
                let pulse_envelope = (-tau * 110.0).exp();
                *slot += (phase * 2.0 * core::f32::consts::PI).sin()
                    * pulse_envelope
                    * amplitude
                    * 0.8;
            }
            echo += 1;
        }
        // Low-level decaying noise bed under the chirps.
        let mut noise = NoiseSource::new(seed);
        let mut band = 0.0f32;
        for (n, slot) in channel.iter_mut().enumerate() {
            let t = n as f32 / sr;
            band += 0.25 * (noise.next() - band);
            *slot += band * decay_envelope(t, seconds) * 0.15;
        }
    }
    let [left, right] = channels;
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: u32 = 48000;
    const BLOCK: usize = 512;

    fn dirac_reverb() -> ConvolutionReverb {
        let mut reverb = ConvolutionReverb::new();
        reverb
            .set_impulse_response(&[1.0], &[1.0], SAMPLE_RATE, SAMPLE_RATE)
            .unwrap();
        reverb
    }

    /// Runs the reverb over deterministic input with blocks of `block`
    /// samples, returning (input, output) lane-0 sample streams.
    fn run_blocks(
        reverb: &mut ConvolutionReverb,
        params: &ConvolutionParams,
        total: usize,
        block: usize,
        signal: impl Fn(usize) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut input_samples = Vec::with_capacity(total);
        let mut output_samples = Vec::with_capacity(total);
        let mut output = vec![PolyF32::ZERO; block];
        let mut n = 0;
        while n < total {
            let count = block.min(total - n);
            let input: Vec<PolyF32> =
                (0..count).map(|i| PolyF32::splat(signal(n + i))).collect();
            reverb.process(params, &input, &mut output[..count]);
            for i in 0..count {
                assert!(output[i].is_finite());
                input_samples.push(input[i].lane(0));
                output_samples.push(output[i].lane(0));
            }
            n += count;
        }
        (input_samples, output_samples)
    }

    fn run(
        reverb: &mut ConvolutionReverb,
        params: &ConvolutionParams,
        total: usize,
        signal: impl Fn(usize) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        run_blocks(reverb, params, total, BLOCK, signal)
    }

    fn impulse(n: usize) -> f32 {
        if n == 0 {
            1.0
        } else {
            0.0
        }
    }

    #[test]
    fn unit_impulse_ir_is_delayed_passthrough() {
        let mut reverb = dirac_reverb();
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        let mut noise = NoiseSource::new(42);
        let signal: Vec<f32> = (0..3 * PARTITION_SIZE).map(|_| noise.next() * 0.5).collect();
        let (input, output) = run(&mut reverb, &params, signal.len(), |n| signal[n]);

        // Before the latency point the wet path is still empty.
        for &sample in &output[..LATENCY_SAMPLES] {
            assert!(sample.abs() < 1e-4, "early output not silent: {sample}");
        }
        // After it, the output is the input delayed by exactly one partition.
        for n in 0..input.len() - LATENCY_SAMPLES {
            let expected = input[n];
            let actual = output[n + LATENCY_SAMPLES];
            assert!(
                (expected - actual).abs() < 1e-3,
                "sample {n}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn partitioned_matches_direct_convolution() {
        // IR longer than one partition, block size that does not divide the
        // partition, deterministic noise input: the partitioned output must
        // equal direct convolution (up to float rounding) after the latency.
        let ir_len = 2 * PARTITION_SIZE + 777;
        let mut ir_noise = NoiseSource::new(7);
        let ir: Vec<f32> = (0..ir_len)
            .map(|n| ir_noise.next() * (-(n as f32) / 3000.0).exp())
            .collect();
        let energy: f64 = ir.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let normalization = (1.0 / energy.sqrt()) as f32;

        let mut reverb = ConvolutionReverb::new();
        reverb
            .set_impulse_response(&ir, &ir, SAMPLE_RATE, SAMPLE_RATE)
            .unwrap();
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        let total = 4 * PARTITION_SIZE + 300;
        let mut in_noise = NoiseSource::new(99);
        let signal: Vec<f32> = (0..total).map(|_| in_noise.next()).collect();
        let (_, output) = run_blocks(&mut reverb, &params, total, 300, |n| signal[n]);

        // Direct convolution in f64.
        let mut direct = vec![0.0f64; total];
        for (n, slot) in direct.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for (k, &h) in ir.iter().enumerate().take(n + 1) {
                sum += h as f64 * normalization as f64 * signal[n - k] as f64;
            }
            *slot = sum;
        }
        // The wet gain ramps in over the first block; compare from the
        // second block after the latency.
        let mut max_error = 0.0f32;
        let mut max_value = 0.0f32;
        for n in 300..total - LATENCY_SAMPLES {
            let expected = direct[n] as f32;
            let actual = output[n + LATENCY_SAMPLES];
            max_error = max_error.max((expected - actual).abs());
            max_value = max_value.max(expected.abs());
        }
        assert!(max_value > 0.01, "direct convolution is degenerate");
        assert!(
            max_error < 2e-3 * max_value.max(1.0),
            "partitioned vs direct: max error {max_error} (peak {max_value})"
        );
    }

    #[test]
    fn dry_mix_is_transparent() {
        let mut reverb = ConvolutionReverb::new();
        let (left, right) = ir_plate(0.5, SAMPLE_RATE);
        reverb
            .set_impulse_response(&left, &right, SAMPLE_RATE, SAMPLE_RATE)
            .unwrap();
        let params = ConvolutionParams {
            dry_wet: 0.0,
            ..ConvolutionParams::default()
        };
        let (input, output) = run(&mut reverb, &params, 4 * BLOCK, |n| {
            (n as f32 * 0.05).sin() * 0.5
        });
        // Skip the first block: the dry gain ramps in from the reset state.
        for n in BLOCK..input.len() {
            assert!(
                (input[n] - output[n]).abs() < 1e-4,
                "sample {n}: dry path not transparent"
            );
        }
    }

    #[test]
    fn exponential_ir_produces_decaying_tail() {
        let mut reverb = ConvolutionReverb::new();
        let (left, right) = ir_plate(0.5, SAMPLE_RATE);
        reverb
            .set_impulse_response(&left, &right, SAMPLE_RATE, SAMPLE_RATE)
            .unwrap();
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        // A single impulse, then silence for a second.
        let total = SAMPLE_RATE as usize;
        let (_, output) = run(&mut reverb, &params, total, impulse);

        // The tail starts after the partition latency; window RMS must be
        // non-zero there and strictly decay across successive windows.
        let window = SAMPLE_RATE as usize / 10;
        let rms: Vec<f32> = (0..5)
            .map(|w| {
                let start = LATENCY_SAMPLES + w * window;
                let slice = &output[start..start + window];
                (slice.iter().map(|s| s * s).sum::<f32>() / window as f32).sqrt()
            })
            .collect();
        assert!(rms[0] > 1e-4, "no tail after the input stopped: {rms:?}");
        for pair in rms.windows(2) {
            assert!(pair[1] < pair[0], "tail RMS not decreasing: {rms:?}");
        }
        // 0.5 s T60: the last window (0.4-0.5 s into the tail) is far down.
        assert!(rms[4] < rms[0] * 0.1, "tail decays too slowly: {rms:?}");
    }

    #[test]
    fn predelay_delays_wet_onset_and_glides() {
        let onset = |predelay_seconds: f32| -> usize {
            let mut reverb = dirac_reverb();
            let params = ConvolutionParams {
                dry_wet: 1.0,
                predelay_seconds,
                ..ConvolutionParams::default()
            };
            // Let the pre-delay smoother settle on silence first.
            let _ = run(&mut reverb, &params, SAMPLE_RATE as usize, |_| 0.0);
            let total = LATENCY_SAMPLES + SAMPLE_RATE as usize / 4;
            let (_, output) = run(&mut reverb, &params, total, impulse);
            output
                .iter()
                .position(|s| s.abs() > 1e-3)
                .expect("wet onset never arrived")
        };

        let base = onset(0.0);
        let delayed = onset(0.05);
        assert_eq!(base, LATENCY_SAMPLES);
        let expected = (0.05 * SAMPLE_RATE as f32).round() as usize;
        assert_eq!(delayed - base, expected);

        // A pre-delay jump on a running signal glides instead of stepping:
        // the delayed wet copy of a slow sine stays continuous.
        let mut reverb = dirac_reverb();
        let mut params = ConvolutionParams {
            dry_wet: 1.0,
            predelay_seconds: 0.0,
            ..ConvolutionParams::default()
        };
        let sine = |n: usize| (n as f32 * 0.002).sin() * 0.5;
        let warm = 2 * LATENCY_SAMPLES;
        let _ = run(&mut reverb, &params, warm, sine);
        params.predelay_seconds = 0.2;
        let (_, output) = run(&mut reverb, &params, 4 * BLOCK, |n| sine(n + warm));
        let max_step = output
            .windows(2)
            .fold(0.0f32, |acc, w| acc.max((w[1] - w[0]).abs()));
        // The sine itself moves by at most 0.001 per sample; the glide of
        // 50 ms time constant reads the delay line a few samples further
        // back per sample at most, never a 0.2 s jump.
        assert!(max_step < 0.05, "pre-delay change stepped: {max_step}");
    }

    #[test]
    fn resampled_ir_is_finite_and_produces_tail() {
        let mut reverb = ConvolutionReverb::new();
        let (left, right) = ir_hall(0.4, 24000);
        reverb
            .set_impulse_response(&left, &right, 24000, SAMPLE_RATE)
            .unwrap();
        let params = ConvolutionParams {
            dry_wet: 0.8,
            ..ConvolutionParams::default()
        };
        let total = SAMPLE_RATE as usize / 2;
        let (_, output) = run(&mut reverb, &params, total, |n| {
            if n < 64 {
                0.5
            } else {
                0.0
            }
        });
        let tail_energy: f32 = output[LATENCY_SAMPLES + 4800..]
            .iter()
            .map(|s| s * s)
            .sum();
        assert!(tail_energy > 1e-6, "resampled IR produced no tail");
    }

    #[test]
    fn invalid_impulse_responses_are_rejected() {
        let mut reverb = ConvolutionReverb::new();
        assert_eq!(
            reverb.set_impulse_response(&[1.0], &[1.0], 0, SAMPLE_RATE),
            Err(IrError::ZeroSampleRate)
        );
        assert_eq!(
            reverb.set_impulse_response(&[1.0], &[1.0], SAMPLE_RATE, 0),
            Err(IrError::ZeroSampleRate)
        );
        assert_eq!(
            reverb.set_impulse_response(&[1.0, f32::NAN], &[1.0, 0.0], SAMPLE_RATE, SAMPLE_RATE),
            Err(IrError::NonFinite)
        );
        assert_eq!(
            reverb.set_impulse_response(&[0.0; 10], &[0.0; 10], SAMPLE_RATE, SAMPLE_RATE),
            Err(IrError::Silent)
        );
        assert!(!reverb.has_engine());
        assert!(resample_linear(&[1.0, 2.0], 0, 48000).is_empty());
    }

    #[test]
    fn set_sample_rate_rebuilds_from_the_retained_ir() {
        let mut reverb = ConvolutionReverb::new();
        let (left, right) = ir_plate(0.3, 44100);
        reverb.set_impulse_response(&left, &right, 44100, 44100).unwrap();
        assert_eq!(reverb.engine.as_ref().unwrap().sample_rate(), 44100.0);
        reverb.set_sample_rate(96000);
        let engine = reverb.engine.as_ref().expect("engine rebuilt");
        assert_eq!(engine.sample_rate(), 96000.0);
        // 0.3 s at 96 kHz = 28800 samples -> 15 partitions of 2048.
        assert_eq!(engine.num_partitions(), 15);
        // The displaced engine crossfades out first, then is parked for
        // off-thread disposal.
        assert!(reverb.take_retired().is_none());
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        let (_, output) = run(&mut reverb, &params, 4 * PARTITION_SIZE, impulse);
        assert!(output[LATENCY_SAMPLES..].iter().any(|s| s.abs() > 1e-4));
        assert!(reverb.take_retired().is_some());
    }

    #[test]
    fn engine_swap_crossfades_without_a_step() {
        // Dirac -> attenuated, delayed Dirac: the wet output must move
        // between the two responses gradually, never jump.
        let mut reverb = dirac_reverb();
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        let sine = |n: usize| (n as f32 * 0.01).sin() * 0.5;
        let warm = 3 * PARTITION_SIZE;
        let _ = run(&mut reverb, &params, warm, sine);

        let mut second = vec![0.0f32; 8];
        second[7] = -1.0;
        let engine = ConvolutionEngine::build(&second, &second, SAMPLE_RATE, SAMPLE_RATE).unwrap();
        assert!(reverb.swap_engine(Some(engine)).is_none());
        let total = 4 * PARTITION_SIZE;
        let (_, output) = run(&mut reverb, &params, total, |n| sine(n + warm));
        let max_step = output
            .windows(2)
            .fold(0.0f32, |acc, w| acc.max((w[1] - w[0]).abs()));
        // The sine moves by ~0.005 per sample; a hard switch between the
        // two responses would step by up to 1.0.
        assert!(max_step < 0.03, "engine swap stepped: {max_step}");
        // After the fade the new response is in charge (inverted, delayed
        // by 7 samples) and the old engine is retired.
        let n = total - 100;
        let expected = -sine(n + warm - 7 - LATENCY_SAMPLES);
        assert!((output[n] - expected).abs() < 1e-3, "{} vs {expected}", output[n]);
        assert!(reverb.take_retired().is_some());
        assert!(reverb.take_retired().is_none());
    }

    #[test]
    fn builtin_irs_are_deterministic_and_bounded() {
        for generate in [ir_plate, ir_hall, ir_spring] {
            let (left_a, right_a) = generate(1.0, 44100);
            let (left_b, right_b) = generate(1.0, 44100);
            assert_eq!(left_a, left_b);
            assert_eq!(right_a, right_b);
            assert_eq!(left_a.len(), 44100);
            assert!(left_a.iter().chain(&right_a).all(|s| s.is_finite()));
            assert!(left_a.iter().chain(&right_a).any(|s| s.abs() > 1e-3));
            assert_ne!(left_a, right_a, "channels must be decorrelated");
        }
    }

    #[test]
    fn hard_reset_clears_the_tail() {
        let mut reverb = ConvolutionReverb::new();
        let (left, right) = ir_plate(1.0, SAMPLE_RATE);
        reverb
            .set_impulse_response(&left, &right, SAMPLE_RATE, SAMPLE_RATE)
            .unwrap();
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        let total = LATENCY_SAMPLES + 2 * BLOCK;
        let _ = run(&mut reverb, &params, total, impulse);
        reverb.hard_reset();
        let (_, output) = run(&mut reverb, &params, LATENCY_SAMPLES, |_| 0.0);
        assert!(
            output.iter().all(|s| s.abs() < 1e-6),
            "tail survived hard_reset"
        );
    }
}
