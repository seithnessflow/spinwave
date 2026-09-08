//! Convolution reverb over a stereo impulse response.
//!
//! Uniform partitioned convolution (overlap-save FFT, partition size
//! [`PARTITION_SIZE`], FFT size `2 * PARTITION_SIZE`). The folded bus signal
//! is convolved as true stereo: the left IR channel processes the left lanes,
//! the right IR channel the right lanes (both voice slots carry the same
//! folded signal, so lanes 0/1 are the authoritative L/R pair and the result
//! is broadcast back to both voices).
//!
//! **Latency**: the wet path is delayed by exactly [`LATENCY_SAMPLES`]
//! (= one partition, 2048 samples) relative to the dry path — the classic
//! input-buffering latency of uniform partitioned convolution. The dry path
//! has zero latency.
//!
//! All FFT plans, spectra, delay lines and scratch buffers are allocated in
//! [`ConvolutionReverb::set_impulse_response`]; `process` never allocates.
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

/// Block-rate convolution reverb parameters.
#[derive(Clone, Copy, Debug)]
pub struct ConvolutionParams {
    /// Dry/wet in [0, 1]; applied as an equal-power crossfade
    /// (0 = fully dry, 1 = fully wet).
    pub dry_wet: f32,
    /// Extra delay on the wet path in seconds, clamped to
    /// [0, [`MAX_PREDELAY_SECONDS`]]. Adds on top of [`LATENCY_SAMPLES`].
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

/// Preallocated per-IR convolution state (built by `set_impulse_response`).
struct Engine {
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    num_partitions: usize,
    /// Per channel: one `NUM_BINS` spectrum per IR partition.
    ir_spectra: [Vec<Vec<Complex<f32>>>; 2],
    /// Per channel: frequency-domain delay line of past input spectra.
    fdl: [Vec<Vec<Complex<f32>>>; 2],
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
    acc: Vec<Complex<f32>>,
    scratch_fwd: Vec<Complex<f32>>,
    scratch_inv: Vec<Complex<f32>>,
    /// Per channel: pre-delay ring for the wet path.
    predelay: [Vec<f32>; 2],
    predelay_pos: usize,
    sample_rate: f32,
}

impl Engine {
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
            self.fdl[channel][self.fdl_pos].copy_from_slice(&self.spec_buf);

            self.acc.fill(Complex::new(0.0, 0.0));
            for partition in 0..self.num_partitions {
                let input =
                    &self.fdl[channel][(self.fdl_pos + partition) % self.num_partitions];
                let ir = &self.ir_spectra[channel][partition];
                for ((acc, x), h) in self.acc.iter_mut().zip(input).zip(ir) {
                    *acc += x * h;
                }
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
            for spectrum in &mut self.fdl[channel] {
                spectrum.fill(Complex::new(0.0, 0.0));
            }
            self.history[channel].fill(0.0);
            self.out_fifo[channel].fill(0.0);
            self.predelay[channel].fill(0.0);
        }
        self.fdl_pos = 0;
        self.in_pos = 0;
        self.predelay_pos = 0;
    }
}

/// Uniform partitioned convolution reverb; see the module docs for the
/// partition scheme and latency.
pub struct ConvolutionReverb {
    engine: Option<Engine>,
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
    /// silent until [`ConvolutionReverb::set_impulse_response`] is called.
    pub fn new() -> ConvolutionReverb {
        ConvolutionReverb {
            engine: None,
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

    /// Loads a stereo impulse response and (re)builds all convolution state.
    ///
    /// The IR is linearly resampled if `sample_rate_of_ir` differs from
    /// `engine_sample_rate`, truncated to [`MAX_IR_SECONDS`] at the engine
    /// rate, and energy-normalized so a unit-impulse IR is transparent-ish
    /// (unity RMS energy per channel pair). A silent or empty IR unloads the
    /// engine. This is the cold path: all allocation happens here.
    pub fn set_impulse_response(
        &mut self,
        left: &[f32],
        right: &[f32],
        sample_rate_of_ir: u32,
        engine_sample_rate: u32,
    ) {
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
            self.engine = None;
            return;
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

        let mut ir_spectra: [Vec<Vec<Complex<f32>>>; 2] = [Vec::new(), Vec::new()];
        for (channel, samples) in [&left, &right].into_iter().enumerate() {
            for partition in 0..num_partitions {
                let start = partition * PARTITION_SIZE;
                let end = (start + PARTITION_SIZE).min(ir_len);
                time_buf.fill(0.0);
                for (slot, &value) in time_buf.iter_mut().zip(&samples[start..end]) {
                    *slot = value * normalization;
                }
                let mut spectrum = vec![Complex::new(0.0, 0.0); NUM_BINS];
                let _ = r2c.process_with_scratch(&mut time_buf, &mut spectrum, &mut scratch_fwd);
                ir_spectra[channel].push(spectrum);
            }
        }

        let empty_fdl = || {
            (0..num_partitions)
                .map(|_| vec![Complex::new(0.0, 0.0); NUM_BINS])
                .collect::<Vec<_>>()
        };
        let predelay_len =
            ((MAX_PREDELAY_SECONDS * engine_sample_rate as f32) as usize).max(1) + 1;

        self.engine = Some(Engine {
            r2c,
            c2r,
            num_partitions,
            ir_spectra,
            fdl: [empty_fdl(), empty_fdl()],
            fdl_pos: 0,
            history: [vec![0.0; FFT_SIZE], vec![0.0; FFT_SIZE]],
            in_pos: 0,
            out_fifo: [vec![0.0; PARTITION_SIZE], vec![0.0; PARTITION_SIZE]],
            time_buf,
            spec_buf: vec![Complex::new(0.0, 0.0); NUM_BINS],
            acc: vec![Complex::new(0.0, 0.0); NUM_BINS],
            scratch_fwd,
            scratch_inv,
            predelay: [vec![0.0; predelay_len], vec![0.0; predelay_len]],
            predelay_pos: 0,
            sample_rate: engine_sample_rate as f32,
        });
    }

    /// Clears every delay line, spectrum history and mix ramp; keeps the
    /// loaded impulse response.
    pub fn hard_reset(&mut self) {
        self.dry = PolyF32::ZERO;
        self.wet = PolyF32::ZERO;
        self.gain = 1.0;
        if let Some(engine) = &mut self.engine {
            engine.clear_runtime_state();
        }
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

        let predelay_len = engine.predelay[0].len();
        let delay_samples = ((params.predelay_seconds.clamp(0.0, MAX_PREDELAY_SECONDS)
            * engine.sample_rate)
            .round() as usize)
            .min(predelay_len - 1);

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            let mut input = sample & first_voice_mask();
            input += input.swap_voices();

            let wet_left = engine.out_fifo[0][engine.in_pos];
            let wet_right = engine.out_fifo[1][engine.in_pos];
            engine.history[0][PARTITION_SIZE + engine.in_pos] = input.lane(0);
            engine.history[1][PARTITION_SIZE + engine.in_pos] = input.lane(1);
            engine.in_pos += 1;
            if engine.in_pos == PARTITION_SIZE {
                engine.step();
                engine.in_pos = 0;
            }

            engine.predelay[0][engine.predelay_pos] = wet_left;
            engine.predelay[1][engine.predelay_pos] = wet_right;
            let read = (engine.predelay_pos + predelay_len - delay_samples) % predelay_len;
            let delayed = PolyF32::stereo(engine.predelay[0][read], engine.predelay[1][read]);
            engine.predelay_pos = (engine.predelay_pos + 1) % predelay_len;

            current_dry += delta_dry;
            current_wet += delta_wet;
            current_gain += delta_gain;
            *out = current_dry * input + current_wet * (delayed * current_gain);
        }
    }
}

fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
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
        reverb.set_impulse_response(&[1.0], &[1.0], SAMPLE_RATE, SAMPLE_RATE);
        reverb
    }

    /// Runs the reverb over deterministic input, returning (input, output)
    /// lane-0 sample streams.
    fn run(
        reverb: &mut ConvolutionReverb,
        params: &ConvolutionParams,
        total: usize,
        signal: impl Fn(usize) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut input_samples = Vec::with_capacity(total);
        let mut output_samples = Vec::with_capacity(total);
        let mut output = vec![PolyF32::ZERO; BLOCK];
        let mut n = 0;
        while n < total {
            let count = BLOCK.min(total - n);
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
    fn dry_mix_is_transparent() {
        let mut reverb = ConvolutionReverb::new();
        let (left, right) = ir_plate(0.5, SAMPLE_RATE);
        reverb.set_impulse_response(&left, &right, SAMPLE_RATE, SAMPLE_RATE);
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
        reverb.set_impulse_response(&left, &right, SAMPLE_RATE, SAMPLE_RATE);
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        // A single impulse, then silence for a second.
        let total = SAMPLE_RATE as usize;
        let (_, output) = run(&mut reverb, &params, total, |n| {
            if n == 0 {
                1.0
            } else {
                0.0
            }
        });

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
    fn predelay_delays_wet_onset() {
        let onset = |predelay_seconds: f32| -> usize {
            let mut reverb = dirac_reverb();
            let params = ConvolutionParams {
                dry_wet: 1.0,
                predelay_seconds,
                ..ConvolutionParams::default()
            };
            let total = LATENCY_SAMPLES + SAMPLE_RATE as usize / 4;
            let (_, output) = run(&mut reverb, &params, total, |n| {
                if n == 0 {
                    1.0
                } else {
                    0.0
                }
            });
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
    }

    #[test]
    fn resampled_ir_is_finite_and_produces_tail() {
        let mut reverb = ConvolutionReverb::new();
        let (left, right) = ir_hall(0.4, 24000);
        reverb.set_impulse_response(&left, &right, 24000, SAMPLE_RATE);
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
        reverb.set_impulse_response(&left, &right, SAMPLE_RATE, SAMPLE_RATE);
        let params = ConvolutionParams {
            dry_wet: 1.0,
            ..ConvolutionParams::default()
        };
        let total = LATENCY_SAMPLES + 2 * BLOCK;
        let _ = run(&mut reverb, &params, total, |n| {
            if n == 0 {
                1.0
            } else {
                0.0
            }
        });
        reverb.hard_reset();
        let (_, output) = run(&mut reverb, &params, LATENCY_SAMPLES, |_| 0.0);
        assert!(
            output.iter().all(|s| s.abs() < 1e-6),
            "tail survived hard_reset"
        );
    }
}
