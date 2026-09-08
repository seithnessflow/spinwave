//! Granular oscillator: scatters short windowed grains read from a loaded
//! [`Sample`] across time, position, pitch and the stereo field.
//!
//! Grains are inherently scalar events, so scheduling and per-grain state
//! run at scalar granularity per voice; output still lands per lane
//! (`[L0, R0, L1, R1]`) so the block interface matches the other
//! oscillators. The grain pool is preallocated — the per-sample loops
//! never allocate.
//!
//! Grain reads reuse the [`Sample`] tier pyramid: each grain picks the
//! band-limited tier matching its playback rate (like `SampleSource`) and
//! interpolates linearly or cubically inside it.

use spinwave_poly::{constants, PolyF32, PolyMask, PolyU32};

use super::sample_source::{
    Sample, BUFFER_SAMPLES, MAX_SAMPLE_AMPLITUDE, MAX_TRANSPOSE, MIN_TRANSPOSE, UPSAMPLE_TIMES,
};
use crate::modulators::RandomGenerator;

/// Hard cap on simultaneous grains per voice pair.
pub const MAX_GRAINS: usize = 64;
/// Highest spawn rate in grains per second.
pub const MAX_GRAIN_DENSITY: f32 = 150.0;
/// Shortest / longest grain length in seconds (params are clamped to this).
pub const MIN_GRAIN_SECONDS: f32 = 0.005;
pub const MAX_GRAIN_SECONDS: f32 = 2.0;

const MAX_BLOCK: usize = constants::MAX_BUFFER_SIZE;
const NUM_VOICES: usize = 2;

/// Per-grain playback direction policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GrainDirection {
    #[default]
    Forward,
    Reverse,
    /// Each grain independently plays reversed with probability 1/2.
    Bidirectional,
}

/// Amplitude envelope applied over a grain's lifetime.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GrainWindow {
    #[default]
    Hann,
    Triangle,
    /// Exponential decay with a few-sample linear attack to avoid an
    /// onset click.
    ExpoDecay,
    /// Tukey (tapered cosine) with a fixed 0.5 taper ratio.
    Tukey,
    /// No envelope; clicky by design (debugging / effect use).
    Rectangular,
}

/// Read quality for the per-grain sample interpolation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GrainInterpolation {
    Linear,
    #[default]
    Cubic,
}

/// Block-rate parameters for [`Granular`]. Per-voice values are read from
/// the voice's left lane (lanes 0 and 2).
#[derive(Clone, Debug)]
pub struct GranularParams {
    /// Grain start position in the source, `0..=1`.
    pub position: PolyF32,
    /// Random position offset per grain, as a fraction of the source
    /// length, `0..=1`.
    pub position_spray: PolyF32,
    /// Nominal grain length in seconds (clamped to
    /// [`MIN_GRAIN_SECONDS`]..=[`MAX_GRAIN_SECONDS`]).
    pub grain_size_seconds: PolyF32,
    /// Relative random grain-length variation, `0..=1`.
    pub size_spray: PolyF32,
    /// Grain spawn rate in grains per second, up to
    /// [`MAX_GRAIN_DENSITY`].
    pub density: PolyF32,
    /// Played MIDI note (used when `keytrack` is on, centered on
    /// `MIDI_TRACK_CENTER` like `SampleSourceParams`).
    pub midi: PolyF32,
    pub keytrack: bool,
    /// Transpose in semitones.
    pub transpose: PolyF32,
    /// Fine tune in semitones.
    pub tune: PolyF32,
    /// Random per-grain detune range in semitones.
    pub pitch_spray_semitones: PolyF32,
    pub direction: GrainDirection,
    pub window: GrainWindow,
    /// Random per-grain pan amount, `0..=1` (constant-power law).
    pub stereo_spray: PolyF32,
    /// Output level (applied squared, like the sample source).
    pub level: PolyF32,
    pub interpolation: GrainInterpolation,
    /// Simultaneous grain cap for the voice pair (clamped to
    /// [`MAX_GRAINS`]).
    pub max_grains: usize,
}

impl Default for GranularParams {
    fn default() -> Self {
        GranularParams {
            position: PolyF32::ZERO,
            position_spray: PolyF32::ZERO,
            grain_size_seconds: PolyF32::splat(0.1),
            size_spray: PolyF32::ZERO,
            density: PolyF32::splat(30.0),
            midi: PolyF32::splat(constants::MIDI_TRACK_CENTER as f32),
            keytrack: false,
            transpose: PolyF32::ZERO,
            tune: PolyF32::ZERO,
            pitch_spray_semitones: PolyF32::ZERO,
            direction: GrainDirection::Forward,
            window: GrainWindow::Hann,
            stereo_spray: PolyF32::ZERO,
            level: PolyF32::ONE,
            interpolation: GrainInterpolation::Cubic,
            max_grains: MAX_GRAINS,
        }
    }
}

/// One active grain: a windowed, panned read head over the sample.
/// Positions are in "active" (upsampled) units like `SampleSource`, kept
/// in `f64` so long samples don't lose read precision.
#[derive(Clone, Copy, Debug, Default)]
struct Grain {
    active: bool,
    voice: u8,
    /// Samples to wait inside the current block before the first output.
    delay: u32,
    /// Output samples rendered so far.
    age: u32,
    /// Total grain length in output samples.
    duration: u32,
    /// Read position in active (upsampled) units.
    position: f64,
    /// Signed step in active units per output sample.
    increment: f64,
    window: GrainWindow,
    interpolation: GrainInterpolation,
    /// Pan and overlap normalization baked in; level stays block-rate.
    gain_l: f32,
    gain_r: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct VoiceState {
    active: bool,
    /// Output samples until the next grain spawns (may span blocks).
    next_spawn: f64,
}

/// The granular playback engine for one voice pair.
pub struct Granular {
    grains: [Grain; MAX_GRAINS],
    voices: [VoiceState; NUM_VOICES],
    /// Per-lane scalar accumulators `[L0, R0, L1, R1]` for one chunk.
    scratch: [[f32; MAX_BLOCK]; 4],
    sample_rate: f32,
    rng: RandomGenerator,
}

impl Default for Granular {
    fn default() -> Self {
        Self::new()
    }
}

fn window_value(window: GrainWindow, w: f32) -> f32 {
    const TAU: f32 = 2.0 * constants::PI;
    match window {
        GrainWindow::Hann => 0.5 - 0.5 * (TAU * w).cos(),
        GrainWindow::Triangle => 1.0 - (2.0 * w - 1.0).abs(),
        GrainWindow::ExpoDecay => (w * 256.0).min(1.0) * (-6.0 * w).exp(),
        GrainWindow::Tukey => {
            const TAPER: f32 = 0.5;
            if w < TAPER * 0.5 {
                0.5 - 0.5 * (TAU * w / TAPER).cos()
            } else if w > 1.0 - TAPER * 0.5 {
                0.5 - 0.5 * (TAU * (1.0 - w) / TAPER).cos()
            } else {
                1.0
            }
        }
        GrainWindow::Rectangular => 1.0,
    }
}

/// Scalar Catmull-Rom over the 4 taps around `index` (guarded buffer).
#[inline]
fn catmull_read(buffer: &[f32], index: usize, t: f32) -> f32 {
    let y0 = buffer[index - 1];
    let y1 = buffer[index];
    let y2 = buffer[index + 1];
    let y3 = buffer[index + 2];
    let c1 = y2 - y0;
    let c2 = 2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3;
    let c3 = 3.0 * (y1 - y2) + y3 - y0;
    y1 + 0.5 * t * (c1 + t * (c2 + t * c3))
}

#[inline]
fn linear_read(buffer: &[f32], index: usize, t: f32) -> f32 {
    let y1 = buffer[index];
    let y2 = buffer[index + 1];
    y1 + t * (y2 - y1)
}

impl Granular {
    /// Auto-seeded instance (seeds from the engine's global counter,
    /// like every other `RandomGenerator::new` user).
    pub fn new() -> Granular {
        Self::from_rng(RandomGenerator::new(-1.0, 1.0))
    }

    /// Explicit seed for reproducible grain clouds (tests).
    pub fn with_seed(seed: u32) -> Granular {
        Self::from_rng(RandomGenerator::with_seed(-1.0, 1.0, seed))
    }

    fn from_rng(rng: RandomGenerator) -> Granular {
        Granular {
            grains: [Grain::default(); MAX_GRAINS],
            voices: [VoiceState::default(); NUM_VOICES],
            scratch: [[0.0; MAX_BLOCK]; 4],
            sample_rate: constants::DEFAULT_SAMPLE_RATE as f32,
            rng,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Number of grains currently sounding.
    pub fn active_grains(&self) -> usize {
        self.grains.iter().filter(|g| g.active).count()
    }

    /// Retriggers the grain clock for the voices in `mask`: pending grains
    /// of those voices are dropped and the first new grain fires at
    /// `sample_offset` (per voice, read from its left lane) inside the
    /// next processed block.
    pub fn note_on(&mut self, mask: PolyMask, sample_offset: PolyU32) {
        for voice in 0..NUM_VOICES {
            if mask.voice_any(voice) {
                self.voices[voice].active = true;
                self.voices[voice].next_spawn = sample_offset.lane(voice * 2) as f64;
                for grain in &mut self.grains {
                    if grain.voice as usize == voice {
                        grain.active = false;
                    }
                }
            }
        }
    }

    /// Renders one block of leveled stereo output for both voice lanes.
    /// Output is overwritten, not accumulated.
    pub fn process(
        &mut self,
        params: &GranularParams,
        sample: &Sample,
        num_samples: usize,
        out: &mut [PolyF32],
    ) {
        assert!(num_samples > 0);
        assert!(out.len() >= num_samples);

        let mut start = 0;
        while start < num_samples {
            let chunk = (num_samples - start).min(MAX_BLOCK);
            self.process_chunk(params, sample, chunk, &mut out[start..start + chunk]);
            start += chunk;
        }
    }

    fn process_chunk(
        &mut self,
        params: &GranularParams,
        sample: &Sample,
        chunk_len: usize,
        out: &mut [PolyF32],
    ) {
        for channel in &mut self.scratch {
            channel[..chunk_len].fill(0.0);
        }

        self.schedule(params, sample, chunk_len);
        self.render_grains(sample, chunk_len);

        let level = params.level.clamp(0.0, MAX_SAMPLE_AMPLITUDE);
        let gain = level * level;
        for (i, value) in out.iter_mut().take(chunk_len).enumerate() {
            *value = PolyF32::from_lanes([
                self.scratch[0][i],
                self.scratch[1][i],
                self.scratch[2][i],
                self.scratch[3][i],
            ]) * gain;
        }
    }

    /// Walks each active voice's grain clock through the chunk, spawning
    /// grains where it fires.
    fn schedule(&mut self, params: &GranularParams, sample: &Sample, chunk_len: usize) {
        for voice in 0..NUM_VOICES {
            if !self.voices[voice].active {
                continue;
            }

            let density = params.density.lane(voice * 2).clamp(0.0, MAX_GRAIN_DENSITY);
            if density < 1e-3 {
                self.voices[voice].next_spawn =
                    (self.voices[voice].next_spawn - chunk_len as f64).max(0.0);
                continue;
            }

            let interval = (self.sample_rate as f64 / density as f64).max(1.0);
            while self.voices[voice].next_spawn < chunk_len as f64 {
                let index = self.voices[voice].next_spawn.max(0.0) as u32;
                self.spawn_grain(voice, index, params, sample);
                self.voices[voice].next_spawn += interval;
            }
            self.voices[voice].next_spawn -= chunk_len as f64;
        }
    }

    /// Draws the jitter values and launches one grain (if the pool and the
    /// cap allow). The RNG is always advanced by the same number of draws
    /// so a seed maps to one deterministic cloud regardless of pool state.
    fn spawn_grain(&mut self, voice: usize, delay: u32, params: &GranularParams, sample: &Sample) {
        let u_position = self.rng.next();
        let u_size = self.rng.next();
        let u_pitch = self.rng.next();
        let u_pan = self.rng.next();
        let u_direction = self.rng.next();

        let max_grains = params.max_grains.min(MAX_GRAINS);
        let active = self.grains.iter().filter(|g| g.active).count();
        if active >= max_grains {
            return;
        }
        let Some(slot) = self.grains.iter().position(|g| !g.active) else {
            return;
        };

        let lane = voice * 2;
        let active_length = sample.active_length() as f64;
        if active_length < 8.0 {
            return;
        }

        // Grain length in output samples.
        let size_spray = params.size_spray.lane(lane).clamp(0.0, 1.0);
        let base_seconds =
            params.grain_size_seconds.lane(lane).clamp(MIN_GRAIN_SECONDS, MAX_GRAIN_SECONDS);
        let seconds = (base_seconds * (1.0 + u_size * size_spray))
            .clamp(MIN_GRAIN_SECONDS, MAX_GRAIN_SECONDS);
        let duration = ((seconds * self.sample_rate) as u32).max(2);

        // Pitch ratio: keytrack on the played note + transpose + tune +
        // per-grain spray, exactly the SampleSourceParams recipe.
        let keytrack_offset = if params.keytrack {
            params.midi.lane(lane) - constants::MIDI_TRACK_CENTER as f32
        } else {
            0.0
        };
        let transpose = (keytrack_offset
            + params.transpose.lane(lane)
            + params.tune.lane(lane)
            + u_pitch * params.pitch_spray_semitones.lane(lane))
        .clamp(MIN_TRANSPOSE, MAX_TRANSPOSE);
        let ratio = (transpose / constants::NOTES_PER_OCTAVE as f32).exp2() as f64;
        let step = ratio * (sample.sample_rate() as f64 / self.sample_rate as f64)
            * (1u32 << UPSAMPLE_TIMES) as f64;

        let reversed = match params.direction {
            GrainDirection::Forward => false,
            GrainDirection::Reverse => true,
            GrainDirection::Bidirectional => u_direction < 0.0,
        };
        let increment = if reversed { -step } else { step };

        // Start position, sprayed then clamped so the whole grain fits.
        let base_position =
            params.position.lane(lane).clamp(0.0, 1.0) as f64 * active_length;
        let spray = params.position_spray.lane(lane).clamp(0.0, 1.0) as f64;
        let mut position = base_position + u_position as f64 * spray * active_length;
        let span = (duration as f64 * step).min(active_length - 1.0);
        if reversed {
            position = position.clamp(span, active_length - 1.0);
        } else {
            position = position.clamp(0.0, (active_length - 1.0 - span).max(0.0));
        }

        // Constant-power pan (unity at center, like the engine pan law)
        // plus overlap normalization so dense clouds stay bounded.
        let pan = (u_pan * params.stereo_spray.lane(lane)).clamp(-1.0, 1.0);
        let angle = (pan + 1.0) * (constants::PI / 4.0);
        let density = params.density.lane(lane).clamp(0.0, MAX_GRAIN_DENSITY);
        let overlap = density * seconds;
        let normalize = 1.0 / overlap.max(1.0).sqrt();
        let gain_l = constants::SQRT_2 * angle.cos() * normalize;
        let gain_r = constants::SQRT_2 * angle.sin() * normalize;

        self.grains[slot] = Grain {
            active: true,
            voice: voice as u8,
            delay,
            age: 0,
            duration,
            position,
            increment,
            window: params.window,
            interpolation: params.interpolation,
            gain_l,
            gain_r,
        };
    }

    /// Renders every active grain into the scalar scratch lanes.
    fn render_grains(&mut self, sample: &Sample, chunk_len: usize) {
        let active_length = sample.active_length() as f64;
        let stereo = sample.stereo();

        for index in 0..MAX_GRAINS {
            if !self.grains[index].active {
                continue;
            }
            let mut grain = self.grains[index];

            // Tier and read scale are re-derived per block so a swapped
            // Sample can't leave stale indices behind.
            let tier = sample.active_index(grain.increment.abs() as f32);
            let phase_mult = 1.0 / (1u64 << tier) as f64;
            let left = sample.left_buffer(tier);
            let right = sample.right_buffer(tier);
            let body_length = (left.len() - 2 * BUFFER_SAMPLES) as isize;

            let start = grain.delay.min(chunk_len as u32) as usize;
            grain.delay = grain.delay.saturating_sub(chunk_len as u32);
            let inv_duration = 1.0 / grain.duration as f32;
            let base_lane = grain.voice as usize * 2;
            let cubic = grain.interpolation == GrainInterpolation::Cubic;

            for i in start..chunk_len {
                if grain.age >= grain.duration
                    || grain.position < 0.0
                    || grain.position >= active_length
                {
                    grain.active = false;
                    break;
                }

                let read = grain.position * phase_mult;
                let floor = read.floor();
                let t = (read - floor) as f32;
                let tap = (floor as isize).clamp(0, body_length - 1) as usize + BUFFER_SAMPLES;

                let value_l = if cubic {
                    catmull_read(left, tap, t)
                } else {
                    linear_read(left, tap, t)
                };
                let value_r = if !stereo {
                    value_l
                } else if cubic {
                    catmull_read(right, tap, t)
                } else {
                    linear_read(right, tap, t)
                };

                let window = window_value(grain.window, grain.age as f32 * inv_duration);
                self.scratch[base_lane][i] += value_l * window * grain.gain_l;
                self.scratch[base_lane + 1][i] += value_r * window * grain.gain_r;

                grain.position += grain.increment;
                grain.age += 1;
            }

            self.grains[index] = grain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spinwave_poly::LANES;

    const SR: f32 = 44100.0;

    fn sine_sample(period: usize) -> Sample {
        let length = 44100;
        let buffer: Vec<f32> = (0..length)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / period as f32).sin())
            .collect();
        let mut sample = Sample::default();
        sample.load_sample(&buffer, 44100);
        sample
    }

    fn ramp_sample() -> Sample {
        let length = 44100;
        let buffer: Vec<f32> = (0..length).map(|i| i as f32 / length as f32).collect();
        let mut sample = Sample::default();
        sample.load_sample(&buffer, 44100);
        sample
    }

    fn render(
        granular: &mut Granular,
        params: &GranularParams,
        sample: &Sample,
        total: usize,
    ) -> Vec<PolyF32> {
        let mut out = vec![PolyF32::ZERO; total];
        granular.process(params, sample, total, &mut out);
        out
    }

    fn lane(buffer: &[PolyF32], index: usize) -> Vec<f32> {
        buffer.iter().map(|v| v.lane(index)).collect()
    }

    /// Coherent grain settings: the spawn interval (400 samples) is a
    /// multiple of the source period so grains sum in phase.
    fn coherent_params() -> GranularParams {
        GranularParams {
            position: PolyF32::splat(0.25),
            density: PolyF32::splat(SR / 400.0),
            grain_size_seconds: PolyF32::splat(0.1),
            ..Default::default()
        }
    }

    /// First strong autocorrelation peak = fundamental period.
    fn dominant_period(x: &[f32], max_lag: usize) -> usize {
        let n = x.len() - max_lag;
        let mut r = vec![0.0f64; max_lag + 1];
        let mut best = 0.0f64;
        for (lag, slot) in r.iter_mut().enumerate().skip(1) {
            let mut sum = 0.0f64;
            for i in 0..n {
                sum += x[i] as f64 * x[i + lag] as f64;
            }
            *slot = sum;
            best = best.max(sum);
        }
        for lag in 2..max_lag {
            if r[lag] >= 0.9 * best && r[lag] >= r[lag - 1] && r[lag] >= r[lag + 1] {
                return lag;
            }
        }
        0
    }

    fn assert_finite(buffer: &[PolyF32]) {
        for value in buffer {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn silent_without_note_on() {
        let sample = sine_sample(100);
        let mut granular = Granular::with_seed(1);
        let out = render(&mut granular, &GranularParams::default(), &sample, 4096);
        for value in &out {
            assert_eq!(value.to_lanes(), [0.0; LANES]);
        }
    }

    #[test]
    fn note_on_offset_delays_first_grain() {
        let sample = sine_sample(100);
        let mut granular = Granular::with_seed(1);
        granular.note_on(PolyMask::all_on(), PolyU32::splat(64));
        let params = GranularParams {
            density: PolyF32::splat(100.0),
            ..Default::default()
        };
        let out = render(&mut granular, &params, &sample, 128);
        for value in &out[..64] {
            assert_eq!(value.to_lanes(), [0.0; LANES]);
        }
        assert!(out[64..].iter().any(|v| v.lane(0).abs() > 1e-4));
    }

    #[test]
    fn voice_masks_are_independent() {
        let sample = sine_sample(100);
        let mut granular = Granular::with_seed(1);
        let voice0 = PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, u32::MAX, 0, 0]));
        granular.note_on(voice0, PolyU32::ZERO);
        let out = render(&mut granular, &coherent_params(), &sample, 4096);
        assert!(out.iter().any(|v| v.lane(0).abs() > 1e-3));
        for value in &out {
            assert_eq!(value.lane(2), 0.0);
            assert_eq!(value.lane(3), 0.0);
        }
    }

    #[test]
    fn pitch_matches_source() {
        let sample = sine_sample(100);
        let mut granular = Granular::with_seed(3);
        granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let out = render(&mut granular, &coherent_params(), &sample, 22050);
        assert_finite(&out);

        let signal = lane(&out[8820..17640], 0);
        let energy: f32 = signal.iter().map(|v| v * v).sum();
        assert!(energy > 1.0, "output too quiet: {energy}");
        let period = dominant_period(&signal, 300);
        assert!(
            (period as i32 - 100).abs() <= 2,
            "expected period ~100, got {period}"
        );
    }

    #[test]
    fn transpose_shifts_pitch_by_ratio() {
        let sample = sine_sample(100);
        let mut granular = Granular::with_seed(3);
        granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let params = GranularParams {
            transpose: PolyF32::splat(12.0),
            ..coherent_params()
        };
        let out = render(&mut granular, &params, &sample, 22050);
        assert_finite(&out);

        let signal = lane(&out[8820..17640], 0);
        let period = dominant_period(&signal, 300);
        assert!(
            (period as i32 - 50).abs() <= 2,
            "expected period ~50 (ratio 2), got {period}"
        );
    }

    #[test]
    fn keytrack_matches_equivalent_transpose() {
        let sample = sine_sample(100);
        let keytracked = GranularParams {
            keytrack: true,
            midi: PolyF32::splat(72.0),
            ..coherent_params()
        };
        let transposed = GranularParams {
            transpose: PolyF32::splat(12.0),
            ..coherent_params()
        };

        let mut a = Granular::with_seed(11);
        let mut b = Granular::with_seed(11);
        a.note_on(PolyMask::all_on(), PolyU32::ZERO);
        b.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let out_a = render(&mut a, &keytracked, &sample, 8192);
        let out_b = render(&mut b, &transposed, &sample, 8192);
        for (x, y) in out_a.iter().zip(&out_b) {
            assert_eq!(x.to_lanes(), y.to_lanes());
        }
    }

    #[test]
    fn density_scales_output_activity() {
        let sample = sine_sample(100);
        let base = GranularParams {
            grain_size_seconds: PolyF32::splat(0.02),
            position: PolyF32::splat(0.25),
            ..Default::default()
        };
        let mut activity = Vec::new();
        for density in [5.0, 50.0] {
            let params = GranularParams { density: PolyF32::splat(density), ..base.clone() };
            let mut granular = Granular::with_seed(5);
            granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
            let out = render(&mut granular, &params, &sample, 44100);
            let active = out[4410..].iter().filter(|v| v.lane(0).abs() > 1e-3).count();
            activity.push(active);
        }
        assert!(
            activity[1] > 2 * activity[0],
            "density 50 should cover much more than density 5: {activity:?}"
        );
    }

    #[test]
    fn spray_differs_across_seeds_but_not_within() {
        let sample = sine_sample(100);
        let params = GranularParams {
            position: PolyF32::splat(0.3),
            position_spray: PolyF32::splat(0.2),
            size_spray: PolyF32::splat(0.5),
            pitch_spray_semitones: PolyF32::splat(7.0),
            stereo_spray: PolyF32::ONE,
            direction: GrainDirection::Bidirectional,
            density: PolyF32::splat(60.0),
            ..Default::default()
        };

        let run = |seed: u32| {
            let mut granular = Granular::with_seed(seed);
            granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
            render(&mut granular, &params, &sample, 8192)
        };

        let a = run(21);
        let b = run(21);
        let c = run(22);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_lanes(), y.to_lanes());
        }
        let difference: f32 = a
            .iter()
            .zip(&c)
            .map(|(x, y)| (*x - *y).abs().sum_lanes())
            .sum();
        assert!(difference > 1e-2, "different seeds should differ: {difference}");
    }

    #[test]
    fn zero_spray_is_seed_invariant() {
        let sample = sine_sample(100);
        let params = coherent_params();
        let run = |seed: u32| {
            let mut granular = Granular::with_seed(seed);
            granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
            render(&mut granular, &params, &sample, 4096)
        };
        let a = run(1);
        let b = run(999);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.to_lanes(), y.to_lanes());
        }
    }

    #[test]
    fn window_prevents_clicks() {
        let sample = sine_sample(100);
        let base = GranularParams {
            grain_size_seconds: PolyF32::splat(0.03),
            ..coherent_params()
        };

        let max_delta = |window: GrainWindow| {
            let params = GranularParams { window, ..base.clone() };
            let mut granular = Granular::with_seed(9);
            granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
            let signal = lane(&render(&mut granular, &params, &sample, 22050), 0);
            signal
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0f32, f32::max)
        };

        let hann = max_delta(GrainWindow::Hann);
        let rectangular = max_delta(GrainWindow::Rectangular);
        assert!(hann > 0.0);
        assert!(
            rectangular > 2.5 * hann,
            "hann delta {hann} should be much smaller than rectangular {rectangular}"
        );
    }

    #[test]
    fn stereo_spray_pans_grains() {
        let sample = sine_sample(100);
        let centered = coherent_params();
        let mut granular = Granular::with_seed(4);
        granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let out = render(&mut granular, &centered, &sample, 4096);
        for value in &out {
            assert_eq!(value.lane(0), value.lane(1), "mono + no spray must be centered");
        }

        let sprayed = GranularParams { stereo_spray: PolyF32::ONE, ..coherent_params() };
        let mut granular = Granular::with_seed(4);
        granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let out = render(&mut granular, &sprayed, &sample, 4096);
        assert!(
            out.iter().any(|v| (v.lane(0) - v.lane(1)).abs() > 1e-3),
            "stereo spray should decorrelate the channels"
        );
    }

    #[test]
    fn reverse_grains_read_backwards() {
        let sample = ramp_sample();
        let base = GranularParams {
            position: PolyF32::splat(0.5),
            density: PolyF32::splat(2.0),
            grain_size_seconds: PolyF32::splat(0.05),
            window: GrainWindow::Rectangular,
            ..Default::default()
        };

        let slope_signs = |direction: GrainDirection| {
            let params = GranularParams { direction, ..base.clone() };
            let mut granular = Granular::with_seed(6);
            granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
            let signal = lane(&render(&mut granular, &params, &sample, 8192), 0);
            let mut positive = 0usize;
            let mut negative = 0usize;
            for w in signal.windows(2) {
                if w[0].abs() > 1e-3 && w[1].abs() > 1e-3 {
                    if w[1] > w[0] {
                        positive += 1;
                    } else if w[1] < w[0] {
                        negative += 1;
                    }
                }
            }
            (positive, negative)
        };

        let (forward_up, forward_down) = slope_signs(GrainDirection::Forward);
        let (reverse_up, reverse_down) = slope_signs(GrainDirection::Reverse);
        assert!(forward_up > 10 * forward_down.max(1), "forward should rise on a ramp");
        assert!(reverse_down > 10 * reverse_up.max(1), "reverse should fall on a ramp");
    }

    #[test]
    fn grain_cap_is_respected() {
        let sample = sine_sample(100);
        let saturated = GranularParams {
            density: PolyF32::splat(150.0),
            grain_size_seconds: PolyF32::splat(1.0),
            ..Default::default()
        };
        let mut granular = Granular::with_seed(8);
        granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let out = render(&mut granular, &saturated, &sample, 44100);
        assert_finite(&out);
        assert!(granular.active_grains() <= MAX_GRAINS);
        assert!(granular.active_grains() > MAX_GRAINS / 2, "expected a saturated pool");

        let capped = GranularParams { max_grains: 8, ..saturated };
        let mut granular = Granular::with_seed(8);
        granular.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let _ = render(&mut granular, &capped, &sample, 44100);
        assert!(granular.active_grains() <= 8);
    }

    #[test]
    fn long_render_stays_finite_and_bounded() {
        let sample = sine_sample(100);
        let params = GranularParams {
            position: PolyF32::splat(0.4),
            position_spray: PolyF32::splat(0.3),
            size_spray: PolyF32::splat(0.8),
            pitch_spray_semitones: PolyF32::splat(12.0),
            stereo_spray: PolyF32::ONE,
            direction: GrainDirection::Bidirectional,
            window: GrainWindow::Tukey,
            density: PolyF32::splat(120.0),
            grain_size_seconds: PolyF32::splat(0.3),
            interpolation: GrainInterpolation::Linear,
            ..Default::default()
        };
        let mut granular = Granular::with_seed(13);
        granular.note_on(PolyMask::all_on(), PolyU32::ZERO);

        let mut out = vec![PolyF32::ZERO; 128];
        for _ in 0..(5 * 44100 / 128) {
            granular.process(&params, &sample, 128, &mut out);
            for value in &out {
                assert!(value.is_finite());
                for lane in value.to_lanes() {
                    assert!(lane.abs() < 8.0, "unbounded output: {lane}");
                }
            }
        }
    }
}
