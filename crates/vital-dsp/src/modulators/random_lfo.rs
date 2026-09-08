//! Random modulation source (port of `RandomLfo`).
//!
//! Styles: perlin-smoothed noise, stepped sample-and-hold, sine-interpolated
//! noise, and a Lorenz attractor. Output is unipolar `[0, 1]`.

use vital_poly::constants::VoiceEvent;
use vital_poly::{math, utils, PolyF32, PolyMask, PolyU32, LANES};

use super::random::RandomGenerator;

const LORENZ_INITIAL1: f32 = 0.0;
const LORENZ_INITIAL2: f32 = 0.0;
const LORENZ_INITIAL3: f32 = 37.6;
const LORENZ_A: f32 = 10.0;
const LORENZ_B: f32 = 28.0;
const LORENZ_C: f32 = 8.0 / 3.0;
const LORENZ_SIZE: f32 = 40.0;
const LORENZ_SCALE: f32 = 1.0 / LORENZ_SIZE;
const LORENZ_MAX_FREQUENCY: f32 = 0.01;

/// Lanes of the first voice (`kFirstMask`).
fn first_voice_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, u32::MAX, 0, 0]))
}

/// Left-channel lanes of both voices (`kLeftMask`).
fn left_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, 0, u32::MAX, 0]))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RandomLfoStyle {
    #[default]
    Perlin,
    SampleAndHold,
    SinInterpolate,
    LorenzAttractor,
}

#[derive(Clone, Copy, Debug)]
pub struct RandomLfoParams {
    pub frequency: PolyF32,
    pub style: RandomLfoStyle,
    /// Independent left/right random streams when true.
    pub stereo: bool,
    /// Follow host transport time instead of free-running/reset triggers.
    pub sync: bool,
}

impl Default for RandomLfoParams {
    fn default() -> Self {
        RandomLfoParams {
            frequency: PolyF32::splat(1.0),
            style: RandomLfoStyle::Perlin,
            stereo: false,
            sync: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RandomState {
    offset: PolyF32,
    last_random_value: PolyF32,
    next_random_value: PolyF32,
    state1: PolyF32,
    state2: PolyF32,
    state3: PolyF32,
}

impl Default for RandomState {
    fn default() -> Self {
        RandomState {
            offset: PolyF32::ZERO,
            last_random_value: PolyF32::ZERO,
            next_random_value: PolyF32::ZERO,
            state1: PolyF32::splat(0.1),
            state2: PolyF32::ZERO,
            state3: PolyF32::ZERO,
        }
    }
}

#[derive(Clone)]
pub struct RandomLfo {
    sample_rate: f32,
    state: RandomState,
    random_generator: RandomGenerator,
    last_value: PolyF32,
    last_output: PolyF32,

    sync_seconds: f64,
    last_sync: f64,

    reset_mask: PolyMask,
    reset_offset: PolyU32,
}

impl RandomLfo {
    pub fn new(sample_rate: f32) -> Self {
        Self::with_generator(sample_rate, RandomGenerator::new(-1.0, 1.0))
    }

    /// Deterministic construction for tests/replays.
    pub fn with_seed(sample_rate: f32, seed: u32) -> Self {
        Self::with_generator(sample_rate, RandomGenerator::with_seed(-1.0, 1.0, seed))
    }

    fn with_generator(sample_rate: f32, random_generator: RandomGenerator) -> Self {
        RandomLfo {
            sample_rate,
            state: RandomState::default(),
            random_generator,
            last_value: PolyF32::ZERO,
            last_output: PolyF32::ZERO,
            sync_seconds: 0.0,
            last_sync: 0.0,
            reset_mask: PolyMask::NONE,
            reset_offset: PolyU32::ZERO,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Sets the host transport time used when `sync` is on.
    pub fn correct_to_time(&mut self, seconds: f64) {
        self.sync_seconds = seconds;
    }

    /// Queues a reset (note-on retrigger) for the next process call.
    pub fn trigger(&mut self, mask: PolyMask, value: PolyF32, sample_offset: usize) {
        self.reset_mask = mask & value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
        self.reset_offset = PolyU32::splat(sample_offset as u32);
    }

    pub fn value(&self) -> PolyF32 {
        self.last_output
    }

    fn do_reset(&mut self, mono: bool, frequency: PolyF32, sync: bool) {
        let reset_mask = self.reset_mask;
        if !reset_mask.any() || sync {
            return;
        }

        let sample_offset = self.reset_offset.to_f32_signed();
        let start_offset = frequency * (1.0 / self.sample_rate) * sample_offset;
        self.state.offset = reset_mask.select(-start_offset, self.state.offset);

        let (from_random, to_random) = if mono {
            (self.random_generator.poly_voice_next(), self.random_generator.poly_voice_next())
        } else {
            (self.random_generator.poly_next(), self.random_generator.poly_next())
        };

        self.state.last_random_value =
            reset_mask.select(from_random, self.state.last_random_value);
        self.state.next_random_value = reset_mask.select(to_random, self.state.next_random_value);
        self.last_value = reset_mask
            .select(self.state.last_random_value * 0.5 + 0.5, self.last_value);
    }

    /// Advances the phase, drawing new random targets on wrap. Returns the
    /// per-lane sample index where the wrap happened (0 when none).
    fn update_phase(&mut self, params: &RandomLfoParams, num_samples: usize) -> PolyU32 {
        let frequency = params.frequency;
        let phase_delta = frequency * (1.0 / self.sample_rate) * num_samples as f32;
        let mono = !params.stereo;
        let mut new_random_mask = PolyMask::NONE;

        if params.sync {
            if self.last_sync != self.sync_seconds {
                let new_offset = utils::cycle_offset_from_seconds(self.sync_seconds, frequency);
                new_random_mask =
                    new_offset.lt(PolyF32::splat(0.5)) & self.state.offset.ge(PolyF32::splat(0.5));
                self.state.offset = new_offset;
            }
        } else {
            self.do_reset(mono, frequency, params.sync);

            self.state.offset += phase_delta;
            new_random_mask = self.state.offset.ge(PolyF32::ONE);
            self.state.offset = self.state.offset.fract();
        }

        if new_random_mask.any() {
            self.state.last_random_value =
                new_random_mask.select(self.state.next_random_value, self.state.last_random_value);
            let next_random = if mono {
                self.random_generator.poly_voice_next()
            } else {
                self.random_generator.poly_next()
            };
            self.state.next_random_value =
                new_random_mask.select(next_random, self.state.next_random_value);

            let delta = phase_delta.le(PolyF32::ZERO).select(PolyF32::ONE, phase_delta);
            let samples_to_wrap = self.state.offset / delta;
            return samples_to_wrap.to_i32_round();
        }

        PolyU32::ZERO
    }

    /// Control-rate tick covering `num_samples` samples; returns the value.
    pub fn process_control(&mut self, params: &RandomLfoParams, num_samples: usize) -> PolyF32 {
        if params.sync {
            if self.last_sync != self.sync_seconds {
                let result = self.compute_control(params, num_samples);
                let first = result & first_voice_mask();
                self.last_output = first + first.swap_voices();
                self.last_sync = self.sync_seconds;
            }
        } else {
            self.last_output = self.compute_control(params, num_samples);
        }
        self.clear_reset();
        self.last_output
    }

    /// Audio-rate processing filling `out` sample by sample.
    pub fn process_audio(&mut self, params: &RandomLfoParams, out: &mut [PolyF32]) {
        if params.sync {
            if self.last_sync != self.sync_seconds {
                self.compute_audio(params, out);
                for sample in out.iter_mut() {
                    let first = *sample & first_voice_mask();
                    *sample = first + first.swap_voices();
                }
                self.last_output = out[out.len() - 1];
                self.last_sync = self.sync_seconds;
            } else {
                out.fill(self.last_output);
            }
        } else {
            self.compute_audio(params, out);
            self.last_output = out[out.len() - 1];
        }
        self.clear_reset();
    }

    fn clear_reset(&mut self) {
        self.reset_mask = PolyMask::NONE;
        self.reset_offset = PolyU32::ZERO;
    }

    fn interpolated_value(&self, style: RandomLfoStyle) -> PolyF32 {
        let result = match style {
            RandomLfoStyle::Perlin => utils::perlin_interpolate(
                self.state.last_random_value,
                self.state.next_random_value,
                self.state.offset,
            ),
            RandomLfoStyle::SinInterpolate => math::sin_interpolate(
                self.state.last_random_value,
                self.state.next_random_value,
                self.state.offset,
            ),
            _ => PolyF32::ZERO,
        };
        result * 0.5 + 0.5
    }

    fn compute_control(&mut self, params: &RandomLfoParams, num_samples: usize) -> PolyF32 {
        match params.style {
            RandomLfoStyle::LorenzAttractor => self.lorenz(params, num_samples, None),
            RandomLfoStyle::SampleAndHold => {
                self.update_phase(params, num_samples);
                self.state.last_random_value * 0.5 + 0.5
            }
            style => {
                self.update_phase(params, num_samples);
                let result = self.interpolated_value(style);
                self.last_value = result;
                result
            }
        }
    }

    fn compute_audio(&mut self, params: &RandomLfoParams, out: &mut [PolyF32]) {
        let num_samples = out.len();
        match params.style {
            RandomLfoStyle::LorenzAttractor => {
                self.lorenz(params, num_samples, Some(out));
            }
            RandomLfoStyle::SampleAndHold => {
                let last_random_value = self.state.last_random_value * 0.5 + 0.5;
                let sample_change = self.update_phase(params, num_samples);
                let current_random_value = self.state.last_random_value * 0.5 + 0.5;

                for (i, sample) in out.iter_mut().enumerate() {
                    let over = greater_than_signed(i as i32, sample_change);
                    *sample = over.select(current_random_value, last_random_value);
                }
            }
            style => {
                self.update_phase(params, num_samples);
                let result = self.interpolated_value(style);

                let mut current_value = self.last_value;
                let delta_value = (result - current_value) * (1.0 / num_samples as f32);
                for sample in out.iter_mut() {
                    current_value += delta_value;
                    *sample = current_value;
                }
                self.last_value = result;
            }
        }
    }

    fn lorenz(
        &mut self,
        params: &RandomLfoParams,
        num_samples: usize,
        mut out: Option<&mut [PolyF32]>,
    ) -> PolyF32 {
        let mono = !params.stereo;
        let stereo_equal_mask = self.state.state1.eq(self.state.state1.swap_stereo());

        let mut state1 = self.state.state1;
        let mut state2 = self.state.state2;
        let mut state3 = self.state.state3;

        let reset_mask = self.reset_mask;
        if reset_mask.any() && !params.sync {
            let (value1, value2, value3) = if mono {
                (
                    self.random_generator.poly_voice_next() + LORENZ_INITIAL1,
                    self.random_generator.poly_voice_next() + LORENZ_INITIAL2,
                    self.random_generator.poly_voice_next() + LORENZ_INITIAL3,
                )
            } else {
                (
                    self.random_generator.poly_next() + LORENZ_INITIAL1,
                    self.random_generator.poly_next() + LORENZ_INITIAL2,
                    self.random_generator.poly_next() + LORENZ_INITIAL3,
                )
            };
            state1 = reset_mask.select(value1, state1);
            state2 = reset_mask.select(value2, state2);
            state3 = reset_mask.select(value3, state3);
        }

        if mono {
            state1 = state1 & left_mask();
            state1 += state1.swap_stereo();
            state2 = state2 & left_mask();
            state2 += state2.swap_stereo();
            state3 = state3 & left_mask();
            state3 += state3.swap_stereo();
        } else {
            state1 -= (state1 * 0.5) & stereo_equal_mask & left_mask();
        }

        let frequency = params.frequency;
        let t = (frequency * (0.5 / self.sample_rate)).min(PolyF32::splat(LORENZ_MAX_FREQUENCY));

        for i in 0..num_samples {
            let delta1 = (state2 - state1) * LORENZ_A;
            let delta2 = (-state3 + LORENZ_B) * state1 - state2;
            let delta3 = state1 * state2 - state3 * LORENZ_C;
            state1 += delta1 * t;
            state2 += delta2 * t;
            state3 += delta3 * t;

            if let Some(out) = out.as_deref_mut() {
                out[i] = state1 * LORENZ_SCALE + 0.5;
            }
        }

        self.state.state1 = state1;
        self.state.state2 = state2;
        self.state.state3 = state3;

        state1 * LORENZ_SCALE + 0.5
    }
}

/// Per-lane signed `i > value` (matching `poly_int::greaterThan`).
fn greater_than_signed(i: i32, value: PolyU32) -> PolyMask {
    let lanes = value.0;
    let mut mask = [0u32; LANES];
    for (out, &lane) in mask.iter_mut().zip(lanes.iter()) {
        if i > lane as i32 {
            *out = u32::MAX;
        }
    }
    PolyMask::from_u32(PolyU32::from_lanes(mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;

    fn on() -> PolyF32 {
        PolyF32::splat(VoiceEvent::On.as_f32())
    }

    #[test]
    fn deterministic_with_same_seed() {
        let params = RandomLfoParams { frequency: PolyF32::splat(50.0), ..Default::default() };
        let mut a = RandomLfo::with_seed(SAMPLE_RATE, 9);
        let mut b = RandomLfo::with_seed(SAMPLE_RATE, 9);
        a.trigger(PolyMask::all_on(), on(), 0);
        b.trigger(PolyMask::all_on(), on(), 0);
        for _ in 0..200 {
            let value_a = a.process_control(&params, 64);
            let value_b = b.process_control(&params, 64);
            assert_eq!(value_a.to_lanes(), value_b.to_lanes());
        }
    }

    #[test]
    fn output_in_unipolar_range() {
        for style in [
            RandomLfoStyle::Perlin,
            RandomLfoStyle::SampleAndHold,
            RandomLfoStyle::SinInterpolate,
        ] {
            let params =
                RandomLfoParams { frequency: PolyF32::splat(100.0), style, ..Default::default() };
            let mut lfo = RandomLfo::with_seed(SAMPLE_RATE, 11);
            lfo.trigger(PolyMask::all_on(), on(), 0);
            for _ in 0..500 {
                let value = lfo.process_control(&params, 32);
                for lane in value.to_lanes() {
                    assert!((-0.001..=1.001).contains(&lane), "{style:?} out of range: {lane}");
                }
            }
        }
    }

    #[test]
    fn sample_and_hold_steps() {
        let params = RandomLfoParams {
            frequency: PolyF32::splat(441.0),
            style: RandomLfoStyle::SampleAndHold,
            ..Default::default()
        };
        let mut lfo = RandomLfo::with_seed(SAMPLE_RATE, 4);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        // 441 Hz at 44.1kHz: period of 100 samples, blocks of 10 samples:
        // the value must be constant within a period and change across it.
        let mut values = Vec::new();
        for _ in 0..100 {
            values.push(lfo.process_control(&params, 10).lane(0));
        }
        let distinct: usize = values.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(distinct > 3, "expected several steps, got {distinct}");
        let stable: usize = values.windows(2).filter(|w| w[0] == w[1]).count();
        assert!(stable > 50, "expected held values between steps, got {stable}");
    }

    #[test]
    fn mono_mode_matches_stereo_lanes() {
        let params = RandomLfoParams { frequency: PolyF32::splat(200.0), ..Default::default() };
        let mut lfo = RandomLfo::with_seed(SAMPLE_RATE, 21);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        for _ in 0..100 {
            let value = lfo.process_control(&params, 64);
            assert_eq!(value.lane(0), value.lane(1));
            assert_eq!(value.lane(2), value.lane(3));
        }
    }

    #[test]
    fn lorenz_stays_bounded() {
        let params = RandomLfoParams {
            frequency: PolyF32::splat(10.0),
            style: RandomLfoStyle::LorenzAttractor,
            ..Default::default()
        };
        let mut lfo = RandomLfo::with_seed(SAMPLE_RATE, 2);
        let mut out = [PolyF32::ZERO; 64];
        for _ in 0..500 {
            lfo.process_audio(&params, &mut out);
        }
        for sample in out {
            for lane in sample.to_lanes() {
                assert!(lane.is_finite());
                assert!((-1.0..=2.0).contains(&lane), "lorenz diverged: {lane}");
            }
        }
    }

    #[test]
    fn audio_rate_ramps_smoothly() {
        let params = RandomLfoParams { frequency: PolyF32::splat(20.0), ..Default::default() };
        let mut lfo = RandomLfo::with_seed(SAMPLE_RATE, 5);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        let mut out = [PolyF32::ZERO; 128];
        lfo.process_audio(&params, &mut out);
        // Per-block linear ramp: constant per-sample delta.
        let d0 = out[1].lane(0) - out[0].lane(0);
        let d1 = out[100].lane(0) - out[99].lane(0);
        assert!((d0 - d1).abs() < 1e-5);
    }
}
