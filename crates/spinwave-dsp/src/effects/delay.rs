//! Delay line effect with mono/stereo/ping-pong and clamped/dampened styles
//! (port of Vital's `Delay`). Tempo-sync lives in the module layer: the
//! params carry the target delay period in samples, per lane.

use spinwave_poly::constants::{MIN_NYQUIST_MULT, NOTES_PER_OCTAVE, SQRT_2};
use spinwave_poly::utils::interpolate;
use spinwave_poly::{math, PolyF32};

use crate::memory::{Memory, StereoMemory};

use super::lanes::{left_mask, right_mask};
use super::one_pole::OnePole;

/// Starting value of the smoothed period frequency (Hz) at construction
/// and after `hard_reset`.
const INITIAL_FREQUENCY: f32 = 2.0;

pub const SPREAD_OCTAVE_RANGE: f32 = 8.0;
pub const DEFAULT_PERIOD: f32 = 100.0;
/// Half-life in seconds of the delay-period (frequency) smoothing.
pub const DELAY_HALF_LIFE: f32 = 0.02;
pub const MIN_DAMP_NOTE: f32 = 60.0;
pub const MAX_DAMP_NOTE: f32 = 136.0;

/// Delay routing/filtering style. `MidPingPong` corresponds to the
/// reference's `kMidPingPong` (stereo-in ping-pong) and `PingPong` to its
/// `kPingPong` (mono-summed ping-pong), matching Vital's UI naming.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DelayStyle {
    #[default]
    Mono,
    Stereo,
    PingPong,
    MidPingPong,
    ClampedDampened,
    ClampedUnfiltered,
    UnclampedUnfiltered,
}

/// Block-rate delay parameters.
#[derive(Clone, Copy, Debug)]
pub struct DelayParams {
    /// Target delay period in samples, per lane (`[L, R, L, R]`). The module
    /// layer resolves tempo sync / aux frequency into this. Must be > 0.
    pub period_samples: PolyF32,
    /// Feedback amount, clamped to [-1, 1].
    pub feedback: PolyF32,
    /// Wet amount in [0, 1]; dry/wet uses an equal-power fade.
    pub wet: PolyF32,
    /// Damping amount in [0, 1] (only used by `ClampedDampened`).
    pub damping: PolyF32,
    /// Band-pass filter center as a MIDI note (filtered styles).
    pub filter_cutoff_midi: PolyF32,
    /// Filter spread in [0, 1] octaves fraction (filtered styles).
    pub filter_spread: PolyF32,
    pub style: DelayStyle,
}

impl Default for DelayParams {
    fn default() -> DelayParams {
        DelayParams {
            period_samples: PolyF32::splat(DEFAULT_PERIOD),
            feedback: PolyF32::ZERO,
            wet: PolyF32::ZERO,
            damping: PolyF32::ZERO,
            filter_cutoff_midi: PolyF32::splat(60.0),
            filter_spread: PolyF32::ONE,
            style: DelayStyle::Mono,
        }
    }
}

/// Delay memory abstraction so the same effect runs on per-lane
/// ([`Memory`]) or global stereo ([`StereoMemory`]) rings.
pub trait DelayMemory {
    fn push(&mut self, sample: PolyF32);
    fn get(&self, period: PolyF32) -> PolyF32;
    fn clear_all(&mut self);
    fn max_period(&self) -> usize;
}

impl DelayMemory for Memory {
    #[inline(always)]
    fn push(&mut self, sample: PolyF32) {
        Memory::push(self, sample);
    }
    #[inline(always)]
    fn get(&self, period: PolyF32) -> PolyF32 {
        Memory::get(self, period)
    }
    fn clear_all(&mut self) {
        Memory::clear_all(self);
    }
    fn max_period(&self) -> usize {
        Memory::max_period(self)
    }
}

impl DelayMemory for StereoMemory {
    #[inline(always)]
    fn push(&mut self, sample: PolyF32) {
        StereoMemory::push(self, sample);
    }
    #[inline(always)]
    fn get(&self, period: PolyF32) -> PolyF32 {
        StereoMemory::get(self, period)
    }
    fn clear_all(&mut self) {
        StereoMemory::clear_all(self);
    }
    fn max_period(&self) -> usize {
        StereoMemory::max_period(self)
    }
}

/// Global stereo delay (shared ring for both voice slots).
pub type StereoDelay = Delay<StereoMemory>;
/// Per-lane delay (independent ring per lane).
pub type MultiDelay = Delay<Memory>;

#[inline(always)]
fn saturate(value: PolyF32) -> PolyF32 {
    math::hard_tanh(value)
}

#[inline(always)]
fn saturate_large(value: PolyF32) -> PolyF32 {
    const RATIO: f32 = 8.0;
    math::hard_tanh(value * (1.0 / RATIO)) * RATIO
}

#[inline(always)]
fn filter_radius(spread: PolyF32) -> PolyF32 {
    (spread * (SPREAD_OCTAVE_RANGE * NOTES_PER_OCTAVE as f32)).max(PolyF32::ZERO)
}

pub struct Delay<M: DelayMemory> {
    memory: M,
    sample_rate: f32,
    last_frequency: PolyF32,
    feedback: PolyF32,
    wet: PolyF32,
    dry: PolyF32,
    period: PolyF32,
    low_coefficient: PolyF32,
    high_coefficient: PolyF32,
    filter_gain: PolyF32,
    low_pass: OnePole,
    high_pass: OnePole,
}

impl Delay<StereoMemory> {
    pub fn new(max_samples: usize, sample_rate: f32) -> StereoDelay {
        Delay::with_memory(StereoMemory::new(max_samples), sample_rate)
    }
}

impl Delay<Memory> {
    pub fn new(max_samples: usize, sample_rate: f32) -> MultiDelay {
        Delay::with_memory(Memory::new(max_samples), sample_rate)
    }
}

impl<M: DelayMemory> Delay<M> {
    pub fn with_memory(memory: M, sample_rate: f32) -> Delay<M> {
        let max_period = memory.max_period() as f32;
        let mut delay = Delay {
            memory,
            sample_rate,
            last_frequency: PolyF32::splat(INITIAL_FREQUENCY),
            feedback: PolyF32::ZERO,
            wet: PolyF32::ZERO,
            dry: PolyF32::ZERO,
            period: PolyF32::splat(DEFAULT_PERIOD.min(max_period)),
            low_coefficient: PolyF32::ZERO,
            high_coefficient: PolyF32::ZERO,
            filter_gain: PolyF32::ZERO,
            low_pass: OnePole::new(),
            high_pass: OnePole::new(),
        };
        delay.hard_reset();
        delay
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn hard_reset(&mut self) {
        self.memory.clear_all();
        self.filter_gain = PolyF32::ZERO;
        self.low_pass.hard_reset();
        self.high_pass.hard_reset();
        // Restart the period smoother from its construction state so no
        // non-finite value can survive a reset (the C++ never reaches a
        // non-finite state; see the period clamp in `process`).
        self.last_frequency = PolyF32::splat(INITIAL_FREQUENCY);
    }

    /// Processes one block; `audio_in` and `audio_out` must be equal length.
    pub fn process(&mut self, params: &DelayParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        assert_eq!(audio_in.len(), audio_out.len());
        let num_samples = audio_in.len();
        if num_samples == 0 {
            return;
        }

        let current_wet = self.wet;
        let current_dry = self.dry;
        let mut current_feedback = self.feedback;
        let current_period = self.period;
        let current_filter_gain = self.filter_gain;
        let current_low_coefficient = self.low_coefficient;
        let current_high_coefficient = self.high_coefficient;

        let style = params.style;
        // A zero/negative period would give an infinite frequency that the
        // smoothing then turns into NaN for good; clamp to one sample.
        let target_frequency =
            PolyF32::splat(self.sample_rate) / params.period_samples.max(PolyF32::ONE);

        let decay = math::exp2(PolyF32::splat(
            -(num_samples as f32) / (DELAY_HALF_LIFE * self.sample_rate),
        ));
        self.last_frequency = interpolate(target_frequency, self.last_frequency, decay);

        let wet_in = params.wet.clamp(0.0, 1.0);
        self.wet = math::equal_power_fade(wet_in);
        self.dry = math::equal_power_fade_inverse(wet_in);
        self.feedback = params.feedback.clamp(-1.0, 1.0);

        let mut samples = PolyF32::splat(self.sample_rate) / self.last_frequency;
        if style == DelayStyle::MidPingPong {
            samples += samples.swap_stereo() & left_mask();
        }
        if style == DelayStyle::PingPong {
            current_feedback = right_mask().select(PolyF32::ONE, current_feedback);
            self.feedback = right_mask().select(PolyF32::ONE, self.feedback);
        }

        self.period = samples.clamp(3.0, self.memory.max_period() as f32);
        self.period = interpolate(current_period, self.period, PolyF32::splat(0.5));

        let filter_cutoff = params.filter_cutoff_midi;
        let radius = filter_radius(params.filter_spread);

        let min_nyquist = self.sample_rate * MIN_NYQUIST_MULT;
        let low_frequency =
            math::midi_note_to_frequency(filter_cutoff + radius).clamp(1.0, min_nyquist);
        self.low_coefficient = OnePole::compute_coefficient(low_frequency, self.sample_rate);

        let high_frequency =
            math::midi_note_to_frequency(filter_cutoff - radius).clamp(1.0, min_nyquist);
        self.high_coefficient = OnePole::compute_coefficient(high_frequency, self.sample_rate);

        self.filter_gain = high_frequency / low_frequency + 1.0;
        let damping = params.damping.clamp(0.0, 1.0);
        let damping_note = interpolate(
            PolyF32::splat(MIN_DAMP_NOTE),
            PolyF32::splat(MAX_DAMP_NOTE),
            damping,
        );
        let damping_frequency = math::midi_note_to_frequency(damping_note);

        match style {
            DelayStyle::Mono | DelayStyle::Stereo => self.process_filtered(
                audio_in,
                audio_out,
                current_period,
                current_feedback,
                current_filter_gain,
                current_low_coefficient,
                current_high_coefficient,
                current_wet,
                current_dry,
            ),
            DelayStyle::PingPong => self.process_mono_ping_pong(
                audio_in,
                audio_out,
                current_period,
                current_feedback,
                current_filter_gain,
                current_low_coefficient,
                current_high_coefficient,
                current_wet,
                current_dry,
            ),
            DelayStyle::MidPingPong => self.process_ping_pong(
                audio_in,
                audio_out,
                current_period,
                current_feedback,
                current_filter_gain,
                current_low_coefficient,
                current_high_coefficient,
                current_wet,
                current_dry,
            ),
            DelayStyle::ClampedDampened => {
                let damping_frequency = damping_frequency.clamp(1.0, min_nyquist);
                self.low_coefficient =
                    OnePole::compute_coefficient(damping_frequency, self.sample_rate);
                self.process_damped(
                    audio_in,
                    audio_out,
                    current_period,
                    current_feedback,
                    current_low_coefficient,
                    current_wet,
                    current_dry,
                );
            }
            DelayStyle::UnclampedUnfiltered => self.process_clean_unfiltered(
                audio_in,
                audio_out,
                current_period,
                current_feedback,
                current_wet,
                current_dry,
            ),
            DelayStyle::ClampedUnfiltered => self.process_unfiltered(
                audio_in,
                audio_out,
                current_period,
                current_feedback,
                current_wet,
                current_dry,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_clean_unfiltered(
        &mut self,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        mut current_period: PolyF32,
        mut current_feedback: PolyF32,
        mut current_wet: PolyF32,
        mut current_dry: PolyF32,
    ) {
        let tick_increment = 1.0 / audio_in.len() as f32;
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;
        let delta_feedback = (self.feedback - current_feedback) * tick_increment;
        let delta_period = (self.period - current_period) * tick_increment;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            current_feedback += delta_feedback;
            current_wet += delta_wet;
            current_dry += delta_dry;

            let read = self.memory.get(current_period);
            self.memory.push(sample + read * current_feedback);
            *out = current_dry * sample + current_wet * read;

            current_period += delta_period;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_unfiltered(
        &mut self,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        mut current_period: PolyF32,
        mut current_feedback: PolyF32,
        mut current_wet: PolyF32,
        mut current_dry: PolyF32,
    ) {
        let tick_increment = 1.0 / audio_in.len() as f32;
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;
        let delta_feedback = (self.feedback - current_feedback) * tick_increment;
        let delta_period = (self.period - current_period) * tick_increment;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            current_feedback += delta_feedback;
            current_wet += delta_wet;
            current_dry += delta_dry;

            let read = self.memory.get(current_period);
            self.memory.push(saturate(sample + read * current_feedback));
            *out = current_dry * sample + current_wet * read;

            current_period += delta_period;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_filtered(
        &mut self,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        mut current_period: PolyF32,
        mut current_feedback: PolyF32,
        mut current_filter_gain: PolyF32,
        mut current_low_coefficient: PolyF32,
        mut current_high_coefficient: PolyF32,
        mut current_wet: PolyF32,
        mut current_dry: PolyF32,
    ) {
        let tick_increment = 1.0 / audio_in.len() as f32;
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;
        let delta_feedback = (self.feedback - current_feedback) * tick_increment;
        let delta_period = (self.period - current_period) * tick_increment;
        let delta_filter_gain = (self.filter_gain - current_filter_gain) * tick_increment;
        let delta_low = (self.low_coefficient - current_low_coefficient) * tick_increment;
        let delta_high = (self.high_coefficient - current_high_coefficient) * tick_increment;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            current_feedback += delta_feedback;
            current_wet += delta_wet;
            current_dry += delta_dry;
            current_filter_gain += delta_filter_gain;
            current_low_coefficient += delta_low;
            current_high_coefficient += delta_high;

            let read = self.memory.get(current_period);
            let write_raw = saturate_large(sample + read * current_feedback);
            let low_pass_result = self
                .low_pass
                .tick_basic(write_raw * current_filter_gain, current_low_coefficient);
            let second_pass_result =
                self.high_pass.tick_basic(low_pass_result, current_high_coefficient);
            self.memory.push(low_pass_result - second_pass_result);
            *out = current_dry * sample + current_wet * read;

            current_period += delta_period;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_damped(
        &mut self,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        mut current_period: PolyF32,
        mut current_feedback: PolyF32,
        mut current_low_coefficient: PolyF32,
        mut current_wet: PolyF32,
        mut current_dry: PolyF32,
    ) {
        let tick_increment = 1.0 / audio_in.len() as f32;
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;
        let delta_feedback = (self.feedback - current_feedback) * tick_increment;
        let delta_period = (self.period - current_period) * tick_increment;
        let delta_low = (self.low_coefficient - current_low_coefficient) * tick_increment;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            current_feedback += delta_feedback;
            current_wet += delta_wet;
            current_dry += delta_dry;
            current_low_coefficient += delta_low;

            let read = self.memory.get(current_period);
            let write_raw = saturate_large(sample + read * current_feedback);
            let low_pass_result = self.low_pass.tick_basic(write_raw, current_low_coefficient);
            self.memory.push(low_pass_result);
            *out = current_dry * sample + current_wet * read;

            current_period += delta_period;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_ping_pong(
        &mut self,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        mut current_period: PolyF32,
        mut current_feedback: PolyF32,
        mut current_filter_gain: PolyF32,
        mut current_low_coefficient: PolyF32,
        mut current_high_coefficient: PolyF32,
        mut current_wet: PolyF32,
        mut current_dry: PolyF32,
    ) {
        let tick_increment = 1.0 / audio_in.len() as f32;
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;
        let delta_feedback = (self.feedback - current_feedback) * tick_increment;
        let delta_period = (self.period - current_period) * tick_increment;
        let delta_filter_gain = (self.filter_gain - current_filter_gain) * tick_increment;
        let delta_low = (self.low_coefficient - current_low_coefficient) * tick_increment;
        let delta_high = (self.high_coefficient - current_high_coefficient) * tick_increment;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            current_feedback += delta_feedback;
            current_wet += delta_wet;
            current_dry += delta_dry;
            current_filter_gain += delta_filter_gain;
            current_low_coefficient += delta_low;
            current_high_coefficient += delta_high;

            let read = self.memory.get(current_period);
            let write_raw = saturate_large(sample + read * current_feedback).swap_stereo();
            let low_pass_result = self
                .low_pass
                .tick_basic(write_raw * current_filter_gain, current_low_coefficient);
            let second_pass_result =
                self.high_pass.tick_basic(low_pass_result, current_high_coefficient);
            self.memory.push(low_pass_result - second_pass_result);
            *out = current_dry * sample + current_wet * read;

            current_period += delta_period;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_mono_ping_pong(
        &mut self,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        mut current_period: PolyF32,
        mut current_feedback: PolyF32,
        mut current_filter_gain: PolyF32,
        mut current_low_coefficient: PolyF32,
        mut current_high_coefficient: PolyF32,
        mut current_wet: PolyF32,
        mut current_dry: PolyF32,
    ) {
        let tick_increment = 1.0 / audio_in.len() as f32;
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;
        let delta_feedback = (self.feedback - current_feedback) * tick_increment;
        let delta_period = (self.period - current_period) * tick_increment;
        let delta_filter_gain = (self.filter_gain - current_filter_gain) * tick_increment;
        let delta_low = (self.low_coefficient - current_low_coefficient) * tick_increment;
        let delta_high = (self.high_coefficient - current_high_coefficient) * tick_increment;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            current_feedback += delta_feedback;
            current_wet += delta_wet;
            current_dry += delta_dry;
            current_filter_gain += delta_filter_gain;
            current_low_coefficient += delta_low;
            current_high_coefficient += delta_high;

            let read = self.memory.get(current_period);
            let mono_in = ((sample + sample.swap_stereo()) * (1.0 / SQRT_2)) & left_mask();
            let write_raw = saturate_large(mono_in + read * current_feedback).swap_stereo();
            let low_pass_result = self
                .low_pass
                .tick_basic(write_raw * current_filter_gain, current_low_coefficient);
            let second_pass_result =
                self.high_pass.tick_basic(low_pass_result, current_high_coefficient);
            self.memory.push(low_pass_result - second_pass_result);
            *out = current_dry * sample + current_wet * read;

            current_period += delta_period;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 64;

    fn warm_up(delay: &mut StereoDelay, params: &DelayParams, blocks: usize) {
        let silence = [PolyF32::ZERO; BLOCK];
        let mut out = [PolyF32::ZERO; BLOCK];
        for _ in 0..blocks {
            delay.process(params, &silence, &mut out);
        }
    }

    #[test]
    fn integer_period_delays_impulse() {
        let period = 100.0f32;
        let mut delay = StereoDelay::new(2048, SAMPLE_RATE);
        let params = DelayParams {
            period_samples: PolyF32::splat(period),
            wet: PolyF32::ONE,
            style: DelayStyle::UnclampedUnfiltered,
            ..DelayParams::default()
        };
        // Let the period smoothing converge (half-life 20 ms).
        warm_up(&mut delay, &params, 2000);

        let mut input = vec![PolyF32::ZERO; 512];
        input[0] = PolyF32::ONE;
        let mut output = vec![PolyF32::ZERO; 512];
        delay.process(&params, &input, &mut output);

        // Wet level is an equal-power fade of 1.0 (~1.0 within approx error).
        // Read happens before push (as in the reference), so an impulse
        // pushed at sample 0 comes back at `period + 1`.
        let expected_index = period as usize + 1;
        let value = output[expected_index].lane(0);
        assert!((value - 1.0).abs() < 1e-2, "delayed impulse was {value}");
        for (i, sample) in output.iter().enumerate() {
            assert!(sample.is_finite());
            if i.abs_diff(expected_index) > 1 && i != 0 {
                assert!(sample.lane(0).abs() < 1e-3, "leakage at {i}: {}", sample.lane(0));
            }
        }
    }

    #[test]
    fn zero_period_never_poisons_the_smoother() {
        let mut delay = StereoDelay::new(2048, SAMPLE_RATE);
        let mut params = DelayParams {
            period_samples: PolyF32::ZERO,
            wet: PolyF32::ONE,
            feedback: PolyF32::splat(0.5),
            style: DelayStyle::UnclampedUnfiltered,
            ..DelayParams::default()
        };
        let mut input = vec![PolyF32::ZERO; BLOCK];
        input[0] = PolyF32::ONE;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for _ in 0..4 {
            delay.process(&params, &input, &mut output);
            assert!(output.iter().all(|v| v.is_finite()), "zero period produced non-finite");
        }
        // Back to a sane period: the delay must recover, not stay NaN.
        params.period_samples = PolyF32::splat(100.0);
        for _ in 0..200 {
            delay.process(&params, &input, &mut output);
            assert!(output.iter().all(|v| v.is_finite()));
        }
        assert!(output.iter().any(|v| v.lane(0).abs() > 1e-3), "delay stayed silent");

        // A hard reset also discards a smoother state the caller may have
        // driven to something absurd through a negative period.
        params.period_samples = PolyF32::splat(-5.0);
        delay.process(&params, &input, &mut output);
        delay.hard_reset();
        params.period_samples = PolyF32::splat(100.0);
        delay.process(&params, &input, &mut output);
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sine_output_finite_and_bounded_with_feedback() {
        let mut delay = StereoDelay::new(2048, SAMPLE_RATE);
        let params = DelayParams {
            period_samples: PolyF32::splat(220.25),
            feedback: PolyF32::splat(0.9),
            wet: PolyF32::splat(0.7),
            style: DelayStyle::Mono,
            filter_cutoff_midi: PolyF32::splat(80.0),
            filter_spread: PolyF32::splat(0.5),
            ..DelayParams::default()
        };
        let mut peak = 0.0f32;
        let mut out = [PolyF32::ZERO; BLOCK];
        for block in 0..400 {
            let mut input = [PolyF32::ZERO; BLOCK];
            for (i, value) in input.iter_mut().enumerate() {
                let t = (block * BLOCK + i) as f32 / SAMPLE_RATE;
                *value = PolyF32::splat((2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.8);
            }
            delay.process(&params, &input, &mut out);
            for sample in &out {
                assert!(sample.is_finite());
                peak = peak.max(sample.abs().lane(0)).max(sample.abs().lane(1));
            }
        }
        assert!(peak < 4.0, "peak {peak}");
    }

    #[test]
    fn ping_pong_alternates_channels() {
        let period = 50.0f32;
        let mut delay = StereoDelay::new(1024, SAMPLE_RATE);
        let params = DelayParams {
            period_samples: PolyF32::splat(period),
            feedback: PolyF32::splat(0.5),
            wet: PolyF32::ONE,
            style: DelayStyle::PingPong,
            filter_cutoff_midi: PolyF32::splat(66.0),
            filter_spread: PolyF32::ONE,
            ..DelayParams::default()
        };
        warm_up(&mut delay, &params, 2000);

        let mut input = vec![PolyF32::ZERO; 512];
        input[0] = PolyF32::stereo(1.0, 1.0);
        let mut output = vec![PolyF32::ZERO; 512];
        delay.process(&params, &input, &mut output);
        for sample in &output {
            assert!(sample.is_finite());
        }
        let total: f32 = output.iter().map(|s| s.abs().sum_lanes()).sum();
        assert!(total > 0.1, "ping-pong produced no signal");
    }
}
