//! Comb / flange filter (port of `comb_filter.{h,cpp}`).

use vital_poly::constants::{MIN_NYQUIST_MULT, NOTES_PER_OCTAVE};
use vital_poly::utils::interpolate;
use vital_poly::{math, PolyF32, PolyMask, LANES};

use crate::memory::{Memory, MIN_PERIOD};

use super::filter_state::{
    frequency_to_midi_note_precise, midi_note_to_frequency_precise, FilterState,
};
use super::one_pole::{OnePoleFilter, Pass};

pub const BAND_OCTAVE_RANGE: f32 = 8.0;
pub const BAND_OCTAVE_MIN: f32 = 0.0;
pub const MIN_PERIOD_SIZE: usize = 2;
pub const INPUT_SCALE: f32 = 0.5;
pub const MAX_FEEDBACK: f32 = 1.0;
const FLANGE_SCALE: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// How the delay feedback is wired (C++ `CombFilter::FeedbackStyle`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FeedbackStyle {
    #[default]
    Comb = 0,
    PositiveFlange = 1,
    NegativeFlange = 2,
}

pub const NUM_FEEDBACK_STYLES: i32 = 3;

/// How the internal one-pole pair filters the feedback path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CombFilterStyle {
    #[default]
    LowHighBlend = 0,
    BandSpread = 1,
}

pub const NUM_FILTER_STYLES: i32 = 2;
pub const NUM_FILTER_TYPES: i32 = NUM_FILTER_STYLES * NUM_FEEDBACK_STYLES;

impl FeedbackStyle {
    /// `style % kNumFeedbackStyles` like the C++ mapping.
    pub fn from_style_index(style: i32) -> FeedbackStyle {
        match style.rem_euclid(NUM_FEEDBACK_STYLES) {
            1 => FeedbackStyle::PositiveFlange,
            2 => FeedbackStyle::NegativeFlange,
            _ => FeedbackStyle::Comb,
        }
    }
}

impl CombFilterStyle {
    /// `style / kNumFeedbackStyles` like the C++ mapping.
    pub fn from_style_index(style: i32) -> CombFilterStyle {
        if style / NUM_FEEDBACK_STYLES >= 1 {
            CombFilterStyle::BandSpread
        } else {
            CombFilterStyle::LowHighBlend
        }
    }
}

#[inline(always)]
fn low_gain(blend: PolyF32) -> PolyF32 {
    (-blend + 2.0).clamp(0.0, 1.0)
}

#[inline(always)]
fn high_gain(blend: PolyF32) -> PolyF32 {
    blend.clamp(0.0, 1.0)
}

#[derive(Clone, Debug)]
pub struct CombFilter {
    state: FilterState,
    sample_rate: f32,

    memory: Memory,

    feedback_style: FeedbackStyle,
    max_period: PolyF32,
    feedback: PolyF32,
    filter_coefficient: PolyF32,
    filter2_coefficient: PolyF32,
    low_gain: PolyF32,
    high_gain: PolyF32,
    scale: PolyF32,

    filter_midi_cutoff: PolyF32,
    filter2_midi_cutoff: PolyF32,
    feedback_filter: OnePoleFilter<Pass>,
    feedback_filter2: OnePoleFilter<Pass>,

    current_feedback: PolyF32,
    current_filter_coefficient: PolyF32,
    current_filter2_coefficient: PolyF32,
    current_scale: PolyF32,
    current_low_gain: PolyF32,
    current_high_gain: PolyF32,
}

impl CombFilter {
    pub fn new(size: usize) -> CombFilter {
        CombFilter {
            state: FilterState::default(),
            sample_rate: 44100.0,
            memory: Memory::new(size.max(MIN_PERIOD_SIZE)),
            feedback_style: FeedbackStyle::Comb,
            max_period: PolyF32::splat(MIN_PERIOD),
            feedback: PolyF32::ZERO,
            filter_coefficient: PolyF32::ZERO,
            filter2_coefficient: PolyF32::ZERO,
            low_gain: PolyF32::ZERO,
            high_gain: PolyF32::ZERO,
            scale: PolyF32::ZERO,
            filter_midi_cutoff: PolyF32::ZERO,
            filter2_midi_cutoff: PolyF32::ZERO,
            feedback_filter: OnePoleFilter::new(),
            feedback_filter2: OnePoleFilter::new(),
            current_feedback: PolyF32::ZERO,
            current_filter_coefficient: PolyF32::ZERO,
            current_filter2_coefficient: PolyF32::ZERO,
            current_scale: PolyF32::ZERO,
            current_low_gain: PolyF32::ZERO,
            current_high_gain: PolyF32::ZERO,
        }
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        let mut max_period = self.max_period.lane(0);
        for i in 1..LANES {
            max_period = max_period.max(self.max_period.lane(i));
        }

        let clear_samples =
            (self.memory.size() as i32 - 1).min(max_period as i32 + 1).max(0) as usize;
        self.memory.clear_memory(clear_samples, reset_mask);

        self.scale = reset_mask.select(PolyF32::ZERO, self.scale);
        self.low_gain = reset_mask.select(PolyF32::ZERO, self.low_gain);
        self.high_gain = reset_mask.select(PolyF32::ZERO, self.high_gain);

        self.feedback_filter.reset(reset_mask);
        self.feedback_filter2.reset(reset_mask);

        self.current_feedback = reset_mask.select(self.feedback, self.current_feedback);
        self.current_filter_coefficient =
            reset_mask.select(self.filter_coefficient, self.current_filter_coefficient);
        self.current_filter2_coefficient =
            reset_mask.select(self.filter2_coefficient, self.current_filter2_coefficient);
        self.current_scale = reset_mask.select(self.scale, self.current_scale);
        self.current_low_gain = reset_mask.select(self.low_gain, self.current_low_gain);
        self.current_high_gain = reset_mask.select(self.high_gain, self.current_high_gain);
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
    }

    /// Per-block parameter update (C++ `setupFilter` + ramp capture).
    pub fn setup(&mut self, filter_state: &FilterState, sample_rate: f32) {
        self.current_feedback = self.feedback;
        self.current_filter_coefficient = self.filter_coefficient;
        self.current_filter2_coefficient = self.filter2_coefficient;
        self.current_scale = self.scale;
        self.current_low_gain = self.low_gain;
        self.current_high_gain = self.high_gain;

        self.state = *filter_state;
        self.sample_rate = sample_rate;

        let style_index = filter_state.style.index();
        self.feedback_style = FeedbackStyle::from_style_index(style_index);
        let resonance = filter_state.resonance_percent.clamp(0.0, 1.0);

        if self.feedback_style == FeedbackStyle::Comb {
            self.feedback =
                interpolate(PolyF32::splat(-MAX_FEEDBACK), PolyF32::splat(MAX_FEEDBACK), resonance);
            self.feedback = self.feedback / (self.feedback.abs() + 0.00001).sqrt();
            self.scale = -self.feedback * self.feedback * INPUT_SCALE + 1.0;
        } else {
            self.feedback =
                interpolate(PolyF32::ZERO, PolyF32::splat(MAX_FEEDBACK), resonance);
            self.scale = PolyF32::ONE / (self.feedback + 1.0);
        }

        let midi_cutoff = filter_state.midi_cutoff;
        let min_nyquist = sample_rate * MIN_NYQUIST_MULT;

        let blend = filter_state.pass_blend;
        let min_cutoff = midi_cutoff - (4 * NOTES_PER_OCTAVE) as f32;

        let filter_style = CombFilterStyle::from_style_index(style_index);
        if filter_style == CombFilterStyle::BandSpread {
            let midi_blend_transpose = filter_state.transpose;
            let center_midi_cutoff = midi_cutoff + midi_blend_transpose;
            let midi_band_range =
                (blend * 0.5 * BAND_OCTAVE_RANGE + BAND_OCTAVE_MIN) * NOTES_PER_OCTAVE as f32;

            self.filter_midi_cutoff = center_midi_cutoff + midi_band_range;
            self.filter2_midi_cutoff = min_cutoff.max(center_midi_cutoff - midi_band_range);
            let mut filter1_cutoff = midi_note_to_frequency_precise(self.filter_midi_cutoff);
            let mut filter2_cutoff = midi_note_to_frequency_precise(self.filter2_midi_cutoff);

            filter1_cutoff = filter1_cutoff.clamp(1.0, sample_rate / 2.1);
            filter2_cutoff = filter2_cutoff.clamp(1.0, sample_rate / 2.1);
            self.filter_midi_cutoff = frequency_to_midi_note_precise(filter1_cutoff);
            self.filter2_midi_cutoff = frequency_to_midi_note_precise(filter2_cutoff);
            self.low_gain = filter2_cutoff / filter1_cutoff + 1.0;
            self.high_gain = PolyF32::ZERO;

            self.filter_coefficient =
                OnePoleFilter::<Pass>::compute_coefficient(filter1_cutoff, sample_rate);
            self.filter2_coefficient =
                OnePoleFilter::<Pass>::compute_coefficient(filter2_cutoff, sample_rate);
        } else {
            self.low_gain = low_gain(blend);
            self.high_gain = high_gain(blend);

            let midi_blend_transpose = filter_state.transpose;
            self.filter_midi_cutoff = midi_cutoff + midi_blend_transpose;
            self.filter2_midi_cutoff = min_cutoff;

            let mut filter_cutoff = midi_note_to_frequency_precise(self.filter_midi_cutoff);
            let mut filter2_cutoff = midi_note_to_frequency_precise(self.filter2_midi_cutoff);
            filter_cutoff = filter_cutoff.clamp(1.0, min_nyquist);
            filter2_cutoff = filter2_cutoff.clamp(1.0, min_nyquist);

            self.filter_coefficient =
                OnePoleFilter::<Pass>::compute_coefficient(filter_cutoff, sample_rate);
            self.filter2_coefficient =
                OnePoleFilter::<Pass>::compute_coefficient(filter2_cutoff, sample_rate);
        }
    }

    pub fn process(&mut self, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        self.process_inner(audio_in, None, audio_out);
    }

    pub fn process_modulated(
        &mut self,
        audio_in: &[PolyF32],
        midi_cutoff: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        assert_eq!(audio_in.len(), midi_cutoff.len());
        self.process_inner(audio_in, Some(midi_cutoff), audio_out);
    }

    /// Recomputes the period bounds from the block's cutoff (done at the top
    /// of the C++ `processFilter`). Runs inside `process`, before the loop.
    fn update_max_period(&mut self, midi_cutoff: Option<&[PolyF32]>, num_samples: usize) {
        let min_midi_cutoff = match midi_cutoff {
            Some(buffer) => buffer[0].min(buffer[num_samples - 1]),
            None => self.state.midi_cutoff,
        };
        let min_frequency = midi_note_to_frequency_precise(min_midi_cutoff);
        let min_nyquist = self.sample_rate * MIN_NYQUIST_MULT;
        self.max_period =
            PolyF32::splat(self.sample_rate) / min_frequency.clamp(1.0, min_nyquist);
        let mut min_period = PolyF32::splat(MIN_PERIOD);
        if self.feedback_style == FeedbackStyle::NegativeFlange {
            min_period *= 2.0;
        }
        let memory_max = self.memory.max_period() as f32 - 5.0;
        self.max_period = self.max_period.max(min_period).min(PolyF32::splat(memory_max));
    }

    fn process_inner(
        &mut self,
        audio_in: &[PolyF32],
        midi_cutoff: Option<&[PolyF32]>,
        audio_out: &mut [PolyF32],
    ) {
        let num_samples = audio_in.len();
        assert_eq!(num_samples, audio_out.len());
        assert!(num_samples > 0);

        self.update_max_period(midi_cutoff, num_samples);

        let mut current_feedback = self.current_feedback;
        let mut current_filter_coefficient = self.current_filter_coefficient;
        let mut current_filter2_coefficient = self.current_filter2_coefficient;
        let mut current_scale = self.current_scale;
        let mut current_low_gain = self.current_low_gain;
        let mut current_high_gain = self.current_high_gain;

        let tick_increment = 1.0 / num_samples as f32;
        let delta_feedback = (self.feedback - current_feedback) * tick_increment;
        let delta_coefficient =
            (self.filter_coefficient - current_filter_coefficient) * tick_increment;
        let delta_coefficient2 =
            (self.filter2_coefficient - current_filter2_coefficient) * tick_increment;
        let delta_scale = (self.scale - current_scale) * tick_increment;
        let delta_low_gain = (self.low_gain - current_low_gain) * tick_increment;
        let delta_high_gain = (self.high_gain - current_high_gain) * tick_increment;

        let base_midi = match midi_cutoff {
            Some(buffer) => buffer[num_samples - 1],
            None => self.state.midi_cutoff,
        };
        let base_frequency = midi_note_to_frequency_precise(base_midi);

        let mut min_period = PolyF32::splat(MIN_PERIOD);
        if self.feedback_style == FeedbackStyle::NegativeFlange {
            min_period *= 2.0;
        }
        let max_period = PolyF32::splat(self.memory.max_period() as f32 - 5.0);
        let sample_rate = PolyF32::splat(self.sample_rate);
        let style = self.feedback_style;

        for i in 0..num_samples {
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_offset = midi - base_midi;
            let frequency = base_frequency * math::midi_offset_to_ratio(midi_offset);
            let period = (sample_rate / frequency).max(min_period).min(max_period);

            current_feedback += delta_feedback;
            current_filter_coefficient += delta_coefficient;
            current_filter2_coefficient += delta_coefficient2;
            current_scale += delta_scale;
            current_low_gain += delta_low_gain;
            current_high_gain += delta_high_gain;

            audio_out[i] = match style {
                FeedbackStyle::Comb => self.tick_comb(
                    audio_in[i],
                    period,
                    current_feedback,
                    current_scale,
                    current_filter_coefficient,
                    current_filter2_coefficient,
                    current_low_gain,
                    current_high_gain,
                ),
                FeedbackStyle::PositiveFlange => self.tick_positive_flange(
                    audio_in[i],
                    period,
                    current_feedback,
                    current_scale,
                    current_filter_coefficient,
                    current_filter2_coefficient,
                    current_low_gain,
                    current_high_gain,
                ),
                FeedbackStyle::NegativeFlange => self.tick_negative_flange(
                    audio_in[i],
                    period,
                    current_feedback,
                    current_scale,
                    current_filter_coefficient,
                    current_filter2_coefficient,
                    current_low_gain,
                    current_high_gain,
                ),
            };
        }

        self.current_feedback = self.feedback;
        self.current_filter_coefficient = self.filter_coefficient;
        self.current_filter2_coefficient = self.filter2_coefficient;
        self.current_scale = self.scale;
        self.current_low_gain = self.low_gain;
        self.current_high_gain = self.high_gain;
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tick_comb(
        &mut self,
        audio_in: PolyF32,
        period: PolyF32,
        feedback: PolyF32,
        scale: PolyF32,
        filter_coefficient: PolyF32,
        filter2_coefficient: PolyF32,
        low_gain: PolyF32,
        high_gain: PolyF32,
    ) -> PolyF32 {
        let read = self.memory.get(period);
        let combine = (scale * audio_in).mul_add(read, feedback);

        let low_output = self.feedback_filter.tick_basic(combine, filter_coefficient);
        let high_output = combine - low_output;
        let stage1_output = (low_gain * low_output).mul_add(high_gain, high_output);
        let stage2_output = self.feedback_filter2.tick_basic(stage1_output, filter2_coefficient);
        let result = stage1_output - stage2_output;
        self.memory.push(math::hard_tanh(result));

        debug_assert!(result.is_finite());
        result
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tick_positive_flange(
        &mut self,
        audio_in: PolyF32,
        period: PolyF32,
        feedback: PolyF32,
        scale: PolyF32,
        filter_coefficient: PolyF32,
        filter2_coefficient: PolyF32,
        low_gain: PolyF32,
        high_gain: PolyF32,
    ) -> PolyF32 {
        let read = self.memory.get(period);
        let low_output = self.feedback_filter.tick_basic(read, filter_coefficient);
        let high_output = read - low_output;
        let stage1_output = (low_gain * low_output).mul_add(high_gain, high_output);
        let stage2_output = self.feedback_filter2.tick_basic(stage1_output, filter2_coefficient);
        let filter_output = stage1_output - stage2_output;
        debug_assert!(filter_output.is_finite());

        let scaled_input = audio_in * FLANGE_SCALE;
        self.memory.push(scaled_input + math::hard_tanh(filter_output * feedback));

        scaled_input * scale + filter_output
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tick_negative_flange(
        &mut self,
        audio_in: PolyF32,
        period: PolyF32,
        feedback: PolyF32,
        scale: PolyF32,
        filter_coefficient: PolyF32,
        filter2_coefficient: PolyF32,
        low_gain: PolyF32,
        high_gain: PolyF32,
    ) -> PolyF32 {
        let read = self.memory.get(period * 0.5);
        let low_output = self.feedback_filter.tick_basic(read, filter_coefficient);
        let high_output = read - low_output;
        let stage1_output = (low_gain * low_output).mul_add(high_gain, high_output);
        let stage2_output = self.feedback_filter2.tick_basic(stage1_output, filter2_coefficient);
        let filter_output = stage1_output - stage2_output;
        debug_assert!(filter_output.is_finite());

        let scaled_input = audio_in * FLANGE_SCALE;
        self.memory.push(scaled_input - math::hard_tanh(filter_output * feedback));

        scaled_input * scale - filter_output
    }

    pub fn drive(&self) -> PolyF32 {
        self.scale
    }

    pub fn resonance(&self) -> PolyF32 {
        self.feedback
    }

    pub fn low_amount(&self) -> PolyF32 {
        self.low_gain
    }

    pub fn high_amount(&self) -> PolyF32 {
        self.high_gain
    }

    pub fn filter_midi_cutoff(&self) -> PolyF32 {
        self.filter_midi_cutoff
    }

    pub fn filter2_midi_cutoff(&self) -> PolyF32 {
        self.filter2_midi_cutoff
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::filters::filter_state::FilterStyle;

    const SAMPLE_RATE: f32 = 48000.0;
    const BLOCK: usize = 128;

    fn comb_state(freq: f32, resonance: f32, style: FilterStyle) -> FilterState {
        let mut state = FilterState::default();
        state.midi_cutoff = frequency_to_midi_note_precise(PolyF32::splat(freq));
        state.resonance_percent = PolyF32::splat(resonance);
        state.set_drive_db(PolyF32::ZERO);
        state.set_pass_blend(PolyF32::ONE);
        state.style = style;
        state
    }

    fn run_sine(filter: &mut CombFilter, state: &FilterState, freq: f32, blocks: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut n = 0usize;
        let mut input = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..blocks {
            for value in input.iter_mut() {
                let phase = 2.0 * core::f32::consts::PI * freq * n as f32 / SAMPLE_RATE;
                *value = PolyF32::splat(0.2 * phase.sin());
                n += 1;
            }
            filter.setup(state, SAMPLE_RATE);
            filter.process(&input, &mut output);
            if block >= blocks / 2 {
                for value in &output {
                    assert!(value.is_finite());
                    sum += value.lane(0) * value.lane(0);
                    count += 1;
                }
            }
        }
        (sum / count as f32).sqrt()
    }

    #[test]
    fn comb_resonates_at_harmonics() {
        // High resonance comb tuned to 375 Hz: harmonics are reinforced,
        // half-harmonics land near notches.
        let f0 = 375.0; // 128-sample period at 48 kHz
        let state = comb_state(f0, 0.95, FilterStyle::TwelveDb);
        let mut filter = CombFilter::new(4096);
        let on_peak = run_sine(&mut filter, &state, f0, 60);
        let mut filter = CombFilter::new(4096);
        let off_peak = run_sine(&mut filter, &state, f0 * 1.5, 60);
        assert!(on_peak > 2.0 * off_peak, "peak {on_peak} vs notch {off_peak}");
    }

    #[test]
    fn flange_styles_are_finite_and_nonzero() {
        for style in [FilterStyle::TwentyFourDb, FilterStyle::NotchPassSwap] {
            let state = comb_state(500.0, 0.7, style);
            let mut filter = CombFilter::new(4096);
            let rms = run_sine(&mut filter, &state, 440.0, 20);
            assert!(rms.is_finite() && rms > 0.0, "style {style:?} rms {rms}");
        }
    }

    #[test]
    fn band_spread_style_is_finite() {
        let state = comb_state(500.0, 0.5, FilterStyle::DualNotchBand);
        let mut filter = CombFilter::new(4096);
        let rms = run_sine(&mut filter, &state, 440.0, 20);
        assert!(rms.is_finite() && rms > 0.0);
    }

    #[test]
    fn reset_clears_delay_memory() {
        let state = comb_state(375.0, 0.9, FilterStyle::TwelveDb);
        let mut filter = CombFilter::new(4096);
        let _ = run_sine(&mut filter, &state, 375.0, 8);
        filter.setup(&state, SAMPLE_RATE);
        filter.reset(PolyMask::all_on());
        let silence = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::splat(1.0); BLOCK];
        filter.process(&silence, &mut output);
        for value in &output {
            assert!(value.lane(0).abs() < 1e-6, "delay ring-out {}", value.lane(0));
        }
    }

    #[test]
    fn style_mapping_matches_reference() {
        assert_eq!(FeedbackStyle::from_style_index(0), FeedbackStyle::Comb);
        assert_eq!(FeedbackStyle::from_style_index(1), FeedbackStyle::PositiveFlange);
        assert_eq!(FeedbackStyle::from_style_index(2), FeedbackStyle::NegativeFlange);
        assert_eq!(FeedbackStyle::from_style_index(4), FeedbackStyle::PositiveFlange);
        assert_eq!(CombFilterStyle::from_style_index(2), CombFilterStyle::LowHighBlend);
        assert_eq!(CombFilterStyle::from_style_index(3), CombFilterStyle::BandSpread);
    }
}
