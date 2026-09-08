//! Diode ladder filter (port of `diode_filter.{h,cpp}`).

use spinwave_poly::utils::interpolate;
use spinwave_poly::{math, PolyF32, PolyMask};

use super::filter_state::{
    coefficient_lookup, midi_note_to_frequency_precise, FilterState, FilterStyle,
};
use super::one_pole::{HardClipSat, OnePoleFilter, Pass, TanhSat};

pub const MIN_RESONANCE: f32 = 0.7;
pub const MAX_RESONANCE: f32 = 17.0;
pub const MIN_CUTOFF: f32 = 1.0;
pub const HIGH_PASS_FREQUENCY: f32 = 20.0;

#[derive(Clone, Debug, Default)]
pub struct DiodeFilter {
    state: FilterState,
    sample_rate: f32,

    resonance: PolyF32,
    drive: PolyF32,
    post_multiply: PolyF32,
    high_pass_ratio: PolyF32,
    high_pass_amount: PolyF32,

    current_resonance: PolyF32,
    current_drive: PolyF32,
    current_post_multiply: PolyF32,
    current_high_pass_ratio: PolyF32,
    current_high_pass_amount: PolyF32,

    high_pass_1: OnePoleFilter<Pass>,
    high_pass_2: OnePoleFilter<Pass>,
    high_pass_feedback: OnePoleFilter<Pass>,
    stage1: OnePoleFilter<TanhSat>,
    stage2: OnePoleFilter<Pass>,
    stage3: OnePoleFilter<Pass>,
    stage4: OnePoleFilter<HardClipSat>,
}

impl DiodeFilter {
    pub fn new() -> DiodeFilter {
        let mut filter = DiodeFilter { sample_rate: 44100.0, ..Default::default() };
        filter.hard_reset();
        filter
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.high_pass_1.reset(reset_mask);
        self.high_pass_2.reset(reset_mask);
        self.stage1.reset(reset_mask);
        self.stage2.reset(reset_mask);
        self.stage3.reset(reset_mask);
        self.stage4.reset(reset_mask);

        self.current_resonance = reset_mask.select(self.resonance, self.current_resonance);
        self.current_drive = reset_mask.select(self.drive, self.current_drive);
        self.current_post_multiply =
            reset_mask.select(self.post_multiply, self.current_post_multiply);
        self.current_high_pass_ratio =
            reset_mask.select(self.high_pass_ratio, self.current_high_pass_ratio);
        self.current_high_pass_amount =
            reset_mask.select(self.high_pass_amount, self.current_high_pass_amount);
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
        self.resonance = PolyF32::ZERO;
        self.drive = PolyF32::ZERO;
        self.post_multiply = PolyF32::ZERO;
        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_high_pass_ratio = self.high_pass_ratio;
        self.current_high_pass_amount = self.high_pass_amount;
    }

    /// Per-block parameter update (C++ `setupFilter` + ramp capture).
    pub fn setup(&mut self, filter_state: &FilterState, sample_rate: f32) {
        const HIGH_PASS_START: f32 = -9.0;
        const HIGH_PASS_END: f32 = -1.0;
        const HIGH_PASS_RANGE: f32 = HIGH_PASS_END - HIGH_PASS_START;

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_high_pass_ratio = self.high_pass_ratio;
        self.current_high_pass_amount = self.high_pass_amount;

        self.state = *filter_state;
        self.sample_rate = sample_rate;

        let mut resonance_percent = filter_state.resonance_percent.clamp(0.0, 1.0);
        resonance_percent = resonance_percent * (resonance_percent * resonance_percent);
        self.resonance = interpolate(
            PolyF32::splat(MIN_RESONANCE),
            PolyF32::splat(MAX_RESONANCE),
            resonance_percent,
        );
        self.drive = (self.resonance * 0.5 + 1.0) * filter_state.drive;
        self.post_multiply = PolyF32::ONE / filter_state.drive.sqrt();

        let blend_amount = filter_state.pass_blend * 0.5;

        if filter_state.style == FilterStyle::TwelveDb {
            self.high_pass_ratio = math::exp2(PolyF32::splat(HIGH_PASS_END));
            self.high_pass_amount = blend_amount * blend_amount;
        } else {
            self.high_pass_ratio =
                math::exp2(blend_amount * HIGH_PASS_RANGE + HIGH_PASS_START);
            self.high_pass_amount = PolyF32::ONE;
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

    fn process_inner(
        &mut self,
        audio_in: &[PolyF32],
        midi_cutoff: Option<&[PolyF32]>,
        audio_out: &mut [PolyF32],
    ) {
        let num_samples = audio_in.len();
        assert_eq!(num_samples, audio_out.len());
        assert!(num_samples > 0);

        let mut current_resonance = self.current_resonance;
        let mut current_drive = self.current_drive;
        let mut current_post_multiply = self.current_post_multiply;
        let mut current_high_pass_ratio = self.current_high_pass_ratio;
        let mut current_high_pass_amount = self.current_high_pass_amount;

        let tick_increment = 1.0 / num_samples as f32;
        let delta_resonance = (self.resonance - current_resonance) * tick_increment;
        let delta_drive = (self.drive - current_drive) * tick_increment;
        let delta_post_multiply = (self.post_multiply - current_post_multiply) * tick_increment;
        let delta_high_pass_ratio =
            (self.high_pass_ratio - current_high_pass_ratio) * tick_increment;
        let delta_high_pass_amount =
            (self.high_pass_amount - current_high_pass_amount) * tick_increment;

        let lookup = coefficient_lookup();
        let base_midi = match midi_cutoff {
            Some(buffer) => buffer[num_samples - 1],
            None => self.state.midi_cutoff,
        };
        let base_frequency =
            midi_note_to_frequency_precise(base_midi) * (1.0 / self.sample_rate);
        let high_pass_frequency_ratio =
            PolyF32::splat(HIGH_PASS_FREQUENCY * (1.0 / self.sample_rate));
        let high_pass_feedback_coefficient = lookup.cubic_lookup(high_pass_frequency_ratio);

        for i in 0..num_samples {
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = lookup.cubic_lookup(frequency);

            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_post_multiply += delta_post_multiply;
            current_high_pass_ratio += delta_high_pass_ratio;
            current_high_pass_amount += delta_high_pass_amount;

            self.tick(
                audio_in[i],
                coefficient,
                current_high_pass_ratio,
                current_high_pass_amount,
                high_pass_feedback_coefficient,
                current_resonance,
                current_drive,
            );
            audio_out[i] = self.stage4.current_state() * current_post_multiply;
        }

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_high_pass_ratio = self.high_pass_ratio;
        self.current_high_pass_amount = self.high_pass_amount;
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tick(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        high_pass_ratio: PolyF32,
        high_pass_amount: PolyF32,
        high_pass_feedback_coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
    ) {
        let high_pass_coefficient = coefficient * high_pass_ratio;
        let high_pass_coefficient2 = high_pass_coefficient * 2.0;
        let high_pass_coefficient_squared = high_pass_coefficient * high_pass_coefficient;
        let high_pass_coefficient_diff = high_pass_coefficient_squared - high_pass_coefficient;
        let high_pass_feedback_mult =
            high_pass_coefficient2 - high_pass_coefficient_squared - 1.0;
        let high_pass_normalizer = PolyF32::ONE / (high_pass_coefficient_diff + 1.0);

        let high_pass_mult_stage2 = -high_pass_coefficient + 1.0;
        let high_pass_feedback = high_pass_feedback_mult * self.high_pass_1.next_state()
            + high_pass_mult_stage2 * self.high_pass_2.next_state();

        let high_pass_input = (audio_in - high_pass_feedback) * high_pass_normalizer;

        let high_pass_1_out = self.high_pass_1.tick_basic(high_pass_input, high_pass_coefficient);
        let high_pass_2_out = self.high_pass_2.tick_basic(high_pass_1_out, high_pass_coefficient);
        let mut high_pass_out = high_pass_input - high_pass_1_out * 2.0 + high_pass_2_out;
        high_pass_out = interpolate(audio_in, high_pass_out, high_pass_amount);

        let filter_state = self.stage4.next_sat_state();
        let filter_input = (drive * high_pass_out - resonance * filter_state) * 0.5;
        let sat_input = math::tanh(filter_input);

        let feedback_input = sat_input + self.stage2.next_sat_state();
        let feedback =
            self.high_pass_feedback.tick_basic(feedback_input, high_pass_feedback_coefficient);
        self.stage1.tick(feedback_input - feedback, coefficient);
        self.stage2
            .tick((self.stage1.current_state() + self.stage3.next_sat_state()) * 0.5, coefficient);
        self.stage3
            .tick((self.stage2.current_state() + self.stage4.next_sat_state()) * 0.5, coefficient);
        self.stage4.tick(self.stage3.current_state(), coefficient);
    }

    pub fn resonance(&self) -> PolyF32 {
        self.resonance
    }

    pub fn drive(&self) -> PolyF32 {
        self.drive
    }

    pub fn high_pass_ratio(&self) -> PolyF32 {
        self.high_pass_ratio
    }

    pub fn high_pass_amount(&self) -> PolyF32 {
        self.high_pass_amount
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::filters::filter_state::frequency_to_midi_note_precise;

    const SAMPLE_RATE: f32 = 48000.0;
    const BLOCK: usize = 128;

    fn state_for(freq: f32) -> FilterState {
        let mut state = FilterState::default();
        state.midi_cutoff = frequency_to_midi_note_precise(PolyF32::splat(freq));
        state.resonance_percent = PolyF32::splat(0.3);
        state.set_drive_db(PolyF32::ZERO);
        state.set_pass_blend(PolyF32::ZERO);
        state.style = FilterStyle::TwelveDb;
        state
    }

    fn run_sine(filter: &mut DiodeFilter, state: &FilterState, freq: f32, blocks: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut n = 0usize;
        let mut input = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..blocks {
            for value in input.iter_mut() {
                let phase = 2.0 * core::f32::consts::PI * freq * n as f32 / SAMPLE_RATE;
                *value = PolyF32::splat(0.5 * phase.sin());
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
    fn low_pass_selectivity() {
        let state = state_for(1000.0);
        let mut filter = DiodeFilter::new();
        let low = run_sine(&mut filter, &state, 100.0, 40);
        let mut filter = DiodeFilter::new();
        let high = run_sine(&mut filter, &state, 8000.0, 40);
        assert!(low > 4.0 * high, "low rms {low} vs high rms {high}");
    }

    #[test]
    fn high_pass_blend_reduces_lows() {
        let mut state = state_for(4000.0);
        state.style = FilterStyle::TwentyFourDb;
        // Full pass_blend raises the internal high-pass cutoff.
        state.set_pass_blend(PolyF32::splat(2.0));
        let mut filter = DiodeFilter::new();
        let low_hp = run_sine(&mut filter, &state, 40.0, 40);

        state.set_pass_blend(PolyF32::ZERO);
        let mut filter = DiodeFilter::new();
        let low_flat = run_sine(&mut filter, &state, 40.0, 40);
        assert!(low_flat > 1.5 * low_hp, "hp blend {low_hp} vs flat {low_flat}");
    }

    #[test]
    fn reset_clears_state() {
        let state = state_for(2000.0);
        let mut filter = DiodeFilter::new();
        let _ = run_sine(&mut filter, &state, 500.0, 4);
        filter.setup(&state, SAMPLE_RATE);
        filter.reset(PolyMask::all_on());
        let silence = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::splat(1.0); BLOCK];
        filter.process(&silence, &mut output);
        // Not exactly zero: like the reference, reset() intentionally leaves
        // the feedback high-pass one-pole untouched, so a small residue decays.
        for value in &output {
            assert!(value.lane(0).abs() < 0.01, "ring-out {}", value.lane(0));
        }
        let first_residue = output[0].lane(0).abs();
        for _ in 0..20 {
            filter.process(&silence, &mut output);
        }
        let late_residue = output[BLOCK - 1].lane(0).abs();
        assert!(
            late_residue < first_residue.max(1e-6),
            "residue grew: {first_residue} -> {late_residue}"
        );
    }
}
