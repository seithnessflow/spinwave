//! "Dirty" analog-style filter (port of `dirty_filter.{h,cpp}`).

use spinwave_poly::utils::interpolate;
use spinwave_poly::{math, PolyF32, PolyMask};

use super::filter_state::{
    coefficient_lookup, midi_note_to_frequency_precise, FilterState, FilterStyle,
};
use super::one_pole::{OnePoleFilter, Pass, QuickTanhSat};

pub const MIN_RESONANCE: f32 = 0.1;
pub const MAX_RESONANCE: f32 = 2.15;
pub const SATURATION_BOOST: f32 = 1.4;
pub const MAX_VISIBLE_RESONANCE: f32 = 2.0;
pub const DRIVE_RESONANCE_BOOST: f32 = 0.05;
pub const MIN_CUTOFF: f32 = 1.0;
pub const MIN_DRIVE: f32 = 0.1;
pub const FLAT_RESONANCE: f32 = 1.0;

#[inline(always)]
fn tune_resonance(resonance: PolyF32, coefficient: PolyF32) -> PolyF32 {
    resonance / PolyF32::ONE.max(coefficient * 0.25 + 0.97)
}

#[derive(Clone, Debug, Default)]
pub struct DirtyFilter {
    state: FilterState,
    sample_rate: f32,

    coefficient: PolyF32,
    resonance: PolyF32,
    drive: PolyF32,
    drive_boost: PolyF32,
    drive_blend: PolyF32,
    drive_mult: PolyF32,

    low_pass_amount: PolyF32,
    band_pass_amount: PolyF32,
    high_pass_amount: PolyF32,

    current_resonance: PolyF32,
    current_drive: PolyF32,
    current_drive_boost: PolyF32,
    current_drive_blend: PolyF32,
    current_drive_mult: PolyF32,
    current_low: PolyF32,
    current_band: PolyF32,
    current_high: PolyF32,

    pre_stage1: OnePoleFilter<Pass>,
    pre_stage2: OnePoleFilter<Pass>,
    stage1: OnePoleFilter<Pass>,
    stage2: OnePoleFilter<Pass>,
    stage3: OnePoleFilter<QuickTanhSat>,
    stage4: OnePoleFilter<QuickTanhSat>,
}

impl DirtyFilter {
    pub fn new() -> DirtyFilter {
        let mut filter = DirtyFilter { sample_rate: 44100.0, ..Default::default() };
        filter.hard_reset();
        filter
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.pre_stage1.reset(reset_mask);
        self.pre_stage2.reset(reset_mask);
        self.stage1.reset(reset_mask);
        self.stage2.reset(reset_mask);
        self.stage3.reset(reset_mask);
        self.stage4.reset(reset_mask);

        self.current_resonance = reset_mask.select(self.resonance, self.current_resonance);
        self.current_drive = reset_mask.select(self.drive, self.current_drive);
        self.current_drive_boost = reset_mask.select(self.drive_boost, self.current_drive_boost);
        self.current_drive_blend = reset_mask.select(self.drive_blend, self.current_drive_blend);
        self.current_drive_mult = reset_mask.select(self.drive_mult, self.current_drive_mult);
        self.current_low = reset_mask.select(self.low_pass_amount, self.current_low);
        self.current_band = reset_mask.select(self.band_pass_amount, self.current_band);
        self.current_high = reset_mask.select(self.high_pass_amount, self.current_high);
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
        self.coefficient = PolyF32::splat(0.1);
        self.resonance = PolyF32::ZERO;
        self.drive = PolyF32::ZERO;
        self.drive_boost = PolyF32::ZERO;
        self.drive_blend = PolyF32::ZERO;
        self.drive_mult = PolyF32::ZERO;
        self.low_pass_amount = PolyF32::ZERO;
        self.band_pass_amount = PolyF32::ZERO;
        self.high_pass_amount = PolyF32::ZERO;

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_drive_boost = self.drive_boost;
        self.current_drive_blend = self.drive_blend;
        self.current_drive_mult = self.drive_mult;
        self.current_low = self.low_pass_amount;
        self.current_band = self.band_pass_amount;
        self.current_high = self.high_pass_amount;
    }

    /// Per-block parameter update (C++ `setupFilter` + ramp capture).
    pub fn setup(&mut self, filter_state: &FilterState, sample_rate: f32) {
        const MAX_MIDI: f32 = 150.0;

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_drive_boost = self.drive_boost;
        self.current_drive_blend = self.drive_blend;
        self.current_drive_mult = self.drive_mult;
        self.current_low = self.low_pass_amount;
        self.current_band = self.band_pass_amount;
        self.current_high = self.high_pass_amount;

        self.state = *filter_state;
        self.sample_rate = sample_rate;

        let cutoff = filter_state.midi_cutoff.clamp(0.0, MAX_MIDI);
        let base_frequency = midi_note_to_frequency_precise(cutoff) * (1.0 / sample_rate);
        self.coefficient = coefficient_lookup().cubic_lookup(base_frequency);

        self.resonance = filter_state.resonance_percent.clamp(0.0, 1.0).sqrt();
        self.drive = (filter_state.drive - 1.0) * 2.0 + 1.0;
        self.drive_boost = filter_state.drive_percent * DRIVE_RESONANCE_BOOST;

        self.drive_blend = PolyF32::ONE;
        self.drive_mult = PolyF32::ONE;

        let blend = (filter_state.pass_blend - 1.0).clamp(-1.0, 1.0);
        match filter_state.style {
            FilterStyle::DualNotchBand => {
                let t = blend * 0.5 + 0.5;
                self.drive_blend = (-blend + 1.0).min(PolyF32::ONE);
                self.drive_mult = -t + 2.0;

                self.low_pass_amount = t;
                self.band_pass_amount = PolyF32::ZERO;
                self.high_pass_amount = PolyF32::ONE;
            }
            FilterStyle::NotchPassSwap => {
                self.drive_blend = blend.abs();

                self.low_pass_amount = (-blend + 1.0).min(PolyF32::ONE);
                self.band_pass_amount = PolyF32::ZERO;
                self.high_pass_amount = (blend + 1.0).min(PolyF32::ONE);
            }
            FilterStyle::BandPeakNotch => {
                self.drive_blend = (-blend + 1.0).min(PolyF32::ONE);

                let drive_inv_t = -self.drive_blend + 1.0;
                let mult = ((drive_inv_t * drive_inv_t) * 0.5 + 0.5).sqrt();
                let peak_band_value = -(-blend).max(PolyF32::ZERO);
                self.low_pass_amount = mult * (peak_band_value + 1.0);
                self.band_pass_amount = mult * (peak_band_value - blend + 1.0) * 2.0;
                self.high_pass_amount = self.low_pass_amount;
            }
            _ => {
                self.band_pass_amount = (-blend * blend + 1.0).sqrt();
                let blend_mask = blend.lt(PolyF32::ZERO);
                self.low_pass_amount = (-blend) & blend_mask;
                self.high_pass_amount = blend & !blend_mask;
            }
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
        let mut current_drive_boost = self.current_drive_boost;
        let mut current_drive_blend = self.current_drive_blend;
        let mut current_drive_mult = self.current_drive_mult;
        let mut current_low = self.current_low;
        let mut current_band = self.current_band;
        let mut current_high = self.current_high;

        let tick_increment = 1.0 / num_samples as f32;
        let delta_resonance = (self.resonance - current_resonance) * tick_increment;
        let delta_drive = (self.drive - current_drive) * tick_increment;
        let delta_drive_boost = (self.drive_boost - current_drive_boost) * tick_increment;
        let delta_drive_blend = (self.drive_blend - current_drive_blend) * tick_increment;
        let delta_drive_mult = (self.drive_mult - current_drive_mult) * tick_increment;
        let delta_low = (self.low_pass_amount - current_low) * tick_increment;
        let delta_band = (self.band_pass_amount - current_band) * tick_increment;
        let delta_high = (self.high_pass_amount - current_high) * tick_increment;

        let lookup = coefficient_lookup();
        let base_midi = match midi_cutoff {
            Some(buffer) => buffer[num_samples - 1],
            None => self.state.midi_cutoff,
        };
        let base_frequency =
            midi_note_to_frequency_precise(base_midi) * (1.0 / self.sample_rate);
        let style = self.state.style;

        for i in 0..num_samples {
            current_drive_boost += delta_drive_boost;
            current_resonance += delta_resonance;

            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = lookup.cubic_lookup(frequency);

            let coefficient_squared = coefficient * coefficient;
            let coefficient2 = coefficient * 2.0;
            let resonance_in = tune_resonance(current_resonance, coefficient2).clamp(0.0, 1.0);
            let resonance = interpolate(
                PolyF32::splat(MIN_RESONANCE),
                PolyF32::splat(MAX_RESONANCE),
                resonance_in,
            ) + current_drive_boost;
            let resonance_squared = resonance * resonance;

            let normalizer = PolyF32::splat(SATURATION_BOOST) / (resonance_squared + 1.0);
            let coefficient_diff = coefficient_squared - coefficient;

            current_drive += delta_drive;
            current_drive_blend += delta_drive_blend;
            current_drive_mult += delta_drive_mult;

            let scaled_drive = PolyF32::splat(MIN_DRIVE).max(current_drive)
                / (resonance_squared * 0.5 + 1.0);

            current_low += delta_low;
            current_band += delta_band;
            current_high += delta_high;

            match style {
                FilterStyle::TwelveDb => {
                    let compute = -resonance * (coefficient - coefficient_squared) + 1.0;
                    let feed_mult = PolyF32::ONE / (compute * (coefficient + 1.0));
                    let drive = interpolate(current_drive, scaled_drive, current_drive_blend);
                    audio_out[i] = self.tick(
                        audio_in[i],
                        coefficient,
                        resonance,
                        drive,
                        feed_mult,
                        normalizer,
                        current_low,
                        current_band,
                        current_high,
                    );
                }
                FilterStyle::DualNotchBand => {
                    let compute = resonance * coefficient_diff + 1.0;
                    let feed_mult = PolyF32::ONE / (compute * (coefficient + 1.0));
                    let pre_feedback = coefficient2 - coefficient_squared - 1.0;
                    let pre_normalizer =
                        PolyF32::ONE / (coefficient_diff * FLAT_RESONANCE + 1.0);
                    let drive = interpolate(
                        current_drive,
                        scaled_drive * current_drive_mult,
                        current_drive_blend,
                    );
                    audio_out[i] = self.tick_dual(
                        audio_in[i],
                        coefficient,
                        resonance,
                        drive,
                        feed_mult,
                        normalizer,
                        pre_feedback,
                        pre_normalizer,
                        current_low,
                        current_high,
                    );
                }
                _ => {
                    let compute = resonance * coefficient_diff + 1.0;
                    let feed_mult = PolyF32::ONE / (compute * (coefficient + 1.0));
                    let pre_feedback = coefficient2 - coefficient_squared - 1.0;
                    let pre_normalizer =
                        PolyF32::ONE / (coefficient_diff * FLAT_RESONANCE + 1.0);
                    let drive = interpolate(current_drive, scaled_drive, current_drive_blend);
                    audio_out[i] = self.tick24(
                        audio_in[i],
                        coefficient,
                        resonance,
                        drive,
                        feed_mult,
                        normalizer,
                        pre_feedback,
                        pre_normalizer,
                        current_low,
                        current_band,
                        current_high,
                    );
                }
            }
        }

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_drive_boost = self.drive_boost;
        self.current_drive_blend = self.drive_blend;
        self.current_drive_mult = self.drive_mult;
        self.current_low = self.low_pass_amount;
        self.current_band = self.band_pass_amount;
        self.current_high = self.high_pass_amount;
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tick24(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        feed_mult: PolyF32,
        normalizer: PolyF32,
        pre_feedback_mult: PolyF32,
        pre_normalizer: PolyF32,
        low: PolyF32,
        band: PolyF32,
        high: PolyF32,
    ) -> PolyF32 {
        let mult_stage2 = -coefficient + 1.0;
        // Faithful to the reference: the pre stages are only ever ticked with
        // `tick_basic`, so their saturated state stays zero and this feedback
        // path is silent. Kept for bit-compatibility.
        let mut feedback = pre_feedback_mult * self.pre_stage1.next_sat_state()
            + mult_stage2 * self.pre_stage2.next_sat_state();

        feedback *= FLAT_RESONANCE;
        let stage1_input = (audio_in - feedback) * pre_normalizer;

        let stage1_out = self.pre_stage1.tick_basic(stage1_input, coefficient);
        let stage2_out = self.pre_stage2.tick_basic(stage1_out, coefficient);

        let band_pass = stage1_out - stage2_out;
        let high_pass = stage1_input - stage1_out - band_pass;
        let pre_out = band * band_pass + high * high_pass + low * stage2_out;

        self.tick(pre_out, coefficient, resonance, drive, feed_mult, normalizer, low, band, high)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tick_dual(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        feed_mult: PolyF32,
        normalizer: PolyF32,
        pre_feedback_mult: PolyF32,
        pre_normalizer: PolyF32,
        low: PolyF32,
        high: PolyF32,
    ) -> PolyF32 {
        let mult_stage2 = -coefficient + 1.0;
        let mut feedback = pre_feedback_mult * self.pre_stage1.next_sat_state()
            + mult_stage2 * self.pre_stage2.next_sat_state();

        feedback *= FLAT_RESONANCE;
        let stage1_input = (audio_in - feedback) * pre_normalizer;

        let stage1_out = self.pre_stage1.tick_basic(stage1_input, coefficient);
        let stage2_out = self.pre_stage2.tick_basic(stage1_out, coefficient);

        let band_pass = stage1_out - stage2_out;
        let high_pass = stage1_input - stage1_out - band_pass;

        let pre_out = low * high_pass + high * stage2_out;

        self.tick(
            pre_out,
            coefficient,
            resonance,
            drive,
            feed_mult,
            normalizer,
            low,
            PolyF32::ZERO,
            high,
        )
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn tick(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        feed_mult: PolyF32,
        normalizer: PolyF32,
        low: PolyF32,
        band: PolyF32,
        high: PolyF32,
    ) -> PolyF32 {
        let stage1_in = normalizer * audio_in;
        let stage1_out = self.stage1.tick_basic(stage1_in, coefficient);
        let stage2_out = self.stage2.tick_basic(stage1_out, coefficient);

        let band_pass = stage1_out - stage2_out;
        let high_pass = stage1_in - stage1_out - band_pass;
        let pass_output = (low * stage2_out).mul_add(band, band_pass).mul_add(high, high_pass);

        let feedback = self.stage4.next_sat_state()
            + pass_output.mul_add(coefficient, pass_output - self.stage3.next_sat_state());

        let loop_input =
            math::tanh((drive * pass_output).mul_add(resonance, feed_mult * feedback));

        let stage3_out = self.stage3.tick(loop_input, coefficient);

        let stage4_in = loop_input - stage3_out;
        self.stage4.tick(stage4_in, coefficient);

        loop_input * (1.0 / SATURATION_BOOST)
    }

    // -- getters mirroring the C++ display helpers ---------------------------

    pub fn resonance(&self) -> PolyF32 {
        let resonance_in =
            tune_resonance(self.resonance, self.coefficient * 2.0).clamp(0.0, 1.0);
        interpolate(
            PolyF32::splat(MIN_RESONANCE),
            PolyF32::splat(MAX_RESONANCE),
            resonance_in,
        ) + self.drive_boost
    }

    pub fn drive(&self) -> PolyF32 {
        let resonance = self.resonance();
        let scaled_drive =
            PolyF32::splat(MIN_DRIVE).max(self.drive) / (resonance * resonance * 0.5 + 1.0);
        interpolate(self.drive, scaled_drive, self.drive_blend)
    }

    pub fn low_amount(&self) -> PolyF32 {
        self.low_pass_amount
    }

    pub fn band_amount(&self) -> PolyF32 {
        self.band_pass_amount
    }

    pub fn high_amount(&self) -> PolyF32 {
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

    fn state_for(freq: f32, style: FilterStyle, pass_blend: f32) -> FilterState {
        let mut state = FilterState::default();
        state.midi_cutoff = frequency_to_midi_note_precise(PolyF32::splat(freq));
        state.resonance_percent = PolyF32::splat(0.3);
        state.set_drive_db(PolyF32::ZERO);
        state.set_pass_blend(PolyF32::splat(pass_blend));
        state.style = style;
        state
    }

    fn run_sine(filter: &mut DirtyFilter, state: &FilterState, freq: f32, blocks: usize) -> f32 {
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
    fn low_pass_12_selectivity() {
        let state = state_for(1000.0, FilterStyle::TwelveDb, 0.0);
        let mut filter = DirtyFilter::new();
        let low = run_sine(&mut filter, &state, 100.0, 40);
        let mut filter = DirtyFilter::new();
        let high = run_sine(&mut filter, &state, 8000.0, 40);
        assert!(low > 3.0 * high, "low rms {low} vs high rms {high}");
    }

    #[test]
    fn all_styles_are_finite_and_nonzero() {
        for style in [
            FilterStyle::TwelveDb,
            FilterStyle::TwentyFourDb,
            FilterStyle::NotchPassSwap,
            FilterStyle::DualNotchBand,
            FilterStyle::BandPeakNotch,
        ] {
            let state = state_for(1000.0, style, 0.6);
            let mut filter = DirtyFilter::new();
            let rms = run_sine(&mut filter, &state, 440.0, 10);
            assert!(rms.is_finite() && rms > 0.0, "style {style:?} rms {rms}");
        }
    }

    #[test]
    fn reset_clears_state() {
        let state = state_for(2000.0, FilterStyle::TwelveDb, 0.0);
        let mut filter = DirtyFilter::new();
        let _ = run_sine(&mut filter, &state, 500.0, 4);
        filter.setup(&state, SAMPLE_RATE);
        filter.reset(PolyMask::all_on());
        let silence = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::splat(1.0); BLOCK];
        filter.process(&silence, &mut output);
        for value in &output {
            assert_eq!(value.lane(0), 0.0);
        }
    }
}
