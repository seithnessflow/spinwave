//! Sallen-Key analog-style filter (port of `sallen_key_filter.{h,cpp}`).

use spinwave_poly::constants::MIN_NYQUIST_MULT;
use spinwave_poly::utils::interpolate;
use spinwave_poly::{math, PolyF32, PolyMask};

use super::filter_state::{
    coefficient_lookup, midi_note_to_frequency_precise, FilterState, FilterStyle,
};
use super::one_pole::{OnePoleFilter, Pass};

pub const MIN_RESONANCE: f32 = 0.0;
pub const MAX_RESONANCE: f32 = 2.15;
pub const DRIVE_RESONANCE_BOOST: f32 = 1.1;
pub const MAX_VISIBLE_RESONANCE: f32 = 2.0;
pub const MIN_CUTOFF: f32 = 1.0;

#[inline(always)]
fn tune_resonance(resonance: PolyF32, coefficient: PolyF32) -> PolyF32 {
    resonance / PolyF32::ONE.max(coefficient * 0.09 + 0.97)
}

#[derive(Clone, Debug, Default)]
pub struct SallenKeyFilter {
    state: FilterState,
    sample_rate: f32,

    cutoff: PolyF32,
    resonance: PolyF32,
    drive: PolyF32,
    post_multiply: PolyF32,
    low_pass_amount: PolyF32,
    band_pass_amount: PolyF32,
    high_pass_amount: PolyF32,

    current_resonance: PolyF32,
    current_drive: PolyF32,
    current_post_multiply: PolyF32,
    current_low: PolyF32,
    current_band: PolyF32,
    current_high: PolyF32,

    stage1_input: PolyF32,

    pre_stage1: OnePoleFilter<Pass>,
    pre_stage2: OnePoleFilter<Pass>,
    stage1: OnePoleFilter<Pass>,
    stage2: OnePoleFilter<Pass>,
}

impl SallenKeyFilter {
    pub fn new() -> SallenKeyFilter {
        let mut filter = SallenKeyFilter { sample_rate: 44100.0, ..Default::default() };
        filter.hard_reset();
        filter
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.stage1_input = reset_mask.select(PolyF32::ZERO, self.stage1_input);

        self.pre_stage1.reset(reset_mask);
        self.pre_stage2.reset(reset_mask);
        self.stage1.reset(reset_mask);
        self.stage2.reset(reset_mask);

        self.current_resonance = reset_mask.select(self.resonance, self.current_resonance);
        self.current_drive = reset_mask.select(self.drive, self.current_drive);
        self.current_post_multiply =
            reset_mask.select(self.post_multiply, self.current_post_multiply);
        self.current_low = reset_mask.select(self.low_pass_amount, self.current_low);
        self.current_band = reset_mask.select(self.band_pass_amount, self.current_band);
        self.current_high = reset_mask.select(self.high_pass_amount, self.current_high);
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
        self.resonance = PolyF32::ZERO;
        self.drive = PolyF32::ZERO;
        self.post_multiply = PolyF32::ZERO;
        self.low_pass_amount = PolyF32::ZERO;
        self.band_pass_amount = PolyF32::ZERO;
        self.high_pass_amount = PolyF32::ZERO;

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_low = self.low_pass_amount;
        self.current_band = self.band_pass_amount;
        self.current_high = self.high_pass_amount;
    }

    /// Per-block parameter update (C++ `setupFilter` + ramp capture).
    pub fn setup(&mut self, filter_state: &FilterState, sample_rate: f32) {
        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_low = self.low_pass_amount;
        self.current_band = self.band_pass_amount;
        self.current_high = self.high_pass_amount;

        self.state = *filter_state;
        self.sample_rate = sample_rate;

        self.cutoff = midi_note_to_frequency_precise(filter_state.midi_cutoff);
        let min_nyquist = sample_rate * MIN_NYQUIST_MULT;
        self.cutoff = self.cutoff.clamp(MIN_CUTOFF, min_nyquist);

        let mut resonance_percent = filter_state.resonance_percent.clamp(0.0, 1.0);
        resonance_percent = resonance_percent.sqrt();
        self.resonance = interpolate(
            PolyF32::splat(MIN_RESONANCE),
            PolyF32::splat(MAX_RESONANCE),
            resonance_percent,
        );
        self.resonance +=
            filter_state.drive_percent * filter_state.resonance_percent * DRIVE_RESONANCE_BOOST;

        let blend = (filter_state.pass_blend - 1.0).clamp(-1.0, 1.0);

        let resonance_scale = resonance_percent * resonance_percent * 2.0 + 1.0;
        self.drive = filter_state.drive / resonance_scale;

        match filter_state.style {
            FilterStyle::DualNotchBand => {
                let t = blend * 0.5 + 0.5;
                let drive_t = (-blend + 1.0).min(PolyF32::ONE);
                let drive_mult = -t + 2.0;
                self.drive = interpolate(filter_state.drive, self.drive * drive_mult, drive_t);

                self.low_pass_amount = t;
                self.band_pass_amount = PolyF32::ZERO;
                self.high_pass_amount = PolyF32::ONE;
            }
            FilterStyle::NotchPassSwap => {
                let drive_t = blend.abs();
                self.drive = interpolate(filter_state.drive, self.drive, drive_t);

                self.low_pass_amount = (-blend + 1.0).min(PolyF32::ONE);
                self.band_pass_amount = PolyF32::ZERO;
                self.high_pass_amount = (blend + 1.0).min(PolyF32::ONE);
            }
            FilterStyle::BandPeakNotch => {
                let drive_t = (-blend + 1.0).min(PolyF32::ONE);
                self.drive = interpolate(filter_state.drive, self.drive, drive_t);

                let drive_inv_t = -drive_t + 1.0;
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

        self.post_multiply = PolyF32::ONE / (resonance_scale * self.drive).sqrt();
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
        let mut current_low = self.current_low;
        let mut current_band = self.current_band;
        let mut current_high = self.current_high;

        let tick_increment = 1.0 / num_samples as f32;
        let delta_resonance = (self.resonance - current_resonance) * tick_increment;
        let delta_drive = (self.drive - current_drive) * tick_increment;
        let delta_post_multiply = (self.post_multiply - current_post_multiply) * tick_increment;
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
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = lookup.cubic_lookup(frequency);
            let coefficient_squared = coefficient * coefficient;
            let coefficient2 = coefficient * 2.0;

            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_post_multiply += delta_post_multiply;
            current_low += delta_low;
            current_band += delta_band;
            current_high += delta_high;

            let resonance = tune_resonance(current_resonance, coefficient2);
            let stage1_feedback_mult = coefficient2 - coefficient_squared - 1.0;
            let normalizer =
                PolyF32::ONE / (resonance * (coefficient_squared - coefficient) + 1.0);

            match style {
                FilterStyle::TwelveDb => {
                    self.tick(
                        audio_in[i],
                        coefficient,
                        resonance,
                        stage1_feedback_mult,
                        current_drive,
                        normalizer,
                    );

                    let stage2_input = self.stage1.current_state();
                    let low_pass = self.stage2.current_state();
                    let band_pass = stage2_input - low_pass;
                    let high_pass = self.stage1_input - stage2_input - band_pass;

                    let low = current_low * low_pass;
                    let band_low = low.mul_add(current_band, band_pass);
                    audio_out[i] = band_low.mul_add(current_high, high_pass) * current_post_multiply;
                }
                FilterStyle::DualNotchBand => {
                    let pre_normalizer =
                        PolyF32::ONE / ((coefficient_squared - coefficient) + 1.0);
                    self.tick24(
                        audio_in[i],
                        coefficient,
                        resonance,
                        stage1_feedback_mult,
                        current_drive,
                        pre_normalizer,
                        normalizer,
                        current_low,
                        PolyF32::ZERO,
                        current_high,
                    );

                    let stage2_input = self.stage1.current_state();
                    let low_pass = self.stage2.current_state();
                    let high_pass =
                        self.stage1_input - stage2_input - stage2_input + low_pass;

                    let low = current_high * low_pass;
                    audio_out[i] = low.mul_add(current_low, high_pass) * current_post_multiply;
                }
                _ => {
                    let pre_normalizer =
                        PolyF32::ONE / ((coefficient_squared - coefficient) + 1.0);
                    self.tick24(
                        audio_in[i],
                        coefficient,
                        resonance,
                        stage1_feedback_mult,
                        current_drive,
                        pre_normalizer,
                        normalizer,
                        current_low,
                        current_band,
                        current_high,
                    );

                    let stage2_input = self.stage1.current_state();
                    let low_pass = self.stage2.current_state();
                    let band_pass = stage2_input - low_pass;
                    let high_pass = self.stage1_input - stage2_input - band_pass;

                    let low = current_low * low_pass;
                    let band_low = low.mul_add(current_band, band_pass);
                    audio_out[i] = band_low.mul_add(current_high, high_pass) * current_post_multiply;
                }
            }
            debug_assert!(audio_out[i].is_finite());
        }

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
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
        stage1_feedback_mult: PolyF32,
        drive: PolyF32,
        pre_normalizer: PolyF32,
        normalizer: PolyF32,
        low: PolyF32,
        band: PolyF32,
        high: PolyF32,
    ) {
        let mult_stage2 = -coefficient + 1.0;
        let feedback = (stage1_feedback_mult * self.pre_stage1.next_state())
            .mul_add(mult_stage2, self.pre_stage2.next_state());

        let stage1_input = (audio_in - feedback) * pre_normalizer;

        let stage1_out = self.pre_stage1.tick_basic(stage1_input, coefficient);
        let stage2_out = self.pre_stage2.tick_basic(stage1_out, coefficient);

        let band_pass_out = stage1_out - stage2_out;
        let high_pass_out = stage1_input - stage1_out - band_pass_out;

        let low_out = low * stage2_out;
        let band_low_out = low_out.mul_add(band, band_pass_out);
        let audio_out = band_low_out.mul_add(high, high_pass_out);

        self.tick(audio_out, coefficient, resonance, stage1_feedback_mult, drive, normalizer);
    }

    #[inline(always)]
    fn tick(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        stage1_feedback_mult: PolyF32,
        drive: PolyF32,
        normalizer: PolyF32,
    ) {
        let mult_stage2 = -coefficient + 1.0;
        let feedback = (stage1_feedback_mult * self.stage1.next_state())
            .mul_add(mult_stage2, self.stage2.next_state());

        self.stage1_input = math::tanh((drive * audio_in - resonance * feedback) * normalizer);

        let stage1_out = self.stage1.tick_basic(self.stage1_input, coefficient);
        self.stage2.tick_basic(stage1_out, coefficient);
    }

    pub fn resonance(&self) -> PolyF32 {
        self.resonance
    }

    pub fn drive(&self) -> PolyF32 {
        self.drive
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

    pub fn low_amount_24(&self, style: FilterStyle) -> PolyF32 {
        if style == FilterStyle::DualNotchBand {
            self.high_pass_amount
        } else {
            self.low_pass_amount
        }
    }

    pub fn high_amount_24(&self, style: FilterStyle) -> PolyF32 {
        if style == FilterStyle::DualNotchBand {
            self.low_pass_amount
        } else {
            self.high_pass_amount
        }
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

    fn run_sine(
        filter: &mut SallenKeyFilter,
        state: &FilterState,
        freq: f32,
        blocks: usize,
    ) -> f32 {
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
        let mut filter = SallenKeyFilter::new();
        let low = run_sine(&mut filter, &state, 100.0, 40);
        let mut filter = SallenKeyFilter::new();
        let high = run_sine(&mut filter, &state, 8000.0, 40);
        assert!(low > 4.0 * high, "low rms {low} vs high rms {high}");
    }

    #[test]
    fn low_pass_24_is_steeper_than_12() {
        let state12 = state_for(1000.0, FilterStyle::TwelveDb, 0.0);
        let mut filter = SallenKeyFilter::new();
        let stop12 = run_sine(&mut filter, &state12, 8000.0, 40);

        let state24 = state_for(1000.0, FilterStyle::TwentyFourDb, 0.0);
        let mut filter = SallenKeyFilter::new();
        let stop24 = run_sine(&mut filter, &state24, 8000.0, 40);
        assert!(stop24 < stop12, "24dB stopband {stop24} not below 12dB {stop12}");
    }

    #[test]
    fn dual_style_is_finite() {
        let state = state_for(1000.0, FilterStyle::DualNotchBand, 0.8);
        let mut filter = SallenKeyFilter::new();
        let rms = run_sine(&mut filter, &state, 440.0, 10);
        assert!(rms.is_finite() && rms > 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let state = state_for(2000.0, FilterStyle::TwelveDb, 0.0);
        let mut filter = SallenKeyFilter::new();
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
