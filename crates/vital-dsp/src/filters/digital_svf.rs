//! Digital state-variable filter (port of `digital_svf.{h,cpp}`).
//!
//! Per-block call order: `setup` (once), then optionally `reset` for
//! newly-triggered voices, then `process`/`process_modulated`. Coefficients
//! and blend values ramp linearly across the block from the previous block's
//! targets, exactly like the reference.

use vital_poly::constants::MIN_NYQUIST_MULT;
use vital_poly::utils::interpolate;
use vital_poly::{math, PolyF32, PolyMask};

use super::filter_state::{
    db_to_magnitude_precise, midi_note_to_frequency_precise, ratio_to_midi_transpose_precise,
    svf_coefficient_lookup, FilterState, FilterStyle,
};

pub const DEFAULT_MIN_RESONANCE: f32 = 0.5;
pub const DEFAULT_MAX_RESONANCE: f32 = 16.0;
pub const MIN_CUTOFF: f32 = 1.0;
pub const MAX_GAIN: f32 = 15.0;
pub const MIN_GAIN: f32 = -15.0;

/// Blend coefficients for the input/band/low taps of the SVF core.
#[derive(Clone, Copy, Debug, Default)]
pub struct FilterValues {
    pub v0: PolyF32,
    pub v1: PolyF32,
    pub v2: PolyF32,
}

impl FilterValues {
    pub fn hard_reset(&mut self) {
        self.v0 = PolyF32::ZERO;
        self.v1 = PolyF32::ZERO;
        self.v2 = PolyF32::ZERO;
    }

    pub fn reset(&mut self, reset_mask: PolyMask, other: &FilterValues) {
        self.v0 = reset_mask.select(other.v0, self.v0);
        self.v1 = reset_mask.select(other.v1, self.v1);
        self.v2 = reset_mask.select(other.v2, self.v2);
    }

    pub fn get_delta(&self, target: &FilterValues, increment: f32) -> FilterValues {
        FilterValues {
            v0: (target.v0 - self.v0) * increment,
            v1: (target.v1 - self.v1) * increment,
            v2: (target.v2 - self.v2) * increment,
        }
    }

    #[inline(always)]
    pub fn increment(&mut self, delta: &FilterValues) {
        self.v0 += delta.v0;
        self.v1 += delta.v1;
        self.v2 += delta.v2;
    }
}

#[derive(Clone, Debug)]
pub struct DigitalSvf {
    state: FilterState,
    sample_rate: f32,

    midi_cutoff: PolyF32,
    resonance: PolyF32,
    blends1: FilterValues,
    blends2: FilterValues,
    drive: PolyF32,
    post_multiply: PolyF32,

    low_amount: PolyF32,
    band_amount: PolyF32,
    high_amount: PolyF32,

    current_resonance: PolyF32,
    current_drive: PolyF32,
    current_post_multiply: PolyF32,
    current_blends1: FilterValues,
    current_blends2: FilterValues,

    ic1eq_pre: PolyF32,
    ic2eq_pre: PolyF32,
    ic1eq: PolyF32,
    ic2eq: PolyF32,

    min_resonance: f32,
    max_resonance: f32,

    basic: bool,
    drive_compensation: bool,
}

impl Default for DigitalSvf {
    fn default() -> DigitalSvf {
        DigitalSvf::new()
    }
}

impl DigitalSvf {
    pub fn new() -> DigitalSvf {
        let mut svf = DigitalSvf {
            state: FilterState::default(),
            sample_rate: 44100.0,
            midi_cutoff: PolyF32::ZERO,
            resonance: PolyF32::ZERO,
            blends1: FilterValues::default(),
            blends2: FilterValues::default(),
            drive: PolyF32::ZERO,
            post_multiply: PolyF32::ZERO,
            low_amount: PolyF32::ZERO,
            band_amount: PolyF32::ZERO,
            high_amount: PolyF32::ZERO,
            current_resonance: PolyF32::ZERO,
            current_drive: PolyF32::ZERO,
            current_post_multiply: PolyF32::ZERO,
            current_blends1: FilterValues::default(),
            current_blends2: FilterValues::default(),
            ic1eq_pre: PolyF32::ZERO,
            ic2eq_pre: PolyF32::ZERO,
            ic1eq: PolyF32::ZERO,
            ic2eq: PolyF32::ZERO,
            min_resonance: DEFAULT_MIN_RESONANCE,
            max_resonance: DEFAULT_MAX_RESONANCE,
            basic: false,
            drive_compensation: true,
        };
        svf.hard_reset();
        svf
    }

    pub fn set_resonance_bounds(&mut self, min: f32, max: f32) {
        self.min_resonance = min;
        self.max_resonance = max;
    }

    pub fn set_basic(&mut self, basic: bool) {
        self.basic = basic;
    }

    pub fn set_drive_compensation(&mut self, drive_compensation: bool) {
        self.drive_compensation = drive_compensation;
    }

    /// Per-block parameter update (C++ `setupFilter` plus the ramp capture
    /// done at the top of `processWithInput`).
    pub fn setup(&mut self, filter_state: &FilterState, sample_rate: f32) {
        // Ramps start from the previous block's targets.
        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_blends1 = self.blends1;
        self.current_blends2 = self.blends2;

        self.state = *filter_state;
        self.sample_rate = sample_rate;

        self.midi_cutoff = filter_state.midi_cutoff;
        let cutoff = midi_note_to_frequency_precise(filter_state.midi_cutoff);
        let min_nyquist = sample_rate * MIN_NYQUIST_MULT;
        let _cutoff = cutoff.clamp(MIN_CUTOFF, min_nyquist);

        let gain_decibels = filter_state.gain.clamp(MIN_GAIN, MAX_GAIN);
        let gain_amplitude = db_to_magnitude_precise(gain_decibels);

        let resonance_percent = filter_state.resonance_percent.clamp(0.0, 1.0);
        let resonance_adjust = resonance_percent * resonance_percent * resonance_percent;
        let resonance = interpolate(
            PolyF32::splat(self.min_resonance),
            PolyF32::splat(self.max_resonance),
            resonance_adjust,
        );
        if self.drive_compensation {
            self.drive = filter_state.drive / (resonance_adjust * 2.0 + 1.0);
        } else {
            self.drive = filter_state.drive;
        }

        self.post_multiply = gain_amplitude / filter_state.drive.sqrt();
        self.resonance = PolyF32::ONE / resonance;

        let blend = (filter_state.pass_blend - 1.0).clamp(-1.0, 1.0);

        match filter_state.style {
            FilterStyle::DualNotchBand => {
                let t = blend * 0.5 + 0.5;
                let drive_t = (-blend + 1.0).min(PolyF32::ONE);
                let drive_mult = -t + 2.0;
                self.drive = interpolate(filter_state.drive, self.drive * drive_mult, drive_t);

                self.low_amount = t;
                self.band_amount = PolyF32::ZERO;
                self.high_amount = PolyF32::ONE;
            }
            FilterStyle::NotchPassSwap => {
                let drive_t = blend.abs();
                self.drive = interpolate(filter_state.drive, self.drive, drive_t);

                self.low_amount = (-blend + 1.0).min(PolyF32::ONE);
                self.band_amount = PolyF32::ZERO;
                self.high_amount = (blend + 1.0).min(PolyF32::ONE);
            }
            FilterStyle::BandPeakNotch => {
                let drive_t = (-blend + 1.0).min(PolyF32::ONE);
                self.drive = interpolate(filter_state.drive, self.drive, drive_t);

                let drive_inv_t = -drive_t + 1.0;
                let mult = ((drive_inv_t * drive_inv_t) * 0.5 + 0.5).sqrt();
                let peak_band_value = -(-blend).max(PolyF32::ZERO);
                self.low_amount = mult * (peak_band_value + 1.0);
                self.band_amount = mult * (peak_band_value - blend + 1.0) * 2.0;
                self.high_amount = self.low_amount;
            }
            FilterStyle::Shelving => {
                self.drive = PolyF32::ONE;
                self.post_multiply = PolyF32::ONE;
                let low_bell_t = (blend + 1.0).clamp(0.0, 1.0);
                let bell_high_t = blend.clamp(0.0, 1.0);
                let band_t = PolyF32::ONE - blend * blend;

                let amplitude_sqrt = gain_amplitude.sqrt();
                let amplitude_quartic = amplitude_sqrt.sqrt();
                let mult_adjust = math::pow(amplitude_quartic, blend);

                self.low_amount = interpolate(gain_amplitude, PolyF32::ONE, low_bell_t);
                self.high_amount = interpolate(PolyF32::ONE, gain_amplitude, bell_high_t);
                self.band_amount = self.resonance
                    * amplitude_sqrt
                    * interpolate(PolyF32::ONE, amplitude_sqrt, band_t);
                self.midi_cutoff += ratio_to_midi_transpose_precise(mult_adjust);
            }
            _ => {
                self.band_amount = (-blend * blend + 1.0).sqrt();
                let blend_mask = blend.lt(PolyF32::ZERO);
                self.low_amount = (-blend) & blend_mask;
                self.high_amount = blend & !blend_mask;
            }
        }

        self.blends1.v0 = PolyF32::ZERO;
        self.blends1.v1 = self.band_amount;
        self.blends1.v2 = self.low_amount;

        self.blends2.v0 = PolyF32::ZERO;
        self.blends2.v1 = self.band_amount;
        self.blends2.v2 = self.high_amount;

        self.blends1.v0 += self.high_amount;
        self.blends1.v1 += -self.resonance * self.high_amount;
        self.blends1.v2 += -self.high_amount;

        self.blends2.v0 += self.low_amount;
        self.blends2.v1 += -self.resonance * self.low_amount;
        self.blends2.v2 += -self.low_amount;
    }

    /// Resets voice state and snaps the ramps to the freshly set-up targets
    /// for the masked lanes. Call after `setup`, before `process`.
    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.ic1eq_pre = reset_mask.select(PolyF32::ZERO, self.ic1eq_pre);
        self.ic2eq_pre = reset_mask.select(PolyF32::ZERO, self.ic2eq_pre);
        self.ic1eq = reset_mask.select(PolyF32::ZERO, self.ic1eq);
        self.ic2eq = reset_mask.select(PolyF32::ZERO, self.ic2eq);

        self.current_blends1.reset(reset_mask, &self.blends1);
        self.current_blends2.reset(reset_mask, &self.blends2);
        self.current_resonance = reset_mask.select(self.resonance, self.current_resonance);
        self.current_drive = reset_mask.select(self.drive, self.current_drive);
        self.current_post_multiply =
            reset_mask.select(self.post_multiply, self.current_post_multiply);
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
        self.resonance = PolyF32::ONE;
        self.blends1.hard_reset();
        self.blends2.hard_reset();

        self.low_amount = PolyF32::ZERO;
        self.band_amount = PolyF32::ZERO;
        self.high_amount = PolyF32::ZERO;

        self.drive = PolyF32::ZERO;
        self.post_multiply = PolyF32::ZERO;

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_blends1 = self.blends1;
        self.current_blends2 = self.blends2;
    }

    /// Processes a block with a constant (per-block) cutoff.
    pub fn process(&mut self, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        self.process_inner(audio_in, None, audio_out);
    }

    /// Processes a block with an audio-rate MIDI cutoff buffer.
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
        assert_eq!(audio_in.len(), audio_out.len());
        assert!(!audio_in.is_empty());

        let blends1 = self.current_blends1;
        let blends2 = self.current_blends2;
        let resonance = self.current_resonance;
        let drive = self.current_drive;
        let post_multiply = self.current_post_multiply;

        if self.state.style == FilterStyle::Shelving || self.basic {
            self.process_basic12(audio_in, midi_cutoff, audio_out, resonance, drive, post_multiply, blends1);
        } else if self.state.style == FilterStyle::DualNotchBand {
            self.process_dual(audio_in, midi_cutoff, audio_out, resonance, drive, post_multiply, blends1, blends2);
        } else if self.state.style == FilterStyle::TwelveDb {
            self.process12(audio_in, midi_cutoff, audio_out, resonance, drive, post_multiply, blends1);
        } else {
            self.process24(audio_in, midi_cutoff, audio_out, resonance, drive, post_multiply, blends1);
        }

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_blends1 = self.blends1;
        self.current_blends2 = self.blends2;
    }

    #[inline(always)]
    fn base_midi(&self, midi_cutoff: Option<&[PolyF32]>, num_samples: usize) -> PolyF32 {
        match midi_cutoff {
            Some(buffer) => buffer[num_samples - 1],
            None => self.state.midi_cutoff,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process12(
        &mut self,
        audio_in: &[PolyF32],
        midi_cutoff: Option<&[PolyF32]>,
        audio_out: &mut [PolyF32],
        mut current_resonance: PolyF32,
        mut current_drive: PolyF32,
        mut current_post_multiply: PolyF32,
        mut blends: FilterValues,
    ) {
        let num_samples = audio_in.len();
        let sample_inc = 1.0 / num_samples as f32;
        let delta_blends = blends.get_delta(&self.blends1, sample_inc);
        let delta_resonance = (self.resonance - current_resonance) * sample_inc;
        let delta_drive = (self.drive - current_drive) * sample_inc;
        let delta_post_multiply = (self.post_multiply - current_post_multiply) * sample_inc;

        let lookup = svf_coefficient_lookup();
        let base_midi = self.base_midi(midi_cutoff, num_samples);
        let base_frequency =
            midi_note_to_frequency_precise(base_midi) * (1.0 / self.sample_rate);

        for i in 0..num_samples {
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi.max(PolyF32::ZERO) - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = lookup.cubic_lookup(frequency);

            blends.increment(&delta_blends);
            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_post_multiply += delta_post_multiply;

            audio_out[i] = self.tick(audio_in[i], coefficient, current_resonance, current_drive, &blends)
                * current_post_multiply;
            debug_assert!(audio_out[i].is_finite());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_basic12(
        &mut self,
        audio_in: &[PolyF32],
        midi_cutoff: Option<&[PolyF32]>,
        audio_out: &mut [PolyF32],
        mut current_resonance: PolyF32,
        mut current_drive: PolyF32,
        mut current_post_multiply: PolyF32,
        mut blends: FilterValues,
    ) {
        let num_samples = audio_in.len();
        let sample_inc = 1.0 / num_samples as f32;
        let delta_blends = blends.get_delta(&self.blends1, sample_inc);
        let delta_resonance = (self.resonance - current_resonance) * sample_inc;
        let delta_drive = (self.drive - current_drive) * sample_inc;
        let delta_post_multiply = (self.post_multiply - current_post_multiply) * sample_inc;

        let lookup = svf_coefficient_lookup();
        let base_midi = self.base_midi(midi_cutoff, num_samples);
        let base_frequency =
            midi_note_to_frequency_precise(base_midi) * (1.0 / self.sample_rate);

        for i in 0..num_samples {
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = lookup.cubic_lookup(frequency);

            blends.increment(&delta_blends);
            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_post_multiply += delta_post_multiply;

            audio_out[i] =
                self.tick_basic(audio_in[i], coefficient, current_resonance, current_drive, &blends)
                    * current_post_multiply;
            debug_assert!(audio_out[i].is_finite());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process24(
        &mut self,
        audio_in: &[PolyF32],
        midi_cutoff: Option<&[PolyF32]>,
        audio_out: &mut [PolyF32],
        mut current_resonance: PolyF32,
        mut current_drive: PolyF32,
        mut current_post_multiply: PolyF32,
        mut blends: FilterValues,
    ) {
        let num_samples = audio_in.len();
        let sample_inc = 1.0 / num_samples as f32;
        let delta_blends = blends.get_delta(&self.blends1, sample_inc);
        let delta_resonance = (self.resonance - current_resonance) * sample_inc;
        let delta_drive = (self.drive - current_drive) * sample_inc;
        let delta_post_multiply = (self.post_multiply - current_post_multiply) * sample_inc;

        let lookup = svf_coefficient_lookup();
        let base_midi = self.base_midi(midi_cutoff, num_samples);
        let base_frequency =
            midi_note_to_frequency_precise(base_midi) * (1.0 / self.sample_rate);

        for i in 0..num_samples {
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = lookup.cubic_lookup(frequency);

            blends.increment(&delta_blends);
            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_post_multiply += delta_post_multiply;

            let result =
                self.tick24(audio_in[i], coefficient, current_resonance, current_drive, &blends);
            audio_out[i] = result * current_post_multiply;
            debug_assert!(audio_out[i].is_finite());
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_dual(
        &mut self,
        audio_in: &[PolyF32],
        midi_cutoff: Option<&[PolyF32]>,
        audio_out: &mut [PolyF32],
        mut current_resonance: PolyF32,
        mut current_drive: PolyF32,
        mut current_post_multiply: PolyF32,
        mut blends1: FilterValues,
        mut blends2: FilterValues,
    ) {
        let num_samples = audio_in.len();
        let sample_inc = 1.0 / num_samples as f32;
        let delta_blends1 = blends1.get_delta(&self.blends1, sample_inc);
        let delta_blends2 = blends2.get_delta(&self.blends2, sample_inc);
        let delta_resonance = (self.resonance - current_resonance) * sample_inc;
        let delta_drive = (self.drive - current_drive) * sample_inc;
        let delta_post_multiply = (self.post_multiply - current_post_multiply) * sample_inc;

        let lookup = svf_coefficient_lookup();
        let base_midi = self.base_midi(midi_cutoff, num_samples);
        let base_frequency =
            midi_note_to_frequency_precise(base_midi) * (1.0 / self.sample_rate);

        for i in 0..num_samples {
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = lookup.cubic_lookup(frequency);

            blends1.increment(&delta_blends1);
            blends2.increment(&delta_blends2);
            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_post_multiply += delta_post_multiply;

            let result = self.tick_dual(
                audio_in[i],
                coefficient,
                current_resonance,
                current_drive,
                &blends1,
                &blends2,
            );
            audio_out[i] = result * current_post_multiply;
            debug_assert!(audio_out[i].is_finite());
        }
    }

    #[inline(always)]
    fn tick(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        blends: &FilterValues,
    ) -> PolyF32 {
        math::hard_tanh(self.tick_basic(audio_in, coefficient, resonance, drive, blends))
    }

    #[inline(always)]
    fn tick_basic(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        blends: &FilterValues,
    ) -> PolyF32 {
        let coefficient_squared = coefficient * coefficient;
        let coefficient_0 = PolyF32::ONE / (coefficient_squared + coefficient * resonance + 1.0);
        let coefficient_1 = coefficient_0 * coefficient;
        let coefficient_2 = coefficient_0 * coefficient_squared;
        let input = drive * audio_in;

        let v3 = input - self.ic2eq;
        let v1 = (coefficient_0 * self.ic1eq).mul_add(coefficient_1, v3);
        let v2 = self.ic2eq.mul_add(coefficient_1, self.ic1eq).mul_add(coefficient_2, v3);
        self.ic1eq = v1 * 2.0 - self.ic1eq;
        self.ic2eq = v2 * 2.0 - self.ic2eq;

        (blends.v0 * input).mul_add(blends.v1, v1).mul_add(blends.v2, v2)
    }

    #[inline(always)]
    fn tick24(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        blends: &FilterValues,
    ) -> PolyF32 {
        let coefficient_squared = coefficient * coefficient;
        let pre_coefficient_0 = PolyF32::ONE / (coefficient_squared + coefficient + 1.0);
        let pre_coefficient_1 = pre_coefficient_0 * coefficient;
        let pre_coefficient_2 = pre_coefficient_0 * coefficient_squared;

        let input = drive * audio_in;

        let v3_pre = input - self.ic2eq_pre;
        let v1_pre = (pre_coefficient_0 * self.ic1eq_pre).mul_add(pre_coefficient_1, v3_pre);
        let v2_pre = self
            .ic2eq_pre
            .mul_add(pre_coefficient_1, self.ic1eq_pre)
            .mul_add(pre_coefficient_2, v3_pre);
        self.ic1eq_pre = v1_pre * 2.0 - self.ic1eq_pre;
        self.ic2eq_pre = v2_pre * 2.0 - self.ic2eq_pre;
        let out_pre = (blends.v0 * input).mul_add(blends.v1, v1_pre).mul_add(blends.v2, v2_pre);

        let distort = math::hard_tanh(out_pre);

        self.tick(distort, coefficient, resonance, PolyF32::ONE, blends)
    }

    #[inline(always)]
    fn tick_dual(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        blends1: &FilterValues,
        blends2: &FilterValues,
    ) -> PolyF32 {
        let coefficient_squared = coefficient * coefficient;
        let pre_coefficient_0 = PolyF32::ONE / (coefficient_squared + coefficient + 1.0);
        let pre_coefficient_1 = pre_coefficient_0 * coefficient;
        let pre_coefficient_2 = pre_coefficient_0 * coefficient_squared;
        let coefficient_0 = PolyF32::ONE / (coefficient_squared + coefficient * resonance + 1.0);
        let coefficient_1 = coefficient_0 * coefficient;
        let coefficient_2 = coefficient_0 * coefficient_squared;

        let input = drive * audio_in;

        let v3_pre = input - self.ic2eq_pre;
        let v1_pre = (pre_coefficient_0 * self.ic1eq_pre).mul_add(pre_coefficient_1, v3_pre);
        let v2_pre = self
            .ic2eq_pre
            .mul_add(pre_coefficient_1, self.ic1eq_pre)
            .mul_add(pre_coefficient_2, v3_pre);
        self.ic1eq_pre = v1_pre * 2.0 - self.ic1eq_pre;
        self.ic2eq_pre = v2_pre * 2.0 - self.ic2eq_pre;
        let out_pre =
            (blends1.v0 * input).mul_add(blends1.v1, v1_pre).mul_add(blends1.v2, v2_pre);

        let distort = math::hard_tanh(out_pre);

        let v3 = distort - self.ic2eq;
        let v1 = (coefficient_0 * self.ic1eq).mul_add(coefficient_1, v3);
        let v2 = self.ic2eq.mul_add(coefficient_1, self.ic1eq).mul_add(coefficient_2, v3);
        self.ic1eq = v1 * 2.0 - self.ic1eq;
        self.ic2eq = v2 * 2.0 - self.ic2eq;

        math::hard_tanh((blends2.v0 * distort).mul_add(blends2.v1, v1).mul_add(blends2.v2, v2))
    }

    // -- getters used by displays/other modules ------------------------------

    pub fn drive(&self) -> PolyF32 {
        self.drive * self.post_multiply
    }

    pub fn midi_cutoff(&self) -> PolyF32 {
        self.midi_cutoff
    }

    pub fn resonance(&self) -> PolyF32 {
        self.resonance
    }

    pub fn low_amount(&self) -> PolyF32 {
        self.low_amount
    }

    pub fn band_amount(&self) -> PolyF32 {
        self.band_amount
    }

    pub fn high_amount(&self) -> PolyF32 {
        self.high_amount
    }

    pub fn low_amount_24(&self, style: FilterStyle) -> PolyF32 {
        if style == FilterStyle::DualNotchBand {
            self.high_amount
        } else {
            self.low_amount
        }
    }

    pub fn high_amount_24(&self, style: FilterStyle) -> PolyF32 {
        if style == FilterStyle::DualNotchBand {
            self.low_amount
        } else {
            self.high_amount
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

    fn midi_for(freq: f32) -> PolyF32 {
        frequency_to_midi_note_precise(PolyF32::splat(freq))
    }

    fn run_sine(svf: &mut DigitalSvf, state: &FilterState, freq: f32, blocks: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut n = 0usize;
        let mut input = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..blocks {
            for value in input.iter_mut() {
                let phase = 2.0 * core::f32::consts::PI * freq * n as f32 / SAMPLE_RATE;
                *value = PolyF32::splat(phase.sin());
                n += 1;
            }
            svf.setup(state, SAMPLE_RATE);
            svf.process(&input, &mut output);
            if block >= blocks / 2 {
                for value in &output {
                    sum += value.lane(0) * value.lane(0);
                    count += 1;
                }
            }
        }
        (sum / count as f32).sqrt()
    }

    fn low_pass_state() -> FilterState {
        let mut state = FilterState::default();
        state.midi_cutoff = midi_for(1000.0);
        state.resonance_percent = PolyF32::splat(0.3);
        state.set_drive_db(PolyF32::ZERO);
        state.set_pass_blend(PolyF32::ZERO); // blend -1: low pass
        state
    }

    #[test]
    fn low_pass_12_selectivity() {
        let state = low_pass_state();
        let mut svf = DigitalSvf::new();
        let low = run_sine(&mut svf, &state, 100.0, 40);
        let mut svf = DigitalSvf::new();
        let high = run_sine(&mut svf, &state, 8000.0, 40);
        assert!(low > 4.0 * high, "low rms {low} vs high rms {high}");
    }

    #[test]
    fn high_pass_12_selectivity() {
        let mut state = low_pass_state();
        state.set_pass_blend(PolyF32::splat(2.0)); // blend 1: high pass
        let mut svf = DigitalSvf::new();
        let low = run_sine(&mut svf, &state, 100.0, 40);
        let mut svf = DigitalSvf::new();
        let high = run_sine(&mut svf, &state, 8000.0, 40);
        assert!(high > 4.0 * low, "high rms {high} vs low rms {low}");
    }

    #[test]
    fn low_pass_24_is_steeper_than_12() {
        let mut state = low_pass_state();
        let mut svf12 = DigitalSvf::new();
        let stop12 = run_sine(&mut svf12, &state, 8000.0, 40);
        state.style = FilterStyle::TwentyFourDb;
        let mut svf24 = DigitalSvf::new();
        let stop24 = run_sine(&mut svf24, &state, 8000.0, 40);
        assert!(stop24 < stop12, "24dB stopband {stop24} not below 12dB {stop12}");
    }

    #[test]
    fn shelving_and_dual_styles_are_finite() {
        for style in [FilterStyle::Shelving, FilterStyle::DualNotchBand, FilterStyle::BandPeakNotch] {
            let mut state = low_pass_state();
            state.style = style;
            state.gain = PolyF32::splat(10.0);
            state.set_pass_blend(PolyF32::splat(0.7));
            let mut svf = DigitalSvf::new();
            let rms = run_sine(&mut svf, &state, 440.0, 10);
            assert!(rms.is_finite() && rms > 0.0, "style {style:?} rms {rms}");
        }
    }

    #[test]
    fn impulse_response_is_bounded_and_nonzero() {
        let state = low_pass_state();
        let mut svf = DigitalSvf::new();
        svf.setup(&state, SAMPLE_RATE);
        let mut input = vec![PolyF32::ZERO; BLOCK];
        input[0] = PolyF32::ONE;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        svf.process(&input, &mut output);
        let energy: f32 = output.iter().map(|v| v.lane(0) * v.lane(0)).sum();
        assert!(energy.is_finite());
        assert!(energy > 0.0);
        for value in &output {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn reset_clears_ring_out() {
        let mut state = low_pass_state();
        state.resonance_percent = PolyF32::splat(0.9);
        let mut svf = DigitalSvf::new();
        svf.setup(&state, SAMPLE_RATE);
        let mut input = vec![PolyF32::splat(0.5); BLOCK];
        input[0] = PolyF32::ONE;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        svf.process(&input, &mut output);

        svf.setup(&state, SAMPLE_RATE);
        svf.reset(PolyMask::all_on());
        let silence = vec![PolyF32::ZERO; BLOCK];
        svf.process(&silence, &mut output);
        for value in &output {
            assert_eq!(value.lane(0), 0.0, "state leaked after reset");
        }
    }

    #[test]
    fn modulated_cutoff_matches_flat_when_constant() {
        let state = low_pass_state();
        let mut input = vec![PolyF32::ZERO; BLOCK];
        for (i, value) in input.iter_mut().enumerate() {
            *value = PolyF32::splat((i as f32 * 0.1).sin());
        }
        let cutoff = vec![state.midi_cutoff; BLOCK];

        let mut svf_flat = DigitalSvf::new();
        svf_flat.setup(&state, SAMPLE_RATE);
        let mut out_flat = vec![PolyF32::ZERO; BLOCK];
        svf_flat.process(&input, &mut out_flat);

        let mut svf_mod = DigitalSvf::new();
        svf_mod.setup(&state, SAMPLE_RATE);
        let mut out_mod = vec![PolyF32::ZERO; BLOCK];
        svf_mod.process_modulated(&input, &cutoff, &mut out_mod);

        for i in 0..BLOCK {
            assert!(
                (out_flat[i].lane(0) - out_mod[i].lane(0)).abs() < 1e-6,
                "sample {i} diverged"
            );
        }
    }
}
