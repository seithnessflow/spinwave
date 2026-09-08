//! Four-stage ladder filter (port of `ladder_filter.{h,cpp}`).

use vital_poly::utils::interpolate;
use vital_poly::{math, PolyF32, PolyMask};

use super::filter_state::{
    coefficient_lookup, midi_note_to_frequency_precise, FilterState, FilterStyle,
};
use super::one_pole::{AlgebraicSat, OnePoleFilter};

pub const NUM_STAGES: usize = 4;
pub const RESONANCE_TUNING: f32 = 1.66;
pub const MIN_RESONANCE: f32 = 0.001;
pub const MAX_RESONANCE: f32 = 4.1;
pub const MAX_COEFFICIENT: f32 = 0.35;
pub const DRIVE_RESONANCE_BOOST: f32 = 5.0;
pub const MIN_CUTOFF: f32 = 1.0;
pub const MAX_CUTOFF: f32 = 20000.0;

const LOW_PASS_24: [f32; NUM_STAGES + 1] = [0.0, 0.0, 0.0, 0.0, 1.0];
const BAND_PASS_24: [f32; NUM_STAGES + 1] = [0.0, 0.0, -1.0, 2.0, -1.0];
const HIGH_PASS_24: [f32; NUM_STAGES + 1] = [1.0, -4.0, 6.0, -4.0, 1.0];
const LOW_PASS_12: [f32; NUM_STAGES + 1] = [0.0, 0.0, 1.0, 0.0, 0.0];
const BAND_PASS_12: [f32; NUM_STAGES + 1] = [0.0, 1.0, -1.0, 0.0, 0.0];
const HIGH_PASS_12: [f32; NUM_STAGES + 1] = [1.0, -2.0, 1.0, 0.0, 0.0];

#[derive(Clone, Debug, Default)]
pub struct LadderFilter {
    state: FilterState,
    sample_rate: f32,

    resonance: PolyF32,
    drive: PolyF32,
    post_multiply: PolyF32,
    stage_scales: [PolyF32; NUM_STAGES + 1],

    current_resonance: PolyF32,
    current_drive: PolyF32,
    current_post_multiply: PolyF32,
    current_stage_scales: [PolyF32; NUM_STAGES + 1],

    stages: [OnePoleFilter<AlgebraicSat>; NUM_STAGES],
    filter_input: PolyF32,
}

impl LadderFilter {
    pub fn new() -> LadderFilter {
        let mut filter = LadderFilter { sample_rate: 44100.0, ..Default::default() };
        filter.hard_reset();
        filter
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.filter_input = reset_mask.select(PolyF32::ZERO, self.filter_input);
        for stage in &mut self.stages {
            stage.reset(reset_mask);
        }

        self.current_resonance = reset_mask.select(self.resonance, self.current_resonance);
        self.current_drive = reset_mask.select(self.drive, self.current_drive);
        self.current_post_multiply =
            reset_mask.select(self.post_multiply, self.current_post_multiply);
        for i in 0..=NUM_STAGES {
            self.current_stage_scales[i] =
                reset_mask.select(self.stage_scales[i], self.current_stage_scales[i]);
        }
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
        self.resonance = PolyF32::ZERO;
        self.drive = PolyF32::ZERO;
        self.post_multiply = PolyF32::ZERO;
        self.stage_scales = [PolyF32::ZERO; NUM_STAGES + 1];
        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_stage_scales = self.stage_scales;
    }

    /// Per-block parameter update (C++ `setupFilter` + ramp capture).
    pub fn setup(&mut self, filter_state: &FilterState, sample_rate: f32) {
        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_stage_scales = self.stage_scales;

        self.state = *filter_state;
        self.sample_rate = sample_rate;

        let resonance_percent = filter_state.resonance_percent.clamp(0.0, 1.0);
        let mut resonance_adjust = resonance_percent;
        if filter_state.style != FilterStyle::TwelveDb {
            resonance_adjust =
                (resonance_percent * (0.5 * vital_poly::constants::PI)).map(f32::sin);
        }

        self.resonance = interpolate(
            PolyF32::splat(MIN_RESONANCE),
            PolyF32::splat(MAX_RESONANCE),
            resonance_adjust,
        );
        self.resonance +=
            filter_state.drive_percent * filter_state.resonance_percent * DRIVE_RESONANCE_BOOST;

        self.set_stage_scales(filter_state);
    }

    fn set_stage_scales(&mut self, filter_state: &FilterState) {
        let blend = (filter_state.pass_blend - 1.0).clamp(-1.0, 1.0);
        let mut band_pass = (-blend * blend + 1.0).sqrt();

        let blend_mask = blend.lt(PolyF32::ZERO);
        let low_pass = (-blend) & blend_mask;
        let high_pass = blend & !blend_mask;

        let resonance_percent = filter_state.resonance_percent.clamp(0.0, 1.0);
        let mut drive_mult = resonance_percent + 1.0;
        if filter_state.style != FilterStyle::TwelveDb {
            drive_mult = resonance_percent.map(f32::sin) + 1.0;
        }

        let resonance_scale = interpolate(drive_mult, PolyF32::ONE, high_pass);
        self.drive = filter_state.drive * resonance_scale;
        self.post_multiply =
            PolyF32::ONE / ((filter_state.drive - 1.0) * 0.5 + 1.0).sqrt();

        match filter_state.style {
            FilterStyle::TwelveDb => {
                for i in 0..=NUM_STAGES {
                    self.stage_scales[i] = low_pass * LOW_PASS_12[i]
                        + band_pass * BAND_PASS_12[i]
                        + high_pass * HIGH_PASS_12[i];
                }
            }
            FilterStyle::TwentyFourDb => {
                band_pass = -blend.abs() + 1.0;
                self.post_multiply =
                    PolyF32::ONE / ((filter_state.drive - 1.0) * 0.25 + 1.0).sqrt();

                for i in 0..=NUM_STAGES {
                    self.stage_scales[i] = low_pass * LOW_PASS_24[i]
                        + band_pass * BAND_PASS_24[i]
                        + high_pass * HIGH_PASS_24[i];
                }
            }
            FilterStyle::DualNotchBand => {
                self.drive = filter_state.drive;
                let low_pass_fade = (blend + 1.0).min(PolyF32::ONE);
                let high_pass_fade = (-blend + 1.0).min(PolyF32::ONE);

                self.stage_scales[0] = low_pass_fade;
                self.stage_scales[1] = low_pass_fade * -4.0;
                self.stage_scales[2] = high_pass_fade * 4.0 + low_pass_fade * 8.0;
                self.stage_scales[3] = high_pass_fade * -8.0 - low_pass_fade * 8.0;
                self.stage_scales[4] = high_pass_fade * 4.0 + low_pass_fade * 4.0;
            }
            FilterStyle::NotchPassSwap => {
                self.post_multiply =
                    PolyF32::ONE / ((filter_state.drive - 1.0) * 0.5 + 1.0).sqrt();

                let low_pass_fade = (blend + 1.0).min(PolyF32::ONE);
                let low_pass_fade2 = low_pass_fade * low_pass_fade;
                let high_pass_fade = (-blend + 1.0).min(PolyF32::ONE);
                let high_pass_fade2 = high_pass_fade * high_pass_fade;
                let low_high_pass_fade = low_pass_fade * high_pass_fade;

                self.stage_scales[0] = low_pass_fade2;
                self.stage_scales[1] = low_pass_fade2 * -4.0;
                self.stage_scales[2] = low_pass_fade2 * 6.0 + low_high_pass_fade * 2.0;
                self.stage_scales[3] = low_pass_fade2 * -4.0 - low_high_pass_fade * 4.0;
                self.stage_scales[4] = low_pass_fade2 + high_pass_fade2 + low_high_pass_fade * 2.0;
            }
            FilterStyle::BandPeakNotch => {
                let drive_t = (-blend + 1.0).min(PolyF32::ONE);
                self.drive = interpolate(filter_state.drive, self.drive, drive_t);

                let drive_inv_t = -drive_t + 1.0;
                let mult = ((drive_inv_t * drive_inv_t) * 0.5 + 0.5).sqrt();
                let peak_band_value = -(-blend).max(PolyF32::ZERO);
                let low_high = mult * (peak_band_value + 1.0);
                let band = mult * (peak_band_value - blend + 1.0) * 2.0;

                for i in 0..=NUM_STAGES {
                    self.stage_scales[i] = low_high * LOW_PASS_12[i]
                        + band * BAND_PASS_12[i]
                        + low_high * HIGH_PASS_12[i];
                }
            }
            FilterStyle::Shelving => {}
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
        let mut current_stage_scales = self.current_stage_scales;

        let tick_increment = 1.0 / num_samples as f32;
        let delta_resonance = (self.resonance - current_resonance) * tick_increment;
        let delta_drive = (self.drive - current_drive) * tick_increment;
        let delta_post_multiply = (self.post_multiply - current_post_multiply) * tick_increment;
        let mut delta_stage_scales = [PolyF32::ZERO; NUM_STAGES + 1];
        for i in 0..=NUM_STAGES {
            delta_stage_scales[i] =
                (self.stage_scales[i] - current_stage_scales[i]) * tick_increment;
        }

        let lookup = coefficient_lookup();
        let base_midi = match midi_cutoff {
            Some(buffer) => buffer[num_samples - 1],
            None => self.state.midi_cutoff,
        };
        let base_frequency =
            midi_note_to_frequency_precise(base_midi) * (1.0 / self.sample_rate);
        let max_frequency = PolyF32::splat(MAX_CUTOFF / self.sample_rate);

        for i in 0..num_samples {
            let midi = midi_cutoff.map_or(base_midi, |b| b[i]);
            let midi_delta = midi - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(max_frequency);
            let coefficient = lookup.cubic_lookup(frequency);

            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_post_multiply += delta_post_multiply;
            for stage in 0..=NUM_STAGES {
                current_stage_scales[stage] += delta_stage_scales[stage];
            }

            self.tick(audio_in[i], coefficient, current_resonance, current_drive);
            let mut total = current_stage_scales[0] * self.filter_input;
            for stage in 0..NUM_STAGES {
                total += current_stage_scales[stage + 1] * self.stages[stage].current_state();
            }

            audio_out[i] = total * current_post_multiply;
        }

        self.current_resonance = self.resonance;
        self.current_drive = self.drive;
        self.current_post_multiply = self.post_multiply;
        self.current_stage_scales = self.stage_scales;
    }

    #[inline(always)]
    fn tick(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
    ) {
        let g1 = coefficient * RESONANCE_TUNING;
        let g2 = g1 * g1;
        let g3 = g1 * g2;

        let filter_state1 =
            self.stages[3].next_sat_state().mul_add(g1, self.stages[2].next_sat_state());
        let filter_state2 = filter_state1.mul_add(g2, self.stages[1].next_sat_state());
        let filter_state = filter_state2.mul_add(g3, self.stages[0].next_sat_state());

        let filter_input = audio_in * drive - resonance * filter_state;
        self.filter_input = math::tanh(filter_input);

        let mut stage_out = self.stages[0].tick(self.filter_input, coefficient);
        stage_out = self.stages[1].tick(stage_out, coefficient);
        stage_out = self.stages[2].tick(stage_out, coefficient);
        self.stages[3].tick(stage_out, coefficient);
    }

    pub fn drive(&self) -> PolyF32 {
        self.drive
    }

    pub fn resonance(&self) -> PolyF32 {
        self.resonance
    }

    pub fn stage_scale(&self, index: usize) -> PolyF32 {
        self.stage_scales[index]
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
        state.resonance_percent = PolyF32::splat(0.4);
        state.set_drive_db(PolyF32::ZERO);
        state.set_pass_blend(PolyF32::splat(pass_blend));
        state.style = style;
        state
    }

    fn run_sine(filter: &mut LadderFilter, state: &FilterState, freq: f32, blocks: usize) -> f32 {
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
    fn low_pass_24_selectivity() {
        let state = state_for(1000.0, FilterStyle::TwentyFourDb, 0.0);
        let mut filter = LadderFilter::new();
        let low = run_sine(&mut filter, &state, 100.0, 40);
        let mut filter = LadderFilter::new();
        let high = run_sine(&mut filter, &state, 8000.0, 40);
        assert!(low > 4.0 * high, "low rms {low} vs high rms {high}");
    }

    #[test]
    fn high_pass_12_selectivity() {
        let state = state_for(1000.0, FilterStyle::TwelveDb, 2.0);
        let mut filter = LadderFilter::new();
        let low = run_sine(&mut filter, &state, 100.0, 40);
        let mut filter = LadderFilter::new();
        let high = run_sine(&mut filter, &state, 8000.0, 40);
        assert!(high > 4.0 * low, "high rms {high} vs low rms {low}");
    }

    #[test]
    fn reset_clears_state() {
        let state = state_for(2000.0, FilterStyle::TwentyFourDb, 0.0);
        let mut filter = LadderFilter::new();
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
