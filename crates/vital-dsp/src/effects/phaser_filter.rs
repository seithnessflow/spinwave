//! 12-stage allpass phaser filter (port of Vital's `PhaserFilter`).
//!
//! Three groups of four one-pole allpass stages produce 1/3/5-notch
//! responses that are blended by `pass_blend`. The resonance path band-passes
//! the previous allpass output and feeds it back, saturated.

use vital_poly::constants::PI;
use vital_poly::{math, PolyF32, PolyMask};

use super::one_pole::OnePole;

pub const MIN_RESONANCE: f32 = 0.0;
pub const MAX_RESONANCE: f32 = 1.0;
pub const CLEAR_RATIO: f32 = 20.0;
pub const PEAK_STAGE: usize = 4;
pub const MAX_STAGES: usize = 3 * PEAK_STAGE;

/// Block-rate phaser filter parameters.
#[derive(Clone, Copy, Debug)]
pub struct PhaserFilterParams {
    /// Resonance (feedback) amount in [0, 1].
    pub resonance_percent: PolyF32,
    /// Input drive as a magnitude (1.0 = 0 dB, the effect default).
    pub drive: PolyF32,
    /// Notch blend in [0, 2]: 0 = 1 peak, 1 = 3 peaks, 2 = 5 peaks.
    pub pass_blend: PolyF32,
    /// Inverts the allpass mix (the reference's non-zero `style`).
    pub invert: bool,
}

impl Default for PhaserFilterParams {
    fn default() -> PhaserFilterParams {
        PhaserFilterParams {
            resonance_percent: PolyF32::ZERO,
            drive: PolyF32::ONE,
            pass_blend: PolyF32::ZERO,
            invert: false,
        }
    }
}

/// One-pole allpass coefficient from a frequency ratio (cutoff / sample rate).
/// Matches the function behind the reference's coefficient lookup, computed
/// directly instead of through the 2048-entry cubic table.
#[inline(always)]
fn one_pole_coefficient(frequency_ratio: PolyF32) -> PolyF32 {
    const MAX_RADS: f32 = 0.499 * PI;
    let scaled = frequency_ratio * PI;
    (scaled / (scaled + 1.0)).min(PolyF32::splat(MAX_RADS)).map(f32::tan)
}

#[inline(always)]
fn pass(value: PolyF32) -> PolyF32 {
    value
}

pub struct PhaserFilter {
    clean: bool,
    sample_rate: f32,

    resonance: PolyF32,
    drive: PolyF32,
    peak1_amount: PolyF32,
    peak3_amount: PolyF32,
    peak5_amount: PolyF32,
    invert_mult: PolyF32,

    stages: [OnePole; MAX_STAGES],
    remove_lows_stage: OnePole,
    remove_highs_stage: OnePole,
    allpass_output: PolyF32,
}

impl PhaserFilter {
    pub fn new(clean: bool, sample_rate: f32) -> PhaserFilter {
        let mut filter = PhaserFilter {
            clean,
            sample_rate,
            resonance: PolyF32::ZERO,
            drive: PolyF32::ZERO,
            peak1_amount: PolyF32::ZERO,
            peak3_amount: PolyF32::ZERO,
            peak5_amount: PolyF32::ZERO,
            invert_mult: PolyF32::ONE,
            stages: [OnePole::new(); MAX_STAGES],
            remove_lows_stage: OnePole::new(),
            remove_highs_stage: OnePole::new(),
            allpass_output: PolyF32::ZERO,
        };
        filter.hard_reset();
        filter
    }

    pub fn set_clean(&mut self, clean: bool) {
        self.clean = clean;
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.allpass_output = reset_mask.select(PolyF32::ZERO, self.allpass_output);
        for stage in &mut self.stages {
            stage.reset(reset_mask);
        }
        self.remove_lows_stage.reset(reset_mask);
        self.remove_highs_stage.reset(reset_mask);
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
        self.resonance = PolyF32::ZERO;
        self.drive = PolyF32::ZERO;
        self.peak1_amount = PolyF32::ZERO;
        self.peak3_amount = PolyF32::ZERO;
        self.peak5_amount = PolyF32::ZERO;
        self.allpass_output = PolyF32::ZERO;
    }

    fn setup(&mut self, params: &PhaserFilterParams) {
        let resonance_percent = params.resonance_percent.clamp(0.0, 1.0);
        self.resonance = vital_poly::utils::interpolate(
            PolyF32::splat(MIN_RESONANCE),
            PolyF32::splat(MAX_RESONANCE),
            resonance_percent,
        );
        self.drive = (self.resonance * 0.5 + 1.0) * params.drive;

        let blend = params.pass_blend.clamp(0.0, 2.0);
        self.peak1_amount = (-blend + 1.0).clamp(0.0, 1.0);
        self.peak5_amount = (blend - 1.0).clamp(0.0, 1.0);
        self.peak3_amount = -self.peak1_amount - self.peak5_amount + 1.0;

        self.invert_mult = if params.invert {
            PolyF32::splat(-1.0)
        } else {
            PolyF32::ONE
        };
    }

    /// Processes one block. `cutoff_midi` carries the per-sample filter
    /// cutoff as MIDI notes (the phaser writes its LFO sweep into it).
    pub fn process(
        &mut self,
        params: &PhaserFilterParams,
        cutoff_midi: &[PolyF32],
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        if self.clean {
            self.process_impl(params, cutoff_midi, audio_in, audio_out, math::tanh, pass);
        } else {
            self.process_impl(params, cutoff_midi, audio_in, audio_out, pass, math::hard_tanh);
        }
    }

    fn process_impl(
        &mut self,
        params: &PhaserFilterParams,
        cutoff_midi: &[PolyF32],
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        saturate_resonance: impl Fn(PolyF32) -> PolyF32,
        saturate_input: impl Fn(PolyF32) -> PolyF32,
    ) {
        let num_samples = audio_in.len();
        assert_eq!(audio_out.len(), num_samples);
        assert_eq!(cutoff_midi.len(), num_samples);
        if num_samples == 0 {
            return;
        }

        let mut current_resonance = self.resonance;
        let mut current_drive = self.drive;
        let mut current_peak1 = self.peak1_amount;
        let mut current_peak3 = self.peak3_amount;
        let mut current_peak5 = self.peak5_amount;

        self.setup(params);

        let tick_increment = 1.0 / num_samples as f32;
        let delta_resonance = (self.resonance - current_resonance) * tick_increment;
        let delta_drive = (self.drive - current_drive) * tick_increment;
        let delta_peak1 = (self.peak1_amount - current_peak1) * tick_increment;
        let delta_peak3 = (self.peak3_amount - current_peak3) * tick_increment;
        let delta_peak5 = (self.peak5_amount - current_peak5) * tick_increment;

        let base_midi = cutoff_midi[num_samples - 1];
        let base_frequency =
            math::midi_note_to_frequency(base_midi) * (1.0 / self.sample_rate);

        for i in 0..num_samples {
            let midi_delta = cutoff_midi[i] - base_midi;
            let frequency =
                (base_frequency * math::midi_offset_to_ratio(midi_delta)).min(PolyF32::ONE);
            let coefficient = one_pole_coefficient(frequency);

            current_resonance += delta_resonance;
            current_drive += delta_drive;
            current_peak1 += delta_peak1;
            current_peak3 += delta_peak3;
            current_peak5 += delta_peak5;

            self.tick(
                audio_in[i],
                coefficient,
                current_resonance,
                current_drive,
                current_peak1,
                current_peak3,
                current_peak5,
                &saturate_resonance,
                &saturate_input,
            );

            audio_out[i] = (audio_in[i] + self.invert_mult * self.allpass_output) * 0.5;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn tick(
        &mut self,
        audio_in: PolyF32,
        coefficient: PolyF32,
        resonance: PolyF32,
        drive: PolyF32,
        peak1: PolyF32,
        peak3: PolyF32,
        peak5: PolyF32,
        saturate_resonance: impl Fn(PolyF32) -> PolyF32,
        saturate_input: impl Fn(PolyF32) -> PolyF32,
    ) {
        let filter_state_lows = self
            .remove_lows_stage
            .tick_basic(self.allpass_output, (coefficient * CLEAR_RATIO).min(PolyF32::splat(0.9)));
        let filter_state_highs = self
            .remove_highs_stage
            .tick_basic(filter_state_lows, coefficient * (1.0 / CLEAR_RATIO));
        let filter_state = saturate_resonance(resonance * (filter_state_lows - filter_state_highs));

        let filter_input = (drive * audio_in).mul_add(self.invert_mult, filter_state);
        let mut all_pass_input = saturate_input(filter_input);

        for i in 0..PEAK_STAGE {
            let stage_out = self.stages[i].tick_basic(all_pass_input, coefficient);
            all_pass_input = all_pass_input.mul_add(stage_out, PolyF32::splat(-2.0));
        }
        let peak1_out = all_pass_input;

        for i in PEAK_STAGE..2 * PEAK_STAGE {
            let stage_out = self.stages[i].tick_basic(all_pass_input, coefficient);
            all_pass_input = all_pass_input.mul_add(stage_out, PolyF32::splat(-2.0));
        }
        let peak3_out = all_pass_input;

        for i in 2 * PEAK_STAGE..3 * PEAK_STAGE {
            let stage_out = self.stages[i].tick_basic(all_pass_input, coefficient);
            all_pass_input = all_pass_input.mul_add(stage_out, PolyF32::splat(-2.0));
        }
        let peak5_out = all_pass_input;

        let all_pass_output_1_3 = (peak1 * peak1_out).mul_add(peak3, peak3_out);
        self.allpass_output = all_pass_output_1_3.mul_add(peak5, peak5_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;

    #[test]
    fn output_finite_on_impulse_and_sine() {
        for clean in [true, false] {
            let mut filter = PhaserFilter::new(clean, SAMPLE_RATE);
            let params = PhaserFilterParams {
                resonance_percent: PolyF32::splat(0.8),
                pass_blend: PolyF32::splat(1.0),
                ..PhaserFilterParams::default()
            };
            let cutoff = vec![PolyF32::splat(60.0); 256];
            let mut input = vec![PolyF32::ZERO; 256];
            input[0] = PolyF32::ONE;
            for (i, value) in input.iter_mut().enumerate().skip(1) {
                let t = i as f32 / SAMPLE_RATE;
                *value = PolyF32::splat((2.0 * core::f32::consts::PI * 330.0 * t).sin() * 0.5);
            }
            let mut output = vec![PolyF32::ZERO; 256];
            for _ in 0..40 {
                filter.process(&params, &cutoff, &input, &mut output);
            }
            for sample in &output {
                assert!(sample.is_finite(), "clean={clean}");
                assert!(sample.abs().lane(0) < 8.0, "clean={clean} unbounded");
            }
        }
    }
}
