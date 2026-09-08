//! Phaser effect (port of Vital's `Phaser`): a triangle LFO sweeps the
//! cutoff of a [`PhaserFilter`], mixed with the dry signal.

use spinwave_poly::constants::{MAX_BUFFER_SIZE, MAX_OVERSAMPLE};
use spinwave_poly::utils::{cycle_offset_from_seconds, interpolate, stereo_split};
use spinwave_poly::{PolyF32, PolyMask, PolyU32};

use super::lanes::u32_gt;
use super::phaser_filter::{PhaserFilter, PhaserFilterParams};

const MAX_BLOCK: usize = MAX_BUFFER_SIZE * MAX_OVERSAMPLE;
const INT_MAX_F: f32 = i32::MAX as f32;
const UINT_MAX_F: f32 = u32::MAX as f32;

/// Block-rate phaser parameters.
#[derive(Clone, Copy, Debug)]
pub struct PhaserParams {
    /// Dry/wet mix in [0, 1] (linear crossfade like the reference).
    pub mix: PolyF32,
    /// LFO rate in Hz.
    pub rate: PolyF32,
    /// Feedback (resonance) amount in [0, 1].
    pub feedback_gain: PolyF32,
    /// Sweep center as a MIDI note.
    pub center_midi: PolyF32,
    /// Sweep depth in semitones.
    pub mod_depth: PolyF32,
    /// Stereo phase offset in [0, 1] (split Â± between L and R).
    pub phase_offset: PolyF32,
    /// Notch blend in [0, 2], forwarded to the filter.
    pub blend: PolyF32,
}

impl Default for PhaserParams {
    fn default() -> PhaserParams {
        PhaserParams {
            mix: PolyF32::ZERO,
            rate: PolyF32::splat(1.0),
            feedback_gain: PolyF32::ZERO,
            center_midi: PolyF32::splat(60.0),
            mod_depth: PolyF32::ZERO,
            phase_offset: PolyF32::ZERO,
            blend: PolyF32::splat(1.0),
        }
    }
}

pub struct Phaser {
    filter: PhaserFilter,
    cutoff: Vec<PolyF32>,
    sample_rate: f32,
    mix: PolyF32,
    mod_depth: PolyF32,
    phase_offset: PolyF32,
    phase: PolyU32,
    last_cutoff: PolyF32,
}

impl Phaser {
    pub fn new(sample_rate: f32) -> Phaser {
        Phaser {
            filter: PhaserFilter::new(true, sample_rate),
            cutoff: vec![PolyF32::ZERO; MAX_BLOCK],
            sample_rate,
            mix: PolyF32::ZERO,
            mod_depth: PolyF32::ZERO,
            phase_offset: PolyF32::ZERO,
            phase: PolyU32::ZERO,
            last_cutoff: PolyF32::ZERO,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.filter.set_sample_rate(sample_rate);
    }

    pub fn hard_reset(&mut self, params: &PhaserParams) {
        self.filter.reset(PolyMask::all_on());
        self.mod_depth = params.mod_depth;
        self.phase_offset = params.phase_offset;
    }

    /// The cutoff (MIDI) the sweep ended on, for UI display.
    pub fn last_cutoff(&self) -> PolyF32 {
        self.last_cutoff
    }

    /// Aligns the LFO phase to a host time in seconds.
    pub fn correct_to_time(&mut self, seconds: f64, rate: PolyF32) {
        let offset = cycle_offset_from_seconds(seconds, rate);
        self.phase = ((offset - 0.5) * UINT_MAX_F).to_i32_round()
            + PolyU32::splat((i32::MAX / 2) as u32);
    }

    pub fn process(&mut self, params: &PhaserParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let num_samples = audio_in.len();
        assert_eq!(audio_out.len(), num_samples);
        assert!(num_samples <= MAX_BLOCK);
        if num_samples == 0 {
            return;
        }

        let tick_delta = params.rate * (1.0 / self.sample_rate);

        let tick_inc = 1.0 / num_samples as f32;
        let phase_spread = self.phase_offset * stereo_split();
        let mut phase_offset = (phase_spread * INT_MAX_F).to_i32_round();
        self.phase_offset = params.phase_offset;
        let end_spread = self.phase_offset * stereo_split();
        let delta_spread = (end_spread - phase_spread) * tick_inc;
        let delta_phase_offset = (delta_spread * INT_MAX_F).to_i32_round();

        let mut current_mod_depth = self.mod_depth;
        self.mod_depth = params.mod_depth;
        let delta_depth = (self.mod_depth - current_mod_depth) * tick_inc;

        let current_phase = self.phase;
        let mut cutoff = core::mem::take(&mut self.cutoff);
        for cutoff_value in cutoff.iter_mut().take(num_samples) {
            phase_offset += delta_phase_offset;
            current_mod_depth += delta_depth;
            let shifted_phase = current_phase + phase_offset;
            let fold_mask = u32_gt(shifted_phase, PolyU32::splat(i32::MAX as u32));
            let folded_phase =
                fold_mask.select_u32(PolyU32::ZERO - shifted_phase, shifted_phase);
            let modulation = folded_phase.to_f32_signed() * (2.0 / INT_MAX_F) - 1.0;
            *cutoff_value = params.center_midi + modulation * current_mod_depth;
        }

        let filter_params = PhaserFilterParams {
            resonance_percent: params.feedback_gain,
            drive: PolyF32::ONE,
            pass_blend: params.blend,
            invert: false,
        };
        self.filter
            .process(&filter_params, &cutoff[..num_samples], audio_in, audio_out);
        self.last_cutoff = cutoff[num_samples - 1];
        self.cutoff = cutoff;

        self.phase += ((tick_delta * num_samples as f32) * UINT_MAX_F).to_i32_round();
        let mut current_mix = self.mix;
        self.mix = params.mix.clamp(0.0, 1.0);
        let delta_mix = (self.mix - current_mix) * (1.0 / num_samples as f32);

        for (out, &dry) in audio_out.iter_mut().zip(audio_in) {
            current_mix += delta_mix;
            *out = interpolate(dry, *out, current_mix);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 128;

    fn sine_block(start: usize, len: usize) -> Vec<PolyF32> {
        (0..len)
            .map(|i| {
                let t = (start + i) as f32 / SAMPLE_RATE;
                PolyF32::splat((2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.5)
            })
            .collect()
    }

    #[test]
    fn mix_zero_is_dry_passthrough() {
        let mut phaser = Phaser::new(SAMPLE_RATE);
        let params = PhaserParams {
            mod_depth: PolyF32::splat(24.0),
            feedback_gain: PolyF32::splat(0.5),
            rate: PolyF32::splat(2.0),
            ..PhaserParams::default()
        };
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..10 {
            let input = sine_block(block * BLOCK, BLOCK);
            phaser.process(&params, &input, &mut output);
            for (out, inp) in output.iter().zip(&input) {
                assert!((out.lane(0) - inp.lane(0)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn wet_output_finite_and_alters_signal() {
        let mut phaser = Phaser::new(SAMPLE_RATE);
        let params = PhaserParams {
            mix: PolyF32::ONE,
            mod_depth: PolyF32::splat(36.0),
            feedback_gain: PolyF32::splat(0.7),
            rate: PolyF32::splat(3.0),
            phase_offset: PolyF32::splat(0.3),
            ..PhaserParams::default()
        };
        let mut difference = 0.0f32;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..80 {
            let input = sine_block(block * BLOCK, BLOCK);
            phaser.process(&params, &input, &mut output);
            for (out, inp) in output.iter().zip(&input) {
                assert!(out.is_finite());
                assert!(out.abs().lane(0) < 4.0);
                difference = difference.max((out.lane(0) - inp.lane(0)).abs());
            }
        }
        assert!(difference > 0.01, "phaser did not alter the signal");
    }

    #[test]
    fn impulse_response_finite() {
        let mut phaser = Phaser::new(SAMPLE_RATE);
        let params = PhaserParams {
            mix: PolyF32::splat(0.8),
            mod_depth: PolyF32::splat(24.0),
            feedback_gain: PolyF32::splat(0.9),
            rate: PolyF32::splat(0.5),
            ..PhaserParams::default()
        };
        let mut input = vec![PolyF32::ZERO; BLOCK];
        input[0] = PolyF32::ONE;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for _ in 0..100 {
            phaser.process(&params, &input, &mut output);
            input[0] = PolyF32::ZERO;
            for sample in &output {
                assert!(sample.is_finite());
            }
        }
    }
}
