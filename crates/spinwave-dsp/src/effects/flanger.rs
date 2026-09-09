//! Flanger effect (port of Vital's `FlangerModule`): a short stereo delay
//! (`ClampedUnfiltered` style, saturated feedback) swept by a block-rate
//! triangle LFO around a MIDI-note center, mixed inside the delay's own
//! wet/dry stage. The comb notches move as the delay time sweeps.

use spinwave_poly::utils::{cycle_offset_from_seconds, triangle_wave};
use spinwave_poly::PolyF32;

use crate::filters::filter_state::midi_note_to_frequency_precise;

use super::delay::{DelayParams, DelayStyle, StereoDelay};
use super::lanes::right_mask;

/// Control range of the center sweep in semitones (C++ `kMaxFlangerSemitoneOffset`).
pub const MAX_FLANGER_SEMITONE_OFFSET: f32 = 24.0;
/// Delay range covered by the center control, seconds (C++ `kFlangerDelayRange`).
pub const FLANGER_DELAY_RANGE: f32 = 0.01;
/// Center of the delay range, seconds (C++ `kFlangerCenter`).
pub const FLANGER_CENTER: f32 = FLANGER_DELAY_RANGE * 0.5 + 0.0005;
/// Minimum delay the modulation cannot cross, seconds (C++ `kModulationDelayBuffer`).
pub const MODULATION_DELAY_BUFFER: f32 = 0.0005;

/// Fixed delay-line size, in samples, regardless of sample rate â€” a quirk
/// kept from the reference (`kMaxSamples` in `FlangerModule::init`).
const MAX_SAMPLES: usize = 40000;
const MAX_FREQUENCY: f32 = 20000.0;

/// Block-rate flanger parameters.
#[derive(Clone, Copy, Debug)]
pub struct FlangerParams {
    /// LFO rate in Hz; tempo sync is resolved by the caller.
    pub frequency: PolyF32,
    /// Sweep center as a MIDI note (the delay time is one period of it).
    pub center_midi: PolyF32,
    /// Feedback amount in [-1, 1].
    pub feedback: PolyF32,
    /// Dry/wet in [0, 1] (equal-power fade inside the delay).
    pub wet: PolyF32,
    /// Modulation depth in [0, 1].
    pub mod_depth: PolyF32,
    /// Stereo LFO phase offset in [0, 1].
    pub phase_offset: PolyF32,
}

impl Default for FlangerParams {
    fn default() -> FlangerParams {
        FlangerParams {
            frequency: PolyF32::splat(2.0),
            center_midi: PolyF32::splat(64.0),
            feedback: PolyF32::splat(0.5),
            wet: PolyF32::splat(0.5),
            mod_depth: PolyF32::splat(0.5),
            phase_offset: PolyF32::ZERO,
        }
    }
}

pub struct Flanger {
    delay: StereoDelay,
    sample_rate: f32,
    phase: PolyF32,
    delay_frequency: PolyF32,
}

impl Flanger {
    pub fn new(sample_rate: f32) -> Flanger {
        Flanger {
            delay: StereoDelay::new(MAX_SAMPLES, sample_rate),
            sample_rate,
            phase: PolyF32::ZERO,
            delay_frequency: PolyF32::ZERO,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.delay.set_sample_rate(sample_rate);
    }

    pub fn hard_reset(&mut self) {
        self.delay.hard_reset();
    }

    /// Current delay read frequency in Hz (C++ `kFrequencyOutput`, the UI
    /// readout of where the comb sits).
    pub fn delay_frequency(&self) -> PolyF32 {
        self.delay_frequency
    }

    /// Aligns the LFO phase to a host time in seconds (C++ `correctToTime`).
    pub fn correct_to_time(&mut self, seconds: f64, frequency: PolyF32) {
        self.phase = cycle_offset_from_seconds(seconds, frequency);
    }

    pub fn process(&mut self, params: &FlangerParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let num_samples = audio_in.len();
        assert_eq!(num_samples, audio_out.len());
        if num_samples == 0 {
            return;
        }

        // Block-rate LFO, like the reference: one delay time per block, with
        // the delay's internal frequency smoothing gliding between blocks.
        let delta_phase = (params.frequency * num_samples as f32) / self.sample_rate;
        self.phase = (self.phase + delta_phase).fract();

        let phase_offset = params.phase_offset;
        let right_offset = phase_offset & right_mask();
        // Quirk kept: the left channel is shifted back by offset/2 and the
        // right forward by offset/2, keeping the pair centered on the phase.
        let phase_total = self.phase - phase_offset * 0.5 + right_offset;

        let modulation =
            params.mod_depth * (triangle_wave(phase_total) * 2.0 - 1.0) + 1.0;
        let delay = PolyF32::ONE / midi_note_to_frequency_precise(params.center_midi);
        let delay = (delay - MODULATION_DELAY_BUFFER) * modulation + MODULATION_DELAY_BUFFER;
        let delay_frequency = PolyF32::ONE / delay.max(PolyF32::splat(1.0 / MAX_FREQUENCY));
        self.delay_frequency = delay_frequency;

        let delay_params = DelayParams {
            period_samples: PolyF32::splat(self.sample_rate) / delay_frequency,
            feedback: params.feedback,
            wet: params.wet,
            style: DelayStyle::ClampedUnfiltered,
            ..DelayParams::default()
        };
        self.delay.process(&delay_params, audio_in, audio_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 128;

    fn sine_block(start: usize, freq: f32) -> Vec<PolyF32> {
        (0..BLOCK)
            .map(|i| {
                let t = (start + i) as f32 / SAMPLE_RATE;
                PolyF32::splat((2.0 * core::f32::consts::PI * freq * t).sin() * 0.5)
            })
            .collect()
    }

    #[test]
    fn delay_frequency_sweeps() {
        let mut flanger = Flanger::new(SAMPLE_RATE);
        let params = FlangerParams {
            frequency: PolyF32::splat(3.0),
            mod_depth: PolyF32::ONE,
            ..FlangerParams::default()
        };
        let input = sine_block(0, 440.0);
        let mut output = vec![PolyF32::ZERO; BLOCK];
        let mut min_frequency = f32::MAX;
        let mut max_frequency = f32::MIN;
        for _ in 0..80 {
            flanger.process(&params, &input, &mut output);
            let read = flanger.delay_frequency().lane(0);
            min_frequency = min_frequency.min(read);
            max_frequency = max_frequency.max(read);
        }
        assert!(
            max_frequency > 1.5 * min_frequency,
            "no sweep: {min_frequency}..{max_frequency}"
        );
    }

    #[test]
    fn notches_sweep_through_a_tone() {
        // With a 50% wet comb, a fixed tone moves in and out of the sweeping
        // notches, so the per-block output level must vary substantially.
        let mut flanger = Flanger::new(SAMPLE_RATE);
        let params = FlangerParams {
            frequency: PolyF32::splat(1.0),
            mod_depth: PolyF32::ONE,
            feedback: PolyF32::ZERO,
            wet: PolyF32::splat(0.5),
            ..FlangerParams::default()
        };
        let mut min_rms = f32::MAX;
        let mut max_rms = 0.0f32;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..200 {
            let input = sine_block(block * BLOCK, 1500.0);
            flanger.process(&params, &input, &mut output);
            if block < 40 {
                continue; // let the delay-period smoothing settle
            }
            let mut sum = 0.0f32;
            for sample in &output {
                assert!(sample.is_finite());
                sum += sample.lane(0) * sample.lane(0);
            }
            let rms = (sum / BLOCK as f32).sqrt();
            min_rms = min_rms.min(rms);
            max_rms = max_rms.max(rms);
        }
        assert!(
            max_rms > 1.4 * min_rms,
            "comb level did not vary: {min_rms}..{max_rms}"
        );
    }

    /// Equal-power dry gain for a wet control value, as
    /// `futils::equalPowerFadeInverse` computes it (`sin1((t + 1) / 4)`).
    fn expected_dry(wet: f32) -> f32 {
        let adjusted = 0.5 - (wet + 1.0) * 0.25;
        let approx = adjusted * (8.0 - 16.0 * adjusted.abs());
        approx * (0.776 + 0.224 * approx.abs())
    }

    #[test]
    fn wet_beyond_the_parameter_range_is_not_clamped() {
        // Vital's `flanger_dry_wet` parameter tops out at 0.5, but nothing
        // in the engine enforces that: `Value::set` does not clamp and
        // `Delay::processWithInput` only does `utils::clamp(wet, 0, 1)`.
        // The golden bench's fx_flanger case drives the control to 0.8, so
        // the effect must reach the 0.8 mix (dry 0.308) and not the 0.5 one
        // (dry 0.708). The delay line is still silent this early - the
        // period smoothing starts at 2 Hz, i.e. thousands of samples - so
        // the output is the dry path alone.
        for wet in [0.5f32, 0.8] {
            let mut flanger = Flanger::new(SAMPLE_RATE);
            let params = FlangerParams {
                wet: PolyF32::splat(wet),
                feedback: PolyF32::ZERO,
                ..FlangerParams::default()
            };
            let mut output = vec![PolyF32::ZERO; BLOCK];
            let mut input = Vec::new();
            for block in 0..3 {
                input = sine_block(block * BLOCK, 440.0);
                flanger.process(&params, &input, &mut output);
            }
            let dry = expected_dry(wet);
            for (out, inp) in output.iter().zip(&input) {
                let expected = dry * inp.lane(0);
                assert!(
                    (out.lane(0) - expected).abs() < 1e-4,
                    "wet {wet}: got {} want {expected}",
                    out.lane(0)
                );
            }
        }
        assert!((expected_dry(0.8) - 0.3084).abs() < 1e-3);
        assert!((expected_dry(0.5) - 0.708).abs() < 1e-3);
    }

    #[test]
    fn dry_at_wet_zero_is_passthrough() {
        let mut flanger = Flanger::new(SAMPLE_RATE);
        let params = FlangerParams { wet: PolyF32::ZERO, ..FlangerParams::default() };
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..6 {
            let input = sine_block(block * BLOCK, 440.0);
            flanger.process(&params, &input, &mut output);
            if block > 1 {
                for (out, inp) in output.iter().zip(&input) {
                    assert!((out.lane(0) - inp.lane(0)).abs() < 1e-3);
                }
            }
        }
    }
}
