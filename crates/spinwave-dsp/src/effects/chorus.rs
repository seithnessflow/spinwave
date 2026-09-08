//! Chorus effect (port of Vital's `ChorusModule`): up to four pairs of
//! band-pass-filtered mono delay lines, swept by a block-rate sine LFO with
//! per-pair phase offsets, summed and equal-power mixed with the dry signal.
//!
//! The module-layer parameter plumbing is gone: tempo sync and control
//! scaling are resolved by the caller into [`ChorusParams`] once per block.

use spinwave_poly::constants::{MAX_BUFFER_SIZE, MAX_OVERSAMPLE, MAX_SAMPLE_RATE, PI};
use spinwave_poly::utils::{cycle_offset_from_seconds, interpolate};
use spinwave_poly::{math, PolyF32};

use super::delay::{DelayParams, DelayStyle, MultiDelay};
use super::lanes::{first_voice_mask, right_mask};

/// Maximum LFO delay modulation in seconds (C++ `kMaxChorusModulation`).
pub const MAX_CHORUS_MODULATION: f32 = 0.03;
/// Maximum base delay in seconds (C++ `kMaxChorusDelay`).
pub const MAX_CHORUS_DELAY: f32 = 0.08;
/// Maximum number of stereo voice pairs (C++ `kMaxDelayPairs`).
pub const MAX_DELAY_PAIRS: usize = 4;

const MAX_BLOCK: usize = MAX_BUFFER_SIZE * MAX_OVERSAMPLE;

/// Block-rate chorus parameters.
#[derive(Clone, Copy, Debug)]
pub struct ChorusParams {
    /// Number of active voice pairs, clamped to `1..=MAX_DELAY_PAIRS`
    /// (C++ `chorus_voices`).
    pub voices: usize,
    /// LFO rate in Hz; tempo sync is resolved by the caller.
    pub frequency: PolyF32,
    /// Delay feedback amount in [-1, 1].
    pub feedback: PolyF32,
    /// Dry/wet in [0, 1]; applied as an equal-power crossfade.
    pub wet: PolyF32,
    /// Delay band-pass filter center as a MIDI note (C++ `chorus_cutoff`).
    pub cutoff_midi: PolyF32,
    /// Delay band-pass filter spread in [0, 1] (C++ `chorus_spread`).
    pub spread: PolyF32,
    /// LFO depth in [0, 1], scaled by [`MAX_CHORUS_MODULATION`].
    pub mod_depth: PolyF32,
    /// Base delay of the first voice of each pair, in seconds.
    pub delay_1: PolyF32,
    /// Base delay of the second voice of each pair, in seconds.
    pub delay_2: PolyF32,
}

impl Default for ChorusParams {
    fn default() -> ChorusParams {
        ChorusParams {
            voices: MAX_DELAY_PAIRS,
            frequency: PolyF32::splat(0.5),
            feedback: PolyF32::splat(0.4),
            wet: PolyF32::splat(0.5),
            cutoff_midi: PolyF32::splat(60.0),
            spread: PolyF32::ONE,
            mod_depth: PolyF32::splat(0.5),
            delay_1: PolyF32::splat(0.002),
            delay_2: PolyF32::splat(0.008),
        }
    }
}

pub struct Chorus {
    delays: [MultiDelay; MAX_DELAY_PAIRS],
    sample_rate: f32,
    phase: PolyF32,
    wet: PolyF32,
    dry: PolyF32,
    last_num_voices: usize,
    delay_frequencies: [PolyF32; MAX_DELAY_PAIRS],
    delay_input: Vec<PolyF32>,
    delay_outputs: [Vec<PolyF32>; MAX_DELAY_PAIRS],
}

impl Chorus {
    pub fn new(sample_rate: f32) -> Chorus {
        // Sized for the largest supported sample rate, like the reference.
        let max_samples = (MAX_CHORUS_DELAY * MAX_SAMPLE_RATE as f32) as usize + 1;
        Chorus {
            delays: core::array::from_fn(|_| MultiDelay::new(max_samples, sample_rate)),
            sample_rate,
            phase: PolyF32::ZERO,
            wet: PolyF32::ZERO,
            dry: PolyF32::ZERO,
            // Starts at 0 so the first block resets every active delay,
            // matching the C++ `last_num_voices_` behavior.
            last_num_voices: 0,
            delay_frequencies: [PolyF32::ZERO; MAX_DELAY_PAIRS],
            delay_input: vec![PolyF32::ZERO; MAX_BLOCK],
            delay_outputs: core::array::from_fn(|_| vec![PolyF32::ZERO; MAX_BLOCK]),
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        for delay in &mut self.delays {
            delay.set_sample_rate(sample_rate);
        }
    }

    /// Reset on (re-)enable: clears the wet/dry ramps and all delay lines
    /// (C++ `ChorusModule::enable(true)`). The LFO phase and the voice-pair
    /// bookkeeping are deliberately kept, as in the reference.
    pub fn hard_reset(&mut self) {
        self.wet = PolyF32::ZERO;
        self.dry = PolyF32::ZERO;
        for delay in &mut self.delays {
            delay.hard_reset();
        }
    }

    /// Per-pair delay read frequency in Hz (`delay_status_outputs_` in the
    /// C++, used by the UI meters). Entries for disabled pairs keep their
    /// last value, matching the reference.
    pub fn delay_frequencies(&self) -> &[PolyF32; MAX_DELAY_PAIRS] {
        &self.delay_frequencies
    }

    /// Aligns the LFO phase to a host time in seconds
    /// (C++ `correctToTime`).
    pub fn correct_to_time(&mut self, seconds: f64, frequency: PolyF32) {
        self.phase = cycle_offset_from_seconds(seconds, frequency);
    }

    /// Resets delay lines that just became active when the pair count grows
    /// (C++ `getNextNumVoicePairs`). Shrinking resets nothing; the pairs are
    /// reset again when re-enabled.
    fn next_num_voice_pairs(&mut self, voices: usize) -> usize {
        let num_voice_pairs = voices.clamp(1, MAX_DELAY_PAIRS);
        for i in self.last_num_voices..num_voice_pairs {
            self.delays[i].hard_reset();
        }
        self.last_num_voices = num_voice_pairs;
        num_voice_pairs
    }

    pub fn process(&mut self, params: &ChorusParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let num_samples = audio_in.len();
        assert_eq!(num_samples, audio_out.len());
        assert!(num_samples <= MAX_BLOCK);
        if num_samples == 0 {
            return;
        }

        // Block-rate LFO: the phase advances once per block; the delay's own
        // frequency smoothing turns the stepped delay times into a sweep.
        let delta_phase = (params.frequency * num_samples as f32) * (1.0 / self.sample_rate);
        self.phase = (self.phase + delta_phase).fract();

        // The chorus only listens to the first voice: it is copied into both
        // voice slots so each delay pair processes the same stereo signal.
        for (dest, &source) in self.delay_input.iter_mut().zip(audio_in) {
            let sample = source & first_voice_mask();
            *dest = sample + sample.swap_voices();
        }

        let num_voices = self.next_num_voice_pairs(params.voices);

        // First voice lanes use delay 1, second voice lanes delay 2.
        let delay_time = first_voice_mask().select(params.delay_1, params.delay_2);
        let average_delay = (delay_time + delay_time.swap_voices()) * 0.5;

        for i in 0..num_voices {
            let pair_offset = i as f32 * 0.25 / num_voices as f32;
            let right_offset = PolyF32::splat(0.25) & right_mask();
            let phase = self.phase
                + right_offset
                + (PolyF32::splat(0.5) & !first_voice_mask())
                + pair_offset;

            let mod_depth = params.mod_depth * MAX_CHORUS_MODULATION;
            // Quirk kept from the reference: the sine is offset to 0.5..1.5,
            // so the "unmodulated" delay center sits above the base delay.
            let modulation = (phase * (PI * 2.0)).map(f32::sin) * 0.5 + 1.0;
            let delay_t = if i > 0 { i as f32 / (num_voices as f32 - 1.0) } else { 0.0 };
            let delay = modulation * mod_depth
                + interpolate(delay_time, average_delay, PolyF32::splat(delay_t));

            let delay_frequency = PolyF32::ONE / delay.max(PolyF32::splat(0.00001));
            self.delay_frequencies[i] = delay_frequency;

            let delay_params = DelayParams {
                period_samples: PolyF32::splat(self.sample_rate) / delay_frequency,
                feedback: params.feedback,
                wet: PolyF32::ONE,
                damping: PolyF32::ZERO,
                filter_cutoff_midi: params.cutoff_midi,
                filter_spread: params.spread,
                style: DelayStyle::Mono,
            };
            let input = &self.delay_input[..num_samples];
            let output = &mut self.delay_outputs[i][..num_samples];
            self.delays[i].process(&delay_params, input, output);
        }

        let mut current_wet = self.wet;
        let mut current_dry = self.dry;

        let wet_value = params.wet.clamp(0.0, 1.0);
        self.wet = math::equal_power_fade(wet_value);
        self.dry = math::equal_power_fade_inverse(wet_value);

        let tick_increment = 1.0 / num_samples as f32;
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;

        for out in audio_out.iter_mut() {
            *out = PolyF32::ZERO;
        }

        for i in 0..num_voices {
            let delay_out = &self.delay_outputs[i][..num_samples];
            for (out, &delayed) in audio_out.iter_mut().zip(delay_out) {
                let sample_out = delayed * 0.5;
                *out += sample_out + sample_out.swap_voices();
            }
        }

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            current_dry += delta_dry;
            current_wet += delta_wet;
            *out = current_dry * sample + current_wet * *out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 128;

    /// One block of a sine whose period (64 samples) divides the block, so
    /// every input block is identical; any change between output blocks must
    /// come from internal modulation.
    fn periodic_block() -> Vec<PolyF32> {
        (0..BLOCK)
            .map(|i| {
                let phase = 2.0 * core::f32::consts::PI * i as f32 / 64.0;
                PolyF32::splat(phase.sin() * 0.5)
            })
            .collect()
    }

    #[test]
    fn produces_movement_for_repeating_input() {
        let mut chorus = Chorus::new(SAMPLE_RATE);
        let params = ChorusParams {
            wet: PolyF32::ONE,
            mod_depth: PolyF32::ONE,
            frequency: PolyF32::splat(2.0),
            feedback: PolyF32::ZERO,
            ..ChorusParams::default()
        };
        let input = periodic_block();
        let mut output = vec![PolyF32::ZERO; BLOCK];
        let mut early = vec![PolyF32::ZERO; BLOCK];

        for block in 0..120 {
            chorus.process(&params, &input, &mut output);
            for sample in &output {
                assert!(sample.is_finite());
            }
            if block == 60 {
                early.copy_from_slice(&output);
            }
        }
        let difference: f32 = output
            .iter()
            .zip(&early)
            .map(|(a, b)| (a.lane(0) - b.lane(0)).abs())
            .sum();
        assert!(difference > 0.05, "no movement between blocks: {difference}");
    }

    #[test]
    fn dry_at_wet_zero_is_passthrough() {
        let mut chorus = Chorus::new(SAMPLE_RATE);
        let params = ChorusParams { wet: PolyF32::ZERO, ..ChorusParams::default() };
        let input = periodic_block();
        let mut output = vec![PolyF32::ZERO; BLOCK];
        // First block ramps wet/dry up from the reset values.
        for _ in 0..4 {
            chorus.process(&params, &input, &mut output);
        }
        for (out, inp) in output.iter().zip(&input) {
            assert!(
                (out.lane(0) - inp.lane(0)).abs() < 1e-3,
                "dry not passed through: {} vs {}",
                out.lane(0),
                inp.lane(0)
            );
        }
    }

    #[test]
    fn full_wet_has_no_dry_leak() {
        let mut chorus = Chorus::new(SAMPLE_RATE);
        let params = ChorusParams {
            wet: PolyF32::ONE,
            mod_depth: PolyF32::ZERO,
            feedback: PolyF32::ZERO,
            delay_1: PolyF32::splat(0.02),
            delay_2: PolyF32::splat(0.02),
            ..ChorusParams::default()
        };
        // Settle the wet/dry ramps and delay-time smoothing on silence.
        let silence = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for _ in 0..50 {
            chorus.process(&params, &silence, &mut output);
        }
        // An impulse must not appear immediately (no dry path at wet = 1).
        let mut input = vec![PolyF32::ZERO; BLOCK];
        input[0] = PolyF32::ONE;
        chorus.process(&params, &input, &mut output);
        for (i, sample) in output.iter().enumerate() {
            assert!(
                sample.abs().lane(0) < 1e-3,
                "dry leak at sample {i}: {}",
                sample.lane(0)
            );
        }
        // The wet signal shows up later (delay is roughly 20ms + modulation
        // center offset, well past one block).
        let mut wet_energy = 0.0f32;
        for _ in 0..20 {
            chorus.process(&params, &silence, &mut output);
            for sample in &output {
                wet_energy += sample.abs().lane(0);
            }
        }
        assert!(wet_energy > 0.05, "wet signal missing: {wet_energy}");
    }

    #[test]
    fn delay_frequency_readouts_active() {
        let mut chorus = Chorus::new(SAMPLE_RATE);
        let params = ChorusParams { voices: 3, ..ChorusParams::default() };
        let input = periodic_block();
        let mut output = vec![PolyF32::ZERO; BLOCK];
        chorus.process(&params, &input, &mut output);
        let frequencies = chorus.delay_frequencies();
        for pair in frequencies.iter().take(3) {
            assert!(pair.lane(0) > 0.0, "inactive readout for active pair");
        }
        assert_eq!(frequencies[3].lane(0), 0.0, "disabled pair readout moved");
    }
}
