//! Bode-style single-sideband frequency shifter.
//!
//! The folded bus signal is turned into an analytic signal with the classic
//! IIR phase-difference network published by Olli Niemitalo: two parallel
//! cascades of four second-order allpass sections (8th order total) whose
//! outputs sit ~90 degrees apart across most of the audio band; the
//! quadrature branch takes one extra sample of delay. The analytic signal is
//! then multiplied by a quadrature oscillator at `shift_hz` and the real part
//! is kept, shifting every partial by the same amount in Hz (not a pitch
//! shift: harmonic relationships are destroyed, which is the point).
//!
//! Per-sample and allocation-free; the oscillator frequency and the dry/wet
//! gains are ramped across each block so shift automation is click-free.

use spinwave_poly::{math, PolyF32};

use super::lanes::first_voice_mask;

/// Maximum |shift| in Hz.
pub const MAX_SHIFT_HZ: f32 = 5000.0;

/// Olli Niemitalo's allpass coefficients (the `a` of each section
/// `H(z) = (a^2 - z^-2) / (1 - a^2 z^-2)`), in-phase branch.
const IN_PHASE_A: [f32; 4] = [
    0.692_387_8, // 0.6923877778065359 rounded to f32
    0.936_065_4,
    0.988_229_5,
    0.998_748_8,
];

/// Quadrature branch coefficients (this branch also gets a one-sample delay).
const QUADRATURE_A: [f32; 4] = [
    0.402_192_1,
    0.856_171_1,
    0.972_291,
    0.995_288_5,
];

/// Block-rate frequency shifter parameters.
#[derive(Clone, Copy, Debug)]
pub struct FrequencyShifterParams {
    /// Shift in Hz, clamped to +/-[`MAX_SHIFT_HZ`]. Positive shifts up.
    pub shift_hz: f32,
    /// Dry/wet in [0, 1]; applied as an equal-power crossfade.
    pub mix: f32,
    /// When true the right lanes shift by `-shift_hz` (barberpole stereo).
    pub stereo: bool,
}

impl Default for FrequencyShifterParams {
    fn default() -> FrequencyShifterParams {
        FrequencyShifterParams {
            shift_hz: 0.0,
            mix: 1.0,
            stereo: false,
        }
    }
}

/// One second-order allpass section `y[n] = c (x[n] + y[n-2]) - x[n-2]`.
#[derive(Clone, Copy, Default)]
struct AllpassSection {
    x1: PolyF32,
    x2: PolyF32,
    y1: PolyF32,
    y2: PolyF32,
}

impl AllpassSection {
    #[inline(always)]
    fn tick(&mut self, x: PolyF32, coefficient: f32) -> PolyF32 {
        let y = (x + self.y2) * coefficient - self.x2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        *self = AllpassSection::default();
    }
}

/// Single-sideband frequency shifter; see the module docs for the design.
pub struct FrequencyShifter {
    sample_rate: f32,
    /// Squared `a` values: the actual section coefficients.
    in_phase_c: [f32; 4],
    quadrature_c: [f32; 4],
    in_phase: [AllpassSection; 4],
    quadrature: [AllpassSection; 4],
    /// One-sample delay completing the quadrature branch.
    quadrature_delay: PolyF32,
    /// Oscillator phase in cycles, per lane, kept in [0, 1).
    phase: PolyF32,
    /// Phase increment in cycles/sample, per lane (ramped per block).
    increment: PolyF32,
    dry: PolyF32,
    wet: PolyF32,
}

impl FrequencyShifter {
    pub fn new(sample_rate: f32) -> FrequencyShifter {
        FrequencyShifter {
            sample_rate,
            in_phase_c: IN_PHASE_A.map(|a| a * a),
            quadrature_c: QUADRATURE_A.map(|a| a * a),
            in_phase: [AllpassSection::default(); 4],
            quadrature: [AllpassSection::default(); 4],
            quadrature_delay: PolyF32::ZERO,
            phase: PolyF32::ZERO,
            increment: PolyF32::ZERO,
            dry: PolyF32::ZERO,
            wet: PolyF32::ZERO,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn hard_reset(&mut self) {
        for section in self.in_phase.iter_mut().chain(&mut self.quadrature) {
            section.reset();
        }
        self.quadrature_delay = PolyF32::ZERO;
        self.phase = PolyF32::ZERO;
        self.increment = PolyF32::ZERO;
        self.dry = PolyF32::ZERO;
        self.wet = PolyF32::ZERO;
    }

    /// Processes one block (any length; designed for blocks of at most 1024
    /// samples). Per-sample, allocation-free.
    pub fn process(
        &mut self,
        params: &FrequencyShifterParams,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        let num_samples = audio_in.len();
        assert_eq!(audio_out.len(), num_samples);
        if num_samples == 0 {
            return;
        }
        let tick_increment = 1.0 / num_samples as f32;

        let mix = PolyF32::splat(params.mix.clamp(0.0, 1.0));
        let mut current_wet = self.wet;
        let mut current_dry = self.dry;
        self.wet = math::equal_power_fade(mix);
        self.dry = math::equal_power_fade_inverse(mix);
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;

        let shift = params.shift_hz.clamp(-MAX_SHIFT_HZ, MAX_SHIFT_HZ);
        let signs = if params.stereo {
            PolyF32::stereo(1.0, -1.0)
        } else {
            PolyF32::ONE
        };
        let target_increment = signs * (shift / self.sample_rate);
        // Ramp the oscillator frequency across the block; the phase itself is
        // always continuous, so shift changes never click.
        let mut current_increment = self.increment;
        let delta_increment = (target_increment - current_increment) * tick_increment;
        self.increment = target_increment;

        let mut phase = self.phase;
        let quarter = PolyF32::splat(0.25);

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            let mut input = sample & first_voice_mask();
            input += input.swap_voices();

            let mut in_phase = input;
            for (section, &c) in self.in_phase.iter_mut().zip(&self.in_phase_c) {
                in_phase = section.tick(in_phase, c);
            }
            let mut quadrature = input;
            for (section, &c) in self.quadrature.iter_mut().zip(&self.quadrature_c) {
                quadrature = section.tick(quadrature, c);
            }
            let quadrature = core::mem::replace(&mut self.quadrature_delay, quadrature);

            current_increment += delta_increment;
            phase = (phase + current_increment).fract();
            let sin = math::sin1(phase);
            let cos = math::sin1((phase + quarter).fract());

            // Re(analytic * e^{j theta}): shifts up for positive increments,
            // down for negative ones (sin flips sign with the increment).
            // In this network the delayed quadrature branch *leads* the
            // in-phase branch by 90 degrees, hence the `+`.
            let shifted = in_phase * cos + quadrature * sin;

            current_dry += delta_dry;
            current_wet += delta_wet;
            *out = current_dry * input + current_wet * shifted;
        }

        self.phase = phase;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 128;

    /// Goertzel amplitude of `samples` at `frequency` (rectangular window;
    /// the tests pick windows holding a whole number of cycles).
    fn goertzel(samples: &[f32], frequency: f32, sample_rate: f32) -> f32 {
        let omega = 2.0 * std::f64::consts::PI * frequency as f64 / sample_rate as f64;
        let coefficient = 2.0 * omega.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &sample in samples {
            let s0 = sample as f64 + coefficient * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = (s1 * s1 + s2 * s2 - coefficient * s1 * s2).max(0.0);
        2.0 * power.sqrt() as f32 / samples.len() as f32
    }

    /// Feeds a 440 Hz sine for `total` samples, returning the requested lane.
    fn run_sine(
        shifter: &mut FrequencyShifter,
        params: &FrequencyShifterParams,
        total: usize,
        lane: usize,
    ) -> Vec<f32> {
        let mut collected = Vec::with_capacity(total);
        let mut output = vec![PolyF32::ZERO; BLOCK];
        let mut n = 0;
        while n < total {
            let count = BLOCK.min(total - n);
            let input: Vec<PolyF32> = (0..count)
                .map(|i| {
                    let t = (n + i) as f32 / SAMPLE_RATE;
                    PolyF32::splat((2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.5)
                })
                .collect();
            shifter.process(params, &input, &mut output[..count]);
            for sample in &output[..count] {
                assert!(sample.is_finite());
                collected.push(sample.lane(lane));
            }
            n += count;
        }
        collected
    }

    /// 0.2 s warm-up (filters + ramps settle), then a 0.5 s analysis window:
    /// 22050 samples hold whole cycles of 440, 340 and 540 Hz exactly.
    const WARMUP: usize = 8820;
    const WINDOW: usize = 22050;

    #[test]
    fn shifts_sine_up_with_image_rejection() {
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let params = FrequencyShifterParams {
            shift_hz: 100.0,
            mix: 1.0,
            stereo: false,
        };
        let output = run_sine(&mut shifter, &params, WARMUP + WINDOW, 0);
        let window = &output[WARMUP..];

        let target = goertzel(window, 540.0, SAMPLE_RATE);
        let image = goertzel(window, 340.0, SAMPLE_RATE);
        let original = goertzel(window, 440.0, SAMPLE_RATE);
        assert!(target > 0.25, "shifted partial too weak: {target}");
        assert!(
            image < target * 0.1,
            "image rejection below 20 dB: target {target}, image {image}"
        );
        assert!(
            original < target * 0.1,
            "carrier leak too strong: target {target}, at 440 Hz {original}"
        );
    }

    #[test]
    fn negative_shift_moves_down() {
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let params = FrequencyShifterParams {
            shift_hz: -100.0,
            mix: 1.0,
            stereo: false,
        };
        let output = run_sine(&mut shifter, &params, WARMUP + WINDOW, 0);
        let window = &output[WARMUP..];

        let target = goertzel(window, 340.0, SAMPLE_RATE);
        let image = goertzel(window, 540.0, SAMPLE_RATE);
        assert!(target > 0.25, "down-shifted partial too weak: {target}");
        assert!(image < target * 0.1, "image rejection below 20 dB");
    }

    #[test]
    fn stereo_mode_inverts_right_lane_shift() {
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let params = FrequencyShifterParams {
            shift_hz: 100.0,
            mix: 1.0,
            stereo: true,
        };
        // Collect both lanes in one pass.
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut output = vec![PolyF32::ZERO; BLOCK];
        let total = WARMUP + WINDOW;
        let mut n = 0;
        while n < total {
            let count = BLOCK.min(total - n);
            let input: Vec<PolyF32> = (0..count)
                .map(|i| {
                    let t = (n + i) as f32 / SAMPLE_RATE;
                    PolyF32::splat((2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.5)
                })
                .collect();
            shifter.process(&params, &input, &mut output[..count]);
            for sample in &output[..count] {
                left.push(sample.lane(0));
                right.push(sample.lane(1));
            }
            n += count;
        }

        let left_up = goertzel(&left[WARMUP..], 540.0, SAMPLE_RATE);
        let left_down = goertzel(&left[WARMUP..], 340.0, SAMPLE_RATE);
        let right_up = goertzel(&right[WARMUP..], 540.0, SAMPLE_RATE);
        let right_down = goertzel(&right[WARMUP..], 340.0, SAMPLE_RATE);
        assert!(left_up > 0.25 && left_down < left_up * 0.1);
        assert!(right_down > 0.25 && right_up < right_down * 0.1);
    }

    #[test]
    fn zero_mix_is_transparent() {
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let params = FrequencyShifterParams {
            shift_hz: 1234.0,
            mix: 0.0,
            stereo: true,
        };
        let total = 8 * BLOCK;
        let output = run_sine(&mut shifter, &params, total, 0);
        // Skip the first block: the dry gain ramps in from the reset state.
        for (n, &sample) in output.iter().enumerate().skip(BLOCK) {
            let t = n as f32 / SAMPLE_RATE;
            let expected = (2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.5;
            assert!(
                (sample - expected).abs() < 1e-4,
                "sample {n}: mix=0 not transparent"
            );
        }
    }

    #[test]
    fn extreme_and_changing_shifts_stay_finite() {
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let mut params = FrequencyShifterParams {
            shift_hz: MAX_SHIFT_HZ,
            mix: 0.7,
            stereo: true,
        };
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..200 {
            // Sweep the shift hard from +5 kHz to -5 kHz and back.
            params.shift_hz = MAX_SHIFT_HZ * ((block as f32 * 0.37).sin());
            let input: Vec<PolyF32> = (0..BLOCK)
                .map(|i| {
                    let t = (block * BLOCK + i) as f32 / SAMPLE_RATE;
                    PolyF32::splat((2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.8)
                })
                .collect();
            shifter.process(&params, &input, &mut output);
            for sample in &output {
                assert!(sample.is_finite());
                assert!(sample.abs().lane(0) < 4.0);
            }
        }
        shifter.hard_reset();
        let silent = vec![PolyF32::ZERO; BLOCK];
        shifter.process(&params, &silent, &mut output);
        // After a reset with silent input, only the (ramping) dry path of
        // silence remains: exact zeros.
        for sample in &output {
            assert!(sample.abs().lane(0) < 1e-6);
        }
    }
}
