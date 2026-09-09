//! Bode-style single-sideband frequency shifter.
//!
//! The folded bus signal is first DC-blocked (an offset would otherwise be
//! shifted into an audible tone at |shift| Hz) and, when shifting upward,
//! low-passed by a 4th-order Butterworth at `fs/2 - |shift|` so nothing is
//! pushed past Nyquist and folded back. It is then turned into an analytic
//! signal with the classic IIR phase-difference network published by Olli
//! Niemitalo: two parallel cascades of four second-order allpass sections
//! (8th order total) whose outputs sit 90 degrees (+/-0.7) apart from
//! 20 Hz to 20 kHz; the in-phase branch takes one extra sample of delay. The
//! analytic signal is multiplied by an exact quadrature carrier — a complex
//! rotation advanced once per sample (a second rotator carries the
//! within-block frequency ramp, so the carrier is a clean linear chirp with
//! no parasitic images) — and the real part is kept, shifting every partial
//! by the same amount in Hz (not a pitch shift: harmonic relationships are
//! destroyed, which is the point).
//!
//! Per-sample and allocation-free; the carrier frequency and the dry/wet
//! gains are ramped across each block so shift automation is click-free.

use spinwave_poly::{math, PolyF32, PolyMask};

use super::lanes::first_voice_mask;

/// Maximum |shift| in Hz.
pub const MAX_SHIFT_HZ: f32 = 5000.0;
/// Corner of the input DC blocker in Hz.
pub const DC_BLOCK_HZ: f32 = 5.0;

/// Olli Niemitalo's allpass coefficients (the `a` of each section
/// `H(z) = (a^2 - z^-2) / (1 - a^2 z^-2)`), in-phase branch (this branch
/// also gets the one-sample delay).
const IN_PHASE_A: [f32; 4] = [
    0.692_387_8, // 0.6923877778065359 rounded to f32
    0.936_065_4,
    0.988_229_5,
    0.998_748_8,
];

/// Quadrature branch coefficients (no extra delay; leads the delayed
/// in-phase branch by 90 degrees).
const QUADRATURE_A: [f32; 4] = [
    0.402_192_1,
    0.856_171_1,
    0.972_291,
    0.995_288_5,
];

/// Q values of the two biquads forming a 4th-order Butterworth low-pass.
const BUTTERWORTH_Q: [f32; 2] = [0.541_196_1, 1.306_563];

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

/// Transposed direct-form II biquad with block-rate scalar coefficients.
#[derive(Clone, Copy, Default)]
struct Biquad {
    s1: PolyF32,
    s2: PolyF32,
}

/// Normalised biquad coefficients (`a0 == 1`).
#[derive(Clone, Copy)]
struct BiquadCoefficients {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl BiquadCoefficients {
    /// RBJ low-pass at `cutoff` Hz.
    fn low_pass(cutoff: f32, sample_rate: f32, q: f32) -> BiquadCoefficients {
        let omega = 2.0 * core::f32::consts::PI * cutoff / sample_rate;
        let (sin, cos) = omega.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b1 = (1.0 - cos) / a0;
        BiquadCoefficients {
            b0: 0.5 * b1,
            b1,
            b2: 0.5 * b1,
            a1: -2.0 * cos / a0,
            a2: (1.0 - alpha) / a0,
        }
    }
}

impl Biquad {
    #[inline(always)]
    fn tick(&mut self, x: PolyF32, c: &BiquadCoefficients) -> PolyF32 {
        let y = x * c.b0 + self.s1;
        self.s1 = x * c.b1 - y * c.a1 + self.s2;
        self.s2 = x * c.b2 - y * c.a2;
        y
    }

    fn reset(&mut self) {
        *self = Biquad::default();
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
    /// One-sample delay completing the in-phase branch.
    in_phase_delay: PolyF32,
    /// DC blocker state (`y = x - x1 + R y1`).
    dc_x1: PolyF32,
    dc_y1: PolyF32,
    dc_coefficient: f32,
    /// Anti-alias low-pass (two cascaded biquads), engaged on upward shifts.
    anti_alias: [Biquad; 2],
    /// Carrier `e^{j theta}` per lane, advanced by one rotation per sample.
    carrier_re: PolyF32,
    carrier_im: PolyF32,
    /// Carrier increment in cycles/sample, per lane (ramped per block).
    increment: PolyF32,
    dry: PolyF32,
    wet: PolyF32,
}

impl FrequencyShifter {
    pub fn new(sample_rate: f32) -> FrequencyShifter {
        let mut shifter = FrequencyShifter {
            sample_rate,
            in_phase_c: IN_PHASE_A.map(|a| a * a),
            quadrature_c: QUADRATURE_A.map(|a| a * a),
            in_phase: [AllpassSection::default(); 4],
            quadrature: [AllpassSection::default(); 4],
            in_phase_delay: PolyF32::ZERO,
            dc_x1: PolyF32::ZERO,
            dc_y1: PolyF32::ZERO,
            dc_coefficient: 0.0,
            anti_alias: [Biquad::default(); 2],
            carrier_re: PolyF32::ONE,
            carrier_im: PolyF32::ZERO,
            increment: PolyF32::ZERO,
            dry: PolyF32::ZERO,
            wet: PolyF32::ZERO,
        };
        shifter.set_sample_rate(sample_rate);
        shifter
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.dc_coefficient =
            1.0 - 2.0 * core::f32::consts::PI * DC_BLOCK_HZ / sample_rate.max(1.0);
    }

    pub fn hard_reset(&mut self) {
        for section in self.in_phase.iter_mut().chain(&mut self.quadrature) {
            section.reset();
        }
        self.in_phase_delay = PolyF32::ZERO;
        self.dc_x1 = PolyF32::ZERO;
        self.dc_y1 = PolyF32::ZERO;
        for biquad in &mut self.anti_alias {
            biquad.reset();
        }
        self.carrier_re = PolyF32::ONE;
        self.carrier_im = PolyF32::ZERO;
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
        // Ramp the carrier frequency across the block. The carrier is a
        // complex rotation z <- z * w; w itself rotates by a fixed step so
        // the frequency sweeps linearly and the phase stays continuous.
        let start_increment = self.increment;
        let delta_increment = (target_increment - start_increment) * tick_increment;
        self.increment = target_increment;
        let tau = 2.0 * core::f32::consts::PI;
        let first_increment = start_increment + delta_increment;
        let mut w_re = (first_increment * tau).map(f32::cos);
        let mut w_im = (first_increment * tau).map(f32::sin);
        let step_re = (delta_increment * tau).map(f32::cos);
        let step_im = (delta_increment * tau).map(f32::sin);
        let mut z_re = self.carrier_re;
        let mut z_im = self.carrier_im;

        // Anti-alias guard for lanes shifting upward: keep everything below
        // Nyquist - shift. The filter always runs (state stays continuous)
        // and is only selected into the lanes that need it.
        let up_mask: PolyMask = target_increment.gt(PolyF32::ZERO);
        let cutoff = (0.5 * self.sample_rate - shift.abs())
            .clamp(0.05 * self.sample_rate, 0.49 * self.sample_rate);
        let anti_alias_c = [
            BiquadCoefficients::low_pass(cutoff, self.sample_rate, BUTTERWORTH_Q[0]),
            BiquadCoefficients::low_pass(cutoff, self.sample_rate, BUTTERWORTH_Q[1]),
        ];
        let dc_coefficient = self.dc_coefficient;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            let mut input = sample & first_voice_mask();
            input += input.swap_voices();

            // DC blocker.
            let blocked = input - self.dc_x1 + self.dc_y1 * dc_coefficient;
            self.dc_x1 = input;
            self.dc_y1 = blocked;

            // Anti-alias low-pass, selected per lane.
            let mut filtered = blocked;
            for (biquad, c) in self.anti_alias.iter_mut().zip(&anti_alias_c) {
                filtered = biquad.tick(filtered, c);
            }
            let network_in = up_mask.select(filtered, blocked);

            let mut in_phase = network_in;
            for (section, &c) in self.in_phase.iter_mut().zip(&self.in_phase_c) {
                in_phase = section.tick(in_phase, c);
            }
            // The one-sample delay completes the in-phase branch: with it
            // the quadrature branch leads by 90 degrees (+/-0.7) from
            // 20 Hz to 20 kHz; on the other branch the error grows to tens
            // of degrees above 200 Hz.
            let in_phase = core::mem::replace(&mut self.in_phase_delay, in_phase);
            let mut quadrature = network_in;
            for (section, &c) in self.quadrature.iter_mut().zip(&self.quadrature_c) {
                quadrature = section.tick(quadrature, c);
            }

            // Advance the carrier: z *= w, then w *= step (linear chirp).
            let next_re = z_re * w_re - z_im * w_im;
            let next_im = z_re * w_im + z_im * w_re;
            z_re = next_re;
            z_im = next_im;
            let w_next_re = w_re * step_re - w_im * step_im;
            let w_next_im = w_re * step_im + w_im * step_re;
            w_re = w_next_re;
            w_im = w_next_im;

            // Re(analytic * e^{j theta}): shifts up for positive increments,
            // down for negative ones (sin flips sign with the increment).
            // The quadrature branch *leads* the delayed in-phase branch by
            // 90 degrees, hence the `+`.
            let shifted = in_phase * z_re + quadrature * z_im;

            current_dry += delta_dry;
            current_wet += delta_wet;
            *out = current_dry * input + current_wet * shifted;
        }

        // Re-normalise the carrier once per block (one Newton step of
        // 1/sqrt) so rounding never lets it drift off the unit circle.
        let norm2 = z_re * z_re + z_im * z_im;
        let correction = PolyF32::splat(1.5) - norm2 * 0.5;
        self.carrier_re = z_re * correction;
        self.carrier_im = z_im * correction;
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

    /// Feeds `signal(n)` for `total` samples, returning the requested lane.
    fn run_signal(
        shifter: &mut FrequencyShifter,
        params: &FrequencyShifterParams,
        total: usize,
        lane: usize,
        signal: impl Fn(usize) -> f32,
    ) -> Vec<f32> {
        let mut collected = Vec::with_capacity(total);
        let mut output = vec![PolyF32::ZERO; BLOCK];
        let mut n = 0;
        while n < total {
            let count = BLOCK.min(total - n);
            let input: Vec<PolyF32> =
                (0..count).map(|i| PolyF32::splat(signal(n + i))).collect();
            shifter.process(params, &input, &mut output[..count]);
            for sample in &output[..count] {
                assert!(sample.is_finite());
                collected.push(sample.lane(lane));
            }
            n += count;
        }
        collected
    }

    fn sine_440(n: usize) -> f32 {
        let t = n as f32 / SAMPLE_RATE;
        (2.0 * core::f32::consts::PI * 440.0 * t).sin() * 0.5
    }

    /// Feeds a 440 Hz sine for `total` samples, returning the requested lane.
    fn run_sine(
        shifter: &mut FrequencyShifter,
        params: &FrequencyShifterParams,
        total: usize,
        lane: usize,
    ) -> Vec<f32> {
        run_signal(shifter, params, total, lane, sine_440)
    }

    /// 0.2 s warm-up (filters + ramps settle), then a 0.5 s analysis window:
    /// 22050 samples hold whole cycles of 440, 340, 540 and 100 Hz exactly.
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
            image < target * 0.01,
            "image rejection below 40 dB: target {target}, image {image}"
        );
        assert!(
            original < target * 0.01,
            "carrier leak above -40 dB: target {target}, at 440 Hz {original}"
        );
        // The exact carrier leaves no harmonic images either.
        let harmonic = goertzel(window, 3.0 * 540.0, SAMPLE_RATE);
        assert!(harmonic < target * 0.01, "carrier harmonic image: {harmonic}");
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
        assert!(image < target * 0.01, "image rejection below 40 dB");
    }

    #[test]
    fn dc_offset_does_not_become_a_tone() {
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let params = FrequencyShifterParams {
            shift_hz: 100.0,
            mix: 1.0,
            stereo: false,
        };
        let output = run_signal(&mut shifter, &params, WARMUP + WINDOW, 0, |n| {
            sine_440(n) + 0.3
        });
        let window = &output[WARMUP..];
        let target = goertzel(window, 540.0, SAMPLE_RATE);
        let dc_tone = goertzel(window, 100.0, SAMPLE_RATE);
        assert!(target > 0.25);
        assert!(
            dc_tone < target * 0.01,
            "DC offset shifted into a tone: {dc_tone} vs target {target}"
        );
    }

    #[test]
    fn upward_shift_guards_against_nyquist_foldover() {
        // 21 kHz + 5 kHz would land at 26 kHz and fold to 18.1 kHz; the
        // anti-alias low-pass (cutoff fs/2 - 5 kHz) must attenuate it well
        // below the unguarded 0.5 amplitude.
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let params = FrequencyShifterParams {
            shift_hz: MAX_SHIFT_HZ,
            mix: 1.0,
            stereo: false,
        };
        let output = run_signal(&mut shifter, &params, WARMUP + WINDOW, 0, |n| {
            let t = n as f32 / SAMPLE_RATE;
            (2.0 * core::f32::consts::PI * 21000.0 * t).sin() * 0.5
        });
        let window = &output[WARMUP..];
        let folded = goertzel(window, SAMPLE_RATE - 26000.0, SAMPLE_RATE);
        assert!(folded < 0.3, "folded image not attenuated: {folded}");

        // Downward shifts are not filtered: a 21 kHz tone shifted down by
        // 5 kHz keeps (most of) its level at 16 kHz.
        let mut shifter = FrequencyShifter::new(SAMPLE_RATE);
        let params = FrequencyShifterParams {
            shift_hz: -MAX_SHIFT_HZ,
            mix: 1.0,
            stereo: false,
        };
        let output = run_signal(&mut shifter, &params, WARMUP + WINDOW, 0, |n| {
            let t = n as f32 / SAMPLE_RATE;
            (2.0 * core::f32::consts::PI * 21000.0 * t).sin() * 0.5
        });
        let down = goertzel(&output[WARMUP..], 16000.0, SAMPLE_RATE);
        assert!(down > 0.35, "downward shift was filtered: {down}");
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
            let input: Vec<PolyF32> = (0..count).map(|i| PolyF32::splat(sine_440(n + i))).collect();
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
        assert!(left_up > 0.25 && left_down < left_up * 0.01);
        assert!(right_down > 0.25 && right_up < right_down * 0.01);
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
            let expected = sine_440(n);
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
        for block in 0..2000 {
            // Sweep the shift hard from +5 kHz to -5 kHz and back.
            params.shift_hz = MAX_SHIFT_HZ * ((block as f32 * 0.37).sin());
            let input: Vec<PolyF32> = (0..BLOCK)
                .map(|i| PolyF32::splat(sine_440(block * BLOCK + i) * 1.6))
                .collect();
            shifter.process(&params, &input, &mut output);
            for sample in &output {
                assert!(sample.is_finite());
                assert!(sample.abs().lane(0) < 4.0);
            }
        }
        // The carrier must still sit on the unit circle after a long run.
        let norm2 = shifter.carrier_re * shifter.carrier_re
            + shifter.carrier_im * shifter.carrier_im;
        for lane in 0..4 {
            assert!((norm2.lane(lane) - 1.0).abs() < 1e-3, "carrier drifted: {norm2:?}");
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
