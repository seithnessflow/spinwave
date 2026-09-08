//! Fast math approximations (port of Vital's `futils.h`).
//!
//! These are deliberately approximate: they trade a few ulps of accuracy for
//! speed, and their exact curves are part of Vital's sound (saturation
//! shapes, envelope scaling). Keep the coefficients bit-identical to the
//! reference when modifying anything here.

use crate::constants::*;
use crate::simd::{PolyF32, PolyU32};

/// Polynomial `2^x` accurate enough for pitch/gain conversions.
#[inline(always)]
pub fn exp2(exponent: PolyF32) -> PolyF32 {
    const C0: f32 = 1.0;
    const C1: f32 = 16970.0 / 24483.0;
    const C2: f32 = 1960.0 / 8161.0;
    const C3: f32 = 1360.0 / 24483.0;
    const C4: f32 = 80.0 / 8161.0;
    const C5: f32 = 32.0 / 24483.0;

    let integer = exponent.to_i32_round();
    let t = exponent - integer.to_f32_signed();
    let int_pow = integer.pow2_to_f32();

    let cubic = t * (t * (t * C5 + C4) + C3) + C2;
    let interpolate = t * (t * cubic + C1) + C0;
    int_pow * interpolate
}

/// Polynomial `log2(x)` matching [`exp2`]'s accuracy class.
#[inline(always)]
pub fn log2(value: PolyF32) -> PolyF32 {
    const C0: f32 = -1819.0 / 651.0;
    const C1: f32 = 5.0;
    const C2: f32 = -10.0 / 3.0;
    const C3: f32 = 10.0 / 7.0;
    const C4: f32 = -1.0 / 3.0;
    const C5: f32 = 1.0 / 31.0;

    let bits = value.to_u32_bits();
    let floored_log2 = bits.shr(23) - PolyU32::splat(0x7f);
    let t_bits = (bits & PolyU32::splat(0x7f_ffff)) | PolyU32::splat(0x7f << 23);
    let t = PolyF32::from_u32_bits(t_bits);

    let cubic = t * (t * (t * C5 + C4) + C3) + C2;
    let interpolate = t * (t * cubic + C1) + C0;
    floored_log2.to_f32_signed() + interpolate
}

/// Cheaper, rougher `2^x` for non-critical scaling.
#[inline(always)]
pub fn cheap_exp2(exponent: PolyF32) -> PolyF32 {
    const C0: f32 = 1.0;
    const C1: f32 = 12.0 / 17.0;
    const C2: f32 = 4.0 / 17.0;

    let integer = exponent.to_i32_round();
    let t = exponent - integer.to_f32_signed();
    let int_pow = integer.pow2_to_f32();

    let interpolate = t * (t * C2 + C1) + C0;
    int_pow * interpolate
}

#[inline(always)]
pub fn cheap_log2(value: PolyF32) -> PolyF32 {
    const C0: f32 = -5.0 / 3.0;
    const C1: f32 = 2.0;
    const C2: f32 = -1.0 / 3.0;

    let bits = value.to_u32_bits();
    let floored_log2 = bits.shr(23) - PolyU32::splat(0x7f);
    let t_bits = (bits & PolyU32::splat(0x7f_ffff)) | PolyU32::splat(0x7f << 23);
    let t = PolyF32::from_u32_bits(t_bits);

    let interpolate = t * (t * C2 + C1) + C0;
    floored_log2.to_f32_signed() + interpolate
}

#[inline(always)]
pub fn exp(exponent: PolyF32) -> PolyF32 {
    exp2(exponent * EXP_CONVERSION_MULT)
}

#[inline(always)]
pub fn ln(value: PolyF32) -> PolyF32 {
    log2(value) * LOG_CONVERSION_MULT
}

#[inline(always)]
pub fn pow(base: PolyF32, exponent: PolyF32) -> PolyF32 {
    exp2(log2(base) * exponent)
}

#[inline(always)]
pub fn cheap_pow(base: PolyF32, exponent: PolyF32) -> PolyF32 {
    cheap_exp2(cheap_log2(base) * exponent)
}

#[inline(always)]
pub fn midi_offset_to_ratio(note_offset: PolyF32) -> PolyF32 {
    exp2(note_offset * (1.0 / NOTES_PER_OCTAVE as f32))
}

#[inline(always)]
pub fn midi_note_to_frequency(note: PolyF32) -> PolyF32 {
    midi_offset_to_ratio(note) * MIDI_0_FREQUENCY
}

#[inline(always)]
pub fn magnitude_to_db(magnitude: PolyF32) -> PolyF32 {
    log2(magnitude) * DB_GAIN_CONVERSION_MULT
}

#[inline(always)]
pub fn db_to_magnitude(decibels: PolyF32) -> PolyF32 {
    exp2(decibels * DB_MAGNITUDE_CONVERSION_MULT)
}

/// Rational tanh approximation, cheap enough for per-sample saturation.
#[inline(always)]
pub fn quick_tanh(value: PolyF32) -> PolyF32 {
    let square = value * value;
    value / (square / PolyF32::splat(3.0).mul_add(square, PolyF32::splat(0.2)) + 1.0)
}

#[inline(always)]
pub fn quick_tanh_derivative(value: PolyF32) -> PolyF32 {
    let square = value * value;
    let fourth = square * square;
    let part_den = square + 2.5;
    let num = PolyF32::splat(6.25)
        .mul_add(fourth, PolyF32::splat(0.166667))
        .mul_add(square, PolyF32::splat(-1.25));
    num / (part_den * part_den)
}

/// Higher-order tanh approximation used by filters and soft clippers.
#[inline(always)]
pub fn tanh(value: PolyF32) -> PolyF32 {
    let abs_value = value.abs();
    let square = value * value;

    let part_num1 = abs_value * 0.821226666969744 + 0.893229853513558;
    let part_num2 = square * part_num1 + 2.45550750702956;
    let num = value * (abs_value * 2.45550750702956 + part_num2);

    let part_den = (abs_value * 0.814642734961073 * value + value).abs();
    let den = part_den * (square + 2.44506634652299) + 2.44506634652299;
    num / den
}

/// Linear up to ±0.66, tanh-saturated beyond (Vital's `hardTanh`).
#[inline(always)]
pub fn hard_tanh(value: PolyF32) -> PolyF32 {
    const HARDNESS: f32 = 0.66;
    const HARDNESS_INV_REC: f32 = 1.0 / (1.0 - HARDNESS);

    let clamped = value.clamp(-HARDNESS, HARDNESS);
    clamped + tanh((value - clamped) * HARDNESS_INV_REC) * (1.0 - HARDNESS)
}

#[inline(always)]
pub fn tanh_derivative_fast(value: PolyF32) -> PolyF32 {
    let square = value * value;
    PolyF32::ONE / PolyF32::splat(2.0).mul_add(square, PolyF32::splat(1.8))
}

/// Smooth algebraic saturation: grows slowly instead of clamping.
#[inline(always)]
pub fn algebraic_sat(value: PolyF32) -> PolyF32 {
    let square = value * value;
    value * square * -0.9 / (square + 3.0) + value
}

#[inline(always)]
pub fn algebraic_sat_derivative(value: PolyF32) -> PolyF32 {
    let square = value * value;
    let fourth = square * square;
    let num = fourth * 0.1 + (square * -2.1 + 9.0);
    let part_den = square + 3.0;
    num / (part_den * part_den)
}

#[inline(always)]
pub fn quadratic_inv_sat(value: PolyF32) -> PolyF32 {
    value / (value * value * 0.25 + 1.0)
}

#[inline(always)]
pub fn bump_sat(value: PolyF32) -> PolyF32 {
    let square = value * value;
    let pow_four = square * square;
    value / (pow_four * 0.1 + 1.0)
}

#[inline(always)]
pub fn bump_sat2(value: PolyF32) -> PolyF32 {
    let square = value * value;
    let pow_four = square * square;
    (value + square * value * 3.0) / (pow_four * 20.0 + 1.0)
}

/// Parabolic sine for phase in `[-0.5, 0.5]` (one period = 1.0).
#[inline(always)]
pub fn quick_sin(phase: PolyF32) -> PolyF32 {
    phase * PolyF32::splat(8.0).mul_add(phase.abs(), PolyF32::splat(-16.0))
}

/// Refined parabolic sine, phase in `[-0.5, 0.5]`.
#[inline(always)]
pub fn sin(phase: PolyF32) -> PolyF32 {
    let approx = quick_sin(phase);
    approx * PolyF32::splat(0.776).mul_add(approx.abs(), PolyF32::splat(0.224))
}

/// Parabolic sine for phase in `[0, 1]`.
#[inline(always)]
pub fn quick_sin1(phase: PolyF32) -> PolyF32 {
    let adjusted = PolyF32::splat(0.5) - phase;
    adjusted * PolyF32::splat(8.0).mul_add(adjusted.abs(), PolyF32::splat(-16.0))
}

/// Refined parabolic sine, phase in `[0, 1]`.
#[inline(always)]
pub fn sin1(phase: PolyF32) -> PolyF32 {
    let approx = quick_sin1(phase);
    approx * PolyF32::splat(0.776).mul_add(approx.abs(), PolyF32::splat(0.224))
}

#[inline(always)]
pub fn sin_interpolate(from: PolyF32, to: PolyF32, t: PolyF32) -> PolyF32 {
    let sin_value = sin(t * 0.5 - 0.25);
    let sin_t = sin_value * 0.5 + 0.5;
    from + (to - from) * sin_t
}

#[inline(always)]
pub fn equal_power_fade(t: PolyF32) -> PolyF32 {
    sin1(t * 0.25)
}

#[inline(always)]
pub fn equal_power_fade_inverse(t: PolyF32) -> PolyF32 {
    sin1((t + 1.0) * 0.25)
}

/// Constant-power pan: `pan` in `[-1, 1]`, lanes follow the stereo split.
#[inline(always)]
pub fn pan_amplitude(pan: PolyF32) -> PolyF32 {
    const SCALE: f32 = SQRT_2;
    let stereo_split = PolyF32::stereo(1.0, -1.0);
    let eighth_phase = PolyF32::splat(0.125);
    sin1(eighth_phase - stereo_split * pan * eighth_phase) * SCALE
}

/// Exponential response curve: identity when `power` ~ 0.
#[inline(always)]
pub fn power_scale(value: PolyF32, power: PolyF32) -> PolyF32 {
    const MIN_POWER_MAG: f32 = 0.005;
    let zero_mask =
        power.lt(PolyF32::splat(MIN_POWER_MAG)) & (-power).lt(PolyF32::splat(MIN_POWER_MAG));
    let numerator = exp(power * value) - 1.0;
    let denominator = exp(power) - 1.0;
    zero_mask.select(value, numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_rel_error(f: impl Fn(PolyF32) -> PolyF32, reference: impl Fn(f32) -> f32, values: &[f32]) -> f32 {
        let mut max_error: f32 = 0.0;
        for &v in values {
            let got = f(PolyF32::splat(v)).lane(0);
            let want = reference(v);
            let denom = want.abs().max(1e-9);
            max_error = max_error.max((got - want).abs() / denom);
        }
        max_error
    }

    #[test]
    fn exp2_accuracy() {
        let values: Vec<f32> = (-100..100).map(|i| i as f32 * 0.1).collect();
        assert!(max_rel_error(exp2, |v| v.exp2(), &values) < 1e-5);
    }

    #[test]
    fn log2_accuracy() {
        // Absolute error: relative error is meaningless around log2(1) = 0.
        let mut max_error: f32 = 0.0;
        for i in 1..1000 {
            let v = i as f32 * 0.01;
            let got = log2(PolyF32::splat(v)).lane(0);
            max_error = max_error.max((got - v.log2()).abs());
        }
        // ~5e-4 is the intrinsic accuracy of the reference polynomial.
        assert!(max_error < 1e-3, "max abs error {max_error}");
    }

    #[test]
    fn tanh_accuracy() {
        let values: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        assert!(max_rel_error(tanh, |v| v.tanh(), &values) < 0.02);
    }

    #[test]
    fn midi_conversions() {
        let a4 = midi_note_to_frequency(PolyF32::splat(69.0)).lane(0);
        assert!((a4 - 440.0).abs() < 0.01, "A4 was {a4}");
        let db = magnitude_to_db(PolyF32::splat(2.0)).lane(0);
        assert!((db - 6.0206).abs() < 0.001);
    }

    #[test]
    fn sin_period() {
        // sin1: full period over [0, 1], peak at 0.25.
        assert!(sin1(PolyF32::splat(0.0)).lane(0).abs() < 1e-6);
        assert!((sin1(PolyF32::splat(0.25)).lane(0) - 1.0).abs() < 1e-3);
        assert!(sin1(PolyF32::splat(0.5)).lane(0).abs() < 1e-6);
        assert!((sin1(PolyF32::splat(0.75)).lane(0) + 1.0).abs() < 1e-3);
    }

    #[test]
    fn power_scale_identity_at_zero_power() {
        let v = PolyF32::splat(0.35);
        assert_eq!(power_scale(v, PolyF32::ZERO).lane(0), 0.35);
        // And behaves exponentially otherwise.
        let scaled = power_scale(PolyF32::splat(0.5), PolyF32::splat(4.0)).lane(0);
        let expected = ((4.0f32 * 0.5).exp() - 1.0) / (4.0f32.exp() - 1.0);
        assert!((scaled - expected).abs() < 1e-3);
    }
}
