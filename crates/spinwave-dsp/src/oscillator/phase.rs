//! Phase distortion and amplitude-window functions of the wavetable
//! oscillator (sync, formant, quantize, bend, squeeze, pulse width, FM/RM).
//!
//! Phase accumulators are wrapping `u32` values covering one cycle, exactly
//! as in the reference; the distortion functions warp that integer phase and
//! the window functions shape the amplitude of the distorted read.

use spinwave_poly::{math, utils, PolyF32, PolyMask, PolyU32};

/// One full cycle of the u32 phase accumulator, as a float.
pub(crate) const PHASE_MULT: f32 = 4_294_967_296.0;
pub(crate) const INV_PHASE_MULT: f32 = 1.0 / PHASE_MULT;
const HALF_PHASE: u32 = 0x8000_0000;

pub(crate) const MAX_SYNC_POWER: f32 = 4.0;
pub(crate) const MAX_SYNC: f32 = 16.0;
const MAX_QUANTIZE: f32 = 0.85;
const MAX_SQUEEZE_PERCENT: f32 = 0.95;
const DISTORT_BITS: f32 = 32.0;
const FM_PHASE_MULT: f32 = PHASE_MULT / 8.0;
const MAX_FM_MODULATION: u32 = 48;

/// Phase distortion modes (Vital's `DistortionType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DistortionType {
    #[default]
    None,
    Sync,
    Formant,
    Quantize,
    Bend,
    Squeeze,
    PulseWidth,
    FmOscillatorA,
    FmOscillatorB,
    FmSample,
    RmOscillatorA,
    RmOscillatorB,
    RmSample,
}

impl DistortionType {
    pub fn is_first_modulation(self) -> bool {
        matches!(self, DistortionType::FmOscillatorA | DistortionType::RmOscillatorA)
    }

    pub fn is_second_modulation(self) -> bool {
        matches!(self, DistortionType::FmOscillatorB | DistortionType::RmOscillatorB)
    }

    /// Whether the mode reads the distortion-phase parameter.
    pub fn uses_distortion_phase(self) -> bool {
        matches!(
            self,
            DistortionType::Sync
                | DistortionType::Formant
                | DistortionType::Quantize
                | DistortionType::Bend
                | DistortionType::Squeeze
                | DistortionType::PulseWidth
        )
    }

    pub fn is_fm(self) -> bool {
        matches!(
            self,
            DistortionType::FmOscillatorA | DistortionType::FmOscillatorB | DistortionType::FmSample
        )
    }

    pub fn is_rm(self) -> bool {
        matches!(
            self,
            DistortionType::RmOscillatorA | DistortionType::RmOscillatorB | DistortionType::RmSample
        )
    }
}

// ---------------------------------------------------------------------------
// Phase distortion functions
// ---------------------------------------------------------------------------

pub(crate) fn pass_through_phase(
    phase: PolyU32,
    _distortion: PolyF32,
    _distortion_phase: PolyU32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyU32 {
    phase
}

pub(crate) fn quantize_phase(
    phase: PolyU32,
    distortion: PolyF32,
    distortion_phase: PolyU32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyU32 {
    let normal_phase = phase.to_f32_signed() * distortion * INV_PHASE_MULT;
    let adjustment = distortion_phase.to_f32_signed() * INV_PHASE_MULT;
    let floored_phase = (normal_phase + adjustment).trunc() - adjustment;
    ((floored_phase / distortion) * PHASE_MULT).to_i32_round() - distortion_phase
}

pub(crate) fn bend_phase(
    phase: PolyU32,
    distortion: PolyF32,
    distortion_phase: PolyU32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyU32 {
    let float_phase = (phase - distortion_phase).to_f32_signed() * INV_PHASE_MULT + 0.5;

    let distortion_offset = (distortion - distortion * distortion) * 2.0;
    let float_phase2 = float_phase * float_phase;
    let float_phase3 = float_phase * float_phase2;

    let distortion_scale = distortion * 3.0;
    let middle_mult1 = distortion_scale + distortion_offset;
    let middle_mult2 = distortion_scale - distortion_offset;
    let middle1 = middle_mult1 * (float_phase2 - float_phase3);
    let middle2 = middle_mult2 * (float_phase - float_phase2 * 2.0 + float_phase3);
    let new_phase = float_phase3 + middle1 + middle2;
    ((new_phase - 0.5) * PHASE_MULT).to_i32_round()
}

pub(crate) fn squeeze_phase(
    phase: PolyU32,
    distortion: PolyF32,
    distortion_phase: PolyU32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyU32 {
    let center_phase = PHASE_MULT / 4.0;
    let max_phase = PHASE_MULT / 2.0;
    let mut float_phase = (phase - distortion_phase).to_f32_signed();
    let positive_mask = float_phase.gt(PolyF32::ZERO);
    float_phase = float_phase.abs();

    let pivot = distortion * center_phase;
    let right_half_mask = float_phase.gt(pivot);

    let left_phase = float_phase / distortion;
    let right_phase =
        PolyF32::splat(max_phase) - (PolyF32::splat(max_phase) - float_phase) / (PolyF32::splat(2.0) - distortion);
    let new_phase = right_half_mask.select(right_phase, left_phase);
    let new_phase = positive_mask.select(new_phase, -new_phase);
    new_phase.to_i32_round()
}

pub(crate) fn sync_phase(
    phase: PolyU32,
    distortion: PolyF32,
    _distortion_phase: PolyU32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyU32 {
    let float_val = (phase + PolyU32::splat(HALF_PHASE)).to_f32_signed() * distortion;
    float_val.to_i32_round() * PolyU32::splat(MAX_SYNC as u32) + PolyU32::splat(HALF_PHASE)
}

pub(crate) fn pulse_width_phase(
    phase: PolyU32,
    distortion: PolyF32,
    _distortion_phase: PolyU32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyU32 {
    let distorted_phase = phase.to_f32_signed() * distortion;
    let clamped_phase = distorted_phase.clamp(i32::MIN as f32, i32::MAX as f32);
    clamped_phase.to_i32_round()
}

pub(crate) fn fm_phase(
    phase: PolyU32,
    distortion: PolyF32,
    _distortion_phase: PolyU32,
    modulation: &[PolyF32],
    i: usize,
) -> PolyU32 {
    let phase_offset = modulation[i] * distortion;
    phase + (phase_offset * FM_PHASE_MULT).to_i32_round() * PolyU32::splat(MAX_FM_MODULATION)
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

pub(crate) fn pass_through_window(
    _phase: PolyU32,
    _distorted_phase: PolyU32,
    _distortion: PolyF32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyF32 {
    PolyF32::ONE
}

pub(crate) fn pulse_width_window(
    _phase: PolyU32,
    distorted_phase: PolyU32,
    _distortion: PolyF32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyF32 {
    PolyF32::ONE & !distorted_phase.eq(PolyU32::splat(HALF_PHASE))
}

pub(crate) fn half_sin_window(
    phase: PolyU32,
    _distorted_phase: PolyU32,
    _distortion: PolyF32,
    _modulation: &[PolyF32],
    _i: usize,
) -> PolyF32 {
    let normal_phase =
        (phase + PolyU32::splat(i32::MAX as u32)).to_f32_signed() * (INV_PHASE_MULT / 2.0);
    math::sin(normal_phase + 0.25)
}

pub(crate) fn rm_window(
    _phase: PolyU32,
    _distorted_phase: PolyU32,
    distortion: PolyF32,
    modulation: &[PolyF32],
    i: usize,
) -> PolyF32 {
    utils::interpolate(PolyF32::ONE, modulation[i], distortion)
}

// ---------------------------------------------------------------------------
// Static helpers matching the reference's public statics
// ---------------------------------------------------------------------------

/// Applies one distortion step outside the audio loop (UI/analysis use).
pub fn adjust_phase(
    distortion_type: DistortionType,
    phase: PolyU32,
    distortion_amount: PolyF32,
    distortion_phase: PolyU32,
) -> PolyU32 {
    let none: &[PolyF32] = &[];
    match distortion_type {
        DistortionType::Sync | DistortionType::Formant => {
            sync_phase(phase, distortion_amount, distortion_phase, none, 0)
        }
        DistortionType::Quantize => quantize_phase(phase, distortion_amount, distortion_phase, none, 0),
        DistortionType::Bend => bend_phase(phase, distortion_amount, distortion_phase, none, 0),
        DistortionType::Squeeze => squeeze_phase(phase, distortion_amount, distortion_phase, none, 0),
        DistortionType::PulseWidth => {
            pulse_width_phase(phase, distortion_amount, distortion_phase, none, 0)
        }
        _ => phase,
    }
}

/// Amplitude window matching [`adjust_phase`].
pub fn phase_window(
    distortion_type: DistortionType,
    phase: PolyU32,
    distorted_phase: PolyU32,
) -> PolyF32 {
    let none: &[PolyF32] = &[];
    match distortion_type {
        DistortionType::Formant => half_sin_window(phase, distorted_phase, PolyF32::ZERO, none, 0),
        DistortionType::PulseWidth => {
            pulse_width_window(phase, distorted_phase, PolyF32::ZERO, none, 0)
        }
        _ => PolyF32::ONE,
    }
}

pub(crate) fn set_power_distortion_values(values: &mut [PolyF32], exponent: f32, spread: bool) {
    if spread {
        for value in values.iter_mut() {
            *value = math::pow(PolyF32::splat(2.0), (*value - 0.5) * 2.0 * exponent);
        }
    } else {
        let value = math::pow(PolyF32::splat(2.0), (values[0] - 0.5) * 2.0 * exponent);
        for slot in values.iter_mut() {
            *slot = value;
        }
    }
}

/// Maps normalized distortion amounts to the per-mode working values.
pub fn shape_distortion_values(
    distortion_type: DistortionType,
    values: &mut [PolyF32],
    spread: bool,
) {
    match distortion_type {
        DistortionType::FmOscillatorA | DistortionType::FmOscillatorB | DistortionType::FmSample => {
            for value in values.iter_mut() {
                *value = *value * *value;
            }
        }
        DistortionType::Sync | DistortionType::Formant => {
            set_power_distortion_values(values, MAX_SYNC_POWER, spread);
            for value in values.iter_mut() {
                *value *= 1.0 / MAX_SYNC;
            }
        }
        DistortionType::Quantize => {
            if spread {
                for value in values.iter_mut() {
                    let mut distortion = PolyF32::ONE - *value;
                    distortion = distortion * distortion * distortion;
                    distortion *= MAX_QUANTIZE;
                    *value = math::pow(PolyF32::splat(2.0), distortion * DISTORT_BITS + 1.0);
                }
            } else {
                let mut distortion = PolyF32::ONE - values[0];
                distortion = distortion * distortion * distortion;
                distortion *= MAX_QUANTIZE;
                let value = math::pow(PolyF32::splat(2.0), distortion * DISTORT_BITS + 1.0);
                for slot in values.iter_mut() {
                    *slot = value;
                }
            }
        }
        DistortionType::Squeeze => {
            for value in values.iter_mut() {
                *value = *value * 2.0 * MAX_SQUEEZE_PERCENT + (1.0 - MAX_SQUEEZE_PERCENT);
            }
        }
        DistortionType::PulseWidth => {
            if spread {
                for value in values.iter_mut() {
                    let distortion = (PolyF32::ONE - *value).max(PolyF32::splat(1.0 / u32::MAX as f32));
                    *value = PolyF32::ONE / distortion;
                }
            } else {
                let distortion = (PolyF32::ONE - values[0]).max(PolyF32::splat(1.0 / u32::MAX as f32));
                let value = PolyF32::ONE / distortion;
                for slot in values.iter_mut() {
                    *slot = value;
                }
            }
        }
        _ => {}
    }
}

/// Signed per-lane `a < b` on phase-like integers.
#[inline(always)]
pub(crate) fn u32_lt_signed(a: PolyU32, b: PolyU32) -> PolyMask {
    let m = |x: u32, y: u32| if (x as i32) < (y as i32) { u32::MAX } else { 0 };
    PolyMask::from_u32(PolyU32::from_lanes([
        m(a.lane(0), b.lane(0)),
        m(a.lane(1), b.lane(1)),
        m(a.lane(2), b.lane(2)),
        m(a.lane(3), b.lane(3)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bend_at_half_is_identity() {
        // distortion = 0.5 collapses the Bezier warp to a straight line.
        for &raw in &[0u32, 0x2000_0000, 0x7fff_ffff, 0x9000_0000, 0xf000_0000] {
            let phase = PolyU32::splat(raw);
            let out = bend_phase(phase, PolyF32::splat(0.5), PolyU32::ZERO, &[], 0);
            let diff = (out.lane(0).wrapping_sub(raw)) as i32;
            assert!(diff.unsigned_abs() < 4096, "raw={raw:#x} diff={diff}");
        }
    }

    #[test]
    fn sync_at_neutral_is_near_identity() {
        // Shaped neutral sync distortion is 1/MAX_SYNC; the phase survives
        // up to the float precision of the 32-bit accumulator.
        let mut values = [PolyF32::splat(0.5)];
        shape_distortion_values(DistortionType::Sync, &mut values, false);
        for &raw in &[0x1234_5678u32, 0x8000_0000, 0xdead_beef] {
            let out = sync_phase(PolyU32::splat(raw), values[0], PolyU32::ZERO, &[], 0);
            let diff = (out.lane(0).wrapping_sub(raw)) as i32;
            assert!(diff.unsigned_abs() < 4096, "raw={raw:#x} diff={diff}");
        }
    }

    #[test]
    fn pulse_width_window_gates_saturated_phase() {
        let gated = pulse_width_window(
            PolyU32::ZERO,
            PolyU32::splat(0x8000_0000),
            PolyF32::ZERO,
            &[],
            0,
        );
        assert_eq!(gated.lane(0), 0.0);
        let open = pulse_width_window(PolyU32::ZERO, PolyU32::splat(42), PolyF32::ZERO, &[], 0);
        assert_eq!(open.lane(0), 1.0);
    }
}
