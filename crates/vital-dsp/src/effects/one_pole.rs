//! Private one-pole low-pass used by the effects (port of Vital's
//! `OnePoleFilter` basic tick). Duplicated here on purpose: the filters
//! module is owned separately and its layout may shift.

use vital_poly::constants::PI;
use vital_poly::{PolyF32, PolyMask};

/// Trapezoidal one-pole low-pass. `tick_basic` returns the mid-step state
/// while advancing the internal state a full step, exactly like the
/// reference's `tickBasic`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OnePole {
    state: PolyF32,
}

impl OnePole {
    pub fn new() -> OnePole {
        OnePole { state: PolyF32::ZERO }
    }

    #[inline(always)]
    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.state = reset_mask.select(PolyF32::ZERO, self.state);
    }

    #[inline(always)]
    pub fn hard_reset(&mut self) {
        self.state = PolyF32::ZERO;
    }

    #[inline(always)]
    pub fn tick_basic(&mut self, audio_in: PolyF32, coefficient: PolyF32) -> PolyF32 {
        let delta = coefficient * (audio_in - self.state);
        self.state += delta;
        let current = self.state;
        self.state += delta;
        current
    }

    /// Warped coefficient from a cutoff in Hz (Vital's `computeCoefficient`).
    /// Block-rate only: the per-lane `tan` is scalar.
    #[inline(always)]
    pub fn compute_coefficient(cutoff_frequency: PolyF32, sample_rate: f32) -> PolyF32 {
        let delta_phase = cutoff_frequency * (PI / sample_rate);
        (delta_phase / (delta_phase + 1.0)).map(f32::tan)
    }
}
