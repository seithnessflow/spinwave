//! Parameter smoothing (port of `SmoothValue` and `cr::SmoothValue`).
//!
//! Exponential one-pole smoothing toward a target, with a linear ramp
//! fallback for lanes where the exponential step stalls (already at the
//! target, or the step is below float precision).

use vital_poly::{math, utils, PolyF32};

/// Audio-rate smoother (5 Hz cutoff).
#[derive(Clone)]
pub struct SmoothValue {
    value: PolyF32,
    current_value: PolyF32,
    sample_rate: f32,
}

impl SmoothValue {
    pub const SMOOTH_CUTOFF: f32 = 5.0;

    pub fn new(value: f32, sample_rate: f32) -> Self {
        SmoothValue {
            value: PolyF32::splat(value),
            current_value: PolyF32::splat(value),
            sample_rate,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Sets the target; the output glides toward it.
    pub fn set(&mut self, value: PolyF32) {
        self.value = value;
    }

    /// Jumps target and output immediately (no glide).
    pub fn set_hard(&mut self, value: PolyF32) {
        self.value = value;
        self.current_value = value;
    }

    #[inline]
    pub fn target(&self) -> PolyF32 {
        self.value
    }

    #[inline]
    pub fn current(&self) -> PolyF32 {
        self.current_value
    }

    pub fn process(&mut self, out: &mut [PolyF32]) {
        if self.current_value.eq(self.value).all() {
            out.fill(self.value);
            return;
        }

        let decay = math::exp(PolyF32::splat(
            -2.0 * core::f32::consts::PI * Self::SMOOTH_CUTOFF / self.sample_rate,
        ));
        let mut current_value = self.current_value;
        let target_value = self.value;
        for sample in out.iter_mut() {
            current_value = utils::interpolate(target_value, current_value, decay);
            *sample = current_value;
        }

        let equal_mask =
            current_value.eq(self.current_value) | self.value.eq(self.current_value);
        if equal_mask.any() {
            self.linear_interpolate(out, equal_mask);
        }

        self.current_value = equal_mask.select(self.current_value, current_value);
    }

    fn linear_interpolate(&mut self, out: &mut [PolyF32], linear_mask: vital_poly::PolyMask) {
        let num_samples = out.len();
        let mut current_value = self.current_value;
        self.current_value = linear_mask.select(self.value, self.current_value);
        let delta_value = (self.value - current_value) * (1.0 / num_samples as f32);

        for sample in out.iter_mut() {
            current_value += delta_value;
            *sample = linear_mask.select(current_value, *sample);
        }
    }
}

/// Control-rate smoother (20 Hz cutoff), one value per block.
#[derive(Clone)]
pub struct ControlSmoothValue {
    value: PolyF32,
    current_value: PolyF32,
    sample_rate: f32,
}

impl ControlSmoothValue {
    pub const SMOOTH_CUTOFF: f32 = 20.0;

    pub fn new(value: f32, sample_rate: f32) -> Self {
        ControlSmoothValue {
            value: PolyF32::splat(value),
            current_value: PolyF32::splat(value),
            sample_rate,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn set(&mut self, value: PolyF32) {
        self.value = value;
    }

    pub fn set_hard(&mut self, value: PolyF32) {
        self.value = value;
        self.current_value = value;
    }

    #[inline]
    pub fn current(&self) -> PolyF32 {
        self.current_value
    }

    /// Advances by `num_samples` samples and returns the smoothed value.
    pub fn tick(&mut self, num_samples: usize) -> PolyF32 {
        let decay = math::exp(PolyF32::splat(
            -2.0 * core::f32::consts::PI * Self::SMOOTH_CUTOFF * num_samples as f32
                / self.sample_rate,
        ));
        self.current_value = utils::interpolate(self.value, self.current_value, decay);
        self.current_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;

    #[test]
    fn converges_to_target() {
        let mut smooth = SmoothValue::new(0.0, SAMPLE_RATE);
        smooth.set(PolyF32::splat(1.0));

        let mut out = [PolyF32::ZERO; 128];
        // 5 Hz cutoff: settles well within a second.
        for _ in 0..400 {
            smooth.process(&mut out);
        }
        let value = smooth.current().lane(0);
        assert!((value - 1.0).abs() < 1e-3, "should converge, got {value}");
    }

    #[test]
    fn monotonic_rise() {
        let mut smooth = SmoothValue::new(0.0, SAMPLE_RATE);
        smooth.set(PolyF32::splat(1.0));
        let mut out = [PolyF32::ZERO; 256];
        smooth.process(&mut out);
        for window in out.windows(2) {
            assert!(window[1].lane(0) >= window[0].lane(0));
        }
        assert!(out[0].lane(0) > 0.0);
        assert!(out[255].lane(0) < 1.0);
    }

    #[test]
    fn set_hard_jumps() {
        let mut smooth = SmoothValue::new(0.0, SAMPLE_RATE);
        smooth.set_hard(PolyF32::splat(0.7));
        let mut out = [PolyF32::ZERO; 32];
        smooth.process(&mut out);
        assert_eq!(out[0].lane(0), 0.7);
        assert_eq!(out[31].lane(0), 0.7);
    }

    #[test]
    fn control_rate_convergence() {
        let mut smooth = ControlSmoothValue::new(0.0, SAMPLE_RATE);
        smooth.set(PolyF32::splat(2.0));
        let mut last = 0.0;
        for _ in 0..200 {
            let value = smooth.tick(64).lane(0);
            assert!(value >= last, "should rise monotonically");
            last = value;
        }
        assert!((last - 2.0).abs() < 1e-3, "should converge, got {last}");
    }
}
