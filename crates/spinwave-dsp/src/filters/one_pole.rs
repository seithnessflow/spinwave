//! One-pole filter with pluggable saturation (port of `one_pole_filter.h`).
//!
//! The C++ template parameter becomes a zero-sized [`Saturation`] type so the
//! saturator inlines into the tick exactly like the template instantiation.

use core::marker::PhantomData;

use spinwave_poly::constants::PI;
use spinwave_poly::{math, PolyF32, PolyMask};

/// Static saturation function applied inside the filter loop.
pub trait Saturation: Copy + Clone + Default {
    fn saturate(value: PolyF32) -> PolyF32;
}

/// Identity (C++ `utils::pass`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Pass;
impl Saturation for Pass {
    #[inline(always)]
    fn saturate(value: PolyF32) -> PolyF32 {
        value
    }
}

/// `futils::algebraicSat` (ladder stages).
#[derive(Clone, Copy, Debug, Default)]
pub struct AlgebraicSat;
impl Saturation for AlgebraicSat {
    #[inline(always)]
    fn saturate(value: PolyF32) -> PolyF32 {
        math::algebraic_sat(value)
    }
}

/// `futils::tanh` (diode stage 1).
#[derive(Clone, Copy, Debug, Default)]
pub struct TanhSat;
impl Saturation for TanhSat {
    #[inline(always)]
    fn saturate(value: PolyF32) -> PolyF32 {
        math::tanh(value)
    }
}

/// `futils::quickTanh` (dirty stages 3/4).
#[derive(Clone, Copy, Debug, Default)]
pub struct QuickTanhSat;
impl Saturation for QuickTanhSat {
    #[inline(always)]
    fn saturate(value: PolyF32) -> PolyF32 {
        math::quick_tanh(value)
    }
}

/// Hard clip to `[-1, 1]` (diode stage 4).
#[derive(Clone, Copy, Debug, Default)]
pub struct HardClipSat;
impl Saturation for HardClipSat {
    #[inline(always)]
    fn saturate(value: PolyF32) -> PolyF32 {
        value.clamp(-1.0, 1.0)
    }
}

/// One-pole filter ticked at double rate (two half-steps per sample).
#[derive(Clone, Copy, Debug, Default)]
pub struct OnePoleFilter<S: Saturation = Pass> {
    current_state: PolyF32,
    filter_state: PolyF32,
    sat_filter_state: PolyF32,
    _saturation: PhantomData<S>,
}

impl<S: Saturation> OnePoleFilter<S> {
    pub fn new() -> OnePoleFilter<S> {
        OnePoleFilter {
            current_state: PolyF32::ZERO,
            filter_state: PolyF32::ZERO,
            sat_filter_state: PolyF32::ZERO,
            _saturation: PhantomData,
        }
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.current_state = reset_mask.select(PolyF32::ZERO, self.current_state);
        self.filter_state = reset_mask.select(PolyF32::ZERO, self.filter_state);
        self.sat_filter_state = reset_mask.select(PolyF32::ZERO, self.sat_filter_state);
    }

    /// Plain tick with no saturation in the state path.
    ///
    /// Note: like the C++, this does NOT update the saturated state; mixing
    /// `tick_basic` with [`OnePoleFilter::next_sat_state`] reads a stale value.
    #[inline(always)]
    pub fn tick_basic(&mut self, audio_in: PolyF32, coefficient: PolyF32) -> PolyF32 {
        let delta = coefficient * (audio_in - self.filter_state);
        self.filter_state += delta;
        self.current_state = self.filter_state;
        self.filter_state += delta;
        self.current_state
    }

    /// Saturating tick: the feedback path reads the saturated state.
    #[inline(always)]
    pub fn tick(&mut self, audio_in: PolyF32, coefficient: PolyF32) -> PolyF32 {
        let delta = coefficient * (audio_in - self.sat_filter_state);
        self.filter_state += delta;
        self.current_state = S::saturate(self.filter_state);
        self.filter_state += delta;
        self.sat_filter_state = S::saturate(self.filter_state);
        self.current_state
    }

    /// Tick scaling the increment by the saturator's derivative shape.
    #[inline(always)]
    pub fn tick_derivative(&mut self, audio_in: PolyF32, coefficient: PolyF32) -> PolyF32 {
        let delta = coefficient * (audio_in - self.filter_state);
        self.filter_state = self.filter_state.mul_add(S::saturate(self.filter_state + delta), delta);
        self.current_state = self.filter_state;
        self.filter_state = self.filter_state.mul_add(S::saturate(self.filter_state + delta), delta);
        self.sat_filter_state = self.filter_state;
        self.current_state
    }

    #[inline(always)]
    pub fn current_state(&self) -> PolyF32 {
        self.current_state
    }

    #[inline(always)]
    pub fn next_sat_state(&self) -> PolyF32 {
        self.sat_filter_state
    }

    #[inline(always)]
    pub fn next_state(&self) -> PolyF32 {
        self.filter_state
    }

    /// Per-block coefficient from a cutoff in Hz (C++ `computeCoefficient`).
    pub fn compute_coefficient(cutoff_frequency: PolyF32, sample_rate: f32) -> PolyF32 {
        let delta_phase = cutoff_frequency * (PI / sample_rate);
        (delta_phase / (delta_phase + 1.0)).map(f32::tan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_response_converges() {
        let mut filter: OnePoleFilter = OnePoleFilter::new();
        let coefficient = OnePoleFilter::<Pass>::compute_coefficient(PolyF32::splat(2000.0), 44100.0);
        let mut out = PolyF32::ZERO;
        for _ in 0..2000 {
            out = filter.tick_basic(PolyF32::ONE, coefficient);
        }
        assert!(out.is_finite());
        assert!((out.lane(0) - 1.0).abs() < 1e-3, "converged to {}", out.lane(0));
    }

    #[test]
    fn low_pass_selectivity() {
        let sample_rate = 44100.0;
        let coefficient = OnePoleFilter::<Pass>::compute_coefficient(PolyF32::splat(500.0), sample_rate);

        let rms = |freq: f32| {
            let mut filter: OnePoleFilter = OnePoleFilter::new();
            let mut sum = 0.0f32;
            let n = 8820;
            for i in 0..n {
                let phase = 2.0 * core::f32::consts::PI * freq * i as f32 / sample_rate;
                let out = filter.tick_basic(PolyF32::splat(phase.sin()), coefficient);
                if i >= n / 2 {
                    sum += out.lane(0) * out.lane(0);
                }
            }
            (sum / (n / 2) as f32).sqrt()
        };

        let low = rms(100.0);
        let high = rms(8000.0);
        assert!(low > 3.0 * high, "low {low} high {high}");
    }

    #[test]
    fn reset_clears_masked_lanes() {
        let mut filter: OnePoleFilter = OnePoleFilter::new();
        let coefficient = PolyF32::splat(0.2);
        for _ in 0..10 {
            filter.tick(PolyF32::ONE, coefficient);
        }
        assert!(filter.next_state().lane(0) > 0.0);
        filter.reset(PolyMask::all_on());
        assert_eq!(filter.next_state().lane(0), 0.0);
        assert_eq!(filter.current_state().lane(0), 0.0);
        assert_eq!(filter.next_sat_state().lane(0), 0.0);
    }

    #[test]
    fn saturating_tick_is_bounded() {
        let mut filter: OnePoleFilter<HardClipSat> = OnePoleFilter::new();
        let coefficient = PolyF32::splat(0.4);
        for _ in 0..100 {
            let out = filter.tick(PolyF32::splat(10.0), coefficient);
            assert!(out.lane(0) <= 1.0 + 1e-6);
        }
    }
}
