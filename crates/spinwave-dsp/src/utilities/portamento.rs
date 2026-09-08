//! Portamento glide between notes (port of `PortamentoSlope`).
//!
//! Interpolates from the previous note to the target over `run_seconds`,
//! with a power-curved slope, optional octave scaling and a force/auto mode.

use spinwave_poly::constants::{VoiceEvent, NOTES_PER_OCTAVE};
use spinwave_poly::{math, utils, PolyF32, PolyMask};

pub const MIN_PORTAMENTO_TIME: f32 = 0.001;

#[derive(Clone, Copy, Debug, Default)]
pub struct PortamentoParams {
    /// Note to glide toward.
    pub target: PolyF32,
    /// Note the glide starts from.
    pub source: PolyF32,
    /// Glide time in seconds (per octave when `scale` is on).
    pub run_seconds: PolyF32,
    /// Curve power (positive bends toward the target early).
    pub slope_power: PolyF32,
    /// Number of held notes, used by auto mode.
    pub num_notes_pressed: PolyF32,
    /// Always glide; when false, only glide while other notes are held.
    pub force: bool,
    /// Scale the glide time by the interval size.
    pub scale: bool,
}

#[derive(Clone)]
pub struct PortamentoSlope {
    sample_rate: f32,
    position: PolyF32,
    pending_reset: PolyMask,
}

impl PortamentoSlope {
    pub fn new(sample_rate: f32) -> Self {
        PortamentoSlope {
            sample_rate,
            position: PolyF32::ZERO,
            pending_reset: PolyMask::NONE,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Note event: `VoiceEvent::On` restarts the glide on the next process.
    pub fn trigger(&mut self, mask: PolyMask, value: PolyF32, _sample_offset: usize) {
        self.pending_reset |= mask & value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
    }

    /// Control-rate tick over `num_samples`; returns the glided note value.
    pub fn process(&mut self, params: &PortamentoParams, num_samples: usize) -> PolyF32 {
        let active_mask = params.run_seconds.gt(PolyF32::splat(MIN_PORTAMENTO_TIME));
        if !active_mask.any() {
            self.pending_reset = PolyMask::NONE;
            self.position = PolyF32::ONE;
            return params.target;
        }

        let mut reset_mask = self.pending_reset;
        self.pending_reset = PolyMask::NONE;
        self.position = reset_mask.select(PolyF32::ZERO, self.position);

        if !params.force {
            // Auto mode: a lone note (nothing else held) starts at the
            // target, no glide.
            reset_mask &= params.num_notes_pressed.eq(PolyF32::ONE);
            self.position = reset_mask.select(PolyF32::ONE, self.position);
        }

        let mut run_seconds = params.run_seconds;
        if params.scale {
            let midi_delta = (params.target - params.source).abs();
            run_seconds *= midi_delta * (1.0 / NOTES_PER_OCTAVE as f32);
        }

        let position_delta =
            PolyF32::splat(num_samples as f32) / (run_seconds * self.sample_rate);
        self.position = (self.position + position_delta).clamp(0.0, 1.0);

        let power = -params.slope_power;
        let adjusted_position = math::power_scale(self.position, power);
        utils::interpolate(params.source, params.target, adjusted_position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;

    fn on() -> PolyF32 {
        PolyF32::splat(VoiceEvent::On.as_f32())
    }

    fn params(run_seconds: f32) -> PortamentoParams {
        PortamentoParams {
            target: PolyF32::splat(72.0),
            source: PolyF32::splat(60.0),
            run_seconds: PolyF32::splat(run_seconds),
            force: true,
            ..Default::default()
        }
    }

    #[test]
    fn slope_timing() {
        let mut slope = PortamentoSlope::new(SAMPLE_RATE);
        let params = params(0.1);
        slope.trigger(PolyMask::all_on(), on(), 0);

        // 0.1s glide, 441-sample blocks: 10 blocks to finish.
        let mut value = 0.0;
        for _ in 0..5 {
            value = slope.process(&params, 441).lane(0);
        }
        // Halfway: linear slope (power 0) sits mid-interval.
        assert!((value - 66.0).abs() < 0.2, "midpoint was {value}");
        for _ in 0..5 {
            value = slope.process(&params, 441).lane(0);
        }
        assert!((value - 72.0).abs() < 1e-3, "endpoint was {value}");
        // Stays clamped at the target afterward.
        value = slope.process(&params, 441).lane(0);
        assert!((value - 72.0).abs() < 1e-3);
    }

    #[test]
    fn bypass_below_min_time() {
        let mut slope = PortamentoSlope::new(SAMPLE_RATE);
        let params = params(0.0);
        slope.trigger(PolyMask::all_on(), on(), 0);
        let value = slope.process(&params, 64);
        assert_eq!(value.lane(0), 72.0);
    }

    #[test]
    fn auto_mode_skips_glide_for_lone_note() {
        let mut slope = PortamentoSlope::new(SAMPLE_RATE);
        let mut auto_params = params(0.5);
        auto_params.force = false;
        auto_params.num_notes_pressed = PolyF32::ONE;
        slope.trigger(PolyMask::all_on(), on(), 0);
        let value = slope.process(&auto_params, 64).lane(0);
        assert!((value - 72.0).abs() < 1e-3, "lone note should jump, got {value}");
    }

    #[test]
    fn scale_mode_extends_time_with_interval() {
        // One octave with scale on and 0.1s/octave: same finish time as
        // unscaled 0.1s.
        let mut slope = PortamentoSlope::new(SAMPLE_RATE);
        let mut scaled = params(0.1);
        scaled.scale = true;
        slope.trigger(PolyMask::all_on(), on(), 0);
        for _ in 0..10 {
            slope.process(&scaled, 441);
        }
        let value = slope.process(&scaled, 441).lane(0);
        assert!((value - 72.0).abs() < 1e-3, "scaled glide end was {value}");
    }

    #[test]
    fn positive_power_bends_curve() {
        let mut slope = PortamentoSlope::new(SAMPLE_RATE);
        let mut bent = params(0.1);
        bent.slope_power = PolyF32::splat(5.0);
        slope.trigger(PolyMask::all_on(), on(), 0);
        for _ in 0..5 {
            slope.process(&bent, 441);
        }
        // Negative applied power (power_scale(-5)) rises fast then flattens:
        // at half time the glide is past the linear midpoint.
        let value = slope.process(&bent, 441).lane(0);
        assert!(value > 66.5, "bent midpoint was {value}");
    }
}
