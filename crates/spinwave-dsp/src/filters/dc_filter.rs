//! DC blocking filter (port of `dc_filter.{h,cpp}`).

use spinwave_poly::{PolyF32, PolyMask};

pub const COEFFICIENT_TO_SR_CONSTANT: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
pub struct DcFilter {
    coefficient: f32,
    past_in: PolyF32,
    past_out: PolyF32,
}

impl DcFilter {
    pub fn new(sample_rate: f32) -> DcFilter {
        let mut filter = DcFilter {
            coefficient: 0.0,
            past_in: PolyF32::ZERO,
            past_out: PolyF32::ZERO,
        };
        filter.set_sample_rate(sample_rate);
        filter
    }

    /// A blocker with an explicit corner frequency instead of the
    /// reference's fixed one. The voice blockers keep Vital's ~0.16 Hz
    /// corner (transparent, but a full second to settle); a master blocker
    /// wants a few Hz so the output reaches true silence promptly after
    /// the last voice dies.
    pub fn with_cutoff(cutoff_hz: f32, sample_rate: f32) -> DcFilter {
        let mut filter = DcFilter::new(sample_rate);
        filter.set_cutoff(cutoff_hz, sample_rate);
        filter
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.coefficient = 1.0 - COEFFICIENT_TO_SR_CONSTANT / sample_rate;
    }

    /// Sets the corner frequency in Hz. The reference's constant
    /// corresponds to `1 / (2 * PI)` Hz.
    pub fn set_cutoff(&mut self, cutoff_hz: f32, sample_rate: f32) {
        let constant = std::f32::consts::TAU * cutoff_hz.max(0.0);
        self.coefficient = (1.0 - constant / sample_rate).clamp(0.0, 1.0);
    }

    #[inline(always)]
    pub fn tick(&mut self, audio_in: PolyF32) -> PolyF32 {
        let audio_out =
            (audio_in - self.past_in).mul_add(self.past_out, PolyF32::splat(self.coefficient));
        self.past_out = audio_out;
        self.past_in = audio_in;
        audio_out
    }

    pub fn process(&mut self, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        for (out, &input) in audio_out.iter_mut().zip(audio_in) {
            *out = self.tick(input);
        }
    }

    /// Filters a buffer in place (the master path has nowhere else to put
    /// the result).
    pub fn process_in_place(&mut self, audio: &mut [PolyF32]) {
        for sample in audio.iter_mut() {
            *sample = self.tick(*sample);
        }
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        self.past_in = reset_mask.select(PolyF32::ZERO, self.past_in);
        // Faithful to the C++ (which loads from the already-updated past_in_):
        // unmasked lanes copy past_in into past_out here. Kept bit-for-bit.
        self.past_out = reset_mask.select(PolyF32::ZERO, self.past_in);
    }

    pub fn hard_reset(&mut self) {
        self.reset(PolyMask::all_on());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_dc() {
        let mut filter = DcFilter::new(44100.0);
        let mut out = PolyF32::ZERO;
        for _ in 0..200_000 {
            out = filter.tick(PolyF32::ONE);
        }
        assert!(out.is_finite());
        assert!(out.lane(0).abs() < 0.02, "residual DC {}", out.lane(0));
    }

    #[test]
    fn passes_high_frequencies() {
        let mut filter = DcFilter::new(44100.0);
        let mut peak = 0.0f32;
        for i in 0..1000 {
            let sample = if i % 2 == 0 { 1.0 } else { -1.0 };
            let out = filter.tick(PolyF32::splat(sample));
            if i > 500 {
                peak = peak.max(out.lane(0).abs());
            }
        }
        assert!(peak > 0.9, "Nyquist content attenuated to {peak}");
    }

    #[test]
    fn reset_zeroes_masked_state() {
        let mut filter = DcFilter::new(44100.0);
        for _ in 0..100 {
            filter.tick(PolyF32::ONE);
        }
        filter.hard_reset();
        // With zeroed state, silence stays silent.
        let out = filter.tick(PolyF32::ZERO);
        assert_eq!(out.lane(0), 0.0);
    }

    /// A higher corner settles far faster, which is why the master path
    /// uses one: the reference's corner needs about a second.
    #[test]
    fn cutoff_sets_how_fast_the_blocker_settles() {
        let settle = |filter: &mut DcFilter| {
            // Feed a constant, then count the samples until the output has
            // fallen to 1% of it.
            let mut samples = 0usize;
            for i in 0..44_100 {
                let out = filter.tick(PolyF32::ONE);
                samples = i;
                if out.lane(0).abs() < 0.01 {
                    break;
                }
            }
            samples
        };
        let mut reference = DcFilter::new(44_100.0);
        let mut master = DcFilter::with_cutoff(5.0, 44_100.0);
        let slow = settle(&mut reference);
        let fast = settle(&mut master);
        // 147 ms against more than a second: fast enough that a stopped
        // voice reaches silence promptly.
        assert!(fast * 5 < slow, "5 Hz settled in {fast}, reference in {slow}");
        // Still slow enough to leave the lowest fundamentals alone.
        assert!(fast > 44_100 / 400, "5 Hz settled suspiciously fast: {fast}");
    }
}
