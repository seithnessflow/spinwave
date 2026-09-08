//! UI peak/RMS metering (port of `PeakMeter`).
//!
//! Tracks a decaying peak and RMS per channel plus a slower "remembered"
//! peak with a hold period. The level output packs peak into the first
//! voice's lanes and RMS into the second voice's lanes, like the reference.

use spinwave_poly::{utils, PolyF32, PolyMask, PolyU32, LANES};

const SAMPLE_DECAY: f32 = 8096.0;
const REMEMBERED_DECAY: f32 = 20000.0;
const REMEMBERED_HOLD: f32 = 50000.0;

/// Lanes of the first voice (`kFirstMask`).
fn first_voice_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, u32::MAX, 0, 0]))
}

/// Per-lane signed `a < b` (matching `poly_int::lessThan`).
fn less_than_signed(a: PolyU32, b: i32) -> PolyMask {
    let mut mask = [0u32; LANES];
    for (out, &lane) in mask.iter_mut().zip(a.0.iter()) {
        if (lane as i32) < b {
            *out = u32::MAX;
        }
    }
    PolyMask::from_u32(PolyU32::from_lanes(mask))
}

#[derive(Clone)]
pub struct PeakMeter {
    oversample_amount: f32,

    current_peak: PolyF32,
    current_square_sum: PolyF32,
    remembered_peak: PolyF32,
    samples_since_remembered: PolyU32,

    level: PolyF32,
}

impl Default for PeakMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl PeakMeter {
    pub fn new() -> Self {
        PeakMeter {
            oversample_amount: 1.0,
            current_peak: PolyF32::ZERO,
            current_square_sum: PolyF32::ZERO,
            remembered_peak: PolyF32::ZERO,
            samples_since_remembered: PolyU32::ZERO,
            level: PolyF32::ZERO,
        }
    }

    pub fn set_oversample_amount(&mut self, oversample_amount: usize) {
        self.oversample_amount = oversample_amount as f32;
    }

    /// Peak in the first voice's lanes, RMS in the second voice's lanes.
    #[inline]
    pub fn level(&self) -> PolyF32 {
        self.level
    }

    /// Held peak with slow decay, for clip/peak indicators.
    #[inline]
    pub fn remembered_peak(&self) -> PolyF32 {
        self.remembered_peak
    }

    pub fn process(&mut self, audio_in: &[PolyF32]) {
        let num_samples = audio_in.len();
        let peak = utils::peak(audio_in, 1);

        let samples = self.oversample_amount * SAMPLE_DECAY;
        let mult = (samples - 1.0) / samples;
        let mut current_peak = self.current_peak;

        let remembered_samples = self.oversample_amount * REMEMBERED_DECAY;
        let remembered_mult = (remembered_samples - 1.0) / remembered_samples;
        let mut current_remembered_peak = self.remembered_peak;

        let mut current_square_sum = self.current_square_sum;

        for &sample in audio_in {
            current_peak *= mult;
            current_remembered_peak *= remembered_mult;
            current_square_sum *= mult;
            current_square_sum += sample * sample;
        }

        self.current_peak = current_peak.max(peak);
        self.samples_since_remembered += PolyU32::splat(num_samples as u32);
        let still_decaying = self.current_peak.lt(current_remembered_peak);
        self.samples_since_remembered = self.samples_since_remembered & still_decaying.to_u32();

        let remembered_hold_samples = (self.oversample_amount * REMEMBERED_HOLD) as i32;
        let hold_mask = less_than_signed(self.samples_since_remembered, remembered_hold_samples);
        current_remembered_peak = hold_mask.select(self.remembered_peak, current_remembered_peak);
        self.remembered_peak = self.current_peak.max(current_remembered_peak);
        self.current_square_sum = current_square_sum;

        let rms = (self.current_square_sum * (1.0 / samples)).sqrt();
        let prepped_rms = rms.swap_voices();
        self.level = first_voice_mask().select(self.current_peak, prepped_rms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_tracks_signal() {
        let mut meter = PeakMeter::new();
        let buffer = [PolyF32::splat(0.5); 64];
        meter.process(&buffer);
        let peak = meter.level().lane(0);
        assert!((peak - 0.5).abs() < 1e-4, "peak was {peak}");
    }

    #[test]
    fn peak_decays_after_silence() {
        let mut meter = PeakMeter::new();
        meter.process(&[PolyF32::splat(1.0); 64]);
        let loud = meter.level().lane(0);
        let silence = [PolyF32::ZERO; 128];
        for _ in 0..200 {
            meter.process(&silence);
        }
        let quiet = meter.level().lane(0);
        assert!(quiet < loud * 0.1, "peak should decay: {loud} -> {quiet}");
    }

    #[test]
    fn rms_in_second_voice_lanes() {
        let mut meter = PeakMeter::new();
        // Steady full-scale square: RMS ~ 1 once the window fills.
        let buffer = [PolyF32::splat(1.0); 128];
        for _ in 0..200 {
            meter.process(&buffer);
        }
        let rms = meter.level().lane(2);
        assert!(rms > 0.5, "rms should build up, got {rms}");
        let peak = meter.level().lane(0);
        assert!((peak - 1.0).abs() < 1e-3);
    }

    #[test]
    fn remembered_peak_holds() {
        let mut meter = PeakMeter::new();
        meter.process(&[PolyF32::splat(0.8); 64]);
        let remembered = meter.remembered_peak().lane(0);
        assert!((remembered - 0.8).abs() < 1e-3);

        // Shortly after, the remembered peak still holds near the maximum
        // while the live peak has begun decaying.
        let silence = [PolyF32::ZERO; 128];
        for _ in 0..20 {
            meter.process(&silence);
        }
        assert!(meter.remembered_peak().lane(0) > 0.75);
        assert!(meter.level().lane(0) < meter.remembered_peak().lane(0));
    }
}
