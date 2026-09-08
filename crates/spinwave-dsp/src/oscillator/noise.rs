//! Dedicated noise source: white/pink blend with a spectral tilt and
//! optional stereo decorrelation. The sixth sound source, Serum-2 style.

use spinwave_poly::{PolyF32, PolyMask};

/// Per-block noise parameters.
#[derive(Clone, Copy, Debug)]
pub struct NoiseParams {
    /// Output level `[0, 1]`.
    pub level: PolyF32,
    /// White (0) → pink (1) blend.
    pub pink: PolyF32,
    /// Spectral tilt: -1 = dark (6 dB/oct low-pass-ish), 0 = neutral,
    /// +1 = bright (high-passed).
    pub tilt: PolyF32,
    /// Constant-power pan `[-1, 1]`.
    pub pan: PolyF32,
    /// 0 = mono (same noise both channels), 1 = fully decorrelated stereo.
    pub stereo: PolyF32,
}

impl Default for NoiseParams {
    fn default() -> Self {
        NoiseParams {
            level: PolyF32::splat(0.5),
            pink: PolyF32::ZERO,
            tilt: PolyF32::ZERO,
            pan: PolyF32::ZERO,
            stereo: PolyF32::ONE,
        }
    }
}

/// Four independent xorshift lanes (one per SIMD lane) feeding white and
/// pink (Voss-McCartney style cascade) generators plus a tilt filter.
pub struct NoiseSource {
    states: [u32; 4],
    /// One-pole cascade for pink noise, per lane.
    pink_rows: [[f32; 3]; 4],
    /// Tilt one-pole state.
    tilt_state: PolyF32,
    level_ramp: PolyF32,
}

impl NoiseSource {
    pub fn new() -> NoiseSource {
        Self::with_seed(0x9E3779B9)
    }

    pub fn with_seed(seed: u32) -> NoiseSource {
        let mut states = [0u32; 4];
        let mut state = seed.max(1);
        for slot in &mut states {
            // Split the seed into four decorrelated streams.
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *slot = state;
        }
        NoiseSource {
            states,
            pink_rows: [[0.0; 3]; 4],
            tilt_state: PolyF32::ZERO,
            level_ramp: PolyF32::ZERO,
        }
    }

    pub fn reset(&mut self, _mask: PolyMask) {
        // Noise is stateless perceptually; only clear the filters so a
        // fresh voice doesn't inherit a tilt transient.
        self.pink_rows = [[0.0; 3]; 4];
        self.tilt_state = PolyF32::ZERO;
    }

    #[inline(always)]
    fn next_white(&mut self) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for (state, value) in self.states.iter_mut().zip(&mut out) {
            let mut s = *state;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *state = s;
            // Map to [-1, 1).
            *value = (s as f32 / 2147483648.0) - 1.0;
        }
        out
    }

    /// Renders one block into `out` (adds nothing — overwrites).
    pub fn process(&mut self, params: &NoiseParams, num_samples: usize, out: &mut [PolyF32]) {
        debug_assert!(out.len() >= num_samples);

        let pink_amount = params.pink.clamp(0.0, 1.0);
        let tilt = params.tilt.clamp(-1.0, 1.0);
        // Tilt via a one-pole shelf blend: dark mixes in a low-passed copy,
        // bright subtracts it.
        let tilt_coeff = PolyF32::splat(0.15);

        let pan = params.pan.clamp(-1.0, 1.0);
        let pan_gain = spinwave_poly::math::pan_amplitude(pan);
        let target_level = params.level.clamp(0.0, 1.0) * pan_gain;
        let mut level = self.level_ramp;
        let level_step = (target_level - level) * (1.0 / num_samples as f32);
        self.level_ramp = target_level;

        let mono_mix = PolyF32::ONE - params.stereo.clamp(0.0, 1.0);

        for sample in out.iter_mut().take(num_samples) {
            level += level_step;
            let white_lanes = self.next_white();

            // Pink: three cascaded one-poles at staggered rates, per lane.
            let mut pink_lanes = [0.0f32; 4];
            for lane in 0..4 {
                // Paul Kellet's economy pink filter: three leaky integrators.
                let rows = &mut self.pink_rows[lane];
                rows[0] = 0.99765 * rows[0] + white_lanes[lane] * 0.0990460;
                rows[1] = 0.96300 * rows[1] + white_lanes[lane] * 0.2965164;
                rows[2] = 0.57000 * rows[2] + white_lanes[lane] * 1.0526913;
                pink_lanes[lane] =
                    (rows[0] + rows[1] + rows[2] + white_lanes[lane] * 0.1848) * 0.25;
            }

            let white = PolyF32::from_lanes(white_lanes);
            let pink = PolyF32::from_lanes(pink_lanes);
            let mut value = spinwave_poly::utils::interpolate(white, pink, pink_amount);

            // Stereo amount: blend toward the left-lane value on both
            // channels for mono.
            let mono = value.swap_stereo();
            value = spinwave_poly::utils::interpolate(value, (value + mono) * 0.5, mono_mix);

            // Tilt.
            self.tilt_state += (value - self.tilt_state) * tilt_coeff;
            let dark = self.tilt_state;
            let bright = value - dark;
            let tilted = value + tilt.max(PolyF32::ZERO) * bright * 1.5
                - (-tilt).max(PolyF32::ZERO) * bright;
            *sample = tilted * level;
        }
    }
}

impl Default for NoiseSource {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(params: &NoiseParams, seed: u32, samples: usize) -> Vec<PolyF32> {
        let mut source = NoiseSource::with_seed(seed);
        let mut out = vec![PolyF32::ZERO; samples];
        let mut start = 0;
        while start < samples {
            let block = 128.min(samples - start);
            source.process(params, block, &mut out[start..start + block]);
            start += block;
        }
        out
    }

    fn lane_rms(buffer: &[PolyF32], lane: usize) -> f32 {
        (buffer.iter().map(|v| v.lane(lane) * v.lane(lane)).sum::<f32>() / buffer.len() as f32)
            .sqrt()
    }

    #[test]
    fn deterministic_per_seed() {
        let params = NoiseParams::default();
        let a = render(&params, 42, 512);
        let b = render(&params, 42, 512);
        let c = render(&params, 43, 512);
        assert_eq!(a[100].to_lanes(), b[100].to_lanes());
        assert_ne!(a[100].to_lanes(), c[100].to_lanes());
    }

    #[test]
    fn output_bounded_and_active() {
        let params = NoiseParams::default();
        let out = render(&params, 1, 4096);
        let rms = lane_rms(&out, 0);
        assert!(rms > 0.01, "rms {rms}");
        assert!(out.iter().all(|v| v.is_finite()));
        let peak = out.iter().fold(0.0f32, |a, v| a.max(v.lane(0).abs()));
        assert!(peak < 2.0, "peak {peak}");
    }

    #[test]
    fn pink_is_darker_than_white() {
        let mut white_params = NoiseParams::default();
        white_params.pink = PolyF32::ZERO;
        let mut pink_params = NoiseParams::default();
        pink_params.pink = PolyF32::ONE;

        // Compare high-frequency energy via first differences (a crude
        // high-pass): pink must have relatively less.
        let ratio = |buffer: &[PolyF32]| {
            let mut diff = 0.0f32;
            let mut total = 0.0f32;
            for pair in buffer.windows(2) {
                let d = pair[1].lane(0) - pair[0].lane(0);
                diff += d * d;
                total += pair[1].lane(0) * pair[1].lane(0);
            }
            diff / total.max(1e-9)
        };
        let white = render(&white_params, 7, 8192);
        let pink = render(&pink_params, 7, 8192);
        assert!(ratio(&pink) < ratio(&white) * 0.7);
    }

    #[test]
    fn stereo_zero_is_mono() {
        let mut params = NoiseParams::default();
        params.stereo = PolyF32::ZERO;
        let out = render(&params, 5, 1024);
        for value in &out {
            assert!((value.lane(0) - value.lane(1)).abs() < 1e-5);
        }
    }

    #[test]
    fn level_zero_is_silent() {
        let mut params = NoiseParams::default();
        params.level = PolyF32::ZERO;
        let out = render(&params, 5, 512);
        // After the short level ramp settles, output is silent.
        assert!(out[256..].iter().all(|v| v.lane(0).abs() < 1e-4));
    }
}
