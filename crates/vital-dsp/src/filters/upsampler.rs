//! Zero-order-hold upsampler (port of `upsampler.{h,cpp}`).

use vital_poly::PolyF32;

/// Repeats each input sample `oversample_amount` times.
#[derive(Clone, Copy, Debug, Default)]
pub struct Upsampler;

impl Upsampler {
    pub fn new() -> Upsampler {
        Upsampler
    }

    /// `audio_out.len()` must be `audio_in.len() * oversample_amount`.
    pub fn process(
        &mut self,
        audio_in: &[PolyF32],
        oversample_amount: usize,
        audio_out: &mut [PolyF32],
    ) {
        assert_eq!(audio_out.len(), audio_in.len() * oversample_amount);

        for (i, &sample) in audio_in.iter().enumerate() {
            let offset = i * oversample_amount;
            for s in 0..oversample_amount {
                audio_out[offset + s] = sample;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_each_sample() {
        let mut upsampler = Upsampler::new();
        let input: Vec<PolyF32> = (0..4).map(|i| PolyF32::splat(i as f32)).collect();
        let mut output = vec![PolyF32::ZERO; 8];
        upsampler.process(&input, 2, &mut output);
        let expected = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        for (out, want) in output.iter().zip(expected) {
            assert_eq!(out.lane(0), want);
        }
    }

    #[test]
    fn identity_at_1x() {
        let mut upsampler = Upsampler::new();
        let input: Vec<PolyF32> = (0..8).map(|i| PolyF32::splat((i as f32 * 0.7).sin())).collect();
        let mut output = vec![PolyF32::ZERO; 8];
        upsampler.process(&input, 1, &mut output);
        for (out, input) in output.iter().zip(&input) {
            assert_eq!(out.lane(0), input.lane(0));
        }
    }

    #[test]
    fn upsample_then_decimate_recovers_low_frequency_shape() {
        use crate::filters::decimator::IirHalfbandDecimator;

        let mut upsampler = Upsampler::new();
        let len = 128;
        let input: Vec<PolyF32> = (0..len)
            .map(|i| {
                let phase = 2.0 * core::f32::consts::PI * 300.0 * i as f32 / 44100.0;
                PolyF32::splat(phase.sin())
            })
            .collect();
        let mut oversampled = vec![PolyF32::ZERO; 2 * len];
        upsampler.process(&input, 2, &mut oversampled);

        let mut decimator = IirHalfbandDecimator::new();
        decimator.set_sharp_cutoff(true);
        let mut roundtrip = vec![PolyF32::ZERO; len];
        decimator.process(&oversampled, &mut roundtrip);

        // Past the transient, the roundtrip should track the input closely.
        for i in len / 2..len {
            let error = (roundtrip[i].lane(0) - input[i].lane(0)).abs();
            assert!(error < 0.2, "sample {i}: error {error}");
        }
    }
}
