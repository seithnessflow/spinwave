//! Waveshaping distortion: soft/hard clip, folds, bit crush and downsample
//! (port of Vital's `Distortion`).
//!
//! Like the reference, samples are compacted two-per-vector before shaping
//! (`[L0, R0, L1, R1]` holds two consecutive stereo samples), which halves
//! the SIMD work and â€” for the downsampler â€” is part of the sound: the
//! second sample of each pair reuses the first, an extra 2x decimation.

use spinwave_poly::constants::{MAX_BUFFER_SIZE, MAX_OVERSAMPLE};
use spinwave_poly::{math, PolyF32};

use super::lanes::first_voice_mask;

pub const MAX_DRIVE_DB: f32 = 30.0;
pub const MIN_DRIVE_DB: f32 = -30.0;
const PERIOD_SCALE: f32 = 1.0 / 88200.0;
const MIN_DISTORTION_MULT: f32 = 32.0 / i32::MAX as f32;
const MAX_BLOCK: usize = MAX_BUFFER_SIZE * MAX_OVERSAMPLE;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DistortionType {
    #[default]
    SoftClip,
    HardClip,
    LinearFold,
    SinFold,
    BitCrush,
    DownSample,
}

#[inline(always)]
fn linear_fold(value: PolyF32, drive: PolyF32) -> PolyF32 {
    let adjust = value * drive * 0.25 + 0.75;
    let range = adjust.fract();
    (range * -4.0 + 2.0).abs() - 1.0
}

#[inline(always)]
fn sin_fold(value: PolyF32, drive: PolyF32) -> PolyF32 {
    let adjust = value * drive * -0.25 + 0.5;
    math::sin1(adjust.fract())
}

#[inline(always)]
fn soft_clip(value: PolyF32, drive: PolyF32) -> PolyF32 {
    math::tanh(value * drive)
}

#[inline(always)]
fn hard_clip(value: PolyF32, drive: PolyF32) -> PolyF32 {
    (value * drive).clamp(-1.0, 1.0)
}

#[inline(always)]
fn bit_crush(value: PolyF32, drive: PolyF32) -> PolyF32 {
    (value / drive).round() * drive
}

/// Drive in dB mapped to a magnitude multiplier.
#[inline(always)]
pub fn drive_db_scale(db: PolyF32) -> PolyF32 {
    math::db_to_magnitude(db.clamp(MIN_DRIVE_DB, MAX_DRIVE_DB))
}

/// Drive in dB mapped to the bit-crush quantization step.
#[inline(always)]
pub fn bit_crush_scale(db: PolyF32) -> PolyF32 {
    const DRIVE_SCALE: f32 = 1.0 / (MAX_DRIVE_DB - MIN_DRIVE_DB);
    let drive = (db - MIN_DRIVE_DB).max(PolyF32::ZERO) * DRIVE_SCALE;
    (drive * drive).clamp(MIN_DISTORTION_MULT, 1.0)
}

/// Drive in dB mapped to the downsample period (times [`PERIOD_SCALE`]).
#[inline(always)]
pub fn down_sample_scale(db: PolyF32) -> PolyF32 {
    const DRIVE_SCALE: f32 = 1.0 / (MAX_DRIVE_DB - MIN_DRIVE_DB);
    let mut drive = (db - MIN_DRIVE_DB).max(PolyF32::ZERO) * DRIVE_SCALE;
    drive = -drive + 1.0;
    drive = PolyF32::ONE / (drive * drive).clamp(MIN_DISTORTION_MULT, 1.0);
    (drive * 0.99).max(PolyF32::ONE) * PERIOD_SCALE
}

/// Maps a raw drive (dB) input to the internal drive value for `dtype`.
pub fn drive_value(dtype: DistortionType, input_drive: PolyF32) -> PolyF32 {
    match dtype {
        DistortionType::BitCrush => bit_crush_scale(input_drive),
        DistortionType::DownSample => down_sample_scale(input_drive),
        _ => drive_db_scale(input_drive),
    }
}

/// Stateless shaping of a single value (used for UI display in Vital).
pub fn driven_value(dtype: DistortionType, value: PolyF32, drive: PolyF32) -> PolyF32 {
    match dtype {
        DistortionType::SoftClip => soft_clip(value, drive),
        DistortionType::HardClip => hard_clip(value, drive),
        DistortionType::LinearFold => linear_fold(value, drive),
        DistortionType::SinFold => sin_fold(value, drive),
        DistortionType::BitCrush => bit_crush(value, drive),
        DistortionType::DownSample => {
            bit_crush(value, PolyF32::splat(1.001) - PolyF32::splat(PERIOD_SCALE) / drive)
        }
    }
}

/// Packs consecutive sample pairs into single vectors; returns packed length.
fn compact_audio(audio: &mut [PolyF32], num_samples: usize) -> usize {
    let num_full = num_samples / 2;
    for i in 0..num_full {
        let in_index = 2 * i;
        audio[i] = PolyF32::compact_first_voices(audio[in_index], audio[in_index + 1]);
    }
    let num_remaining = num_samples % 2;
    if num_remaining != 0 {
        audio[num_full] = audio[num_samples - 1];
    }
    num_full + num_remaining
}

fn compact_into(dest: &mut [PolyF32], source: &[PolyF32]) -> usize {
    let num_samples = source.len();
    let num_full = num_samples / 2;
    for (i, out) in dest.iter_mut().enumerate().take(num_full) {
        let in_index = 2 * i;
        *out = PolyF32::compact_first_voices(source[in_index], source[in_index + 1]);
    }
    let num_remaining = num_samples % 2;
    if num_remaining != 0 {
        dest[num_full] = source[num_samples - 1];
    }
    num_full + num_remaining
}

/// Expands packed pairs back to one sample per vector, in place.
fn expand_audio(audio: &mut [PolyF32], num_samples: usize) {
    let num_full = num_samples / 2;
    if !num_samples.is_multiple_of(2) {
        audio[num_samples - 1] = audio[num_full];
    }
    for i in (0..num_full).rev() {
        let out_index = 2 * i;
        let value = audio[i];
        audio[out_index] = value;
        audio[out_index + 1] = value.swap_voices();
    }
}

pub struct Distortion {
    sample_rate: f32,
    last_distorted_value: PolyF32,
    current_samples: PolyF32,
    current_type: Option<DistortionType>,
    drive_scratch: Vec<PolyF32>,
}

impl Distortion {
    pub fn new(sample_rate: f32) -> Distortion {
        Distortion {
            sample_rate,
            last_distorted_value: PolyF32::ZERO,
            current_samples: PolyF32::ZERO,
            current_type: None,
            drive_scratch: vec![PolyF32::ZERO; MAX_BLOCK],
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn hard_reset(&mut self) {
        self.last_distorted_value = PolyF32::ZERO;
        self.current_samples = PolyF32::ZERO;
        self.current_type = None;
    }

    /// Shapes `audio` in place. `drive_db` is the audio-rate drive input in
    /// dB (pre-scaling), one value per input sample.
    pub fn process(&mut self, dtype: DistortionType, drive_db: &[PolyF32], audio: &mut [PolyF32]) {
        let num_samples = audio.len();
        assert_eq!(drive_db.len(), num_samples);
        assert!(num_samples <= MAX_BLOCK);
        if num_samples == 0 {
            return;
        }

        let compact_samples = compact_audio(audio, num_samples);
        let mut drive_scratch = core::mem::take(&mut self.drive_scratch);
        compact_into(&mut drive_scratch, drive_db);
        let drive = &drive_scratch[..compact_samples];

        if self.current_type != Some(dtype) {
            self.current_type = Some(dtype);
            self.last_distorted_value = PolyF32::ZERO;
            self.current_samples = PolyF32::ZERO;
        }

        let audio_compact = &mut audio[..compact_samples];
        match dtype {
            DistortionType::SoftClip => {
                Self::process_time_invariant(audio_compact, drive, soft_clip, drive_db_scale)
            }
            DistortionType::HardClip => {
                Self::process_time_invariant(audio_compact, drive, hard_clip, drive_db_scale)
            }
            DistortionType::LinearFold => {
                Self::process_time_invariant(audio_compact, drive, linear_fold, drive_db_scale)
            }
            DistortionType::SinFold => {
                Self::process_time_invariant(audio_compact, drive, sin_fold, drive_db_scale)
            }
            DistortionType::BitCrush => {
                Self::process_time_invariant(audio_compact, drive, bit_crush, bit_crush_scale)
            }
            DistortionType::DownSample => self.process_down_sample(audio_compact, drive),
        }
        self.drive_scratch = drive_scratch;

        expand_audio(audio, num_samples);
    }

    fn process_time_invariant(
        audio: &mut [PolyF32],
        drive: &[PolyF32],
        distort: impl Fn(PolyF32, PolyF32) -> PolyF32,
        scale: impl Fn(PolyF32) -> PolyF32,
    ) {
        for (sample, &drive_db) in audio.iter_mut().zip(drive) {
            let current_drive = scale(drive_db);
            *sample = distort(*sample, current_drive);
            debug_assert!(sample.is_finite());
        }
    }

    fn process_down_sample(&mut self, audio: &mut [PolyF32], drive: &[PolyF32]) {
        let sample_rate = self.sample_rate;
        let mut current_samples = self.current_samples;

        for (sample, &drive_db) in audio.iter_mut().zip(drive) {
            let current_period = down_sample_scale(drive_db) * sample_rate;
            current_samples += 1.0;

            let current_sample = *sample;
            let mut current_downsample = current_sample & first_voice_mask();
            current_downsample += current_downsample.swap_voices();

            let update = current_samples.ge(current_period);
            self.last_distorted_value =
                update.select(current_downsample, self.last_distorted_value);
            current_samples = update.select(current_samples - current_period, current_samples);
            *sample = self.last_distorted_value;
        }

        self.current_samples = current_samples;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 88200.0;

    fn sine_block(len: usize, amplitude: f32) -> Vec<PolyF32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE;
                PolyF32::splat((2.0 * core::f32::consts::PI * 440.0 * t).sin() * amplitude)
            })
            .collect()
    }

    #[test]
    fn hard_clip_at_zero_drive_is_transparent() {
        let mut distortion = Distortion::new(SAMPLE_RATE);
        let input = sine_block(128, 0.5);
        let mut audio = input.clone();
        let drive = vec![PolyF32::ZERO; 128];
        distortion.process(DistortionType::HardClip, &drive, &mut audio);
        for (out, inp) in audio.iter().zip(&input) {
            // 0 dB drive = unit gain; below the clip point this is identity.
            assert!((out.lane(0) - inp.lane(0)).abs() < 1e-5);
            assert!((out.lane(1) - inp.lane(1)).abs() < 1e-5);
        }
    }

    #[test]
    fn all_types_finite_and_bounded_on_sine_and_impulse() {
        let types = [
            DistortionType::SoftClip,
            DistortionType::HardClip,
            DistortionType::LinearFold,
            DistortionType::SinFold,
            DistortionType::BitCrush,
            DistortionType::DownSample,
        ];
        for dtype in types {
            let mut distortion = Distortion::new(SAMPLE_RATE);
            let mut audio = sine_block(127, 0.9);
            audio[0] = PolyF32::ONE;
            let drive = vec![PolyF32::splat(20.0); 127];
            distortion.process(dtype, &drive, &mut audio);
            for sample in &audio {
                assert!(sample.is_finite(), "{dtype:?} not finite");
                assert!(sample.abs().lane(0) <= 8.0, "{dtype:?} unbounded");
            }
        }
    }

    #[test]
    fn down_sample_holds_values() {
        let mut distortion = Distortion::new(SAMPLE_RATE);
        let mut audio: Vec<PolyF32> =
            (0..128).map(|i| PolyF32::splat(i as f32 / 128.0)).collect();
        // High drive = long hold periods (~16 packed samples at +15 dB).
        let drive = vec![PolyF32::splat(15.0); 128];
        distortion.process(DistortionType::DownSample, &drive, &mut audio);
        let mut distinct = 1;
        for i in 1..128 {
            if (audio[i].lane(0) - audio[i - 1].lane(0)).abs() > 1e-9 {
                distinct += 1;
            }
        }
        assert!(distinct < 12, "downsampler held {distinct} distinct values");
        for sample in &audio {
            assert!(sample.is_finite());
        }
    }
}
