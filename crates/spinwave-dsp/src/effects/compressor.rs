//! Per-band compressor/expander core (port of Vital's `Compressor`).
//!
//! Envelope follower on the squared signal plus an upper (compress) and
//! lower (expand) threshold/ratio pair. The multiband wiring â€”
//! Linkwitz-Riley band splitting and packing two bands into the voice
//! lanes â€” lives in the module layer; this processes one packed band pair
//! (first voice = first band, second voice = second band).

use spinwave_poly::{math, PolyF32};

use super::lanes::first_voice_mask;

const RMS_TIME: f32 = 0.025;
const MAX_EXPAND_MULT: f32 = 32.0;
const MIN_GAIN_DB: f32 = -30.0;
const MAX_GAIN_DB: f32 = 30.0;
const MIN_THRESHOLD_DB: f32 = -100.0;
const MAX_THRESHOLD_DB: f32 = 12.0;
const MIN_SAMPLE_ENVELOPE: f32 = 5.0;
const MS_PER_SEC: f32 = 1000.0;

/// Vital's multiband attack/release baselines (milliseconds).
pub const LOW_ATTACK_MS: f32 = 2.8;
pub const BAND_ATTACK_MS: f32 = 1.4;
pub const HIGH_ATTACK_MS: f32 = 0.7;
pub const LOW_RELEASE_MS: f32 = 40.0;
pub const BAND_RELEASE_MS: f32 = 28.0;
pub const HIGH_RELEASE_MS: f32 = 15.0;

/// Block-rate compressor parameters.
#[derive(Clone, Copy, Debug)]
pub struct CompressorParams {
    /// Threshold (dB) above which the upper ratio compresses.
    pub upper_threshold_db: PolyF32,
    /// Threshold (dB) below which the lower ratio expands.
    pub lower_threshold_db: PolyF32,
    /// Upper (compression) ratio control in [0, 1].
    pub upper_ratio: PolyF32,
    /// Lower (expansion) ratio control in [-1, 1].
    pub lower_ratio: PolyF32,
    /// Output gain in dB, clamped to [-30, 30].
    pub output_gain_db: PolyF32,
    /// Attack control in [0, 1] (exponentially scales the base attack).
    pub attack: PolyF32,
    /// Release control in [0, 1] (exponentially scales the base release).
    pub release: PolyF32,
    /// Dry/wet mix in [0, 1].
    pub mix: PolyF32,
}

impl Default for CompressorParams {
    fn default() -> CompressorParams {
        CompressorParams {
            upper_threshold_db: PolyF32::ZERO,
            lower_threshold_db: PolyF32::splat(MIN_THRESHOLD_DB),
            upper_ratio: PolyF32::ZERO,
            lower_ratio: PolyF32::ZERO,
            output_gain_db: PolyF32::ZERO,
            attack: PolyF32::splat(0.5),
            release: PolyF32::splat(0.5),
            mix: PolyF32::ONE,
        }
    }
}

pub struct Compressor {
    sample_rate: f32,
    base_attack_ms: PolyF32,
    base_release_ms: PolyF32,

    input_mean_squared: PolyF32,
    output_mean_squared: PolyF32,
    high_enveloped_mean_squared: PolyF32,
    low_enveloped_mean_squared: PolyF32,

    mix: PolyF32,
    output_mult: PolyF32,
}

impl Compressor {
    /// `first` values apply to the first voice lanes, `second` to the
    /// second voice lanes (two bands packed per vector).
    pub fn new(
        base_attack_ms_first: f32,
        base_release_ms_first: f32,
        base_attack_ms_second: f32,
        base_release_ms_second: f32,
        sample_rate: f32,
    ) -> Compressor {
        let first = first_voice_mask();
        Compressor {
            sample_rate,
            base_attack_ms: first.select(
                PolyF32::splat(base_attack_ms_first),
                PolyF32::splat(base_attack_ms_second),
            ),
            base_release_ms: first.select(
                PolyF32::splat(base_release_ms_first),
                PolyF32::splat(base_release_ms_second),
            ),
            input_mean_squared: PolyF32::ZERO,
            output_mean_squared: PolyF32::ZERO,
            high_enveloped_mean_squared: PolyF32::ZERO,
            low_enveloped_mean_squared: PolyF32::ZERO,
            mix: PolyF32::ZERO,
            output_mult: PolyF32::ZERO,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn reset(&mut self) {
        self.input_mean_squared = PolyF32::ZERO;
        self.output_mean_squared = PolyF32::ZERO;
        self.output_mult = PolyF32::ZERO;
        self.mix = PolyF32::ZERO;
        self.high_enveloped_mean_squared = PolyF32::ZERO;
        self.low_enveloped_mean_squared = PolyF32::ZERO;
    }

    pub fn input_mean_squared(&self) -> PolyF32 {
        self.input_mean_squared
    }

    pub fn output_mean_squared(&self) -> PolyF32 {
        self.output_mean_squared
    }

    pub fn process(
        &mut self,
        params: &CompressorParams,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        assert_eq!(audio_in.len(), audio_out.len());
        if audio_in.is_empty() {
            return;
        }
        self.process_rms(params, audio_in, audio_out);
        self.input_mean_squared = self.compute_mean_squared(audio_in, self.input_mean_squared);
        self.output_mean_squared = self.compute_mean_squared(audio_out, self.output_mean_squared);
        self.scale_output(params, audio_in, audio_out);
    }

    fn process_rms(&mut self, params: &CompressorParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let samples_per_ms = self.sample_rate / MS_PER_SEC;
        let attack_mult = self.base_attack_ms * samples_per_ms;
        let release_mult = self.base_release_ms * samples_per_ms;
        let attack_exponent = params.attack.clamp(0.0, 1.0) * 8.0 - 4.0;
        let release_exponent = params.release.clamp(0.0, 1.0) * 8.0 - 4.0;
        let envelope_attack_samples =
            (math::exp(attack_exponent) * attack_mult).max(PolyF32::splat(MIN_SAMPLE_ENVELOPE));
        let envelope_release_samples =
            (math::exp(release_exponent) * release_mult).max(PolyF32::splat(MIN_SAMPLE_ENVELOPE));

        let attack_scale = PolyF32::ONE / (envelope_attack_samples + 1.0);
        let release_scale = PolyF32::ONE / (envelope_release_samples + 1.0);

        let mut upper_threshold = math::db_to_magnitude(
            params.upper_threshold_db.clamp(MIN_THRESHOLD_DB, MAX_THRESHOLD_DB),
        );
        upper_threshold *= upper_threshold;
        let mut lower_threshold = math::db_to_magnitude(
            params.lower_threshold_db.clamp(MIN_THRESHOLD_DB, MAX_THRESHOLD_DB),
        );
        lower_threshold *= lower_threshold;

        let upper_ratio = params.upper_ratio.clamp(0.0, 1.0) * 0.5;
        let lower_ratio = params.lower_ratio.clamp(-1.0, 1.0) * 0.5;

        let mut low_enveloped_mean_squared = self.low_enveloped_mean_squared;
        let mut high_enveloped_mean_squared = self.high_enveloped_mean_squared;

        for (out, &sample) in audio_out.iter_mut().zip(audio_in) {
            let sample_squared = sample * sample;

            let high_attack_mask = sample_squared.gt(high_enveloped_mean_squared);
            let high_samples =
                high_attack_mask.select(envelope_attack_samples, envelope_release_samples);
            let high_scale = high_attack_mask.select(attack_scale, release_scale);

            high_enveloped_mean_squared =
                (sample_squared + high_enveloped_mean_squared * high_samples) * high_scale;
            high_enveloped_mean_squared = high_enveloped_mean_squared.max(upper_threshold);

            let upper_mag_delta = upper_threshold / high_enveloped_mean_squared;
            let upper_mult = math::pow(upper_mag_delta, upper_ratio);

            let low_attack_mask = sample_squared.gt(low_enveloped_mean_squared);
            let low_samples =
                low_attack_mask.select(envelope_attack_samples, envelope_release_samples);
            let low_scale = low_attack_mask.select(attack_scale, release_scale);

            low_enveloped_mean_squared =
                (sample_squared + low_enveloped_mean_squared * low_samples) * low_scale;
            low_enveloped_mean_squared = low_enveloped_mean_squared.min(lower_threshold);

            let lower_mag_delta = lower_threshold / low_enveloped_mean_squared;
            let lower_mult = math::pow(lower_mag_delta, lower_ratio);

            let gain_compression = (upper_mult * lower_mult).clamp(0.0, MAX_EXPAND_MULT);
            *out = gain_compression * sample;
            debug_assert!(out.is_finite());
        }

        self.low_enveloped_mean_squared = low_enveloped_mean_squared;
        self.high_enveloped_mean_squared = high_enveloped_mean_squared;
    }

    fn scale_output(&mut self, params: &CompressorParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let num_samples = audio_in.len();

        let mut current_output_mult = self.output_mult;
        let gain = params.output_gain_db.clamp(MIN_GAIN_DB, MAX_GAIN_DB);
        self.output_mult = math::db_to_magnitude(gain);
        let delta_output_mult = (self.output_mult - current_output_mult) * (1.0 / num_samples as f32);

        let mut current_mix = self.mix;
        self.mix = params.mix.clamp(0.0, 1.0);
        let delta_mix = (self.mix - current_mix) * (1.0 / num_samples as f32);

        for (out, &dry) in audio_out.iter_mut().zip(audio_in) {
            current_output_mult += delta_output_mult;
            current_mix += delta_mix;
            *out = spinwave_poly::utils::interpolate(dry, *out * current_output_mult, current_mix);
            debug_assert!(out.is_finite());
        }
    }

    fn compute_mean_squared(&self, audio: &[PolyF32], mut mean_squared: PolyF32) -> PolyF32 {
        let rms_samples = (RMS_TIME * self.sample_rate) as i32 as f32;
        let rms_adjusted = rms_samples - 1.0;
        let input_scale = 1.0 / rms_samples;

        for &sample in audio {
            let sample_squared = sample * sample;
            mean_squared = (mean_squared * rms_adjusted + sample_squared) * input_scale;
        }
        mean_squared
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 128;

    fn run_sine(compressor: &mut Compressor, params: &CompressorParams, amplitude: f32, blocks: usize)
        -> (f32, f32) {
        let mut in_rms = 0.0f64;
        let mut out_rms = 0.0f64;
        let mut count = 0usize;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..blocks {
            let input: Vec<PolyF32> = (0..BLOCK)
                .map(|i| {
                    let t = (block * BLOCK + i) as f32 / SAMPLE_RATE;
                    PolyF32::splat((2.0 * core::f32::consts::PI * 220.0 * t).sin() * amplitude)
                })
                .collect();
            compressor.process(params, &input, &mut output);
            if block >= blocks / 2 {
                for (out, inp) in output.iter().zip(&input) {
                    assert!(out.is_finite());
                    in_rms += (inp.lane(0) as f64) * (inp.lane(0) as f64);
                    out_rms += (out.lane(0) as f64) * (out.lane(0) as f64);
                    count += 1;
                }
            }
        }
        (
            (in_rms / count as f64).sqrt() as f32,
            (out_rms / count as f64).sqrt() as f32,
        )
    }

    #[test]
    fn reduces_gain_above_threshold() {
        let mut compressor =
            Compressor::new(BAND_ATTACK_MS, BAND_RELEASE_MS, BAND_ATTACK_MS, BAND_RELEASE_MS, SAMPLE_RATE);
        let params = CompressorParams {
            upper_threshold_db: PolyF32::splat(-30.0),
            upper_ratio: PolyF32::ONE,
            ..CompressorParams::default()
        };
        let (in_rms, out_rms) = run_sine(&mut compressor, &params, 0.8, 60);
        assert!(
            out_rms < in_rms * 0.7,
            "no compression: in {in_rms} out {out_rms}"
        );
    }

    #[test]
    fn transparent_below_threshold() {
        let mut compressor =
            Compressor::new(BAND_ATTACK_MS, BAND_RELEASE_MS, BAND_ATTACK_MS, BAND_RELEASE_MS, SAMPLE_RATE);
        let params = CompressorParams {
            upper_threshold_db: PolyF32::splat(0.0),
            upper_ratio: PolyF32::ONE,
            ..CompressorParams::default()
        };
        let (in_rms, out_rms) = run_sine(&mut compressor, &params, 0.05, 60);
        assert!(
            (out_rms - in_rms).abs() < in_rms * 0.02,
            "not transparent: in {in_rms} out {out_rms}"
        );
    }

    #[test]
    fn expander_reduces_quiet_signal() {
        let mut compressor =
            Compressor::new(LOW_ATTACK_MS, LOW_RELEASE_MS, HIGH_ATTACK_MS, HIGH_RELEASE_MS, SAMPLE_RATE);
        let params = CompressorParams {
            lower_threshold_db: PolyF32::splat(-20.0),
            // Negative lower ratio = downward expansion of quiet signals.
            lower_ratio: PolyF32::splat(-1.0),
            ..CompressorParams::default()
        };
        let (in_rms, out_rms) = run_sine(&mut compressor, &params, 0.02, 60);
        assert!(
            out_rms < in_rms * 0.9,
            "no expansion: in {in_rms} out {out_rms}"
        );
    }

    #[test]
    fn meters_track_signal() {
        let mut compressor =
            Compressor::new(BAND_ATTACK_MS, BAND_RELEASE_MS, BAND_ATTACK_MS, BAND_RELEASE_MS, SAMPLE_RATE);
        let params = CompressorParams::default();
        run_sine(&mut compressor, &params, 0.5, 40);
        assert!(compressor.input_mean_squared().lane(0) > 0.0);
        assert!(compressor.output_mean_squared().lane(0) > 0.0);
    }
}
