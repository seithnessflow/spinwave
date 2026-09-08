//! Multiband compressor (port of Vital's `MultibandCompressor` plus the
//! `CompressorModule` wiring).
//!
//! Two Linkwitz-Riley crossovers (120 Hz and 2500 Hz) split the signal; the
//! bands are packed into the two voice slots so two [`Compressor`] cores
//! process all three bands: the "low/band" core carries the low band in its
//! first voice and the mid band in its second, the "band/high" core carries
//! the mid band first and the high band second. Per-band input/output
//! mean-squared levels are exposed for the UI meters.

use spinwave_poly::constants::{MAX_BUFFER_SIZE, MAX_OVERSAMPLE};
use spinwave_poly::{PolyF32, PolyMask};

use crate::filters::LinkwitzRileyFilter;

use super::compressor::{
    Compressor, CompressorParams, BAND_ATTACK_MS, BAND_RELEASE_MS, HIGH_ATTACK_MS,
    HIGH_RELEASE_MS, LOW_ATTACK_MS, LOW_RELEASE_MS,
};
use super::lanes::first_voice_mask;

/// Crossover frequency between the low and mid bands (Hz).
pub const LOW_BAND_CROSSOVER: f32 = 120.0;
/// Crossover frequency between the mid and high bands (Hz).
pub const BAND_HIGH_CROSSOVER: f32 = 2500.0;

const MAX_BLOCK: usize = MAX_BUFFER_SIZE * MAX_OVERSAMPLE;

/// Which bands are active (C++ `MultibandCompressor::BandOptions`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BandOptions {
    /// All three bands compress independently.
    #[default]
    Multiband,
    /// Split at 120 Hz only: "low" and "rest" bands.
    LowBand,
    /// Split at 2500 Hz only: "rest" and "high" bands.
    HighBand,
    /// No band split; single full-range compressor.
    SingleBand,
}

/// Block-rate multiband compressor parameters. Thresholds in dB, ratio
/// controls in [0, 1] (upper) / [-1, 1] (lower), gains in dB.
#[derive(Clone, Copy, Debug)]
pub struct MultibandCompressorParams {
    pub enabled_bands: BandOptions,
    /// Attack control in [0, 1] (exponentially scales each band's base attack).
    pub attack: PolyF32,
    /// Release control in [0, 1] (exponentially scales each band's base release).
    pub release: PolyF32,
    /// Dry/wet mix in [0, 1].
    pub mix: PolyF32,
    pub low_upper_ratio: PolyF32,
    pub band_upper_ratio: PolyF32,
    pub high_upper_ratio: PolyF32,
    pub low_lower_ratio: PolyF32,
    pub band_lower_ratio: PolyF32,
    pub high_lower_ratio: PolyF32,
    pub low_upper_threshold_db: PolyF32,
    pub band_upper_threshold_db: PolyF32,
    pub high_upper_threshold_db: PolyF32,
    pub low_lower_threshold_db: PolyF32,
    pub band_lower_threshold_db: PolyF32,
    pub high_lower_threshold_db: PolyF32,
    pub low_output_gain_db: PolyF32,
    pub band_output_gain_db: PolyF32,
    pub high_output_gain_db: PolyF32,
}

impl Default for MultibandCompressorParams {
    fn default() -> MultibandCompressorParams {
        MultibandCompressorParams {
            enabled_bands: BandOptions::Multiband,
            attack: PolyF32::splat(0.5),
            release: PolyF32::splat(0.5),
            mix: PolyF32::ONE,
            low_upper_ratio: PolyF32::ZERO,
            band_upper_ratio: PolyF32::ZERO,
            high_upper_ratio: PolyF32::ZERO,
            low_lower_ratio: PolyF32::ZERO,
            band_lower_ratio: PolyF32::ZERO,
            high_lower_ratio: PolyF32::ZERO,
            low_upper_threshold_db: PolyF32::ZERO,
            band_upper_threshold_db: PolyF32::ZERO,
            high_upper_threshold_db: PolyF32::ZERO,
            low_lower_threshold_db: PolyF32::splat(-100.0),
            band_lower_threshold_db: PolyF32::splat(-100.0),
            high_lower_threshold_db: PolyF32::splat(-100.0),
            low_output_gain_db: PolyF32::ZERO,
            band_output_gain_db: PolyF32::ZERO,
            high_output_gain_db: PolyF32::ZERO,
        }
    }
}

pub struct MultibandCompressor {
    low_band_filter: LinkwitzRileyFilter,
    band_high_filter: LinkwitzRileyFilter,
    low_band_compressor: Compressor,
    band_high_compressor: Compressor,

    was_low_enabled: bool,
    was_high_enabled: bool,

    low_input_mean_squared: PolyF32,
    band_input_mean_squared: PolyF32,
    high_input_mean_squared: PolyF32,
    low_output_mean_squared: PolyF32,
    band_output_mean_squared: PolyF32,
    high_output_mean_squared: PolyF32,

    low_buffer: Vec<PolyF32>,
    high_buffer: Vec<PolyF32>,
    packed_buffer: Vec<PolyF32>,
    low_band_out: Vec<PolyF32>,
    band_high_out: Vec<PolyF32>,
}

impl MultibandCompressor {
    pub fn new(sample_rate: f32) -> MultibandCompressor {
        MultibandCompressor {
            low_band_filter: LinkwitzRileyFilter::new(LOW_BAND_CROSSOVER, sample_rate),
            band_high_filter: LinkwitzRileyFilter::new(BAND_HIGH_CROSSOVER, sample_rate),
            low_band_compressor: Compressor::new(
                LOW_ATTACK_MS,
                LOW_RELEASE_MS,
                BAND_ATTACK_MS,
                BAND_RELEASE_MS,
                sample_rate,
            ),
            band_high_compressor: Compressor::new(
                BAND_ATTACK_MS,
                BAND_RELEASE_MS,
                HIGH_ATTACK_MS,
                HIGH_RELEASE_MS,
                sample_rate,
            ),
            // false/false so the first block in the default multiband mode
            // triggers a full reset, matching the reference.
            was_low_enabled: false,
            was_high_enabled: false,
            low_input_mean_squared: PolyF32::ZERO,
            band_input_mean_squared: PolyF32::ZERO,
            high_input_mean_squared: PolyF32::ZERO,
            low_output_mean_squared: PolyF32::ZERO,
            band_output_mean_squared: PolyF32::ZERO,
            high_output_mean_squared: PolyF32::ZERO,
            low_buffer: vec![PolyF32::ZERO; MAX_BLOCK],
            high_buffer: vec![PolyF32::ZERO; MAX_BLOCK],
            packed_buffer: vec![PolyF32::ZERO; MAX_BLOCK],
            low_band_out: vec![PolyF32::ZERO; MAX_BLOCK],
            band_high_out: vec![PolyF32::ZERO; MAX_BLOCK],
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.low_band_filter.set_sample_rate(sample_rate);
        self.band_high_filter.set_sample_rate(sample_rate);
        self.low_band_compressor.set_sample_rate(sample_rate);
        self.band_high_compressor.set_sample_rate(sample_rate);
    }

    pub fn reset(&mut self) {
        self.low_band_filter.reset(PolyMask::all_on());
        self.band_high_filter.reset(PolyMask::all_on());
        self.low_band_compressor.reset();
        self.band_high_compressor.reset();

        self.low_input_mean_squared = PolyF32::ZERO;
        self.band_input_mean_squared = PolyF32::ZERO;
        self.high_input_mean_squared = PolyF32::ZERO;
        self.low_output_mean_squared = PolyF32::ZERO;
        self.band_output_mean_squared = PolyF32::ZERO;
        self.high_output_mean_squared = PolyF32::ZERO;
    }

    // -- UI meter readouts (mean squared levels, first-voice lanes) ----------

    pub fn low_input_mean_squared(&self) -> PolyF32 {
        self.low_input_mean_squared
    }

    pub fn band_input_mean_squared(&self) -> PolyF32 {
        self.band_input_mean_squared
    }

    pub fn high_input_mean_squared(&self) -> PolyF32 {
        self.high_input_mean_squared
    }

    pub fn low_output_mean_squared(&self) -> PolyF32 {
        self.low_output_mean_squared
    }

    pub fn band_output_mean_squared(&self) -> PolyF32 {
        self.band_output_mean_squared
    }

    pub fn high_output_mean_squared(&self) -> PolyF32 {
        self.high_output_mean_squared
    }

    pub fn process(
        &mut self,
        params: &MultibandCompressorParams,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        let num_samples = audio_in.len();
        assert_eq!(num_samples, audio_out.len());
        assert!(num_samples <= MAX_BLOCK);
        if num_samples == 0 {
            return;
        }

        let low_enabled = matches!(
            params.enabled_bands,
            BandOptions::Multiband | BandOptions::LowBand
        );
        let high_enabled = matches!(
            params.enabled_bands,
            BandOptions::Multiband | BandOptions::HighBand
        );

        // Pack per-band controls into the two voices of each compressor core.
        let first = first_voice_mask();
        let low_band_params = CompressorParams {
            upper_threshold_db: first
                .select(params.low_upper_threshold_db, params.band_upper_threshold_db),
            lower_threshold_db: first
                .select(params.low_lower_threshold_db, params.band_lower_threshold_db),
            upper_ratio: first.select(params.low_upper_ratio, params.band_upper_ratio),
            lower_ratio: first.select(params.low_lower_ratio, params.band_lower_ratio),
            output_gain_db: first.select(params.low_output_gain_db, params.band_output_gain_db),
            attack: params.attack,
            release: params.release,
            mix: params.mix,
        };
        let band_high_params = CompressorParams {
            upper_threshold_db: first
                .select(params.band_upper_threshold_db, params.high_upper_threshold_db),
            lower_threshold_db: first
                .select(params.band_lower_threshold_db, params.high_lower_threshold_db),
            upper_ratio: first.select(params.band_upper_ratio, params.high_upper_ratio),
            lower_ratio: first.select(params.band_lower_ratio, params.high_lower_ratio),
            output_gain_db: first.select(params.band_output_gain_db, params.high_output_gain_db),
            attack: params.attack,
            release: params.release,
            mix: params.mix,
        };

        if low_enabled != self.was_low_enabled || high_enabled != self.was_high_enabled {
            self.low_band_filter.reset(PolyMask::all_on());
            self.band_high_filter.reset(PolyMask::all_on());
            self.low_band_compressor.reset();
            self.band_high_compressor.reset();
            self.was_low_enabled = low_enabled;
            self.was_high_enabled = high_enabled;
        }

        if low_enabled && high_enabled {
            // Split at 120 Hz, pack [low | >120], split the packed signal at
            // 2500 Hz, then compress [low | band] and [band | high].
            self.low_band_filter.process(
                audio_in,
                &mut self.low_buffer[..num_samples],
                &mut self.high_buffer[..num_samples],
            );
            pack_filter_output(
                &self.low_buffer[..num_samples],
                &self.high_buffer[..num_samples],
                &mut self.packed_buffer[..num_samples],
            );
            self.band_high_filter.process(
                &self.packed_buffer[..num_samples],
                &mut self.low_buffer[..num_samples],
                &mut self.high_buffer[..num_samples],
            );
            // Quirk kept from the reference: the sub-2500 residue of the
            // high-packed voice is folded back into the low band's input.
            for i in 0..num_samples {
                self.packed_buffer[i] =
                    self.low_buffer[i] + (self.high_buffer[i] & first_voice_mask());
            }

            self.low_band_compressor.process(
                &low_band_params,
                &self.packed_buffer[..num_samples],
                &mut self.low_band_out[..num_samples],
            );
            self.band_high_compressor.process(
                &band_high_params,
                &self.high_buffer[..num_samples],
                &mut self.band_high_out[..num_samples],
            );

            for (i, out) in audio_out.iter_mut().enumerate() {
                let mut low_band_sample = self.low_band_out[i];
                low_band_sample += low_band_sample.swap_voices();
                let high_sample = self.band_high_out[i].swap_voices();
                *out = low_band_sample + high_sample;
            }
        } else if low_enabled {
            self.low_band_filter.process(
                audio_in,
                &mut self.low_buffer[..num_samples],
                &mut self.high_buffer[..num_samples],
            );
            pack_filter_output(
                &self.low_buffer[..num_samples],
                &self.high_buffer[..num_samples],
                &mut self.packed_buffer[..num_samples],
            );
            self.low_band_compressor.process(
                &low_band_params,
                &self.packed_buffer[..num_samples],
                &mut self.low_band_out[..num_samples],
            );
            write_compressor_outputs(&self.low_band_out[..num_samples], audio_out);
        } else if high_enabled {
            self.band_high_filter.process(
                audio_in,
                &mut self.low_buffer[..num_samples],
                &mut self.high_buffer[..num_samples],
            );
            pack_filter_output(
                &self.low_buffer[..num_samples],
                &self.high_buffer[..num_samples],
                &mut self.packed_buffer[..num_samples],
            );
            self.band_high_compressor.process(
                &band_high_params,
                &self.packed_buffer[..num_samples],
                &mut self.band_high_out[..num_samples],
            );
            write_compressor_outputs(&self.band_high_out[..num_samples], audio_out);
        } else {
            // Single band: the band/high core compresses the full signal
            // (band settings in the first voice, high settings in the second,
            // both voices copied out as-is â€” a quirk kept from the reference).
            self.band_high_compressor.process(
                &band_high_params,
                audio_in,
                &mut self.band_high_out[..num_samples],
            );
            audio_out.copy_from_slice(&self.band_high_out[..num_samples]);
        }

        let low_band_input_ms = self.low_band_compressor.input_mean_squared();
        let band_high_input_ms = self.band_high_compressor.input_mean_squared();
        let low_band_output_ms = self.low_band_compressor.output_mean_squared();
        let band_high_output_ms = self.band_high_compressor.output_mean_squared();

        self.low_input_mean_squared = low_band_input_ms;
        self.low_output_mean_squared = low_band_output_ms;

        if low_enabled {
            self.band_input_mean_squared = low_band_input_ms.swap_voices();
            self.band_output_mean_squared = low_band_output_ms.swap_voices();
        } else {
            self.band_input_mean_squared = band_high_input_ms;
            self.band_output_mean_squared = band_high_output_ms;
        }

        self.high_input_mean_squared = band_high_input_ms.swap_voices();
        self.high_output_mean_squared = band_high_output_ms.swap_voices();
    }
}

/// Packs a crossover split into voice slots: first voice = low output,
/// second voice = high output (C++ `packFilterOutput`).
fn pack_filter_output(low: &[PolyF32], high: &[PolyF32], dest: &mut [PolyF32]) {
    for i in 0..dest.len() {
        let low_sample = low[i];
        let high_sample = high[i].swap_voices();
        dest[i] = first_voice_mask().select(low_sample, high_sample);
    }
}

/// Sums both packed voices of a compressor output into each voice slot
/// (C++ `writeCompressorOutputs`).
fn write_compressor_outputs(compressor_out: &[PolyF32], dest: &mut [PolyF32]) {
    for (out, &sample) in dest.iter_mut().zip(compressor_out) {
        *out = sample + sample.swap_voices();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 128;

    fn run_sine(
        compressor: &mut MultibandCompressor,
        params: &MultibandCompressorParams,
        freq: f32,
        amplitude: f32,
        blocks: usize,
    ) -> (f32, f32) {
        let mut in_sum = 0.0f64;
        let mut out_sum = 0.0f64;
        let mut count = 0usize;
        let mut n = 0usize;
        let mut input = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..blocks {
            for value in input.iter_mut() {
                let phase = 2.0 * core::f32::consts::PI * freq * n as f32 / SAMPLE_RATE;
                *value = PolyF32::splat(phase.sin() * amplitude);
                n += 1;
            }
            compressor.process(params, &input, &mut output);
            if block >= blocks / 2 {
                for (out, inp) in output.iter().zip(&input) {
                    assert!(out.is_finite());
                    in_sum += (inp.lane(0) as f64) * (inp.lane(0) as f64);
                    out_sum += (out.lane(0) as f64) * (out.lane(0) as f64);
                    count += 1;
                }
            }
        }
        (
            (in_sum / count as f64).sqrt() as f32,
            (out_sum / count as f64).sqrt() as f32,
        )
    }

    /// Params that compress nothing: max thresholds, zero ratios.
    fn transparent_params() -> MultibandCompressorParams {
        MultibandCompressorParams {
            low_upper_threshold_db: PolyF32::splat(12.0),
            band_upper_threshold_db: PolyF32::splat(12.0),
            high_upper_threshold_db: PolyF32::splat(12.0),
            ..MultibandCompressorParams::default()
        }
    }

    #[test]
    fn compresses_low_band_and_leaves_high_band() {
        let mut params = transparent_params();
        params.low_upper_threshold_db = PolyF32::splat(-40.0);
        params.low_upper_ratio = PolyF32::ONE;

        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (in_low, out_low) = run_sine(&mut compressor, &params, 60.0, 0.5, 80);
        assert!(out_low < 0.8 * in_low, "low band not compressed: {in_low} -> {out_low}");

        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (in_high, out_high) = run_sine(&mut compressor, &params, 8000.0, 0.5, 80);
        assert!(
            (out_high - in_high).abs() < 0.1 * in_high,
            "high band affected by low settings: {in_high} -> {out_high}"
        );
    }

    #[test]
    fn compresses_band_and_high_bands() {
        let mut params = transparent_params();
        params.band_upper_threshold_db = PolyF32::splat(-40.0);
        params.band_upper_ratio = PolyF32::ONE;
        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (input, output) = run_sine(&mut compressor, &params, 500.0, 0.5, 80);
        assert!(output < 0.8 * input, "mid band not compressed: {input} -> {output}");

        let mut params = transparent_params();
        params.high_upper_threshold_db = PolyF32::splat(-40.0);
        params.high_upper_ratio = PolyF32::ONE;
        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (input, output) = run_sine(&mut compressor, &params, 8000.0, 0.5, 80);
        assert!(output < 0.8 * input, "high band not compressed: {input} -> {output}");
    }

    #[test]
    fn low_band_mode_splits_at_low_crossover_only() {
        // In LowBand mode the second voice ("rest of the spectrum") uses the
        // band controls; compressing only the low controls must leave a mid
        // tone alone but squash a sub tone.
        let mut params = transparent_params();
        params.enabled_bands = BandOptions::LowBand;
        params.low_upper_threshold_db = PolyF32::splat(-40.0);
        params.low_upper_ratio = PolyF32::ONE;

        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (in_low, out_low) = run_sine(&mut compressor, &params, 60.0, 0.5, 80);
        assert!(out_low < 0.8 * in_low, "low not compressed: {in_low} -> {out_low}");

        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (in_mid, out_mid) = run_sine(&mut compressor, &params, 1000.0, 0.5, 80);
        assert!(
            (out_mid - in_mid).abs() < 0.1 * in_mid,
            "rest band affected: {in_mid} -> {out_mid}"
        );
    }

    #[test]
    fn high_band_mode_splits_at_high_crossover_only() {
        let mut params = transparent_params();
        params.enabled_bands = BandOptions::HighBand;
        params.high_upper_threshold_db = PolyF32::splat(-40.0);
        params.high_upper_ratio = PolyF32::ONE;

        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (in_high, out_high) = run_sine(&mut compressor, &params, 8000.0, 0.5, 80);
        assert!(out_high < 0.8 * in_high, "high not compressed: {in_high} -> {out_high}");

        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (in_mid, out_mid) = run_sine(&mut compressor, &params, 500.0, 0.5, 80);
        assert!(
            (out_mid - in_mid).abs() < 0.1 * in_mid,
            "rest band affected: {in_mid} -> {out_mid}"
        );
    }

    #[test]
    fn single_band_mode_compresses_full_range() {
        let mut params = transparent_params();
        params.enabled_bands = BandOptions::SingleBand;
        params.band_upper_threshold_db = PolyF32::splat(-40.0);
        params.band_upper_ratio = PolyF32::ONE;

        for freq in [60.0, 1000.0, 8000.0] {
            let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
            let (input, output) = run_sine(&mut compressor, &params, freq, 0.5, 80);
            assert!(
                output < 0.8 * input,
                "single band left {freq} Hz uncompressed: {input} -> {output}"
            );
        }
    }

    #[test]
    fn meters_track_per_band_levels() {
        let params = transparent_params();
        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        run_sine(&mut compressor, &params, 500.0, 0.5, 40);
        // A 500 Hz tone lives in the mid band.
        let band = compressor.band_input_mean_squared().lane(0);
        let low = compressor.low_input_mean_squared().lane(0);
        let high = compressor.high_input_mean_squared().lane(0);
        assert!(band > 0.0);
        assert!(band > 10.0 * low, "band {band} vs low {low}");
        assert!(band > 10.0 * high, "band {band} vs high {high}");
        assert!(compressor.band_output_mean_squared().lane(0) > 0.0);
    }

    #[test]
    fn transparent_when_ratios_are_zero() {
        let params = transparent_params();
        let mut compressor = MultibandCompressor::new(SAMPLE_RATE);
        let (input, output) = run_sine(&mut compressor, &params, 1000.0, 0.5, 80);
        assert!(
            (output - input).abs() < 0.05 * input,
            "not transparent: {input} -> {output}"
        );
    }
}
