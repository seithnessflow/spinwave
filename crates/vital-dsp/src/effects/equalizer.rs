//! Three-band equalizer (port of Vital's `EqualizerModule`).
//!
//! Three [`DigitalSvf`] stages in series — low (shelf or high pass), band
//! (shelf or notch), high (shelf or low pass) — with six persistent filters
//! so switching modes keeps the inactive filter's state, exactly like the
//! reference's idle processors. The output is also pushed into a
//! [`StereoMemory`] ring for the spectrogram display.

use vital_poly::constants::{MAX_BUFFER_SIZE, MAX_OVERSAMPLE};
use vital_poly::{PolyF32, PolyMask};

use crate::filters::{DigitalSvf, FilterState, FilterStyle};
use crate::memory::StereoMemory;

/// Size of the spectrogram audio memory (C++ `kAudioMemorySamples`).
pub const AUDIO_MEMORY_SAMPLES: usize = 1 << 15;

const MAX_BLOCK: usize = MAX_BUFFER_SIZE * MAX_OVERSAMPLE;

/// Block-rate equalizer parameters.
#[derive(Clone, Copy, Debug)]
pub struct EqualizerParams {
    /// Low band mode: `false` = low shelf, `true` = high pass (C++ `eq_low_mode`).
    pub low_mode: bool,
    /// Band mode: `false` = band shelf, `true` = notch (C++ `eq_band_mode`).
    pub band_mode: bool,
    /// High band mode: `false` = high shelf, `true` = low pass (C++ `eq_high_mode`).
    pub high_mode: bool,
    pub low_cutoff_midi: PolyF32,
    pub band_cutoff_midi: PolyF32,
    pub high_cutoff_midi: PolyF32,
    /// Resonance controls in [0, 1].
    pub low_resonance: PolyF32,
    pub band_resonance: PolyF32,
    pub high_resonance: PolyF32,
    /// Shelf gains in dB (ignored by the pass/notch modes, as in the C++
    /// where those filters have no gain input plugged).
    pub low_gain_db: PolyF32,
    pub band_gain_db: PolyF32,
    pub high_gain_db: PolyF32,
}

impl Default for EqualizerParams {
    fn default() -> EqualizerParams {
        EqualizerParams {
            low_mode: false,
            band_mode: false,
            high_mode: false,
            low_cutoff_midi: PolyF32::splat(40.0),
            band_cutoff_midi: PolyF32::splat(80.0),
            high_cutoff_midi: PolyF32::splat(100.0),
            low_resonance: PolyF32::splat(0.5),
            band_resonance: PolyF32::splat(0.5),
            high_resonance: PolyF32::splat(0.5),
            low_gain_db: PolyF32::ZERO,
            band_gain_db: PolyF32::ZERO,
            high_gain_db: PolyF32::ZERO,
        }
    }
}

pub struct Equalizer {
    high_pass: DigitalSvf,
    low_shelf: DigitalSvf,
    notch: DigitalSvf,
    band_shelf: DigitalSvf,
    low_pass: DigitalSvf,
    high_shelf: DigitalSvf,
    audio_memory: StereoMemory,
    sample_rate: f32,
    buffer_low: Vec<PolyF32>,
    buffer_band: Vec<PolyF32>,
}

fn band_state(
    style: FilterStyle,
    pass_blend: f32,
    midi_cutoff: PolyF32,
    resonance: PolyF32,
    gain_db: PolyF32,
) -> FilterState {
    FilterState {
        midi_cutoff,
        resonance_percent: resonance,
        gain: gain_db,
        style,
        pass_blend: PolyF32::splat(pass_blend),
        ..FilterState::default()
    }
}

impl Equalizer {
    pub fn new(sample_rate: f32) -> Equalizer {
        let mut high_pass = DigitalSvf::new();
        let mut notch = DigitalSvf::new();
        let mut low_pass = DigitalSvf::new();
        // The pass/notch filters are "basic" (no saturation) and skip drive
        // compensation, like the reference.
        high_pass.set_drive_compensation(false);
        high_pass.set_basic(true);
        notch.set_drive_compensation(false);
        notch.set_basic(true);
        low_pass.set_drive_compensation(false);
        low_pass.set_basic(true);

        Equalizer {
            high_pass,
            low_shelf: DigitalSvf::new(),
            notch,
            band_shelf: DigitalSvf::new(),
            low_pass,
            high_shelf: DigitalSvf::new(),
            audio_memory: StereoMemory::new(AUDIO_MEMORY_SAMPLES),
            sample_rate,
            buffer_low: vec![PolyF32::ZERO; MAX_BLOCK],
            buffer_band: vec![PolyF32::ZERO; MAX_BLOCK],
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Audio memory of the (post-EQ) output, for the spectrogram display.
    pub fn audio_memory(&self) -> &StereoMemory {
        &self.audio_memory
    }

    pub fn hard_reset(&mut self) {
        let mask = PolyMask::all_on();
        self.high_pass.reset(mask);
        self.low_shelf.reset(mask);
        self.band_shelf.reset(mask);
        self.notch.reset(mask);
        self.low_pass.reset(mask);
        self.high_shelf.reset(mask);
    }

    pub fn process(&mut self, params: &EqualizerParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let num_samples = audio_in.len();
        assert_eq!(num_samples, audio_out.len());
        assert!(num_samples <= MAX_BLOCK);
        if num_samples == 0 {
            return;
        }
        let sample_rate = self.sample_rate;

        // Low stage: audio_in -> buffer_low.
        {
            let (filter, state) = if params.low_mode {
                let state = band_state(
                    FilterStyle::TwelveDb,
                    2.0,
                    params.low_cutoff_midi,
                    params.low_resonance,
                    PolyF32::ZERO,
                );
                (&mut self.high_pass, state)
            } else {
                let state = band_state(
                    FilterStyle::Shelving,
                    0.0,
                    params.low_cutoff_midi,
                    params.low_resonance,
                    params.low_gain_db,
                );
                (&mut self.low_shelf, state)
            };
            filter.setup(&state, sample_rate);
            filter.process(audio_in, &mut self.buffer_low[..num_samples]);
        }

        // Band stage: buffer_low -> buffer_band.
        {
            let (filter, state) = if params.band_mode {
                let state = band_state(
                    FilterStyle::NotchPassSwap,
                    1.0,
                    params.band_cutoff_midi,
                    params.band_resonance,
                    PolyF32::ZERO,
                );
                (&mut self.notch, state)
            } else {
                let state = band_state(
                    FilterStyle::Shelving,
                    1.0,
                    params.band_cutoff_midi,
                    params.band_resonance,
                    params.band_gain_db,
                );
                (&mut self.band_shelf, state)
            };
            filter.setup(&state, sample_rate);
            filter.process(&self.buffer_low[..num_samples], &mut self.buffer_band[..num_samples]);
        }

        // High stage: buffer_band -> audio_out.
        {
            let (filter, state) = if params.high_mode {
                let state = band_state(
                    FilterStyle::TwelveDb,
                    0.0,
                    params.high_cutoff_midi,
                    params.high_resonance,
                    PolyF32::ZERO,
                );
                (&mut self.low_pass, state)
            } else {
                let state = band_state(
                    FilterStyle::Shelving,
                    2.0,
                    params.high_cutoff_midi,
                    params.high_resonance,
                    params.high_gain_db,
                );
                (&mut self.high_shelf, state)
            };
            filter.setup(&state, sample_rate);
            filter.process(&self.buffer_band[..num_samples], audio_out);
        }

        for &sample in audio_out.iter() {
            self.audio_memory.push(sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::filter_state::frequency_to_midi_note_precise;

    const SAMPLE_RATE: f32 = 48000.0;
    const BLOCK: usize = 128;

    fn midi_for(freq: f32) -> PolyF32 {
        frequency_to_midi_note_precise(PolyF32::splat(freq))
    }

    fn run_sine(eq: &mut Equalizer, params: &EqualizerParams, freq: f32, blocks: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut n = 0usize;
        let mut input = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..blocks {
            for value in input.iter_mut() {
                let phase = 2.0 * core::f32::consts::PI * freq * n as f32 / SAMPLE_RATE;
                *value = PolyF32::splat(phase.sin() * 0.5);
                n += 1;
            }
            eq.process(params, &input, &mut output);
            if block >= blocks / 2 {
                for value in &output {
                    assert!(value.is_finite());
                    sum += value.lane(0) * value.lane(0);
                    count += 1;
                }
            }
        }
        (sum / count as f32).sqrt()
    }

    fn shelf_params() -> EqualizerParams {
        EqualizerParams {
            low_cutoff_midi: midi_for(200.0),
            band_cutoff_midi: midi_for(1000.0),
            high_cutoff_midi: midi_for(5000.0),
            ..EqualizerParams::default()
        }
    }

    #[test]
    fn transparent_at_neutral_gains() {
        let params = shelf_params();
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let input_rms = 0.5 / core::f32::consts::SQRT_2;
        for freq in [100.0, 1000.0, 8000.0] {
            let rms = run_sine(&mut eq, &params, freq, 30);
            assert!(
                (rms - input_rms).abs() < 0.02 * input_rms,
                "freq {freq}: rms {rms} vs {input_rms}"
            );
        }
    }

    #[test]
    fn band_shelf_boosts_and_cuts_at_center() {
        let neutral = shelf_params();
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let reference = run_sine(&mut eq, &neutral, 1000.0, 30);

        let mut boosted = neutral;
        boosted.band_gain_db = PolyF32::splat(12.0);
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let boost = run_sine(&mut eq, &boosted, 1000.0, 30);
        assert!(boost > 1.5 * reference, "boost {boost} vs {reference}");

        let mut cut = neutral;
        cut.band_gain_db = PolyF32::splat(-12.0);
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let cut_rms = run_sine(&mut eq, &cut, 1000.0, 30);
        assert!(cut_rms < 0.7 * reference, "cut {cut_rms} vs {reference}");
    }

    #[test]
    fn low_and_high_shelves_boost_their_bands() {
        let neutral = shelf_params();
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let low_reference = run_sine(&mut eq, &neutral, 100.0, 30);
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let high_reference = run_sine(&mut eq, &neutral, 10000.0, 30);

        let mut low_boost = neutral;
        low_boost.low_gain_db = PolyF32::splat(12.0);
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let low = run_sine(&mut eq, &low_boost, 100.0, 30);
        assert!(low > 1.5 * low_reference, "low shelf {low} vs {low_reference}");

        let mut high_boost = neutral;
        high_boost.high_gain_db = PolyF32::splat(12.0);
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let high = run_sine(&mut eq, &high_boost, 10000.0, 30);
        assert!(high > 1.5 * high_reference, "high shelf {high} vs {high_reference}");
    }

    #[test]
    fn pass_modes_attenuate_out_of_band() {
        let mut params = shelf_params();
        params.low_mode = true; // high pass at 200 Hz
        params.high_mode = true; // low pass at 5 kHz
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let sub = run_sine(&mut eq, &params, 40.0, 30);
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let mid = run_sine(&mut eq, &params, 1000.0, 30);
        let mut eq = Equalizer::new(SAMPLE_RATE);
        let air = run_sine(&mut eq, &params, 15000.0, 30);
        assert!(mid > 3.0 * sub, "high pass failed: mid {mid} vs sub {sub}");
        assert!(mid > 3.0 * air, "low pass failed: mid {mid} vs air {air}");
    }

    #[test]
    fn audio_memory_records_output() {
        let params = shelf_params();
        let mut eq = Equalizer::new(SAMPLE_RATE);
        run_sine(&mut eq, &params, 500.0, 4);
        let mut samples = [0.0f32; 64];
        eq.audio_memory().read_samples(&mut samples, 0, 0);
        assert!(samples.iter().any(|s| s.abs() > 1e-4), "memory stayed silent");
    }
}
