//! One wavetable frame: a 2048-sample cycle plus its spectrum.
//!
//! Rework of Vital's `wave_frame.{h,cpp}`. The FFT convention matches the
//! reference: the forward transform is an unnormalized real DFT producing
//! bins `0..=N/2`, and the inverse divides by `N`, so a bin value of `N/2`
//! renders a unit-amplitude cosine and forwardâ†’inverse is the identity.

use std::sync::{Arc, OnceLock};

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use spinwave_poly::{math, PolyF32};

pub const WAVEFORM_BITS: usize = 11;
pub const WAVEFORM_SIZE: usize = 1 << WAVEFORM_BITS;
/// Number of complex bins of the real FFT: `N/2 + 1`.
pub const NUM_REAL_COMPLEX: usize = WAVEFORM_SIZE / 2 + 1;
pub const DEFAULT_FREQUENCY_RATIO: f32 = 1.0;
pub const DEFAULT_WAVE_SAMPLE_RATE: f32 = 44100.0;

/// Shared FFT plans for the 2048-sample frame size.
pub(crate) struct WaveFft {
    pub r2c: Arc<dyn RealToComplex<f32>>,
    pub c2r: Arc<dyn ComplexToReal<f32>>,
}

pub(crate) fn wave_fft() -> &'static WaveFft {
    static FFT: OnceLock<WaveFft> = OnceLock::new();
    FFT.get_or_init(|| {
        let mut planner = RealFftPlanner::<f32>::new();
        WaveFft {
            r2c: planner.plan_fft_forward(WAVEFORM_SIZE),
            c2r: planner.plan_fft_inverse(WAVEFORM_SIZE),
        }
    })
}

/// A single frame of a wavetable, kept in both time and frequency domain.
#[derive(Clone)]
pub struct WaveFrame {
    /// Position of this frame inside its wavetable.
    pub index: usize,
    pub frequency_ratio: f32,
    pub sample_rate: f32,
    /// One cycle of the waveform, `WAVEFORM_SIZE` samples.
    pub time_domain: Vec<f32>,
    /// Spectrum bins `0..=N/2` (unnormalized real-DFT convention).
    pub frequency_domain: Vec<Complex<f32>>,
}

impl Default for WaveFrame {
    fn default() -> Self {
        WaveFrame {
            index: 0,
            frequency_ratio: DEFAULT_FREQUENCY_RATIO,
            sample_rate: DEFAULT_WAVE_SAMPLE_RATE,
            time_domain: vec![0.0; WAVEFORM_SIZE],
            frequency_domain: vec![Complex::new(0.0, 0.0); NUM_REAL_COMPLEX],
        }
    }
}

impl WaveFrame {
    pub fn new() -> Self {
        Self::default()
    }

    /// Largest absolute sample value.
    pub fn max_zero_offset(&self) -> f32 {
        self.time_domain.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    pub fn clear(&mut self) {
        self.frequency_ratio = DEFAULT_FREQUENCY_RATIO;
        self.sample_rate = DEFAULT_WAVE_SAMPLE_RATE;
        self.time_domain.fill(0.0);
        self.frequency_domain.fill(Complex::new(0.0, 0.0));
    }

    pub fn multiply(&mut self, value: f32) {
        for sample in &mut self.time_domain {
            *sample *= value;
        }
        for bin in &mut self.frequency_domain {
            *bin *= value;
        }
    }

    /// Copies a time-domain cycle in and refreshes the spectrum.
    pub fn load_time_domain(&mut self, buffer: &[f32]) {
        self.time_domain.copy_from_slice(&buffer[..WAVEFORM_SIZE]);
        self.to_frequency_domain();
    }

    /// Normalizes peak to 1. With `allow_positive_gain`, quiet frames are
    /// boosted; otherwise only attenuation is applied.
    pub fn normalize(&mut self, allow_positive_gain: bool) {
        const MAX_INVERSE_MULT: f32 = 0.000_000_1;
        let max = self.max_zero_offset();
        let min = if allow_positive_gain { MAX_INVERSE_MULT } else { 1.0 };
        let normalization = 1.0 / min.max(max);
        for sample in &mut self.time_domain {
            *sample *= normalization;
        }
    }

    pub fn add_from(&mut self, source: &WaveFrame) {
        for (dest, src) in self.time_domain.iter_mut().zip(&source.time_domain) {
            *dest += *src;
        }
        for (dest, src) in self.frequency_domain.iter_mut().zip(&source.frequency_domain) {
            *dest += *src;
        }
    }

    pub fn copy_from(&mut self, other: &WaveFrame) {
        self.time_domain.copy_from_slice(&other.time_domain);
        self.frequency_domain.copy_from_slice(&other.frequency_domain);
    }

    /// Recomputes the spectrum from the time-domain cycle.
    pub fn to_frequency_domain(&mut self) {
        let fft = wave_fft();
        let mut input = self.time_domain.clone();
        fft.r2c
            .process(&mut input, &mut self.frequency_domain)
            .expect("forward FFT");
    }

    /// Recomputes the time-domain cycle from the spectrum.
    pub fn to_time_domain(&mut self) {
        let fft = wave_fft();
        let mut input = self.frequency_domain.clone();
        input[0].im = 0.0;
        input[NUM_REAL_COMPLEX - 1].im = 0.0;
        fft.c2r
            .process(&mut input, &mut self.time_domain)
            .expect("inverse FFT");
        let scale = 1.0 / WAVEFORM_SIZE as f32;
        for sample in &mut self.time_domain {
            *sample *= scale;
        }
    }

    /// Removes the DC bin exactly like Vital's `WaveFrame::removedDc`: bin 0
    /// of the spectrum is zeroed and the time domain is shifted by
    /// `frequency_domain[0].im`, which is always 0 for a real cycle, so the
    /// time domain is effectively left alone. The oscillator only reads the
    /// spectrum, hence the DC is gone from what is heard; callers that need
    /// a DC-free time domain must use [`WaveFrame::remove_time_domain_dc`]
    /// or re-run [`WaveFrame::to_time_domain`].
    pub fn remove_dc(&mut self) {
        let offset = self.frequency_domain[0].im;
        self.frequency_domain[0] = Complex::new(0.0, 0.0);
        for sample in &mut self.time_domain {
            *sample -= offset;
        }
    }

    /// Subtracts the mean of the time-domain cycle (the spectrum is not
    /// touched; call [`WaveFrame::to_frequency_domain`] afterwards). Not a
    /// reference operation: used by the Rust-only importers before they
    /// build the spectrum.
    pub fn remove_time_domain_dc(&mut self) {
        let mean = self.time_domain.iter().sum::<f32>() / self.time_domain.len() as f32;
        for sample in &mut self.time_domain {
            *sample -= mean;
        }
    }

    /// Builds one of the classic starting shapes.
    pub fn predefined(shape: WaveShape) -> WaveFrame {
        let mut frame = WaveFrame::new();
        let half = (WAVEFORM_SIZE / 2) as f32;
        match shape {
            WaveShape::Sin => {
                frame.frequency_domain[1] = Complex::new(half, 0.0);
                frame.to_time_domain();
            }
            WaveShape::SaturatedSin => {
                frame.frequency_domain[1] = Complex::new(WAVEFORM_SIZE as f32, 0.0);
                frame.to_time_domain();
                for sample in &mut frame.time_domain {
                    *sample = math::tanh(PolyF32::splat(*sample)).lane(0);
                }
                frame.to_frequency_domain();
            }
            WaveShape::Triangle => {
                let section = WAVEFORM_SIZE / 4;
                for i in 0..section {
                    let t = i as f32 / section as f32;
                    frame.time_domain[i] = 1.0 - t;
                    frame.time_domain[i + section] = -t;
                    frame.time_domain[i + 2 * section] = t - 1.0;
                    frame.time_domain[i + 3 * section] = t;
                }
                frame.to_frequency_domain();
            }
            WaveShape::Square => {
                let section = WAVEFORM_SIZE / 4;
                for i in 0..section {
                    frame.time_domain[i] = 1.0;
                    frame.time_domain[i + section] = -1.0;
                    frame.time_domain[i + 2 * section] = -1.0;
                    frame.time_domain[i + 3 * section] = 1.0;
                }
                frame.to_frequency_domain();
            }
            WaveShape::Pulse => {
                let sections = 4;
                let pulse_size = WAVEFORM_SIZE / sections;
                for i in 0..pulse_size {
                    frame.time_domain[i + (sections - 1) * pulse_size] = 1.0;
                    for s in 0..sections - 1 {
                        frame.time_domain[i + s * pulse_size] = -1.0;
                    }
                }
                frame.to_frequency_domain();
            }
            WaveShape::Saw => {
                let section = WAVEFORM_SIZE / 2;
                let quarter = WAVEFORM_SIZE / 4;
                for i in 0..section {
                    let t = i as f32 / section as f32;
                    frame.time_domain[(i + quarter) % WAVEFORM_SIZE] = t - 1.0;
                    frame.time_domain[(i + section + quarter) % WAVEFORM_SIZE] = t;
                }
                frame.to_frequency_domain();
            }
        }
        frame
    }
}

/// Classic single-cycle shapes (Vital's `PredefinedWaveFrames`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveShape {
    Sin,
    SaturatedSin,
    Triangle,
    Square,
    Pulse,
    Saw,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_frame_is_unit_cosine() {
        let frame = WaveFrame::predefined(WaveShape::Sin);
        // Bin 1 = N/2 renders cos(2*pi*n/N) with unit amplitude.
        assert!((frame.time_domain[0] - 1.0).abs() < 1e-4);
        assert!((frame.time_domain[WAVEFORM_SIZE / 2] + 1.0).abs() < 1e-4);
        assert!(frame.time_domain[WAVEFORM_SIZE / 4].abs() < 1e-3);
        assert!((frame.max_zero_offset() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn pulse_frame_keeps_the_reference_dc() {
        // `PredefinedWaveFrames::createPulse` (wave_frame.cpp) writes -1
        // over the first three quarters and +1 over the last, then calls
        // `toFrequencyDomain` WITHOUT removing the DC. The frame therefore
        // has mean -0.5, bin 0 carries -N/2, and Vital's oscillator plays
        // that offset: `Wavetable::loadFrequencyAmplitudes` starts at
        // harmonic 0. Verified against the golden reference render, whose
        // output holds a rock-steady -0.245 offset for the whole note.
        // Nothing here may zero that bin.
        let frame = WaveFrame::predefined(WaveShape::Pulse);
        let quarter = WAVEFORM_SIZE / 4;
        assert!(frame.time_domain[..3 * quarter].iter().all(|&v| v == -1.0));
        assert!(frame.time_domain[3 * quarter..].iter().all(|&v| v == 1.0));

        let mean = frame.time_domain.iter().sum::<f32>() / WAVEFORM_SIZE as f32;
        assert!((mean + 0.5).abs() < 1e-6, "pulse mean {mean}");
        let dc = frame.frequency_domain[0];
        assert!((dc.re + WAVEFORM_SIZE as f32 / 2.0).abs() < 1e-2, "dc bin {dc}");
        assert!(dc.im.abs() < 1e-3, "dc bin {dc}");
        // The 3/4 duty cycle has no Nyquist content, and every even
        // harmonic beyond the fundamental group survives.
        assert!(frame.frequency_domain[NUM_REAL_COMPLEX - 1].norm() < 1e-2);
        assert!(frame.frequency_domain[1].norm() > 1.0);
    }

    #[test]
    fn remove_dc_zeroes_bin_zero_only() {
        // Pure DC: bin 0 carries everything; after remove_dc the spectrum
        // is silent (the time domain keeps the offset, as in Vital).
        let mut dc = WaveFrame::new();
        dc.time_domain.fill(0.25);
        dc.to_frequency_domain();
        assert!(dc.frequency_domain[0].re.abs() > 1.0);
        dc.remove_dc();
        assert!(dc.frequency_domain.iter().all(|bin| bin.norm() < 1e-6));
        assert!(dc.time_domain.iter().all(|&v| (v - 0.25).abs() < 1e-6));

        // AC-only frame: untouched in both domains.
        let mut ac = WaveFrame::predefined(WaveShape::Sin);
        let spectrum_before = ac.frequency_domain.clone();
        let time_before = ac.time_domain.clone();
        ac.remove_dc();
        assert_eq!(ac.frequency_domain, spectrum_before);
        assert_eq!(ac.time_domain, time_before);

        // The time-domain helper is the importer-side complement.
        let mut offset_sine = WaveFrame::predefined(WaveShape::Sin);
        for value in &mut offset_sine.time_domain {
            *value += 0.5;
        }
        offset_sine.remove_time_domain_dc();
        let mean: f32 =
            offset_sine.time_domain.iter().sum::<f32>() / offset_sine.time_domain.len() as f32;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn fft_roundtrip_is_identity() {
        let mut frame = WaveFrame::predefined(WaveShape::Saw);
        let original = frame.time_domain.clone();
        frame.to_frequency_domain();
        frame.to_time_domain();
        for (a, b) in frame.time_domain.iter().zip(&original) {
            assert!((a - b).abs() < 1e-4, "roundtrip mismatch: {a} vs {b}");
        }
    }
}
