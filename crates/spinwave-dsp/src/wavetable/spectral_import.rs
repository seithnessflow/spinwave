//! Spectral import: turning audio samples and images into playable wavetables.
//!
//! Two entry points:
//! * [`wavetable_from_audio`] slices a recording into frames, either by
//!   pitch-tracked resynthesis (Serum-style: harmonics of the detected
//!   fundamental are measured with a windowed FFT and mapped onto the
//!   frame's harmonic bins, everything non-harmonic is discarded) or by
//!   raw single-period slicing with cubic resampling.
//! * [`wavetable_from_png`] reads an image as a drawn spectrum: X is frame
//!   position, Y is harmonic number (bottom row = fundamental), and pixel
//!   brightness is harmonic amplitude.

use std::f32::consts::{PI, TAU};

use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

use super::wave_frame::{WaveFrame, NUM_REAL_COMPLEX, WAVEFORM_SIZE};
use super::wavetable::{Wavetable, NUM_OSCILLATOR_WAVE_FRAMES};

/// Fundamental used when detection fails and no usable override is given.
pub const DEFAULT_FUNDAMENTAL_HZ: f32 = 220.0;

/// Peak below which the input counts as silence.
const SILENCE_PEAK: f32 = 1e-5;
/// Inputs shorter than this fall back to the quiet sine table.
const MIN_IMPORT_SAMPLES: usize = 64;
/// Amplitude of the fallback table returned for silent/empty input.
const QUIET_SINE_LEVEL: f32 = 0.1;

const PITCH_MIN_HZ: f32 = 30.0;
const PITCH_MAX_HZ: f32 = 2000.0;
/// Normalized autocorrelation below this counts as "no pitch found".
const MIN_PITCH_CONFIDENCE: f32 = 0.5;
/// Window used for the whole-file pitch estimate.
const GLOBAL_PITCH_WINDOW: usize = 8192;
/// Window used for the per-frame pitch refinement.
const LOCAL_PITCH_WINDOW: usize = 4096;

/// How the audio is turned into frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioImportMode {
    /// Pitch-tracked resynthesis: per frame, FFT a window around the frame
    /// position and rebuild one clean fundamental period from the harmonic
    /// bins only. The best mode for pitched material.
    Spectral,
    /// Slice exactly one detected period and resample it to the frame size
    /// with cubic interpolation.
    RawSlice,
}

/// Options for [`wavetable_from_audio`].
#[derive(Clone, Debug)]
pub struct AudioImportOptions {
    /// Frames to generate, clamped to `1..=NUM_OSCILLATOR_WAVE_FRAMES`.
    /// Short files may produce fewer.
    pub num_frames: usize,
    /// Autocorrelation pitch detection; when off, `fundamental_hz` is used
    /// directly.
    pub pitch_detection: bool,
    /// Fundamental used when detection is off or fails.
    pub fundamental_hz: f32,
    pub mode: AudioImportMode,
}

impl Default for AudioImportOptions {
    fn default() -> Self {
        AudioImportOptions {
            num_frames: NUM_OSCILLATOR_WAVE_FRAMES,
            pitch_detection: true,
            fundamental_hz: DEFAULT_FUNDAMENTAL_HZ,
            mode: AudioImportMode::Spectral,
        }
    }
}

/// Phase assignment for spectra built from magnitudes only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralPhase {
    /// All harmonics in cosine phase; bright but spiky.
    Zero,
    /// Schroeder phase spread (`-pi * k^2 / K`), keeps the crest factor low.
    Schroeder,
}

/// Options for [`wavetable_from_png`].
#[derive(Clone, Debug)]
pub struct ImageImportOptions {
    /// Frames to generate (image width is resampled onto this), clamped to
    /// `1..=NUM_OSCILLATOR_WAVE_FRAMES`.
    pub num_frames: usize,
    /// Highest harmonic the image height maps to. Images shorter than this
    /// map one row per harmonic; taller images are band-averaged down.
    pub max_harmonics: usize,
    /// Applied to brightness after invert: `amplitude = brightness^gamma`.
    pub gamma: f32,
    /// Treat dark pixels as loud instead of bright ones.
    pub invert: bool,
    pub phase: SpectralPhase,
}

impl Default for ImageImportOptions {
    fn default() -> Self {
        ImageImportOptions {
            num_frames: NUM_OSCILLATOR_WAVE_FRAMES,
            max_harmonics: 256,
            gamma: 1.0,
            invert: false,
            phase: SpectralPhase::Zero,
        }
    }
}

// ---------------------------------------------------------------------------
// Audio import

struct AudioContext<'a> {
    samples: &'a [f32],
    sample_rate: f32,
    global_f0: f32,
    track_pitch: bool,
    num_frames: usize,
}

/// Builds a wavetable from a mono recording.
///
/// Silent, empty or non-finite input never panics: it yields a one-frame
/// quiet sine table instead.
pub fn wavetable_from_audio(
    samples: &[f32],
    sample_rate: u32,
    options: &AudioImportOptions,
) -> Wavetable {
    let sample_rate = if sample_rate == 0 { 44100.0 } else { sample_rate as f32 };
    let clean: Vec<f32> = samples
        .iter()
        .map(|v| if v.is_finite() { *v } else { 0.0 })
        .collect();
    let peak = clean.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if clean.len() < MIN_IMPORT_SAMPLES || peak < SILENCE_PEAK {
        return quiet_sine_table("Audio Import");
    }

    let fallback = sanitize_hz(options.fundamental_hz, sample_rate);
    let global_f0 = if options.pitch_detection {
        detect_f0_global(&clean, sample_rate).unwrap_or(fallback)
    } else {
        fallback
    };

    // Each frame should consume at least one fundamental period of fresh
    // audio; short files produce fewer frames.
    let period = sample_rate / global_f0;
    let frame_cap = ((clean.len() as f32 / period) as usize).max(1);
    let num_frames = options
        .num_frames
        .clamp(1, NUM_OSCILLATOR_WAVE_FRAMES)
        .min(frame_cap);

    let context = AudioContext {
        samples: &clean,
        sample_rate,
        global_f0,
        track_pitch: options.pitch_detection,
        num_frames,
    };

    let mut wavetable = Wavetable::new(NUM_OSCILLATOR_WAVE_FRAMES);
    wavetable.name = "Audio Import".to_string();
    wavetable.set_num_frames(num_frames);
    match options.mode {
        AudioImportMode::Spectral => fill_spectral_frames(&mut wavetable, &context),
        AudioImportMode::RawSlice => fill_raw_slices(&mut wavetable, &context),
    }
    wavetable.post_process(0.0);
    wavetable
}

/// Sample position of a frame's center, spread evenly across the file.
fn frame_center(context: &AudioContext, index: usize) -> f32 {
    if context.num_frames > 1 {
        index as f32 * (context.samples.len() - 1) as f32 / (context.num_frames - 1) as f32
    } else {
        context.samples.len() as f32 * 0.5
    }
}

/// Per-frame fundamental: local refinement around the global estimate when
/// tracking, otherwise the global/fallback value.
fn frame_fundamental(context: &AudioContext, center: f32) -> f32 {
    if context.track_pitch {
        detect_f0_near(context.samples, context.sample_rate, center, context.global_f0)
            .unwrap_or(context.global_f0)
    } else {
        context.global_f0
    }
}

fn fill_spectral_frames(wavetable: &mut Wavetable, context: &AudioContext) {
    let mut planner = RealFftPlanner::<f32>::new();
    let mut frame = WaveFrame::new();
    for i in 0..context.num_frames {
        let center = frame_center(context, i);
        let f0 = frame_fundamental(context, center);
        frame.clear();
        frame.index = i;
        resynthesize_frame(&mut frame, context, center, f0, &mut planner);
        frame.normalize(true);
        frame.to_frequency_domain();
        wavetable.load_wave_frame_at(&frame, i);
    }
}

/// FFTs a Hann window around `center` and rebuilds one fundamental period:
/// the energy near each `k * f0` becomes harmonic bin `k`, everything else
/// is dropped. Phases are re-referenced so the fundamental sits in cosine
/// phase, keeping frames coherent with each other.
fn resynthesize_frame(
    frame: &mut WaveFrame,
    context: &AudioContext,
    center: f32,
    f0: f32,
    planner: &mut RealFftPlanner<f32>,
) {
    let period = context.sample_rate / f0;
    let window_size = analysis_window_size(period);
    let start = (center - window_size as f32 * 0.5).round() as isize;

    let mut segment = vec![0.0f32; window_size];
    for (j, value) in segment.iter_mut().enumerate() {
        let hann = 0.5 - 0.5 * (TAU * j as f32 / window_size as f32).cos();
        *value = hann * sample_at(context.samples, start + j as isize);
    }

    let r2c = planner.plan_fft_forward(window_size);
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut segment, &mut spectrum).expect("forward FFT");

    let bins_per_harmonic = f0 * window_size as f32 / context.sample_rate;
    let highest_bin = (window_size / 2).saturating_sub(2);
    let max_harmonic =
        (NUM_REAL_COMPLEX - 1).min((highest_bin as f32 / bins_per_harmonic) as usize);

    let mut fundamental_phase = 0.0f32;
    for k in 1..=max_harmonic {
        let target_bin = bins_per_harmonic * k as f32;
        let center_bin = target_bin.round() as usize;

        // Gather the Hann mainlobe: incoherent sum for a magnitude that is
        // flat against fractional bin offsets, sign-corrected coherent sum
        // for the phase (adjacent FFT bins alternate sign under Hann).
        let mut coherent = Complex::new(0.0f32, 0.0);
        let mut energy = 0.0f32;
        let low = center_bin.saturating_sub(1);
        let high = (center_bin + 1).min(spectrum.len() - 1);
        for (offset, bin) in spectrum[low..=high].iter().enumerate() {
            let sign = if (low + offset).is_multiple_of(2) { 1.0 } else { -1.0 };
            coherent += bin * sign;
            energy += bin.norm_sqr();
        }

        let magnitude = energy.sqrt();
        let mut phase = coherent.arg();
        if k == 1 {
            fundamental_phase = phase;
        }
        // Remove the time-reference ramp: harmonic k keeps only its phase
        // relative to the fundamental.
        phase -= k as f32 * fundamental_phase;
        frame.frequency_domain[k] = Complex::from_polar(magnitude, phase);
    }
    frame.to_time_domain();
}

/// Smallest power of two holding several fundamental periods, so harmonics
/// land on well-separated bins.
fn analysis_window_size(period: f32) -> usize {
    let target = period * 6.0;
    let mut size = 2048usize;
    while (size as f32) < target && size < 16384 {
        size *= 2;
    }
    size
}

fn fill_raw_slices(wavetable: &mut Wavetable, context: &AudioContext) {
    let mut frame = WaveFrame::new();
    let mut cycle = vec![0.0f32; WAVEFORM_SIZE];
    for i in 0..context.num_frames {
        let center = frame_center(context, i);
        let f0 = frame_fundamental(context, center);
        let period = context.sample_rate / f0;
        let max_start = (context.samples.len() as f32 - period).max(0.0);
        let start = (center - period * 0.5).clamp(0.0, max_start);
        for (j, value) in cycle.iter_mut().enumerate() {
            let position = start + j as f32 * period / WAVEFORM_SIZE as f32;
            *value = sample_cubic(context.samples, position);
        }
        frame.clear();
        frame.index = i;
        frame.load_time_domain(&cycle);
        frame.remove_dc();
        frame.normalize(true);
        frame.to_frequency_domain();
        wavetable.load_wave_frame_at(&frame, i);
    }
}

// ---------------------------------------------------------------------------
// Pitch detection

fn sanitize_hz(hz: f32, sample_rate: f32) -> f32 {
    let max_hz = (sample_rate * 0.25).max(20.0);
    if hz.is_finite() && hz > 0.0 {
        hz.clamp(10.0, max_hz)
    } else {
        DEFAULT_FUNDAMENTAL_HZ.min(max_hz)
    }
}

/// Whole-file fundamental estimate from a centered analysis window.
fn detect_f0_global(samples: &[f32], sample_rate: f32) -> Option<f32> {
    let length = samples.len().min(GLOBAL_PITCH_WINDOW);
    let start = (samples.len() - length) / 2;
    let window = &samples[start..start + length];
    let min_lag = (sample_rate / PITCH_MAX_HZ) as usize;
    let max_lag = (sample_rate / PITCH_MIN_HZ).ceil() as usize;
    autocorrelation_pitch(window, sample_rate, min_lag, max_lag)
        .filter(|&(_, confidence)| confidence >= MIN_PITCH_CONFIDENCE)
        .map(|(f0, _)| sanitize_hz(f0, sample_rate))
}

/// Local fundamental near `center`, searched in a narrow band around the
/// global estimate so octave jumps between frames are ruled out.
fn detect_f0_near(samples: &[f32], sample_rate: f32, center: f32, global_f0: f32) -> Option<f32> {
    let length = samples.len().min(LOCAL_PITCH_WINDOW);
    let max_start = samples.len() - length;
    let start = ((center - length as f32 * 0.5).max(0.0) as usize).min(max_start);
    let window = &samples[start..start + length];
    let global_lag = sample_rate / global_f0;
    let min_lag = (global_lag * 0.75) as usize;
    let max_lag = (global_lag * 1.34).ceil() as usize;
    autocorrelation_pitch(window, sample_rate, min_lag, max_lag)
        .filter(|&(_, confidence)| confidence >= MIN_PITCH_CONFIDENCE)
        .map(|(f0, _)| sanitize_hz(f0, sample_rate))
}

/// Normalized autocorrelation pitch pick: the smallest lag that is a local
/// maximum within 90% of the best correlation (avoids octave-down errors),
/// refined with parabolic interpolation. Returns `(f0, confidence)`.
fn autocorrelation_pitch(
    window: &[f32],
    sample_rate: f32,
    min_lag: usize,
    max_lag: usize,
) -> Option<(f32, f32)> {
    let n = window.len();
    let min_lag = min_lag.max(2);
    let max_lag = max_lag.min(n / 2);
    if n < 8 || max_lag <= min_lag {
        return None;
    }

    let mean = window.iter().sum::<f32>() / n as f32;
    let x: Vec<f32> = window.iter().map(|v| v - mean).collect();

    // Prefix sums of squares normalize each lag's correlation.
    let mut prefix = vec![0.0f64; n + 1];
    for (i, value) in x.iter().enumerate() {
        prefix[i + 1] = prefix[i] + f64::from(*value) * f64::from(*value);
    }

    let mut corr = vec![0.0f32; max_lag - min_lag + 1];
    for (index, value) in corr.iter_mut().enumerate() {
        let lag = min_lag + index;
        let overlap = n - lag;
        let mut dot = 0.0f64;
        for (a, b) in x[..overlap].iter().zip(&x[lag..]) {
            dot += f64::from(*a) * f64::from(*b);
        }
        let energy = prefix[overlap] * (prefix[n] - prefix[lag]);
        *value = if energy > 1e-12 { (dot / energy.sqrt()) as f32 } else { 0.0 };
    }

    let best = corr.iter().fold(f32::MIN, |m, &v| m.max(v));
    if best < 0.4 {
        return None;
    }
    let threshold = best * 0.9;
    let mut pick = None;
    for i in 1..corr.len() {
        // The zero-lag autocorrelation tail decays through min_lag, so the
        // first index is never a real peak: require a rise into the maximum.
        let left = corr[i - 1];
        let right = if i + 1 < corr.len() { corr[i + 1] } else { f32::MIN };
        if corr[i] >= threshold && corr[i] >= left && corr[i] >= right {
            pick = Some(i);
            break;
        }
    }
    let i = pick?;

    let mut lag = (min_lag + i) as f32;
    if i > 0 && i + 1 < corr.len() {
        let denom = corr[i - 1] - 2.0 * corr[i] + corr[i + 1];
        if denom.abs() > 1e-9 {
            lag += (0.5 * (corr[i - 1] - corr[i + 1]) / denom).clamp(-0.5, 0.5);
        }
    }
    Some((sample_rate / lag, corr[i]))
}

// ---------------------------------------------------------------------------
// Interpolated sample access

#[inline]
fn sample_at(samples: &[f32], index: isize) -> f32 {
    if index < 0 || index as usize >= samples.len() {
        0.0
    } else {
        samples[index as usize]
    }
}

/// Catmull-Rom cubic read at a fractional position; outside the buffer
/// reads as silence.
fn sample_cubic(samples: &[f32], position: f32) -> f32 {
    let base = position.floor();
    let t = position - base;
    let i = base as isize;
    let p0 = sample_at(samples, i - 1);
    let p1 = sample_at(samples, i);
    let p2 = sample_at(samples, i + 1);
    let p3 = sample_at(samples, i + 2);
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = 0.5 * (p2 - p0);
    ((a * t + b) * t + c) * t + p1
}

/// Fallback table for silent or unusable input: one frame of a 0.1-peak
/// sine, so downstream code always has something finite and audible-safe.
fn quiet_sine_table(name: &str) -> Wavetable {
    let mut wavetable = Wavetable::new(NUM_OSCILLATOR_WAVE_FRAMES);
    wavetable.name = name.to_string();
    let mut frame = WaveFrame::new();
    frame.frequency_domain[1] =
        Complex::new(QUIET_SINE_LEVEL * (WAVEFORM_SIZE / 2) as f32, 0.0);
    frame.to_time_domain();
    wavetable.load_wave_frame_at(&frame, 0);
    wavetable.post_process(0.0);
    wavetable
}

// ---------------------------------------------------------------------------
// Image import

/// Builds a wavetable from a PNG interpreted as a drawn spectrum: X maps to
/// frame position, Y to harmonic number (bottom row = harmonic 1), and
/// brightness (luminance, weighted by alpha) to harmonic amplitude.
pub fn wavetable_from_png(
    png_bytes: &[u8],
    options: &ImageImportOptions,
) -> Result<Wavetable, String> {
    let mut decoder = png::Decoder::new(png_bytes);
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("invalid PNG: {e}"))?;
    let mut buffer = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|e| format!("PNG decode failed: {e}"))?;
    let width = info.width as usize;
    let height = info.height as usize;
    if width == 0 || height == 0 {
        return Err("PNG has no pixels".to_string());
    }
    let channels = info.color_type.samples();
    let bytes = &buffer[..info.buffer_size()];
    let gamma = if options.gamma.is_finite() && options.gamma > 0.0 {
        options.gamma
    } else {
        1.0
    };

    let mut luminance = vec![0.0f32; width * height];
    for (pixel, value) in luminance.iter_mut().enumerate() {
        let p = pixel * channels;
        let (luma, alpha) = match info.color_type {
            png::ColorType::Grayscale => (f32::from(bytes[p]) / 255.0, 1.0),
            png::ColorType::GrayscaleAlpha => {
                (f32::from(bytes[p]) / 255.0, f32::from(bytes[p + 1]) / 255.0)
            }
            png::ColorType::Rgb => (rgb_luma(&bytes[p..p + 3]), 1.0),
            png::ColorType::Rgba => {
                (rgb_luma(&bytes[p..p + 3]), f32::from(bytes[p + 3]) / 255.0)
            }
            other => return Err(format!("unsupported PNG color type {other:?}")),
        };
        let bright = if options.invert { 1.0 - luma } else { luma };
        *value = (bright * alpha).clamp(0.0, 1.0).powf(gamma);
    }

    let max_harmonics = options.max_harmonics.clamp(1, NUM_REAL_COMPLEX - 1);
    // Images no taller than the harmonic budget map one row per harmonic
    // exactly; taller images are band-averaged down.
    let num_harmonics = max_harmonics.min(height);
    let num_frames = options.num_frames.clamp(1, NUM_OSCILLATOR_WAVE_FRAMES);

    let mut wavetable = Wavetable::new(NUM_OSCILLATOR_WAVE_FRAMES);
    wavetable.name = "Image Import".to_string();
    wavetable.set_num_frames(num_frames);

    let scale = (WAVEFORM_SIZE / 2) as f32;
    let mut frame = WaveFrame::new();
    for i in 0..num_frames {
        let t = if num_frames > 1 {
            i as f32 / (num_frames - 1) as f32
        } else {
            0.5
        };
        let x = t * (width - 1) as f32;
        let x0 = x.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let frac = x - x0 as f32;
        let column_value = |row: usize| -> f32 {
            let a = luminance[row * width + x0];
            let b = luminance[row * width + x1];
            a + (b - a) * frac
        };

        frame.clear();
        frame.index = i;
        for k in 1..=num_harmonics {
            let amplitude = if num_harmonics == height {
                // One source row per harmonic, bottom row = harmonic 1.
                column_value(height - k)
            } else {
                // Average the band of source rows feeding harmonic k.
                let y_top = height as f32 * (1.0 - k as f32 / num_harmonics as f32);
                let y_bottom = height as f32 * (1.0 - (k - 1) as f32 / num_harmonics as f32);
                let row_first = (y_top.max(0.0) as usize).min(height - 1);
                let row_last = ((y_bottom.ceil() as usize).max(row_first + 1)).min(height);
                let mut sum = 0.0;
                for row in row_first..row_last {
                    sum += column_value(row);
                }
                sum / (row_last - row_first) as f32
            };
            if amplitude <= 0.0 {
                continue;
            }
            let phase = match options.phase {
                SpectralPhase::Zero => 0.0,
                SpectralPhase::Schroeder => -PI * (k * k) as f32 / num_harmonics as f32,
            };
            frame.frequency_domain[k] = Complex::from_polar(amplitude * scale, phase);
        }
        frame.to_time_domain();
        frame.normalize(true);
        frame.to_frequency_domain();
        wavetable.load_wave_frame_at(&frame, i);
    }
    wavetable.post_process(0.0);
    Ok(wavetable)
}

fn rgb_luma(rgb: &[u8]) -> f32 {
    (0.2126 * f32::from(rgb[0]) + 0.7152 * f32::from(rgb[1]) + 0.0722 * f32::from(rgb[2])) / 255.0
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::wavetable::NUM_HARMONICS;
    use super::*;

    fn sine(sample_rate: f32, hz: f32, length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| (TAU * hz * i as f32 / sample_rate).sin())
            .collect()
    }

    fn saw(sample_rate: f32, hz: f32, length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| 2.0 * (hz * i as f32 / sample_rate).fract() - 1.0)
            .collect()
    }

    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    fn pearson(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() as f32;
        let mean_a = a.iter().sum::<f32>() / n;
        let mean_b = b.iter().sum::<f32>() / n;
        let mut num = 0.0f32;
        let mut var_a = 0.0f32;
        let mut var_b = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            let da = x - mean_a;
            let db = y - mean_b;
            num += da * db;
            var_a += da * da;
            var_b += db * db;
        }
        num / (var_a.sqrt() * var_b.sqrt()).max(1e-12)
    }

    fn harmonic_energy_ratio(wavetable: &Wavetable, frame: usize, harmonic: usize) -> f32 {
        let amps = wavetable.data().frequency_amplitudes(frame);
        let total: f32 = (1..NUM_HARMONICS).map(|k| amps[2 * k] * amps[2 * k]).sum();
        if total <= 0.0 {
            return 0.0;
        }
        amps[2 * harmonic] * amps[2 * harmonic] / total
    }

    fn dominant_harmonic(wavetable: &Wavetable, frame: usize) -> usize {
        let amps = wavetable.data().frequency_amplitudes(frame);
        (1..NUM_HARMONICS)
            .max_by(|&a, &b| amps[2 * a].partial_cmp(&amps[2 * b]).unwrap())
            .unwrap()
    }

    fn assert_finite(wavetable: &Wavetable) {
        for frame in 0..wavetable.num_frames() {
            assert!(
                wavetable
                    .data()
                    .wave_data(frame)
                    .iter()
                    .all(|v| v.is_finite()),
                "frame {frame} not finite"
            );
        }
    }

    fn encode_gray_png(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(pixels).unwrap();
        }
        bytes
    }

    #[test]
    fn spectral_sine_is_dominated_by_fundamental() {
        let samples = sine(44100.0, 220.0, 22050);
        for pitch_detection in [true, false] {
            let options = AudioImportOptions {
                num_frames: 8,
                pitch_detection,
                fundamental_hz: 220.0,
                mode: AudioImportMode::Spectral,
            };
            let wavetable = wavetable_from_audio(&samples, 44100, &options);
            assert_eq!(wavetable.num_frames(), 8);
            assert_finite(&wavetable);
            for frame in 0..wavetable.num_frames() {
                let ratio = harmonic_energy_ratio(&wavetable, frame, 1);
                assert!(
                    ratio > 0.9,
                    "pitch_detection={pitch_detection} frame {frame}: ratio {ratio}"
                );
            }
        }
    }

    #[test]
    fn spectral_saw_harmonics_follow_one_over_k() {
        let samples = saw(44100.0, 110.0, 22050);
        let options = AudioImportOptions {
            num_frames: 6,
            pitch_detection: true,
            fundamental_hz: 110.0,
            mode: AudioImportMode::Spectral,
        };
        let wavetable = wavetable_from_audio(&samples, 44100, &options);
        assert_eq!(wavetable.num_frames(), 6);
        assert_finite(&wavetable);

        let amps = wavetable.data().frequency_amplitudes(3);
        let measured: Vec<f32> = (1..=8).map(|k| amps[2 * k]).collect();
        let ideal: Vec<f32> = (1..=8).map(|k| 1.0 / k as f32).collect();
        let corr = pearson(&measured, &ideal);
        assert!(corr > 0.95, "harmonic profile vs 1/k: {corr}, measured {measured:?}");
        assert!(
            measured[0] > measured[3] && measured[3] > measured[7],
            "amplitudes should decrease: {measured:?}"
        );
    }

    #[test]
    fn raw_slice_reproduces_known_period() {
        // 100 Hz at 44.1 kHz: one period is exactly 441 samples.
        let period = 441usize;
        let wave = |t: f32| (TAU * t).sin() + 0.5 * (2.0 * TAU * t).sin();
        let samples: Vec<f32> = (0..period * 20)
            .map(|i| wave((i % period) as f32 / period as f32))
            .collect();
        let options = AudioImportOptions {
            num_frames: 3,
            pitch_detection: true,
            fundamental_hz: 100.0,
            mode: AudioImportMode::RawSlice,
        };
        let wavetable = wavetable_from_audio(&samples, 44100, &options);
        assert_finite(&wavetable);
        let reference: Vec<f32> = (0..WAVEFORM_SIZE)
            .map(|j| wave(j as f32 / WAVEFORM_SIZE as f32))
            .collect();
        let corr = correlation(wavetable.data().wave_data(0), &reference);
        assert!(corr > 0.95, "raw slice vs source period: {corr}");
    }

    #[test]
    fn png_bright_bottom_row_gives_pure_sine() {
        let mut pixels = vec![0u8; 8 * 8];
        pixels[7 * 8..].fill(255);
        let bytes = encode_gray_png(8, 8, &pixels);
        let options = ImageImportOptions {
            num_frames: 4,
            ..Default::default()
        };
        let wavetable = wavetable_from_png(&bytes, &options).unwrap();
        assert_eq!(wavetable.num_frames(), 4);
        assert_finite(&wavetable);
        let reference: Vec<f32> = (0..WAVEFORM_SIZE)
            .map(|j| (TAU * j as f32 / WAVEFORM_SIZE as f32).cos())
            .collect();
        for frame in 0..wavetable.num_frames() {
            let ratio = harmonic_energy_ratio(&wavetable, frame, 1);
            assert!(ratio > 0.99, "frame {frame}: ratio {ratio}");
            let corr = correlation(wavetable.data().wave_data(frame), &reference);
            assert!(corr > 0.99, "frame {frame}: correlation {corr}");
        }
    }

    #[test]
    fn png_columns_map_to_frames() {
        // Column x holds a single bright pixel at harmonic x + 1.
        let (width, height) = (5usize, 8usize);
        let mut pixels = vec![0u8; width * height];
        for x in 0..width {
            pixels[(height - 1 - x) * width + x] = 255;
        }
        let bytes = encode_gray_png(width as u32, height as u32, &pixels);
        let options = ImageImportOptions {
            num_frames: 5,
            ..Default::default()
        };
        let wavetable = wavetable_from_png(&bytes, &options).unwrap();
        assert_eq!(wavetable.num_frames(), 5);
        for frame in 0..wavetable.num_frames() {
            assert_eq!(dominant_harmonic(&wavetable, frame), frame + 1, "frame {frame}");
        }
    }

    #[test]
    fn png_options_invert_gamma_schroeder() {
        // Dark bottom row on a white background: with invert, harmonic 1
        // dominates. Gamma and Schroeder phases must stay well-behaved.
        let mut pixels = vec![255u8; 8 * 8];
        pixels[7 * 8..].fill(0);
        let bytes = encode_gray_png(8, 8, &pixels);
        let options = ImageImportOptions {
            num_frames: 2,
            invert: true,
            gamma: 2.0,
            phase: SpectralPhase::Schroeder,
            ..Default::default()
        };
        let wavetable = wavetable_from_png(&bytes, &options).unwrap();
        assert_finite(&wavetable);
        for frame in 0..wavetable.num_frames() {
            assert!(harmonic_energy_ratio(&wavetable, frame, 1) > 0.9);
        }

        // Tall image exercises the band-averaging path (512 rows -> 256
        // harmonics) without panicking.
        let mut tall = vec![0u8; 512];
        tall[256..].fill(200);
        let bytes = encode_gray_png(1, 512, &tall);
        let options = ImageImportOptions {
            num_frames: 3,
            phase: SpectralPhase::Schroeder,
            ..Default::default()
        };
        let wavetable = wavetable_from_png(&bytes, &options).unwrap();
        assert_finite(&wavetable);
        assert!(dominant_harmonic(&wavetable, 1) <= 128);
    }

    #[test]
    fn png_rejects_invalid_input() {
        let options = ImageImportOptions::default();
        assert!(wavetable_from_png(&[], &options).is_err());
        assert!(wavetable_from_png(&[1, 2, 3, 4], &options).is_err());
        assert!(wavetable_from_png(&[0x89, b'P', b'N', b'G'], &options).is_err());
    }

    #[test]
    fn degenerate_audio_never_panics() {
        for mode in [AudioImportMode::Spectral, AudioImportMode::RawSlice] {
            for samples in [Vec::new(), vec![0.0; 4410], vec![f32::NAN; 1000]] {
                let options = AudioImportOptions {
                    mode,
                    ..Default::default()
                };
                let wavetable = wavetable_from_audio(&samples, 44100, &options);
                assert!(wavetable.num_frames() >= 1);
                assert_finite(&wavetable);
                let peak = wavetable
                    .data()
                    .wave_data(0)
                    .iter()
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                assert!(peak > 0.01, "silence should fall back to a quiet sine");
            }
        }

        // Short files yield fewer frames than requested, all finite.
        let short = sine(44100.0, 220.0, 500);
        let options = AudioImportOptions {
            num_frames: 64,
            ..Default::default()
        };
        let wavetable = wavetable_from_audio(&short, 44100, &options);
        assert!(wavetable.num_frames() >= 1 && wavetable.num_frames() < 64);
        assert_finite(&wavetable);

        // A zero sample rate falls back to 44.1 kHz instead of dividing by 0.
        let wavetable = wavetable_from_audio(&short, 0, &AudioImportOptions::default());
        assert!(wavetable.num_frames() >= 1);
        assert_finite(&wavetable);
    }
}
