//! Spectral morphing: rebuilds a playable wave frame from a wavetable
//! frame's spectrum with a frequency-domain transformation applied.
//!
//! Rework of Vital's `spectral_morph.h`. Each morph writes interleaved
//! `(re, im)` bins into a spectrum scratch buffer, band-limited to
//! `last_harmonic` (the mip level for the current pitch); the caller then
//! inverse-FFTs the spectrum into a wrapped time-domain frame.

use realfft::num_complex::Complex;
use realfft::ComplexToReal;
use spinwave_poly::{math, utils, PolyF32, PolyMask, PolyU32, LANES};

use crate::wavetable::wavetable::{
    WavetableData, FREQUENCY_BINS, NUM_HARMONICS, NUM_OSCILLATOR_WAVE_FRAMES,
};
use crate::wavetable::WAVEFORM_SIZE;

use super::phase::set_power_distortion_values;

pub const MAX_FORMANT_SHIFT: f32 = 1.0;
pub const MAX_EVEN_ODD_FORMANT_SHIFT: f32 = 2.0;
pub const MAX_HARMONIC_SCALE: f32 = 4.0;
pub const MAX_INHARMONIC_SCALE: f32 = 12.0;
pub const RANDOM_AMPLITUDE_STAGES: usize = 16;
pub const PHASE_DISPERSE_SCALE: f32 = 0.05;
pub const SKEW_SCALE: f32 = 16.0;

/// Guard samples on each side of a rendered wave frame so 4-tap reads at
/// any phase stay in bounds.
pub(crate) const FRAME_GUARD: usize = LANES;
/// Full length of a rendered wave frame including guards.
pub(crate) const FRAME_LEN: usize = WAVEFORM_SIZE + 2 * FRAME_GUARD;
/// Length of the interleaved spectrum scratch (matches the padded
/// per-frame arrays of [`WavetableData`]).
pub(crate) const SPECTRUM_LEN: usize = crate::wavetable::POLY_FREQUENCY_FLOATS;

/// Spectral morph modes (Vital's `SpectralMorph`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpectralMorph {
    #[default]
    None,
    Vocode,
    FormScale,
    HarmonicScale,
    InharmonicScale,
    Smear,
    RandomAmplitudes,
    LowPass,
    HighPass,
    PhaseDisperse,
    ShepardTone,
    Skew,
}

/// Maps normalized morph amounts to the per-mode working values.
pub fn shape_spectral_morph_values(morph: SpectralMorph, values: &mut [PolyF32], spread: bool) {
    match morph {
        SpectralMorph::Vocode => set_power_distortion_values(values, -MAX_FORMANT_SHIFT, spread),
        SpectralMorph::FormScale => {
            set_power_distortion_values(values, -MAX_EVEN_ODD_FORMANT_SHIFT, spread)
        }
        SpectralMorph::HarmonicScale => {
            set_power_distortion_values(values, MAX_HARMONIC_SCALE, spread)
        }
        SpectralMorph::InharmonicScale => {
            set_power_distortion_values(values, MAX_INHARMONIC_SCALE, spread)
        }
        SpectralMorph::Smear => {
            for value in values.iter_mut() {
                let invert = PolyF32::ONE - *value;
                *value = PolyF32::ONE - invert * invert * invert;
            }
        }
        SpectralMorph::RandomAmplitudes => {
            for value in values.iter_mut() {
                *value *= RANDOM_AMPLITUDE_STAGES as f32 - 1.0;
            }
        }
        SpectralMorph::PhaseDisperse => {
            for value in values.iter_mut() {
                *value = -(*value * 2.0 - 1.0) * PHASE_DISPERSE_SCALE;
            }
        }
        SpectralMorph::Skew => {
            for value in values.iter_mut() {
                *value = *value * *value * SKEW_SCALE;
            }
        }
        SpectralMorph::ShepardTone => {
            for value in values.iter_mut() {
                *value = PolyF32::ONE - *value;
            }
        }
        _ => {}
    }
}

#[inline(always)]
fn quad(buffer: &[f32], i: usize) -> PolyF32 {
    PolyF32::from_lanes([buffer[4 * i], buffer[4 * i + 1], buffer[4 * i + 2], buffer[4 * i + 3]])
}

#[inline(always)]
fn set_quad(buffer: &mut [f32], i: usize, value: PolyF32) {
    let lanes = value.to_lanes();
    buffer[4 * i] = lanes[0];
    buffer[4 * i + 1] = lanes[1];
    buffer[4 * i + 2] = lanes[2];
    buffer[4 * i + 3] = lanes[3];
}

#[inline(always)]
fn left_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, 0, u32::MAX, 0]))
}

fn scalar_sin1(phase: f32) -> f32 {
    math::sin1(PolyF32::splat(phase)).lane(0)
}

/// Renders the morphed spectrum for one wavetable frame into `spectrum`
/// (interleaved `(re, im)` floats, length [`SPECTRUM_LEN`]).
pub fn run_spectral_morph(
    morph: SpectralMorph,
    data: &WavetableData,
    frame_index: usize,
    shift: f32,
    last_harmonic: usize,
    random_buffer: &[f32],
    spectrum: &mut [f32],
) {
    spectrum.fill(0.0);
    let frame = data.clamp_frame(frame_index);
    let last_harmonic = last_harmonic.min(WAVEFORM_SIZE / 2);
    match morph {
        SpectralMorph::Vocode | SpectralMorph::FormScale => {
            even_odd_vocode_morph(data, frame, shift, last_harmonic, spectrum)
        }
        SpectralMorph::HarmonicScale => {
            harmonic_scale_morph(data, frame, shift, last_harmonic, spectrum)
        }
        SpectralMorph::InharmonicScale => {
            inharmonic_scale_morph(data, frame, shift, last_harmonic, spectrum)
        }
        SpectralMorph::Smear => smear_morph(data, frame, shift, last_harmonic, spectrum),
        SpectralMorph::RandomAmplitudes => {
            random_amplitude_morph(data, frame, shift, last_harmonic, random_buffer, spectrum)
        }
        SpectralMorph::LowPass => low_pass_morph(data, frame, shift, last_harmonic, spectrum),
        SpectralMorph::HighPass => high_pass_morph(data, frame, shift, last_harmonic, spectrum),
        SpectralMorph::PhaseDisperse => phase_morph(data, frame, shift, last_harmonic, spectrum),
        SpectralMorph::ShepardTone => shepard_morph(data, frame, shift, last_harmonic, spectrum),
        SpectralMorph::Skew => skew_morph(data, frame, shift, last_harmonic, spectrum),
        SpectralMorph::None => passthrough_morph(data, frame, last_harmonic, spectrum),
    }
}

/// Inverse-FFTs an interleaved spectrum into a wrapped wave frame:
/// `frame[FRAME_GUARD..FRAME_GUARD + N]` holds the cycle and the guards
/// repeat its edges so 4-tap interpolation never leaves the buffer.
pub(crate) fn spectrum_to_frame(
    spectrum: &[f32],
    c2r: &dyn ComplexToReal<f32>,
    c2r_input: &mut [Complex<f32>],
    c2r_scratch: &mut [Complex<f32>],
    frame: &mut [f32],
) {
    for (i, bin) in c2r_input.iter_mut().enumerate() {
        *bin = Complex::new(spectrum[2 * i], spectrum[2 * i + 1]);
    }
    c2r_input[0].im = 0.0;
    c2r_input[NUM_HARMONICS - 1].im = 0.0;

    let wave = &mut frame[FRAME_GUARD..FRAME_GUARD + WAVEFORM_SIZE];
    c2r.process_with_scratch(c2r_input, wave, c2r_scratch)
        .expect("inverse FFT");
    let scale = 1.0 / WAVEFORM_SIZE as f32;
    for sample in wave.iter_mut() {
        *sample *= scale;
    }

    for i in 0..FRAME_GUARD {
        frame[i] = frame[WAVEFORM_SIZE + i];
        frame[WAVEFORM_SIZE + FRAME_GUARD + i] = frame[FRAME_GUARD + i];
    }
}

fn passthrough_morph(data: &WavetableData, frame: usize, last_harmonic: usize, spectrum: &mut [f32]) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);
    let last_index = 2 * last_harmonic / LANES;
    for i in 0..=last_index {
        set_quad(spectrum, i, quad(amplitudes, i) * quad(normalized, i));
    }
}

fn shepard_morph(
    data: &WavetableData,
    frame: usize,
    shift: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    const MIN_AMPLITUDE_RATIO: f32 = 2.0;
    const MIN_AMPLITUDE_ADD: f32 = 0.001;

    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);
    let phases = data.phases(frame);

    // Odd harmonics keep the regular spectrum, faded out as the tone slides.
    let regular_amount = 1.0 - shift;
    let last_index = 2 * last_harmonic / LANES;
    let second_mask = PolyMask::from_u32(PolyU32::from_lanes([0, 0, u32::MAX, u32::MAX]));
    for i in 0..=last_index {
        let value = quad(amplitudes, i) * quad(normalized, i) * regular_amount;
        set_quad(spectrum, i, value & second_mask);
    }

    // Even harmonics blend toward the octave-down copy of the spectrum.
    let mut i = 0;
    while i <= last_harmonic {
        let real_index = 2 * i;
        let imag_index = real_index + 1;

        let fundamental_amplitude = amplitudes[real_index];
        let shepard_amplitude = amplitudes[i];
        let amplitude = fundamental_amplitude + (shepard_amplitude - fundamental_amplitude) * shift;

        let ratio = (fundamental_amplitude + MIN_AMPLITUDE_ADD) / (shepard_amplitude + MIN_AMPLITUDE_ADD);
        let (real, imag) = if ratio < MIN_AMPLITUDE_RATIO && ratio > 1.0 / MIN_AMPLITUDE_RATIO {
            let fundamental_phase = phases[real_index] * (0.5 / std::f32::consts::PI);
            let shepard_phase = phases[i] * (0.5 / std::f32::consts::PI);
            let mut delta_phase = shepard_phase - fundamental_phase;
            let mut wraps = delta_phase as i32;
            wraps = (wraps + 1) / 2;
            delta_phase -= 2.0 * wraps as f32;

            let phase = fundamental_phase + delta_phase * shift;
            let real = scalar_sin1((phase + 0.75).rem_euclid(1.0));
            let imag = scalar_sin1((phase + 0.5).rem_euclid(1.0));
            (real, imag)
        } else {
            let fundamental_real = normalized[real_index];
            let real = (normalized[i] - fundamental_real) * shift + fundamental_real;
            let fundamental_imag = normalized[real_index + 1];
            let imag = (normalized[i + 1] - fundamental_imag) * shift + fundamental_imag;
            (real, imag)
        };

        spectrum[real_index] = amplitude * real;
        spectrum[imag_index] = amplitude * imag;
        i += 2;
    }
}

fn skew_morph(
    data: &WavetableData,
    frame: usize,
    shift: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    let num_frames = data.num_frames();
    if num_frames <= 1 {
        passthrough_morph(data, frame, last_harmonic, spectrum);
        return;
    }

    let dc_amplitude = data.frequency_amplitudes(frame)[0];
    spectrum[0] = dc_amplitude * data.normalized_frequencies(frame)[0];
    spectrum[1] = dc_amplitude * data.normalized_frequencies(frame)[1];

    let max_frame = (NUM_OSCILLATOR_WAVE_FRAMES - 1) as f32;
    let base_wavetable_t = frame as f32 / max_frame;
    for i in 1..=last_harmonic {
        let shift_scale = (i as f32).log2() / FREQUENCY_BINS as f32;
        let base_value =
            1.0 - ((base_wavetable_t + shift * shift_scale) * 0.5).rem_euclid(1.0) * 2.0;
        let shifted_index = (1.0 - base_value.abs()) * max_frame;
        let from_index = (shifted_index as usize).min(num_frames - 2);
        let t = (shifted_index - from_index as f32).min(1.0);
        let to_index = from_index + 1;

        let real_index = 2 * i;
        let imaginary_index = real_index + 1;
        let from_amplitudes = data.frequency_amplitudes(from_index);
        let to_amplitudes = data.frequency_amplitudes(to_index);
        let amplitude =
            from_amplitudes[real_index] + t * (to_amplitudes[real_index] - from_amplitudes[real_index]);

        let from_normalized = data.normalized_frequencies(from_index);
        let to_normalized = data.normalized_frequencies(to_index);
        let real =
            from_normalized[real_index] + t * (to_normalized[real_index] - from_normalized[real_index]);
        let imag = from_normalized[imaginary_index]
            + t * (to_normalized[imaginary_index] - from_normalized[imaginary_index]);

        spectrum[real_index] = amplitude * real;
        spectrum[imaginary_index] = amplitude * imag;
    }
}

fn phase_morph(
    data: &WavetableData,
    frame: usize,
    phase_shift: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    const CENTER_MORPH: f32 = 24.0;

    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let last_index = 2 * last_harmonic / LANES;
    let offset = -(CENTER_MORPH - 1.0) * (CENTER_MORPH - 1.0) * phase_shift;
    let value_offset = PolyF32::from_lanes([0.0, 0.0, 1.0, 1.0]);
    let phase_offset = PolyF32::from_lanes([0.25, 0.0, 0.25, 0.0]);
    let scale = 0.5 / std::f32::consts::PI;
    for i in 0..=last_index {
        let amplitude = quad(amplitudes, i);
        let norm = quad(normalized, i);
        let index = value_offset + 2.0 * i as f32;

        let delta_center = (index - CENTER_MORPH) * (index - CENTER_MORPH) * phase_shift + offset;
        let phase = (delta_center * scale + phase_offset).fract();
        let shift = math::sin1(phase);

        let match_mult = norm * shift;
        let switch_mult = norm.swap_stereo() * shift;
        let real = match_mult - match_mult.swap_stereo();
        let imag = switch_mult + switch_mult.swap_stereo();

        set_quad(spectrum, i, amplitude * left_mask().select(real, imag));
    }
}

fn smear_morph(
    data: &WavetableData,
    frame: usize,
    smear: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let last_index = 2 * last_harmonic / LANES;
    let mut amplitude = quad(amplitudes, 0) * (1.0 - smear);
    set_quad(spectrum, 0, amplitude * quad(normalized, 0));

    for i in 1..=last_index {
        let original_amplitude = quad(amplitudes, i);
        amplitude = utils::interpolate(original_amplitude, amplitude, PolyF32::splat(smear));
        set_quad(spectrum, i, amplitude * quad(normalized, i));
        amplitude *= (i as f32 + 0.25) / i as f32;
    }
}

fn low_pass_morph(
    data: &WavetableData,
    frame: usize,
    cutoff_t: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let cutoff = ((FREQUENCY_BINS - 1) as f32 * cutoff_t).exp2() + 1.0;
    let mut last_index = (2 * last_harmonic / LANES) as i32;
    let poly_cutoff = (last_index as f32 + 1.0).min(2.0 * cutoff / LANES as f32);
    last_index = last_index.min(poly_cutoff as i32);
    let t = LANES as f32 * (poly_cutoff - last_index as f32) / 2.0;

    for i in 0..=last_index as usize {
        set_quad(spectrum, i, quad(amplitudes, i) * quad(normalized, i));
    }

    let last_mult = if t >= 1.0 {
        PolyF32::from_lanes([1.0, 1.0, t - 1.0, t - 1.0])
    } else {
        PolyF32::from_lanes([t, t, 0.0, 0.0])
    };
    let li = last_index as usize;
    set_quad(spectrum, li, quad(spectrum, li) * last_mult);
}

fn high_pass_morph(
    data: &WavetableData,
    frame: usize,
    cutoff_t: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let mut cutoff = ((FREQUENCY_BINS - 1) as f32 * cutoff_t).exp2();
    cutoff *= (NUM_HARMONICS as f32 + 1.0) / NUM_HARMONICS as f32;
    let last_index = (2 * last_harmonic / LANES) as i32;
    let poly_cutoff = (last_index as f32 + 1.0).min(2.0 * cutoff / LANES as f32);
    let start_index = poly_cutoff as i32;
    let t = LANES as f32 * (poly_cutoff - start_index as f32) / 2.0;

    if start_index <= last_index {
        for i in start_index as usize..=last_index as usize {
            set_quad(spectrum, i, quad(amplitudes, i) * quad(normalized, i));
        }
    }

    let last_mult = if t >= 1.0 {
        PolyF32::from_lanes([0.0, 0.0, 2.0 - t, 2.0 - t])
    } else {
        PolyF32::from_lanes([1.0 - t, 1.0 - t, 1.0, 1.0])
    };
    let si = start_index as usize;
    set_quad(spectrum, si, quad(spectrum, si) * last_mult);
}

fn even_odd_vocode_morph(
    data: &WavetableData,
    frame: usize,
    shift: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let shifted_limit = WAVEFORM_SIZE as f32 / (2.0 * shift);
    let last_index = last_harmonic.min(shifted_limit as usize);

    let dc_amplitude = amplitudes[0];
    spectrum[0] = dc_amplitude * normalized[0];
    spectrum[1] = dc_amplitude * normalized[1];

    for i in 1..=last_index {
        let shifted_index = (i as f32 * shift).max(1.0);
        let mut index_start = shifted_index as i32;
        index_start -= (i as i32 + index_start) % 2;
        let index_start = (index_start.max(0) as usize).min(NUM_HARMONICS - 1);

        let t = (shifted_index - index_start as f32) * 0.5;
        let real_index1 = 2 * index_start;
        let real_index2 = real_index1 + 4;
        let amplitude_from = amplitudes[real_index1];
        let amplitude_to = amplitudes[real_index2];
        let real_from = amplitude_from * normalized[real_index1];
        let real_to = amplitude_to * normalized[real_index2];
        let imag_from = amplitude_from * normalized[real_index1 + 1];
        let imag_to = amplitude_to * normalized[real_index2 + 1];

        let real_index = 2 * i;
        spectrum[real_index] = shift * (real_from + t * (real_to - real_from));
        spectrum[real_index + 1] = shift * (imag_from + t * (imag_to - imag_from));
    }
}

fn harmonic_scale_morph(
    data: &WavetableData,
    frame: usize,
    shift: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let harmonics = NUM_HARMONICS.min(((last_harmonic as f32 - 1.0) / shift + 1.0) as usize);

    let dc_amplitude = amplitudes[0];
    spectrum[0] = dc_amplitude * normalized[0];
    spectrum[1] = dc_amplitude * normalized[1];

    for i in 1..=harmonics {
        let shifted_index = ((i as f32 - 1.0) * shift + 1.0).max(1.0);
        let dest_index = (shifted_index as usize).min(NUM_HARMONICS);

        let t = shifted_index - dest_index as f32;
        let real_amount = normalized[2 * i];
        let imag_amount = normalized[2 * i + 1];
        let amplitude = amplitudes[2 * i];
        let amplitude1 = (1.0 - t) * amplitude;
        let amplitude2 = t * amplitude;

        let real_index1 = 2 * dest_index;
        spectrum[real_index1] += amplitude1 * real_amount;
        spectrum[real_index1 + 1] += amplitude1 * imag_amount;
        spectrum[real_index1 + 2] += amplitude2 * real_amount;
        spectrum[real_index1 + 3] += amplitude2 * imag_amount;
    }
}

fn inharmonic_scale_morph(
    data: &WavetableData,
    frame: usize,
    mult: f32,
    last_harmonic: usize,
    spectrum: &mut [f32],
) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let dc_amplitude = amplitudes[0];
    spectrum[0] = dc_amplitude * normalized[0];
    spectrum[1] = dc_amplitude * normalized[1];

    for i in 1..=NUM_HARMONICS {
        let octave = (i as f32).log2();
        let power = octave / (FREQUENCY_BINS as f32 - 1.0);
        let shift = mult.powf(power);
        let shifted_index = (shift * (i as f32 - 1.0) + 1.0).max(1.0);
        let dest_index = shifted_index as usize;
        // Bins above Nyquist would only be audible in the reference's
        // kissfft build; every other backend drops them, and so do we.
        if dest_index > 2 * last_harmonic || dest_index >= NUM_HARMONICS {
            break;
        }

        let t = shifted_index - dest_index as f32;
        let amplitude = amplitudes[2 * i];
        let real = normalized[2 * i];
        let imag = normalized[2 * i + 1];

        let real_index = 2 * dest_index;
        let value1 = (1.0 - t) * amplitude;
        spectrum[real_index] += value1 * real;
        spectrum[real_index + 1] += value1 * imag;
        let value2 = t * amplitude;
        spectrum[real_index + 2] += value2 * real;
        spectrum[real_index + 3] += value2 * imag;
    }
}

fn random_amplitude_morph(
    data: &WavetableData,
    frame: usize,
    shift: f32,
    last_harmonic: usize,
    random_buffer: &[f32],
    spectrum: &mut [f32],
) {
    let amplitudes = data.frequency_amplitudes(frame);
    let normalized = data.normalized_frequencies(frame);

    let last_index = 2 * last_harmonic / LANES;
    let index = (shift as usize).min(RANDOM_AMPLITUDE_STAGES - 2);
    let t = shift - index as f32;
    let scale = PolyF32::splat(shift);
    let center = PolyF32::ONE - scale;
    let mult = PolyF32::splat(1.0 + shift);

    let stage_quads = NUM_HARMONICS / LANES;
    let buffer1 = &random_buffer[index * stage_quads * LANES..];
    let buffer2 = &random_buffer[(index + 1) * stage_quads * LANES..];

    for i in 0..=last_index {
        let mut random_value1 = quad(buffer1, i) & left_mask();
        random_value1 = random_value1 + random_value1.swap_stereo();
        let mut random_value2 = quad(buffer2, i) & left_mask();
        random_value2 = random_value2 + random_value2.swap_stereo();
        let random1 = mult * (center - scale * random_value1).max(PolyF32::ZERO);
        let random2 = mult * (center - scale * random_value2).max(PolyF32::ZERO);
        let interpolated = utils::interpolate(random1, random2, PolyF32::splat(t));
        let amplitude = (interpolated * quad(amplitudes, i)).min(PolyF32::splat(1024.0));

        set_quad(spectrum, i, amplitude * quad(normalized, i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavetable::wave_frame::wave_fft;
    use crate::wavetable::{WaveFrame, WaveShape, Wavetable};

    fn morph_to_wave(morph: SpectralMorph, shift: f32, wavetable: &Wavetable) -> Vec<f32> {
        let mut spectrum = vec![0.0f32; SPECTRUM_LEN];
        run_spectral_morph(
            morph,
            wavetable.data(),
            0,
            shift,
            WAVEFORM_SIZE / 2,
            &[0.0; 32],
            &mut spectrum,
        );
        let fft = wave_fft();
        let mut input = vec![Complex::new(0.0f32, 0.0); NUM_HARMONICS];
        let mut scratch =
            vec![Complex::new(0.0f32, 0.0); fft.c2r.get_scratch_len()];
        let mut frame = vec![0.0f32; FRAME_LEN];
        spectrum_to_frame(&spectrum, fft.c2r.as_ref(), &mut input, &mut scratch, &mut frame);
        frame
    }

    #[test]
    fn passthrough_morph_roundtrips_frame() {
        let mut wavetable = Wavetable::new(1);
        wavetable.load_wave_frame(&WaveFrame::predefined(WaveShape::Saw));
        let frame = morph_to_wave(SpectralMorph::None, 0.0, &wavetable);
        let original = wavetable.data().wave_data(0);
        for i in 0..WAVEFORM_SIZE {
            let diff = (frame[FRAME_GUARD + i] - original[i]).abs();
            assert!(diff < 1e-3, "sample {i}: {} vs {}", frame[FRAME_GUARD + i], original[i]);
        }
        // The guards must wrap the cycle.
        for i in 0..FRAME_GUARD {
            assert_eq!(frame[i], frame[WAVEFORM_SIZE + i]);
            assert_eq!(frame[WAVEFORM_SIZE + FRAME_GUARD + i], frame[FRAME_GUARD + i]);
        }
    }

    #[test]
    fn smear_at_zero_is_passthrough() {
        let mut wavetable = Wavetable::new(1);
        wavetable.load_wave_frame(&WaveFrame::predefined(WaveShape::Square));
        // Shaped smear amount for a 0 input is 0 -> identity.
        let mut values = [PolyF32::ZERO];
        shape_spectral_morph_values(SpectralMorph::Smear, &mut values, false);
        let smear = morph_to_wave(SpectralMorph::Smear, values[0].lane(0), &wavetable);
        let passthrough = morph_to_wave(SpectralMorph::None, 0.0, &wavetable);
        for i in 0..FRAME_LEN {
            assert!((smear[i] - passthrough[i]).abs() < 1e-4);
        }
    }
}
