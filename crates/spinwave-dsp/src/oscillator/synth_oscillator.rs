//! The wavetable oscillator: band-limited playback of morphed wavetable
//! frames with unison, phase distortion, spectral morphing, stereo spread
//! and crossfaded frame updates.
//!
//! Rework of Vital's `synth_oscillator.{h,cpp}`. Phase accumulators are
//! wrapping `u32` values; each `PolyF32` vector carries two unison
//! oscillators (left/right lanes) for each of the two synth voices, so
//! eight vectors cover the full 16-voice unison. Frames are re-rendered
//! from the wavetable spectrum every fade period (7 ms) and crossfaded,
//! which is how continuous frame/morph motion stays click free.
//!
//! Reworked from the reference: no Processor graph â€” parameters arrive in
//! [`SynthOscillatorParams`] once per block (control-rate values are ramped
//! across the block), note events arrive through [`SynthOscillator::note_on`]
//! / [`SynthOscillator::retrigger`], and both outputs are explicit buffers.
//! The single-active-voice lane-compaction optimization of the reference is
//! dropped (both voice slots always process), which changes performance
//! only, never the audio.

use std::sync::{Arc, OnceLock};

use realfft::num_complex::Complex;
use realfft::ComplexToReal;
use spinwave_poly::utils::{catmull_interpolation_matrix, linear_interpolation_matrix};
use spinwave_poly::{constants, math, utils, Matrix, PolyF32, PolyMask, PolyU32, LANES};

use crate::wavetable::wave_frame::wave_fft;
use crate::wavetable::{
    Wavetable, FREQUENCY_BINS, NUM_HARMONICS, NUM_OSCILLATOR_WAVE_FRAMES, WAVEFORM_BITS,
    WAVEFORM_SIZE,
};

use super::phase::{self, u32_lt_signed, DistortionType, INV_PHASE_MULT, PHASE_MULT};
use super::rng::Xorshift32;
use super::spectral_morph::{
    run_spectral_morph, shape_spectral_morph_values, spectrum_to_frame, SpectralMorph,
    FRAME_GUARD, FRAME_LEN, RANDOM_AMPLITUDE_STAGES, SPECTRUM_LEN,
};

pub const MAX_UNISON: usize = 16;
/// Unison-pair vectors: two unison oscillators per vector per voice.
pub const NUM_POLY_PHASE: usize = MAX_UNISON / 2;
const NUM_BUFFERS: usize = NUM_POLY_PHASE * LANES;

const INTERMEDIATE_BITS: u32 = 32 - WAVEFORM_BITS as u32;
const INTERMEDIATE_MASK: u32 = (1 << INTERMEDIATE_BITS) - 1;
const INTERMEDIATE_MULT: f32 = (1u32 << INTERMEDIATE_BITS) as f32;

const CENTER_LOW_AMPLITUDE: f32 = 0.4;
const DETUNED_HIGH_AMPLITUDE: f32 = 0.6;
const WAVETABLE_FADE_TIME: f32 = 0.007;
const NO_MIDI_TRACK_DEFAULT: f32 = 48.0;

const FIFTH_MULT: f32 = 1.498_307_1;
const MAJOR_THIRD_MULT: f32 = 1.259_921_1;
const MINOR_THIRD_MULT: f32 = 1.189_207_1;

const MAX_BUFFER: usize = constants::MAX_BUFFER_SIZE * constants::MAX_OVERSAMPLE;

static ZERO_WAVEFORM: [f32; WAVEFORM_SIZE + 3] = [0.0; WAVEFORM_SIZE + 3];
static ZERO_MODULATION: [PolyF32; MAX_BUFFER] = [PolyF32::ZERO; MAX_BUFFER];

/// Unison stack tunings (Vital's `UnisonStackType`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnisonStackType {
    #[default]
    Normal,
    CenterDropOctave,
    CenterDropOctave2,
    Octave,
    Octave2,
    PowerChord,
    PowerChord2,
    MajorChord,
    MinorChord,
    HarmonicSeries,
    OddHarmonicSeries,
}

const STACK_MULTIPLIERS: [[f32; NUM_POLY_PHASE]; 11] = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
    [1.0, FIFTH_MULT, 2.0, 1.0, FIFTH_MULT, 2.0, 1.0, FIFTH_MULT],
    [1.0, FIFTH_MULT, 2.0, 2.0 * FIFTH_MULT, 4.0, 1.0, FIFTH_MULT, 2.0],
    [1.0, MAJOR_THIRD_MULT, FIFTH_MULT, 2.0, 1.0, MAJOR_THIRD_MULT, FIFTH_MULT, 2.0],
    [1.0, MINOR_THIRD_MULT, FIFTH_MULT, 2.0, 1.0, MINOR_THIRD_MULT, FIFTH_MULT, 2.0],
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
];

impl UnisonStackType {
    fn multipliers(self) -> &'static [f32; NUM_POLY_PHASE] {
        &STACK_MULTIPLIERS[self as usize]
    }
}

/// Block-rate parameters of the oscillator. Continuous values are ramped
/// across the block from their previous value.
#[derive(Clone, Debug)]
pub struct SynthOscillatorParams {
    /// Wavetable frame position, `0..NUM_OSCILLATOR_WAVE_FRAMES`.
    pub wave_frame: PolyF32,
    pub midi_note: PolyF32,
    pub midi_track: bool,
    pub transpose: PolyF32,
    /// Scale bitfield (bits 0..12) plus the global flag at bit 12.
    pub transpose_quantize: u32,
    pub tune: PolyF32,
    pub amplitude: PolyF32,
    pub pan: PolyF32,
    pub unison_voices: usize,
    pub unison_detune: PolyF32,
    /// Manual oscillator phase, in cycles.
    pub phase: PolyF32,
    /// Phase-distortion anchor, in cycles.
    pub distortion_phase: PolyF32,
    /// Amount of phase randomization on note-on, `0..=1`.
    pub random_phase: PolyF32,
    pub blend: PolyF32,
    pub stereo_spread: PolyF32,
    pub stack_style: UnisonStackType,
    pub detune_power: PolyF32,
    /// Detune span scale: `detune_range * unison_detune` is the maximum
    /// detune in cents.
    pub detune_range: PolyF32,
    pub frame_spread: PolyF32,
    pub distortion_spread: PolyF32,
    pub spectral_morph_spread: PolyF32,
    pub spectral_morph_type: SpectralMorph,
    pub spectral_morph_amount: PolyF32,
    pub spectral_unison: bool,
    pub distortion_type: DistortionType,
    pub distortion_amount: PolyF32,
}

impl Default for SynthOscillatorParams {
    fn default() -> Self {
        SynthOscillatorParams {
            wave_frame: PolyF32::ZERO,
            midi_note: PolyF32::splat(60.0),
            midi_track: true,
            transpose: PolyF32::ZERO,
            transpose_quantize: 0,
            tune: PolyF32::ZERO,
            amplitude: PolyF32::ONE,
            pan: PolyF32::ZERO,
            unison_voices: 1,
            unison_detune: PolyF32::ZERO,
            phase: PolyF32::ZERO,
            distortion_phase: PolyF32::splat(0.5),
            random_phase: PolyF32::ZERO,
            blend: PolyF32::splat(0.8),
            stereo_spread: PolyF32::ONE,
            stack_style: UnisonStackType::Normal,
            detune_power: PolyF32::ZERO,
            detune_range: PolyF32::splat(2.0),
            frame_spread: PolyF32::ZERO,
            distortion_spread: PolyF32::ZERO,
            spectral_morph_spread: PolyF32::ZERO,
            spectral_morph_type: SpectralMorph::None,
            spectral_morph_amount: PolyF32::ZERO,
            spectral_unison: true,
            distortion_type: DistortionType::None,
            distortion_amount: PolyF32::splat(0.5),
        }
    }
}

/// Reference into the rendered-frame store (or the silent waveform).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufRef {
    Null,
    Frame(u16),
}

/// Deterministic random table for the random-amplitudes spectral morph.
fn random_amplitude_table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let len = (RANDOM_AMPLITUDE_STAGES + 1) * (NUM_HARMONICS + 1) / LANES * LANES;
        let mut rng = Xorshift32::new(0x4);
        (0..len).map(|_| rng.next_in(-1.0, 1.0)).collect()
    })
}

#[inline(always)]
fn mask_select(condition: PolyMask, if_true: PolyMask, if_false: PolyMask) -> PolyMask {
    PolyMask::from_u32(condition.select_u32(if_true.to_u32(), if_false.to_u32()))
}

#[inline(always)]
fn left_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, 0, u32::MAX, 0]))
}

#[inline(always)]
fn cents_to_ratio(cents: PolyF32) -> PolyF32 {
    math::exp2(cents * (1.0 / constants::CENTS_PER_OCTAVE as f32))
}

/// Band-limit: number of usable harmonics for a normalized phase increment.
pub fn band_limited_harmonics(phase_inc: f32) -> usize {
    let bin = Wavetable::frequency_float_bin(phase_inc);
    let bin_shift = FREQUENCY_BINS as f32 + 1.0 - bin;
    let harmonics = WAVEFORM_SIZE as f32 * (-bin_shift).exp2();
    (harmonics.max(0.0) as usize).min(WAVEFORM_SIZE / 2)
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

#[inline(always)]
fn interpolation_t(indices: PolyU32) -> PolyF32 {
    (indices & PolyU32::splat(INTERMEDIATE_MASK)).to_f32_signed() * (1.0 / INTERMEDIATE_MULT)
}

#[inline(always)]
pub(crate) fn value_matrix_multi(buffers: &[&[f32]; LANES], indices: PolyU32) -> Matrix {
    let row = |lane: usize| {
        let start = indices.lane(lane) as usize;
        let buffer = buffers[lane];
        PolyF32::from_lanes([
            buffer[start],
            buffer[start + 1],
            buffer[start + 2],
            buffer[start + 3],
        ])
    };
    Matrix::new(row(0), row(1), row(2), row(3))
}

/// Linear per-lane read of a single wrapped wave buffer at integer-phase
/// `indices` (Vital's `SynthOscillator::interpolate`). The buffer must
/// carry at least 3 guard samples past the cycle.
pub fn linearly_interpolate_buffer(buffer: &[f32], indices: PolyU32) -> PolyF32 {
    let start_indices = indices.shr(INTERMEDIATE_BITS);
    let t = interpolation_t(indices);
    let interpolation_matrix = linear_interpolation_matrix(t);
    let mut values = spinwave_poly::utils::value_matrix(buffer, start_indices);
    values.transpose();
    interpolation_matrix.multiply_and_sum_rows(&values)
}

#[inline(always)]
fn interpolate_single(buffers: &[&[f32]; LANES], indices: PolyU32) -> PolyF32 {
    let start_indices = indices.shr(INTERMEDIATE_BITS);
    let t = interpolation_t(indices);
    let interpolation_matrix = catmull_interpolation_matrix(t);
    let mut values = value_matrix_multi(buffers, start_indices);
    values.transpose();
    interpolation_matrix.multiply_and_sum_rows(&values)
}

#[inline(always)]
fn interpolate_multi(
    from: &[&[f32]; LANES],
    to: &[&[f32]; LANES],
    indices: PolyU32,
    buffer_t: PolyF32,
) -> PolyF32 {
    let start_indices = indices.shr(INTERMEDIATE_BITS);
    let t = interpolation_t(indices);
    let interpolation_matrix = catmull_interpolation_matrix(t);
    let mut values = value_matrix_multi(from, start_indices);
    values.interpolate_rows(&value_matrix_multi(to, start_indices), buffer_t);
    values.transpose();
    interpolation_matrix.multiply_and_sum_rows(&values)
}

#[inline(always)]
fn interpolate_shepard(
    from: &[&[f32]; LANES],
    to: &[&[f32]; LANES],
    indices: PolyU32,
    buffer_t: PolyF32,
    double_mask: PolyMask,
    half_mask: PolyMask,
) -> PolyF32 {
    let mut adjusted = double_mask.select_u32(indices + indices, indices);
    adjusted = half_mask.select_u32(adjusted.shr(1), adjusted);
    let from_value = interpolate_single(from, adjusted);
    let to_value = interpolate_single(to, indices);
    utils::interpolate(from_value, to_value, buffer_t)
}

// ---------------------------------------------------------------------------
// Per-chunk voice state and sample loops
// ---------------------------------------------------------------------------

struct VoiceRun<'a> {
    start_sample: usize,
    end_sample: usize,
    total_samples: usize,
    phase: PolyU32,
    phase_inc_mult: PolyF32,
    from_phase_inc_mult: PolyF32,
    shepard_double_mask: PolyMask,
    shepard_half_mask: PolyMask,
    distortion_phase: PolyU32,
    last_distortion_phase: PolyU32,
    distortion: PolyF32,
    last_distortion: PolyF32,
    num_buffer_samples: usize,
    current_buffer_sample: PolyU32,
    from_buffers: [&'a [f32]; LANES],
    to_buffers: [&'a [f32]; LANES],
    is_static: bool,
    modulation: &'a [PolyF32],
    phase_inc_buffer: &'a [PolyF32],
    phase_buffer: &'a [PolyU32],
}

#[inline(always)]
fn run_detuned_body<PD, WI>(
    run: &VoiceRun,
    audio_out: &mut [PolyF32],
    pd: PD,
    wi: WI,
    interp: impl Fn(PolyU32, PolyF32) -> PolyF32,
) -> PolyU32
where
    PD: Fn(PolyU32, PolyF32, PolyU32, &[PolyF32], usize) -> PolyU32 + Copy,
    WI: Fn(PolyU32, PolyU32, PolyF32, &[PolyF32], usize) -> PolyF32 + Copy,
{
    let start = run.start_sample;
    let t_inc = PolyF32::splat(1.0 / run.num_buffer_samples as f32);
    let mut t = (run.current_buffer_sample + PolyU32::splat(1)).to_f32_signed() * t_inc;
    let sample_inc = 1.0 / run.total_samples as f32;

    let mut phase = run.phase;
    let mut current_mult = run.from_phase_inc_mult;
    let delta_mult = (run.phase_inc_mult - current_mult) * sample_inc;
    current_mult += delta_mult * start as f32;

    let mut current_dist_phase = run.last_distortion_phase;
    let delta_dist_phase = ((run.distortion_phase - current_dist_phase).to_f32_signed()
        * sample_inc)
        .to_i32_round();
    current_dist_phase += delta_dist_phase * PolyU32::splat(start as u32);

    let mut current_distortion = run.last_distortion;
    let distortion_inc = (run.distortion - current_distortion) * sample_inc;
    current_distortion += distortion_inc * start as f32;

    let modulation = &run.modulation[start..];
    let phase_inc = &run.phase_inc_buffer[start..run.end_sample];
    let phase_off = &run.phase_buffer[start..run.end_sample];
    for (i, ((&inc, &off), out)) in phase_inc
        .iter()
        .zip(phase_off)
        .zip(audio_out.iter_mut())
        .enumerate()
    {
        current_mult += delta_mult;
        phase += (inc * current_mult).to_i32_round();
        let adjusted_phase = phase + off;
        current_distortion += distortion_inc;
        current_dist_phase += delta_dist_phase;
        let distorted = pd(adjusted_phase, current_distortion, current_dist_phase, modulation, i);
        let result = interp(distorted + current_dist_phase, t);
        *out += wi(adjusted_phase, distorted, current_distortion, modulation, i) * result;
        t += t_inc;
    }

    phase
}

fn run_detuned_shepard(run: &VoiceRun, audio_out: &mut [PolyF32]) -> PolyU32 {
    let start = run.start_sample;
    let t_inc = PolyF32::splat(1.0 / run.num_buffer_samples as f32);
    let mut t = (run.current_buffer_sample + PolyU32::splat(1)).to_f32_signed() * t_inc;
    let sample_inc = 1.0 / run.total_samples as f32;

    let mut phase = run.phase;
    let mut current_mult = run.from_phase_inc_mult;
    let delta_mult = (run.phase_inc_mult - current_mult) * sample_inc;
    current_mult += delta_mult * start as f32;

    let phase_inc = &run.phase_inc_buffer[start..run.end_sample];
    let phase_off = &run.phase_buffer[start..run.end_sample];
    for ((&inc, &off), out) in phase_inc.iter().zip(phase_off).zip(audio_out.iter_mut()) {
        current_mult += delta_mult;
        phase += (inc * current_mult).to_i32_round();
        let adjusted_phase = phase + off;
        *out += interpolate_shepard(
            &run.from_buffers,
            &run.to_buffers,
            adjusted_phase,
            t,
            run.shepard_double_mask,
            run.shepard_half_mask,
        );
        t += t_inc;
    }

    phase
}

fn run_detuned<PD, WI>(run: &VoiceRun, audio_out: &mut [PolyF32], pd: PD, wi: WI) -> PolyU32
where
    PD: Fn(PolyU32, PolyF32, PolyU32, &[PolyF32], usize) -> PolyU32 + Copy,
    WI: Fn(PolyU32, PolyU32, PolyF32, &[PolyF32], usize) -> PolyF32 + Copy,
{
    if run.is_static {
        run_detuned_body(run, audio_out, pd, wi, |indices, _t| {
            interpolate_single(&run.to_buffers, indices)
        })
    } else if run.shepard_double_mask.any() || run.shepard_half_mask.any() {
        run_detuned_shepard(run, audio_out)
    } else {
        run_detuned_body(run, audio_out, pd, wi, |indices, t| {
            interpolate_multi(&run.from_buffers, &run.to_buffers, indices, t)
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn run_center_body<PD, WI>(
    run: &VoiceRun,
    audio_out: &mut [PolyF32],
    pd: PD,
    wi: WI,
    mut center_amplitude: PolyF32,
    delta_center_amplitude: PolyF32,
    mut detuned_amplitude: PolyF32,
    delta_detuned_amplitude: PolyF32,
    interp: impl Fn(PolyU32, PolyF32) -> PolyF32,
) -> PolyU32
where
    PD: Fn(PolyU32, PolyF32, PolyU32, &[PolyF32], usize) -> PolyU32 + Copy,
    WI: Fn(PolyU32, PolyU32, PolyF32, &[PolyF32], usize) -> PolyF32 + Copy,
{
    let start = run.start_sample;
    let t_inc = PolyF32::splat(1.0 / run.num_buffer_samples as f32);
    let mut t = (run.current_buffer_sample + PolyU32::splat(1)).to_f32_signed() * t_inc;
    let sample_inc = 1.0 / run.total_samples as f32;

    let mut phase = run.phase;
    let mut current_mult = run.from_phase_inc_mult;
    let delta_mult = (run.phase_inc_mult - current_mult) * sample_inc;
    current_mult += delta_mult * start as f32;

    let mut current_dist_phase = run.last_distortion_phase;
    let delta_dist_phase = ((run.distortion_phase - current_dist_phase).to_f32_signed()
        * sample_inc)
        .to_i32_round();
    current_dist_phase += delta_dist_phase * PolyU32::splat(start as u32);

    let mut current_distortion = run.last_distortion;
    let distortion_inc = (run.distortion - current_distortion) * sample_inc;
    current_distortion += distortion_inc * start as f32;

    let modulation = &run.modulation[start..];
    let phase_inc = &run.phase_inc_buffer[start..run.end_sample];
    let phase_off = &run.phase_buffer[start..run.end_sample];
    for (i, ((&inc, &off), out)) in phase_inc
        .iter()
        .zip(phase_off)
        .zip(audio_out.iter_mut())
        .enumerate()
    {
        current_mult += delta_mult;
        phase += (inc * current_mult).to_i32_round();
        let adjusted_phase = phase + off;
        current_distortion += distortion_inc;
        current_dist_phase += delta_dist_phase;
        center_amplitude += delta_center_amplitude;
        detuned_amplitude += delta_detuned_amplitude;
        let distorted = pd(adjusted_phase, current_distortion, current_dist_phase, modulation, i);
        let mult = wi(adjusted_phase, distorted, current_distortion, modulation, i);
        let read = mult * interp(distorted + current_dist_phase, t);
        *out = center_amplitude * read + detuned_amplitude * *out;
        t += t_inc;
    }

    phase
}

fn run_center_shepard(
    run: &VoiceRun,
    audio_out: &mut [PolyF32],
    mut center_amplitude: PolyF32,
    delta_center_amplitude: PolyF32,
    mut detuned_amplitude: PolyF32,
    delta_detuned_amplitude: PolyF32,
) -> PolyU32 {
    let start = run.start_sample;
    let t_inc = PolyF32::splat(1.0 / run.num_buffer_samples as f32);
    let mut t = (run.current_buffer_sample + PolyU32::splat(1)).to_f32_signed() * t_inc;
    let sample_inc = 1.0 / run.total_samples as f32;

    let mut phase = run.phase;
    let mut current_mult = run.from_phase_inc_mult;
    let delta_mult = (run.phase_inc_mult - current_mult) * sample_inc;
    current_mult += delta_mult * start as f32;

    let phase_inc = &run.phase_inc_buffer[start..run.end_sample];
    let phase_off = &run.phase_buffer[start..run.end_sample];
    for ((&inc, &off), out) in phase_inc.iter().zip(phase_off).zip(audio_out.iter_mut()) {
        current_mult += delta_mult;
        phase += (inc * current_mult).to_i32_round();
        let adjusted_phase = phase + off;
        center_amplitude += delta_center_amplitude;
        detuned_amplitude += delta_detuned_amplitude;
        let read = interpolate_shepard(
            &run.from_buffers,
            &run.to_buffers,
            adjusted_phase,
            t,
            run.shepard_double_mask,
            run.shepard_half_mask,
        );
        *out = center_amplitude * read + detuned_amplitude * *out;
        t += t_inc;
    }

    phase
}

#[allow(clippy::too_many_arguments)]
fn run_center<PD, WI>(
    run: &VoiceRun,
    audio_out: &mut [PolyF32],
    pd: PD,
    wi: WI,
    center_amplitude: PolyF32,
    delta_center_amplitude: PolyF32,
    detuned_amplitude: PolyF32,
    delta_detuned_amplitude: PolyF32,
) -> PolyU32
where
    PD: Fn(PolyU32, PolyF32, PolyU32, &[PolyF32], usize) -> PolyU32 + Copy,
    WI: Fn(PolyU32, PolyU32, PolyF32, &[PolyF32], usize) -> PolyF32 + Copy,
{
    if run.is_static {
        run_center_body(
            run,
            audio_out,
            pd,
            wi,
            center_amplitude,
            delta_center_amplitude,
            detuned_amplitude,
            delta_detuned_amplitude,
            |indices, _t| interpolate_single(&run.to_buffers, indices),
        )
    } else if run.shepard_double_mask.any() || run.shepard_half_mask.any() {
        run_center_shepard(
            run,
            audio_out,
            center_amplitude,
            delta_center_amplitude,
            detuned_amplitude,
            delta_detuned_amplitude,
        )
    } else {
        run_center_body(
            run,
            audio_out,
            pd,
            wi,
            center_amplitude,
            delta_center_amplitude,
            detuned_amplitude,
            delta_detuned_amplitude,
            |indices, t| interpolate_multi(&run.from_buffers, &run.to_buffers, indices, t),
        )
    }
}

// ---------------------------------------------------------------------------
// The oscillator
// ---------------------------------------------------------------------------

pub struct SynthOscillator {
    phases: [PolyU32; NUM_POLY_PHASE],
    detunings: [PolyF32; NUM_POLY_PHASE],
    phase_inc_mults: [PolyF32; NUM_POLY_PHASE],
    from_phase_inc_mults: [PolyF32; NUM_POLY_PHASE],
    shepard_double_masks: [PolyMask; NUM_POLY_PHASE],
    shepard_half_masks: [PolyMask; NUM_POLY_PHASE],
    waiting_shepard_double_masks: [PolyMask; NUM_POLY_PHASE],
    waiting_shepard_half_masks: [PolyMask; NUM_POLY_PHASE],
    spectral_morph_values: [PolyF32; NUM_POLY_PHASE],
    last_spectral_morph_values: [PolyF32; NUM_POLY_PHASE],
    distortion_values: [PolyF32; NUM_POLY_PHASE],
    last_distortion_values: [PolyF32; NUM_POLY_PHASE],

    pan_amplitude: PolyF32,
    center_amplitude: PolyF32,
    detuned_amplitude: PolyF32,
    midi_total: PolyF32,
    distortion_phase: PolyF32,
    blend_stereo_multiply: PolyF32,
    blend_center_multiply: PolyF32,
    last_amplitude: PolyF32,
    last_quantize_ratio: PolyF32,

    wave_buffers: [BufRef; NUM_BUFFERS],
    last_buffers: [BufRef; NUM_BUFFERS],
    frames: Vec<Vec<f32>>,

    vb_current_buffer_sample: PolyU32,
    vb_num_buffer_samples: usize,

    unison: usize,
    active_oscillators: usize,
    wavetable_version: Option<u32>,
    sample_rate: f32,

    pending_reset: PolyMask,
    pending_reset_offset: PolyU32,
    pending_retrigger: PolyMask,

    spectrum: Vec<f32>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    c2r_input: Vec<Complex<f32>>,
    c2r_scratch: Vec<Complex<f32>>,

    phase_inc_buffer: Vec<PolyF32>,
    phase_buffer: Vec<PolyU32>,

    rng: Xorshift32,
}

impl Default for SynthOscillator {
    fn default() -> Self {
        Self::new()
    }
}

impl SynthOscillator {
    pub fn new() -> SynthOscillator {
        let fft = wave_fft();
        SynthOscillator {
            phases: [PolyU32::ZERO; NUM_POLY_PHASE],
            detunings: [PolyF32::ONE; NUM_POLY_PHASE],
            phase_inc_mults: [PolyF32::ONE; NUM_POLY_PHASE],
            from_phase_inc_mults: [PolyF32::ONE; NUM_POLY_PHASE],
            shepard_double_masks: [PolyMask::NONE; NUM_POLY_PHASE],
            shepard_half_masks: [PolyMask::NONE; NUM_POLY_PHASE],
            waiting_shepard_double_masks: [PolyMask::NONE; NUM_POLY_PHASE],
            waiting_shepard_half_masks: [PolyMask::NONE; NUM_POLY_PHASE],
            spectral_morph_values: [PolyF32::ZERO; NUM_POLY_PHASE],
            last_spectral_morph_values: [PolyF32::ONE; NUM_POLY_PHASE],
            distortion_values: [PolyF32::ZERO; NUM_POLY_PHASE],
            last_distortion_values: [PolyF32::ZERO; NUM_POLY_PHASE],

            pan_amplitude: PolyF32::ZERO,
            center_amplitude: PolyF32::ZERO,
            detuned_amplitude: PolyF32::ZERO,
            midi_total: PolyF32::ZERO,
            distortion_phase: PolyF32::ZERO,
            blend_stereo_multiply: PolyF32::ZERO,
            blend_center_multiply: PolyF32::ZERO,
            last_amplitude: PolyF32::ZERO,
            last_quantize_ratio: PolyF32::ONE,

            wave_buffers: [BufRef::Null; NUM_BUFFERS],
            last_buffers: [BufRef::Null; NUM_BUFFERS],
            frames: vec![vec![0.0; FRAME_LEN]; 2 * NUM_BUFFERS],

            vb_current_buffer_sample: PolyU32::ZERO,
            vb_num_buffer_samples: 0,

            unison: 1,
            active_oscillators: 2,
            wavetable_version: None,
            sample_rate: constants::DEFAULT_SAMPLE_RATE as f32,

            pending_reset: PolyMask::NONE,
            pending_reset_offset: PolyU32::ZERO,
            pending_retrigger: PolyMask::NONE,

            spectrum: vec![0.0; SPECTRUM_LEN],
            c2r: fft.c2r.clone(),
            c2r_input: vec![Complex::new(0.0, 0.0); NUM_HARMONICS],
            c2r_scratch: vec![Complex::new(0.0, 0.0); fft.c2r.get_scratch_len()],

            phase_inc_buffer: vec![PolyF32::ZERO; MAX_BUFFER],
            phase_buffer: vec![PolyU32::ZERO; MAX_BUFFER],

            rng: Xorshift32::new(0x9e37_79b9),
        }
    }

    /// Effective (post-oversampling) sample rate for the audio loops.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Schedules a note-on reset for the lanes in `mask`, taking effect at
    /// `sample_offset` (per lane) inside the next processed block:
    /// phases re-randomize, ramps snap, and pre-trigger output is cleared.
    pub fn note_on(&mut self, mask: PolyMask, sample_offset: PolyU32) {
        self.pending_reset |= mask;
        self.pending_reset_offset = mask.select_u32(sample_offset, self.pending_reset_offset);
    }

    /// Schedules a legato retrigger: pitch ramps snap without resetting
    /// phase or fades.
    pub fn retrigger(&mut self, mask: PolyMask) {
        self.pending_retrigger |= mask;
    }

    /// Renders one block. `raw_out` receives the post-spread unison mix,
    /// `leveled_out` the same signal after pan and amplitude
    /// (Vital's `kRaw` / `kLevelled` outputs). `modulation` feeds the
    /// FM/RM distortion modes.
    pub fn process(
        &mut self,
        params: &SynthOscillatorParams,
        wavetable: &Wavetable,
        modulation: Option<&[PolyF32]>,
        num_samples: usize,
        raw_out: &mut [PolyF32],
        leveled_out: &mut [PolyF32],
    ) {
        assert!(num_samples > 0 && num_samples <= MAX_BUFFER);
        assert!(raw_out.len() >= num_samples && leveled_out.len() >= num_samples);

        if self.wavetable_version != Some(wavetable.version()) {
            self.wavetable_version = Some(wavetable.version());
            self.reset_wavetable_buffers();
        }

        self.unison = params.unison_voices.clamp(1, MAX_UNISON);
        self.set_active_oscillators(self.unison + self.unison % 2);
        self.set_spectral_morph_values(params, wavetable);
        self.set_distortion_values(params);

        let modulation = modulation.unwrap_or(&ZERO_MODULATION);
        assert!(modulation.len() >= num_samples);

        match params.distortion_type {
            DistortionType::Sync => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::sync_phase, phase::pass_through_window,
            ),
            DistortionType::Formant => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::sync_phase, phase::half_sin_window,
            ),
            DistortionType::Quantize => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::quantize_phase, phase::pass_through_window,
            ),
            DistortionType::Bend => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::bend_phase, phase::pass_through_window,
            ),
            DistortionType::Squeeze => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::squeeze_phase, phase::pass_through_window,
            ),
            DistortionType::PulseWidth => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::pulse_width_phase, phase::pulse_width_window,
            ),
            DistortionType::FmOscillatorA
            | DistortionType::FmOscillatorB
            | DistortionType::FmSample => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::fm_phase, phase::pass_through_window,
            ),
            DistortionType::RmOscillatorA
            | DistortionType::RmOscillatorB
            | DistortionType::RmSample => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::pass_through_phase, phase::rm_window,
            ),
            DistortionType::None => self.process_oscillators(
                params, wavetable, modulation, num_samples, raw_out, leveled_out,
                phase::pass_through_phase, phase::pass_through_window,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_oscillators<PD, WI>(
        &mut self,
        params: &SynthOscillatorParams,
        wavetable: &Wavetable,
        modulation: &[PolyF32],
        num_samples: usize,
        raw_out: &mut [PolyF32],
        leveled_out: &mut [PolyF32],
        pd: PD,
        wi: WI,
    ) where
        PD: Fn(PolyU32, PolyF32, PolyU32, &[PolyF32], usize) -> PolyU32 + Copy,
        WI: Fn(PolyU32, PolyU32, PolyF32, &[PolyF32], usize) -> PolyF32 + Copy,
    {
        let mut current_center_amplitude = self.center_amplitude;
        let mut current_detuned_amplitude = self.detuned_amplitude;
        self.set_amplitude(params);

        let reset_mask = self.pending_reset;
        let trigger_offset = self.pending_reset_offset;
        let retrigger_mask = self.pending_retrigger & !reset_mask;
        self.pending_reset = PolyMask::NONE;
        self.pending_retrigger = PolyMask::NONE;

        current_center_amplitude = reset_mask.select(self.center_amplitude, current_center_amplitude);
        current_detuned_amplitude =
            reset_mask.select(self.detuned_amplitude, current_detuned_amplitude);

        self.set_phase_inc_mults(params);
        self.set_phase_inc_buffer(params, num_samples, reset_mask, trigger_offset);

        let mut current_distortion_phase = self.distortion_phase;
        self.distortion_phase = if params.distortion_type.uses_distortion_phase() {
            params.distortion_phase - 0.5
        } else {
            PolyF32::ZERO
        };
        current_distortion_phase =
            reset_mask.select(self.distortion_phase, current_distortion_phase);

        let last_distortion_phase = (current_distortion_phase * PHASE_MULT).to_i32_round();
        let distortion_phase = (self.distortion_phase * PHASE_MULT).to_i32_round();

        let wave_buffer_mask = reset_mask | retrigger_mask;
        let buffer_phase_inc = self.phase_inc_buffer[num_samples - 1] * INV_PHASE_MULT;
        if wave_buffer_mask.to_u32().lane(0) != 0 {
            self.set_wave_buffers(params, wavetable, buffer_phase_inc, 0);
        }
        if wave_buffer_mask.to_u32().lane(2) != 0 {
            self.set_wave_buffers(params, wavetable, buffer_phase_inc, 2);
        }

        if reset_mask.any() {
            self.reset(params.random_phase, reset_mask, trigger_offset);
        }

        if retrigger_mask.any() {
            for i in 0..NUM_POLY_PHASE {
                self.from_phase_inc_mults[i] =
                    retrigger_mask.select(self.phase_inc_mults[i], self.from_phase_inc_mults[i]);
            }
        }

        let num_buffer_samples = (WAVETABLE_FADE_TIME * self.sample_rate) as usize;
        if self.vb_num_buffer_samples != num_buffer_samples {
            self.vb_num_buffer_samples = num_buffer_samples.max(1);
            self.vb_current_buffer_sample = PolyU32::ZERO;
        }

        let shepard = params.spectral_morph_type == SpectralMorph::ShepardTone;
        if shepard {
            self.setup_shepard_wrap();
        } else {
            self.clear_shepard_wrap();
        }

        let mut start_sample = 0;
        while start_sample < num_samples {
            let remaining0 = self.vb_num_buffer_samples as i64
                - self.vb_current_buffer_sample.lane(0) as i32 as i64;
            let remaining2 = self.vb_num_buffer_samples as i64
                - self.vb_current_buffer_sample.lane(2) as i32 as i64;
            let min_remaining = remaining0.min(remaining2).max(1) as usize;
            let samples = min_remaining.min(num_samples - start_sample);
            let end_sample = start_sample + samples;

            self.process_chunk(
                modulation,
                start_sample,
                end_sample,
                num_samples,
                current_center_amplitude,
                current_detuned_amplitude,
                last_distortion_phase,
                distortion_phase,
                raw_out,
                pd,
                wi,
            );

            self.vb_current_buffer_sample += PolyU32::splat(samples as u32);
            start_sample = end_sample;

            let new_buffer_mask = self
                .vb_current_buffer_sample
                .eq(PolyU32::splat(self.vb_num_buffer_samples as u32));
            if shepard && new_buffer_mask.any() {
                self.do_shepard_wrap(new_buffer_mask);
            }
            if new_buffer_mask.to_u32().lane(0) != 0 {
                self.set_wave_buffers(params, wavetable, buffer_phase_inc, 0);
            }
            if new_buffer_mask.to_u32().lane(2) != 0 {
                self.set_wave_buffers(params, wavetable, buffer_phase_inc, 2);
            }
        }

        if reset_mask.any() {
            for (i, out) in raw_out.iter_mut().take(num_samples).enumerate() {
                let zero_mask =
                    u32_lt_signed(PolyU32::splat(i as u32), trigger_offset) & reset_mask;
                *out = *out & !zero_mask;
            }
        }

        self.process_blend(params, num_samples, reset_mask, raw_out, leveled_out);
    }

    #[allow(clippy::too_many_arguments)]
    fn process_chunk<PD, WI>(
        &mut self,
        modulation: &[PolyF32],
        start: usize,
        end: usize,
        total: usize,
        current_center_amplitude: PolyF32,
        current_detuned_amplitude: PolyF32,
        last_distortion_phase: PolyU32,
        distortion_phase: PolyU32,
        raw_out: &mut [PolyF32],
        pd: PD,
        wi: WI,
    ) where
        PD: Fn(PolyU32, PolyF32, PolyU32, &[PolyF32], usize) -> PolyU32 + Copy,
        WI: Fn(PolyU32, PolyU32, PolyF32, &[PolyF32], usize) -> PolyF32 + Copy,
    {
        for out in &mut raw_out[start..end] {
            *out = PolyF32::ZERO;
        }

        let num_phase_updates = self.active_oscillators / 2;
        for p in 1..num_phase_updates {
            let new_phase = {
                let run = self.voice_run(
                    p,
                    start,
                    end,
                    total,
                    last_distortion_phase,
                    distortion_phase,
                    modulation,
                );
                run_detuned(&run, &mut raw_out[start..end], pd, wi)
            };
            self.phases[p] = new_phase;
        }

        let sample_inc = 1.0 / total as f32;
        let delta_center = (self.center_amplitude - current_center_amplitude) * sample_inc;
        let delta_detuned = (self.detuned_amplitude - current_detuned_amplitude) * sample_inc;
        let chunk_center = current_center_amplitude + delta_center * start as f32;
        let chunk_detuned = current_detuned_amplitude + delta_detuned * start as f32;

        let new_phase = {
            let run = self.voice_run(
                0,
                start,
                end,
                total,
                last_distortion_phase,
                distortion_phase,
                modulation,
            );
            run_center(
                &run,
                &mut raw_out[start..end],
                pd,
                wi,
                chunk_center,
                delta_center,
                chunk_detuned,
                delta_detuned,
            )
        };
        self.phases[0] = new_phase;
    }

    fn resolve(&self, buffer: BufRef) -> &[f32] {
        match buffer {
            BufRef::Null => &ZERO_WAVEFORM,
            BufRef::Frame(id) => &self.frames[id as usize][FRAME_GUARD - 1..],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn voice_run<'a>(
        &'a self,
        index: usize,
        start: usize,
        end: usize,
        total: usize,
        last_distortion_phase: PolyU32,
        distortion_phase: PolyU32,
        modulation: &'a [PolyF32],
    ) -> VoiceRun<'a> {
        let buffer_index = index * LANES;
        let from_refs = [
            self.last_buffers[buffer_index],
            self.last_buffers[buffer_index + 1],
            self.last_buffers[buffer_index + 2],
            self.last_buffers[buffer_index + 3],
        ];
        let to_refs = [
            self.wave_buffers[buffer_index],
            self.wave_buffers[buffer_index + 1],
            self.wave_buffers[buffer_index + 2],
            self.wave_buffers[buffer_index + 3],
        ];
        let is_static = from_refs == to_refs;

        VoiceRun {
            start_sample: start,
            end_sample: end,
            total_samples: total,
            phase: self.phases[index],
            phase_inc_mult: self.phase_inc_mults[index],
            from_phase_inc_mult: self.from_phase_inc_mults[index],
            shepard_double_mask: self.shepard_double_masks[index],
            shepard_half_mask: self.shepard_half_masks[index],
            distortion_phase,
            last_distortion_phase,
            distortion: self.distortion_values[index],
            last_distortion: self.last_distortion_values[index],
            num_buffer_samples: self.vb_num_buffer_samples,
            current_buffer_sample: self.vb_current_buffer_sample,
            from_buffers: from_refs.map(|r| self.resolve(r)),
            to_buffers: to_refs.map(|r| self.resolve(r)),
            is_static,
            modulation,
            phase_inc_buffer: &self.phase_inc_buffer,
            phase_buffer: &self.phase_buffer,
        }
    }

    fn reset_wavetable_buffers(&mut self) {
        self.wave_buffers = [BufRef::Null; NUM_BUFFERS];
        self.last_buffers = [BufRef::Null; NUM_BUFFERS];
    }

    fn set_active_oscillators(&mut self, new_active_oscillators: usize) {
        for i in self.active_oscillators..new_active_oscillators {
            self.wave_buffers[2 * i] = BufRef::Null;
            self.wave_buffers[2 * i + 1] = BufRef::Null;
        }
        self.active_oscillators = new_active_oscillators;
    }

    fn set_amplitude(&mut self, params: &SynthOscillatorParams) {
        if self.unison <= 2 {
            self.center_amplitude = PolyF32::ONE;
            self.detuned_amplitude = PolyF32::ZERO;
            return;
        }

        let blend = params.blend;
        let center = utils::interpolate(PolyF32::ONE, PolyF32::splat(CENTER_LOW_AMPLITUDE), blend);
        let mut detuned_blend = -blend + 1.0;
        detuned_blend = detuned_blend * detuned_blend;
        let detuned =
            utils::interpolate(PolyF32::splat(DETUNED_HIGH_AMPLITUDE), PolyF32::ZERO, detuned_blend);

        let half_oscillators = (self.active_oscillators / 2) as f32;
        let square_sums = center * center + detuned * detuned * (half_oscillators - 1.0);
        let adjustment = PolyF32::ONE / square_sums.sqrt();
        self.center_amplitude = adjustment * center;
        self.detuned_amplitude = adjustment * detuned;
    }

    fn set_phase_inc_mults(&mut self, params: &SynthOscillatorParams) {
        let range = params.detune_range;
        let cents = range * params.unison_detune;
        let power = params.detune_power;
        let stack_settings = params.stack_style.multipliers();

        let divisor = (self.unison as f32 - 1.0).max(1.0);
        let bump = if self.unison.is_multiple_of(2) { 1 } else { 0 };

        let mut sharp_flat_mask = left_mask();
        let num_updates = self.active_oscillators / 2;
        for (i, &stack_multiplier) in stack_settings.iter().enumerate().take(num_updates) {
            let t = (2 * i + bump) as f32 / divisor;
            let adjusted_t = math::power_scale(PolyF32::splat(t), power);
            let oscillator_cents = adjusted_t * cents;

            let up_ratio = cents_to_ratio(oscillator_cents);
            let down_ratio = PolyF32::ONE / up_ratio;
            self.detunings[i] = sharp_flat_mask.select(down_ratio, up_ratio) * stack_multiplier;
            self.from_phase_inc_mults[i] = self.phase_inc_mults[i];
            self.phase_inc_mults[i] = self.detunings[i];

            sharp_flat_mask = !sharp_flat_mask;
        }
    }

    /// Builds the per-sample phase increment and manual-phase-offset
    /// buffers from the block's pitch parameters, ramping pitch from the
    /// previous block's value.
    fn set_phase_inc_buffer(
        &mut self,
        params: &SynthOscillatorParams,
        num_samples: usize,
        reset_mask: PolyMask,
        trigger_offset: PolyU32,
    ) {
        let midi_note = if params.midi_track {
            params.midi_note
        } else {
            PolyF32::splat(NO_MIDI_TRACK_DEFAULT)
        };

        let quantize = params.transpose_quantize;
        let snapping = quantize & ((1 << constants::NOTES_PER_OCTAVE) - 1) != 0;
        let global = quantize >> constants::NOTES_PER_OCTAVE != 0;
        let pitch = if snapping {
            if global {
                utils::snap_transpose(midi_note + params.transpose, quantize)
            } else {
                midi_note + utils::snap_transpose(params.transpose, quantize)
            }
        } else {
            midi_note + params.transpose
        };
        let target = pitch + params.tune;

        let mut current = reset_mask.select(target, self.midi_total);
        let delta = (target - current) * (1.0 / num_samples as f32);
        self.midi_total = target;

        let sample_rate_scale = PHASE_MULT / self.sample_rate;
        let shift_phase = params.phase.fract() - 0.5;
        let phase_value = (shift_phase * PHASE_MULT).to_i32_round();

        for i in 0..num_samples {
            self.phase_buffer[i] = phase_value;
            current += delta;
            let frequency = math::midi_note_to_frequency(current);
            let zero_mask = u32_lt_signed(PolyU32::splat(i as u32), trigger_offset) & reset_mask;
            self.phase_inc_buffer[i] = (frequency * sample_rate_scale) & !zero_mask;
        }
    }

    fn set_spectral_morph_values(&mut self, params: &SynthOscillatorParams, wavetable: &Wavetable) {
        const MOD_MULT: f32 = 0.99;
        let spectral_morph = params.spectral_morph_type;
        let amount = params.spectral_morph_amount;
        let morph_spread = params.spectral_morph_spread;
        let num_phase_updates = self.active_oscillators / 2;

        for v in 0..NUM_POLY_PHASE {
            let t = (v as f32 / (num_phase_updates.max(2) as f32 - 1.0)) * 2.0;
            self.last_spectral_morph_values[v] = self.spectral_morph_values[v];
            self.spectral_morph_values[v] = amount + morph_spread * t;
        }

        if spectral_morph == SpectralMorph::ShepardTone {
            for value in &mut self.spectral_morph_values {
                *value = (*value * MOD_MULT).fract() * (1.0 / MOD_MULT);
            }
        } else {
            for value in &mut self.spectral_morph_values {
                *value = value.clamp(0.0, 1.0);
            }
        }

        let is_spread = morph_spread.ne(PolyF32::ZERO).any();
        shape_spectral_morph_values(spectral_morph, &mut self.spectral_morph_values, is_spread);

        if spectral_morph == SpectralMorph::Vocode {
            const DEFAULT_VOCODE_SAMPLE_RATE: f32 = 88200.0;
            let mut wave_sample_rate = wavetable.data().sample_rate;
            if wave_sample_rate <= 0.0 {
                wave_sample_rate = DEFAULT_VOCODE_SAMPLE_RATE;
            }
            let sample_rate_ratio = self.sample_rate / wave_sample_rate;
            let frequency_ratio = sample_rate_ratio * wavetable.data().frequency_ratio;
            for value in &mut self.spectral_morph_values {
                *value *= frequency_ratio;
            }
        }
    }

    fn set_distortion_values(&mut self, params: &SynthOscillatorParams) {
        let distortion_type = params.distortion_type;
        let amount = params.distortion_amount;
        let spread = params.distortion_spread;
        let num_phase_updates = self.active_oscillators / 2;

        for v in 0..NUM_POLY_PHASE {
            let t = v as f32 / (num_phase_updates.max(2) as f32 - 1.0) * 2.0;
            self.last_distortion_values[v] = self.distortion_values[v];
            self.distortion_values[v] = (amount + spread * t).clamp(0.0, 1.0);
        }

        if distortion_type == DistortionType::Quantize {
            for value in &mut self.last_distortion_values {
                *value = value.max(PolyF32::splat(1.5));
            }
        }

        let is_spread = spread.ne(PolyF32::ZERO).any();
        phase::shape_distortion_values(distortion_type, &mut self.distortion_values, is_spread);
    }

    fn phase_inc_adjustment(&self) -> f32 {
        let mut adjustment = 1.0f32;
        let mut sample_rate_mult = self.sample_rate as i32 / constants::DEFAULT_SAMPLE_RATE as i32;
        while sample_rate_mult > 1 {
            sample_rate_mult >>= 1;
            adjustment *= 2.0;
        }
        adjustment
    }

    /// Recomputes the morphed wave buffers for one voice's lanes
    /// (`index` is 0 or 2) and restarts that voice's crossfade.
    fn set_wave_buffers(
        &mut self,
        params: &SynthOscillatorParams,
        wavetable: &Wavetable,
        phase_inc: PolyF32,
        index: usize,
    ) {
        let morph = params.spectral_morph_type;
        let formant_shift = morph == SpectralMorph::Vocode;
        let phase_inc = phase_inc.max(PolyF32::ZERO);
        let phase_inc_adjustment = self.phase_inc_adjustment();

        let (distortion_frequency, distortion_mult) = match params.distortion_type {
            DistortionType::Formant | DistortionType::Sync => (true, phase::MAX_SYNC),
            _ => (false, 1.0),
        };

        let spectral_morph_mask = self.spectral_morph_values[0]
            .ne(self.spectral_morph_values[1])
            .any();
        let frame_spread_any = params.frame_spread.ne(PolyF32::ZERO).any();
        let spectral_unison = params.spectral_unison
            && (spectral_morph_mask || frame_spread_any || morph == SpectralMorph::Vocode);

        let num_phase_updates = self.active_oscillators / 2;
        let max_frame = (NUM_OSCILLATOR_WAVE_FRAMES - 1) as f32;
        if spectral_unison {
            let t_inc = 1.0 / (num_phase_updates.max(2) as f32 - 1.0);
            for v in 0..num_phase_updates {
                let frequency_mult = if distortion_frequency {
                    self.distortion_values[v] * distortion_mult
                } else {
                    PolyF32::ONE
                };

                let morph_amount = self.spectral_morph_values[v];
                let voice_increment = phase_inc * self.detunings[v] * frequency_mult;
                let t = v as f32 * t_inc;
                let frame = params.wave_frame + params.frame_spread * t;
                let wave_index = frame.clamp(0.0, max_frame).to_i32_round();
                self.compute_spectral_wave_buffer_pair(
                    wavetable,
                    morph,
                    v,
                    index,
                    formant_shift,
                    phase_inc_adjustment,
                    wave_index,
                    voice_increment,
                    morph_amount,
                );
            }
        } else {
            let frequency_mult = if distortion_frequency {
                self.distortion_values[0] * distortion_mult
            } else {
                PolyF32::ONE
            };

            let morph_amount = self.spectral_morph_values[0];
            let voice_increment = phase_inc * self.detunings[0] * frequency_mult;
            let wave_index = params.wave_frame.clamp(0.0, max_frame).to_i32_round();

            self.compute_spectral_wave_buffer_pair(
                wavetable,
                morph,
                0,
                index,
                formant_shift,
                phase_inc_adjustment,
                wave_index,
                voice_increment,
                morph_amount,
            );

            for v in 1..num_phase_updates {
                for i in index..index + 2 {
                    let buffer_index = v * LANES + i;
                    self.last_buffers[buffer_index] = self.wave_buffers[buffer_index];
                    self.wave_buffers[buffer_index] = self.wave_buffers[i];
                }
            }
        }

        self.vb_current_buffer_sample.set_lane(index, 0);
        self.vb_current_buffer_sample.set_lane(index + 1, 0);
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_spectral_wave_buffer_pair(
        &mut self,
        wavetable: &Wavetable,
        morph: SpectralMorph,
        phase_update: usize,
        index: usize,
        formant_shift: bool,
        phase_adjustment: f32,
        wave_index: PolyU32,
        voice_increment: PolyF32,
        morph_amount: PolyF32,
    ) {
        for i in index..index + 2 {
            let adjust_phase_inc = voice_increment.lane(i) * phase_adjustment;
            let formant_adjustment = voice_increment.lane(i) * WAVEFORM_SIZE as f32;
            let buffer_index = phase_update * LANES + i;
            self.last_buffers[buffer_index] = self.wave_buffers[buffer_index];

            let candidate = (2 * buffer_index) as u16;
            let frame_id = if self.wave_buffers[buffer_index] == BufRef::Frame(candidate) {
                candidate + 1
            } else {
                candidate
            };

            let mut shift = morph_amount.lane(i);
            if formant_shift {
                shift *= formant_adjustment;
            }
            let table_index = (wave_index.lane(i) as usize).min(wavetable.num_frames() - 1);
            let last_harmonic = band_limited_harmonics(adjust_phase_inc);

            run_spectral_morph(
                morph,
                wavetable.data(),
                table_index,
                shift,
                last_harmonic,
                random_amplitude_table(),
                &mut self.spectrum,
            );
            spectrum_to_frame(
                &self.spectrum,
                self.c2r.as_ref(),
                &mut self.c2r_input,
                &mut self.c2r_scratch,
                &mut self.frames[frame_id as usize],
            );
            self.wave_buffers[buffer_index] = BufRef::Frame(frame_id);

            if i == index
                && morph_amount.lane(i) == morph_amount.lane(i + 1)
                && wave_index.lane(i) == wave_index.lane(i + 1)
            {
                self.last_buffers[buffer_index + 1] = self.wave_buffers[buffer_index + 1];
                self.wave_buffers[buffer_index + 1] = self.wave_buffers[buffer_index];
                return;
            }
        }
    }

    fn setup_shepard_wrap(&mut self) {
        let num_phase_updates = self.active_oscillators / 2;
        let ratio_div = PolyF32::ONE / self.last_quantize_ratio;
        for i in 0..num_phase_updates {
            let spectral_diff = self.last_spectral_morph_values[i] - self.spectral_morph_values[i];
            let mult = math::exp2(-self.spectral_morph_values[i]);
            self.phase_inc_mults[i] *= mult;
            self.detunings[i] *= mult;

            let double_mask =
                self.waiting_shepard_double_masks[i] | spectral_diff.lt(PolyF32::splat(-0.6));
            let half_mask =
                self.waiting_shepard_half_masks[i] | spectral_diff.gt(PolyF32::splat(0.6));

            self.phase_inc_mults[i] =
                double_mask.select(self.phase_inc_mults[i] * 2.0, self.phase_inc_mults[i]);
            self.phase_inc_mults[i] =
                half_mask.select(self.phase_inc_mults[i] * 0.5, self.phase_inc_mults[i]);
            let reset_phase_inc_mult = self.from_phase_inc_mults[i] * ratio_div;
            self.from_phase_inc_mults[i] = (half_mask | double_mask)
                .select(reset_phase_inc_mult, self.from_phase_inc_mults[i]);

            self.waiting_shepard_double_masks[i] = double_mask;
            self.waiting_shepard_half_masks[i] = half_mask;
        }
    }

    fn clear_shepard_wrap(&mut self) {
        let num_phase_updates = self.active_oscillators / 2;
        for i in 0..num_phase_updates {
            self.shepard_double_masks[i] = PolyMask::NONE;
            self.shepard_half_masks[i] = PolyMask::NONE;
            self.waiting_shepard_double_masks[i] = PolyMask::NONE;
            self.waiting_shepard_half_masks[i] = PolyMask::NONE;
        }
    }

    /// Octave-wraps voices whose shepard slide crossed a boundary. (The
    /// reference has a quantized variant too, but its `transpose_quantize_`
    /// member is never written, so only this branch is live there.)
    fn do_shepard_wrap(&mut self, new_buffer_mask: PolyMask) {
        let num_phase_updates = self.active_oscillators / 2;
        for i in 0..num_phase_updates {
            let double_mask = self.waiting_shepard_double_masks[i] & new_buffer_mask;
            let half_mask = self.waiting_shepard_half_masks[i] & new_buffer_mask;
            self.waiting_shepard_double_masks[i] &= !new_buffer_mask;
            self.waiting_shepard_half_masks[i] &= !new_buffer_mask;

            self.phase_inc_mults[i] =
                double_mask.select(self.phase_inc_mults[i] * 0.5, self.phase_inc_mults[i]);
            self.from_phase_inc_mults[i] = double_mask
                .select(self.from_phase_inc_mults[i] * 0.5, self.from_phase_inc_mults[i]);
            self.phases[i] = double_mask.select_u32(self.phases[i].shr(1), self.phases[i]);

            self.phase_inc_mults[i] =
                half_mask.select(self.phase_inc_mults[i] * 2.0, self.phase_inc_mults[i]);
            self.from_phase_inc_mults[i] = half_mask
                .select(self.from_phase_inc_mults[i] * 2.0, self.from_phase_inc_mults[i]);
            self.phases[i] = half_mask.select_u32(self.phases[i].shl(1), self.phases[i]);

            self.shepard_double_masks[i] =
                mask_select(new_buffer_mask, double_mask, self.shepard_double_masks[i]);
            self.shepard_half_masks[i] =
                mask_select(new_buffer_mask, half_mask, self.shepard_half_masks[i]);
        }
    }

    /// Note-on reset: randomizes phases, snaps smoothing ramps and marks
    /// the crossfade start (`sample` is the trigger offset in the block).
    fn reset(&mut self, random_amount: PolyF32, reset_mask: PolyMask, sample: PolyU32) {
        self.last_quantize_ratio = reset_mask.select(PolyF32::ONE, self.last_quantize_ratio);

        for v in 0..2 {
            if reset_mask.to_u32().lane(2 * v) == 0 {
                continue;
            }
            for i in 0..NUM_POLY_PHASE {
                let left = self.rng.next_in(-1.0, 1.0) * random_amount.lane(2 * v) * i32::MAX as f32;
                let right =
                    self.rng.next_in(-1.0, 1.0) * random_amount.lane(2 * v + 1) * i32::MAX as f32;
                self.phases[i].set_lane(2 * v, left as i64 as u32);
                self.phases[i].set_lane(2 * v + 1, right as i64 as u32);

                let buffer_index = i * LANES + 2 * v;
                self.last_buffers[buffer_index] = self.wave_buffers[buffer_index];
                self.last_buffers[buffer_index + 1] = self.wave_buffers[buffer_index + 1];
            }

            // Odd unison counts: the extra center lane mirrors its pair.
            if self.unison < self.active_oscillators {
                let value = self.phases[0].lane(2 * v + 1);
                self.phases[0].set_lane(2 * v, value);
            }
        }

        for i in 0..NUM_POLY_PHASE {
            self.last_distortion_values[i] =
                reset_mask.select(self.distortion_values[i], self.last_distortion_values[i]);
            self.last_spectral_morph_values[i] = reset_mask
                .select(self.spectral_morph_values[i], self.last_spectral_morph_values[i]);
            self.from_phase_inc_mults[i] =
                reset_mask.select(self.phase_inc_mults[i], self.from_phase_inc_mults[i]);
            self.shepard_double_masks[i] &= !reset_mask;
            self.shepard_half_masks[i] &= !reset_mask;
            self.waiting_shepard_double_masks[i] &= !reset_mask;
            self.waiting_shepard_half_masks[i] &= !reset_mask;
        }

        let negative_sample = PolyU32::ZERO - sample;
        self.vb_current_buffer_sample =
            reset_mask.select_u32(negative_sample, self.vb_current_buffer_sample);
    }

    fn process_blend(
        &mut self,
        params: &SynthOscillatorParams,
        num_samples: usize,
        reset_mask: PolyMask,
        raw_out: &mut [PolyF32],
        leveled_out: &mut [PolyF32],
    ) {
        self.stereo_blend(params, num_samples, reset_mask, raw_out);
        self.level_output(params, num_samples, reset_mask, raw_out, leveled_out);
    }

    /// Spreads unison voices across the stereo field by mixing each lane
    /// with its swapped-stereo counterpart.
    fn stereo_blend(
        &mut self,
        params: &SynthOscillatorParams,
        num_samples: usize,
        reset_mask: PolyMask,
        audio_out: &mut [PolyF32],
    ) {
        let stereo_spread = params.stereo_spread.clamp(0.0, 1.0);

        let mut current_stereo_mult = self.blend_stereo_multiply;
        let mut current_center_mult = self.blend_center_multiply;
        self.blend_stereo_multiply = math::equal_power_fade(stereo_spread * 0.5 + 0.5);
        self.blend_center_multiply = math::equal_power_fade_inverse(stereo_spread * 0.5 + 0.5);

        current_stereo_mult = reset_mask.select(self.blend_stereo_multiply, current_stereo_mult);
        current_center_mult = reset_mask.select(self.blend_center_multiply, current_center_mult);
        let inv_samples = 1.0 / num_samples as f32;
        let delta_stereo_mult = (self.blend_stereo_multiply - current_stereo_mult) * inv_samples;
        let delta_center_mult = (self.blend_center_multiply - current_center_mult) * inv_samples;

        if delta_stereo_mult.sum_lanes() + delta_center_mult.sum_lanes() == 0.0
            && stereo_spread.eq(PolyF32::ONE).all()
        {
            return;
        }

        for out in audio_out.iter_mut().take(num_samples) {
            current_stereo_mult += delta_stereo_mult;
            current_center_mult += delta_center_mult;
            let value = *out;
            let swap = value.swap_stereo();
            *out = value * current_stereo_mult + swap * current_center_mult;
        }
    }

    /// Applies pan and squared amplitude to produce the leveled output.
    fn level_output(
        &mut self,
        params: &SynthOscillatorParams,
        num_samples: usize,
        reset_mask: PolyMask,
        raw_out: &[PolyF32],
        leveled_out: &mut [PolyF32],
    ) {
        let mut current_pan_amplitude = self.pan_amplitude;
        self.pan_amplitude = math::pan_amplitude(params.pan.clamp(-1.0, 1.0));
        current_pan_amplitude = reset_mask.select(self.pan_amplitude, current_pan_amplitude);
        let delta_pan_amplitude =
            (self.pan_amplitude - current_pan_amplitude) * (1.0 / num_samples as f32);

        let target_amplitude = params.amplitude.max(PolyF32::ZERO);
        let mut current_amplitude = reset_mask.select(target_amplitude, self.last_amplitude);
        let delta_amplitude = (target_amplitude - current_amplitude) * (1.0 / num_samples as f32);
        self.last_amplitude = target_amplitude;

        for (out, &raw) in leveled_out
            .iter_mut()
            .zip(raw_out.iter())
            .take(num_samples)
        {
            current_pan_amplitude += delta_pan_amplitude;
            current_amplitude += delta_amplitude;
            *out = current_pan_amplitude * raw * current_amplitude * current_amplitude;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wavetable::{WaveFrame, WaveShape};

    fn sine_wavetable() -> Wavetable {
        let mut wavetable = Wavetable::new(1);
        wavetable.load_wave_frame(&WaveFrame::predefined(WaveShape::Sin));
        wavetable
    }

    #[test]
    fn band_limit_shrinks_with_pitch() {
        let low = band_limited_harmonics(55.0 / 44100.0);
        let mid = band_limited_harmonics(440.0 / 44100.0);
        let high = band_limited_harmonics(3520.0 / 44100.0);
        assert!(low > mid && mid > high, "low={low} mid={mid} high={high}");
        // Band limit keeps harmonics at or below Nyquist.
        let f = 440.0 / 44100.0;
        let harmonics = band_limited_harmonics(f);
        assert!(harmonics as f32 * 440.0 <= 22050.0 * 1.01);
        assert!(harmonics as f32 * 440.0 >= 22050.0 * 0.8);
    }

    #[test]
    fn sine_table_plays_at_440_hz() {
        let wavetable = sine_wavetable();
        let mut oscillator = SynthOscillator::new();
        oscillator.set_sample_rate(44100.0);
        let params = SynthOscillatorParams {
            midi_note: PolyF32::splat(69.0), // A4 = 440 Hz
            ..Default::default()
        };

        oscillator.note_on(PolyMask::all_on(), PolyU32::ZERO);

        const BLOCK: usize = 128;
        const BLOCKS: usize = 400;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];
        let mut samples = Vec::with_capacity(BLOCK * BLOCKS);
        for _ in 0..BLOCKS {
            oscillator.process(&params, &wavetable, None, BLOCK, &mut raw, &mut leveled);
            for value in &raw {
                samples.push(value.lane(0));
            }
        }

        // Count zero crossings over exactly one second, skipping the attack.
        let start = 1000;
        let end = start + 44100;
        let mut crossings = 0;
        for i in start + 1..end {
            if (samples[i - 1] >= 0.0) != (samples[i] >= 0.0) {
                crossings += 1;
            }
        }
        assert!(
            (860..=900).contains(&crossings),
            "expected ~880 crossings for 440 Hz, got {crossings}"
        );

        // The sine should reach close to full scale in the raw output.
        let peak = samples[start..end].iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(peak > 0.9 && peak < 1.1, "peak={peak}");
    }

    #[test]
    fn unison_detune_ratios() {
        let mut oscillator = SynthOscillator::new();
        oscillator.unison = 2;
        oscillator.active_oscillators = 2;
        let params = SynthOscillatorParams {
            unison_voices: 2,
            unison_detune: PolyF32::ONE,
            detune_range: PolyF32::splat(100.0), // 100 cents at full detune
            ..Default::default()
        };
        oscillator.set_phase_inc_mults(&params);

        let mults = oscillator.phase_inc_mults[0];
        let expected_up = 2.0f32.powf(100.0 / 1200.0);
        // Left lane detunes flat, right lane sharp.
        assert!((mults.lane(0) - 1.0 / expected_up).abs() < 1e-3, "flat={}", mults.lane(0));
        assert!((mults.lane(1) - expected_up).abs() < 1e-3, "sharp={}", mults.lane(1));

        // Four-voice unison: outer pair reaches the full detune range.
        oscillator.unison = 4;
        oscillator.active_oscillators = 4;
        oscillator.set_phase_inc_mults(&params);
        let inner = oscillator.phase_inc_mults[0];
        let outer = oscillator.phase_inc_mults[1];
        let expected_inner = 2.0f32.powf(100.0 / 3.0 / 1200.0);
        assert!((inner.lane(1) - expected_inner).abs() < 1e-3);
        // The sharp/flat lane assignment alternates between pairs.
        assert!((outer.lane(0) - expected_up).abs() < 1e-3);
        assert!((outer.lane(1) - 1.0 / expected_up).abs() < 1e-3);
        // Detuned pairs stay symmetric around the center.
        assert!((inner.lane(0) * inner.lane(1) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn octave_stack_doubles_frequency() {
        let mut oscillator = SynthOscillator::new();
        oscillator.unison = 4;
        oscillator.active_oscillators = 4;
        let params = SynthOscillatorParams {
            unison_voices: 4,
            stack_style: UnisonStackType::Octave,
            ..Default::default()
        };
        oscillator.set_phase_inc_mults(&params);
        assert!((oscillator.phase_inc_mults[0].lane(0) - 1.0).abs() < 1e-4);
        assert!((oscillator.phase_inc_mults[1].lane(0) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn phase_distortion_neutral_bend_keeps_sine() {
        // Bend at 0.5 is the identity warp; the output must still be the
        // clean 440 Hz sine.
        let wavetable = sine_wavetable();
        let mut oscillator = SynthOscillator::new();
        oscillator.set_sample_rate(44100.0);
        let params = SynthOscillatorParams {
            midi_note: PolyF32::splat(69.0),
            distortion_type: DistortionType::Bend,
            distortion_amount: PolyF32::splat(0.5),
            ..Default::default()
        };
        oscillator.note_on(PolyMask::all_on(), PolyU32::ZERO);

        const BLOCK: usize = 128;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];
        let mut samples = Vec::new();
        for _ in 0..80 {
            oscillator.process(&params, &wavetable, None, BLOCK, &mut raw, &mut leveled);
            for value in &raw {
                samples.push(value.lane(0));
            }
        }
        // Compare against a pure sine of the same phase development: a
        // clean sinusoid satisfies x[n-1] + x[n+1] = 2 cos(w) x[n].
        let w = std::f32::consts::TAU * 440.0 / 44100.0;
        let k = 2.0 * w.cos();
        for i in 2000..6000 {
            let recurrence = (samples[i - 1] + samples[i + 1] - k * samples[i]).abs();
            assert!(recurrence < 5e-3, "sample {i} deviates: {recurrence}");
        }
    }

    #[test]
    fn levelled_output_applies_squared_amplitude() {
        let wavetable = sine_wavetable();
        let mut oscillator = SynthOscillator::new();
        oscillator.set_sample_rate(44100.0);
        let params = SynthOscillatorParams {
            midi_note: PolyF32::splat(69.0),
            amplitude: PolyF32::splat(0.5),
            ..Default::default()
        };
        oscillator.note_on(PolyMask::all_on(), PolyU32::ZERO);

        const BLOCK: usize = 128;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];
        for _ in 0..40 {
            oscillator.process(&params, &wavetable, None, BLOCK, &mut raw, &mut leveled);
        }
        // pan center => pan gain 1 per channel; leveled = raw * amp^2.
        for i in 0..BLOCK {
            let expected = raw[i].lane(0) * 0.25;
            assert!((leveled[i].lane(0) - expected).abs() < 2e-2);
        }
    }
}
