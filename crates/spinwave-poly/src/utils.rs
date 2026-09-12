//! Vector helpers shared across the engine (port of Vital's `poly_utils.h`).

use crate::constants::*;
use crate::matrix::Matrix;
use crate::simd::{PolyF32, PolyMask, PolyU32, LANES};

/// `[1, -1, 1, -1]`: multiplies left/right channels with opposite signs.
#[inline(always)]
pub fn stereo_split() -> PolyF32 {
    PolyF32::stereo(1.0, -1.0)
}

#[inline(always)]
pub fn interpolate(from: PolyF32, to: PolyF32, t: PolyF32) -> PolyF32 {
    from.mul_add(to - from, t)
}

/// Smoothstep-style interpolation used by the random LFO.
#[inline(always)]
pub fn perlin_interpolate(from: PolyF32, to: PolyF32, t: PolyF32) -> PolyF32 {
    let interpolate_from = from * t;
    let interpolate_to = to * (t - 1.0);
    let interpolate_t = t * t * (t * -2.0 + 3.0);
    interpolate(interpolate_from, interpolate_to, interpolate_t) * 2.0
}

/// Wraps a value one period down if it exceeds 1 (cheaper than fract).
#[inline(always)]
pub fn mod_once(value: PolyF32) -> PolyF32 {
    let less_mask = value.lt(PolyF32::ONE);
    less_mask.select(value, value - 1.0)
}

#[inline(always)]
pub fn close_to_zero_mask(value: PolyF32) -> PolyMask {
    value.abs().lt(PolyF32::splat(EPSILON))
}

/// All-lanes-silent mask over a buffer.
pub fn silent_mask(buffer: &[PolyF32]) -> PolyMask {
    let mut silent = PolyMask::all_on();
    for &value in buffer {
        silent &= close_to_zero_mask(value);
    }
    silent
}

// -- 4-point interpolation ---------------------------------------------------

/// Cubic Lagrange coefficients for the 4 taps around `t`.
#[inline(always)]
pub fn cubic_interpolation_values(t: f32) -> PolyF32 {
    let t = PolyF32::splat(t);
    let lagrange_one = PolyF32::from_lanes([0.0, 1.0, 0.0, 0.0]);
    let lagrange_two = PolyF32::from_lanes([-1.0, -1.0, 1.0, 1.0]);
    let lagrange_three = PolyF32::from_lanes([-2.0, -2.0, -2.0, -1.0]);
    let lagrange_mult =
        PolyF32::from_lanes([-1.0 / 6.0, 1.0 / 2.0, -1.0 / 2.0, 1.0 / 6.0]);
    lagrange_mult * (t + lagrange_one) * (t + lagrange_two) * (t + lagrange_three)
}

/// "Optimal" 4-tap coefficients (minimized aliasing for resampling).
#[inline(always)]
pub fn optimal_interpolation_values(t: f32) -> PolyF32 {
    let t = PolyF32::splat(t);
    let one = PolyF32::from_lanes([
        0.00224072707074864375,
        0.20184198969656244725,
        0.59244492420272312725,
        0.20345744715566445625,
    ]);
    let two = PolyF32::from_lanes([
        -0.0059513775678254975,
        -0.456633315206820491,
        -0.035736698832993691,
        0.4982319203618311775,
    ]);
    let three = PolyF32::from_lanes([
        0.093515484757265265,
        0.294278871937834749,
        -0.786648885977648931,
        0.398765058036740415,
    ]);
    let four = PolyF32::from_lanes([
        -0.10174985775982505,
        0.36030925263849456,
        -0.36030925263849456,
        0.10174985775982505,
    ]);
    ((four * t + three) * t + two) * t + one
}

/// Per-lane cubic Lagrange coefficient matrix, one row per tap.
#[inline(always)]
pub fn polynomial_interpolation_matrix(t_from: PolyF32) -> Matrix {
    const MULT_PREV: f32 = -1.0 / 6.0;
    const MULT_FROM: f32 = 1.0 / 2.0;
    const MULT_TO: f32 = -1.0 / 2.0;
    const MULT_NEXT: f32 = 1.0 / 6.0;

    let t_prev = t_from + 1.0;
    let t_to = t_from - 1.0;
    let t_next = t_from - 2.0;

    let t_prev_from = t_prev * t_from;
    let t_to_next = t_to * t_next;

    Matrix::new(
        t_from * t_to_next * MULT_PREV,
        t_prev * t_to_next * MULT_FROM,
        t_prev_from * t_next * MULT_TO,
        t_prev_from * t_to * MULT_NEXT,
    )
}

/// Catmull-Rom coefficient matrix.
#[inline(always)]
pub fn catmull_interpolation_matrix(t: PolyF32) -> Matrix {
    let half_t = t * 0.5;
    let half_t2 = t * half_t;
    let half_t3 = half_t2 * t;
    let half_three_t3 = half_t3 * 3.0;

    Matrix::new(
        half_t2 * 2.0 - half_t3 - half_t,
        half_three_t3.mul_sub(half_t2, PolyF32::splat(5.0)) + 1.0,
        half_t.mul_add(half_t2, PolyF32::splat(4.0)) - half_three_t3,
        half_t3 - half_t2,
    )
}

#[inline(always)]
pub fn linear_interpolation_matrix(t: PolyF32) -> Matrix {
    Matrix::new(PolyF32::ZERO, PolyF32::ONE - t, t, PolyF32::ZERO)
}

/// Loads 4 consecutive samples starting at each lane's index.
#[inline(always)]
pub fn value_matrix(buffer: &[f32], indices: PolyU32) -> Matrix {
    let row = |i: usize| {
        let start = indices.lane(i) as usize;
        PolyF32::from_lanes([
            buffer[start],
            buffer[start + 1],
            buffer[start + 2],
            buffer[start + 3],
        ])
    };
    Matrix::new(row(0), row(1), row(2), row(3))
}

/// Per-lane single-sample gather.
#[inline(always)]
pub fn gather(buffer: &[f32], indices: PolyU32) -> PolyF32 {
    PolyF32::from_lanes([
        buffer[indices.lane(0) as usize],
        buffer[indices.lane(1) as usize],
        buffer[indices.lane(2) as usize],
        buffer[indices.lane(3) as usize],
    ])
}

/// Per-lane gather of a sample and its successor.
#[inline(always)]
pub fn adjacent_gather(buffer: &[f32], indices: PolyU32) -> (PolyF32, PolyF32) {
    let mut value = [0.0; LANES];
    let mut next = [0.0; LANES];
    for i in 0..LANES {
        let index = indices.lane(i) as usize;
        value[i] = buffer[index];
        next[i] = buffer[index + 1];
    }
    (PolyF32::from_lanes(value), PolyF32::from_lanes(next))
}

// -- Stereo helpers ----------------------------------------------------------

/// Sums the two lanes of each voice and lays the per-voice totals side by
/// side, duplicated: `[L0+R0, L1+R1, L0+R0, L1+R1]` (Vital's
/// `sumSplitAudio`, used after a stereo split where lanes 0/1 hold one
/// signal and lanes 2/3 the other).
#[inline(always)]
pub fn sum_split_audio(sum: PolyF32) -> PolyF32 {
    let totals = sum + sum.swap_stereo();
    totals.swap_inner()
}

#[inline(always)]
pub fn max_lane(values: PolyF32) -> f32 {
    let max_voice = values.max(values.swap_voices());
    max_voice.max(max_voice.swap_stereo()).lane(0)
}

#[inline(always)]
pub fn min_lane(values: PolyF32) -> f32 {
    let min_voice = values.min(values.swap_voices());
    min_voice.min(min_voice.swap_stereo()).lane(0)
}

#[inline(always)]
pub fn encode_mid_side(value: PolyF32) -> PolyF32 {
    (value + stereo_split() * value.swap_stereo()) * 0.5
}

#[inline(always)]
pub fn decode_mid_side(value: PolyF32) -> PolyF32 {
    value + (stereo_split() * value).swap_stereo()
}

/// Peak magnitude over a buffer with an optional stride.
/// Per-lane absolute peak over every `skip`-th sample. `skip` must be at
/// least 1; 0 is treated as 1 (and asserted in debug builds) so the scan
/// can never loop forever.
pub fn peak(buffer: &[PolyF32], skip: usize) -> PolyF32 {
    debug_assert!(skip > 0, "peak(): skip must be >= 1");
    let skip = skip.max(1);
    let mut peak = PolyF32::ZERO;
    let mut i = 0;
    while i < buffer.len() {
        peak = peak.max(buffer[i]).max(-buffer[i]);
        i += skip;
    }
    peak
}

// -- Phase and pitch helpers -------------------------------------------------

#[inline(always)]
pub fn triangle_wave(t: PolyF32) -> PolyF32 {
    let range = (t + 0.75).fract();
    PolyF32::splat(-1.0).mul_add(range, PolyF32::splat(2.0)).abs()
}

pub fn cycle_offset_from_seconds(seconds: f64, frequency: PolyF32) -> PolyF32 {
    let mut offset = [0.0f32; LANES];
    for (i, lane) in offset.iter_mut().enumerate() {
        let cycles = frequency.lane(i) as f64 * seconds;
        *lane = (cycles - cycles.floor()) as f32;
    }
    PolyF32::from_lanes(offset)
}

pub fn cycle_offset_from_samples(
    samples: i64,
    frequency: PolyF32,
    sample_rate: u32,
    oversample_amount: u32,
) -> PolyF32 {
    let tick_time = oversample_amount as f64 / sample_rate as f64;
    cycle_offset_from_seconds(tick_time * samples as f64, frequency)
}

/// Snaps a transpose value to the enabled scale bits in `quantize` by the
/// nearest enabled note to the fractional value (Vital's
/// `utils::snapTranspose`, which its sample source uses). The wavetable
/// oscillator snaps differently: see [`SnapBuffer`].
pub fn snap_transpose(transpose: PolyF32, quantize: u32) -> PolyF32 {
    let notes = NOTES_PER_OCTAVE as f32;
    let octave_floored = (transpose * (1.0 / notes)).floor() * notes;
    let transpose_from_octave = transpose - octave_floored;
    let mut min_distance = PolyF32::splat(notes);
    let mut transpose_in_octave = transpose_from_octave;
    for i in 0..=NOTES_PER_OCTAVE {
        if (quantize >> (i % NOTES_PER_OCTAVE)) & 1 == 1 {
            let distance = (transpose_from_octave - i as f32).abs();
            let best_mask = distance.lt(min_distance);
            min_distance = best_mask.select(distance, min_distance);
            transpose_in_octave = best_mask.select(PolyF32::splat(i as f32), transpose_in_octave);
        }
    }
    octave_floored + transpose_in_octave
}

/// The wavetable oscillator's transpose snap (`fillSnapBuffer` plus
/// `localTransposeSnap` / `globalTransposeSnap` in Vital's
/// `synth_oscillator.cpp`). It is NOT [`snap_transpose`]: the value is
/// first rounded to the nearest integer note, then that note is looked up
/// in a table that maps each of the 13 notes of an octave (12 wraps to
/// the next octave's root) to its nearest enabled note — the distance is
/// measured from the rounded note, not from the fractional value, and a
/// tie goes to the note below. Filled once per block, looked up per
/// sample.
const SNAP_NOTES: usize = NOTES_PER_OCTAVE as usize + 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapBuffer([f32; SNAP_NOTES]);

impl SnapBuffer {
    /// Whether any note bit of `quantize` is set (`isTransposeSnapping`).
    pub fn snapping(quantize: u32) -> bool {
        quantize & ((1 << NOTES_PER_OCTAVE) - 1) != 0
    }

    /// Whether the snap applies to note + transpose rather than to the
    /// transpose alone (`isTransposeQuantizeGlobal`).
    pub fn global(quantize: u32) -> bool {
        quantize >> NOTES_PER_OCTAVE != 0
    }

    /// Literal port of `fillSnapBuffer`.
    pub fn new(quantize: u32) -> Self {
        let notes = NOTES_PER_OCTAVE as usize;
        let mut min_snap = 0.0f32;
        let mut max_snap = 0.0f32;
        for i in 0..notes {
            if (quantize >> i) & 1 == 1 {
                max_snap = i as f32;
                if min_snap == 0.0 {
                    min_snap = i as f32;
                }
            }
        }

        let mut buffer = [0.0f32; SNAP_NOTES];
        // First pass, upwards: the distance down to the previous enabled
        // note (wrapping from the top of the octave).
        let mut offset = notes as f32 - max_snap;
        for (i, slot) in buffer.iter_mut().enumerate() {
            if (quantize >> (i % notes)) & 1 == 1 {
                offset = 0.0;
            }
            *slot = offset;
            offset += 1.0;
        }
        // Second pass, downwards, `offset` now the distance up to the next
        // enabled note: pick the nearer neighbour, the one below on a tie.
        // (`min_snap` is left at 0 by an enabled root, so the seed is the
        // second enabled note — reproduced, since the first iteration only
        // uses it when note 12 is not enabled.)
        let mut offset = min_snap;
        for i in (0..=notes).rev() {
            let down = buffer[i];
            if offset < down {
                buffer[i] = i as f32 + offset;
            } else if down != 0.0 {
                buffer[i] = i as f32 - down;
            } else {
                buffer[i] = i as f32;
                offset = 0.0;
            }
            offset += 1.0;
        }
        Self(buffer)
    }

    fn lookup(&self, note_offset: PolyF32) -> PolyF32 {
        // `roundToInt` is `floorToInt(value + 0.5)`.
        let index = (note_offset + 0.5).to_i32_floor();
        let mut lanes = [0.0f32; LANES];
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = self.0[(index.lane(lane) as usize).min(SNAP_NOTES - 1)];
        }
        PolyF32::from_lanes(lanes)
    }

    /// `localTransposeSnap`: the transpose is snapped on its own, then
    /// added to the note.
    pub fn snap_local(&self, midi: PolyF32, transpose: PolyF32) -> PolyF32 {
        let notes = NOTES_PER_OCTAVE as f32;
        let note_offset = (transpose * (1.0 / notes)).fract() * notes;
        let octave_snap = transpose - note_offset;
        midi + octave_snap + self.lookup(note_offset)
    }

    /// `globalTransposeSnap`: note + transpose is snapped as one pitch.
    pub fn snap_global(&self, midi: PolyF32, transpose: PolyF32) -> PolyF32 {
        let notes = NOTES_PER_OCTAVE as f32;
        let total = midi + transpose;
        let note_offset = (total * (1.0 / notes)).fract() * notes;
        let octave_snap = total - note_offset;
        octave_snap + self.lookup(note_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_endpoints() {
        let a = PolyF32::splat(2.0);
        let b = PolyF32::splat(6.0);
        assert_eq!(interpolate(a, b, PolyF32::ZERO).lane(0), 2.0);
        assert_eq!(interpolate(a, b, PolyF32::ONE).lane(0), 6.0);
        assert_eq!(interpolate(a, b, PolyF32::splat(0.5)).lane(0), 4.0);
    }

    #[test]
    fn catmull_is_interpolating() {
        // At t=0 the Catmull-Rom matrix must select the second tap exactly.
        let m = catmull_interpolation_matrix(PolyF32::ZERO);
        let taps = Matrix::new(
            PolyF32::splat(10.0),
            PolyF32::splat(20.0),
            PolyF32::splat(30.0),
            PolyF32::splat(40.0),
        );
        assert!((m.multiply_and_sum_rows(&taps).lane(0) - 20.0).abs() < 1e-5);

        // At t=1 it must select the third tap.
        let m = catmull_interpolation_matrix(PolyF32::ONE);
        assert!((m.multiply_and_sum_rows(&taps).lane(0) - 30.0).abs() < 1e-4);
    }

    #[test]
    fn cubic_lagrange_sums_to_one() {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let sum = cubic_interpolation_values(t).sum_lanes();
            assert!((sum - 1.0).abs() < 1e-5, "t={t} sum={sum}");
        }
    }

    #[test]
    fn mid_side_roundtrip() {
        let value = PolyF32::from_lanes([0.8, -0.2, 0.1, 0.5]);
        let decoded = decode_mid_side(encode_mid_side(value));
        for i in 0..LANES {
            assert!((decoded.lane(i) - value.lane(i)).abs() < 1e-6);
        }
    }

    #[test]
    fn sum_split_audio_totals() {
        // Reference semantics: per-voice L+R totals, interleaved
        // [v0, v1, v0, v1].
        let v = PolyF32::from_lanes([1.0, 10.0, 2.0, 20.0]);
        let summed = sum_split_audio(v);
        assert_eq!(summed.to_lanes(), [11.0, 22.0, 11.0, 22.0]);
    }

    #[test]
    fn peak_scans_with_skip_and_survives_zero_skip() {
        let buffer = [
            PolyF32::from_lanes([0.5, -2.0, 0.0, 1.0]),
            PolyF32::from_lanes([-3.0, 0.1, 0.0, -1.5]),
            PolyF32::from_lanes([0.2, 0.2, 4.0, 0.0]),
        ];
        assert_eq!(peak(&buffer, 1).to_lanes(), [3.0, 2.0, 4.0, 1.5]);
        // skip = 2 only visits samples 0 and 2.
        assert_eq!(peak(&buffer, 2).to_lanes(), [0.5, 2.0, 4.0, 1.0]);
        // skip = 0 must terminate (treated as 1) instead of looping forever.
        if !cfg!(debug_assertions) {
            assert_eq!(peak(&buffer, 0).to_lanes(), [3.0, 2.0, 4.0, 1.5]);
        }
    }

    #[test]
    fn snap_transpose_to_octave() {
        // Only bit 0 set: snap to octaves (multiples of 12).
        let snapped = snap_transpose(PolyF32::splat(13.2), 1);
        assert_eq!(snapped.lane(0), 12.0);
    }

    #[test]
    fn oscillator_snap_rounds_then_looks_up() {
        // A major triad: bits 0, 4, 7. The table the reference builds for
        // it maps the 13 notes to [0 0 0 4 4 4 7 7 7 7 12 12 12].
        let triad = SnapBuffer::new(0b1001_0001);
        assert!(SnapBuffer::snapping(0b1001_0001));
        assert!(!SnapBuffer::global(0b1001_0001));
        assert!(SnapBuffer::global(0b1001_0001 | (1 << NOTES_PER_OCTAVE)));
        let snap = |t: f32| triad.snap_local(PolyF32::ZERO, PolyF32::splat(t)).lane(0);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-4;
        // Enabled notes stay put, across octaves (to float rounding: the
        // octave split multiplies by 1/12 and back, as the reference does).
        for t in [0.0, 4.0, 7.0, 12.0, 16.0, -5.0] {
            assert!(near(snap(t), t), "{t} -> {}", snap(t));
        }
        // Rounded first, then looked up: 2.4 and 1.6 both round to 2,
        // which is equidistant from 0 and 4, and the table sends a tie
        // DOWN. The fractional rule would have sent 2.4 to 4.
        assert!(near(snap(2.4), 0.0), "2.4 -> {}", snap(2.4));
        assert!(near(snap(1.6), 0.0), "1.6 -> {}", snap(1.6));
        assert!(near(snap(2.6), 4.0), "2.6 -> {}", snap(2.6));
        // 9 is two steps from 7 and three from 12: down. 10 is nearer 12.
        assert!(near(snap(9.0), 7.0), "9 -> {}", snap(9.0));
        assert!(near(snap(10.0), 12.0), "10 -> {}", snap(10.0));
        assert!(near(snap(11.2), 12.0), "11.2 -> {}", snap(11.2));
        // Global: the note takes part in the snap.
        let global = triad.snap_global(PolyF32::splat(45.0), PolyF32::splat(1.0)).lane(0);
        assert!(near(global, 48.0), "45 + 1 = 46 = 12*3 + 10 -> next root, 48; got {global}");
    }

    #[test]
    fn mod_once_wraps() {
        assert_eq!(mod_once(PolyF32::splat(1.25)).lane(0), 0.25);
        assert_eq!(mod_once(PolyF32::splat(0.75)).lane(0), 0.75);
    }
}
