//! Shared filter parameter state, styles and cutoff-coefficient lookups
//! (rework of Vital's `synth_filter.{h,cpp}` plus the `OneDimLookup` table).

use std::sync::LazyLock;

use spinwave_poly::constants::{DB_GAIN_CONVERSION_MULT, MIDI_0_FREQUENCY, PI};
use spinwave_poly::utils::{catmull_interpolation_matrix, value_matrix};
use spinwave_poly::{math, PolyF32, PolyU32};

pub const MIN_DRIVE_GAIN: f32 = 0.0;
pub const MAX_DRIVE_GAIN: f32 = 36.0;

/// Filter response style shared by the main voice filters.
///
/// The discriminants match the C++ `SynthFilter::Style` indices; the comb
/// filter reinterprets the raw index through its own style mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum FilterStyle {
    #[default]
    TwelveDb = 0,
    TwentyFourDb = 1,
    NotchPassSwap = 2,
    DualNotchBand = 3,
    BandPeakNotch = 4,
    Shelving = 5,
}

pub const NUM_FILTER_STYLES: usize = 6;

impl FilterStyle {
    #[inline]
    pub fn from_index(index: i32) -> FilterStyle {
        match index {
            1 => FilterStyle::TwentyFourDb,
            2 => FilterStyle::NotchPassSwap,
            3 => FilterStyle::DualNotchBand,
            4 => FilterStyle::BandPeakNotch,
            5 => FilterStyle::Shelving,
            _ => FilterStyle::TwelveDb,
        }
    }

    #[inline]
    pub fn index(self) -> i32 {
        self as i32
    }
}

/// Per-block filter settings (rework of `SynthFilter::FilterState`).
///
/// Instead of reading processor inputs, fill the fields directly (or through
/// the helpers) and hand the struct to a filter's `setup` once per block.
#[derive(Clone, Copy, Debug)]
pub struct FilterState {
    /// MIDI note of the cutoff frequency.
    pub midi_cutoff: PolyF32,
    /// Resonance amount in `[0, 1]`.
    pub resonance_percent: PolyF32,
    /// Drive as a magnitude (see [`FilterState::set_drive_db`]).
    pub drive: PolyF32,
    /// Drive as a percent of the allowed dB range.
    pub drive_percent: PolyF32,
    /// Post gain in decibels (used by shelving styles).
    pub gain: PolyF32,
    /// Response style.
    pub style: FilterStyle,
    /// Low/band/high blend in `[0, 2]`.
    pub pass_blend: PolyF32,
    /// Formant X interpolation in `[0, 1]`.
    pub interpolate_x: PolyF32,
    /// Formant Y interpolation in `[0, 1]`.
    pub interpolate_y: PolyF32,
    /// Extra cutoff transpose in semitones (comb/formant).
    pub transpose: PolyF32,
}

impl Default for FilterState {
    fn default() -> FilterState {
        FilterState {
            midi_cutoff: PolyF32::splat(1.0),
            resonance_percent: PolyF32::ZERO,
            drive: PolyF32::ONE,
            drive_percent: PolyF32::ZERO,
            gain: PolyF32::ZERO,
            style: FilterStyle::TwelveDb,
            pass_blend: PolyF32::ZERO,
            interpolate_x: PolyF32::splat(0.5),
            interpolate_y: PolyF32::splat(0.5),
            transpose: PolyF32::ZERO,
        }
    }
}

impl FilterState {
    /// Sets `drive`/`drive_percent` from a drive gain in decibels, mirroring
    /// the clamping in the C++ `FilterState::loadSettings`.
    pub fn set_drive_db(&mut self, drive_gain_db: PolyF32) {
        let input_drive = drive_gain_db.clamp(MIN_DRIVE_GAIN, MAX_DRIVE_GAIN);
        self.drive_percent =
            (input_drive - MIN_DRIVE_GAIN) * (1.0 / (MAX_DRIVE_GAIN - MIN_DRIVE_GAIN));
        self.drive = math::db_to_magnitude(input_drive);
    }

    /// Sets the pass blend, clamped to `[0, 2]` like `loadSettings` does.
    pub fn set_pass_blend(&mut self, pass_blend: PolyF32) {
        self.pass_blend = pass_blend.clamp(0.0, 2.0);
    }
}

// ---------------------------------------------------------------------------
// Precise (per-block) conversion helpers, ports of the scalar `utils::` maps.
// The per-sample paths use the approximations in `spinwave_poly::math` instead.
// ---------------------------------------------------------------------------

/// Accurate MIDI note to frequency (C++ `utils::midiNoteToFrequency`).
#[inline]
pub fn midi_note_to_frequency_precise(note: PolyF32) -> PolyF32 {
    note.map(|n| MIDI_0_FREQUENCY * (n * (1.0 / 12.0)).exp2())
}

/// Accurate frequency to MIDI note (C++ `utils::frequencyToMidiNote`).
#[inline]
pub fn frequency_to_midi_note_precise(frequency: PolyF32) -> PolyF32 {
    frequency.map(|f| 12.0 * (f / MIDI_0_FREQUENCY).log2())
}

/// Accurate frequency-ratio to semitone offset (C++ `utils::ratioToMidiTranspose`).
#[inline]
pub fn ratio_to_midi_transpose_precise(ratio: PolyF32) -> PolyF32 {
    ratio.map(|r| 12.0 * r.log2())
}

/// Accurate decibels to magnitude (C++ `utils::dbToMagnitude`).
#[inline]
pub fn db_to_magnitude_precise(decibels: PolyF32) -> PolyF32 {
    decibels.map(|db| (db * (1.0 / DB_GAIN_CONVERSION_MULT)).exp2())
}

// ---------------------------------------------------------------------------
// Coefficient lookup tables (port of `OneDimLookup<function, 2048>`).
// ---------------------------------------------------------------------------

pub const COEFFICIENT_LOOKUP_RESOLUTION: usize = 2048;
const LOOKUP_EXTRA_VALUES: usize = 4;

/// Catmull-Rom interpolated lookup of a scalar coefficient function over
/// `[0, scale]`, sampled at 2048 points like the C++ `OneDimLookup`.
pub struct CoefficientLookup {
    table: [f32; COEFFICIENT_LOOKUP_RESOLUTION + LOOKUP_EXTRA_VALUES],
    scale: f32,
}

impl CoefficientLookup {
    pub fn new(function: impl Fn(f32) -> f32, scale: f32) -> CoefficientLookup {
        let resolution = COEFFICIENT_LOOKUP_RESOLUTION;
        let mut table = [0.0; COEFFICIENT_LOOKUP_RESOLUTION + LOOKUP_EXTRA_VALUES];
        for (i, entry) in table.iter_mut().enumerate() {
            let t = (i as f32 - 1.0) / (resolution as f32 - 1.0);
            *entry = function(t * scale);
        }
        CoefficientLookup { table, scale: resolution as f32 / scale }
    }

    #[inline(always)]
    pub fn cubic_lookup(&self, value: PolyF32) -> PolyF32 {
        let boost = value * self.scale;
        let indices = clamp_signed(boost.to_i32_round(), 0, COEFFICIENT_LOOKUP_RESOLUTION as i32);
        let t = boost - indices.to_f32_signed();

        let interpolation_matrix = catmull_interpolation_matrix(t);
        let mut values = value_matrix(&self.table, indices);
        values.transpose();

        interpolation_matrix.multiply_and_sum_rows(&values)
    }
}

#[inline(always)]
fn clamp_signed(value: PolyU32, min: i32, max: i32) -> PolyU32 {
    PolyU32::from_lanes([
        (value.lane(0) as i32).clamp(min, max) as u32,
        (value.lane(1) as i32).clamp(min, max) as u32,
        (value.lane(2) as i32).clamp(min, max) as u32,
        (value.lane(3) as i32).clamp(min, max) as u32,
    ])
}

/// One-pole coefficient curve (C++ `SynthFilter::computeOnePoleFilterCoefficient`).
pub fn compute_one_pole_filter_coefficient(frequency_ratio: f32) -> f32 {
    const MAX_RADS: f32 = 0.499 * PI;
    let scaled = frequency_ratio * PI;
    (scaled / (scaled + 1.0)).min(MAX_RADS).tan()
}

/// SVF coefficient curve (C++ `DigitalSvf::computeSvfOnePoleFilterCoefficient`).
pub fn compute_svf_one_pole_filter_coefficient(frequency_ratio: f32) -> f32 {
    const MAX_RATIO: f32 = 0.499;
    (frequency_ratio.min(MAX_RATIO) * PI).tan()
}

static COEFFICIENT_LOOKUP: LazyLock<CoefficientLookup> =
    LazyLock::new(|| CoefficientLookup::new(compute_one_pole_filter_coefficient, 1.0));

static SVF_COEFFICIENT_LOOKUP: LazyLock<CoefficientLookup> =
    LazyLock::new(|| CoefficientLookup::new(compute_svf_one_pole_filter_coefficient, 1.0));

/// Shared one-pole coefficient lookup (`SynthFilter::getCoefficientLookup`).
pub fn coefficient_lookup() -> &'static CoefficientLookup {
    &COEFFICIENT_LOOKUP
}

/// Shared SVF coefficient lookup (`DigitalSvf::getSvfCoefficientLookup`).
pub fn svf_coefficient_lookup() -> &'static CoefficientLookup {
    &SVF_COEFFICIENT_LOOKUP
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_matches_direct_function() {
        let lookup = svf_coefficient_lookup();
        for i in 0..100 {
            let ratio = i as f32 * 0.004;
            let direct = compute_svf_one_pole_filter_coefficient(ratio);
            let looked_up = lookup.cubic_lookup(PolyF32::splat(ratio)).lane(0);
            let tolerance = 5e-3 * (1.0 + direct.abs());
            assert!(
                (direct - looked_up).abs() < tolerance,
                "ratio {ratio}: direct {direct} vs lookup {looked_up}"
            );
        }
    }

    #[test]
    fn lookup_handles_bounds() {
        let lookup = coefficient_lookup();
        let low = lookup.cubic_lookup(PolyF32::ZERO);
        let high = lookup.cubic_lookup(PolyF32::ONE);
        assert!(low.is_finite());
        assert!(high.is_finite());
    }

    #[test]
    fn drive_db_conversion() {
        let mut state = FilterState::default();
        state.set_drive_db(PolyF32::splat(MAX_DRIVE_GAIN));
        assert!((state.drive_percent.lane(0) - 1.0).abs() < 1e-6);
        // 36 dB is a magnitude of ~63.1.
        assert!((state.drive.lane(0) - 63.1).abs() < 0.5);

        state.set_drive_db(PolyF32::splat(-10.0));
        assert_eq!(state.drive_percent.lane(0), 0.0);
        assert!((state.drive.lane(0) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn precise_midi_conversions_roundtrip() {
        let note = PolyF32::splat(69.0);
        let freq = midi_note_to_frequency_precise(note);
        assert!((freq.lane(0) - 440.0).abs() < 0.01);
        let back = frequency_to_midi_note_precise(freq);
        assert!((back.lane(0) - 69.0).abs() < 1e-4);
        let transpose = ratio_to_midi_transpose_precise(PolyF32::splat(2.0));
        assert!((transpose.lane(0) - 12.0).abs() < 1e-4);
    }
}
