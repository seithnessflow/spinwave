//! Tempo sync: the beat-ratio table and sync-mode resolution shared by the
//! engine's bus effects and the kernel's LFOs (port of the `TempoChooser`
//! resolution math from `operators.cpp`).
//!
//! Lives outside [`crate::engine`] so the kernel can use it without an
//! engine ← kernel import cycle; the engine re-exports everything here for
//! compatibility.

use spinwave_poly::PolyF32;

/// Beat-sync ratios, copied from `spinwave_params::constants` (which mirrors
/// `vital::constants::kSyncedFrequencyRatios`) to keep this crate free of a
/// params dependency.
pub const NUM_SYNCED_FREQUENCY_RATIOS: usize = 13;
pub static SYNCED_FREQUENCY_RATIOS: [f32; NUM_SYNCED_FREQUENCY_RATIOS] = [
    0.0, // Freeze
    0.0078125, // 1/128
    0.015625,  // 1/64
    0.03125,   // 1/32
    0.0625,    // 1/16
    0.125,     // 1/8
    0.25,      // 1/4
    0.5,       // 1/2
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
];

/// Tempo sync mode, matching the reference `TempoChooser` modes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SyncMode {
    /// Free-running frequency in Hz.
    #[default]
    Frequency,
    /// Beat-synced: `ratio * beats_per_second`.
    Tempo,
    /// Dotted beat sync (`ratio * 2/3`).
    DottedTempo,
    /// Triplet beat sync (`ratio * 3/2`).
    TripletTempo,
    /// Keytracked: the frequency of `bent midi + keytrack_transpose +
    /// keytrack_tune` through the exact `utils::midiNoteToFrequency`
    /// (`TempoChooser::process`). LFOs and random LFOs only; the bus
    /// effects never wire a MIDI input into their chooser.
    Keytrack,
}

/// `cr::ExponentialScale`: the reference's scaling of every
/// Exponential-table control, modulated or not — the stored value plus
/// its modulation offsets, clamped to the table range, through the
/// POLYNOMIAL `futils::pow(2, x)` (`exp2(log2(2) · x)` with the
/// polynomial `log2`, whose `log2(2)` is 1 to a few ulp, not exactly).
#[inline]
pub fn exponential_scale(stored: f32, range: (f32, f32)) -> f32 {
    spinwave_poly::math::pow(PolyF32::splat(2.0), PolyF32::splat(stored.clamp(range.0, range.1))).lane(0)
}

/// One tempo-syncable frequency control (port of the `TempoChooser`
/// resolution math from `operators.cpp`).
#[derive(Clone, Copy, Debug)]
pub struct SyncedFrequency {
    pub sync: SyncMode,
    /// The stored value of `<name>_frequency`, log2 Hz, used in
    /// [`SyncMode::Frequency`] through [`exponential_scale`].
    pub frequency_log2: f32,
    /// The table range of that control (the `ExponentialScale` clamp).
    pub range: (f32, f32),
    /// Index into [`SYNCED_FREQUENCY_RATIOS`], used in the tempo modes.
    pub tempo_index: f32,
}

impl SyncedFrequency {
    pub fn free(frequency_hz: f32) -> SyncedFrequency {
        SyncedFrequency {
            sync: SyncMode::Frequency,
            frequency_log2: frequency_hz.log2(),
            range: (f32::MIN, f32::MAX),
            tempo_index: 8.0,
        }
    }

    /// Resolves to a frequency in Hz, exactly like `TempoChooser::process`.
    pub fn frequency_hz(&self, beats_per_second: f32) -> f32 {
        self.frequency_hz_with(beats_per_second, 0.0, 0.0)
    }

    /// The same with this block's modulation offsets on the frequency
    /// (log2 domain, `<name>_frequency`) and on the ratio index
    /// (`<name>_tempo`, added before the clamp and the `+ 0.3` truncation).
    pub fn frequency_hz_with(&self, beats_per_second: f32, log2_offset: f32, tempo_offset: f32) -> f32 {
        match self.sync {
            SyncMode::Frequency | SyncMode::Keytrack => {
                exponential_scale(self.frequency_log2 + log2_offset, self.range)
            }
            _ => {
                let tempo = (self.tempo_index + tempo_offset)
                    .clamp(0.0, (NUM_SYNCED_FREQUENCY_RATIOS - 1) as f32);
                // `utils::toInt(tempo + 0.3)`: `_mm_cvtps_epi32`, which ROUNDS
                // to nearest — not a truncation. Integer indices from the
                // table never showed it; a modulated index of 6.5 did
                // (mono_macro_to_chorus_tempo at 3.8e-1 with `as usize`).
                let index = (tempo + 0.3).round_ties_even() as usize;
                let ratio = SYNCED_FREQUENCY_RATIOS[index.min(NUM_SYNCED_FREQUENCY_RATIOS - 1)];
                let sync_mult = match self.sync {
                    SyncMode::DottedTempo => 2.0 / 3.0,
                    SyncMode::TripletTempo => 3.0 / 2.0,
                    _ => 1.0,
                };
                ratio * sync_mult * beats_per_second
            }
        }
    }
}

/// Per-LFO tempo sync selection. The free-running frequency lives in the
/// LFO's own params; this carries the mode, the ratio index and the
/// keytrack offsets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LfoSync {
    pub mode: SyncMode,
    /// Index into [`SYNCED_FREQUENCY_RATIOS`], used in the tempo modes.
    pub tempo_index: f32,
    /// `<name>_keytrack_transpose` (semitones) and `_keytrack_tune`
    /// (semitones, [-1, 1]), used in [`SyncMode::Keytrack`].
    pub keytrack_transpose: f32,
    pub keytrack_tune: f32,
}

impl Default for LfoSync {
    fn default() -> LfoSync {
        LfoSync { mode: SyncMode::Frequency, tempo_index: 8.0, keytrack_transpose: -12.0, keytrack_tune: 0.0 }
    }
}

impl LfoSync {
    /// Resolves the effective LFO frequency: free mode keeps
    /// `free_frequency` (with any modulation offsets already applied); the
    /// tempo modes resolve the ratio table against the tempo, exactly like
    /// [`SyncedFrequency::frequency_hz`].
    pub fn resolve(&self, free_frequency: PolyF32, beats_per_second: f32) -> PolyF32 {
        self.resolve_with(free_frequency, beats_per_second, PolyF32::ZERO, PolyF32::ZERO, PolyF32::ZERO)
    }

    /// The same with per-lane offsets on the ratio index (`lfo_N_tempo` as
    /// a modulation destination) and on the keytrack transpose, and the
    /// voice's bent MIDI for the keytrack mode.
    pub fn resolve_with(
        &self,
        free_frequency: PolyF32,
        beats_per_second: f32,
        tempo_offset: PolyF32,
        keytrack_transpose_offset: PolyF32,
        midi: PolyF32,
    ) -> PolyF32 {
        match self.mode {
            SyncMode::Frequency => free_frequency,
            SyncMode::Keytrack => {
                // `midiNoteToFrequency(keytrack_transpose + keytrack_tune +
                // midi)`, the exact one (operators.cpp:341).
                let note = (PolyF32::splat(self.keytrack_transpose) + keytrack_transpose_offset)
                    + PolyF32::splat(self.keytrack_tune)
                    + midi;
                spinwave_dsp::filters::filter_state::midi_note_to_frequency_precise(note)
            }
            mode => {
                let synced = SyncedFrequency { sync: mode, tempo_index: self.tempo_index, ..SyncedFrequency::free(1.0) };
                tempo_offset.map(|offset| synced.frequency_hz_with(beats_per_second, 0.0, offset))
            }
        }
    }
}

/// `ExponentialScale` on a per-lane sum: `stored + offset`, clamped to the
/// range, through the polynomial `pow(2, x)`.
pub trait AddThenExponentialScale {
    fn add_then_exponential_scale(self, offset: PolyF32, range: (f32, f32)) -> PolyF32;
}

impl AddThenExponentialScale for PolyF32 {
    fn add_then_exponential_scale(self, offset: PolyF32, range: (f32, f32)) -> PolyF32 {
        let clamped = (self + offset).clamp(range.0, range.1);
        spinwave_poly::math::pow(PolyF32::splat(2.0), clamped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfo_sync_resolves_expected_hz_at_bpm() {
        let bps = 90.0 / 60.0; // 90 bpm

        // Free mode passes the free frequency through untouched.
        let free = LfoSync::default();
        assert_eq!(free.resolve(PolyF32::splat(5.5), bps).lane(0), 5.5);

        // Index 6 is the 1/4 ratio: 0.25 * 1.5 bps = 0.375 Hz; the free
        // frequency is ignored in tempo modes.
        let synced = LfoSync { mode: SyncMode::Tempo, tempo_index: 6.0, ..LfoSync::default() };
        assert!((synced.resolve(PolyF32::splat(5.5), bps).lane(0) - 0.375).abs() < 1e-6);

        let dotted = LfoSync { mode: SyncMode::DottedTempo, tempo_index: 6.0, ..LfoSync::default() };
        assert!((dotted.resolve(PolyF32::ZERO, bps).lane(0) - 0.25).abs() < 1e-6);

        let triplet = LfoSync { mode: SyncMode::TripletTempo, tempo_index: 6.0, ..LfoSync::default() };
        assert!((triplet.resolve(PolyF32::ZERO, bps).lane(0) - 0.5625).abs() < 1e-6);
    }
}
