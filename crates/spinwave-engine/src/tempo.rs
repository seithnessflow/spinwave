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
// TODO(fidelity): the reference also has a keytrack sync mode; bus effects
// never wire a MIDI input into it, so it is omitted here.
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
}

/// One tempo-syncable frequency control (port of the `TempoChooser`
/// resolution math from `operators.cpp`).
#[derive(Clone, Copy, Debug)]
pub struct SyncedFrequency {
    pub sync: SyncMode,
    /// Frequency in Hz, used in [`SyncMode::Frequency`].
    pub frequency_hz: f32,
    /// Index into [`SYNCED_FREQUENCY_RATIOS`], used in the tempo modes.
    pub tempo_index: f32,
}

impl SyncedFrequency {
    pub const fn free(frequency_hz: f32) -> SyncedFrequency {
        SyncedFrequency { sync: SyncMode::Frequency, frequency_hz, tempo_index: 8.0 }
    }

    /// Resolves to a frequency in Hz, exactly like `TempoChooser::process`.
    pub fn frequency_hz(&self, beats_per_second: f32) -> f32 {
        match self.sync {
            SyncMode::Frequency => self.frequency_hz,
            _ => {
                let tempo = self
                    .tempo_index
                    .clamp(0.0, (NUM_SYNCED_FREQUENCY_RATIOS - 1) as f32);
                let ratio = SYNCED_FREQUENCY_RATIOS[(tempo + 0.3) as usize];
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
/// LFO's own params; this only carries the mode and the ratio index.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LfoSync {
    pub mode: SyncMode,
    /// Index into [`SYNCED_FREQUENCY_RATIOS`], used in the tempo modes.
    pub tempo_index: f32,
}

impl Default for LfoSync {
    fn default() -> LfoSync {
        LfoSync { mode: SyncMode::Frequency, tempo_index: 8.0 }
    }
}

impl LfoSync {
    /// Resolves the effective LFO frequency: free mode keeps
    /// `free_frequency` (with any modulation offsets already applied); the
    /// tempo modes resolve the ratio table against the tempo, exactly like
    /// [`SyncedFrequency::frequency_hz`].
    pub fn resolve(&self, free_frequency: PolyF32, beats_per_second: f32) -> PolyF32 {
        match self.mode {
            SyncMode::Frequency => free_frequency,
            mode => PolyF32::splat(
                SyncedFrequency { sync: mode, frequency_hz: 0.0, tempo_index: self.tempo_index }
                    .frequency_hz(beats_per_second),
            ),
        }
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
        let synced = LfoSync { mode: SyncMode::Tempo, tempo_index: 6.0 };
        assert!((synced.resolve(PolyF32::splat(5.5), bps).lane(0) - 0.375).abs() < 1e-6);

        let dotted = LfoSync { mode: SyncMode::DottedTempo, tempo_index: 6.0 };
        assert!((dotted.resolve(PolyF32::ZERO, bps).lane(0) - 0.25).abs() < 1e-6);

        let triplet = LfoSync { mode: SyncMode::TripletTempo, tempo_index: 6.0 };
        assert!((triplet.resolve(PolyF32::ZERO, bps).lane(0) - 0.5625).abs() < 1e-6);
    }
}
