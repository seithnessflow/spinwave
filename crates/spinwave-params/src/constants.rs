//! Engine constants, mirroring `src/common/synth_constants.h` from the C++
//! reference (plus the modulation source list the engine builds at runtime).

/// `vital::kNumLfos`
pub const NUM_LFOS: usize = 8;
/// `vital::kNumOscillators`
pub const NUM_OSCILLATORS: usize = 3;
/// `vital::kNumOscillatorWaveFrames`
pub const NUM_OSCILLATOR_WAVE_FRAMES: usize = 257;
/// `vital::kNumEnvelopes`
pub const NUM_ENVELOPES: usize = 6;
/// `vital::kNumRandomLfos`
pub const NUM_RANDOM_LFOS: usize = 4;
/// `vital::kNumMacros`
pub const NUM_MACROS: usize = 4;
/// `vital::kNumFilters`
pub const NUM_FILTERS: usize = 2;
/// `vital::kNumFormants`
pub const NUM_FORMANTS: usize = 4;
/// `vital::kNumChannels`
pub const NUM_CHANNELS: usize = 2;
/// `vital::kMaxPolyphony`
pub const MAX_POLYPHONY: usize = 33;
/// `vital::kMaxActivePolyphony`
pub const MAX_ACTIVE_POLYPHONY: usize = 32;
/// `vital::kLfoDataResolution`
pub const LFO_DATA_RESOLUTION: usize = 2048;
/// `vital::kMaxModulationConnections`
pub const MAX_MODULATION_CONNECTIONS: usize = 64;

/// `vital::kPresetExtension`
pub const PRESET_EXTENSION: &str = "vital";
/// `vital::kWavetableExtension`
pub const WAVETABLE_EXTENSION: &str = "vitaltable";
/// `vital::kSkinExtension`
pub const SKIN_EXTENSION: &str = "vitalskin";
/// `vital::kLfoExtension`
pub const LFO_EXTENSION: &str = "vitallfo";
/// `vital::kBankExtension`
pub const BANK_EXTENSION: &str = "vitalbank";

/// Routing destinations before the effects, mirroring
/// `vital::constants::SourceDestination`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SourceDestination {
    Filter1 = 0,
    Filter2 = 1,
    DualFilters = 2,
    Effects = 3,
    DirectOut = 4,
}

/// `vital::constants::kNumSourceDestinations`
pub const NUM_SOURCE_DESTINATIONS: usize = 5;

/// Effects in enum order, mirroring `vital::constants::Effect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Effect {
    Chorus = 0,
    Compressor = 1,
    Delay = 2,
    Distortion = 3,
    Eq = 4,
    FilterFx = 5,
    Flanger = 6,
    Phaser = 7,
    Reverb = 8,
}

/// `vital::constants::kNumEffects`
pub const NUM_EFFECTS: usize = 9;

/// Effect id strings in enum order (`strings::kEffectOrder`); these are also
/// the prefixes of the effect parameters in the preset settings map.
pub static EFFECT_ORDER: [&str; NUM_EFFECTS] = [
    "chorus",
    "compressor",
    "delay",
    "distortion",
    "eq",
    "filter_fx",
    "flanger",
    "phaser",
    "reverb",
];

/// Filter models in enum order, mirroring `vital::constants::FilterModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FilterModel {
    Analog = 0,
    Dirty = 1,
    Ladder = 2,
    Digital = 3,
    Diode = 4,
    Formant = 5,
    Comb = 6,
    Phase = 7,
}

/// `vital::constants::kNumFilterModels`
pub const NUM_FILTER_MODELS: usize = 8;

/// `vital::constants::kNumSyncedFrequencyRatios`
pub const NUM_SYNCED_FREQUENCY_RATIOS: usize = 13;

/// `vital::constants::kSyncedFrequencyRatios` — beat-sync ratios matching
/// `strings::SYNCED_FREQUENCY_NAMES` ("Freeze", "32/1" ... "1/64").
pub static SYNCED_FREQUENCY_RATIOS: [f32; NUM_SYNCED_FREQUENCY_RATIOS] = [
    0.0, // Freeze
    0.0078125, // 1/128
    0.015625,  // 1/64
    0.03125,   // 1/32
    0.0625,    // 1/16
    0.125,     // 1/8
    0.25,      // 1/4
    0.5,       // 1/2
    1.0, 2.0, 4.0, 8.0, 16.0,
];

/// Spinwave engine limits (the Rust kernel is larger than Vital's: see
/// `spinwave_engine::kernel::mod_matrix`). Parameters generated for the
/// extra slots are flagged `spinwave_only` in the table so the preset
/// writer can keep `.vital` files loadable in Vital.
pub const SPINWAVE_NUM_OSCILLATORS: usize = 4;
/// Spinwave envelope count (Vital: 6).
pub const SPINWAVE_NUM_ENVELOPES: usize = 8;
/// Spinwave LFO count (Vital: 8).
pub const SPINWAVE_NUM_LFOS: usize = 12;
/// Spinwave macro count (Vital: 4).
pub const SPINWAVE_NUM_MACROS: usize = 8;
/// Spinwave maximum active polyphony (Vital: 32).
pub const SPINWAVE_MAX_ACTIVE_POLYPHONY: usize = 64;

/// Modulation source names the way the C++ engine registers them in
/// `SynthVoiceHandler` (`data_->mod_sources[...]`): the numbered families use
/// 1-based suffixes (`lfo_1` .. `lfo_8`, `env_1` .. `env_6`,
/// `random_1` .. `random_4`, `macro_control_1` .. `macro_control_4`) plus the
/// fixed per-voice/monophonic sources.
pub static VITAL_MODULATION_SOURCES: [&str; 32] = [
    "lfo_1", "lfo_2", "lfo_3", "lfo_4", "lfo_5", "lfo_6", "lfo_7", "lfo_8",
    "env_1", "env_2", "env_3", "env_4", "env_5", "env_6",
    "random_1", "random_2", "random_3", "random_4",
    "macro_control_1", "macro_control_2", "macro_control_3", "macro_control_4",
    "note", "note_in_octave", "aftertouch", "velocity", "slide", "lift", "mod_wheel",
    "pitch_wheel", "random", "stereo",
];

/// Every source the Spinwave engine exposes: Vital's list plus the extra
/// LFOs (`lfo_9..12`), envelopes (`env_7..8`) and macros
/// (`macro_control_5..8`).
pub static MODULATION_SOURCES: [&str; 42] = [
    "lfo_1", "lfo_2", "lfo_3", "lfo_4", "lfo_5", "lfo_6", "lfo_7", "lfo_8", "lfo_9", "lfo_10",
    "lfo_11", "lfo_12",
    "env_1", "env_2", "env_3", "env_4", "env_5", "env_6", "env_7", "env_8",
    "random_1", "random_2", "random_3", "random_4",
    "macro_control_1", "macro_control_2", "macro_control_3", "macro_control_4",
    "macro_control_5", "macro_control_6", "macro_control_7", "macro_control_8",
    "note", "note_in_octave", "aftertouch", "velocity", "slide", "lift", "mod_wheel",
    "pitch_wheel", "random", "stereo",
];

/// Modulation source names the Spinwave engine registers (Vital's plus the
/// Spinwave-only slots), as a static slice: no allocation per call.
#[must_use]
pub fn modulation_source_names() -> &'static [&'static str] {
    &MODULATION_SOURCES
}

/// Whether `name` is a modulation source the Spinwave engine registers.
#[must_use]
pub fn is_modulation_source(name: &str) -> bool {
    MODULATION_SOURCES.contains(&name)
}

/// Whether `name` is a modulation source Vital itself knows (a preset using
/// only these stays loadable in Vital).
#[must_use]
pub fn is_vital_modulation_source(name: &str) -> bool {
    VITAL_MODULATION_SOURCES.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulation_sources() {
        assert_eq!(
            VITAL_MODULATION_SOURCES.len(),
            NUM_LFOS + NUM_ENVELOPES + NUM_RANDOM_LFOS + NUM_MACROS + 10
        );
        assert_eq!(
            modulation_source_names().len(),
            SPINWAVE_NUM_LFOS + SPINWAVE_NUM_ENVELOPES + NUM_RANDOM_LFOS + SPINWAVE_NUM_MACROS + 10
        );
        assert!(is_modulation_source("lfo_8"));
        assert!(is_modulation_source("env_6"));
        assert!(is_modulation_source("random_4"));
        assert!(is_modulation_source("macro_control_1"));
        assert!(is_modulation_source("pitch_wheel"));
        // Spinwave-only slots are sources too, but not Vital ones.
        assert!(is_modulation_source("lfo_9"));
        assert!(is_modulation_source("env_8"));
        assert!(is_modulation_source("macro_control_8"));
        assert!(!is_vital_modulation_source("lfo_9"));
        assert!(is_vital_modulation_source("lfo_8"));
        assert!(!is_modulation_source("lfo_13"));
        assert!(!is_modulation_source("osc_1_level"));
    }

    #[test]
    fn synced_ratios_match_names() {
        assert_eq!(
            SYNCED_FREQUENCY_RATIOS.len(),
            crate::strings::SYNCED_FREQUENCY_NAMES.len()
        );
        assert_eq!(SYNCED_FREQUENCY_RATIOS[1], 1.0 / 128.0);
        assert_eq!(SYNCED_FREQUENCY_RATIOS[8], 1.0);
    }
}
