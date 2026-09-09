//! Display string tables for indexed parameters, mirroring
//! `src/interface/look_and_feel/synth_strings.h` from the C++ reference.

/// `strings::kOffOnNames`
pub static OFF_ON_NAMES: [&str; 2] = ["Off", "On"];

/// `strings::kOversamplingNames`
pub static OVERSAMPLING_NAMES: [&str; 4] = ["1x", "2x", "4x", "8x"];

/// `strings::kDelayStyleNames`
pub static DELAY_STYLE_NAMES: [&str; 4] = ["Mono", "Stereo", "Ping Pong", "Mid Ping Pong"];

/// `strings::kCompressorBandNames`
pub static COMPRESSOR_BAND_NAMES: [&str; 4] = ["Multiband", "Low Band", "High Band", "Single Band"];

/// `strings::kPresetStyleNames`
pub static PRESET_STYLE_NAMES: [&str; 9] = [
    "Bass",
    "Lead",
    "Keys",
    "Pad",
    "Percussion",
    "Sequence",
    "Experiment",
    "SFX",
    "Template",
];

/// `strings::kUnisonStackNames`
pub static UNISON_STACK_NAMES: [&str; 11] = [
    "Unison",
    "Center Drop 12",
    "Center Drop 24",
    "Octave",
    "2x Octave",
    "Power Chord",
    "2x Power Chord",
    "Major Chord",
    "Minor Chord",
    "Harmonics",
    "Odd Harmonics",
];

/// `strings::kFilterStyleNames`
pub static FILTER_STYLE_NAMES: [&str; 5] =
    ["12dB", "24dB", "Notch Blend", "Notch Spread", "B/P/N"];

/// `strings::kFrequencySyncNames`
pub static FREQUENCY_SYNC_NAMES: [&str; 5] =
    ["Seconds", "Tempo", "Tempo Dotted", "Tempo Triplets", "Keytrack"];

/// `strings::kDistortionTypeNames`
pub static DISTORTION_TYPE_NAMES: [&str; 6] = [
    "Soft Clip",
    "Hard Clip",
    "Linear Fold",
    "Sine Fold",
    "Bit Crush",
    "Down Sample",
];

/// `strings::kDistortionFilterOrderNames`
pub static DISTORTION_FILTER_ORDER_NAMES: [&str; 3] = ["None", "Pre", "Post"];

/// `strings::kFilterModelNames`
pub static FILTER_MODEL_NAMES: [&str; 8] = [
    "Analog", "Dirty", "Ladder", "Digital", "Diode", "Formant", "Comb", "Phaser",
];

/// `strings::kPredefinedWaveformNames`
pub static PREDEFINED_WAVEFORM_NAMES: [&str; 6] =
    ["Sin", "Saturated Sin", "Triangle", "Square", "Pulse", "Saw"];

/// `strings::kSyncedFrequencyNames`
pub static SYNCED_FREQUENCY_NAMES: [&str; 13] = [
    "Freeze", "32/1", "16/1", "8/1", "4/1", "2/1", "1/1", "1/2", "1/4", "1/8", "1/16", "1/32",
    "1/64",
];

/// `strings::kStereoModeNames`
pub static STEREO_MODE_NAMES: [&str; 2] = ["SPREAD", "ROTATE"];

/// `strings::kSmoothModeNames`
pub static SMOOTH_MODE_NAMES: [&str; 2] = ["FADE IN", "SMOOTH"];

/// `strings::kSyncNames` (LFO sync types)
pub static SYNC_NAMES: [&str; 6] = [
    "Trigger",
    "Sync",
    "Envelope",
    "Sustain Envelope",
    "Loop Point",
    "Loop Hold",
];

/// `strings::kRandomNames` (random LFO styles)
pub static RANDOM_NAMES: [&str; 4] = [
    "Perlin",
    "Sample & Hold",
    "Sine Interpolate",
    "Lorenz Attractor",
];

/// `strings::kVoicePriorityNames`
pub static VOICE_PRIORITY_NAMES: [&str; 5] = ["Newest", "Oldest", "Highest", "Lowest", "Round Robin"];

/// `strings::kVoiceOverrideNames`
pub static VOICE_OVERRIDE_NAMES: [&str; 2] = ["Kill", "Steal"];

/// `strings::kEqHighModeNames`
pub static EQ_HIGH_MODE_NAMES: [&str; 2] = ["Shelf", "Low Pass"];

/// `strings::kEqBandModeNames`
pub static EQ_BAND_MODE_NAMES: [&str; 2] = ["Shelf", "Notch"];

/// `strings::kEqLowModeNames`
pub static EQ_LOW_MODE_NAMES: [&str; 2] = ["Shelf", "High Pass"];

/// `strings::kDestinationNames` — the 5 routing destinations followed by the
/// 9 effects (`kNumSourceDestinations + kNumEffects` entries).
pub static DESTINATION_NAMES: [&str; 14] = [
    "FILTER 1",
    "FILTER 2",
    "FILTER 1+2",
    "EFFECTS",
    "DIRECT OUT",
    "CHORUS",
    "COMPRESSOR",
    "DELAY",
    "DISTORTION",
    "EQ",
    "FX FILTER",
    "FLANGER",
    "PHASER",
    "REVERB",
];

/// `strings::kPhaseDistortionNames`
pub static PHASE_DISTORTION_NAMES: [&str; 13] = [
    "None",
    "Sync",
    "Formant",
    "Quantize",
    "Bend",
    "Squeeze",
    "Pulse",
    "FM <- Osc",
    "FM <- Osc",
    "FM <- Sample",
    "RM <- Osc",
    "RM <- Osc",
    "RM <- Sample",
];

/// `strings::kSpectralMorphNames`
pub static SPECTRAL_MORPH_NAMES: [&str; 12] = [
    "None",
    "Vocode",
    "Formant Scale",
    "Harmonic Stretch",
    "Inharmonic Stretch",
    "Smear",
    "Random Amplitudes",
    "Low Pass",
    "High Pass",
    "Phase Disperse",
    "Shepard Tone",
    "Spectral Time Skew",
];

// -- Spinwave-only indexed parameters --------------------------------------

/// `osc_N_engine`: the oscillator slot engine (`OscEngineKind` order).
pub static OSC_ENGINE_NAMES: [&str; 4] = ["Wavetable", "Sample", "Granular", "Multisample"];

/// `osc_N_gran_window`: grain window (`GrainWindow` declaration order).
pub static GRAIN_WINDOW_NAMES: [&str; 5] =
    ["Hann", "Triangle", "Expo Decay", "Tukey", "Rectangular"];

/// `osc_N_gran_direction`: grain playback direction (`GrainDirection` order).
pub static GRAIN_DIRECTION_NAMES: [&str; 3] = ["Forward", "Reverse", "Bidirectional"];

/// `lfo_N_generator`: LFO value generator (`LfoGeneratorMode` order minus
/// the control-rate-only `Path` mode).
pub static LFO_GENERATOR_NAMES: [&str; 4] =
    ["Shape", "Sample & Hold", "Chaos Lorenz", "Chaos Rossler"];

/// `fx_split_<effect>`: per-effect signal split (`SplitMode` order).
pub static SPLIT_MODE_NAMES: [&str; 5] = ["Full", "Mid", "Side", "Low", "High"];

/// `bus_a_output` / `bus_b_output`: where a send bus returns (`BusOutput`).
pub static BUS_OUTPUT_NAMES: [&str; 2] = ["Master", "Main Chain"];

/// `noise_destination` as the Spinwave engine reads it (`ProducerDestination`
/// order: Vital's five routings plus the two send buses).
pub static PRODUCER_DESTINATION_NAMES: [&str; 7] =
    ["FILTER 1", "FILTER 2", "FILTER 1+2", "EFFECTS", "DIRECT OUT", "BUS A", "BUS B"];
