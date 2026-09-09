//! The complete Vital parameter table, mirroring
//! `src/common/synth_parameters.cpp` from the C++ reference.
//!
//! Like the C++ (`ValueDetailsLookup`), the global parameters are listed once
//! and the per-instance families (`osc_1_*`, `env_3_*`, `modulation_17_*`, ...)
//! are generated from group templates with 1-based ids.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::constants::{
    EFFECT_ORDER, MAX_MODULATION_CONNECTIONS, NUM_ENVELOPES, NUM_FILTERS, NUM_LFOS,
    NUM_OSCILLATORS, NUM_RANDOM_LFOS, SPINWAVE_MAX_ACTIVE_POLYPHONY, SPINWAVE_NUM_ENVELOPES,
    SPINWAVE_NUM_LFOS, SPINWAVE_NUM_MACROS, SPINWAVE_NUM_OSCILLATORS,
};
use crate::details::ParamDetails;
use crate::scale::ParamScale;
use crate::strings;

/// Static template for one parameter definition (borrowed strings; instances
/// in the table own their generated names).
struct ParamDef {
    name: &'static str,
    version_added: u32,
    min: f32,
    max: f32,
    default_value: f32,
    post_offset: f32,
    display_multiply: f32,
    scale: ParamScale,
    display_invert: bool,
    display_units: &'static str,
    display_name: &'static str,
    string_lookup: Option<&'static [&'static str]>,
}

/// Shorthand constructor keeping the field order of the C++ aggregate
/// initializers: name, version, min, max, default, post_offset,
/// display_multiply, scale, display_invert, units, display name, lookup.
#[allow(clippy::too_many_arguments)]
const fn p(
    name: &'static str,
    version_added: u32,
    min: f32,
    max: f32,
    default_value: f32,
    post_offset: f32,
    display_multiply: f32,
    scale: ParamScale,
    display_invert: bool,
    display_units: &'static str,
    display_name: &'static str,
    string_lookup: Option<&'static [&'static str]>,
) -> ParamDef {
    ParamDef {
        name,
        version_added,
        min,
        max,
        default_value,
        post_offset,
        display_multiply,
        scale,
        display_invert,
        display_units,
        display_name,
        string_lookup,
    }
}

use ParamScale::{Exponential, Indexed, Linear, Quadratic, Quartic, SquareRoot};

// Constants folded into the C++ table from other headers:
// Distortion::kMinDrive/kMaxDrive = -30/30, DigitalSvf::kMinGain/kMaxGain = -15/15,
// PredefinedWaveFrames::kNumShapes = 6, MultibandCompressor::kNumBandOptions = 4,
// SynthLfo::kNumSyncTypes = 6, SynthLfo::kNumSyncOptions = 5,
// RandomLfo::kNumStyles = 4, SynthOscillator::kNumUnisonStackTypes = 11,
// kNumDistortionTypes = 13, kNumSpectralMorphTypes = 12,
// VoiceHandler::kNumVoicePriorities = 5 (kRoundRobin = 4), kNumVoiceOverrides = 2
// (kKill = 0), kDegreesPerCycle = 360, factorial(kNumEffects) - 1 = 362879,
// kMaxPolyphony - 1 = 32, kNumOscillatorWaveFrames - 1 = 256,
// kNumOscillatorWaveFrames / 2 = 128 (integer division),
// kNumSourceDestinations + kNumEffects = 14, kNumFilterModels - 1 = 7.
const SQRT_2: f32 = std::f32::consts::SQRT_2; // 1.4142135624 in the C++ table.
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2; // 0.70710678119 in the C++ table.

/// `ValueDetailsLookup::parameter_list` — the global (non-grouped) parameters.
// Float literals are kept verbatim from the C++ table for auditability.
#[allow(clippy::excessive_precision)]
static PARAMETER_LIST: [ParamDef; 145] = [
    p("bypass", 0x000702, 0.0, 1.0, 0.0, 0.0, 60.0, Indexed, false, "", "Bypass", None),
    p("beats_per_minute", 0x000000, 0.333333333, 5.0, 2.0, 0.0, 60.0, Linear, false, "", "Beats Per Minute", None),
    p("delay_dry_wet", 0x000000, 0.0, 1.0, 0.3334, 0.0, 100.0, Linear, false, "%", "Delay Mix", None),
    p("delay_feedback", 0x000000, -1.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Delay Feedback", None),
    p("delay_frequency", 0x000000, -2.0, 9.0, 2.0, 0.0, 1.0, Exponential, true, " secs", "Delay Frequency", None),
    p("delay_aux_frequency", 0x000507, -2.0, 9.0, 2.0, 0.0, 1.0, Exponential, true, " secs", "Delay Frequency 2", None),
    p("delay_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Delay Switch", Some(&strings::OFF_ON_NAMES)),
    p("delay_style", 0x000000, 0.0, 3.0, 0.0, 0.0, 1.0, Indexed, false, "", "Delay Style", Some(&strings::DELAY_STYLE_NAMES)),
    p("delay_filter_cutoff", 0x000000, 8.0, 136.0, 60.0, 0.0, 1.0, Linear, false, "", "Delay Filter Cutoff", None),
    p("delay_filter_spread", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Linear, false, "", "Delay Filter Spread", None),
    p("delay_sync", 0x000000, 0.0, 3.0, 1.0, 0.0, 1.0, Indexed, false, "", "Delay Sync", Some(&strings::FREQUENCY_SYNC_NAMES)),
    p("delay_tempo", 0x000000, 4.0, 12.0, 9.0, 0.0, 1.0, Indexed, false, "", "Delay Tempo", Some(&strings::SYNCED_FREQUENCY_NAMES)),
    p("delay_aux_sync", 0x000507, 0.0, 3.0, 1.0, 0.0, 1.0, Indexed, false, "", "Delay Sync 2", Some(&strings::FREQUENCY_SYNC_NAMES)),
    p("delay_aux_tempo", 0x000507, 4.0, 12.0, 9.0, 0.0, 1.0, Indexed, false, "", "Delay Tempo 2", Some(&strings::SYNCED_FREQUENCY_NAMES)),
    p("distortion_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Distortion Switch", Some(&strings::OFF_ON_NAMES)),
    p("distortion_type", 0x000000, 0.0, 5.0, 0.0, 0.0, 1.0, Indexed, false, "", "Distortion Type", Some(&strings::DISTORTION_TYPE_NAMES)),
    p("distortion_drive", 0x000000, -30.0, 30.0, 0.0, 0.0, 1.0, Linear, false, " dB", "Distortion Drive", None),
    p("distortion_mix", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Linear, false, "", "Distortion Mix", None),
    p("distortion_filter_order", 0x000000, 0.0, 2.0, 0.0, 0.0, 1.0, Indexed, false, "", "Distortion Filter Order", Some(&strings::DISTORTION_FILTER_ORDER_NAMES)),
    p("distortion_filter_cutoff", 0x000000, 8.0, 136.0, 80.0, 0.0, 1.0, Linear, false, " semitones", "Distortion Filter Cutoff", None),
    p("distortion_filter_resonance", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Distortion Filter Resonance", None),
    p("distortion_filter_blend", 0x000000, 0.0, 2.0, 0.0, 0.0, 1.0, Linear, false, "", "Distortion Filter Blend", None),
    p("legato", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Legato", Some(&strings::OFF_ON_NAMES)),
    p("macro_control_1", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Macro 1", None),
    p("macro_control_2", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Macro 2", None),
    p("macro_control_3", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Macro 3", None),
    p("macro_control_4", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Macro 4", None),
    p("pitch_bend_range", 0x000000, 0.0, 48.0, 2.0, 0.0, 1.0, Indexed, false, " semitones", "Pitch Bend Range", None),
    // Vital caps this at 32; the Spinwave allocator goes to 64 (Vital clamps
    // larger values on load).
    p("polyphony", 0x000000, 1.0, SPINWAVE_MAX_ACTIVE_POLYPHONY as f32, 8.0, 0.0, 1.0, Indexed, false, " voices", "Polyphony", None),
    p("voice_tune", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, " cents", "Voice Tune", None),
    p("voice_transpose", 0x000604, -48.0, 48.0, 0.0, 0.0, 1.0, Indexed, false, "", "Voice Transpose", None),
    p("voice_amplitude", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Linear, false, "", "Voice Amplitude", None),
    p("stereo_routing", 0x000000, 0.0, 1.0, 1.0, 0.0, 100.0, Linear, false, "%", "Stereo Routing", None),
    p("stereo_mode", 0x000605, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Stereo Mode", Some(&strings::STEREO_MODE_NAMES)),
    p("portamento_time", 0x000000, -10.0, 4.0, -10.0, 0.0, 1.0, Exponential, false, " secs", "Portamento Time", None),
    p("portamento_slope", 0x000000, -8.0, 8.0, 0.0, 0.0, 1.0, Linear, false, "", "Portamento Slope", None),
    p("portamento_force", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Portamento Force", Some(&strings::OFF_ON_NAMES)),
    p("portamento_scale", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Portamento Scale", Some(&strings::OFF_ON_NAMES)),
    p("reverb_pre_low_cutoff", 0x000000, 0.0, 128.0, 0.0, 0.0, 1.0, Linear, false, " semitones", "Reverb Pre Low Cutoff", None),
    p("reverb_pre_high_cutoff", 0x000000, 0.0, 128.0, 110.0, 0.0, 1.0, Linear, false, " semitones", "Reverb Pre High Cutoff", None),
    p("reverb_low_shelf_cutoff", 0x000000, 0.0, 128.0, 0.0, 0.0, 1.0, Linear, false, " semitones", "Reverb Low Cutoff", None),
    p("reverb_low_shelf_gain", 0x000000, -6.0, 0.0, 0.0, 0.0, 1.0, Linear, false, " dB", "Reverb Low Gain", None),
    p("reverb_high_shelf_cutoff", 0x000000, 0.0, 128.0, 90.0, 0.0, 1.0, Linear, false, " semitones", "Reverb High Cutoff", None),
    p("reverb_high_shelf_gain", 0x000000, -6.0, 0.0, -1.0, 0.0, 1.0, Linear, false, " dB", "Reverb High Gain", None),
    p("reverb_dry_wet", 0x000000, 0.0, 1.0, 0.25, 0.0, 100.0, Linear, false, "%", "Reverb Mix", None),
    p("reverb_delay", 0x000609, 0.0, 0.3, 0.0, 0.0, 1.0, Linear, false, " secs", "Reverb Delay", None),
    p("reverb_decay_time", 0x000000, -6.0, 6.0, 0.0, 0.0, 1.0, Exponential, false, " secs", "Reverb Decay Time", None),
    p("reverb_size", 0x000506, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Reverb Size", None),
    p("reverb_chorus_amount", 0x000000, 0.0, 1.0, 0.223607, 0.0, 100.0, Quadratic, false, "%", "Reverb Chorus Amount", None),
    p("reverb_chorus_frequency", 0x000000, -8.0, 3.0, -2.0, 0.0, 1.0, Exponential, false, " Hz", "Reverb Chorus Frequency", None),
    p("reverb_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Reverb Switch", Some(&strings::OFF_ON_NAMES)),
    p("sub_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sub Switch", Some(&strings::OFF_ON_NAMES)),
    p("sub_direct_out", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sub Direct Out", None),
    p("sub_transpose", 0x000000, -48.0, 48.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sub Transpose", None),
    p("sub_transpose_quantize", 0x000000, 0.0, 8191.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sub Transpose Quantize", None),
    p("sub_tune", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "", "Sub Tune", None),
    p("sub_level", 0x000000, 0.0, 1.0, FRAC_1_SQRT_2, 0.0, 1.0, Quadratic, false, "", "Sub Level", None),
    p("sub_pan", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Sub Pan", None),
    p("sub_waveform", 0x000000, 0.0, 5.0, 2.0, 0.0, 1.0, Indexed, false, "", "Sub Osc Waveform", None),
    p("sample_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sample Switch", Some(&strings::OFF_ON_NAMES)),
    p("sample_random_phase", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sample Random Phase", Some(&strings::OFF_ON_NAMES)),
    p("sample_keytrack", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sample Keytrack", Some(&strings::OFF_ON_NAMES)),
    p("sample_loop", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Indexed, false, "", "Sample Loop", Some(&strings::OFF_ON_NAMES)),
    p("sample_bounce", 0x000603, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sample Bounce", Some(&strings::OFF_ON_NAMES)),
    p("sample_transpose", 0x000000, -48.0, 48.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sample Transpose", None),
    p("sample_transpose_quantize", 0x000000, 0.0, 8191.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sample Transpose Quantize", None),
    p("sample_tune", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "", "Sample Tune", None),
    p("sample_level", 0x000000, 0.0, 1.0, FRAC_1_SQRT_2, 0.0, 1.0, Quadratic, false, "", "Sample Level", None),
    p("sample_destination", 0x000500, 0.0, 14.0, 3.0, 0.0, 1.0, Indexed, false, "", "Sample Destination", Some(&strings::DESTINATION_NAMES)),
    p("sample_pan", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Sample Pan", None),
    p("velocity_track", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Velocity Track", None),
    p("volume", 0x000000, 0.0, 7399.4404, 5473.0404, -80.0, 1.0, SquareRoot, false, "dB", "Volume", None),
    p("phaser_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Phaser Switch", Some(&strings::OFF_ON_NAMES)),
    p("phaser_dry_wet", 0x000000, 0.0, 1.0, 1.0, 0.0, 100.0, Linear, false, "%", "Phaser Mix", None),
    p("phaser_feedback", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Phaser Feedback", None),
    p("phaser_frequency", 0x000000, -5.0, 2.0, -3.0, 0.0, 1.0, Exponential, true, " secs", "Phaser Frequency", None),
    p("phaser_sync", 0x000000, 0.0, 3.0, 1.0, 0.0, 1.0, Indexed, false, "", "Phaser Sync", Some(&strings::FREQUENCY_SYNC_NAMES)),
    p("phaser_tempo", 0x000000, 0.0, 10.0, 3.0, 0.0, 1.0, Indexed, false, "", "Phaser Tempo", Some(&strings::SYNCED_FREQUENCY_NAMES)),
    p("phaser_center", 0x000000, 8.0, 136.0, 80.0, 0.0, 1.0, Linear, false, " semitones", "Phaser Center", None),
    p("phaser_blend", 0x000509, 0.0, 2.0, 1.0, 0.0, 1.0, Linear, false, "", "Phaser Blend", None),
    p("phaser_mod_depth", 0x000000, 0.0, 48.0, 24.0, 0.0, 1.0, Linear, false, " semitones", "Phaser Mod Depth", None),
    p("phaser_phase_offset", 0x000000, 0.0, 1.0, 0.33333333, 0.0, 360.0, Linear, false, "", "Phaser Phase Offset", None),
    p("flanger_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Flanger Switch", Some(&strings::OFF_ON_NAMES)),
    p("flanger_dry_wet", 0x000000, 0.0, 0.5, 0.5, 0.0, 200.0, Linear, false, "%", "Flanger Mix", None),
    p("flanger_feedback", 0x000000, -1.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Flanger Feedback", None),
    p("flanger_frequency", 0x000000, -5.0, 2.0, 2.0, 0.0, 1.0, Exponential, true, " secs", "Flanger Frequency", None),
    p("flanger_sync", 0x000000, 0.0, 3.0, 1.0, 0.0, 1.0, Indexed, false, "", "Flanger Sync", Some(&strings::FREQUENCY_SYNC_NAMES)),
    p("flanger_tempo", 0x000000, 0.0, 10.0, 4.0, 0.0, 1.0, Indexed, false, "", "Flanger Tempo", Some(&strings::SYNCED_FREQUENCY_NAMES)),
    p("flanger_center", 0x000505, 8.0, 136.0, 64.0, 0.0, 1.0, Linear, false, " semitones", "Flanger Center", None),
    p("flanger_mod_depth", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Flanger Mod Depth", None),
    p("flanger_phase_offset", 0x000000, 0.0, 1.0, 0.33333333, 0.0, 360.0, Linear, false, "", "Flanger Phase Offset", None),
    p("chorus_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Chorus Switch", Some(&strings::OFF_ON_NAMES)),
    p("chorus_dry_wet", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Chorus Mix", None),
    p("chorus_feedback", 0x000000, -0.95, 0.95, 0.4, 0.0, 100.0, Linear, false, "%", "Chorus Feedback", None),
    p("chorus_cutoff", 0x000000, 8.0, 136.0, 60.0, 0.0, 1.0, Linear, false, "", "Chorus Filter Cutoff", None),
    p("chorus_spread", 0x000607, 0.0, 1.0, 1.0, 0.0, 1.0, Linear, false, "", "Chorus Filter Spread", None),
    p("chorus_voices", 0x000508, 1.0, 4.0, 4.0, 0.0, 4.0, Indexed, false, "", "Chorus Voices", None),
    p("chorus_frequency", 0x000000, -6.0, 3.0, -3.0, 0.0, 1.0, Exponential, true, " secs", "Chorus Frequency", None),
    p("chorus_sync", 0x000000, 0.0, 3.0, 1.0, 0.0, 1.0, Indexed, false, "", "Chorus Sync", Some(&strings::FREQUENCY_SYNC_NAMES)),
    p("chorus_tempo", 0x000000, 0.0, 10.0, 4.0, 0.0, 1.0, Indexed, false, "", "Chorus Tempo", Some(&strings::SYNCED_FREQUENCY_NAMES)),
    p("chorus_mod_depth", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Chorus Mod Depth", None),
    p("chorus_delay_1", 0x000000, -10.0, -5.64386, -9.0, 0.0, 1000.0, Exponential, false, "ms", "Chorus Delay 1", None),
    p("chorus_delay_2", 0x000000, -10.0, -5.64386, -7.0, 0.0, 1000.0, Exponential, false, " ms", "Chorus Delay 2", None),
    p("compressor_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Compressor Switch", Some(&strings::OFF_ON_NAMES)),
    p("compressor_low_upper_threshold", 0x000000, -80.0, 0.0, -28.0, 0.0, 1.0, Linear, false, " dB", "Low Upper Threshold", None),
    p("compressor_band_upper_threshold", 0x000000, -80.0, 0.0, -25.0, 0.0, 1.0, Linear, false, " dB", "Band Upper Threshold", None),
    p("compressor_high_upper_threshold", 0x000000, -80.0, 0.0, -30.0, 0.0, 1.0, Linear, false, " dB", "High Upper Threshold", None),
    p("compressor_low_lower_threshold", 0x000000, -80.0, 0.0, -35.0, 0.0, 1.0, Linear, false, " dB", "Low Lower Threshold", None),
    p("compressor_band_lower_threshold", 0x000000, -80.0, 0.0, -36.0, 0.0, 1.0, Linear, false, " dB", "Band Lower Threshold", None),
    p("compressor_high_lower_threshold", 0x000000, -80.0, 0.0, -35.0, 0.0, 1.0, Linear, false, " dB", "High Lower Threshold", None),
    p("compressor_low_upper_ratio", 0x000000, 0.0, 1.0, 0.9, 0.0, 1.0, Linear, false, "", "Low Upper Ratio", None),
    p("compressor_band_upper_ratio", 0x000000, 0.0, 1.0, 0.857, 0.0, 1.0, Linear, false, "", "Band Upper Ratio", None),
    p("compressor_high_upper_ratio", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Linear, false, "", "High Upper Ratio", None),
    p("compressor_low_lower_ratio", 0x000000, -1.0, 1.0, 0.8, 0.0, 1.0, Linear, false, "", "Low Lower Ratio", None),
    p("compressor_band_lower_ratio", 0x000000, -1.0, 1.0, 0.8, 0.0, 1.0, Linear, false, "", "Band Lower Ratio", None),
    p("compressor_high_lower_ratio", 0x000000, -1.0, 1.0, 0.8, 0.0, 1.0, Linear, false, "", "High Lower Ratio", None),
    p("compressor_low_gain", 0x000000, -30.0, 30.0, 16.3, 0.0, 1.0, Linear, false, " dB", "Compressor Low Gain", None),
    p("compressor_band_gain", 0x000000, -30.0, 30.0, 11.7, 0.0, 1.0, Linear, false, " dB", "Compressor Band Gain", None),
    p("compressor_high_gain", 0x000000, -30.0, 30.0, 16.3, 0.0, 1.0, Linear, false, " dB", "Compressor High Gain", None),
    p("compressor_attack", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Compressor Attack", None),
    p("compressor_release", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Compressor Release", None),
    p("compressor_enabled_bands", 0x000000, 0.0, 3.0, 0.0, 0.0, 1.0, Indexed, false, "", "Compressor Enabled Bands", Some(&strings::COMPRESSOR_BAND_NAMES)),
    p("compressor_mix", 0x000602, 0.0, 1.0, 1.0, 0.0, 1.0, Linear, false, "", "Compressor Mix", None),
    p("compressor_low_band_unused", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Indexed, false, "", "Compressor Unused", None),
    p("eq_on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "EQ Switch", Some(&strings::OFF_ON_NAMES)),
    p("eq_low_mode", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "EQ Low Mode", Some(&strings::EQ_LOW_MODE_NAMES)),
    p("eq_low_cutoff", 0x000000, 8.0, 136.0, 40.0, 0.0, 1.0, Linear, false, " semitones", "EQ Low Cutoff", None),
    p("eq_low_gain", 0x000000, -15.0, 15.0, 0.0, 0.0, 1.0, Linear, false, " dB", "EQ Low Gain", None),
    p("eq_low_resonance", 0x000000, 0.0, 1.0, 0.3163, 0.0, 100.0, Quadratic, false, "%", "EQ Low Resonance", None),
    p("eq_band_mode", 0x000506, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "EQ Band Mode", Some(&strings::EQ_BAND_MODE_NAMES)),
    p("eq_band_cutoff", 0x000000, 8.0, 136.0, 80.0, 0.0, 1.0, Linear, false, " semitones", "EQ Band Cutoff", None),
    p("eq_band_gain", 0x000000, -15.0, 15.0, 0.0, 0.0, 1.0, Linear, false, " dB", "EQ Band Gain", None),
    p("eq_band_resonance", 0x000000, 0.0, 1.0, 0.4473, 0.0, 100.0, Quadratic, false, "", "EQ Band Resonance", None),
    p("eq_high_mode", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "EQ High Mode", Some(&strings::EQ_HIGH_MODE_NAMES)),
    p("eq_high_cutoff", 0x000000, 8.0, 136.0, 100.0, 0.0, 1.0, Linear, false, " semitones", "EQ High Cutoff", None),
    p("eq_high_gain", 0x000000, -15.0, 15.0, 0.0, 0.0, 1.0, Linear, false, " dB", "EQ High Gain", None),
    p("eq_high_resonance", 0x000000, 0.0, 1.0, 0.3163, 0.0, 100.0, Quadratic, false, "", "EQ High Resonance", None),
    p("effect_chain_order", 0x000000, 0.0, 362879.0, 0.0, 0.0, 1.0, Indexed, false, "", "Effect Chain Order", None),
    p("voice_priority", 0x000000, 0.0, 4.0, 4.0, 0.0, 1.0, Indexed, false, "", "Voice Priority", Some(&strings::VOICE_PRIORITY_NAMES)),
    p("voice_override", 0x000700, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Voice Override", Some(&strings::VOICE_OVERRIDE_NAMES)),
    p("oversampling", 0x000000, 0.0, 3.0, 1.0, 0.0, 1.0, Indexed, false, "", "Oversampling", Some(&strings::OVERSAMPLING_NAMES)),
    p("pitch_wheel", 0x000400, -1.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Pitch Wheel", None),
    p("mod_wheel", 0x000400, 0.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Mod Wheel", None),
    p("mpe_enabled", 0x000501, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "MPE Enabled", Some(&strings::OFF_ON_NAMES)),
    p("view_spectrogram", 0x000803, 0.0, 2.0, 0.0, 0.0, 1.0, Indexed, false, "", "View Spectrogram", Some(&strings::OFF_ON_NAMES)),
];

/// `ValueDetailsLookup::env_parameter_list` — template for `env_N_*`.
static ENV_PARAMETER_LIST: [ParamDef; 9] = [
    p("delay", 0x000503, 0.0, SQRT_2, 0.0, 0.0, 1.0, Quartic, false, " secs", "Delay", None),
    p("attack", 0x000000, 0.0, 2.37842, 0.1495, 0.0, 1.0, Quartic, false, " secs", "Attack", None),
    p("hold", 0x000504, 0.0, SQRT_2, 0.0, 0.0, 1.0, Quartic, false, " secs", "Hold", None),
    p("decay", 0x000000, 0.0, 2.37842, 1.0, 0.0, 1.0, Quartic, false, " secs", "Decay", None),
    p("release", 0x000000, 0.0, 2.37842, 0.5476, 0.0, 1.0, Quartic, false, " secs", "Release", None),
    p("attack_power", 0x000000, -20.0, 20.0, 0.0, 0.0, 1.0, Linear, false, "", "Attack Power", None),
    p("decay_power", 0x000000, -20.0, 20.0, -2.0, 0.0, 1.0, Linear, false, "", "Decay Power", None),
    p("release_power", 0x000000, -20.0, 20.0, -2.0, 0.0, 1.0, Linear, false, "", "Release Power", None),
    p("sustain", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Linear, false, "", "Sustain", None),
];

/// `ValueDetailsLookup::lfo_parameter_list` — template for `lfo_N_*`.
static LFO_PARAMETER_LIST: [ParamDef; 12] = [
    p("phase", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Phase", None),
    p("sync_type", 0x000000, 0.0, 5.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sync Type", Some(&strings::SYNC_NAMES)),
    p("frequency", 0x000000, -7.0, 9.0, 1.0, 0.0, 1.0, Exponential, true, " secs", "Frequency", None),
    p("sync", 0x000000, 0.0, 4.0, 1.0, 0.0, 1.0, Indexed, false, "", "Sync", Some(&strings::FREQUENCY_SYNC_NAMES)),
    p("tempo", 0x000000, 0.0, 12.0, 7.0, 0.0, 1.0, Indexed, false, "", "Tempo", Some(&strings::SYNCED_FREQUENCY_NAMES)),
    p("fade_time", 0x000000, 0.0, 8.0, 0.0, 0.0, 1.0, Linear, false, " secs", "Fade In", None),
    p("smooth_mode", 0x000801, 0.0, 1.0, 1.0, 0.0, 1.0, Indexed, false, "", "Smooth Mode", Some(&strings::OFF_ON_NAMES)),
    p("smooth_time", 0x000801, -10.0, 4.0, -7.5, 0.0, 1.0, Exponential, false, " secs", "Smooth Time", None),
    p("delay_time", 0x000000, 0.0, 4.0, 0.0, 0.0, 1.0, Linear, false, " secs", "Delay", None),
    p("stereo", 0x000406, -0.5, 0.5, 0.0, 0.0, 1.0, Linear, false, "", "Stereo", None),
    p("keytrack_transpose", 0x000704, -60.0, 36.0, -12.0, 0.0, 1.0, Indexed, false, "", "Transpose", None),
    p("keytrack_tune", 0x000704, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "", "Tune", None),
];

/// `ValueDetailsLookup::random_lfo_parameter_list` — template for `random_N_*`.
static RANDOM_LFO_PARAMETER_LIST: [ParamDef; 8] = [
    p("style", 0x000401, 0.0, 3.0, 0.0, 0.0, 1.0, Indexed, false, "", "Style", Some(&strings::RANDOM_NAMES)),
    p("frequency", 0x000401, -7.0, 9.0, 1.0, 0.0, 1.0, Exponential, true, " secs", "Frequency", None),
    p("sync", 0x000401, 0.0, 4.0, 1.0, 0.0, 1.0, Indexed, false, "", "Sync", Some(&strings::FREQUENCY_SYNC_NAMES)),
    p("tempo", 0x000401, 0.0, 12.0, 8.0, 0.0, 1.0, Indexed, false, "", "Tempo", Some(&strings::SYNCED_FREQUENCY_NAMES)),
    p("stereo", 0x000401, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Stereo", Some(&strings::OFF_ON_NAMES)),
    p("sync_type", 0x000600, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sync Type", Some(&strings::OFF_ON_NAMES)),
    p("keytrack_transpose", 0x000704, -60.0, 36.0, -12.0, 0.0, 1.0, Indexed, false, "", "Transpose", None),
    p("keytrack_tune", 0x000704, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "", "Tune", None),
];

/// `ValueDetailsLookup::filter_parameter_list` — template for `filter_N_*` and
/// `filter_fx_*`.
static FILTER_PARAMETER_LIST: [ParamDef; 20] = [
    p("mix", 0x000000, 0.0, 1.0, 1.0, 0.0, 100.0, Linear, false, "%", "Mix", None),
    p("cutoff", 0x000000, 8.0, 136.0, 60.0, -60.0, 1.0, Linear, false, " semitones", "Cutoff", None),
    p("resonance", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Resonance", None),
    p("drive", 0x000000, 0.0, 20.0, 0.0, 0.0, 1.0, Linear, false, "dB", "Drive", None),
    p("blend", 0x000000, 0.0, 2.0, 0.0, 0.0, 1.0, Linear, false, "", "Blend", None),
    p("style", 0x000000, 0.0, 9.0, 0.0, 0.0, 1.0, Indexed, false, "", "Style", Some(&strings::FILTER_STYLE_NAMES)),
    p("model", 0x000000, 0.0, 7.0, 0.0, 0.0, 1.0, Indexed, false, "", "Model", Some(&strings::FILTER_MODEL_NAMES)),
    p("on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Switch", Some(&strings::OFF_ON_NAMES)),
    p("blend_transpose", 0x000000, 0.0, 84.0, 42.0, 0.0, 1.0, Linear, false, " semitones", "Comb Blend Offset", None),
    p("keytrack", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Key Track", None),
    p("formant_x", 0x000000, 0.0, 1.0, 0.5, 0.0, 1.0, Linear, false, "", "Formant X", None),
    p("formant_y", 0x000000, 0.0, 1.0, 0.5, 0.0, 1.0, Linear, false, "", "Formant Y", None),
    p("formant_transpose", 0x000000, -12.0, 12.0, 0.0, 0.0, 1.0, Linear, false, "", "Formant Transpose", None),
    p("formant_resonance", 0x000000, 0.3, 1.0, 0.85, 0.0, 100.0, Linear, false, "%", "Formant Resonance", None),
    p("formant_spread", 0x000707, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Formant Spread", None),
    p("osc1_input", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "OSC 1 Input", Some(&strings::OFF_ON_NAMES)),
    p("osc2_input", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "OSC 2 Input", Some(&strings::OFF_ON_NAMES)),
    p("osc3_input", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "OSC 3 Input", Some(&strings::OFF_ON_NAMES)),
    p("sample_input", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "SAMPLE Input", Some(&strings::OFF_ON_NAMES)),
    p("filter_input", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "FILTER Input", Some(&strings::OFF_ON_NAMES)),
];

/// `ValueDetailsLookup::osc_parameter_list` — template for `osc_N_*`.
// Float literals are kept verbatim from the C++ table for auditability.
#[allow(clippy::excessive_precision)]
static OSC_PARAMETER_LIST: [ParamDef; 29] = [
    p("on", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Switch", Some(&strings::OFF_ON_NAMES)),
    p("transpose", 0x000000, -48.0, 48.0, 0.0, 0.0, 1.0, Indexed, false, "", "Transpose", None),
    p("transpose_quantize", 0x000000, 0.0, 8191.0, 0.0, 0.0, 1.0, Indexed, false, "", "Transpose Quantize", None),
    p("tune", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "", "Tune", None),
    p("pan", 0x000000, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Pan", None),
    p("stack_style", 0x000000, 0.0, 10.0, 0.0, 0.0, 1.0, Indexed, false, "", "Stack Style", Some(&strings::UNISON_STACK_NAMES)),
    p("unison_detune", 0x000000, 0.0, 10.0, 4.472135955, 0.0, 1.0, Quadratic, false, "%", "Unison Detune", None),
    p("unison_voices", 0x000000, 1.0, 16.0, 1.0, 0.0, 1.0, Indexed, false, "v", "Unison Voices", None),
    p("unison_blend", 0x000000, 0.0, 1.0, 0.8, 0.0, 100.0, Linear, false, "%", "Blend", None),
    p("detune_power", 0x000000, -5.0, 5.0, 1.5, 0.0, 1.0, Linear, false, "", "Detune Power", None),
    p("detune_range", 0x000000, 0.0, 48.0, 2.0, 0.0, 1.0, Linear, false, "", "Detune Range", None),
    p("level", 0x000000, 0.0, 1.0, FRAC_1_SQRT_2, 0.0, 1.0, Quadratic, false, "", "Level", None),
    p("midi_track", 0x000000, 0.0, 1.0, 1.0, 0.0, 1.0, Indexed, false, "", "Midi Track", Some(&strings::OFF_ON_NAMES)),
    p("smooth_interpolation", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Smooth Interpolation", Some(&strings::OFF_ON_NAMES)),
    p("spectral_unison", 0x000500, 0.0, 1.0, 1.0, 0.0, 1.0, Indexed, false, "", "Spectral Unison", Some(&strings::OFF_ON_NAMES)),
    p("wave_frame", 0x000000, 0.0, 256.0, 0.0, 0.0, 1.0, Linear, false, "", "Wave Frame", None),
    p("frame_spread", 0x000000, -128.0, 128.0, 0.0, 0.0, 1.0, Linear, false, "", "Unison Frame Spread", None),
    p("stereo_spread", 0x000000, 0.0, 1.0, 1.0, 0.0, 100.0, Linear, false, "%", "Stereo Spread", None),
    p("phase", 0x000000, 0.0, 1.0, 0.5, 0.0, 360.0, Linear, false, "", "Phase", None),
    p("distortion_phase", 0x000000, 0.0, 1.0, 0.5, 0.0, 360.0, Linear, false, "", "Distortion Phase", None),
    p("random_phase", 0x000000, 0.0, 1.0, 1.0, 0.0, 100.0, Linear, false, "%", "Phase Randomization", None),
    p("distortion_type", 0x000000, 0.0, 12.0, 0.0, 0.0, 1.0, Indexed, false, "", "Distortion Type", Some(&strings::PHASE_DISTORTION_NAMES)),
    p("distortion_amount", 0x000000, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Distortion Amount", None),
    p("distortion_spread", 0x000000, -0.5, 0.5, 0.0, 0.0, 200.0, Linear, false, "%", "Distortion Spread", None),
    p("spectral_morph_type", 0x000407, 0.0, 11.0, 0.0, 0.0, 1.0, Indexed, false, "", "Frequency Morph Type", Some(&strings::SPECTRAL_MORPH_NAMES)),
    p("spectral_morph_amount", 0x000407, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Frequency Morph Amount", None),
    p("spectral_morph_spread", 0x000407, -0.5, 0.5, 0.0, 0.0, 200.0, Linear, false, "%", "Frequency Morph Spread", None),
    p("destination", 0x000500, 0.0, 14.0, 0.0, 0.0, 1.0, Indexed, false, "", "Destination", Some(&strings::DESTINATION_NAMES)),
    p("view_2d", 0x000402, 0.0, 2.0, 1.0, 0.0, 1.0, Indexed, false, "", "View 2D", Some(&strings::OFF_ON_NAMES)),
];

/// `ValueDetailsLookup::mod_parameter_list` — template for `modulation_N_*`.
static MOD_PARAMETER_LIST: [ParamDef; 5] = [
    p("amount", 0x000000, -1.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "Amount", None),
    p("power", 0x000000, -10.0, 10.0, 0.0, 0.0, 1.0, Linear, false, "", "Power", None),
    p("bipolar", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Bipolar", Some(&strings::OFF_ON_NAMES)),
    p("stereo", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Stereo", Some(&strings::OFF_ON_NAMES)),
    p("bypass", 0x000000, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Bypass", Some(&strings::OFF_ON_NAMES)),
];

// -- Spinwave-only namespace ---------------------------------------------------
//
// Parameters the Rust engine reads beyond Vital's table. Names and defaults
// match what `spinwave_plugin::patch` parses; ranges come from the DSP param
// structs (`GranularParams`, `SampleSourceParams`, `NoiseParams`,
// `SynthLfoParams`, `BusParams`, `EffectSplit`).

/// Version marker for every Spinwave-only parameter (sorts after Vital's).
const SPINWAVE_VERSION: u32 = 0x010000;

/// Per-oscillator Spinwave keys: engine selection, Sample engine
/// (`_smp_*`) and Granular engine (`_gran_*`) controls.
static SPINWAVE_OSC_PARAMETER_LIST: [ParamDef; 14] = [
    p("engine", SPINWAVE_VERSION, 0.0, 3.0, 0.0, 0.0, 1.0, Indexed, false, "", "Engine", Some(&strings::OSC_ENGINE_NAMES)),
    p("smp_rate", SPINWAVE_VERSION, 0.25, 4.0, 1.0, 0.0, 1.0, Linear, false, "x", "Sample Rate Mult", None),
    p("smp_loop", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Sample Loop", Some(&strings::OFF_ON_NAMES)),
    p("smp_slice", SPINWAVE_VERSION, -1.0, 4096.0, -1.0, 0.0, 1.0, Indexed, false, "", "Sample Slice", None),
    p("smp_offset", SPINWAVE_VERSION, 0.0, 1764000.0, 0.0, 0.0, 1.0, Indexed, false, " frames", "Sample Start Offset", None),
    p("gran_position", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Grain Position", None),
    p("gran_position_spray", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Grain Position Spray", None),
    p("gran_size", SPINWAVE_VERSION, 0.005, 2.0, 0.1, 0.0, 1000.0, Linear, false, " ms", "Grain Size", None),
    p("gran_size_spray", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Grain Size Spray", None),
    p("gran_density", SPINWAVE_VERSION, 0.1, 150.0, 30.0, 0.0, 1.0, Linear, false, " grains/s", "Grain Density", None),
    p("gran_pitch_spray", SPINWAVE_VERSION, 0.0, 48.0, 0.0, 0.0, 1.0, Linear, false, " semitones", "Grain Pitch Spray", None),
    p("gran_window", SPINWAVE_VERSION, 0.0, 4.0, 0.0, 0.0, 1.0, Indexed, false, "", "Grain Window", Some(&strings::GRAIN_WINDOW_NAMES)),
    p("gran_direction", SPINWAVE_VERSION, 0.0, 2.0, 0.0, 0.0, 1.0, Indexed, false, "", "Grain Direction", Some(&strings::GRAIN_DIRECTION_NAMES)),
    p("gran_stereo_spray", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Grain Stereo Spray", None),
];

/// Per-LFO Spinwave keys: value generator, sample & hold glide, chaos rate.
static SPINWAVE_LFO_PARAMETER_LIST: [ParamDef; 3] = [
    p("generator", SPINWAVE_VERSION, 0.0, 3.0, 0.0, 0.0, 1.0, Indexed, false, "", "Generator", Some(&strings::LFO_GENERATOR_NAMES)),
    p("sh_glide", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "S&H Glide", None),
    p("chaos_speed", SPINWAVE_VERSION, 0.01, 16.0, 1.0, 0.0, 1.0, Linear, false, "x", "Chaos Speed", None),
];

/// Dedicated noise source (`noise_*`).
static SPINWAVE_NOISE_PARAMETER_LIST: [ParamDef; 7] = [
    p("noise_on", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Noise Switch", Some(&strings::OFF_ON_NAMES)),
    p("noise_destination", SPINWAVE_VERSION, 0.0, 6.0, 0.0, 0.0, 1.0, Indexed, false, "", "Noise Destination", Some(&strings::PRODUCER_DESTINATION_NAMES)),
    p("noise_level", SPINWAVE_VERSION, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Noise Level", None),
    p("noise_pink", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Noise Pink", None),
    p("noise_tilt", SPINWAVE_VERSION, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Noise Tilt", None),
    p("noise_pan", SPINWAVE_VERSION, -1.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Noise Pan", None),
    p("noise_stereo", SPINWAVE_VERSION, 0.0, 1.0, 1.0, 0.0, 100.0, Linear, false, "%", "Noise Stereo", None),
];

/// The two Spinwave-only effect slots: a partitioned convolution reverb
/// and a Bode frequency shifter. Both sit in every chain (main, bus A,
/// bus B) alongside the nine reference effects.
static SPINWAVE_EFFECT_PARAMETER_LIST: [ParamDef; 8] = [
    p("convolution_on", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Convolution Switch", Some(&strings::OFF_ON_NAMES)),
    p("convolution_impulse", SPINWAVE_VERSION, 0.0, 2.0, 0.0, 0.0, 1.0, Indexed, false, "", "Convolution Impulse", Some(&strings::CONVOLUTION_IMPULSE_NAMES)),
    p("convolution_size", SPINWAVE_VERSION, 0.1, 10.0, 2.0, 0.0, 1.0, Linear, false, " s", "Convolution Size", None),
    p("convolution_dry_wet", SPINWAVE_VERSION, 0.0, 1.0, 0.5, 0.0, 100.0, Linear, false, "%", "Convolution Mix", None),
    p("convolution_predelay", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1000.0, Linear, false, " ms", "Convolution Predelay", None),
    p("convolution_gain", SPINWAVE_VERSION, -60.0, 24.0, 0.0, 0.0, 1.0, Linear, false, " dB", "Convolution Gain", None),
    p("frequency_shifter_on", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Frequency Shifter Switch", Some(&strings::OFF_ON_NAMES)),
    p("frequency_shifter_shift", SPINWAVE_VERSION, -5000.0, 5000.0, 0.0, 0.0, 1.0, Linear, false, " Hz", "Frequency Shifter Shift", None),
];

/// The frequency shifter's remaining two controls, kept out of the list
/// above only because the mix/stereo names would collide with a prefix
/// scan; they are registered the same way.
static SPINWAVE_SHIFTER_PARAMETER_LIST: [ParamDef; 2] = [
    p("frequency_shifter_mix", SPINWAVE_VERSION, 0.0, 1.0, 1.0, 0.0, 100.0, Linear, false, "%", "Frequency Shifter Mix", None),
    p("frequency_shifter_stereo", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Frequency Shifter Stereo", Some(&strings::OFF_ON_NAMES)),
];

/// Effects-mixer send bus (`bus_a_*` / `bus_b_*` mixer keys).
static SPINWAVE_BUS_PARAMETER_LIST: [ParamDef; 4] = [
    p("on", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Switch", Some(&strings::OFF_ON_NAMES)),
    p("send", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 100.0, Linear, false, "%", "Send", None),
    p("return_db", SPINWAVE_VERSION, -60.0, 12.0, 0.0, 0.0, 1.0, Linear, false, " dB", "Return", None),
    p("output", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Indexed, false, "", "Output", Some(&strings::BUS_OUTPUT_NAMES)),
];

/// Per-effect signal split (`fx_split_<effect>` / `fx_split_<effect>_crossover`).
static SPINWAVE_SPLIT_PARAMETER_LIST: [ParamDef; 2] = [
    p("", SPINWAVE_VERSION, 0.0, 4.0, 0.0, 0.0, 1.0, Indexed, false, "", "Split", Some(&strings::SPLIT_MODE_NAMES)),
    p("_crossover", SPINWAVE_VERSION, 20.0, 20000.0, 1000.0, 0.0, 1.0, Linear, false, " Hz", "Split Crossover", None),
];

/// Whether a global (non-grouped) Vital parameter belongs to the bus effect
/// chain (and therefore also exists as `bus_a_<name>` / `bus_b_<name>`).
fn is_effect_chain_parameter(name: &str) -> bool {
    name == "effect_chain_order"
        || EFFECT_ORDER.iter().any(|effect| name.starts_with(&format!("{effect}_")))
}

impl ParamDef {
    fn to_details(&self) -> ParamDetails {
        ParamDetails {
            name: self.name.to_string(),
            version_added: self.version_added,
            min: self.min,
            max: self.max,
            default_value: self.default_value,
            post_offset: self.post_offset,
            display_multiply: self.display_multiply,
            scale: self.scale,
            display_invert: self.display_invert,
            display_units: self.display_units.to_string(),
            display_name: self.display_name.to_string(),
            string_lookup: self.string_lookup,
            local_description: String::new(),
            spinwave_only: false,
        }
    }
}

/// The full parameter table (name lookup plus deterministic iteration order).
pub struct ParamTable {
    by_name: HashMap<String, ParamDetails>,
    /// Names sorted like the C++ `details_list_`: by `version_added`, then by
    /// name (byte-wise).
    ordered: Vec<String>,
}

impl ParamTable {
    fn build() -> Self {
        let mut table = ParamTable { by_name: HashMap::new(), ordered: Vec::new() };

        for def in &PARAMETER_LIST {
            let details = def.to_details();
            debug_assert!(details.default_value <= details.max);
            debug_assert!(details.default_value >= details.min);
            table.by_name.insert(details.name.clone(), details);
        }

        // Version markers from the C++ constructor: oscillator 3 and
        // modulations 33..64 were added later than their group templates.
        const NUM_OSCILLATORS_OLD: usize = 2;
        const NEW_OSCILLATOR_VERSION: u32 = 0x000500;
        const OLD_MAX_MODULATIONS: usize = 32;
        const NEW_MODULATION_VERSION: u32 = 0x000601;

        for env in 1..=NUM_ENVELOPES {
            table.add_group_indexed(&ENV_PARAMETER_LIST, env, "env", "Envelope", None);
        }
        for lfo in 1..=NUM_LFOS {
            table.add_group_indexed(&LFO_PARAMETER_LIST, lfo, "lfo", "LFO", None);
        }
        for lfo in 1..=NUM_RANDOM_LFOS {
            table.add_group_indexed(&RANDOM_LFO_PARAMETER_LIST, lfo, "random", "Random LFO", None);
        }
        for osc in 1..=NUM_OSCILLATORS_OLD {
            table.add_group_indexed(&OSC_PARAMETER_LIST, osc, "osc", "Oscillator", None);
        }
        for osc in (NUM_OSCILLATORS_OLD + 1)..=NUM_OSCILLATORS {
            table.add_group_indexed(
                &OSC_PARAMETER_LIST,
                osc,
                "osc",
                "Oscillator",
                Some(NEW_OSCILLATOR_VERSION),
            );
        }
        for filter in 1..=NUM_FILTERS {
            table.add_group_indexed(&FILTER_PARAMETER_LIST, filter, "filter", "Filter", None);
        }
        table.add_group(&FILTER_PARAMETER_LIST, "fx", "filter", "Filter", None);
        for modulation in 1..=OLD_MAX_MODULATIONS {
            table.add_group_indexed(&MOD_PARAMETER_LIST, modulation, "modulation", "Modulation", None);
        }
        for modulation in (OLD_MAX_MODULATIONS + 1)..=MAX_MODULATION_CONNECTIONS {
            table.add_group_indexed(
                &MOD_PARAMETER_LIST,
                modulation,
                "modulation",
                "Modulation",
                Some(NEW_MODULATION_VERSION),
            );
        }

        // Post-generation default overrides from the C++ constructor.
        for (name, default_value) in [
            ("osc_1_on", 1.0),
            ("osc_2_destination", 1.0),
            ("osc_3_destination", 3.0),
            ("filter_1_osc1_input", 1.0),
            ("filter_2_osc2_input", 1.0),
        ] {
            table
                .by_name
                .get_mut(name)
                .expect("override target must exist")
                .default_value = default_value;
        }

        table.add_spinwave_namespace();

        let mut ordered: Vec<String> = table.by_name.keys().cloned().collect();
        ordered.sort_by(|a, b| {
            let va = table.by_name[a].version_added;
            let vb = table.by_name[b].version_added;
            va.cmp(&vb).then_with(|| a.cmp(b))
        });
        table.ordered = ordered;
        table
    }

    /// `addParameterGroup` with a 1-based numeric id (the C++ passes a 0-based
    /// index and adds one when formatting).
    fn add_group_indexed(
        &mut self,
        defs: &[ParamDef],
        one_based_index: usize,
        id_prefix: &str,
        name_prefix: &str,
        version: Option<u32>,
    ) {
        self.add_group(defs, &one_based_index.to_string(), id_prefix, name_prefix, version);
    }

    /// `addParameterGroup` with a string id (used for the `filter_fx_` group).
    fn add_group(
        &mut self,
        defs: &[ParamDef],
        id: &str,
        id_prefix: &str,
        name_prefix: &str,
        version: Option<u32>,
    ) {
        for def in defs {
            let mut details = def.to_details();
            if let Some(version) = version {
                if version > details.version_added {
                    details.version_added = version;
                }
            }
            details.name = format!("{id_prefix}_{id}_{}", def.name);
            details.local_description = def.display_name.to_string();
            details.display_name = format!("{name_prefix} {id} {}", def.display_name);
            self.by_name.insert(details.name.clone(), details);
        }
    }

    /// Inserts one Spinwave-only parameter: `prefix + def.name`, flagged so
    /// the preset writer can strip it at its default.
    fn add_spinwave(&mut self, def: &ParamDef, prefix: &str, display_prefix: &str) {
        let mut details = def.to_details();
        details.name = format!("{prefix}{}", def.name);
        details.local_description = def.display_name.to_string();
        details.display_name = if display_prefix.is_empty() {
            def.display_name.to_string()
        } else {
            format!("{display_prefix} {}", def.display_name)
        };
        details.spinwave_only = true;
        details.version_added = details.version_added.max(SPINWAVE_VERSION);
        self.by_name.insert(details.name.clone(), details);
    }

    /// Everything the Rust engine reads beyond Vital's table. Extra slots
    /// (`osc_4`, `env_7..8`, `lfo_9..12`, `macro_control_5..8`) reuse their
    /// siblings' templates (same scales); the Spinwave namespace (engine
    /// selection, sample/granular, noise, LFO generators, send buses, effect
    /// splits and the `bus_a_` / `bus_b_` effect chains) gets its own
    /// definitions.
    fn add_spinwave_namespace(&mut self) {
        // Extra oscillator / envelope / LFO slots with Vital's templates.
        for osc in (NUM_OSCILLATORS + 1)..=SPINWAVE_NUM_OSCILLATORS {
            for def in &OSC_PARAMETER_LIST {
                self.add_spinwave(def, &format!("osc_{osc}_"), &format!("Oscillator {osc}"));
            }
        }
        for env in (NUM_ENVELOPES + 1)..=SPINWAVE_NUM_ENVELOPES {
            for def in &ENV_PARAMETER_LIST {
                self.add_spinwave(def, &format!("env_{env}_"), &format!("Envelope {env}"));
            }
        }
        for lfo in (NUM_LFOS + 1)..=SPINWAVE_NUM_LFOS {
            for def in &LFO_PARAMETER_LIST {
                self.add_spinwave(def, &format!("lfo_{lfo}_"), &format!("LFO {lfo}"));
            }
        }
        for index in (crate::constants::NUM_MACROS + 1)..=SPINWAVE_NUM_MACROS {
            let def = p("", SPINWAVE_VERSION, 0.0, 1.0, 0.0, 0.0, 1.0, Linear, false, "", "", None);
            let mut details = def.to_details();
            details.name = format!("macro_control_{index}");
            details.display_name = format!("Macro {index}");
            details.spinwave_only = true;
            self.by_name.insert(details.name.clone(), details);
        }

        // Spinwave namespace per oscillator / LFO.
        for osc in 1..=SPINWAVE_NUM_OSCILLATORS {
            for def in &SPINWAVE_OSC_PARAMETER_LIST {
                self.add_spinwave(def, &format!("osc_{osc}_"), &format!("Oscillator {osc}"));
            }
        }
        for lfo in 1..=SPINWAVE_NUM_LFOS {
            for def in &SPINWAVE_LFO_PARAMETER_LIST {
                self.add_spinwave(def, &format!("lfo_{lfo}_"), &format!("LFO {lfo}"));
            }
        }
        for def in &SPINWAVE_NOISE_PARAMETER_LIST {
            self.add_spinwave(def, "", "");
        }

        // The two Spinwave-only effects on the main chain.
        for def in SPINWAVE_EFFECT_PARAMETER_LIST.iter().chain(&SPINWAVE_SHIFTER_PARAMETER_LIST) {
            self.add_spinwave(def, "", "");
        }

        // Effect splits on the main chain, then the two bus chains: mixer
        // keys, a full copy of every effect parameter, and their splits.
        for effect in EFFECT_ORDER {
            for def in &SPINWAVE_SPLIT_PARAMETER_LIST {
                self.add_spinwave(def, &format!("fx_split_{effect}"), &title_case(effect));
            }
        }
        let effect_defs: Vec<&ParamDef> =
            PARAMETER_LIST.iter().filter(|def| is_effect_chain_parameter(def.name)).collect();
        for (bus, label) in [("bus_a_", "Bus A"), ("bus_b_", "Bus B")] {
            for def in &SPINWAVE_BUS_PARAMETER_LIST {
                self.add_spinwave(def, bus, label);
            }
            for def in &effect_defs {
                self.add_spinwave(def, bus, label);
            }
            for def in
                SPINWAVE_EFFECT_PARAMETER_LIST.iter().chain(&SPINWAVE_SHIFTER_PARAMETER_LIST)
            {
                self.add_spinwave(def, bus, label);
            }
            for def in &FILTER_PARAMETER_LIST {
                self.add_spinwave(def, &format!("{bus}filter_fx_"), &format!("{label} Filter fx"));
            }
            for effect in EFFECT_ORDER {
                for def in &SPINWAVE_SPLIT_PARAMETER_LIST {
                    self.add_spinwave(
                        def,
                        &format!("{bus}fx_split_{effect}"),
                        &format!("{label} {}", title_case(effect)),
                    );
                }
            }
        }
    }

    /// Whether `name` is a Spinwave-only parameter (absent from Vital).
    #[must_use]
    pub fn is_spinwave_only(&self, name: &str) -> bool {
        self.lookup(name).is_some_and(|d| d.spinwave_only)
    }

    /// Number of parameters Vital itself defines (`getNumParameters` in the
    /// reference: 794).
    #[must_use]
    pub fn len_vital(&self) -> usize {
        self.by_name.values().filter(|d| !d.spinwave_only).count()
    }

    /// Iterates Vital's own parameters only, in `iter` order.
    pub fn iter_vital(&self) -> impl Iterator<Item = &ParamDetails> {
        self.iter().filter(|d| !d.spinwave_only)
    }

    /// Looks up a parameter by machine name.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&ParamDetails> {
        self.by_name.get(name)
    }

    /// Whether `name` is a known parameter (`ValueDetailsLookup::isParameter`).
    #[must_use]
    pub fn is_parameter(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Number of parameters (`getNumParameters`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// `max - min` for the named parameter (`getParameterRange`).
    #[must_use]
    pub fn parameter_range(&self, name: &str) -> Option<f32> {
        self.lookup(name).map(|d| d.max - d.min)
    }

    /// Display name for the parameter (`getDisplayName`).
    #[must_use]
    pub fn display_name(&self, name: &str) -> Option<&str> {
        self.lookup(name).map(|d| d.display_name.as_str())
    }

    /// Iterates parameters in the C++ `details_list_` order: sorted by
    /// (`version_added`, name).
    pub fn iter(&self) -> impl Iterator<Item = &ParamDetails> {
        self.ordered.iter().map(move |name| &self.by_name[name])
    }
}

/// `"filter_fx"` → `"Filter Fx"`, `"eq"` → `"Eq"` (display prefixes).
fn title_case(id: &str) -> String {
    id.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

static TABLE: LazyLock<ParamTable> = LazyLock::new(ParamTable::build);

/// The global parameter table (built lazily on first access, like the C++
/// static `Parameters::lookup_`).
#[must_use]
pub fn parameters() -> &'static ParamTable {
    &TABLE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_parameter_count() {
        // 145 globals + 9*6 env + 12*8 lfo + 8*4 random + 29*3 osc
        // + 20*3 filter (1, 2, fx) + 5*64 modulation = 794.
        let expected = PARAMETER_LIST.len()
            + ENV_PARAMETER_LIST.len() * NUM_ENVELOPES
            + LFO_PARAMETER_LIST.len() * NUM_LFOS
            + RANDOM_LFO_PARAMETER_LIST.len() * NUM_RANDOM_LFOS
            + OSC_PARAMETER_LIST.len() * NUM_OSCILLATORS
            + FILTER_PARAMETER_LIST.len() * (NUM_FILTERS + 1)
            + MOD_PARAMETER_LIST.len() * MAX_MODULATION_CONNECTIONS;
        assert_eq!(expected, 794);
        assert_eq!(parameters().len_vital(), expected);
        assert_eq!(parameters().iter_vital().count(), expected);
        assert!(parameters().len() > expected);
    }

    #[test]
    fn spinwave_namespace_is_flagged() {
        let table = parameters();
        for name in [
            "osc_4_level",
            "osc_4_unison_detune",
            "env_7_attack",
            "env_8_release",
            "lfo_9_frequency",
            "lfo_12_smooth_time",
            "macro_control_5",
            "macro_control_8",
            "osc_1_engine",
            "osc_4_gran_density",
            "osc_2_smp_rate",
            "lfo_3_generator",
            "lfo_12_chaos_speed",
            "noise_on",
            "noise_destination",
            "bus_a_on",
            "bus_b_return_db",
            "bus_a_delay_on",
            "bus_b_reverb_dry_wet",
            "bus_a_filter_fx_cutoff",
            "bus_a_effect_chain_order",
            "fx_split_delay",
            "fx_split_reverb_crossover",
            "bus_b_fx_split_chorus",
        ] {
            let details = table.lookup(name).unwrap_or_else(|| panic!("missing {name}"));
            assert!(details.spinwave_only, "{name} must be spinwave_only");
            assert!(table.is_spinwave_only(name));
        }
        // Extra slots keep their siblings' scales and defaults.
        let detune = table.lookup("osc_4_unison_detune").unwrap();
        assert_eq!(detune.scale, ParamScale::Quadratic);
        assert_eq!(detune.default_value, table.lookup("osc_1_unison_detune").unwrap().default_value);
        assert_eq!(table.lookup("env_7_attack").unwrap().scale, ParamScale::Quartic);
        assert_eq!(table.lookup("lfo_11_smooth_time").unwrap().scale, ParamScale::Exponential);
        // Bus copies keep the main-chain definition.
        let bus_delay = table.lookup("bus_a_delay_frequency").unwrap();
        let main_delay = table.lookup("delay_frequency").unwrap();
        assert_eq!(bus_delay.scale, main_delay.scale);
        assert_eq!(bus_delay.default_value, main_delay.default_value);
        // Vital's own entries stay unflagged.
        assert!(!table.is_spinwave_only("osc_3_level"));
        assert!(!table.is_spinwave_only("delay_frequency"));
        assert!(!table.is_spinwave_only("macro_control_4"));
        assert_eq!(table.lookup("polyphony").unwrap().max, 64.0);
    }

    #[test]
    fn defaults_within_range() {
        for details in parameters().iter() {
            assert!(
                details.default_value >= details.min && details.default_value <= details.max,
                "{} default {} outside [{}, {}]",
                details.name,
                details.default_value,
                details.min,
                details.max
            );
        }
    }

    #[test]
    fn known_parameter_values() {
        let table = parameters();

        let osc_level = table.lookup("osc_1_level").unwrap();
        assert_eq!(osc_level.min, 0.0);
        assert_eq!(osc_level.max, 1.0);
        assert_eq!(osc_level.default_value, std::f32::consts::FRAC_1_SQRT_2);
        assert_eq!(osc_level.scale, ParamScale::Quadratic);
        assert_eq!(osc_level.display_name, "Oscillator 1 Level");
        assert_eq!(osc_level.local_description, "Level");

        let cutoff = table.lookup("filter_1_cutoff").unwrap();
        assert_eq!(cutoff.min, 8.0);
        assert_eq!(cutoff.max, 136.0);
        assert_eq!(cutoff.default_value, 60.0);
        assert_eq!(cutoff.post_offset, -60.0);
        assert_eq!(cutoff.scale, ParamScale::Linear);

        let attack = table.lookup("env_1_attack").unwrap();
        assert_eq!(attack.min, 0.0);
        assert_eq!(attack.max, 2.37842);
        assert_eq!(attack.default_value, 0.1495);
        assert_eq!(attack.scale, ParamScale::Quartic);
        assert_eq!(attack.display_units, " secs");

        let reverb_mix = table.lookup("reverb_dry_wet").unwrap();
        assert_eq!(reverb_mix.min, 0.0);
        assert_eq!(reverb_mix.max, 1.0);
        assert_eq!(reverb_mix.default_value, 0.25);
        assert_eq!(reverb_mix.display_multiply, 100.0);
        assert_eq!(reverb_mix.display_name, "Reverb Mix");

        let delay_tempo = table.lookup("delay_tempo").unwrap();
        assert_eq!(delay_tempo.min, 4.0);
        assert_eq!(delay_tempo.max, 12.0);
        assert_eq!(delay_tempo.default_value, 9.0);
        assert_eq!(delay_tempo.scale, ParamScale::Indexed);
        assert_eq!(delay_tempo.string_lookup.unwrap()[9], "1/8");

        let volume = table.lookup("volume").unwrap();
        assert_eq!(volume.max, 7399.4404);
        assert_eq!(volume.default_value, 5473.0404);
        assert_eq!(volume.post_offset, -80.0);
        assert_eq!(volume.scale, ParamScale::SquareRoot);

        let delay_frequency = table.lookup("delay_frequency").unwrap();
        assert_eq!(delay_frequency.scale, ParamScale::Exponential);
        assert!(delay_frequency.display_invert);
    }

    #[test]
    fn generated_name_families() {
        let table = parameters();
        for osc in 1..=3 {
            for local in ["on", "level", "wave_frame", "destination", "view_2d"] {
                let name = format!("osc_{osc}_{local}");
                assert!(table.is_parameter(&name), "missing {name}");
            }
        }
        assert!(table.is_parameter("osc_4_level"));
        assert!(!table.is_parameter("osc_5_level"));
        assert!(table.is_parameter("env_6_release"));
        assert!(table.is_parameter("env_8_release"));
        assert!(!table.is_parameter("env_9_release"));
        assert!(table.is_parameter("lfo_8_frequency"));
        assert!(table.is_parameter("lfo_12_frequency"));
        assert!(!table.is_parameter("lfo_13_frequency"));
        assert!(table.is_parameter("random_4_style"));
        assert!(table.is_parameter("filter_fx_cutoff"));
        assert_eq!(table.display_name("filter_fx_cutoff").unwrap(), "Filter fx Cutoff");
        assert!(table.is_parameter("modulation_64_amount"));
        assert!(!table.is_parameter("modulation_65_amount"));
    }

    #[test]
    fn group_version_overrides() {
        let table = parameters();
        // Oscillator 3 was added in 0.5.0; oscillators 1-2 keep template versions.
        assert_eq!(table.lookup("osc_1_level").unwrap().version_added, 0x000000);
        assert_eq!(table.lookup("osc_3_level").unwrap().version_added, 0x000500);
        // A template entry newer than the group version keeps its own version.
        assert_eq!(
            table.lookup("osc_3_spectral_morph_type").unwrap().version_added,
            0x000500
        );
        // Modulations 33+ were added in 0.6.1.
        assert_eq!(table.lookup("modulation_32_amount").unwrap().version_added, 0x000000);
        assert_eq!(table.lookup("modulation_33_amount").unwrap().version_added, 0x000601);
    }

    #[test]
    fn default_overrides() {
        let table = parameters();
        assert_eq!(table.lookup("osc_1_on").unwrap().default_value, 1.0);
        assert_eq!(table.lookup("osc_2_on").unwrap().default_value, 0.0);
        assert_eq!(table.lookup("osc_2_destination").unwrap().default_value, 1.0);
        assert_eq!(table.lookup("osc_3_destination").unwrap().default_value, 3.0);
        assert_eq!(table.lookup("filter_1_osc1_input").unwrap().default_value, 1.0);
        assert_eq!(table.lookup("filter_2_osc2_input").unwrap().default_value, 1.0);
        assert_eq!(table.lookup("filter_2_osc1_input").unwrap().default_value, 0.0);
    }

    #[test]
    fn ordered_iteration_is_sorted() {
        let table = parameters();
        let mut previous: Option<(&u32, &str)> = None;
        for details in table.iter() {
            if let Some((prev_version, prev_name)) = previous {
                assert!(
                    (*prev_version, prev_name) <= (details.version_added, details.name.as_str())
                );
            }
            previous = Some((&details.version_added, details.name.as_str()));
        }
    }

    #[test]
    fn normalization_examples() {
        let table = parameters();
        let cutoff = table.lookup("filter_1_cutoff").unwrap();
        assert_eq!(cutoff.to_engine(0.0), 8.0);
        assert_eq!(cutoff.to_engine(1.0), 136.0);
        let mid = cutoff.to_normalized(60.0);
        assert!((cutoff.to_engine(mid) - 60.0).abs() < 1e-4);

        let tempo = table.lookup("delay_tempo").unwrap();
        assert_eq!(tempo.to_engine(0.5), 8.0);
    }
}
