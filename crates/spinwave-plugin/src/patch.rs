//! Loads a `.vital` preset into the Spinwave voice kernel.
//!
//! The preset's `settings` map stores engine values; most feed the DSP
//! param structs directly. Exceptions, mirroring the reference:
//! envelope times are stored as the quartic root of seconds, LFO and
//! random-LFO frequencies as log2(Hz), `Quadratic` parameters as the
//! square root of the engine value (`cr::Square` in `SynthModule`) and
//! `Exponential` ones as log2 of it.
//!
//! [`load_preset`] parses + migrates a preset and reports what could not
//! be mapped ([`LoadReport`]); [`BuiltPatch::build`] turns a preset into
//! every prebuilt structure the audio thread swaps in.

use std::sync::Arc;

use spinwave_dsp::effects::{
    ir_hall, ir_plate, ir_spring, BandOptions, ConvolutionReverb, DelayStyle,
    DistortionType as FxDistortionType,
};
use spinwave_dsp::modulators::line_generator::MAX_POINTS;
use spinwave_dsp::modulators::{LfoGeneratorMode, LineGenerator, RandomLfoStyle};
use spinwave_dsp::oscillator::{
    DistortionType, GrainDirection, GrainWindow, MultisampleSource, Sample, SpectralMorph,
    UnisonStackType,
};
use spinwave_dsp::wavetable::Wavetable;
use spinwave_engine::allocator::{VoiceOverride, VoicePriority, MAX_ACTIVE_POLYPHONY, PARALLEL_VOICES};
use spinwave_engine::effect_chain::DistortionFilterOrder;
use spinwave_engine::engine::{
    decode_order, BusOutput, BusParams, ChainId, Effect, EffectsConnection, EffectsModDest,
    EffectsParams,
    MixerParams, SplitMode, StereoMode, SyncMode, SyncedFrequency, DEFAULT_SPLIT_CROSSOVER_HZ,
};
use spinwave_engine::kernel::mod_matrix::{
    Connection, ModDest, ModSource, NUM_LFOS, NUM_MACROS, NUM_OSCILLATORS, NUM_RANDOM_LFOS,
};
use spinwave_engine::kernel::voice_filter::{FilterModel, VoiceFilterParams};
use spinwave_engine::kernel::{FilterRouting, KernelParams, OscEngineKind, ProducerDestination};
use spinwave_engine::modulation::{ModulationTransform, RemapCurve};
use spinwave_engine::tempo::LfoSync;
use spinwave_params::preset::{LineShape, LoadReport, Preset, SampleJson};
use spinwave_params::{parameters, ParamDetails};
use spinwave_poly::PolyF32;

use spinwave_engine::kernel::mod_matrix::NUM_ENVELOPES;

use crate::materials;

/// Upper bound on voice-pair kernels: enough prebuilt per-kernel structures
/// for every kernel at maximum polyphony.
pub const MAX_KERNELS: usize = spinwave_engine::allocator::MAX_POLYPHONY.div_ceil(PARALLEL_VOICES);

/// Parses a preset from JSON text (a UTF-8 BOM is tolerated), applies the
/// version migrations and reports the unknown keys. Connection-level
/// findings are added by [`BuiltPatch::build`] / [`connections_report`].
pub fn load_preset(text: &str) -> Result<(Preset, LoadReport), String> {
    let text = text.trim_start_matches('\u{feff}');
    let mut preset = Preset::from_json(text).map_err(|e| format!("invalid preset JSON: {e}"))?;
    let report = migrate_and_report(&mut preset);
    Ok((preset, report))
}

/// Same as [`load_preset`] for an already-parsed preset value.
pub fn load_preset_value(value: serde_json::Value) -> Result<(Preset, LoadReport), String> {
    let mut preset: Preset =
        serde_json::from_value(value).map_err(|e| format!("invalid preset: {e}"))?;
    let report = migrate_and_report(&mut preset);
    Ok((preset, report))
}

fn migrate_and_report(preset: &mut Preset) -> LoadReport {
    let mut report = LoadReport::default();
    let original_version = preset.synth_version.clone();
    let migrations = preset.upgrade();
    if !migrations.is_empty() {
        report.migrated_from = Some(original_version);
        report.migrations = migrations;
    }
    report.unknown_params = preset.unknown_parameters();
    connections_report(preset, &mut report);
    report
}

/// Adds the modulation connections the engine cannot route (unknown
/// source or destination) and the remap curves it cannot apply yet to a
/// report.
pub fn connections_report(preset: &Preset, report: &mut LoadReport) {
    for modulation in preset.settings.modulations.iter() {
        if !modulation.is_connected() {
            continue;
        }
        let source_ok = parse_mod_source(&modulation.source).is_some();
        let dest_ok = parse_mod_dest(&modulation.destination).is_some()
            || parse_effects_mod_dest(&modulation.destination).is_some();
        if !source_ok || !dest_ok {
            report
                .ignored_connections
                .push(format!("{} -> {}", modulation.source, modulation.destination));
        }
    }
}

/// Whether a `line_mapping` is the identity ramp Vital writes by default.
fn is_linear_shape(shape: &LineShape) -> bool {
    shape.num_points == 2
        && shape.points.len() >= 4
        && (shape.points[0]).abs() < 1e-6
        && (shape.points[1] - 1.0).abs() < 1e-6
        && (shape.points[2] - 1.0).abs() < 1e-6
        && (shape.points[3]).abs() < 1e-6
        && shape.powers.iter().all(|p| p.abs() < 1e-6)
}

/// Settings reader with table-backed defaults. A non-empty `prefix` (e.g.
/// `"bus_a_"`) is prepended to every key read from the preset, while table
/// defaults still resolve against the UNPREFIXED name — so a bus chain reads
/// `bus_a_delay_on` but falls back to `delay_on`'s table default.
struct Reader<'a> {
    preset: &'a Preset,
    prefix: &'a str,
}

thread_local! {
    /// The names `Reader::setting` was asked for, while recording.
    static READS: std::cell::RefCell<Option<std::collections::HashSet<String>>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f` and returns every parameter name the preset reader was asked
/// for meanwhile, on this thread. The direct observation behind the
/// read-parameter audit: a table parameter never asked for is one the
/// engine cannot receive — the formant filter's controls and the
/// distortion's filter were exactly that, for months, and no render-based
/// test could see it without the right context. Being *read* does not
/// depend on being audible, so this has no context false positives.
pub fn record_reads(f: impl FnOnce()) -> std::collections::HashSet<String> {
    READS.with(|reads| *reads.borrow_mut() = Some(std::collections::HashSet::new()));
    f();
    READS.with(|reads| reads.borrow_mut().take().unwrap_or_default())
}

impl<'a> Reader<'a> {
    fn new(preset: &'a Preset) -> Reader<'a> {
        Reader { preset, prefix: "" }
    }

    fn prefixed(preset: &'a Preset, prefix: &'a str) -> Reader<'a> {
        Reader { preset, prefix }
    }

    /// Raw preset value of `{prefix}{name}`, if set. Every read of a
    /// preset value goes through here, which is what lets
    /// [`record_reads`] prove which parameters the engine can receive.
    fn setting(&self, name: &str) -> Option<f32> {
        let full;
        let name = if self.prefix.is_empty() {
            name
        } else {
            full = format!("{}{name}", self.prefix);
            &full
        };
        READS.with(|reads| {
            if let Some(set) = reads.borrow_mut().as_mut() {
                set.insert(name.to_string());
            }
        });
        self.preset.settings.parameter(name)
    }

    fn get(&self, name: &str) -> f32 {
        if let Some(value) = self.setting(name) {
            return value;
        }
        parameters()
            .lookup(name)
            .map(|d: &ParamDetails| d.default_value)
            .unwrap_or(0.0)
    }

    /// Spinwave-namespace key (absent from the vital parameter table): the
    /// preset value if set, else the given default.
    fn raw(&self, name: &str, default: f32) -> f32 {
        self.setting(name).unwrap_or(default)
    }

    fn poly(&self, name: &str) -> PolyF32 {
        PolyF32::splat(self.get(name))
    }

    fn on(&self, name: &str) -> bool {
        self.get(name) > 0.5
    }
}

/// Reads `{group}_{slot+1}_{suffix}` with table-backed defaults; slots the
/// table doesn't know (osc_4, lfo_9..lfo_12) default from the group's slot-1
/// table entry instead.
fn get_slot(reader: &Reader, group: &str, slot: usize, suffix: &str) -> f32 {
    let name = format!("{group}_{}_{suffix}", slot + 1);
    if let Some(value) = reader.setting(&name) {
        return value;
    }
    let table = parameters();
    table
        .lookup(&name)
        .or_else(|| table.lookup(&format!("{group}_1_{suffix}")))
        .map(|d| d.default_value)
        .unwrap_or(0.0)
}

/// Reads `osc_{slot+1}_{suffix}`; slot 3 (osc_4) falls back to osc_1's
/// table defaults for absent keys.
fn get_osc(reader: &Reader, slot: usize, suffix: &str) -> f32 {
    get_slot(reader, "osc", slot, suffix)
}

/// Reads `lfo_{slot+1}_{suffix}`; slots 8..12 (lfo_9..lfo_12) fall back to
/// lfo_1's table defaults for absent keys.
fn get_lfo(reader: &Reader, slot: usize, suffix: &str) -> f32 {
    get_slot(reader, "lfo", slot, suffix)
}

fn distortion_type_from_index(index: i32) -> DistortionType {
    use DistortionType::*;
    [
        None, Sync, Formant, Quantize, Bend, Squeeze, PulseWidth, FmOscillatorA, FmOscillatorB,
        FmSample, RmOscillatorA, RmOscillatorB, RmSample,
    ]
    .get(index.max(0) as usize)
    .copied()
    .unwrap_or(None)
}

fn spectral_morph_from_index(index: i32) -> SpectralMorph {
    use SpectralMorph::*;
    [
        None, Vocode, FormScale, HarmonicScale, InharmonicScale, Smear, RandomAmplitudes,
        LowPass, HighPass, PhaseDisperse, ShepardTone, Skew,
    ]
    .get(index.max(0) as usize)
    .copied()
    .unwrap_or(None)
}

fn stack_type_from_index(index: i32) -> UnisonStackType {
    use UnisonStackType::*;
    [
        Normal, CenterDropOctave, CenterDropOctave2, Octave, Octave2, PowerChord, PowerChord2,
        MajorChord, MinorChord, HarmonicSeries, OddHarmonicSeries,
    ]
    .get(index.max(0) as usize)
    .copied()
    .unwrap_or(Normal)
}

/// `osc_N_engine` order: 0 Wavetable, 1 Sample, 2 Granular, 3 Multisample.
fn osc_engine_from_index(index: i32) -> OscEngineKind {
    use OscEngineKind::*;
    [Wavetable, Sample, Granular, Multisample]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Wavetable)
}

/// `osc_N_gran_window` order (the [`GrainWindow`] declaration order).
fn grain_window_from_index(index: i32) -> GrainWindow {
    use GrainWindow::*;
    [Hann, Triangle, ExpoDecay, Tukey, Rectangular]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Hann)
}

/// `osc_N_gran_direction` order (the [`GrainDirection`] declaration order).
fn grain_direction_from_index(index: i32) -> GrainDirection {
    use GrainDirection::*;
    [Forward, Reverse, Bidirectional]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Forward)
}

/// `lfo_N_generator` order: 0 Shape, 1 SampleHold, 2 Chaos1 (Lorenz),
/// 3 Chaos2 (Rossler). [`LfoGeneratorMode::Path`] is control-rate-only and
/// NOT preset-wired yet (it needs a second shape per LFO), so no index maps
/// to it.
fn lfo_generator_from_index(index: i32) -> LfoGeneratorMode {
    use LfoGeneratorMode::*;
    [Shape, SampleHold, Chaos1, Chaos2]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Shape)
}

/// `fx_split_<name>` order: 0 Full, 1 Mid, 2 Side, 3 Low, 4 High.
fn split_mode_from_index(index: i32) -> SplitMode {
    use SplitMode::*;
    [Full, Mid, Side, Low, High]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Full)
}

fn random_style_from_index(index: i32) -> RandomLfoStyle {
    use RandomLfoStyle::*;
    [Perlin, SampleAndHold, SinInterpolate, LorenzAttractor]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Perlin)
}

/// `strings::FREQUENCY_SYNC_NAMES` order. The bus effect sync parameters
/// only reach index 3; the LFO-only "Keytrack" mode (4) has no engine
/// equivalent and falls back to free-running.
fn sync_mode_from_index(index: i32) -> SyncMode {
    use SyncMode::*;
    [Frequency, Tempo, DottedTempo, TripletTempo]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Frequency)
}

/// `strings::DELAY_STYLE_NAMES` order (the remaining `DelayStyle` variants
/// are internal to the flanger/comb paths).
fn delay_style_from_index(index: i32) -> DelayStyle {
    use DelayStyle::*;
    [Mono, Stereo, PingPong, MidPingPong]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Mono)
}

/// `strings::COMPRESSOR_BAND_NAMES` order.
fn band_options_from_index(index: i32) -> BandOptions {
    use BandOptions::*;
    [Multiband, LowBand, HighBand, SingleBand]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(Multiband)
}

/// `strings::DISTORTION_TYPE_NAMES` order (bus distortion, not the
/// oscillator phase distortion).
fn fx_distortion_type_from_index(index: i32) -> FxDistortionType {
    use FxDistortionType::*;
    [SoftClip, HardClip, LinearFold, SinFold, BitCrush, DownSample]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(SoftClip)
}

/// `strings::VOICE_PRIORITY_NAMES` order.
fn voice_priority_from_index(index: i32) -> VoicePriority {
    use VoicePriority::*;
    [Newest, Oldest, Highest, Lowest, RoundRobin]
        .get(index.max(0) as usize)
        .copied()
        .unwrap_or(RoundRobin)
}

/// Quartic-root seconds → seconds.
#[inline]
fn env_seconds(stored: f32) -> PolyF32 {
    let squared = stored * stored;
    PolyF32::splat(squared * squared)
}

/// log2(Hz) → Hz.
#[inline]
fn exp_frequency(stored: f32) -> PolyF32 {
    PolyF32::splat(stored.exp2())
}

/// log2(seconds) → seconds (chorus delays, reverb decay time).
#[inline]
fn exp_seconds(stored: f32) -> PolyF32 {
    PolyF32::splat(stored.exp2())
}

/// Square-scaled parameter (table `Quadratic` scale: the setting stores the
/// square root of the engine value, like `cr::Quadratic` in the reference).
#[inline]
fn quadratic(stored: f32) -> PolyF32 {
    PolyF32::splat(stored * stored)
}

/// Builds a [`SyncedFrequency`] from `<prefix>_frequency` (stored as
/// log2 Hz), `<prefix>_sync` and `<prefix>_tempo`.
fn synced_frequency(reader: &Reader, prefix: &str) -> SyncedFrequency {
    SyncedFrequency {
        sync: sync_mode_from_index(reader.get(&format!("{prefix}_sync")) as i32),
        frequency_hz: reader.get(&format!("{prefix}_frequency")).exp2(),
        tempo_index: reader.get(&format!("{prefix}_tempo")),
    }
}

/// Fills a voice-filter param struct from `{prefix}on`, `{prefix}cutoff`, ...
/// — shared by the two voice filters and the `filter_fx_` bus effect (the
/// table generates the same group for all three).
fn fill_filter_params(reader: &Reader, prefix: &str, params: &mut VoiceFilterParams) {
    let p = |suffix: &str| format!("{prefix}{suffix}");
    params.on = reader.on(&p("on"));
    params.model = FilterModel::from_index(reader.get(&p("model")) as i32);
    params.mix = reader.poly(&p("mix"));
    let state = &mut params.state;
    state.midi_cutoff = reader.poly(&p("cutoff"));
    state.resonance_percent = reader.poly(&p("resonance"));
    state.set_drive_db(reader.poly(&p("drive")));
    state.set_pass_blend(reader.poly(&p("blend")));
    state.transpose = reader.poly(&p("blend_transpose"));
    state.style = spinwave_dsp::filters::FilterStyle::from_index(reader.get(&p("style")) as i32);
    // The formant model's five controls (`FormantModule` plugs them into
    // the formant filter's X, Y, transpose, resonance and spread inputs).
    // The DSP had the fields from the start; nothing filled them until the
    // read-parameter audit listed the names, months after the sensitivity
    // sweep had reported them inert.
    state.interpolate_x = reader.poly(&p("formant_x"));
    state.interpolate_y = reader.poly(&p("formant_y"));
    state.formant_transpose = reader.poly(&p("formant_transpose"));
    state.formant_resonance = reader.poly(&p("formant_resonance"));
    state.formant_spread = reader.poly(&p("formant_spread"));
}

fn lfo_sync_type_from_index(index: i32) -> spinwave_dsp::modulators::LfoSyncType {
    use spinwave_dsp::modulators::LfoSyncType::*;
    match index {
        1 => Sync,
        2 => Envelope,
        3 => SustainEnvelope,
        4 => LoopPoint,
        5 => LoopHold,
        _ => Trigger,
    }
}

/// Builds a line generator from a preset shape. `num_points` is clamped to
/// the generator's capacity (`MAX_POINTS`) and to the data actually
/// present; fewer than two usable points fall back to the triangle.
pub fn line_shape_to_generator(shape: &LineShape) -> LineGenerator {
    let mut generator = LineGenerator::new(2048);
    let num_points = (shape.num_points as usize).min(shape.points.len() / 2).min(MAX_POINTS);
    if num_points >= 2 {
        generator.set_num_points(num_points);
        for i in 0..num_points {
            generator.set_point(i, (shape.points[2 * i], shape.points[2 * i + 1]));
            if let Some(&power) = shape.powers.get(i) {
                generator.set_power(i, power);
            }
        }
    } else {
        generator.init_triangle();
    }
    generator.set_smooth(shape.smooth);
    generator.render();
    generator
}

/// Parses a modulation source name (`lfo_3`, `env_1`, `macro_control_5`,
/// `velocity`...). Numbered names are 1-based: `lfo_0` is not a source.
pub fn parse_mod_source(name: &str) -> Option<ModSource> {
    let indexed = |prefix: &str| -> Option<usize> {
        let n: usize = name.strip_prefix(prefix)?.parse().ok()?;
        n.checked_sub(1)
    };
    if let Some(i) = indexed("lfo_") {
        return (i < NUM_LFOS).then_some(ModSource::Lfo(i));
    }
    if let Some(i) = indexed("env_") {
        return (i < NUM_ENVELOPES).then_some(ModSource::Envelope(i));
    }
    if let Some(i) = indexed("random_") {
        return (i < NUM_RANDOM_LFOS).then_some(ModSource::RandomLfo(i));
    }
    if let Some(i) = indexed("macro_control_") {
        return (i < NUM_MACROS).then_some(ModSource::Macro(i));
    }
    match name {
        "note" => Some(ModSource::Note),
        "note_in_octave" => Some(ModSource::NoteInOctave),
        "velocity" => Some(ModSource::Velocity),
        "lift" => Some(ModSource::Lift),
        "mod_wheel" => Some(ModSource::ModWheel),
        "pitch_wheel" => Some(ModSource::PitchWheel),
        "aftertouch" => Some(ModSource::Aftertouch),
        "slide" => Some(ModSource::Slide),
        "random" => Some(ModSource::Random),
        "stereo" => Some(ModSource::Stereo),
        _ => Option::None,
    }
}

/// Parses a per-voice modulation destination name.
pub fn parse_mod_dest(name: &str) -> Option<ModDest> {
    let osc = |suffix: &str, make: fn(usize) -> ModDest| -> Option<ModDest> {
        for i in 0..NUM_OSCILLATORS {
            if name == format!("osc_{}_{}", i + 1, suffix) {
                return Some(make(i));
            }
        }
        Option::None
    };
    let filter = |suffix: &str, make: fn(usize) -> ModDest| -> Option<ModDest> {
        for i in 0..2 {
            if name == format!("filter_{}_{}", i + 1, suffix) {
                return Some(make(i));
            }
        }
        Option::None
    };
    let env = |suffix: &str, make: fn(usize) -> ModDest| -> Option<ModDest> {
        for i in 0..NUM_ENVELOPES {
            if name == format!("env_{}_{}", i + 1, suffix) {
                return Some(make(i));
            }
        }
        Option::None
    };
    let lfo = |suffix: &str, make: fn(usize) -> ModDest| -> Option<ModDest> {
        for i in 0..NUM_LFOS {
            if name == format!("lfo_{}_{}", i + 1, suffix) {
                return Some(make(i));
            }
        }
        Option::None
    };
    let random = |suffix: &str, make: fn(usize) -> ModDest| -> Option<ModDest> {
        for i in 0..NUM_RANDOM_LFOS {
            if name == format!("random_{}_{}", i + 1, suffix) {
                return Some(make(i));
            }
        }
        Option::None
    };

    osc("level", ModDest::OscLevel)
        .or_else(|| osc("transpose", ModDest::OscTranspose))
        .or_else(|| osc("tune", ModDest::OscTune))
        .or_else(|| osc("wave_frame", ModDest::OscFrame))
        .or_else(|| osc("frame_spread", ModDest::OscFrameSpread))
        .or_else(|| osc("pan", ModDest::OscPan))
        .or_else(|| osc("unison_detune", ModDest::OscUnisonDetune))
        .or_else(|| osc("unison_blend", ModDest::OscUnisonBlend))
        .or_else(|| osc("stereo_spread", ModDest::OscStereoSpread))
        .or_else(|| osc("distortion_amount", ModDest::OscDistortionAmount))
        .or_else(|| osc("distortion_phase", ModDest::OscDistortionPhase))
        .or_else(|| osc("spectral_morph_amount", ModDest::OscSpectralMorphAmount))
        .or_else(|| osc("phase", ModDest::OscPhase))
        .or_else(|| filter("cutoff", ModDest::FilterCutoff))
        .or_else(|| filter("resonance", ModDest::FilterResonance))
        .or_else(|| filter("drive", ModDest::FilterDrive))
        .or_else(|| filter("blend", ModDest::FilterBlend))
        .or_else(|| filter("blend_transpose", ModDest::FilterBlendTranspose))
        .or_else(|| filter("keytrack", ModDest::FilterKeytrack))
        .or_else(|| filter("mix", ModDest::FilterMix))
        .or_else(|| env("delay", ModDest::EnvDelay))
        .or_else(|| env("attack", ModDest::EnvAttack))
        .or_else(|| env("attack_power", ModDest::EnvAttackPower))
        .or_else(|| env("hold", ModDest::EnvHold))
        .or_else(|| env("decay", ModDest::EnvDecay))
        .or_else(|| env("decay_power", ModDest::EnvDecayPower))
        .or_else(|| env("sustain", ModDest::EnvSustain))
        .or_else(|| env("release", ModDest::EnvRelease))
        .or_else(|| env("release_power", ModDest::EnvReleasePower))
        .or_else(|| lfo("frequency", ModDest::LfoFrequency))
        .or_else(|| lfo("phase", ModDest::LfoPhase))
        .or_else(|| random("frequency", ModDest::RandomLfoFrequency))
        .or(match name {
            "sample_level" => Some(ModDest::SampleLevel),
            "sample_transpose" => Some(ModDest::SampleTranspose),
            "sample_tune" => Some(ModDest::SampleTune),
            "sample_pan" => Some(ModDest::SamplePan),
            "volume" => Some(ModDest::VolumeAmp),
            "pitch_wheel" => Some(ModDest::PitchBend),
            _ => Option::None,
        })
}

/// Bus-effect (mono) modulation destinations, matched against the same
/// preset destination names as the parameter table.
pub fn parse_effects_mod_dest(name: &str) -> Option<EffectsModDest> {
    use EffectsModDest::*;
    match name {
        "delay_feedback" => Some(DelayFeedback),
        "delay_dry_wet" => Some(DelayDryWet),
        "delay_frequency" => Some(DelayFrequency),
        "delay_aux_frequency" => Some(DelayAuxFrequency),
        "reverb_dry_wet" => Some(ReverbDryWet),
        "reverb_decay_time" => Some(ReverbDecayTime),
        "reverb_size" => Some(ReverbSize),
        "chorus_dry_wet" => Some(ChorusDryWet),
        "chorus_feedback" => Some(ChorusFeedback),
        "chorus_mod_depth" => Some(ChorusModDepth),
        "chorus_frequency" => Some(ChorusFrequency),
        "flanger_dry_wet" => Some(FlangerDryWet),
        "flanger_feedback" => Some(FlangerFeedback),
        "flanger_mod_depth" => Some(FlangerModDepth),
        "flanger_frequency" => Some(FlangerFrequency),
        "flanger_phase_offset" => Some(FlangerPhaseOffset),
        "phaser_dry_wet" => Some(PhaserDryWet),
        "phaser_feedback" => Some(PhaserFeedback),
        "phaser_mod_depth" => Some(PhaserModDepth),
        "phaser_frequency" => Some(PhaserFrequency),
        "phaser_blend" => Some(PhaserBlend),
        "phaser_center" => Some(PhaserCenter),
        "distortion_filter_cutoff" => Some(DistortionFilterCutoff),
        "distortion_drive" => Some(DistortionDrive),
        "distortion_mix" => Some(DistortionMix),
        "filter_fx_cutoff" => Some(FilterFxCutoff),
        "filter_fx_resonance" => Some(FilterFxResonance),
        "filter_fx_blend" => Some(FilterFxBlend),
        "eq_low_cutoff" => Some(EqLowCutoff),
        "eq_band_cutoff" => Some(EqBandCutoff),
        "eq_high_cutoff" => Some(EqHighCutoff),
        "eq_low_gain" => Some(EqLowGain),
        "eq_band_gain" => Some(EqBandGain),
        "eq_high_gain" => Some(EqHighGain),
        "compressor_mix" => Some(CompressorMix),
        "compressor_low_gain" => Some(CompressorLowGain),
        "compressor_band_gain" => Some(CompressorBandGain),
        "compressor_high_gain" => Some(CompressorHighGain),
        _ => Option::None,
    }
}

/// Renders the per-slot wavetables embedded in the preset
/// (`settings.wavetables[0..3]`, `spinwave_materials.slots[3].wavetable`),
/// shared via `Arc` across voice kernels and memoized on content.
pub fn wavetables_from_preset(preset: &Preset) -> Vec<(usize, Arc<Wavetable>)> {
    let mut tables = Vec::new();
    for slot in 0..materials::NUM_SLOTS {
        let Some(json) = materials::slot_wavetable_json(preset, slot) else { continue };
        if json.is_null() {
            continue;
        }
        if let Some(table) = materials::cached_wavetable(json) {
            tables.push((slot, table));
        }
    }
    tables
}

/// Decodes the per-slot sample material (`spinwave_materials.slots[n].sample`).
pub fn slot_samples_from_preset(preset: &Preset) -> Vec<(usize, Arc<Sample>)> {
    let mut samples = Vec::new();
    let Some(block) = &preset.settings.spinwave_materials else { return samples };
    for slot in 0..materials::NUM_SLOTS {
        let Some(payload) = block.slot(slot).and_then(|s| s.sample.as_ref()) else { continue };
        if let Some(sample) = materials::cached_sample(payload) {
            samples.push((slot, sample));
        }
    }
    samples
}

/// Decodes Vital's global `settings.sample` payload (the SMP section).
/// The material is shared by every kernel's sampler (one band-limited
/// pyramid, refcounted).
pub fn global_sample_from_preset(preset: &Preset) -> Option<Arc<Sample>> {
    let value = preset.settings.sample.as_ref()?;
    let payload = SampleJson::from_value(value)?;
    materials::sample_from_json(&payload).map(Arc::new)
}

/// Builds the per-slot SFZ playback sources (`spinwave_materials.slots[n].sfz`),
/// `count` per slot (one per kernel: the zone playback state is private,
/// the zone audio itself is shared). `decode` resolves zone files
/// ([`materials::decode_wav_zone`] in the plugin; the MCP server plugs its
/// any-format decoder).
pub fn multisamples_from_preset(
    preset: &Preset,
    count: usize,
    report: &mut LoadReport,
    decode: &mut dyn FnMut(&std::path::Path) -> Option<materials::ZoneFrames>,
) -> Vec<(usize, Vec<MultisampleSource>)> {
    let mut out = Vec::new();
    let Some(block) = &preset.settings.spinwave_materials else { return out };
    for slot in 0..materials::NUM_SLOTS {
        let Some(sfz) = block.slot(slot).and_then(|s| s.sfz.as_ref()) else { continue };
        let base_dir = materials::sfz_base_dir(sfz);
        match materials::multisample_sources_from_sfz(&sfz.text, &base_dir, count, &mut *decode) {
            Ok(sources) => out.push((slot, sources)),
            Err(e) => report.notes.push(format!("osc {} SFZ '{}' skipped: {e}", slot + 1, sfz.path)),
        }
    }
    out
}

/// Everything the audio thread swaps in for a patch, prebuilt off it:
/// one `KernelParams` and one connection list PER KERNEL (no clone on the
/// audio thread), the effect chains, master settings and materials.
pub struct BuiltPatch {
    pub kernels: Vec<KernelParams>,
    /// One shared connection list, copied into every kernel matrix.
    pub connections: Vec<Connection>,
    pub effects_connections: Vec<EffectsConnection>,
    pub effects: Box<EffectsParams>,
    pub bus_a: Box<EffectsParams>,
    pub bus_b: Box<EffectsParams>,
    pub master: MasterFromPreset,
    pub wavetables: Vec<(usize, Arc<Wavetable>)>,
    pub samples: Vec<(usize, Arc<Sample>)>,
    /// The global sampler material (`settings.sample`), shared by every
    /// kernel; `None` when the preset embeds none.
    pub global_sample: Option<Arc<Sample>>,
    /// One prebuilt playback source per kernel, per slot.
    pub multisamples: Vec<(usize, Vec<MultisampleSource>)>,
    /// Convolution engines with their impulse response already rendered and
    /// transformed, one per chain whose convolution is on and whose impulse
    /// changed. Building one costs an FFT per partition, so it happens here
    /// and the audio thread only swaps it in.
    pub convolutions: Vec<(ChainId, ConvolutionReverb)>,
}

/// The built-in impulse responses, selected by `convolution_impulse`.
fn impulse_response(index: usize, seconds: f32, sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let seconds = seconds.clamp(0.1, 10.0);
    match index {
        1 => ir_plate(seconds, sample_rate),
        2 => ir_spring(seconds, sample_rate),
        _ => ir_hall(seconds, sample_rate),
    }
}

/// Renders the convolution impulse for every chain whose convolution is
/// on. Never called on the audio thread (allocates and runs FFTs).
fn convolutions_from_preset(
    preset: &Preset,
    engine_rate: u32,
    report: &mut LoadReport,
) -> Vec<(ChainId, ConvolutionReverb)> {
    let mut built = Vec::new();
    for (chain, prefix) in
        [(ChainId::Main, ""), (ChainId::BusA, "bus_a_"), (ChainId::BusB, "bus_b_")]
    {
        let reader = Reader::prefixed(preset, prefix);
        if !reader.on("convolution_on") {
            continue;
        }
        let index = reader.get("convolution_impulse") as usize;
        let seconds = reader.get("convolution_size");
        let (left, right) = impulse_response(index, seconds, engine_rate);
        let mut reverb = ConvolutionReverb::new();
        match reverb.set_impulse_response(&left, &right, engine_rate, engine_rate) {
            Ok(()) => built.push((chain, reverb)),
            Err(e) => report
                .notes
                .push(format!("{prefix}convolution impulse not loaded: {e:?}")),
        }
    }
    built
}

impl BuiltPatch {
    /// Kernel count a patch needs: the current pool (never shrinks) or the
    /// polyphony's pairs, whichever is larger.
    #[must_use]
    pub fn kernel_count(current_kernels: usize, polyphony: usize) -> usize {
        polyphony
            .clamp(1, MAX_ACTIVE_POLYPHONY)
            .div_ceil(PARALLEL_VOICES)
            .max(current_kernels)
            .min(MAX_KERNELS)
    }

    /// Builds the patch for `kernel_count` kernels. `report` collects the
    /// connections that could not be routed and the materials that failed.
    /// SFZ zone files decode through the engine's WAV parser.
    #[must_use]
    pub fn build(
        preset: &Preset,
        kernel_count: usize,
        engine_rate: u32,
        report: &mut LoadReport,
    ) -> BuiltPatch {
        BuiltPatch::build_with(
            preset,
            kernel_count,
            engine_rate,
            report,
            &mut materials::decode_wav_zone,
        )
    }

    /// [`BuiltPatch::build`] with a custom SFZ zone decoder.
    #[must_use]
    pub fn build_with(
        preset: &Preset,
        kernel_count: usize,
        engine_rate: u32,
        report: &mut LoadReport,
        decode: &mut dyn FnMut(&std::path::Path) -> Option<materials::ZoneFrames>,
    ) -> BuiltPatch {
        let kernel = kernel_params_from_preset(preset);
        let master = master_from_preset(preset);
        let kernel_count = kernel_count.clamp(1, MAX_KERNELS);
        let mut kernels = Vec::with_capacity(kernel_count);
        for _ in 1..kernel_count {
            kernels.push(kernel.clone());
        }
        kernels.push(kernel);
        BuiltPatch {
            kernels,
            // One list: the audio thread copies it into each kernel's
            // fixed-capacity matrix storage (no allocation).
            connections: connections_from_preset(preset),
            effects_connections: effects_connections_from_preset(preset),
            effects: Box::new(effects_params_from_preset(preset)),
            bus_a: Box::new(effects_params_from_preset_prefixed(preset, "bus_a_")),
            bus_b: Box::new(effects_params_from_preset_prefixed(preset, "bus_b_")),
            master,
            wavetables: wavetables_from_preset(preset),
            samples: slot_samples_from_preset(preset),
            global_sample: global_sample_from_preset(preset),
            multisamples: multisamples_from_preset(preset, kernel_count, report, decode),
            convolutions: convolutions_from_preset(preset, engine_rate, report),
        }
    }
}

fn destination_scale(name: &str) -> f32 {
    parameters()
        .lookup(name)
        .map(|d| d.max - d.min)
        .unwrap_or(1.0)
}

/// Builds full kernel params (and modulation matrix) from a preset.
pub fn kernel_params_from_preset(preset: &Preset) -> KernelParams {
    let reader = Reader::new(preset);
    let mut params = KernelParams::default();

    for i in 0..NUM_OSCILLATORS {
        // Table-backed keys, osc_4 defaulting from the osc_1 table entries.
        let g = |suffix: &str| get_osc(&reader, i, suffix);
        let gp = |suffix: &str| PolyF32::splat(get_osc(&reader, i, suffix));
        let g_on = |suffix: &str| get_osc(&reader, i, suffix) > 0.5;
        // Spinwave-namespace keys (engine selection, sample/granular).
        let raw = |suffix: &str, default: f32| {
            reader.raw(&format!("osc_{}_{}", i + 1, suffix), default)
        };
        let section = &mut params.oscillators[i];
        section.on = g_on("on");
        // `osc_N_engine`: 0 Wavetable / 1 Sample / 2 Granular / 3
        // Multisample (default 0).
        section.engine = osc_engine_from_index(raw("engine", 0.0) as i32);
        section.destination = ProducerDestination::from_index(g("destination") as i32);
        let osc = &mut section.params;
        osc.amplitude = gp("level");
        osc.transpose = gp("transpose");
        osc.transpose_quantize = g("transpose_quantize") as u32;
        osc.tune = gp("tune");
        osc.pan = gp("pan");
        osc.wave_frame = gp("wave_frame");
        osc.frame_spread = gp("frame_spread");
        osc.unison_voices = g("unison_voices").max(1.0) as usize;
        // `unison_detune` is table-`Quadratic` (default 4.472 = 20 real):
        // Vital inserts `cr::Square` AFTER the modulation sum, so the
        // stored value is passed raw and the kernel squares
        // `(stored + offset).clamp(0, 10)` each block.
        osc.unison_detune = gp("unison_detune");
        osc.detune_power = gp("detune_power");
        osc.detune_range = gp("detune_range");
        osc.blend = gp("unison_blend");
        osc.stereo_spread = gp("stereo_spread");
        osc.phase = gp("phase");
        osc.random_phase = gp("random_phase");
        osc.distortion_phase = gp("distortion_phase");
        osc.midi_track = g_on("midi_track");
        osc.spectral_unison = g_on("spectral_unison");
        osc.stack_style = stack_type_from_index(g("stack_style") as i32);
        osc.distortion_type = distortion_type_from_index(g("distortion_type") as i32);
        osc.distortion_amount = gp("distortion_amount");
        osc.distortion_spread = gp("distortion_spread");
        osc.spectral_morph_type = spectral_morph_from_index(g("spectral_morph_type") as i32);
        osc.spectral_morph_amount = gp("spectral_morph_amount");
        osc.spectral_morph_spread = gp("spectral_morph_spread");

        // Sample engine: the common osc keys map onto the slot's sample
        // params (level→level, transpose, tune, pan; keytrack follows
        // `osc_N_midi_track`), plus the spinwave-namespace `osc_N_smp_*`
        // keys: `_smp_rate` (tape-style rate, default 1.0, clamped
        // 0.25..=4.0), `_smp_loop` (bool, default 0), `_smp_slice` (slice
        // marker index, default -1 = none), `_smp_offset` (note-on start in
        // sample frames, default 0).
        let sample = &mut section.sample_params;
        sample.level = gp("level");
        sample.transpose = gp("transpose");
        sample.transpose_quantize = g("transpose_quantize") as u32;
        sample.tune = gp("tune");
        sample.pan = gp("pan");
        sample.keytrack = g_on("midi_track");
        sample.rate = raw("smp_rate", 1.0).clamp(0.25, 4.0);
        sample.loop_sample = raw("smp_loop", 0.0) > 0.5;
        let slice = raw("smp_slice", -1.0);
        sample.slice = (slice >= 0.0).then_some(slice as usize);
        sample.start_offset = raw("smp_offset", 0.0).max(0.0) as usize;

        // Granular engine: level/transpose/tune from the common osc keys
        // (granular has no pan), keytrack from `osc_N_midi_track`, plus the
        // spinwave-namespace `osc_N_gran_*` keys with `GranularParams`
        // defaults: `_gran_position` (0.0), `_gran_position_spray` (0.0),
        // `_gran_size` (seconds, 0.1), `_gran_size_spray` (0.0),
        // `_gran_density` (grains/s, 30.0), `_gran_pitch_spray` (semitones,
        // 0.0), `_gran_window` (index, 0 = Hann), `_gran_direction` (index,
        // 0 = Forward), `_gran_stereo_spray` (0.0).
        let granular = &mut section.granular_params;
        granular.level = gp("level");
        granular.transpose = gp("transpose");
        granular.tune = gp("tune");
        granular.keytrack = g_on("midi_track");
        granular.position = PolyF32::splat(raw("gran_position", 0.0));
        granular.position_spray = PolyF32::splat(raw("gran_position_spray", 0.0));
        granular.grain_size_seconds = PolyF32::splat(raw("gran_size", 0.1));
        granular.size_spray = PolyF32::splat(raw("gran_size_spray", 0.0));
        granular.density = PolyF32::splat(raw("gran_density", 30.0));
        granular.pitch_spray_semitones = PolyF32::splat(raw("gran_pitch_spray", 0.0));
        granular.window = grain_window_from_index(raw("gran_window", 0.0) as i32);
        granular.direction = grain_direction_from_index(raw("gran_direction", 0.0) as i32);
        granular.stereo_spray = PolyF32::splat(raw("gran_stereo_spray", 0.0));
    }

    params.sample.on = reader.on("sample_on");
    params.sample.destination =
        ProducerDestination::from_index(reader.get("sample_destination") as i32);
    params.sample.params.level = reader.poly("sample_level");
    params.sample.params.keytrack = reader.on("sample_keytrack");
    params.sample.params.transpose = reader.poly("sample_transpose");
    params.sample.params.tune = reader.poly("sample_tune");
    params.sample.params.loop_sample = reader.on("sample_loop");
    params.sample.params.bounce = reader.on("sample_bounce");
    params.sample.params.random_phase = reader.on("sample_random_phase");
    params.sample.params.pan = reader.poly("sample_pan");
    params.sample.params.transpose_quantize = reader.get("sample_transpose_quantize") as u32;

    // Dedicated noise source, spinwave-namespace keys with
    // `NoiseParams::default` defaults: `noise_on` (0), `noise_destination`
    // (producer destination index, 0 = Filter1), `noise_level` (0.5),
    // `noise_pink` (white→pink blend, 0.0), `noise_tilt` (-1..1, 0.0),
    // `noise_pan` (-1..1, 0.0), `noise_stereo` (decorrelation, 1.0).
    params.noise.on = reader.raw("noise_on", 0.0) > 0.5;
    params.noise.destination =
        ProducerDestination::from_index(reader.raw("noise_destination", 0.0) as i32);
    let noise = &mut params.noise.params;
    noise.level = PolyF32::splat(reader.raw("noise_level", 0.5));
    noise.pink = PolyF32::splat(reader.raw("noise_pink", 0.0));
    noise.tilt = PolyF32::splat(reader.raw("noise_tilt", 0.0));
    noise.pan = PolyF32::splat(reader.raw("noise_pan", 0.0));
    noise.stereo = PolyF32::splat(reader.raw("noise_stereo", 1.0));

    for i in 0..2 {
        let prefix = format!("filter_{}_", i + 1);
        let section = &mut params.filters[i];
        fill_filter_params(&reader, &prefix, &mut section.params);
        section.keytrack = reader.get(&format!("{prefix}keytrack"));
    }

    // Serial routing from the filter-input switches, in the reference's
    // order (`FiltersModule::process`): filter 1 taking filter 2's output
    // wins when both switches are set. Both are read whatever the first
    // says, so the read audit sees them both.
    let backward = reader.on("filter_1_filter_input");
    let forward = reader.on("filter_2_filter_input");
    if backward {
        params.filter_routing = FilterRouting::SerialBackward;
    } else if forward {
        params.filter_routing = FilterRouting::SerialForward;
    }

    for i in 0..NUM_ENVELOPES {
        let p = |suffix: &str| format!("env_{}_{}", i + 1, suffix);
        let env = &mut params.envelopes[i];
        env.delay = env_seconds(reader.get(&p("delay")));
        env.attack = env_seconds(reader.get(&p("attack")));
        env.hold = env_seconds(reader.get(&p("hold")));
        env.decay = env_seconds(reader.get(&p("decay")));
        env.release = env_seconds(reader.get(&p("release")));
        env.sustain = reader.poly(&p("sustain"));
        env.attack_power = reader.poly(&p("attack_power"));
        env.decay_power = reader.poly(&p("decay_power"));
        env.release_power = reader.poly(&p("release_power"));
    }

    for i in 0..NUM_LFOS {
        // Table-backed keys, lfo_9..lfo_12 defaulting from the lfo_1 table
        // entries.
        let g = |suffix: &str| get_lfo(&reader, i, suffix);
        let gp = |suffix: &str| PolyF32::splat(get_lfo(&reader, i, suffix));
        // Spinwave-namespace per-LFO keys: `lfo_N_generator` (index, default
        // 0 = Shape), `lfo_N_sh_glide` (SampleHold glide portion 0..1,
        // default 0.0), `lfo_N_chaos_speed` (chaos rate multiplier, default
        // 1.0).
        let raw = |suffix: &str, default: f32| {
            reader.raw(&format!("lfo_{}_{}", i + 1, suffix), default)
        };
        let lfo = &mut params.lfos[i];
        lfo.params.frequency = exp_frequency(g("frequency"));
        lfo.params.phase = gp("phase");
        lfo.params.stereo_phase = gp("stereo");
        // fade_time / delay_time are Linear seconds; smooth_time is
        // Exponential (log2 seconds, default -7.5 = ~5.5 ms).
        lfo.params.fade_time = gp("fade_time");
        lfo.params.delay_time = gp("delay_time");
        lfo.params.smooth_mode = g("smooth_mode") > 0.5;
        lfo.params.smooth_time = exp_seconds(g("smooth_time"));
        // `lfo_N_sync_type`: trigger / sync / envelope / sustain envelope /
        // loop point / loop hold (`strings::kSyncNames`). The DSP had every
        // mode; the reader never asked for the index.
        lfo.params.sync_type = lfo_sync_type_from_index(g("sync_type") as i32);
        lfo.params.generator = lfo_generator_from_index(raw("generator", 0.0) as i32);
        lfo.params.sample_hold_glide = PolyF32::splat(raw("sh_glide", 0.0));
        lfo.params.chaos_speed = PolyF32::splat(raw("chaos_speed", 1.0));
        // Tempo sync from the existing table keys: `lfo_N_sync` (0 freq,
        // 1 tempo, 2 dotted, 3 triplet; the LFO-only keytrack index 4 falls
        // back to free-running) and `lfo_N_tempo` (ratio index).
        lfo.sync = LfoSync {
            mode: sync_mode_from_index(g("sync") as i32),
            tempo_index: g("tempo"),
        };
        if let Some(shape) = preset.settings.lfos.get(i) {
            lfo.shape = line_shape_to_generator(shape);
        }
    }

    for i in 0..NUM_RANDOM_LFOS {
        let p = |suffix: &str| format!("random_{}_{}", i + 1, suffix);
        let section = &mut params.random_lfos[i];
        section.params.frequency = exp_frequency(reader.get(&p("frequency")));
        section.params.style = random_style_from_index(reader.get(&p("style")) as i32);
        section.params.stereo = reader.on(&p("stereo"));
        // `random_N_sync_type`: 1 = one instance follows the transport and
        // every voice reads its value.
        section.params.sync = reader.on(&p("sync_type"));
        // Tempo sync from the existing table keys, like the LFOs.
        section.sync = LfoSync {
            mode: sync_mode_from_index(reader.get(&p("sync")) as i32),
            tempo_index: reader.get(&p("tempo")),
        };
    }

    params.velocity_track = reader.get("velocity_track");
    // `pitch_bend_range` (0..48 semitones, default 2); `pitch_wheel` is the
    // wheel POSITION, a modulation source, not the range.
    params.pitch_bend_range = reader.get("pitch_bend_range").clamp(0.0, 48.0);
    // Voice-level settings (reference `SynthVoiceHandler`: portamento slope,
    // then `voice_transpose + voice_tune`, then bend).
    params.voice_amplitude = reader.get("voice_amplitude").clamp(0.0, 1.0);
    params.voice_transpose = reader.get("voice_transpose");
    params.voice_tune = reader.get("voice_tune");
    // `portamento_time` is Exponential (log2 seconds).
    params.portamento_time = reader.get("portamento_time").exp2();
    params.portamento_slope = reader.get("portamento_slope");
    params.portamento_force = reader.on("portamento_force");
    params.portamento_scale = reader.on("portamento_scale");
    // `macro_control_1..8` are all table keys (5..8 flagged spinwave_only,
    // default 0).
    for i in 0..NUM_MACROS {
        params.macros[i] = reader.get(&format!("macro_control_{}", i + 1));
    }

    params
}

/// Builds one connection's transform from its `modulation_N_*` settings
/// and its optional drawn remap (`line_mapping`; a linear default shape
/// installs no curve).
fn read_transform(
    reader: &Reader,
    index: usize,
    destination: &str,
    line_mapping: Option<&LineShape>,
) -> ModulationTransform {
    let n = index + 1;
    let mut transform = ModulationTransform::with_amount(
        reader.get(&format!("modulation_{n}_amount")),
        destination_scale(destination),
    );
    transform.power = PolyF32::splat(reader.get(&format!("modulation_{n}_power")));
    transform.bipolar = reader.on(&format!("modulation_{n}_bipolar"));
    transform.stereo = reader.on(&format!("modulation_{n}_stereo"));
    transform.bypass = reader.on(&format!("modulation_{n}_bypass"));
    if let Some(shape) = line_mapping.filter(|shape| !is_linear_shape(shape)) {
        let generator = line_shape_to_generator(shape);
        transform.remap = Some(Arc::new(RemapCurve::from_line_generator(&generator)));
    }
    transform
}

/// Builds the (per-voice) modulation matrix from the preset's connection
/// list. Connections whose destination is a bus-effect parameter go to
/// [`effects_connections_from_preset`] instead.
pub fn connections_from_preset(preset: &Preset) -> Vec<Connection> {
    let reader = Reader::new(preset);
    let mut connections = Vec::new();
    for (index, modulation) in preset.settings.modulations.iter().enumerate() {
        let (Some(source), Some(dest)) = (
            parse_mod_source(&modulation.source),
            parse_mod_dest(&modulation.destination),
        ) else {
            continue;
        };
        let transform = read_transform(&reader, index, &modulation.destination, modulation.line_mapping.as_ref());
        connections.push(Connection { source, dest, transform });
    }
    connections
}

/// Builds the bus-effect (mono) modulation matrix from the preset's
/// connection list: every connection whose destination does not parse as a
/// voice destination but does parse as an effect destination (a connection
/// tries the voice matrix first, then the effects matrix).
pub fn effects_connections_from_preset(preset: &Preset) -> Vec<EffectsConnection> {
    let reader = Reader::new(preset);
    let mut connections = Vec::new();
    for (index, modulation) in preset.settings.modulations.iter().enumerate() {
        if parse_mod_dest(&modulation.destination).is_some() {
            continue; // routed to the voice kernel matrix
        }
        let (Some(source), Some(dest)) = (
            parse_mod_source(&modulation.source),
            parse_effects_mod_dest(&modulation.destination),
        ) else {
            continue;
        };
        let transform = read_transform(&reader, index, &modulation.destination, modulation.line_mapping.as_ref());
        connections.push(EffectsConnection { source, dest, transform });
    }
    connections
}

/// Builds the bus effect chain parameters from a preset.
///
/// Unit conversions (settings → dsp struct), mirroring the module wrappers:
/// tempo-syncable `*_frequency` values and the chorus delays / reverb decay
/// time are stored as log2 of the engine unit; the EQ resonances and the
/// reverb chorus amount are stored square-root scaled (table `Quadratic`).
/// Everything else is stored in the unit the dsp struct expects.
pub fn effects_params_from_preset(preset: &Preset) -> EffectsParams {
    effects_params_from_reader(&Reader::new(preset))
}

/// Bus-chain variant of [`effects_params_from_preset`]: every key is read
/// with `prefix` prepended (e.g. `bus_a_delay_on`, `bus_a_reverb_dry_wet`),
/// falling back to the UNPREFIXED table default when a prefixed key is
/// absent — so an untouched bus chain matches the main chain's defaults.
pub fn effects_params_from_preset_prefixed(preset: &Preset, prefix: &str) -> EffectsParams {
    effects_params_from_reader(&Reader::prefixed(preset, prefix))
}

fn effects_params_from_reader(reader: &Reader) -> EffectsParams {
    let mut params = EffectsParams {
        order: decode_order(reader.get("effect_chain_order") as u32),
        ..EffectsParams::default()
    };

    params.chorus_on = reader.on("chorus_on");
    let chorus = &mut params.chorus;
    chorus.voices = reader.get("chorus_voices").max(1.0) as usize;
    chorus.feedback = reader.poly("chorus_feedback");
    chorus.wet = reader.poly("chorus_dry_wet");
    chorus.cutoff_midi = reader.poly("chorus_cutoff");
    chorus.spread = reader.poly("chorus_spread");
    chorus.mod_depth = reader.poly("chorus_mod_depth");
    chorus.delay_1 = exp_seconds(reader.get("chorus_delay_1"));
    chorus.delay_2 = exp_seconds(reader.get("chorus_delay_2"));
    // `chorus.frequency` is overwritten from the sync struct every block.
    params.chorus_sync = synced_frequency(reader, "chorus");

    params.compressor_on = reader.on("compressor_on");
    let compressor = &mut params.compressor;
    compressor.enabled_bands =
        band_options_from_index(reader.get("compressor_enabled_bands") as i32);
    compressor.attack = reader.poly("compressor_attack");
    compressor.release = reader.poly("compressor_release");
    compressor.mix = reader.poly("compressor_mix");
    compressor.low_upper_ratio = reader.poly("compressor_low_upper_ratio");
    compressor.band_upper_ratio = reader.poly("compressor_band_upper_ratio");
    compressor.high_upper_ratio = reader.poly("compressor_high_upper_ratio");
    compressor.low_lower_ratio = reader.poly("compressor_low_lower_ratio");
    compressor.band_lower_ratio = reader.poly("compressor_band_lower_ratio");
    compressor.high_lower_ratio = reader.poly("compressor_high_lower_ratio");
    compressor.low_upper_threshold_db = reader.poly("compressor_low_upper_threshold");
    compressor.band_upper_threshold_db = reader.poly("compressor_band_upper_threshold");
    compressor.high_upper_threshold_db = reader.poly("compressor_high_upper_threshold");
    compressor.low_lower_threshold_db = reader.poly("compressor_low_lower_threshold");
    compressor.band_lower_threshold_db = reader.poly("compressor_band_lower_threshold");
    compressor.high_lower_threshold_db = reader.poly("compressor_high_lower_threshold");
    compressor.low_output_gain_db = reader.poly("compressor_low_gain");
    compressor.band_output_gain_db = reader.poly("compressor_band_gain");
    compressor.high_output_gain_db = reader.poly("compressor_high_gain");

    params.delay_on = reader.on("delay_on");
    let delay = &mut params.delay;
    delay.feedback = reader.poly("delay_feedback");
    delay.wet = reader.poly("delay_dry_wet");
    delay.filter_cutoff_midi = reader.poly("delay_filter_cutoff");
    delay.filter_spread = reader.poly("delay_filter_spread");
    delay.style = delay_style_from_index(reader.get("delay_style") as i32);
    // `delay.period_samples` is resolved from the sync structs every block.
    params.delay_sync = synced_frequency(reader, "delay");
    params.delay_aux_sync = synced_frequency(reader, "delay_aux");

    params.distortion_on = reader.on("distortion_on");
    params.distortion_type = fx_distortion_type_from_index(reader.get("distortion_type") as i32);
    params.distortion_drive_db = reader.get("distortion_drive");
    params.distortion_mix = reader.get("distortion_mix");
    // The distortion's own filter. Its four controls were never read here
    // until the golden bench had a case for it: the engine carried the
    // fields, the preset never reached them (fx_distortion_filter_pre/post
    // at 2.3e-1 with the filter silently off, 4e-4 with it on).
    params.distortion_filter_order =
        DistortionFilterOrder::from_index(reader.get("distortion_filter_order") as i32);
    params.distortion_filter_cutoff = reader.get("distortion_filter_cutoff");
    params.distortion_filter_resonance = reader.get("distortion_filter_resonance");
    params.distortion_filter_blend = reader.get("distortion_filter_blend");

    params.eq_on = reader.on("eq_on");
    let eq = &mut params.eq;
    eq.low_mode = reader.on("eq_low_mode");
    eq.band_mode = reader.on("eq_band_mode");
    eq.high_mode = reader.on("eq_high_mode");
    eq.low_cutoff_midi = reader.poly("eq_low_cutoff");
    eq.band_cutoff_midi = reader.poly("eq_band_cutoff");
    eq.high_cutoff_midi = reader.poly("eq_high_cutoff");
    eq.low_resonance = quadratic(reader.get("eq_low_resonance"));
    eq.band_resonance = quadratic(reader.get("eq_band_resonance"));
    eq.high_resonance = quadratic(reader.get("eq_high_resonance"));
    eq.low_gain_db = reader.poly("eq_low_gain");
    eq.band_gain_db = reader.poly("eq_band_gain");
    eq.high_gain_db = reader.poly("eq_high_gain");

    params.filter_fx_on = reader.on("filter_fx_on");
    fill_filter_params(reader, "filter_fx_", &mut params.filter_fx);
    params.filter_fx_keytrack = reader.get("filter_fx_keytrack");

    params.flanger_on = reader.on("flanger_on");
    let flanger = &mut params.flanger;
    flanger.center_midi = reader.poly("flanger_center");
    flanger.feedback = reader.poly("flanger_feedback");
    // Table range is [0, 0.5]: 0.5 is the 50/50 equal-power point.
    flanger.wet = reader.poly("flanger_dry_wet");
    flanger.mod_depth = reader.poly("flanger_mod_depth");
    flanger.phase_offset = reader.poly("flanger_phase_offset");
    params.flanger_sync = synced_frequency(reader, "flanger");

    params.phaser_on = reader.on("phaser_on");
    let phaser = &mut params.phaser;
    phaser.mix = reader.poly("phaser_dry_wet");
    phaser.feedback_gain = reader.poly("phaser_feedback");
    phaser.center_midi = reader.poly("phaser_center");
    phaser.mod_depth = reader.poly("phaser_mod_depth");
    phaser.phase_offset = reader.poly("phaser_phase_offset");
    phaser.blend = reader.poly("phaser_blend");
    params.phaser_sync = synced_frequency(reader, "phaser");

    params.reverb_on = reader.on("reverb_on");
    let reverb = &mut params.reverb;
    reverb.decay_time = exp_seconds(reader.get("reverb_decay_time"));
    reverb.pre_low_cutoff = reader.poly("reverb_pre_low_cutoff");
    reverb.pre_high_cutoff = reader.poly("reverb_pre_high_cutoff");
    reverb.low_cutoff = reader.poly("reverb_low_shelf_cutoff");
    reverb.low_gain = reader.poly("reverb_low_shelf_gain");
    reverb.high_cutoff = reader.poly("reverb_high_shelf_cutoff");
    reverb.high_gain = reader.poly("reverb_high_shelf_gain");
    reverb.chorus_amount = quadratic(reader.get("reverb_chorus_amount"));
    reverb.chorus_frequency = exp_frequency(reader.get("reverb_chorus_frequency"));
    reverb.size = reader.poly("reverb_size");
    reverb.delay = reader.poly("reverb_delay");
    reverb.wet = reader.poly("reverb_dry_wet");

    // The two Spinwave-only effects. The convolution's impulse response is
    // NOT built here: it is rendered off the audio thread and installed
    // with `SoundEngine::set_convolution_engine` (see `ConvolutionIr`).
    params.convolution_on = reader.on("convolution_on");
    let convolution = &mut params.convolution;
    convolution.dry_wet = reader.get("convolution_dry_wet");
    convolution.predelay_seconds = reader.get("convolution_predelay");
    convolution.ir_gain_db = reader.get("convolution_gain");

    params.frequency_shifter_on = reader.on("frequency_shifter_on");
    let shifter = &mut params.frequency_shifter;
    shifter.shift_hz = reader.get("frequency_shifter_shift");
    shifter.mix = reader.get("frequency_shifter_mix");
    shifter.stereo = reader.on("frequency_shifter_stereo");

    // Per-effect signal splits, spinwave-namespace keys: `fx_split_<name>`
    // (SplitMode index, default 0 = Full) and `fx_split_<name>_crossover`
    // (Hz, default 1000, only used by the Low/High modes). The `split`
    // array is indexed by `Effect as usize`.
    const FX_SPLIT_EFFECTS: [(&str, Effect); 9] = [
        ("chorus", Effect::Chorus),
        ("compressor", Effect::Compressor),
        ("delay", Effect::Delay),
        ("distortion", Effect::Distortion),
        ("eq", Effect::Eq),
        ("filter_fx", Effect::FilterFx),
        ("flanger", Effect::Flanger),
        ("phaser", Effect::Phaser),
        ("reverb", Effect::Reverb),
    ];
    for (name, effect) in FX_SPLIT_EFFECTS {
        let split = &mut params.split[effect as usize];
        split.mode = split_mode_from_index(reader.raw(&format!("fx_split_{name}"), 0.0) as i32);
        split.crossover_hz = reader.raw(
            &format!("fx_split_{name}_crossover"),
            DEFAULT_SPLIT_CROSSOVER_HZ,
        );
    }

    params
}

/// Master / output-section settings mapped from a preset. Callers apply them
/// to `SoundEngine::master`, the effects mixer (`SoundEngine::mixer`), the
/// polyphony and the voice allocator.
#[derive(Clone, Copy, Debug)]
pub struct MasterFromPreset {
    /// Master volume in dB. The `volume` setting is stored square-root
    /// scaled with a -80 post offset (`cr::Root` in the reference):
    /// `dB = sqrt(stored) - 80`, e.g. the default 5473.0404 → ~-6.02 dB.
    pub volume_db: f32,
    /// `stereo_routing` in `[0, 1]`.
    pub stereo_routing: f32,
    pub stereo_mode: StereoMode,
    /// Voice count in `1..=64`.
    pub polyphony: usize,
    pub legato: bool,
    pub voice_priority: VoicePriority,
    pub voice_override: VoiceOverride,
    /// Effects-mixer send buses, spinwave-namespace keys with
    /// `BusParams::default` defaults: `bus_a_on` (0), `bus_a_send` (0..1,
    /// default 0), `bus_a_return_db` (dB, default 0), `bus_a_output`
    /// (0 = Master parallel, 1 = MainChain serial; default 0) — and the
    /// same `bus_b_*` set.
    pub mixer: MixerParams,
}

/// Reads one send bus's spinwave-namespace settings (`{prefix}on`, ...).
fn bus_params(reader: &Reader, prefix: &str) -> BusParams {
    BusParams {
        on: reader.raw(&format!("{prefix}on"), 0.0) > 0.5,
        send: reader.raw(&format!("{prefix}send"), 0.0),
        return_gain_db: reader.raw(&format!("{prefix}return_db"), 0.0),
        output: if reader.raw(&format!("{prefix}output"), 0.0) > 0.5 {
            BusOutput::MainChain
        } else {
            BusOutput::Master
        },
    }
}

/// Reads the master / voice-allocation settings from a preset.
pub fn master_from_preset(preset: &Preset) -> MasterFromPreset {
    let reader = Reader::new(preset);
    let volume_post_offset = parameters()
        .lookup("volume")
        .map(|d| d.post_offset)
        .unwrap_or(-80.0);
    MasterFromPreset {
        mixer: MixerParams {
            bus_a: bus_params(&reader, "bus_a_"),
            bus_b: bus_params(&reader, "bus_b_"),
        },
        volume_db: reader.get("volume").max(0.0).sqrt() + volume_post_offset,
        stereo_routing: reader.get("stereo_routing"),
        stereo_mode: if reader.on("stereo_mode") {
            StereoMode::Rotate
        } else {
            StereoMode::Spread
        },
        polyphony: (reader.get("polyphony").max(1.0) as usize).min(MAX_ACTIVE_POLYPHONY),
        legato: reader.on("legato"),
        voice_priority: voice_priority_from_index(reader.get("voice_priority") as i32),
        voice_override: if reader.on("voice_override") {
            VoiceOverride::Steal
        } else {
            VoiceOverride::Kill
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset(json: &str) -> Preset {
        Preset::from_json(json).expect("valid preset")
    }

    #[test]
    fn defaults_apply_without_settings() {
        let preset = preset(r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#);
        let params = kernel_params_from_preset(&preset);
        // Table defaults: osc 1 on, others off.
        assert!(params.oscillators[0].on);
        assert!(!params.oscillators[1].on);
        // Default filter cutoff from the table (midi 60).
        assert!(params.filters[0].params.state.midi_cutoff.lane(0) > 0.0);
    }

    #[test]
    fn envelope_times_are_quartic() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"env_1_attack": 2.0}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        assert!((params.envelopes[0].attack.lane(0) - 16.0).abs() < 1e-4);
    }

    #[test]
    fn lfo_frequency_is_exponential() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"lfo_1_frequency": 3.0}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        assert!((params.lfos[0].params.frequency.lane(0) - 8.0).abs() < 1e-4);
    }

    #[test]
    fn modulations_map_to_connections() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "modulation_1_amount": 0.5,
                  "modulation_1_bipolar": 1.0,
                  "modulations":[
                    {"source":"lfo_1","destination":"filter_1_cutoff"},
                    {"source":"env_2","destination":"osc_1_level"},
                    {"source":"unknown_thing","destination":"filter_1_cutoff"}
                  ]}}"#,
        );
        let connections = connections_from_preset(&preset);
        assert_eq!(connections.len(), 2);
        assert_eq!(connections[0].source, ModSource::Lfo(0));
        assert_eq!(connections[0].dest, ModDest::FilterCutoff(0));
        assert!(connections[0].transform.bipolar);
        // filter_1_cutoff range 8..136 => scale 128.
        let mut t = connections[0].transform.clone();
        let out = t.process_control(PolyF32::splat(1.0));
        assert!((out.scaled.lane(0) - 0.5 * 0.5 * 128.0).abs() < 1.0);
    }

    #[test]
    fn expanded_voice_destinations_parse() {
        for (name, expected) in [
            ("osc_2_frame_spread", ModDest::OscFrameSpread(1)),
            ("osc_1_distortion_phase", ModDest::OscDistortionPhase(0)),
            ("osc_3_stereo_spread", ModDest::OscStereoSpread(2)),
            ("osc_1_unison_blend", ModDest::OscUnisonBlend(0)),
            ("filter_2_drive", ModDest::FilterDrive(1)),
            ("filter_1_blend_transpose", ModDest::FilterBlendTranspose(0)),
            ("filter_2_keytrack", ModDest::FilterKeytrack(1)),
            ("env_3_delay", ModDest::EnvDelay(2)),
            ("env_1_hold", ModDest::EnvHold(0)),
            ("env_2_attack_power", ModDest::EnvAttackPower(1)),
            ("env_4_decay_power", ModDest::EnvDecayPower(3)),
            ("env_6_release_power", ModDest::EnvReleasePower(5)),
            ("lfo_5_phase", ModDest::LfoPhase(4)),
            ("random_3_frequency", ModDest::RandomLfoFrequency(2)),
            ("sample_transpose", ModDest::SampleTranspose),
            ("sample_tune", ModDest::SampleTune),
            ("sample_pan", ModDest::SamplePan),
            ("pitch_wheel", ModDest::PitchBend),
        ] {
            assert_eq!(parse_mod_dest(name), Some(expected), "{name}");
        }
    }

    #[test]
    fn modulations_split_between_voice_and_effects_matrices() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "modulation_2_amount": 1.0,
                  "modulation_4_amount": 0.5,
                  "modulations":[
                    {"source":"lfo_1","destination":"filter_1_cutoff"},
                    {"source":"lfo_2","destination":"delay_dry_wet"},
                    {"source":"env_2","destination":"env_1_attack_power"},
                    {"source":"macro_control_1","destination":"distortion_drive"},
                    {"source":"random_1","destination":"chorus_feedback"},
                    {"source":"lfo_1","destination":"not_a_destination"}
                  ]}}"#,
        );

        let voice = connections_from_preset(&preset);
        assert_eq!(voice.len(), 2);
        assert_eq!(voice[0].dest, ModDest::FilterCutoff(0));
        assert_eq!(voice[1].dest, ModDest::EnvAttackPower(0));

        let effects = effects_connections_from_preset(&preset);
        assert_eq!(effects.len(), 3);
        assert_eq!(effects[0].source, ModSource::Lfo(1));
        assert_eq!(effects[0].dest, EffectsModDest::DelayDryWet);
        assert_eq!(effects[1].source, ModSource::Macro(0));
        assert_eq!(effects[1].dest, EffectsModDest::DistortionDrive);
        assert_eq!(effects[2].source, ModSource::RandomLfo(0));
        assert_eq!(effects[2].dest, EffectsModDest::ChorusFeedback);

        // Slot numbering follows the modulation list index: the delay
        // connection reads modulation_2_*, the distortion one modulation_4_*.
        // delay_dry_wet range 0..1 -> scale 1; distortion_drive -30..30 -> 60.
        let mut delay_transform = effects[0].transform.clone();
        let out = delay_transform.process_control(PolyF32::splat(1.0));
        assert!((out.scaled.lane(0) - 1.0).abs() < 1e-4);
        let mut drive_transform = effects[1].transform.clone();
        let out = drive_transform.process_control(PolyF32::splat(1.0));
        assert!((out.scaled.lane(0) - 30.0).abs() < 1e-3);
    }

    #[test]
    fn effects_defaults_apply_without_settings() {
        let preset = preset(r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#);
        let params = effects_params_from_preset(&preset);
        assert_eq!(params.order, spinwave_engine::engine::DEFAULT_ORDER);
        assert!(!params.chorus_on && !params.delay_on && !params.reverb_on);
        // Table defaults: chorus mix 0.5, 4 voices; delay mix 0.3334 with
        // straight tempo sync at index 9 (1/8); reverb mix 0.25.
        assert_eq!(params.chorus.wet.lane(0), 0.5);
        assert_eq!(params.chorus.voices, 4);
        assert_eq!(params.delay.wet.lane(0), 0.3334);
        assert_eq!(params.delay_sync.sync, SyncMode::Tempo);
        assert_eq!(params.delay_sync.tempo_index, 9.0);
        assert_eq!(params.delay_aux_sync.tempo_index, 9.0);
        assert_eq!(params.reverb.wet.lane(0), 0.25);
        assert_eq!(params.compressor.band_upper_threshold_db.lane(0), -25.0);
        assert_eq!(params.distortion_mix, 1.0);
        // eq_low_resonance default 0.3163 is square-root scaled -> ~0.1.
        assert!((params.eq.low_resonance.lane(0) - 0.1).abs() < 1e-3);
    }

    /// The two Spinwave-only effects are reachable from a preset, and the
    /// convolution's impulse is rendered off the audio thread.
    #[test]
    fn spinwave_effects_are_reachable_from_a_preset() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "convolution_on": 1.0, "convolution_impulse": 2.0,
                  "convolution_size": 0.5, "convolution_dry_wet": 0.8,
                  "convolution_predelay": 0.02, "convolution_gain": -3.0,
                  "frequency_shifter_on": 1.0, "frequency_shifter_shift": -220.0,
                  "frequency_shifter_mix": 0.6, "frequency_shifter_stereo": 1.0,
                  "bus_a_convolution_on": 1.0}}"#,
        );
        let params = effects_params_from_preset(&preset);
        assert!(params.convolution_on);
        assert_eq!(params.convolution.dry_wet, 0.8);
        assert_eq!(params.convolution.predelay_seconds, 0.02);
        assert_eq!(params.convolution.ir_gain_db, -3.0);
        assert!(params.frequency_shifter_on);
        assert_eq!(params.frequency_shifter.shift_hz, -220.0);
        assert_eq!(params.frequency_shifter.mix, 0.6);
        assert!(params.frequency_shifter.stereo);

        // Both chains that switched it on get a loaded engine.
        let mut report = LoadReport::default();
        let built = BuiltPatch::build(&preset, 1, 88_200, &mut report);
        let chains: Vec<ChainId> = built.convolutions.iter().map(|(chain, _)| *chain).collect();
        assert_eq!(chains, vec![ChainId::Main, ChainId::BusA]);
        assert!(built.convolutions.iter().all(|(_, reverb)| reverb.has_engine()));
        assert!(report.notes.is_empty(), "{:?}", report.notes);

        // Defaults leave both off and build nothing.
        let quiet = BuiltPatch::build(&preset_default(), 1, 88_200, &mut LoadReport::default());
        assert!(quiet.convolutions.is_empty());
        assert!(!effects_params_from_preset(&preset_default()).convolution_on);
    }

    #[test]
    fn effect_chain_order_decodes_from_settings() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"effect_chain_order": 1.0}}"#,
        );
        let params = effects_params_from_preset(&preset);
        // Code 1 is a single inversion at the last legacy position
        // (phaser <-> reverb); the two Spinwave effects stay anchored after
        // the flanger and after the reverb.
        use spinwave_engine::effect_chain::Effect;
        let expected = [
            Effect::Chorus,
            Effect::Compressor,
            Effect::Delay,
            Effect::Distortion,
            Effect::Eq,
            Effect::FilterFx,
            Effect::Flanger,
            Effect::FrequencyShifter,
            Effect::Reverb,
            Effect::Convolution,
            Effect::Phaser,
        ];
        assert_eq!(params.order, expected);
    }

    #[test]
    fn delay_sync_maps_main_and_aux_lines() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "delay_sync": 0.0, "delay_frequency": 3.0, "delay_tempo": 6.0,
                  "delay_aux_sync": 3.0, "delay_aux_tempo": 5.0}}"#,
        );
        let params = effects_params_from_preset(&preset);
        assert_eq!(params.delay_sync.sync, SyncMode::Frequency);
        assert!((params.delay_sync.frequency_hz - 8.0).abs() < 1e-4);
        assert_eq!(params.delay_sync.tempo_index, 6.0);
        assert_eq!(params.delay_aux_sync.sync, SyncMode::TripletTempo);
        assert_eq!(params.delay_aux_sync.tempo_index, 5.0);
    }

    #[test]
    fn effect_params_route_from_settings() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "chorus_on": 1.0, "chorus_voices": 2.0, "chorus_delay_1": -9.0,
                  "chorus_feedback": -0.3,
                  "compressor_enabled_bands": 3.0, "compressor_high_gain": -7.5,
                  "compressor_low_upper_ratio": 0.65,
                  "delay_on": 1.0, "delay_style": 2.0, "delay_feedback": 0.25,
                  "distortion_type": 4.0, "distortion_drive": 12.5,
                  "eq_high_mode": 1.0, "eq_band_resonance": 0.5, "eq_band_cutoff": 92.0,
                  "filter_fx_on": 1.0, "filter_fx_cutoff": 90.0,
                  "flanger_center": 70.0, "flanger_phase_offset": 0.25,
                  "phaser_dry_wet": 0.75, "phaser_blend": 1.5,
                  "reverb_decay_time": 2.0, "reverb_chorus_amount": 0.5,
                  "reverb_size": 0.8}}"#,
        );
        let params = effects_params_from_preset(&preset);

        assert!(params.chorus_on);
        assert_eq!(params.chorus.voices, 2);
        // Chorus delays are stored as log2(seconds).
        assert!((params.chorus.delay_1.lane(0) - 0.001953125).abs() < 1e-7);
        assert_eq!(params.chorus.feedback.lane(0), -0.3);

        assert_eq!(params.compressor.enabled_bands, BandOptions::SingleBand);
        assert_eq!(params.compressor.high_output_gain_db.lane(0), -7.5);
        assert_eq!(params.compressor.low_upper_ratio.lane(0), 0.65);

        assert!(params.delay_on);
        assert_eq!(params.delay.style, DelayStyle::PingPong);
        assert_eq!(params.delay.feedback.lane(0), 0.25);

        assert_eq!(params.distortion_type, FxDistortionType::BitCrush);
        assert_eq!(params.distortion_drive_db, 12.5);

        assert!(params.eq.high_mode);
        assert!(!params.eq.low_mode);
        assert_eq!(params.eq.band_cutoff_midi.lane(0), 92.0);
        // EQ resonances are square-root scaled in the settings.
        assert_eq!(params.eq.band_resonance.lane(0), 0.25);

        assert!(params.filter_fx_on);
        assert_eq!(params.filter_fx.state.midi_cutoff.lane(0), 90.0);

        assert_eq!(params.flanger.center_midi.lane(0), 70.0);
        assert_eq!(params.flanger.phase_offset.lane(0), 0.25);

        assert_eq!(params.phaser.mix.lane(0), 0.75);
        assert_eq!(params.phaser.blend.lane(0), 1.5);

        // Reverb decay time is stored as log2(seconds); chorus amount is
        // square-root scaled.
        assert!((params.reverb.decay_time.lane(0) - 4.0).abs() < 1e-5);
        assert_eq!(params.reverb.chorus_amount.lane(0), 0.25);
        assert_eq!(params.reverb.size.lane(0), 0.8);
    }

    #[test]
    fn master_volume_converts_square_root_scale() {
        // sqrt(6400) - 80 = 0 dB.
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"volume": 6400.0}}"#,
        );
        assert!(master_from_preset(&preset).volume_db.abs() < 1e-4);
    }

    #[test]
    fn master_settings_apply() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "stereo_routing": 0.3, "stereo_mode": 1.0, "polyphony": 16.0,
                  "legato": 1.0, "voice_priority": 1.0, "voice_override": 1.0}}"#,
        );
        let master = master_from_preset(&preset);
        assert_eq!(master.stereo_routing, 0.3);
        assert_eq!(master.stereo_mode, StereoMode::Rotate);
        assert_eq!(master.polyphony, 16);
        assert!(master.legato);
        assert_eq!(master.voice_priority, VoicePriority::Oldest);
        assert_eq!(master.voice_override, VoiceOverride::Steal);
    }

    #[test]
    fn master_defaults_from_table() {
        let preset = preset(r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#);
        let master = master_from_preset(&preset);
        // Default volume 5473.0404 -> sqrt - 80 = ~-6.02 dB.
        assert!((master.volume_db + 6.02).abs() < 0.01);
        assert_eq!(master.stereo_routing, 1.0);
        assert_eq!(master.stereo_mode, StereoMode::Spread);
        assert_eq!(master.polyphony, 8);
        assert!(!master.legato);
        assert_eq!(master.voice_priority, VoicePriority::RoundRobin);
        assert_eq!(master.voice_override, VoiceOverride::Kill);
    }

    #[test]
    fn osc_engine_and_granular_keys_map() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "osc_1_engine": 1.0,
                  "osc_2_engine": 2.0,
                  "osc_2_level": 0.5,
                  "osc_2_transpose": 12.0,
                  "osc_2_gran_position": 0.3,
                  "osc_2_gran_size": 0.25,
                  "osc_2_gran_density": 12.0,
                  "osc_2_gran_window": 3.0,
                  "osc_2_gran_direction": 1.0,
                  "osc_2_gran_stereo_spray": 0.4,
                  "osc_2_smp_rate": 2.0,
                  "osc_2_smp_loop": 1.0,
                  "osc_2_smp_slice": 3.0,
                  "osc_4_on": 1.0}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        assert_eq!(params.oscillators[0].engine, OscEngineKind::Sample);
        assert_eq!(params.oscillators[1].engine, OscEngineKind::Granular);
        assert_eq!(params.oscillators[2].engine, OscEngineKind::Wavetable);

        let granular = &params.oscillators[1].granular_params;
        assert_eq!(granular.position.lane(0), 0.3);
        assert_eq!(granular.grain_size_seconds.lane(0), 0.25);
        assert_eq!(granular.density.lane(0), 12.0);
        assert_eq!(granular.window, GrainWindow::Tukey);
        assert_eq!(granular.direction, GrainDirection::Reverse);
        assert_eq!(granular.stereo_spray.lane(0), 0.4);
        // Common osc keys land in the granular params too.
        assert_eq!(granular.level.lane(0), 0.5);
        assert_eq!(granular.transpose.lane(0), 12.0);

        let sample = &params.oscillators[1].sample_params;
        assert_eq!(sample.rate, 2.0);
        assert!(sample.loop_sample);
        assert_eq!(sample.slice, Some(3));
        assert_eq!(sample.level.lane(0), 0.5);
        assert_eq!(sample.start_offset, 0);

        // Untouched slots keep the GranularParams defaults.
        assert_eq!(params.oscillators[0].granular_params.density.lane(0), 30.0);
        assert_eq!(params.oscillators[0].sample_params.slice, None);

        // osc_4 has no table entries: absent keys fall back to osc_1's
        // table defaults (level 1/sqrt(2), midi_track on).
        assert!(params.oscillators[3].on);
        let level = params.oscillators[3].params.amplitude.lane(0);
        assert!((level - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(params.oscillators[3].params.midi_track);
    }

    #[test]
    fn lfo_generator_and_sync_keys_map() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "lfo_1_sync": 0.0,
                  "lfo_2_generator": 1.0,
                  "lfo_2_sh_glide": 0.4,
                  "lfo_2_sync": 3.0,
                  "lfo_2_tempo": 6.0,
                  "lfo_3_generator": 2.0,
                  "lfo_3_chaos_speed": 3.0,
                  "lfo_10_frequency": 2.0,
                  "random_2_sync": 2.0,
                  "random_2_tempo": 5.0}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        assert_eq!(params.lfos[0].sync.mode, SyncMode::Frequency);
        assert_eq!(params.lfos[0].params.generator, LfoGeneratorMode::Shape);

        assert_eq!(params.lfos[1].params.generator, LfoGeneratorMode::SampleHold);
        assert_eq!(params.lfos[1].params.sample_hold_glide.lane(0), 0.4);
        assert_eq!(params.lfos[1].sync.mode, SyncMode::TripletTempo);
        assert_eq!(params.lfos[1].sync.tempo_index, 6.0);

        assert_eq!(params.lfos[2].params.generator, LfoGeneratorMode::Chaos1);
        assert_eq!(params.lfos[2].params.chaos_speed.lane(0), 3.0);

        // Table defaults: lfo_N_sync 1 (Tempo), lfo_N_tempo 7.
        assert_eq!(params.lfos[3].sync.mode, SyncMode::Tempo);
        assert_eq!(params.lfos[3].sync.tempo_index, 7.0);

        // lfo_10 has no table entries but its set key applies; absent keys
        // fall back to lfo_1's table defaults.
        assert_eq!(params.lfos[9].params.frequency.lane(0), 4.0);
        assert_eq!(params.lfos[9].sync.mode, SyncMode::Tempo);
        assert_eq!(params.lfos[9].params.chaos_speed.lane(0), 1.0);

        assert_eq!(params.random_lfos[1].sync.mode, SyncMode::DottedTempo);
        assert_eq!(params.random_lfos[1].sync.tempo_index, 5.0);
        // random_N_tempo table default is 8.
        assert_eq!(params.random_lfos[0].sync.mode, SyncMode::Tempo);
        assert_eq!(params.random_lfos[0].sync.tempo_index, 8.0);
    }

    #[test]
    fn noise_keys_map() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "noise_on": 1.0,
                  "noise_destination": 4.0,
                  "noise_level": 0.8,
                  "noise_pink": 0.5,
                  "noise_tilt": -0.3,
                  "noise_pan": 0.2,
                  "noise_stereo": 0.0}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        assert!(params.noise.on);
        assert_eq!(params.noise.destination, ProducerDestination::DirectOut);
        assert_eq!(params.noise.params.level.lane(0), 0.8);
        assert_eq!(params.noise.params.pink.lane(0), 0.5);
        assert_eq!(params.noise.params.tilt.lane(0), -0.3);
        assert_eq!(params.noise.params.pan.lane(0), 0.2);
        assert_eq!(params.noise.params.stereo.lane(0), 0.0);

        // NoiseParams defaults without settings (off, level 0.5, stereo 1).
        let init = preset_default();
        let params = kernel_params_from_preset(&init);
        assert!(!params.noise.on);
        assert_eq!(params.noise.params.level.lane(0), 0.5);
        assert_eq!(params.noise.params.stereo.lane(0), 1.0);
    }

    fn preset_default() -> Preset {
        preset(r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#)
    }

    #[test]
    fn bus_prefixed_effects_read_prefixed_keys() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "bus_a_delay_on": 1.0,
                  "bus_a_delay_feedback": 0.7,
                  "bus_a_fx_split_delay": 3.0}}"#,
        );
        let bus_a = effects_params_from_preset_prefixed(&preset, "bus_a_");
        assert!(bus_a.delay_on);
        assert_eq!(bus_a.delay.feedback.lane(0), 0.7);
        // Absent prefixed keys fall back to the unprefixed table default.
        assert_eq!(bus_a.delay.wet.lane(0), 0.3334);
        assert_eq!(bus_a.delay_sync.tempo_index, 9.0);
        // Splits work through the prefix too.
        assert_eq!(bus_a.split[Effect::Delay as usize].mode, SplitMode::Low);
        // The main chain is untouched by bus_a_* keys.
        let main = effects_params_from_preset(&preset);
        assert!(!main.delay_on);
        assert_eq!(main.split[Effect::Delay as usize].mode, SplitMode::Full);
    }

    #[test]
    fn effect_splits_map() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "fx_split_delay": 2.0,
                  "fx_split_reverb": 4.0,
                  "fx_split_reverb_crossover": 250.0}}"#,
        );
        let params = effects_params_from_preset(&preset);
        assert_eq!(params.split[Effect::Delay as usize].mode, SplitMode::Side);
        assert_eq!(params.split[Effect::Reverb as usize].mode, SplitMode::High);
        assert_eq!(params.split[Effect::Reverb as usize].crossover_hz, 250.0);
        // Untouched effects keep the Full default and 1 kHz crossover.
        assert_eq!(params.split[Effect::Chorus as usize].mode, SplitMode::Full);
        assert_eq!(
            params.split[Effect::Chorus as usize].crossover_hz,
            DEFAULT_SPLIT_CROSSOVER_HZ
        );
    }

    #[test]
    fn master_mixer_fields_map() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{
                  "bus_a_on": 1.0,
                  "bus_a_send": 0.5,
                  "bus_a_return_db": -3.0,
                  "bus_a_output": 1.0,
                  "bus_b_send": 0.2}}"#,
        );
        let master = master_from_preset(&preset);
        assert!(master.mixer.bus_a.on);
        assert_eq!(master.mixer.bus_a.send, 0.5);
        assert_eq!(master.mixer.bus_a.return_gain_db, -3.0);
        assert_eq!(master.mixer.bus_a.output, BusOutput::MainChain);
        assert!(!master.mixer.bus_b.on);
        assert_eq!(master.mixer.bus_b.send, 0.2);
        assert_eq!(master.mixer.bus_b.output, BusOutput::Master);
        assert_eq!(master.mixer.bus_b.return_gain_db, 0.0);
    }

    #[test]
    fn macros_extend_to_eight() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"macro_control_2": 0.4, "macro_control_6": 0.9}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        assert_eq!(params.macros[1], 0.4);
        assert_eq!(params.macros[5], 0.9);
        assert_eq!(params.macros[7], 0.0);
    }

    #[test]
    fn filter_settings_apply() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"filter_1_on": 1.0, "filter_1_cutoff": 80.0,
                            "filter_1_model": 3.0, "filter_2_filter_input": 1.0}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        assert!(params.filters[0].params.on);
        assert_eq!(params.filters[0].params.state.midi_cutoff.lane(0), 80.0);
        assert_eq!(params.filters[0].params.model, FilterModel::Digital);
        assert_eq!(params.filter_routing, FilterRouting::SerialForward);
    }

    #[test]
    fn quadratic_and_exponential_scales_are_applied() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"osc_1_unison_detune": 3.0, "osc_1_level": 0.5,
                            "lfo_1_smooth_time": -2.0, "lfo_1_fade_time": 1.5}}"#,
        );
        let params = kernel_params_from_preset(&preset);
        // Table Quadratic: passed raw (3.0); the kernel squares the
        // modulated sum (see synth_voice.rs `modulated_osc_params`).
        assert!((params.oscillators[0].params.unison_detune.lane(0) - 3.0).abs() < 1e-5);
        let init = preset_default();
        let defaults = kernel_params_from_preset(&init);
        assert!((defaults.oscillators[0].params.unison_detune.lane(0) - 4.472).abs() < 1e-3);
        // osc level is squared by the DSP itself: passed raw.
        assert_eq!(params.oscillators[0].params.amplitude.lane(0), 0.5);
        // smooth_time is Exponential (log2 s): -2 -> 0.25 s; fade_time linear.
        assert!((params.lfos[0].params.smooth_time.lane(0) - 0.25).abs() < 1e-6);
        assert_eq!(params.lfos[0].params.fade_time.lane(0), 1.5);
    }

    #[test]
    fn pitch_bend_range_reads_the_range_parameter() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"pitch_bend_range": 12.0, "pitch_wheel": 0.7}}"#,
        );
        assert_eq!(kernel_params_from_preset(&preset).pitch_bend_range, 12.0);
        // Table default 2, wheel position ignored.
        let init = preset_default();
        assert_eq!(kernel_params_from_preset(&init).pitch_bend_range, 2.0);
        let wheel_only = self::preset(
            r#"{"synth_version":"1.0.7","preset_name":"t","settings":{"pitch_wheel": 0.9}}"#,
        );
        assert_eq!(kernel_params_from_preset(&wheel_only).pitch_bend_range, 2.0);
    }

    #[test]
    fn line_shapes_are_clamped_and_sources_validated() {
        // 300 points claimed: clamped to MAX_POINTS without panicking.
        let mut points = Vec::new();
        for i in 0..300 {
            points.push(i as f32 / 299.0);
            points.push(0.5);
        }
        let shape = LineShape {
            num_points: 300,
            points,
            powers: vec![0.0; 300],
            name: None,
            smooth: false,
            extra: Default::default(),
        };
        let generator = line_shape_to_generator(&shape);
        assert_eq!(generator.resolution(), 2048);
        // Claimed points beyond the data: only the data counts.
        let short = LineShape { num_points: 50, points: vec![0.0, 1.0, 1.0, 0.0], ..LineShape::linear() };
        let _ = line_shape_to_generator(&short);
        // A preset lfo shape goes through the same path.
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"lfos":[{"num_points": 999, "points":[0,1,1,0], "powers":[0,0]}]}}"#,
        );
        let _ = kernel_params_from_preset(&preset);

        assert_eq!(parse_mod_source("lfo_0"), Option::None);
        assert_eq!(parse_mod_source("lfo_1"), Some(ModSource::Lfo(0)));
        assert_eq!(parse_mod_source("lfo_12"), Some(ModSource::Lfo(11)));
        assert_eq!(parse_mod_source("lfo_13"), Option::None);
        assert_eq!(parse_mod_source("env_0"), Option::None);
        assert_eq!(parse_mod_source("macro_control_8"), Some(ModSource::Macro(7)));
        assert_eq!(parse_mod_source("macro_control_9"), Option::None);
        assert_eq!(parse_mod_dest("osc_4_level"), Some(ModDest::OscLevel(3)));
    }

    #[test]
    fn load_preset_reports_drops_and_migrations() {
        let text = r#"{"synth_version":"0.8.0","preset_name":"old",
            "settings":{
              "filter_1_model": 4.0, "filter_1_blend": 1.0,
              "mystery_knob": 0.3,
              "modulations":[
                {"source":"lfo_1","destination":"filter_1_cutoff"},
                {"source":"lfo_0","destination":"filter_1_cutoff"},
                {"source":"env_1","destination":"no_such_param"},
                {"source":"lfo_2","destination":"osc_1_level",
                 "line_mapping":{"num_points":3,"points":[0,1,0.5,0.2,1,0],"powers":[0,0,0]}}
              ]}}"#;
        let (preset, report) = load_preset(text).unwrap();
        assert_eq!(preset.synth_version, spinwave_params::migrate::CURRENT_FORMAT_VERSION);
        assert_eq!(report.migrated_from.as_deref(), Some("0.8.0"));
        assert!(report.migrations.iter().any(|m| m.starts_with("0.9.0")));
        assert_eq!(preset.settings.parameter("filter_1_blend"), Some(0.0));
        assert_eq!(report.unknown_params, vec!["mystery_knob".to_string()]);
        assert_eq!(
            report.ignored_connections,
            vec!["lfo_0 -> filter_1_cutoff".to_string(), "env_1 -> no_such_param".to_string()]
        );
        assert!(report.notes.is_empty(), "{:?}", report.notes);
        assert!(!report.is_clean());

        // The drawn `line_mapping` becomes the connection's remap curve.
        let connections = connections_from_preset(&preset);
        let remapped = connections
            .iter()
            .find(|c| c.source == ModSource::Lfo(1))
            .expect("lfo_2 -> osc_1_level survived");
        assert!(remapped.transform.remap.is_some());
        assert!(connections
            .iter()
            .find(|c| c.source == ModSource::Lfo(0))
            .is_some_and(|c| c.transform.remap.is_none()));

        // A BOM and a current preset: clean report.
        let (_, clean) = load_preset("\u{feff}{\"synth_version\":\"1.0.7\",\"settings\":{}}").unwrap();
        assert!(clean.is_clean());
        assert!(load_preset("not json").is_err());
    }

    #[test]
    fn embedded_sample_and_materials_are_built() {
        let mono: Vec<f32> = (0..200).map(|i| (i as f32 * 0.3).sin()).collect();
        let payload = SampleJson::from_channels("bell", &mono, Option::None, 22050);
        let mut preset = preset_default();
        preset.settings.sample = Some(serde_json::to_value(&payload).unwrap());
        materials::set_slot_sample_json(&mut preset, 1, payload.clone());
        let mut report = LoadReport::default();
        let built = BuiltPatch::build(&preset, 3, 88_200, &mut report);
        assert_eq!(built.kernels.len(), 3);
        assert!(built.connections.is_empty());
        assert!(built.global_sample.is_some());
        assert_eq!(built.global_sample.as_ref().unwrap().original_length(), 200);
        assert_eq!(built.global_sample.as_ref().unwrap().sample_rate(), 22050);
        assert_eq!(built.samples.len(), 1);
        assert_eq!(built.samples[0].0, 1);
        assert_eq!(built.samples[0].1.name, "bell");
        assert!(built.multisamples.is_empty());
        assert!(report.is_clean());

        assert_eq!(BuiltPatch::kernel_count(4, 8), 4);
        assert_eq!(BuiltPatch::kernel_count(2, 16), 8);
        assert_eq!(BuiltPatch::kernel_count(0, 64), 32);
        assert!(BuiltPatch::kernel_count(0, 1000) <= MAX_KERNELS);
    }
}

#[cfg(test)]
mod read_audit {
    //! Maintenance point 2 of `notes/operations-design.md`: every table
    //! parameter must be READ somewhere on the patch → engine path, or be
    //! on the list below with a reason. Not a source scan — the reader
    //! records what it was asked for while a preset that sets every
    //! parameter is applied.

    use super::*;
    use spinwave_engine::engine::SoundEngine;

    /// Table parameters the reader legitimately never asks for, each with
    /// its reason. Two kinds, kept apart: names that are not engine
    /// values at all, and controls the engine does not implement yet —
    /// the second kind is a finding list, like `KNOWN_DIVERGENCES` on the
    /// golden bench, and a name leaves it when the control is wired.
    /// Matched by family: `N` stands for any slot index, `X` for `a`/`b`.
    const NEVER_READ: &[(&str, &str)] = &[
        // -- not engine values
        ("osc_N_view_2d", "GUI: how the oscillator is drawn"),
        ("view_spectrogram", "GUI: which analyser is shown"),
        ("bypass", "the host's bypass, applied by the plugin wrapper, not by the patch"),
        ("beats_per_minute", "the host transport's tempo (`SoundEngine::set_bpm`), never a patch value"),
        ("mod_wheel", "a live MIDI controller value, not a patch value"),
        ("pitch_wheel", "a live MIDI controller value, not a patch value"),
        ("mpe_enabled", "MIDI input configuration, the plugin wrapper's"),
        ("oversampling", "read by the offline session (`Session::render_samples_probed`); the plugin follows its host"),
        ("compressor_low_band_unused", "the reference's own unused parameter"),
        ("bus_X_compressor_low_band_unused", "the reference's own unused parameter"),
        ("filter_N_osc1_input", "pre-1.0 routing flag, converted to osc_N_destination by migrate.rs"),
        ("filter_N_osc2_input", "pre-1.0 routing flag, converted by migrate.rs"),
        ("filter_N_osc3_input", "pre-1.0 routing flag, converted by migrate.rs"),
        ("filter_N_sample_input", "pre-1.0 routing flag, converted by migrate.rs"),
        ("filter_fx_osc1_input", "the bus filter has no per-oscillator input in the reference either (FilterFxModule takes the mix)"),
        ("filter_fx_osc2_input", "as above"),
        ("filter_fx_osc3_input", "as above"),
        ("filter_fx_sample_input", "as above"),
        ("filter_fx_filter_input", "as above: no second filter to chain from"),
        ("bus_X_filter_fx_osc1_input", "as above, on the send buses"),
        ("bus_X_filter_fx_osc2_input", "as above"),
        ("bus_X_filter_fx_osc3_input", "as above"),
        ("bus_X_filter_fx_sample_input", "as above"),
        ("bus_X_filter_fx_filter_input", "as above"),
        // -- NOT IMPLEMENTED: findings of the read audit, 2026-09-12
        ("sub_on", "FINDING: the reference's sub oscillator has no engine here; every sub_* control is dropped"),
        ("sub_level", "FINDING: sub oscillator, see sub_on"),
        ("sub_pan", "FINDING: sub oscillator, see sub_on"),
        ("sub_transpose", "FINDING: sub oscillator, see sub_on"),
        ("sub_transpose_quantize", "FINDING: sub oscillator, see sub_on"),
        ("sub_tune", "FINDING: sub oscillator, see sub_on"),
        ("sub_waveform", "FINDING: sub oscillator, see sub_on"),
        ("sub_direct_out", "FINDING: sub oscillator, see sub_on"),
        ("osc_N_smooth_interpolation", "FINDING: the oscillator has no smooth-frame-interpolation mode (`kSmoothlyInterpolate`)"),
        ("lfo_N_keytrack_transpose", "FINDING: the keytracked LFO rate (sync index 4) falls back to free-running"),
        ("lfo_N_keytrack_tune", "FINDING: keytracked LFO rate, see lfo_N_keytrack_transpose"),
        ("random_N_keytrack_transpose", "FINDING: keytracked random LFO rate, as for the LFOs"),
        ("random_N_keytrack_tune", "FINDING: keytracked random LFO rate, as for the LFOs"),
    ];

    /// `lfo_3_sync` → `lfo_N_sync`, `bus_b_x` → `bus_X_x`: a slot index is
    /// a whole `_<digits>_` segment (so `view_2d` keeps its 2).
    fn family(name: &str) -> String {
        let parts: Vec<&str> = name.split('_').collect();
        let last = parts.len() - 1;
        let mapped: Vec<&str> = parts
            .iter()
            .enumerate()
            .map(|(i, p)| if i > 0 && i < last && !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) { "N" } else { p })
            .collect();
        mapped.join("_").replace("bus_a_", "bus_X_").replace("bus_b_", "bus_X_")
    }

    #[test]
    fn every_table_parameter_is_read_by_the_engine() {
        let table = parameters();
        let mut preset = Preset::default();
        for details in table.iter() {
            // Away from the default, so a read that depends on a value
            // (an engine switch, a model) sees the non-default too; and the
            // switches on, so the modules behind them are read.
            let value = if details.name.ends_with("_on") || details.name.ends_with("_enabled") {
                1.0
            } else if details.default_value != details.min {
                details.min
            } else {
                details.max
            };
            preset.settings.values.insert(details.name.clone(), serde_json::Value::from(value as f64));
        }
        // A connection in every slot, so the per-slot modulation controls
        // are asked for too.
        for i in 0..spinwave_engine::kernel::mod_matrix::MAX_MODULATION_CONNECTIONS {
            preset.settings.modulations.push(spinwave_params::preset::ModulationConnection {
                source: format!("lfo_{}", i % 8 + 1),
                destination: "filter_1_cutoff".into(),
                ..Default::default()
            });
        }
        let mut engine = SoundEngine::with_pool(44100, 2);
        let reads = record_reads(|| {
            let _ = crate::apply_preset_with(&preset, &mut engine, &mut |_| None);
        });
        assert!(reads.len() > 100, "the reader recorded only {} names: is recording wired?", reads.len());
        let mut unread: Vec<&str> = table
            .iter()
            .map(|d| d.name.as_str())
            .filter(|name| !reads.contains(*name))
            .filter(|name| !NEVER_READ.iter().any(|(n, _)| *n == family(name)))
            .collect();
        unread.sort();
        let stale: Vec<&str> = NEVER_READ
            .iter()
            .filter(|(n, _)| reads.iter().any(|r| family(r) == *n))
            .map(|(n, _)| *n)
            .collect();
        assert!(stale.is_empty(), "listed as never read but read: {stale:?}");
        assert!(unread.is_empty(), "{} table parameters the engine never reads:\n  {}", unread.len(), unread.join("\n  "));
    }
}
