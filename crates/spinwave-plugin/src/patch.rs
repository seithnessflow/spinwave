//! Loads a `.vital` preset into the Spinwave voice kernel.
//!
//! The preset's `settings` map stores engine values; most feed the DSP
//! param structs directly. Exceptions, mirroring the reference:
//! envelope times are stored as the quartic root of seconds, LFO and
//! random-LFO frequencies as log2(Hz).

use spinwave_dsp::effects::{BandOptions, DelayStyle, DistortionType as FxDistortionType};
use spinwave_dsp::modulators::{LineGenerator, RandomLfoStyle};
use spinwave_dsp::oscillator::{DistortionType, SpectralMorph, UnisonStackType};
use spinwave_engine::allocator::{VoiceOverride, VoicePriority};
use spinwave_engine::engine::{
    decode_order, EffectsParams, StereoMode, SyncMode, SyncedFrequency,
};
use spinwave_engine::kernel::mod_matrix::{
    Connection, ModDest, ModSource, NUM_LFOS, NUM_OSCILLATORS, NUM_RANDOM_LFOS,
};
use spinwave_engine::kernel::voice_filter::{FilterModel, VoiceFilterParams};
use spinwave_engine::kernel::{FilterRouting, KernelParams, ProducerDestination};
use spinwave_engine::modulation::ModulationTransform;
use spinwave_params::preset::{LineShape, Preset};
use spinwave_params::{parameters, ParamDetails};
use spinwave_poly::PolyF32;

use spinwave_engine::kernel::mod_matrix::NUM_ENVELOPES;

/// Settings reader with table-backed defaults.
struct Reader<'a> {
    preset: &'a Preset,
}

impl Reader<'_> {
    fn get(&self, name: &str) -> f32 {
        if let Some(value) = self.preset.settings.parameter(name) {
            return value;
        }
        parameters()
            .lookup(name)
            .map(|d: &ParamDetails| d.default_value)
            .unwrap_or(0.0)
    }

    fn poly(&self, name: &str) -> PolyF32 {
        PolyF32::splat(self.get(name))
    }

    fn on(&self, name: &str) -> bool {
        self.get(name) > 0.5
    }
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
}

fn line_shape_to_generator(shape: &LineShape) -> LineGenerator {
    let mut generator = LineGenerator::new(2048);
    let num_points = (shape.num_points as usize).min(shape.points.len() / 2);
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

fn parse_mod_source(name: &str) -> Option<ModSource> {
    let indexed = |prefix: &str| -> Option<usize> {
        name.strip_prefix(prefix)?.parse::<usize>().ok().map(|n| n - 1)
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
        return (i < 4).then_some(ModSource::Macro(i));
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

fn parse_mod_dest(name: &str) -> Option<ModDest> {
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

    osc("level", ModDest::OscLevel)
        .or_else(|| osc("transpose", ModDest::OscTranspose))
        .or_else(|| osc("tune", ModDest::OscTune))
        .or_else(|| osc("wave_frame", ModDest::OscFrame))
        .or_else(|| osc("pan", ModDest::OscPan))
        .or_else(|| osc("unison_detune", ModDest::OscUnisonDetune))
        .or_else(|| osc("distortion_amount", ModDest::OscDistortionAmount))
        .or_else(|| osc("spectral_morph_amount", ModDest::OscSpectralMorphAmount))
        .or_else(|| osc("phase", ModDest::OscPhase))
        .or_else(|| filter("cutoff", ModDest::FilterCutoff))
        .or_else(|| filter("resonance", ModDest::FilterResonance))
        .or_else(|| filter("drive", ModDest::FilterDrive))
        .or_else(|| filter("blend", ModDest::FilterBlend))
        .or_else(|| filter("mix", ModDest::FilterMix))
        .or_else(|| env("attack", ModDest::EnvAttack))
        .or_else(|| env("decay", ModDest::EnvDecay))
        .or_else(|| env("sustain", ModDest::EnvSustain))
        .or_else(|| env("release", ModDest::EnvRelease))
        .or_else(|| lfo("frequency", ModDest::LfoFrequency))
        .or(match name {
            "sample_level" => Some(ModDest::SampleLevel),
            "volume" => Some(ModDest::VolumeAmp),
            _ => Option::None,
        })
}

fn destination_scale(name: &str) -> f32 {
    parameters()
        .lookup(name)
        .map(|d| d.max - d.min)
        .unwrap_or(1.0)
}

/// Builds full kernel params (and modulation matrix) from a preset.
pub fn kernel_params_from_preset(preset: &Preset) -> KernelParams {
    let reader = Reader { preset };
    let mut params = KernelParams::default();

    for i in 0..NUM_OSCILLATORS {
        let p = |suffix: &str| format!("osc_{}_{}", i + 1, suffix);
        let section = &mut params.oscillators[i];
        section.on = reader.on(&p("on"));
        section.destination =
            ProducerDestination::from_index(reader.get(&p("destination")) as i32);
        let osc = &mut section.params;
        osc.amplitude = reader.poly(&p("level"));
        osc.transpose = reader.poly(&p("transpose"));
        osc.transpose_quantize = reader.get(&p("transpose_quantize")) as u32;
        osc.tune = reader.poly(&p("tune"));
        osc.pan = reader.poly(&p("pan"));
        osc.wave_frame = reader.poly(&p("wave_frame"));
        osc.frame_spread = reader.poly(&p("frame_spread"));
        osc.unison_voices = reader.get(&p("unison_voices")).max(1.0) as usize;
        osc.unison_detune = reader.poly(&p("unison_detune"));
        osc.detune_power = reader.poly(&p("detune_power"));
        osc.detune_range = reader.poly(&p("detune_range"));
        osc.blend = reader.poly(&p("unison_blend"));
        osc.stereo_spread = reader.poly(&p("stereo_spread"));
        osc.phase = reader.poly(&p("phase"));
        osc.random_phase = reader.poly(&p("random_phase"));
        osc.distortion_phase = reader.poly(&p("distortion_phase"));
        osc.midi_track = reader.on(&p("midi_track"));
        osc.spectral_unison = reader.on(&p("spectral_unison"));
        osc.stack_style = stack_type_from_index(reader.get(&p("stack_style")) as i32);
        osc.distortion_type =
            distortion_type_from_index(reader.get(&p("distortion_type")) as i32);
        osc.distortion_amount = reader.poly(&p("distortion_amount"));
        osc.distortion_spread = reader.poly(&p("distortion_spread"));
        osc.spectral_morph_type =
            spectral_morph_from_index(reader.get(&p("spectral_morph_type")) as i32);
        osc.spectral_morph_amount = reader.poly(&p("spectral_morph_amount"));
        osc.spectral_morph_spread = reader.poly(&p("spectral_morph_spread"));
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

    for i in 0..2 {
        let prefix = format!("filter_{}_", i + 1);
        let section = &mut params.filters[i];
        fill_filter_params(&reader, &prefix, &mut section.params);
        section.keytrack = reader.get(&format!("{prefix}keytrack"));
    }

    // Serial routing from the filter-input switches.
    if reader.on("filter_2_filter_input") {
        params.filter_routing = FilterRouting::SerialForward;
    } else if reader.on("filter_1_filter_input") {
        params.filter_routing = FilterRouting::SerialBackward;
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
        let p = |suffix: &str| format!("lfo_{}_{}", i + 1, suffix);
        let lfo = &mut params.lfos[i];
        lfo.params.frequency = exp_frequency(reader.get(&p("frequency")));
        lfo.params.phase = reader.poly(&p("phase"));
        lfo.params.stereo_phase = reader.poly(&p("stereo"));
        lfo.params.fade_time = reader.poly(&p("fade_time"));
        lfo.params.delay_time = reader.poly(&p("delay_time"));
        lfo.params.smooth_mode = reader.on(&p("smooth_mode"));
        lfo.params.smooth_time = reader.poly(&p("smooth_time"));
        if let Some(shape) = preset.settings.lfos.get(i) {
            lfo.shape = line_shape_to_generator(shape);
        }
    }

    for i in 0..NUM_RANDOM_LFOS {
        let p = |suffix: &str| format!("random_{}_{}", i + 1, suffix);
        let random = &mut params.random_lfos[i];
        random.frequency = exp_frequency(reader.get(&p("frequency")));
        random.style = random_style_from_index(reader.get(&p("style")) as i32);
        random.stereo = reader.on(&p("stereo"));
    }

    params.velocity_track = reader.get("velocity_track");
    params.pitch_bend_range = reader.get("pitch_wheel").max(2.0);
    for i in 0..4 {
        params.macros[i] = reader.get(&format!("macro_control_{}", i + 1));
    }

    params
}

/// Builds the modulation matrix from the preset's connection list.
pub fn connections_from_preset(preset: &Preset) -> Vec<Connection> {
    let reader = Reader { preset };
    let mut connections = Vec::new();
    for (index, modulation) in preset.settings.modulations.iter().enumerate() {
        let (Some(source), Some(dest)) = (
            parse_mod_source(&modulation.source),
            parse_mod_dest(&modulation.destination),
        ) else {
            continue;
        };
        let n = index + 1;
        let mut transform = ModulationTransform::with_amount(
            reader.get(&format!("modulation_{n}_amount")),
            destination_scale(&modulation.destination),
        );
        transform.power = PolyF32::splat(reader.get(&format!("modulation_{n}_power")));
        transform.bipolar = reader.on(&format!("modulation_{n}_bipolar"));
        transform.stereo = reader.on(&format!("modulation_{n}_stereo"));
        transform.bypass = reader.on(&format!("modulation_{n}_bypass"));
        connections.push(Connection { source, dest, transform });
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
    let reader = Reader { preset };
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
    params.chorus_sync = synced_frequency(&reader, "chorus");

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
    params.delay_sync = synced_frequency(&reader, "delay");
    params.delay_aux_sync = synced_frequency(&reader, "delay_aux");

    params.distortion_on = reader.on("distortion_on");
    params.distortion_type = fx_distortion_type_from_index(reader.get("distortion_type") as i32);
    params.distortion_drive_db = reader.get("distortion_drive");
    params.distortion_mix = reader.get("distortion_mix");

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
    fill_filter_params(&reader, "filter_fx_", &mut params.filter_fx);

    params.flanger_on = reader.on("flanger_on");
    let flanger = &mut params.flanger;
    flanger.center_midi = reader.poly("flanger_center");
    flanger.feedback = reader.poly("flanger_feedback");
    // Table range is [0, 0.5]: 0.5 is the 50/50 equal-power point.
    flanger.wet = reader.poly("flanger_dry_wet");
    flanger.mod_depth = reader.poly("flanger_mod_depth");
    flanger.phase_offset = reader.poly("flanger_phase_offset");
    params.flanger_sync = synced_frequency(&reader, "flanger");

    params.phaser_on = reader.on("phaser_on");
    let phaser = &mut params.phaser;
    phaser.mix = reader.poly("phaser_dry_wet");
    phaser.feedback_gain = reader.poly("phaser_feedback");
    phaser.center_midi = reader.poly("phaser_center");
    phaser.mod_depth = reader.poly("phaser_mod_depth");
    phaser.phase_offset = reader.poly("phaser_phase_offset");
    phaser.blend = reader.poly("phaser_blend");
    params.phaser_sync = synced_frequency(&reader, "phaser");

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

    params
}

/// Master / output-section settings mapped from a preset. Callers apply them
/// to `SoundEngine::master`, the polyphony and the voice allocator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MasterFromPreset {
    /// Master volume in dB. The `volume` setting is stored square-root
    /// scaled with a -80 post offset (`cr::Root` in the reference):
    /// `dB = sqrt(stored) - 80`, e.g. the default 5473.0404 → ~-6.02 dB.
    pub volume_db: f32,
    /// `stereo_routing` in `[0, 1]`.
    pub stereo_routing: f32,
    pub stereo_mode: StereoMode,
    /// Voice count in `1..=32`.
    pub polyphony: usize,
    pub legato: bool,
    pub voice_priority: VoicePriority,
    pub voice_override: VoiceOverride,
}

/// Reads the master / voice-allocation settings from a preset.
pub fn master_from_preset(preset: &Preset) -> MasterFromPreset {
    let reader = Reader { preset };
    let volume_post_offset = parameters()
        .lookup("volume")
        .map(|d| d.post_offset)
        .unwrap_or(-80.0);
    MasterFromPreset {
        volume_db: reader.get("volume").max(0.0).sqrt() + volume_post_offset,
        stereo_routing: reader.get("stereo_routing"),
        stereo_mode: if reader.on("stereo_mode") {
            StereoMode::Rotate
        } else {
            StereoMode::Spread
        },
        polyphony: reader.get("polyphony").max(1.0) as usize,
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
        let out = t.process_control(PolyF32::splat(1.0), Option::None);
        assert!((out.scaled.lane(0) - 0.5 * 0.5 * 128.0).abs() < 1.0);
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

    #[test]
    fn effect_chain_order_decodes_from_settings() {
        let preset = preset(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"effect_chain_order": 1.0}}"#,
        );
        let params = effects_params_from_preset(&preset);
        // Code 1 is a single inversion at the last position.
        let mut expected = spinwave_engine::engine::DEFAULT_ORDER;
        expected.swap(7, 8);
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
}
