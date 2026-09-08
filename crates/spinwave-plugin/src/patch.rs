//! Loads a `.vital` preset into the Spinwave voice kernel.
//!
//! The preset's `settings` map stores engine values; most feed the DSP
//! param structs directly. Exceptions, mirroring the reference:
//! envelope times are stored as the quartic root of seconds, LFO and
//! random-LFO frequencies as log2(Hz).

use spinwave_dsp::modulators::{LineGenerator, RandomLfoStyle};
use spinwave_dsp::oscillator::{DistortionType, SpectralMorph, UnisonStackType};
use spinwave_engine::kernel::mod_matrix::{
    Connection, ModDest, ModSource, NUM_LFOS, NUM_OSCILLATORS, NUM_RANDOM_LFOS,
};
use spinwave_engine::kernel::voice_filter::FilterModel;
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
        .or_else(|| match name {
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
        let p = |suffix: &str| format!("filter_{}_{}", i + 1, suffix);
        let section = &mut params.filters[i];
        section.params.on = reader.on(&p("on"));
        section.params.model = FilterModel::from_index(reader.get(&p("model")) as i32);
        section.params.mix = reader.poly(&p("mix"));
        section.keytrack = reader.get(&p("keytrack"));
        let state = &mut section.params.state;
        state.midi_cutoff = reader.poly(&p("cutoff"));
        state.resonance_percent = reader.poly(&p("resonance"));
        state.set_drive_db(reader.poly(&p("drive")));
        state.set_pass_blend(reader.poly(&p("blend")));
        state.transpose = reader.poly(&p("blend_transpose"));
        state.style = spinwave_dsp::filters::FilterStyle::from_index(
            reader.get(&p("style")) as i32,
        );
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
