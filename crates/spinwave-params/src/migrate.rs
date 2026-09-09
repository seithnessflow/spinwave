//! Preset version migrations: a port of `LoadSave::updateFromOldVersion`
//! (`src/common/load_save.cpp` in the C++ reference). Every step is gated
//! on the preset's `synth_version` exactly like the C++, so a preset saved
//! by any Vital release converts the same way here.
//!
//! Where the C++ reads a key that an old preset may lack (nlohmann would
//! throw), the table default is used instead. Two C++ quirks are noted
//! inline: the 0.3.0 step writes the (unused) `decay_time` key, kept for
//! fidelity; the 0.4.7 FM remap loop skips its index increment on
//! zero-amount connections, which is NOT reproduced (it would corrupt the
//! following connections' amounts).

use serde_json::{Map, Value};

use crate::base64;
use crate::constants::{NUM_ENVELOPES, NUM_LFOS, NUM_RANDOM_LFOS};
use crate::preset::{LineShape, ModulationConnection, Preset};
use crate::table::parameters;

/// `synth_version` written for presets produced by this code base (all
/// migrations below target older versions).
pub const CURRENT_FORMAT_VERSION: &str = "1.0.7";

/// Parses `"a.b.c"` (missing parts read as 0). `None` when the first part
/// is not a number.
#[must_use]
pub fn parse_version(text: &str) -> Option<(u32, u32, u32)> {
    let mut parts = text.trim().split('.').map(|p| p.trim().parse::<u32>());
    let major = parts.next()?.ok()?;
    let minor = parts.next().and_then(Result::ok).unwrap_or(0);
    let patch = parts.next().and_then(Result::ok).unwrap_or(0);
    Some((major, minor, patch))
}

/// `compareVersionStrings(a, b) < 0`.
fn version_before(version: (u32, u32, u32), gate: &str) -> bool {
    match parse_version(gate) {
        Some(gate) => version < gate,
        None => false,
    }
}

/// Lehmer-code order encoding (`vital::utils::encodeOrderToFloat`).
#[must_use]
pub fn encode_order(order: &[usize]) -> f32 {
    let mut code: u32 = 0;
    for i in 1..order.len() {
        let index = order[..i].iter().filter(|&&earlier| order[i] < earlier).count() as u32;
        code *= i as u32 + 1;
        code += index;
    }
    code as f32
}

/// Lehmer-code order decoding (`vital::utils::decodeFloatToOrder`).
#[must_use]
pub fn decode_order(float_code: f32, size: usize) -> Vec<usize> {
    let mut code = float_code.max(0.0) as u32;
    let mut order: Vec<usize> = (0..size).collect();
    for i in 0..size {
        let remaining = (size - i) as u32;
        let index = remaining as usize - 1;
        let inversions = (code % remaining) as usize;
        code /= remaining;
        let placement = order[index - inversions];
        for j in index - inversions..index {
            order[j] = order[j + 1];
        }
        order[index] = placement;
    }
    order
}

/// Working view over the preset parts the C++ mutates.
struct Migration<'a> {
    settings: &'a mut Map<String, Value>,
    modulations: &'a mut Vec<ModulationConnection>,
    sample: &'a mut Option<Value>,
    wavetables: &'a mut Option<Value>,
    applied: Vec<String>,
}

impl Migration<'_> {
    fn has(&self, name: &str) -> bool {
        self.settings.contains_key(name)
    }

    /// The stored value, else the table default, else 0 (the C++ would
    /// throw on a missing key; old presets are complete in practice).
    fn get(&self, name: &str) -> f32 {
        self.settings
            .get(name)
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .or_else(|| parameters().lookup(name).map(|d| d.default_value))
            .unwrap_or(0.0)
    }

    fn set(&mut self, name: &str, value: f32) {
        self.settings.insert(name.to_string(), Value::from(f64::from(value)));
    }

    fn rename_destinations(&mut self, renames: &[(&str, &str)]) {
        for modulation in self.modulations.iter_mut() {
            for (from, to) in renames {
                if modulation.destination == *from {
                    modulation.destination = (*to).to_string();
                }
            }
        }
    }

    /// Scales `modulation_N_amount` of every connection into `destination`
    /// (N is the 1-based slot index like the C++ loops).
    fn scale_amounts_into(&mut self, destination: &str, multiply: f32) {
        let indices: Vec<usize> = self
            .modulations
            .iter()
            .enumerate()
            .filter(|(_, m)| m.destination == destination)
            .map(|(i, _)| i + 1)
            .collect();
        for index in indices {
            let name = format!("modulation_{index}_amount");
            let amount = self.get(&name);
            self.set(&name, amount * multiply);
        }
    }

    fn done(&mut self, label: &str) {
        self.applied.push(label.to_string());
    }
}

/// Applies every migration older than the preset's `synth_version`, in the
/// C++ order. Returns the labels of the steps applied (empty when the
/// preset was current). Afterwards `synth_version` is set to
/// [`CURRENT_FORMAT_VERSION`] so a re-load does not convert twice.
pub fn upgrade(preset: &mut Preset) -> Vec<String> {
    let has_sub_octave = preset.settings.values.contains_key("sub_octave");
    let version = match parse_version(&preset.synth_version) {
        Some(version) => version,
        // Unparsable version: the C++ compare yields 0, no update.
        None if has_sub_octave => (0, 0, 0),
        None => return Vec::new(),
    };
    if !version_before(version, "0.9.0") && !has_sub_octave {
        return Vec::new();
    }

    let Preset { settings, .. } = preset;
    let mut m = Migration {
        settings: &mut settings.values,
        modulations: &mut settings.modulations,
        sample: &mut settings.sample,
        wavetables: &mut settings.wavetables,
        applied: Vec::new(),
    };
    let before = |gate: &str| version_before(version, gate);

    if before("0.2.0") || has_sub_octave {
        let sub_waveform = m.get("sub_waveform") as i32;
        let swapped = match sub_waveform {
            4 => 5,
            5 => 4,
            other => other,
        };
        m.set("sub_waveform", swapped as f32);
        let sub_octave = m.get("sub_octave") as i32;
        m.set("sub_transpose", 12.0 * sub_octave as f32);

        let osc_1 = m.get("osc_1_filter_routing") as i32;
        let osc_2 = m.get("osc_2_filter_routing") as i32;
        let sample = m.get("sample_filter_routing") as i32;
        let sub = m.get("sub_filter_routing") as i32;
        m.set("filter_1_osc1_input", (1 - osc_1) as f32);
        m.set("filter_1_osc2_input", (1 - osc_2) as f32);
        m.set("filter_1_sample_input", (1 - sample) as f32);
        m.set("filter_1_sub_input", (1 - sub) as f32);
        m.set("filter_2_osc1_input", osc_1 as f32);
        m.set("filter_2_osc2_input", osc_2 as f32);
        m.set("filter_2_sample_input", sample as f32);
        m.set("filter_2_sub_input", sub as f32);

        for name in ["filter_1_style", "filter_2_style"] {
            let style = m.get(name) as i32;
            let swapped = match style {
                2 => 3,
                3 => 2,
                other => other,
            };
            m.set(name, swapped as f32);
        }
        m.done("0.2.0 sub octave / filter routing / filter styles");
    }

    if before("0.2.1") {
        for i in 1..=NUM_ENVELOPES {
            if !m.has(&format!("env_{i}_attack")) {
                break;
            }
            for stage in ["attack", "decay", "release"] {
                let name = format!("env_{i}_{stage}");
                let value = m.get(&name);
                m.set(&name, value.max(0.0).powf(1.0 / 1.5));
            }
        }
        if let Some(tables) = m.settings.remove("wave_tables") {
            *m.wavetables = Some(tables);
        }
        m.done("0.2.1 envelope curve / wave_tables key");
    }

    if before("0.2.4") {
        let portamento_type = m.get("portamento_type") as i32;
        m.set("portamento_force", (portamento_type - 1).max(0) as f32);
        if portamento_type == 0 {
            m.set("portamento_time", -10.0);
        }
        m.done("0.2.4 portamento type");
    }

    if before("0.2.5") {
        for i in 1..=NUM_ENVELOPES {
            if !m.has(&format!("env_{i}_attack")) {
                break;
            }
            for stage in ["attack", "decay", "release"] {
                let name = format!("env_{i}_{stage}");
                let value = m.get(&name);
                m.set(&name, value.max(0.0).powf(0.75));
            }
        }
        m.done("0.2.5 envelope quartic times");
    }

    if before("0.2.6") {
        for i in 1..=NUM_LFOS {
            m.set(&format!("lfo_{i}_fade_time"), 0.0);
            m.set(&format!("lfo_{i}_delay_time"), 0.0);
        }
        m.done("0.2.6 lfo fade/delay reset");
    }

    if before("0.2.7") {
        const ADJUSTMENT: f32 = std::f32::consts::FRAC_1_SQRT_2;
        for name in ["osc_1_level", "osc_2_level", "sub_level", "sample_level"] {
            let level = m.get(name);
            m.set(name, (ADJUSTMENT * level * level).sqrt());
        }
        m.done("0.2.7 levels sqrt");
    }

    if before("0.3.0") {
        let damping = m.get("reverb_damping");
        let feedback = m.get("reverb_feedback");
        // The C++ writes the unused `decay_time` key here (not
        // `reverb_decay_time`); kept verbatim so the result matches Vital.
        m.set("decay_time", (feedback - 0.8) * 10.0);
        m.set("reverb_high_shelf_gain", -damping * 4.0);
        m.set("reverb_pre_high_cutoff", 128.0);
        m.rename_destinations(&[
            ("reverb_damping", "reverb_high_shelf_gain"),
            ("reverb_feedback", "reverb_decay_time"),
        ]);
        m.done("0.3.0 reverb damping -> high shelf");
    }

    if before("0.3.1") {
        if m.get("sample_keytrack") != 0.0 {
            let transpose = m.get("sample_transpose");
            m.set("sample_transpose", transpose + 28.0);
        }
        m.done("0.3.1 sample keytrack transpose");
    }

    if before("0.3.2") {
        for osc in ["osc_1", "osc_2"] {
            if m.get(&format!("{osc}_midi_track")) == 0.0 {
                let transpose = m.get(&format!("{osc}_transpose"));
                m.set(&format!("{osc}_transpose"), transpose - 48.0);
            }
        }
        m.done("0.3.2 untracked oscillator transpose -48");
    }

    if before("0.3.4") {
        const NUM_EFFECTS: usize = 9;
        const FILTER_FX: usize = 5;
        let mut order = decode_order(m.get("effect_chain_order"), NUM_EFFECTS - 1);
        for slot in order.iter_mut() {
            if *slot >= FILTER_FX {
                *slot += 1;
            }
        }
        order.push(FILTER_FX);
        m.set("effect_chain_order", encode_order(&order));
        m.done("0.3.4 effect chain gains filter fx");
    }

    if before("0.3.5") {
        for osc in ["osc_1", "osc_2"] {
            let name = format!("{osc}_distortion_type");
            let value = m.get(&name);
            if value >= 10.0 {
                m.set(&name, value + 1.0);
            }
        }
        m.done("0.3.5 distortion type shift");
    }

    if before("0.3.6") {
        for i in 1..=NUM_LFOS {
            let name = format!("lfo_{i}_sync_type");
            if m.has(&name) {
                let value = m.get(&name);
                if value >= 2.0 {
                    m.set(&name, value - 1.0);
                }
            }
        }
        m.done("0.3.6 lfo sync type shift");
    }

    if before("0.3.7") {
        if let Some(Value::Object(sample)) = m.sample.as_mut() {
            for field in ["samples", "samples_stereo"] {
                if let Some(Value::String(encoded)) = sample.get(field) {
                    if let Some(floats) = base64::decode_f32(encoded) {
                        sample.insert(field.to_string(), Value::String(base64::encode_pcm16(&floats)));
                    }
                }
            }
        }
        m.done("0.3.7 sample float32 -> pcm16");
    }

    if before("0.4.1") {
        let mut update = false;
        for modulation in m.modulations.iter_mut() {
            if modulation.source == "perlin" {
                update = true;
                modulation.source = "random_1".to_string();
            }
        }
        if update {
            m.set("random_1_sync", 0.0);
            // 1.65149612947 in the C++ (log2 of ~3.14 Hz).
            m.set("random_1_frequency", 1.651_496);
            m.set("random_1_stereo", 1.0);
        }
        m.done("0.4.1 perlin -> random_1");
    }

    // 0.4.3 / 0.4.4: formant (2) and sync (1) distortion amounts were
    // re-ranged. The C++ compares against the CURRENT enum values, before
    // the 0.4.7 renumbering — reproduced as is.
    for (gate, distortion_type, label) in [
        ("0.4.3", 2.0, "0.4.3 formant distortion range"),
        ("0.4.4", 1.0, "0.4.4 sync distortion range"),
    ] {
        if before(gate) {
            for osc in ["osc_1", "osc_2"] {
                if m.get(&format!("{osc}_distortion_type")) == distortion_type {
                    let name = format!("{osc}_distortion_amount");
                    let amount = m.get(&name);
                    m.set(&name, 0.5 + 0.5 * amount);
                    m.scale_amounts_into(&name, 0.5);
                }
            }
            m.done(label);
        }
    }

    if before("0.4.5") {
        let low = m.get("compressor_low_band") != 0.0;
        let high = m.get("compressor_high_band") != 0.0;
        let bands = match (low, high) {
            (true, true) => 0.0,
            (true, false) => 1.0,
            (false, true) => 2.0,
            (false, false) => 3.0,
        };
        m.set("compressor_enabled_bands", bands);
        m.done("0.4.5 compressor enabled bands");
    }

    if before("0.4.7") && m.has("osc_1_distortion_type") {
        const LOW_PASS_MORPH: f32 = 7.0;
        for osc in ["osc_1", "osc_2"] {
            let type_name = format!("{osc}_distortion_type");
            let original = m.get(&type_name);
            if original != 0.0 {
                m.set(&type_name, original - 1.0);
            }
            if original == 1.0 {
                m.set(&format!("{osc}_spectral_morph_type"), LOW_PASS_MORPH);
                let from = format!("{osc}_distortion_amount");
                let to = format!("{osc}_spectral_morph_amount");
                m.rename_destinations(&[(&from, &to)]);
            }
        }
        for osc in ["osc_1", "osc_2"] {
            let distortion_type = m.get(&format!("{osc}_distortion_type"));
            if !(7.0..=9.0).contains(&distortion_type) {
                continue;
            }
            let amount_name = format!("{osc}_distortion_amount");
            let original_fm = m.get(&amount_name);
            let new_fm = original_fm.max(0.0).sqrt();
            m.set(&amount_name, new_fm);
            remap_fm_connections(&mut m, &amount_name, original_fm, new_fm);
        }
        m.done("0.4.7 distortion renumbering / fm amount sqrt");
    }

    if before("0.5.0") && m.has("sub_on") {
        for (from, to) in [
            ("sub_on", "osc_3_on"),
            ("sub_level", "osc_3_level"),
            ("sub_pan", "osc_3_pan"),
            ("sub_transpose", "osc_3_transpose"),
            ("sub_tune", "osc_3_tune"),
        ] {
            let value = m.get(from);
            m.set(to, value);
        }
        if m.has("sub_transpose_quantize") {
            let value = m.get("sub_transpose_quantize");
            m.set("osc_3_transpose_quantize", value);
        }
        m.set("osc_3_phase", 0.25);
        m.set("osc_3_random_phase", 0.0);
        let sub_waveform = m.get("sub_waveform");
        m.set("osc_3_wave_frame", sub_waveform * 257.0 / 6.0);

        let destination = |filter1: bool, filter2: bool| -> f32 {
            match (filter1, filter2) {
                (true, true) => 2.0,
                (false, true) => 1.0,
                (true, false) => 0.0,
                (false, false) => 3.0,
            }
        };
        let sub_filter1 = m.get("filter_1_sub_input") != 0.0;
        let sub_filter2 = m.get("filter_2_sub_input") != 0.0;
        if m.get("sub_direct_out") != 0.0 {
            m.set("osc_3_destination", 4.0);
        } else {
            m.set("osc_3_destination", destination(sub_filter1, sub_filter2));
        }
        for (osc, input) in [("osc_1", "osc1"), ("osc_2", "osc2"), ("sample", "sample")] {
            let filter1 = m.get(&format!("filter_1_{input}_input")) != 0.0;
            let filter2 = m.get(&format!("filter_2_{input}_input")) != 0.0;
            m.set(&format!("{osc}_destination"), destination(filter1, filter2));
        }

        // The C++ rebuilds the wavetable list back to front, then appends
        // the predefined "Sub" table for the new oscillator 3.
        let mut tables: Vec<Value> = match m.wavetables.take() {
            Some(Value::Array(tables)) => tables,
            _ => Vec::new(),
        };
        tables.reverse();
        tables.push(sub_wavetable_json());
        *m.wavetables = Some(Value::Array(tables));

        m.rename_destinations(&[
            ("sub_transpose", "osc_3_transpose"),
            ("sub_tune", "osc_3_tune"),
            ("sub_level", "osc_3_level"),
            ("sub_pan", "osc_3_pan"),
        ]);
        m.done("0.5.0 sub oscillator -> osc_3");
    }

    if before("0.5.5") {
        for name in ["flanger_tempo", "phaser_tempo", "chorus_tempo", "delay_tempo"] {
            let tempo = m.get(name);
            m.set(name, tempo + 1.0);
        }
        for i in 1..=NUM_LFOS {
            let name = format!("lfo_{i}_tempo");
            if m.has(&name) {
                let tempo = m.get(&name);
                m.set(&name, tempo + 1.0);
            }
        }
        for i in 1..=NUM_RANDOM_LFOS {
            let name = format!("random_{i}_tempo");
            if m.has(&name) {
                let tempo = m.get(&name);
                m.set(&name, tempo + 1.0);
            }
        }
        m.done("0.5.5 tempo index +1");
    }

    if before("0.5.7") {
        for (from, to) in [
            ("delay_sync", "delay_aux_sync"),
            ("delay_frequency", "delay_aux_frequency"),
            ("delay_tempo", "delay_aux_tempo"),
        ] {
            let value = m.get(from);
            m.set(to, value);
        }
        let style = m.get("delay_style");
        if style != 0.0 {
            m.set("delay_style", style + 1.0);
        }
        m.done("0.5.7 delay aux line / style");
    }

    if before("0.5.8") {
        m.set("chorus_damping", 1.0);
        m.done("0.5.8 chorus damping");
    }

    if before("0.6.5") {
        m.set("stereo_mode", 1.0);
        let routing = m.get("stereo_routing") * 0.125;
        m.set("stereo_routing", if routing < 0.0 { 1.0 - routing } else { routing });
        m.done("0.6.5 stereo routing rescale");
    }

    if before("0.6.6") {
        if m.get("stereo_mode") == 0.0 {
            let routing = m.get("stereo_routing");
            m.set("stereo_routing", 1.0 - routing);
        }
        m.done("0.6.6 stereo routing spread flip");
    }

    if before("0.6.7") {
        let damping = m.get("chorus_damping");
        m.set("chorus_cutoff", 20.0);
        m.set("chorus_spread", damping);
        m.rename_destinations(&[("chorus_damping", "chorus_spread")]);
        m.done("0.6.7 chorus damping -> spread");
    }

    if before("0.7.1") {
        for osc in ["osc_1", "osc_2"] {
            let type_name = format!("{osc}_spectral_morph_type");
            let morph_type = if m.has(&type_name) { m.get(&type_name) } else { 0.0 };
            if morph_type == 9.0 {
                let name = format!("{osc}_spectral_morph_amount");
                let amount = m.get(&name);
                m.set(&name, -0.5 * amount + 0.5);
                m.scale_amounts_into(&name, -0.5);
            }
        }
        m.done("0.7.1 phase disperse morph inversion");
    }

    if before("0.7.5") {
        const CENTER_MULTIPLY: f32 = 48.0 / 128.0;
        const FLANGER_CENTER_OFFSET: f32 = 53.69;
        if m.has("flanger_center") {
            let center = m.get("flanger_center");
            m.set("flanger_center", center + FLANGER_CENTER_OFFSET);
        }
        m.scale_amounts_into("flanger_center", CENTER_MULTIPLY);
        m.scale_amounts_into("phaser_center", CENTER_MULTIPLY);
        m.done("0.7.5 flanger/phaser center");
    }

    if before("0.7.6") {
        for filter in ["filter_1", "filter_2"] {
            if m.get(&format!("{filter}_model")) as i32 == 6
                && m.get(&format!("{filter}_style")) as i32 == 1
            {
                m.set(&format!("{filter}_style"), 3.0);
            }
        }
        if m.has("filter_fx_model")
            && m.get("filter_fx_model") as i32 == 6
            && m.get("filter_fx_style") as i32 == 1
        {
            m.set("filter_fx_style", 3.0);
        }
        for osc in ["osc_1", "osc_2", "osc_3"] {
            let name = format!("{osc}_spectral_morph_type");
            if m.has(&name) && m.get(&name) == 10.0 {
                m.set(&name, 7.0);
            }
        }
        m.done("0.7.6 comb style / shepard morph renumbering");
    }

    if before("0.8.1") {
        // The C++ loop formats `lfo_<i>` with a 0-based i, so it writes
        // lfo_0 (unused) .. lfo_7 and leaves lfo_8 at its default. Same
        // effective result here without the junk key.
        for i in 1..NUM_LFOS {
            m.set(&format!("lfo_{i}_smooth_mode"), 0.0);
        }
        m.done("0.8.1 lfo smooth mode off");
    }

    if before("0.9.0") {
        for filter in ["filter_1", "filter_2"] {
            if m.get(&format!("{filter}_model")) == 4.0 {
                let blend = format!("{filter}_blend");
                m.set(&blend, 0.0);
                m.set(&format!("{filter}_style"), 0.0);
                m.scale_amounts_into(&blend, 0.0);
            }
        }
        m.done("0.9.0 diode filter blend reset");
    }

    let applied = std::mem::take(&mut m.applied);
    if !applied.is_empty() {
        preset.synth_version = CURRENT_FORMAT_VERSION.to_string();
    }
    applied
}

/// The 0.4.7 FM-amount conversion: the connection amounts into
/// `amount_name` are re-derived in the square-root domain and a 32-point
/// remap curve reproduces the old linear response.
fn remap_fm_connections(m: &mut Migration, amount_name: &str, original_fm: f32, new_fm: f32) {
    const REMAP_RESOLUTION: usize = 32;
    for index in 0..m.modulations.len() {
        if m.modulations[index].destination != amount_name {
            continue;
        }
        let number = index + 1;
        let amount_key = format!("modulation_{number}_amount");
        let last_amount = m.get(&amount_key);
        if last_amount == 0.0 {
            continue;
        }
        let bipolar = m.get(&format!("modulation_{number}_bipolar")) != 0.0;
        let (min, max) = if bipolar {
            let a = original_fm + last_amount * 0.5;
            let b = original_fm - last_amount * 0.5;
            (a.min(b), a.max(b))
        } else {
            (original_fm.min(original_fm + last_amount), original_fm.max(original_fm + last_amount))
        };
        let min_target = min.max(0.0).sqrt();
        let max_target = max.max(0.0).sqrt();
        let mut new_amount = max_target - min_target;
        if bipolar {
            new_amount = 2.0 * (new_fm - min_target).max(max_target - new_fm);
        }
        m.set(&amount_key, new_amount);

        let mut points = Vec::with_capacity(REMAP_RESOLUTION * 2);
        for i in 0..REMAP_RESOLUTION {
            let t = i as f32 / (REMAP_RESOLUTION - 1) as f32;
            let old_value = min + (max - min) * t;
            let adjusted = old_value.max(0.0).sqrt();
            let y = if new_amount.abs() > 1e-9 {
                1.0 - (adjusted - min_target) / new_amount
            } else {
                1.0
            };
            points.push(t);
            points.push(y);
        }
        m.modulations[index].line_mapping = Some(LineShape {
            num_points: REMAP_RESOLUTION as u32,
            points,
            powers: vec![0.0; REMAP_RESOLUTION],
            name: Some("Linear".to_string()),
            smooth: false,
            extra: Map::new(),
        });
    }
}

/// `WavetableCreator::initPredefinedWaves()` serialized: one Wave Source
/// with the six predefined shapes (sin, saturated sin, triangle, square,
/// pulse, saw) at `257 * i / 6`, step interpolation, no normalization.
fn sub_wavetable_json() -> Value {
    const WAVEFORM_SIZE: usize = 2048;
    const NUM_SHAPES: usize = 6;
    let mut keyframes = Vec::with_capacity(NUM_SHAPES);
    for shape in 0..NUM_SHAPES {
        let position = (257 * shape) / NUM_SHAPES;
        let wave = predefined_shape(shape, WAVEFORM_SIZE);
        keyframes.push(serde_json::json!({
            "position": position,
            "wave_data": base64::encode_f32(&wave),
        }));
    }
    serde_json::json!({
        "name": "Sub",
        "author": "",
        "version": CURRENT_FORMAT_VERSION,
        "remove_all_dc": false,
        "full_normalize": false,
        "groups": [{
            "components": [{
                "type": "Wave Source",
                "interpolation_style": 0,
                "interpolation": 1,
                "keyframes": keyframes,
            }]
        }]
    })
}

/// `PredefinedWaveFrames` shapes in enum order.
fn predefined_shape(shape: usize, size: usize) -> Vec<f32> {
    let mut wave = vec![0.0f32; size];
    let quarter = size / 4;
    match shape {
        0 => {
            for (i, value) in wave.iter_mut().enumerate() {
                *value = (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos();
            }
        }
        1 => {
            for (i, value) in wave.iter_mut().enumerate() {
                *value = (2.0 * (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()).tanh();
            }
        }
        2 => {
            for i in 0..quarter {
                let t = i as f32 / quarter as f32;
                wave[i] = 1.0 - t;
                wave[i + quarter] = -t;
                wave[i + 2 * quarter] = t - 1.0;
                wave[i + 3 * quarter] = t;
            }
        }
        3 => {
            for i in 0..quarter {
                wave[i] = 1.0;
                wave[i + quarter] = -1.0;
                wave[i + 2 * quarter] = -1.0;
                wave[i + 3 * quarter] = 1.0;
            }
        }
        4 => {
            for i in 0..quarter {
                wave[i + 3 * quarter] = 1.0;
                for section in 0..3 {
                    wave[i + section * quarter] = -1.0;
                }
            }
        }
        _ => {
            let half = size / 2;
            for i in 0..half {
                let t = i as f32 / half as f32;
                wave[(i + quarter) % size] = t - 1.0;
                wave[(i + half + quarter) % size] = t;
            }
        }
    }
    wave
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset(version: &str, settings: Value) -> Preset {
        let text = serde_json::json!({
            "synth_version": version,
            "preset_name": "old",
            "settings": settings,
        })
        .to_string();
        Preset::from_json(&text).unwrap()
    }

    #[test]
    fn version_parsing_and_order_codes() {
        assert_eq!(parse_version("0.4.0"), Some((0, 4, 0)));
        assert_eq!(parse_version("1.0"), Some((1, 0, 0)));
        assert_eq!(parse_version("junk"), None);
        assert!(version_before((0, 4, 0), "0.5.0"));
        assert!(!version_before((1, 0, 7), "0.9.0"));
        // Lehmer round trip for every size the C++ uses.
        for size in [8usize, 9] {
            for code in [0.0f32, 1.0, 5.0, 362879.0_f32.min((1..=size as u32).product::<u32>() as f32 - 1.0)] {
                let order = decode_order(code, size);
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..size).collect::<Vec<_>>());
                assert_eq!(encode_order(&order), code);
            }
        }
    }

    #[test]
    fn current_preset_is_untouched() {
        let mut current = preset(
            "1.0.7",
            serde_json::json!({"osc_1_level": 0.5, "flanger_center": 60.0, "delay_tempo": 9.0}),
        );
        let snapshot = current.clone();
        assert!(upgrade(&mut current).is_empty());
        assert_eq!(current, snapshot);
        // 0.9.x presets sit past the last migration too.
        let mut recent = preset("0.9.2", serde_json::json!({"osc_1_level": 0.5}));
        assert!(upgrade(&mut recent).is_empty());
        assert_eq!(recent.synth_version, "0.9.2");
    }

    #[test]
    fn synthetic_0_4_0_preset_converts() {
        let mut old = preset(
            "0.4.0",
            serde_json::json!({
                "osc_1_level": 0.5,
                "osc_2_level": 0.8,
                "osc_1_midi_track": 1.0,
                "osc_2_midi_track": 0.0,
                "osc_2_transpose": 12.0,
                "osc_1_distortion_type": 8.0,   // FM osc B after the -1 shift
                "osc_1_distortion_amount": 0.25,
                "osc_2_distortion_type": 0.0,
                "osc_2_distortion_amount": 0.5,
                "compressor_low_band": 1.0,
                "compressor_high_band": 0.0,
                "flanger_center": 10.0,
                "flanger_tempo": 3.0,
                "phaser_tempo": 3.0,
                "chorus_tempo": 3.0,
                "delay_tempo": 8.0,
                "delay_sync": 1.0,
                "delay_frequency": 2.0,
                "delay_style": 1.0,
                "chorus_damping": 0.4,
                "stereo_routing": 4.0,
                "filter_1_model": 4.0,
                "filter_1_blend": 1.5,
                "filter_1_style": 2.0,
                "filter_2_model": 6.0,
                "filter_2_style": 1.0,
                "modulation_1_amount": 0.5,
                "modulation_2_amount": 0.8,
                "modulation_3_amount": 0.3,
                "modulations": [
                    {"source": "lfo_1", "destination": "osc_1_distortion_amount"},
                    {"source": "env_2", "destination": "flanger_center"},
                    {"source": "lfo_2", "destination": "filter_1_blend"},
                    {"source": "perlin", "destination": "chorus_damping"}
                ]
            }),
        );
        let applied = upgrade(&mut old);
        assert!(!applied.is_empty());
        assert_eq!(old.synth_version, CURRENT_FORMAT_VERSION);
        let get = |name: &str| old.settings.parameter(name).unwrap_or_else(|| panic!("{name}"));

        // 0.3.2 is older than 0.4.0: no transpose shift. 0.4.x steps only.
        assert_eq!(get("osc_2_transpose"), 12.0);
        // 0.4.1 perlin -> random_1 with its defaults.
        assert_eq!(old.settings.modulations[3].source, "random_1");
        assert_eq!(get("random_1_stereo"), 1.0);
        // 0.4.5 bands: low only -> 1.
        assert_eq!(get("compressor_enabled_bands"), 1.0);
        // 0.4.7: type 8 -> 7 (FM osc A), amount sqrt, remap curve on the
        // connection into it, its amount re-derived.
        assert_eq!(get("osc_1_distortion_type"), 7.0);
        assert!((get("osc_1_distortion_amount") - 0.5).abs() < 1e-6);
        let remap = old.settings.modulations[0].line_mapping.as_ref().expect("remap curve");
        assert_eq!(remap.num_points, 32);
        assert!((get("modulation_1_amount") - (0.75f32.sqrt() - 0.5)).abs() < 1e-5);
        // 0.5.5 tempo +1, 0.5.7 aux delay copy + style shift.
        assert_eq!(get("delay_tempo"), 9.0);
        assert_eq!(get("delay_aux_tempo"), 9.0);
        assert_eq!(get("delay_style"), 2.0);
        // 0.6.5 / 0.6.6 stereo routing: 4 * 0.125 = 0.5, mode forced to 1.
        assert_eq!(get("stereo_mode"), 1.0);
        assert_eq!(get("stereo_routing"), 0.5);
        // 0.5.8 forces chorus_damping to 1 first, then 0.6.7 copies it to
        // spread (+ mod destination rename).
        assert_eq!(get("chorus_spread"), 1.0);
        assert_eq!(get("chorus_cutoff"), 20.0);
        assert_eq!(old.settings.modulations[3].destination, "chorus_spread");
        // 0.7.5 flanger center offset and the connection amount rescale.
        assert!((get("flanger_center") - 63.69).abs() < 1e-3);
        assert!((get("modulation_2_amount") - 0.8 * 0.375).abs() < 1e-6);
        // 0.7.6 comb style 1 -> 3.
        assert_eq!(get("filter_2_style"), 3.0);
        // 0.8.1 smooth mode off on lfo 1..7 (C++ off-by-one), lfo 8 untouched.
        assert_eq!(get("lfo_1_smooth_mode"), 0.0);
        assert_eq!(get("lfo_7_smooth_mode"), 0.0);
        assert_eq!(old.settings.parameter("lfo_8_smooth_mode"), None);
        // 0.9.0 diode model: blend/style reset, its connection zeroed.
        assert_eq!(get("filter_1_blend"), 0.0);
        assert_eq!(get("filter_1_style"), 0.0);
        assert_eq!(get("modulation_3_amount"), 0.0);
        // Re-running is a no-op now that the version is current.
        assert!(upgrade(&mut old).is_empty());
    }

    #[test]
    fn pre_0_5_0_sub_becomes_osc_3() {
        let mut old = preset(
            "0.2.0",
            serde_json::json!({
                "sub_on": 1.0, "sub_level": 0.5, "sub_pan": 0.1, "sub_transpose": -12.0,
                "sub_tune": 0.0, "sub_waveform": 3.0, "sub_direct_out": 1.0,
                "filter_1_sub_input": 1.0, "filter_2_sub_input": 0.0,
                "filter_1_osc1_input": 1.0, "filter_2_osc1_input": 1.0,
                "filter_1_osc2_input": 0.0, "filter_2_osc2_input": 0.0,
                "filter_1_sample_input": 0.0, "filter_2_sample_input": 1.0,
                "osc_1_level": 1.0, "osc_2_level": 1.0, "sample_level": 1.0,
                "env_1_attack": 0.5, "env_1_decay": 0.5, "env_1_release": 0.5,
                "reverb_damping": 0.5, "reverb_feedback": 0.9,
                "wavetables": [{"name": "A"}, {"name": "B"}],
                "modulations": [{"source": "lfo_1", "destination": "sub_level"}]
            }),
        );
        let applied = upgrade(&mut old);
        assert!(applied.iter().any(|s| s.starts_with("0.5.0")));
        let get = |name: &str| old.settings.parameter(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(get("osc_3_on"), 1.0);
        assert_eq!(get("osc_3_destination"), 4.0);
        assert_eq!(get("osc_1_destination"), 2.0);
        assert_eq!(get("osc_2_destination"), 3.0);
        assert_eq!(get("sample_destination"), 1.0);
        assert!((get("osc_3_wave_frame") - 3.0 * 257.0 / 6.0).abs() < 1e-4);
        assert_eq!(get("osc_3_phase"), 0.25);
        assert_eq!(old.settings.modulations[0].destination, "osc_3_level");
        // 0.2.7 level adjust: sqrt(0.7071 * 1 * 1) then 0.5.0 copies sub.
        assert!((get("osc_1_level") - 0.8409).abs() < 1e-3);
        // 0.2.1 then 0.2.5 envelope conversions: 0.5^(1/1.5)^(0.75) = 0.5^0.5.
        assert!((get("env_1_attack") - 0.5f32.sqrt()).abs() < 1e-5);
        // 0.3.0 reverb (with the C++ `decay_time` key) and destination rename.
        assert_eq!(get("reverb_high_shelf_gain"), -2.0);
        assert!((get("decay_time") - 1.0).abs() < 1e-5);
        // Wavetables: reversed, plus the generated "Sub" table.
        let tables = old.settings.wavetables.as_ref().unwrap().as_array().unwrap();
        assert_eq!(tables.len(), 3);
        assert_eq!(tables[0]["name"], "B");
        assert_eq!(tables[2]["name"], "Sub");
        let keyframes = tables[2]["groups"][0]["components"][0]["keyframes"].as_array().unwrap();
        assert_eq!(keyframes.len(), 6);
        assert_eq!(keyframes[3]["position"], 128);
        let square = base64::decode_f32(keyframes[3]["wave_data"].as_str().unwrap()).unwrap();
        assert_eq!(square.len(), 2048);
        assert_eq!(square[0], 1.0);
        assert_eq!(square[1024], -1.0);
    }
}
