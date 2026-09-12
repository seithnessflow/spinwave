//! Preset → text. A canonical serializer: stable module order, stable key
//! order, one key per line, comments derived from the data. Two presets
//! that differ by one parameter produce texts that differ by one line.

use std::collections::BTreeMap;

use serde_json::Value as Json;
use spinwave_engine::kernel::mod_matrix::Connection;
use spinwave_engine::modulation::ModulationTransform;
use spinwave_params::preset::LineShape;
use spinwave_params::{parameters, ParamDetails, Preset};
use spinwave_plugin::patch::{parse_mod_dest, parse_mod_source};

use super::layout::{module_rank, place};
use super::units::{self, Spelling, Value};
use super::{factory_name_of, Blobs, Written, FORMAT_VERSION};

/// One line of a module: key, value, optional comment.
struct Line {
    key: String,
    value: String,
    comment: Option<String>,
}

/// A modulation connection, resolved for writing.
struct Mod {
    slot: usize,
    to_key: String,
    from: String,
    amount: f32,
    bipolar: bool,
    power: f32,
    stereo: bool,
    bypass: bool,
    audio_rate: bool,
    curve: Option<LineShape>,
    extra: serde_json::Map<String, Json>,
}

fn toml_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn value_text(value: &Value) -> String {
    match value {
        Value::Bool(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        // A bare number must read back as a float: TOML makes `128` an
        // integer and the reader accepts both, but `1e-3` needs no fixing.
        Value::Number(s) => s.clone(),
        Value::Str(s) => toml_string(s),
    }
}

/// A float as TOML always accepts it (with a decimal point or exponent).
fn toml_float(x: f32) -> String {
    let s = format!("{x}");
    if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    }
}

/// The `[-1, 1]` amount as a signed percentage, exact or `raw:`.
fn amount_text(amount: f32) -> String {
    // Reuse the unit machinery with a percent spelling: multiply 100, "%".
    let details = ParamDetails {
        name: "amount".into(),
        version_added: 0,
        min: -1.0,
        max: 1.0,
        default_value: 0.0,
        post_offset: 0.0,
        display_multiply: 100.0,
        scale: spinwave_params::ParamScale::Linear,
        display_invert: false,
        display_units: "%".into(),
        display_name: "Amount".into(),
        string_lookup: None,
        local_description: String::new(),
        spinwave_only: false,
    };
    match units::write(&details, "mod", "amount", amount).0 {
        Value::Str(s) | Value::Number(s) => {
            if s.starts_with("raw:") || s.starts_with('-') || s.starts_with('+') {
                s
            } else {
                format!("+{s}")
            }
        }
        other => value_text(&other),
    }

}

/// The reach of an amount in the destination's own unit, for the comment
/// on a Linear destination where that unit means something.
fn reach_comment(dest: &ParamDetails, dest_module: &str, dest_key: &str, amount: f32) -> Option<String> {
    let reach = amount * (dest.max - dest.min);
    match units::spelling(dest, dest_module, dest_key) {
        Spelling::CutoffHz => Some(format!("{} st", trim_float(reach))),
        Spelling::Unit(units::UnitKind::Semitones) => Some(format!("{} st", trim_float(reach * dest.display_multiply))),
        Spelling::Unit(units::UnitKind::Decibel) => Some(format!("{} dB", trim_float(reach * dest.display_multiply))),
        _ => None,
    }
}

fn trim_float(x: f32) -> String {
    let s = format!("{:+.2}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Writes a preset as `.spinwave` text plus its sidecar.
pub fn write(preset: &Preset) -> Written {
    let table = parameters();
    let mut blobs = Blobs::default();

    // ---- values, grouped by module, only where they depart from the default
    let mut modules: BTreeMap<String, Vec<(usize, Line)>> = BTreeMap::new();
    let mut requires_spinwave: Vec<String> = Vec::new();
    let mut unknown_settings: Vec<(String, Json)> = Vec::new();

    for (order, details) in table.iter().enumerate() {
        let name = &details.name;
        if name.starts_with("modulation_") {
            continue; // spelled out as connections
        }
        let Some(json) = preset.settings.values.get(name) else { continue };
        let Some(engine) = json.as_f64().map(|v| v as f32) else { continue };
        if engine.to_bits() == details.default_value.to_bits() || (engine == details.default_value) {
            continue;
        }
        let at = place(name);
        let (value, comment) = units::write(details, &at.module, &at.key, engine);
        if details.spinwave_only {
            requires_spinwave.push(name.clone());
        }
        modules.entry(at.module).or_default().push((order, Line { key: at.key, value: value_text(&value), comment }));
    }
    for (name, json) in &preset.settings.values {
        if table.lookup(name).is_none() {
            unknown_settings.push((name.clone(), json.clone()));
        }
    }

    // ---- connections, under their destination's module
    let mut mods: Vec<(String, Mod)> = Vec::new();
    for (slot, connection) in preset.settings.modulations.iter().enumerate() {
        if !connection.is_connected() {
            continue;
        }
        let n = slot + 1;
        let get = |suffix: &str| preset.settings.values.get(&format!("modulation_{n}_{suffix}")).and_then(Json::as_f64).map(|v| v as f32);
        let at = place(&connection.destination);
        let audio_rate = match (parse_mod_source(&connection.source), parse_mod_dest(&connection.destination)) {
            (Some(source), Some(dest)) => Connection { source, dest, transform: ModulationTransform::default() }.is_audio_rate(),
            _ => false,
        };
        let curve = connection.line_mapping.clone().filter(|shape| !is_linear(shape));
        if parse_mod_source(&connection.source).is_none() || parse_mod_dest(&connection.destination).is_none() {
            // The engine cannot route it, but the file carries it; keep it.
        }
        mods.push((
            at.module,
            Mod {
                slot,
                to_key: at.key,
                from: connection.source.clone(),
                amount: get("amount").unwrap_or(0.0),
                bipolar: get("bipolar").unwrap_or(0.0) >= 0.5,
                power: get("power").unwrap_or(0.0),
                stereo: get("stereo").unwrap_or(0.0) >= 0.5,
                bypass: get("bypass").unwrap_or(0.0) >= 0.5,
                audio_rate,
                curve,
                extra: connection.extra.clone(),
            },
        ));
    }
    // Document order: by destination module, then by slot. A slot that is
    // not where document order would put it is written explicitly.
    mods.sort_by(|a, b| module_rank(&a.0).cmp(&module_rank(&b.0)).then(a.1.slot.cmp(&b.1.slot)));

    // ---- LFO shapes
    let mut shapes: Vec<(usize, &LineShape)> = Vec::new();
    for (i, shape) in preset.settings.lfos.iter().enumerate() {
        shapes.push((i, shape));
        if i >= 8 {
            requires_spinwave.push(format!("lfo_{}", i + 1));
        }
    }

    // ---- material
    let mut material: Vec<(String, String)> = Vec::new();
    if let Some(wavetables) = &preset.settings.wavetables {
        material.push(("wavetables".into(), blobs.put(wavetables)));
    }
    if let Some(sample) = &preset.settings.sample {
        material.push(("sample".into(), blobs.put(sample)));
    }
    if let Some(materials) = &preset.settings.spinwave_materials {
        let json = serde_json::to_value(materials).expect("materials serialise");
        material.push(("spinwave".into(), blobs.put(&json)));
        requires_spinwave.push("spinwave_materials".into());
    }

    // Summarised by module: naming every key would run to hundreds on a
    // fuzzed patch, and the module is what a reader has to remove.
    let mut requires_spinwave: Vec<String> = requires_spinwave
        .iter()
        .map(|name| match name.as_str() {
            "spinwave_materials" => name.clone(),
            n if n.starts_with("lfo_") && !n.contains("_generator") && !n.contains("_chaos") && !n.contains("_sh_") => place(n).module,
            n => {
                let at = place(n);
                if at.module == "global" { n.to_string() } else { at.module }
            }
        })
        .collect();
    requires_spinwave.sort();
    requires_spinwave.dedup();

    // ---- emit
    let mut out = String::new();
    out.push_str("# spinwave preset (TOML)\n");
    out.push_str(&format!("format = {FORMAT_VERSION}\n"));
    out.push_str(&format!("synth_version = {}\n", toml_string(&preset.synth_version)));
    if requires_spinwave.is_empty() {
        out.push_str("requires = \"vital\"\n");
    } else {
        out.push_str(&format!("requires = \"spinwave\"   # {}\n", requires_spinwave.join(", ")));
    }

    out.push_str("\n[preset]\n");
    out.push_str(&format!("name = {}\n", toml_string(&preset.preset_name)));
    if !preset.author.is_empty() {
        out.push_str(&format!("author = {}\n", toml_string(&preset.author)));
    }
    if !preset.preset_style.is_empty() {
        out.push_str(&format!("style = {}\n", toml_string(&preset.preset_style)));
    }
    if !preset.comments.is_empty() {
        out.push_str(&format!("comments = {}\n", toml_string(&preset.comments)));
    }
    let macros = [&preset.macro1, &preset.macro2, &preset.macro3, &preset.macro4];
    if macros.iter().any(|m| !m.is_empty()) {
        let list: Vec<String> = macros.iter().map(|m| toml_string(m)).collect();
        out.push_str(&format!("macros = [{}]\n", list.join(", ")));
    }

    // Every module that has lines, connections or a shape, in canonical order.
    let mut module_names: Vec<String> = modules.keys().cloned().collect();
    for (module, _) in &mods {
        if !module_names.contains(module) {
            module_names.push(module.clone());
        }
    }
    for (i, _) in &shapes {
        let module = format!("lfo_{}", i + 1);
        if !module_names.contains(&module) {
            module_names.push(module);
        }
    }
    module_names.sort_by_key(|m| module_rank(m));

    for module in &module_names {
        out.push('\n');
        out.push_str(&format!("[{module}]\n"));
        let mut lines: Vec<&(usize, Line)> = modules.get(module).map(|v| v.iter().collect()).unwrap_or_default();
        // `on` opens a module; the rest is alphabetical, which a reader can
        // predict (the table's own order is Vital's panel layout, and means
        // nothing away from it).
        lines.sort_by(|(_, a), (_, b)| (a.key != "on", &a.key).cmp(&(b.key != "on", &b.key)));
        let width = lines.iter().map(|(_, l)| l.key.len() + 3 + l.value.len()).max().unwrap_or(0);
        for (_, line) in lines {
            let mut text = format!("{} = {}", line.key, line.value);
            if let Some(comment) = &line.comment {
                text = format!("{text:width$}   # {comment}", width = width);
            }
            out.push_str(text.trim_end());
            out.push('\n');
        }
        // The module's LFO shape, if drawn.
        if let Some(index) = module.strip_prefix("lfo_").and_then(|n| n.parse::<usize>().ok()) {
            if let Some((_, shape)) = shapes.iter().find(|(i, _)| i + 1 == index) {
                write_shape(&mut out, module, "shape", shape, true);
            }
        }
        // Its connections.
        for (ordinal, (_, connection)) in mods.iter().filter(|(m, _)| m == module).enumerate() {
            out.push('\n');
            out.push_str(&format!("[[{module}.mod]]\n"));
            out.push_str(&format!("to = {}\n", toml_string(&connection.to_key)));
            out.push_str(&format!("from = {}\n", toml_string(&connection.from)));
            let dest_name = super::layout::table_name(module, &connection.to_key);
            let reach = table.lookup(&dest_name).and_then(|d| reach_comment(d, module, &connection.to_key, connection.amount));
            match reach {
                Some(reach) => out.push_str(&format!("amount = {:<10}   # {reach}\n", toml_string(&amount_text(connection.amount)))),
                None => out.push_str(&format!("amount = {}\n", toml_string(&amount_text(connection.amount)))),
            }
            // `bipolar` is always written for the sources Vital creates
            // bipolar by default (lfo, random, stereo, pitch), because a
            // reader's prior there points the other way from the table.
            let born_bipolar = ["lfo", "random", "stereo", "pitch"].iter().any(|p| connection.from.starts_with(p));
            if connection.bipolar || born_bipolar {
                out.push_str(&format!("bipolar = {}\n", connection.bipolar));
            }
            if connection.power != 0.0 {
                out.push_str(&format!("power = {}\n", toml_float(connection.power)));
            }
            if connection.stereo {
                out.push_str("stereo = true\n");
            }
            if connection.bypass {
                out.push_str("bypass = true\n");
            }
            if connection.audio_rate {
                out.push_str("rate = \"audio\"\n");
            }
            let canonical_slot = canonical_slot_for(&mods, module, ordinal);
            if connection.slot != canonical_slot {
                out.push_str(&format!("slot = {}\n", connection.slot + 1));
            }
            if !connection.extra.is_empty() {
                out.push_str(&format!("extra = {}\n", toml_string(&Json::Object(connection.extra.clone()).to_string())));
            }
            if let Some(curve) = &connection.curve {
                write_shape(&mut out, &format!("{module}.mod"), "curve", curve, false);
            }
        }
    }

    if !material.is_empty() {
        out.push_str("\n[material]\n");
        for (key, reference) in &material {
            out.push_str(&format!("{key} = {}\n", toml_string(reference)));
        }
    }

    if !unknown_settings.is_empty() {
        out.push_str("\n[vital.settings]\n");
        for (name, json) in &unknown_settings {
            out.push_str(&format!("{} = {}\n", toml_key(name), toml_string(&json.to_string())));
        }
    }
    if !preset.extra.is_empty() {
        out.push_str("\n[vital.extra]\n");
        for (name, json) in &preset.extra {
            out.push_str(&format!("{} = {}\n", toml_key(name), toml_string(&json.to_string())));
        }
    }

    Written { text: out, blobs }
}

/// The slot document order assigns to the `ordinal`-th connection of a
/// module: its position among all connections in document order.
fn canonical_slot_for(mods: &[(String, Mod)], module: &str, ordinal: usize) -> usize {
    let mut position = 0usize;
    let mut seen = 0usize;
    for (m, _) in mods {
        if m == module {
            if seen == ordinal {
                return position;
            }
            seen += 1;
        }
        position += 1;
    }
    position
}

fn toml_key(name: &str) -> String {
    if name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') && !name.is_empty() {
        name.to_string()
    } else {
        toml_string(name)
    }
}

fn is_linear(shape: &LineShape) -> bool {
    factory_name_of(shape) == Some("linear")
}

/// Writes a shape: by name when it is a factory shape, as a sub-table with
/// its points otherwise. `inline_name` lets a factory shape be a plain
/// key (`shape = "triangle"`); a curve under a connection is always a
/// sub-table, because a key would have to precede the `[[...]]` header.
fn write_shape(out: &mut String, parent: &str, key: &str, shape: &LineShape, inline_name: bool) {
    if let Some(name) = factory_name_of(shape) {
        if inline_name {
            out.push_str(&format!("{key} = {}\n", toml_string(name)));
            return;
        }
        out.push_str(&format!("[{parent}.{key}]\nfactory = {}\n", toml_string(name)));
        return;
    }
    out.push_str(&format!("\n[{parent}.{key}]\n"));
    let n = shape.num_points as usize;
    let pairs: Vec<String> = (0..n)
        .filter_map(|i| Some(format!("[{}, {}]", toml_float(*shape.points.get(2 * i)?), toml_float(*shape.points.get(2 * i + 1)?))))
        .collect();
    out.push_str(&format!("points = [{}]\n", pairs.join(", ")));
    let powers: Vec<String> = shape.powers.iter().map(|p| toml_float(*p)).collect();
    out.push_str(&format!("powers = [{}]\n", powers.join(", ")));
    if shape.smooth {
        out.push_str("smooth = true\n");
    }
    if let Some(name) = &shape.name {
        out.push_str(&format!("name = {}\n", toml_string(name)));
    }
    if !shape.extra.is_empty() {
        out.push_str(&format!("extra = {}\n", toml_string(&Json::Object(shape.extra.clone()).to_string())));
    }
    // Points beyond num_points and a powers list longer than num_points are
    // carried by the .vital; keep them for the round trip.
    if shape.points.len() > 2 * n || shape.powers.len() > n {
        out.push_str(&format!("raw_points = [{}]\n", shape.points.iter().map(|p| toml_float(*p)).collect::<Vec<_>>().join(", ")));
        out.push_str(&format!("raw_powers = [{}]\n", shape.powers.iter().map(|p| toml_float(*p)).collect::<Vec<_>>().join(", ")));
    }
}
