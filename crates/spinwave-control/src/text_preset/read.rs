//! Text → preset, with a report that says everything the parser
//! normalised or refused, precisely enough for a program to fix its file.
//!
//! Generous about spelling, strict about meaning. A refused value is an
//! error and the preset does not load; nothing is guessed silently.

use std::collections::BTreeMap;

use serde_json::Value as Json;
use spinwave_engine::kernel::mod_matrix::Connection;
use spinwave_engine::modulation::ModulationTransform;
use spinwave_params::preset::{Correction, CorrectionKind, ErrorCode, LineShape, LoadError, ModulationConnection};
use spinwave_params::{parameters, LoadReport, ParamDetails, Preset};
use spinwave_plugin::patch::{parse_mod_dest, parse_mod_source};
use toml::de::{DeArray, DeTable, DeValue};

use super::layout::{module_rank, place, table_name, FLAT_FAMILIES};
use super::units::{self, Scalar, Spelling, UnitKind};
use super::{factory_name_of, factory_shape, Blobs, ReadResult, FORMAT_VERSION};

struct Reader<'a> {
    text: &'a str,
    blobs: &'a Blobs,
    report: LoadReport,
    values: BTreeMap<String, Json>,
    connections: Vec<(Option<usize>, ModulationConnection, ModValues)>,
    shapes: BTreeMap<usize, LineShape>,
    spinwave_used: Vec<String>,
}

#[derive(Default)]
struct ModValues {
    amount: f32,
    bipolar: bool,
    power: f32,
    stereo: bool,
    bypass: bool,
}

impl<'a> Reader<'a> {
    fn line_of(&self, span: &std::ops::Range<usize>) -> u32 {
        self.text[..span.start.min(self.text.len())].matches('\n').count() as u32 + 1
    }

    fn error(&mut self, line: u32, key: &str, code: ErrorCode, message: String, expected: Option<String>, suggestion: Option<String>) {
        self.report.errors.push(LoadError { line, key: key.to_string(), code, message, expected, suggestion });
    }

    fn correction(&mut self, line: u32, key: &str, written: &str, read_as: &str, kind: CorrectionKind) {
        self.report.corrections.push(Correction {
            line,
            key: key.to_string(),
            written: written.to_string(),
            read_as: read_as.to_string(),
            kind,
        });
    }
}

/// Edit distance, for "did you mean".
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            cur.push((prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost));
        }
        prev = cur;
    }
    prev[b.len()]
}

/// The nearest candidates to `name`: `Some(one)` when a single best exists
/// within reach, `Err(several)` on a tie, `Ok(None)` when nothing is close.
fn nearest<'c>(name: &str, candidates: impl Iterator<Item = &'c str>) -> Result<Option<String>, Vec<String>> {
    let mut best: Vec<(usize, String)> = candidates.map(|c| (distance(name, c), c.to_string())).collect();
    best.sort();
    let reach = (name.len() / 3).max(2);
    let close: Vec<(usize, String)> = best.into_iter().take_while(|(d, _)| *d <= reach).collect();
    match close.as_slice() {
        [] => Ok(None),
        [(_, one)] => Ok(Some(one.clone())),
        [(d0, first), (d1, _), ..] if d0 < d1 => Ok(Some(first.clone())),
        many => Err(many.iter().map(|(_, n)| n.clone()).collect()),
    }
}

/// Keys the table knows for a module.
fn module_keys(module: &str) -> Vec<String> {
    parameters()
        .iter()
        .filter(|d| !d.name.starts_with("modulation_"))
        .map(|d| place(&d.name))
        .filter(|p| p.module == module)
        .map(|p| p.key)
        .collect()
}

fn module_is_known(module: &str) -> bool {
    module == "global" || module == "macros" || FLAT_FAMILIES.contains(&module) || !module_keys(module).is_empty()
}

fn all_modules() -> Vec<String> {
    let mut modules: Vec<String> = parameters().iter().map(|d| place(&d.name).module).collect();
    modules.sort();
    modules.dedup();
    modules
}

/// Reads `.spinwave` text. `blobs` resolves material references.
pub fn read(text: &str, blobs: &Blobs) -> ReadResult {
    let mut reader = Reader {
        text,
        blobs,
        report: LoadReport::default(),
        values: BTreeMap::new(),
        connections: Vec::new(),
        shapes: BTreeMap::new(),
        spinwave_used: Vec::new(),
    };

    let doc = match DeTable::parse(text) {
        Ok(doc) => doc,
        Err(e) => {
            let line = e.span().map(|s| reader.line_of(&s)).unwrap_or(0);
            reader.error(line, "", ErrorCode::Syntax, e.message().to_string(), None, None);
            return ReadResult { preset: None, report: reader.report };
        }
    };
    let doc = doc.get_ref();

    let mut preset = Preset::default();
    let mut declared_requires: Option<(u32, String)> = None;

    // ---- header
    match doc.get("format") {
        None => reader.error(1, "format", ErrorCode::MissingFormatVersion, "no `format` key; this file does not say which version of the format it is".into(), Some(format!("format = {FORMAT_VERSION}")), None),
        Some(v) => {
            let line = reader.line_of(&v.span());
            match integer_of(v.get_ref()) {
                Some(n) if n == FORMAT_VERSION => {}
                Some(n) => reader.error(line, "format", ErrorCode::UnsupportedFormatVersion, format!("format {n} is not supported"), Some(format!("format = {FORMAT_VERSION}")), None),
                None => reader.error(line, "format", ErrorCode::BadValue, "`format` must be an integer".into(), Some(format!("format = {FORMAT_VERSION}")), None),
            }
        }
    }
    if let Some(v) = doc.get("synth_version") {
        preset.synth_version = v.get_ref().as_str().unwrap_or("").to_string();
    }
    if let Some(v) = doc.get("requires") {
        let line = reader.line_of(&v.span());
        match v.get_ref().as_str() {
            Some("vital") | Some("spinwave") => declared_requires = Some((line, v.get_ref().as_str().unwrap().to_string())),
            _ => reader.error(line, "requires", ErrorCode::BadValue, "`requires` must be \"vital\" or \"spinwave\"".into(), None, None),
        }
    }

    // ---- [preset]
    if let Some(section) = doc.get("preset").and_then(|v| v.get_ref().as_table()) {
        let get = |key: &str| section.get(key).and_then(|v| v.get_ref().as_str()).map(str::to_string);
        preset.preset_name = get("name").unwrap_or_default();
        preset.author = get("author").unwrap_or_default();
        preset.preset_style = get("style").unwrap_or_default();
        preset.comments = get("comments").unwrap_or_default();
        if let Some(list) = section.get("macros").and_then(|v| v.get_ref().as_array()) {
            let names: Vec<String> = list.iter().map(|v| v.get_ref().as_str().unwrap_or("").to_string()).collect();
            preset.macro1 = names.first().cloned().unwrap_or_default();
            preset.macro2 = names.get(1).cloned().unwrap_or_default();
            preset.macro3 = names.get(2).cloned().unwrap_or_default();
            preset.macro4 = names.get(3).cloned().unwrap_or_default();
        }
    }

    // ---- modules, in canonical order. The TOML map does not keep document
    // order and a hand-written file may put modules anywhere; slots are
    // assigned in the order the writer uses, so both sides agree.
    let table = parameters();
    let mut entries: Vec<_> = doc.iter().collect();
    entries.sort_by_key(|(key, _)| module_rank(key.get_ref().as_ref()));
    for (key, value) in entries {
        let module = key.get_ref().as_ref();
        if matches!(module, "format" | "synth_version" | "requires" | "preset" | "material" | "vital") {
            continue;
        }
        let line = reader.line_of(&key.span());
        let Some(section) = value.get_ref().as_table() else {
            reader.error(line, module, ErrorCode::BadValue, format!("`{module}` should be a table `[{module}]`"), None, None);
            continue;
        };
        if !module_is_known(module) {
            let suggestion = match nearest(module, all_modules().iter().map(String::as_str)) {
                Ok(s) => s,
                Err(several) => Some(several.join(" or ")),
            };
            reader.error(line, module, ErrorCode::UnknownModule, format!("unknown module `[{module}]`"), None, suggestion);
            continue;
        }
        read_module(&mut reader, table, module, section);
    }

    // ---- [material]
    if let Some(section) = doc.get("material").and_then(|v| v.get_ref().as_table()) {
        for (key, value) in section.iter() {
            let line = reader.line_of(&key.span());
            let name = key.get_ref().as_ref();
            let Some(reference) = value.get_ref().as_str() else {
                reader.error(line, name, ErrorCode::BadValue, "a material entry is a blob reference string".into(), None, None);
                continue;
            };
            let Some(json) = reader.blobs.get(reference) else {
                reader.error(line, name, ErrorCode::BlobMissing, format!("{reference} is not in the sidecar"), None, None);
                continue;
            };
            match name {
                "wavetables" => preset.settings.wavetables = Some(json),
                "sample" => preset.settings.sample = Some(json),
                "spinwave" => match serde_json::from_value(json) {
                    Ok(materials) => {
                        preset.settings.spinwave_materials = Some(materials);
                        reader.spinwave_used.push("spinwave_materials".into());
                    }
                    Err(e) => reader.error(line, name, ErrorCode::BadValue, format!("spinwave material blob does not parse: {e}"), None, None),
                },
                other => reader.error(line, other, ErrorCode::UnknownKey, format!("unknown material `{other}`"), Some("wavetables, sample, spinwave".into()), None),
            }
        }
    }

    // ---- [vital.settings] and [vital.extra]: raw JSON, kept verbatim
    if let Some(vital) = doc.get("vital").and_then(|v| v.get_ref().as_table()) {
        if let Some(settings) = vital.get("settings").and_then(|v| v.get_ref().as_table()) {
            for (key, value) in settings.iter() {
                if let Some(json) = value.get_ref().as_str().and_then(|s| serde_json::from_str(s).ok()) {
                    reader.values.insert(key.get_ref().to_string(), json);
                }
            }
        }
        if let Some(extra) = vital.get("extra").and_then(|v| v.get_ref().as_table()) {
            for (key, value) in extra.iter() {
                if let Some(json) = value.get_ref().as_str().and_then(|s| serde_json::from_str(s).ok()) {
                    preset.extra.insert(key.get_ref().to_string(), json);
                }
            }
        }
    }

    // ---- requires: derived, then checked against what was declared
    reader.spinwave_used.sort();
    reader.spinwave_used.dedup();
    let derived = if reader.spinwave_used.is_empty() { "vital" } else { "spinwave" };
    if let Some((line, declared)) = &declared_requires {
        if declared != derived {
            let message = if derived == "spinwave" {
                format!("declared `requires = \"vital\"` but uses Spinwave-only features: {}", reader.spinwave_used.join(", "))
            } else {
                "declared `requires = \"spinwave\"` but uses nothing Spinwave-specific".to_string()
            };
            let code = ErrorCode::RequiresMismatch;
            if derived == "spinwave" {
                reader.error(*line, "requires", code, message, Some("requires = \"spinwave\"".into()), None);
            } else {
                reader.report.notes.push(message);
            }
        }
    }
    if derived == "spinwave" {
        reader.report.notes.push(format!("requires spinwave: {}", reader.spinwave_used.join(", ")));
    }

    // ---- assemble the connections into slots
    let mut slots: BTreeMap<usize, (ModulationConnection, ModValues)> = BTreeMap::new();
    let mut unplaced: Vec<(ModulationConnection, ModValues)> = Vec::new();
    for (explicit, connection, values) in std::mem::take(&mut reader.connections) {
        match explicit {
            Some(slot) if !slots.contains_key(&slot) => {
                slots.insert(slot, (connection, values));
            }
            Some(slot) => {
                reader.error(0, "slot", ErrorCode::BadValue, format!("slot {} is used twice", slot + 1), None, None);
            }
            None => unplaced.push((connection, values)),
        }
    }
    let mut next = 0usize;
    for entry in unplaced {
        while slots.contains_key(&next) {
            next += 1;
        }
        slots.insert(next, entry);
        next += 1;
    }
    let count = slots.keys().max().map(|m| m + 1).unwrap_or(0);
    let mut modulations = vec![ModulationConnection::default(); count];
    for (slot, (connection, values)) in slots {
        let n = slot + 1;
        reader.values.insert(format!("modulation_{n}_amount"), Json::from(values.amount as f64));
        if values.bipolar {
            reader.values.insert(format!("modulation_{n}_bipolar"), Json::from(1.0));
        }
        if values.power != 0.0 {
            reader.values.insert(format!("modulation_{n}_power"), Json::from(values.power as f64));
        }
        if values.stereo {
            reader.values.insert(format!("modulation_{n}_stereo"), Json::from(1.0));
        }
        if values.bypass {
            reader.values.insert(format!("modulation_{n}_bypass"), Json::from(1.0));
        }
        modulations[slot] = connection;
    }
    preset.settings.modulations = modulations;

    // ---- LFO shapes: absent means the engine's default, a triangle
    if let Some(&last) = reader.shapes.keys().max() {
        let triangle = factory_shape("triangle").expect("factory triangle");
        preset.settings.lfos = (0..=last).map(|i| reader.shapes.remove(&i).unwrap_or_else(|| triangle.clone())).collect();
    }

    preset.settings.values = std::mem::take(&mut reader.values).into_iter().collect();

    let ok = reader.report.errors.is_empty();
    ReadResult { preset: ok.then_some(preset), report: reader.report }
}

fn integer_of(value: &DeValue) -> Option<i64> {
    let i = value.as_integer()?;
    i64::from_str_radix(i.as_str(), i.radix()).ok()
}

fn float_of(value: &DeValue) -> Option<f64> {
    if let Some(i) = integer_of(value) {
        return Some(i as f64);
    }
    value.as_float()?.as_str().parse().ok()
}

fn scalar_of(value: &DeValue) -> Option<Scalar> {
    if let Some(b) = value.as_bool() {
        return Some(Scalar::Bool(b));
    }
    if let Some(i) = integer_of(value) {
        return Some(Scalar::Integer(i));
    }
    value.as_float().and_then(|f| f.as_str().parse().ok()).map(Scalar::Float)
}

fn read_module(reader: &mut Reader, table: &spinwave_params::ParamTable, module: &str, section: &DeTable) {
    for (key, value) in section.iter() {
        let key_name = key.get_ref().as_ref();
        let line = reader.line_of(&key.span());

        if key_name == "mod" {
            match value.get_ref().as_array() {
                Some(list) => read_connections(reader, table, module, list),
                None => reader.error(line, "mod", ErrorCode::BadValue, "`mod` is an array of tables: `[[module.mod]]`".into(), None, None),
            }
            continue;
        }
        if key_name == "shape" && module.starts_with("lfo_") {
            let index = module[4..].parse::<usize>().unwrap_or(0).saturating_sub(1);
            if let Some(shape) = read_shape(reader, line, "shape", value.get_ref()) {
                if index >= 8 {
                    reader.spinwave_used.push(module.to_string());
                }
                reader.shapes.insert(index, shape);
            }
            continue;
        }

        let name = table_name(module, key_name);
        let Some(details) = table.lookup(&name) else {
            let keys = module_keys(module);
            let (suggestion, code) = match nearest(key_name, keys.iter().map(String::as_str)) {
                Ok(s) => (s, ErrorCode::UnknownKey),
                Err(several) => (Some(several.join(" or ")), ErrorCode::AmbiguousName),
            };
            reader.error(line, &format!("{module}.{key_name}"), code, format!("unknown key `{key_name}` in [{module}]"), None, suggestion);
            continue;
        };
        let details: &ParamDetails = details;
        let full_key = format!("{module}.{key_name}");
        let result = match value.get_ref().as_str() {
            Some(text) => units::read_str(details, module, key_name, text).map(|r| (r, text.to_string())),
            None => match scalar_of(value.get_ref()) {
                Some(scalar) => units::read_scalar(details, module, key_name, scalar).map(|r| (r, scalar_text(scalar))),
                None => Err(units::UnitError::Bad { message: "not a value".into() }),
            },
        };
        match result {
            Ok((read, written)) => {
                if let Some(canonical) = &read.normalised {
                    reader.correction(line, &full_key, &written, canonical, CorrectionKind::UnitNormalised);
                }
                if details.spinwave_only && read.engine != details.default_value {
                    reader.spinwave_used.push(name.clone());
                }
                reader.values.insert(name.clone(), Json::from(read.engine as f64));
            }
            Err(e) => {
                let (code, message, expected) = match e {
                    units::UnitError::MissingUnit { expected } => (ErrorCode::MissingUnit, format!("`{key_name}` needs a unit"), Some(expected)),
                    units::UnitError::WrongUnit { found, expected } => (ErrorCode::WrongUnit, format!("`{key_name}` does not take `{found}`"), Some(expected)),
                    units::UnitError::OutOfRange { value, range } => (ErrorCode::OutOfRange, format!("`{key_name}` = {value} is out of range"), Some(range)),
                    units::UnitError::Bad { message } => (ErrorCode::BadValue, message, None),
                };
                reader.error(line, &full_key, code, message, expected, None);
            }
        }
    }
}

fn scalar_text(scalar: Scalar) -> String {
    match scalar {
        Scalar::Bool(b) => b.to_string(),
        Scalar::Integer(i) => i.to_string(),
        Scalar::Float(f) => f.to_string(),
    }
}

/// Reads an amount: a signed percentage, or on a Linear destination a
/// value in the destination's unit, converted and reported.
fn read_amount(reader: &mut Reader, line: u32, dest: Option<&ParamDetails>, dest_module: &str, dest_key: &str, text: &str) -> Option<f32> {
    let trimmed = text.trim();
    if let Some(raw) = trimmed.strip_prefix("raw:") {
        return raw.trim().parse().ok();
    }
    let number_end = trimmed
        .char_indices()
        .take_while(|(_, c)| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
        .map(|(i, c)| i + c.len_utf8())
        .last()
        .unwrap_or(0);
    let (number, unit) = trimmed.split_at(number_end);
    let Ok(number) = number.parse::<f64>() else {
        reader.error(line, "amount", ErrorCode::BadValue, format!("amount `{trimmed}` is not a number"), Some("a signed percentage, e.g. \"+70%\"".into()), None);
        return None;
    };
    let unit = unit.trim().to_lowercase();
    let amount = match unit.as_str() {
        "%" | "pct" | "percent" => number / 100.0,
        "" => {
            reader.error(line, "amount", ErrorCode::MissingUnit, "amount needs a unit".into(), Some("a signed percentage of the destination's range, e.g. \"+70%\"".into()), None);
            return None;
        }
        other => {
            // A unit of the destination, on a Linear destination only.
            let Some(dest) = dest else {
                reader.error(line, "amount", ErrorCode::WrongUnit, format!("amount in `{other}` needs a known destination"), Some("%".into()), None);
                return None;
            };
            let span = (dest.max - dest.min) as f64 * dest.display_multiply as f64;
            let converted = match (units::spelling(dest, dest_module, dest_key), other) {
                (Spelling::CutoffHz, "st" | "semi" | "semitones") => number / (dest.max - dest.min) as f64,
                (Spelling::CutoffHz, "oct" | "octave" | "octaves") => number * 12.0 / (dest.max - dest.min) as f64,
                (Spelling::Unit(UnitKind::Semitones), "st" | "semi" | "semitones") => number / span,
                (Spelling::Unit(UnitKind::Semitones), "oct" | "octave" | "octaves") => number * 12.0 / span,
                (Spelling::Unit(UnitKind::Decibel), "db") => number / span,
                _ => {
                    reader.error(line, "amount", ErrorCode::WrongUnit, format!("amount in `{other}` does not apply to `{dest_key}`"), Some("a signed percentage, e.g. \"+70%\"".into()), None);
                    return None;
                }
            };
            let percent = format!("{}%", trim(converted * 100.0));
            let percent = if converted >= 0.0 { format!("+{percent}") } else { percent };
            reader.correction(line, "amount", trimmed, &percent, CorrectionKind::UnitNormalised);
            converted
        }
    };
    if !(-1.0..=1.0).contains(&amount) {
        reader.error(line, "amount", ErrorCode::OutOfRange, format!("amount {trimmed} is out of range"), Some("-100% to +100%".into()), None);
        return None;
    }
    Some(amount as f32)
}

fn trim(x: f64) -> String {
    let s = format!("{x:.4}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn read_connections(reader: &mut Reader, table: &spinwave_params::ParamTable, module: &str, list: &DeArray) {
    for item in list.iter() {
        let line = reader.line_of(&item.span());
        let Some(entry) = item.get_ref().as_table() else {
            reader.error(line, "mod", ErrorCode::BadValue, "each `[[module.mod]]` entry is a table".into(), None, None);
            continue;
        };
        let get_str = |key: &str| entry.get(key).and_then(|v| v.get_ref().as_str());
        let Some(to_key) = get_str("to") else {
            reader.error(line, "mod.to", ErrorCode::BadValue, "a connection needs `to = \"<key>\"`".into(), None, None);
            continue;
        };
        let Some(from) = get_str("from") else {
            reader.error(line, "mod.from", ErrorCode::BadValue, "a connection needs `from = \"<source>\"`".into(), None, None);
            continue;
        };
        let destination = table_name(module, to_key);
        let dest_details = table.lookup(&destination);
        if dest_details.is_none() {
            let keys = module_keys(module);
            let suggestion = nearest(to_key, keys.iter().map(String::as_str)).unwrap_or_else(|s| Some(s.join(" or ")));
            reader.error(line, "mod.to", ErrorCode::UnknownKey, format!("no parameter `{to_key}` in [{module}] to modulate"), None, suggestion);
            continue;
        }
        let source_known = parse_mod_source(from).is_some();
        if !source_known {
            let sources = ["lfo_1", "env_1", "random_1", "macro_control_1", "note", "velocity", "lift", "mod_wheel", "pitch_wheel", "aftertouch", "slide", "random", "stereo"];
            let suggestion = nearest(from, sources.iter().copied()).unwrap_or(None);
            reader.error(line, "mod.from", ErrorCode::UnknownKey, format!("unknown modulation source `{from}`"), None, suggestion);
            continue;
        }

        let mut values = ModValues::default();
        match entry.get("amount") {
            Some(v) => {
                let amount_line = reader.line_of(&v.span());
                let text = match v.get_ref().as_str() {
                    Some(s) => s.to_string(),
                    None => match float_of(v.get_ref()) {
                        Some(_) => {
                            reader.error(amount_line, "amount", ErrorCode::MissingUnit, "amount needs a unit".into(), Some("a signed percentage, e.g. \"+70%\"".into()), None);
                            continue;
                        }
                        None => {
                            reader.error(amount_line, "amount", ErrorCode::BadValue, "amount is not a value".into(), None, None);
                            continue;
                        }
                    },
                };
                match read_amount(reader, amount_line, dest_details, module, to_key, &text) {
                    Some(a) => values.amount = a,
                    None => continue,
                }
            }
            None => {
                reader.error(line, "amount", ErrorCode::BadValue, "a connection needs `amount`".into(), Some("a signed percentage, e.g. \"+70%\"".into()), None);
                continue;
            }
        }
        values.bipolar = entry.get("bipolar").and_then(|v| v.get_ref().as_bool()).unwrap_or(false);
        values.stereo = entry.get("stereo").and_then(|v| v.get_ref().as_bool()).unwrap_or(false);
        values.bypass = entry.get("bypass").and_then(|v| v.get_ref().as_bool()).unwrap_or(false);
        values.power = entry.get("power").and_then(|v| float_of(v.get_ref())).unwrap_or(0.0) as f32;

        // Regime: derived from the pair, and checked if written.
        let audio_rate = match (parse_mod_source(from), parse_mod_dest(&destination)) {
            (Some(source), Some(dest)) => Connection { source, dest, transform: ModulationTransform::default() }.is_audio_rate(),
            _ => false,
        };
        if let Some(rate) = entry.get("rate") {
            let rate_line = reader.line_of(&rate.span());
            match rate.get_ref().as_str() {
                Some("audio") if !audio_rate => reader.error(rate_line, "rate", ErrorCode::BadRegime, format!("`{from} -> {destination}` runs at control rate; audio rate needs an envelope or LFO source into a filter cutoff"), Some("omit `rate`".into()), None),
                Some("control") if audio_rate => reader.error(rate_line, "rate", ErrorCode::BadRegime, format!("`{from} -> {destination}` runs at audio rate: an envelope or LFO into a filter cutoff is evaluated per sample"), Some("rate = \"audio\"".into()), None),
                Some("audio") | Some("control") => {}
                _ => reader.error(rate_line, "rate", ErrorCode::BadValue, "`rate` is \"audio\" or \"control\"".into(), None, None),
            }
        }

        let slot = match entry.get("slot") {
            Some(v) => match integer_of(v.get_ref()) {
                Some(n) if n >= 1 => Some((n - 1) as usize),
                _ => {
                    reader.error(reader.line_of(&v.span()), "slot", ErrorCode::BadValue, "`slot` is a 1-based integer".into(), None, None);
                    continue;
                }
            },
            None => None,
        };

        let curve = match entry.get("curve") {
            Some(v) => read_shape(reader, reader.line_of(&v.span()), "curve", v.get_ref()),
            None => None,
        };
        let extra = entry
            .get("extra")
            .and_then(|v| v.get_ref().as_str())
            .and_then(|s| serde_json::from_str::<Json>(s).ok())
            .and_then(|j| j.as_object().cloned())
            .unwrap_or_default();

        if table.lookup(&destination).is_some_and(|d| d.spinwave_only) {
            reader.spinwave_used.push(destination.clone());
        }
        if from.starts_with("macro_control_") && from[14..].parse::<usize>().map(|n| n > 4).unwrap_or(false) {
            reader.spinwave_used.push(from.to_string());
        }

        reader.connections.push((
            slot,
            ModulationConnection { source: from.to_string(), destination, line_mapping: curve, extra },
            values,
        ));
    }
}

/// Reads a shape: a factory name as a string, or a table with `factory`,
/// or a table with `points` / `powers` / `smooth` / `name`.
fn read_shape(reader: &mut Reader, line: u32, key: &str, value: &DeValue) -> Option<LineShape> {
    if let Some(name) = value.as_str() {
        return match factory_shape(name) {
            Some(shape) => Some(shape),
            None => {
                reader.error(line, key, ErrorCode::BadValue, format!("`{name}` is not a factory shape"), Some("linear, triangle, square, sin, saw_up, saw_down, or a table with points".into()), None);
                None
            }
        };
    }
    let Some(t) = value.as_table() else {
        reader.error(line, key, ErrorCode::BadValue, "a shape is a factory name or a table".into(), None, None);
        return None;
    };
    if let Some(name) = t.get("factory").and_then(|v| v.get_ref().as_str()) {
        return factory_shape(name).or_else(|| {
            reader.error(line, key, ErrorCode::BadValue, format!("`{name}` is not a factory shape"), None, None);
            None
        });
    }
    // `LineShape::default()` is the linear shape; a drawn one starts empty.
    let mut shape = LineShape { num_points: 0, points: Vec::new(), powers: Vec::new(), ..Default::default() };
    if let Some(raw) = t.get("raw_points").and_then(|v| v.get_ref().as_array()) {
        shape.points = raw.iter().filter_map(|v| float_of(v.get_ref())).map(|f| f as f32).collect();
        if let Some(raw) = t.get("raw_powers").and_then(|v| v.get_ref().as_array()) {
            shape.powers = raw.iter().filter_map(|v| float_of(v.get_ref())).map(|f| f as f32).collect();
        }
        shape.num_points = t.get("points").and_then(|v| v.get_ref().as_array()).map(|a| a.iter().count() as u32).unwrap_or(0);
    } else {
        let Some(points) = t.get("points").and_then(|v| v.get_ref().as_array()) else {
            reader.error(line, key, ErrorCode::BadValue, "a drawn shape needs `points = [[x, y], ...]`".into(), None, None);
            return None;
        };
        for pair in points.iter() {
            let Some(xy) = pair.get_ref().as_array() else {
                reader.error(line, key, ErrorCode::BadValue, "each point is `[x, y]`".into(), None, None);
                return None;
            };
            let coords: Vec<f32> = xy.iter().filter_map(|v| float_of(v.get_ref())).map(|f| f as f32).collect();
            if coords.len() != 2 {
                reader.error(line, key, ErrorCode::BadValue, "each point is `[x, y]`".into(), None, None);
                return None;
            }
            shape.points.extend(coords);
        }
        shape.num_points = (shape.points.len() / 2) as u32;
        shape.powers = t
            .get("powers")
            .and_then(|v| v.get_ref().as_array())
            .map(|a| a.iter().filter_map(|v| float_of(v.get_ref())).map(|f| f as f32).collect())
            .unwrap_or_else(|| vec![0.0; shape.num_points as usize]);
    }
    shape.smooth = t.get("smooth").and_then(|v| v.get_ref().as_bool()).unwrap_or(false);
    shape.name = t.get("name").and_then(|v| v.get_ref().as_str()).map(str::to_string);
    if let Some(extra) = t.get("extra").and_then(|v| v.get_ref().as_str()).and_then(|s| serde_json::from_str::<Json>(s).ok()).and_then(|j| j.as_object().cloned()) {
        shape.extra = extra;
    }
    if let Some(name) = factory_name_of(&shape) {
        reader.correction(line, key, "drawn points", name, CorrectionKind::FactoryShape);
    }
    Some(shape)
}
