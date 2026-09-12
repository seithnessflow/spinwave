//! The non-negotiable property, as tests: `.vital` → text → `.vital` is the
//! same patch, text → `.vital` → text is the same text, and the two
//! `.vital`s render the same samples bit for bit. Over every pack preset
//! and over fuzzed patches, which cover the parameter space far wider
//! than anything written by hand.

use std::collections::BTreeMap;

use serde_json::Value as Json;
use spinwave_params::preset::LineShape;
use spinwave_params::{parameters, Preset};

use super::{factory_shape, read, write, Blobs};
use crate::fuzz::{patch_for_seed, Wildness};
use crate::session::{NoteSpec, Session};

/// The engine-relevant state of a preset, in a form two presets can be
/// compared on: every table parameter at its effective value (absent is
/// the default), connections by slot with their five values, LFO shapes
/// with a trailing run of engine-default triangles trimmed, material and
/// unknown fields verbatim.
/// One connection's engine-relevant values: source, destination, amount,
/// bipolar, power, stereo, bypass, remap curve.
type ConnectionState = (String, String, f32, bool, f32, bool, bool, Option<LineShape>);

#[derive(Debug, PartialEq)]
struct State {
    values: BTreeMap<String, f32>,
    connections: BTreeMap<usize, ConnectionState>,
    lfos: Vec<LineShape>,
    wavetables: Option<Json>,
    sample: Option<Json>,
    materials: Option<Json>,
    unknown: BTreeMap<String, Json>,
    extra: BTreeMap<String, Json>,
    header: (String, String, String, String, String, [String; 4]),
}

fn state(preset: &Preset) -> State {
    let table = parameters();
    let mut values = BTreeMap::new();
    let mut unknown = BTreeMap::new();
    for details in table.iter() {
        if details.name.starts_with("modulation_") {
            continue;
        }
        let engine = preset.settings.values.get(&details.name).and_then(Json::as_f64).map(|v| v as f32).unwrap_or(details.default_value);
        values.insert(details.name.clone(), engine);
    }
    for (name, json) in &preset.settings.values {
        if table.lookup(name).is_none() {
            unknown.insert(name.clone(), json.clone());
        }
    }
    let mut connections = BTreeMap::new();
    for (slot, c) in preset.settings.modulations.iter().enumerate() {
        if !c.is_connected() {
            continue;
        }
        let n = slot + 1;
        let get = |s: &str| preset.settings.values.get(&format!("modulation_{n}_{s}")).and_then(Json::as_f64).map(|v| v as f32).unwrap_or(0.0);
        let curve = c.line_mapping.clone().filter(|s| super::factory_name_of(s) != Some("linear"));
        connections.insert(slot, (c.source.clone(), c.destination.clone(), get("amount"), get("bipolar") >= 0.5, get("power"), get("stereo") >= 0.5, get("bypass") >= 0.5, curve));
    }
    let triangle = factory_shape("triangle").unwrap();
    let mut lfos = preset.settings.lfos.clone();
    while lfos.last().is_some_and(|s| *s == triangle) {
        lfos.pop();
    }
    State {
        values,
        connections,
        lfos,
        wavetables: preset.settings.wavetables.clone(),
        sample: preset.settings.sample.clone(),
        materials: preset.settings.spinwave_materials.as_ref().map(|m| serde_json::to_value(m).unwrap()),
        unknown,
        extra: preset.extra.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        header: (
            preset.synth_version.clone(),
            preset.preset_name.clone(),
            preset.author.clone(),
            preset.preset_style.clone(),
            preset.comments.clone(),
            [preset.macro1.clone(), preset.macro2.clone(), preset.macro3.clone(), preset.macro4.clone()],
        ),
    }
}

/// Renders a preset the way the product does. Returns the samples and
/// their peak; the caller decides what a peak that low means.
fn render(preset: &Preset, label: &str) -> (Vec<f32>, f32) {
    let mut session = Session::with_output_dir(std::env::temp_dir());
    session.load_preset_json(&preset.to_json().unwrap()).unwrap_or_else(|e| panic!("{label}: {e}"));
    let notes = [NoteSpec { note: 48, velocity: 0.9, start: 0.0, duration: 0.6, channel: 0 }];
    let samples = session.render_samples(&notes, 1.0, 120.0);
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    (samples, peak)
}

/// -60 dBFS: below this a render is not a quiet sound but a broken
/// measurement, for any patch that is supposed to be audible.
const AUDIBLE: f32 = 1.0e-3;

/// Round-trips one preset and returns whether its render was audible.
/// Raw fallbacks split by cause. Only `continuous` is the format's own
/// precision at work; the other two are values the table has no spelling
/// for (a fraction where an index is expected, an index with no name).
#[derive(Default, Debug)]
struct RawCount {
    continuous: usize,
    fractional_index: usize,
    unnamed_index: usize,
}

fn count_raw(text: &str, raw: &mut RawCount, digit_hist: &mut BTreeMap<usize, usize>) {
    let table = parameters();
    let mut module = String::new();
    for line in text.lines() {
        if let Some(header) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            module = header.trim_matches('[').split('.').next().unwrap_or("").to_string();
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { continue };
        let value = value.trim();
        if value.starts_with('"') && value.chars().any(|c| c.is_ascii_digit()) {
            *digit_hist.entry(value.chars().filter(char::is_ascii_digit).count()).or_default() += 1;
        }
        if !value.starts_with("\"raw:") {
            continue;
        }
        let name = super::layout::table_name(&module, key.trim());
        match table.lookup(&name) {
            Some(d) if d.scale == spinwave_params::ParamScale::Indexed => {
                let engine: f32 = value.trim_matches('"').trim_start_matches("raw:").trim_end_matches('"').parse().unwrap_or(0.0);
                if engine.fract() != 0.0 {
                    raw.fractional_index += 1;
                } else {
                    raw.unnamed_index += 1;
                }
            }
            _ => raw.continuous += 1,
        }
    }
}

fn assert_round_trip(preset: &Preset, label: &str, raw_count: &mut RawCount, digit_hist: &mut BTreeMap<usize, usize>) -> bool {
    let written = write(preset);
    count_raw(&written.text, raw_count, digit_hist);

    let back = read(&written.text, &written.blobs);
    assert!(back.report.errors.is_empty(), "{label}: text does not read back: {:#?}\n{}", back.report.errors, written.text);
    let rebuilt = back.preset.expect("no errors, so a preset");
    let (a, b) = (state(preset), state(&rebuilt));
    if a != b {
        let mut lines = Vec::new();
        for (k, v) in &a.values {
            if b.values.get(k) != Some(v) {
                lines.push(format!("  {k}: {v} -> {:?}", b.values.get(k)));
            }
        }
        for (slot, c) in &a.connections {
            if b.connections.get(slot) != Some(c) {
                lines.push(format!("  slot {slot}: {c:?} -> {:?}", b.connections.get(slot)));
            }
        }
        for (slot, c) in &b.connections {
            if !a.connections.contains_key(slot) {
                lines.push(format!("  slot {slot}: (absent) -> {c:?}"));
            }
        }
        if a.lfos != b.lfos {
            lines.push(format!("  lfos: {:?} -> {:?}", a.lfos, b.lfos));
        }
        if a.header != b.header {
            lines.push(format!("  header: {:?} -> {:?}", a.header, b.header));
        }
        if a.unknown != b.unknown {
            lines.push("  unknown settings differ".into());
        }
        if a.extra != b.extra {
            lines.push("  extra differs".into());
        }
        if a.wavetables != b.wavetables || a.sample != b.sample || a.materials != b.materials {
            lines.push("  material differs".into());
        }
        panic!("{label}: .vital -> text -> .vital changed the patch:\n{}", lines.join("\n"));
    }

    let again = write(&rebuilt);
    assert_eq!(again.text, written.text, "{label}: text -> .vital -> text is not stable");
    assert_eq!(again.blobs, written.blobs, "{label}: sidecar changed across the round trip");

    let (original, peak) = render(preset, label);
    let (reconstructed, _) = render(&rebuilt, label);
    assert_eq!(original.len(), reconstructed.len());
    let first_difference = original.iter().zip(&reconstructed).position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(first_difference.is_none(), "{label}: renders differ at sample {}", first_difference.unwrap());
    peak > AUDIBLE
}

#[test]
fn every_pack_preset_round_trips_and_renders_identically() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../presets/packs");
    let mut raw = RawCount::default();
    let mut hist = BTreeMap::new();
    let mut count = 0;
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("vital") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let preset = Preset::from_json(text.trim_start_matches('\u{feff}')).unwrap();
        let label = path.file_name().unwrap().to_str().unwrap();
        // A pack preset is a sound somebody chose; a silent render of one
        // is the measurement failing, and two silent renders would compare
        // equal for the wrong reason.
        let audible = assert_round_trip(&preset, label, &mut raw, &mut hist);
        assert!(audible, "{label}: rendered below -60 dBFS; refusing a comparison that would pass on silence");
        count += 1;
    }
    assert!(count >= 5, "expected the five packs, found {count}");
    // Measured, committed: no hand-authored pack needs the raw fallback.
    assert_eq!(raw.continuous + raw.fractional_index + raw.unnamed_index, 0, "raw: fallbacks on the packs: {raw:?}");
}

#[test]
fn fuzzed_patches_round_trip_and_render_identically() {
    let mut raw = RawCount::default();
    let mut hist = BTreeMap::new();
    let mut lines = 0usize;
    let mut audible = 0usize;
    const SEEDS: u64 = 24;
    for seed in 0..SEEDS {
        let preset = patch_for_seed(seed, Wildness::Full);
        lines += write(&preset).text.lines().count();
        if assert_round_trip(&preset, &format!("fuzz seed {seed}"), &mut raw, &mut hist) {
            audible += 1;
        }
    }
    // A fully random patch may legitimately be silent (a level at zero, a
    // send to a bus that is off). Silent-against-silent proves nothing, so
    // the test needs most seeds audible to mean anything, and says so.
    assert!(audible * 2 >= SEEDS as usize, "only {audible} of {SEEDS} fuzzed patches were audible; the render comparison is not testing enough");
    // The raw fallback is measured here, over the widest input we have.
    // The bound is loose on purpose; the number itself is what matters,
    // and it is printed so a change shows.
    // The raw fallback, measured over the widest input we have, split by
    // cause. Measured on 2026-09-12: continuous 10, fractional index 404,
    // unnamed index 104, over 24313 lines — the format's own precision
    // misses on 0.04% of continuous values, and the rest is the fuzzer
    // writing fractions where the engine reads an index, or an index the
    // table has no name for. Both bounds are loose so a real regression
    // shows and noise does not.
    eprintln!("fuzz: {raw:?} over {lines} lines, {audible}/{SEEDS} audible; digits per value: {hist:?}");
    assert!(raw.continuous * 1000 < lines, "{} raw fallbacks on continuous values over {lines} lines is not rare", raw.continuous);
    assert!((raw.fractional_index + raw.unnamed_index) * 20 < lines, "{raw:?} over {lines} lines");
}

#[test]
fn a_hand_written_patch_reads_writes_and_reads_the_same() {
    let text = r#"
format = 1
synth_version = "1.0.7"
requires = "vital"

[preset]
name = "Hand"

[osc_1]
on = true
level = "-6.2 dB"
wave_frame = 128

[filter_1]
on = true
model = "ladder"
cutoff = "440 Hz"
resonance = "50%"

[[filter_1.mod]]
to = "cutoff"
from = "lfo_1"
amount = "+2 oct"
bipolar = false
rate = "audio"

[env_1]
attack = "90 ms"
release = "300 ms"
"#;
    let first = read(text, &Blobs::default());
    assert!(first.report.errors.is_empty(), "{:#?}", first.report.errors);
    let preset = first.preset.unwrap();
    // "+2 oct" was converted and the conversion reported.
    assert!(first.report.corrections.iter().any(|c| c.written == "+2 oct" && c.read_as == "+18.75%"), "{:#?}", first.report.corrections);
    let written = write(&preset);
    assert!(written.text.contains("amount = \"+18.75%\""), "{}", written.text);
    assert!(written.text.contains("rate = \"audio\""));
    assert!(written.text.contains("bipolar = false"), "an LFO connection always states its polarity\n{}", written.text);
    let second = read(&written.text, &written.blobs);
    assert!(second.report.errors.is_empty());
    assert_eq!(state(&second.preset.unwrap()), state(&preset));
    assert_eq!(write(&preset).text, written.text);
}

#[test]
fn refusals_carry_what_a_program_needs_to_fix_them() {
    let text = r#"
format = 1
[filter_1]
cutof = "440 Hz"
cutoff = 440
resonance = "500%"

[[filter_1.mod]]
to = "cutoff"
from = "macro_control_1"
amount = "+50%"
rate = "audio"
"#;
    let result = read(text, &Blobs::default());
    assert!(result.preset.is_none());
    let codes: Vec<_> = result.report.errors.iter().map(|e| (e.code, e.line)).collect();
    use spinwave_params::preset::ErrorCode::*;
    assert!(codes.contains(&(UnknownKey, 4)), "{codes:?}");
    assert!(codes.contains(&(MissingUnit, 5)), "{codes:?}");
    assert!(codes.contains(&(OutOfRange, 6)), "{codes:?}");
    assert!(codes.contains(&(BadRegime, 12)), "{codes:?}");
    let unknown = result.report.errors.iter().find(|e| e.code == UnknownKey).unwrap();
    assert_eq!(unknown.suggestion.as_deref(), Some("cutoff"));
    let missing = result.report.errors.iter().find(|e| e.code == MissingUnit).unwrap();
    assert!(missing.expected.as_deref().unwrap().contains("Hz"));
    // The report serialises, so an agent can read it.
    let json = serde_json::to_string(&result.report).unwrap();
    assert!(json.contains("\"unknown_key\""));
}

#[test]
fn a_missing_format_version_is_refused_not_defaulted() {
    let result = read("[osc_1]\non = true\n", &Blobs::default());
    assert!(result.preset.is_none());
    assert_eq!(result.report.errors[0].code, spinwave_params::preset::ErrorCode::MissingFormatVersion);
}

#[test]
fn requires_is_derived_and_checked() {
    let text = "format = 1\nrequires = \"vital\"\n[osc_4]\non = true\n";
    let result = read(text, &Blobs::default());
    assert!(result.preset.is_none());
    assert_eq!(result.report.errors[0].code, spinwave_params::preset::ErrorCode::RequiresMismatch);

    let mut preset = Preset::default();
    preset.settings.values.insert("osc_4_on".into(), Json::from(1.0));
    let written = write(&preset);
    assert!(written.text.contains("requires = \"spinwave\""), "{}", written.text);
}


#[test]
fn the_committed_examples_read_clean_and_round_trip() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../presets/text");
    let mut count = 0;
    for entry in std::fs::read_dir(&dir).unwrap().flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("spinwave") {
            continue;
        }
        let label = path.file_name().unwrap().to_str().unwrap().to_string();
        let text = std::fs::read_to_string(&path).unwrap();
        let first = read(&text, &Blobs::default());
        assert!(first.report.errors.is_empty(), "{label}: {:#?}", first.report.errors);
        let preset = first.preset.unwrap();
        let written = write(&preset);
        let second = read(&written.text, &written.blobs);
        assert!(second.report.errors.is_empty(), "{label}: rewritten text does not read back");
        assert_eq!(state(&second.preset.unwrap()), state(&preset), "{label}");
        assert_eq!(write(&preset).text, written.text, "{label}: not stable");
        count += 1;
    }
    assert_eq!(count, 3, "three examples are documented: simple, modulated, extensions");
}
