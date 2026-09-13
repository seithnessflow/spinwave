//! The knowledge base's third store, `declared/`: the dictionary of the
//! trade's words — growl, reese, pluck, wub, pad, supersaw — each
//! translated into descriptors this engine measures and into the patch
//! structure expected, with its sources (cited, never copied) and its
//! trust. What a source says is a hypothesis until `knowledge validate`
//! builds the patch the entry describes, renders it, and compares the
//! measurement to `expects`: every descriptor inside its range and the
//! entry is `validated`, any outside and it is `refuted` with the
//! measured value kept beside the claim. A refuted claim is never
//! edited to fit. Measured beats declared.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use spinwave_params::preset::ModulationConnection;
use spinwave_params::Preset;

use super::{engine_stamp, today, EngineStamp};
#[cfg(test)]
use super::SCHEMA_VERSION;
use crate::ops::explain::Quality;
use crate::ops::descriptors::describe_with;
use crate::ops::{render, render_seed, Descriptors, RenderMode, Scenario};
use crate::session::{NoteSpec, SAMPLE_RATE};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// The measurable expectation: each named descriptor inside `[min, max]`
/// (a missing bound is open), and the structure the patch must show.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Expects {
    #[serde(default)]
    pub descriptors: BTreeMap<String, [Option<f32>; 2]>,
    #[serde(default)]
    pub modules_on: Vec<String>,
    #[serde(default)]
    pub destinations_modulated: Vec<String>,
}

/// The patch the entry describes, in engine values, plus its scenario.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TermPatch {
    pub settings: BTreeMap<String, f32>,
    #[serde(default)]
    pub modulations: Vec<TermModulation>,
    /// MIDI note, hold seconds, render seconds; Faithful mode.
    pub note: i32,
    pub hold: f32,
    pub seconds: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TermModulation {
    pub source: String,
    pub destination: String,
    pub amount: f32,
    #[serde(default)]
    pub bipolar: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Validation {
    /// `unvalidated` | `validated` | `refuted`.
    pub status: String,
    #[serde(default)]
    pub measured: BTreeMap<String, f32>,
    #[serde(default)]
    pub failed: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<EngineStamp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date: Option<String>,
}

/// `knowledge/declared/terms/<term>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Term {
    pub kind: String,
    pub schema: u32,
    pub term: String,
    /// The claim, in our words.
    pub says: String,
    pub sources: Vec<Source>,
    /// `high` | `medium` | `low` — how far the sources are trusted.
    pub trust: String,
    pub expects: Expects,
    pub patch: TermPatch,
    #[serde(default)]
    pub validation: Validation,
}

/// The descriptor vocabulary `expects` may name, resolved on a render.
pub const VOCABULARY: [&str; 20] = [
    "f0_hz", "centroid_hz", "rolloff_hz", "attack_s", "decay_20db_s", "tail_db", "peak_db", "rms_db", "width",
    "mono_compat_db", "harmonicity", "inharmonicity", "flatness", "movement_db", "movement_held_db",
    "brightness_movement_held_st", "movement_rate_hz", "brightness_st", "warmth_db", "harshness_db",
];

/// `movement_db` over the whole render counts the release and the
/// silence after it, so any decaying sound "moves" by tens of dB; the
/// held part alone (from 100 ms after the note-on to the note-off, in
/// the 50 ms windows of the trajectory) is what a beating or a wobble
/// shows on.
fn movement_held_db(d: &Descriptors, hold: f32) -> Option<f32> {
    let first = 2usize;
    let last = ((hold / 0.05).floor() as usize).min(d.rms_trajectory_dbfs.len());
    if last <= first + 2 {
        return None;
    }
    let held = &d.rms_trajectory_dbfs[first..last];
    let mean = held.iter().sum::<f32>() / held.len() as f32;
    Some((held.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / held.len() as f32).sqrt())
}

/// The same window on the centroid trajectory, in semitones: a wobble
/// whose level a compressor has flattened still moves here.
fn brightness_movement_held_st(d: &Descriptors, hold: f32) -> Option<f32> {
    let first = 2usize;
    let last = ((hold / 0.05).floor() as usize).min(d.centroid_trajectory_hz.len());
    if last <= first + 2 {
        return None;
    }
    let held: Vec<f32> = d.centroid_trajectory_hz[first..last].iter().map(|hz| 12.0 * hz.max(1.0).log2()).collect();
    let mean = held.iter().sum::<f32>() / held.len() as f32;
    Some((held.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / held.len() as f32).sqrt())
}

fn measure_all(d: &Descriptors, hold: f32) -> BTreeMap<String, f32> {
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: Option<f32>| {
        if let Some(v) = v {
            m.insert(k.to_string(), v);
        }
    };
    put("f0_hz", d.f0_hz);
    put("centroid_hz", Some(d.centroid_hz));
    put("rolloff_hz", Some(d.rolloff_hz));
    put("attack_s", Some(d.attack_seconds));
    put("decay_20db_s", d.decay_20db_seconds);
    put("tail_db", Some(d.tail_dbfs));
    put("peak_db", Some(d.peak_dbfs));
    put("rms_db", Some(d.rms_dbfs));
    put("width", Some(d.stereo_width));
    put("mono_compat_db", Some(d.mono_compatibility_db));
    put("harmonicity", d.harmonicity);
    put("inharmonicity", d.inharmonicity);
    put("flatness", Some(d.spectral_flatness));
    put("movement_db", Some(Quality::Movement.measure(d)));
    put("movement_held_db", movement_held_db(d, hold));
    put("brightness_movement_held_st", brightness_movement_held_st(d, hold));
    // The strongest periodic movement the analysis found, Hz (an LFO's
    // rate; a unison's beating), when it found one.
    put("movement_rate_hz", d.movement_rates_hz.first().copied());
    put("brightness_st", Some(Quality::Brightness.measure(d)));
    put("warmth_db", Some(Quality::Warmth.measure(d)));
    put("harshness_db", Some(Quality::Harshness.measure(d)));
    m
}

pub fn terms_dir(dir: &Path) -> PathBuf {
    dir.join("declared").join("terms")
}

pub fn load_all(dir: &Path) -> Vec<Term> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(terms_dir(dir)) else { return out };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(term) = serde_json::from_str::<Term>(&text) {
                out.push(term);
            }
        }
    }
    out.sort_by(|a, b| a.term.cmp(&b.term));
    out
}

pub fn save(dir: &Path, term: &Term) -> Result<(), String> {
    let folder = terms_dir(dir);
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    let text = serde_json::to_string_pretty(term).map_err(|e| e.to_string())?;
    std::fs::write(folder.join(format!("{}.json", term.term)), text + "\n").map_err(|e| e.to_string())
}

/// The entry's patch as a preset.
pub fn preset_of(term: &Term) -> Preset {
    let mut preset = Preset { preset_name: term.term.clone(), ..Default::default() };
    for (k, v) in &term.patch.settings {
        preset.settings.values.insert(k.clone(), Json::from(*v as f64));
    }
    for (i, m) in term.patch.modulations.iter().enumerate() {
        preset.settings.modulations.push(ModulationConnection { source: m.source.clone(), destination: m.destination.clone(), ..Default::default() });
        preset.settings.values.insert(format!("modulation_{}_amount", i + 1), Json::from(m.amount as f64));
        preset.settings.values.insert(format!("modulation_{}_bipolar", i + 1), Json::from(if m.bipolar { 1.0 } else { 0.0 }));
    }
    preset
}

/// Builds, renders (Faithful mode, the entry's note), measures, and
/// writes the verdict into the entry: every named descriptor inside its
/// range and every structural expectation met → `validated`; otherwise
/// `refuted`, with what failed. The claim itself is never touched.
pub fn validate(term: &mut Term) -> Result<(), String> {
    let preset = preset_of(term);
    let scenario = Scenario {
        notes: vec![NoteSpec { note: term.patch.note, start: 0.0, duration: term.patch.hold, velocity: 0.8, channel: 0 }],
        seconds: term.patch.seconds,
        bpm: 120.0,
        mode: RenderMode::Faithful,
    };
    let mut session = crate::ops::session();
    let r = render(&mut session, &preset, &scenario, render_seed(1, 0)).map_err(|e| format!("{}: {e:?}", term.term))?;
    let d = describe_with(&r.samples, SAMPLE_RATE, true);
    let measured = measure_all(&d, term.patch.hold);
    let mut failed = Vec::new();
    for (name, [min, max]) in &term.expects.descriptors {
        if !VOCABULARY.contains(&name.as_str()) {
            return Err(format!("{}: `{name}` is not a descriptor the dictionary knows ({})", term.term, VOCABULARY.join(", ")));
        }
        match measured.get(name) {
            None => failed.push(format!("{name}: not measurable on this render")),
            Some(v) => {
                if min.is_some_and(|lo| *v < lo) || max.is_some_and(|hi| *v > hi) {
                    failed.push(format!("{name}: measured {v:.4}, expected [{}, {}]", min.map_or("-".into(), |x| format!("{x}")), max.map_or("-".into(), |x| format!("{x}"))));
                }
            }
        }
    }
    for module in &term.expects.modules_on {
        if term.patch.settings.get(&format!("{module}_on")).copied().unwrap_or(0.0) < 0.5 {
            failed.push(format!("structure: {module} is not on in the entry's patch"));
        }
    }
    for dest in &term.expects.destinations_modulated {
        if !term.patch.modulations.iter().any(|m| &m.destination == dest) {
            failed.push(format!("structure: nothing modulates {dest} in the entry's patch"));
        }
    }
    term.validation = Validation {
        status: if failed.is_empty() { "validated".into() } else { "refuted".into() },
        measured,
        failed,
        engine: Some(engine_stamp()),
        date: Some(today()),
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_claim_is_validated_when_the_patch_measures_as_it_says_and_refuted_when_not() {
        let mut settings = BTreeMap::new();
        for (k, v) in [("osc_1_on", 1.0), ("osc_1_level", 0.7), ("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 60.0), ("env_1_sustain", 1.0)] {
            settings.insert(k.to_string(), v);
        }
        let mut term = Term {
            kind: "declared.term".into(),
            schema: SCHEMA_VERSION,
            term: "test".into(),
            says: "a filtered saw is darker than 2 kHz".into(),
            sources: vec![],
            trust: "high".into(),
            expects: Expects { descriptors: BTreeMap::from([("centroid_hz".to_string(), [None, Some(2000.0)])]), modules_on: vec!["filter_1".into()], destinations_modulated: vec![] },
            patch: TermPatch { settings, modulations: vec![], note: 48, hold: 0.5, seconds: 0.8 },
            validation: Validation::default(),
        };
        validate(&mut term).unwrap();
        assert_eq!(term.validation.status, "validated", "{:?}", term.validation.failed);
        assert!(term.validation.measured.contains_key("centroid_hz"));
        term.expects.descriptors.insert("width".into(), [Some(0.5), None]);
        validate(&mut term).unwrap();
        assert_eq!(term.validation.status, "refuted");
        assert!(term.validation.failed[0].starts_with("width: measured"));
        term.expects.descriptors.insert("nonsense".into(), [None, None]);
        assert!(validate(&mut term).is_err());
    }
}
