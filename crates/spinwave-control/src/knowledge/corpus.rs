//! The knowledge base's second store, `corpus/`: the STRUCTURE of real
//! patches — which modules are on, which coexist, which destinations are
//! modulated from which sources, how many connections a patch carries —
//! and, as the one concession to values, the range each parameter is
//! used in (p10 / p50 / p90 over the patches where it is active). The
//! structure is dictated by the engine and survives a mediocre author;
//! the values are taste, and only their spread is kept. No patch enters
//! the repo: the statistics do, the source path stays local.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use spinwave_params::{parameters, Preset};

use super::{engine_stamp, preset_hash, today, EngineStamp, SCHEMA_VERSION};
use crate::ops::explain::{active_parameters_for, value_of};
use crate::ops::{describe_without_pitch, render, render_seed, RenderMode, Scenario};

/// The modules a patch switches on, by their `*_on` key.
const MODULES: &[&str] = &[
    "osc_1", "osc_2", "osc_3", "osc_4", "sample", "noise", "filter_1", "filter_2", "chorus", "compressor", "delay",
    "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb", "convolution", "frequency_shifter", "bus_a", "bus_b",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Range {
    pub p10: f32,
    pub p50: f32,
    pub p90: f32,
    /// Patches the parameter was active in.
    pub n: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorpusInfo {
    pub id: String,
    pub patches: usize,
    /// Excluded by the quality filter, with the reason counts.
    pub excluded: BTreeMap<String, usize>,
    pub path_local: bool,
}

/// `knowledge/corpus/<id>/structure.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Structure {
    pub kind: String,
    pub schema: u32,
    pub corpus: CorpusInfo,
    pub engine: EngineStamp,
    pub date: String,
    /// Fraction of patches with the module on.
    pub modules_on: BTreeMap<String, f32>,
    /// Fraction of patches with both modules on (`a&b`, a < b).
    pub cooccurrence: BTreeMap<String, f32>,
    /// Fraction of patches with at least one connection into the destination.
    pub destinations_modulated: BTreeMap<String, f32>,
    /// Fraction of patches with the connection `source->destination`.
    pub source_to_destination: BTreeMap<String, f32>,
    pub connections_per_patch: Range,
    /// The used range of every parameter active in at least three patches.
    pub value_ranges_used: BTreeMap<String, Range>,
}

/// One patch's line in `catalogue.json`: what it measures as, and the
/// flags the quality filter reads.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CatalogueEntry {
    pub label: String,
    pub preset_hash: String,
    pub peak_dbfs: f32,
    pub rms_dbfs: f32,
    pub centroid_hz: f32,
    pub attack_seconds: f32,
    pub stereo_width: f32,
    pub clipping_ratio: f32,
    pub silent: bool,
    pub modules_on: Vec<String>,
    pub connections: usize,
    /// Why it was excluded from the structure, if it was.
    pub excluded: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Catalogue {
    pub kind: String,
    pub schema: u32,
    pub corpus: String,
    pub engine: EngineStamp,
    pub date: String,
    pub entries: Vec<CatalogueEntry>,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = p * (sorted.len() - 1) as f32;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let t = pos - lo as f32;
    sorted[lo] * (1.0 - t) + sorted[hi.min(sorted.len() - 1)] * t
}

fn range_of(values: &mut [f32]) -> Range {
    values.sort_by(|a, b| a.total_cmp(b));
    Range { p10: percentile(values, 0.1), p50: percentile(values, 0.5), p90: percentile(values, 0.9), n: values.len() }
}

fn on(preset: &Preset, module: &str) -> bool {
    value_of(preset, &format!("{module}_on")) >= 0.5
}

/// Builds the structure and the catalogue of `patches` (label, path),
/// rendering each once at the Lite scenario for its flags. A patch that
/// clips (clipping ratio above 1 %) or renders silent is catalogued,
/// flagged, and left out of the structure; `min_quality` off keeps them.
pub fn build(id: &str, patches: &[(String, PathBuf)], min_quality: bool, mut report: impl FnMut(&str)) -> Result<(Structure, Catalogue), String> {
    let scenario = Scenario::lite();
    let table = parameters();
    let mut entries = Vec::new();
    let mut kept: Vec<Preset> = Vec::new();
    let mut excluded: BTreeMap<String, usize> = BTreeMap::new();
    let mut session = crate::ops::session();
    for (label, path) in patches {
        let preset = match crate::ops::load_patch(&path.to_string_lossy()) {
            Ok(p) => p,
            Err(e) => {
                *excluded.entry("load".into()).or_insert(0) += 1;
                report(&format!("{label}: {e}"));
                continue;
            }
        };
        let modules_on: Vec<String> = MODULES.iter().filter(|m| on(&preset, m)).map(|m| m.to_string()).collect();
        let connections = preset.settings.modulations.iter().filter(|m| m.is_connected()).count();
        let d = match render(&mut session, &preset, &scenario, render_seed(1, 0)) {
            Ok(r) => Some(describe_without_pitch(&r)),
            Err(crate::ops::OpError::Silent { .. }) => None,
            Err(e) => {
                *excluded.entry("render".into()).or_insert(0) += 1;
                report(&format!("{label}: {e:?}"));
                continue;
            }
        };
        let silent = d.is_none();
        let clipping_ratio = d.as_ref().map_or(0.0, |d| d.clipping.ratio);
        let mut why = None;
        if silent {
            why = Some("silent".to_string());
        } else if clipping_ratio > 0.01 {
            why = Some("clips".to_string());
        }
        if min_quality {
            if let Some(w) = &why {
                *excluded.entry(w.clone()).or_insert(0) += 1;
            }
        }
        entries.push(CatalogueEntry {
            label: label.clone(),
            preset_hash: preset_hash(&preset),
            peak_dbfs: d.as_ref().map_or(-180.0, |d| d.peak_dbfs),
            rms_dbfs: d.as_ref().map_or(-180.0, |d| d.rms_dbfs),
            centroid_hz: d.as_ref().map_or(0.0, |d| d.centroid_hz),
            attack_seconds: d.as_ref().map_or(0.0, |d| d.attack_seconds),
            stereo_width: d.as_ref().map_or(0.0, |d| d.stereo_width),
            clipping_ratio,
            silent,
            modules_on,
            connections,
            excluded: if min_quality { why.clone() } else { None },
        });
        if !(min_quality && why.is_some()) {
            kept.push(preset);
        }
    }
    let n = kept.len().max(1) as f32;
    let mut modules_on: BTreeMap<String, f32> = BTreeMap::new();
    let mut cooccurrence: BTreeMap<String, f32> = BTreeMap::new();
    let mut destinations: BTreeMap<String, f32> = BTreeMap::new();
    let mut pairs: BTreeMap<String, f32> = BTreeMap::new();
    let mut counts: Vec<f32> = Vec::new();
    let mut values: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for preset in &kept {
        let live: Vec<&str> = MODULES.iter().copied().filter(|m| on(preset, m)).collect();
        for m in &live {
            *modules_on.entry(m.to_string()).or_insert(0.0) += 1.0;
        }
        for (i, a) in live.iter().enumerate() {
            for b in &live[i + 1..] {
                *cooccurrence.entry(format!("{a}&{b}")).or_insert(0.0) += 1.0;
            }
        }
        let mut dests_seen = std::collections::BTreeSet::new();
        let mut pairs_seen = std::collections::BTreeSet::new();
        let mut count = 0usize;
        for m in preset.settings.modulations.iter().filter(|m| m.is_connected()) {
            count += 1;
            dests_seen.insert(m.destination.clone());
            pairs_seen.insert(format!("{}->{}", m.source, m.destination));
        }
        for d in dests_seen {
            *destinations.entry(d).or_insert(0.0) += 1.0;
        }
        for p in pairs_seen {
            *pairs.entry(p).or_insert(0.0) += 1.0;
        }
        counts.push(count as f32);
        for details in active_parameters_for(preset, RenderMode::Faithful) {
            if table.lookup(&details.name).is_some_and(|d| d.max > d.min) {
                values.entry(details.name.clone()).or_default().push(value_of(preset, &details.name));
            }
        }
    }
    for map in [&mut modules_on, &mut cooccurrence, &mut destinations, &mut pairs] {
        for v in map.values_mut() {
            *v /= n;
        }
    }
    let value_ranges_used: BTreeMap<String, Range> =
        values.into_iter().filter(|(_, v)| v.len() >= 3).map(|(k, mut v)| (k, range_of(&mut v))).collect();
    let stamp = engine_stamp();
    let date = today();
    let structure = Structure {
        kind: "corpus.structure".into(),
        schema: SCHEMA_VERSION,
        corpus: CorpusInfo { id: id.into(), patches: kept.len(), excluded, path_local: true },
        engine: stamp.clone(),
        date: date.clone(),
        modules_on,
        cooccurrence,
        destinations_modulated: destinations,
        source_to_destination: pairs,
        connections_per_patch: range_of(&mut counts),
        value_ranges_used,
    };
    let catalogue = Catalogue { kind: "corpus.catalogue".into(), schema: SCHEMA_VERSION, corpus: id.into(), engine: stamp, date, entries };
    Ok((structure, catalogue))
}

pub fn save(dir: &Path, structure: &Structure, catalogue: &Catalogue) -> Result<(), String> {
    let folder = dir.join("corpus").join(&structure.corpus.id);
    std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
    for (name, text) in [
        ("structure.json", serde_json::to_string_pretty(structure).map_err(|e| e.to_string())?),
        ("catalogue.json", serde_json::to_string_pretty(catalogue).map_err(|e| e.to_string())?),
    ] {
        std::fs::write(folder.join(name), text + "\n").map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(())
}

/// The structure of corpus `id`, if it exists and is fresh.
pub fn load(dir: &Path, id: &str) -> Option<Structure> {
    let text = std::fs::read_to_string(dir.join("corpus").join(id).join("structure.json")).ok()?;
    let structure: Structure = serde_json::from_str(&text).ok()?;
    (structure.engine.fingerprint == super::ENGINE_FINGERPRINT).then_some(structure)
}

/// The value ranges of the corpora present under `dir`, fresh ones only;
/// several corpora take the widest range per parameter.
pub fn ranges(dir: &Path) -> BTreeMap<String, Range> {
    let mut out: BTreeMap<String, Range> = BTreeMap::new();
    let Ok(entries) = std::fs::read_dir(dir.join("corpus")) else { return out };
    for entry in entries.flatten() {
        let id = entry.file_name().to_string_lossy().to_string();
        let Some(structure) = load(dir, &id) else { continue };
        for (name, range) in structure.value_ranges_used {
            out.entry(name)
                .and_modify(|r| {
                    r.p10 = r.p10.min(range.p10);
                    r.p90 = r.p90.max(range.p90);
                    r.n += range.n;
                })
                .or_insert(range);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_and_ranges_read_as_expected() {
        let mut v = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let r = range_of(&mut v);
        assert_eq!((r.p50, r.n), (3.0, 5));
        assert!((r.p10 - 1.4).abs() < 1e-6 && (r.p90 - 4.6).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn the_structure_of_two_saws_says_what_they_share() {
        let dir = std::env::temp_dir().join(format!("spinwave-corpus-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = crate::ops::tests::saw_patch();
        let b = a.clone();
        a.settings.values.insert("reverb_on".into(), 1.0.into());
        a.settings.modulations.push(spinwave_params::preset::ModulationConnection { source: "lfo_1".into(), destination: "filter_1_cutoff".into(), ..Default::default() });
        a.settings.values.insert("modulation_1_amount".into(), 0.4.into());
        let pa = dir.join("a.vital");
        let pb = dir.join("b.vital");
        std::fs::write(&pa, a.to_json().unwrap()).unwrap();
        std::fs::write(&pb, b.to_json().unwrap()).unwrap();
        let (s, c) = build("test", &[("a".into(), pa), ("b".into(), pb)], true, |_| {}).unwrap();
        assert_eq!(s.corpus.patches, 2);
        assert_eq!(s.modules_on["osc_1"], 1.0);
        assert_eq!(s.modules_on["reverb"], 0.5);
        assert_eq!(s.cooccurrence["filter_1&reverb"], 0.5);
        assert_eq!(s.destinations_modulated["filter_1_cutoff"], 0.5);
        assert_eq!(s.source_to_destination["lfo_1->filter_1_cutoff"], 0.5);
        assert_eq!(c.entries.len(), 2);
        assert!(c.entries.iter().all(|e| e.excluded.is_none() && !e.silent));
        save(&dir, &s, &c).unwrap();
        assert!(load(&dir, "test").is_some());
        assert!(ranges(&dir).is_empty(), "two patches are under the three the ranges need");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
