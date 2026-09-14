//! The knowledge base's first store, `measured/`: what each parameter
//! does in this engine, measured and kept (`notes/knowledge-base-design.md`).
//!
//! An observation is one parameter moved a quarter of its range in one
//! patch under one scenario: the band distance the move makes and the
//! signed change of every quality. It carries the CONTEXT it was made in
//! — the switches, models and connections that gate the parameter,
//! evaluated on that patch ([`context_key`]) — and the engine fingerprint
//! that produced it. A consumer asks for a parameter in a context and
//! gets the fresh observations with that exact key, or nothing: no
//! observation is passed off as valid in a context it was not made in.
//!
//! Everything derived regenerates by command (`spinwave-cli knowledge
//! measure`); an observation whose fingerprint is not the running
//! engine's is stale — readable, counted, never weighed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use spinwave_params::{parameters, ParamDetails, Preset};

use crate::ops::explain::{active_parameters_for, step, value_of, Quality};
use crate::ops::{describe_without_pitch, distance, parallel, render, render_seed, Budget, Descriptors, OpError, RenderMode, Scenario};
use crate::sensitivity::{base_settings, build, context_for, split_indexed};
use crate::session::{Session, SAMPLE_RATE};

pub mod corpus;
pub mod terms;

/// The engine's fingerprint: sources and data of the engine crates plus
/// the toolchain (`build.rs`).
pub const ENGINE_FINGERPRINT: &str = env!("SPINWAVE_ENGINE_FINGERPRINT");
/// The descriptors' own fingerprint, apart from the engine's.
pub const DESCRIPTORS_FINGERPRINT: &str = env!("SPINWAVE_DESCRIPTORS_FINGERPRINT");
pub const GIT_COMMIT: &str = env!("SPINWAVE_GIT_COMMIT");
pub const GIT_DIRTY: bool = matches!(env!("SPINWAVE_GIT_DIRTY").as_bytes(), b"1");

/// The schema of the files under `knowledge/`; bumped with any
/// incompatible change to the structs below.
pub const SCHEMA_VERSION: u32 = 1;

/// Which engine produced an entry. `fingerprint` decides validity, the
/// rest is provenance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EngineStamp {
    pub fingerprint: String,
    pub descriptors: String,
    pub git: String,
    pub dirty: bool,
}

pub fn engine_stamp() -> EngineStamp {
    EngineStamp {
        fingerprint: ENGINE_FINGERPRINT.to_string(),
        descriptors: DESCRIPTORS_FINGERPRINT.to_string(),
        git: GIT_COMMIT.to_string(),
        dirty: GIT_DIRTY,
    }
}

/// Today as `YYYY-MM-DD` (UTC), from the system clock; no calendar crate.
pub fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = (secs / 86_400) as i64;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

/// Where the observation was made: its context key, the patch's label
/// and hash.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextRef {
    pub key: String,
    pub origin: String,
    pub preset_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Step {
    pub from: f32,
    pub to: f32,
    pub fraction_of_range: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Effect {
    /// The band distance between the render before and after the step.
    pub distance_db: f32,
    /// Signed change of each quality (`Quality::measure`), in its unit.
    pub deltas: BTreeMap<String, f32>,
    /// Per octave band, after minus before, dB.
    pub bands_db: [f32; 8],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub context: ContextRef,
    pub scenario: String,
    pub step: Step,
    pub effect: Effect,
    pub renders: u32,
    pub engine: EngineStamp,
    pub date: String,
}

impl Observation {
    pub fn is_fresh(&self) -> bool {
        self.engine.fingerprint == ENGINE_FINGERPRINT
    }
}

/// `knowledge/measured/params/<name>.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParameterFile {
    pub kind: String,
    pub schema: u32,
    pub name: String,
    pub observations: Vec<Observation>,
}

impl ParameterFile {
    fn new(name: &str) -> ParameterFile {
        ParameterFile { kind: "measured.parameter".into(), schema: SCHEMA_VERSION, name: name.into(), observations: Vec::new() }
    }
}

/// `SPINWAVE_KNOWLEDGE`, or `knowledge/` beside `presets/` at the repo
/// root (found the way the racks are).
pub fn knowledge_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SPINWAVE_KNOWLEDGE") {
        return dir.into();
    }
    Session::racks_dir().parent().and_then(|p| p.parent()).map(|root| root.join("knowledge")).unwrap_or_else(|| "knowledge".into())
}

/// The `measured/` store, loaded whole (a few hundred small files).
#[derive(Default)]
pub struct Store {
    pub params: BTreeMap<String, ParameterFile>,
}

impl Store {
    pub fn load(dir: &Path) -> Store {
        let mut store = Store::default();
        let Ok(entries) = std::fs::read_dir(dir.join("measured").join("params")) else { return store };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            if let Ok(file) = serde_json::from_str::<ParameterFile>(&text) {
                store.params.insert(file.name.clone(), file);
            }
        }
        store
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Fresh observations of `name` made in context `key`, minus those
    /// from `exclude_origin` (the leave-one-out of the agreement test).
    pub fn lookup(&self, name: &str, key: &str, exclude_origin: Option<&str>) -> Vec<&Observation> {
        self.params
            .get(name)
            .map(|f| {
                f.observations
                    .iter()
                    .filter(|o| o.is_fresh() && o.context.key == key && exclude_origin.is_none_or(|x| o.context.origin != x))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The prior weight of `name` in context `key`: the median distance
    /// of the fresh observations, shrunk by their count (`n / (n + 5)`:
    /// three patches must not rank as confidently as fifty), with `n`.
    pub fn prior(&self, name: &str, key: &str, exclude_origin: Option<&str>) -> Option<(f32, usize)> {
        let found = self.lookup(name, key, exclude_origin);
        if found.is_empty() {
            return None;
        }
        let mut distances: Vec<f32> = found.iter().map(|o| o.effect.distance_db).collect();
        distances.sort_by(|a, b| a.total_cmp(b));
        let n = distances.len();
        let median = if n % 2 == 1 { distances[n / 2] } else { 0.5 * (distances[n / 2 - 1] + distances[n / 2]) };
        Some((median * n as f32 / (n as f32 + SHRINK_K), n))
    }

    fn upsert(&mut self, name: &str, observation: Observation) {
        let file = self.params.entry(name.to_string()).or_insert_with(|| ParameterFile::new(name));
        // One observation per (origin, scenario): a re-measurement replaces.
        file.observations.retain(|o| !(o.context.origin == observation.context.origin && o.scenario == observation.scenario));
        file.observations.push(observation);
        file.observations.sort_by(|a, b| a.context.origin.cmp(&b.context.origin).then(a.scenario.cmp(&b.scenario)));
    }

    fn save_param(&self, dir: &Path, name: &str) -> Result<(), String> {
        let Some(file) = self.params.get(name) else { return Ok(()) };
        let folder = dir.join("measured").join("params");
        std::fs::create_dir_all(&folder).map_err(|e| format!("{}: {e}", folder.display()))?;
        let path = folder.join(format!("{name}.json"));
        let text = serde_json::to_string_pretty(file).map_err(|e| e.to_string())?;
        std::fs::write(&path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
impl Store {
    pub(crate) fn upsert_for_test(&mut self, name: &str, observation: Observation) {
        self.upsert(name, observation);
    }
    pub(crate) fn save_for_test(&self, dir: &Path, name: &str) {
        self.save_param(dir, name).unwrap();
    }
}

/// The evidence shrinkage constant of [`Store::prior`].
pub const SHRINK_K: f32 = 5.0;

/// The content-addressed cache of measured effects: a patch, a parameter
/// moved to a value, a scenario, the engine and the descriptors that
/// measured it, give the same [`Effect`] every time. Keyed by the
/// SHA-256 of all of those; the value is the effect (thirty floats),
/// never the audio. Lives outside the repo: `SPINWAVE_RENDER_CACHE`, or
/// `%LOCALAPPDATA%/spinwave/render-cache` (`~/.cache/spinwave/render-cache`
/// elsewhere); `SPINWAVE_RENDER_CACHE=off` disables it. A sensitivity
/// sweep that revisits a patch hits it on every parameter and renders
/// nothing (`knowledge measure --patches` a second time: seconds where
/// the first took minutes).
pub struct EffectCache {
    dir: Option<PathBuf>,
}

impl EffectCache {
    pub fn open() -> EffectCache {
        let dir = match std::env::var("SPINWAVE_RENDER_CACHE") {
            Ok(v) if v == "off" || v == "0" => None,
            Ok(v) => Some(PathBuf::from(v)),
            Err(_) => {
                let base = std::env::var_os("LOCALAPPDATA")
                    .map(PathBuf::from)
                    .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
                    .unwrap_or_else(std::env::temp_dir);
                Some(base.join("spinwave").join("render-cache"))
            }
        };
        EffectCache { dir }
    }

    pub fn disabled() -> EffectCache {
        EffectCache { dir: None }
    }

    pub fn is_on(&self) -> bool {
        self.dir.is_some()
    }

    /// The key of one measured step.
    pub fn key(preset_hash: &str, name: &str, to: f32, scenario_id: &str) -> String {
        let mut hasher = Sha256::new();
        for part in [preset_hash, name, &format!("{to:.6}"), scenario_id, ENGINE_FINGERPRINT, DESCRIPTORS_FINGERPRINT] {
            hasher.update(part.as_bytes());
            hasher.update([0u8]);
        }
        format!("{:x}", hasher.finalize())
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        self.dir.as_ref().map(|d| d.join(&key[..2]).join(format!("{key}.json")))
    }

    pub fn get(&self, key: &str) -> Option<Effect> {
        let path = self.path(key)?;
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn put(&self, key: &str, effect: &Effect) {
        let Some(path) = self.path(key) else { return };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string(effect) {
            let _ = std::fs::write(path, text);
        }
    }
}

/// SHA-256 of the preset's JSON, the observation's handle back to its patch.
pub fn preset_hash(preset: &Preset) -> String {
    let json = preset.to_json().unwrap_or_default();
    format!("sha256:{:x}", Sha256::digest(json.as_bytes()))
}

fn fmt(v: f32) -> String {
    if v == v.round() { format!("{}", v as i64) } else { format!("{v:.3}") }
}

/// The context of a parameter on a patch: every switch, model, engine,
/// routing and connection that gates it, as `active_parameters_for`
/// reads them, in one canonical string. Two patches with the same key
/// are the same context for that parameter as far as the dependency
/// table knows. A parameter nothing gates (a global) has key `global`.
pub fn context_key(preset: &Preset, details: &ParamDetails) -> String {
    let mut parts = gating_parts(preset, details);
    // A rate control is read through its sync switch: `delay_frequency`
    // is inert while `delay_sync` is tempo-synced and `delay_tempo` while
    // it is not (every `*_frequency` / `*_tempo` of the table has a
    // `*_sync` sibling: the LFOs, the randoms, the delay and its aux, the
    // chorus, the flanger, the phaser). The switch is part of the context.
    for suffix in ["_frequency", "_tempo"] {
        if let Some(prefix) = details.name.strip_suffix(suffix) {
            let sync = format!("{prefix}_sync");
            if parameters().is_parameter(&sync) {
                parts.push(format!("{sync}={}", fmt(value_of(preset, &sync))));
            }
        }
    }
    if parts.is_empty() { "global".to_string() } else { parts.join(";") }
}

/// The switches, models, engines, routings and connections that gate
/// `details` on `preset`, as `key=value` parts.
fn gating_parts(preset: &Preset, details: &ParamDetails) -> Vec<String> {
    let name = details.name.as_str();
    let mut parts: Vec<String> = Vec::new();
    let setting = |key: &str| format!("{key}={}", fmt(value_of(preset, key)));
    let effects = ["chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb", "convolution", "frequency_shifter"];

    for bus in ["bus_a", "bus_b"] {
        if let Some(rest) = name.strip_prefix(&format!("{bus}_")) {
            parts.push(setting(&format!("{bus}_on")));
            for effect in effects {
                if rest.starts_with(&format!("{effect}_")) {
                    parts.push(setting(&format!("{bus}_{effect}_on")));
                    if effect == "distortion" {
                        parts.push(setting(&format!("{bus}_distortion_type")));
                    }
                }
            }
            return parts;
        }
    }
    for effect in effects {
        if name.starts_with(&format!("{effect}_")) {
            parts.push(setting(&format!("{effect}_on")));
            match effect {
                "distortion" => parts.push(setting("distortion_type")),
                "filter_fx" => {
                    parts.push(setting("filter_fx_model"));
                    parts.push(setting("filter_fx_style"));
                }
                _ => {}
            }
            return parts;
        }
    }
    if name.starts_with("sample_") {
        parts.push(setting("sample_on"));
        parts.push(setting("sample_destination"));
        return parts;
    }
    if name.starts_with("noise_") {
        parts.push(setting("noise_on"));
        parts.push(setting("noise_destination"));
        return parts;
    }
    if let Some((family, index, rest)) = split_indexed(name) {
        match family {
            "osc" => {
                parts.push(setting(&format!("osc_{index}_on")));
                parts.push(setting(&format!("osc_{index}_engine")));
                parts.push(setting(&format!("osc_{index}_destination")));
                let unison = value_of(preset, &format!("osc_{index}_unison_voices")) > 1.0;
                parts.push(format!("osc_{index}_unison={}", if unison { "many" } else { "one" }));
                if rest.starts_with("distortion_") {
                    parts.push(setting(&format!("osc_{index}_distortion_type")));
                }
                if rest.starts_with("spectral_morph_") {
                    parts.push(setting(&format!("osc_{index}_spectral_morph_type")));
                }
            }
            "filter" => {
                parts.push(setting(&format!("filter_{index}_on")));
                parts.push(setting(&format!("filter_{index}_model")));
                parts.push(setting(&format!("filter_{index}_style")));
            }
            "env" | "lfo" | "random" | "macro_control" => {
                let source = format!("{family}_{index}");
                let mut dests: Vec<String> = preset
                    .settings
                    .modulations
                    .iter()
                    .filter(|m| m.is_connected() && m.source == source)
                    .map(|m| m.destination.clone())
                    .collect();
                dests.sort();
                dests.dedup();
                parts.push(format!("{source}->{}", if dests.is_empty() { "nothing".to_string() } else { dests.join("|") }));
            }
            _ => {}
        }
    }
    parts
}

/// The qualities an observation records (every `Quality` but the bands
/// and the aliasing, which need their own renders).
const QUALITIES: [(&str, Quality); 9] = [
    ("level_db", Quality::Level),
    ("brightness_st", Quality::Brightness),
    ("harshness_db", Quality::Harshness),
    ("warmth_db", Quality::Warmth),
    ("width", Quality::Width),
    ("attack_s", Quality::Attack),
    ("sustain_s", Quality::Sustain),
    ("noise", Quality::Noise),
    ("movement_db", Quality::Movement),
];

fn effect_between(before: &[f32], after: &[f32], d0: &Descriptors, d1: &Descriptors) -> Effect {
    let mut deltas = BTreeMap::new();
    for (name, quality) in QUALITIES {
        deltas.insert(name.to_string(), quality.measure(d1) - quality.measure(d0));
    }
    let mut bands = [0.0f32; 8];
    for (i, b) in bands.iter_mut().enumerate() {
        *b = d1.bands_dbfs[i] - d0.bands_dbfs[i];
    }
    Effect { distance_db: distance(before, after, SAMPLE_RATE, Default::default()).total_db, deltas, bands_db: bands }
}

/// Measures every active parameter of `preset` (Lite mode) — or only
/// those `only` names — one base render, one per parameter, in
/// parallel. Returns (name, observation).
pub fn measure_patch(preset: &Preset, origin: &str, scenario: &Scenario, scenario_id: &str, seed: u64, budget: Budget, only: Option<&[&str]>) -> Result<Vec<(String, Observation)>, OpError> {
    let active: Vec<&ParamDetails> = active_parameters_for(preset, RenderMode::Lite)
        .into_iter()
        .filter(|d| only.is_none_or(|names| names.contains(&d.name.as_str())))
        .collect();
    if active.is_empty() {
        return Ok(Vec::new());
    }
    let hash = preset_hash(preset);
    let stamp = engine_stamp();
    let date = today();
    let cache = EffectCache::open();
    // The step of each parameter, and the cached effect where the cache
    // has it: those need no render at all.
    let steps: Vec<Option<(f32, f32, String)>> = active
        .iter()
        .map(|details| {
            let from = value_of(preset, &details.name);
            let to = step(details, from, true).or_else(|| step(details, from, false))?;
            Some((from, to, EffectCache::key(&hash, &details.name, to, scenario_id)))
        })
        .collect();
    let mut effects: Vec<Option<(Effect, u32)>> = steps.iter().map(|s| s.as_ref().and_then(|(_, _, key)| cache.get(key)).map(|e| (e, 0))).collect();
    let misses: Vec<usize> = (0..active.len()).filter(|&i| steps[i].is_some() && effects[i].is_none()).collect();
    if !misses.is_empty() {
        let mut session = crate::ops::session();
        let base = render(&mut session, preset, scenario, render_seed(seed, 0))?;
        let base_descriptors = describe_without_pitch(&base);
        let base_samples = base.samples;
        let (results, _) = parallel(misses.len(), budget, |j, session| {
            let i = misses[j];
            let details = active[i];
            let (_, to, key) = steps[i].as_ref()?;
            let mut p = preset.clone();
            p.settings.values.insert(details.name.clone(), Json::from(*to as f64));
            let r = render(session, &p, scenario, render_seed(seed, 0)).ok()?;
            let effect = effect_between(&base_samples, &r.samples, &base_descriptors, &describe_without_pitch(&r));
            cache.put(key, &effect);
            Some(effect)
        });
        for (j, r) in results.into_iter().enumerate() {
            if let Some(effect) = r.flatten() {
                effects[misses[j]] = Some((effect, 1));
            }
        }
    }
    let mut out = Vec::new();
    for (i, details) in active.iter().enumerate() {
        let (Some((from, to, _)), Some((effect, renders))) = (&steps[i], effects[i].take()) else { continue };
        out.push((
            details.name.clone(),
            Observation {
                context: ContextRef { key: context_key(preset, details), origin: origin.to_string(), preset_hash: hash.clone() },
                scenario: scenario_id.to_string(),
                step: Step { from: *from, to: *to, fraction_of_range: (to - from) / (details.max - details.min).max(1e-9) },
                effect,
                renders,
                engine: stamp.clone(),
                date: date.clone(),
            },
        ));
    }
    Ok(out)
}

/// The canonical patch of a parameter: the sweep's base patch plus the
/// context the parameter needs (`context_for`).
pub fn canonical_patch(name: &str) -> Preset {
    let context = context_for(name);
    let mut settings = base_settings();
    for (key, value) in &context.settings {
        settings.insert(key.clone(), Json::from(*value));
    }
    build(settings, context.connection, context.slot)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MeasureReport {
    pub parameters_written: usize,
    pub observations_written: usize,
    /// Renders actually made; an observation served by the effect cache
    /// costs none (`cache_hits`).
    pub renders: usize,
    pub cache_hits: usize,
    pub skipped_fresh: usize,
    pub origins: Vec<String>,
    /// (parameter or patch, why) for what could not be measured: a
    /// canonical patch that renders silent (the granular engine with no
    /// sample), a patch that fails to load. Counted, never hidden.
    pub failed: Vec<(String, String)>,
}

/// What to measure: the canonical contexts, and/or a list of patches.
pub struct MeasureOptions {
    pub canonical: bool,
    pub patches: Vec<(String, PathBuf)>,
    /// Restrict to parameters whose name contains this.
    pub only: Option<String>,
    /// Skip a (parameter, origin, scenario) that is already fresh.
    pub stale_only: bool,
    pub seed: u64,
    pub budget: Budget,
}

/// The self-test every `measure` runs before writing: a parameter with a
/// known strong effect must measure one, and a parameter gated off must
/// measure exactly nothing.
fn self_test(scenario: &Scenario, seed: u64) -> Result<(), String> {
    let cutoff = parameters().lookup("filter_1_cutoff").ok_or("no filter_1_cutoff")?;
    let obs = measure_patch(&canonical_patch("filter_1_cutoff"), "self-test", scenario, "lite", seed, Budget::default(), Some(&["filter_1_cutoff"])).map_err(|e| format!("{e:?}"))?;
    let strong = obs.iter().find(|(n, _)| n == &cutoff.name).map(|(_, o)| o.effect.distance_db).unwrap_or(0.0);
    if strong < 0.5 {
        return Err(format!("self-test: filter_1_cutoff moved a quarter range reads {strong:.3} dB, expected > 0.5"));
    }
    // filter_2 is off in the base patch: its cutoff is not active there
    // and must not appear; and a stepped inert parameter must read 0.
    let base = build(base_settings(), None, 1);
    let mut stepped = base.clone();
    stepped.settings.values.insert("filter_2_cutoff".into(), Json::from(100.0));
    let mut session = crate::ops::session();
    let a = render(&mut session, &base, scenario, render_seed(seed, 0)).map_err(|e| format!("{e:?}"))?;
    let b = render(&mut session, &stepped, scenario, render_seed(seed, 0)).map_err(|e| format!("{e:?}"))?;
    let inert = distance(&a.samples, &b.samples, SAMPLE_RATE, Default::default()).total_db;
    if inert != 0.0 {
        return Err(format!("self-test: filter_2_cutoff with filter 2 off reads {inert} dB, expected 0"));
    }
    Ok(())
}

/// Runs the measurements and writes `measured/params/*.json` under `dir`.
pub fn measure(dir: &Path, options: &MeasureOptions, mut report: impl FnMut(&str)) -> Result<MeasureReport, String> {
    let scenario = Scenario::lite();
    self_test(&scenario, options.seed)?;
    report("self-test passed");
    let mut store = Store::load(dir);
    let mut out = MeasureReport::default();
    let table = parameters();
    let wanted = |name: &str| options.only.as_deref().is_none_or(|f| name.contains(f));
    let fresh = |store: &Store, name: &str, origin: &str| {
        store.params.get(name).is_some_and(|f| f.observations.iter().any(|o| o.context.origin == origin && o.scenario == "lite" && o.is_fresh()))
    };

    let mut touched: std::collections::BTreeSet<String> = Default::default();
    if options.canonical {
        // One patch per parameter (its own context): measured one at a
        // time, only that parameter, so the cost is two renders each.
        let names: Vec<String> = table.iter().map(|d| d.name.clone()).filter(|n| wanted(n)).collect();
        let mut done = 0usize;
        for name in &names {
            if options.stale_only && fresh(&store, name, "canonical") {
                out.skipped_fresh += 1;
                continue;
            }
            let patch = canonical_patch(name);
            let observations = match measure_patch(&patch, "canonical", &scenario, "lite", options.seed, options.budget, Some(&[name.as_str()])) {
                Ok(o) => o,
                Err(e) => {
                    out.failed.push((name.clone(), format!("{e:?}")));
                    continue;
                }
            };
            out.cache_hits += observations.iter().filter(|(_, o)| o.renders == 0).count();
            out.renders += observations.iter().map(|(_, o)| o.renders as usize).sum::<usize>() + usize::from(observations.iter().any(|(_, o)| o.renders > 0));
            if let Some((_, o)) = observations.into_iter().find(|(n, _)| n == name) {
                store.upsert(name, o);
                touched.insert(name.clone());
                out.observations_written += 1;
            }
            done += 1;
            if done.is_multiple_of(100) {
                report(&format!("canonical: {done}/{}", names.len()));
            }
        }
        out.origins.push("canonical".into());
    }
    for (label, path) in &options.patches {
        let origin = format!("patch:{label}");
        let preset = match crate::ops::load_patch(&path.to_string_lossy()) {
            Ok(p) => p,
            Err(e) => {
                out.failed.push((label.clone(), e));
                continue;
            }
        };
        if options.stale_only && active_parameters_for(&preset, RenderMode::Lite).iter().all(|d| !wanted(&d.name) || fresh(&store, &d.name, &origin)) {
            out.skipped_fresh += 1;
            continue;
        }
        let observations = match measure_patch(&preset, &origin, &scenario, "lite", options.seed, options.budget, None) {
            Ok(o) => o,
            Err(e) => {
                out.failed.push((label.clone(), format!("{e:?}")));
                continue;
            }
        };
        out.cache_hits += observations.iter().filter(|(_, o)| o.renders == 0).count();
        out.renders += observations.iter().map(|(_, o)| o.renders as usize).sum::<usize>() + usize::from(observations.iter().any(|(_, o)| o.renders > 0));
        for (name, o) in observations {
            if !wanted(&name) {
                continue;
            }
            store.upsert(&name, o);
            touched.insert(name);
            out.observations_written += 1;
        }
        out.origins.push(origin.clone());
        report(&format!("measured {origin}"));
    }
    for name in &touched {
        store.save_param(dir, name)?;
    }
    out.parameters_written = touched.len();
    Ok(out)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Status {
    pub engine: Option<EngineStamp>,
    pub parameters: usize,
    pub observations: usize,
    pub fresh: usize,
    pub stale: usize,
    /// Stale over all, 0..1: the number to watch.
    pub stale_fraction: f32,
    pub origins: BTreeMap<String, usize>,
    pub malformed: Vec<String>,
    /// Per corpus: patches, and whether its engine is the running one.
    pub corpora: BTreeMap<String, (usize, bool)>,
    /// Declared terms by validation status, plus `stale`: verdicts
    /// written by another engine or descriptor set.
    pub terms: BTreeMap<String, usize>,
}

/// Counts the store: parameters, observations, how many are stale
/// against the running engine, and per-origin counts.
pub fn status(dir: &Path) -> Status {
    let mut s = Status { engine: Some(engine_stamp()), ..Default::default() };
    let folder = dir.join("measured").join("params");
    let Ok(entries) = std::fs::read_dir(&folder) else { return s };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        match serde_json::from_str::<ParameterFile>(&text) {
            Ok(file) if file.schema == SCHEMA_VERSION && file.kind == "measured.parameter" => {
                s.parameters += 1;
                for o in &file.observations {
                    s.observations += 1;
                    if o.is_fresh() { s.fresh += 1 } else { s.stale += 1 }
                    *s.origins.entry(o.context.origin.clone()).or_insert(0) += 1;
                }
            }
            _ => s.malformed.push(path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()),
        }
    }
    s.stale_fraction = if s.observations > 0 { s.stale as f32 / s.observations as f32 } else { 0.0 };
    if let Ok(entries) = std::fs::read_dir(dir.join("corpus")) {
        for entry in entries.flatten() {
            let id = entry.file_name().to_string_lossy().to_string();
            let text = std::fs::read_to_string(entry.path().join("structure.json")).unwrap_or_default();
            if let Ok(structure) = serde_json::from_str::<corpus::Structure>(&text) {
                s.corpora.insert(id, (structure.corpus.patches, structure.engine.fingerprint == ENGINE_FINGERPRINT));
            }
        }
    }
    for term in terms::load_all(dir) {
        let status = if term.validation.status.is_empty() { "unvalidated".to_string() } else { term.validation.status.clone() };
        *s.terms.entry(status).or_insert(0) += 1;
        // A verdict from another engine or another descriptor set is
        // stale, whatever it says: `knowledge validate --all` renews it.
        let stale = term.validation.engine.as_ref().is_none_or(|e| e.fingerprint != ENGINE_FINGERPRINT || e.descriptors != DESCRIPTORS_FINGERPRINT);
        if stale && !term.validation.status.is_empty() {
            *s.terms.entry("stale".to_string()).or_insert(0) += 1;
        }
    }
    s
}

/// The agreement test of the design note: on each patch, the weights
/// `explore` measures live against the store's priors for the same
/// parameters (the patch's own observations left out), as a Spearman
/// rank correlation, and the renders each side spends.
#[derive(Clone, Debug, Serialize)]
pub struct Agreement {
    pub label: String,
    pub active: usize,
    pub known: usize,
    pub renders_live: usize,
    pub renders_prior: usize,
    pub spearman: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AgreementReport {
    pub patches: Vec<Agreement>,
    pub mean_spearman: Option<f32>,
    pub renders_live: usize,
    pub renders_prior: usize,
}

fn spearman(a: &[f32], b: &[f32]) -> Option<f32> {
    let n = a.len();
    if n < 3 || n != b.len() {
        return None;
    }
    let ranks = |v: &[f32]| -> Vec<f32> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
        let mut r = vec![0.0f32; v.len()];
        let mut i = 0;
        while i < idx.len() {
            let mut j = i;
            while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                j += 1;
            }
            let mean = (i + j) as f32 / 2.0 + 1.0;
            for &k in &idx[i..=j] {
                r[k] = mean;
            }
            i = j + 1;
        }
        r
    };
    let (ra, rb) = (ranks(a), ranks(b));
    let mean = |r: &[f32]| r.iter().sum::<f32>() / r.len() as f32;
    let (ma, mb) = (mean(&ra), mean(&rb));
    let cov: f32 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f32 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f32 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    if va == 0.0 || vb == 0.0 { None } else { Some(cov / (va * vb).sqrt()) }
}

pub fn agreement(store: &Store, patches: &[(String, PathBuf)], seed: u64, budget: Budget) -> Result<AgreementReport, String> {
    let scenario = Scenario::lite();
    let mut out = AgreementReport { patches: Vec::new(), mean_spearman: None, renders_live: 0, renders_prior: 0 };
    for (label, path) in patches {
        let origin = format!("patch:{label}");
        let preset = crate::ops::load_patch(&path.to_string_lossy()).map_err(|e| format!("{label}: {e}"))?;
        let live = measure_patch(&preset, &origin, &scenario, "lite", seed, budget, None).map_err(|e| format!("{label}: {e:?}"))?;
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut known = 0;
        for (name, o) in &live {
            if let Some((prior, _)) = store.prior(name, &o.context.key, Some(&origin)) {
                a.push(o.effect.distance_db);
                b.push(prior);
                known += 1;
            }
        }
        let active = live.len();
        // What each side WOULD render without the effect cache: the
        // comparison is of the two strategies, not of a warm cache.
        let entry = Agreement {
            label: label.clone(),
            active,
            known,
            renders_live: active + 1,
            renders_prior: 1 + (active - known),
            spearman: spearman(&a, &b),
        };
        out.renders_live += entry.renders_live;
        out.renders_prior += entry.renders_prior;
        out.patches.push(entry);
    }
    let rhos: Vec<f32> = out.patches.iter().filter_map(|p| p.spearman).collect();
    out.mean_spearman = if rhos.is_empty() { None } else { Some(rhos.iter().sum::<f32>() / rhos.len() as f32) };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fingerprints_are_sha256_and_the_stamp_reads_them() {
        for f in [ENGINE_FINGERPRINT, DESCRIPTORS_FINGERPRINT] {
            assert!(f.starts_with("sha256:") && f.len() == 7 + 64, "{f}");
            assert!(f[7..].chars().all(|c| c.is_ascii_hexdigit()));
        }
        assert_ne!(ENGINE_FINGERPRINT, DESCRIPTORS_FINGERPRINT);
        let stamp = engine_stamp();
        assert_eq!(stamp.fingerprint, ENGINE_FINGERPRINT);
        assert_eq!(today().len(), 10);
    }

    #[test]
    fn the_context_key_names_what_gates_a_parameter_and_nothing_else() {
        let table = parameters();
        let mut preset = canonical_patch("filter_1_cutoff");
        let cutoff = table.lookup("filter_1_cutoff").unwrap();
        let key = context_key(&preset, cutoff);
        assert_eq!(key, "filter_1_on=1;filter_1_model=0;filter_1_style=0", "{key}");
        // A change in the filter's model changes the key; one in its
        // resonance does not.
        preset.settings.values.insert("filter_1_resonance".into(), Json::from(0.9));
        assert_eq!(context_key(&preset, cutoff), key);
        preset.settings.values.insert("filter_1_model".into(), Json::from(6.0));
        assert_ne!(context_key(&preset, cutoff), key);
        // A source's key is where it goes.
        let lfo = table.lookup("lfo_1_frequency").unwrap();
        assert_eq!(context_key(&preset, lfo), "lfo_1->nothing;lfo_1_sync=1");
        let wired = canonical_patch("lfo_1_frequency");
        assert_eq!(context_key(&wired, lfo), "lfo_1->filter_1_cutoff;lfo_1_sync=1");
        // A rate control carries its sync switch; its neighbour does not.
        let delay = table.lookup("delay_frequency").unwrap();
        assert_eq!(context_key(&preset, delay), "delay_on=0;delay_sync=1");
        let feedback = table.lookup("delay_feedback").unwrap();
        assert_eq!(context_key(&preset, feedback), "delay_on=0");
        // A global has the global key.
        let volume = table.lookup("volume").unwrap();
        assert_eq!(context_key(&preset, volume), "global");
    }

    #[test]
    fn the_prior_is_the_median_shrunk_by_its_evidence() {
        let mut store = Store::default();
        let mk = |d: f32, origin: &str| Observation {
            context: ContextRef { key: "k".into(), origin: origin.into(), preset_hash: String::new() },
            scenario: "lite".into(),
            step: Step { from: 0.0, to: 1.0, fraction_of_range: 0.25 },
            effect: Effect { distance_db: d, deltas: BTreeMap::new(), bands_db: [0.0; 8] },
            renders: 2,
            engine: engine_stamp(),
            date: today(),
        };
        for (i, d) in [1.0, 5.0, 3.0].iter().enumerate() {
            store.upsert("p", mk(*d, &format!("o{i}")));
        }
        let (w, n) = store.prior("p", "k", None).unwrap();
        assert_eq!(n, 3);
        assert!((w - 3.0 * 3.0 / 8.0).abs() < 1e-6, "{w}");
        // Leave-one-out drops that origin; a stale entry is never counted.
        assert_eq!(store.prior("p", "k", Some("o1")).unwrap().1, 2);
        let mut stale = mk(9.0, "o9");
        stale.engine.fingerprint = "sha256:old".into();
        store.upsert("p", stale);
        assert_eq!(store.prior("p", "k", None).unwrap().1, 3);
        assert!(store.prior("p", "other", None).is_none());
    }

    #[test]
    fn a_store_round_trips_through_its_files() {
        let dir = std::env::temp_dir().join(format!("spinwave-knowledge-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::default();
        store.upsert(
            "filter_1_cutoff",
            Observation {
                context: ContextRef { key: "k".into(), origin: "canonical".into(), preset_hash: "h".into() },
                scenario: "lite".into(),
                step: Step { from: 72.0, to: 104.0, fraction_of_range: 0.25 },
                effect: Effect { distance_db: 4.0, deltas: BTreeMap::new(), bands_db: [0.0; 8] },
                renders: 2,
                engine: engine_stamp(),
                date: today(),
            },
        );
        store.save_param(&dir, "filter_1_cutoff").unwrap();
        let back = Store::load(&dir);
        assert_eq!(back.lookup("filter_1_cutoff", "k", None).len(), 1);
        let s = status(&dir);
        assert_eq!((s.parameters, s.observations, s.fresh, s.stale), (1, 1, 1, 0));
        assert!(s.malformed.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn spearman_agrees_with_itself_and_disagrees_with_its_reverse() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((spearman(&a, &a).unwrap() - 1.0).abs() < 1e-6);
        let r = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((spearman(&a, &r).unwrap() + 1.0).abs() < 1e-6);
        assert!(spearman(&a[..2], &r[..2]).is_none());
    }

    #[test]
    fn measuring_the_canonical_cutoff_passes_the_self_test() {
        self_test(&Scenario::lite(), 1).unwrap();
    }
}
