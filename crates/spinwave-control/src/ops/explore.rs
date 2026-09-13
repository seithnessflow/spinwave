//! Operation 5: variations around a patch, and the line between two.
//!
//! The fuzzer's lesson: a uniform mutation over the whole table is noise.
//! Here only the patch's active parameters move, each by an amplitude
//! weighted by its sensitivity *measured on this patch* (one Lite render
//! per parameter, the band distance of a quarter-range step), so a
//! parameter that does nothing is not touched and one that does a lot is
//! touched gently. Continuous values draw from a triangular distribution
//! around the current value; indexed parameters and switches (a filter
//! model, a distortion type, an effect's on/off) hold unless
//! [`ExploreSpec::switch_indexed`] says how often they may switch — zero
//! by default, because a topology change is a jump, not a variation.
//!
//! Interpolation: continuous values lerp; booleans and indexed values are
//! frozen at `a`'s for `t < 0.5` and take `b`'s from `t ≥ 0.5`; the
//! connections are the union, amounts lerped with an absent connection
//! counting as zero, and a connection whose amount interpolates below
//! 1e-3 in magnitude is dropped.

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use spinwave_params::preset::ModulationConnection;
use spinwave_params::{parameters, ParamDetails, ParamScale, Preset};

use super::diff::{connections_of, param_diff, ParamChange};
use super::distance::{distance, Options};
use super::explain::{active_parameters, step};
use super::{describe_without_pitch, parallel, render, render_seed, Budget, Descriptors, OpError, Scenario};
use crate::fuzz::Rng;
use crate::session::SAMPLE_RATE;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExploreSpec {
    pub count: usize,
    /// 0..1: the fraction of a parameter's range the most sensitive
    /// parameter may move (others move less, by their measured weight).
    pub amplitude: f32,
    pub seed: u64,
    /// Probability that an indexed parameter or a switch changes to
    /// another option; 0 keeps the topology.
    #[serde(default)]
    pub switch_indexed: f32,
    #[serde(default)]
    pub budget: Budget,
    /// Where the weights come from (`knowledge/measured/`, live renders,
    /// or the store first and live renders for what it lacks).
    #[serde(default)]
    pub prior: Prior,
    /// Let a continuous parameter leave the range real patches use it in
    /// (`knowledge/corpus/*/value_ranges_used`, p10..p90). Off by
    /// default: a variation stays where patches live, without copying
    /// any patch's values.
    #[serde(default)]
    pub free_ranges: bool,
}

/// The source of a parameter's weight.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Prior {
    /// One Lite render per active parameter, on this patch (today's
    /// behaviour, and what a store-less checkout does).
    Live,
    /// The store's fresh observations in this parameter's context; a
    /// parameter the store does not know gets weight 0 and is reported.
    Measured,
    /// The store where it knows, a live render where it does not.
    #[default]
    MeasuredThenLive,
}

#[derive(Clone, Debug, Serialize)]
pub struct Variant {
    pub index: usize,
    pub preset: Preset,
    pub diff: Vec<ParamChange>,
    pub distance_from_origin_db: f32,
    pub descriptors: Descriptors,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exploration {
    pub seed: u64,
    /// Per active parameter, the measured weight (0..1) the mutation used.
    pub weights: Vec<(String, f32)>,
    /// Per active parameter, where its weight came from: `store:n=12`,
    /// `live`, or `unknown` (weight 0 under `Prior::Measured`).
    pub weight_sources: Vec<(String, String)>,
    /// Active parameters whose draws were held to a corpus range.
    pub corpus_ranges_used: usize,
    pub variants: Vec<Variant>,
    /// Variants dropped because every attempt failed to load or sounded.
    pub dropped: usize,
    pub renders: usize,
    pub truncated: bool,
}

fn value_of(preset: &Preset, details: &ParamDetails) -> f32 {
    preset.settings.values.get(&details.name).and_then(Json::as_f64).map(|v| v as f32).unwrap_or(details.default_value)
}

/// A parameter and its measured weight, 0..1.
type Weight = (&'static ParamDetails, f32);

/// The weight of every active parameter, normalised to the largest, with
/// its source. Under `Prior::Live`: one Lite render per parameter moved
/// a quarter of its range, the band distance as the weight. Under a
/// measured prior: the store's fresh observations in the parameter's
/// context on THIS patch (`knowledge::context_key`), the median shrunk
/// by their count; what the store lacks is rendered live
/// (`MeasuredThenLive`) or weighs nothing (`Measured`). The two scales
/// agree: both are the band distance of a quarter-range step. Returns
/// the weights, the sources, the render count, and whether the budget
/// cut it short.
pub(crate) fn sensitivity_weights(
    preset: &Preset,
    scenario: &Scenario,
    seed: u64,
    budget: Budget,
    prior: Prior,
) -> Result<(Vec<Weight>, Vec<String>, usize, bool), OpError> {
    let active = active_parameters(preset);
    let store = match prior {
        Prior::Live => None,
        _ => Some(crate::knowledge::Store::load(&crate::knowledge::knowledge_dir())),
    };
    let mut raw: Vec<Option<f32>> = vec![None; active.len()];
    let mut sources: Vec<String> = vec![String::new(); active.len()];
    if let Some(store) = &store {
        for (i, details) in active.iter().enumerate() {
            let key = crate::knowledge::context_key(preset, details);
            if let Some((weight, n)) = store.prior(&details.name, &key, None) {
                raw[i] = Some(weight);
                sources[i] = format!("store:n={n}");
            }
        }
    }
    let to_render: Vec<usize> = (0..active.len()).filter(|&i| raw[i].is_none() && prior != Prior::Measured).collect();
    let mut renders = 0;
    let mut truncated = false;
    if !to_render.is_empty() || prior == Prior::Live {
        let mut session = super::session();
        let base = render(&mut session, preset, scenario, render_seed(seed, 0))?;
        let base_samples = base.samples;
        let (results, ran) = parallel(to_render.len(), budget, |j, session| {
            let details = active[to_render[j]];
            let from = value_of(preset, details);
            let to = step(details, from, true).or_else(|| step(details, from, false))?;
            let mut p = preset.clone();
            p.settings.values.insert(details.name.clone(), Json::from(to as f64));
            render(session, &p, scenario, render_seed(seed, 0))
                .ok()
                .map(|r| distance(&base_samples, &r.samples, SAMPLE_RATE, Options::default()).total_db)
        });
        for (j, r) in results.into_iter().enumerate() {
            raw[to_render[j]] = Some(r.flatten().unwrap_or(0.0));
            sources[to_render[j]] = "live".into();
        }
        renders = ran + 1;
        truncated = ran < to_render.len();
    }
    for (i, s) in sources.iter_mut().enumerate() {
        if s.is_empty() {
            *s = if raw[i].is_none() { "unknown".into() } else { "live".into() };
        }
    }
    let raw: Vec<f32> = raw.into_iter().map(|r| r.unwrap_or(0.0)).collect();
    let top = raw.iter().cloned().fold(0.0f32, f32::max);
    let weights = active.into_iter().zip(raw).map(|(d, w)| (d, if top > 0.0 { w / top } else { 0.0 })).collect();
    Ok((weights, sources, renders, truncated))
}

/// The corpus range a continuous parameter is held to, if any.
type Ranges = std::collections::BTreeMap<String, crate::knowledge::corpus::Range>;

/// One mutated copy of `preset`.
fn mutate(preset: &Preset, weights: &[(&ParamDetails, f32)], ranges: &Ranges, spec: &ExploreSpec, rng: &mut Rng) -> Preset {
    let mut p = preset.clone();
    for (details, weight) in weights {
        if *weight <= 0.0 {
            continue;
        }
        let from = value_of(preset, details);
        let to = if details.scale == ParamScale::Indexed {
            if spec.switch_indexed > 0.0 && rng.chance(spec.switch_indexed) {
                let options = (details.max - details.min) as usize + 1;
                if options < 2 {
                    continue;
                }
                let mut pick = details.min + rng.below(options) as f32;
                if pick == from.round() {
                    pick = details.min + ((pick - details.min + 1.0) % options as f32);
                }
                pick
            } else {
                continue;
            }
        } else {
            // Triangular around the current value: the sum of two uniforms.
            let reach = (details.max - details.min) * spec.amplitude * weight;
            let offset = (rng.unit() + rng.unit() - 1.0) * reach;
            let (low, high) = match ranges.get(&details.name) {
                // A value already outside the corpus range is not pulled
                // in: the range only bounds the move.
                Some(r) if !spec.free_ranges => (r.p10.min(from), r.p90.max(from)),
                _ => (details.min, details.max),
            };
            (from + offset).clamp(low.max(details.min), high.min(details.max))
        };
        p.settings.values.insert(details.name.clone(), Json::from(to as f64));
    }
    p
}

pub fn explore(preset: &Preset, scenario: &Scenario, spec: &ExploreSpec) -> Result<Exploration, OpError> {
    if spec.count == 0 || !(0.0..=1.0).contains(&spec.amplitude) {
        return Err(OpError::BadScenario { message: "count > 0 and amplitude in 0..=1".into() });
    }
    let (weights, sources, weight_renders, weights_truncated) = sensitivity_weights(preset, scenario, spec.seed, spec.budget, spec.prior)?;
    if weights.iter().all(|(_, w)| *w <= 0.0) {
        return Err(OpError::Nothing { message: "no active parameter changes the sound".into() });
    }
    let ranges: Ranges = if spec.free_ranges { Ranges::new() } else { crate::knowledge::corpus::ranges(&crate::knowledge::knowledge_dir()) };
    let corpus_ranges_used = weights.iter().filter(|(d, w)| *w > 0.0 && d.scale != ParamScale::Indexed && ranges.contains_key(&d.name)).count();
    let mut session = super::session();
    let origin = render(&mut session, preset, scenario, render_seed(spec.seed, 0))?;
    let origin_samples = origin.samples;
    let remaining = Budget {
        max_renders: spec.budget.max_renders.saturating_sub(weight_renders + 1),
        max_seconds: spec.budget.max_seconds,
    };
    let (results, ran) = parallel(spec.count, remaining, |i, session| {
        // A stream per variant, from the spec's seed and the index: the
        // variant is the same whichever thread draws it.
        let mut rng = Rng::new(spec.seed ^ ((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        for _attempt in 0..8 {
            let candidate = mutate(preset, &weights, &ranges, spec, &mut rng);
            if let Ok(r) = render(session, &candidate, scenario, render_seed(spec.seed, 0)) {
                let (diff, _) = param_diff(preset, &candidate);
                return Some(Variant {
                    index: i,
                    distance_from_origin_db: distance(&origin_samples, &r.samples, SAMPLE_RATE, Options::default()).total_db,
                    // Without the pitch detector: a variant's f0 is read
                    // by `measure` when someone picks it.
                    descriptors: describe_without_pitch(&r),
                    preset: candidate,
                    diff,
                });
            }
        }
        None
    });
    let variants: Vec<Variant> = results.into_iter().flatten().flatten().collect();
    Ok(Exploration {
        seed: spec.seed,
        weights: weights.iter().map(|(d, w)| (d.name.clone(), *w)).collect(),
        weight_sources: weights.iter().zip(sources).map(|((d, _), s)| (d.name.clone(), s)).collect(),
        corpus_ranges_used,
        dropped: ran - variants.len(),
        variants,
        renders: weight_renders + 1 + ran,
        truncated: weights_truncated || ran < spec.count,
    })
}

/// The patches at `t` in `ts` between `a` (t = 0) and `b` (t = 1).
pub fn interpolate(a: &Preset, b: &Preset, ts: &[f32]) -> Result<Vec<Preset>, OpError> {
    let table = parameters();
    let mut out = Vec::new();
    for &t in ts {
        if !(0.0..=1.0).contains(&t) {
            return Err(OpError::BadScenario { message: format!("t = {t} outside 0..=1") });
        }
        let mut p = a.clone();
        p.settings.modulations.clear();
        p.settings.values.retain(|k, _| !k.starts_with("modulation_"));
        for details in table.iter() {
            if details.name.starts_with("modulation_") {
                continue;
            }
            let (x, y) = (value_of(a, details), value_of(b, details));
            if x == y {
                continue;
            }
            let v = if details.scale == ParamScale::Indexed {
                if t < 0.5 { x } else { y }
            } else {
                x + (y - x) * t
            };
            p.settings.values.insert(details.name.clone(), Json::from(v as f64));
        }
        // Connections: the union, amounts lerped, absent = 0.
        let (ca, cb) = (connections_of(a), connections_of(b));
        let mut routes: Vec<(String, String)> = ca.iter().chain(cb.iter()).map(|(r, _)| r.clone()).collect();
        routes.dedup();
        let mut seen = Vec::new();
        for route in routes {
            if seen.contains(&route) {
                continue;
            }
            seen.push(route.clone());
            let amount_a = ca.iter().find(|(r, _)| *r == route).map(|(_, v)| *v).unwrap_or(0.0);
            let amount_b = cb.iter().find(|(r, _)| *r == route).map(|(_, v)| *v).unwrap_or(0.0);
            let amount = amount_a + (amount_b - amount_a) * t;
            if amount.abs() < 1e-3 {
                continue;
            }
            // The other slot fields (bipolar, power…) come from whichever
            // side owns the route at this t, like an indexed value.
            let owner = if t < 0.5 && amount_a != 0.0 || amount_b == 0.0 { a } else { b };
            let slot_in_owner = owner.settings.modulations.iter().position(|m| (m.source.clone(), m.destination.clone()) == route);
            p.settings.modulations.push(ModulationConnection { source: route.0.clone(), destination: route.1.clone(), ..Default::default() });
            let n = p.settings.modulations.len();
            if let Some(s) = slot_in_owner {
                let prefix = format!("modulation_{}_", s + 1);
                for (k, v) in &owner.settings.values {
                    if let Some(field) = k.strip_prefix(&prefix) {
                        if field != "amount" {
                            p.settings.values.insert(format!("modulation_{n}_{field}"), v.clone());
                        }
                    }
                }
            }
            p.settings.values.insert(format!("modulation_{n}_amount"), Json::from(amount as f64));
        }
        out.push(p);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::saw_patch;

    #[test]
    fn variants_stay_close_keep_the_topology_and_are_reproducible() {
        let p = saw_patch();
        let spec = ExploreSpec { count: 6, amplitude: 0.3, seed: 11, switch_indexed: 0.0, budget: Budget::default(), prior: Prior::Live, free_ranges: true };
        let e = explore(&p, &Scenario::lite(), &spec).expect("explores");
        assert!(!e.truncated, "{} renders", e.renders);
        assert_eq!(e.variants.len(), 6, "dropped {}", e.dropped);
        for v in &e.variants {
            assert!(!v.diff.is_empty());
            assert!(v.diff.iter().all(|c| !c.indexed), "no indexed switch by default: {:?}", v.diff.iter().filter(|c| c.indexed).map(|c| &c.name).collect::<Vec<_>>());
            assert!(v.distance_from_origin_db > 0.0);
        }
        let again = explore(&p, &Scenario::lite(), &spec).unwrap();
        for (x, y) in e.variants.iter().zip(&again.variants) {
            assert_eq!(x.diff, y.diff, "same seed, same variant");
            assert_eq!(x.distance_from_origin_db, y.distance_from_origin_db);
        }
    }

    /// The store's prior replaces the live render of a parameter it
    /// knows in THIS patch's context, and only that one; the weight it
    /// gives is the stored distance on the same scale as the live ones,
    /// so the variants are the same kind of variation.
    #[test]
    fn a_measured_prior_spares_the_live_render_of_what_the_store_knows() {
        use crate::knowledge::{context_key, engine_stamp, today, ContextRef, Effect, Observation, Step, Store};
        let p = saw_patch();
        let cutoff = parameters().lookup("filter_1_cutoff").unwrap();
        // The live weight of the cutoff on this patch, to store as if
        // another patch of the same context had measured it.
        let (live, _, live_renders, _) = sensitivity_weights(&p, &Scenario::lite(), 11, Budget::default(), Prior::Live).unwrap();
        let active = live.len();
        assert_eq!(live_renders, active + 1);
        let dir = std::env::temp_dir().join(format!("spinwave-explore-prior-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = Store::default();
        store.upsert_for_test(
            "filter_1_cutoff",
            Observation {
                context: ContextRef { key: context_key(&p, cutoff), origin: "patch:elsewhere".into(), preset_hash: "h".into() },
                scenario: "lite".into(),
                step: Step { from: 72.0, to: 104.0, fraction_of_range: 0.25 },
                effect: Effect { distance_db: 7.0, deltas: Default::default(), bands_db: [0.0; 8] },
                renders: 2,
                engine: engine_stamp(),
                date: today(),
            },
        );
        store.save_for_test(&dir, "filter_1_cutoff");
        // SPINWAVE_KNOWLEDGE is process-wide: this is the only test that
        // sets it, and Live never reads it.
        std::env::set_var("SPINWAVE_KNOWLEDGE", &dir);
        let (weights, sources, renders, _) = sensitivity_weights(&p, &Scenario::lite(), 11, Budget::default(), Prior::MeasuredThenLive).unwrap();
        std::env::remove_var("SPINWAVE_KNOWLEDGE");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(renders, active, "one live render spared: {sources:?}");
        let source_of = |name: &str| sources[weights.iter().position(|(d, _)| d.name == name).unwrap()].clone();
        assert_eq!(source_of("filter_1_cutoff"), "store:n=1");
        assert!(sources.iter().filter(|s| *s == "live").count() == active - 1, "{sources:?}");
        // 7 dB shrunk by 1/6, normalised against the live top.
        let stored = weights.iter().find(|(d, _)| d.name == "filter_1_cutoff").unwrap().1;
        assert!(stored > 0.0 && stored <= 1.0, "{stored}");
    }

    #[test]
    fn interpolation_lerps_continuous_and_switches_indexed_at_half() {
        let a = saw_patch();
        let mut b = a.clone();
        b.settings.values.insert("filter_1_cutoff".into(), 40.0.into());
        b.settings.values.insert("filter_1_model".into(), 2.0.into());
        b.settings.modulations.push(ModulationConnection { source: "lfo_1".into(), destination: "filter_1_cutoff".into(), ..Default::default() });
        b.settings.values.insert("modulation_1_amount".into(), 0.8.into());
        let ps = interpolate(&a, &b, &[0.0, 0.25, 0.5, 1.0]).unwrap();
        let cutoff = |p: &Preset| p.settings.values["filter_1_cutoff"].as_f64().unwrap() as f32;
        assert_eq!(cutoff(&ps[0]), 80.0);
        assert_eq!(cutoff(&ps[1]), 70.0);
        assert_eq!(cutoff(&ps[3]), 40.0);
        let model = |p: &Preset| p.settings.values.get("filter_1_model").and_then(|v| v.as_f64()).unwrap_or(0.0);
        assert_eq!(model(&ps[1]), 0.0, "frozen before t = 0.5");
        assert_eq!(model(&ps[2]), 2.0, "switched at t = 0.5");
        let amount = |p: &Preset| p.settings.values.get("modulation_1_amount").and_then(|v| v.as_f64()).map(|v| v as f32);
        assert_eq!(amount(&ps[0]), None, "absent at t = 0");
        assert!((amount(&ps[1]).unwrap() - 0.2).abs() < 1e-6);
        assert!((amount(&ps[3]).unwrap() - 0.8).abs() < 1e-6);
        assert!(interpolate(&a, &b, &[1.5]).is_err());
    }
}
