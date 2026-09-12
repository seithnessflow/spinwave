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
use super::{describe, parallel, render, render_seed, Budget, Descriptors, OpError, Scenario};
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

/// One Lite render per active parameter moved a quarter of its range:
/// the band distance is its weight, normalised to the largest. Returns
/// the weights, the render count, and whether the budget cut it short.
fn sensitivity_weights(preset: &Preset, scenario: &Scenario, seed: u64, budget: Budget) -> Result<(Vec<Weight>, usize, bool), OpError> {
    let active = active_parameters(preset);
    let mut session = super::session();
    let base = render(&mut session, preset, scenario, render_seed(seed, 0))?;
    let base_samples = base.samples;
    let (results, ran) = parallel(active.len(), budget, |i, session| {
        let details = active[i];
        let from = value_of(preset, details);
        let to = step(details, from, true).or_else(|| step(details, from, false))?;
        let mut p = preset.clone();
        p.settings.values.insert(details.name.clone(), Json::from(to as f64));
        render(session, &p, scenario, render_seed(seed, 0))
            .ok()
            .map(|r| distance(&base_samples, &r.samples, SAMPLE_RATE, Options::default()).total_db)
    });
    let raw: Vec<f32> = results.into_iter().map(|r| r.flatten().unwrap_or(0.0)).collect();
    let top = raw.iter().cloned().fold(0.0f32, f32::max);
    let count = active.len();
    let weights = active.into_iter().zip(raw).map(|(d, w)| (d, if top > 0.0 { w / top } else { 0.0 })).collect();
    Ok((weights, ran + 1, ran < count))
}

/// One mutated copy of `preset`.
fn mutate(preset: &Preset, weights: &[(&ParamDetails, f32)], spec: &ExploreSpec, rng: &mut Rng) -> Preset {
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
            (from + offset).clamp(details.min, details.max)
        };
        p.settings.values.insert(details.name.clone(), Json::from(to as f64));
    }
    p
}

pub fn explore(preset: &Preset, scenario: &Scenario, spec: &ExploreSpec) -> Result<Exploration, OpError> {
    if spec.count == 0 || !(0.0..=1.0).contains(&spec.amplitude) {
        return Err(OpError::BadScenario { message: "count > 0 and amplitude in 0..=1".into() });
    }
    let (weights, weight_renders, weights_truncated) = sensitivity_weights(preset, scenario, spec.seed, spec.budget)?;
    if weights.iter().all(|(_, w)| *w <= 0.0) {
        return Err(OpError::Nothing { message: "no active parameter changes the sound".into() });
    }
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
            let candidate = mutate(preset, &weights, spec, &mut rng);
            if let Ok(r) = render(session, &candidate, scenario, render_seed(spec.seed, 0)) {
                let (diff, _) = param_diff(preset, &candidate);
                return Some(Variant {
                    index: i,
                    distance_from_origin_db: distance(&origin_samples, &r.samples, SAMPLE_RATE, Options::default()).total_db,
                    descriptors: describe(&r),
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
        let spec = ExploreSpec { count: 6, amplitude: 0.3, seed: 11, switch_indexed: 0.0, budget: Budget::default() };
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
