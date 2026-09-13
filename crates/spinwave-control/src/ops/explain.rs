//! Operation 3: which parameters make THIS patch sound the way it does,
//! for a named quality — and the inverse, which moves would push the
//! quality the way the caller wants.
//!
//! Explain neutralises one active parameter at a time (its table default,
//! or the value that removes its effect: a switch off, an amount at zero)
//! and re-renders in the scenario's mode; the contribution is the change
//! of the quality's measure. Suggest moves each active parameter a step in
//! each direction instead. Both reuse the sensitivity sweep's idea of
//! context, inverted: a parameter is *active* when the module it belongs
//! to is on and, for a source, connected — see [`active_parameters`].

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use spinwave_params::{parameters, ParamDetails, ParamScale, Preset};

use super::diff::{connections_of, spell};
use super::distance::{distance, Options};
use super::{describe_without_pitch, parallel, render, render_seed, Budget, Descriptors, OpError, Scenario};
use crate::sensitivity::split_indexed;
use crate::session::{Session, SAMPLE_RATE};

/// A sound quality with a measure and a unit, each mapped to descriptors
/// so the mapping is in one place and readable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    /// RMS level, dBFS.
    Level,
    /// Spectral centroid in semitones (`12·log2(centroid_hz)`).
    Brightness,
    /// Presence region: mean of the 2.56–20 kHz bands minus the mean of
    /// the bands below, dB.
    Harshness,
    /// Low-mid weight: mean of the 80–320 Hz bands minus the mean of all
    /// bands, dB.
    Warmth,
    /// `stereo_width`, 0..1.
    Width,
    /// Attack time, seconds (`Less` = sharper).
    Attack,
    /// Time to fall 20 dB after the peak, seconds; the render length when
    /// it never does.
    Sustain,
    /// Spectral flatness, 0..1.
    Noise,
    /// Standard deviation of the 50 ms RMS trajectory, dB.
    Movement,
    /// One octave band's level, dBFS (index into `bands_dbfs`, 0..8).
    Band(u8),
    /// The honest aliasing ratio (`ops::aliasing`): two renders a
    /// semitone apart per measurement, the power of the prominent peaks
    /// above 2 kHz that do not follow the key over all of them.
    Aliasing,
}

impl Quality {
    pub fn unit(self) -> &'static str {
        match self {
            Quality::Level | Quality::Harshness | Quality::Warmth | Quality::Movement | Quality::Band(_) => "dB",
            Quality::Brightness => "st",
            Quality::Width | Quality::Noise | Quality::Aliasing => "ratio",
            Quality::Attack | Quality::Sustain => "s",
        }
    }

    /// The quality's measure on a set of descriptors.
    pub fn measure(self, d: &Descriptors) -> f32 {
        let mean = |b: &[f32]| b.iter().sum::<f32>() / b.len().max(1) as f32;
        match self {
            Quality::Level => d.rms_dbfs,
            Quality::Brightness => 12.0 * d.centroid_hz.max(1.0).log2(),
            Quality::Harshness => mean(&d.bands_dbfs[6..8]) - mean(&d.bands_dbfs[0..6]),
            Quality::Warmth => mean(&d.bands_dbfs[1..3]) - mean(&d.bands_dbfs),
            Quality::Width => d.stereo_width,
            Quality::Attack => d.attack_seconds,
            Quality::Sustain => d.decay_20db_seconds.unwrap_or(d.duration_seconds),
            Quality::Noise => d.spectral_flatness,
            Quality::Movement => {
                let t = &d.rms_trajectory_dbfs;
                if t.len() < 2 {
                    return 0.0;
                }
                let m = mean(t);
                (t.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / t.len() as f32).sqrt()
            }
            Quality::Band(b) => d.bands_dbfs[(b as usize).min(7)],
            // Not a function of one render's descriptors: see `evaluate`.
            Quality::Aliasing => f32::NAN,
        }
    }

    /// Renders per measurement: two for aliasing, one otherwise.
    pub fn renders(self) -> usize {
        if self == Quality::Aliasing { 2 } else { 1 }
    }

    pub fn from_id(id: &str) -> Option<Quality> {
        Some(match id {
            "level" => Quality::Level,
            "brightness" => Quality::Brightness,
            "harshness" => Quality::Harshness,
            "warmth" => Quality::Warmth,
            "width" => Quality::Width,
            "attack" => Quality::Attack,
            "sustain" => Quality::Sustain,
            "noise" => Quality::Noise,
            "movement" => Quality::Movement,
            "aliasing" => Quality::Aliasing,
            other => {
                let n: u8 = other.strip_prefix("band")?.parse().ok()?;
                if n < 8 { Quality::Band(n) } else { return None }
            }
        })
    }

    pub const ALL: [&'static str; 11] =
        ["level", "brightness", "harshness", "warmth", "width", "attack", "sustain", "noise", "movement", "aliasing", "bandN (0..7)"];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    More,
    Less,
}

/// One parameter's measured share of a quality in this patch.
#[derive(Clone, Debug, Serialize)]
pub struct Contribution {
    pub name: String,
    pub value: f32,
    pub value_text: String,
    /// What it was set to, to measure without it.
    pub neutral_value: f32,
    pub neutral_text: String,
    /// `measure(with) − measure(without)`, in the quality's unit: positive
    /// means the parameter, as set, adds to the quality.
    pub effect: f32,
    /// How much of everything else moved when it was neutralised, dB
    /// (the band distance between the two renders).
    pub distance_db: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Explanation {
    pub seed: u64,
    pub quality: Quality,
    pub unit: &'static str,
    /// The quality on the patch as it is.
    pub measured: f32,
    /// Ranked by |effect|, largest first.
    pub ranked: Vec<Contribution>,
    /// Active parameters already at their neutral value: nothing to take away.
    pub at_neutral: usize,
    /// Parameters skipped because their module is off or their source is
    /// unconnected in this patch.
    pub skipped_inert: usize,
    pub renders: usize,
    pub truncated: bool,
}

/// One move, ranked by its measured effect in the requested direction.
#[derive(Clone, Debug, Serialize)]
pub struct Move {
    pub name: String,
    pub from: f32,
    pub to: f32,
    pub from_text: String,
    pub to_text: String,
    /// Change of the quality's measure, in its unit, signed towards the
    /// requested direction (positive = helps).
    pub effect: f32,
    /// Band distance between the two renders, dB: everything that moved.
    pub distance_db: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Suggestion {
    pub seed: u64,
    pub quality: Quality,
    pub direction: Direction,
    pub unit: &'static str,
    pub measured: f32,
    /// Moves that help, best first.
    pub moves: Vec<Move>,
    pub renders: usize,
    pub truncated: bool,
}

// ------------------------------------------------------------ activity

pub(crate) fn value_of(preset: &Preset, name: &str) -> f32 {
    preset
        .settings
        .values
        .get(name)
        .and_then(Json::as_f64)
        .map(|v| v as f32)
        .or_else(|| parameters().lookup(name).map(|d| d.default_value))
        .unwrap_or(0.0)
}

fn on(preset: &Preset, name: &str) -> bool {
    value_of(preset, name) >= 0.5
}

/// Names no single offline render can hear, whatever the patch (the
/// sensitivity sweep's list, same reasons).
const NEVER_ACTIVE: &[&str] = &[
    "polyphony", "beats_per_minute", "bpm", "voice_priority", "voice_override",
    "mpe_enabled", "pitch_bend_range", "velocity_track", "view_spectrogram", "legato",
];

/// Names the Lite render mode overrides (it pins them), so a move on
/// them cannot be heard in that mode.
const PINNED_BY_LITE: &[&str] = &["oversampling", "polyphony"];

/// The parameters that can change this patch's sound: the inverse of the
/// sensitivity sweep's `context_for`. A module's parameters are active
/// when the module is on; a source's when it is connected; a model's
/// when that model is selected; GUI-only names never.
pub fn active_parameters(preset: &Preset) -> Vec<&'static ParamDetails> {
    active_parameters_for(preset, super::RenderMode::Faithful)
}

/// [`active_parameters`], minus what the render mode pins.
pub fn active_parameters_for(preset: &Preset, mode: super::RenderMode) -> Vec<&'static ParamDetails> {
    let table = parameters();
    let connected_sources: Vec<String> = connections_of(preset).into_iter().map(|((s, _), _)| s).collect();
    let connected = |source: &str| connected_sources.iter().any(|s| s == source);
    let effects = ["chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb", "convolution", "frequency_shifter"];
    let mut active = Vec::new();
    'params: for details in table.iter() {
        let name = details.name.as_str();
        if NEVER_ACTIVE.contains(&name) || name.contains("view") || name.starts_with("modulation_") {
            continue;
        }
        if mode == super::RenderMode::Lite && PINNED_BY_LITE.contains(&name) {
            continue;
        }
        for bus in ["bus_a", "bus_b"] {
            if let Some(rest) = name.strip_prefix(&format!("{bus}_")) {
                if !on(preset, &format!("{bus}_on")) {
                    continue 'params;
                }
                for effect in effects {
                    if rest.starts_with(&format!("{effect}_")) && rest != format!("{effect}_on") && !on(preset, &format!("{bus}_{effect}_on")) {
                        continue 'params;
                    }
                }
                active.push(details);
                continue 'params;
            }
        }
        for effect in effects {
            if name.starts_with(&format!("{effect}_")) && name != format!("{effect}_on") && !on(preset, &format!("{effect}_on")) {
                continue 'params;
            }
        }
        if name.starts_with("sample_") && name != "sample_on" && !on(preset, "sample_on") {
            continue;
        }
        if name.starts_with("noise_") && name != "noise_on" && !on(preset, "noise_on") {
            continue;
        }
        if let Some((family, index, rest)) = split_indexed(name) {
            match family {
                "osc" => {
                    if rest != "on" && !on(preset, &format!("osc_{index}_on")) {
                        continue;
                    }
                    let engine = value_of(preset, &format!("osc_{index}_engine")) as i32;
                    if rest.starts_with("smp_") && engine != 1 || rest.starts_with("gran_") && engine != 2 {
                        continue;
                    }
                    let unison = value_of(preset, &format!("osc_{index}_unison_voices")) > 1.0;
                    if (rest.starts_with("unison_detune") || rest.starts_with("detune_") || rest.starts_with("stack_") || rest == "stereo_spread" || rest == "frame_spread" || rest == "unison_blend") && !unison {
                        continue;
                    }
                    if rest.starts_with("distortion_") && rest != "distortion_type" && value_of(preset, &format!("osc_{index}_distortion_type")) == 0.0 {
                        continue;
                    }
                    if rest.starts_with("spectral_morph_") && rest != "spectral_morph_type" && value_of(preset, &format!("osc_{index}_spectral_morph_type")) == 0.0 {
                        continue;
                    }
                }
                "filter" => {
                    if rest != "on" && !on(preset, &format!("filter_{index}_on")) {
                        continue;
                    }
                    let model = value_of(preset, &format!("filter_{index}_model")) as i32;
                    if rest.starts_with("formant_") && model != 5 || rest.starts_with("comb_") && model != 6 {
                        continue;
                    }
                }
                "env" => {
                    if index != 1 && !connected(&format!("env_{index}")) {
                        continue;
                    }
                }
                "lfo" | "random" => {
                    if !connected(&format!("{family}_{index}")) {
                        continue;
                    }
                    if family == "lfo" {
                        let generator = value_of(preset, &format!("lfo_{index}_generator")) as i32;
                        if rest.starts_with("sh_") && generator != 1 || rest.starts_with("chaos_") && generator != 2 {
                            continue;
                        }
                        // Frequency and tempo: one of them is read, by sync type.
                        let synced = value_of(preset, &format!("lfo_{index}_sync")) >= 0.5;
                        if rest == "frequency" && synced || rest == "tempo" && !synced {
                            continue;
                        }
                    }
                }
                "macro_control" if !connected(&format!("macro_control_{index}")) => continue,
                _ => {}
            }
        }
        active.push(details);
    }
    active
}

/// The value at which a parameter stops contributing: off for a switch,
/// zero for an amount or a level, the table default otherwise.
pub fn neutral_value(details: &ParamDetails) -> f32 {
    let name = details.name.as_str();
    if details.is_boolean() {
        return 0.0;
    }
    if name.ends_with("_level") || name.ends_with("_amount") || name.ends_with("_mix") || name.ends_with("_dry_wet") || name.ends_with("_send") {
        return details.min.max(0.0);
    }
    details.default_value
}

// ------------------------------------------------------------ explain

struct Alternative {
    name: String,
    from: f32,
    to: f32,
}

/// What one alternative measured: the quality delta `measure(alt) −
/// measure(base)` and the band distance to the base; `None` when the
/// alternative could not be measured (silent, rejected).
type Measured = Option<(f32, f32)>;

/// One quality measurement of one patch: the value, and the render at
/// the scenario's own pitch (for the distance). `None` when the patch
/// could not be measured — silent (a level to zero says nothing about
/// the quality), rejected: a fact about the move, not a failure.
fn evaluate(session: &mut Session, preset: &Preset, scenario: &Scenario, seed: u64, quality: Quality) -> Option<(f32, Vec<f32>)> {
    if quality == Quality::Aliasing {
        let (report, samples) = super::aliasing::aliasing_with_samples(session, preset, scenario, seed).ok()?;
        return Some((report.ratio, samples));
    }
    // The same render seed for every patch: only the parameters differ.
    let r = render(session, preset, scenario, render_seed(seed, 0)).ok()?;
    if r.self_test.peak_dbfs < -60.0 {
        return None;
    }
    let value = quality.measure(&describe_without_pitch(&r));
    Some((value, r.samples))
}

/// Renders the patch once (twice for aliasing) per alternative, each
/// with one parameter changed, in parallel. Returns the base measure, one
/// [`Measured`] per alternative, the render count, and whether the budget
/// cut it short.
fn measure_alternatives(
    preset: &Preset,
    scenario: &Scenario,
    seed: u64,
    quality: Quality,
    alternatives: &[Alternative],
    budget: Budget,
) -> Result<(f32, Vec<Measured>, usize, bool), OpError> {
    let mut session = super::session();
    let per = quality.renders();
    let (base_measure, base_samples) = evaluate(&mut session, preset, scenario, seed, quality)
        .ok_or_else(|| OpError::Silent { peak_dbfs: -180.0 })?;
    let inner = Budget { max_renders: budget.max_renders.saturating_sub(per) / per, max_seconds: budget.max_seconds };
    let (results, ran) = parallel(alternatives.len(), inner, |i, session| {
        let alt = &alternatives[i];
        let mut p = preset.clone();
        p.settings.values.insert(alt.name.clone(), Json::from(alt.to as f64));
        let (value, samples) = evaluate(session, &p, scenario, seed, quality)?;
        let dist = distance(&base_samples, &samples, SAMPLE_RATE, Options::default());
        Some((value - base_measure, dist.total_db))
    });
    let truncated = ran < alternatives.len();
    Ok((base_measure, results.into_iter().map(|r| r.flatten()).collect(), (ran + 1) * per, truncated))
}

pub fn explain(preset: &Preset, scenario: &Scenario, quality: Quality, seed: u64, budget: Budget) -> Result<Explanation, OpError> {
    let active = active_parameters_for(preset, scenario.mode);
    let total = parameters().len();
    let mut alternatives = Vec::new();
    let mut at_neutral = 0;
    for details in &active {
        let from = value_of(preset, &details.name);
        let to = neutral_value(details);
        if from == to {
            at_neutral += 1;
            continue;
        }
        alternatives.push(Alternative { name: details.name.clone(), from, to });
    }
    if alternatives.is_empty() {
        return Err(OpError::Nothing { message: "no active parameter departs from its neutral value".into() });
    }
    let (measured, results, renders, truncated) = measure_alternatives(preset, scenario, seed, quality, &alternatives, budget)?;
    let mut ranked: Vec<Contribution> = alternatives
        .iter()
        .zip(results)
        .filter_map(|(alt, r)| {
            let (delta, distance_db) = r?;
            Some(Contribution {
                name: alt.name.clone(),
                value: alt.from,
                value_text: spell(&alt.name, alt.from).unwrap_or_default(),
                neutral_value: alt.to,
                neutral_text: spell(&alt.name, alt.to).unwrap_or_default(),
                // With minus without: the parameter's share.
                effect: -delta,
                distance_db,
            })
        })
        .collect();
    ranked.sort_by(|a, b| b.effect.abs().partial_cmp(&a.effect.abs()).unwrap_or(core::cmp::Ordering::Equal));
    Ok(Explanation {
        seed,
        quality,
        unit: quality.unit(),
        measured,
        ranked,
        at_neutral,
        skipped_inert: total - active.len(),
        renders,
        truncated,
    })
}

// ------------------------------------------------------------ suggest

/// A step of a quarter of the range for a continuous parameter, one
/// index for a discrete one, a flip for a switch; `None` when the value
/// cannot move that way.
pub(crate) fn step(details: &ParamDetails, from: f32, up: bool) -> Option<f32> {
    let to = if details.is_boolean() {
        if up == (from >= 0.5) { return None } else { 1.0 - from.round() }
    } else if details.scale == ParamScale::Indexed {
        from.round() + if up { 1.0 } else { -1.0 }
    } else {
        from + (details.max - details.min) * 0.25 * if up { 1.0 } else { -1.0 }
    };
    let to = to.clamp(details.min, details.max);
    if (to - from).abs() < 1e-6 { None } else { Some(to) }
}

/// `include_switches`: also try indexed parameters and switches (a filter
/// model, an effect's on/off). Off by default — switching a module on is
/// a jump, not a move, and it dominates every ranking it enters.
pub fn suggest(preset: &Preset, scenario: &Scenario, quality: Quality, direction: Direction, include_switches: bool, seed: u64, budget: Budget) -> Result<Suggestion, OpError> {
    let active = active_parameters_for(preset, scenario.mode);
    let mut alternatives = Vec::new();
    for details in &active {
        if details.scale == ParamScale::Indexed && !include_switches {
            continue;
        }
        let from = value_of(preset, &details.name);
        for up in [true, false] {
            if let Some(to) = step(details, from, up) {
                alternatives.push(Alternative { name: details.name.clone(), from, to });
            }
        }
    }
    if alternatives.is_empty() {
        return Err(OpError::Nothing { message: "no active parameter can move".into() });
    }
    let (measured, results, renders, truncated) = measure_alternatives(preset, scenario, seed, quality, &alternatives, budget)?;
    let sign = if direction == Direction::More { 1.0 } else { -1.0 };
    let mut moves: Vec<Move> = alternatives
        .iter()
        .zip(results)
        .filter_map(|(alt, r)| {
            let (delta, distance_db) = r?;
            let effect = sign * delta;
            (effect > 0.0).then(|| Move {
                name: alt.name.clone(),
                from: alt.from,
                to: alt.to,
                from_text: spell(&alt.name, alt.from).unwrap_or_default(),
                to_text: spell(&alt.name, alt.to).unwrap_or_default(),
                effect,
                distance_db,
            })
        })
        .collect();
    moves.sort_by(|a, b| b.effect.partial_cmp(&a.effect).unwrap_or(core::cmp::Ordering::Equal));
    Ok(Suggestion { seed, quality, direction, unit: quality.unit(), measured, moves, renders, truncated })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::saw_patch;

    #[test]
    fn active_parameters_follow_the_patch() {
        let p = saw_patch();
        let names: Vec<&str> = active_parameters(&p).iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"filter_1_cutoff"));
        assert!(names.contains(&"osc_1_wave_frame"));
        assert!(!names.contains(&"filter_2_cutoff"), "filter 2 is off");
        assert!(!names.contains(&"lfo_1_frequency"), "lfo 1 is unconnected");
        assert!(!names.contains(&"osc_2_level"), "osc 2 is off");
        assert!(!names.contains(&"filter_1_formant_x"), "not the formant model");
        assert!(!names.contains(&"reverb_dry_wet"), "reverb off");
        assert!(names.contains(&"env_1_attack"), "the amp envelope is always heard");
    }

    #[test]
    fn the_cutoff_explains_the_brightness_of_a_filtered_saw() {
        let mut p = saw_patch();
        p.settings.values.insert("filter_1_cutoff".into(), 55.0.into());
        let e = explain(&p, &Scenario::lite(), Quality::Brightness, 1, Budget::default()).expect("explains");
        assert!(!e.truncated);
        let top = &e.ranked[0];
        // What moves brightness most is the filter: its switch (on → off),
        // its mix (→ dry) or its cutoff (55 → the default).
        assert!(["filter_1_cutoff", "filter_1_on", "filter_1_mix"].contains(&top.name.as_str()), "{}", top.name);
        assert!(e.ranked.iter().any(|c| c.name == "filter_1_on" && c.effect < 0.0), "the filter, as set, darkens: {:?}", e.ranked.iter().map(|c| (&c.name, c.effect)).collect::<Vec<_>>());
    }

    #[test]
    fn suggest_finds_the_cutoff_to_brighten() {
        let mut p = saw_patch();
        p.settings.values.insert("filter_1_cutoff".into(), 55.0.into());
        let s = suggest(&p, &Scenario::lite(), Quality::Brightness, Direction::More, false, 1, Budget::default()).expect("suggests");
        let cutoff = s.moves.iter().find(|m| m.name == "filter_1_cutoff").expect("cutoff up helps");
        assert!(cutoff.to > cutoff.from);
        assert!(cutoff.effect > 1.0, "at least a semitone of centroid: {}", cutoff.effect);
        assert!(s.moves.iter().position(|m| m.name == "filter_1_cutoff").unwrap() < 3, "in the top three: {:?}", s.moves.iter().take(6).map(|m| (&m.name, m.effect)).collect::<Vec<_>>());
    }
}
