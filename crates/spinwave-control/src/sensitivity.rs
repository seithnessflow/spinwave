//! Sweeps the parameter table asking one question of every entry: does
//! moving it change the sound at all?
//!
//! The golden bench found the formant filter's controls were wired to
//! nothing — `fill_filter_params` simply never read them. No test noticed,
//! and no test could have, because every test was written by someone who
//! already believed the parameter worked. With 800-odd parameters there is
//! no reason to think it was the only one.
//!
//! So: render a patch, move one parameter, render again, and compare. A
//! parameter that changes nothing is either dead or needs a context it did
//! not get. That distinction is the whole difficulty, and it lives in
//! [`context_for`]: a filter's formant controls do nothing unless the
//! filter is ON and its model IS the formant one, an LFO's shape does
//! nothing unless the LFO is CONNECTED to something. A sweep without that
//! context reports hundreds of false positives, everyone stops reading it,
//! and the one real dead control goes back to hiding in the noise.
//!
//! What is left over after the context rules is [`EXPECTED_INERT`]:
//! parameters that genuinely cannot show up in one offline note. Adding a
//! name there is a claim, so each one carries its reason.
//!
//! # State: NOT yet a clean signal
//!
//! The sweep reports **393 of 1313** parameters inert. That is not 393
//! bugs, and the number must come down before this can be read as a list
//! of findings — a tool that cries wolf gets ignored, which is the one
//! failure mode that matters here. It already earns its keep by
//! reproducing the formant filter's five dead controls from a cold start,
//! and the count has come down 964 → 595 → 401 → 393 as context rules
//! were added, so what remains is mostly more of the same work:
//!
//! * `lfo_N_frequency` (12) is inert because the default `sync_type` is
//!   tempo-synced, so the rate comes from `lfo_N_tempo` and the frequency
//!   knob is correctly ignored. Verified: with the SAME wiring
//!   `modulation_1_amount` does change the sound, so the connection is
//!   live and the LFO is simply not reading that control. Needs a rule
//!   that sets the free-running sync type when testing `frequency` and a
//!   synced one when testing `tempo`. The same likely covers `fade_time`,
//!   `sync_type` and the two `keytrack_*`.
//! * `osc_N_smp_*` and `osc_N_gran_*` (about 40) get their engine
//!   switched, but no sample is loaded, so there is nothing to play.
//! * `env_N_decay`, `_hold`, `_decay_power` (24) need a sustain below 1:
//!   the base patch holds the note at full sustain, where decay has
//!   nothing to decay to.
//!
//! Until those are done, read the sweep as a worklist, not a verdict. It
//! exits 0 for that reason.

use serde_json::{Map, Value};
use spinwave_params::{parameters, ParamDetails, Preset};

use crate::session::{NoteSpec, Session};

/// Parameters that cannot change this render, with the reason why.
///
/// Every entry is a claim that the parameter is fine and the SWEEP is
/// blind to it, not that the parameter does not matter. Anything added
/// here without a reason is a bug being hidden.
const EXPECTED_INERT: &[(&str, &str)] = &[
    ("polyphony", "one note: voice stealing never happens"),
    ("oversampling", "the bench renders at a fixed rate"),
    ("beats_per_minute", "nothing here is tempo-synced by default"),
    ("bpm", "same"),
    ("voice_priority", "one note: nothing to prioritise"),
    ("voice_override", "one note: nothing to override"),
    ("mpe_enabled", "no per-channel MIDI in an offline render"),
    ("pitch_bend_range", "the render sends no bend"),
    ("stereo_routing", "measured on the summed difference"),
    ("velocity_track", "every note here has the same velocity"),
];

/// One parameter that moved without the sound moving.
#[derive(Clone, Debug)]
pub struct Finding {
    pub name: String,
    /// The value it was moved to, for reproducing by hand.
    pub moved_to: f32,
    pub default_value: f32,
}

/// A patch that gives most of the engine something to do: one oscillator
/// at a real level through a filter, with an envelope that opens and
/// stays open for the length of the render.
fn base_settings() -> Map<String, Value> {
    let mut settings = Map::new();
    for (name, value) in [
        ("osc_1_on", 1.0),
        ("osc_1_level", 0.7),
        ("osc_1_wave_frame", 128.0),
        ("osc_1_random_phase", 0.0),
        ("filter_1_on", 1.0),
        ("filter_1_cutoff", 72.0),
        ("filter_1_resonance", 0.5),
        ("env_1_attack", 0.0),
        ("env_1_decay", 0.5),
        ("env_1_sustain", 1.0),
        ("env_1_release", 0.3),
    ] {
        settings.insert(name.into(), Value::from(value));
    }
    settings
}

/// Splits `prefix_N_rest` into its family, its index and the rest, so the
/// context rules can be written once per family instead of once per slot.
pub(crate) fn split_indexed(name: &str) -> Option<(&str, usize, &str)> {
    let mut best = None;
    for (position, _) in name.match_indices('_') {
        let tail = &name[position + 1..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            continue;
        }
        let rest_at = position + 1 + digits.len();
        let rest = name[rest_at..].strip_prefix('_').unwrap_or("");
        if let Ok(index) = digits.parse::<usize>() {
            best = Some((&name[..position], index, rest));
        }
    }
    best
}

/// What a parameter needs switched on before it can possibly be heard: a
/// source's own controls are inert until something listens to it, and a
/// model's controls are inert until that model is the one selected.
struct Context {
    settings: Vec<(String, f32)>,
    /// The connection to wire, and which slot to wire it in.
    connection: Option<(String, String)>,
    slot: usize,
}

/// What a parameter needs switched on before it can possibly be heard.
fn context_for(name: &str) -> Context {
    let mut settings: Vec<(String, f32)> = Vec::new();
    let mut connection = None;
    let mut slot = 1usize;

    const EFFECTS: [&str; 12] = [
        "chorus", "compressor", "delay", "distortion", "equalizer", "eq", "filter_fx",
        "flanger", "phaser", "reverb", "convolution", "frequency_shifter",
    ];

    // The send buses carry no signal until they are switched on AND an
    // oscillator is routed to one; their effect chains then sit behind
    // their own switches, exactly like the main chain's.
    for (bus, destination) in [("bus_a", 5.0), ("bus_b", 6.0)] {
        let Some(rest) = name.strip_prefix(&format!("{bus}_")) else { continue };
        settings.push((format!("{bus}_on"), 1.0));
        settings.push(("osc_1_destination".into(), destination));
        for effect in EFFECTS {
            if rest.starts_with(&format!("{effect}_")) {
                settings.push((format!("{bus}_{effect}_on"), 1.0));
            }
        }
    }

    // Effects sit behind their own on/off. The name of the switch is the
    // family plus `_on`, which is also how the preset format spells it.
    for effect in EFFECTS {
        if name.starts_with(&format!("{effect}_")) {
            settings.push((format!("{effect}_on"), 1.0));
        }
    }

    if let Some((family, index, rest)) = split_indexed(name) {
        match family {
            "osc" => {
                settings.push((format!("osc_{index}_on"), 1.0));
                settings.push((format!("osc_{index}_level"), 0.7));
                // Detune, blend, stack and spread describe a relationship
                // BETWEEN unison voices; with one voice there is nothing
                // for them to describe and they are correctly inert.
                if rest.starts_with("unison_")
                    || rest.starts_with("detune_")
                    || rest.starts_with("stack_")
                    || rest == "stereo_spread"
                    || rest == "frame_spread"
                {
                    settings.push((format!("osc_{index}_unison_voices"), 4.0));
                    settings.push((format!("osc_{index}_unison_detune"), 4.0));
                }
                // Phase distortion has an amount only once it has a type.
                if rest.starts_with("distortion_") && rest != "distortion_type" {
                    settings.push((format!("osc_{index}_distortion_type"), 1.0));
                }
                if rest.starts_with("spectral_morph_") && rest != "spectral_morph_type" {
                    settings.push((format!("osc_{index}_spectral_morph_type"), 1.0));
                }
                // The sample, granular and multisample controls belong to
                // an engine the oscillator is not running by default.
                if rest.starts_with("smp_") {
                    settings.push((format!("osc_{index}_engine"), 1.0));
                } else if rest.starts_with("gran_") {
                    settings.push((format!("osc_{index}_engine"), 2.0));
                }
            }
            "filter" => {
                settings.push((format!("filter_{index}_on"), 1.0));
                // A filter model reads only its own controls. This is the
                // rule that would have caught the formant filter: without
                // it those five names look dead in every model but one.
                if rest.starts_with("formant_") {
                    settings.push((format!("filter_{index}_model"), 5.0));
                } else if rest.starts_with("comb_") {
                    settings.push((format!("filter_{index}_model"), 6.0));
                }
            }
            // A modulation source changes nothing until it is connected.
            // Wire it to the filter cutoff, which is audible and accepts
            // every source, and give the connection a real amount.
            "env" | "lfo" | "random" => {
                let source = match family {
                    "env" => format!("env_{index}"),
                    "lfo" => format!("lfo_{index}"),
                    _ => format!("random_{index}"),
                };
                settings.push(("filter_1_on".into(), 1.0));
                settings.push(("filter_1_cutoff".into(), 60.0));
                settings.push(("modulation_1_amount".into(), 0.8));
                // An LFO's generator decides what produces its value, and
                // each generator reads only its own controls: the glide
                // belongs to sample-and-hold, the chaos speed to the
                // attractors. Under the default drawn-shape generator both
                // are correctly ignored.
                if family == "lfo" {
                    if rest.starts_with("sh_") {
                        settings.push((format!("lfo_{index}_generator"), 1.0));
                    } else if rest.starts_with("chaos_") {
                        settings.push((format!("lfo_{index}_generator"), 2.0));
                    }
                }
                connection = Some((source, "filter_1_cutoff".into()));
            }
            // `macro_control_N` splits as family "macro_control", not
            // "macro": the digit is the LAST underscore-delimited number.
            "macro_control" => {
                settings.push(("filter_1_on".into(), 1.0));
                settings.push(("filter_1_cutoff".into(), 60.0));
                settings.push(("modulation_1_amount".into(), 0.8));
                connection = Some((format!("macro_control_{index}"), "filter_1_cutoff".into()));
            }
            // `modulation_N_*` shapes the connection in SLOT N, so the
            // connection has to be made in that slot and not the first.
            "modulation" => {
                settings.push((format!("modulation_{index}_amount"), 0.8));
                settings.push(("filter_1_on".into(), 1.0));
                settings.push(("filter_1_cutoff".into(), 60.0));
                slot = index;
                connection = Some(("lfo_1".into(), "filter_1_cutoff".into()));
            }
            _ => {}
        }
    }
    Context { settings, connection, slot }
}

/// A value as far as the range allows from the one the patch ALREADY has,
/// so a parameter that does anything at all has to show it.
///
/// Moving away from the table default is not enough: the base patch and
/// the context rules set parameters themselves, and a "move" that lands on
/// the value already in place renders the same audio twice and reports the
/// parameter dead. `filter_1_on` did exactly that on the first run.
fn moved_value(details: &ParamDetails, current: f32) -> f32 {
    if details.is_boolean() {
        return if current >= 0.5 { 0.0 } else { 1.0 };
    }
    let low = details.min;
    let high = details.max;
    if (current - low).abs() >= (high - current).abs() {
        low
    } else {
        high
    }
}

fn build(settings: Map<String, Value>, connection: Option<(String, String)>, slot: usize) -> Preset {
    let mut preset = Preset { preset_name: "sensitivity".into(), ..Default::default() };
    preset.settings.values = settings;
    // A preset with no `lfos` array leaves every LFO on whatever shape the
    // generator was constructed with, and a flat shape makes an LFO a
    // constant: its rate, its fade and its sync then genuinely change
    // nothing, and the sweep blames the parameters. Give each one the
    // triangle the reference starts from.
    preset.settings.lfos = (0..12)
        .map(|_| spinwave_params::preset::LineShape {
            num_points: 3,
            points: vec![0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
            powers: vec![0.0, 0.0, 0.0],
            name: Some("Triangle".into()),
            smooth: false,
            ..Default::default()
        })
        .collect();
    if let Some((source, destination)) = connection {
        // Slots are positional: slot N is the Nth entry, so the ones
        // before it are present and empty rather than absent.
        while preset.settings.modulations.len() + 1 < slot {
            preset.settings.modulations.push(Default::default());
        }
        preset.settings.modulations.push(spinwave_params::preset::ModulationConnection {
            source,
            destination,
            ..Default::default()
        });
    }
    preset
}

fn render(preset: &Preset) -> Result<Vec<f32>, String> {
    let mut session = Session::with_output_dir(std::env::temp_dir());
    session.load_preset_json(&preset.to_json().map_err(|e| e.to_string())?)?;
    let notes = [NoteSpec { note: 57, velocity: 0.9, start: 0.0, duration: 0.45, channel: 0 }];
    Ok(session.render_samples(&notes, 0.6, 120.0))
}

fn identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1.0e-9)
}

/// Runs the sweep. `only` restricts it to parameters whose name contains
/// that substring, which is how you check one family after a fix.
pub fn sweep(only: Option<&str>, mut report: impl FnMut(&str)) -> Result<Vec<Finding>, String> {
    let table = parameters();
    let expected: Vec<&str> = EXPECTED_INERT.iter().map(|(name, _)| *name).collect();
    let mut findings = Vec::new();
    let mut checked = 0usize;

    for details in table.iter() {
        if only.is_some_and(|filter| !details.name.contains(filter)) {
            continue;
        }
        if expected.contains(&details.name.as_str()) {
            continue;
        }
        if details.max <= details.min {
            continue;
        }

        let context = context_for(&details.name);
        let mut settings = base_settings();
        for (name, value) in &context.settings {
            settings.insert(name.clone(), Value::from(*value));
        }

        let current = settings
            .get(&details.name)
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
            .unwrap_or(details.default_value);

        let before = render(&build(settings.clone(), context.connection.clone(), context.slot))?;
        let moved = moved_value(details, current);
        settings.insert(details.name.clone(), Value::from(moved));
        let after = render(&build(settings, context.connection, context.slot))?;

        checked += 1;
        if identical(&before, &after) {
            report(&format!(
                "{:<34} {} -> {} changes nothing",
                details.name, current, moved
            ));
            findings.push(Finding {
                name: details.name.clone(),
                moved_to: moved,
                default_value: current,
            });
        }
    }
    report(&format!("checked {checked} parameters, {} inert", findings.len()));
    Ok(findings)
}
