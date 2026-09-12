//! The operations that make the synth answerable: measure, compare,
//! explain, apply, explore. Design: `notes/operations-design.md`.
//!
//! Library first. The CLI and the MCP server are clients; a GUI or a
//! hosted API would be two more. Nothing here needs a model, and every
//! result is a structure with units and codes, not prose.
//!
//! Three rules every operation obeys, each learned the hard way:
//!
//! * **self-test before numbers** — a render that is silent while an
//!   oscillator is on, a buffer too short to frame, a preset that did not
//!   load cleanly: the operation refuses ([`OpError`]) instead of
//!   returning a plausible value;
//! * **a seed in, the seed out** — every render reseeds every generator
//!   from a seed derived from the operation's seed and the render's own
//!   index, so the result depends on neither the process history nor the
//!   thread that ran it ([`render_seed`]);
//! * **a budget** — a search never blocks unbounded; what it could not
//!   finish it says ([`Budget`]).

pub mod aliasing;
pub mod apply;
pub mod compare;
pub mod descriptors;
pub mod diff;
pub mod distance;
pub mod explain;
pub mod explore;
pub mod measure;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use spinwave_params::{parameters, Preset};

use crate::session::{NoteSpec, Session, SAMPLE_RATE};

pub use aliasing::{aliasing, AliasingReport};
pub use apply::{apply, Applied, GoalCheck};
pub use compare::{compare, Comparison};
pub use descriptors::Descriptors;
pub use diff::{param_diff, Change, ConnectionChange, Diff, ParamChange};
pub use distance::{distance, Distance, Options as DistanceOptions};
pub use explain::{explain, suggest, Contribution, Direction, Explanation, Move, Quality};
pub use explore::{explore, interpolate, ExploreSpec, Variant};
pub use measure::{measure, Measurement};

/// How a scenario is rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderMode {
    /// Polyphony and oversampling as the patch says, stereo.
    #[default]
    Faithful,
    /// Polyphony 1, oversampling 1×, one short note; stereo is kept
    /// (width is a descriptor). What searches spend.
    Lite,
}

/// What is played: notes, length, tempo, and the render mode.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scenario {
    pub notes: Vec<NoteSpec>,
    /// Total render length; 0 = 1.5 s after the last note ends.
    pub seconds: f32,
    pub bpm: f32,
    #[serde(default)]
    pub mode: RenderMode,
}

impl Scenario {
    /// One note, C3 at 0.8, held for `hold` seconds, rendered for
    /// `seconds`.
    pub fn one_note(midi: i32, hold: f32, seconds: f32, mode: RenderMode) -> Scenario {
        Scenario {
            notes: vec![NoteSpec { note: midi, start: 0.0, duration: hold, velocity: 0.8, channel: 0 }],
            seconds,
            bpm: 120.0,
            mode,
        }
    }

    /// The Lite scenario: C3 for 0.4 s, 0.6 s of audio.
    pub fn lite() -> Scenario {
        Scenario::one_note(60, 0.4, 0.6, RenderMode::Lite)
    }

    /// The Faithful default: C3 for 1.5 s, 2.5 s of audio.
    pub fn faithful() -> Scenario {
        Scenario::one_note(60, 1.5, 2.5, RenderMode::Faithful)
    }
}

/// What a search may spend. Exceeding either bound ends the search with
/// what it has and `truncated = true`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Budget {
    pub max_renders: usize,
    pub max_seconds: f32,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget { max_renders: 400, max_seconds: 60.0 }
    }
}

/// Why an operation refused. Serialised with a `code` an agent can act on.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum OpError {
    /// The render peaks below −60 dBFS while a source is on.
    Silent { peak_dbfs: f32 },
    NotFinite,
    /// Too short to frame (fewer than two analysis frames).
    Empty { frames: usize },
    /// The preset did not load cleanly; the messages are the format's.
    Rejected { messages: Vec<String> },
    UnknownParameter { name: String },
    BadValue { name: String, message: String },
    NotInterpolable { name: String },
    /// Nothing to do: no active parameter, no change in the diff…
    Nothing { message: String },
    /// A note outside 0..=127, a negative duration…
    BadScenario { message: String },
}

impl core::fmt::Display for OpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OpError::Silent { peak_dbfs } => write!(f, "silent render ({peak_dbfs:.1} dBFS) while a source is on"),
            OpError::NotFinite => write!(f, "the render is not finite"),
            OpError::Empty { frames } => write!(f, "too short to measure ({frames} frames)"),
            OpError::Rejected { messages } => write!(f, "preset rejected: {}", messages.join("; ")),
            OpError::UnknownParameter { name } => write!(f, "unknown parameter `{name}`"),
            OpError::BadValue { name, message } => write!(f, "`{name}`: {message}"),
            OpError::NotInterpolable { name } => write!(f, "`{name}` cannot be interpolated"),
            OpError::Nothing { message } => write!(f, "{message}"),
            OpError::BadScenario { message } => write!(f, "bad scenario: {message}"),
        }
    }
}

/// What the render self-test saw, kept in every result so the caller can
/// see the check happened.
#[derive(Clone, Debug, Serialize)]
pub struct SelfTest {
    pub peak_dbfs: f32,
    pub frames: usize,
    /// Whether a source was on, so silence would have been refused.
    pub source_on: bool,
}

/// The seed of one render: the operation's seed mixed with the render's
/// index in the operation, so the same render gets the same seed on any
/// thread and in any order (SplitMix64 finaliser, truncated).
pub fn render_seed(seed: u64, index: u64) -> u32 {
    let mut z = seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

/// A checked render: the preset applied with the scenario's mode, the
/// engine reseeded, the audio inspected before it is returned.
pub struct Render {
    pub samples: Vec<f32>,
    pub self_test: SelfTest,
    pub cost: Duration,
}

/// Whether the patch has any source switched on with a level above zero:
/// the condition under which a silent render is an error.
pub fn source_on(preset: &Preset) -> bool {
    let table = parameters();
    let get = |name: &str| -> f32 {
        preset
            .settings
            .values
            .get(name)
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .or_else(|| table.lookup(name).map(|d| d.default_value))
            .unwrap_or(0.0)
    };
    let oscillators = (1..=4).any(|i| get(&format!("osc_{i}_on")) >= 0.5 && get(&format!("osc_{i}_level")) > 0.0);
    oscillators || (get("sample_on") >= 0.5 && get("sample_level") > 0.0) || get("noise_on") >= 0.5
}

/// Applies the render mode to a copy of the preset.
fn for_mode(preset: &Preset, mode: RenderMode) -> Preset {
    let mut p = preset.clone();
    if mode == RenderMode::Lite {
        p.settings.values.insert("polyphony".into(), 1.0.into());
        p.settings.values.insert("oversampling".into(), 0.0.into());
    }
    p
}

fn check_scenario(scenario: &Scenario) -> Result<(), OpError> {
    if scenario.notes.is_empty() {
        return Err(OpError::BadScenario { message: "no notes".into() });
    }
    for n in &scenario.notes {
        if !(0..=127).contains(&n.note) {
            return Err(OpError::BadScenario { message: format!("note {} outside 0..=127", n.note) });
        }
        if n.duration <= 0.0 || n.start < 0.0 {
            return Err(OpError::BadScenario { message: "a note needs a start >= 0 and a duration > 0".into() });
        }
    }
    if scenario.seconds < 0.0 || scenario.bpm <= 0.0 {
        return Err(OpError::BadScenario { message: "seconds >= 0 and bpm > 0".into() });
    }
    Ok(())
}

/// Renders `preset` under `scenario` in `session`, reseeded with `seed`,
/// and refuses what cannot be measured.
pub fn render(session: &mut Session, preset: &Preset, scenario: &Scenario, seed: u32) -> Result<Render, OpError> {
    check_scenario(scenario)?;
    let started = Instant::now();
    let staged = for_mode(preset, scenario.mode);
    let json = staged.to_json().map_err(|e| OpError::Rejected { messages: vec![e.to_string()] })?;
    session.set_random_seed(Some(seed));
    session.load_preset_json(&json).map_err(|e| OpError::Rejected { messages: vec![e] })?;
    let report = &session.last_report;
    if !report.errors.is_empty() || !report.ignored_connections.is_empty() {
        let mut messages: Vec<String> = report.errors.iter().map(|e| format!("{}: {}", e.key, e.message)).collect();
        messages.extend(report.ignored_connections.iter().map(|c| format!("ignored connection {c}")));
        return Err(OpError::Rejected { messages });
    }
    let samples = session.render_samples(&scenario.notes, scenario.seconds, scenario.bpm);
    let cost = started.elapsed();
    if !samples.iter().all(|s| s.is_finite()) {
        return Err(OpError::NotFinite);
    }
    let frames = samples.len() / 2;
    if frames < 2 * 2048 {
        return Err(OpError::Empty { frames });
    }
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let peak_dbfs = descriptors::db(peak);
    let on = source_on(preset);
    if on && peak_dbfs < -60.0 {
        return Err(OpError::Silent { peak_dbfs });
    }
    Ok(Render { samples, self_test: SelfTest { peak_dbfs, frames, source_on: on }, cost })
}

/// Loads a `.vital` or `.spinwave` patch (the sidecar `<path>.d/` for the
/// latter), refusing what does not read cleanly.
pub fn load_patch(path: &str) -> Result<Preset, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    if path.ends_with(".spinwave") {
        let blobs = crate::text_preset::Blobs::from_dir(&std::path::PathBuf::from(format!("{path}.d")))?;
        let result = crate::text_preset::read(&text, &blobs);
        result.preset.ok_or_else(|| format!("{path}: {}", result.report.summary()))
    } else {
        Preset::from_json(text.trim_start_matches('\u{feff}')).map_err(|e| format!("{path}: {e}"))
    }
}

/// Writes a patch as `.spinwave` (with its sidecar) or `.vital`, by the
/// path's extension.
pub fn save_patch(preset: &Preset, path: &str) -> Result<(), String> {
    if path.ends_with(".spinwave") {
        let written = crate::text_preset::write(preset);
        std::fs::write(path, written.text).map_err(|e| format!("{path}: {e}"))?;
        if !written.blobs.is_empty() {
            written.blobs.write_dir(&std::path::PathBuf::from(format!("{path}.d")))?;
        }
    } else {
        let text = preset.for_vital_file().to_json_pretty().map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))?;
    }
    Ok(())
}

/// A session per thread, made on demand.
pub(crate) fn session() -> Session {
    Session::with_output_dir(std::env::temp_dir())
}

/// Runs `job(index, session)` for every index in `0..count` on every
/// core, each thread owning one [`Session`], results in index order. Stops
/// handing out work once `budget` is spent; the second element says how
/// many jobs ran.
pub(crate) fn parallel<T: Send>(
    count: usize,
    budget: Budget,
    job: impl Fn(usize, &mut Session) -> T + Sync,
) -> (Vec<Option<T>>, usize) {
    let threads = thread_count().min(count.max(1));
    parallel_on(threads, count, budget, job)
}

/// Workers beyond this made the exploration SLOWER, measured
/// (`examples/render_cost.rs`, 2026-09-12, 16-core Ryzen): 43 renders/s
/// on 1 thread, 94 on 4, 80 on 8, 52 on 16, 31 on 32. The block loop
/// itself scales (6× at 16 threads); what does not is rebuilding the
/// voice kernels per render — allocation and first-touch page faults,
/// which every thread pays through the same memory system. The next lever
/// is reusing kernels the way the effect chains are reused
/// (`SoundEngine::recycle`); until then, four.
pub const DEFAULT_THREADS: usize = 4;

/// The worker count: [`DEFAULT_THREADS`] capped by the cores, or
/// `SPINWAVE_THREADS` when set (the determinism test runs the same
/// operation on 1 and on several; a machine where more helps sets it).
pub fn thread_count() -> usize {
    std::env::var("SPINWAVE_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or_else(|| DEFAULT_THREADS.min(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)))
}

pub(crate) fn parallel_on<T: Send>(
    threads: usize,
    count: usize,
    budget: Budget,
    job: impl Fn(usize, &mut Session) -> T + Sync,
) -> (Vec<Option<T>>, usize) {
    let next = AtomicUsize::new(0);
    let started = Instant::now();
    let deadline = Duration::from_secs_f32(budget.max_seconds.max(0.0));
    let limit = count.min(budget.max_renders);
    let mut results: Vec<Option<T>> = (0..count).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<Option<T>>> = (0..count).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut session = session();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= limit || started.elapsed() > deadline {
                        break;
                    }
                    let value = job(i, &mut session);
                    *slots[i].lock().expect("slot") = Some(value);
                }
            });
        }
    });
    let mut ran = 0;
    for (slot, out) in slots.into_iter().zip(results.iter_mut()) {
        *out = slot.into_inner().expect("slot");
        ran += usize::from(out.is_some());
    }
    (results, ran)
}

/// Descriptors of a checked render.
pub(crate) fn describe(render: &Render) -> Descriptors {
    descriptors::describe(&render.samples, SAMPLE_RATE)
}

/// Descriptors without the pitch detector, for the searches (explain,
/// suggest, explore): no quality reads `f0`, and YIN is a third of a
/// Lite measurement.
pub(crate) fn describe_without_pitch(render: &Render) -> Descriptors {
    descriptors::describe_with(&render.samples, SAMPLE_RATE, false)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A patch that makes a sound: one saw through a filter.
    pub(crate) fn saw_patch() -> Preset {
        let mut p = Preset::default();
        for (k, v) in [
            ("osc_1_on", 1.0),
            ("osc_1_level", 0.7),
            ("osc_1_wave_frame", 128.0),
            ("filter_1_on", 1.0),
            ("filter_1_cutoff", 80.0),
            ("filter_1_resonance", 0.3),
            ("env_1_attack", 0.05),
            ("env_1_release", 0.3),
        ] {
            p.settings.values.insert(k.into(), serde_json::Value::from(v));
        }
        p
    }

    #[test]
    fn a_silent_patch_with_a_source_on_is_refused_and_one_with_none_is_not() {
        let mut p = saw_patch();
        p.settings.values.insert("osc_1_level".into(), 0.0.into());
        let mut session = session();
        assert!(!source_on(&p));
        assert!(render(&mut session, &p, &Scenario::lite(), 1).is_ok(), "no source on: silence is fine");
        p.settings.values.insert("osc_1_level".into(), 0.7.into());
        p.settings.values.insert("volume".into(), 0.0.into());
        assert!(source_on(&p));
        match render(&mut session, &p, &Scenario::lite(), 1) {
            Err(OpError::Silent { .. }) => {}
            other => panic!("expected Silent, got {:?}", other.map(|_| ())),
        }
    }

    /// The rule the bench learned: the same operation gives the same
    /// bytes whatever ran before, in whatever order, on however many
    /// threads. Descriptors compared as serialised JSON, exactly.
    #[test]
    fn the_same_operation_gives_the_same_bytes_in_any_order_on_any_thread_count() {
        let base = saw_patch();
        let mut patches = Vec::new();
        for i in 0..6 {
            let mut p = base.clone();
            p.settings.values.insert("filter_1_cutoff".into(), (50.0 + 8.0 * i as f64).into());
            if i % 2 == 0 {
                // A random source, so the seed matters.
                p.settings.modulations.push(spinwave_params::preset::ModulationConnection {
                    source: "random_1".into(),
                    destination: "filter_1_cutoff".into(),
                    ..Default::default()
                });
                p.settings.values.insert("modulation_1_amount".into(), 0.5.into());
            }
            patches.push(p);
        }
        let json = |p: &Preset| serde_json::to_string(&measure(p, &Scenario::lite(), 42).unwrap().descriptors).unwrap();
        let forward: Vec<String> = patches.iter().map(json).collect();
        let backward: Vec<String> = patches.iter().rev().map(json).collect();
        for (i, (f, b)) in forward.iter().zip(backward.iter().rev()).enumerate() {
            assert_eq!(f, b, "patch {i} measured differently in reverse order");
        }
        // The same exploration on one thread and on four: byte-identical.
        let spec = ExploreSpec { count: 5, amplitude: 0.2, seed: 9, switch_indexed: 0.0, budget: Budget::default() };
        let (one, _) = parallel_on(1, 1, Budget::default(), |_, _| serde_json::to_string(&explore(&patches[0], &Scenario::lite(), &spec).unwrap().variants.iter().map(|v| (&v.diff, v.distance_from_origin_db)).collect::<Vec<_>>()).unwrap());
        let (four, _) = parallel_on(4, 1, Budget::default(), |_, _| serde_json::to_string(&explore(&patches[0], &Scenario::lite(), &spec).unwrap().variants.iter().map(|v| (&v.diff, v.distance_from_origin_db)).collect::<Vec<_>>()).unwrap());
        assert_eq!(one[0], four[0]);
    }

    /// The distance's scale, measured (design note §2): what "the same"
    /// and "different" read as. The numbers are asserted loosely and
    /// printed exactly; the note quotes them.
    #[test]
    fn distance_scale() {
        use super::distance::{distance, Options};
        let sr = crate::session::SAMPLE_RATE;
        let sc = Scenario::faithful();
        let mut session = session();
        let saw = saw_patch();
        let r = |session: &mut Session, p: &Preset, seed: u32| render(session, p, &sc, seed).unwrap().samples;
        let a = r(&mut session, &saw, 1);
        let d = |x: &[f32], y: &[f32]| distance(x, y, sr, Options::default()).total_db;

        // 1. The same patch, two seeds, a random LFO on the cutoff.
        let mut random = saw.clone();
        random.settings.modulations.push(spinwave_params::preset::ModulationConnection { source: "random_1".into(), destination: "filter_1_cutoff".into(), ..Default::default() });
        random.settings.values.insert("modulation_1_amount".into(), 0.3.into());
        let two_seeds = d(&r(&mut session, &random, 1), &r(&mut session, &random, 2));
        // 2. A −1 dB copy.
        let mut quieter = saw.clone();
        quieter.settings.values.insert("osc_1_level".into(), (0.7 * 10f64.powf(-1.0 / 40.0)).into());
        let minus_one_db = d(&a, &r(&mut session, &quieter, 1));
        // 3. Saw against square.
        let mut square = saw.clone();
        square.settings.values.insert("osc_1_wave_frame".into(), 191.0.into());
        let saw_vs_square = d(&a, &r(&mut session, &square, 1));
        // 4. Level: the same comparison 20 dB down must read the same.
        let mut a20 = saw.clone();
        a20.settings.values.insert("volume".into(), 3000.0.into());
        let mut sq20 = square.clone();
        sq20.settings.values.insert("volume".into(), 3000.0.into());
        let saw_vs_square_20db_down = d(&r(&mut session, &a20, 1), &r(&mut session, &sq20, 1));
        // 5. Phase: random_phase on, two seeds — perceptually identical,
        // sample by sample different.
        let mut phased = saw.clone();
        phased.settings.values.insert("osc_1_random_phase".into(), 1.0.into());
        let pa = r(&mut session, &phased, 1);
        let pb = r(&mut session, &phased, 2);
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let waveform_rms = rms(&pa.iter().zip(&pb).map(|(x, y)| x - y).collect::<Vec<_>>()) / rms(&pa);
        let random_phase = d(&pa, &pb);

        println!("distance scale (dB): two seeds of a 30% random LFO {two_seeds:.3}, -1 dB {minus_one_db:.3}, saw/square {saw_vs_square:.3}, saw/square 20 dB down {saw_vs_square_20db_down:.3}, random phase {random_phase:.3} (relative waveform RMS {waveform_rms:.3})");
        assert!(random_phase < 0.3, "a phase change must be near zero: {random_phase}");
        assert!(waveform_rms > 0.1, "the waveforms did differ (relative RMS): {waveform_rms}");
        assert!((minus_one_db - 1.0).abs() < 0.3, "a −1 dB copy reads about 1 dB: {minus_one_db}");
        assert!(saw_vs_square > 3.0 * minus_one_db, "a timbre change dwarfs a level change: {saw_vs_square}");
        assert!((saw_vs_square_20db_down - saw_vs_square).abs() < 0.15 * saw_vs_square, "the floor holds at −20 dB: {saw_vs_square_20db_down} vs {saw_vs_square}");
    }

    /// The engine reuse in `Session::render_samples_probed` is only
    /// allowed because of this: a patch rendered on a session that just
    /// rendered something else (every effect on, buses, modulation,
    /// portamento, a different polyphony) gives the same bytes as on a
    /// fresh session. Fails on the first bit that differs.
    #[test]
    fn a_recycled_engine_renders_the_same_bytes_as_a_fresh_one() {
        let mut heavy = saw_patch();
        for (k, v) in [
            ("chorus_on", 1.0), ("delay_on", 1.0), ("reverb_on", 1.0), ("distortion_on", 1.0), ("distortion_mix", 0.7),
            ("phaser_on", 1.0), ("flanger_on", 1.0), ("eq_on", 1.0), ("filter_fx_on", 1.0), ("compressor_on", 1.0),
            ("bus_a_on", 1.0), ("bus_a_reverb_on", 1.0), ("osc_2_on", 1.0), ("osc_2_destination", 5.0),
            ("filter_2_on", 1.0), ("filter_2_filter_input", 1.0), ("osc_1_unison_voices", 4.0),
            ("portamento_time", -1.0), ("polyphony", 4.0), ("osc_1_random_phase", 1.0),
        ] {
            heavy.settings.values.insert(k.into(), serde_json::Value::from(v));
        }
        heavy.settings.modulations.push(spinwave_params::preset::ModulationConnection { source: "random_1".into(), destination: "filter_1_cutoff".into(), ..Default::default() });
        heavy.settings.values.insert("modulation_1_amount".into(), 0.5.into());
        heavy.settings.modulations.push(spinwave_params::preset::ModulationConnection { source: "lfo_1".into(), destination: "reverb_dry_wet".into(), ..Default::default() });
        heavy.settings.values.insert("modulation_2_amount".into(), 0.5.into());
        let light = saw_patch();
        let two_notes = Scenario { notes: vec![
            NoteSpec { note: 48, start: 0.0, duration: 0.5, velocity: 0.9, channel: 0 },
            NoteSpec { note: 60, start: 0.3, duration: 0.5, velocity: 0.7, channel: 0 },
        ], seconds: 1.2, bpm: 100.0, mode: RenderMode::Faithful };

        let fresh = |p: &Preset, sc: &Scenario| render(&mut session(), p, sc, 5).unwrap().samples;
        let mut reused = session();
        let after = |s: &mut Session, warm: &Preset, warm_sc: &Scenario, p: &Preset, sc: &Scenario| {
            let _ = render(s, warm, warm_sc, 7).unwrap();
            render(s, p, sc, 5).unwrap().samples
        };
        let cases: [(&str, &Preset, &Scenario, &Preset, &Scenario); 4] = [
            ("light after heavy", &heavy, &two_notes, &light, &Scenario::lite()),
            ("heavy after light", &light, &Scenario::lite(), &heavy, &two_notes),
            ("heavy after heavy", &heavy, &two_notes, &heavy, &two_notes),
            ("light after light (lite after faithful)", &light, &Scenario::faithful(), &light, &Scenario::lite()),
        ];
        for (name, warm, warm_sc, p, sc) in cases {
            let a = fresh(p, sc);
            let b = after(&mut reused, warm, warm_sc, p, sc);
            assert_eq!(a.len(), b.len(), "{name}: lengths");
            if let Some(i) = a.iter().zip(&b).position(|(x, y)| x.to_bits() != y.to_bits()) {
                panic!("{name}: differs at sample {i}: fresh {} recycled {}", a[i], b[i]);
            }
        }
    }

    #[test]
    fn render_seed_depends_on_index_not_on_order() {
        assert_ne!(render_seed(7, 0), render_seed(7, 1));
        assert_eq!(render_seed(7, 3), render_seed(7, 3));
        assert_ne!(render_seed(7, 3), render_seed(8, 3));
    }
}
