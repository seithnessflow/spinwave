//! Random-patch stress testing: build patches from the whole parameter
//! space, render them, and judge what comes out.
//!
//! Hand-written tests only visit the parameter combinations someone
//! thought of. A synth's ugly failures, a reachable panic or a NaN that
//! poisons every later sample, live in the combinations nobody would
//! write down: the first run of this module crashed the engine. This
//! module walks the table itself, so every parameter the engine reads is
//! reachable, including the Spinwave-only namespace.
//!
//! Every patch comes from a seed, so a failure replays exactly: the report
//! names the seed, and [`patch_for_seed`] rebuilds it.

use serde_json::{Map, Value};
use spinwave_params::{parameters, ParamDetails, Preset};

use crate::analysis::analyze;
use crate::session::{NoteSpec, Session};

/// Deterministic, portable RNG (xoshiro-style splitmix): the corpus must
/// replay identically on any machine, which rules out `HashMap` iteration
/// order and platform RNGs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // Avoid the zero state, which splitmix maps to a poor sequence.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + self.unit() * (high - low)
    }

    pub fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() % bound as u64) as usize
        }
    }

    /// True with probability `chance`.
    pub fn chance(&mut self, chance: f32) -> bool {
        self.unit() < chance
    }
}

/// Parameters the generator must not touch, because randomising them
/// tests the harness rather than the synth: polyphony and oversampling
/// change cost rather than sound, and the master volume would just mask
/// every level check behind a random gain.
fn is_excluded(name: &str) -> bool {
    matches!(name, "polyphony" | "oversampling" | "volume" | "beats_per_minute" | "bpm")
}

/// A value for one parameter. Discrete parameters land exactly on an
/// index; continuous ones are uniform over the engine range.
fn random_value(details: &ParamDetails, rng: &mut Rng) -> f32 {
    if details.is_boolean() {
        return if rng.chance(0.5) { 1.0 } else { 0.0 };
    }
    if details.is_discrete() {
        let steps = (details.max - details.min).max(0.0) as usize + 1;
        return details.min + rng.below(steps) as f32;
    }
    rng.range(details.min, details.max)
}

/// How adventurous a patch is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wildness {
    /// Every parameter randomised. Finds the ugly corners; most patches
    /// sound like noise, which is the point.
    Full,
    /// Randomise a subset and leave the rest at its default, which
    /// produces patches closer to something a person would build.
    Sparse,
}

/// Builds a random preset. The result always makes sound: one oscillator
/// and the amplitude envelope are forced on, so a silent render is a
/// finding rather than an accident of the dice.
pub fn patch_for_seed(seed: u64, wildness: Wildness) -> Preset {
    let mut rng = Rng::new(seed);
    let table = parameters();
    let mut settings = Map::new();

    for details in table.iter() {
        if is_excluded(&details.name) {
            continue;
        }
        let touch = match wildness {
            Wildness::Full => true,
            Wildness::Sparse => rng.chance(0.25),
        };
        if !touch {
            continue;
        }
        let value = random_value(details, &mut rng);
        settings.insert(details.name.clone(), Value::from(value));
    }

    // Guarantee an audible path: a wavetable oscillator at a real level,
    // routed to the filters, with an envelope that opens and sustains.
    settings.insert("osc_1_on".into(), Value::from(1.0));
    settings.insert("osc_1_engine".into(), Value::from(0.0));
    settings.insert("osc_1_level".into(), Value::from(rng.range(0.4, 1.0)));
    settings.insert("osc_1_wave_frame".into(), Value::from(rng.range(0.0, 255.0)));
    settings.insert("osc_1_destination".into(), Value::from(0.0));
    settings.insert("env_1_attack".into(), Value::from(rng.range(0.0, 0.4)));
    settings.insert("env_1_decay".into(), Value::from(rng.range(0.2, 1.0)));
    settings.insert("env_1_sustain".into(), Value::from(rng.range(0.5, 1.0)));
    settings.insert("env_1_release".into(), Value::from(rng.range(0.1, 0.6)));

    let mut preset = Preset {
        preset_name: format!("fuzz-{seed}"),
        ..Default::default()
    };
    preset.settings.values = settings;
    preset
}

/// One thing that went wrong.
///
/// The thresholds are deliberately conservative. A random patch is allowed
/// to be loud, ugly, and to ring for a long time: those are sounds, not
/// bugs. Only what no parameter setting should be able to cause counts as
/// fatal, because a fuzzer that cries wolf gets ignored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Defect {
    /// A NaN or infinity reached the output. Always a bug: once one
    /// appears it usually poisons every later sample.
    NonFinite,
    /// The level never fell after every note was released. Usually not a
    /// bug: a resonant filter past its threshold self-oscillates and rings
    /// forever at an amplitude its own saturator sets, which is what an
    /// analog filter does and what the reference does too. Reported
    /// because a patch that drones after every note is worth knowing
    /// about, and because a genuine runaway would look the same at first.
    NeverDecays,
    /// A steady offset survived the master blocker.
    DcOffset,
    /// The render was slower than the audio it produced, which points at
    /// denormals rather than an honestly expensive patch.
    Sluggish,
    /// The signal was louder than the output can carry, so the clamp had
    /// to hold it. Informational: the clamp exists for this, and a random
    /// patch with everything at maximum is expected to reach it.
    ExceedsFullScale,
}

impl Defect {
    /// Whether this makes the render unusable, as opposed to merely ugly
    /// or surprising. Only a NaN qualifies: every other finding here has a
    /// legitimate cause reachable from the parameter space.
    pub fn is_fatal(&self) -> bool {
        matches!(self, Defect::NonFinite)
    }

    pub fn describe(&self) -> &'static str {
        match self {
            Defect::NonFinite => "non-finite samples reached the output",
            Defect::NeverDecays => "level never fell after every note was released",
            Defect::DcOffset => "steady DC offset survived the master blocker",
            Defect::Sluggish => "render slower than realtime: suspect denormals",
            Defect::ExceedsFullScale => "louder than full scale: held by the clamp",
        }
    }
}

/// What one random patch did.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub seed: u64,
    pub defects: Vec<Defect>,
    pub peak: f32,
    pub rms_db: f32,
    pub dc: f32,
    /// Audio seconds rendered per second of CPU.
    pub realtime_factor: f32,
    /// True when the patch made no sound at all, which is suspicious but
    /// reachable honestly (a closed filter, a zeroed modulation).
    pub silent: bool,
}

impl Verdict {
    pub fn is_clean(&self) -> bool {
        self.defects.is_empty()
    }

    pub fn has_fatal(&self) -> bool {
        self.defects.iter().any(Defect::is_fatal)
    }
}

/// The clamp the engine applies on its way out; sitting on it means the
/// signal upstream was louder than the output can carry.
const OUTPUT_CLAMP: f32 = 2.1;

/// Renders one seeded patch and judges it.
pub fn run_seed(seed: u64, wildness: Wildness, seconds: f32) -> Verdict {
    let preset = patch_for_seed(seed, wildness);
    let mut session = Session::with_output_dir(std::env::temp_dir());
    let text = preset.to_json().unwrap_or_default();
    // A generated patch is always loadable; a parse failure is the
    // generator's bug, not the engine's, and shows up as silence.
    let _ = session.load_preset_json(&text);

    // Notes are released a quarter of the way in, leaving three quarters
    // of tail: long enough to tell a slow release apart from something
    // that never stops.
    let notes = vec![
        NoteSpec { note: 45, velocity: 0.9, start: 0.0, duration: seconds * 0.25, channel: 0 },
        NoteSpec { note: 57, velocity: 0.8, start: 0.02, duration: seconds * 0.22, channel: 0 },
        NoteSpec { note: 64, velocity: 0.7, start: 0.04, duration: seconds * 0.2, channel: 0 },
    ];

    let start = std::time::Instant::now();
    let stereo = session.render_samples(&notes, seconds, 120.0);
    let elapsed = start.elapsed().as_secs_f32().max(1e-6);
    let realtime_factor = seconds / elapsed;

    judge(seed, &stereo, session.sample_rate(), seconds, realtime_factor)
}

/// The checks themselves, separated from rendering so they can be run
/// against any buffer (including a preset's render, not just a fuzz one).
pub fn judge(
    seed: u64,
    stereo: &[f32],
    sample_rate: u32,
    seconds: f32,
    realtime_factor: f32,
) -> Verdict {
    let mut defects = Vec::new();

    let non_finite = stereo.iter().any(|v| !v.is_finite());
    if non_finite {
        defects.push(Defect::NonFinite);
    }

    // Everything below needs finite numbers to mean anything.
    let finite: Vec<f32> = stereo.iter().copied().filter(|v| v.is_finite()).collect();
    let peak = finite.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let mean_square = if finite.is_empty() {
        0.0
    } else {
        finite.iter().map(|v| v * v).sum::<f32>() / finite.len() as f32
    };
    let rms_db = if mean_square > 0.0 { 10.0 * mean_square.log10() } else { -180.0 };
    let dc = if finite.is_empty() {
        0.0
    } else {
        finite.iter().sum::<f32>() / finite.len() as f32
    };

    // Sitting on the clamp means the patch is louder than the output can
    // carry. Worth reporting, not a defect: the clamp is there for it.
    let pinned = finite.iter().filter(|v| v.abs() >= OUTPUT_CLAMP - 1e-3).count();
    if !finite.is_empty() && pinned as f32 / finite.len() as f32 > 0.02 {
        defects.push(Defect::ExceedsFullScale);
    }

    // Self-oscillation versus a long tail. Both keep ringing after the
    // last note, so comparing against the peak cannot tell them apart: a
    // big reverb is still near its peak seconds later. What separates
    // them is the SLOPE. Anything decaying, however slowly, is quieter at
    // the end of the tail than earlier in it; something oscillating on its
    // own is just as loud. Both windows sit after every note is released.
    let frames = finite.len() / 2;
    if frames > 200 {
        let window = (frames / 12).max(1);
        let rms_of = |from: usize| -> f32 {
            let start = (from * 2).min(finite.len());
            let end = ((from + window) * 2).min(finite.len());
            let slice = &finite[start..end];
            if slice.is_empty() {
                return 0.0;
            }
            (slice.iter().map(|v| v * v).sum::<f32>() / slice.len() as f32).sqrt()
        };
        // Half the render apart, both after every note is released. A slow
        // release or a long reverb still loses several dB over that span;
        // only a self-sustaining loop holds its level.
        let early_tail = rms_of(frames / 2);
        let late_tail = rms_of(frames - window);
        if late_tail > 1e-3 && late_tail > early_tail * 0.9 {
            defects.push(Defect::NeverDecays);
        }
    }

    if dc.abs() > 0.02 {
        defects.push(Defect::DcOffset);
    }

    // Denormals show up as a render slower than the audio it produced. A
    // legitimately heavy patch still beats realtime by a wide margin on
    // any machine that can host the synth at all.
    if realtime_factor < 1.0 {
        defects.push(Defect::Sluggish);
    }

    let silent = peak < 1e-5;
    let _ = (sample_rate, seconds);
    Verdict { seed, defects, peak, rms_db, dc, realtime_factor, silent }
}

/// Runs a range of seeds and returns every verdict.
pub fn run_range(first_seed: u64, count: usize, wildness: Wildness, seconds: f32) -> Vec<Verdict> {
    (0..count as u64).map(|i| run_seed(first_seed + i, wildness, seconds)).collect()
}

/// A one-line summary of a batch, for the command line.
pub fn summarize(verdicts: &[Verdict]) -> String {
    let total = verdicts.len();
    let clean = verdicts.iter().filter(|v| v.is_clean()).count();
    let silent = verdicts.iter().filter(|v| v.silent).count();
    let fatal = verdicts.iter().filter(|v| v.has_fatal()).count();
    format!("{total} patches: {clean} clean, {silent} silent, {fatal} with a fatal defect")
}

/// Renders one patch and analyses it the way a listener would, for the
/// cases a verdict flags and a person then wants to hear.
pub fn describe_seed(seed: u64, wildness: Wildness, seconds: f32) -> String {
    let preset = patch_for_seed(seed, wildness);
    let mut session = Session::with_output_dir(std::env::temp_dir());
    let _ = session.load_preset_json(&preset.to_json().unwrap_or_default());
    let notes = vec![NoteSpec {
        note: 45,
        velocity: 0.9,
        start: 0.0,
        duration: seconds * 0.6,
        channel: 0,
    }];
    let stereo = session.render_samples(&notes, seconds, 120.0);
    let analysis = analyze(&stereo, session.sample_rate());
    format!(
        "seed {seed}: peak {:.3}, rms {:.1} dB, centroid {:.0} Hz, flatness {:.3}",
        analysis.peak,
        analysis.rms_db,
        analysis.spectral_centroid_hz,
        analysis.texture.spectral_flatness
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_builds_the_same_patch() {
        let a = patch_for_seed(7, Wildness::Full);
        let b = patch_for_seed(7, Wildness::Full);
        assert_eq!(a.settings.values, b.settings.values);
        let c = patch_for_seed(8, Wildness::Full);
        assert_ne!(a.settings.values, c.settings.values);
    }

    /// The standing guarantee: no parameter combination may produce a NaN,
    /// an infinity, or a panic. A fixed seed range keeps this reproducible;
    /// widen it with `spinwave-cli fuzz` when hunting.
    #[test]
    fn no_random_patch_produces_a_non_finite_sample() {
        // Small on purpose: this runs in debug on every `cargo test`. The
        // real sweep is `spinwave-cli fuzz --count 500`, which is where a
        // wide hunt belongs.
        let verdicts = run_range(0, 4, Wildness::Full, 1.5);
        let broken: Vec<u64> = verdicts
            .iter()
            .filter(|v| v.defects.contains(&Defect::NonFinite))
            .map(|v| v.seed)
            .collect();
        assert!(broken.is_empty(), "non-finite output from seeds {broken:?}");
        // A generator that made nothing but silence would pass the check
        // above while testing nothing at all.
        let audible = verdicts.iter().filter(|v| !v.silent).count();
        assert!(audible > 0, "every patch was silent: the generator stopped exercising the engine");
    }

    #[test]
    fn generated_patches_stay_inside_every_parameter_range() {
        let table = parameters();
        for seed in 0..40u64 {
            let preset = patch_for_seed(seed, Wildness::Full);
            for (name, value) in &preset.settings.values {
                let Some(details) = table.lookup(name) else { continue };
                let value = value.as_f64().unwrap_or_default() as f32;
                assert!(
                    value >= details.min - 1e-3 && value <= details.max + 1e-3,
                    "seed {seed}: {name} = {value} outside [{}, {}]",
                    details.min,
                    details.max
                );
            }
        }
    }
}
