//! The judge for the ten-sounds test: twelve targets, each a description
//! a person would give plus a criterion a program can measure.
//!
//! Every target carries its own render recipe (which notes, how loud, how
//! long) and its own thresholds. The thresholds live HERE and nowhere
//! else: the model that writes the patch sees the description and never
//! the numbers, otherwise the test measures its ability to hit a figure
//! rather than to understand a sentence. See `notes/ten-sounds-protocol.md`.
//!
//! The judge self-tests before it judges: a render that peaks below
//! -60 dBFS is refused, not scored, because two silent renders would
//! satisfy half these criteria for the wrong reason.

use serde::Serialize;
use serde_json::Value as Json;
use spinwave_params::Preset;

use crate::analysis::{analyze, Analysis};
use crate::session::{NoteSpec, Session, SAMPLE_RATE};

/// -60 dBFS: below this a render is a broken measurement, not a sound.
const AUDIBLE: f32 = 1.0e-3;

/// The twelve targets. Ten sounds and two controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    SubBass,
    Pluck,
    Pad,
    FmBell,
    Lead,
    FilterSweepUp,
    VelocityDarkSoft,
    KeytrackBrightHigh,
    NoiseRiser,
    TempoWobble,
    /// Control: describe an existing preset in words, measure the distance.
    Reconstruct,
    /// Control: "darker, softer attack" on an existing patch; the diff must
    /// be small and go the right way.
    Edit,
}

impl Target {
    pub const ALL: [Target; 12] = [
        Target::SubBass,
        Target::Pluck,
        Target::Pad,
        Target::FmBell,
        Target::Lead,
        Target::FilterSweepUp,
        Target::VelocityDarkSoft,
        Target::KeytrackBrightHigh,
        Target::NoiseRiser,
        Target::TempoWobble,
        Target::Reconstruct,
        Target::Edit,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Target::SubBass => "sub_bass",
            Target::Pluck => "pluck",
            Target::Pad => "pad",
            Target::FmBell => "fm_bell",
            Target::Lead => "lead",
            Target::FilterSweepUp => "filter_sweep_up",
            Target::VelocityDarkSoft => "velocity_dark_soft",
            Target::KeytrackBrightHigh => "keytrack_bright_high",
            Target::NoiseRiser => "noise_riser",
            Target::TempoWobble => "tempo_wobble",
            Target::Reconstruct => "reconstruct",
            Target::Edit => "edit",
        }
    }

    pub fn from_id(id: &str) -> Option<Target> {
        Target::ALL.into_iter().find(|t| t.id() == id)
    }

    /// The description the model is given. No numbers from the criteria.
    pub fn description(self) -> &'static str {
        match self {
            Target::SubBass => "A sub bass with body: a deep fundamental you feel more than hear, a touch of warmth from the first harmonics, and nothing bright on top. Played low.",
            Target::Pluck => "A short plucked sound that dies away quickly after each note, like a muted string.",
            Target::Pad => "A slow, wide pad that swells in gently and sits in stereo without losing its body when summed to mono.",
            Target::FmBell => "A bell made with FM: a metallic, inharmonic clang with a fast strike and a ringing decay.",
            Target::Lead => "A cutting lead that carries in a mix: its energy sits in the upper mids, where a melody is heard.",
            Target::FilterSweepUp => "A sound whose filter opens over the course of a held note, going from dark to bright.",
            Target::VelocityDarkSoft => "A sound that responds to how hard you play: a soft note is darker, a hard note is brighter.",
            Target::KeytrackBrightHigh => "A sound that follows the keyboard: notes high up are noticeably brighter than notes low down.",
            Target::NoiseRiser => "A filtered noise riser: noise, not a tone, that sweeps upward in brightness over a few seconds.",
            Target::TempoWobble => "A timbre that evolves in a cycle locked to the tempo, one full cycle every eight beats at 120 BPM.",
            Target::Reconstruct => "Rebuild the described preset as closely as you can.",
            Target::Edit => "Make this patch darker, with a softer attack. Change as little as possible.",
        }
    }
}

/// One measured criterion.
#[derive(Clone, Debug, Serialize)]
pub struct Check {
    pub name: String,
    pub measured: f32,
    pub threshold: f32,
    pub pass: bool,
}

/// The judge's verdict on one patch for one target.
#[derive(Clone, Debug, Serialize)]
pub struct Verdict {
    pub target: Target,
    pub pass: bool,
    pub checks: Vec<Check>,
    /// The analysis of the main render, for the loop condition: the model
    /// gets this, never the checks.
    pub analysis: Analysis,
}

fn render(preset: &Preset, notes: &[NoteSpec], seconds: f32) -> Result<Vec<f32>, String> {
    let mut session = Session::with_output_dir(std::env::temp_dir());
    session.load_preset_json(&preset.to_json().map_err(|e| e.to_string())?)?;
    let samples = session.render_samples(notes, seconds, 120.0);
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    if !samples.iter().all(|s| s.is_finite()) {
        return Err("the render is not finite; refusing to judge it".into());
    }
    if peak < AUDIBLE {
        return Err(format!("the render peaks at {peak:.2e}, below -60 dBFS; refusing to judge silence"));
    }
    Ok(samples)
}

fn note(midi: i32, velocity: f32, duration: f32) -> NoteSpec {
    NoteSpec { note: midi, velocity, start: 0.0, duration, channel: 0 }
}

fn check(name: &str, measured: f32, threshold: f32, pass: bool) -> Check {
    Check { name: name.into(), measured, threshold, pass }
}

/// Level in dB relative to the peak, `seconds` after the peak.
fn level_after_peak_db(samples: &[f32], seconds: f32) -> f32 {
    let frames = samples.len() / 2;
    let mono = |i: usize| 0.5 * (samples[2 * i] + samples[2 * i + 1]);
    let (mut peak, mut at) = (0.0f32, 0usize);
    for i in 0..frames {
        let v = mono(i).abs();
        if v > peak {
            peak = v;
            at = i;
        }
    }
    let start = (at + (seconds * SAMPLE_RATE as f32) as usize).min(frames.saturating_sub(1));
    let window = (0.02 * SAMPLE_RATE as f32) as usize;
    let end = (start + window).min(frames);
    let rms = (start..end).map(|i| mono(i) * mono(i)).sum::<f32>() / (end - start).max(1) as f32;
    20.0 * (rms.sqrt() / peak.max(1e-9)).log10()
}

/// RMS of the mono sum against the RMS of the stereo signal, in dB. Near
/// zero means nothing cancels; strongly negative means the width is fake.
fn mono_collapse_db(samples: &[f32]) -> f32 {
    let mut stereo = 0.0f64;
    let mut mono = 0.0f64;
    for frame in samples.chunks_exact(2) {
        let (l, r) = (frame[0] as f64, frame[1] as f64);
        stereo += 0.5 * (l * l + r * r);
        let m = 0.5 * (l + r);
        mono += m * m;
    }
    (10.0 * (mono / stereo.max(1e-18)).log10()) as f32
}

fn centroid_of(samples: &[f32]) -> f32 {
    analyze(samples, SAMPLE_RATE).spectral_centroid_hz
}

/// Judges a patch against a target. `reference` is required for the two
/// controls: the preset being reconstructed, or the patch being edited.
pub fn judge(preset: &Preset, target: Target, reference: Option<&Preset>) -> Result<Verdict, String> {
    let mut checks = Vec::new();
    let analysis;

    match target {
        Target::SubBass => {
            // The judge plays C2 (65 Hz); what the patch can get wrong is
            // to transpose up or to add harmonics. The autocorrelation
            // pitch detector returns None on a pure low sine, so the
            // fundamental is judged by where the energy sits, not by it.
            let s = render(preset, &[note(36, 0.9, 2.0)], 2.5)?;
            analysis = analyze(&s, SAMPLE_RATE);
            checks.push(check("centroid_below_150hz", analysis.spectral_centroid_hz, 150.0, analysis.spectral_centroid_hz < 150.0));
            checks.push(check("rolloff_below_2khz", analysis.spectral_rolloff_hz, 2000.0, analysis.spectral_rolloff_hz < 2000.0));
            // Discrimination (2026-09-13): the init patch — a bare sine at
            // the note — passed the two checks above, so the target
            // measured nothing. "Body" is the first harmonics: the
            // rolloff has to clear the fundamental's octave. Init reads
            // 75 Hz at C2; a saw through a low-pass at MIDI 48 reads 194.
            checks.push(check("rolloff_above_100hz", analysis.spectral_rolloff_hz, 100.0, analysis.spectral_rolloff_hz > 100.0));
        }
        Target::Pluck => {
            let s = render(preset, &[note(60, 0.9, 1.5)], 2.0)?;
            analysis = analyze(&s, SAMPLE_RATE);
            let drop = level_after_peak_db(&s, 0.3);
            checks.push(check("drop_300ms_after_peak_db", drop, -20.0, drop <= -20.0));
        }
        Target::Pad => {
            let s = render(preset, &[note(60, 0.8, 4.0)], 5.0)?;
            analysis = analyze(&s, SAMPLE_RATE);
            checks.push(check("attack_seconds", analysis.envelope.attack_seconds, 0.5, analysis.envelope.attack_seconds > 0.5));
            checks.push(check("stereo_width", analysis.stereo_width, 0.3, analysis.stereo_width > 0.3));
            let collapse = mono_collapse_db(&s);
            checks.push(check("mono_sum_within_6db", collapse, -6.0, collapse > -6.0));
        }
        Target::FmBell => {
            let s = render(preset, &[note(72, 0.9, 0.05)], 3.0)?;
            analysis = analyze(&s, SAMPLE_RATE);
            let harmonicity = analysis.texture.harmonicity.unwrap_or(0.0);
            checks.push(check("inharmonic", harmonicity, 0.5, harmonicity < 0.5));
            checks.push(check("attack_seconds", analysis.envelope.attack_seconds, 0.05, analysis.envelope.attack_seconds < 0.05));
            let ring = level_after_peak_db(&s, 1.0);
            checks.push(check("still_ringing_1s_after_peak_db", ring, -40.0, ring > -40.0));
            checks.push(check("dies_by_the_end_db", analysis.envelope.tail_db, -20.0, analysis.envelope.tail_db < -20.0));
        }
        Target::Lead => {
            let s = render(preset, &[note(64, 0.9, 2.0)], 2.5)?;
            analysis = analyze(&s, SAMPLE_RATE);
            let b = &analysis.bands_db;
            let others = [b.sub_0_60, b.bass_60_250, b.low_mid_250_1k, b.high_4k_12k, b.air_12k_up];
            let loudest_other = others.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let margin = b.mid_1k_4k - loudest_other;
            checks.push(check("upper_mids_lead_by_db", margin, -3.0, margin > -3.0));
        }
        Target::FilterSweepUp => {
            let s = render(preset, &[note(48, 0.9, 3.0)], 3.5)?;
            analysis = analyze(&s, SAMPLE_RATE);
            let t = &analysis.movement.centroid_trajectory_hz;
            let (first, last) = (t.first().copied().unwrap_or(1.0), t.get(t.len().saturating_sub(3)).copied().unwrap_or(1.0));
            let ratio = last / first.max(1.0);
            checks.push(check("centroid_end_over_start", ratio, 2.0, ratio > 2.0));
        }
        Target::VelocityDarkSoft => {
            let soft = render(preset, &[note(60, 0.3, 1.5)], 2.0)?;
            let loud = render(preset, &[note(60, 1.0, 1.5)], 2.0)?;
            analysis = analyze(&loud, SAMPLE_RATE);
            let ratio = centroid_of(&soft) / centroid_of(&loud).max(1.0);
            checks.push(check("soft_centroid_over_loud", ratio, 0.7, ratio < 0.7));
        }
        Target::KeytrackBrightHigh => {
            let low = render(preset, &[note(36, 0.9, 1.5)], 2.0)?;
            let high = render(preset, &[note(72, 0.9, 1.5)], 2.0)?;
            analysis = analyze(&high, SAMPLE_RATE);
            // Brighter than the pitch ratio alone would give: three octaves
            // is 8x, and a bare sine's centroid climbs by that (measured
            // 4.6 on the init patch, the low note's centroid being smeared
            // upward by the analysis window), so the threshold sits at 8:
            // the init fails, the keytracked ceiling reads 26.8. It was
            // 4.5 until the discrimination rule (2026-09-13), and the init
            // patch passed it.
            let ratio = centroid_of(&high) / centroid_of(&low).max(1.0);
            checks.push(check("high_centroid_over_low", ratio, 8.0, ratio > 8.0));
        }
        Target::NoiseRiser => {
            let s = render(preset, &[note(60, 0.9, 4.0)], 4.5)?;
            analysis = analyze(&s, SAMPLE_RATE);
            let harmonicity = analysis.texture.harmonicity.unwrap_or(0.0);
            checks.push(check("not_a_tone", harmonicity, 0.3, harmonicity < 0.3));
            checks.push(check("spectral_flatness", analysis.texture.spectral_flatness, 0.2, analysis.texture.spectral_flatness > 0.2));
            let t = &analysis.movement.centroid_trajectory_hz;
            let (first, last) = (t.first().copied().unwrap_or(1.0), t.get(t.len().saturating_sub(3)).copied().unwrap_or(1.0));
            let ratio = last / first.max(1.0);
            checks.push(check("centroid_end_over_start", ratio, 1.5, ratio > 1.5));
        }
        Target::TempoWobble => {
            // Eight beats at 120 BPM is a 4 s cycle: 0.25 Hz.
            let s = render(preset, &[note(48, 0.9, 12.0)], 12.5)?;
            analysis = analyze(&s, SAMPLE_RATE);
            let wanted = 0.25f32;
            let best = analysis
                .movement
                .mod_rates_hz
                .iter()
                .filter(|r| (r.hz / wanted - 1.0).abs() < 0.08)
                .map(|r| r.strength)
                .fold(0.0f32, f32::max);
            checks.push(check("rate_at_0_25hz_strength", best, 0.3, best > 0.3));
        }
        Target::Reconstruct => {
            let truth = reference.ok_or("reconstruct needs --reference <the described preset>")?;
            let ours = render(preset, &[note(57, 0.85, 2.0)], 3.0)?;
            let theirs = render(truth, &[note(57, 0.85, 2.0)], 3.0)?;
            analysis = analyze(&ours, SAMPLE_RATE);
            let target_analysis = analyze(&theirs, SAMPLE_RATE);
            let distance = analysis_distance(&analysis, &target_analysis);
            checks.push(check("analysis_distance", distance, 1.0, distance < 1.0));
            let changed = differing_parameters(preset, truth);
            checks.push(check("parameters_differing", changed as f32, 0.0, true));
        }
        Target::Edit => {
            let origin = reference.ok_or("edit needs --reference <the original patch>")?;
            let ours = render(preset, &[note(57, 0.85, 2.0)], 3.0)?;
            let before = render(origin, &[note(57, 0.85, 2.0)], 3.0)?;
            analysis = analyze(&ours, SAMPLE_RATE);
            let previous = analyze(&before, SAMPLE_RATE);
            let darker = analysis.spectral_centroid_hz / previous.spectral_centroid_hz.max(1.0);
            checks.push(check("centroid_ratio_after_over_before", darker, 0.85, darker < 0.85));
            let softer = analysis.envelope.attack_seconds - previous.envelope.attack_seconds;
            checks.push(check("attack_longer_by_seconds", softer, 0.02, softer > 0.02));
            let changed = differing_parameters(preset, origin);
            checks.push(check("parameters_changed", changed as f32, 8.0, changed <= 8));
        }
    }

    let pass = checks.iter().all(|c| c.pass);
    Ok(Verdict { target, pass, checks, analysis })
}

/// A scale-free distance between two analyses: log ratios of centroid,
/// rolloff and attack, plus band differences in dB over 10. Zero is the
/// same sound; one is clearly a different one.
fn analysis_distance(a: &Analysis, b: &Analysis) -> f32 {
    let log_ratio = |x: f32, y: f32| ((x.max(1.0) / y.max(1.0)).ln()).abs();
    let bands = |x: &Analysis| [x.bands_db.sub_0_60, x.bands_db.bass_60_250, x.bands_db.low_mid_250_1k, x.bands_db.mid_1k_4k, x.bands_db.high_4k_12k, x.bands_db.air_12k_up];
    let band_term: f32 = bands(a).iter().zip(bands(b)).map(|(x, y)| (x - y).abs() / 10.0).sum::<f32>() / 6.0;
    let attack_term = log_ratio(a.envelope.attack_seconds + 0.01, b.envelope.attack_seconds + 0.01) / 2.0;
    log_ratio(a.spectral_centroid_hz, b.spectral_centroid_hz) + log_ratio(a.spectral_rolloff_hz, b.spectral_rolloff_hz) / 2.0 + band_term + attack_term
}

/// How many table parameters hold a different value in the two presets
/// (absent counts as the default), plus connections present in one only.
pub fn differing_parameters(a: &Preset, b: &Preset) -> usize {
    let table = spinwave_params::parameters();
    let get = |p: &Preset, name: &str| p.settings.values.get(name).and_then(Json::as_f64).map(|v| v as f32);
    let mut count = 0;
    for d in table.iter() {
        let x = get(a, &d.name).unwrap_or(d.default_value);
        let y = get(b, &d.name).unwrap_or(d.default_value);
        if x != y {
            count += 1;
        }
    }
    let routes = |p: &Preset| -> Vec<(String, String)> {
        p.settings.modulations.iter().filter(|m| m.is_connected()).map(|m| (m.source.clone(), m.destination.clone())).collect()
    };
    let (ra, rb) = (routes(a), routes(b));
    count += ra.iter().filter(|r| !rb.contains(r)).count() + rb.iter().filter(|r| !ra.contains(r)).count();
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn patch(values: &[(&str, f64)], mods: &[(&str, &str, f64)]) -> Preset {
        let mut p = Preset::default();
        p.settings.values.insert("osc_1_on".into(), json!(1.0));
        for (k, v) in values {
            p.settings.values.insert((*k).into(), json!(v));
        }
        for (i, (s, d, amount)) in mods.iter().enumerate() {
            p.settings.modulations.push(spinwave_params::preset::ModulationConnection { source: (*s).into(), destination: (*d).into(), ..Default::default() });
            p.settings.values.insert(format!("modulation_{}_amount", i + 1), json!(amount));
        }
        p
    }

    #[test]
    fn a_silent_patch_is_refused_not_scored() {
        let silent = patch(&[("osc_1_level", 0.0)], &[]);
        let err = judge(&silent, Target::SubBass, None).unwrap_err();
        assert!(err.contains("refusing"), "{err}");
    }

    #[test]
    fn sub_bass_passes_a_filtered_saw_and_fails_a_bare_sine_and_a_bright_saw() {
        // A saw through a low-pass an octave above the fundamental: body
        // from the first harmonics, nothing on top.
        let sub = patch(&[("osc_1_wave_frame", 128.0), ("osc_1_level", 0.8), ("filter_1_on", 1.0), ("filter_1_cutoff", 48.0), ("filter_1_resonance", 0.3)], &[]);
        let v = judge(&sub, Target::SubBass, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        // Discrimination: the init patch (a bare sine) must FAIL, or the
        // target measures nothing — it passed until 2026-09-13.
        let init = patch(&[], &[]);
        let v = judge(&init, Target::SubBass, None).unwrap();
        assert!(!v.pass, "the init patch is not a sub bass with body: {:#?}", v.checks);
        let saw = patch(&[("osc_1_wave_frame", 128.0)], &[]);
        let v = judge(&saw, Target::SubBass, None).unwrap();
        assert!(!v.pass, "a bare saw is not a sub bass: {:#?}", v.checks);
    }

    #[test]
    fn pluck_passes_a_fast_decay_and_fails_a_sustained_tone() {
        let pluck = patch(&[("osc_1_wave_frame", 128.0), ("env_1_attack", 0.0), ("env_1_decay", 0.55), ("env_1_sustain", 0.0), ("env_1_release", 0.3)], &[]);
        assert!(judge(&pluck, Target::Pluck, None).unwrap().pass);
        let organ = patch(&[("osc_1_wave_frame", 128.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&organ, Target::Pluck, None).unwrap().pass);
    }

    #[test]
    fn pad_needs_a_slow_attack_and_real_width() {
        let pad = patch(
            &[("osc_1_wave_frame", 128.0), ("osc_1_unison_voices", 6.0), ("osc_1_unison_detune", 3.0), ("osc_1_stereo_spread", 1.0), ("env_1_attack", 1.05), ("env_1_sustain", 1.0)],
            &[],
        );
        let v = judge(&pad, Target::Pad, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        let mono_stab = patch(&[("osc_1_wave_frame", 128.0), ("env_1_attack", 0.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&mono_stab, Target::Pad, None).unwrap().pass);
    }

    #[test]
    fn filter_sweep_needs_a_rising_centroid() {
        let sweep = patch(
            &[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 30.0), ("env_1_sustain", 1.0), ("env_2_attack", 1.2), ("env_2_sustain", 1.0)],
            &[("env_2", "filter_1_cutoff", 0.8)],
        );
        let v = judge(&sweep, Target::FilterSweepUp, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        let flat = patch(&[("osc_1_wave_frame", 128.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&flat, Target::FilterSweepUp, None).unwrap().pass);
    }

    #[test]
    fn velocity_and_keytrack_compare_two_renders() {
        let vel = patch(
            &[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 40.0), ("env_1_sustain", 1.0)],
            &[("velocity", "filter_1_cutoff", 0.7)],
        );
        assert!(judge(&vel, Target::VelocityDarkSoft, None).unwrap().pass);
        let key = patch(
            &[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 30.0), ("filter_1_keytrack", 1.0), ("env_1_sustain", 1.0)],
            &[("note", "filter_1_cutoff", 0.9)],
        );
        let v = judge(&key, Target::KeytrackBrightHigh, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        let no_track = patch(&[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 60.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&no_track, Target::VelocityDarkSoft, None).unwrap().pass);
    }

    #[test]
    fn the_edit_control_wants_a_small_diff_in_the_right_direction() {
        let origin = patch(&[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 90.0), ("env_1_attack", 0.0), ("env_1_sustain", 1.0)], &[]);
        let mut good = origin.clone();
        good.settings.values.insert("filter_1_cutoff".into(), json!(60.0));
        good.settings.values.insert("env_1_attack".into(), json!(0.8));
        let v = judge(&good, Target::Edit, Some(&origin)).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        assert_eq!(differing_parameters(&good, &origin), 2);
        // Rewriting everything fails even if the sound goes the right way.
        let mut rewrite = good.clone();
        for i in 1..=12 {
            rewrite.settings.values.insert(format!("lfo_{i}_frequency"), json!(3.0));
        }
        assert!(!judge(&rewrite, Target::Edit, Some(&origin)).unwrap().pass);
    }

    #[test]
    fn fm_bell_lead_noise_and_wobble_have_known_positives() {
        // FM from oscillator A - slot 2 for slot 1 (the reference's
        // wiring: type 7; type 8 is slot 3, off here) - at a non-integer
        // ratio, struck and ringing.
        let bell = patch(
            &[("osc_1_wave_frame", 0.0), ("osc_1_distortion_type", 7.0), ("osc_1_distortion_amount", 0.6),
              ("osc_2_on", 1.0), ("osc_2_wave_frame", 0.0), ("osc_2_level", 0.0), ("osc_2_transpose", 19.0), ("osc_2_tune", 0.3),
              ("env_1_attack", 0.0), ("env_1_decay", 1.4), ("env_1_sustain", 0.0), ("env_1_release", 1.5)],
            &[],
        );
        let v = judge(&bell, Target::FmBell, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        let flute = patch(&[("osc_1_wave_frame", 0.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&flute, Target::FmBell, None).unwrap().pass);

        // A saw through a band-pass parked in the upper mids.
        let lead = patch(&[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_blend", 1.0), ("filter_1_cutoff", 96.0), ("filter_1_resonance", 0.6), ("env_1_sustain", 1.0)], &[]);
        let v = judge(&lead, Target::Lead, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        let sub = patch(&[("osc_1_wave_frame", 0.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&sub, Target::Lead, None).unwrap().pass);

        // The dedicated noise source, low-passed, with the filter opening.
        let riser = patch(
            &[("osc_1_level", 0.0), ("noise_on", 1.0), ("noise_level", 0.8), ("noise_destination", 0.0),
              ("filter_1_on", 1.0), ("filter_1_cutoff", 40.0), ("env_1_sustain", 1.0), ("env_2_attack", 1.3), ("env_2_sustain", 1.0)],
            &[("env_2", "filter_1_cutoff", 0.7)],
        );
        let v = judge(&riser, Target::NoiseRiser, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        let tone = patch(&[("osc_1_wave_frame", 128.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&tone, Target::NoiseRiser, None).unwrap().pass);

        // An LFO synced to 2/1 — two bars, eight beats, 4 s at 120 BPM — on
        // the cutoff. (An LFO is tempo-synced by default, so its frequency
        // knob is ignored; the division is what counts.)
        let wobble = patch(
            &[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 50.0), ("lfo_1_tempo", 5.0), ("env_1_sustain", 1.0)],
            &[("lfo_1", "filter_1_cutoff", 0.6)],
        );
        let v = judge(&wobble, Target::TempoWobble, None).unwrap();
        assert!(v.pass, "{:#?}", v.checks);
        let steady = patch(&[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 50.0), ("env_1_sustain", 1.0)], &[]);
        assert!(!judge(&steady, Target::TempoWobble, None).unwrap().pass);
    }

    #[test]
    fn reconstruct_scores_distance_to_the_truth() {
        let truth = patch(&[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 70.0), ("env_1_sustain", 0.8)], &[]);
        let same = judge(&truth, Target::Reconstruct, Some(&truth)).unwrap();
        assert!(same.pass);
        assert_eq!(same.checks[0].measured, 0.0);
        let far = patch(&[("osc_1_wave_frame", 0.0), ("env_1_attack", 1.2), ("env_1_sustain", 1.0)], &[]);
        let v = judge(&far, Target::Reconstruct, Some(&truth)).unwrap();
        assert!(!v.pass, "{:#?}", v.checks);
    }

    /// The discrimination rule (2026-09-13): every criterion must be
    /// FAILED by the init patch and PASSED by a hand-written patch — both.
    /// A criterion the init passes measures nothing (sub_bass did, until
    /// its body check); one only an impossible patch satisfies is as
    /// useless. The passing half is each target's known positive above;
    /// this is the failing half, on every target, with the reference the
    /// two controls need.
    #[test]
    fn every_target_fails_the_init_patch() {
        let init = patch(&[], &[]);
        let truth = patch(&[("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 70.0), ("env_1_sustain", 0.8)], &[]);
        for target in Target::ALL {
            let reference = match target {
                Target::Reconstruct | Target::Edit => Some(&truth),
                _ => None,
            };
            let verdict = judge(&init, target, reference).unwrap();
            assert!(!verdict.pass, "{}: the init patch passes — the target measures nothing: {:#?}", target.id(), verdict.checks);
        }
    }

    #[test]
    fn every_target_has_a_description_without_its_numbers() {
        for t in Target::ALL {
            let d = t.description();
            assert!(!d.is_empty());
            assert!(!d.contains("dB") && !d.contains("Hz") && !d.contains("ms"), "{}: the description leaks a criterion: {d}", t.id());
            assert_eq!(Target::from_id(t.id()), Some(t));
        }
    }
}
