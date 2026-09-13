//! The Rust half of the golden bench: renders a case through Spinwave and
//! compares it against audio rendered by Vital's own DSP core.
//!
//! Spinwave claims its DSP is a faithful port: same approximations, same
//! sound. Nothing verified that claim. The test suite checks the engine
//! against itself, which is why it stayed green while the LFOs ran at half
//! depth and the detune was four times too narrow. Only a comparison
//! against the reference catches that class of bug, and only a committed
//! comparison stops it coming back.
//!
//! A case file is shared by both halves, so a difference in the audio is a
//! difference in the DSP. The reference renders live in
//! `tools/golden/reference/`, generated once by `tools/golden` (see its
//! README) and committed; running the comparison needs no C++ compiler.

use std::path::{Path, PathBuf};

use spinwave_dsp::wavetable::{WaveFrame, WaveShape};
use spinwave_engine::kernel::ModSource;
use spinwave_params::Preset;

use crate::session::{NoteSpec, Session};

/// One note event in a case.
#[derive(Clone, Copy, Debug)]
pub struct CaseNote {
    pub midi: i32,
    pub velocity: f32,
    pub start_seconds: f32,
    pub hold_seconds: f32,
}

/// A case: what to render, in the exact terms both halves understand.
#[derive(Clone, Debug)]
pub struct Case {
    pub sample_rate: u32,
    pub seconds: f32,
    /// The single-cycle shape loaded into every oscillator's table. Both
    /// halves load one predefined shape rather than building a morphing
    /// wavetable, so the comparison is about the DSP and not about two
    /// different wavetable builders.
    pub shape: WaveShape,
    pub notes: Vec<CaseNote>,
    pub controls: Vec<(String, f32)>,
    /// Modulation connections: `(source, destination, amount)`. Not
    /// controls on either side, so both harnesses wire them separately.
    pub modulations: Vec<(String, String, f32)>,
    /// A flat LFO shape at this value instead of the triangle, from
    /// `lfo_shape flat <v>`. An LFO that does not move separates "the
    /// source varies" from "the source is an LFO", which every earlier
    /// case changed together.
    pub lfo_flat: Option<f32>,
    /// `random_seed <n>`: the seed the voice's `random_1` generator starts
    /// from. Both engines seed each generator from a process-global
    /// counter (`next_seed_++`), so which seed a voice's random LFO holds
    /// depends on how many generators were built before it — a fact about
    /// construction order, not DSP, and different on the two sides. The
    /// reference cannot be told its seed; this pins Spinwave's to the one
    /// the reference happens to use, which `tools/golden/random_seed.py`
    /// recovers from a `--probe random_1` curve.
    pub random_seed: Option<u32>,
    /// Seconds excluded from the front of the comparison.
    ///
    /// This is an escape hatch for DELIBERATE, RECORDED deviations from the
    /// reference, and for nothing else. A case that uses it must say in a
    /// comment which deviation it is covering and where that deviation is
    /// documented in the code. Reaching for it to make a case pass is how a
    /// golden bench stops being one.
    pub skip_seconds: f32,
}

impl Default for Case {
    fn default() -> Case {
        Case {
            sample_rate: 44100,
            seconds: 1.0,
            shape: WaveShape::Saw,
            notes: Vec::new(),
            controls: Vec::new(),
            modulations: Vec::new(),
            lfo_flat: None,
            random_seed: None,
            skip_seconds: 0.0,
        }
    }
}

fn parse_shape(name: &str) -> Result<WaveShape, String> {
    match name {
        "sin" => Ok(WaveShape::Sin),
        "triangle" => Ok(WaveShape::Triangle),
        "square" => Ok(WaveShape::Square),
        "pulse" => Ok(WaveShape::Pulse),
        "saw" => Ok(WaveShape::Saw),
        other => Err(format!("unknown wave shape '{other}'")),
    }
}

impl Case {
    /// Parses the shared case format. The C++ harness parses the same
    /// text; keep the two readers in step.
    pub fn parse(text: &str) -> Result<Case, String> {
        let mut case = Case::default();
        for (number, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("");
            let mut words = line.split_whitespace();
            let Some(directive) = words.next() else { continue };
            let number = number + 1;
            let next = |words: &mut std::str::SplitWhitespace| -> Result<f32, String> {
                words
                    .next()
                    .ok_or_else(|| format!("line {number}: missing value"))?
                    .parse()
                    .map_err(|_| format!("line {number}: not a number"))
            };

            match directive {
                "rate" => case.sample_rate = next(&mut words)? as u32,
                "seconds" => case.seconds = next(&mut words)?,
                "skip" => case.skip_seconds = next(&mut words)?,
                "wave" => {
                    let name = words.next().unwrap_or("");
                    case.shape = parse_shape(name).map_err(|e| format!("line {number}: {e}"))?;
                }
                "note" => {
                    case.notes.push(CaseNote {
                        midi: next(&mut words)? as i32,
                        velocity: next(&mut words)?,
                        start_seconds: next(&mut words)?,
                        hold_seconds: next(&mut words)?,
                    });
                }
                "random_seed" => {
                    let seed: f32 = next(&mut words)?;
                    if seed < 0.0 || seed.fract() != 0.0 {
                        return Err(format!("line {number}: random_seed must be a whole number"));
                    }
                    case.random_seed = Some(seed as u32);
                }
                "lfo_shape" => {
                    let kind = words.next().unwrap_or("");
                    if kind != "flat" {
                        return Err(format!("line {number}: unknown lfo shape '{kind}'"));
                    }
                    case.lfo_flat = Some(next(&mut words)?);
                }
                "modulate" => {
                    let source = words
                        .next()
                        .ok_or_else(|| format!("line {number}: missing source"))?;
                    let destination = words
                        .next()
                        .ok_or_else(|| format!("line {number}: missing destination"))?;
                    case.modulations.push((
                        source.to_string(),
                        destination.to_string(),
                        next(&mut words)?,
                    ));
                }
                "set" => {
                    let name = words
                        .next()
                        .ok_or_else(|| format!("line {number}: missing control name"))?;
                    case.controls.push((name.to_string(), next(&mut words)?));
                }
                other => return Err(format!("line {number}: unknown directive '{other}'")),
            }
        }
        Ok(case)
    }

    pub fn read(path: &Path) -> Result<Case, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Case::parse(&text)
    }

    /// Renders the case through Spinwave, returning interleaved stereo.
    pub fn render(&self) -> Result<Vec<f32>, String> {
        self.render_probed(&[]).map(|(samples, _)| samples)
    }

    /// Renders and returns the values that left their range on the way
    /// (`bounds`): a case with any is measuring a clamp, not what it
    /// says, unless the corpus test's allowlist says the clamp is the
    /// point.
    pub fn render_checked(&self) -> Result<(Vec<f32>, Vec<crate::bounds::Excursion>), String> {
        self.render_full(&[]).map(|(samples, _, excursions)| (samples, excursions))
    }

    /// Renders, and reads back the control-rate value of each named
    /// modulation source once per block.
    ///
    /// Comparing the audio says a case diverges; comparing the modulation
    /// curve says where. The reference harness writes the same readings
    /// from Vital's own sources (`vital_golden case out.raw --probe lfo_1`),
    /// and the shape of the difference names the mechanism: an exponential
    /// means a one-pole smoother whose coefficient can be read off, a
    /// linear ramp per block means a buffer interpolation, a step one block
    /// late means a reset that fires at the wrong time.
    pub fn render_probed(
        &self,
        probes: &[ModSource],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>), String> {
        self.render_full(probes).map(|(samples, curves, _)| (samples, curves))
    }

    #[allow(clippy::type_complexity)]
    fn render_full(
        &self,
        probes: &[ModSource],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<crate::bounds::Excursion>), String> {
        let mut preset = Preset::default();
        for (name, value) in &self.controls {
            preset.settings.values.insert(name.clone(), (*value).into());
        }
        // A preset carries its connections in `settings.modulations`, and
        // their amounts in `modulation_N_amount` controls, which is how the
        // reference's bank hands out slots in order too.
        for (index, (source, destination, amount)) in self.modulations.iter().enumerate() {
            preset.settings.modulations.push(spinwave_params::preset::ModulationConnection {
                source: source.clone(),
                destination: destination.clone(),
                ..Default::default()
            });
            preset
                .settings
                .values
                .insert(format!("modulation_{}_amount", index + 1), (*amount).into());
        }

        // The y axis is inverted (a value of 0 is drawn at 1.0), so a
        // flat line at v sits at 1 - v. Every LFO gets it; the cases that
        // use this have one connection.
        if let Some(value) = self.lfo_flat {
            let y = 1.0 - value;
            preset.settings.lfos = (0..12)
                .map(|_| spinwave_params::preset::LineShape {
                    num_points: 2,
                    points: vec![0.0, y, 1.0, y],
                    powers: vec![0.0, 0.0],
                    name: Some("Flat".into()),
                    smooth: false,
                    ..Default::default()
                })
                .collect();
        }

        let mut session = Session::with_output_dir(std::env::temp_dir());
        // Spinwave blocks DC, per voice and on the master output; the
        // reference wires no DC filter at all, though it ships the class.
        // Both are deliberate additions, so the bench pins them off and
        // compares the DSP path the two engines actually share. Leaving
        // them on cost 99.7% of the residual on every filter case.
        session.set_dc_blockers(false);
        session.set_random_seed(self.random_seed);
        session.set_check_bounds(true);
        session.load_preset_json(&preset.to_json().map_err(|e| e.to_string())?)?;
        // A connection the engine cannot route would leave the case
        // measuring something other than what it says. Two of the mono
        // destinations were missing for as long as this did not refuse.
        if !session.last_report.ignored_connections.is_empty() {
            return Err(format!(
                "case wires a connection Spinwave ignores: {}",
                session.last_report.ignored_connections.join(", ")
            ));
        }

        // The same single-cycle shape the reference loads, through the same
        // steps: one frame, then the band-limited post-process.
        let mut frame = WaveFrame::predefined(self.shape);
        frame.to_frequency_domain();
        session.load_single_frame_wavetables(&frame);

        let notes: Vec<NoteSpec> = self
            .notes
            .iter()
            .map(|note| NoteSpec {
                note: note.midi,
                velocity: note.velocity,
                start: note.start_seconds,
                duration: note.hold_seconds,
                channel: 0,
            })
            .collect();
        let (samples, curves) = session.render_samples_probed(&notes, self.seconds, 120.0, probes);
        let excursions = std::mem::take(&mut session.last_excursions);
        Ok((samples, curves, excursions))
    }
}

/// Parses a modulation source name for `--probe`, in the same spelling the
/// case files and the reference harness use (`lfo_1`, `env_2`, `random_1`,
/// `macro_control_1`, `velocity`...).
pub fn parse_probe(name: &str) -> Result<ModSource, String> {
    spinwave_plugin::patch::parse_mod_source(name)
        .ok_or_else(|| format!("unknown modulation source '{name}'"))
}

/// How far apart two renders are.
#[derive(Clone, Copy, Debug)]
pub struct Difference {
    /// Largest absolute difference on any sample.
    pub peak: f32,
    /// Root mean square of the difference.
    pub rms: f32,
    /// Sample index where `peak` occurs, in frames.
    pub worst_frame: usize,
    /// Peak level of the reference, for context: a large difference on a
    /// loud render can still be a small relative error.
    pub reference_peak: f32,
    /// Root mean square of the reference, for the relative reading below.
    pub reference_rms: f32,
}

impl Difference {
    /// Whether the two renders agree within `tolerance`, in absolute
    /// sample units.
    pub fn within(&self, tolerance: f32) -> bool {
        self.peak <= tolerance
    }

    /// The residual relative to the reference, in dB.
    ///
    /// The tolerances are absolute, which is right for the bound but hides
    /// something: 1e-3 of error against a render peaking near full scale is
    /// -60 dB and inaudible, while the same 1e-3 against a reference at
    /// -40 dBFS is only -20 dB relative and plainly wrong. A quiet case can
    /// therefore pass on absolute RMS while agreeing far less well than a
    /// loud one. This number makes that visible without loosening anything:
    /// read it to find cases that pass too easily, not to decide pass/fail.
    ///
    /// Silence in the reference has no relative reading; `None` says so
    /// rather than reporting an infinity.
    pub fn relative_db(&self) -> Option<f32> {
        if self.reference_rms <= 0.0 || self.rms <= 0.0 {
            return None;
        }
        Some(20.0 * (self.rms / self.reference_rms).log10())
    }

    pub fn describe(&self) -> String {
        let relative = match self.relative_db() {
            Some(db) => format!("{db:+.0} dB rel"),
            None => "no relative reading".to_string(),
        };
        format!(
            "peak difference {:.2e} at frame {}, rms {:.2e} ({relative}, reference peaks at {:.4})",
            self.peak, self.worst_frame, self.rms, self.reference_peak
        )
    }
}

/// Compares two interleaved renders. A length mismatch is an error rather
/// than a difference: it means the two halves disagree about the case, not
/// about the DSP.
pub fn compare(ours: &[f32], reference: &[f32]) -> Result<Difference, String> {
    if ours.len() != reference.len() {
        return Err(format!(
            "length mismatch: rendered {} samples, reference has {}",
            ours.len(),
            reference.len()
        ));
    }
    let mut peak = 0.0f32;
    let mut worst_frame = 0usize;
    let mut sum_squares = 0.0f64;
    let mut reference_peak = 0.0f32;
    let mut reference_squares = 0.0f64;
    for (index, (&ours, &theirs)) in ours.iter().zip(reference).enumerate() {
        let difference = (ours - theirs).abs();
        if difference > peak {
            peak = difference;
            worst_frame = index / 2;
        }
        sum_squares += (difference as f64) * (difference as f64);
        reference_squares += (theirs as f64) * (theirs as f64);
        reference_peak = reference_peak.max(theirs.abs());
    }
    let samples = ours.len().max(1) as f64;
    let rms = (sum_squares / samples).sqrt() as f32;
    let reference_rms = (reference_squares / samples).sqrt() as f32;
    Ok(Difference { peak, rms, worst_frame, reference_peak, reference_rms })
}

/// Reads a reference render: raw little-endian f32, interleaved stereo.
pub fn read_reference(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: not a whole number of f32 samples", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Writes a render in the reference format, for regenerating the corpus.
pub fn write_raw(path: &Path, samples: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The bench's directory, found from the crate rather than the working
/// directory so the test runs from anywhere.
pub fn bench_dir() -> PathBuf {
    // `SPINWAVE_BENCH_DIR` points the bench at another cases/reference
    // pair (an older revision's, extracted with `git show`, to measure
    // what a case's reconstruction changed with the SAME engine).
    if let Some(dir) = std::env::var_os("SPINWAVE_BENCH_DIR") {
        return PathBuf::from(dir);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(|root| root.join("tools").join("golden"))
        .unwrap_or_default()
}

/// Every case in the corpus, as `(name, case path, reference path)`.
pub fn corpus() -> Vec<(String, PathBuf, PathBuf)> {
    let bench = bench_dir();
    let cases = bench.join("cases");
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(&cases) else { return found };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|e| e == "txt"))
        .collect();
    paths.sort();
    for path in paths {
        let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().to_string()) else {
            continue;
        };
        let reference = bench.join("reference").join(format!("{stem}.raw"));
        found.push((stem, path, reference));
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_case_format_parses() {
        let case = Case::parse(
            "rate 48000\nseconds 2.5\nwave square\n\
             note 60 0.8 0.1 1.0   # a comment\n\
             set osc_1_level 0.75\n",
        )
        .unwrap();
        assert_eq!(case.sample_rate, 48000);
        assert_eq!(case.seconds, 2.5);
        assert_eq!(case.shape, WaveShape::Square);
        assert_eq!(case.notes.len(), 1);
        assert_eq!(case.notes[0].midi, 60);
        assert_eq!(case.notes[0].hold_seconds, 1.0);
        assert_eq!(case.controls, vec![("osc_1_level".to_string(), 0.75)]);

        assert!(Case::parse("nonsense 1").is_err());
        assert!(Case::parse("wave triangular").is_err());
    }

    #[test]
    fn comparing_reports_the_worst_sample() {
        let ours = vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.0];
        let reference = vec![0.0, 0.0, 0.3, 0.0, 0.0, 0.0];
        let difference = compare(&ours, &reference).unwrap();
        assert!((difference.peak - 0.2).abs() < 1e-6);
        assert_eq!(difference.worst_frame, 1);
        assert!(!difference.within(0.1));
        assert!(difference.within(0.3));

        assert!(compare(&ours, &reference[..2]).is_err());
    }
}

#[cfg(test)]
mod corpus_tests {
    use super::*;

    /// Two renders agree when the error has the shape of rounding rather
    /// than of a different algorithm.
    ///
    /// The bounds were RMS 1e-3 / peak 2e-2 for months, argued from a
    /// floor of 4e-4 RMS on `osc_saw_dry` explained as "0.015 samples of
    /// timing between two phase accumulators". The floor was one wrong
    /// call (the base frequency through the approximate `exp2`, see
    /// `notes/audio-rate-audit.md` §2), and once it was gone the passing
    /// residuals split in two with nothing between: 64 cases at or below
    /// 1e-5 (59 of them below 1e-6, float noise), and a handful between
    /// 1e-4 and 1e-3 — every one of which turned out to be a real, small
    /// error (the phaser's coefficient computed instead of looked up, a
    /// per-block destination, a ramp with the one-block lead). So the
    /// bounds sit in the gap, tightened 2026-09-12 after the phaser fix:
    /// RMS 1e-4, peak 2e-3 × max(reference peak, 1). The peak bound stays
    /// relative to the case's level because edge error is slope times
    /// timing, and a floor keeps quiet cases from an impossible absolute.
    ///
    /// Tighten these if a case ever passes that should not. Do not loosen
    /// them to make one pass. A passing case above 1e-5 RMS is saying
    /// something — and as of 2026-09-13 what the seven above 1e-5 say is
    /// known: the compressor family (~1e-5) is the reference's own
    /// direct-form crossover at 120 Hz amplifying the engines' float
    /// noise (bit-identical on identical input, measured by
    /// perturbation: notes/exact-vs-polynomial.md), and macro_dest_lfo
    /// (4.5e-5) is the LFO value held across the silence. Neither is a
    /// correction waiting to be made, so 1e-4 stays: 1e-5 would sit
    /// against the compressor's floor.
    const TOLERANCE_RMS: f32 = 1.0e-4;
    const TOLERANCE_PEAK: f32 = 2.0e-3;

    fn peak_allowance(reference_peak: f32) -> f32 {
        TOLERANCE_PEAK * reference_peak.max(1.0)
    }

    /// Cases that do NOT match the reference yet, with what is wrong.
    ///
    /// The corpus was written in one go and turned up a landscape rather
    /// than a single bug. Hiding that would waste it, and deleting the
    /// failing cases would waste it twice, so they stay and this list
    /// records them. The test asserts a known case still diverges: fixing
    /// one has to be noticed and removed from here, or the bench quietly
    /// stops testing it.
    ///
    /// Ordered worst first by RMS. Everything not listed must match.
    const KNOWN_DIVERGENCES: &[(&str, &str)] = &[
        // The modulation matrix. It was the BENCH, not the engine — and
        // more narrowly than first written.
        //
        // Vital's own ModulationConnectionBank::createConnection gives a
        // NEW connection its bipolar flag from the source's prefix:
        // `lfo`, `random`, `stereo` and `pitch` are born bipolar (see
        // kBipolarModulationSourcePrefixes in synth_types.cpp). A loaded
        // .vital overrides that with the stored flag; the harness created
        // its connections fresh and never wrote the flag, so it inherited
        // the creation default on exactly those sources. Measured under
        // the old harness: lfo and random CENTRED, macro / note / velocity
        // / envelope not — which is why the constant-source cases passed
        // all along and why "the poly route is sound" still stands.
        //
        // It hid because setting `modulation_1_bipolar` from a case did
        // nothing either (the harness set it before the connection
        // existed): the reference rendered `mod_lfo_bipolar_low` BYTE FOR
        // BYTE identically to `mod_lfo_to_cutoff`. Two "bipolar" cases were
        // testing the unipolar path.
        //
        // The harness now re-runs its whole initialisation after wiring,
        // and AUDITS it: every control the case does not name must hold
        // its table default, or the run aborts. Six references changed;
        // `mod_lfo_to_cutoff` went from rms 1.6e-1 to 2.8e-4, with
        // `mod_lfo_to_cutoff_high` and `mod_two_sources_one_dest`.
        //
        // Five diagnoses died on the way, the last carefully measured and
        // still wrong because the instrument was. Two rules to keep: when
        // two cases differ by ONE setting, check their references differ;
        // and assert the harness's initialisation instead of discovering
        // its gaps one wrong reference at a time.
        //
        // What still fails, against references that are now right:
        //
        // The level cases had a second bug, found by probing the
        // DESTINATION per lane (and the probe first had to learn to read
        // the active slot's lanes, not lanes 0 and 1: a retriggered note
        // can land on slot 1, and the dead voice's lanes read zero).
        // Source, transform and route agreed to 0.1%; the consumer did
        // not. Spinwave clamped the summed level to [0, 1] where the
        // reference's oscillator does `max(amplitude, 0)` and squares it —
        // no ceiling — so a 0.7 level plus an envelope at 0.75 peaks at
        // 2.1 there and at 1.0 here. Removing the ceiling took
        // mod_lfo_to_level from 2.2e-1 to 1.7e-3 and mod_env_to_level from
        // 7.3e-2 to 1.25e-2. What is left on both is that `osc_N_level` is
        // an AUDIO-RATE destination in the reference (createPolyModControl
        // with audio_rate = true; the oscillator reads an amplitude
        // buffer per sample) while Spinwave ramps it once per block —
        // visible as one block at the onset and as the curvature of an
        // attack. DONE: `osc_N_level` is now an audio-rate destination
        // here too, with a per-sample offset buffer into the oscillator's
        // level stage. mod_env_to_level went to 7.0e-4 and is delisted;
        // mod_lfo_to_level sits at 1.07e-3, a hair over.
        //
        // mod_env_to_pitch was the same bug on the other per-sample
        // inputs: the reference's oscillator evaluates level, transpose,
        // tune and phase per sample (`createPolyModControl(..., true)` for
        // all four), reading a transpose buffer inside its phase-increment
        // loop, where Spinwave ramped the whole pitch once per block. A
        // 48-semitone chirp over 25 ms is a dozen blocks; the per-block
        // ramp lands the phase somewhere else and the sustain, at the right
        // pitch, never lines up again. Measured before the fix: the sustain
        // pitch matched to 0.01 st, the divergence was all phase. With the
        // four inputs audio-rate: 2.6e-1 -> 5.3e-4, delisted. The three
        // cases added for the inputs that had none (tune, phase, snapped
        // transpose) pass at 1.3e-4, 3.3e-4 and 5.4e-4 — the snapped one
        // only after replacing the snap itself: the oscillator rounds to
        // the nearest note THEN looks the note up in a table
        // (`fillSnapBuffer`), which the sample source's nearest-by-distance
        // `snapTranspose` — what Spinwave used for both — does not
        // reproduce (3.1e-1 with the wrong snap).
        //
        // mod_random_to_cutoff was not DSP either. Its residual changed
        // with the ORDER the bench ran in (6.7e-2, 4.4e-2, 4.9e-2 for the
        // same code): both engines seed each RandomGenerator from a
        // process-global counter, the reference runs one process per case
        // and Spinwave ran 73 cases in one. Rewinding the counter per
        // render made it stable at 6.3e-2; fitting the Perlin curve of
        // `--probe random_1` on both sides (tools/golden/random_seed.py,
        // residual 1e-9) gave the draws, and the draws gave the seeds: 18
        // in the reference, 684 here. Same draw indices on both sides for
        // both notes; only the seed differed. `random_seed 18` in the case
        // pins ours: 1.6e-4, delisted. The "unipolar contribution DC
        // offset" reading of it was a diagnosis of noise.
        //
        // The one-block lead on every source, measured by the probe, was
        // tried both ways — resolving the matrix before advancing the
        // modulators, and reading each control-rate modulator before its
        // advance — and neither helps: the first delays the constants
        // (note, velocity) that the reference applies at once and turns
        // two green cases red; the second moves the envelope cases by a
        // few percent. It is real, but it is not what any tracked case is
        // made of.
        // The mono effect destinations the reference creates audio-rate
        // (notes/audio-rate-audit.md). Their ModulationSum ramps the
        // control part across the block and adds audio-rate sources per
        // sample; Spinwave resolves the bus chain once per block. The
        // control on the same route, mod_lfo_to_distortion_mix (a
        // control-rate destination), passes at 3.2e-4, so what is measured
        // here is the rate and nothing else. phaser_center and
        // distortion_filter_cutoff were MISSING as destinations before
        // these cases (2.0e-1 and 1.3e-1, the modulation dropped), and
        // the distortion's own filter was never read from the preset
        // (fx_distortion_filter_pre/post 2.3e-1 -> 7e-5).
        // THE RESIDUAL FLOOR WAS THE BASE FREQUENCY. Every oscillator case
        // sat at ~4e-4 RMS (-67 dB) with its peak on the saw's edges,
        // read for months as "0.015 samples of timing between two phase
        // accumulators" and accepted as the floor. It was one call: the
        // reference converts the block's base note with the EXACT
        // `utils::midiNoteToFrequency` (powf) and scales it per sample by
        // the polynomial `futils::midiOffsetToRatio` of the small offset;
        // Spinwave converted the note with the polynomial. The survey of
        // "two reference functions, one port" (notes/audio-rate-audit.md)
        // found it. osc_saw_dry 3.3e-4 -> 4.4e-8; 60 of 70 cases improved
        // more than tenfold, none got worse; seven tracked cases fell to
        // float noise at once — mod_lfo_to_level (1.2e-7, so the "LFO
        // one-block lead" was never what it was made of), the two-voice
        // LFO, the inharmonic stretch ("a term near Nyquist growing
        // across the note" was the base pitch error integrating), the
        // three warps, and distortion_drive's near miss. The delay's
        // three filter frequencies had the same wrong call; fx_delay did
        // not move, so its cause is elsewhere.

        // Promoted by the 2026-09-12 tightening (1e-3 -> 1e-4): the cases
        // that sat between the two bounds, each measured, not all
        // diagnosed. The hypotheses are labelled as such.
        // The three pitch ramps (mod_env_to_pitch 3.6e-4, _snapped 4.4e-4,
        // mod_env_to_tune 1.2e-4) were NOT the one-block lead they were
        // listed as: the exact note-to-frequency conversion computed
        // `note * (1/12)` where the reference computes `(note * 100) /
        // 1200`, a last-bit difference on a transposed note that drifted
        // the phase over the sustain. 7.4e-8, 7.6e-8, and 5.8e-5 for the
        // tune — which sat in the gap until the bounds check
        // (`bounds.rs`) showed the case pushing the tune to 1.2 in
        // [-1, 1]. It was first written up as "the two engines leave a
        // clamp differently"; that was WRONG. Re-running the old case
        // through the engine of 2026-09-13 (SPINWAVE_BENCH_DIR on the
        // pre-reconstruction files): 4.4e-8, and putting the exact log2
        // back into Wavetable::frequency_float_bin brings the 5.8e-5
        // back. The one localized event was the tune's ramp crossing a
        // mip-bin boundary, where the exact and the polynomial log2
        // place the crossing a sample apart (notes/exact-vs-polynomial.md).
        // The "lead" the probe reported on every source was the probe:
        // the reference's status outputs read one block late.
        // osc_unison (3.2e-4) was the unison detune ratio through the
        // polynomial exp2 where the reference's setPhaseIncMults uses the
        // exact utils::centsToRatio — the base-frequency floor one
        // function over. 5.5e-8, delisted.
        // Meta-modulation: ported 2026-09-12 against the cases of
        // notes/meta-modulation.md, every one at float noise. What the
        // port found beyond the note: a connection from a MONO source
        // reads its meta-modulated amount one block late (the reference
        // evaluates those before the voices), and an audio-rate source's
        // control value is its buffer's first sample, not its last.
        // fx_chorus (1.1e-4) was the exponential scale after all — not
        // the polynomial exp2 versus the exact one, which had been tried,
        // but the reference's ExponentialScale being futils::pow(2, x) =
        // exp2(log2(2) * x) with the POLYNOMIAL log2(2), one to a few ulp,
        // where Spinwave had the polynomial exp2(x). Those ulp on a
        // chorus delay time of 2^-9 s moved the fractional delay by a
        // last bit and the chorus by 1.1e-4; with pow: 3.7e-6, delisted.
        // Every Exponential-scale control now goes through
        // tempo::exponential_scale (notes/exact-vs-polynomial.md).
        // fx_delay (1.4e-2) and fx_reverb (6.4e-3) were the PRIMER: the
        // first note differs between the engines by construction (the
        // reference glides it from MIDI 0) and `skip` hides it, but a
        // 250 ms delay at feedback 0.5 brings it back into the window at
        // 0.5^4 and a reverb tail lasts seconds. Skipping 5 s instead of
        // 1 on the cases with a delay line or a reverb: 2.2e-7 and
        // 4.0e-7, delisted. The mono destinations of 2026-09-13 (34 of
        // them, a macro each with a static twin) found the tempo index
        // rounded to nearest by the reference's toInt where Spinwave
        // truncated (chorus_tempo 3.8e-1 -> 9.6e-7).
        //
        // The five mono audio-rate destinations (2026-09-13, evening):
        // distortion_drive, distortion_filter_cutoff, eq_low_cutoff,
        // phaser_center and filter_fx_cutoff were resolved per block
        // (9.1e-4, 5.3e-3, 1.1e-3, 2.2e-3, 5.8e-3). Now the engine builds
        // them a per-sample buffer, the reference's ModulationSum: the
        // control part ramped across the block, the audio-rate
        // connections added per sample from the last active voice
        // (`EffectsModMatrix::ramp_control`, `add_audio_sources`): 9.4e-8,
        // 8.7e-8, 8.7e-8, 1.2e-6, and the filter fx at 1.2e-3 STILL. Its
        // cutoff buffer dumped from both engines was bit-identical, and a
        // free 32 Hz LFO made the residual 20 dB louder: the reference's
        // filter fx reads its cutoff one block late (its FilterModule
        // orders the filter before the sum feeding it; the router's order
        // printed from the reference: SallenKeyFilter, SmoothValue,
        // ModulationSum, cr::Multiply - the keytrack multiply after the
        // sum, so a note change reaches the filter two blocks late).
        // With the lag and the ramped keytrack (EffectChain::process):
        // 2.8e-8, the 32 Hz twin 3.2e-8, fx_filter_fx_keytrack 3.0e-8.
        //
        // The diode (filter_diode_low_q 2.3e-3, _high_q 1.9e-2, the
        // whole residual in the note's first 100 ms), same evening: the
        // filter alone is bit-identical to the reference
        // (`vital_golden --diode` / the `diode_probe` example, both
        // resonances, transient included), but in the voice its output
        // was non-zero on a zero input right after the note-on reset -
        // the reference's `DiodeFilter::reset` leaves its feedback
        // high-pass alone (diode_filter.cpp:26), and ours too, so that
        // one-pole carries the IDLE lane's pre-note input into the note.
        // Both engines carry junk there; the junk differed: the reference
        // oscillator, with one voice of a pair active, folds that voice's
        // output into the idle lanes (`convertVoiceChannels`, `out +=
        // swapVoices(out)`) while Spinwave's idle lane ran its own MIDI-0
        // saw. Mirroring the active voice into the idle lanes
        // (SynthVoiceKernel::run_producers): 5.4e-8 and 5.5e-8. The
        // reference's own onset depends on when the previous voice of the
        // pair died (two releases of the primer, 0.3 and 0.45: the second
        // note's onset differs by rms 7.6e-3, 0.62 dB); Spinwave now
        // shares that dependency, being the same machine.
        //
        // A pitch that keeps moving (mod_lfo_to_transpose_sweep,
        // mod_env_to_pitch_slow) read 3.1e-4 and 2.1e-4 against a RELEASE
        // build of the reference during the bank's bisection and 4.1e-8
        // and 7.4e-8 against the Debug build the goldens come from: the
        // residual was the reference compiler's, a few ulps in the
        // base-times-polynomial-ratio path integrated by the phase while
        // the pitch moves (it plateaued once the pitch settled). The
        // Release harness stays a bisection tool, not a judge.
        //
        // The downsample distortion under an LFO on its drive (the bank's
        // Staggered Phrases, 21 dB, is this: lfo_1 -> distortion_drive on
        // a type-5 distortion). The unit is bit-identical
        // (`vital_golden --distortion 5 12 88200` against the
        // `distortion_probe` example), the static case reads 4e-8 once the
        // base glides like the reference's SmoothValue, and the drive
        // buffers dumped from both engines are identical from sample 256
        // on. They differ in the primer's FIRST block only (up to 7.8e-3
        // mid-block, equal at its ends): the audio-rate LFO's part of
        // the drive at the very start of the render, which every other
        // case hides under the skip and the downsample keeps as its hold
        // counter's residual - a phase of the hold grid, for good. The
        // note plays through a grid offset by a fraction of a sample:
        // 8.4e-2. Next: the connection's amount ramp and the LFO's own
        // first audio-rate block, compared sample by sample at t = 0.
        ("mod_lfo_to_downsample_drive", "rms 8.4e-2: the hold grid's residual from the render's first block"),
        //
        // -- Gate 3's second round (2026-09-13): the bank's residual
        // classes that a case reproduces and this pass did not close.
        // Each is named after its preset. What the pass DID close, for
        // the record (each at the floor now, its case in the corpus):
        // random_2..4 and the per-note `random` seeded 19 - N and 19;
        // random LFOs rendered per sample (sample-and-hold steps mid
        // block); the `stereo` source [1, 0]; FM / RM wired by the
        // reference's A / B / sample map with its render order and its
        // silent cycle; a mono-source plug into a modulator parameter
        // leaving the poly connections where they were; an envelope
        // into voice_transpose read the same block, an LFO a block late;
        // a filter's setup reading its cutoff buffer's first sample (the
        // comb); a stereo sample loaded uncapped.
        //
        // A macro as a destination with a moving source, heard on a
        // PITCH. `macro_dest_lfo` (lfo -> macro -> cutoff) reads 4.5e-5;
        // the same chain into osc_1_transpose integrates its timing
        // error into a phase offset for the whole note: 3.7e-2 with a
        // unipolar LFO, the same with a bipolar one, 2.2e-1 with a
        // sample-and-hold random either polarity (the steps are
        // larger). The macro chain's two-block lag was measured on
        // static steps (macro_dest_step); what it does around a note-on,
        // where the reference's mono chain holds the previous voice's
        // last readout, is not pinned. Not the polarity, not the random.
        ("bank_cursed_random_to_macro", "rms 2.2e-1: a random into a macro into a pitch (Cursed Steps)"),
        ("bank_cursed_random_to_macro_unipolar", "rms 2.2e-1: the same, unipolar"),
        ("bank_cursed_lfo_bipolar_to_macro", "rms 3.7e-2: an LFO into a macro into a pitch"),
        ("bank_cursed_lfo_to_macro_to_transpose", "rms 3.7e-2: the same, unipolar"),
        //
        // An LFO sweeping osc_1_unison_voices over its full range (1 to
        // 16 in a quarter second). The residual sits ENTIRELY in the
        // blocks where the count is 3, 4 or 5 (1.1e-1 to 1.5e-1 per
        // block there, the floor before and after; blocks alternate
        // between a 4 % level difference at correlation 1.000 and a
        // shape difference at 0.75-0.96); the same sweep narrowed to
        // 1..4, 3..6 or 6..9 reads 2e-6, and an envelope over the full
        // range too. Something about a fast rise through 3..5 -
        // `setActiveOscillators` gives the new oscillators null wave
        // buffers until the next fade boundary (kWavetableFadeTime,
        // 7 ms) - not located.
        ("bank_remedial_lfo_to_unison_voices", "rms 2.7e-2: a fast LFO sweep of the unison count, counts 3..5 only (Remedial Shikari)"),
        //
        // An LFO into the delay's time (delay_frequency). Alive, the two
        // engines agree to 1e-3 per block (the delay's per-block period
        // interpolation under a moving target, itself above the floor);
        // from the voice's death on, the modulation each engine HOLDS
        // differs (echo periods 686 and 370 samples measured by
        // autocorrelation after the death) and the echoes drift apart at
        // 3e-1. The reference writes its non-accumulated readouts only
        // while a voice is active, so the held value is the block before
        // the death - which Spinwave models the same way; why the held
        // values still differ is not established.
        ("bank_squish_lfo_to_delay_frequency", "rms 1.2e-1: the delay time's held modulation after the voice dies (Squish Clicker)"),
        //
        // An LFO into filter_fx_blend and filter_fx_resonance together,
        // interior values: 8.4e-4, peak at the note's first block. Both
        // are control-rate mono destinations (the blend interpolates the
        // filter's two outputs, the resonance its coefficients per
        // block); the phaser's four control-rate destinations from the
        // same LFO read 6e-6 to 5.5e-5. Not located.
        ("bank_phaser_entropy_lfo_to_filter_fx_blend", "rms 8.4e-4: an LFO into the filter fx's blend and resonance (Phaser Entropy)"),
        //
        // A poly envelope into the mono filter_fx_cutoff with its amount
        // from the VELOCITY through a meta connection, two voices of
        // different velocities: 2.9e-4, peak 3.9e-3 at the second note.
        // Which voice's velocity the mono connection's amount reads, and
        // when, is not pinned (the reference's poly readouts are the last
        // active voice's, written after the voices).
        ("bank_dispersed_env_to_filter_fx_meta_velocity", "rms 2.9e-4: a meta amount from the velocity on a mono destination, two voices (Dispersed Grit)"),
    ];

    fn is_known(name: &str) -> Option<&'static str> {
        KNOWN_DIVERGENCES.iter().find(|(case, _)| *case == name).map(|(_, why)| *why)
    }

    /// Cases whose values are allowed against a bound, because the bound
    /// is what they measure. Everything else must keep every control and
    /// every modulated value interior to its range for the whole render
    /// (`bounds.rs`): the rule of 2026-09-12, born of a meta chain whose
    /// summed amounts saturated at 1 and hid the chain behind the clamp.
    /// The first automatic pass flagged 28 cases — among them the gap
    /// occupant mod_env_to_tune (the tune at 1.2), the four LFO -> cutoff
    /// cases (cutoff 150 and 170 in [8, 136]), both level cases (1.4 and
    /// 1.5 in [0, 1]) and fx_flanger's dry/wet set to 0.8 in [0, 0.5]; all
    /// were redesigned interior and re-rendered, and the gap emptied.
    const BOUNDED_BY_DESIGN: &[(&str, &str)] = &[
        ("meta_bounds", "the amount pushed to 2.5: measures the clamp at 1"),
        ("meta_bounds_twin_clamped", "static amount 1.0, the clamped twin (cutoff to 183)"),
        ("meta_bounds_twin_overflow", "static amount 2.5, the overflowed twin"),
        ("meta_power_bounds", "the power pushed to 20: measures that it is NOT clamped"),
        ("meta_power_bounds_twin_overflow", "static power 20, the overflowed twin"),
        (
            "meta_true_cycle",
            "a two-cycle's loop gain is 4 x source_1 x source_2, above 1 for any two \
             sources past 0.5; its amounts saturate whatever the bases, and it proves \
             bounded / deterministic / NaN-free, nothing else",
        ),
    ];

    fn is_bounded_by_design(name: &str) -> bool {
        BOUNDED_BY_DESIGN.iter().any(|(case, _)| *case == name)
    }

    /// Spinwave must render every case in the corpus the way Vital does,
    /// except the ones listed above, which must keep diverging until
    /// somebody fixes them and says so here.
    #[test]
    fn spinwave_matches_the_reference_on_every_case() {
        let cases = corpus();
        assert!(!cases.is_empty(), "no cases found under {}", bench_dir().display());

        let mut failures = Vec::new();
        for (name, case_path, reference_path) in cases {
            if !reference_path.exists() {
                failures.push(format!("{name}: no reference render; see tools/golden/README.md"));
                continue;
            }
            let case = match Case::read(&case_path) {
                Ok(case) => case,
                Err(e) => {
                    failures.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let skip = (case.skip_seconds.max(0.0) * case.sample_rate as f32) as usize * 2;
            let ours = match case.render_checked() {
                Ok((samples, excursions)) => {
                    // A value against a bound measures the clamp, not what
                    // the case says it measures.
                    if !excursions.is_empty() && !is_bounded_by_design(&name) {
                        let list: Vec<String> = excursions.iter().map(|e| e.describe()).collect();
                        failures.push(format!("{name}: value against a bound: {}", list.join("; ")));
                        continue;
                    }
                    if excursions.is_empty() && is_bounded_by_design(&name) {
                        failures.push(format!(
                            "{name} is listed in BOUNDED_BY_DESIGN but nothing in it reaches a bound"
                        ));
                        continue;
                    }
                    samples
                }
                // A case the engine cannot route yet is a tracked divergence
                // of the strongest kind; listed, it must still fail to render
                // (a listed case that renders and matches is caught below
                // like any other).
                Err(e) if is_known(&name).is_some() => {
                    let _ = e;
                    continue;
                }
                Err(e) => {
                    failures.push(format!("{name}: render failed: {e}"));
                    continue;
                }
            };
            let reference = match read_reference(&reference_path) {
                Ok(samples) => samples,
                Err(e) => {
                    failures.push(format!("{name}: {e}"));
                    continue;
                }
            };
            // The reference file already starts after the skip: the
            // harness renders the primer but does not write it, so the
            // corpus stores only what gets compared.
            let ours = &ours[skip.min(ours.len())..];
            let reference = &reference[..];
            match compare(ours, reference) {
                Ok(difference) => {
                    let matches = difference.rms <= TOLERANCE_RMS
                        && difference.peak <= peak_allowance(difference.reference_peak);
                    match (matches, is_known(&name)) {
                        (true, None) => {}
                        (false, Some(_)) => {}
                        (false, None) => {
                            failures.push(format!("{name}: {}", difference.describe()))
                        }
                        (true, Some(why)) => failures.push(format!(
                            "{name} now MATCHES but is listed as diverging ({why});                              remove it from KNOWN_DIVERGENCES"
                        )),
                    }
                }
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }

        assert!(failures.is_empty(), "diverged from the reference:\n  {}", failures.join("\n  "));
    }
    /// A case that does not render the same bytes twice supports no
    /// conclusion at all — and the order the cases run in must not matter
    /// either. mod_random_to_cutoff changed its residual with the run
    /// order for as long as the random seed counter was process-global;
    /// every diagnosis made of it in that state was a diagnosis of noise.
    /// The second pass runs the corpus backwards, so any state one case
    /// leaves for the next lands on a different case.
    #[test]
    fn every_case_renders_the_same_bytes_in_any_order() {
        let cases = corpus();
        assert!(!cases.is_empty());
        let mut first = std::collections::HashMap::new();
        for (name, case_path, _) in &cases {
            let case = Case::read(case_path).unwrap_or_else(|e| panic!("{name}: {e}"));
            // A case the engine refuses (a tracked unrouted destination)
            // has no bytes to compare; the matching test covers it.
            if let Ok(samples) = case.render() {
                first.insert(name.clone(), samples);
            }
        }
        assert!(first.len() > cases.len() / 2, "most cases must render");
        let mut unstable = Vec::new();
        for (name, case_path, _) in cases.iter().rev() {
            let Some(before) = first.get(name) else { continue };
            let case = Case::read(case_path).unwrap();
            let again = case.render().unwrap();
            if again.len() != before.len()
                || again.iter().zip(before).any(|(a, b)| a.to_bits() != b.to_bits())
            {
                let worst = again
                    .iter()
                    .zip(before)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                unstable.push(format!("{name}: two renders differ, worst sample {worst:.3e}"));
            }
        }
        assert!(unstable.is_empty(), "non-deterministic cases:\n  {}", unstable.join("\n  "));
    }
}
