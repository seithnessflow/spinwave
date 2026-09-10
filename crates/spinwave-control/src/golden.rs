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
        session.load_preset_json(&preset.to_json().map_err(|e| e.to_string())?)?;

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
        Ok(session.render_samples_probed(&notes, self.seconds, 120.0, probes))
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

    /// Two renders agree when the error is inaudible and, more to the
    /// point, when it has the shape of rounding rather than of a different
    /// algorithm.
    ///
    /// The RMS bound is the real test: -60 dB below a signal that peaks
    /// near 1 leaves no room for a wrong coefficient, a wrong branch or a
    /// wrong constant. The peak bound is looser on purpose. Error at a
    /// waveform's discontinuity is the local slope times the timing
    /// difference, so on a band-limited saw whose edge moves 0.5 per
    /// sample, agreeing to a hundredth of a sample still shows up as
    /// several thousandths of amplitude. Measured on `osc_saw_dry`: RMS
    /// 4.1e-4, peak 8.8e-3, and every one of the ten worst samples sits on
    /// an edge. That is 0.015 samples of timing difference between two
    /// phase accumulators, which is where two implementations of the same
    /// arithmetic land.
    ///
    /// The peak bound is RELATIVE to how loud the case is, because the
    /// error it bounds is slope times timing and the slope scales with
    /// amplitude: a case peaking at 1.8 gets edges twice as steep as one
    /// peaking at 0.9, for the same timing agreement. A floor keeps quiet
    /// cases from being held to an impossible absolute.
    ///
    /// Tighten these if a case ever passes that should not. Do not loosen
    /// them to make one pass.
    const TOLERANCE_RMS: f32 = 1.0e-3;
    const TOLERANCE_PEAK: f32 = 2.0e-2;

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
        // The modulation matrix. SOLVED as a rule, not yet as a fix.
        //
        // Four wrong diagnoses died here before the right one, each
        // because two things changed at once. In order: "an onset ramp"
        // (the sources agree to 3e-4); "the poly route" (note and velocity
        // are poly and match); "the polarity branch" (macro, note and
        // velocity are unipolar and match); "unipolar AND a source that
        // varies" (a FLAT LFO diverges by the full amount).
        //
        // The flat-LFO case is what settled it. Draw a horizontal line in
        // the LFO editor (`lfo_shape flat <v>`) so the source cannot move,
        // and read the contribution at both extremes, amount 0.7:
        //
        //     flat at 0.0   reference -0.35   Spinwave  0.00
        //     flat at 1.0   reference +0.35   Spinwave +0.70
        //
        // The reference computes `amount * (source - 0.5)`, Spinwave
        // computes `amount * source`. Nothing varies; the whole error is
        // there. Crossed against the destination to be sure, an LFO is
        // centred into BOTH the cutoff and the level, and a macro into
        // NEITHER, so it is the source and not the destination.
        //
        // Reading the reference's own contribution for every source:
        //
        //     lfo_1     CENTRED     (amount * (source - 0.5))
        //     random_1  CENTRED
        //     env_2     uncentred   (amount * source)
        //     velocity  uncentred
        //     note      uncentred
        //     macro     uncentred
        //
        // So the reference centres exactly the two oscillating sources,
        // and Spinwave centres none. That is also where an earlier fix
        // stopped short: it moved the LFO and random sources out of a
        // [0.5, 1] range into [0, 1] and went no further.
        //
        // BEFORE FIXING, mind the constraint: `mod_lfo_bipolar` and
        // `mod_lfo_bipolar_low` PASS today, so whatever centres a source
        // must leave the bipolar branch where it is. Note that for an LFO
        // the reference's unipolar result already equals Spinwave's
        // bipolar result, which is why those two cases are green — so the
        // mechanism may be "the flag defaults on for oscillating sources"
        // rather than "the source value is centred". Those two are
        // distinguishable: set `modulation_1_bipolar 1` on a flat LFO and
        // see whether the reference shifts again or stays at +-0.35.
        //
        // The envelope cases fail for some OTHER reason: env is uncentred
        // on both sides. Suspect the one control block (2.9 ms) by which
        // `--probe` found Spinwave running ahead of the reference on every
        // source - structural, and the fix is to resolve the matrix BEFORE
        // `update_modulators` in the voice kernel.
        //
        // This one reaches past the bench: a unipolar LFO into the cutoff
        // is probably the commonest modulation in real patches.
        ("mod_env_to_pitch", "rms 2.6e-1, +2 dB rel: env is uncentred on BOTH sides, so           this one is something else"),
        ("mod_two_voices_one_lfo", "rms 2.5e-1, +1 dB rel: same, two voices sounding"),
        ("mod_random_to_cutoff", "rms 2.3e-1, +1 dB rel: unipolar contribution DC offset"),
        ("mod_lfo_to_level", "rms 1.6e-1, -5 dB rel: control-rate destination, so it is           not the audio-rate path"),
        ("mod_lfo_to_cutoff", "rms 1.6e-1, -1 dB rel: the LFO source is not centred"),
        ("mod_lfo_to_cutoff_high", "rms 1.3e-1, -3 dB rel: the base cutoff changes           nothing, so the destination value is not what decides it"),
        ("mod_two_sources_one_dest", "rms 1.2e-1, -4 dB rel: two summed"),
        ("mod_env_to_level", "rms 7.4e-2, -16 dB rel: env is uncentred on both sides"),
        ("osc_morph_inharmonic_stretch",
         "rms 6.1e-2: a term near Nyquist that grows across the note; the scratch           buffer aliasing into the inverse transform is fixed, the rest is not"),
        ("filter_diode_high_q", "rms 1.9e-2: diode filter, worse at high resonance"),
        ("fx_delay", "rms 1.4e-2: the delay disagrees"),
        ("osc_warp_sync", "rms 1.6e-3: one oversampled sample of gate timing at the           hard edge, every 11 cycles"),
        ("osc_warp_quantize",
         "rms 6.8e-4 passes; the PEAK does not, on a deliberately stepped waveform           where slope times timing is largest"),
        ("fx_reverb", "rms 6.4e-3: the reverb disagrees"),
        ("filter_diode_low_q", "rms 2.3e-3: diode filter"),
        ("osc_warp_pulse_width", "rms 2.1e-3: the pulse-width warp disagrees"),
    ];

    fn is_known(name: &str) -> Option<&'static str> {
        KNOWN_DIVERGENCES.iter().find(|(case, _)| *case == name).map(|(_, why)| *why)
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
            let ours = match case.render() {
                Ok(samples) => samples,
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
}
