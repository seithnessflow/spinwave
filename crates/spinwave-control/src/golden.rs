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
        Ok(session.render_samples(&notes, self.seconds, 120.0))
    }
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
}

impl Difference {
    /// Whether the two renders agree within `tolerance`, in absolute
    /// sample units.
    pub fn within(&self, tolerance: f32) -> bool {
        self.peak <= tolerance
    }

    pub fn describe(&self) -> String {
        format!(
            "peak difference {:.2e} at frame {}, rms {:.2e} (reference peaks at {:.4})",
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
    for (index, (&ours, &theirs)) in ours.iter().zip(reference).enumerate() {
        let difference = (ours - theirs).abs();
        if difference > peak {
            peak = difference;
            worst_frame = index / 2;
        }
        sum_squares += (difference as f64) * (difference as f64);
        reference_peak = reference_peak.max(theirs.abs());
    }
    let rms = (sum_squares / ours.len().max(1) as f64).sqrt() as f32;
    Ok(Difference { peak, rms, worst_frame, reference_peak })
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
        // The modulation matrix. Diagnosed: from about 300 ms into a note
        // the two engines are IDENTICAL (-12.1 dB both, measured on
        // mod_lfo_to_cutoff), and the whole divergence lives in the first
        // 200 ms, where the reference ramps a modulated destination up
        // from below while Spinwave arrives at its value immediately. It
        // is a smoothing difference at every note onset, not a wrong
        // transform: the transform itself was read line by line against
        // modulation_connection_processor.cpp and matches. The bipolar
        // case passes, which rules out the source range and the polarity
        // maths.
        ("mod_env_to_pitch", "rms 2.6e-1: onset ramp on a modulated destination"),
        ("mod_random_to_cutoff", "rms 2.3e-1: onset ramp on a modulated destination"),
        ("mod_lfo_to_cutoff", "rms 1.6e-1: onset ramp on a modulated destination"),
        ("mod_two_sources_one_dest", "rms 1.2e-1: onset ramp, two sources summed"),
        ("mod_env_to_level", "rms 7.3e-2: onset ramp on a modulated destination"),
        ("osc_morph_inharmonic_stretch", "rms 1.2: partial positions are wrong, and it clips"),
        ("osc_morph_random_amplitudes", "rms 2.0e-1: the random table still differs"),
        ("filter_diode_high_q", "rms 1.9e-2: diode filter, worse at high resonance"),
        ("fx_delay", "rms 1.4e-2: the delay disagrees"),
        ("osc_warp_sync", "rms 9.5e-3: the sync warp disagrees"),
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
