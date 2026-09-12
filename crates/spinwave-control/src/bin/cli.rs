//! Command-line access to the same session the MCP server drives: render a
//! preset to a WAV, and read any audio file back as analysis.
//!
//! The MCP server is a long-lived process that holds its binary open, so a
//! rebuilt engine only reaches it after a restart. This tool always runs
//! the engine it was just built with, which makes it the honest way to
//! judge a change.
//!
//! ```text
//! spinwave-cli render <preset.vital> <out.wav> [--seconds N] [--notes 45,57,64]
//! spinwave-cli analyze <file.wav> [--start S] [--duration D]
//! ```

use spinwave_control::analysis::analyze;
use spinwave_control::fuzz::{patch_for_seed, run_seed, summarize, Wildness};
use spinwave_control::golden;
use spinwave_control::decode::decode_file;
use spinwave_control::session::{NoteSpec, Session};

fn flag(args: &[String], name: &str) -> Option<String> {
    let position = args.iter().position(|a| a == name)?;
    args.get(position + 1).cloned()
}

/// `45,57,64` or `45:0.9,57:0.8` (note or note:velocity), held for the
/// whole render minus a release tail.
fn parse_notes(spec: &str, seconds: f32, hold: Option<f32>) -> Vec<NoteSpec> {
    let hold = hold.unwrap_or((seconds - 1.0).max(0.2));
    spec.split(',')
        .filter_map(|token| {
            let mut parts = token.split(':');
            let note: i32 = parts.next()?.trim().parse().ok()?;
            let velocity = parts.next().and_then(|v| v.trim().parse().ok()).unwrap_or(0.85);
            Some(NoteSpec { note, velocity, start: 0.0, duration: hold, channel: 0 })
        })
        .collect()
}

fn print_analysis(label: &str, analysis: &spinwave_control::analysis::Analysis) {
    println!("{label}");
    println!("  peak {:.3}  rms {:.1} dB  dc {:+.5}", analysis.peak, analysis.rms_db, analysis.dc_offset);
    println!(
        "  centroid {:.0} Hz  rolloff {:.0} Hz  width {:.2}",
        analysis.spectral_centroid_hz, analysis.spectral_rolloff_hz, analysis.stereo_width
    );
    let bands = &analysis.bands_db;
    println!(
        "  bands  sub {:.1}  bass {:.1}  lowmid {:.1}  mid {:.1}  high {:.1}  air {:.1}",
        bands.sub_0_60,
        bands.bass_60_250,
        bands.low_mid_250_1k,
        bands.mid_1k_4k,
        bands.high_4k_12k,
        bands.air_12k_up
    );
    println!(
        "  envelope  attack {:.2}s  post-peak {:.1} dB  tail {:.1} dB",
        analysis.envelope.attack_seconds,
        analysis.envelope.post_peak_250ms_db,
        analysis.envelope.tail_db
    );
    let texture = &analysis.texture;
    println!(
        "  texture  harmonicity {}  odd/even {}  flatness {:.4}",
        texture.harmonicity.map_or("-".to_string(), |v| format!("{v:.2}")),
        texture.odd_even_ratio.map_or("-".to_string(), |v| format!("{v:.2}")),
        texture.spectral_flatness
    );
    if let Some(pitch) = analysis.pitch_hz {
        println!("  pitch {pitch:.1} Hz");
    }
    let rates: Vec<String> = analysis
        .movement
        .mod_rates_hz
        .iter()
        .map(|rate| format!("{:.2} Hz ({:.0}%)", rate.hz, rate.strength * 100.0))
        .collect();
    if !rates.is_empty() {
        println!("  movement  {}", rates.join(", "));
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("render") => {
            let preset = args.get(1).ok_or("usage: render <preset.vital> <out.wav>")?;
            let out = args.get(2).ok_or("usage: render <preset.vital> <out.wav>")?;
            let seconds: f32 =
                flag(&args, "--seconds").and_then(|v| v.parse().ok()).unwrap_or(4.0);
            let notes = parse_notes(
                &flag(&args, "--notes").unwrap_or_else(|| "45,57,64".to_string()),
                seconds,
                flag(&args, "--hold").and_then(|v| v.parse().ok()),
            );

            let mut session = Session::with_output_dir(std::env::current_dir().unwrap_or_default());
            let text = std::fs::read_to_string(preset).map_err(|e| format!("{preset}: {e}"))?;
            let summary = session.load_preset_json(&text)?;
            println!("{summary}");

            session.render_to(&notes, Some(seconds), 120.0, out, true)?;
            let analysis = session.analyze_last()?;
            print_analysis(out, &analysis);
            Ok(())
        }
        Some("fuzz") => {
            let count: usize = flag(&args, "--count").and_then(|v| v.parse().ok()).unwrap_or(200);
            let first: u64 = flag(&args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(0);
            let seconds: f32 =
                flag(&args, "--seconds").and_then(|v| v.parse().ok()).unwrap_or(4.0);
            let wildness = match flag(&args, "--wildness").as_deref() {
                Some("sparse") => Wildness::Sparse,
                _ => Wildness::Full,
            };

            println!("fuzzing {count} patches from seed {first} ({wildness:?}, {seconds}s each)");
            let mut verdicts = Vec::with_capacity(count);
            for i in 0..count as u64 {
                let verdict = run_seed(first + i, wildness, seconds);
                if !verdict.is_clean() {
                    let names: Vec<&str> =
                        verdict.defects.iter().map(|d| d.describe()).collect();
                    println!(
                        "  seed {:>6}  peak {:>7.3}  rms {:>7.1} dB  {:>5.1}x  {}",
                        verdict.seed,
                        verdict.peak,
                        verdict.rms_db,
                        verdict.realtime_factor,
                        names.join("; ")
                    );
                }
                verdicts.push(verdict);
            }
            println!("{}", summarize(&verdicts));

            // Save the worst offenders so they can be loaded and heard.
            if let Some(dir) = flag(&args, "--save-failures") {
                std::fs::create_dir_all(&dir).map_err(|e| format!("{dir}: {e}"))?;
                let mut saved = 0usize;
                for verdict in verdicts.iter().filter(|v| v.has_fatal()) {
                    let preset = patch_for_seed(verdict.seed, wildness);
                    let path = format!("{dir}/fuzz-{}.vital", verdict.seed);
                    let json = preset.to_json().map_err(|e| e.to_string())?;
                    std::fs::write(&path, json).map_err(|e| format!("{path}: {e}"))?;
                    saved += 1;
                }
                println!("saved {saved} failing patches to {dir}");
            }
            Ok(())
        }
        Some("golden") => {
            // Renders every corpus case through Spinwave and reports how
            // far each sits from Vital's own render. `--write DIR` dumps
            // our audio next to the reference so the two can be inspected.
            let write = flag(&args, "--write");
            if let Some(dir) = &write {
                std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
            }
            let only = flag(&args, "--case");
            // `--probe lfo_1,env_2` prints the control-rate value of each
            // source once per block instead of comparing audio. Run the
            // reference harness with the same `--probe` and diff the two
            // curves: that is what says WHERE a case diverges.
            let probe_names: Vec<String> = flag(&args, "--probe")
                .map(|list| list.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();
            let probes: Vec<_> = probe_names
                .iter()
                .map(|name| golden::parse_probe(name))
                .collect::<Result<_, _>>()?;
            for (name, case_path, reference_path) in golden::corpus() {
                if only.as_ref().is_some_and(|wanted| wanted != &name) {
                    continue;
                }
                let case = golden::Case::read(&case_path)?;
                if !probes.is_empty() {
                    let (_, curves) = case.render_probed(&probes)?;
                    let mut headers = probe_names.clone();
                    headers.extend(
                        ["cutoff_lane0", "cutoff_lane1", "level_lane0", "level_lane1"]
                            .map(String::from),
                    );
                    println!("block,{}", headers.join(","));
                    for row in 0..curves[0].len() {
                        let values: Vec<String> =
                            curves.iter().map(|c| format!("{}", c[row])).collect();
                        println!("{row},{}", values.join(","));
                    }
                    continue;
                }
                let ours = case.render()?;
                if let Some(dir) = &write {
                    golden::write_raw(std::path::Path::new(&format!("{dir}/{name}.ours.raw")), &ours)?;
                }
                if !reference_path.exists() {
                    println!("{name:<20} no reference render yet");
                    continue;
                }
                let reference = golden::read_reference(&reference_path)?;
                let skip =
                    (case.skip_seconds.max(0.0) * case.sample_rate as f32) as usize * 2;
                let ours = &ours[skip.min(ours.len())..];
                let reference = &reference[..];
                match golden::compare(ours, reference) {
                    Ok(difference) => println!("{name:<20} {}", difference.describe()),
                    Err(e) => println!("{name:<20} {e}"),
                }
            }
            Ok(())
        }
        Some("to-text") => {
            // .vital -> .spinwave, plus a sidecar directory `<out>.d/` when
            // the preset carries wavetables, samples or Spinwave material.
            let input = args.get(1).ok_or("usage: to-text <in.vital> <out.spinwave>")?;
            let output = args.get(2).ok_or("usage: to-text <in.vital> <out.spinwave>")?;
            let json = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
            let preset = spinwave_params::Preset::from_json(json.trim_start_matches('\u{feff}'))
                .map_err(|e| format!("{input}: {e}"))?;
            let written = spinwave_control::text_preset::write(&preset);
            std::fs::write(output, written.text.as_bytes()).map_err(|e| format!("{output}: {e}"))?;
            let sidecar = std::path::PathBuf::from(format!("{output}.d"));
            written.blobs.write_dir(&sidecar)?;
            let blobs = written.blobs.iter().count();
            println!(
                "wrote {output} ({} lines{})",
                written.text.lines().count(),
                if blobs > 0 { format!(", {blobs} blob(s) in {}", sidecar.display()) } else { String::new() }
            );
            Ok(())
        }
        Some("from-text") => {
            // .spinwave -> .vital. The report goes to stderr as JSON so an
            // agent can read it; a refused file writes nothing.
            let input = args.get(1).ok_or("usage: from-text <in.spinwave> <out.vital>")?;
            let output = args.get(2).ok_or("usage: from-text <in.spinwave> <out.vital>")?;
            let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
            let blobs = spinwave_control::text_preset::Blobs::from_dir(&std::path::PathBuf::from(format!("{input}.d")))?;
            let result = spinwave_control::text_preset::read(&text, &blobs);
            eprintln!("{}", serde_json::to_string_pretty(&result.report).unwrap_or_default());
            match result.preset {
                Some(preset) => {
                    let json = preset.to_json().map_err(|e| e.to_string())?;
                    std::fs::write(output, json.as_bytes()).map_err(|e| format!("{output}: {e}"))?;
                    println!("wrote {output}");
                    Ok(())
                }
                None => Err(format!("{input}: {} error(s); nothing written", result.report.errors.len())),
            }
        }
        Some("check") => {
            // Parse only; the report as JSON on stdout. Exit 1 on any error.
            let input = args.get(1).ok_or("usage: check <in.spinwave>")?;
            let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
            let blobs = spinwave_control::text_preset::Blobs::from_dir(&std::path::PathBuf::from(format!("{input}.d")))?;
            let result = spinwave_control::text_preset::read(&text, &blobs);
            println!("{}", serde_json::to_string_pretty(&result.report).unwrap_or_default());
            if result.preset.is_some() {
                Ok(())
            } else {
                Err(format!("{} error(s)", result.report.errors.len()))
            }
        }
        Some("sensitivity") => {
            // Moves every parameter and checks the sound moves too. The
            // formant filter's controls were wired to nothing for months
            // and no test could see it; this is the test that can.
            let only = flag(&args, "--only");
            // Exits 0 even with findings: the context rules are not
            // complete yet, so this is a worklist and not a verdict. See
            // the module docs before reading a name here as a bug.
            spinwave_control::sensitivity::sweep(only.as_deref(), |line| {
                println!("{line}");
            })?;
            Ok(())
        }
        Some("analyze") => {
            let path = args.get(1).ok_or("usage: analyze <file.wav>")?;
            let start = flag(&args, "--start").and_then(|v| v.parse().ok());
            let duration = flag(&args, "--duration").and_then(|v| v.parse().ok());
            let (stereo, sample_rate) = decode_file(path, start, duration)?;
            print_analysis(path, &analyze(&stereo, sample_rate));
            Ok(())
        }
        _ => Err("usage: spinwave-cli render <preset> <out.wav> | analyze <file> | fuzz [--count N] [--seed S] [--wildness full|sparse] [--save-failures DIR] | golden [--case NAME] [--probe SRC] | sensitivity [--only SUBSTR] | to-text <in.vital> <out.spinwave> | from-text <in.spinwave> <out.vital> | check <in.spinwave>".to_string()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("spinwave-cli: {e}");
        std::process::exit(1);
    }
}
