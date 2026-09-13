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
use spinwave_control::ops;
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
                // A case the engine cannot render is reported like a
                // divergence, and the run goes on: the corpus is the
                // report, not the first refusal.
                let (ours, excursions) = match case.render_checked() {
                    Ok(rendered) => rendered,
                    Err(e) => {
                        println!("{name:<20} REFUSED: {e}");
                        continue;
                    }
                };
                // A value against a bound measures the clamp, not the case
                // (the corpus test allows the twins that saturate on
                // purpose; here every one is printed).
                for excursion in &excursions {
                    println!("{name:<20} BOUND: {}", excursion.describe());
                }
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
        Some("preset-render") => {
            // The twin of `vital_golden --preset`: a real preset rendered
            // the way the golden cases are (DC blockers off, a primer note
            // hidden by the skip, then the note), raw stereo f32 out, so
            // the two engines can be compared on what a preset sounds
            // like. `--dump-tables <dir>` writes each oscillator's built
            // wavetable in the same layout as the reference's dump.
            let input = args.get(1).ok_or("usage: preset-render <in.vital> <out.raw> [--note N] [--velocity V] [--hold S] [--seconds S] [--skip S] [--dump-tables DIR]")?;
            let output = args.get(2).ok_or("usage: preset-render <in.vital> <out.raw> [options]")?;
            let note = flag(&args, "--note").and_then(|v| v.parse::<i32>().ok()).unwrap_or(48);
            let velocity = flag(&args, "--velocity").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.8);
            let hold = flag(&args, "--hold").and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.5);
            let seconds = flag(&args, "--seconds").and_then(|v| v.parse::<f32>().ok()).unwrap_or(2.5);
            let skip = flag(&args, "--skip").and_then(|v| v.parse::<f32>().ok()).unwrap_or(1.0);
            let text = std::fs::read_to_string(input).map_err(|e| format!("{input}: {e}"))?;
            let text = text.trim_start_matches('\u{feff}');
            if let Some(dir) = flag(&args, "--dump-tables") {
                let preset = spinwave_params::Preset::from_json(text).map_err(|e| format!("{input}: {e}"))?;
                if let Some(tables) = preset.settings.wavetables.as_ref().and_then(|v| v.as_array()) {
                    for (i, table) in tables.iter().enumerate() {
                        let Some(built) = spinwave_dsp::wavetable::wavetable_from_json(table) else { continue };
                        let mut bytes: Vec<u8> = (built.num_frames() as f32).to_le_bytes().to_vec();
                        for frame in 0..built.num_frames() {
                            for v in built.data().wave_data(frame) {
                                bytes.extend_from_slice(&v.to_le_bytes());
                            }
                        }
                        std::fs::write(format!("{dir}/osc_{}.raw", i + 1), bytes).map_err(|e| e.to_string())?;
                    }
                }
                // The sample (the SMP section) as built: its length, then
                // the original-rate left buffer with its guard samples.
                if let Some(sample) = spinwave_plugin::patch::global_sample_from_preset(&preset) {
                    let index = spinwave_dsp::oscillator::sample_source::UPSAMPLE_TIMES;
                    let mut bytes: Vec<u8> = (sample.original_length() as f32).to_le_bytes().to_vec();
                    for v in sample.left_buffer(index) {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    std::fs::write(format!("{dir}/sample.raw"), bytes).map_err(|e| e.to_string())?;
                }
            }
            let mut session = spinwave_control::session::Session::with_output_dir(std::env::temp_dir());
            session.set_dc_blockers(false);
            // The reference's random LFOs draw from seed 18 onwards in
            // its construction order (tools/golden/random_seed.py).
            session.set_random_seed(Some(18));
            // `--fixed-phase`: random phases off on every oscillator and
            // the sample, as `vital_golden --preset --fixed-phase` does.
            let text = if args.iter().any(|a| a == "--fixed-phase") {
                let mut value: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{input}: {e}"))?;
                if let Some(settings) = value.get_mut("settings").and_then(|v| v.as_object_mut()) {
                    for key in ["osc_1_random_phase", "osc_2_random_phase", "osc_3_random_phase", "sample_random_phase"] {
                        settings.insert(key.into(), serde_json::json!(0.0));
                    }
                }
                value.to_string()
            } else {
                text.to_string()
            };
            session.load_preset_json(&text)?;
            let notes = vec![
                spinwave_control::session::NoteSpec { note: 45, velocity: 0.9, start: 0.0, duration: 0.3, channel: 0 },
                spinwave_control::session::NoteSpec { note, velocity, start: skip + 0.1, duration: hold, channel: 0 },
            ];
            let stereo = session.render_samples(&notes, skip + seconds, 120.0);
            let skip_samples = ((skip * 44100.0) as usize) * 2;
            golden::write_raw(std::path::Path::new(output), &stereo[skip_samples.min(stereo.len())..])?;
            Ok(())
        }
        Some("raw-distance") => {
            // Two raw stereo f32 renders: sample RMS and peak of the
            // difference, and the ops band distance (dB).
            let a = golden::read_reference(std::path::Path::new(args.get(1).ok_or("usage: raw-distance <a.raw> <b.raw>")?))?;
            let b = golden::read_reference(std::path::Path::new(args.get(2).ok_or("usage: raw-distance <a.raw> <b.raw>")?))?;
            let difference = golden::compare(&a, &b)?;
            let distance = spinwave_control::ops::distance::distance(&a, &b, 44100, Default::default());
            println!(
                "{{\"rms\":{},\"peak\":{},\"reference_peak\":{},\"band_db\":{}}}",
                difference.rms, difference.peak, difference.reference_peak, distance.total_db
            );
            Ok(())
        }
        Some("bank") => {
            // Loads every .vital under a directory and reports what the
            // engine could not route: one line per preset with ignored
            // connections, then the ignored destinations by count. The
            // real bank (~/Documents/Vital, 75 presets) is the standard
            // for "loads": zero ignored connections on all of them.
            let dir = args.get(1).ok_or("usage: bank <dir> [--render]")?;
            // --render: also build every embedded wavetable through the
            // creator (its warnings name the components the construction
            // could not handle: a wavetable error, not a DSP one) and
            // render one C3 note, printing peak and loudness.
            let render = args.iter().any(|a| a == "--render");
            let mut paths = Vec::new();
            fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            walk(&path, out);
                        } else if path.extension().is_some_and(|e| e == "vital") {
                            out.push(path);
                        }
                    }
                }
            }
            walk(std::path::Path::new(dir), &mut paths);
            paths.sort();
            let mut session = spinwave_control::session::Session::with_output_dir(std::env::temp_dir());
            let mut by_destination: std::collections::BTreeMap<String, usize> = Default::default();
            let (mut clean, mut refused, mut failed) = (0usize, 0usize, 0usize);
            for path in &paths {
                let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let text = match std::fs::read_to_string(path) {
                    Ok(text) => text,
                    Err(e) => {
                        println!("{name:<32} FAILED: {e}");
                        failed += 1;
                        continue;
                    }
                };
                if let Err(e) = session.load_preset_json(text.trim_start_matches('\u{feff}')) {
                    println!("{name:<32} FAILED: {e}");
                    failed += 1;
                    continue;
                }
                if render {
                    let preset = session.preset.clone();
                    let mut wavetable_notes = Vec::new();
                    if let Some(tables) = preset.settings.wavetables.as_ref().and_then(|v| v.as_array()) {
                        for (i, table) in tables.iter().enumerate() {
                            match spinwave_dsp::wavetable::wavetable_from_json_with_warnings(table) {
                                Some((_, warnings)) => {
                                    for warning in warnings {
                                        wavetable_notes.push(format!("osc {}: {warning}", i + 1));
                                    }
                                }
                                None => wavetable_notes.push(format!("osc {}: wavetable did not build", i + 1)),
                            }
                        }
                    }
                    let scenario = spinwave_control::ops::Scenario::one_note(
                        48,
                        1.5,
                        2.5,
                        spinwave_control::ops::RenderMode::Faithful,
                    );
                    match spinwave_control::ops::measure::measure(&preset, &scenario, 0) {
                        Ok(measurement) => {
                            let d = &measurement.descriptors;
                            println!(
                                "{name:<32} peak {:>7.1} dBFS  loudness {:>7.1} LUFS  {}",
                                d.peak_dbfs,
                                d.loudness_lufs.unwrap_or(f32::NEG_INFINITY),
                                if wavetable_notes.is_empty() { String::new() } else { format!("WAVETABLE: {}", wavetable_notes.join("; ")) }
                            );
                        }
                        Err(e) => println!("{name:<32} RENDER FAILED: {e:?}"),
                    }
                }
                let ignored = &session.last_report.ignored_connections;
                if ignored.is_empty() {
                    clean += 1;
                } else {
                    refused += 1;
                    println!("{name:<32} {} ignored: {}", ignored.len(), ignored.join(", "));
                    for connection in ignored {
                        let destination = connection.rsplit(" -> ").next().unwrap_or(connection);
                        // Family key: osc_2_level -> osc_N_level.
                        let mut key = String::new();
                        for (i, part) in destination.split('_').enumerate() {
                            if i > 0 {
                                key.push('_');
                            }
                            key.push_str(if part.parse::<u32>().is_ok() { "N" } else { part });
                        }
                        *by_destination.entry(key).or_default() += 1;
                    }
                }
            }
            println!();
            println!("{} presets: {clean} load clean, {refused} with ignored connections, {failed} failed", paths.len());
            let mut rows: Vec<(String, usize)> = by_destination.into_iter().collect();
            rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let total: usize = rows.iter().map(|r| r.1).sum();
            if total > 0 {
                println!("{total} ignored connections by destination:");
                for (destination, count) in rows {
                    println!("  {count:4}  {destination}");
                }
            }
            Ok(())
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
        Some("judge") => {
            // Judges a patch (.vital or .spinwave) against one of the
            // ten-sounds targets. JSON verdict on stdout: the checks with
            // their thresholds, and the analysis. `--analysis-only` hides
            // the checks, which is what the loop condition feeds back to
            // the model: the measurement, never the pass/fail.
            let input = args.get(1).ok_or("usage: judge <patch> --target <id> [--reference <patch>] [--analysis-only]")?;
            let target_id = flag(&args, "--target").ok_or("judge needs --target <id>")?;
            let target = spinwave_control::judge::Target::from_id(&target_id).ok_or_else(|| {
                let ids: Vec<&str> = spinwave_control::judge::Target::ALL.iter().map(|t| t.id()).collect();
                format!("unknown target `{target_id}`; one of: {}", ids.join(", "))
            })?;
            let load = |path: &str| -> Result<spinwave_params::Preset, String> {
                let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                if path.ends_with(".spinwave") {
                    let blobs = spinwave_control::text_preset::Blobs::from_dir(&std::path::PathBuf::from(format!("{path}.d")))?;
                    let result = spinwave_control::text_preset::read(&text, &blobs);
                    result.preset.ok_or_else(|| format!("{path}: {}", result.report.summary()))
                } else {
                    spinwave_params::Preset::from_json(text.trim_start_matches('\u{feff}')).map_err(|e| format!("{path}: {e}"))
                }
            };
            let preset = load(input)?;
            let reference = match flag(&args, "--reference") {
                Some(path) => Some(load(&path)?),
                None => None,
            };
            let verdict = spinwave_control::judge::judge(&preset, target, reference.as_ref())?;
            if args.iter().any(|a| a == "--analysis-only") {
                println!("{}", serde_json::to_string_pretty(&verdict.analysis).unwrap_or_default());
            } else {
                println!("{}", serde_json::to_string_pretty(&verdict).unwrap_or_default());
            }
            if verdict.pass { Ok(()) } else { Err(format!("{}: not met", target.id())) }
        }
        Some("targets") => {
            // The twelve target descriptions, as the model receives them.
            for t in spinwave_control::judge::Target::ALL {
                println!("{:<22} {}", t.id(), t.description());
            }
            Ok(())
        }
        // -- The operations (notes/operations-design.md) ------------------
        // Every one prints one JSON document on stdout, the same struct the
        // MCP tool returns; a refusal is a JSON error with a `code`.
        Some("measure") => {
            let path = args.get(1).ok_or("usage: measure <patch> [--lite] [--notes 60:0.8,64] [--hold S] [--seconds S] [--bpm B] [--seed N] [--wav out.wav]")?;
            let preset = ops::load_patch(path)?;
            let scenario = scenario_from_args(&args);
            let seed = seed_from_args(&args);
            let (m, audio) = report(ops::measure::measure_with_audio(&preset, &scenario, seed))?;
            if let Some(out) = flag(&args, "--wav") {
                spinwave_plugin::materials::write_wav(&out, &audio, spinwave_control::session::SAMPLE_RATE).map_err(|e| format!("{out}: {e}"))?;
            }
            println!("{}", serde_json::to_string_pretty(&m).unwrap_or_default());
            Ok(())
        }
        Some("compare") => {
            let a = args.get(1).ok_or("usage: compare <a> <b> [--normalize-loudness] [scenario flags]")?;
            let b = args.get(2).ok_or("usage: compare <a> <b>")?;
            let (pa, pb) = (ops::load_patch(a)?, ops::load_patch(b)?);
            let options = ops::DistanceOptions { normalize_loudness: args.iter().any(|x| x == "--normalize-loudness") };
            let c = report(ops::compare(&pa, &pb, &scenario_from_args(&args), seed_from_args(&args), options))?;
            println!("{}", serde_json::to_string_pretty(&c).unwrap_or_default());
            Ok(())
        }
        Some("aliasing") => {
            // Two renders a semitone apart: the partials that do not follow
            // the key. `--quality aliasing` in explain / suggest uses the same.
            let path = args.get(1).ok_or("usage: aliasing <patch> [scenario flags]")?;
            let preset = ops::load_patch(path)?;
            let r = report(ops::aliasing(&preset, &scenario_from_args(&args), seed_from_args(&args)))?;
            println!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
            Ok(())
        }
        Some("explain") => {
            let path = args.get(1).ok_or("usage: explain <patch> --quality Q [--max-renders N] [scenario flags]")?;
            let preset = ops::load_patch(path)?;
            let quality = quality_from_args(&args)?;
            let e = report(ops::explain(&preset, &scenario_from_args(&args), quality, seed_from_args(&args), budget_from_args(&args)))?;
            println!("{}", serde_json::to_string_pretty(&e).unwrap_or_default());
            Ok(())
        }
        Some("suggest") => {
            let path = args.get(1).ok_or("usage: suggest <patch> --quality Q --more|--less [--switches] [scenario flags]")?;
            let preset = ops::load_patch(path)?;
            let quality = quality_from_args(&args)?;
            let direction = if args.iter().any(|x| x == "--less") { ops::Direction::Less } else { ops::Direction::More };
            let switches = args.iter().any(|x| x == "--switches");
            let s = report(ops::suggest(&preset, &scenario_from_args(&args), quality, direction, switches, seed_from_args(&args), budget_from_args(&args)))?;
            println!("{}", serde_json::to_string_pretty(&s).unwrap_or_default());
            Ok(())
        }
        Some("apply") => {
            // `apply <patch> <diff.spinwave>` or `apply <patch> --set name=value --set ...`;
            // `--goal brightness:more` checks the direction; `--out` writes the result.
            let path = args.get(1).ok_or("usage: apply <patch> [<diff.spinwave>] [--set name=value]... [--goal Q:more|less] [--out patch] [scenario flags]")?;
            let preset = ops::load_patch(path)?;
            let sets: Vec<ops::Change> = args
                .windows(2)
                .filter(|w| w[0] == "--set")
                .filter_map(|w| {
                    let (name, value) = w[1].split_once('=')?;
                    let value = match value.parse::<f32>() {
                        Ok(v) => ops::diff::ChangeValue::Engine(v),
                        Err(_) => ops::diff::ChangeValue::Text(value.to_string()),
                    };
                    Some(ops::Change { name: name.to_string(), value })
                })
                .collect();
            let diff = if let Some(fragment) = args.get(2).filter(|a| !a.starts_with("--")) {
                ops::Diff::Fragment(std::fs::read_to_string(fragment).map_err(|e| format!("{fragment}: {e}"))?)
            } else {
                ops::Diff::Changes(sets)
            };
            let goal = match flag(&args, "--goal") {
                Some(spec) => {
                    let (q, d) = spec.split_once(':').ok_or("--goal takes quality:more|less")?;
                    let quality = ops::Quality::from_id(q).ok_or_else(|| format!("unknown quality `{q}`"))?;
                    Some((quality, if d == "less" { ops::Direction::Less } else { ops::Direction::More }))
                }
                None => None,
            };
            let applied = report(ops::apply(&preset, &diff, &scenario_from_args(&args), goal, seed_from_args(&args)))?;
            if let Some(out) = flag(&args, "--out") {
                ops::save_patch(&applied.preset, &out)?;
            }
            // The patch itself is on disk (or in `--out`); the report stands alone.
            let mut json = serde_json::to_value(&applied).unwrap_or_default();
            json.as_object_mut().map(|o| o.remove("preset"));
            println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
            Ok(())
        }
        Some("explore") => {
            let path = args.get(1).ok_or("usage: explore <patch> --count N [--amplitude A] [--seed S] [--switch-indexed P] --out DIR [scenario flags]")?;
            let preset = ops::load_patch(path)?;
            let spec = ops::ExploreSpec {
                count: flag(&args, "--count").and_then(|v| v.parse().ok()).unwrap_or(8),
                amplitude: flag(&args, "--amplitude").and_then(|v| v.parse().ok()).unwrap_or(0.25),
                seed: seed_from_args(&args),
                switch_indexed: flag(&args, "--switch-indexed").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                budget: budget_from_args(&args),
                prior: match flag(&args, "--prior").as_deref() {
                    Some("live") => ops::Prior::Live,
                    Some("measured") => ops::Prior::Measured,
                    Some("measured-then-live") | None => ops::Prior::MeasuredThenLive,
                    Some(other) => return Err(format!("--prior {other}: live | measured | measured-then-live")),
                },
                free_ranges: args.iter().any(|a| a == "--free-ranges"),
            };
            let e = report(ops::explore(&preset, &scenario_from_args(&args), &spec))?;
            let out = flag(&args, "--out");
            if let Some(dir) = &out {
                std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
            }
            let mut json = serde_json::to_value(&e).unwrap_or_default();
            for (i, v) in e.variants.iter().enumerate() {
                if let Some(dir) = &out {
                    let file = format!("{dir}/variant_{:03}.spinwave", v.index);
                    ops::save_patch(&v.preset, &file)?;
                    json["variants"][i]["path"] = serde_json::Value::from(file);
                }
                json["variants"][i].as_object_mut().map(|o| o.remove("preset"));
            }
            println!("{}", serde_json::to_string_pretty(&json).unwrap_or_default());
            Ok(())
        }
        Some("interpolate") => {
            let a = args.get(1).ok_or("usage: interpolate <a> <b> --steps N --out DIR")?;
            let b = args.get(2).ok_or("usage: interpolate <a> <b> --steps N --out DIR")?;
            let (pa, pb) = (ops::load_patch(a)?, ops::load_patch(b)?);
            let steps: usize = flag(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(5);
            let ts: Vec<f32> = (0..steps).map(|i| i as f32 / (steps.max(2) - 1) as f32).collect();
            let patches = report(ops::interpolate(&pa, &pb, &ts))?;
            let dir = flag(&args, "--out").ok_or("interpolate needs --out DIR")?;
            std::fs::create_dir_all(&dir).map_err(|e| format!("{dir}: {e}"))?;
            let mut files = Vec::new();
            for (t, p) in ts.iter().zip(&patches) {
                let file = format!("{dir}/t_{:.3}.spinwave", t);
                ops::save_patch(p, &file)?;
                files.push(serde_json::json!({ "t": t, "path": file }));
            }
            println!("{}", serde_json::to_string_pretty(&serde_json::Value::Array(files)).unwrap_or_default());
            Ok(())
        }
        Some("fingerprint") => {
            let stamp = spinwave_control::knowledge::engine_stamp();
            println!("{}", serde_json::to_string_pretty(&stamp).unwrap_or_default());
            Ok(())
        }
        Some("knowledge") => {
            use spinwave_control::knowledge;
            let dir = flag(&args, "--dir").map(std::path::PathBuf::from).unwrap_or_else(knowledge::knowledge_dir);
            // `--patches DIR`: every .vital / .spinwave under it, labelled
            // by file stem, as measurement contexts.
            let patches = |args: &[String]| -> Result<Vec<(String, std::path::PathBuf)>, String> {
                let Some(root) = flag(args, "--patches") else { return Ok(Vec::new()) };
                let mut found = Vec::new();
                fn walk(dir: &std::path::Path, out: &mut Vec<(String, std::path::PathBuf)>) {
                    let Ok(entries) = std::fs::read_dir(dir) else { return };
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            walk(&path, out);
                        } else if path.extension().is_some_and(|e| e == "vital" || e == "spinwave") {
                            out.push((path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(), path));
                        }
                    }
                }
                walk(std::path::Path::new(&root), &mut found);
                found.sort();
                if found.is_empty() {
                    return Err(format!("{root}: no .vital or .spinwave under it"));
                }
                Ok(found)
            };
            match args.get(1).map(String::as_str) {
                Some("measure") => {
                    let options = knowledge::MeasureOptions {
                        canonical: args.iter().any(|a| a == "--canonical"),
                        patches: patches(&args)?,
                        only: flag(&args, "--only"),
                        stale_only: args.iter().any(|a| a == "--stale-only"),
                        seed: seed_from_args(&args),
                        budget: budget_from_args(&args),
                    };
                    if !options.canonical && options.patches.is_empty() {
                        return Err("knowledge measure: --canonical and/or --patches DIR".into());
                    }
                    let report = knowledge::measure(&dir, &options, |line| eprintln!("{line}"))?;
                    println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
                    Ok(())
                }
                Some("status") => {
                    println!("{}", serde_json::to_string_pretty(&knowledge::status(&dir)).unwrap_or_default());
                    Ok(())
                }
                Some("corpus") => {
                    let list = patches(&args)?;
                    let id = flag(&args, "--id").unwrap_or_else(|| "corpus".into());
                    let min_quality = !args.iter().any(|a| a == "--keep-all");
                    let (structure, catalogue) = knowledge::corpus::build(&id, &list, min_quality, |line| eprintln!("{line}"))?;
                    knowledge::corpus::save(&dir, &structure, &catalogue)?;
                    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                        "corpus": structure.corpus,
                        "modules_on": structure.modules_on,
                        "destinations_modulated": structure.destinations_modulated.len(),
                        "value_ranges_used": structure.value_ranges_used.len(),
                        "connections_per_patch": structure.connections_per_patch,
                    })).unwrap_or_default());
                    Ok(())
                }
                Some("agreement") => {
                    let store = knowledge::Store::load(&dir);
                    if store.is_empty() {
                        return Err(format!("{}: no measured store; run knowledge measure first", dir.display()));
                    }
                    let list = patches(&args)?;
                    let report = knowledge::agreement(&store, &list, seed_from_args(&args), budget_from_args(&args))?;
                    for p in &report.patches {
                        eprintln!("{:<32} active {:>3} known {:>3} renders {:>3} -> {:>3} spearman {}", p.label, p.active, p.known, p.renders_live, p.renders_prior, p.spearman.map(|r| format!("{r:.3}")).unwrap_or_else(|| "n/a".into()));
                    }
                    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                        "patches": report.patches.len(),
                        "mean_spearman": report.mean_spearman,
                        "renders_live": report.renders_live,
                        "renders_prior": report.renders_prior,
                    })).unwrap_or_default());
                    Ok(())
                }
                _ => Err("usage: knowledge measure [--canonical] [--patches DIR] [--only SUBSTR] [--stale-only] | status | agreement --patches DIR | corpus --patches DIR --id ID [--keep-all]   [--dir KNOWLEDGE]".into()),
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
        _ => Err("usage: spinwave-cli render <preset> <out.wav> | analyze <file> | fuzz [--count N] [--seed S] [--wildness full|sparse] [--save-failures DIR] | golden [--case NAME] [--probe SRC] | sensitivity [--only SUBSTR] | to-text <in.vital> <out.spinwave> | from-text <in.spinwave> <out.vital> | check <in.spinwave> | judge <patch> --target ID [--reference P] [--analysis-only] | targets | measure <patch> | aliasing <patch> | compare <a> <b> | explain <patch> --quality Q | suggest <patch> --quality Q --more|--less | apply <patch> [diff] [--set n=v] | explore <patch> --count N --out DIR [--prior live|measured|measured-then-live] | interpolate <a> <b> --steps N --out DIR | knowledge measure|status|agreement | fingerprint   (scenario flags: --lite --notes 60:0.8,64 --hold S --seconds S --bpm B --seed N --max-renders N --max-seconds S)".to_string()),
    }
}

/// A refusal prints as JSON with its code and exits non-zero, so a caller
/// can act on it without parsing prose.
fn report<T>(result: Result<T, ops::OpError>) -> Result<T, String> {
    result.map_err(|e| {
        let json = serde_json::to_string(&e).unwrap_or_default();
        format!("refused: {e}\n{json}")
    })
}

fn scenario_from_args(args: &[String]) -> ops::Scenario {
    let lite = args.iter().any(|a| a == "--lite");
    let mut scenario = if lite { ops::Scenario::lite() } else { ops::Scenario::faithful() };
    if let Some(seconds) = flag(args, "--seconds").and_then(|v| v.parse().ok()) {
        scenario.seconds = seconds;
    }
    let hold = flag(args, "--hold").and_then(|v| v.parse().ok());
    if let Some(notes) = flag(args, "--notes") {
        scenario.notes = parse_notes(&notes, scenario.seconds, hold);
    } else if let Some(hold) = hold {
        for n in &mut scenario.notes {
            n.duration = hold;
        }
    }
    if let Some(bpm) = flag(args, "--bpm").and_then(|v| v.parse().ok()) {
        scenario.bpm = bpm;
    }
    scenario
}

fn seed_from_args(args: &[String]) -> u64 {
    flag(args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(0)
}

fn budget_from_args(args: &[String]) -> ops::Budget {
    let mut budget = ops::Budget::default();
    if let Some(n) = flag(args, "--max-renders").and_then(|v| v.parse().ok()) {
        budget.max_renders = n;
    }
    if let Some(s) = flag(args, "--max-seconds").and_then(|v| v.parse().ok()) {
        budget.max_seconds = s;
    }
    budget
}

fn quality_from_args(args: &[String]) -> Result<ops::Quality, String> {
    let id = flag(args, "--quality").ok_or("needs --quality Q")?;
    ops::Quality::from_id(&id).ok_or_else(|| format!("unknown quality `{id}`; one of: {}", ops::Quality::ALL.join(", ")))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("spinwave-cli: {e}");
        std::process::exit(1);
    }
}
