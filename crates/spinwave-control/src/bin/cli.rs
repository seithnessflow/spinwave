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
use spinwave_control::decode::decode_file;
use spinwave_control::session::{NoteSpec, Session};

fn flag(args: &[String], name: &str) -> Option<String> {
    let position = args.iter().position(|a| a == name)?;
    args.get(position + 1).cloned()
}

/// `45,57,64` or `45:0.9,57:0.8` (note or note:velocity), held for the
/// whole render minus a release tail.
fn parse_notes(spec: &str, seconds: f32) -> Vec<NoteSpec> {
    let hold = (seconds - 1.0).max(0.2);
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
        Some("analyze") => {
            let path = args.get(1).ok_or("usage: analyze <file.wav>")?;
            let start = flag(&args, "--start").and_then(|v| v.parse().ok());
            let duration = flag(&args, "--duration").and_then(|v| v.parse().ok());
            let (stereo, sample_rate) = decode_file(path, start, duration)?;
            print_analysis(path, &analyze(&stereo, sample_rate));
            Ok(())
        }
        _ => Err("usage: spinwave-cli render <preset> <out.wav> | analyze <file>".to_string()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("spinwave-cli: {e}");
        std::process::exit(1);
    }
}
