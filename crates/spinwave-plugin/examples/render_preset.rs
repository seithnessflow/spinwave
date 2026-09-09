//! Renders a `.vital` preset to a WAV file through the complete engine
//! (voices, modulation, bus effects, master path, 2x oversampling).
//!
//! Usage: `cargo run -p spinwave-plugin --example render_preset [preset.vital]`
//! Without an argument, renders a built-in demo preset to
//! `spinwave-preset-demo.wav`.

use spinwave_engine::engine::SoundEngine;
use spinwave_params::Preset;
use spinwave_plugin::materials::write_wav;
use spinwave_plugin::{apply_preset, patch::load_preset};

const SAMPLE_RATE: u32 = 44100;

const DEMO_PRESET: &str = r#"{
  "synth_version": "1.0.7",
  "preset_name": "Spinwave Demo",
  "author": "spinwave",
  "settings": {
    "osc_1_on": 1.0,
    "osc_1_level": 0.6,
    "osc_1_unison_voices": 5.0,
    "osc_1_unison_detune": 2.5,
    "osc_2_on": 1.0,
    "osc_2_level": 0.35,
    "osc_2_transpose": -12.0,
    "osc_2_destination": 0.0,
    "filter_1_on": 1.0,
    "filter_1_model": 3.0,
    "filter_1_cutoff": 70.0,
    "filter_1_resonance": 0.45,
    "env_1_attack": 0.35,
    "env_1_decay": 1.0,
    "env_1_sustain": 0.75,
    "env_1_release": 0.85,
    "lfo_1_frequency": 1.0,
    "modulation_1_amount": 0.7,
    "delay_on": 1.0,
    "delay_sync": 1.0,
    "delay_tempo": 9.0,
    "delay_feedback": 0.4,
    "delay_dry_wet": 0.3,
    "reverb_on": 1.0,
    "reverb_dry_wet": 0.35,
    "reverb_decay_time": 0.5,
    "modulations": [
      {"source": "lfo_1", "destination": "filter_1_cutoff"}
    ]
  }
}"#;

fn main() {
    let path = std::env::args().nth(1);
    let json = match &path {
        Some(path) => std::fs::read_to_string(path).expect("read preset file"),
        None => DEMO_PRESET.to_string(),
    };
    let (preset, report): (Preset, _) = load_preset(&json).expect("parse preset");
    println!("preset: {}", preset.preset_name);
    if !report.is_clean() {
        println!("load report: {}", report.summary());
    }

    let mut engine = SoundEngine::new(SAMPLE_RATE);
    engine.set_bpm(120.0);
    apply_preset(&preset, &mut engine);
    println!(
        "polyphony {}, delay on: {}, reverb on: {}",
        engine.allocator().polyphony(),
        engine.params().delay_on,
        engine.params().reverb_on,
    );

    let notes = [
        (0.0, 1.8, 45, 0.9),
        (0.2, 1.6, 57, 0.8),
        (0.4, 1.4, 64, 0.8),
        (2.0, 1.5, 50, 0.9),
        (2.2, 1.3, 62, 0.8),
        (2.4, 1.1, 69, 0.8),
    ];

    let total_samples = (6.0 * SAMPLE_RATE as f32) as usize;
    let block_size = 128usize;
    let mut stereo = Vec::with_capacity(total_samples * 2);
    let mut left = vec![0.0f32; block_size];
    let mut right = vec![0.0f32; block_size];

    let mut position = 0usize;
    while position < total_samples {
        let block = block_size.min(total_samples - position);
        for &(start, duration, note, velocity) in &notes {
            let start_sample = (start * SAMPLE_RATE as f32) as usize;
            let end_sample = ((start + duration) * SAMPLE_RATE as f32) as usize;
            if start_sample >= position && start_sample < position + block {
                engine.note_on(note, velocity, start_sample - position, 0);
            }
            if end_sample >= position && end_sample < position + block {
                engine.note_off(note, 0.5, end_sample - position, 0);
            }
        }

        engine.process(block, &mut left[..block], &mut right[..block]);
        for i in 0..block {
            stereo.push(left[i]);
            stereo.push(right[i]);
        }
        position += block;
    }

    let peak = stereo.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let rms = (stereo.iter().map(|v| v * v).sum::<f32>() / stereo.len() as f32).sqrt();
    let non_finite = stereo.iter().filter(|v| !v.is_finite()).count();
    println!("peak {peak:.4}, rms {rms:.4}, non-finite {non_finite}");
    assert_eq!(non_finite, 0);
    assert!(peak > 0.01, "silent render");

    // Reverb + delay must leave a tail after the last note-off (~3.5 s).
    let tail_start = (5.0 * SAMPLE_RATE as f32) as usize * 2;
    let tail_rms = (stereo[tail_start..].iter().map(|v| v * v).sum::<f32>()
        / (stereo.len() - tail_start) as f32)
        .sqrt();
    println!("tail rms (5.0s..6.0s): {tail_rms:.6}");

    write_wav("spinwave-preset-demo.wav", &stereo, SAMPLE_RATE).expect("write wav");
    println!("wrote spinwave-preset-demo.wav");
}
