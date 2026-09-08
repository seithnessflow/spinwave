//! Renders a `.vital` preset to a WAV file through the full voice kernel.
//!
//! Usage: `cargo run -p spinwave-plugin --example render_preset [preset.vital]`
//! Without an argument, renders a built-in demo preset (wavetable saw with
//! unison, serial filters, LFO-swept cutoff) to `spinwave-preset-demo.wav`.

use std::io::Write;

use spinwave_engine::VoiceAllocator;
use spinwave_engine::kernel::SynthVoiceKernel;
use spinwave_params::Preset;
use spinwave_plugin::patch::{connections_from_preset, kernel_params_from_preset};
use spinwave_poly::constants::MAX_BUFFER_SIZE;
use spinwave_poly::PolyF32;

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
    let preset = Preset::from_json(&json).expect("parse preset");
    println!("preset: {}", preset.preset_name);

    let params = kernel_params_from_preset(&preset);
    let connections = connections_from_preset(&preset);
    println!("modulation connections mapped: {}", connections.len());

    let mut allocator = VoiceAllocator::new(16, || {
        let mut kernel = SynthVoiceKernel::new(SAMPLE_RATE);
        kernel.params = params.clone();
        kernel.matrix.connections = connections.clone();
        kernel
    });
    allocator.set_sample_rate(SAMPLE_RATE);

    let notes = [
        (0.0, 1.8, 45, 0.9),
        (0.2, 1.6, 57, 0.8),
        (0.4, 1.4, 64, 0.8),
        (2.0, 1.5, 50, 0.9),
        (2.2, 1.3, 62, 0.8),
        (2.4, 1.1, 69, 0.8),
    ];

    let total_samples = (4.5 * SAMPLE_RATE as f32) as usize;
    let mut stereo = Vec::with_capacity(total_samples * 2);
    let mut position = 0usize;
    while position < total_samples {
        let block = MAX_BUFFER_SIZE.min(total_samples - position);
        for &(start, duration, note, velocity) in &notes {
            let start_sample = (start * SAMPLE_RATE as f32) as usize;
            let end_sample = ((start + duration) * SAMPLE_RATE as f32) as usize;
            if start_sample >= position && start_sample < position + block {
                allocator.note_on(note, velocity, start_sample - position, 0);
            }
            if end_sample >= position && end_sample < position + block {
                allocator.note_off(note, 0.5, end_sample - position, 0);
            }
        }

        let mut mix = vec![PolyF32::ZERO; block];
        allocator.process(block, |out| {
            for (dest, src) in mix.iter_mut().zip(out) {
                *dest += *src;
            }
        });
        for value in &mix {
            let folded = *value + value.swap_voices();
            stereo.push(folded.lane(0) * 0.5);
            stereo.push(folded.lane(1) * 0.5);
        }
        position += block;
    }

    let peak = stereo.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let rms = (stereo.iter().map(|v| v * v).sum::<f32>() / stereo.len() as f32).sqrt();
    let non_finite = stereo.iter().filter(|v| !v.is_finite()).count();
    println!("peak {peak:.4}, rms {rms:.4}, non-finite {non_finite}");
    assert_eq!(non_finite, 0);
    assert!(peak > 0.01, "silent render");

    write_wav("spinwave-preset-demo.wav", &stereo, SAMPLE_RATE);
    println!("wrote spinwave-preset-demo.wav");
}

/// Minimal 32-bit float stereo WAV writer.
fn write_wav(path: &str, interleaved: &[f32], sample_rate: u32) {
    let mut file = std::fs::File::create(path).expect("create wav");
    let data_bytes = (interleaved.len() * 4) as u32;
    let byte_rate = sample_rate * 2 * 4;

    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    header.extend_from_slice(b"WAVEfmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&3u16.to_le_bytes());
    header.extend_from_slice(&2u16.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&8u16.to_le_bytes());
    header.extend_from_slice(&32u16.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_bytes.to_le_bytes());
    file.write_all(&header).expect("write header");
    for value in interleaved {
        file.write_all(&value.to_le_bytes()).expect("write sample");
    }
}
