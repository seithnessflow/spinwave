//! Offline render smoke test: plays a chord and a melody through the full
//! voice kernel and writes `spinwave-demo.wav` next to the workspace root.
//!
//! Run: `cargo run -p spinwave-engine --example render_demo`

use std::io::Write;

use spinwave_dsp::modulators::EnvelopeParams;
use spinwave_engine::kernel::{Connection, ModDest, ModSource, SynthVoiceKernel};
use spinwave_engine::modulation::ModulationTransform;
use spinwave_engine::VoiceAllocator;
use spinwave_poly::constants::MAX_BUFFER_SIZE;
use spinwave_poly::PolyF32;

const SAMPLE_RATE: u32 = 44100;

fn make_kernel() -> SynthVoiceKernel {
    let mut kernel = SynthVoiceKernel::new(SAMPLE_RATE);
    kernel.params.oscillators[0].on = true;
    kernel.params.oscillators[0].params.amplitude = PolyF32::splat(0.7);
    kernel.params.oscillators[0].params.unison_voices = 4;
    kernel.params.oscillators[0].params.unison_detune = PolyF32::splat(0.3);
    kernel.params.envelopes[0] = EnvelopeParams {
        attack: PolyF32::splat(0.01),
        decay: PolyF32::splat(0.4),
        sustain: PolyF32::splat(0.7),
        release: PolyF32::splat(0.4),
        ..Default::default()
    };
    // Low-pass filter swept by LFO 1.
    kernel.params.filters[0].params.on = true;
    kernel.params.filters[0].params.state.midi_cutoff = PolyF32::splat(80.0);
    kernel.params.filters[0].params.model = spinwave_engine::kernel::FilterModel::Digital;
    kernel.params.lfos[0].params.frequency = PolyF32::splat(0.5);
    kernel.matrix.connections.push(Connection {
        source: ModSource::Lfo(0),
        dest: ModDest::FilterCutoff(0),
        transform: ModulationTransform::with_amount(0.8, 48.0),
    });
    kernel
}

fn main() {
    let mut allocator = VoiceAllocator::new(16, make_kernel);
    allocator.set_sample_rate(SAMPLE_RATE);

    // (start_seconds, duration_seconds, midi_note, velocity)
    let notes = [
        (0.0, 2.5, 48, 0.9),
        (0.0, 2.5, 60, 0.8),
        (0.0, 2.5, 64, 0.8),
        (0.0, 2.5, 67, 0.8),
        (2.6, 0.4, 72, 0.9),
        (3.0, 0.4, 74, 0.9),
        (3.4, 0.9, 76, 0.9),
    ];

    let total_seconds = 5.0;
    let total_samples = (total_seconds * SAMPLE_RATE as f32) as usize;
    let mut stereo = Vec::with_capacity(total_samples * 2);

    let mut sample_position = 0usize;
    while sample_position < total_samples {
        let block = MAX_BUFFER_SIZE.min(total_samples - sample_position);
        let time = |s: usize| s as f32 / SAMPLE_RATE as f32;

        for &(start, duration, note, velocity) in &notes {
            let start_sample = (start * SAMPLE_RATE as f32) as usize;
            let end_sample = ((start + duration) * SAMPLE_RATE as f32) as usize;
            if start_sample >= sample_position && start_sample < sample_position + block {
                allocator.note_on(note, velocity, start_sample - sample_position, 0);
            }
            if end_sample >= sample_position && end_sample < sample_position + block {
                allocator.note_off(note, 0.5, end_sample - sample_position, 0);
            }
        }
        let _ = time;

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
        sample_position += block;
    }

    let peak = stereo.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let rms = (stereo.iter().map(|v| v * v).sum::<f32>() / stereo.len() as f32).sqrt();
    let non_finite = stereo.iter().filter(|v| !v.is_finite()).count();
    println!("rendered {} samples, peak {peak:.4}, rms {rms:.4}, non-finite {non_finite}", stereo.len());
    assert_eq!(non_finite, 0, "non-finite samples in render");
    assert!(peak > 0.01, "render is silent");
    assert!(peak <= 1.0, "render clips: peak {peak}");

    write_wav("spinwave-demo.wav", &stereo, SAMPLE_RATE);
    println!("wrote spinwave-demo.wav");
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
    header.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    header.extend_from_slice(&2u16.to_le_bytes()); // stereo
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&8u16.to_le_bytes()); // block align
    header.extend_from_slice(&32u16.to_le_bytes()); // bits
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_bytes.to_le_bytes());
    file.write_all(&header).expect("write header");
    for value in interleaved {
        file.write_all(&value.to_le_bytes()).expect("write sample");
    }
}
