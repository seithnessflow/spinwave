//! Worst-case CPU probe: how much realtime headroom the engine keeps as
//! polyphony grows, with everything a live patch turns on.
//!
//! Usage: `cargo run --release -p spinwave-engine --example bench_voices`
//!
//! Prints the realtime factor per voice count. A factor of 1.0 means the
//! engine needs exactly one second of CPU per second of audio (dropouts);
//! anything under ~4x is uncomfortable for live playing, because a host
//! block must finish well inside its deadline.

use std::time::Instant;

use spinwave_dsp::modulators::EnvelopeParams;
use spinwave_engine::engine::SoundEngine;
use spinwave_engine::kernel::mod_matrix::{Connection, ModDest, ModSource};
use spinwave_engine::modulation::ModulationTransform;
use spinwave_poly::PolyF32;

const SAMPLE_RATE: u32 = 48_000;
const BLOCK: usize = 128;
const SECONDS: f32 = 4.0;

/// A patch that exercises the expensive paths: unison oscillators, both
/// filters, every envelope and LFO running, and an audio-rate modulation
/// on the filter cutoff (the per-sample matrix path).
fn build_engine(polyphony: usize, audio_rate: bool) -> SoundEngine {
    let mut engine = SoundEngine::new(SAMPLE_RATE);
    engine.set_polyphony(polyphony);
    engine.set_bpm(120.0);

    engine.kernel_params_mut(|params| {
        for osc in params.oscillators.iter_mut() {
            osc.on = true;
            osc.params.unison_voices = 7;
            osc.params.unison_detune = PolyF32::splat(4.0);
            osc.params.amplitude = PolyF32::splat(0.4);
        }
        params.filters[0].params.on = true;
        params.filters[1].params.on = true;
        for env in params.envelopes.iter_mut() {
            *env = EnvelopeParams {
                attack: PolyF32::splat(0.2),
                decay: PolyF32::splat(0.6),
                sustain: PolyF32::splat(0.8),
                release: PolyF32::splat(0.5),
                ..Default::default()
            };
        }
        for lfo in params.lfos.iter_mut() {
            lfo.params.frequency = PolyF32::splat(2.0);
        }
    });

    // One audio-rate connection (envelope -> cutoff) plus a control-rate
    // one per LFO, so both matrix paths run.
    let mut connections = Vec::new();
    if audio_rate {
        connections.push(Connection {
            source: ModSource::Envelope(1),
            dest: ModDest::FilterCutoff(0),
            transform: ModulationTransform::with_amount(0.8, 128.0),
        });
    }
    for lfo in 0..4 {
        connections.push(Connection {
            source: ModSource::Lfo(lfo),
            dest: ModDest::OscLevel(lfo % 4),
            transform: ModulationTransform::with_amount(0.3, 1.0),
        });
    }
    for kernel in engine.allocator_mut().kernels_mut() {
        kernel.matrix.set_connections(&connections);
    }
    engine
}

fn measure(polyphony: usize, audio_rate: bool) -> f32 {
    let mut engine = build_engine(polyphony, audio_rate);
    for i in 0..polyphony {
        engine.note_on(36 + (i as i32 % 40), 0.9, 0, 0);
    }
    let total = (SECONDS * SAMPLE_RATE as f32) as usize;
    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];
    // One pass to settle envelopes and fill the lookup caches.
    for _ in 0..(SAMPLE_RATE as usize / BLOCK) {
        engine.process(BLOCK, &mut left, &mut right);
    }
    let start = Instant::now();
    let mut done = 0usize;
    while done < total {
        let block = BLOCK.min(total - done);
        engine.process(block, &mut left[..block], &mut right[..block]);
        done += block;
    }
    SECONDS / start.elapsed().as_secs_f32()
}

fn main() {
    println!("{:>6}  {:>10}  {:>12}  {:>9}", "voices", "realtime", "control-rate", "audio cost");
    for &polyphony in &[1usize, 4, 8, 16, 32, 64] {
        let audio = measure(polyphony, true);
        let control = measure(polyphony, false);
        let cost = (control / audio - 1.0) * 100.0;
        println!("{polyphony:>6}  {audio:>9.1}x  {control:>11.1}x  {cost:>8.0}%");
    }
}

