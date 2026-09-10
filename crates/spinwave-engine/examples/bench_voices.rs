//! Worst-case CPU probe: how much realtime headroom the engine keeps as
//! polyphony grows, with everything a live patch turns on.
//!
//! Usage: `cargo run --release -p spinwave-engine --example bench_voices`
//!
//! Prints the realtime factor per voice count, and — the number that
//! actually decides whether a patch crackles — the WORST single block.
//!
//! The average is the reassuring number and the wrong one. A host hands
//! the engine one block and a deadline; miss it once and there is an
//! audible click, however comfortable the mean was. So this reports the
//! mean, the 99th percentile and the maximum as a fraction of the block's
//! own deadline (its duration at the sample rate). Anything approaching
//! 100% on the max will click under load, and a mean of "1.1x realtime"
//! means the margin is already gone.

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

/// What one voice count costs: the mean realtime factor, and the block
/// deadline usage at the 99th percentile and at the worst block.
struct Cost {
    realtime: f32,
    p99_percent: f32,
    max_percent: f32,
    /// Worst block of the block every voice STARTS in. Voices allocate,
    /// reset their filters and fill their first buffers all at once there,
    /// and a host gives that block no more time than any other.
    onset_percent: f32,
    /// Worst block deep in the release tail, where levels decay toward
    /// zero. Denormal arithmetic can cost tens of times normal there, and
    /// nothing in this engine sets flush-to-zero, so this column is the
    /// one that says whether that is theory or a real cliff.
    tail_percent: f32,
}

fn block_percent(elapsed: f64, block: usize) -> f32 {
    let deadline = block as f64 / SAMPLE_RATE as f64;
    (elapsed / deadline) as f32 * 100.0
}

fn measure(polyphony: usize, audio_rate: bool) -> Cost {
    let mut engine = build_engine(polyphony, audio_rate);
    let mut left = vec![0.0f32; BLOCK];
    let mut right = vec![0.0f32; BLOCK];

    // Every voice starts in the same block, which is both the realistic
    // worst case (a chord, a sequencer step) and the block a mean hides.
    for i in 0..polyphony {
        engine.note_on(36 + (i as i32 % 40), 0.9, 0, 0);
    }
    let onset_start = Instant::now();
    engine.process(BLOCK, &mut left, &mut right);
    let onset_percent = block_percent(onset_start.elapsed().as_secs_f64(), BLOCK);

    let total = (SECONDS * SAMPLE_RATE as f32) as usize;
    // One pass to settle envelopes and fill the lookup caches.
    for _ in 0..(SAMPLE_RATE as usize / BLOCK) {
        engine.process(BLOCK, &mut left, &mut right);
    }
    // Time every block separately: the distribution is the point, and a
    // single total would hide the one block that overruns.
    let mut block_times = Vec::with_capacity(total / BLOCK + 1);
    let start = Instant::now();
    let mut done = 0usize;
    while done < total {
        let block = BLOCK.min(total - done);
        let block_start = Instant::now();
        engine.process(block, &mut left[..block], &mut right[..block]);
        block_times.push((block_start.elapsed().as_secs_f64(), block));
        done += block;
    }
    let realtime = SECONDS / start.elapsed().as_secs_f32();

    // Each block's cost as a fraction of its own deadline, so a short
    // final block is judged against the time IT had, not a full one.
    let mut usage: Vec<f32> = block_times
        .iter()
        .map(|&(elapsed, block)| {
            let deadline = block as f64 / SAMPLE_RATE as f64;
            (elapsed / deadline) as f32 * 100.0
        })
        .collect();
    usage.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99 = usage[(usage.len() as f32 * 0.99) as usize % usage.len()];
    let max = *usage.last().unwrap_or(&0.0);

    // Release everything and run well past the tail's audible life. Levels
    // fall through the denormal range on the way down; if that costs, it
    // costs here.
    for i in 0..polyphony {
        engine.note_off(36 + (i as i32 % 40), 0.5, 0, 0);
    }
    let mut tail = 0.0f32;
    for _ in 0..(4 * SAMPLE_RATE as usize / BLOCK) {
        let start = Instant::now();
        engine.process(BLOCK, &mut left, &mut right);
        tail = tail.max(block_percent(start.elapsed().as_secs_f64(), BLOCK));
    }

    Cost {
        realtime,
        p99_percent: p99,
        max_percent: max,
        onset_percent,
        tail_percent: tail,
    }
}

fn main() {
    println!(
        "{:>6}  {:>9}  {:>7}  {:>7}  {:>7}  {:>7}",
        "voices", "realtime", "p99", "worst", "onset", "tail"
    );
    for &polyphony in &[1usize, 4, 8, 16, 32, 64] {
        let audio = measure(polyphony, true);
        println!(
            "{polyphony:>6}  {:>8.1}x  {:>6.0}%  {:>6.0}%  {:>6.0}%  {:>6.0}%",
            audio.realtime,
            audio.p99_percent,
            audio.max_percent,
            audio.onset_percent,
            audio.tail_percent
        );
    }
    println!();
    println!("Every percentage is one block's cost as a share of ITS OWN deadline.");
    println!("Past 100% the engine has missed it, and a missed block is a click.");
    println!("`onset` is the block every voice starts in; `tail` is the worst block");
    println!("of a four-second release, where denormals would show. Nothing in this");
    println!("engine sets flush-to-zero, so `tail` is the column that tells you");
    println!("whether that matters in practice.");
}

