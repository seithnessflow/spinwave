//! The cost of a render, published as ratios (the CPU rule: no absolute
//! number from a noisy machine): Lite against Faithful, and the parallel
//! throughput against the single-thread one. Run twice back to back and
//! trust the ratios, not the milliseconds.
use std::time::Instant;

use spinwave_control::ops::{self, explore, Budget, ExploreSpec, Scenario};
use spinwave_params::Preset;

fn main() {
    let mut p = Preset::default();
    for (k, v) in [("osc_1_on", 1.0), ("osc_1_level", 0.7), ("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 80.0), ("env_1_release", 0.3)] {
        p.settings.values.insert(k.into(), serde_json::Value::from(v));
    }
    let time = |scenario: &Scenario, n: usize| {
        let t = Instant::now();
        for i in 0..n {
            ops::measure(&p, scenario, i as u64).unwrap();
        }
        t.elapsed().as_secs_f32() / n as f32
    };
    let lite = time(&Scenario::lite(), 10);
    let faithful = time(&Scenario::faithful(), 10);
    println!("one render + descriptors, single thread: lite {:.1} ms, faithful {:.1} ms, ratio {:.2}", lite * 1000.0, faithful * 1000.0, lite / faithful);

    // Throughput: an exploration of 64 variants in Lite, on every core.
    let spec = ExploreSpec { count: 64, amplitude: 0.2, seed: 1, switch_indexed: 0.0, prior: spinwave_control::ops::Prior::Live, free_ranges: true, budget: Budget { max_renders: 10_000, max_seconds: 600.0 } };
    let t = Instant::now();
    let e = explore(&p, &Scenario::lite(), &spec).unwrap();
    let wall = t.elapsed().as_secs_f32();
    let threads = ops::thread_count();
    println!("explore: {} renders on {} threads in {:.2} s = {:.0} renders/s, {:.1} renders/s/thread ({:.2} of the single-thread rate)",
        e.renders, threads, wall, e.renders as f32 / wall, e.renders as f32 / wall / threads as f32, (e.renders as f32 / wall / threads as f32) * lite);
}
