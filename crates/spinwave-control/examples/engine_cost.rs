//! Which phase of a render stops scaling across threads.
use std::time::Instant;
use spinwave_control::ops::{render, render_seed, Scenario};
use spinwave_control::session::Session;
use spinwave_params::Preset;

fn patch() -> Preset {
    let mut p = Preset::default();
    for (k, v) in [("osc_1_on", 1.0), ("osc_1_level", 0.7), ("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 80.0)] {
        p.settings.values.insert(k.into(), serde_json::Value::from(v));
    }
    p
}

fn bench(name: &str, threads: usize, per_thread: usize, job: impl Fn(&mut Session) + Sync) {
    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut session = Session::with_output_dir(std::env::temp_dir());
                for _ in 0..per_thread { job(&mut session); }
            });
        }
    });
    let total = threads * per_thread;
    println!("{name:<28} {threads:>2} threads: {:>7.1} ms per job per thread, {:>6.0} jobs/s", t.elapsed().as_secs_f32() * 1000.0 / per_thread as f32, total as f32 / t.elapsed().as_secs_f32());
}

fn main() {
    let p = patch();
    let json = p.to_json().unwrap();
    let sc = Scenario::lite();
    for threads in [1, 4, 8, 16] {
        bench("kernel new", threads, 20, |_| { std::hint::black_box(spinwave_engine::kernel::SynthVoiceKernel::new(88200)); });
        bench("engine with_pool(1)", threads, 20, |_| { std::hint::black_box(spinwave_engine::engine::SoundEngine::with_pool(44100, 1)); });
        bench("recycle(1)", threads, 20, |s| { s.engine_mut().recycle(1); });
        bench("load_preset_json", threads, 20, |s| { s.load_preset_json(&json).unwrap(); });
        bench("run_blocks only", threads, 20, |s| { s.run_blocks(&sc.notes, sc.seconds, &[]); });
        bench("render (lite, no describe)", threads, 20, |s| { render(s, &p, &sc, render_seed(1, 0)).unwrap(); });
        bench("describe only", threads, 20, |s| {
            let r = render(s, &p, &sc, render_seed(1, 0)).unwrap();
            spinwave_control::ops::descriptors::describe(&r.samples, 44100);
        });
    }
}
