//! Where a render's fixed cost goes: building the engine, recycling it,
//! applying the preset, and the audio itself.
use spinwave_control::ops::{render, Scenario};
use spinwave_control::session::{Session, SAMPLE_RATE};
use spinwave_engine::engine::SoundEngine;
use std::time::Instant;

fn main() {
    let n = 100;
    let t = Instant::now();
    for _ in 0..n {
        let e = SoundEngine::with_pool(SAMPLE_RATE, 1);
        std::hint::black_box(&e);
    }
    println!("engine with_pool(1): {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    let mut e = SoundEngine::with_pool(SAMPLE_RATE, 1);
    let t = Instant::now();
    for _ in 0..n {
        e.recycle(1);
    }
    println!("engine recycle(1): {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    let t = Instant::now();
    for _ in 0..n {
        let k = spinwave_engine::kernel::SynthVoiceKernel::new(SAMPLE_RATE);
        std::hint::black_box(&k);
    }
    println!("kernel new: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    let mut preset = spinwave_params::Preset::default();
    for (k, v) in [("osc_1_on", 1.0), ("osc_1_level", 0.7), ("osc_1_wave_frame", 128.0), ("filter_1_on", 1.0), ("filter_1_cutoff", 80.0), ("filter_1_resonance", 0.3), ("env_1_attack", 0.05), ("env_1_release", 0.3)] {
        preset.settings.values.insert(k.into(), serde_json::Value::from(v));
    }
    let mut session = Session::with_output_dir(std::env::temp_dir());
    let scenario = Scenario::lite();
    let t = Instant::now();
    for i in 0..n {
        let r = render(&mut session, &preset, &scenario, i as u32).unwrap();
        std::hint::black_box(&r);
    }
    println!("lite render (all in): {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    let json = spinwave_control::ops::for_mode_json(&preset, spinwave_control::ops::RenderMode::Lite);
    let t = Instant::now();
    for _ in 0..n {
        session.load_preset_json(&json).unwrap();
    }
    println!("  load_preset_json (parse + build + recycle + apply): {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    let t = Instant::now();
    for _ in 0..n {
        let s = session.render_samples(&scenario.notes, scenario.seconds, scenario.bpm);
        std::hint::black_box(&s);
    }
    println!("  render_samples (recycle + apply + blocks): {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    effects_breakdown();
}
#[allow(dead_code)]
pub fn effects_breakdown() {
    use spinwave_dsp::effects::*;
    let er = SAMPLE_RATE as f32;
    let n = 200;
    macro_rules! time {
        ($label:expr, $body:expr) => {{
            let t = Instant::now();
            for _ in 0..n {
                let v = $body;
                std::hint::black_box(&v);
            }
            println!("  {:<28} {:.3} ms", $label, t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        }};
    }
    time!("Chorus::new", Chorus::new(er));
    time!("MultibandCompressor::new", MultibandCompressor::new(er));
    time!("Distortion::new", Distortion::new(er));
    time!("Equalizer::new", Equalizer::new(er));
    time!("Flanger::new", Flanger::new(er));
    time!("Phaser::new", Phaser::new(er));
    time!("FrequencyShifter::new", FrequencyShifter::new(er));
    time!("ConvolutionReverb::new", ConvolutionReverb::new());
    let mut delay = StereoDelay::new((4.0 * er) as usize + 1, er);
    time!("delay reset_for_reuse", { delay.reset_for_reuse(); 0 });
    let mut reverb = Reverb::new(er);
    time!("reverb reset_for_reuse", { reverb.reset_for_reuse(); 0 });
    time!("VoiceFilter::new", spinwave_engine::kernel::VoiceFilter::new(er));
    let mut chain = spinwave_engine::effect_chain::EffectChain::new(er, 1024);
    time!("EffectChain reset_for_reuse", { chain.reset_for_reuse(); 0 });
    time!("VoiceAllocator::new(1)", spinwave_engine::VoiceAllocator::new(1, || spinwave_engine::kernel::SynthVoiceKernel::new(SAMPLE_RATE)));
    let mut alloc = spinwave_engine::VoiceAllocator::new(1, || spinwave_engine::kernel::SynthVoiceKernel::new(SAMPLE_RATE));
    time!("allocator set_sample_rate", { alloc.set_sample_rate(SAMPLE_RATE); 0 });
    time!("Decimator::new(3)", spinwave_dsp::filters::Decimator::new(3));
    let mut engine = SoundEngine::with_pool(SAMPLE_RATE, 1);
    time!("engine recycle(1)", { engine.recycle(1); 0 });
}
