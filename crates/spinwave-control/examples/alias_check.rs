use spinwave_control::ops::{aliasing, Scenario, RenderMode};
use spinwave_params::Preset;
fn main() {
    let mut p = Preset::default();
    for (k, v) in [("osc_1_on", 1.0), ("osc_1_level", 0.7), ("osc_1_wave_frame", 128.0), ("filter_1_on", 0.0), ("env_1_release", 0.3),
                   ("osc_1_distortion_type", 7.0), ("osc_1_distortion_amount", 0.6), ("osc_2_on", 1.0), ("osc_2_level", 0.0), ("osc_2_transpose", 17.0), ("osc_2_wave_frame", 0.0)] {
        p.settings.values.insert(k.into(), serde_json::Value::from(v));
    }
    for (name, os) in [("1x (lite)", 0.0), ("2x", 1.0), ("4x", 2.0), ("8x", 3.0)] {
        p.settings.values.insert("oversampling".into(), os.into());
        let mut sc = Scenario::one_note(72, 0.4, 0.6, RenderMode::Faithful);
        sc.mode = RenderMode::Faithful;
        let r = aliasing(&p, &sc, 1).unwrap();
        println!("{name:<10} ratio {:.3}  explained {}  unexplained {} (top {:?})", r.ratio, r.explained_peaks, r.unexplained.len(), r.unexplained.iter().take(3).map(|u| (u.0.round(), u.1.round())).collect::<Vec<_>>());
    }
}
