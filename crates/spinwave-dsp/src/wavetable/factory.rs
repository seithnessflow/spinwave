//! Procedural factory wavetables: multi-frame tables built at startup so
//! init patches morph out of the box, without any preset JSON.

use realfft::num_complex::Complex;

use super::components::InterpolationStyle;
use super::creator::{Component, Creator, Group};
use super::sources::{InterpolationMode, WaveSource};
use super::wave_frame::{WaveFrame, WaveShape, NUM_REAL_COMPLEX, WAVEFORM_SIZE};
use super::wavetable::{Wavetable, NUM_OSCILLATOR_WAVE_FRAMES};

const LAST_FRAME: usize = NUM_OSCILLATOR_WAVE_FRAMES - 1;

/// Builds a full-size table by generating one frame per position,
/// normalizing each frame to unit peak.
fn table_from_frames(
    name: &str,
    mut build_frame: impl FnMut(f32, &mut WaveFrame),
) -> Wavetable {
    let mut wavetable = Wavetable::new(NUM_OSCILLATOR_WAVE_FRAMES);
    wavetable.name = name.to_string();
    wavetable.set_num_frames(NUM_OSCILLATOR_WAVE_FRAMES);

    let mut frame = WaveFrame::new();
    for position in 0..NUM_OSCILLATOR_WAVE_FRAMES {
        let t = position as f32 / LAST_FRAME as f32;
        frame.clear();
        frame.index = position;
        build_frame(t, &mut frame);
        frame.normalize(true);
        frame.to_frequency_domain();
        wavetable.load_wave_frame_at(&frame, position);
    }
    wavetable.post_process(0.0);
    wavetable
}

/// The classic morph: sin, triangle, saw, square and pulse anchors spread
/// over the frame range with smooth spectral interpolation between them
/// (Vital's init-style table, but morphing instead of stepped).
pub fn basic_shapes() -> Wavetable {
    let shapes = [
        WaveShape::Sin,
        WaveShape::Triangle,
        WaveShape::Saw,
        WaveShape::Square,
        WaveShape::Pulse,
    ];
    let keyframes = shapes
        .iter()
        .enumerate()
        .map(|(i, &shape)| {
            let position = (LAST_FRAME * i / (shapes.len() - 1)) as i32;
            (position, WaveFrame::predefined(shape))
        })
        .collect();

    let source = WaveSource::from_frames(
        keyframes,
        InterpolationStyle::Linear,
        InterpolationMode::Frequency,
    );
    let creator = Creator {
        name: "Basic Shapes".to_string(),
        author: String::new(),
        groups: vec![Group {
            components: vec![Component::Wave(source)],
        }],
        remove_all_dc: false,
        full_normalize: false,
    };
    creator.render()
}

/// Pulse-width sweep from a 50% square down to a 2% needle pulse. Frames
/// are DC-free and peak-normalized.
pub fn pwm() -> Wavetable {
    table_from_frames("PWM", |t, frame| {
        let width = 0.5 + (0.02 - 0.5) * t;
        let dc = 2.0 * width - 1.0;
        for (i, sample) in frame.time_domain.iter_mut().enumerate() {
            let phase = i as f32 / WAVEFORM_SIZE as f32;
            *sample = if phase < width { 1.0 } else { -1.0 } - dc;
        }
    })
}

/// Frames adding harmonics progressively: frame 0 is a pure tone, the
/// last frame carries the first 16 harmonics at saw-style `1/k` weights.
/// The newest harmonic fades in fractionally so the morph stays smooth.
pub fn harmonic_series() -> Wavetable {
    const MAX_HARMONICS: usize = 16;
    table_from_frames("Harmonic Series", |t, frame| {
        let count = 1.0 + (MAX_HARMONICS as f32 - 1.0) * t;
        let full = count as usize;
        let fraction = count - full as f32;

        let scale = (WAVEFORM_SIZE / 2) as f32;
        for k in 1..=full.min(MAX_HARMONICS) {
            frame.frequency_domain[k] = Complex::new(scale / k as f32, 0.0);
        }
        if full < MAX_HARMONICS && fraction > 0.0 {
            let k = full + 1;
            frame.frequency_domain[k] = Complex::new(fraction * scale / k as f32, 0.0);
        }
        frame.to_time_domain();
    })
}

/// A saw spectrum shaped by a sweeping vocal formant pair (roughly an
/// "oo" to "ah" motion), built for bass sound design. Uses Schroeder
/// phases to keep the crest factor low.
pub fn formant_growl() -> Wavetable {
    const NUM_HARMONICS_USED: usize = 64;
    const FUNDAMENTAL_HZ: f32 = 55.0;

    let gaussian = |harmonic: f32, center: f32, width: f32| -> f32 {
        let delta = (harmonic - center) / width;
        (-0.5 * delta * delta).exp()
    };

    table_from_frames("Formant Growl", |t, frame| {
        // Formant pair in harmonics of a low bass fundamental.
        let formant1 = (300.0 + (800.0 - 300.0) * t) / FUNDAMENTAL_HZ;
        let formant2 = (2200.0 + (1100.0 - 2200.0) * t) / FUNDAMENTAL_HZ;
        let width1 = 2.0;
        let width2 = 4.0;

        let scale = (WAVEFORM_SIZE / 2) as f32;
        for k in 1..=NUM_HARMONICS_USED.min(NUM_REAL_COMPLEX - 1) {
            let harmonic = k as f32;
            let saw = 1.0 / harmonic;
            let envelope = 0.05
                + gaussian(harmonic, formant1, width1)
                + 0.8 * gaussian(harmonic, formant2, width2);
            let amplitude = saw * envelope * scale;
            // Schroeder phase spread keeps the waveform from spiking.
            let phase = -std::f32::consts::PI * (harmonic * harmonic)
                / NUM_HARMONICS_USED as f32;
            frame.frequency_domain[k] = Complex::from_polar(amplitude, phase);
        }
        frame.to_time_domain();
    })
}

/// All factory table names, as accepted by [`factory_table`].
pub const FACTORY_TABLE_NAMES: [&str; 4] =
    ["Basic Shapes", "PWM", "Harmonic Series", "Formant Growl"];

/// Looks up a factory table by name (case-insensitive; spaces,
/// underscores and hyphens are interchangeable).
pub fn factory_table(name: &str) -> Option<Wavetable> {
    let normalized: String = name
        .trim()
        .chars()
        .map(|c| match c {
            '_' | '-' => ' ',
            _ => c.to_ascii_lowercase(),
        })
        .collect();
    match normalized.as_str() {
        "basic shapes" => Some(basic_shapes()),
        "pwm" => Some(pwm()),
        "harmonic series" => Some(harmonic_series()),
        "formant growl" => Some(formant_growl()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    fn assert_frames_sane(wavetable: &Wavetable) {
        assert_eq!(wavetable.num_frames(), NUM_OSCILLATOR_WAVE_FRAMES);
        for frame in 0..wavetable.num_frames() {
            let data = wavetable.data().wave_data(frame);
            let peak = data.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            assert!(
                data.iter().all(|value| value.is_finite()),
                "{}: frame {frame} not finite",
                wavetable.name
            );
            assert!(
                (0.3..=2.0).contains(&peak),
                "{}: frame {frame} peak {peak} out of range",
                wavetable.name
            );
        }
    }

    #[test]
    fn factory_tables_are_sane() {
        for table in [basic_shapes(), pwm(), harmonic_series(), formant_growl()] {
            assert_frames_sane(&table);
        }
    }

    #[test]
    fn basic_shapes_starts_at_sine() {
        let wavetable = basic_shapes();
        let sin = WaveFrame::predefined(WaveShape::Sin);
        let corr = correlation(wavetable.data().wave_data(0), &sin.time_domain);
        assert!(corr > 0.99, "first frame vs sine: {corr}");

        // The morph actually moves: the middle is not still a sine.
        let mid = correlation(wavetable.data().wave_data(128), &sin.time_domain);
        assert!(mid < 0.95, "table does not morph: {mid}");
    }

    #[test]
    fn pwm_sweeps_pulse_width() {
        let wavetable = pwm();
        let duty = |frame: usize| -> f32 {
            let data = wavetable.data().wave_data(frame);
            let (min, max) = data
                .iter()
                .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));
            let middle = 0.5 * (min + max);
            data.iter().filter(|&&v| v > middle).count() as f32 / data.len() as f32
        };
        let first = duty(0);
        let last = duty(NUM_OSCILLATOR_WAVE_FRAMES - 1);
        assert!((first - 0.5).abs() < 0.02, "first frame duty {first}");
        assert!((0.005..=0.05).contains(&last), "last frame duty {last}");
    }

    #[test]
    fn harmonic_series_adds_harmonics() {
        let wavetable = harmonic_series();
        let sin = WaveFrame::predefined(WaveShape::Sin);
        let corr = correlation(wavetable.data().wave_data(0), &sin.time_domain);
        assert!(corr > 0.99, "first frame vs pure tone: {corr}");

        let data = wavetable.data();
        // Harmonic 16 present at the end, absent at the start.
        assert!(data.frequency_amplitudes(0)[2 * 16] < 1e-3);
        assert!(data.frequency_amplitudes(256)[2 * 16] > 1e-3);
    }

    #[test]
    fn formant_growl_moves_formants() {
        let wavetable = formant_growl();
        let data = wavetable.data();
        let first = data.wave_data(0);
        let last = data.wave_data(NUM_OSCILLATOR_WAVE_FRAMES - 1);
        let corr = correlation(first, last);
        assert!(corr < 0.9, "formant sweep too static: {corr}");
    }

    #[test]
    fn factory_registry_finds_tables() {
        for name in FACTORY_TABLE_NAMES {
            assert!(factory_table(name).is_some(), "missing {name}");
        }
        assert!(factory_table("basic_shapes").is_some());
        assert!(factory_table("PWM").is_some());
        assert!(factory_table(" Formant-Growl ").is_some());
        assert!(factory_table("does not exist").is_none());
    }
}
