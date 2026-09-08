//! Audio descriptors for LLM ears: level, spectrum, envelope, pitch.

use realfft::RealFftPlanner;
use serde::Serialize;

const FRAME_SIZE: usize = 4096;
const HOP_SIZE: usize = 2048;

#[derive(Serialize)]
pub struct Analysis {
    pub duration_seconds: f32,
    pub peak: f32,
    pub rms_db: f32,
    pub dc_offset: f32,
    /// Spectral centroid in Hz — perceptual brightness.
    pub spectral_centroid_hz: f32,
    /// Frequency below which 85% of the energy sits.
    pub spectral_rolloff_hz: f32,
    /// Energy per band in dB relative to the loudest band.
    pub bands_db: Bands,
    pub envelope: Envelope,
    /// Autocorrelation pitch estimate on the loudest section, if voiced.
    pub pitch_hz: Option<f32>,
    /// Stereo width: 1 - |correlation| between channels (0 = mono).
    pub stereo_width: f32,
}

#[derive(Serialize)]
pub struct Bands {
    pub sub_0_60: f32,
    pub bass_60_250: f32,
    pub low_mid_250_1k: f32,
    pub mid_1k_4k: f32,
    pub high_4k_12k: f32,
    pub air_12k_up: f32,
}

#[derive(Serialize)]
pub struct Envelope {
    /// Seconds from first sound to 90% of the peak level.
    pub attack_seconds: f32,
    /// Level 250 ms after the peak, relative to the peak (dB).
    pub post_peak_250ms_db: f32,
    /// RMS of the final 10% of the file relative to the overall peak (dB);
    /// very negative = the sound fully dies out.
    pub tail_db: f32,
}

/// Analyzes an interleaved stereo buffer.
pub fn analyze(interleaved: &[f32], sample_rate: u32) -> Analysis {
    let frames = interleaved.len() / 2;
    let mono: Vec<f32> = (0..frames)
        .map(|i| (interleaved[2 * i] + interleaved[2 * i + 1]) * 0.5)
        .collect();

    let peak = mono.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let rms = (mono.iter().map(|v| v * v).sum::<f32>() / frames.max(1) as f32).sqrt();
    let dc = mono.iter().sum::<f32>() / frames.max(1) as f32;

    let (centroid, rolloff, bands) = spectral(&mono, sample_rate);
    let envelope = envelope(&mono, sample_rate, peak);
    let pitch = pitch(&mono, sample_rate, peak);

    // Stereo width via inter-channel correlation.
    let mut sum_lr = 0.0f64;
    let mut sum_ll = 0.0f64;
    let mut sum_rr = 0.0f64;
    for i in 0..frames {
        let l = interleaved[2 * i] as f64;
        let r = interleaved[2 * i + 1] as f64;
        sum_lr += l * r;
        sum_ll += l * l;
        sum_rr += r * r;
    }
    let denom = (sum_ll * sum_rr).sqrt();
    let correlation = if denom > 1e-12 { (sum_lr / denom) as f32 } else { 1.0 };
    let stereo_width = (1.0 - correlation.abs()).clamp(0.0, 1.0);

    Analysis {
        duration_seconds: frames as f32 / sample_rate as f32,
        peak,
        rms_db: to_db(rms),
        dc_offset: dc,
        spectral_centroid_hz: centroid,
        spectral_rolloff_hz: rolloff,
        bands_db: bands,
        envelope,
        pitch_hz: pitch,
        stereo_width,
    }
}

fn to_db(magnitude: f32) -> f32 {
    20.0 * magnitude.max(1e-9).log10()
}

fn spectral(mono: &[f32], sample_rate: u32) -> (f32, f32, Bands) {
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FRAME_SIZE);
    let mut spectrum_sum = vec![0.0f32; FRAME_SIZE / 2 + 1];
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    let window: Vec<f32> = (0..FRAME_SIZE)
        .map(|i| {
            let t = i as f32 / (FRAME_SIZE - 1) as f32;
            0.5 - 0.5 * (2.0 * core::f32::consts::PI * t).cos()
        })
        .collect();

    let mut num_frames = 0usize;
    let mut start = 0usize;
    while start + FRAME_SIZE <= mono.len() {
        for (dst, (&sample, &w)) in
            input.iter_mut().zip(mono[start..].iter().zip(&window))
        {
            *dst = sample * w;
        }
        fft.process(&mut input, &mut output).expect("fft");
        for (sum, bin) in spectrum_sum.iter_mut().zip(&output) {
            *sum += bin.norm();
        }
        num_frames += 1;
        start += HOP_SIZE;
    }
    if num_frames == 0 {
        return (
            0.0,
            0.0,
            Bands {
                sub_0_60: 0.0,
                bass_60_250: 0.0,
                low_mid_250_1k: 0.0,
                mid_1k_4k: 0.0,
                high_4k_12k: 0.0,
                air_12k_up: 0.0,
            },
        );
    }

    let bin_hz = sample_rate as f32 / FRAME_SIZE as f32;
    let total: f32 = spectrum_sum.iter().sum();
    let centroid = if total > 1e-9 {
        spectrum_sum
            .iter()
            .enumerate()
            .map(|(i, &m)| i as f32 * bin_hz * m)
            .sum::<f32>()
            / total
    } else {
        0.0
    };

    let mut cumulative = 0.0f32;
    let mut rolloff = 0.0f32;
    for (i, &m) in spectrum_sum.iter().enumerate() {
        cumulative += m;
        if cumulative >= total * 0.85 {
            rolloff = i as f32 * bin_hz;
            break;
        }
    }

    let mut band_energy = [0.0f32; 6];
    const EDGES: [f32; 5] = [60.0, 250.0, 1000.0, 4000.0, 12000.0];
    for (i, &m) in spectrum_sum.iter().enumerate() {
        let hz = i as f32 * bin_hz;
        let band = EDGES.iter().position(|&edge| hz < edge).unwrap_or(5);
        band_energy[band] += m * m;
    }
    let max_energy = band_energy.iter().fold(1e-12f32, |a, &v| a.max(v));
    let band_db = |i: usize| 10.0 * (band_energy[i] / max_energy).max(1e-12).log10();

    (
        centroid,
        rolloff,
        Bands {
            sub_0_60: band_db(0),
            bass_60_250: band_db(1),
            low_mid_250_1k: band_db(2),
            mid_1k_4k: band_db(3),
            high_4k_12k: band_db(4),
            air_12k_up: band_db(5),
        },
    )
}

fn envelope(mono: &[f32], sample_rate: u32, peak: f32) -> Envelope {
    let frame = (sample_rate as usize / 100).max(1); // 10 ms frames
    let rms_frames: Vec<f32> = mono
        .chunks(frame)
        .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let frame_seconds = frame as f32 / sample_rate as f32;

    let peak_frame = rms_frames
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let peak_rms = rms_frames.get(peak_frame).copied().unwrap_or(0.0);

    let first_sound = rms_frames
        .iter()
        .position(|&v| v > peak_rms * 0.01)
        .unwrap_or(0);
    let attack_end = rms_frames[first_sound..]
        .iter()
        .position(|&v| v >= peak_rms * 0.9)
        .unwrap_or(0)
        + first_sound;
    let attack_seconds = (attack_end - first_sound) as f32 * frame_seconds;

    let post_index = peak_frame + (0.25 / frame_seconds) as usize;
    let post_level = rms_frames.get(post_index).copied().unwrap_or(0.0);
    let post_peak_250ms_db = to_db(post_level / peak_rms.max(1e-9));

    let tail_start = rms_frames.len().saturating_sub(rms_frames.len() / 10).max(1);
    let tail_rms = rms_frames[tail_start..]
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let tail_db = to_db(tail_rms / peak.max(1e-9));

    Envelope { attack_seconds, post_peak_250ms_db, tail_db }
}

/// Autocorrelation pitch on a 100 ms window around the loudest point.
fn pitch(mono: &[f32], sample_rate: u32, peak: f32) -> Option<f32> {
    if peak < 1e-4 {
        return None;
    }
    let window = (sample_rate as usize / 10).min(mono.len());
    if window < 256 {
        return None;
    }
    // Center on the loudest 100 ms.
    let mut best_start = 0usize;
    let mut best_level = 0.0f32;
    let mut start = 0usize;
    while start + window <= mono.len() {
        let level: f32 = mono[start..start + window].iter().map(|v| v.abs()).sum();
        if level > best_level {
            best_level = level;
            best_start = start;
        }
        start += window / 2;
    }
    let segment = &mono[best_start..best_start + window];

    let min_lag = (sample_rate / 2000).max(2) as usize; // up to 2 kHz
    let max_lag = (sample_rate / 30) as usize; // down to 30 Hz
    let max_lag = max_lag.min(window / 2);

    let energy: f32 = segment.iter().map(|v| v * v).sum();
    if energy < 1e-9 {
        return None;
    }

    let mut best_lag = 0usize;
    let mut best_corr = 0.0f32;
    for lag in min_lag..max_lag {
        let mut corr = 0.0f32;
        for i in 0..window - lag {
            corr += segment[i] * segment[i + lag];
        }
        let normalized = corr / energy;
        if normalized > best_corr {
            best_corr = normalized;
            best_lag = lag;
        }
    }

    if best_corr > 0.5 && best_lag > 0 {
        Some(sample_rate as f32 / best_lag as f32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, seconds: f32, sample_rate: u32) -> Vec<f32> {
        let frames = (seconds * sample_rate as f32) as usize;
        let mut out = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let v =
                (2.0 * core::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin() * 0.5;
            out.push(v);
            out.push(v);
        }
        out
    }

    #[test]
    fn sine_analysis_finds_pitch_and_centroid() {
        let audio = sine(440.0, 1.0, 44100);
        let analysis = analyze(&audio, 44100);
        let pitch = analysis.pitch_hz.expect("pitch detected");
        assert!((pitch - 440.0).abs() < 8.0, "pitch {pitch}");
        assert!(
            (analysis.spectral_centroid_hz - 440.0).abs() < 100.0,
            "centroid {}",
            analysis.spectral_centroid_hz
        );
        assert!(analysis.stereo_width < 0.01);
        assert!((analysis.peak - 0.5).abs() < 0.01);
    }

    #[test]
    fn band_energy_concentrates_correctly() {
        let audio = sine(100.0, 1.0, 44100);
        let analysis = analyze(&audio, 44100);
        assert_eq!(analysis.bands_db.bass_60_250, 0.0); // loudest band
        assert!(analysis.bands_db.high_4k_12k < -40.0);
    }
}
