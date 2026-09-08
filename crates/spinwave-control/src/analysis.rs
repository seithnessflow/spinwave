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
    /// How the sound moves over time (wobbles, sweeps, rhythm).
    pub movement: Movement,
    /// Harmonic texture of the sound.
    pub texture: Texture,
}

#[derive(Serialize)]
pub struct ModRate {
    pub hz: f32,
    /// Relative strength, 1.0 = the dominant rate.
    pub strength: f32,
}

#[derive(Serialize)]
pub struct Movement {
    /// Dominant modulation rates detected in the upper-spectrum energy
    /// envelope (filter wobbles, tremolo, rhythmic gating), 0.2–16 Hz.
    pub mod_rates_hz: Vec<ModRate>,
    /// Spectral centroid per 250 ms window (brightness trajectory).
    pub centroid_trajectory_hz: Vec<f32>,
    /// RMS per 250 ms window in dB (dynamics trajectory).
    pub rms_trajectory_db: Vec<f32>,
    /// Note/percussion onsets per second.
    pub onset_density_per_second: f32,
}

#[derive(Serialize)]
pub struct Texture {
    /// 0 = purely tonal/harmonic, 1 = noise.
    pub spectral_flatness: f32,
    /// Energy at harmonic multiples of the pitch / total energy (0..1).
    pub harmonicity: Option<f32>,
    /// Odd-harmonic energy over even-harmonic energy (square-ish > 1,
    /// saw-ish ≈ 1); needs a detected pitch.
    pub odd_even_ratio: Option<f32>,
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

    let (centroid, rolloff, bands, mean_spectrum) = spectral(&mono, sample_rate);
    let envelope = envelope(&mono, sample_rate, peak);
    let pitch = pitch(&mono, sample_rate, peak);
    let movement = movement(&mono, sample_rate);
    let texture = texture(&mean_spectrum, sample_rate, pitch);

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
        movement,
        texture,
    }
}

fn to_db(magnitude: f32) -> f32 {
    20.0 * magnitude.max(1e-9).log10()
}

fn spectral(mono: &[f32], sample_rate: u32) -> (f32, f32, Bands, Vec<f32>) {
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
            spectrum_sum,
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
        spectrum_sum.iter().map(|m| m / num_frames as f32).collect(),
    )
}

/// Movement analysis: modulation rates in the upper-spectrum energy
/// envelope, brightness/level trajectories, onset density.
fn movement(mono: &[f32], sample_rate: u32) -> Movement {
    // Upper-band (>500 Hz) energy envelope via short frames — this is
    // where filter wobbles and gating show up strongest.
    const ENV_FRAME: usize = 1024;
    const ENV_HOP: usize = 512;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(ENV_FRAME);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let bin_hz = sample_rate as f32 / ENV_FRAME as f32;
    let first_bin = (500.0 / bin_hz) as usize;

    let mut band_envelope: Vec<f32> = Vec::new();
    let mut start = 0usize;
    while start + ENV_FRAME <= mono.len() {
        input.copy_from_slice(&mono[start..start + ENV_FRAME]);
        if fft.process(&mut input, &mut output).is_ok() {
            let energy: f32 = output[first_bin..].iter().map(|c| c.norm_sqr()).sum();
            band_envelope.push(energy.sqrt());
        }
        start += ENV_HOP;
    }

    let envelope_rate = sample_rate as f32 / ENV_HOP as f32;
    let mod_rates_hz = modulation_peaks(&band_envelope, envelope_rate);

    // 250 ms trajectories.
    let window = (sample_rate as usize / 4).max(1);
    let mut centroid_trajectory_hz = Vec::new();
    let mut rms_trajectory_db = Vec::new();
    for chunk in mono.chunks(window) {
        if chunk.len() < window / 2 {
            break;
        }
        let rms = (chunk.iter().map(|v| v * v).sum::<f32>() / chunk.len() as f32).sqrt();
        rms_trajectory_db.push(to_db(rms));
        centroid_trajectory_hz.push(window_centroid(chunk, sample_rate));
    }

    // Onsets: 10 ms RMS frames; a frame > 1.6x the mean of the previous
    // 50 ms counts once (with a 50 ms refractory period).
    let frame = (sample_rate as usize / 100).max(1);
    let frames: Vec<f32> = mono
        .chunks(frame)
        .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
        .collect();
    let mut onsets = 0usize;
    let mut last_onset = 0isize;
    for i in 5..frames.len() {
        let history = frames[i - 5..i].iter().sum::<f32>() / 5.0;
        if frames[i] > history * 1.6 && frames[i] > 1e-4 && i as isize - last_onset >= 5 {
            onsets += 1;
            last_onset = i as isize;
        }
    }
    let seconds = mono.len() as f32 / sample_rate as f32;

    Movement {
        mod_rates_hz,
        centroid_trajectory_hz,
        rms_trajectory_db,
        onset_density_per_second: onsets as f32 / seconds.max(0.001),
    }
}

/// FFT of the (mean-removed, windowed) energy envelope; returns the top
/// local maxima between 0.2 and 16 Hz.
fn modulation_peaks(envelope: &[f32], envelope_rate: f32) -> Vec<ModRate> {
    if envelope.len() < 16 {
        return Vec::new();
    }
    let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
    let padded_len = (envelope.len() * 2).next_power_of_two().max(256);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(padded_len);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    input.fill(0.0);
    for (i, (&value, slot)) in envelope.iter().zip(input.iter_mut()).enumerate() {
        let t = i as f32 / (envelope.len() - 1) as f32;
        let hann = 0.5 - 0.5 * (2.0 * core::f32::consts::PI * t).cos();
        *slot = (value - mean) * hann;
    }
    if fft.process(&mut input, &mut output).is_err() {
        return Vec::new();
    }

    let bin_hz = envelope_rate / padded_len as f32;
    let magnitudes: Vec<f32> = output.iter().map(|c| c.norm()).collect();
    let low_bin = (0.2 / bin_hz).ceil() as usize;
    let high_bin = ((16.0 / bin_hz) as usize).min(magnitudes.len().saturating_sub(2));
    if low_bin + 1 >= high_bin {
        return Vec::new();
    }

    let mut peaks: Vec<(f32, f32)> = Vec::new();
    for bin in low_bin.max(1)..high_bin {
        let value = magnitudes[bin];
        if value > magnitudes[bin - 1] && value >= magnitudes[bin + 1] {
            peaks.push((bin as f32 * bin_hz, value));
        }
    }
    peaks.sort_by(|a, b| b.1.total_cmp(&a.1));
    let strongest = peaks.first().map(|p| p.1).unwrap_or(0.0);
    if strongest <= 1e-9 {
        return Vec::new();
    }
    peaks
        .into_iter()
        .take(3)
        .filter(|(_, strength)| *strength > strongest * 0.2)
        .map(|(hz, strength)| ModRate { hz, strength: strength / strongest })
        .collect()
}

fn window_centroid(chunk: &[f32], sample_rate: u32) -> f32 {
    // Coarse per-window centroid via a single 1024-point FFT.
    const N: usize = 1024;
    if chunk.len() < N {
        return 0.0;
    }
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    input.copy_from_slice(&chunk[..N]);
    if fft.process(&mut input, &mut output).is_err() {
        return 0.0;
    }
    let bin_hz = sample_rate as f32 / N as f32;
    let total: f32 = output.iter().map(|c| c.norm()).sum();
    if total < 1e-9 {
        return 0.0;
    }
    output
        .iter()
        .enumerate()
        .map(|(i, c)| i as f32 * bin_hz * c.norm())
        .sum::<f32>()
        / total
}

/// Harmonic texture from the averaged magnitude spectrum.
fn texture(mean_spectrum: &[f32], sample_rate: u32, pitch: Option<f32>) -> Texture {
    let usable = &mean_spectrum[1..];
    let count = usable.len() as f32;
    let spectral_flatness = if usable.iter().all(|&m| m > 0.0) && !usable.is_empty() {
        let log_mean = usable.iter().map(|m| m.max(1e-12).ln()).sum::<f32>() / count;
        let mean = usable.iter().sum::<f32>() / count;
        (log_mean.exp() / mean.max(1e-12)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let (harmonicity, odd_even_ratio) = match pitch {
        Some(f0) if f0 > 20.0 => {
            let bin_hz = sample_rate as f32 / (2 * (mean_spectrum.len() - 1)) as f32;
            let total_energy: f32 = usable.iter().map(|m| m * m).sum();
            let mut harmonic = 0.0f32;
            let mut odd = 0.0f32;
            let mut even = 0.0f32;
            for k in 1..=16usize {
                let bin = (k as f32 * f0 / bin_hz).round() as usize;
                if bin == 0 || bin + 1 >= mean_spectrum.len() {
                    break;
                }
                // ±1 bin window around each harmonic.
                let energy: f32 = (bin - 1..=bin + 1)
                    .map(|b| mean_spectrum[b] * mean_spectrum[b])
                    .sum();
                harmonic += energy;
                if k % 2 == 1 {
                    odd += energy;
                } else {
                    even += energy;
                }
            }
            let harmonicity = (harmonic / total_energy.max(1e-12)).clamp(0.0, 1.0);
            let ratio = if even > 1e-12 { Some(odd / even) } else { None };
            (Some(harmonicity), ratio)
        }
        _ => (None, None),
    };

    Texture { spectral_flatness, harmonicity, odd_even_ratio }
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

    let mut correlations = vec![0.0f32; max_lag];
    let mut best_corr = 0.0f32;
    for lag in min_lag..max_lag {
        let mut corr = 0.0f32;
        for i in 0..window - lag {
            corr += segment[i] * segment[i + lag];
        }
        let normalized = corr / energy;
        correlations[lag] = normalized;
        best_corr = best_corr.max(normalized);
    }

    if best_corr < 0.5 {
        return None;
    }
    // The first LOCAL maximum (ascending) above the threshold is the
    // fundamental period: requiring a local max rejects the descending
    // tail of the zero-lag peak (the old spurious-2kHz artifact), and
    // taking the first one avoids sub-octave picks on periodic signals.
    let threshold = best_corr * 0.9;
    let mut chosen_lag = 0usize;
    for lag in min_lag + 1..max_lag - 1 {
        let value = correlations[lag];
        if value >= threshold
            && value >= correlations[lag - 1]
            && value >= correlations[lag + 1]
        {
            chosen_lag = lag;
            break;
        }
    }
    if chosen_lag == 0 {
        return None;
    }
    Some(sample_rate as f32 / chosen_lag as f32)
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
