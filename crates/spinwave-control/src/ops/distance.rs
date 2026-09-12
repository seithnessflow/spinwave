//! The perceptual distance of `notes/operations-design.md` §2: a
//! multi-resolution spectrogram folded into half-octave bands, compared in
//! dB, with the two decompositions that say WHERE two renders differ.
//!
//! Two STFT resolutions (4096/1024 and 1024/256 samples), 24 half-octave
//! bands from 40 Hz to 16 kHz, each band's power in dB. The distance is
//! the mean over resolutions of the mean over (frame, band) of |ΔdB| —
//! the log-STFT-magnitude term of Yamamoto, Song & Kim (ICASSP 2020) on a
//! log frequency axis, a per-frame log-spectral distance (Gray & Markel
//! 1976). Two renders of the same scenario are aligned by construction.
//!
//! Two things a plain mean of |ΔdB| gets wrong, both from the review:
//!
//! * quiet cells. A 10 dB difference between −85 and −95 dBFS is not a
//!   timbre change, and on a quiet patch such cells outnumber the loud
//!   ones. Every cell is weighted by the louder of the two renders'
//!   power in it, relative to the loudest cell of either render: a cell
//!   40 dB or more under that peak weighs nothing, a cell at the peak
//!   weighs one, linear in dB between. (Relative to the *global* peak,
//!   not the frame's: a silent tail must weigh nothing, and relative to
//!   its own peak every frame looks loud.) Measured: a pure −1 dB gain
//!   reads 1.0 dB, and saw-against-square reads the same 20 dB down as
//!   at nominal level (`ops::tests::distance_scale`).
//! * loudness. Two identical patches 3 dB apart should not count as
//!   different when the question is timbre. [`Options::normalize_loudness`]
//!   aligns the two renders' integrated loudness before comparing; it is
//!   off by default, so `measure`-style questions keep the level, and on
//!   for searches by reference.

use realfft::RealFftPlanner;
use serde::Serialize;

use super::descriptors::{hann, loudness_lufs};

/// Half-octave band edges from 40 Hz: 40·2^(n/2), 25 edges for 24 bands.
pub fn band_edges_hz() -> [f32; 25] {
    core::array::from_fn(|i| 40.0 * 2f32.powf(i as f32 / 2.0))
}

const RESOLUTIONS: [(usize, usize); 2] = [(4096, 1024), (1024, 256)];
/// A cell this far under the loudest cell of either render weighs nothing.
const WEIGHT_RANGE_DB: f32 = 40.0;
/// Well under anything the weighting can reach at any level a render is
/// measured at (a −60 dBFS render still has 40 dB of window above this).
const FLOOR_DB: f32 = -120.0;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Options {
    /// Scale `b` to `a`'s integrated loudness before comparing.
    pub normalize_loudness: bool,
}

/// How far apart two renders are, and where.
#[derive(Clone, Debug, Serialize)]
pub struct Distance {
    /// Weighted mean |ΔdB| over resolutions, frames and bands.
    pub total_db: f32,
    /// The same, per band (time-averaged): `(band_low_hz, db)`.
    pub per_band_db: Vec<(f32, f32)>,
    /// The same, per time (band-averaged), at the coarse resolution's
    /// hop: `(seconds, db)`.
    pub per_time_db: Vec<(f32, f32)>,
    /// The gain applied to `b` for the comparison, dB (0 unless
    /// normalised).
    pub b_gain_db: f32,
    pub options: Options,
}

/// Band-power spectrogram at one resolution: `frames × 24`, dB.
struct BandSpectrogram {
    frames: Vec<[f32; 24]>,
    hop_seconds: f32,
}

fn band_spectrogram(mono: &[f32], sample_rate: u32, frame: usize, hop: usize) -> BandSpectrogram {
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(frame);
    let window = hann(frame);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let edges = band_edges_hz();
    let bin_hz = sample_rate as f32 / frame as f32;
    // Which band each bin falls in, once.
    let bands: Vec<Option<usize>> = (0..=frame / 2)
        .map(|i| {
            let hz = i as f32 * bin_hz;
            edges.windows(2).position(|e| hz >= e[0] && hz < e[1])
        })
        .collect();
    let window_power: f32 = window.iter().map(|w| w * w).sum();
    let scale = 2.0 / (frame as f32 * window_power);
    let mut frames = Vec::new();
    let mut start = 0;
    while start + frame <= mono.len() {
        for (dst, (&s, &w)) in input.iter_mut().zip(mono[start..].iter().zip(&window)) {
            *dst = s * w;
        }
        fft.process(&mut input, &mut output).expect("fft");
        let mut power = [0.0f32; 24];
        for (bin, band) in output.iter().zip(&bands) {
            if let Some(b) = band {
                power[*b] += bin.norm_sqr() * scale;
            }
        }
        frames.push(power.map(|p| (10.0 * p.max(1e-18).log10()).max(FLOOR_DB)));
        start += hop;
    }
    BandSpectrogram { frames, hop_seconds: hop as f32 / sample_rate as f32 }
}

fn mono_of(interleaved: &[f32], gain: f32) -> Vec<f32> {
    interleaved.chunks_exact(2).map(|f| 0.5 * (f[0] + f[1]) * gain).collect()
}

/// The distance between two interleaved stereo renders of the same
/// scenario (same length, same rate). Shorter of the two decides the
/// number of frames compared.
pub fn distance(a: &[f32], b: &[f32], sample_rate: u32, options: Options) -> Distance {
    let b_gain_db = if options.normalize_loudness {
        match (loudness_lufs(a, sample_rate), loudness_lufs(b, sample_rate)) {
            (Some(la), Some(lb)) => la - lb,
            _ => 0.0,
        }
    } else {
        0.0
    };
    let mono_a = mono_of(a, 1.0);
    let mono_b = mono_of(b, 10f32.powf(b_gain_db / 20.0));

    let mut per_band_sum = [0.0f32; 24];
    let mut per_band_weight = [0.0f32; 24];
    let mut per_time: Vec<(f32, f32)> = Vec::new();
    let mut total_sum = 0.0f32;
    let mut total_weight = 0.0f32;

    for (r, &(frame, hop)) in RESOLUTIONS.iter().enumerate() {
        let sa = band_spectrogram(&mono_a, sample_rate, frame, hop);
        let sb = band_spectrogram(&mono_b, sample_rate, frame, hop);
        let n = sa.frames.len().min(sb.frames.len());
        let peak = sa.frames[..n]
            .iter()
            .chain(sb.frames[..n].iter())
            .flat_map(|f| f.iter().cloned())
            .fold(FLOOR_DB, f32::max);
        let mut resolution_sum = 0.0f32;
        let mut resolution_weight = 0.0f32;
        for t in 0..n {
            let (fa, fb) = (&sa.frames[t], &sb.frames[t]);
            let mut frame_sum = 0.0f32;
            let mut frame_weight = 0.0f32;
            for band in 0..24 {
                let louder = fa[band].max(fb[band]);
                let weight = ((louder - (peak - WEIGHT_RANGE_DB)) / WEIGHT_RANGE_DB).clamp(0.0, 1.0);
                let delta = (fa[band] - fb[band]).abs();
                frame_sum += weight * delta;
                frame_weight += weight;
                per_band_sum[band] += weight * delta;
                per_band_weight[band] += weight;
            }
            if r == 0 {
                per_time.push((t as f32 * sa.hop_seconds, safe_div(frame_sum, frame_weight)));
            }
            resolution_sum += frame_sum;
            resolution_weight += frame_weight;
        }
        // Each resolution counts once, whatever its frame count.
        total_sum += safe_div(resolution_sum, resolution_weight);
        total_weight += 1.0;
    }
    let edges = band_edges_hz();
    Distance {
        total_db: safe_div(total_sum, total_weight),
        per_band_db: (0..24).map(|b| (edges[b], safe_div(per_band_sum[b], per_band_weight[b]))).collect(),
        per_time_db: per_time,
        b_gain_db,
        options,
    }
}

fn safe_div(sum: f32, weight: f32) -> f32 {
    if weight > 0.0 { sum / weight } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, amplitude: f32, harmonics: &[f32]) -> Vec<f32> {
        let sr = 44100.0;
        (0..44100)
            .flat_map(|i| {
                let t = i as f32 / sr;
                let v: f32 = harmonics
                    .iter()
                    .enumerate()
                    .map(|(k, a)| a * (2.0 * core::f32::consts::PI * hz * (k + 1) as f32 * t).sin())
                    .sum();
                [amplitude * v, amplitude * v]
            })
            .collect()
    }

    #[test]
    fn identical_renders_are_at_zero_and_a_gain_is_a_flat_offset() {
        let a = tone(220.0, 0.5, &[1.0, 0.5, 0.25]);
        assert_eq!(distance(&a, &a, 44100, Options::default()).total_db, 0.0);
        let b = tone(220.0, 0.5 * 10f32.powf(-3.0 / 20.0), &[1.0, 0.5, 0.25]);
        let d = distance(&a, &b, 44100, Options::default());
        assert!((d.total_db - 3.0).abs() < 0.1, "a −3 dB copy sits 3 dB away: {}", d.total_db);
        let n = distance(&a, &b, 44100, Options { normalize_loudness: true });
        assert!(n.total_db < 0.05, "normalised, the same timbre is at zero: {}", n.total_db);
        assert!((n.b_gain_db - 3.0).abs() < 0.1);
    }

    #[test]
    fn a_timbre_change_lands_in_its_bands() {
        let a = tone(220.0, 0.5, &[1.0]);
        let b = tone(220.0, 0.5, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]);
        let d = distance(&a, &b, 44100, Options::default());
        assert!(d.total_db > 0.5, "{}", d.total_db);
        // The added partial is at 1760 Hz: the band holding it moves most.
        let worst = d.per_band_db.iter().cloned().fold((0.0, 0.0), |m, x| if x.1 > m.1 { x } else { m });
        assert!(worst.0 <= 1760.0 && 1760.0 < worst.0 * 2f32.sqrt(), "worst band starts at {} Hz", worst.0);
    }
}
