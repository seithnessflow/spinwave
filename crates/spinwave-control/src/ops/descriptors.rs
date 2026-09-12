//! The descriptor set of `notes/operations-design.md` §1: every number an
//! operation reports about a render, with its unit in its name and its
//! definition here. What `analysis::analyze` already computes is reused
//! (centroid, rolloff, flatness, odd/even, width, movement); what it does
//! not — loudness, YIN, octave bands, clipping, the −20 dB decay,
//! inharmonicity, the trajectories — is added here. Aliasing is not a
//! descriptor of one render: it takes two, a semitone apart
//! (`ops::aliasing`); the single-render proxy this once carried blamed
//! FM sidebands and was removed.
//!
//! Frames are 2048-sample Hann windows with a hop of 512 unless a
//! descriptor says otherwise. The mono sum `(L + R) / 2` unless stated.

use realfft::RealFftPlanner;
use serde::Serialize;

use crate::analysis::{analyze, Analysis};

const FRAME: usize = 2048;
const HOP: usize = 512;
/// Peaks below this (relative to the strongest bin of the mean spectrum)
/// are not partials.
const PEAK_FLOOR_DB: f32 = -60.0;
/// A partial within this fraction of `n·f0` counts as harmonic `n`.
const HARMONIC_TOLERANCE: f32 = 0.03;
/// Octave band edges: 40·2ⁿ Hz, the last band open to 20 kHz.
pub const BAND_EDGES_HZ: [f32; 9] =
    [40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 20000.0];

/// Everything measured on one render. Field names carry the unit.
#[derive(Clone, Debug, Serialize)]
pub struct Descriptors {
    pub duration_seconds: f32,
    /// Largest |sample| over both channels.
    pub peak_dbfs: f32,
    /// RMS of the mono sum; a full-scale sine reads −3.01.
    pub rms_dbfs: f32,
    /// Integrated loudness, ITU-R BS.1770-4 (K-weighting, 400 ms blocks
    /// at 75 % overlap, −70 LKFS absolute gate, −10 LU relative gate).
    /// `None` when no block passes the absolute gate.
    pub loudness_lufs: Option<f32>,
    /// Mean of the mono sum, floored at −120.
    pub dc_offset_dbfs: f32,
    pub clipping: Clipping,
    /// Mean power per octave band (see [`BAND_EDGES_HZ`]), dBFS.
    pub bands_dbfs: [f32; 8],
    /// Power-weighted mean frequency (Peeters 2004).
    pub centroid_hz: f32,
    /// Frequency below which 85 % of the power sits (Peeters 2004).
    pub rolloff_hz: f32,
    /// Centroid per 50 ms window.
    pub centroid_trajectory_hz: Vec<f32>,
    /// RMS per 50 ms window, dBFS.
    pub rms_trajectory_dbfs: Vec<f32>,
    /// RMS envelope (5 ms hop) from 10 % to 90 % of its maximum — the
    /// MPEG-7 LogAttackTime thresholds, reported linear.
    pub attack_seconds: f32,
    /// From the envelope maximum to the first point 20 dB below it;
    /// `None` when it never gets there before the render ends.
    pub decay_20db_seconds: Option<f32>,
    /// RMS of the last 10 % of the render.
    pub tail_dbfs: f32,
    /// YIN (de Cheveigné & Kawahara 2002), threshold 0.15, on the loudest
    /// 100 ms; `None` when unvoiced.
    pub f0_hz: Option<f32>,
    /// Power in the bins within ±3 % of some `n·f0` (at least ±2 bins, the
    /// window's main lobe) over total power; `None` without `f0`.
    pub harmonicity: Option<f32>,
    /// Peeters 2004: `2/f0 · Σ|f_k − n_k f0|·a_k² / Σ a_k²` over the 20
    /// strongest peaks, 0 = harmonic; `None` without `f0`.
    pub inharmonicity: Option<f32>,
    /// Geometric over arithmetic mean of the power spectrum.
    pub spectral_flatness: f32,
    pub odd_even_ratio: Option<f32>,
    /// `1 − |corr(L, R)|`.
    pub stereo_width: f32,
    /// RMS(mono sum) − RMS(stereo), dB; strongly negative = cancels in mono.
    pub mono_compatibility_db: f32,
    /// Dominant modulation rates 0.2–16 Hz (from `analysis`).
    pub movement_rates_hz: Vec<f32>,
    pub onset_density_per_second: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Clipping {
    /// Samples at |x| ≥ 0.999 inside a run of at least three such samples.
    pub samples: usize,
    pub ratio: f32,
}

/// Computes every descriptor on an interleaved stereo buffer. The caller
/// checks the buffer is finite, non-empty and audible first (see
/// `ops::render`); this only measures.
pub fn describe(interleaved: &[f32], sample_rate: u32) -> Descriptors {
    describe_with(interleaved, sample_rate, true)
}

/// [`describe`] with the pitch detector optional. YIN on a 4096-sample
/// window is the largest single cost of a measurement (about a third of
/// a Lite render); a search that reads bands, centroid or level a
/// thousand times does not need `f0`, so it asks without. The
/// pitch-dependent fields (`f0_hz`, `harmonicity`, `inharmonicity`) are
/// then `None`.
pub fn describe_with(interleaved: &[f32], sample_rate: u32, pitch: bool) -> Descriptors {
    let frames = interleaved.len() / 2;
    let mono: Vec<f32> = (0..frames).map(|i| 0.5 * (interleaved[2 * i] + interleaved[2 * i + 1])).collect();
    let base: Analysis = analyze(interleaved, sample_rate);

    let peak = interleaved.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let rms = (mono.iter().map(|v| v * v).sum::<f32>() / frames.max(1) as f32).sqrt();
    let dc = mono.iter().sum::<f32>() / frames.max(1) as f32;

    let spectrum = MeanSpectrum::new(&mono, sample_rate);
    let f0 = if pitch { yin(&mono, sample_rate) } else { None };
    let peaks = spectrum.peaks();
    let (harmonicity, inharmonicity) = match f0 {
        Some(f0) => {
            let (h, i) = harmonic_measures(&peaks, &spectrum, f0);
            (Some(h), Some(i))
        }
        None => (None, None),
    };
    let env = Envelope::new(&mono, sample_rate);

    Descriptors {
        duration_seconds: frames as f32 / sample_rate as f32,
        peak_dbfs: db(peak),
        rms_dbfs: db(rms),
        loudness_lufs: loudness_lufs(interleaved, sample_rate),
        dc_offset_dbfs: db(dc.abs()).max(-120.0),
        clipping: clipping(interleaved),
        bands_dbfs: spectrum.octave_bands(),
        centroid_hz: base.spectral_centroid_hz,
        rolloff_hz: base.spectral_rolloff_hz,
        centroid_trajectory_hz: trajectory(&mono, sample_rate, |chunk| centroid_of(chunk, sample_rate)),
        rms_trajectory_dbfs: trajectory(&mono, sample_rate, |chunk| {
            db((chunk.iter().map(|v| v * v).sum::<f32>() / chunk.len().max(1) as f32).sqrt())
        }),
        attack_seconds: env.attack_seconds(),
        decay_20db_seconds: env.decay_seconds(20.0),
        tail_dbfs: {
            let start = frames - frames / 10;
            db((mono[start..].iter().map(|v| v * v).sum::<f32>() / (frames - start).max(1) as f32).sqrt())
        },
        f0_hz: f0,
        harmonicity,
        inharmonicity,
        spectral_flatness: base.texture.spectral_flatness,
        odd_even_ratio: base.texture.odd_even_ratio,
        stereo_width: base.stereo_width,
        mono_compatibility_db: mono_compatibility_db(interleaved),
        movement_rates_hz: base.movement.mod_rates_hz.iter().map(|r| r.hz).collect(),
        onset_density_per_second: base.movement.onset_density_per_second,
    }
}

pub(crate) fn db(magnitude: f32) -> f32 {
    20.0 * magnitude.max(1e-9).log10()
}

pub(crate) fn hann(size: usize) -> Vec<f32> {
    (0..size).map(|i| 0.5 - 0.5 * (2.0 * core::f32::consts::PI * i as f32 / size as f32).cos()).collect()
}

// ------------------------------------------------------------ loudness

/// One biquad, direct form I.
struct Biquad {
    b: [f64; 3],
    a: [f64; 3],
    x: [f64; 2],
    y: [f64; 2],
}

impl Biquad {
    fn tick(&mut self, input: f64) -> f64 {
        let out = (self.b[0] * input + self.b[1] * self.x[0] + self.b[2] * self.x[1]
            - self.a[1] * self.y[0]
            - self.a[2] * self.y[1])
            / self.a[0];
        self.x = [input, self.x[0]];
        self.y = [out, self.y[0]];
        out
    }
}

/// The K-weighting of BS.1770-4 at any sample rate: the standard gives
/// the 48 kHz coefficients; these are the filters they come from (a
/// high shelf, +4 dB at 1681.97 Hz, Q 0.7072; a high-pass at 38.135 Hz,
/// Q 0.5003), which reproduce the published coefficients at 48 kHz.
fn k_weighting(sample_rate: f64) -> [Biquad; 2] {
    let pi = core::f64::consts::PI;
    let shelf = {
        let f0 = 1_681.974_450_955_533;
        let gain_db = 3.999_843_853_973_347;
        let q = 0.707_175_236_955_419_6;
        let k = (pi * f0 / sample_rate).tan();
        let vh = 10f64.powf(gain_db / 20.0);
        let vb = vh.powf(0.499_666_774_154_541_6);
        let a0 = 1.0 + k / q + k * k;
        Biquad {
            b: [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0],
            a: [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    };
    // The RLB stage keeps its numerator [1, −2, 1] unnormalised, as the
    // standard's 48 kHz table does (its high-frequency gain is then
    // 4 / (1 − a1 + a2) ≈ 0 dB).
    let high_pass = {
        let f0 = 38.135_470_876_024_44;
        let q = 0.500_327_037_323_877_3;
        let k = (pi * f0 / sample_rate).tan();
        let a0 = 1.0 + k / q + k * k;
        Biquad {
            b: [1.0, -2.0, 1.0],
            a: [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0],
            x: [0.0; 2],
            y: [0.0; 2],
        }
    };
    [shelf, high_pass]
}

/// Integrated loudness per ITU-R BS.1770-4, stereo, channel weights 1.
pub fn loudness_lufs(interleaved: &[f32], sample_rate: u32) -> Option<f32> {
    let frames = interleaved.len() / 2;
    let sr = sample_rate as f64;
    let block = (0.4 * sr) as usize;
    let step = (0.1 * sr) as usize;
    if frames < block || block == 0 {
        return None;
    }
    let mut weighted = vec![[0.0f64; 2]; frames];
    for channel in 0..2 {
        let [mut shelf, mut hp] = k_weighting(sr);
        for i in 0..frames {
            let x = interleaved[2 * i + channel] as f64;
            weighted[i][channel] = hp.tick(shelf.tick(x));
        }
    }
    // Block loudness: −0.691 + 10 log10(Σ_channels mean square).
    let mut blocks = Vec::new();
    let mut start = 0;
    while start + block <= frames {
        let mut sum = 0.0;
        for w in &weighted[start..start + block] {
            sum += w[0] * w[0] + w[1] * w[1];
        }
        blocks.push(-0.691 + 10.0 * (sum / block as f64).max(1e-30).log10());
        start += step;
    }
    let absolute: Vec<f64> = blocks.iter().copied().filter(|&l| l > -70.0).collect();
    if absolute.is_empty() {
        return None;
    }
    let mean_power = |ls: &[f64]| ls.iter().map(|l| 10f64.powf((l + 0.691) / 10.0)).sum::<f64>() / ls.len() as f64;
    let relative_gate = -0.691 + 10.0 * mean_power(&absolute).log10() - 10.0;
    let gated: Vec<f64> = absolute.into_iter().filter(|&l| l > relative_gate).collect();
    if gated.is_empty() {
        return None;
    }
    Some((-0.691 + 10.0 * mean_power(&gated).log10()) as f32)
}

// ------------------------------------------------------------ clipping

fn clipping(interleaved: &[f32]) -> Clipping {
    let mut samples = 0usize;
    for channel in 0..2 {
        let mut run = 0usize;
        let flush = |run: usize, samples: &mut usize| {
            if run >= 3 {
                *samples += run;
            }
        };
        for frame in interleaved.chunks_exact(2) {
            if frame[channel].abs() >= 0.999 {
                run += 1;
            } else {
                flush(run, &mut samples);
                run = 0;
            }
        }
        flush(run, &mut samples);
    }
    Clipping { samples, ratio: samples as f32 / interleaved.len().max(1) as f32 }
}

fn mono_compatibility_db(interleaved: &[f32]) -> f32 {
    let mut stereo = 0.0f64;
    let mut mono = 0.0f64;
    for frame in interleaved.chunks_exact(2) {
        let (l, r) = (frame[0] as f64, frame[1] as f64);
        stereo += 0.5 * (l * l + r * r);
        let m = 0.5 * (l + r);
        mono += m * m;
    }
    (10.0 * (mono.max(1e-18) / stereo.max(1e-18)).log10()) as f32
}

// ------------------------------------------------------------ spectrum

/// The mean power spectrum over the render's frames, plus what the
/// band, peak and proxy measures read off it.
pub(crate) struct MeanSpectrum {
    /// Mean power per bin, scaled so that Σ over bins is the mean square
    /// of the signal (Parseval with the window's power).
    power: Vec<f32>,
    bin_hz: f32,
}

impl MeanSpectrum {
    pub(crate) fn new(mono: &[f32], sample_rate: u32) -> MeanSpectrum {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FRAME);
        let window = hann(FRAME);
        let window_power: f32 = window.iter().map(|w| w * w).sum();
        let mut input = fft.make_input_vec();
        let mut output = fft.make_output_vec();
        let mut power = vec![0.0f32; FRAME / 2 + 1];
        let mut count = 0usize;
        let mut start = 0usize;
        while start + FRAME <= mono.len() {
            for (dst, (&s, &w)) in input.iter_mut().zip(mono[start..].iter().zip(&window)) {
                *dst = s * w;
            }
            fft.process(&mut input, &mut output).expect("fft");
            for (p, bin) in power.iter_mut().zip(&output) {
                *p += bin.norm_sqr();
            }
            count += 1;
            start += HOP;
        }
        if count > 0 {
            // One-sided spectrum: bins 1..N/2−1 stand for two.
            let scale = 1.0 / (count as f32 * FRAME as f32 * window_power);
            for (i, p) in power.iter_mut().enumerate() {
                let sides = if i == 0 || i == FRAME / 2 { 1.0 } else { 2.0 };
                *p *= sides * scale;
            }
        }
        MeanSpectrum { power, bin_hz: sample_rate as f32 / FRAME as f32 }
    }

    pub(crate) fn total_power(&self) -> f32 {
        self.power.iter().sum()
    }

    pub(crate) fn power(&self) -> &[f32] {
        &self.power
    }

    pub(crate) fn bin_hz(&self) -> f32 {
        self.bin_hz
    }

    fn octave_bands(&self) -> [f32; 8] {
        let mut bands = [0.0f32; 8];
        for (i, p) in self.power.iter().enumerate() {
            let hz = i as f32 * self.bin_hz;
            if let Some(b) = BAND_EDGES_HZ.windows(2).position(|e| hz >= e[0] && hz < e[1]) {
                bands[b] += p;
            }
        }
        bands.map(|p| 10.0 * p.max(1e-18).log10())
    }

    /// Local maxima above the floor: `(hz, power)`, strongest first.
    pub(crate) fn peaks(&self) -> Vec<(f32, f32)> {
        let top = self.power.iter().cloned().fold(0.0f32, f32::max);
        let floor = top * 10f32.powf(PEAK_FLOOR_DB / 10.0);
        let mut peaks = Vec::new();
        for i in 1..self.power.len() - 1 {
            let p = self.power[i];
            if p > floor && p > self.power[i - 1] && p >= self.power[i + 1] {
                // Parabolic refinement of the bin position, in log power.
                let (l, c, r) = (self.power[i - 1].max(1e-30).ln(), p.ln(), self.power[i + 1].max(1e-30).ln());
                let denom = l - 2.0 * c + r;
                let delta = if denom.abs() > 1e-9 { 0.5 * (l - r) / denom } else { 0.0 };
                peaks.push(((i as f32 + delta) * self.bin_hz, p));
            }
        }
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        peaks
    }
}

fn nearest_harmonic(hz: f32, f0: f32) -> (f32, f32) {
    let n = (hz / f0).round().max(1.0);
    (n, n * f0)
}

/// `(harmonicity, inharmonicity)`: the first over the spectrum's bins,
/// the second over the peak list.
fn harmonic_measures(peaks: &[(f32, f32)], spectrum: &MeanSpectrum, f0: f32) -> (f32, f32) {
    let mut harmonic_power = 0.0f32;
    for (i, &p) in spectrum.power.iter().enumerate() {
        let hz = i as f32 * spectrum.bin_hz;
        let (_, target) = nearest_harmonic(hz, f0);
        let reach = (HARMONIC_TOLERANCE * target).max(2.0 * spectrum.bin_hz);
        if (hz - target).abs() <= reach {
            harmonic_power += p;
        }
    }
    let total_power = spectrum.total_power();
    let harmonicity = if total_power > 0.0 { (harmonic_power / total_power).min(1.0) } else { 0.0 };
    let strongest: Vec<&(f32, f32)> = peaks.iter().take(20).collect();
    let weight: f32 = strongest.iter().map(|(_, p)| p).sum();
    let deviation: f32 = strongest.iter().map(|&&(hz, p)| (hz - nearest_harmonic(hz, f0).1).abs() * p).sum();
    let inharmonicity = if weight > 0.0 { (2.0 / f0 * deviation / weight).min(1.0) } else { 0.0 };
    (harmonicity, inharmonicity)
}

fn centroid_of(chunk: &[f32], sample_rate: u32) -> f32 {
    // A small DFT on the chunk; 50 ms windows are too short for FRAME.
    let n = chunk.len();
    if n < 64 {
        return 0.0;
    }
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let window = hann(n);
    let mut input: Vec<f32> = chunk.iter().zip(&window).map(|(s, w)| s * w).collect();
    let mut output = fft.make_output_vec();
    fft.process(&mut input, &mut output).expect("fft");
    let bin_hz = sample_rate as f32 / n as f32;
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for (i, bin) in output.iter().enumerate() {
        let p = bin.norm_sqr();
        num += i as f32 * bin_hz * p;
        den += p;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

fn trajectory(mono: &[f32], sample_rate: u32, f: impl Fn(&[f32]) -> f32) -> Vec<f32> {
    let window = (0.05 * sample_rate as f32) as usize;
    mono.chunks(window.max(1)).filter(|c| c.len() == window).map(f).collect()
}

// ------------------------------------------------------------ envelope

struct Envelope {
    /// RMS per 5 ms hop, linear.
    rms: Vec<f32>,
    hop_seconds: f32,
}

impl Envelope {
    fn new(mono: &[f32], sample_rate: u32) -> Envelope {
        let hop = ((0.005 * sample_rate as f32) as usize).max(1);
        let rms = mono
            .chunks(hop)
            .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        Envelope { rms, hop_seconds: hop as f32 / sample_rate as f32 }
    }

    fn peak_index(&self) -> usize {
        self.rms.iter().enumerate().fold((0, 0.0f32), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0
    }

    fn attack_seconds(&self) -> f32 {
        let at = self.peak_index();
        let peak = self.rms.get(at).copied().unwrap_or(0.0);
        if peak <= 0.0 {
            return 0.0;
        }
        let low = (0..=at).find(|&i| self.rms[i] >= 0.1 * peak).unwrap_or(at);
        let high = (low..=at).find(|&i| self.rms[i] >= 0.9 * peak).unwrap_or(at);
        (high - low) as f32 * self.hop_seconds
    }

    fn decay_seconds(&self, drop_db: f32) -> Option<f32> {
        let at = self.peak_index();
        let peak = self.rms.get(at).copied().unwrap_or(0.0);
        let target = peak * 10f32.powf(-drop_db / 20.0);
        (at..self.rms.len()).find(|&i| self.rms[i] <= target).map(|i| (i - at) as f32 * self.hop_seconds)
    }
}

// ------------------------------------------------------------ YIN

/// YIN on the loudest 100 ms: window 4096, lags for 30 Hz..4 kHz,
/// cumulative-mean-normalised difference, absolute threshold 0.15 with
/// the local minimum rule, parabolic refinement.
pub fn yin(mono: &[f32], sample_rate: u32) -> Option<f32> {
    const WINDOW: usize = 4096;
    const THRESHOLD: f32 = 0.15;
    if mono.len() < 2 * WINDOW {
        return None;
    }
    // The loudest 100 ms decides where to look.
    let region = (0.1 * sample_rate as f32) as usize;
    let hop = region / 4;
    let mut best = (0usize, -1.0f32);
    let mut start = 0;
    while start + region <= mono.len() {
        let energy: f32 = mono[start..start + region].iter().map(|v| v * v).sum();
        if energy > best.1 {
            best = (start, energy);
        }
        start += hop.max(1);
    }
    let start = best.0.min(mono.len() - 2 * WINDOW);
    let x = &mono[start..start + 2 * WINDOW];

    let min_lag = (sample_rate as f32 / 4000.0) as usize;
    let max_lag = ((sample_rate as f32 / 30.0) as usize).min(WINDOW - 1);
    let mut difference = vec![0.0f32; max_lag + 1];
    for (tau, d) in difference.iter_mut().enumerate().skip(1) {
        let mut sum = 0.0f32;
        for j in 0..WINDOW {
            let delta = x[j] - x[j + tau];
            sum += delta * delta;
        }
        *d = sum;
    }
    let mut cmnd = vec![1.0f32; max_lag + 1];
    let mut running = 0.0f32;
    for tau in 1..=max_lag {
        running += difference[tau];
        cmnd[tau] = if running > 0.0 { difference[tau] * tau as f32 / running } else { 1.0 };
    }
    let mut tau = min_lag.max(2);
    let mut chosen = None;
    while tau < max_lag {
        if cmnd[tau] < THRESHOLD {
            while tau + 1 < max_lag && cmnd[tau + 1] < cmnd[tau] {
                tau += 1;
            }
            chosen = Some(tau);
            break;
        }
        tau += 1;
    }
    let tau = chosen?;
    let (l, c, r) = (cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]);
    let denom = l - 2.0 * c + r;
    let refined = tau as f32 + if denom.abs() > 1e-9 { 0.5 * (l - r) / denom } else { 0.0 };
    Some(sample_rate as f32 / refined)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, seconds: f32, amplitude: f32, sample_rate: u32) -> Vec<f32> {
        let n = (seconds * sample_rate as f32) as usize;
        (0..n)
            .flat_map(|i| {
                let v = amplitude * (2.0 * core::f32::consts::PI * hz * i as f32 / sample_rate as f32).sin();
                [v, v]
            })
            .collect()
    }

    #[test]
    fn a_full_scale_sine_reads_minus_three_dbfs_rms_and_its_band() {
        let d = describe(&sine(1000.0, 1.0, 1.0, 44100), 44100);
        assert!((d.rms_dbfs + 3.01).abs() < 0.05, "{}", d.rms_dbfs);
        assert!((d.peak_dbfs).abs() < 0.01);
        // 1 kHz sits in band 640–1280: that band carries the whole power.
        assert!((d.bands_dbfs[4] + 3.01).abs() < 0.2, "{:?}", d.bands_dbfs);
        assert!(d.bands_dbfs[0] < -60.0 && d.bands_dbfs[7] < -60.0);
    }

    #[test]
    fn loudness_of_a_997hz_sine_matches_bs1770() {
        // BS.1770-4 Annex: a 997 Hz sine at −20 dBFS reads −20.0 LKFS
        // (the K-weighting is 0 dB at 1 kHz by construction).
        let d = describe(&sine(997.0, 2.0, 0.1, 48000), 48000);
        let lufs = d.loudness_lufs.expect("gated loudness");
        assert!((lufs + 20.0).abs() < 0.15, "{lufs}");
        let d = describe(&sine(997.0, 2.0, 0.1, 44100), 44100);
        assert!((d.loudness_lufs.unwrap() + 20.0).abs() < 0.15, "{:?}", d.loudness_lufs);
    }

    #[test]
    fn yin_finds_a_sine_and_declines_noise() {
        let d = describe(&sine(220.0, 1.0, 0.5, 44100), 44100);
        let f0 = d.f0_hz.expect("voiced");
        assert!((f0 - 220.0).abs() < 0.5, "{f0}");
        assert!(d.harmonicity.unwrap() > 0.95);
        assert!(d.inharmonicity.unwrap() < 0.01);
        let mut seed = 12345u32;
        let noise: Vec<f32> = (0..88200)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed as f32 / u32::MAX as f32) - 0.5
            })
            .collect();
        assert!(describe(&noise, 44100).f0_hz.is_none());
    }

    #[test]
    fn clipping_counts_flat_tops_not_single_peaks() {
        // At 0.99 nothing reaches the threshold; at 1.0 a 100 Hz sine
        // spends six samples per half cycle above 0.999, which IS a flat
        // top for this purpose — the test is about the run rule, so the
        // clean case stays below the threshold.
        let mut s = sine(100.0, 0.5, 0.99, 44100);
        assert_eq!(describe(&s, 44100).clipping.samples, 0);
        for v in s.iter_mut() {
            *v = (*v * 4.0).clamp(-1.0, 1.0);
        }
        assert!(describe(&s, 44100).clipping.ratio > 0.5);
    }

    #[test]
    fn attack_and_decay_follow_the_envelope() {
        let sr = 44100;
        let mut s = sine(440.0, 1.0, 0.8, sr);
        for (i, v) in s.iter_mut().enumerate() {
            let t = (i / 2) as f32 / sr as f32;
            let env = if t < 0.1 { t / 0.1 } else { (-(t - 0.1) * 23.0).exp() };
            *v *= env;
        }
        let d = describe(&s, sr);
        assert!((d.attack_seconds - 0.08).abs() < 0.015, "{}", d.attack_seconds);
        // exp(−23 t) hits −20 dB at t = ln(10)/23 = 0.100 s.
        let decay = d.decay_20db_seconds.expect("decays");
        assert!((decay - 0.1).abs() < 0.02, "{decay}");
    }
}
