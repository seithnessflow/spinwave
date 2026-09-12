//! The honest aliasing measure: the same patch a semitone apart, and the
//! partials that do not follow the key.
//!
//! A harmonic, an FM sideband, a ring-mod product, a formant peak on a
//! keytracked source: all of these move up by the same ratio when the
//! note moves up a semitone. An aliased component is a reflection off
//! Nyquist, so it moves the other way, or by a different amount. Noise
//! has no prominent peaks at all. So: render at `n` and at `n + 1`, take
//! the prominent peaks of the higher render above 2 kHz, and count as
//! aliasing the power of those with no counterpart in the lower render at
//! `f / 2^(1/12)`. A single-render proxy (non-harmonic peaks above 4 kHz)
//! was tried first and removed: it blamed FM sidebands, and above a few
//! kHz every peak sits within 3 % of some harmonic anyway. This costs one
//! more render and measures the thing itself — validated by the fact
//! that its reading falls to zero as the oversampling rises.
//!
//! Prominence keeps noise out: a peak counts only if it stands 8 dB above
//! the median of its surroundings (±24 bins, ~500 Hz at 2048/44.1k).
//!
//! What it measures is exactly "partials that do not follow the key", so
//! a source that does not track the key reads as aliasing: an FM
//! modulator with `midi_track` off, a sample played at a fixed rate, a
//! noise with resonant peaks. Explain will rank such a switch first
//! (measured: `osc_2_midi_track` off on the FM bell reads 0.33). That is
//! the measure being literal, not wrong; read the ranked names.

use serde::Serialize;
use spinwave_params::Preset;

use super::descriptors::MeanSpectrum;
use super::{render, render_seed, OpError, Scenario};
use crate::session::{Session, SAMPLE_RATE};

/// Peaks above this are examined; below it, aliasing of a synth's
/// harmonics is rare and pitch tracking is ambiguous.
const MIN_HZ: f32 = 2000.0;
const PROMINENCE_DB: f32 = 8.0;
const SURROUND_BINS: usize = 24;
/// A counterpart within this fraction of the expected frequency explains
/// the peak.
const TRACK_TOLERANCE: f32 = 0.015;

#[derive(Clone, Debug, Serialize)]
pub struct AliasingReport {
    /// Power of the unexplained prominent peaks above 2 kHz over the
    /// power of all prominent peaks above 2 kHz, in the higher render.
    pub ratio: f32,
    /// The unexplained peaks: `(hz, dBFS)`, strongest first.
    pub unexplained: Vec<(f32, f32)>,
    pub explained_peaks: usize,
    pub renders: usize,
}

/// Prominent peaks of a mean spectrum above `min_hz`: `(hz, power)`.
fn prominent_peaks(spectrum: &MeanSpectrum, min_hz: f32) -> Vec<(f32, f32)> {
    let power = spectrum.power();
    let bin_hz = spectrum.bin_hz();
    let mut peaks = Vec::new();
    for i in 1..power.len() - 1 {
        let hz = i as f32 * bin_hz;
        if hz < min_hz {
            continue;
        }
        let p = power[i];
        if p <= power[i - 1] || p < power[i + 1] || p <= 0.0 {
            continue;
        }
        let lo = i.saturating_sub(SURROUND_BINS);
        let hi = (i + SURROUND_BINS).min(power.len() - 1);
        let mut around: Vec<f32> = power[lo..=hi].to_vec();
        around.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let median = around[around.len() / 2].max(1e-18);
        if 10.0 * (p / median).log10() >= PROMINENCE_DB {
            peaks.push((hz, p));
        }
    }
    peaks
}

/// Measures aliasing on two renders a semitone apart (the scenario's
/// notes, then every note one semitone up), with the same seed.
pub fn aliasing(preset: &Preset, scenario: &Scenario, seed: u64) -> Result<AliasingReport, OpError> {
    let mut session = super::session();
    aliasing_in(&mut session, preset, scenario, seed)
}

pub(crate) fn aliasing_in(session: &mut Session, preset: &Preset, scenario: &Scenario, seed: u64) -> Result<AliasingReport, OpError> {
    aliasing_with_samples(session, preset, scenario, seed).map(|(report, _)| report)
}

/// The report plus the lower render's samples, for callers that also
/// need the audio at the scenario's own pitch (explain's distance).
pub(crate) fn aliasing_with_samples(session: &mut Session, preset: &Preset, scenario: &Scenario, seed: u64) -> Result<(AliasingReport, Vec<f32>), OpError> {
    let low = render(session, preset, scenario, render_seed(seed, 0))?;
    // Silence has no peaks and would read as perfectly clean.
    if low.self_test.peak_dbfs < -60.0 {
        return Err(OpError::Silent { peak_dbfs: low.self_test.peak_dbfs });
    }
    let mut up = scenario.clone();
    for n in &mut up.notes {
        n.note = (n.note + 1).min(127);
    }
    let high = render(session, preset, &up, render_seed(seed, 0))?;
    let mono = |s: &[f32]| -> Vec<f32> { s.chunks_exact(2).map(|f| 0.5 * (f[0] + f[1])).collect() };
    let spectrum_low = MeanSpectrum::new(&mono(&low.samples), SAMPLE_RATE);
    let spectrum_high = MeanSpectrum::new(&mono(&high.samples), SAMPLE_RATE);
    let peaks_low = prominent_peaks(&spectrum_low, MIN_HZ / 2.0);
    let peaks_high = prominent_peaks(&spectrum_high, MIN_HZ);
    let ratio = 2f32.powf(1.0 / 12.0);
    let mut unexplained = Vec::new();
    let mut explained = 0usize;
    let mut total = 0.0f32;
    let mut stray = 0.0f32;
    for &(hz, p) in &peaks_high {
        total += p;
        let expected = hz / ratio;
        let tracks = peaks_low.iter().any(|&(h, _)| (h - expected).abs() <= TRACK_TOLERANCE * expected);
        if tracks {
            explained += 1;
        } else {
            stray += p;
            unexplained.push((hz, 10.0 * p.max(1e-18).log10()));
        }
    }
    unexplained.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    let report = AliasingReport {
        ratio: if total > 0.0 { stray / total } else { 0.0 },
        unexplained,
        explained_peaks: explained,
        renders: 2,
    };
    Ok((report, low.samples))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::saw_patch;
    use crate::ops::RenderMode;

    fn high_scenario() -> Scenario {
        // High enough that a saw's harmonics reach Nyquist and alias
        // without oversampling; the lite mode has none.
        Scenario::one_note(96, 0.4, 0.6, RenderMode::Lite)
    }

    #[test]
    fn aliasing_falls_with_oversampling() {
        let saw = saw_patch();
        let clean = aliasing(&saw, &high_scenario(), 1).expect("measures");
        assert!(clean.ratio < 0.05, "band-limited saw at C7: {:?}", clean);

        // An FM bell: a saw carrier phase-modulated by a sine 17 semitones
        // up (an inharmonic ratio). Its sidebands track the key, so they
        // are not aliasing; what IS aliasing is the fold-over of the
        // sidebands past Nyquist, which oversampling removes. Measured
        // 2026-09-12: ratio 0.104 at 1x, 0.007 at 2x, 0.000 at 4x and 8x.
        let mut fm = saw_patch();
        for (k, v) in [
            ("osc_1_distortion_type", 7.0), ("osc_1_distortion_amount", 0.6), ("osc_2_on", 1.0), ("osc_2_level", 0.0),
            ("osc_2_transpose", 17.0), ("osc_2_wave_frame", 0.0), ("filter_1_on", 0.0),
        ] {
            fm.settings.values.insert(k.into(), serde_json::Value::from(v));
        }
        let scenario = Scenario::one_note(72, 0.4, 0.6, RenderMode::Faithful);
        let mut ratios = Vec::new();
        for os in [0.0, 1.0, 2.0] {
            fm.settings.values.insert("oversampling".into(), os.into());
            ratios.push(aliasing(&fm, &scenario, 1).expect("measures").ratio);
        }
        assert!(ratios[0] > 0.05, "no oversampling: the bell aliases: {ratios:?}");
        assert!(ratios[1] < ratios[0] / 5.0 && ratios[2] <= ratios[1], "oversampling removes it: {ratios:?}");
        assert!(ratios[2] < 0.005, "at 4x nothing is left: {ratios:?}");
    }
}
