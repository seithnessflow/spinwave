//! Clock-based listening: walk a track with a virtual clock, observe a
//! frame of descriptors per step, then interpret the timeline into musical
//! structure (sections, entries, drops) the way a listener follows a song.

use realfft::RealFftPlanner;
use serde::Serialize;

const FFT_SIZE: usize = 2048;

/// One observation step (default 0.5 s of audio).
#[derive(Serialize, Clone)]
pub struct Frame {
    /// Time in seconds from the start of the listened segment.
    pub t: f32,
    /// Absolute levels (dBFS).
    pub rms_db: f32,
    pub sub_db: f32,
    pub bass_db: f32,
    pub mid_db: f32,
    pub high_db: f32,
    pub centroid_hz: f32,
    pub flatness: f32,
    pub width: f32,
    pub onsets: u32,
}

#[derive(Serialize)]
pub struct ListenEvent {
    pub t: f32,
    pub label: String,
}

#[derive(Serialize)]
pub struct Timeline {
    pub step_seconds: f32,
    pub bpm_estimate: Option<f32>,
    pub events: Vec<ListenEvent>,
    /// Narrative summary of the structure, one line per section.
    pub narrative: Vec<String>,
    pub frames: Vec<Frame>,
}

fn to_db(x: f32) -> f32 {
    20.0 * x.max(1e-9).log10()
}

/// Walks the interleaved stereo audio with a virtual clock.
pub fn listen(interleaved: &[f32], sample_rate: u32, step_seconds: f32) -> Timeline {
    let step_seconds = step_seconds.clamp(0.1, 2.0);
    let frames_len = interleaved.len() / 2;
    let step = ((step_seconds * sample_rate as f32) as usize).max(FFT_SIZE);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let bin_hz = sample_rate as f32 / FFT_SIZE as f32;

    let onset_frame = (sample_rate as usize / 100).max(1);

    let mut frames: Vec<Frame> = Vec::new();
    let mut start = 0usize;
    while start + step <= frames_len {
        let mut rms = 0.0f32;
        let mut sum_lr = 0.0f64;
        let mut sum_ll = 0.0f64;
        let mut sum_rr = 0.0f64;
        for i in start..start + step {
            let l = interleaved[2 * i];
            let r = interleaved[2 * i + 1];
            let mono = (l + r) * 0.5;
            rms += mono * mono;
            sum_lr += (l as f64) * (r as f64);
            sum_ll += (l as f64) * (l as f64);
            sum_rr += (r as f64) * (r as f64);
        }
        rms = (rms / step as f32).sqrt();
        let denom = (sum_ll * sum_rr).sqrt();
        let width = if denom > 1e-12 {
            (1.0 - ((sum_lr / denom) as f32).abs()).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Spectrum of the step's center window.
        let window_start = start + step / 2 - FFT_SIZE.min(step) / 2;
        for (slot, i) in input.iter_mut().zip(window_start..) {
            let mono = (interleaved[2 * i] + interleaved[2 * i + 1]) * 0.5;
            let t = (i - window_start) as f32 / (FFT_SIZE - 1) as f32;
            let hann = 0.5 - 0.5 * (2.0 * core::f32::consts::PI * t).cos();
            *slot = mono * hann;
        }
        let (mut sub, mut bass, mut mid, mut high) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut centroid_num = 0.0f32;
        let mut magnitude_sum = 0.0f32;
        let mut log_sum = 0.0f32;
        let mut bins = 0usize;
        if fft.process(&mut input, &mut output).is_ok() {
            for (bin, value) in output.iter().enumerate().skip(1) {
                let hz = bin as f32 * bin_hz;
                let power = value.norm_sqr();
                match hz {
                    h if h < 60.0 => sub += power,
                    h if h < 250.0 => bass += power,
                    h if h < 4000.0 => mid += power,
                    _ => high += power,
                }
                let magnitude = power.sqrt();
                centroid_num += hz * magnitude;
                magnitude_sum += magnitude;
                log_sum += magnitude.max(1e-12).ln();
                bins += 1;
            }
        }
        // Band level calibration: a lone sine of amplitude A under a Hann
        // window puts |X| = A*N/4 in its bin, so amplitude = 4*sqrt(P)/N
        // and RMS = amplitude / sqrt(2). A full-scale sine reads ~-3 dBFS.
        let amp_scale = 4.0 / (FFT_SIZE as f32 * core::f32::consts::SQRT_2);
        let band_db = |power: f32| to_db(power.sqrt() * amp_scale);
        let flatness = if bins > 0 && magnitude_sum > 1e-9 {
            ((log_sum / bins as f32).exp() / (magnitude_sum / bins as f32)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Onsets inside the step (10 ms energy rises).
        let mut onsets = 0u32;
        let mut history = [0.0f32; 5];
        let mut history_pos = 0usize;
        let mut refractory = 0i32;
        let mut position = start;
        while position + onset_frame <= start + step {
            let mut energy = 0.0f32;
            for i in position..position + onset_frame {
                let mono = (interleaved[2 * i] + interleaved[2 * i + 1]) * 0.5;
                energy += mono * mono;
            }
            energy = (energy / onset_frame as f32).sqrt();
            let mean: f32 = history.iter().sum::<f32>() / 5.0;
            if refractory <= 0 && energy > mean * 1.6 && energy > 1e-4 {
                onsets += 1;
                refractory = 5;
            }
            refractory -= 1;
            history[history_pos % 5] = energy;
            history_pos += 1;
            position += onset_frame;
        }

        frames.push(Frame {
            t: start as f32 / sample_rate as f32,
            rms_db: to_db(rms),
            sub_db: band_db(sub),
            bass_db: band_db(bass),
            mid_db: band_db(mid),
            high_db: band_db(high),
            centroid_hz: if magnitude_sum > 1e-9 { centroid_num / magnitude_sum } else { 0.0 },
            flatness,
            width,
            onsets,
        });
        start += step;
    }

    let bpm_estimate = estimate_bpm(interleaved, sample_rate);
    let events = detect_events(&frames, step_seconds);
    let narrative = narrate(&frames, &events);

    Timeline { step_seconds, bpm_estimate, events, narrative, frames }
}

/// BPM from the autocorrelation of a 10 ms onset-strength envelope.
fn estimate_bpm(interleaved: &[f32], sample_rate: u32) -> Option<f32> {
    let frame = (sample_rate as usize / 100).max(1);
    let frames_len = interleaved.len() / 2;
    let mut envelope: Vec<f32> = Vec::with_capacity(frames_len / frame);
    let mut previous = 0.0f32;
    let mut position = 0usize;
    while position + frame <= frames_len {
        let mut energy = 0.0f32;
        for i in position..position + frame {
            let mono = (interleaved[2 * i] + interleaved[2 * i + 1]) * 0.5;
            energy += mono * mono;
        }
        energy = (energy / frame as f32).sqrt();
        envelope.push((energy - previous).max(0.0)); // onset strength
        previous = energy;
        position += frame;
    }
    if envelope.len() < 400 {
        return None;
    }
    let mean = envelope.iter().sum::<f32>() / envelope.len() as f32;
    for value in &mut envelope {
        *value -= mean;
    }
    // Beat period search: 60â€“200 BPM â†’ 100..333 envelope frames per beat*?
    // envelope rate = 100 fps â†’ beat lag = 6000/bpm frames... 100*60/bpm.
    let energy: f32 = envelope.iter().map(|v| v * v).sum();
    if energy < 1e-9 {
        return None;
    }
    let mut correlations = vec![0.0f32; 101];
    let mut best = 0.0f32;
    for lag in 30..=100usize {
        // 200 down to 60 BPM
        let mut corr = 0.0f32;
        for i in 0..envelope.len() - lag {
            corr += envelope[i] * envelope[i + lag];
        }
        correlations[lag] = corr;
        best = best.max(corr);
    }
    if best <= 0.0 {
        return None;
    }
    // Among near-equal peaks, prefer the SHORTEST lag (the true beat
    // period rather than its multiples), with parabolic refinement to
    // resolve fractional periods.
    let mut chosen = 0usize;
    for lag in 31..100usize {
        let value = correlations[lag];
        if value >= best * 0.85
            && value >= correlations[lag - 1]
            && value >= correlations[lag + 1]
        {
            chosen = lag;
            break;
        }
    }
    if chosen == 0 {
        return None;
    }
    let previous = correlations[chosen - 1];
    let next = correlations[chosen + 1];
    let denom = previous - 2.0 * correlations[chosen] + next;
    let refine = if denom.abs() > 1e-9 { 0.5 * (previous - next) / denom } else { 0.0 };
    let mut bpm = 6000.0 / (chosen as f32 + refine.clamp(-0.5, 0.5));
    // Fold into the DnB/house-friendly 80..190 window.
    while bpm < 80.0 {
        bpm *= 2.0;
    }
    while bpm > 190.0 {
        bpm /= 2.0;
    }
    Some(bpm)
}

fn detect_events(frames: &[Frame], step_seconds: f32) -> Vec<ListenEvent> {
    let mut events = Vec::new();
    if frames.len() < 8 {
        return events;
    }

    // A listener hears trends, not per-beat jitter: smooth every series
    // over ~2 s and compare against a ~2 s trailing baseline, then debounce
    // per event type.
    let window = ((2.0 / step_seconds) as usize).clamp(2, 8);
    let smooth = |select: &dyn Fn(&Frame) -> f32| -> Vec<f32> {
        (0..frames.len())
            .map(|i| {
                let from = i.saturating_sub(window - 1);
                let slice = &frames[from..=i];
                slice.iter().map(|f| select(f)).sum::<f32>() / slice.len() as f32
            })
            .collect()
    };
    let rms = smooth(&|f| f.rms_db);
    let sub = smooth(&|f| f.sub_db);
    let width = smooth(&|f| f.width);
    let onsets = smooth(&|f| f.onsets as f32);

    let debounce_steps = (6.0 / step_seconds) as usize;
    let mut last: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut fire = |events: &mut Vec<ListenEvent>, kind: &'static str, i: usize, label: String| {
        if last.get(kind).map(|&j| i - j >= debounce_steps).unwrap_or(true) {
            events.push(ListenEvent { t: frames[i].t, label });
            last.insert(kind, i);
        }
    };

    let sustain = (1.5 / step_seconds).ceil() as usize;
    for i in window..frames.len().saturating_sub(sustain) {
        let baseline = rms[i - window + 1..i].iter().sum::<f32>() / (window - 1).max(1) as f32;
        let jump = rms[i + sustain - 1] - baseline;

        if jump >= 5.0 && rms[i..i + sustain].iter().all(|&v| v > baseline + 3.0) {
            let frame = &frames[i + sustain - 1];
            let mono = if width[i + sustain - 1] < 0.12 { ", mono" } else { "" };
            fire(
                &mut events,
                "drop",
                i,
                format!(
                    "DROP (+{:.0} dB, sub {:.0} dBFS, flatness {:.2}{})",
                    jump, frame.sub_db, frame.flatness, mono
                ),
            );
        } else if jump <= -6.0 && rms[i..i + sustain].iter().all(|&v| v < baseline - 4.0) {
            fire(&mut events, "drop", i, "breakdown (energy falls)".into());
        }

        // Sub entry: quiet lows for the whole baseline window, then present.
        let sub_baseline = sub[i - window + 1..i].iter().cloned().fold(f32::MIN, f32::max);
        if sub_baseline < -35.0 && sub[i] > -25.0 {
            fire(&mut events, "sub", i, "sub/bass enters".into());
        }

        // Stereo stance changes with hysteresis on the smoothed width.
        let width_before = width[i - window + 1];
        if width_before < 0.15 && width[i] > 0.30 {
            fire(&mut events, "width", i, "stereo opens up".into());
        } else if width_before > 0.30 && width[i] < 0.12 {
            fire(&mut events, "width", i, "collapses to mono".into());
        }

        if onsets[i] - onsets[i - window + 1] >= 3.0 {
            fire(&mut events, "drums", i, "drums/percussion densify".into());
        }
    }

    // Buildups: sustained centroid + level rise over >= 4 steps ending in a
    // DROP event.
    let drop_times: Vec<f32> = events
        .iter()
        .filter(|e| e.label.starts_with("DROP"))
        .map(|e| e.t)
        .collect();
    for &drop_t in &drop_times {
        let end = frames.iter().position(|f| f.t >= drop_t).unwrap_or(0);
        if end >= 5 {
            let run = &frames[end - 5..end];
            let rising = run.windows(2).filter(|w| w[1].centroid_hz > w[0].centroid_hz).count();
            if rising >= 3 {
                events.push(ListenEvent {
                    t: run[0].t,
                    label: "buildup (brightness/tension rising)".into(),
                });
            }
        }
    }
    events.sort_by(|a, b| a.t.total_cmp(&b.t));
    events
}

fn narrate(frames: &[Frame], events: &[ListenEvent]) -> Vec<String> {
    let mut narrative = Vec::new();
    if let Some(first) = frames.first() {
        narrative.push(format!(
            "0:00 start â€” rms {:.0} dB, centroid {:.0} Hz, width {:.2}, flatness {:.2}",
            first.rms_db, first.centroid_hz, first.width, first.flatness
        ));
    }
    for event in events {
        let minutes = (event.t / 60.0) as u32;
        let seconds = event.t % 60.0;
        narrative.push(format!("{}:{:04.1} {}", minutes, seconds, event.label));
    }
    narrative
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo(mono: &[f32]) -> Vec<f32> {
        mono.iter().flat_map(|&v| [v, v]).collect()
    }

    #[test]
    fn detects_a_drop_between_quiet_and_loud_sections() {
        let sr = 44100usize;
        let mut mono = Vec::new();
        // 4 s quiet sine, then 4 s loud saw-ish buzz.
        for i in 0..sr * 4 {
            mono.push((2.0 * core::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin() * 0.05);
        }
        for i in 0..sr * 4 {
            let phase = (i as f32 * 110.0 / sr as f32).fract();
            mono.push((phase * 2.0 - 1.0) * 0.6);
        }
        let timeline = listen(&stereo(&mono), sr as u32, 0.5);
        assert!(
            timeline.events.iter().any(|e| e.label.starts_with("DROP") && (e.t - 4.0).abs() < 1.0),
            "no drop detected: {:?}",
            timeline.events.iter().map(|e| format!("{} {}", e.t, e.label)).collect::<Vec<_>>()
        );
        // Loud section frames are measurably louder and brighter.
        let quiet = &timeline.frames[2];
        let loud = timeline.frames.iter().find(|f| f.t > 5.0).unwrap();
        assert!(loud.rms_db > quiet.rms_db + 10.0);
    }

    #[test]
    fn estimates_a_plausible_bpm_for_a_click_track() {
        let sr = 44100usize;
        let bpm = 174.0f32;
        let beat = (sr as f32 * 60.0 / bpm) as usize;
        let mut mono = vec![0.0f32; sr * 20];
        let mut i = 0;
        while i < mono.len() {
            for j in 0..200.min(mono.len() - i) {
                mono[i + j] = (1.0 - j as f32 / 200.0) * 0.8;
            }
            i += beat;
        }
        let timeline = listen(&stereo(&mono), sr as u32, 0.5);
        let estimate = timeline.bpm_estimate.expect("bpm detected");
        let ratio = estimate / bpm;
        let near = |x: f32, target: f32| (x / target - 1.0).abs() < 0.05;
        assert!(
            near(ratio, 1.0) || near(ratio, 0.5) || near(ratio, 2.0),
            "bpm {estimate} vs {bpm}"
        );
    }

    #[test]
    fn empty_audio_yields_empty_timeline() {
        let timeline = listen(&[], 44100, 0.5);
        assert!(timeline.frames.is_empty());
        assert!(timeline.events.is_empty());
    }
}

