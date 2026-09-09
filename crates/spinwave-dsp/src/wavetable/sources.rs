//! Source components: Wave Source, Shepard Tone Source, Line Source and
//! Audio File Source.
//!
//! Rework of Vital's `wave_source.cpp`, `shepard_tone_source.cpp`,
//! `wave_line_source.cpp` (+ `line_generator.cpp`) and `file_source.cpp`.

use realfft::num_complex::Complex;
use serde_json::Value;

use super::codec::{base64_decode, bytes_to_f32, pcm16_round_trip, pcm_bytes_to_f32, Mt19937};
use super::components::{
    cubic_tween, json_bool, json_f32, json_f64, json_str, json_u64, linear_tween, locate,
    parse_keyframes, power_scale, InterpolationStyle, ScalarKeyframe, Segment,
    LAST_FRAME_POSITION,
};
use super::wave_frame::{WaveFrame, NUM_REAL_COMPLEX, WAVEFORM_SIZE};

/// How the JSON payload encodes embedded buffers; depends on the preset's
/// version (see `WavetableCreator::updateJson` in the reference).
#[derive(Clone, Copy, Default)]
pub(crate) struct LoadContext {
    /// `wave_data` stored as 16-bit PCM (versions `0.3.7..0.3.9`).
    pub wave_data_pcm: bool,
    /// `audio_file` stored as raw floats (versions before `0.3.7`).
    pub audio_file_float: bool,
    /// Line Source keyframes in the pre-`0.7.7` points format.
    pub line_old_format: bool,
}

// ---------------------------------------------------------------------------
// Frequency/time interpolation between whole wave frames (wave_source.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum InterpolationMode {
    Time,
    Frequency,
}

fn linear_time_interpolate(dest: &mut WaveFrame, from: &WaveFrame, to: &WaveFrame, t: f32) {
    for i in 0..WAVEFORM_SIZE {
        dest.time_domain[i] = linear_tween(from.time_domain[i], to.time_domain[i], t);
    }
    dest.to_frequency_domain();
}

#[allow(clippy::too_many_arguments)]
fn cubic_time_interpolate(
    dest: &mut WaveFrame,
    prev: &WaveFrame,
    from: &WaveFrame,
    to: &WaveFrame,
    next: &WaveFrame,
    range_prev: f32,
    range: f32,
    range_next: f32,
    t: f32,
) {
    for i in 0..WAVEFORM_SIZE {
        dest.time_domain[i] = cubic_tween(
            prev.time_domain[i],
            from.time_domain[i],
            to.time_domain[i],
            next.time_domain[i],
            range_prev,
            range,
            range_next,
            t,
        );
    }
    dest.to_frequency_domain();
}

pub(crate) fn linear_frequency_interpolate(
    dest: &mut WaveFrame,
    from: &WaveFrame,
    to: &WaveFrame,
    t: f32,
) {
    for i in 0..NUM_REAL_COMPLEX {
        let amplitude_from = from.frequency_domain[i].norm().sqrt();
        let amplitude_to = to.frequency_domain[i].norm().sqrt();
        let mut amplitude = linear_tween(amplitude_from, amplitude_to, t);
        amplitude *= amplitude;

        let phase_from = from.frequency_domain[i].arg();
        let phase_delta = (from.frequency_domain[i].conj() * to.frequency_domain[i]).arg();
        let mut phase = phase_from + t * phase_delta;
        if amplitude_from == 0.0 {
            phase = to.frequency_domain[i].arg();
        }
        dest.frequency_domain[i] = Complex::from_polar(amplitude, phase);
    }

    let dc = linear_tween(from.frequency_domain[0].re, to.frequency_domain[0].re, t);
    dest.frequency_domain[0] = Complex::new(dc, 0.0);

    let last = NUM_REAL_COMPLEX - 1;
    let last_harmonic =
        linear_tween(from.frequency_domain[last].re, to.frequency_domain[last].re, t);
    dest.frequency_domain[last] = Complex::new(last_harmonic, 0.0);

    dest.to_time_domain();
}

#[allow(clippy::too_many_arguments)]
fn cubic_frequency_interpolate(
    dest: &mut WaveFrame,
    prev: &WaveFrame,
    from: &WaveFrame,
    to: &WaveFrame,
    next: &WaveFrame,
    range_prev: f32,
    range: f32,
    range_next: f32,
    t: f32,
) {
    for i in 0..NUM_REAL_COMPLEX {
        let amplitude_prev = prev.frequency_domain[i].norm().sqrt();
        let amplitude_from = from.frequency_domain[i].norm().sqrt();
        let amplitude_to = to.frequency_domain[i].norm().sqrt();
        let amplitude_next = next.frequency_domain[i].norm().sqrt();
        let mut amplitude = cubic_tween(
            amplitude_prev,
            amplitude_from,
            amplitude_to,
            amplitude_next,
            range_prev,
            range,
            range_next,
            t,
        );
        amplitude *= amplitude;

        let phase_delta_from = (prev.frequency_domain[i].conj() * from.frequency_domain[i]).arg();
        let phase_delta_to = (from.frequency_domain[i].conj() * to.frequency_domain[i]).arg();
        let phase_delta_next = (to.frequency_domain[i].conj() * next.frequency_domain[i]).arg();
        let phase_prev = prev.frequency_domain[i].arg();
        let mut phase_from = phase_prev;
        if amplitude_from != 0.0 {
            phase_from += phase_delta_from;
        }
        let mut phase_to = phase_from;
        if amplitude_to != 0.0 {
            phase_to += phase_delta_to;
        }
        let mut phase_next = phase_to;
        if amplitude_next != 0.0 {
            phase_next += phase_delta_next;
        }

        let phase = cubic_tween(
            phase_prev, phase_from, phase_to, phase_next, range_prev, range, range_next, t,
        );
        dest.frequency_domain[i] = Complex::from_polar(amplitude, phase);
    }

    let dc = cubic_tween(
        prev.frequency_domain[0].re,
        from.frequency_domain[0].re,
        to.frequency_domain[0].re,
        next.frequency_domain[0].re,
        range_prev,
        range,
        range_next,
        t,
    );
    dest.frequency_domain[0] = Complex::new(dc, 0.0);

    let last = NUM_REAL_COMPLEX - 1;
    let last_harmonic = cubic_tween(
        prev.frequency_domain[last].re,
        from.frequency_domain[last].re,
        to.frequency_domain[last].re,
        next.frequency_domain[last].re,
        range_prev,
        range,
        range_next,
        t,
    );
    dest.frequency_domain[last] = Complex::new(last_harmonic, 0.0);

    dest.to_time_domain();
}

// ---------------------------------------------------------------------------
// Wave Source
// ---------------------------------------------------------------------------

pub(crate) struct WaveSource {
    positions: Vec<i32>,
    frames: Vec<WaveFrame>,
    style: InterpolationStyle,
    mode: InterpolationMode,
}

impl WaveSource {
    /// Builds a wave source directly from `(position, frame)` pairs, for
    /// procedurally generated tables.
    pub(crate) fn from_frames(
        mut keyframes: Vec<(i32, WaveFrame)>,
        style: InterpolationStyle,
        mode: InterpolationMode,
    ) -> WaveSource {
        keyframes.sort_by_key(|(position, _)| *position);
        let mut positions = Vec::with_capacity(keyframes.len());
        let mut frames = Vec::with_capacity(keyframes.len());
        for (position, frame) in keyframes {
            positions.push(position.clamp(0, LAST_FRAME_POSITION));
            frames.push(frame);
        }
        WaveSource {
            positions,
            frames,
            style,
            mode,
        }
    }

    pub(crate) fn from_json(data: &Value, context: &LoadContext) -> Option<WaveSource> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            let encoded = json_str(keyframe, "wave_data")?;
            let bytes = base64_decode(encoded)?;
            let samples = if context.wave_data_pcm {
                pcm_bytes_to_f32(&bytes)
            } else {
                bytes_to_f32(&bytes)
            };
            if samples.len() < WAVEFORM_SIZE {
                return None;
            }
            let mut frame = WaveFrame::new();
            frame.load_time_domain(&samples[..WAVEFORM_SIZE]);
            Some(frame)
        })?;
        let mode = if json_u64(data, "interpolation").unwrap_or(1) == 0 {
            InterpolationMode::Time
        } else {
            InterpolationMode::Frequency
        };
        Some(WaveSource {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
            mode,
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(segment) = locate(&self.positions, &self.frames, self.style, position) else {
            return;
        };
        match segment {
            Segment::Copy(keyframe) => frame.copy_from(keyframe),
            Segment::Linear { from, to, t } => match self.mode {
                InterpolationMode::Frequency => linear_frequency_interpolate(frame, from, to, t),
                InterpolationMode::Time => linear_time_interpolate(frame, from, to, t),
            },
            Segment::Cubic {
                prev,
                from,
                to,
                next,
                range_prev,
                range,
                range_next,
                t,
            } => match self.mode {
                InterpolationMode::Frequency => cubic_frequency_interpolate(
                    frame, prev, from, to, next, range_prev, range, range_next, t,
                ),
                InterpolationMode::Time => cubic_time_interpolate(
                    frame, prev, from, to, next, range_prev, range, range_next, t,
                ),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Shepard Tone Source (shepard_tone_source.cpp)
// ---------------------------------------------------------------------------

pub(crate) struct ShepardSource {
    source: WaveSource,
}

impl ShepardSource {
    pub(crate) fn from_json(data: &Value, context: &LoadContext) -> Option<ShepardSource> {
        Some(ShepardSource {
            source: WaveSource::from_json(data, context)?,
        })
    }

    /// The reference reports no user keyframes for Shepard sources, so the
    /// component spans the whole table when it holds any keyframe.
    pub(crate) fn last_position(&self) -> i32 {
        if self.source.frames.is_empty() {
            0
        } else {
            LAST_FRAME_POSITION
        }
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) = self.source.frames.first() else {
            return;
        };

        // The loop frame doubles every harmonic's index (an octave up),
        // which the source then cross-fades into over the table.
        let mut loop_frame = WaveFrame::new();
        let mut i = 0;
        while 2 * i < NUM_REAL_COMPLEX {
            loop_frame.frequency_domain[2 * i] = keyframe.frequency_domain[i];
            i += 1;
        }
        loop_frame.to_time_domain();

        let t = position / LAST_FRAME_POSITION as f32;
        match self.source.mode {
            InterpolationMode::Frequency => {
                linear_frequency_interpolate(frame, keyframe, &loop_frame, t)
            }
            InterpolationMode::Time => linear_time_interpolate(frame, keyframe, &loop_frame, t),
        }
    }
}

// ---------------------------------------------------------------------------
// Line Source (wave_line_source.cpp + line_generator.cpp)
// ---------------------------------------------------------------------------

/// State of Vital's `LineGenerator` (non-looping, wavetable flavor).
#[derive(Clone)]
pub(crate) struct LineGenerator {
    points: Vec<(f32, f32)>,
    powers: Vec<f32>,
    smooth: bool,
}

impl LineGenerator {
    fn smooth_transition(t: f32) -> f32 {
        0.5 * ((t - 0.5) * std::f32::consts::PI).sin() + 0.5
    }

    pub(crate) fn from_json(data: &Value) -> Option<LineGenerator> {
        let num_points = json_u64(data, "num_points")? as usize;
        let point_data = data.get("points")?.as_array()?;
        let power_data = data.get("powers")?.as_array()?;
        let num_points = num_points.min(power_data.len()).min(point_data.len() / 2);

        let mut points = Vec::with_capacity(num_points);
        let mut powers = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let x = point_data[2 * i].as_f64()? as f32;
            let y = point_data[2 * i + 1].as_f64()? as f32;
            points.push((x, y));
            powers.push(power_data[i].as_f64()? as f32);
        }
        Some(LineGenerator {
            points,
            powers,
            smooth: json_bool(data, "smooth").unwrap_or(false),
        })
    }

    /// Line-generator state converted from the pre-`0.7.7` keyframe format
    /// (`WavetableCreator::updateJson`, `< 0.7.7` block).
    pub(crate) fn from_old_json(keyframe: &Value, num_points: usize) -> Option<LineGenerator> {
        let point_data = keyframe.get("points")?.as_array()?;
        let power_data = keyframe.get("powers")?.as_array()?;
        let num_points = num_points.min(power_data.len()).min(point_data.len() / 2);
        if num_points == 0 {
            return None;
        }

        let mut points = vec![(0.0f32, 0.0f32); num_points + 2];
        let mut powers = vec![0.0f32; num_points + 2];
        for i in 0..num_points {
            let x = point_data[2 * i].as_f64()? as f32;
            let y = point_data[2 * i + 1].as_f64()? as f32;
            points[i + 1] = (x, y * 0.5 + 0.5);
            powers[i + 1] = power_data[i].as_f64()? as f32;
        }

        let (start_x, start_y) = (
            point_data[0].as_f64()? as f32,
            point_data[1].as_f64()? as f32,
        );
        let (end_x, end_y) = (
            point_data[2 * (num_points - 1)].as_f64()? as f32,
            point_data[2 * (num_points - 1) + 1].as_f64()? as f32,
        );

        let range_x = start_x - end_x + 1.0;
        let y = if range_x < 0.001 {
            0.5 * (start_y + end_y)
        } else {
            let t = (1.0 - end_x) / range_x;
            linear_tween(end_y, start_y, t)
        };

        points[0] = (0.0, y * 0.5 + 0.5);
        points[num_points + 1] = (1.0, y * 0.5 + 0.5);
        Some(LineGenerator {
            points,
            powers,
            smooth: false,
        })
    }

    /// Renders the line into `WAVEFORM_SIZE` values in `[0, 1]`
    /// (`LineGenerator::render`, non-looping).
    fn render_into(&self, buffer: &mut [f32]) {
        let num_points = self.points.len();
        if num_points == 0 {
            buffer.fill(0.5);
            return;
        }

        let resolution = buffer.len();
        let mut point_index = 0usize;
        let mut last_point = self.points[0];
        let mut current_power = 0.0f32;
        let mut current_point = self.points[0];

        for (i, value) in buffer.iter_mut().enumerate() {
            let x = i as f32 / (resolution as f32 - 1.0);
            let mut t = 1.0;
            if current_point.0 > last_point.0 {
                t = (x - last_point.0) / (current_point.0 - last_point.0);
            }
            if self.smooth {
                t = Self::smooth_transition(t);
            }
            t = power_scale(t, current_power).clamp(0.0, 1.0);

            let y = last_point.1 + t * (current_point.1 - last_point.1);
            *value = 1.0 - y;

            while x > current_point.0 && point_index < num_points {
                current_power = self.powers[point_index % num_points];
                point_index += 1;
                last_point = current_point;
                current_point = self.points[point_index % num_points];
                if point_index >= num_points {
                    current_point.0 += 1.0;
                    break;
                }
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct LineKeyframe {
    line: LineGenerator,
    pull_power: f32,
}

impl ScalarKeyframe for LineKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        let relative_power = from.pull_power - to.pull_power;
        let adjusted_t = power_scale(t, relative_power);

        let num_points = from.line.points.len().min(to.line.points.len());
        let mut points = Vec::with_capacity(num_points);
        let mut powers = Vec::with_capacity(num_points);
        for i in 0..num_points {
            points.push((
                linear_tween(from.line.points[i].0, to.line.points[i].0, adjusted_t),
                linear_tween(from.line.points[i].1, to.line.points[i].1, adjusted_t),
            ));
            powers.push(linear_tween(from.line.powers[i], to.line.powers[i], adjusted_t));
        }
        LineKeyframe {
            line: LineGenerator {
                points,
                powers,
                smooth: from.line.smooth,
            },
            pull_power: linear_tween(from.pull_power, to.pull_power, t),
        }
    }
}

pub(crate) struct LineSource {
    positions: Vec<i32>,
    frames: Vec<LineKeyframe>,
    style: InterpolationStyle,
}

impl LineSource {
    pub(crate) fn from_json(data: &Value, context: &LoadContext) -> Option<LineSource> {
        let num_points = json_u64(data, "num_points").unwrap_or(0) as usize;
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            let line = if context.line_old_format {
                LineGenerator::from_old_json(keyframe, num_points)?
            } else {
                LineGenerator::from_json(keyframe.get("line")?)?
            };
            Some(LineKeyframe {
                line,
                pull_power: json_f32(keyframe, "pull_power").unwrap_or(0.0),
            })
        })?;
        Some(LineSource {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) =
            super::components::interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        let mut buffer = vec![0.0f32; WAVEFORM_SIZE];
        keyframe.line.render_into(&mut buffer);
        for (sample, value) in frame.time_domain.iter_mut().zip(&buffer) {
            *sample = value * 2.0 - 1.0;
        }
        frame.to_frequency_domain();
    }
}

// ---------------------------------------------------------------------------
// Audio File Source (file_source.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FadeStyle {
    WaveBlend,
    NoInterpolate,
    TimeInterpolate,
    FreqInterpolate,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FilePhaseStyle {
    None,
    Clear,
    Vocode,
}

#[derive(Clone)]
pub(crate) struct FileKeyframe {
    start_position: f64,
    window_fade: f64,
}

impl ScalarKeyframe for FileKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        FileKeyframe {
            start_position: linear_tween(from.start_position as f32, to.start_position as f32, t)
                as f64,
            window_fade: linear_tween(from.window_fade as f32, to.window_fade as f32, t) as f64,
        }
    }
}

pub(crate) struct FileSource {
    positions: Vec<i32>,
    frames: Vec<FileKeyframe>,
    style: InterpolationStyle,
    /// Padded sample data: one duplicated leading sample (so cubic
    /// interpolation can look one sample back) plus trailing repeats.
    padded: Vec<f32>,
    size: usize,
    sample_rate: f32,
    window_size: f64,
    fade_style: FadeStyle,
    phase_style: FilePhaseStyle,
    normalize_gain: bool,
    normalize_mult: bool,
    overridden_phase: Vec<f32>,
}

impl FileSource {
    const EXTRA_BUFFER_SAMPLES: usize = 4;
    /// Smallest accepted `window_size` (samples).
    const MIN_WINDOW_SIZE: f64 = 1.0;

    pub(crate) fn from_json(data: &Value, context: &LoadContext) -> Option<FileSource> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            Some(FileKeyframe {
                start_position: json_f64(keyframe, "start_position")?,
                // The fade is a fraction of the frame; anything outside
                // [0, 1] (or NaN) would make the fade loop run for ~1e18
                // iterations, so clamp at parse time.
                window_fade: {
                    let fade = json_f64(keyframe, "window_fade")?;
                    if fade.is_finite() { fade.clamp(0.0, 1.0) } else { 0.0 }
                },
            })
        })?;

        // A window below one sample (or non-finite) would divide the frame
        // positions by ~0 and fill the table with NaN; clamp like the
        // editor's minimum window.
        let window_size = json_f64(data, "window_size")?;
        let window_size = if window_size.is_finite() {
            window_size.max(Self::MIN_WINDOW_SIZE)
        } else {
            WAVEFORM_SIZE as f64
        };
        let fade_style = match json_u64(data, "fade_style").unwrap_or(0) {
            1 => FadeStyle::NoInterpolate,
            2 => FadeStyle::TimeInterpolate,
            3 => FadeStyle::FreqInterpolate,
            _ => FadeStyle::WaveBlend,
        };
        let phase_style = match json_u64(data, "phase_style").unwrap_or(0) {
            1 => FilePhaseStyle::Clear,
            2 => FilePhaseStyle::Vocode,
            _ => FilePhaseStyle::None,
        };
        let random_seed = data
            .get("random_seed")
            .and_then(|value| value.as_i64())
            .unwrap_or(0);

        let encoded = json_str(data, "audio_file")?;
        let bytes = base64_decode(encoded)?;
        let samples = if context.audio_file_float {
            // Pre-0.3.7 presets store raw floats; Vital's updateJson
            // converts them to 16-bit PCM first (clamping to +/-1 and
            // quantising), so replay that round trip to sound the same.
            pcm16_round_trip(&bytes_to_f32(&bytes))
        } else {
            pcm_bytes_to_f32(&bytes)
        };

        let size = samples.len();
        let mut padded = vec![0.0f32; size + Self::EXTRA_BUFFER_SAMPLES];
        padded[1..1 + size].copy_from_slice(&samples);
        padded[0] = padded.get(1).copied().unwrap_or(0.0);
        let last = padded.get(size).copied().unwrap_or(0.0);
        for value in padded[size + 1..].iter_mut() {
            *value = last;
        }

        let mut overridden_phase = vec![0.0f32; WAVEFORM_SIZE];
        match phase_style {
            FilePhaseStyle::Clear => {
                for i in 0..WAVEFORM_SIZE / 2 {
                    overridden_phase[2 * i] = -0.5 * std::f32::consts::PI;
                    overridden_phase[2 * i + 1] = 0.5 * std::f32::consts::PI;
                }
            }
            FilePhaseStyle::Vocode => {
                let mut rng = Mt19937::new(random_seed as u32);
                for value in &mut overridden_phase {
                    *value = rng.next_in_range(-std::f32::consts::PI, std::f32::consts::PI);
                }
            }
            FilePhaseStyle::None => {}
        }

        Some(FileSource {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
            padded,
            size,
            sample_rate: json_f64(data, "audio_sample_rate").unwrap_or(44100.0) as f32,
            window_size,
            fade_style,
            phase_style,
            normalize_gain: json_bool(data, "normalize_gain").unwrap_or(false),
            normalize_mult: json_bool(data, "normalize_mult").unwrap_or(true),
            overridden_phase,
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    /// Catmull-Rom interpolated sample lookup
    /// (`getScaledInterpolatedSample`).
    fn sample_at(&self, position: f64) -> f32 {
        let clamped = position.clamp(0.0, (self.size.max(1) - 1) as f64);
        let start = clamped as usize;
        let t = (clamped - start as f64) as f32;

        let half_t = t * 0.5;
        let half_t2 = t * half_t;
        let half_t3 = half_t2 * t;
        let half_three_t3 = half_t3 * 3.0;
        let w0 = half_t2 * 2.0 - half_t3 - half_t;
        let w1 = half_three_t3 - half_t2 * 5.0 + 1.0;
        let w2 = half_t + half_t2 * 4.0 - half_three_t3;
        let w3 = half_t3 - half_t2;

        w0 * self.padded[start]
            + w1 * self.padded[start + 1]
            + w2 * self.padded[start + 2]
            + w3 * self.padded[start + 3]
    }

    fn render_wave_blend(&self, frame: &mut WaveFrame, keyframe: &FileKeyframe) {
        let window_ratio = self.window_size / WAVEFORM_SIZE as f64;
        let waveform_middle = WAVEFORM_SIZE / 2;
        let start_index = (keyframe.start_position / window_ratio
            + self.window_size / 2.0
            + waveform_middle as f64) as usize
            % WAVEFORM_SIZE;

        for i in 0..WAVEFORM_SIZE {
            let t = i as f64 / WAVEFORM_SIZE as f64;
            let position = keyframe.start_position + t * self.window_size;
            let write_index = (start_index + i) % WAVEFORM_SIZE;
            frame.time_domain[write_index] = self.sample_at(position);
        }

        let fade_samples = (keyframe.window_fade * WAVEFORM_SIZE as f64) as usize;
        if fade_samples > 1 {
            let fade_size = fade_samples as f64 * window_ratio;
            for i in 0..fade_samples {
                let t = i as f64 / (fade_samples as f64 - 1.0);
                let fade = 0.5 + 0.5 * (std::f64::consts::PI * t).cos();

                let write_index = (start_index + i) % WAVEFORM_SIZE;
                let position = keyframe.start_position + self.window_size + t * fade_size;
                let existing = frame.time_domain[write_index] as f64;
                let fade_value = self.sample_at(position) as f64;
                frame.time_domain[write_index] =
                    (existing + (fade_value - existing) * fade) as f32;
            }
        }
        frame.to_frequency_domain();
    }

    fn render_no_interpolate(&self, frame: &mut WaveFrame, keyframe: &FileKeyframe) {
        let cycle = (keyframe.start_position / self.window_size) as i64 as f64;
        let start_index = cycle * self.window_size;
        for i in 0..WAVEFORM_SIZE {
            let t = i as f64 / WAVEFORM_SIZE as f64;
            frame.time_domain[i] = self.sample_at(start_index + t * self.window_size);
        }
        frame.to_frequency_domain();
    }

    fn render_time_interpolate(&self, frame: &mut WaveFrame, keyframe: &FileKeyframe) {
        let cycles_in = keyframe.start_position / self.window_size;
        let from_cycle = cycles_in as i64;
        let to_cycle = from_cycle + 1;
        let transition = (cycles_in - from_cycle as f64) as f32;

        let start_from = from_cycle as f64 * self.window_size;
        let start_to = to_cycle as f64 * self.window_size;
        for i in 0..WAVEFORM_SIZE {
            let t = i as f64 / WAVEFORM_SIZE as f64;
            let from_sample = self.sample_at(start_from + t * self.window_size);
            let to_sample = self.sample_at(start_to + t * self.window_size);
            frame.time_domain[i] = linear_tween(from_sample, to_sample, transition);
        }
        frame.to_frequency_domain();
    }

    fn render_freq_interpolate(&self, frame: &mut WaveFrame, keyframe: &FileKeyframe) {
        let cycles_in = keyframe.start_position / self.window_size;
        let from_cycle = cycles_in as i64;
        let to_cycle = from_cycle + 1;
        let transition = (cycles_in - from_cycle as f64) as f32;

        let start_from = from_cycle as f64 * self.window_size;
        let start_to = to_cycle as f64 * self.window_size;

        let mut from_frame = WaveFrame::new();
        let mut to_frame = WaveFrame::new();
        for i in 0..WAVEFORM_SIZE {
            let t = i as f64 / WAVEFORM_SIZE as f64;
            from_frame.time_domain[i] = self.sample_at(start_from + t * self.window_size);
            to_frame.time_domain[i] = self.sample_at(start_to + t * self.window_size);
        }
        from_frame.to_frequency_domain();
        to_frame.to_frequency_domain();
        linear_frequency_interpolate(frame, &from_frame, &to_frame, transition);
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        if self.size == 0 {
            frame.clear();
            return;
        }

        let Some(keyframe) =
            super::components::interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        match self.fade_style {
            FadeStyle::WaveBlend => self.render_wave_blend(frame, &keyframe),
            FadeStyle::NoInterpolate => self.render_no_interpolate(frame, &keyframe),
            FadeStyle::TimeInterpolate => self.render_time_interpolate(frame, &keyframe),
            FadeStyle::FreqInterpolate => self.render_freq_interpolate(frame, &keyframe),
        }

        if self.phase_style != FilePhaseStyle::None {
            for i in 0..NUM_REAL_COMPLEX {
                let amplitude = frame.frequency_domain[i].norm();
                frame.frequency_domain[i] =
                    Complex::from_polar(amplitude, self.overridden_phase[i]);
            }
        }
        frame.to_time_domain();

        frame.frequency_ratio = (self.window_size / WAVEFORM_SIZE as f64) as f32;
        frame.sample_rate = self.sample_rate;
        if self.normalize_mult {
            frame.normalize(self.normalize_gain);
        }
        frame.to_frequency_domain();
    }
}
