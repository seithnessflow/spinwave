//! Shared wavetable-component machinery plus the modifier components.
//!
//! Rework of Vital's `wavetable_component.{h,cpp}`, `wavetable_keyframe`
//! and the modifier implementations (`phase_modifier`,
//! `wave_window_modifier`, `frequency_filter_modifier`,
//! `slew_limit_modifier`, `wave_fold_modifier`, `wave_warp_modifier`).
//! Every component holds a sorted keyframe list over positions
//! `0..=NUM_OSCILLATOR_WAVE_FRAMES - 1` and renders the state
//! interpolated at a fractional position into a [`WaveFrame`].

use realfft::num_complex::Complex;
use serde_json::Value;

use super::wave_frame::{WaveFrame, NUM_REAL_COMPLEX, WAVEFORM_SIZE};
use super::wavetable::NUM_OSCILLATOR_WAVE_FRAMES;

/// Highest valid keyframe position (256 in the reference).
pub(crate) const LAST_FRAME_POSITION: i32 = NUM_OSCILLATOR_WAVE_FRAMES as i32 - 1;

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

pub(crate) fn json_f32(data: &Value, key: &str) -> Option<f32> {
    data.get(key)?.as_f64().map(|v| v as f32)
}

pub(crate) fn json_f64(data: &Value, key: &str) -> Option<f64> {
    data.get(key)?.as_f64()
}

pub(crate) fn json_u64(data: &Value, key: &str) -> Option<u64> {
    data.get(key)?.as_u64()
}

pub(crate) fn json_bool(data: &Value, key: &str) -> Option<bool> {
    data.get(key)?.as_bool()
}

pub(crate) fn json_str<'a>(data: &'a Value, key: &str) -> Option<&'a str> {
    data.get(key)?.as_str()
}

// ---------------------------------------------------------------------------
// Interpolation primitives (wavetable_keyframe.cpp)
// ---------------------------------------------------------------------------

/// Keyframe interpolation styles (`WavetableComponent::InterpolationStyle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InterpolationStyle {
    None,
    Linear,
    Cubic,
}

impl InterpolationStyle {
    pub(crate) fn from_json(data: &Value) -> InterpolationStyle {
        match json_u64(data, "interpolation_style") {
            Some(0) => InterpolationStyle::None,
            Some(2) => InterpolationStyle::Cubic,
            _ => InterpolationStyle::Linear,
        }
    }
}

#[inline]
pub(crate) fn linear_tween(from: f32, to: f32, t: f32) -> f32 {
    from + (to - from) * t
}

/// The reference's `WavetableKeyframe::cubicTween`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn cubic_tween(
    prev: f32,
    from: f32,
    to: f32,
    next: f32,
    range_prev: f32,
    range: f32,
    range_next: f32,
    t: f32,
) -> f32 {
    let mut slope_from = 0.0;
    let mut slope_to = 0.0;
    if range_prev > 0.0 {
        slope_from = (to - prev) / (1.0 + range_prev / range);
    }
    if range_next > 0.0 {
        slope_to = (next - from) / (1.0 + range_next / range);
    }
    let delta = to - from;
    let movement = linear_tween(from, to, t);
    let smooth = t * (1.0 - t) * ((1.0 - t) * (slope_from - delta) + t * (delta - slope_to));
    movement + smooth
}

/// The reference's `futils::powerScale` (no absolute value).
#[inline]
pub(crate) fn power_scale(value: f32, power: f32) -> f32 {
    const MIN_POWER: f32 = 0.01;
    if power.abs() < MIN_POWER {
        return value;
    }
    let numerator = (power * value).exp() - 1.0;
    let denominator = power.exp() - 1.0;
    numerator / denominator
}

/// The double-precision symmetric power scale used by the frequency
/// filter's comb shape and by Wave Warp (`highResPowerScale`).
#[inline]
pub(crate) fn power_scale_symmetric(value: f64, power: f64) -> f64 {
    const MIN_POWER: f64 = 0.01;
    if power.abs() < MIN_POWER {
        return value;
    }
    let abs_value = value.abs();
    let numerator = (power * abs_value).exp() - 1.0;
    let denominator = power.exp() - 1.0;
    if value >= 0.0 {
        numerator / denominator
    } else {
        -numerator / denominator
    }
}

/// Where a render position falls inside a keyframe list
/// (`WavetableComponent::interpolate`).
pub(crate) enum Segment<'a, K> {
    /// Before the first keyframe, past the last one, or style `None`.
    Copy(&'a K),
    Linear {
        from: &'a K,
        to: &'a K,
        t: f32,
    },
    Cubic {
        prev: &'a K,
        from: &'a K,
        to: &'a K,
        next: &'a K,
        range_prev: f32,
        range: f32,
        range_next: f32,
        t: f32,
    },
}

/// Locates the interpolation segment for `position`, mirroring the index
/// arithmetic of `WavetableComponent::interpolate` (including its
/// truncation of the position when searching keyframes).
pub(crate) fn locate<'a, K>(
    positions: &[i32],
    frames: &'a [K],
    style: InterpolationStyle,
    position: f32,
) -> Option<Segment<'a, K>> {
    let num_frames = frames.len();
    if num_frames == 0 {
        return None;
    }

    let int_position = position as i32;
    let mut index: isize = -1;
    for &keyframe_position in positions {
        if int_position < keyframe_position {
            break;
        }
        index += 1;
    }

    let clamped = index.clamp(0, num_frames as isize - 1) as usize;
    if index < 0 || index >= num_frames as isize - 1 || style == InterpolationStyle::None {
        return Some(Segment::Copy(&frames[clamped]));
    }

    let index = index as usize;
    let from_position = positions[index];
    let to_position = positions[index + 1];
    let t = (position - from_position as f32) / (to_position - from_position) as f32;

    match style {
        InterpolationStyle::Linear => Some(Segment::Linear {
            from: &frames[index],
            to: &frames[index + 1],
            t,
        }),
        InterpolationStyle::Cubic => {
            let next_index = if index + 2 >= num_frames { index } else { index + 2 };
            let prev_index = if index == 0 { index + 1 } else { index - 1 };
            Some(Segment::Cubic {
                prev: &frames[prev_index],
                from: &frames[index],
                to: &frames[index + 1],
                next: &frames[next_index],
                range_prev: (from_position - positions[prev_index]) as f32,
                range: (to_position - from_position) as f32,
                range_next: (positions[next_index] - to_position) as f32,
                t,
            })
        }
        InterpolationStyle::None => unreachable!(),
    }
}

/// Keyframes whose state interpolates field-by-field. The reference's
/// scalar keyframes only implement linear interpolation (cubic style is a
/// no-op for them), so cubic falls back to linear here.
pub(crate) trait ScalarKeyframe: Clone {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self;
}

pub(crate) fn interpolate_scalar<K: ScalarKeyframe>(
    positions: &[i32],
    frames: &[K],
    style: InterpolationStyle,
    position: f32,
) -> Option<K> {
    match locate(positions, frames, style, position)? {
        Segment::Copy(keyframe) => Some(keyframe.clone()),
        Segment::Linear { from, to, t } | Segment::Cubic { from, to, t, .. } => {
            Some(K::lerp(from, to, t))
        }
    }
}

/// Parses the `keyframes` array of a component, returning positions
/// (sorted, clamped to the valid frame range) and per-keyframe payloads.
pub(crate) fn parse_keyframes<K>(
    data: &Value,
    mut parse: impl FnMut(&Value) -> Option<K>,
) -> Option<(Vec<i32>, Vec<K>)> {
    let keyframes = data.get("keyframes")?.as_array()?;
    let mut entries: Vec<(i32, K)> = Vec::with_capacity(keyframes.len());
    for keyframe in keyframes {
        let position = keyframe.get("position")?.as_i64()? as i32;
        let position = position.clamp(0, LAST_FRAME_POSITION);
        entries.push((position, parse(keyframe)?));
    }
    entries.sort_by_key(|(position, _)| *position);
    let mut positions = Vec::with_capacity(entries.len());
    let mut frames = Vec::with_capacity(entries.len());
    for (position, frame) in entries {
        positions.push(position);
        frames.push(frame);
    }
    Some((positions, frames))
}

// ---------------------------------------------------------------------------
// Phase Shift (phase_modifier.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) enum PhaseStyle {
    Normal,
    EvenOdd,
    Harmonic,
    HarmonicEvenOdd,
    Clear,
}

#[derive(Clone)]
pub(crate) struct PhaseKeyframe {
    phase: f32,
    mix: f32,
}

impl ScalarKeyframe for PhaseKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        PhaseKeyframe {
            phase: linear_tween(from.phase, to.phase, t),
            mix: linear_tween(from.mix, to.mix, t),
        }
    }
}

pub(crate) struct PhaseShift {
    positions: Vec<i32>,
    frames: Vec<PhaseKeyframe>,
    style: InterpolationStyle,
    phase_style: PhaseStyle,
}

impl PhaseShift {
    pub(crate) fn from_json(data: &Value) -> Option<PhaseShift> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            Some(PhaseKeyframe {
                phase: json_f32(keyframe, "phase")?,
                mix: json_f32(keyframe, "mix")?,
            })
        })?;
        let phase_style = match json_u64(data, "style").unwrap_or(0) {
            1 => PhaseStyle::EvenOdd,
            2 => PhaseStyle::Harmonic,
            3 => PhaseStyle::HarmonicEvenOdd,
            4 => PhaseStyle::Clear,
            _ => PhaseStyle::Normal,
        };
        Some(PhaseShift {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
            phase_style,
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) = interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        let mix = keyframe.mix;
        let mix_bin = |bin: Complex<f32>, shift: Complex<f32>| -> Complex<f32> {
            let result = bin * shift;
            result * mix + bin * (1.0 - mix)
        };

        let phase_shift = Complex::from_polar(1.0, -keyframe.phase);
        match self.phase_style {
            PhaseStyle::Harmonic => {
                for bin in frame.frequency_domain.iter_mut() {
                    *bin = mix_bin(*bin, phase_shift);
                }
            }
            PhaseStyle::HarmonicEvenOdd => {
                let odd_shift = Complex::new(1.0, 0.0) / phase_shift;
                for (i, bin) in frame.frequency_domain.iter_mut().enumerate() {
                    let shift = if i % 2 == 0 { phase_shift } else { odd_shift };
                    *bin = mix_bin(*bin, shift);
                }
            }
            PhaseStyle::Normal => {
                let mut current = Complex::new(1.0, 0.0);
                for bin in frame.frequency_domain.iter_mut() {
                    *bin = mix_bin(*bin, current);
                    current *= phase_shift;
                }
            }
            PhaseStyle::EvenOdd => {
                let mut current = Complex::new(1.0, 0.0);
                let mut i = 0;
                while i < frame.frequency_domain.len() {
                    frame.frequency_domain[i] = mix_bin(frame.frequency_domain[i], current);
                    if i + 1 < frame.frequency_domain.len() {
                        let odd_shift = Complex::new(1.0, 0.0) / (current * phase_shift);
                        frame.frequency_domain[i + 1] =
                            mix_bin(frame.frequency_domain[i + 1], odd_shift);
                    }
                    current *= phase_shift * phase_shift;
                    i += 2;
                }
            }
            PhaseStyle::Clear => {
                for bin in frame.frequency_domain.iter_mut() {
                    *bin = Complex::new(bin.norm(), 0.0);
                }
            }
        }
        frame.to_time_domain();
    }
}

// ---------------------------------------------------------------------------
// Wave Window (wave_window_modifier.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) enum WindowShape {
    Cos,
    HalfSin,
    Linear,
    Square,
    Wiggle,
}

fn apply_window(shape: WindowShape, t: f32) -> f32 {
    match shape {
        WindowShape::Cos => 0.5 - 0.5 * (std::f32::consts::PI * t).cos(),
        WindowShape::HalfSin => (std::f32::consts::PI * t / 2.0).sin(),
        WindowShape::Square => {
            if t < 1.0 {
                0.0
            } else {
                1.0
            }
        }
        WindowShape::Wiggle => t * (std::f32::consts::PI * (t * 1.5 + 0.5)).cos(),
        WindowShape::Linear => t,
    }
}

#[derive(Clone)]
pub(crate) struct WindowKeyframe {
    left: f32,
    right: f32,
}

impl ScalarKeyframe for WindowKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        WindowKeyframe {
            left: linear_tween(from.left, to.left, t),
            right: linear_tween(from.right, to.right, t),
        }
    }
}

pub(crate) struct WaveWindow {
    positions: Vec<i32>,
    frames: Vec<WindowKeyframe>,
    style: InterpolationStyle,
    shape: WindowShape,
}

impl WaveWindow {
    pub(crate) fn from_json(data: &Value) -> Option<WaveWindow> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            Some(WindowKeyframe {
                left: json_f32(keyframe, "left_position")?,
                right: json_f32(keyframe, "right_position")?,
            })
        })?;
        let shape = match json_u64(data, "window_shape").unwrap_or(0) {
            1 => WindowShape::HalfSin,
            2 => WindowShape::Linear,
            3 => WindowShape::Square,
            4 => WindowShape::Wiggle,
            _ => WindowShape::Cos,
        };
        Some(WaveWindow {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
            shape,
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) = interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        for i in 0..WAVEFORM_SIZE {
            let t = i as f32 / (WAVEFORM_SIZE as f32 - 1.0);
            if t >= keyframe.left {
                break;
            }
            frame.time_domain[i] *= apply_window(self.shape, t / keyframe.left);
        }

        for i in (0..WAVEFORM_SIZE).rev() {
            let t = i as f32 / (WAVEFORM_SIZE as f32 - 1.0);
            if t <= keyframe.right {
                break;
            }
            frame.time_domain[i] *= apply_window(self.shape, (1.0 - t) / (1.0 - keyframe.right));
        }

        frame.to_frequency_domain();
    }
}

// ---------------------------------------------------------------------------
// Frequency Filter (frequency_filter_modifier.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum FilterStyle {
    LowPass,
    BandPass,
    HighPass,
    Comb,
}

#[derive(Clone)]
pub(crate) struct FilterKeyframe {
    cutoff: f32,
    shape: f32,
}

impl ScalarKeyframe for FilterKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        FilterKeyframe {
            cutoff: linear_tween(from.cutoff, to.cutoff, t),
            shape: linear_tween(from.shape, to.shape, t),
        }
    }
}

pub(crate) struct FrequencyFilter {
    positions: Vec<i32>,
    frames: Vec<FilterKeyframe>,
    style: InterpolationStyle,
    filter_style: FilterStyle,
    normalize: bool,
}

impl FrequencyFilter {
    pub(crate) fn from_json(data: &Value) -> Option<FrequencyFilter> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            Some(FilterKeyframe {
                cutoff: json_f32(keyframe, "cutoff")?,
                shape: json_f32(keyframe, "shape")?,
            })
        })?;
        let filter_style = match json_u64(data, "style").unwrap_or(0) {
            1 => FilterStyle::BandPass,
            2 => FilterStyle::HighPass,
            3 => FilterStyle::Comb,
            _ => FilterStyle::LowPass,
        };
        Some(FrequencyFilter {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
            filter_style,
            normalize: json_bool(data, "normalize").unwrap_or(true),
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    fn multiplier(&self, keyframe: &FilterKeyframe, index: f32) -> f32 {
        const MIN_POWER: f32 = -9.0;
        const MAX_POWER: f32 = 9.0;
        const MAX_SLOPE_REACH: f32 = 128.0;

        let cutoff_index = 2.0f32.powf(keyframe.cutoff);
        let cutoff_delta = index - cutoff_index;

        let slope = 1.0 / linear_tween(1.0, MAX_SLOPE_REACH, keyframe.shape * keyframe.shape);
        let power = linear_tween(MIN_POWER, MAX_POWER, keyframe.shape);

        match self.filter_style {
            FilterStyle::LowPass => (1.0 - slope * cutoff_delta).clamp(0.0, 1.0),
            FilterStyle::BandPass => (1.0 - (slope * cutoff_delta).abs()).clamp(0.0, 1.0),
            FilterStyle::HighPass => (1.0 + slope * cutoff_delta).clamp(0.0, 1.0),
            FilterStyle::Comb => {
                let t = index / (cutoff_index * 2.0);
                let range = t - t.floor();
                2.0 * power_scale_symmetric(
                    (1.0 - (2.0 * range - 1.0).abs()) as f64,
                    power as f64,
                ) as f32
            }
        }
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) = interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        for i in 0..NUM_REAL_COMPLEX {
            let multiplier = self.multiplier(&keyframe, i as f32);
            frame.frequency_domain[i] *= multiplier;
        }
        frame.to_time_domain();

        if self.normalize {
            frame.normalize(true);
            frame.to_frequency_domain();
        }
    }
}

// ---------------------------------------------------------------------------
// Slew Limiter (slew_limit_modifier.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct SlewKeyframe {
    up_run_rise: f32,
    down_run_rise: f32,
}

impl ScalarKeyframe for SlewKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        SlewKeyframe {
            up_run_rise: linear_tween(from.up_run_rise, to.up_run_rise, t),
            down_run_rise: linear_tween(from.down_run_rise, to.down_run_rise, t),
        }
    }
}

pub(crate) struct SlewLimiter {
    positions: Vec<i32>,
    frames: Vec<SlewKeyframe>,
    style: InterpolationStyle,
}

impl SlewLimiter {
    pub(crate) fn from_json(data: &Value) -> Option<SlewLimiter> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            Some(SlewKeyframe {
                up_run_rise: json_f32(keyframe, "up_run_rise")?,
                down_run_rise: json_f32(keyframe, "down_run_rise")?,
            })
        })?;
        Some(SlewLimiter {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) = interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        let min_slew_limit = 1.0 / WAVEFORM_SIZE as f32;
        let max_up_delta =
            (2.0 / WAVEFORM_SIZE as f32) / keyframe.up_run_rise.max(min_slew_limit);
        let max_down_delta =
            (2.0 / WAVEFORM_SIZE as f32) / keyframe.down_run_rise.max(min_slew_limit);

        let mut current_value = frame.time_domain[0];
        for i in 1..2 * WAVEFORM_SIZE {
            let index = i % WAVEFORM_SIZE;
            let target_value = frame.time_domain[index];
            let delta = target_value - current_value;
            if delta > 0.0 {
                current_value += delta.min(max_up_delta);
            } else {
                current_value -= (-delta).min(max_down_delta);
            }
            frame.time_domain[index] = current_value;
        }
        frame.to_frequency_domain();
    }
}

// ---------------------------------------------------------------------------
// Wave Folder (wave_fold_modifier.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct FoldKeyframe {
    fold_boost: f32,
}

impl ScalarKeyframe for FoldKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        FoldKeyframe {
            fold_boost: linear_tween(from.fold_boost, to.fold_boost, t),
        }
    }
}

pub(crate) struct WaveFolder {
    positions: Vec<i32>,
    frames: Vec<FoldKeyframe>,
    style: InterpolationStyle,
}

impl WaveFolder {
    pub(crate) fn from_json(data: &Value) -> Option<WaveFolder> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            Some(FoldKeyframe {
                fold_boost: json_f32(keyframe, "fold_boost")?,
            })
        })?;
        Some(WaveFolder {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) = interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        let max_value = frame.max_zero_offset().max(1.0);
        for sample in &mut frame.time_domain {
            let value = (*sample / max_value).clamp(-1.0, 1.0);
            let adjusted = max_value * keyframe.fold_boost * value.asin();
            *sample = adjusted.sin();
        }
        frame.to_frequency_domain();
    }
}

// ---------------------------------------------------------------------------
// Wave Warp (wave_warp_modifier.cpp)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct WarpKeyframe {
    horizontal_power: f32,
    vertical_power: f32,
}

impl ScalarKeyframe for WarpKeyframe {
    fn lerp(from: &Self, to: &Self, t: f32) -> Self {
        WarpKeyframe {
            horizontal_power: linear_tween(from.horizontal_power, to.horizontal_power, t),
            vertical_power: linear_tween(from.vertical_power, to.vertical_power, t),
        }
    }
}

pub(crate) struct WaveWarp {
    positions: Vec<i32>,
    frames: Vec<WarpKeyframe>,
    style: InterpolationStyle,
    horizontal_asymmetric: bool,
    vertical_asymmetric: bool,
}

impl WaveWarp {
    pub(crate) fn from_json(data: &Value) -> Option<WaveWarp> {
        let (positions, frames) = parse_keyframes(data, |keyframe| {
            Some(WarpKeyframe {
                horizontal_power: json_f32(keyframe, "horizontal_power")?,
                vertical_power: json_f32(keyframe, "vertical_power")?,
            })
        })?;
        Some(WaveWarp {
            positions,
            frames,
            style: InterpolationStyle::from_json(data),
            horizontal_asymmetric: json_bool(data, "horizontal_asymmetric").unwrap_or(false),
            vertical_asymmetric: json_bool(data, "vertical_asymmetric").unwrap_or(false),
        })
    }

    pub(crate) fn last_position(&self) -> i32 {
        self.positions.last().copied().unwrap_or(0)
    }

    pub(crate) fn render(&self, frame: &mut WaveFrame, position: f32) {
        let Some(keyframe) = interpolate_scalar(&self.positions, &self.frames, self.style, position)
        else {
            return;
        };

        // The reference stashes the original cycle in the (larger)
        // frequency-domain scratch; a plain copy is equivalent.
        let original = frame.time_domain.clone();

        for i in 0..WAVEFORM_SIZE {
            let horizontal = i as f32 / (WAVEFORM_SIZE as f32 - 1.0);
            let warped_horizontal = if self.horizontal_asymmetric {
                power_scale_symmetric(horizontal as f64, keyframe.horizontal_power as f64) as f32
            } else {
                0.5 * power_scale_symmetric(
                    (2.0 * horizontal - 1.0) as f64,
                    keyframe.horizontal_power as f64,
                ) as f32
                    + 0.5
            };

            let float_index = (WAVEFORM_SIZE as f32 - 1.0) * warped_horizontal;
            let index = (float_index as i32).clamp(0, WAVEFORM_SIZE as i32 - 2) as usize;

            let vertical_from = original[index];
            let vertical_to = original[index + 1];
            let vertical = linear_tween(vertical_from, vertical_to, float_index - index as f32)
                .clamp(-1.0, 1.0);
            frame.time_domain[i] = if self.vertical_asymmetric {
                2.0 * power_scale_symmetric(
                    (0.5 * vertical + 0.5) as f64,
                    keyframe.vertical_power as f64,
                ) as f32
                    - 1.0
            } else {
                power_scale_symmetric(vertical as f64, keyframe.vertical_power as f64) as f32
            };
        }
        frame.to_frequency_domain();
    }
}
