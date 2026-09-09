//! Renders a `.vital` wavetable JSON payload (preset
//! `settings.wavetables[i]`) into a [`Wavetable`].
//!
//! Rework of Vital's `wavetable_creator.{h,cpp}` and
//! `wavetable_group.{h,cpp}`: parse the component groups, then render each
//! frame position by running every group's component chain over a compute
//! frame, averaging the groups, and loading the result into the table.

use serde_json::Value;

use super::components::{
    json_bool, json_str, FrequencyFilter, PhaseShift, SlewLimiter, WaveFolder, WaveWarp,
    WaveWindow, LAST_FRAME_POSITION,
};
use super::sources::{FileSource, LineSource, LoadContext, ShepardSource, WaveSource};
use super::wave_frame::WaveFrame;
use super::wavetable::{Wavetable, NUM_OSCILLATOR_WAVE_FRAMES};

/// Component type strings before Vital 0.3.3 were indices into this order
/// (`WavetableCreator::updateJson`).
const OLD_TYPE_ORDER: [&str; 9] = [
    "Wave Source",
    "Line Source",
    "Audio File Source",
    "Phase Shift",
    "Wave Window",
    "Frequency Filter",
    "Slew Limiter",
    "Wave Folder",
    "Wave Warp",
];

/// One entry of a group's component chain.
pub(crate) enum Component {
    Wave(WaveSource),
    Shepard(ShepardSource),
    Line(LineSource),
    File(FileSource),
    Phase(PhaseShift),
    Window(WaveWindow),
    Filter(FrequencyFilter),
    Slew(SlewLimiter),
    Fold(WaveFolder),
    Warp(WaveWarp),
}

impl Component {
    fn render(&self, frame: &mut WaveFrame, position: f32) {
        match self {
            Component::Wave(component) => component.render(frame, position),
            Component::Shepard(component) => component.render(frame, position),
            Component::Line(component) => component.render(frame, position),
            Component::File(component) => component.render(frame, position),
            Component::Phase(component) => component.render(frame, position),
            Component::Window(component) => component.render(frame, position),
            Component::Filter(component) => component.render(frame, position),
            Component::Slew(component) => component.render(frame, position),
            Component::Fold(component) => component.render(frame, position),
            Component::Warp(component) => component.render(frame, position),
        }
    }

    fn last_position(&self) -> i32 {
        match self {
            Component::Wave(component) => component.last_position(),
            Component::Shepard(component) => component.last_position(),
            Component::Line(component) => component.last_position(),
            Component::File(component) => component.last_position(),
            Component::Phase(component) => component.last_position(),
            Component::Window(component) => component.last_position(),
            Component::Filter(component) => component.last_position(),
            Component::Slew(component) => component.last_position(),
            Component::Fold(component) => component.last_position(),
            Component::Warp(component) => component.last_position(),
        }
    }

    fn is_shepard(&self) -> bool {
        matches!(self, Component::Shepard(_))
    }
}

/// A group of components rendered in sequence over one compute frame.
pub(crate) struct Group {
    pub(crate) components: Vec<Component>,
}

impl Group {
    fn render(&self, frame: &mut WaveFrame, position: f32) {
        frame.index = position as usize;
        for component in &self.components {
            component.render(frame, position);
        }
    }

    fn last_position(&self) -> i32 {
        self.components
            .iter()
            .map(Component::last_position)
            .max()
            .unwrap_or(0)
    }

    fn is_shepard(&self) -> bool {
        self.components.iter().all(Component::is_shepard)
    }
}

/// The full creator state: groups plus post-processing flags.
pub(crate) struct Creator {
    pub(crate) name: String,
    pub(crate) author: String,
    pub(crate) groups: Vec<Group>,
    pub(crate) remove_all_dc: bool,
    pub(crate) full_normalize: bool,
}

impl Creator {
    /// Renders all frame positions into a new wavetable
    /// (`WavetableCreator::render`).
    pub(crate) fn render(&self) -> Wavetable {
        let mut wavetable = Wavetable::new(NUM_OSCILLATOR_WAVE_FRAMES);
        wavetable.name = self.name.clone();
        wavetable.author = self.author.clone();

        let last_position = self
            .groups
            .iter()
            .map(Group::last_position)
            .max()
            .unwrap_or(0)
            .clamp(0, LAST_FRAME_POSITION) as usize;
        wavetable.set_num_frames(last_position + 1);

        let shepard = !self.groups.is_empty() && self.groups.iter().all(Group::is_shepard);
        wavetable.set_shepard_table(shepard);

        let mut compute = WaveFrame::new();
        let mut combine = WaveFrame::new();
        let mut max_span = 0.0f32;

        for position in 0..=last_position {
            combine.clear();
            combine.index = position;
            compute.index = position;

            for group in &self.groups {
                group.render(&mut compute, position as f32);
                combine.add_from(&compute);
            }

            if self.groups.len() > 1 {
                combine.multiply(1.0 / self.groups.len() as f32);
            }
            if self.remove_all_dc {
                combine.remove_dc();
            }

            let mut max_value = 0.0f32;
            let mut min_value = 0.0f32;
            for &value in &combine.time_domain {
                max_value = max_value.max(value);
                min_value = min_value.min(value);
            }
            max_span = max_span.max(max_value - min_value);

            wavetable.load_wave_frame_at(&combine, position);
        }

        wavetable.set_frequency_ratio(compute.frequency_ratio);
        wavetable.set_sample_rate(compute.sample_rate);
        wavetable.post_process(if self.full_normalize { max_span } else { 0.0 });
        wavetable
    }
}

// ---------------------------------------------------------------------------
// Version handling (`WavetableCreator::updateJson`)
// ---------------------------------------------------------------------------

fn parse_version(version: &str) -> (u32, u32, u32) {
    let mut parts = version.split('.').map(|part| part.parse::<u32>().unwrap_or(0));
    (
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
        parts.next().unwrap_or(0),
    )
}

fn context_for_version(version: (u32, u32, u32)) -> LoadContext {
    LoadContext {
        wave_data_pcm: ((0, 3, 7)..(0, 3, 9)).contains(&version),
        audio_file_float: version < (0, 3, 7),
        line_old_format: version < (0, 7, 7),
    }
}

// ---------------------------------------------------------------------------
// JSON parsing
// ---------------------------------------------------------------------------

fn component_type_name(data: &Value) -> Option<String> {
    let type_value = data.get("type")?;
    if let Some(name) = type_value.as_str() {
        return Some(name.to_string());
    }
    // Pre-0.3.3 presets store an index instead of a name.
    let index = type_value.as_u64()? as usize;
    OLD_TYPE_ORDER.get(index).map(|name| (*name).to_string())
}

fn parse_component(
    data: &Value,
    context: &LoadContext,
    warnings: &mut Vec<String>,
) -> Option<Component> {
    let Some(type_name) = component_type_name(data) else {
        warnings.push("wavetable component with missing or invalid type".to_string());
        return None;
    };

    let component = match type_name.as_str() {
        "Wave Source" => WaveSource::from_json(data, context).map(Component::Wave),
        "Shepard Tone Source" => ShepardSource::from_json(data, context).map(Component::Shepard),
        "Line Source" => LineSource::from_json(data, context).map(Component::Line),
        "Audio File Source" => FileSource::from_json(data, context).map(Component::File),
        "Phase Shift" => PhaseShift::from_json(data).map(Component::Phase),
        "Wave Window" => WaveWindow::from_json(data).map(Component::Window),
        "Frequency Filter" => FrequencyFilter::from_json(data).map(Component::Filter),
        "Slew Limiter" => SlewLimiter::from_json(data).map(Component::Slew),
        "Wave Folder" => WaveFolder::from_json(data).map(Component::Fold),
        "Wave Warp" => WaveWarp::from_json(data).map(Component::Warp),
        other => {
            warnings.push(format!("unsupported wavetable component type \"{other}\""));
            return None;
        }
    };

    if component.is_none() {
        warnings.push(format!("failed to parse \"{type_name}\" component"));
    }
    component
}

/// A bare `LineGenerator` state (LFO-shape JSON) is a valid wavetable
/// source in Vital (`LineGenerator::isValidJson` in `jsonToState`).
fn is_line_generator_json(data: &Value) -> bool {
    data.get("num_points").is_some()
        && data.get("points").is_some_and(Value::is_array)
        && data.get("powers").is_some_and(Value::is_array)
}

fn creator_from_json(data: &Value, warnings: &mut Vec<String>) -> Option<Creator> {
    if is_line_generator_json(data) {
        let line_json = serde_json::json!({
            "num_points": 2,
            "keyframes": [{ "position": 0, "line": data }],
        });
        let context = LoadContext::default();
        let component = LineSource::from_json(&line_json, &context)?;
        return Some(Creator {
            name: json_str(data, "name").unwrap_or("").to_string(),
            author: String::new(),
            groups: vec![Group {
                components: vec![Component::Line(component)],
            }],
            remove_all_dc: true,
            full_normalize: true,
        });
    }

    let groups_data = data.get("groups")?.as_array()?;
    let version = parse_version(json_str(data, "version").unwrap_or("0.0.0"));
    let context = context_for_version(version);

    let remove_all_dc = if version < (0, 3, 8) {
        false
    } else {
        json_bool(data, "remove_all_dc").unwrap_or(true)
    };
    let full_normalize = if version < (0, 4, 7) {
        false
    } else {
        json_bool(data, "full_normalize").unwrap_or(false)
    };

    let mut groups = Vec::with_capacity(groups_data.len());
    for group_data in groups_data {
        let Some(components_data) = group_data.get("components").and_then(Value::as_array) else {
            warnings.push("wavetable group without components".to_string());
            continue;
        };
        let mut components = Vec::with_capacity(components_data.len());
        for component_data in components_data {
            if let Some(component) = parse_component(component_data, &context, warnings) {
                components.push(component);
            }
        }
        groups.push(Group { components });
    }

    Some(Creator {
        name: json_str(data, "name").unwrap_or("").to_string(),
        author: json_str(data, "author").unwrap_or("").to_string(),
        groups,
        remove_all_dc,
        full_normalize,
    })
}

/// Renders a `.vital` wavetable JSON payload into a wavetable.
///
/// Returns `None` only when the payload is not a wavetable at all (no
/// `groups` array and not a line-generator shape). Unsupported or broken
/// components are skipped so the rest of the table still renders; use
/// [`wavetable_from_json_with_warnings`] to see what was skipped.
pub fn wavetable_from_json(value: &Value) -> Option<Wavetable> {
    wavetable_from_json_with_warnings(value).map(|(wavetable, _)| wavetable)
}

/// Like [`wavetable_from_json`], also returning a warning per component
/// that could not be handled.
pub fn wavetable_from_json_with_warnings(value: &Value) -> Option<(Wavetable, Vec<String>)> {
    let mut warnings = Vec::new();
    let creator = creator_from_json(value, &mut warnings)?;
    Some((creator.render(), warnings))
}

#[cfg(test)]
mod tests {
    use super::super::codec::{base64_encode, f32_to_bytes};
    use super::super::wave_frame::{WaveFrame, WaveShape, WAVEFORM_SIZE};
    use super::*;
    use serde_json::json;

    fn encode_wave(samples: &[f32]) -> String {
        base64_encode(&f32_to_bytes(samples))
    }

    fn encode_pcm(samples: &[f32]) -> String {
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &sample in samples {
            let pcm = (sample * 32767.0).clamp(-32767.0, 32767.0) as i16;
            bytes.extend_from_slice(&pcm.to_le_bytes());
        }
        base64_encode(&bytes)
    }

    fn correlation(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    fn two_keyframe_wave_source_json() -> serde_json::Value {
        let sin = WaveFrame::predefined(WaveShape::Sin);
        let saw = WaveFrame::predefined(WaveShape::Saw);
        json!({
            "name": "Test Table",
            "author": "spinwave",
            "version": "1.5.5",
            "remove_all_dc": true,
            "full_normalize": true,
            "groups": [{
                "components": [{
                    "type": "Wave Source",
                    "interpolation": 1,
                    "interpolation_style": 1,
                    "keyframes": [
                        { "position": 0, "wave_data": encode_wave(&sin.time_domain) },
                        { "position": 256, "wave_data": encode_wave(&saw.time_domain) },
                    ],
                }],
            }],
        })
    }

    #[test]
    fn renders_two_keyframe_wave_source() {
        let (wavetable, warnings) =
            wavetable_from_json_with_warnings(&two_keyframe_wave_source_json()).unwrap();
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(wavetable.num_frames(), 257);
        assert_eq!(wavetable.name, "Test Table");
        assert_eq!(wavetable.author, "spinwave");

        let sin = WaveFrame::predefined(WaveShape::Sin);
        let saw = WaveFrame::predefined(WaveShape::Saw);
        let data = wavetable.data();
        let first_corr = correlation(data.wave_data(0), &sin.time_domain);
        let last_corr = correlation(data.wave_data(256), &saw.time_domain);
        assert!(first_corr > 0.95, "first frame vs sin: {first_corr}");
        assert!(last_corr > 0.95, "last frame vs saw: {last_corr}");

        // The interpolated middle differs from both anchors. A pure
        // correlation bound is too blunt here (bin 1 dominates the
        // energy either way), so check both correlation and waveform
        // distance.
        let mid = data.wave_data(128);
        let mid_sin = correlation(mid, &sin.time_domain);
        let mid_saw = correlation(mid, &saw.time_domain);
        assert!(mid_sin < 0.99, "midpoint too close to sin: {mid_sin}");
        assert!(mid_saw < 0.99, "midpoint too close to saw: {mid_saw}");
        let max_diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
        };
        assert!(max_diff(mid, data.wave_data(0)) > 0.05, "midpoint equals first frame");
        assert!(max_diff(mid, data.wave_data(256)) > 0.05, "midpoint equals last frame");

        for frame in [0, 64, 128, 192, 256] {
            assert!(data.wave_data(frame).iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn unsupported_component_is_skipped_with_warning() {
        let sin = WaveFrame::predefined(WaveShape::Sin);
        let data = json!({
            "name": "Odd",
            "version": "1.0.0",
            "groups": [{
                "components": [
                    {
                        "type": "Chaos Reactor",
                        "keyframes": [{ "position": 0 }],
                    },
                    {
                        "type": "Wave Source",
                        "interpolation": 1,
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "wave_data": encode_wave(&sin.time_domain) },
                        ],
                    },
                ],
            }],
        });
        let (wavetable, warnings) = wavetable_from_json_with_warnings(&data).unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("Chaos Reactor"));
        assert_eq!(wavetable.num_frames(), 1);
        let corr = correlation(
            wavetable.data().wave_data(0),
            &WaveFrame::predefined(WaveShape::Sin).time_domain,
        );
        assert!(corr > 0.95, "sin survived the broken component: {corr}");
    }

    #[test]
    fn only_unsupported_components_still_returns_a_table() {
        let data = json!({
            "name": "Empty",
            "version": "1.0.0",
            "groups": [{ "components": [{ "type": "Nope" }] }],
        });
        let (wavetable, warnings) = wavetable_from_json_with_warnings(&data).unwrap();
        assert!(!warnings.is_empty());
        assert_eq!(wavetable.num_frames(), 1);
        assert!(wavetable.data().wave_data(0).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn not_a_wavetable_returns_none() {
        assert!(wavetable_from_json(&json!({ "foo": 1 })).is_none());
        assert!(wavetable_from_json(&json!(42)).is_none());
    }

    #[test]
    fn pcm_wave_data_version_path() {
        let sin = WaveFrame::predefined(WaveShape::Sin);
        let data = json!({
            "name": "Old",
            "version": "0.3.8",
            "groups": [{
                "components": [{
                    "type": "Wave Source",
                    "interpolation": 1,
                    "interpolation_style": 0,
                    "keyframes": [
                        { "position": 0, "wave_data": encode_pcm(&sin.time_domain) },
                    ],
                }],
            }],
        });
        let wavetable = wavetable_from_json(&data).unwrap();
        let corr = correlation(wavetable.data().wave_data(0), &sin.time_domain);
        assert!(corr > 0.99, "PCM-decoded sin: {corr}");
    }

    #[test]
    fn old_integer_component_type_is_mapped() {
        let sin = WaveFrame::predefined(WaveShape::Sin);
        let data = json!({
            "name": "Ancient",
            "version": "0.3.0",
            "groups": [{
                "components": [{
                    "type": 0,
                    "interpolation": 1,
                    "interpolation_style": 0,
                    "keyframes": [
                        { "position": 0, "wave_data": encode_wave(&sin.time_domain) },
                    ],
                }],
            }],
        });
        let (wavetable, warnings) = wavetable_from_json_with_warnings(&data).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        let corr = correlation(wavetable.data().wave_data(0), &sin.time_domain);
        assert!(corr > 0.95);
    }

    #[test]
    fn line_source_renders() {
        let data = json!({
            "name": "Lines",
            "version": "1.5.5",
            "remove_all_dc": true,
            "full_normalize": true,
            "groups": [{
                "components": [{
                    "type": "Line Source",
                    "num_points": 3,
                    "interpolation_style": 1,
                    "keyframes": [{
                        "position": 0,
                        "pull_power": 0.0,
                        "line": {
                            "num_points": 3,
                            "points": [0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
                            "powers": [0.0, 0.0, 0.0],
                            "smooth": false,
                        },
                    }],
                }],
            }],
        });
        let wavetable = wavetable_from_json(&data).unwrap();
        let frame = wavetable.data().wave_data(0);
        assert!(frame.iter().all(|value| value.is_finite()));
        let (min, max) = frame
            .iter()
            .fold((f32::MAX, f32::MIN), |(min, max), &v| (min.min(v), max.max(v)));
        assert!(max - min > 0.5, "line rendered flat: {min}..{max}");
    }

    #[test]
    fn file_source_validates_window_and_old_float_audio() {
        // Old (pre-0.3.7) float audio: values beyond +/-1 go through the
        // PCM16 round trip like Vital's updateJson, so they clamp.
        let audio: Vec<f32> = (0..4096)
            .map(|i| 3.0 * (2.0 * std::f32::consts::PI * i as f32 / 512.0).sin())
            .collect();
        let round_trip = super::super::codec::pcm16_round_trip(&audio);
        assert!(round_trip.iter().all(|v| v.abs() <= 1.0));
        assert!((round_trip[128] - 1.0).abs() < 1e-6);
        assert!((round_trip[384] + 1.0).abs() < 1e-6);
        assert!((round_trip[10] - (audio[10] * 32767.0) as i16 as f32 / 32767.0).abs() < 1e-7);

        // window_size 0 and an absurd window_fade must neither NaN the
        // frames nor spin for ~1e18 iterations.
        let data = json!({
            "name": "Hostile File",
            "version": "0.3.6",
            "groups": [{
                "components": [{
                    "type": "Audio File Source",
                    "interpolation": 1,
                    "interpolation_style": 0,
                    "fade_style": 0,
                    "phase_style": 0,
                    "window_size": 0.0,
                    "audio_sample_rate": 44100,
                    "audio_file": encode_wave(&audio),
                    "keyframes": [
                        { "position": 0, "start_position": 0.0, "window_fade": 1.0e18 },
                        { "position": 256.0, "start_position": 1024.0, "window_fade": -5.0 },
                    ],
                }],
            }],
        });
        let (wavetable, warnings) = wavetable_from_json_with_warnings(&data).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        for frame in [0, 100, 256] {
            let wave = wavetable.data().wave_data(frame);
            assert!(wave.iter().all(|v| v.is_finite()), "frame {frame} is not finite");
        }
    }

    #[test]
    fn modifier_chain_does_not_panic() {
        let saw = WaveFrame::predefined(WaveShape::Saw);
        let data = json!({
            "name": "Chain",
            "version": "1.5.5",
            "remove_all_dc": true,
            "full_normalize": true,
            "groups": [{
                "components": [
                    {
                        "type": "Wave Source",
                        "interpolation": 1,
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "wave_data": encode_wave(&saw.time_domain) },
                            { "position": 256, "wave_data": encode_wave(&saw.time_domain) },
                        ],
                    },
                    {
                        "type": "Phase Shift",
                        "style": 0,
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "phase": 0.0, "mix": 1.0 },
                            { "position": 256, "phase": std::f32::consts::PI, "mix": 1.0 },
                        ],
                    },
                    {
                        "type": "Frequency Filter",
                        "style": 0,
                        "normalize": true,
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "cutoff": 9.0, "shape": 0.5 },
                            { "position": 256, "cutoff": 3.0, "shape": 0.5 },
                        ],
                    },
                    {
                        "type": "Wave Folder",
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "fold_boost": 1.0 },
                            { "position": 256, "fold_boost": 3.0 },
                        ],
                    },
                    {
                        "type": "Slew Limiter",
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "up_run_rise": 0.0, "down_run_rise": 0.0 },
                            { "position": 256, "up_run_rise": 0.5, "down_run_rise": 0.1 },
                        ],
                    },
                    {
                        "type": "Wave Window",
                        "window_shape": 0,
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "left_position": 0.25, "right_position": 0.75 },
                        ],
                    },
                    {
                        "type": "Wave Warp",
                        "horizontal_asymmetric": false,
                        "vertical_asymmetric": false,
                        "interpolation_style": 1,
                        "keyframes": [
                            { "position": 0, "horizontal_power": 0.0, "vertical_power": 0.0 },
                            { "position": 256, "horizontal_power": 2.0, "vertical_power": -1.0 },
                        ],
                    },
                ],
            }],
        });
        let (wavetable, warnings) = wavetable_from_json_with_warnings(&data).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(wavetable.num_frames(), 257);
        for frame in 0..wavetable.num_frames() {
            assert!(
                wavetable.data().wave_data(frame).iter().all(|v| v.is_finite()),
                "frame {frame} has non-finite samples"
            );
        }
    }

    #[test]
    fn bare_line_generator_json_is_accepted() {
        let data = json!({
            "name": "Saw Line",
            "num_points": 2,
            "points": [0.0, 1.0, 1.0, 0.0],
            "powers": [0.0, 0.0],
            "smooth": false,
        });
        let wavetable = wavetable_from_json(&data).unwrap();
        assert_eq!(wavetable.num_frames(), 1);
        let frame = wavetable.data().wave_data(0);
        // A 0..1 descending line becomes a rising ramp after the
        // "1 - y" flip and 2x - 1 scaling.
        assert!(frame[10] < frame[WAVEFORM_SIZE - 10]);
    }

    #[test]
    fn shepard_tone_source_spans_table() {
        let sin = WaveFrame::predefined(WaveShape::Sin);
        let data = json!({
            "name": "Shepard",
            "version": "1.5.5",
            "groups": [{
                "components": [{
                    "type": "Shepard Tone Source",
                    "interpolation": 1,
                    "interpolation_style": 1,
                    "keyframes": [
                        { "position": 0, "wave_data": encode_wave(&sin.time_domain) },
                    ],
                }],
            }],
        });
        let wavetable = wavetable_from_json(&data).unwrap();
        assert_eq!(wavetable.num_frames(), 257);
        assert!(wavetable.is_shepard_table());
        // Last frame is the octave-up loop frame: bin 2 carries the energy.
        let amps = wavetable.data().frequency_amplitudes(256);
        assert!(amps[4] > amps[2], "octave shift missing: {} vs {}", amps[4], amps[2]);
    }
}
