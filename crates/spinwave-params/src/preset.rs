//! Serde model of the `.vital` preset JSON format, mirroring the structure
//! written by `LoadSave::stateToJson` in the C++ reference.
//!
//! Top-level object:
//!
//! ```json
//! {
//!   "synth_version": "1.0.7",
//!   "preset_name": "...", "author": "...", "comments": "...",
//!   "preset_style": "...",
//!   "macro1": "...", "macro2": "...", "macro3": "...", "macro4": "...",
//!   "settings": { ... }
//! }
//! ```
//!
//! The `settings` object maps every parameter name to its **engine** value
//! (not normalized), and additionally holds:
//!
//! * `"modulations"`: array of `{ "source", "destination" }` objects, one per
//!   modulation slot (64 in current versions; unused slots have empty
//!   strings). A non-linear modulation remap curve adds a `"line_mapping"`
//!   object. Note that per-connection amount/power/bipolar/stereo/bypass are
//!   *not* stored here — they live in the settings map itself as the
//!   `modulation_<n>_amount` (etc.) parameters.
//! * `"lfos"`: array of line-generator shapes (one per LFO).
//! * `"wavetables"`: array of wavetable-creator states (one per oscillator).
//!   Kept as raw JSON here.
//! * `"sample"`: sample payload (name, length, sample_rate and base64 sample
//!   data). Kept as raw JSON here.
//!
//! Unknown fields at every level are preserved through flattened maps, so a
//! load → save round trip does not lose data.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A full `.vital` preset file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Preset {
    /// Vital version string that saved the preset, e.g. `"1.0.7"`.
    #[serde(default)]
    pub synth_version: String,
    #[serde(default)]
    pub preset_name: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub comments: String,
    /// One of `strings::PRESET_STYLE_NAMES` (or empty).
    #[serde(default)]
    pub preset_style: String,
    /// Display names of the four macro controls (keys `macro1` .. `macro4`).
    #[serde(default, rename = "macro1")]
    pub macro1: String,
    #[serde(default, rename = "macro2")]
    pub macro2: String,
    #[serde(default, rename = "macro3")]
    pub macro3: String,
    #[serde(default, rename = "macro4")]
    pub macro4: String,
    /// All engine state.
    #[serde(default)]
    pub settings: Settings,
    /// Any fields this model does not know about (forward compatibility).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// The `"settings"` object of a preset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Settings {
    /// Modulation slots in bank order.
    #[serde(default)]
    pub modulations: Vec<ModulationConnection>,
    /// LFO shapes in `lfo_1` .. `lfo_8` order.
    #[serde(default)]
    pub lfos: Vec<LineShape>,
    /// Wavetable-creator states, one per oscillator (raw passthrough).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wavetables: Option<Value>,
    /// Sample payload (raw passthrough).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<Value>,
    /// Everything else in `settings` — in practice the parameter values
    /// (`name -> engine value`), plus any unknown future keys. Use
    /// [`Settings::parameter`] / [`Settings::set_parameter`] for typed access.
    #[serde(flatten)]
    pub values: Map<String, Value>,
}

impl Settings {
    /// Reads a parameter's engine value from the settings map.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<f32> {
        self.values.get(name)?.as_f64().map(|v| v as f32)
    }

    /// Writes a parameter's engine value into the settings map.
    pub fn set_parameter(&mut self, name: &str, value: f32) {
        self.values.insert(
            name.to_string(),
            Value::from(f64::from(value)),
        );
    }

    /// A parameter's engine value, falling back to the table default when the
    /// preset does not store it (mirrors `LoadSave::loadControls`).
    #[must_use]
    pub fn parameter_or_default(&self, name: &str) -> Option<f32> {
        self.parameter(name)
            .or_else(|| crate::table::parameters().lookup(name).map(|d| d.default_value))
    }
}

/// One modulation slot: `{ "source": ..., "destination": ... }`, with an
/// optional `"line_mapping"` remap curve. Empty source/destination strings
/// mean the slot is unused (the C++ saves all 64 slots unconditionally).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ModulationConnection {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub destination: String,
    /// Only present when the modulation remap curve is not the default linear
    /// line (`LineGenerator::linear()` in the C++).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_mapping: Option<LineShape>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ModulationConnection {
    /// Whether this slot actually connects a source to a destination.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        !self.source.is_empty() && !self.destination.is_empty()
    }
}

/// A line-generator shape (`LineGenerator::stateToJson`), used for LFO shapes
/// and modulation remap curves:
///
/// ```json
/// {
///   "num_points": 3,
///   "points": [x0, y0, x1, y1, x2, y2],
///   "powers": [p0, p1, p2],
///   "name": "Triangle",
///   "smooth": false
/// }
/// ```
///
/// `points` is a flat interleaved array of `2 * num_points` coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineShape {
    pub num_points: u32,
    pub points: Vec<f32>,
    pub powers: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub smooth: bool,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl LineShape {
    /// The default linear ramp, matching `LineGenerator::initLinear`.
    #[must_use]
    pub fn linear() -> Self {
        LineShape {
            num_points: 2,
            points: vec![0.0, 1.0, 1.0, 0.0],
            powers: vec![0.0, 0.0],
            name: Some("Linear".to_string()),
            smooth: false,
            extra: Map::new(),
        }
    }

    /// Structural validity, mirroring `LineGenerator::isValidJson`: the point
    /// and power arrays must cover `num_points` entries.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.points.len() >= 2 * self.num_points as usize
            && self.powers.len() >= self.num_points as usize
    }

    /// The `(x, y)` coordinates of point `index`.
    #[must_use]
    pub fn point(&self, index: usize) -> Option<(f32, f32)> {
        if index >= self.num_points as usize {
            return None;
        }
        Some((*self.points.get(2 * index)?, *self.points.get(2 * index + 1)?))
    }
}

impl Default for LineShape {
    fn default() -> Self {
        LineShape::linear()
    }
}

impl Preset {
    /// Parses a `.vital` preset from JSON text.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Serializes the preset to JSON text.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serializes the preset to pretty-printed JSON text.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_preset_json() -> String {
        r#"{
            "author": "Tester",
            "comments": "roundtrip test",
            "macro1": "MACRO 1",
            "macro2": "MACRO 2",
            "macro3": "CUSTOM",
            "macro4": "MACRO 4",
            "preset_name": "Test Preset",
            "preset_style": "Lead",
            "synth_version": "1.0.7",
            "future_top_level_key": {"nested": [1, 2, 3]},
            "settings": {
                "osc_1_on": 1.0,
                "osc_1_level": 0.70710678119,
                "filter_1_cutoff": 30.0,
                "env_1_attack": 0.1495,
                "delay_tempo": 9.0,
                "some_future_param": 0.25,
                "modulations": [
                    {"source": "env_2", "destination": "filter_1_cutoff"},
                    {"source": "lfo_1", "destination": "osc_1_tune",
                     "line_mapping": {"num_points": 2, "points": [0.0, 1.0, 1.0, 0.0],
                                      "powers": [0.0, 0.0], "name": "Linear", "smooth": false}},
                    {"source": "", "destination": ""}
                ],
                "lfos": [
                    {"num_points": 3, "points": [0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
                     "powers": [0.0, 0.0, 0.0], "name": "Triangle", "smooth": false,
                     "future_lfo_key": true}
                ],
                "sample": {"name": "White Noise", "length": 4, "sample_rate": 44100,
                           "samples": "AAAA"},
                "wavetables": [{"name": "Init", "groups": []}]
            }
        }"#
        .to_string()
    }

    #[test]
    fn deserialize_minimal_preset() {
        let preset = Preset::from_json(&minimal_preset_json()).unwrap();
        assert_eq!(preset.preset_name, "Test Preset");
        assert_eq!(preset.author, "Tester");
        assert_eq!(preset.preset_style, "Lead");
        assert_eq!(preset.macro3, "CUSTOM");
        assert_eq!(preset.synth_version, "1.0.7");

        assert_eq!(preset.settings.parameter("osc_1_on"), Some(1.0));
        assert_eq!(preset.settings.parameter("filter_1_cutoff"), Some(30.0));
        assert_eq!(preset.settings.parameter("env_1_attack"), Some(0.1495));
        // Missing parameter falls back to the table default.
        assert_eq!(
            preset.settings.parameter_or_default("reverb_dry_wet"),
            Some(0.25)
        );

        assert_eq!(preset.settings.modulations.len(), 3);
        assert!(preset.settings.modulations[0].is_connected());
        assert_eq!(preset.settings.modulations[0].source, "env_2");
        assert_eq!(preset.settings.modulations[0].destination, "filter_1_cutoff");
        assert!(preset.settings.modulations[1].line_mapping.is_some());
        assert!(!preset.settings.modulations[2].is_connected());

        assert_eq!(preset.settings.lfos.len(), 1);
        let lfo = &preset.settings.lfos[0];
        assert_eq!(lfo.num_points, 3);
        assert!(lfo.is_valid());
        assert_eq!(lfo.point(1), Some((0.5, 0.0)));
        assert_eq!(lfo.name.as_deref(), Some("Triangle"));
        assert!(lfo.extra.contains_key("future_lfo_key"));

        assert!(preset.settings.sample.is_some());
        assert!(preset.settings.wavetables.is_some());
        assert!(preset.extra.contains_key("future_top_level_key"));
    }

    #[test]
    fn roundtrip_is_stable() {
        let original: Value = serde_json::from_str(&minimal_preset_json()).unwrap();
        let preset = Preset::from_json(&minimal_preset_json()).unwrap();
        let reserialized: Value = serde_json::from_str(&preset.to_json().unwrap()).unwrap();
        // Known and unknown fields must survive the load -> save round trip.
        assert_eq!(original, reserialized);

        // A second trip through the typed model must be identical too.
        let preset2 = Preset::from_json(&preset.to_json().unwrap()).unwrap();
        assert_eq!(preset, preset2);
    }

    #[test]
    fn set_parameter_roundtrip() {
        let mut preset = Preset::default();
        preset.settings.set_parameter("osc_1_level", 0.5);
        assert_eq!(preset.settings.parameter("osc_1_level"), Some(0.5));
        let text = preset.to_json().unwrap();
        let parsed = Preset::from_json(&text).unwrap();
        assert_eq!(parsed.settings.parameter("osc_1_level"), Some(0.5));
    }

    #[test]
    fn default_line_shape_matches_init_linear() {
        let shape = LineShape::default();
        assert_eq!(shape.num_points, 2);
        assert_eq!(shape.points, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(shape.powers, vec![0.0, 0.0]);
        assert!(!shape.smooth);
        assert!(shape.is_valid());
    }
}
