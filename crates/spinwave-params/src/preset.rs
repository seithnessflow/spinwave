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
    /// Sample payload (raw passthrough): `{name, length, sample_rate,
    /// samples (base64 PCM16), samples_stereo?}` — see
    /// [`SampleJson`] for the typed view.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<Value>,
    /// Spinwave-only material block: per-oscillator-slot samples, SFZ
    /// instruments and the 4th slot's wavetable. Vital ignores unknown
    /// settings keys, so its presence keeps the file loadable there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spinwave_materials: Option<SpinwaveMaterials>,
    /// Everything else in `settings` — in practice the parameter values
    /// (`name -> engine value`), plus any unknown future keys. Use
    /// [`Settings::parameter`] / [`Settings::set_parameter`] for typed access.
    #[serde(flatten)]
    pub values: Map<String, Value>,
}

/// Typed view of Vital's `settings.sample` payload (`Sample::stateToJson`):
/// PCM16 base64 per channel, `length` frames at `sample_rate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SampleJson {
    #[serde(default)]
    pub name: String,
    pub length: u64,
    pub sample_rate: u32,
    /// Left (or mono) channel, base64 little-endian PCM16.
    pub samples: String,
    /// Right channel when the sample is stereo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples_stereo: Option<String>,
}

impl SampleJson {
    /// Builds the payload from float channels (`None` right = mono).
    #[must_use]
    pub fn from_channels(name: &str, left: &[f32], right: Option<&[f32]>, sample_rate: u32) -> Self {
        SampleJson {
            name: name.to_string(),
            length: left.len() as u64,
            sample_rate,
            samples: crate::base64::encode_pcm16(left),
            samples_stereo: right.map(crate::base64::encode_pcm16),
        }
    }

    /// Decodes the channels: `(left, Some(right))` for stereo. `None` when
    /// the base64 is invalid or the payload is empty; channels are cut to
    /// `length` frames like the C++ (`memcpy(..., length * sizeof(int16_t))`).
    #[must_use]
    pub fn decode(&self) -> Option<(Vec<f32>, Option<Vec<f32>>)> {
        let length = self.length as usize;
        let mut left = crate::base64::decode_pcm16(&self.samples)?;
        if left.is_empty() {
            return None;
        }
        left.truncate(length.max(1));
        let right = match &self.samples_stereo {
            Some(text) => {
                let mut right = crate::base64::decode_pcm16(text)?;
                right.truncate(left.len());
                if right.len() < left.len() {
                    right.resize(left.len(), 0.0);
                }
                Some(right)
            }
            None => None,
        };
        Some((left, right))
    }

    /// Parses `settings.sample`.
    #[must_use]
    pub fn from_value(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }
}

/// Spinwave material descriptors for the oscillator slots, stored under
/// `settings.spinwave_materials`. One entry per slot (`osc_1` .. `osc_4`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SpinwaveMaterials {
    #[serde(default)]
    pub slots: Vec<SlotMaterials>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl SpinwaveMaterials {
    /// The slot's descriptors, if any were stored.
    #[must_use]
    pub fn slot(&self, slot: usize) -> Option<&SlotMaterials> {
        self.slots.get(slot)
    }

    /// Mutable access, growing the slot list as needed.
    pub fn slot_mut(&mut self, slot: usize) -> &mut SlotMaterials {
        if self.slots.len() <= slot {
            self.slots.resize_with(slot + 1, SlotMaterials::default);
        }
        &mut self.slots[slot]
    }

    /// Whether nothing at all is stored (the block can be dropped).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(SlotMaterials::is_empty) && self.extra.is_empty()
    }
}

/// Material loaded into one oscillator slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SlotMaterials {
    /// Sample audio for the Sample / Granular engines, embedded in Vital's
    /// `settings.sample` format (PCM16 base64).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample: Option<SampleJson>,
    /// SFZ instrument for the Multisample engine: the file path (zone
    /// samples resolve relative to it) and the SFZ text as loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sfz: Option<SfzMaterial>,
    /// Wavetable-creator JSON for slots Vital has no `settings.wavetables`
    /// entry for (slot 3 = `osc_4`); slots 0..2 use `settings.wavetables`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wavetable: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl SlotMaterials {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sample.is_none() && self.sfz.is_none() && self.wavetable.is_none()
    }
}

/// An SFZ instrument reference: path (for relative sample opcodes) plus
/// the text itself so the preset stays self-describing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SfzMaterial {
    pub path: String,
    #[serde(default)]
    pub text: String,
}

/// What a preset load had to drop or change. Surfaced by the MCP
/// `load_preset` / `set_patch` tools and the live `set_patch` reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct LoadReport {
    /// Modulation connections whose source or destination the engine does
    /// not expose (`"source -> destination"`).
    #[serde(default)]
    pub ignored_connections: Vec<String>,
    /// Numeric settings keys absent from the parameter table.
    #[serde(default)]
    pub unknown_params: Vec<String>,
    /// The `synth_version` the preset was written by when migrations ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migrated_from: Option<String>,
    /// Human-readable list of the migrations applied.
    #[serde(default)]
    pub migrations: Vec<String>,
    /// Anything else worth telling the user (embedded sample decoded,
    /// remap curves the engine cannot apply yet, ...).
    #[serde(default)]
    pub notes: Vec<String>,
    /// What the text-format parser accepted with a normalisation: a unit
    /// alias, a name alias, a factory shape recognised by its points. Each
    /// says what was written and what it was read as, so nothing is
    /// corrected silently.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrections: Vec<Correction>,
    /// What the text-format parser refused, with enough for a program to
    /// fix its own file: the line, the key, a code, the valid range or the
    /// nearest name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<LoadError>,
}

/// A normalisation the text parser applied and is telling you about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Correction {
    pub line: u32,
    pub key: String,
    pub written: String,
    pub read_as: String,
    pub kind: CorrectionKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionKind {
    /// `8kHz` read as `8000 Hz`, `0.09s` as `90 ms`, `+2 oct` as `+18.75%`.
    UnitNormalised,
    /// A display name or an old spelling resolved to the table name.
    AliasResolved,
    /// A drawn shape whose points equal a factory shape, named as such.
    FactoryShape,
    /// A modulation slot moved to its canonical position.
    SlotRenumbered,
}

/// Something the text parser refused. `expected` carries the valid range
/// or the expected unit in the same terms as the input; `suggestion` the
/// nearest known name when there is exactly one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadError {
    pub line: u32,
    pub key: String,
    pub code: ErrorCode,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    UnknownKey,
    UnknownModule,
    MissingUnit,
    WrongUnit,
    OutOfRange,
    AmbiguousName,
    BadValue,
    BadRegime,
    RequiresMismatch,
    UnsupportedFormatVersion,
    MissingFormatVersion,
    BlobMissing,
    Syntax,
}

impl LoadReport {
    /// Whether the load was lossless (nothing dropped, nothing migrated).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.ignored_connections.is_empty()
            && self.unknown_params.is_empty()
            && self.migrated_from.is_none()
            && self.notes.is_empty()
            && self.corrections.is_empty()
            && self.errors.is_empty()
    }

    /// One-line summary for tool replies (empty when clean).
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if let Some(from) = &self.migrated_from {
            parts.push(format!(
                "migrated from {from} ({} step(s): {})",
                self.migrations.len(),
                self.migrations.join(", ")
            ));
        }
        if !self.ignored_connections.is_empty() {
            parts.push(format!(
                "{} modulation(s) ignored: {}",
                self.ignored_connections.len(),
                self.ignored_connections.join(", ")
            ));
        }
        if !self.unknown_params.is_empty() {
            parts.push(format!(
                "{} unknown parameter(s): {}",
                self.unknown_params.len(),
                self.unknown_params.join(", ")
            ));
        }
        parts.extend(self.notes.iter().cloned());
        if !self.corrections.is_empty() {
            parts.push(format!("{} value(s) normalised", self.corrections.len()));
        }
        for error in &self.errors {
            let mut line = format!("line {}: {}", error.line, error.message);
            if let Some(suggestion) = &error.suggestion {
                line.push_str(&format!(" (did you mean {suggestion}?)"));
            }
            parts.push(line);
        }
        parts.join("; ")
    }
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

    /// Converts a preset written by an older Vital version in place
    /// (`LoadSave::updateFromOldVersion`), returning the migrations applied
    /// — empty for a current preset. See [`crate::migrate`].
    pub fn upgrade(&mut self) -> Vec<String> {
        crate::migrate::upgrade(self)
    }

    /// The preset's `synth_version` parsed as `(major, minor, patch)`;
    /// missing parts read as 0, an unparsable string as `None`.
    #[must_use]
    pub fn version_tuple(&self) -> Option<(u32, u32, u32)> {
        crate::migrate::parse_version(&self.synth_version)
    }

    /// Settings keys that hold a number but name no table parameter
    /// (candidates for [`LoadReport::unknown_params`]).
    #[must_use]
    pub fn unknown_parameters(&self) -> Vec<String> {
        let table = crate::table::parameters();
        let mut names: Vec<String> = self
            .settings
            .values
            .iter()
            .filter(|(name, value)| value.is_number() && !table.is_parameter(name))
            .map(|(name, _)| name.clone())
            .collect();
        names.sort();
        names
    }

    /// A copy for writing a Vital-loadable `.vital` file: Spinwave-only
    /// parameters sitting at their default are dropped (Vital would ignore
    /// them anyway; dropping keeps the file lean), everything else — set
    /// Spinwave-only values, `spinwave_materials`, unknown keys — is kept.
    #[must_use]
    pub fn for_vital_file(&self) -> Preset {
        let table = crate::table::parameters();
        let mut copy = self.clone();
        copy.settings.values.retain(|name, value| match table.lookup(name) {
            Some(details) if details.spinwave_only => {
                let stored = value.as_f64().map(|v| v as f32);
                stored.is_none_or(|v| (v - details.default_value).abs() > 1e-6)
            }
            _ => true,
        });
        if copy.settings.spinwave_materials.as_ref().is_some_and(SpinwaveMaterials::is_empty) {
            copy.settings.spinwave_materials = None;
        }
        copy
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
    fn sample_json_round_trips_channels() {
        let left = [0.0f32, 0.25, -0.5, 1.0];
        let right = [0.1f32, -0.1, 0.2, -0.2];
        let payload = SampleJson::from_channels("kick", &left, Some(&right), 48000);
        assert_eq!(payload.length, 4);
        let value = serde_json::to_value(&payload).unwrap();
        assert!(value["samples_stereo"].is_string());
        let parsed = SampleJson::from_value(&value).unwrap();
        let (l, r) = parsed.decode().unwrap();
        for (a, b) in left.iter().zip(&l) {
            assert!((a - b).abs() < 1e-4);
        }
        let r = r.unwrap();
        for (a, b) in right.iter().zip(&r) {
            assert!((a - b).abs() < 1e-4);
        }
        // Mono payload, as Vital writes for mono samples.
        let mono = SampleJson::from_channels("m", &left, None, 44100);
        assert!(mono.decode().unwrap().1.is_none());
    }

    #[test]
    fn spinwave_materials_survive_the_round_trip() {
        let mut preset = Preset::default();
        preset.settings.set_parameter("osc_1_engine", 1.0);
        let materials = preset.settings.spinwave_materials.get_or_insert_with(Default::default);
        materials.slot_mut(0).sample = Some(SampleJson::from_channels("s", &[0.5, -0.5], None, 44100));
        materials.slot_mut(3).wavetable = Some(serde_json::json!({"name": "T", "groups": []}));
        let text = preset.to_json().unwrap();
        let parsed = Preset::from_json(&text).unwrap();
        let block = parsed.settings.spinwave_materials.as_ref().unwrap();
        assert_eq!(block.slot(0).unwrap().sample.as_ref().unwrap().name, "s");
        assert!(block.slot(1).unwrap().is_empty());
        assert!(block.slot(3).unwrap().wavetable.is_some());
        assert_eq!(parsed.settings.parameter("osc_1_engine"), Some(1.0));
    }

    #[test]
    fn vital_file_view_drops_default_spinwave_params_only() {
        let mut preset = Preset::default();
        preset.settings.set_parameter("osc_1_engine", 0.0); // spinwave default
        preset.settings.set_parameter("osc_4_on", 1.0); // spinwave, set
        preset.settings.set_parameter("noise_level", 0.5); // spinwave default
        preset.settings.set_parameter("osc_1_level", 0.6); // vital param, kept
        preset.settings.set_parameter("mystery_key", 3.0); // unknown, kept
        let vital = preset.for_vital_file();
        assert_eq!(vital.settings.parameter("osc_1_engine"), None);
        assert_eq!(vital.settings.parameter("noise_level"), None);
        assert_eq!(vital.settings.parameter("osc_4_on"), Some(1.0));
        assert_eq!(vital.settings.parameter("osc_1_level"), Some(0.6));
        assert_eq!(vital.settings.parameter("mystery_key"), Some(3.0));
        assert_eq!(preset.unknown_parameters(), vec!["mystery_key".to_string()]);
    }

    #[test]
    fn load_report_summary() {
        let mut report = LoadReport::default();
        assert!(report.is_clean());
        assert_eq!(report.summary(), "");
        report.ignored_connections.push("lfo_1 -> nope".into());
        report.migrated_from = Some("0.4.0".into());
        report.migrations.push("0.5.0 sub -> osc_3".into());
        assert!(!report.is_clean());
        let summary = report.summary();
        assert!(summary.contains("migrated from 0.4.0"));
        assert!(summary.contains("lfo_1 -> nope"));
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
