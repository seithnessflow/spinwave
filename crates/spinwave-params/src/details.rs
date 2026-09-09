//! Per-parameter metadata, mirroring `vital::ValueDetails`.

use crate::scale::ParamScale;

/// Full metadata for one synth parameter, mirroring `vital::ValueDetails`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDetails {
    /// Machine name and preset key, e.g. `"osc_1_level"`.
    pub name: String,
    /// Version the parameter was added in, encoded like the C++ constants
    /// (`0x000502` == version 0.5.2).
    pub version_added: u32,
    /// Minimum engine value.
    pub min: f32,
    /// Maximum engine value.
    pub max: f32,
    /// Default engine value.
    pub default_value: f32,
    /// Offset added after display skewing (offsets quadratic/exponential
    /// display values, e.g. `-60` for filter cutoff, `-80` for volume).
    pub post_offset: f32,
    /// Multiplier applied to the skewed value for display.
    pub display_multiply: f32,
    /// Value scale (see [`ParamScale`]).
    pub scale: ParamScale,
    /// When true, exponential display uses `1 / 2^value` instead of `2^value`.
    pub display_invert: bool,
    /// Units suffix for display, e.g. `" secs"`, `"%"`.
    pub display_units: String,
    /// Human-readable name, e.g. `"Oscillator 1 Level"`.
    pub display_name: String,
    /// For indexed parameters: the display strings for each step.
    pub string_lookup: Option<&'static [&'static str]>,
    /// Group-local display name, e.g. `"Level"` for `"osc_1_level"`.
    /// Empty for non-grouped parameters (matches the C++ behavior where
    /// `local_description` is only filled by `addParameterGroup`).
    pub local_description: String,
    /// `true` for parameters that only exist in Spinwave (extra oscillator /
    /// envelope / LFO / macro slots, the `noise_*`, `bus_*`, `fx_split_*`,
    /// `osc_N_engine`, `*_gran_*`, `*_smp_*`, `lfo_N_generator` families).
    /// The preset writer may omit them at their default so the `.vital`
    /// file stays loadable in Vital.
    pub spinwave_only: bool,
}

impl ParamDetails {
    /// The value span used for normalization (`max - min`, rounded for
    /// indexed parameters).
    #[must_use]
    pub fn span(&self) -> f32 {
        self.scale.span(self.min, self.max)
    }

    /// Converts a normalized (0..1) value to an engine value
    /// (`ValueBridge::convertToEngineValue`).
    #[must_use]
    pub fn to_engine(&self, normalized: f32) -> f32 {
        self.scale.to_engine(normalized, self.min, self.max)
    }

    /// Converts an engine value to a normalized (0..1) value
    /// (`ValueBridge::convertToPluginValue`).
    #[must_use]
    pub fn to_normalized(&self, engine: f32) -> f32 {
        self.scale.to_normalized(engine, self.min, self.max)
    }

    /// Normalized default value (`ValueBridge::getDefaultValue`).
    #[must_use]
    pub fn default_normalized(&self) -> f32 {
        self.to_normalized(self.default_value)
    }

    /// The numeric value shown in the UI for an engine value:
    /// `display_multiply * skew(engine) + post_offset`.
    #[must_use]
    pub fn display_value(&self, engine: f32) -> f32 {
        self.display_multiply * self.scale.skew(engine, self.display_invert) + self.post_offset
    }

    /// The display text for an engine value, mirroring `ValueBridge::getText`:
    /// indexed parameters with a string table show the string, everything else
    /// shows the skewed numeric value followed by the units.
    #[must_use]
    pub fn display_string(&self, engine: f32) -> String {
        if let Some(lookup) = self.string_lookup {
            let index = engine.min(self.max).max(0.0) as usize;
            let index = index.min(lookup.len().saturating_sub(1));
            return lookup.get(index).copied().unwrap_or_default().to_string();
        }
        format!("{}{}", self.display_value(engine), self.display_units)
            .trim()
            .to_string()
    }

    /// Whether the parameter is a discrete stepped control
    /// (`ValueBridge::isDiscrete`).
    #[must_use]
    pub fn is_discrete(&self) -> bool {
        const MAX_INDEXED_STEPS: f32 = 300.0;
        self.scale == ParamScale::Indexed && self.span() < MAX_INDEXED_STEPS
    }

    /// Whether the parameter is a two-state switch (`ValueBridge::isBoolean`).
    #[must_use]
    pub fn is_boolean(&self) -> bool {
        self.is_discrete() && self.span() == 1.0
    }
}
