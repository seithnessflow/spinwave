//! Parameter value scales, mirroring `vital::ValueDetails::ValueScale`.
//!
//! Vital stores every parameter as a plain engine value inside `[min, max]`.
//! The *normalized* (0..1, "plugin") representation is always a linear mapping
//! of that range — the scale only changes two things, exactly as in the C++
//! reference (`value_bridge.h`):
//!
//! * `Indexed` parameters round the engine value (and the span) to integers.
//! * The scale's *skew* is applied on top of the engine value for display
//!   purposes only (`display = display_multiply * skew(engine) + post_offset`).

/// The value scale of a parameter, mirroring `vital::ValueDetails::ValueScale`.
///
/// Discriminant order matches the C++ enum (`kIndexed`, `kLinear`,
/// `kQuadratic`, `kCubic`, `kQuartic`, `kSquareRoot`, `kExponential`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ParamScale {
    /// Discrete/stepped parameter; engine values are rounded to integers.
    Indexed,
    /// Plain linear parameter.
    #[default]
    Linear,
    /// Displayed as `value^2`.
    Quadratic,
    /// Displayed as `value^3`.
    Cubic,
    /// Displayed as `value^4`.
    Quartic,
    /// Displayed as `sqrt(value)`.
    SquareRoot,
    /// Displayed as `2^value` (or `1 / 2^value` when the parameter sets
    /// `display_invert`).
    Exponential,
}

impl ParamScale {
    /// The value span used for normalization, mirroring the `ValueBridge`
    /// constructor: `max - min`, rounded for indexed parameters.
    #[must_use]
    pub fn span(self, min: f32, max: f32) -> f32 {
        let span = max - min;
        match self {
            ParamScale::Indexed => span.round(),
            _ => span,
        }
    }

    /// Converts a normalized (0..1) value to an engine value, mirroring
    /// `ValueBridge::convertToEngineValue`.
    #[must_use]
    pub fn to_engine(self, normalized: f32, min: f32, max: f32) -> f32 {
        let value = normalized * self.span(min, max) + min;
        match self {
            ParamScale::Indexed => value.round(),
            _ => value,
        }
    }

    /// Converts an engine value to a normalized (0..1) value, mirroring
    /// `ValueBridge::convertToPluginValue`.
    #[must_use]
    pub fn to_normalized(self, engine: f32, min: f32, max: f32) -> f32 {
        (engine - min) / self.span(min, max)
    }

    /// Applies the display skew to an engine value, mirroring
    /// `ValueBridge::skewValue`.
    ///
    /// `display_invert` only affects [`ParamScale::Exponential`], where the
    /// displayed value becomes `1 / 2^value` (used for frequency parameters
    /// displayed in seconds).
    #[must_use]
    pub fn skew(self, value: f32, display_invert: bool) -> f32 {
        match self {
            ParamScale::Quadratic => value * value,
            ParamScale::Cubic => value * value * value,
            ParamScale::Quartic => {
                let squared = value * value;
                squared * squared
            }
            ParamScale::Exponential => {
                if display_invert {
                    1.0 / 2.0_f32.powf(value)
                } else {
                    2.0_f32.powf(value)
                }
            }
            ParamScale::SquareRoot => value.sqrt(),
            ParamScale::Indexed | ParamScale::Linear => value,
        }
    }

    /// Inverts the display skew, mirroring `ValueBridge::unskewValue`.
    ///
    /// Note: the C++ reference has no `kSquareRoot` case in `unskewValue`
    /// (falls through to identity). We reproduce that behavior exactly, so
    /// `unskew(skew(x))` does *not* round-trip for [`ParamScale::SquareRoot`]
    /// (only `volume` uses this scale, and Vital never parses its display text
    /// through this path with the square applied).
    #[must_use]
    pub fn unskew(self, value: f32, display_invert: bool) -> f32 {
        match self {
            ParamScale::Quadratic => value.sqrt(),
            ParamScale::Cubic => value.powf(1.0 / 3.0),
            ParamScale::Quartic => value.powf(1.0 / 4.0),
            ParamScale::Exponential => {
                if display_invert {
                    (1.0 / value).log2()
                } else {
                    value.log2()
                }
            }
            ParamScale::Indexed | ParamScale::Linear | ParamScale::SquareRoot => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_SCALES: [ParamScale; 7] = [
        ParamScale::Indexed,
        ParamScale::Linear,
        ParamScale::Quadratic,
        ParamScale::Cubic,
        ParamScale::Quartic,
        ParamScale::SquareRoot,
        ParamScale::Exponential,
    ];

    #[test]
    fn engine_normalized_roundtrip() {
        for scale in ALL_SCALES {
            let (min, max) = match scale {
                ParamScale::Indexed => (0.0, 12.0),
                _ => (-2.0, 9.0),
            };
            for i in 0..=20 {
                let normalized = i as f32 / 20.0;
                let engine = scale.to_engine(normalized, min, max);
                let back = scale.to_normalized(engine, min, max);
                let engine_again = scale.to_engine(back, min, max);
                // Indexed rounds, so compare through the engine value.
                assert!(
                    (engine - engine_again).abs() < 1e-4,
                    "{scale:?}: {engine} != {engine_again}"
                );
                if scale != ParamScale::Indexed {
                    assert!(
                        (normalized - back).abs() < 1e-5,
                        "{scale:?}: {normalized} != {back}"
                    );
                }
            }
        }
    }

    #[test]
    fn indexed_rounds_to_integers() {
        let value = ParamScale::Indexed.to_engine(0.49, 0.0, 10.0);
        assert_eq!(value, 5.0);
        let value = ParamScale::Indexed.to_engine(0.0, -48.0, 48.0);
        assert_eq!(value, -48.0);
    }

    #[test]
    fn skew_unskew_roundtrip() {
        // SquareRoot deliberately mirrors the C++ asymmetry, so skip it here.
        let invertible = [
            ParamScale::Indexed,
            ParamScale::Linear,
            ParamScale::Quadratic,
            ParamScale::Cubic,
            ParamScale::Quartic,
            ParamScale::Exponential,
        ];
        for scale in invertible {
            for invert in [false, true] {
                for i in 1..=10 {
                    let value = i as f32 * 0.3;
                    let skewed = scale.skew(value, invert);
                    let back = scale.unskew(skewed, invert);
                    assert!(
                        (value - back).abs() < 1e-4,
                        "{scale:?} invert={invert}: {value} != {back}"
                    );
                }
            }
        }
    }

    #[test]
    fn exponential_display_invert() {
        // "delay_frequency" style: displayed as seconds = 1 / 2^value.
        let displayed = ParamScale::Exponential.skew(2.0, true);
        assert!((displayed - 0.25).abs() < 1e-6);
        let displayed = ParamScale::Exponential.skew(2.0, false);
        assert!((displayed - 4.0).abs() < 1e-6);
    }

    #[test]
    fn square_root_matches_cpp_quirk() {
        // C++ unskewValue has no kSquareRoot case: identity on the way back.
        assert_eq!(ParamScale::SquareRoot.skew(4.0, false), 2.0);
        assert_eq!(ParamScale::SquareRoot.unskew(2.0, false), 2.0);
    }
}
