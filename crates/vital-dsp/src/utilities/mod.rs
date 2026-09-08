//! Small shared processors: smoothed values, portamento, legato, metering.

pub mod legato;
pub mod peak_meter;
pub mod portamento;
pub mod smooth_value;

pub use legato::{LegatoFilter, TriggerEvent};
pub use peak_meter::PeakMeter;
pub use portamento::{PortamentoParams, PortamentoSlope};
pub use smooth_value::{ControlSmoothValue, SmoothValue};
