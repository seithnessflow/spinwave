//! Parameter table and `.vital` preset (de)serialization for the spinwave
//! engine.
//!
//! This crate is the pure data layer of the port: it knows every synth
//! parameter (name, range, default, scale, display metadata - mirroring
//! `src/common/synth_parameters.cpp` from the C++ reference), the engine
//! constants (`synth_constants.h`), and the `.vital` preset JSON structure
//! (`LoadSave::stateToJson` / `jsonToState`). It has no dependency on the DSP
//! crates.
//!
//! # Quick tour
//!
//! ```
//! use spinwave_params::{parameters, Preset};
//!
//! // Parameter table.
//! let cutoff = parameters().lookup("filter_1_cutoff").unwrap();
//! assert_eq!(cutoff.min, 8.0);
//! assert_eq!(cutoff.max, 136.0);
//! let engine = cutoff.to_engine(0.5); // normalized 0..1 -> engine value
//! assert_eq!(cutoff.to_normalized(engine), 0.5);
//!
//! // Preset parsing.
//! let preset = Preset::from_json(
//!     r#"{"synth_version": "1.0.7", "preset_name": "Init", "settings": {"osc_1_level": 0.5}}"#,
//! ).unwrap();
//! assert_eq!(preset.settings.parameter("osc_1_level"), Some(0.5));
//! ```

pub mod base64;
pub mod constants;
pub mod details;
pub mod migrate;
pub mod preset;
pub mod scale;
pub mod strings;
pub mod table;

pub use details::ParamDetails;
pub use preset::{
    LineShape, LoadReport, ModulationConnection, Preset, Settings, SlotMaterials,
    SpinwaveMaterials,
};
pub use scale::ParamScale;
pub use table::{parameters, ParamTable};
