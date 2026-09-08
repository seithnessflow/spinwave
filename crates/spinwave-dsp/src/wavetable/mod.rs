//! Wavetable data model: frames, tables, band-limited lookups, plus the
//! creator pipeline that renders `.vital` wavetable JSON and the
//! procedural factory tables.

mod codec;
mod components;
mod sources;

pub mod creator;
pub mod factory;
pub mod lookup_table;
pub mod wave_frame;
#[allow(clippy::module_inception)]
pub mod wavetable;

pub use creator::{wavetable_from_json, wavetable_from_json_with_warnings};
pub use factory::{
    basic_shapes, factory_table, formant_growl, harmonic_series, pwm, FACTORY_TABLE_NAMES,
};
pub use lookup_table::OneDimLookup;
pub use wave_frame::{
    WaveFrame, WaveShape, NUM_REAL_COMPLEX, WAVEFORM_BITS, WAVEFORM_SIZE,
};
pub use wavetable::{
    Wavetable, WavetableData, FREQUENCY_BINS, NUM_HARMONICS, NUM_OSCILLATOR_WAVE_FRAMES,
    POLY_FREQUENCY_FLOATS,
};
