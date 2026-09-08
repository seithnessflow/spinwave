//! Wavetable data model: frames, tables, band-limited lookups.

pub mod lookup_table;
pub mod wave_frame;
#[allow(clippy::module_inception)]
pub mod wavetable;

pub use lookup_table::OneDimLookup;
pub use wave_frame::{
    WaveFrame, WaveShape, NUM_REAL_COMPLEX, WAVEFORM_BITS, WAVEFORM_SIZE,
};
pub use wavetable::{
    Wavetable, WavetableData, FREQUENCY_BINS, NUM_HARMONICS, NUM_OSCILLATOR_WAVE_FRAMES,
    POLY_FREQUENCY_FLOATS,
};
