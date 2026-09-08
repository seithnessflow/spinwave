//! DSP building blocks for the spinwave synth engine.
//!
//! Everything here is framework-free: plain structs with explicit state,
//! processing `PolyF32` buffers (two stereo voices per vector). Voice
//! management, modulation routing and graph wiring live in `spinwave-engine`.

pub mod effects;
pub mod filters;
pub mod memory;
pub mod modulators;
pub mod oscillator;
pub mod utilities;
pub mod wavetable;

pub use spinwave_poly::{constants, math, utils, Matrix, PolyF32, PolyMask, PolyU32, LANES};
