//! DSP building blocks for the vital-rs synth engine.
//!
//! Everything here is framework-free: plain structs with explicit state,
//! processing `PolyF32` buffers (two stereo voices per vector). Voice
//! management, modulation routing and graph wiring live in `vital-engine`.

pub mod memory;

pub use vital_poly::{constants, math, utils, Matrix, PolyF32, PolyMask, PolyU32, LANES};
