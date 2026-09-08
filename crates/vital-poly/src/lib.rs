//! SIMD voice-pair primitives and fast math for the vital-rs synth engine.
//!
//! Modern Rust rework of Vital's `poly_values.h` / `futils.h` / `poly_utils.h`
//! (GPLv3). The engine computes two stereo voices per SIMD vector; every
//! module downstream builds on the [`PolyF32`] / [`PolyMask`] types here.

pub mod constants;
pub mod math;
pub mod matrix;
pub mod simd;
pub mod utils;

pub use matrix::Matrix;
pub use simd::{PolyF32, PolyMask, PolyU32, LANES};
