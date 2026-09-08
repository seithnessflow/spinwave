//! SIMD voice-pair primitives and fast math for the spinwave synth engine.
//!
//! Modern Rust rework of Vital's `poly_values.h` / `futils.h` / `poly_utils.h`
//! (GPLv3). The engine computes two stereo voices per SIMD vector; every
//! module downstream builds on the [`PolyF32`] / [`PolyMask`] types here.

// Coefficients are transcribed digit-for-digit from the C++ reference;
// keeping the full spelled-out precision is deliberate.
#![allow(clippy::excessive_precision)]

pub mod constants;
pub mod math;
pub mod matrix;
pub mod simd;
pub mod utils;

pub use matrix::Matrix;
pub use simd::{PolyF32, PolyMask, PolyU32, LANES};
