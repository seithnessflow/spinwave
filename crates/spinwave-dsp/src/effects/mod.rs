//! Bus effects: delay, distortion, phaser, reverb, compressor.
//!
//! Effects run on the summed voice signal, but still use [`spinwave_poly::PolyF32`]
//! lanes as `[L, R, L, R]`. Parameters arrive once per block through plain
//! param structs; every effect ramps its internal coefficients across the
//! block so automation stays click-free.

mod lanes;
mod one_pole;

pub mod chorus;
pub mod compressor;
pub mod delay;
pub mod distortion;
pub mod equalizer;
pub mod flanger;
pub mod multiband_compressor;
pub mod phaser;
pub mod phaser_filter;
pub mod reverb;

pub use chorus::{Chorus, ChorusParams};
pub use compressor::{Compressor, CompressorParams};
pub use delay::{Delay, DelayParams, DelayStyle, MultiDelay, StereoDelay};
pub use distortion::{Distortion, DistortionType};
pub use equalizer::{Equalizer, EqualizerParams};
pub use flanger::{Flanger, FlangerParams};
pub use multiband_compressor::{BandOptions, MultibandCompressor, MultibandCompressorParams};
pub use phaser::{Phaser, PhaserParams};
pub use phaser_filter::{PhaserFilter, PhaserFilterParams};
pub use reverb::{Reverb, ReverbParams};
