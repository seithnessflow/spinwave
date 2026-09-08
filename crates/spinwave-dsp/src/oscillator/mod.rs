//! The wavetable oscillator: phase distortion, spectral morphing, sampler.

pub mod granular;
pub mod multisample;
pub mod noise;
pub mod phase;
mod rng;
pub mod sample_source;
pub mod spectral_morph;
pub mod synth_oscillator;

pub use granular::{GrainDirection, GrainInterpolation, GrainWindow, Granular, GranularParams, MAX_GRAINS};
pub use multisample::{Multisample, MultisampleSource, MultisampleZone, SfzLoopMode};
pub use phase::{adjust_phase, phase_window, shape_distortion_values, DistortionType};
pub use sample_source::{Sample, SampleSource, SampleSourceParams};
pub use spectral_morph::{
    run_spectral_morph, shape_spectral_morph_values, SpectralMorph, RANDOM_AMPLITUDE_STAGES,
};
pub use synth_oscillator::{
    band_limited_harmonics, linearly_interpolate_buffer, SynthOscillator, SynthOscillatorParams,
    UnisonStackType, MAX_UNISON, NUM_POLY_PHASE,
};
