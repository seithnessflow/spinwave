//! The synth voice kernel: statically wired producers → filters → amp
//! path with modulators and the per-voice modulation matrix. This replaces
//! Vital's per-voice Processor graph.

pub mod mod_matrix;
pub mod synth_voice;
pub mod voice_filter;

pub use mod_matrix::{Connection, ModDest, ModMatrix, ModOffsets, ModSource, SourceValues};
pub use synth_voice::{
    FilterRouting, FilterSection, KernelParams, LfoSection, NoiseSection, OscEngineKind,
    OscSection, ProducerDestination, RandomLfoSection, SampleSection, SynthVoiceKernel,
};
pub use voice_filter::{FilterModel, VoiceFilter, VoiceFilterParams};
