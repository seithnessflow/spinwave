//! Voice management, modulation routing and the sound engine for spinwave.
//!
//! Rework of Vital's engine layer: the dynamic Processor graph is replaced
//! by a statically wired voice kernel driven by [`allocator::VoiceAllocator`];
//! only modulation connections stay dynamic.

pub mod allocator;
pub mod effect_chain;
pub mod engine;
pub mod kernel;
pub mod modulation;
pub mod tempo;
pub mod tuning;
pub mod voice;

pub use allocator::{VoiceAllocator, VoiceKernel, VoiceOverride, VoicePriority};
pub use engine::{
    decode_order, BusParams, ChainId, Effect, EffectChain, EffectSplit, EffectsParams,
    MasterParams, MixerParams, SoundEngine, SplitMode, StereoMode, SyncMode, SyncedFrequency,
};
pub use tempo::LfoSync;
pub use voice::{KeyState, Trigger, Voice, VoiceControls};
