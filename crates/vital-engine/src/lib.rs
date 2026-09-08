//! Voice management, modulation routing and the sound engine for vital-rs.
//!
//! Rework of Vital's engine layer: the dynamic Processor graph is replaced
//! by a statically wired voice kernel driven by [`allocator::VoiceAllocator`];
//! only modulation connections stay dynamic.

pub mod allocator;
pub mod modulation;
pub mod voice;

pub use allocator::{VoiceAllocator, VoiceKernel, VoiceOverride, VoicePriority};
pub use voice::{KeyState, Trigger, Voice, VoiceControls};
