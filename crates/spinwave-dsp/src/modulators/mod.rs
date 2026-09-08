//! Modulation sources: envelopes, drawable LFOs, random LFOs.

pub mod envelope;
pub mod line_generator;
pub mod line_map;
pub mod random;
pub mod random_lfo;
pub mod synth_lfo;
pub mod trigger_random;

pub use envelope::{Envelope, EnvelopeParams};
pub use line_generator::LineGenerator;
pub use random::RandomGenerator;
pub use random_lfo::{RandomLfo, RandomLfoParams, RandomLfoStyle};
pub use synth_lfo::{LfoSyncType, SynthLfo, SynthLfoParams};
pub use trigger_random::TriggerRandom;
