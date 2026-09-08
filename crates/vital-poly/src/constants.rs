//! Engine-wide constants (port of Vital's `common.h`).

pub const PI: f32 = core::f32::consts::PI;
pub const SQRT_2: f32 = core::f32::consts::SQRT_2;
pub const EPSILON: f32 = 1e-16;

/// Maximum samples processed per block, before oversampling.
pub const MAX_BUFFER_SIZE: usize = 128;
pub const MAX_OVERSAMPLE: usize = 8;
pub const DEFAULT_SAMPLE_RATE: u32 = 44100;
pub const MAX_SAMPLE_RATE: u32 = 192_000;
/// Highest playable ratio of a frequency to the sample rate before culling.
pub const MIN_NYQUIST_MULT: f32 = 0.45351473923;

pub const MIDI_SIZE: usize = 128;
pub const MIDI_TRACK_CENTER: i32 = 60;
pub const MIDI_0_FREQUENCY: f32 = 8.1757989156;
pub const NOTES_PER_OCTAVE: i32 = 12;
pub const CENTS_PER_NOTE: i32 = 100;
pub const CENTS_PER_OCTAVE: i32 = NOTES_PER_OCTAVE * CENTS_PER_NOTE;

pub const DB_GAIN_CONVERSION_MULT: f32 = 6.02059991329;
pub const DB_MAGNITUDE_CONVERSION_MULT: f32 = 1.0 / DB_GAIN_CONVERSION_MULT;
pub const EXP_CONVERSION_MULT: f32 = core::f32::consts::LOG2_E;
pub const LOG_CONVERSION_MULT: f32 = core::f32::consts::LN_2;

/// Pulses per quarter note used for host-sync phase math.
pub const PPQ: u32 = 960;
/// Seconds to fade out a voice being stolen.
pub const VOICE_KILL_TIME: f32 = 0.05;
pub const NUM_MIDI_CHANNELS: usize = 16;

/// Lifecycle values carried by note triggers through the voice graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum VoiceEvent {
    Invalid = 0,
    Idle,
    On,
    Hold,
    Decay,
    Off,
    Kill,
}

impl VoiceEvent {
    #[inline(always)]
    pub fn as_f32(self) -> f32 {
        self as u32 as f32
    }
}
