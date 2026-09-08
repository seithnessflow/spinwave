//! Per-voice state: lifecycle, note articulation, MPE expression.
//!
//! Two voices share one SIMD kernel instance (a "pair"); each voice owns
//! a lane mask covering its stereo half of the vector.

use spinwave_poly::constants::VoiceEvent;
use spinwave_poly::{PolyF32, PolyMask};

pub const DEFAULT_LIFT_VELOCITY: f32 = 0.5;

/// Key lifecycle, distinct from the audio event: a voice can be Released
/// while its release envelope still sounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyState {
    Triggering,
    Held,
    Sustained,
    Released,
    Dead,
}

/// Snapshot of the articulation values a voice was started with.
#[derive(Clone, Copy, Debug)]
pub struct VoiceNote {
    pub event: VoiceEvent,
    pub midi_note: i32,
    pub tuned_note: f32,
    /// Previous note per lane, for portamento glides.
    pub last_note: PolyF32,
    pub velocity: f32,
    pub lift: f32,
    pub local_pitch_bend: f32,
    /// 1-based position among currently pressed notes.
    pub note_pressed: i32,
    /// Monotonic note counter (voice age).
    pub note_count: i32,
    pub channel: usize,
    pub sostenuto_pressed: bool,
}

impl Default for VoiceNote {
    fn default() -> Self {
        VoiceNote {
            event: VoiceEvent::Off,
            midi_note: 0,
            tuned_note: 0.0,
            last_note: PolyF32::ZERO,
            velocity: 0.0,
            lift: 0.0,
            local_pitch_bend: 0.0,
            note_pressed: 0,
            note_count: 0,
            channel: 0,
            sostenuto_pressed: false,
        }
    }
}

/// One playable voice: half the lanes of a [`VoicePair`] kernel.
#[derive(Clone, Debug)]
pub struct Voice {
    /// Index of the kernel pair this voice runs on.
    pub pair: usize,
    /// Slot within the pair (0 or 1) â€” selects the lane mask.
    pub slot: usize,
    pub state: VoiceNote,
    key_state: KeyState,
    last_key_state: KeyState,
    /// Sample offset of a pending event, or `None`.
    pub event_sample: Option<usize>,
    pub aftertouch: f32,
    pub aftertouch_sample: Option<usize>,
    pub slide: f32,
    pub slide_sample: Option<usize>,
}

impl Voice {
    pub fn new(pair: usize, slot: usize) -> Voice {
        Voice {
            pair,
            slot,
            state: VoiceNote::default(),
            key_state: KeyState::Dead,
            last_key_state: KeyState::Dead,
            event_sample: None,
            aftertouch: 0.0,
            aftertouch_sample: None,
            slide: 0.0,
            slide_sample: None,
        }
    }

    /// Lanes `[2*slot, 2*slot+1]` â€” this voice's stereo half of the vector.
    #[inline(always)]
    pub fn mask(&self) -> PolyMask {
        let lane = self.slot as f32;
        PolyF32::from_lanes([0.0, 0.0, 1.0, 1.0]).eq(PolyF32::splat(lane))
    }

    pub fn key_state(&self) -> KeyState {
        self.key_state
    }

    pub fn last_key_state(&self) -> KeyState {
        self.last_key_state
    }

    fn set_key_state(&mut self, key_state: KeyState) {
        self.last_key_state = self.key_state;
        self.key_state = key_state;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn activate(
        &mut self,
        midi_note: i32,
        tuned_note: f32,
        velocity: f32,
        last_note: PolyF32,
        note_pressed: i32,
        note_count: i32,
        sample: usize,
        channel: usize,
    ) {
        self.event_sample = Some(sample);
        self.state = VoiceNote {
            event: VoiceEvent::On,
            midi_note,
            tuned_note,
            last_note,
            velocity,
            lift: DEFAULT_LIFT_VELOCITY,
            local_pitch_bend: 0.0,
            note_pressed,
            note_count,
            channel,
            sostenuto_pressed: false,
        };
        self.aftertouch = 0.0;
        self.aftertouch_sample = None;
        self.slide = 0.0;
        self.slide_sample = None;
        self.set_key_state(KeyState::Triggering);
    }

    pub fn sustain(&mut self) {
        self.set_key_state(KeyState::Sustained);
    }

    pub fn sustained(&self) -> bool {
        self.key_state == KeyState::Sustained
    }

    pub fn held(&self) -> bool {
        self.key_state == KeyState::Held
    }

    pub fn released(&self) -> bool {
        self.key_state == KeyState::Released
    }

    pub fn dead(&self) -> bool {
        self.key_state == KeyState::Dead
    }

    pub fn deactivate(&mut self, sample: usize) {
        self.event_sample = Some(sample);
        self.state.event = VoiceEvent::Off;
        self.set_key_state(KeyState::Released);
    }

    pub fn kill(&mut self, sample: usize) {
        self.event_sample = Some(sample);
        self.state.event = VoiceEvent::Kill;
    }

    pub fn mark_dead(&mut self) {
        self.set_key_state(KeyState::Dead);
    }

    pub fn set_aftertouch(&mut self, value: f32, sample: usize) {
        self.aftertouch = value;
        self.aftertouch_sample = Some(sample);
    }

    pub fn set_slide(&mut self, value: f32, sample: usize) {
        self.slide = value;
        self.slide_sample = Some(sample);
    }

    /// Consumes the pending event once triggered into a block.
    pub fn complete_voice_event(&mut self) {
        self.event_sample = None;
        if self.key_state == KeyState::Triggering {
            self.set_key_state(KeyState::Held);
        }
    }
}

/// A trigger carried into a processing block: which lanes fire, with what
/// value, at which sample offset. `value` doubles as the steady
/// control-rate value for the block (reference `cr::Output` semantics).
#[derive(Clone, Copy, Debug, Default)]
pub struct Trigger {
    pub mask: PolyMask,
    pub value: PolyF32,
    pub offset: spinwave_poly::PolyU32,
}

impl Trigger {
    #[inline(always)]
    pub fn clear(&mut self) {
        self.mask = PolyMask::NONE;
        self.offset = spinwave_poly::PolyU32::ZERO;
        // `value` is intentionally kept: it is the running control value.
    }

    #[inline(always)]
    pub fn fire(&mut self, mask: PolyMask, value: PolyF32, offset: usize) {
        self.mask |= mask;
        self.value = mask.select(value, self.value);
        self.offset = mask.select_u32(spinwave_poly::PolyU32::splat(offset as u32), self.offset);
    }

    #[inline(always)]
    pub fn set_value(&mut self, mask: PolyMask, value: PolyF32) {
        self.value = mask.select(value, self.value);
    }
}

/// All control signals a voice kernel reads for one block.
#[derive(Clone, Copy, Debug, Default)]
pub struct VoiceControls {
    /// VoiceEvent as float (On/Off/Kill) with per-lane offsets.
    pub voice_event: Trigger,
    /// Fires on every articulation unless suppressed by legato.
    pub retrigger: Trigger,
    /// Fires only when the voice rises from Dead (full state reset).
    pub reset: Trigger,
    pub note: Trigger,
    pub last_note: Trigger,
    pub velocity: Trigger,
    pub lift: Trigger,
    pub aftertouch: Trigger,
    pub slide: Trigger,
    pub channel: Trigger,
    pub note_pressed: PolyF32,
    pub note_count: PolyF32,
    pub note_in_octave: PolyF32,
    /// 1.0 on lanes with a non-dead voice.
    pub active_mask: PolyF32,
    pub mod_wheel: PolyF32,
    pub pitch_wheel: PolyF32,
    pub pitch_wheel_percent: PolyF32,
    pub local_pitch_bend: PolyF32,
}
