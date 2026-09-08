//! Legato note filtering (port of `LegatoFilter`).
//!
//! Passes note triggers through, suppressing retriggers when legato is on
//! and a voice is already sounding (note-on while note-on).

use spinwave_poly::constants::VoiceEvent;
use spinwave_poly::{PolyF32, PolyMask};

/// A (possibly filtered) trigger event to forward downstream.
#[derive(Clone, Copy, Debug)]
pub struct TriggerEvent {
    pub mask: PolyMask,
    pub value: PolyF32,
    pub sample_offset: usize,
}

impl TriggerEvent {
    pub fn none() -> Self {
        TriggerEvent { mask: PolyMask::NONE, value: PolyF32::ZERO, sample_offset: 0 }
    }
}

#[derive(Clone)]
pub struct LegatoFilter {
    last_value: PolyF32,
}

impl Default for LegatoFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl LegatoFilter {
    pub fn new() -> Self {
        LegatoFilter { last_value: PolyF32::splat(VoiceEvent::Off.as_f32()) }
    }

    /// Filters a note trigger. With `legato` on, a note-on landing on a
    /// still-sounding voice is swallowed; note-offs always pass.
    pub fn process_trigger(
        &mut self,
        legato: bool,
        mask: PolyMask,
        value: PolyF32,
        sample_offset: usize,
    ) -> TriggerEvent {
        if !mask.any() {
            return TriggerEvent::none();
        }

        let on = PolyF32::splat(VoiceEvent::On.as_f32());
        let mut legato_mask = if legato { PolyMask::NONE } else { PolyMask::all_on() };
        legato_mask |= value.ne(on);
        legato_mask |= self.last_value.ne(on);
        let trigger_mask = mask & legato_mask;

        self.last_value = trigger_mask.select(value, self.last_value);
        TriggerEvent { mask: trigger_mask, value, sample_offset }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> PolyF32 {
        PolyF32::splat(VoiceEvent::On.as_f32())
    }

    fn off() -> PolyF32 {
        PolyF32::splat(VoiceEvent::Off.as_f32())
    }

    #[test]
    fn passes_first_note_on() {
        let mut filter = LegatoFilter::new();
        let event = filter.process_trigger(true, PolyMask::all_on(), on(), 3);
        assert!(event.mask.all());
        assert_eq!(event.sample_offset, 3);
    }

    #[test]
    fn legato_swallows_retrigger() {
        let mut filter = LegatoFilter::new();
        filter.process_trigger(true, PolyMask::all_on(), on(), 0);
        // Second note-on while sounding: filtered out.
        let event = filter.process_trigger(true, PolyMask::all_on(), on(), 0);
        assert!(!event.mask.any());
        // Note-off always passes.
        let event = filter.process_trigger(true, PolyMask::all_on(), off(), 0);
        assert!(event.mask.all());
        // After the off, a new note-on passes again.
        let event = filter.process_trigger(true, PolyMask::all_on(), on(), 0);
        assert!(event.mask.all());
    }

    #[test]
    fn non_legato_always_retriggers() {
        let mut filter = LegatoFilter::new();
        filter.process_trigger(false, PolyMask::all_on(), on(), 0);
        let event = filter.process_trigger(false, PolyMask::all_on(), on(), 0);
        assert!(event.mask.all());
    }
}
