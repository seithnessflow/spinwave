//! Per-note random value source (port of `TriggerRandom`).
//!
//! Draws one uniform `[0, 1)` value per voice on every note-on trigger and
//! holds it until the next trigger.

use spinwave_poly::constants::VoiceEvent;
use spinwave_poly::{PolyF32, PolyMask, PolyU32, LANES};

use super::random::RandomGenerator;

#[derive(Clone)]
pub struct TriggerRandom {
    value: PolyF32,
    random_generator: RandomGenerator,
}

impl Default for TriggerRandom {
    fn default() -> Self {
        Self::new()
    }
}

impl TriggerRandom {
    pub fn new() -> Self {
        TriggerRandom { value: PolyF32::ZERO, random_generator: RandomGenerator::new(0.0, 1.0) }
    }

    /// Deterministic construction for tests/replays.
    pub fn with_seed(seed: u32) -> Self {
        TriggerRandom {
            value: PolyF32::ZERO,
            random_generator: RandomGenerator::with_seed(0.0, 1.0, seed),
        }
    }

    /// Note event: draws a new value for voices triggering `VoiceEvent::On`.
    pub fn trigger(&mut self, mask: PolyMask, value: PolyF32, _sample_offset: usize) {
        self.trigger_at(mask, value, PolyU32::ZERO);
    }

    /// Per-lane-offset form of [`Self::trigger`] (same signature family as
    /// the other modulators; the value is drawn immediately, so the offsets
    /// are irrelevant here).
    pub fn trigger_at(&mut self, mask: PolyMask, value: PolyF32, _sample_offsets: PolyU32) {
        let trigger_mask = mask & value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
        if !trigger_mask.any() {
            return;
        }
        let lanes = trigger_mask.to_u32().0;
        for voice in 0..LANES / 2 {
            if lanes[voice * 2] != 0 {
                let rand_value = self.random_generator.next();
                self.value.set_lane(voice * 2, rand_value);
                self.value.set_lane(voice * 2 + 1, rand_value);
            }
        }
    }

    /// Control-rate output: the held per-voice random values.
    #[inline]
    pub fn process_control(&self) -> PolyF32 {
        self.value
    }

    #[inline]
    pub fn value(&self) -> PolyF32 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> PolyF32 {
        PolyF32::splat(VoiceEvent::On.as_f32())
    }

    #[test]
    fn holds_value_between_triggers() {
        let mut trigger_random = TriggerRandom::with_seed(1);
        trigger_random.trigger(PolyMask::all_on(), on(), 0);
        let first = trigger_random.process_control();
        assert_eq!(first.to_lanes(), trigger_random.process_control().to_lanes());

        trigger_random.trigger(PolyMask::all_on(), on(), 0);
        let second = trigger_random.process_control();
        assert_ne!(first.to_lanes(), second.to_lanes());
    }

    #[test]
    fn stereo_lanes_share_voice_value() {
        let mut trigger_random = TriggerRandom::with_seed(2);
        trigger_random.trigger(PolyMask::all_on(), on(), 0);
        let value = trigger_random.value();
        assert_eq!(value.lane(0), value.lane(1));
        assert_eq!(value.lane(2), value.lane(3));
        assert_ne!(value.lane(0), value.lane(2));
    }

    #[test]
    fn only_triggered_voice_changes() {
        let mut trigger_random = TriggerRandom::with_seed(3);
        trigger_random.trigger(PolyMask::all_on(), on(), 0);
        let before = trigger_random.value();

        let voice1_mask =
            PolyMask::from_u32(PolyU32::from_lanes([0, 0, u32::MAX, u32::MAX]));
        trigger_random.trigger(voice1_mask, on(), 0);
        let after = trigger_random.value();
        assert_eq!(before.lane(0), after.lane(0));
        assert_ne!(before.lane(2), after.lane(2));
    }

    #[test]
    fn release_does_not_redraw() {
        let mut trigger_random = TriggerRandom::with_seed(4);
        trigger_random.trigger(PolyMask::all_on(), on(), 0);
        let before = trigger_random.value();
        trigger_random.trigger(
            PolyMask::all_on(),
            PolyF32::splat(VoiceEvent::Off.as_f32()),
            0,
        );
        assert_eq!(before.to_lanes(), trigger_random.value().to_lanes());
    }
}
