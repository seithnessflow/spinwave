//! DAHDSR envelope with power-curved segments (port of `Envelope`).
//!
//! Stage state lives in `poly_state` as `VoiceEvent` values per lane, so two
//! voices advance independently through delay/attack/hold/decay/release/kill.

use spinwave_poly::constants::{VoiceEvent, VOICE_KILL_TIME};
use spinwave_poly::{math, utils, PolyF32, PolyMask, PolyU32};

/// Per-block envelope parameters (times in seconds, powers unitless).
#[derive(Clone, Copy, Debug, Default)]
pub struct EnvelopeParams {
    pub delay: PolyF32,
    pub attack: PolyF32,
    pub attack_power: PolyF32,
    pub hold: PolyF32,
    pub decay: PolyF32,
    pub decay_power: PolyF32,
    pub sustain: PolyF32,
    pub release: PolyF32,
    pub release_power: PolyF32,
}

#[derive(Clone)]
pub struct Envelope {
    sample_rate: f32,

    position: PolyF32,
    value: PolyF32,
    poly_state: PolyF32,
    start_value: PolyF32,

    attack_power: PolyF32,
    decay_power: PolyF32,
    release_power: PolyF32,
    sustain: PolyF32,

    trigger_mask: PolyMask,
    trigger_value: PolyF32,
    trigger_offset: PolyU32,
}

impl Envelope {
    pub fn new(sample_rate: f32) -> Self {
        Envelope {
            sample_rate,
            position: PolyF32::ZERO,
            value: PolyF32::ZERO,
            poly_state: PolyF32::ZERO,
            start_value: PolyF32::ZERO,
            attack_power: PolyF32::ZERO,
            decay_power: PolyF32::ZERO,
            release_power: PolyF32::ZERO,
            sustain: PolyF32::ZERO,
            trigger_mask: PolyMask::NONE,
            trigger_value: PolyF32::ZERO,
            trigger_offset: PolyU32::ZERO,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Queues a voice event for the next process call: `value` is a
    /// `VoiceEvent` as float (`On` starts the envelope, `Off` releases,
    /// `Kill` fast-fades), applied to lanes in `mask` at `sample_offset`.
    pub fn trigger(&mut self, mask: PolyMask, value: PolyF32, sample_offset: usize) {
        self.trigger_mask = mask;
        self.trigger_value = value;
        self.trigger_offset = PolyU32::splat(sample_offset as u32);
    }

    /// Current envelope output.
    #[inline]
    pub fn value(&self) -> PolyF32 {
        self.value
    }

    /// Stage + intra-stage position, matching the C++ phase output.
    #[inline]
    pub fn phase(&self) -> PolyF32 {
        self.poly_state + self.position
    }

    fn take_trigger(&mut self) -> (PolyMask, PolyF32, PolyU32) {
        let trigger = (self.trigger_mask, self.trigger_value, self.trigger_offset);
        self.trigger_mask = PolyMask::NONE;
        self.trigger_value = PolyF32::ZERO;
        self.trigger_offset = PolyU32::ZERO;
        trigger
    }

    /// Control-rate tick over `num_samples` samples; returns the new value.
    pub fn process_control(&mut self, params: &EnvelopeParams, num_samples: usize) -> PolyF32 {
        let (trigger_mask, mut trigger_value, trigger_offset) = self.take_trigger();

        let delay_time = params.delay.max(PolyF32::ZERO);
        let has_delay_mask = delay_time.ne(PolyF32::ZERO);
        let note_on_mask = trigger_value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
        trigger_value = (has_delay_mask & note_on_mask)
            .select(PolyF32::splat(VoiceEvent::Idle.as_f32()), trigger_value);

        self.poly_state = trigger_mask.select(trigger_value, self.poly_state);
        self.position = trigger_mask.select(PolyF32::ZERO, self.position);

        let triggered_remaining = PolyU32::splat(num_samples as u32) - trigger_offset;
        let remaining_samples =
            trigger_mask.select_u32(triggered_remaining, PolyU32::splat(num_samples as u32));
        self.start_value = trigger_mask.select(self.value, self.start_value);

        let state = self.poly_state;
        let delay_mask = state.eq(PolyF32::splat(VoiceEvent::Idle.as_f32()));
        let attack_mask = state.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
        let hold_mask = state.eq(PolyF32::splat(VoiceEvent::Hold.as_f32()));
        let decay_mask = state.eq(PolyF32::splat(VoiceEvent::Decay.as_f32()));
        let release_mask = state.eq(PolyF32::splat(VoiceEvent::Off.as_f32()));
        let kill_mask = state.eq(PolyF32::splat(VoiceEvent::Kill.as_f32()));

        let delta_time = remaining_samples.to_f32_signed() * (1.0 / self.sample_rate);

        let delta_delay = delta_time / delay_time.max(PolyF32::splat(0.0000001));
        self.position += delta_delay & delay_mask;

        let attack_time = params.attack.max(PolyF32::splat(0.000000001));
        let delta_attack = delta_time / attack_time;
        self.position += delta_attack & attack_mask;

        let hold_time = params.hold.max(PolyF32::ZERO);
        let has_hold_mask = hold_time.ne(PolyF32::ZERO);
        let delta_hold = delta_time / hold_time.max(PolyF32::splat(0.0000001));
        self.position += delta_hold & hold_mask;

        let decay_time = params.decay.max(PolyF32::splat(0.000000001));
        let delta_decay = delta_time / decay_time;
        self.position += delta_decay & decay_mask;

        let release_time = params.release.max(PolyF32::splat(0.000000001));
        let delta_release = delta_time / release_time;
        self.position += delta_release & release_mask;

        let delta_kill = delta_time * (1.0 / VOICE_KILL_TIME);
        self.position += delta_kill & kill_mask;
        self.position = self.position.clamp(0.0, 1.0);

        let power = (-params.attack_power & attack_mask)
            + (params.decay_power & decay_mask)
            + (params.release_power & release_mask);

        let attack_value = math::power_scale(self.position, power);

        let sustain = params.sustain;
        let decay_value = PolyF32::ONE - (PolyF32::ONE - sustain) * attack_value;
        let release_value = self.start_value * (PolyF32::ONE - attack_value);
        let kill_value = self.start_value * (PolyF32::ONE - attack_value);

        self.value = (attack_value & attack_mask)
            + (PolyF32::ONE & hold_mask)
            + (decay_value & decay_mask)
            + (release_value & release_mask)
            + (kill_value & kill_mask);
        self.value = self.value.clamp(0.0, 1.0);

        let at_end = self.position.eq(PolyF32::ONE);
        let attack_transition_mask = delay_mask & at_end;
        let hold_transition_mask = attack_mask & at_end & has_hold_mask;
        let decay_turn_mask = (attack_mask & !has_hold_mask) | hold_mask;
        let decay_transition_mask = decay_turn_mask & at_end;
        self.poly_state = attack_transition_mask
            .select(PolyF32::splat(VoiceEvent::On.as_f32()), self.poly_state);
        self.poly_state = hold_transition_mask
            .select(PolyF32::splat(VoiceEvent::Hold.as_f32()), self.poly_state);
        self.poly_state = decay_transition_mask
            .select(PolyF32::splat(VoiceEvent::Decay.as_f32()), self.poly_state);

        let transition_mask = attack_transition_mask | hold_transition_mask | decay_transition_mask;
        self.position = self.position & !transition_mask;

        let dead_transition_mask = release_mask & at_end;
        self.poly_state = dead_transition_mask
            .select(PolyF32::splat(VoiceEvent::Kill.as_f32()), self.poly_state);

        self.value
    }

    #[allow(clippy::too_many_arguments)]
    fn process_section(
        &self,
        audio_out: &mut [PolyF32],
        from: usize,
        to: usize,
        power: PolyF32,
        delta_power: PolyF32,
        position: PolyF32,
        delta_position: PolyF32,
        start: PolyF32,
        end: PolyF32,
        delta_end: PolyF32,
    ) -> PolyF32 {
        let num_samples = (to - from) as f32;

        let mut current_power = power;
        let mut current_position = position;
        let mut current_end = end;
        for out in audio_out.iter_mut().take(to).skip(from) {
            let t = math::power_scale(current_position, current_power);
            *out = utils::interpolate(start, current_end, t);

            current_power += delta_power;
            current_position = (current_position + delta_position).clamp(0.0, 1.0);
            current_end += delta_end;
        }

        (position + delta_position * num_samples).clamp(0.0, 1.0)
    }

    /// Audio-rate processing: fills `out` with per-sample envelope values.
    /// Parameter changes and the section powers are interpolated across the
    /// block for click-free automation, exactly like the reference.
    pub fn process_audio(&mut self, params: &EnvelopeParams, out: &mut [PolyF32]) {
        let num_samples = out.len();
        let delta_time = PolyF32::splat(1.0 / self.sample_rate);
        let delta_sample = 1.0 / num_samples as f32;

        let sustain_end = params.sustain.clamp(0.0, 1.0);

        let delay_time = params.delay.max(PolyF32::ZERO);
        let delta_delay = delta_time / delay_time.max(PolyF32::splat(0.0000001));

        let attack_time = params.attack.max(PolyF32::splat(0.000000001));
        let delta_attack = delta_time / attack_time;
        let attack_power_end = -params.attack_power;

        let hold_time = params.hold.max(PolyF32::ZERO);
        let has_hold_mask = hold_time.ne(PolyF32::ZERO);
        let delta_hold = delta_time / hold_time.max(PolyF32::splat(0.0000001));

        let decay_time = params.decay.max(PolyF32::splat(0.000000001));
        let delta_decay = delta_time / decay_time;
        let decay_power_end = params.decay_power;

        let release_time = params.release.max(PolyF32::splat(0.000000001));
        let delta_release = delta_time / release_time;
        let release_power_end = params.release_power;

        let delta_kill = delta_time * (1.0 / VOICE_KILL_TIME);

        let (trigger_mask, mut trigger_value, trigger_offset) = self.take_trigger();
        let has_delay_mask = delay_time.ne(PolyF32::ZERO);
        let note_on_mask = trigger_value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
        trigger_value = (has_delay_mask & note_on_mask)
            .select(PolyF32::splat(VoiceEvent::Idle.as_f32()), trigger_value);

        let mut triggered_position =
            trigger_mask.select_u32(trigger_offset, PolyU32::splat(num_samples as u32));

        let mut current_position = self.position;

        let mut i = 0usize;
        while i < num_samples {
            let triggering = trigger_mask & PolyU32::splat(i as u32).eq(triggered_position);
            triggered_position =
                triggering.select_u32(PolyU32::splat(num_samples as u32), triggered_position);
            self.poly_state = triggering.select(trigger_value, self.poly_state);
            current_position = triggering.select(PolyF32::ZERO, current_position);

            self.start_value = triggering.select(self.value, self.start_value);
            self.attack_power = triggering.select(attack_power_end, self.attack_power);
            self.decay_power = triggering.select(decay_power_end, self.decay_power);
            self.release_power = triggering.select(release_power_end, self.release_power);
            self.sustain = triggering.select(sustain_end, self.sustain);

            let state = self.poly_state;
            let delay_mask = state.eq(PolyF32::splat(VoiceEvent::Idle.as_f32()));
            let attack_mask = state.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
            let hold_mask = state.eq(PolyF32::splat(VoiceEvent::Hold.as_f32()));
            let decay_mask = state.eq(PolyF32::splat(VoiceEvent::Decay.as_f32()));
            let release_mask = state.eq(PolyF32::splat(VoiceEvent::Off.as_f32()));
            let kill_mask = state.eq(PolyF32::splat(VoiceEvent::Kill.as_f32()));

            let delta_position = (delta_delay & delay_mask)
                + (delta_attack & attack_mask)
                + (delta_hold & hold_mask)
                + (delta_decay & decay_mask)
                + (delta_release & release_mask)
                + (delta_kill & kill_mask);

            let from_power = (self.attack_power & attack_mask)
                + (self.decay_power & decay_mask)
                + (self.release_power & release_mask);
            let to_power = (attack_power_end & attack_mask)
                + (decay_power_end & decay_mask)
                + (release_power_end & release_mask);

            let block_t = PolyF32::splat(i as f32 / num_samples as f32);
            let power = utils::interpolate(from_power, to_power, block_t);
            let delta_power = (to_power - from_power) * delta_sample;

            let cycles_remaining =
                (current_position.ceil() - current_position) / delta_position;
            let end_cycle = attack_mask
                .select(cycles_remaining + PolyF32::splat(i as f32), PolyF32::splat(num_samples as f32));
            let end_cycle = end_cycle.min(triggered_position.to_f32_signed());
            let last_cycle = (utils::min_lane(end_cycle) as usize).max(i + 1);

            let current_sustain = utils::interpolate(self.sustain, sustain_end, block_t);
            let start = (decay_mask | hold_mask).select(PolyF32::ONE, self.start_value);
            let end =
                (PolyF32::ONE & (attack_mask | hold_mask)) + (current_sustain & decay_mask);
            let delta_end = ((sustain_end - self.sustain) * delta_sample) & decay_mask;

            current_position = self.process_section(
                out,
                i,
                last_cycle,
                power,
                delta_power,
                current_position,
                delta_position,
                start,
                end,
                delta_end,
            );
            i = last_cycle;

            self.value = out[i - 1];

            let at_end = current_position.eq(PolyF32::ONE);
            let attack_transition_mask = delay_mask & at_end;
            let hold_transition_mask = attack_mask & at_end & has_hold_mask;
            let decay_turn_mask = (attack_mask & !has_hold_mask) | hold_mask;
            let decay_transition_mask = decay_turn_mask & at_end;

            self.poly_state = attack_transition_mask
                .select(PolyF32::splat(VoiceEvent::On.as_f32()), self.poly_state);
            self.poly_state = hold_transition_mask
                .select(PolyF32::splat(VoiceEvent::Hold.as_f32()), self.poly_state);
            self.poly_state = decay_transition_mask
                .select(PolyF32::splat(VoiceEvent::Decay.as_f32()), self.poly_state);

            let transition_mask =
                attack_transition_mask | hold_transition_mask | decay_transition_mask;
            current_position = current_position & !transition_mask;

            let dead_transition_mask = release_mask & at_end;
            self.poly_state = dead_transition_mask
                .select(PolyF32::splat(VoiceEvent::Kill.as_f32()), self.poly_state);
        }

        self.position = current_position;
        self.attack_power = attack_power_end;
        self.decay_power = decay_power_end;
        self.release_power = release_power_end;
        self.sustain = sustain_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spinwave_poly::PolyMask;

    const SAMPLE_RATE: f32 = 44100.0;

    fn params() -> EnvelopeParams {
        EnvelopeParams {
            delay: PolyF32::ZERO,
            attack: PolyF32::splat(0.01),
            attack_power: PolyF32::ZERO,
            hold: PolyF32::ZERO,
            decay: PolyF32::splat(0.05),
            decay_power: PolyF32::ZERO,
            sustain: PolyF32::splat(0.5),
            release: PolyF32::splat(0.02),
            release_power: PolyF32::ZERO,
        }
    }

    fn on() -> PolyF32 {
        PolyF32::splat(VoiceEvent::On.as_f32())
    }

    fn off() -> PolyF32 {
        PolyF32::splat(VoiceEvent::Off.as_f32())
    }

    #[test]
    fn stage_progression() {
        let mut envelope = Envelope::new(SAMPLE_RATE);
        let params = params();
        envelope.trigger(PolyMask::all_on(), on(), 0);

        // Attack: value rises toward 1 over ~441 samples (0.01s).
        let mut last = 0.0f32;
        let mut rising = true;
        for _ in 0..25 {
            let value = envelope.process_control(&params, 16).lane(0);
            rising &= value >= last - 1e-6;
            last = value;
        }
        assert!(rising, "attack should be monotonic");
        assert!(last > 0.85, "attack should approach 1, got {last}");

        // Decay: falls to sustain level.
        for _ in 0..300 {
            envelope.process_control(&params, 16);
        }
        let sustained = envelope.value().lane(0);
        assert!((sustained - 0.5).abs() < 0.01, "sustain was {sustained}");
        assert!(envelope.phase().lane(0) >= VoiceEvent::Decay.as_f32());
    }

    #[test]
    fn release_to_zero() {
        let mut envelope = Envelope::new(SAMPLE_RATE);
        let params = params();
        envelope.trigger(PolyMask::all_on(), on(), 0);
        for _ in 0..400 {
            envelope.process_control(&params, 16);
        }

        envelope.trigger(PolyMask::all_on(), off(), 0);
        let mut last = envelope.process_control(&params, 16).lane(0);
        for _ in 0..200 {
            let value = envelope.process_control(&params, 16).lane(0);
            assert!(value <= last + 1e-6, "release should be monotonic");
            last = value;
        }
        assert!(last.abs() < 1e-6, "release should reach zero, got {last}");
    }

    #[test]
    fn delay_defers_attack() {
        let mut envelope = Envelope::new(SAMPLE_RATE);
        let mut delayed = params();
        delayed.delay = PolyF32::splat(0.05);
        envelope.trigger(PolyMask::all_on(), on(), 0);

        // During the delay the envelope stays at zero (Idle state).
        for _ in 0..30 {
            let value = envelope.process_control(&delayed, 16);
            assert_eq!(value.lane(0), 0.0);
        }
        // After 0.05s it starts the attack.
        for _ in 0..200 {
            envelope.process_control(&delayed, 16);
        }
        assert!(envelope.value().lane(0) > 0.1);
    }

    #[test]
    fn trigger_masks_are_per_lane() {
        let mut envelope = Envelope::new(SAMPLE_RATE);
        let params = params();
        // Only voice 0 (lanes 0, 1) triggers.
        let voice0 = PolyF32::from_lanes([1.0, 1.0, 0.0, 0.0]).ne(PolyF32::ZERO);
        envelope.trigger(voice0, on(), 0);
        for _ in 0..40 {
            envelope.process_control(&params, 16);
        }
        assert!(envelope.value().lane(0) > 0.1);
        assert_eq!(envelope.value().lane(2), 0.0);
    }

    #[test]
    fn audio_rate_matches_control_direction() {
        let mut envelope = Envelope::new(SAMPLE_RATE);
        let params = params();
        envelope.trigger(PolyMask::all_on(), on(), 0);

        let mut out = [PolyF32::ZERO; 64];
        envelope.process_audio(&params, &mut out);
        // Rising attack from zero.
        assert!(out[0].lane(0) < out[63].lane(0));
        assert!(out[63].lane(0) > 0.0);
        for window in out.windows(2) {
            assert!(window[1].lane(0) >= window[0].lane(0) - 1e-6);
        }
    }

    #[test]
    fn audio_rate_trigger_offset() {
        let mut envelope = Envelope::new(SAMPLE_RATE);
        let params = params();
        envelope.trigger(PolyMask::all_on(), on(), 32);

        let mut out = [PolyF32::ZERO; 64];
        envelope.process_audio(&params, &mut out);
        // Nothing before the offset, envelope starts after.
        assert_eq!(out[16].lane(0), 0.0);
        assert!(out[63].lane(0) > 0.0);
    }
}
