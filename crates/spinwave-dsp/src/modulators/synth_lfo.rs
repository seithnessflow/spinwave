//! LFO reading a [`LineGenerator`] shape (port of `SynthLfo`).
//!
//! Supports six sync modes (free/host-synced/one-shot envelope variants and
//! loop-point styles), delay and fade-in, per-voice stereo phase offset and
//! an exponential smoothing mode. Control-rate and audio-rate paths keep
//! separate state, both updated like the reference so switching rates is
//! seamless.

use spinwave_poly::constants::VoiceEvent;
use spinwave_poly::{math, utils, PolyF32, PolyMask, PolyU32};

use super::line_generator::LineGenerator;

pub const HALF_LIFE_RATIO: f32 = 0.2;
pub const MIN_HALF_LIFE: f32 = 0.0002;

/// How the LFO phase restarts and advances.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoSyncType {
    /// Restart phase on every note trigger.
    #[default]
    Trigger,
    /// Phase follows host transport time.
    Sync,
    /// One-shot: play the shape once and stop at the end.
    Envelope,
    /// Play up to the loop point, hold while the note is held, finish on
    /// release.
    SustainEnvelope,
    /// Loop back to the loop point after each full cycle.
    LoopPoint,
    /// Loop before the loop point while held, play out after release.
    LoopHold,
}

/// Per-block parameters. `phase` is the shape phase offset (also the loop
/// point for the loop modes), `stereo_phase` splits left/right phases.
#[derive(Clone, Copy, Debug, Default)]
pub struct SynthLfoParams {
    pub frequency: PolyF32,
    pub phase: PolyF32,
    pub stereo_phase: PolyF32,
    pub sync_type: LfoSyncType,
    pub smooth_mode: bool,
    pub fade_time: PolyF32,
    pub smooth_time: PolyF32,
    pub delay_time: PolyF32,
}

#[derive(Clone, Copy, Debug, Default)]
struct LfoState {
    delay_time_passed: PolyF32,
    fade_amplitude: PolyF32,
    smooth_value: PolyF32,
    offset: PolyF32,
    phase: PolyF32,
}

/// Catmull-Rom lookup into a line generator's guarded buffer.
pub fn value_at_phase(source: &LineGenerator, phase: PolyF32) -> PolyF32 {
    let resolution = source.resolution();
    let buffer = source.cubic_interpolation_buffer();
    let resolution_f = resolution as f32;

    let boost = (phase * resolution_f).clamp(0.0, resolution_f);
    let indices = boost.to_i32_round().min(PolyU32::splat(resolution as u32 - 1));
    let t = boost - indices.to_f32_signed();

    let interpolation_matrix = utils::catmull_interpolation_matrix(t);
    let mut values = utils::value_matrix(buffer, indices);
    values.transpose();
    interpolation_matrix.multiply_and_sum_rows(&values)
}

struct AudioSetup {
    delta_phase: PolyF32,
    delay_time: PolyF32,
    tick_time: PolyF32,
    fade_increase: PolyF32,
    smooth_mult: PolyF32,
    current_amplitude: PolyF32,
    delay_time_passed: PolyF32,
    current_value: PolyF32,
}

#[derive(Clone)]
pub struct SynthLfo {
    sample_rate: f32,

    was_control_rate: bool,
    control_rate_state: LfoState,
    audio_rate_state: LfoState,

    held_mask: PolyMask,
    trigger_sample: PolyU32,
    trigger_delay: PolyF32,

    sync_seconds: f64,

    trigger_mask: PolyMask,
    trigger_value: PolyF32,
    trigger_offset: PolyU32,
    /// Reset mask of the current block, consumed by the audio path.
    block_reset_mask: PolyMask,

    last_phase: PolyF32,
    last_value: PolyF32,
}

impl SynthLfo {
    pub fn new(sample_rate: f32) -> Self {
        SynthLfo {
            sample_rate,
            was_control_rate: true,
            control_rate_state: LfoState::default(),
            audio_rate_state: LfoState::default(),
            held_mask: PolyMask::NONE,
            trigger_sample: PolyU32::ZERO,
            trigger_delay: PolyF32::ZERO,
            sync_seconds: 0.0,
            trigger_mask: PolyMask::NONE,
            trigger_value: PolyF32::ZERO,
            trigger_offset: PolyU32::ZERO,
            block_reset_mask: PolyMask::NONE,
            last_phase: PolyF32::ZERO,
            last_value: PolyF32::ZERO,
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    /// Sets the host transport time used by [`LfoSyncType::Sync`].
    pub fn correct_to_time(&mut self, seconds: f64) {
        self.sync_seconds = seconds;
    }

    /// Queues a note event (`VoiceEvent::On` retriggers, `Off` releases)
    /// applied on the next process call at `sample_offset`.
    pub fn trigger(&mut self, mask: PolyMask, value: PolyF32, sample_offset: usize) {
        self.trigger_mask = mask;
        self.trigger_value = value;
        self.trigger_offset = PolyU32::splat(sample_offset as u32);
    }

    /// Phase used for the last output (for UI/oscillator sync feedback).
    pub fn phase(&self) -> PolyF32 {
        self.last_phase
    }

    pub fn value(&self) -> PolyF32 {
        self.last_value
    }

    fn process_trigger(&mut self, params: &SynthLfoParams) {
        let trigger_mask = self.trigger_mask;
        let trigger_value = self.trigger_value;
        let trigger_offset = self.trigger_offset;
        self.trigger_mask = PolyMask::NONE;
        self.trigger_value = PolyF32::ZERO;
        self.trigger_offset = PolyU32::ZERO;

        let reset_mask = trigger_mask & trigger_value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));
        let release_mask =
            trigger_mask & trigger_value.eq(PolyF32::splat(VoiceEvent::Off.as_f32()));
        self.block_reset_mask = reset_mask;
        self.held_mask = (self.held_mask | reset_mask) & !release_mask;
        self.trigger_sample =
            (reset_mask | release_mask).select_u32(trigger_offset, self.trigger_sample);

        self.control_rate_state.delay_time_passed =
            reset_mask.select(PolyF32::ZERO, self.control_rate_state.delay_time_passed);
        self.control_rate_state.fade_amplitude =
            reset_mask.select(PolyF32::ZERO, self.control_rate_state.fade_amplitude);
        self.control_rate_state.smooth_value =
            reset_mask.select(PolyF32::ZERO, self.control_rate_state.smooth_value);
        self.audio_rate_state.delay_time_passed =
            reset_mask.select(PolyF32::ZERO, self.audio_rate_state.delay_time_passed);
        self.audio_rate_state.fade_amplitude =
            reset_mask.select(PolyF32::ZERO, self.audio_rate_state.fade_amplitude);
        self.audio_rate_state.smooth_value =
            reset_mask.select(PolyF32::ZERO, self.audio_rate_state.smooth_value);

        let trigger_delay = trigger_offset.to_f32_signed() * (1.0 / self.sample_rate);
        self.trigger_delay = reset_mask.select(trigger_delay, self.trigger_delay);

        if reset_mask.any() {
            let frequency = params.frequency;

            if params.sync_type == LfoSyncType::Sync {
                let sync_phase = utils::cycle_offset_from_seconds(self.sync_seconds, frequency);
                self.control_rate_state.offset =
                    reset_mask.select(sync_phase, self.control_rate_state.offset);
                self.audio_rate_state.offset =
                    reset_mask.select(sync_phase, self.audio_rate_state.offset);
            } else {
                self.control_rate_state.offset =
                    reset_mask.select(PolyF32::ZERO, self.control_rate_state.offset);

                let sample_offset = trigger_offset.to_f32_signed() & reset_mask;
                let offset_start = frequency * sample_offset * (1.0 / self.sample_rate);
                self.audio_rate_state.offset =
                    reset_mask.select(-offset_start, self.audio_rate_state.offset);
            }
        }
    }

    /// Control-rate tick covering `num_samples` samples.
    pub fn process_control(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        num_samples: usize,
    ) -> PolyF32 {
        self.was_control_rate = true;
        self.process_trigger(params);
        self.control_step(source, params, num_samples, true)
    }

    fn control_step(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        num_samples: usize,
        store_output: bool,
    ) -> PolyF32 {
        let delay_time = params.delay_time;

        let tick_time = 1.0 / self.sample_rate;
        let mut time_passed = PolyF32::splat(tick_time * num_samples as f32);
        self.control_rate_state.delay_time_passed += time_passed;
        time_passed = (self.control_rate_state.delay_time_passed - delay_time)
            .max(PolyF32::ZERO)
            .min(time_passed);

        let stereo_phase = params.stereo_phase;
        let phase = params.phase + stereo_phase * PolyF32::stereo(0.5, -0.5);
        let frequency = params.frequency;
        let current_offset = self.control_rate_state.offset;
        self.control_rate_state.offset += frequency * time_passed;

        let phased_offset;
        match params.sync_type {
            LfoSyncType::Envelope => {
                self.control_rate_state.offset = self.control_rate_state.offset.min(PolyF32::ONE);
                phased_offset = (current_offset + phase).min(PolyF32::ONE);
            }
            LfoSyncType::SustainEnvelope => {
                let held_limit = self.held_mask.select(phase, PolyF32::ONE);
                self.control_rate_state.offset = self.control_rate_state.offset.min(held_limit);
                phased_offset = current_offset;
            }
            LfoSyncType::Trigger | LfoSyncType::Sync => {
                self.control_rate_state.offset = self.control_rate_state.offset.fract();
                phased_offset = (current_offset + phase).fract();
            }
            LfoSyncType::LoopPoint => {
                let over = self.control_rate_state.offset.ge(PolyF32::ONE);
                self.control_rate_state.offset = over.select(
                    self.control_rate_state.offset - PolyF32::ONE + phase,
                    self.control_rate_state.offset,
                );
                phased_offset = current_offset.min(PolyF32::ONE);
            }
            LfoSyncType::LoopHold => {
                let over = self.held_mask & self.control_rate_state.offset.ge(phase);
                self.control_rate_state.offset = over
                    .select(
                        self.control_rate_state.offset - phase,
                        self.control_rate_state.offset,
                    )
                    .min(PolyF32::ONE);
                phased_offset =
                    self.held_mask.select(current_offset.min(phase), current_offset);
            }
        }

        let fade_time = params.fade_time;
        let fade_increase =
            time_passed / time_passed.max(PolyF32::splat(tick_time)).max(fade_time);
        self.control_rate_state.fade_amplitude =
            (self.control_rate_state.fade_amplitude + fade_increase).min(PolyF32::ONE);
        self.control_rate_state.fade_amplitude = fade_time
            .eq(PolyF32::ZERO)
            .select(PolyF32::ONE, self.control_rate_state.fade_amplitude);

        let value = value_at_phase(source, phased_offset);
        let result = if params.smooth_mode {
            let half_life = params.smooth_time * HALF_LIFE_RATIO;
            let smooth_mask = half_life.gt(PolyF32::splat(MIN_HALF_LIFE));
            let exponent = -time_passed / half_life.max(PolyF32::splat(MIN_HALF_LIFE));
            let ratio = math::exp2(exponent) & smooth_mask;
            let result = utils::interpolate(value, self.control_rate_state.smooth_value, ratio);
            self.control_rate_state.smooth_value = result;
            result
        } else {
            let start_value = value_at_phase(source, phase);
            utils::interpolate(start_value, value, self.control_rate_state.fade_amplitude)
        };

        let result = result.clamp(-1.0, 1.0);
        if store_output {
            self.last_value = result;
            self.last_phase = phased_offset;
        }
        result
    }

    /// Audio-rate processing filling `out` with one value per sample. Also
    /// ticks the control-rate state so switching rates stays consistent.
    pub fn process_audio(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        out: &mut [PolyF32],
    ) {
        if self.was_control_rate {
            self.audio_rate_state = self.control_rate_state;
        }
        self.was_control_rate = false;

        self.process_trigger(params);
        self.audio_step(source, params, out);
        self.control_step(source, params, out.len(), false);
    }

    fn audio_setup(&self, params: &SynthLfoParams, current_phase: PolyF32, num_samples: usize) -> AudioSetup {
        let tick_time = PolyF32::splat(1.0 / self.sample_rate);
        let fade_time = params.fade_time;
        let delay_time = params.delay_time + self.trigger_delay;
        let mut current_amplitude = self.audio_rate_state.fade_amplitude;
        let fade_increase = tick_time / tick_time.max(fade_time);

        let mut smooth_mult = PolyF32::ZERO;
        if params.smooth_mode {
            let half_life = params.smooth_time * HALF_LIFE_RATIO;
            let smooth_mask = half_life.gt(PolyF32::splat(MIN_HALF_LIFE));
            let exponent = -tick_time / half_life.max(PolyF32::splat(MIN_HALF_LIFE));
            smooth_mult = math::exp2(exponent) & smooth_mask;
            current_amplitude = PolyF32::ONE;
        }

        AudioSetup {
            delta_phase: (self.audio_rate_state.phase - current_phase)
                * (1.0 / num_samples as f32),
            delay_time,
            tick_time,
            fade_increase,
            smooth_mult,
            current_amplitude,
            delay_time_passed: self.audio_rate_state.delay_time_passed,
            current_value: self.audio_rate_state.smooth_value,
        }
    }

    fn audio_step(&mut self, source: &LineGenerator, params: &SynthLfoParams, out: &mut [PolyF32]) {
        let num_samples = out.len();
        let stereo_phase = params.stereo_phase;
        let mut current_phase = self.audio_rate_state.phase;
        self.audio_rate_state.phase = params.phase + stereo_phase * PolyF32::stereo(0.5, -0.5);

        let reset_mask = self.block_reset_mask;
        if params.sync_type == LfoSyncType::SustainEnvelope {
            current_phase = reset_mask.select(PolyF32::ZERO, current_phase);
        } else {
            current_phase = reset_mask.select(self.audio_rate_state.phase, current_phase);
        }

        let frequency = params.frequency;
        let tick_time = 1.0 / self.sample_rate;
        let delta_offset = frequency * tick_time;

        let offset = self.audio_rate_state.offset.max(PolyF32::ZERO);
        let output_phase = match params.sync_type {
            LfoSyncType::Envelope => {
                self.audio_envelope(source, params, out, current_phase, offset, delta_offset)
            }
            LfoSyncType::SustainEnvelope => self.audio_sustain_envelope(
                source,
                params,
                out,
                current_phase,
                offset,
                delta_offset,
            ),
            LfoSyncType::Trigger | LfoSyncType::Sync => {
                self.audio_lfo(source, params, out, current_phase, offset, delta_offset)
            }
            LfoSyncType::LoopPoint => {
                self.audio_loop_point(source, params, out, current_phase, offset, delta_offset)
            }
            LfoSyncType::LoopHold => {
                self.audio_loop_hold(source, params, out, current_phase, offset, delta_offset)
            }
        };

        self.last_phase = output_phase;
        self.last_value = out[num_samples - 1];
    }

    fn audio_envelope(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        out: &mut [PolyF32],
        mut current_phase: PolyF32,
        mut current_offset: PolyF32,
        delta_offset: PolyF32,
    ) -> PolyF32 {
        let mut setup = self.audio_setup(params, current_phase, out.len());
        let mut phased_offset = PolyF32::ZERO;

        for sample in out.iter_mut() {
            setup.delay_time_passed += setup.tick_time;
            let past_delay_mask = setup.delay_time_passed.ge(setup.delay_time);
            setup.current_amplitude = (setup.current_amplitude
                + (setup.fade_increase & past_delay_mask))
                .clamp(0.0, 1.0);
            phased_offset = (current_offset + current_phase).min(PolyF32::ONE);
            let value = value_at_phase(source, phased_offset);
            setup.current_value = utils::interpolate(value, setup.current_value, setup.smooth_mult);
            *sample = setup.current_amplitude * setup.current_value;

            current_offset =
                (current_offset + (delta_offset & past_delay_mask)).min(PolyF32::ONE);
            current_phase += setup.delta_phase;
        }

        self.audio_rate_state.smooth_value = setup.current_value;
        self.audio_rate_state.fade_amplitude = setup.current_amplitude;
        self.audio_rate_state.delay_time_passed = setup.delay_time_passed;
        self.audio_rate_state.offset = current_offset.min(PolyF32::ONE);
        phased_offset
    }

    fn audio_sustain_envelope(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        out: &mut [PolyF32],
        mut current_phase: PolyF32,
        mut current_offset: PolyF32,
        delta_offset: PolyF32,
    ) -> PolyF32 {
        let mut setup = self.audio_setup(params, current_phase, out.len());

        let mut current_hold_mask = PolyMask::NONE;
        let held_mask = self.held_mask;
        let trigger_sample = self.trigger_sample;

        for (i, sample) in out.iter_mut().enumerate() {
            setup.delay_time_passed += setup.tick_time;
            let past_delay_mask = setup.delay_time_passed.ge(setup.delay_time);
            setup.current_amplitude = (setup.current_amplitude
                + (setup.fade_increase & past_delay_mask))
                .clamp(0.0, 1.0);

            let at_trigger = PolyU32::splat(i as u32).eq(trigger_sample);
            current_hold_mask = PolyMask::from_u32(
                at_trigger.select_u32(held_mask.to_u32(), current_hold_mask.to_u32()),
            );
            let max = current_hold_mask.select(current_phase, PolyF32::ONE);
            let value = value_at_phase(source, current_offset);
            setup.current_value = utils::interpolate(value, setup.current_value, setup.smooth_mult);
            *sample = setup.current_amplitude * setup.current_value;

            current_offset = (current_offset + (delta_offset & past_delay_mask)).min(max);
            current_phase += setup.delta_phase;
        }

        self.audio_rate_state.smooth_value = setup.current_value;
        self.audio_rate_state.fade_amplitude = setup.current_amplitude;
        self.audio_rate_state.delay_time_passed = setup.delay_time_passed;
        let last_max = current_hold_mask.select(current_phase, PolyF32::ONE);
        self.audio_rate_state.offset = current_offset.min(last_max);
        current_offset
    }

    fn audio_lfo(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        out: &mut [PolyF32],
        mut current_phase: PolyF32,
        mut current_offset: PolyF32,
        delta_offset: PolyF32,
    ) -> PolyF32 {
        let num_samples = out.len();
        let mut setup = self.audio_setup(params, current_phase, num_samples);
        let delaying_mask = setup.delay_time.gt(setup.delay_time_passed);
        let mut phased_offset = PolyF32::ZERO;

        for sample in out.iter_mut() {
            setup.delay_time_passed += setup.tick_time;
            let past_delay_mask = setup.delay_time_passed.ge(setup.delay_time);
            setup.current_amplitude = (setup.current_amplitude
                + (setup.fade_increase & past_delay_mask))
                .clamp(0.0, 1.0);

            phased_offset = (current_offset + current_phase).fract();
            let value = value_at_phase(source, phased_offset);
            setup.current_value = utils::interpolate(value, setup.current_value, setup.smooth_mult);
            *sample = setup.current_amplitude * setup.current_value;

            current_offset = (current_offset + (delta_offset & past_delay_mask)).fract();
            current_phase += setup.delta_phase;
        }

        self.audio_rate_state.smooth_value = setup.current_value;
        self.audio_rate_state.fade_amplitude = setup.current_amplitude;
        self.audio_rate_state.delay_time_passed = setup.delay_time_passed;
        let undelayed_offset =
            (self.audio_rate_state.offset + delta_offset * num_samples as f32).fract();
        self.audio_rate_state.offset = delaying_mask.select(current_offset, undelayed_offset);
        phased_offset
    }

    fn audio_loop_point(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        out: &mut [PolyF32],
        mut current_phase: PolyF32,
        mut current_offset: PolyF32,
        delta_offset: PolyF32,
    ) -> PolyF32 {
        let mut setup = self.audio_setup(params, current_phase, out.len());

        for sample in out.iter_mut() {
            setup.delay_time_passed += setup.tick_time;
            let past_delay_mask = setup.delay_time_passed.ge(setup.delay_time);
            setup.current_amplitude = (setup.current_amplitude
                + (setup.fade_increase & past_delay_mask))
                .clamp(0.0, 1.0);

            current_offset += delta_offset & past_delay_mask;
            let over = current_offset.ge(PolyF32::ONE);
            current_offset = over
                .select(current_offset - PolyF32::ONE + current_phase, current_offset)
                .min(PolyF32::ONE);
            let value = value_at_phase(source, current_offset);
            setup.current_value = utils::interpolate(value, setup.current_value, setup.smooth_mult);
            *sample = setup.current_amplitude * setup.current_value;
            current_phase += setup.delta_phase;
        }

        self.audio_rate_state.smooth_value = setup.current_value;
        self.audio_rate_state.fade_amplitude = setup.current_amplitude;
        self.audio_rate_state.delay_time_passed = setup.delay_time_passed;
        self.audio_rate_state.offset = current_offset;
        current_offset
    }

    fn audio_loop_hold(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        out: &mut [PolyF32],
        mut current_phase: PolyF32,
        mut current_offset: PolyF32,
        delta_offset: PolyF32,
    ) -> PolyF32 {
        let mut setup = self.audio_setup(params, current_phase, out.len());
        let held_mask = self.held_mask;

        for sample in out.iter_mut() {
            setup.delay_time_passed += setup.tick_time;
            let past_delay_mask = setup.delay_time_passed.ge(setup.delay_time);
            setup.current_amplitude = (setup.current_amplitude
                + (setup.fade_increase & past_delay_mask))
                .clamp(0.0, 1.0);

            current_offset += delta_offset & past_delay_mask;
            let over = held_mask & current_offset.ge(current_phase);
            current_offset = over
                .select(current_offset - current_phase, current_offset)
                .min(PolyF32::ONE);
            let value = value_at_phase(source, current_offset);
            setup.current_value = utils::interpolate(value, setup.current_value, setup.smooth_mult);
            *sample = setup.current_amplitude * setup.current_value;
            current_phase += setup.delta_phase;
        }

        self.audio_rate_state.smooth_value = setup.current_value;
        self.audio_rate_state.fade_amplitude = setup.current_amplitude;
        self.audio_rate_state.delay_time_passed = setup.delay_time_passed;
        self.audio_rate_state.offset = current_offset;
        current_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;

    fn on() -> PolyF32 {
        PolyF32::splat(VoiceEvent::On.as_f32())
    }

    fn params(frequency: f32) -> SynthLfoParams {
        SynthLfoParams { frequency: PolyF32::splat(frequency), ..Default::default() }
    }

    #[test]
    fn triangle_shape_tracks_phase() {
        let source = LineGenerator::triangle();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let params = params(1.0);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        // 1 Hz: a quarter cycle per 11025 samples. Sample the value at
        // phases 0.25 / 0.5 / 0.75 of the triangle (values 0.5 / 1.0 / 0.5).
        let mut values = Vec::new();
        for _ in 0..44 {
            values.push(lfo.process_control(&source, &params, 1000).lane(0));
        }
        let at = |phase: f32| values[(phase * 44.1) as usize];
        assert!((at(0.25) - 0.5).abs() < 0.05, "quarter {}", at(0.25));
        assert!((at(0.5) - 1.0).abs() < 0.05, "half {}", at(0.5));
        assert!((at(0.75) - 0.5).abs() < 0.05, "three-quarter {}", at(0.75));
    }

    #[test]
    fn square_shape_tracks_phase() {
        let source = LineGenerator::square();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let params = params(1.0);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        let mut values = Vec::new();
        for _ in 0..44 {
            values.push(lfo.process_control(&source, &params, 1000).lane(0));
        }
        let at = |phase: f32| values[(phase * 44.1) as usize];
        assert!((at(0.25) - 1.0).abs() < 0.02, "first half {}", at(0.25));
        assert!((at(0.75) - 0.0).abs() < 0.02, "second half {}", at(0.75));
    }

    #[test]
    fn sync_frequency_math() {
        // In Sync mode a retrigger derives phase from transport time:
        // 1.5 Hz at t = 2.5s -> 3.75 cycles -> phase 0.75.
        let source = LineGenerator::linear();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let mut sync_params = params(1.5);
        sync_params.sync_type = LfoSyncType::Sync;

        lfo.correct_to_time(2.5);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        let value = lfo.process_control(&source, &sync_params, 1).lane(0);
        // Linear shape ramps 0..1 with phase; one control sample later the
        // phase is still ~0.75.
        assert!((value - 0.75).abs() < 1e-3, "sync phase value {value}");
    }

    #[test]
    fn trigger_resets_phase() {
        let source = LineGenerator::linear();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let params = params(2.0);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        for _ in 0..10 {
            lfo.process_control(&source, &params, 1000);
        }
        let advanced = lfo.phase().lane(0);
        assert!(advanced > 0.2);

        lfo.trigger(PolyMask::all_on(), on(), 0);
        lfo.process_control(&source, &params, 1);
        assert!(lfo.phase().lane(0) < 0.01);
    }

    #[test]
    fn envelope_mode_stops_at_end() {
        let source = LineGenerator::linear();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let mut envelope_params = params(4.0);
        envelope_params.sync_type = LfoSyncType::Envelope;
        lfo.trigger(PolyMask::all_on(), on(), 0);

        // 4 Hz one-shot finishes after 0.25s; run 1s.
        let mut last = 0.0;
        for _ in 0..44 {
            last = lfo.process_control(&source, &envelope_params, 1000).lane(0);
        }
        assert!((last - 1.0).abs() < 1e-3, "one-shot should hold the end value, got {last}");
    }

    #[test]
    fn stereo_phase_splits_channels() {
        let source = LineGenerator::linear();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let mut stereo_params = params(1.0);
        stereo_params.stereo_phase = PolyF32::splat(0.5);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        lfo.process_control(&source, &stereo_params, 4410);
        let value = lfo.value();
        assert!(
            (value.lane(0) - value.lane(1)).abs() > 0.1,
            "stereo phase should separate L/R: {:?}",
            value.to_lanes()
        );
    }

    #[test]
    fn audio_rate_matches_control_trend() {
        let source = LineGenerator::linear();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let params = params(10.0);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        let mut out = [PolyF32::ZERO; 128];
        lfo.process_audio(&source, &params, &mut out);
        // Linear ramp shape at 10 Hz: rising within the block.
        assert!(out[127].lane(0) > out[8].lane(0));
    }

    #[test]
    fn fade_ramps_amplitude() {
        let source = LineGenerator::square();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let mut fade_params = params(0.1);
        fade_params.fade_time = PolyF32::splat(1.0);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        // Early in the fade the output hugs the start value; later it
        // reaches the full shape value (square's first half is 1).
        let early = lfo.process_control(&source, &fade_params, 441).lane(0);
        for _ in 0..200 {
            lfo.process_control(&source, &fade_params, 441);
        }
        let late = lfo.value().lane(0);
        assert!(early < 0.2, "faded-in output should start low, got {early}");
        assert!(late > 0.9, "output should reach full level, got {late}");
    }
}
