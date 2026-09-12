//! LFO reading a [`LineGenerator`] shape (port of `SynthLfo`).
//!
//! Supports six sync modes (free/host-synced/one-shot envelope variants and
//! loop-point styles), delay and fade-in, per-voice stereo phase offset and
//! an exponential smoothing mode. Control-rate and audio-rate paths keep
//! separate state, both updated like the reference so switching rates is
//! seamless.
//!
//! Serum-2-style generator modes ([`LfoGeneratorMode`]) swap what produces
//! the value while keeping all of that machinery: the drawn shape (default),
//! sample-and-hold with glide, two chaotic attractors (Lorenz/Rossler), and
//! a 2D path of two shapes via [`SynthLfo::process_control_path`].

use spinwave_poly::constants::VoiceEvent;
use spinwave_poly::{math, utils, PolyF32, PolyMask, PolyU32};

use super::line_generator::LineGenerator;
use super::random::RandomGenerator;

pub const HALF_LIFE_RATIO: f32 = 0.2;
pub const MIN_HALF_LIFE: f32 = 0.0002;

/// Below this glide amount SampleHold steps instantly.
pub const MIN_SH_GLIDE: f32 = 1e-4;

// Chaos1: Lorenz attractor (constants shared with `random_lfo`).
const LORENZ_A: f32 = 10.0;
const LORENZ_B: f32 = 28.0;
const LORENZ_C: f32 = 8.0 / 3.0;
const LORENZ_INITIAL1: f32 = 0.0;
const LORENZ_INITIAL2: f32 = 0.0;
const LORENZ_INITIAL3: f32 = 37.6;
/// Bipolar output scale: `state1` swings roughly +-20.
const LORENZ_OUT_SCALE: f32 = 1.0 / 20.0;
/// Time units advanced per second at 1 Hz (matching `random_lfo`).
const LORENZ_RATE: f32 = 0.5;
const LORENZ_MAX_STEP: f32 = 0.01;

// Chaos2: Rossler attractor.
const ROSSLER_A: f32 = 0.2;
const ROSSLER_B: f32 = 0.2;
const ROSSLER_C: f32 = 5.7;
const ROSSLER_INITIAL1: f32 = 0.1;
const ROSSLER_INITIAL2: f32 = 0.0;
const ROSSLER_INITIAL3: f32 = 0.0;
/// Bipolar output scale: `state1` swings roughly -10..12.
const ROSSLER_OUT_SCALE: f32 = 1.0 / 12.0;
/// Rossler orbits ~10x slower than Lorenz, so it gets a faster clock.
const ROSSLER_RATE: f32 = 6.0;
const ROSSLER_MAX_STEP: f32 = 0.05;

/// Keeps chaotic states finite through parameter jumps or character switches.
const CHAOS_STATE_LIMIT: f32 = 1000.0;

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

/// What produces the LFO's value at a given phase (Serum-2-style generator
/// modes). Every mode runs through the same sync/trigger/fade/delay/smooth
/// and stereo-phase machinery.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LfoGeneratorMode {
    /// The drawn [`LineGenerator`] shape (default, original behavior).
    #[default]
    Shape,
    /// A new random target each cycle (`frequency` is the hold rate), with
    /// optional glide via [`SynthLfoParams::sample_hold_glide`].
    SampleHold,
    /// Lorenz attractor, rate scaled by the LFO frequency.
    Chaos1,
    /// Rossler attractor, rate scaled by the LFO frequency.
    Chaos2,
    /// 2D path: two [`LineGenerator`] shapes x(t)/y(t) evaluated at the same
    /// phase, producing two outputs. Control-rate only, via
    /// [`SynthLfo::process_control_path`]; the kernel can route X and Y as
    /// separate modulation sources.
    Path,
}

/// Per-block parameters. `phase` is the shape phase offset (also the loop
/// point for the loop modes), `stereo_phase` splits left/right phases.
#[derive(Clone, Copy, Debug)]
pub struct SynthLfoParams {
    pub frequency: PolyF32,
    pub phase: PolyF32,
    pub stereo_phase: PolyF32,
    pub sync_type: LfoSyncType,
    pub smooth_mode: bool,
    pub fade_time: PolyF32,
    pub smooth_time: PolyF32,
    pub delay_time: PolyF32,
    /// Value generator driving the LFO (default: the drawn shape).
    pub generator: LfoGeneratorMode,
    /// SampleHold only: portion of each cycle (0..1) spent gliding from the
    /// previous value to the new target. 0 steps instantly.
    pub sample_hold_glide: PolyF32,
    /// Chaos1/Chaos2 only: rate multiplier applied on top of `frequency`
    /// (default 1). The chaotic clock is `frequency * chaos_speed`, so tempo
    /// sync of `frequency` scales the attractor too.
    pub chaos_speed: PolyF32,
}

impl Default for SynthLfoParams {
    fn default() -> Self {
        SynthLfoParams {
            frequency: PolyF32::ZERO,
            phase: PolyF32::ZERO,
            stereo_phase: PolyF32::ZERO,
            sync_type: LfoSyncType::default(),
            smooth_mode: false,
            fade_time: PolyF32::ZERO,
            smooth_time: PolyF32::ZERO,
            delay_time: PolyF32::ZERO,
            generator: LfoGeneratorMode::Shape,
            sample_hold_glide: PolyF32::ZERO,
            chaos_speed: PolyF32::ONE,
        }
    }
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

    /// Random stream for SampleHold targets and chaos initial states.
    random_generator: RandomGenerator,
    /// Pristine copy of the stream: retrigger restores it so the random
    /// sequence restarts deterministically on every note.
    seed_generator: RandomGenerator,
    /// SampleHold: value gliding from / target / phase seen last evaluation.
    sh_from: PolyF32,
    sh_target: PolyF32,
    sh_prev_phase: PolyF32,
    /// Chaos1/Chaos2 attractor state (shared; switching characters continues
    /// from the current state).
    chaos_state1: PolyF32,
    chaos_state2: PolyF32,
    chaos_state3: PolyF32,
    /// Second smoothing state for the Y output of Path mode.
    path_smooth_y: PolyF32,
}

impl SynthLfo {
    pub fn new(sample_rate: f32) -> Self {
        Self::with_generator(sample_rate, RandomGenerator::new(-1.0, 1.0))
    }

    /// Deterministic construction for tests/replays: SampleHold and chaos
    /// draw from a seeded stream.
    pub fn with_seed(sample_rate: f32, seed: u32) -> Self {
        Self::with_generator(sample_rate, RandomGenerator::with_seed(-1.0, 1.0, seed))
    }

    /// Restarts the sample-and-hold / chaos seed generator from `seed`:
    /// the next trigger draws the values a generator built with that seed
    /// would draw first.
    pub fn reseed(&mut self, seed: u32) {
        self.seed_generator.seed(seed);
        self.random_generator = self.seed_generator.clone();
    }

    fn with_generator(sample_rate: f32, random_generator: RandomGenerator) -> Self {
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
            seed_generator: random_generator.clone(),
            random_generator,
            sh_from: PolyF32::ZERO,
            sh_target: PolyF32::ZERO,
            sh_prev_phase: PolyF32::ZERO,
            chaos_state1: PolyF32::splat(0.1),
            chaos_state2: PolyF32::ZERO,
            chaos_state3: PolyF32::ZERO,
            path_smooth_y: PolyF32::ZERO,
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
    /// applied on the next process call at `sample_offset` (the same
    /// offset for every lane; see [`Self::trigger_at`]).
    pub fn trigger(&mut self, mask: PolyMask, value: PolyF32, sample_offset: usize) {
        self.trigger_at(mask, value, PolyU32::splat(sample_offset as u32));
    }

    /// Like [`Self::trigger`] with a per-lane sample offset (the reference
    /// `trigger_offset` semantics). Lanes outside `mask` keep any trigger
    /// already queued; queuing twice for the same lane before a process
    /// call is a caller bug (debug-asserted).
    pub fn trigger_at(&mut self, mask: PolyMask, value: PolyF32, sample_offsets: PolyU32) {
        debug_assert!(
            !(self.trigger_mask & mask).any(),
            "LFO trigger overwritten before being processed"
        );
        self.trigger_mask |= mask;
        self.trigger_value = mask.select(value, self.trigger_value);
        self.trigger_offset = mask.select_u32(sample_offsets, self.trigger_offset);
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
        self.path_smooth_y = reset_mask.select(PolyF32::ZERO, self.path_smooth_y);

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
    ///
    /// [`LfoGeneratorMode::Path`] has no single-output form; when selected
    /// here it falls back to Shape behavior on `source` (use
    /// [`Self::process_control_path`] for the two-output path).
    pub fn process_control(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        num_samples: usize,
    ) -> PolyF32 {
        self.was_control_rate = true;
        self.process_trigger(params);
        match params.generator {
            LfoGeneratorMode::Shape | LfoGeneratorMode::Path => {
                self.control_step(source, params, num_samples, true)
            }
            LfoGeneratorMode::SampleHold => self.control_step_sample_hold(params, num_samples),
            LfoGeneratorMode::Chaos1 | LfoGeneratorMode::Chaos2 => {
                self.control_step_chaos(params, num_samples)
            }
        }
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
    /// keeps the control-rate state in step so switching rates stays
    /// consistent.
    ///
    /// SampleHold and Chaos1/Chaos2 run dedicated per-sample loops here;
    /// [`LfoGeneratorMode::Path`] is control-rate only and falls back to
    /// Shape behavior on `source`.
    pub fn process_audio(
        &mut self,
        source: &LineGenerator,
        params: &SynthLfoParams,
        out: &mut [PolyF32],
    ) {
        match params.generator {
            LfoGeneratorMode::Shape | LfoGeneratorMode::Path => {
                if self.was_control_rate {
                    self.audio_rate_state = self.control_rate_state;
                }
                self.was_control_rate = false;

                self.process_trigger(params);
                self.audio_step(source, params, out);
                self.control_step(source, params, out.len(), false);
            }
            LfoGeneratorMode::SampleHold => self.process_audio_sample_hold(params, out),
            LfoGeneratorMode::Chaos1 | LfoGeneratorMode::Chaos2 => {
                self.process_audio_chaos(params, out)
            }
        }
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

    // -- Generator modes (SampleHold / Chaos / Path) -------------------------

    /// The phase/delay/fade machinery of `control_step` without the shape
    /// evaluation, shared by the non-Shape generator modes. Returns the
    /// phased offset and the delay-gated time advanced (seconds).
    fn control_phase_step(
        &mut self,
        params: &SynthLfoParams,
        num_samples: usize,
    ) -> (PolyF32, PolyF32) {
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

        (phased_offset, time_passed)
    }

    /// Smoothing/fade tail shared by the non-Shape control paths, mirroring
    /// the end of `control_step`.
    fn finish_control_value(
        &mut self,
        raw: PolyF32,
        start_value: PolyF32,
        time_passed: PolyF32,
        params: &SynthLfoParams,
    ) -> PolyF32 {
        let result = if params.smooth_mode {
            let half_life = params.smooth_time * HALF_LIFE_RATIO;
            let smooth_mask = half_life.gt(PolyF32::splat(MIN_HALF_LIFE));
            let exponent = -time_passed / half_life.max(PolyF32::splat(MIN_HALF_LIFE));
            let ratio = math::exp2(exponent) & smooth_mask;
            let result = utils::interpolate(raw, self.control_rate_state.smooth_value, ratio);
            self.control_rate_state.smooth_value = result;
            result
        } else {
            utils::interpolate(start_value, raw, self.control_rate_state.fade_amplitude)
        };
        result.clamp(-1.0, 1.0)
    }

    /// Retrigger for SampleHold: restores the pristine random stream so the
    /// sequence restarts deterministically, then draws fresh from/target
    /// values (per voice, both stereo lanes coherent).
    fn reset_sample_hold(&mut self) {
        let reset_mask = self.block_reset_mask;
        if !reset_mask.any() {
            return;
        }
        self.random_generator = self.seed_generator.clone();
        let from = self.random_generator.poly_voice_next();
        let target = self.random_generator.poly_voice_next();
        self.sh_from = reset_mask.select(from, self.sh_from);
        self.sh_target = reset_mask.select(target, self.sh_target);
        self.sh_prev_phase = reset_mask.select(PolyF32::ZERO, self.sh_prev_phase);
    }

    /// Draws a new SampleHold target for lanes whose phase wrapped (like
    /// `RandomLfo::update_phase`, the draw happens when any lane wraps).
    fn advance_sample_hold(&mut self, phased_offset: PolyF32) {
        let wrap = phased_offset.lt(self.sh_prev_phase);
        if wrap.any() {
            let next = self.random_generator.poly_voice_next();
            self.sh_from = wrap.select(self.sh_target, self.sh_from);
            self.sh_target = wrap.select(next, self.sh_target);
        }
        self.sh_prev_phase = phased_offset;
    }

    fn control_step_sample_hold(&mut self, params: &SynthLfoParams, num_samples: usize) -> PolyF32 {
        self.reset_sample_hold();
        let (phased_offset, time_passed) = self.control_phase_step(params, num_samples);
        self.advance_sample_hold(phased_offset);
        let raw =
            sample_hold_value(self.sh_from, self.sh_target, phased_offset, params.sample_hold_glide);
        let result = self.finish_control_value(raw, PolyF32::ZERO, time_passed, params);
        self.last_value = result;
        self.last_phase = phased_offset;
        result
    }

    /// Retrigger for chaos: restores the pristine random stream and lands the
    /// attractor on a deterministic randomized initial state.
    fn reset_chaos(&mut self, generator: LfoGeneratorMode) {
        let reset_mask = self.block_reset_mask;
        if !reset_mask.any() {
            return;
        }
        self.random_generator = self.seed_generator.clone();
        let (initial1, initial2, initial3) = if generator == LfoGeneratorMode::Chaos2 {
            (ROSSLER_INITIAL1, ROSSLER_INITIAL2, ROSSLER_INITIAL3)
        } else {
            (LORENZ_INITIAL1, LORENZ_INITIAL2, LORENZ_INITIAL3)
        };
        let value1 = self.random_generator.poly_voice_next() + initial1;
        let value2 = self.random_generator.poly_voice_next() + initial2;
        let value3 = self.random_generator.poly_voice_next() + initial3;
        self.chaos_state1 = reset_mask.select(value1, self.chaos_state1);
        self.chaos_state2 = reset_mask.select(value2, self.chaos_state2);
        self.chaos_state3 = reset_mask.select(value3, self.chaos_state3);
    }

    fn control_step_chaos(&mut self, params: &SynthLfoParams, num_samples: usize) -> PolyF32 {
        self.reset_chaos(params.generator);
        let (phased_offset, time_passed) = self.control_phase_step(params, num_samples);

        let (rate, max_step, out_scale) = chaos_config(params.generator);
        // Delay gating: while delaying, time_passed is 0 and the attractor
        // holds still, matching the delay behavior of the other modes.
        let step_seconds = time_passed * (1.0 / num_samples as f32);
        let t = (params.frequency * params.chaos_speed * rate * step_seconds).clamp(0.0, max_step);

        let mut state1 = self.chaos_state1;
        let mut state2 = self.chaos_state2;
        let mut state3 = self.chaos_state3;
        for _ in 0..num_samples {
            let (delta1, delta2, delta3) = chaos_delta(params.generator, state1, state2, state3);
            state1 += delta1 * t;
            state2 += delta2 * t;
            state3 += delta3 * t;
        }
        self.chaos_state1 = state1.clamp(-CHAOS_STATE_LIMIT, CHAOS_STATE_LIMIT);
        self.chaos_state2 = state2.clamp(-CHAOS_STATE_LIMIT, CHAOS_STATE_LIMIT);
        self.chaos_state3 = state3.clamp(-CHAOS_STATE_LIMIT, CHAOS_STATE_LIMIT);

        let raw = (self.chaos_state1 * out_scale).clamp(-1.0, 1.0);
        let result = self.finish_control_value(raw, PolyF32::ZERO, time_passed, params);
        self.last_value = result;
        self.last_phase = phased_offset;
        result
    }

    /// Control-rate tick for [`LfoGeneratorMode::Path`]: evaluates a 2D path
    /// drawn as two [`LineGenerator`] shapes `x(t)` / `y(t)` at the LFO phase
    /// and returns `(x, y)`. All sync/trigger/fade/delay/smooth/stereo-phase
    /// machinery applies; X shares the LFO's primary state (`value()` reports
    /// X), Y smooths through its own state. Path is control-rate only — the
    /// kernel can route the two outputs as separate modulation sources.
    pub fn process_control_path(
        &mut self,
        x: &LineGenerator,
        y: &LineGenerator,
        params: &SynthLfoParams,
        num_samples: usize,
    ) -> (PolyF32, PolyF32) {
        self.was_control_rate = true;
        self.process_trigger(params);
        let (phased_offset, time_passed) = self.control_phase_step(params, num_samples);

        let raw_x = value_at_phase(x, phased_offset);
        let raw_y = value_at_phase(y, phased_offset);

        let (out_x, out_y) = if params.smooth_mode {
            let half_life = params.smooth_time * HALF_LIFE_RATIO;
            let smooth_mask = half_life.gt(PolyF32::splat(MIN_HALF_LIFE));
            let exponent = -time_passed / half_life.max(PolyF32::splat(MIN_HALF_LIFE));
            let ratio = math::exp2(exponent) & smooth_mask;
            let out_x = utils::interpolate(raw_x, self.control_rate_state.smooth_value, ratio);
            self.control_rate_state.smooth_value = out_x;
            let out_y = utils::interpolate(raw_y, self.path_smooth_y, ratio);
            self.path_smooth_y = out_y;
            (out_x, out_y)
        } else {
            let static_phase = params.phase + params.stereo_phase * PolyF32::stereo(0.5, -0.5);
            let fade = self.control_rate_state.fade_amplitude;
            (
                utils::interpolate(value_at_phase(x, static_phase), raw_x, fade),
                utils::interpolate(value_at_phase(y, static_phase), raw_y, fade),
            )
        };

        let out_x = out_x.clamp(-1.0, 1.0);
        let out_y = out_y.clamp(-1.0, 1.0);
        self.last_value = out_x;
        self.last_phase = phased_offset;
        (out_x, out_y)
    }

    /// Audio-rate SampleHold: per-sample phase advance and wrap-triggered
    /// target draws, with the standard delay/fade/smooth machinery. Instead
    /// of the parallel control tick Shape mode uses (which would double-draw
    /// the random stream), the control-rate state is synced afterwards.
    fn process_audio_sample_hold(&mut self, params: &SynthLfoParams, out: &mut [PolyF32]) {
        if self.was_control_rate {
            self.audio_rate_state = self.control_rate_state;
        }
        self.was_control_rate = false;
        self.process_trigger(params);
        self.reset_sample_hold();

        let num_samples = out.len();
        let stereo_phase = params.stereo_phase;
        let mut current_phase = self.audio_rate_state.phase;
        self.audio_rate_state.phase = params.phase + stereo_phase * PolyF32::stereo(0.5, -0.5);
        let reset_mask = self.block_reset_mask;
        current_phase = reset_mask.select(self.audio_rate_state.phase, current_phase);

        let delta_offset = params.frequency * (1.0 / self.sample_rate);
        let mut current_offset = self.audio_rate_state.offset.max(PolyF32::ZERO);
        let mut setup = self.audio_setup(params, current_phase, num_samples);
        let held_mask = self.held_mask;
        let glide = params.sample_hold_glide;
        let sync_type = params.sync_type;

        let mut sh_from = self.sh_from;
        let mut sh_target = self.sh_target;
        let mut prev_phased = self.sh_prev_phase;
        let mut phased_offset = prev_phased;

        for sample in out.iter_mut() {
            setup.delay_time_passed += setup.tick_time;
            let past_delay_mask = setup.delay_time_passed.ge(setup.delay_time);
            setup.current_amplitude = (setup.current_amplitude
                + (setup.fade_increase & past_delay_mask))
                .clamp(0.0, 1.0);

            phased_offset = phased_for_sync(sync_type, current_offset, current_phase, held_mask);
            let wrap = phased_offset.lt(prev_phased);
            if wrap.any() {
                let next = self.random_generator.poly_voice_next();
                sh_from = wrap.select(sh_target, sh_from);
                sh_target = wrap.select(next, sh_target);
            }
            prev_phased = phased_offset;

            let raw = sample_hold_value(sh_from, sh_target, phased_offset, glide);
            setup.current_value = utils::interpolate(raw, setup.current_value, setup.smooth_mult);
            *sample = (setup.current_amplitude * setup.current_value).clamp(-1.0, 1.0);

            current_offset = advance_offset_for_sync(
                sync_type,
                current_offset,
                delta_offset & past_delay_mask,
                current_phase,
                held_mask,
            );
            current_phase += setup.delta_phase;
        }

        self.sh_from = sh_from;
        self.sh_target = sh_target;
        self.sh_prev_phase = prev_phased;
        self.audio_rate_state.smooth_value = setup.current_value;
        self.audio_rate_state.fade_amplitude = setup.current_amplitude;
        self.audio_rate_state.delay_time_passed = setup.delay_time_passed;
        self.audio_rate_state.offset = current_offset;
        self.last_phase = phased_offset;
        self.last_value = out[num_samples - 1];
        self.control_rate_state = self.audio_rate_state;
    }

    /// Audio-rate chaos: one Euler step per sample, delay gating freezing the
    /// attractor, fade/smooth applied like the other audio paths. The
    /// control-rate state is synced afterwards (no parallel control tick, so
    /// the attractor advances exactly once per sample).
    fn process_audio_chaos(&mut self, params: &SynthLfoParams, out: &mut [PolyF32]) {
        if self.was_control_rate {
            self.audio_rate_state = self.control_rate_state;
        }
        self.was_control_rate = false;
        self.process_trigger(params);
        self.reset_chaos(params.generator);

        let num_samples = out.len();
        let current_phase = self.audio_rate_state.phase;
        self.audio_rate_state.phase =
            params.phase + params.stereo_phase * PolyF32::stereo(0.5, -0.5);
        let mut setup = self.audio_setup(params, current_phase, num_samples);

        let (rate, max_step, out_scale) = chaos_config(params.generator);
        let t = (params.frequency * params.chaos_speed * (rate / self.sample_rate))
            .clamp(0.0, max_step);
        let generator = params.generator;

        let mut state1 = self.chaos_state1;
        let mut state2 = self.chaos_state2;
        let mut state3 = self.chaos_state3;

        for sample in out.iter_mut() {
            setup.delay_time_passed += setup.tick_time;
            let past_delay_mask = setup.delay_time_passed.ge(setup.delay_time);
            setup.current_amplitude = (setup.current_amplitude
                + (setup.fade_increase & past_delay_mask))
                .clamp(0.0, 1.0);

            let step = t & past_delay_mask;
            let (delta1, delta2, delta3) = chaos_delta(generator, state1, state2, state3);
            state1 += delta1 * step;
            state2 += delta2 * step;
            state3 += delta3 * step;

            let raw = (state1 * out_scale).clamp(-1.0, 1.0);
            setup.current_value = utils::interpolate(raw, setup.current_value, setup.smooth_mult);
            *sample = (setup.current_amplitude * setup.current_value).clamp(-1.0, 1.0);
        }

        self.chaos_state1 = state1.clamp(-CHAOS_STATE_LIMIT, CHAOS_STATE_LIMIT);
        self.chaos_state2 = state2.clamp(-CHAOS_STATE_LIMIT, CHAOS_STATE_LIMIT);
        self.chaos_state3 = state3.clamp(-CHAOS_STATE_LIMIT, CHAOS_STATE_LIMIT);
        self.audio_rate_state.smooth_value = setup.current_value;
        self.audio_rate_state.fade_amplitude = setup.current_amplitude;
        self.audio_rate_state.delay_time_passed = setup.delay_time_passed;
        self.last_value = out[num_samples - 1];
        self.control_rate_state = self.audio_rate_state;
    }
}

/// SampleHold output: glide over the first `glide` portion of the cycle,
/// then hold the target. `glide` below [`MIN_SH_GLIDE`] steps instantly.
#[inline]
fn sample_hold_value(
    from: PolyF32,
    target: PolyF32,
    phased_offset: PolyF32,
    glide: PolyF32,
) -> PolyF32 {
    let glide = glide.clamp(0.0, 1.0);
    let glide_mask = glide.gt(PolyF32::splat(MIN_SH_GLIDE));
    let ratio = (phased_offset / glide.max(PolyF32::splat(MIN_SH_GLIDE))).min(PolyF32::ONE);
    let ratio = glide_mask.select(ratio, PolyF32::ONE);
    utils::interpolate(from, target, ratio)
}

/// `(rate, max Euler step, bipolar output scale)` per chaos character.
#[inline]
fn chaos_config(generator: LfoGeneratorMode) -> (f32, f32, f32) {
    if generator == LfoGeneratorMode::Chaos2 {
        (ROSSLER_RATE, ROSSLER_MAX_STEP, ROSSLER_OUT_SCALE)
    } else {
        (LORENZ_RATE, LORENZ_MAX_STEP, LORENZ_OUT_SCALE)
    }
}

/// One derivative evaluation of the selected attractor.
#[inline]
fn chaos_delta(
    generator: LfoGeneratorMode,
    state1: PolyF32,
    state2: PolyF32,
    state3: PolyF32,
) -> (PolyF32, PolyF32, PolyF32) {
    if generator == LfoGeneratorMode::Chaos2 {
        (
            -(state2 + state3),
            state1 + state2 * ROSSLER_A,
            state3 * (state1 + (-ROSSLER_C)) + ROSSLER_B,
        )
    } else {
        (
            (state2 - state1) * LORENZ_A,
            (-state3 + LORENZ_B) * state1 - state2,
            state1 * state2 - state3 * LORENZ_C,
        )
    }
}

/// Per-sample phased offset for the audio SampleHold loop, mirroring the
/// sync-type semantics of `control_step`.
#[inline]
fn phased_for_sync(
    sync_type: LfoSyncType,
    offset: PolyF32,
    phase: PolyF32,
    held: PolyMask,
) -> PolyF32 {
    match sync_type {
        LfoSyncType::Trigger | LfoSyncType::Sync => (offset + phase).fract(),
        LfoSyncType::Envelope => (offset + phase).min(PolyF32::ONE),
        LfoSyncType::SustainEnvelope | LfoSyncType::LoopPoint => offset.min(PolyF32::ONE),
        LfoSyncType::LoopHold => held.select(offset.min(phase), offset).min(PolyF32::ONE),
    }
}

/// Per-sample offset advance for the audio SampleHold loop, mirroring the
/// sync-type semantics of `control_step`.
#[inline]
fn advance_offset_for_sync(
    sync_type: LfoSyncType,
    offset: PolyF32,
    delta: PolyF32,
    phase: PolyF32,
    held: PolyMask,
) -> PolyF32 {
    let next = offset + delta;
    match sync_type {
        LfoSyncType::Trigger | LfoSyncType::Sync => next.fract(),
        LfoSyncType::Envelope => next.min(PolyF32::ONE),
        LfoSyncType::SustainEnvelope => next.min(held.select(phase, PolyF32::ONE)),
        LfoSyncType::LoopPoint => {
            let over = next.ge(PolyF32::ONE);
            over.select(next - PolyF32::ONE + phase, next).min(PolyF32::ONE)
        }
        LfoSyncType::LoopHold => {
            let over = held & next.ge(phase);
            over.select(next - phase, next).min(PolyF32::ONE)
        }
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

    // -- Shape-mode regression: reference vectors captured from the build
    // BEFORE generator modes were added (f32 bit patterns). Shape mode (the
    // default) must stay byte-identical.

    #[test]
    fn shape_mode_regression_control() {
        const REFERENCE: [u32; 32] = [
            0, 1013834608, 1022223218, 1026714261, 1030611826, 1033154086, 1035102868, 1037051650,
            1039000433, 1040568304, 1041542695, 1042517091, 1043491483, 1044465873, 1045440265,
            1046414657, 1047389049, 1048363440, 1048956916, 1049444112, 1049931308, 1050418504,
            1050905701, 1051392897, 1051880092, 1052367288, 1052854482, 1053341681, 1053828875,
            1054316071, 1054803267, 1055290463,
        ];
        let source = LineGenerator::triangle();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let p = params(5.0);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        for (i, expected) in REFERENCE.iter().enumerate() {
            let bits = lfo.process_control(&source, &p, 64).lane(0).to_bits();
            assert_eq!(bits, *expected, "control regression at block {i}");
        }
    }

    #[test]
    fn shape_mode_regression_audio() {
        const REFERENCE: [u32; 16] = [
            0, 1007988263, 1016376870, 1021248824, 1024765477, 1027201455, 1029637430, 1031936096,
            1033154086, 1034372077, 1035590067, 1036808058, 1038026048, 1039244039, 1040324710,
            1040933704,
        ];
        let source = LineGenerator::triangle();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let p = params(50.0);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        let mut out = [PolyF32::ZERO; 64];
        lfo.process_audio(&source, &p, &mut out);
        for (i, (sample, expected)) in out.iter().step_by(4).zip(REFERENCE.iter()).enumerate() {
            assert_eq!(sample.lane(0).to_bits(), *expected, "audio regression at sample {}", i * 4);
        }
    }

    #[test]
    fn shape_mode_regression_fade_smooth_stereo() {
        const REFERENCE_L: [u32; 16] = [
            1007100192, 1015548424, 1020093632, 1024052384, 1026379824, 1028733336, 1031112056,
            1032656968, 1033870304, 1035095040, 1036330800, 1037577220, 1038833948, 1040100644,
            1040782186, 1041425014,
        ];
        const REFERENCE_R: [u32; 16] = [
            1007065600, 1015103136, 1018823200, 1022214960, 1024349712, 1025733352, 1026968552,
            1028060206, 1029013048, 1029831654, 1030520450, 1031083717, 1031525594, 1031824434,
            1031929921, 1031980522,
        ];
        let source = LineGenerator::triangle();
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        let mut p = params(2.0);
        p.fade_time = PolyF32::splat(0.5);
        p.smooth_mode = true;
        p.smooth_time = PolyF32::splat(0.3);
        p.stereo_phase = PolyF32::splat(0.25);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        for i in 0..16 {
            let value = lfo.process_control(&source, &p, 128);
            assert_eq!(value.lane(0).to_bits(), REFERENCE_L[i], "left regression at block {i}");
            assert_eq!(value.lane(1).to_bits(), REFERENCE_R[i], "right regression at block {i}");
        }
    }

    #[test]
    fn default_params_select_shape_mode() {
        let p = SynthLfoParams::default();
        assert_eq!(p.generator, LfoGeneratorMode::Shape);
        assert_eq!(p.chaos_speed.lane(0), 1.0);
        assert_eq!(p.sample_hold_glide.lane(0), 0.0);
        assert_eq!(p.frequency.lane(0), 0.0);
    }

    // -- SampleHold ----------------------------------------------------------

    fn sample_hold_params(frequency: f32, glide: f32) -> SynthLfoParams {
        SynthLfoParams {
            frequency: PolyF32::splat(frequency),
            generator: LfoGeneratorMode::SampleHold,
            sample_hold_glide: PolyF32::splat(glide),
            ..Default::default()
        }
    }

    /// Runs SampleHold at control rate and returns lane-0 values per block.
    fn run_sample_hold(seed: u32, glide: f32, blocks: usize, block_size: usize) -> Vec<f32> {
        let source = LineGenerator::triangle();
        let p = sample_hold_params(44.1, glide);
        let mut lfo = SynthLfo::with_seed(SAMPLE_RATE, seed);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        (0..blocks).map(|_| lfo.process_control(&source, &p, block_size).lane(0)).collect()
    }

    #[test]
    fn sample_hold_changes_once_per_cycle() {
        // 44.1 Hz hold rate: 1000-sample cycles; 500 blocks of 10 samples
        // cover 5 cycles. Without glide the value must step exactly once per
        // cycle and hold in between.
        let values = run_sample_hold(7, 0.0, 500, 10);
        let change_blocks: Vec<usize> = values
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] != w[1])
            .map(|(i, _)| i)
            .collect();
        assert!(
            (4..=5).contains(&change_blocks.len()),
            "expected one change per cycle over 5 cycles, got {change_blocks:?}"
        );
        for gap in change_blocks.windows(2) {
            let blocks_between = gap[1] - gap[0];
            assert!(
                (95..=105).contains(&blocks_between),
                "steps should be one hold period apart, got {blocks_between} blocks"
            );
        }
    }

    #[test]
    fn sample_hold_glide_smooths_steps() {
        let stepped = run_sample_hold(7, 0.0, 500, 10);
        let glided = run_sample_hold(7, 0.8, 500, 10);
        let max_delta = |values: &[f32]| {
            values.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max)
        };
        let step_delta = max_delta(&stepped);
        let glide_delta = max_delta(&glided);
        assert!(step_delta > 0.05, "steps should jump, got {step_delta}");
        assert!(
            glide_delta < step_delta * 0.5,
            "glide should shrink the max block delta: glide {glide_delta} vs step {step_delta}"
        );
    }

    #[test]
    fn sample_hold_deterministic_per_seed() {
        let a = run_sample_hold(21, 0.0, 300, 10);
        let b = run_sample_hold(21, 0.0, 300, 10);
        assert_eq!(a, b, "same seed must reproduce the sequence");
        let c = run_sample_hold(22, 0.0, 300, 10);
        assert_ne!(a, c, "different seed should produce a different sequence");
    }

    #[test]
    fn sample_hold_retrigger_resets_sequence() {
        let source = LineGenerator::triangle();
        let p = sample_hold_params(44.1, 0.0);
        let mut lfo = SynthLfo::with_seed(SAMPLE_RATE, 5);

        lfo.trigger(PolyMask::all_on(), on(), 0);
        let first: Vec<f32> =
            (0..300).map(|_| lfo.process_control(&source, &p, 10).lane(0)).collect();

        lfo.trigger(PolyMask::all_on(), on(), 0);
        let second: Vec<f32> =
            (0..300).map(|_| lfo.process_control(&source, &p, 10).lane(0)).collect();

        assert_eq!(first, second, "retrigger must restart the random sequence");
    }

    #[test]
    fn sample_hold_audio_steps_at_rate() {
        // 441 Hz hold rate: a new value every 100 samples of a 512-sample
        // audio buffer, stepwise (constant between changes).
        let source = LineGenerator::triangle();
        let p = sample_hold_params(441.0, 0.0);
        let mut lfo = SynthLfo::with_seed(SAMPLE_RATE, 13);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        let mut out = [PolyF32::ZERO; 512];
        lfo.process_audio(&source, &p, &mut out);
        let change_samples: Vec<usize> = out
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0].lane(0) != w[1].lane(0))
            .map(|(i, _)| i)
            .collect();
        assert!(
            (4..=6).contains(&change_samples.len()),
            "expected ~5 steps in 512 samples at 441 Hz, got {change_samples:?}"
        );
        for gap in change_samples.windows(2) {
            let samples_between = gap[1] - gap[0];
            assert!(
                (95..=105).contains(&samples_between),
                "audio steps should be one hold period apart, got {samples_between}"
            );
        }
    }

    // -- Chaos ---------------------------------------------------------------

    fn chaos_params(generator: LfoGeneratorMode, frequency: f32) -> SynthLfoParams {
        SynthLfoParams {
            frequency: PolyF32::splat(frequency),
            generator,
            ..Default::default()
        }
    }

    fn run_chaos_audio(generator: LfoGeneratorMode, frequency: f32, samples: usize) -> Vec<f32> {
        let source = LineGenerator::triangle();
        let p = chaos_params(generator, frequency);
        let mut lfo = SynthLfo::with_seed(SAMPLE_RATE, 3);
        lfo.trigger(PolyMask::all_on(), on(), 0);
        let mut collected = Vec::with_capacity(samples);
        let mut out = [PolyF32::ZERO; 256];
        while collected.len() < samples {
            lfo.process_audio(&source, &p, &mut out);
            collected.extend(out.iter().map(|v| v.lane(0)));
        }
        collected.truncate(samples);
        collected
    }

    #[test]
    fn chaos_bounded_and_nonrepeating() {
        for generator in [LfoGeneratorMode::Chaos1, LfoGeneratorMode::Chaos2] {
            let values = run_chaos_audio(generator, 100.0, 16384);
            for &value in &values {
                assert!(value.is_finite(), "{generator:?} not finite");
                assert!((-1.0..=1.0).contains(&value), "{generator:?} out of range: {value}");
            }

            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let centered: Vec<f32> = values.iter().map(|v| v - mean).collect();
            let energy: f32 = centered.iter().map(|v| v * v).sum();
            assert!(
                energy / values.len() as f32 > 1e-3,
                "{generator:?} output should not be (near-)constant"
            );

            // No dominant periodic peak: past the short-lag neighborhood
            // (lags start beyond one attractor oscillation) the normalized
            // autocorrelation must stay clearly below 1 for every candidate
            // period. A periodic signal would score ~1 at its period.
            let mut max_correlation = 0.0f32;
            let mut lag = 1000;
            while lag <= 7000 {
                let n = centered.len() - lag;
                let correlation: f32 =
                    (0..n).map(|i| centered[i] * centered[i + lag]).sum::<f32>();
                let normalized = correlation / energy;
                max_correlation = max_correlation.max(normalized);
                lag += 25;
            }
            assert!(
                max_correlation < 0.9,
                "{generator:?} looks periodic: max autocorrelation {max_correlation}"
            );
        }
    }

    #[test]
    fn chaos_rate_scales_with_frequency() {
        for generator in [LfoGeneratorMode::Chaos1, LfoGeneratorMode::Chaos2] {
            let mean_delta = |values: &[f32]| {
                values.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>()
                    / (values.len() - 1) as f32
            };
            let slow = mean_delta(&run_chaos_audio(generator, 4.0, 8192));
            let fast = mean_delta(&run_chaos_audio(generator, 16.0, 8192));
            assert!(
                fast > slow * 2.0,
                "{generator:?} rate should scale with frequency: slow {slow}, fast {fast}"
            );
        }
    }

    // -- Path ----------------------------------------------------------------

    #[test]
    fn path_returns_both_coordinates() {
        let x_shape = LineGenerator::triangle();
        let y_shape = LineGenerator::sin();
        let mut p = params(1.0);
        p.generator = LfoGeneratorMode::Path;
        let mut lfo = SynthLfo::new(SAMPLE_RATE);
        lfo.trigger(PolyMask::all_on(), on(), 0);

        // 1 Hz in 441-sample blocks: phase advances 0.01 per block. Both
        // outputs must match their shapes evaluated at the LFO phase.
        let mut distinct = 0usize;
        for _ in 0..99 {
            let (x, y) = lfo.process_control_path(&x_shape, &y_shape, &p, 441);
            let phase = lfo.phase().lane(0);
            let expected_x = x_shape.value_at_phase(phase);
            let expected_y = y_shape.value_at_phase(phase);
            assert!(
                (x.lane(0) - expected_x).abs() < 5e-3,
                "x mismatch at phase {phase}: {} vs {expected_x}",
                x.lane(0)
            );
            assert!(
                (y.lane(0) - expected_y).abs() < 5e-3,
                "y mismatch at phase {phase}: {} vs {expected_y}",
                y.lane(0)
            );
            if (x.lane(0) - y.lane(0)).abs() > 1e-3 {
                distinct += 1;
            }
        }
        assert!(distinct > 10, "x and y should follow different shapes");
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
