//! Feedback-network reverb (port of Vital's `Reverb`).
//!
//! Pre low/high one-pole filtering feeds a 16-way network (4 containers of
//! 4 lanes): per-line Schroeder allpasses, chorused feedback-delay reads,
//! high/low shelving damping and a Householder-style mixing of container
//! rows and adjacent lanes. The network output is pushed through a stereo
//! pre-delay memory before the wet/dry mix.

use spinwave_poly::constants::{MAX_SAMPLE_RATE, PI};
use spinwave_poly::utils::{interpolate, polynomial_interpolation_matrix};
use spinwave_poly::{math, Matrix, PolyF32, PolyU32};

use crate::memory::StereoMemory;

use super::lanes::first_voice_mask;
use super::one_pole::OnePole;
use crate::filters::filter_state::{db_to_magnitude_precise, midi_note_to_frequency_precise};

pub const T60_AMPLITUDE: f32 = 0.001;
pub const ALLPASS_FEEDBACK: f32 = 0.6;
pub const MIN_DELAY: f32 = 3.0;

pub const BASE_SAMPLE_RATE: usize = 44100;
pub const NETWORK_SIZE: usize = 16;
const BASE_FEEDBACK_BITS: usize = 14;
const EXTRA_LOOKUP_SAMPLE: usize = 4;
const BASE_ALLPASS_BITS: usize = 10;
const NETWORK_CONTAINERS: usize = NETWORK_SIZE / 4;
const MIN_SIZE_POWER: f32 = -3.0;
const MAX_SIZE_POWER: i32 = 1;
const SIZE_POWER_RANGE: f32 = MAX_SIZE_POWER as f32 - MIN_SIZE_POWER;

const MAX_CHORUS_DRIFT: f32 = 2500.0;
const MIN_DECAY_TIME: f32 = 0.1;
const MAX_DECAY_TIME: f32 = 100.0;
const MAX_CHORUS_FREQUENCY: f32 = 16.0;
const SAMPLE_DELAY_MULTIPLIER: f32 = 0.05;
const SAMPLE_INCREMENT_MULTIPLIER: f32 = 0.05;

const ALLPASS_DELAYS: [[u32; 4]; NETWORK_CONTAINERS] = [
    [1001, 799, 933, 876],
    [895, 807, 907, 853],
    [957, 1019, 711, 567],
    [833, 779, 663, 997],
];

const FEEDBACK_DELAYS: [[f32; 4]; NETWORK_CONTAINERS] = [
    [6753.2, 9278.4, 7704.5, 11328.5],
    [9701.12, 5512.5, 8480.45, 5638.65],
    [3120.73, 3429.5, 3626.37, 7713.52],
    [4521.54, 6518.97, 5265.56, 5630.25],
];

/// Block-rate reverb parameters.
#[derive(Clone, Copy, Debug)]
pub struct ReverbParams {
    /// T60 decay time in seconds, clamped to [0.1, 100].
    pub decay_time: PolyF32,
    /// Input high-pass cutoff (MIDI note), removes lows before the network.
    pub pre_low_cutoff: PolyF32,
    /// Input low-pass cutoff (MIDI note), removes highs before the network.
    pub pre_high_cutoff: PolyF32,
    /// In-network low shelf cutoff (MIDI note).
    pub low_cutoff: PolyF32,
    /// In-network low shelf gain in dB, clamped to [-24, 0].
    pub low_gain: PolyF32,
    /// In-network high shelf cutoff (MIDI note).
    pub high_cutoff: PolyF32,
    /// In-network high shelf gain in dB, clamped to [-24, 0].
    pub high_gain: PolyF32,
    /// Chorusing amount in [0, 1] (only lane 0 is read).
    pub chorus_amount: PolyF32,
    /// Chorusing rate in Hz (only lane 0 is read).
    pub chorus_frequency: PolyF32,
    /// Room size in [0, 1], maps to a 2^[-3, 1] length multiplier.
    pub size: PolyF32,
    /// Pre-delay in seconds.
    pub delay: PolyF32,
    /// Wet amount in [0, 1]; dry/wet uses an equal-power fade.
    pub wet: PolyF32,
}

impl Default for ReverbParams {
    fn default() -> ReverbParams {
        ReverbParams {
            decay_time: PolyF32::splat(1.0),
            pre_low_cutoff: PolyF32::ZERO,
            pre_high_cutoff: PolyF32::splat(130.0),
            low_cutoff: PolyF32::ZERO,
            low_gain: PolyF32::ZERO,
            high_cutoff: PolyF32::splat(130.0),
            high_gain: PolyF32::ZERO,
            chorus_amount: PolyF32::ZERO,
            chorus_frequency: PolyF32::splat(1.0),
            size: PolyF32::splat(0.5),
            delay: PolyF32::ZERO,
            wet: PolyF32::ZERO,
        }
    }
}

#[inline(always)]
fn pow_exact(base: f32, exponent: PolyF32) -> PolyF32 {
    exponent.map(|e| base.powf(e))
}

#[inline(always)]
fn transpose4(values: [PolyF32; 4]) -> [PolyF32; 4] {
    let mut matrix = Matrix::new(values[0], values[1], values[2], values[3]);
    matrix.transpose();
    matrix.rows
}

pub struct Reverb {
    memory: StereoMemory,
    /// Per container: `max_allpass_size` vectors stored as interleaved lanes.
    allpass_lookups: [Vec<f32>; NETWORK_CONTAINERS],
    /// 16 mono rings, each `max_feedback_size + 4` samples (index 0 and the
    /// last 3 mirror the ring for interpolation reads).
    feedback_memories: Vec<Vec<f32>>,
    decays: [PolyF32; NETWORK_CONTAINERS],

    low_shelf_filters: [OnePole; NETWORK_CONTAINERS],
    high_shelf_filters: [OnePole; NETWORK_CONTAINERS],
    low_pre_filter: OnePole,
    high_pre_filter: OnePole,

    low_pre_coefficient: PolyF32,
    high_pre_coefficient: PolyF32,
    low_coefficient: PolyF32,
    low_amplitude: PolyF32,
    high_coefficient: PolyF32,
    high_amplitude: PolyF32,

    chorus_phase: f32,
    chorus_amount: PolyF32,
    sample_delay: PolyF32,
    sample_delay_increment: PolyF32,
    dry: PolyF32,
    wet: PolyF32,
    write_index: usize,

    sample_rate: f32,
    max_allpass_size: usize,
    max_feedback_size: usize,
    feedback_mask: usize,
    allpass_mask: u32,
    poly_allpass_mask: usize,
}

impl Reverb {
    pub fn new(sample_rate: f32) -> Reverb {
        let mut reverb = Reverb {
            memory: StereoMemory::new(MAX_SAMPLE_RATE as usize),
            allpass_lookups: core::array::from_fn(|_| Vec::new()),
            feedback_memories: Vec::new(),
            decays: [PolyF32::ZERO; NETWORK_CONTAINERS],
            low_shelf_filters: [OnePole::new(); NETWORK_CONTAINERS],
            high_shelf_filters: [OnePole::new(); NETWORK_CONTAINERS],
            low_pre_filter: OnePole::new(),
            high_pre_filter: OnePole::new(),
            low_pre_coefficient: PolyF32::splat(0.1),
            high_pre_coefficient: PolyF32::splat(0.1),
            low_coefficient: PolyF32::splat(0.1),
            low_amplitude: PolyF32::ZERO,
            high_coefficient: PolyF32::splat(0.1),
            high_amplitude: PolyF32::ZERO,
            chorus_phase: 0.0,
            chorus_amount: PolyF32::ZERO,
            sample_delay: PolyF32::splat(MIN_DELAY),
            sample_delay_increment: PolyF32::ZERO,
            dry: PolyF32::ZERO,
            wet: PolyF32::ZERO,
            write_index: 0,
            sample_rate,
            max_allpass_size: 0,
            max_feedback_size: 0,
            feedback_mask: 0,
            allpass_mask: 0,
            poly_allpass_mask: 0,
        };
        reverb.setup_buffers_for_sample_rate(sample_rate);
        reverb
    }

    fn sample_rate_ratio(&self, sample_rate: f32) -> f32 {
        sample_rate / BASE_SAMPLE_RATE as f32
    }

    fn buffer_scale(&self, sample_rate: f32) -> usize {
        let ratio = self.sample_rate_ratio(sample_rate);
        let mut scale = 1usize;
        while (scale as f32) < ratio {
            scale *= 2;
        }
        scale
    }

    fn setup_buffers_for_sample_rate(&mut self, sample_rate: f32) {
        let buffer_scale = self.buffer_scale(sample_rate);
        let max_feedback_size =
            buffer_scale * (1usize << (BASE_FEEDBACK_BITS as i32 + MAX_SIZE_POWER));
        if self.max_feedback_size == max_feedback_size {
            return;
        }

        self.max_feedback_size = max_feedback_size;
        self.feedback_mask = max_feedback_size - 1;
        self.feedback_memories = (0..NETWORK_SIZE)
            .map(|_| vec![0.0f32; max_feedback_size + EXTRA_LOOKUP_SAMPLE])
            .collect();

        self.max_allpass_size = buffer_scale * (1usize << BASE_ALLPASS_BITS);
        self.poly_allpass_mask = self.max_allpass_size - 1;
        self.allpass_mask = (self.max_allpass_size * 4 - 1) as u32;
        for lookup in &mut self.allpass_lookups {
            *lookup = vec![0.0f32; self.max_allpass_size * 4];
        }

        self.write_index &= self.feedback_mask;
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.setup_buffers_for_sample_rate(sample_rate);
    }

    /// Full reset that also primes the chorus amount from the current
    /// parameters exactly like `Reverb::hardReset` (reverb.cpp:376):
    /// `chorus_amount_ = clamp(chorus_amount, 0, 1) * kMaxChorusDrift`, so
    /// the first block after the reset does not ramp the chorus depth up
    /// from zero. Prefer this over [`Reverb::hard_reset`] when the block
    /// parameters are at hand.
    pub fn hard_reset_with(&mut self, params: &ReverbParams) {
        self.hard_reset();
        self.chorus_amount =
            PolyF32::splat(params.chorus_amount.lane(0).clamp(0.0, 1.0) * MAX_CHORUS_DRIFT);
    }

    /// Clears all state. Unlike the C++ this cannot read the chorus amount
    /// input, so it leaves `chorus_amount` untouched; see
    /// [`Reverb::hard_reset_with`] for the faithful priming.
    pub fn hard_reset(&mut self) {
        self.wet = PolyF32::ZERO;
        self.dry = PolyF32::ZERO;
        self.low_pre_filter.hard_reset();
        self.high_pre_filter.hard_reset();

        for i in 0..NETWORK_CONTAINERS {
            self.low_shelf_filters[i].hard_reset();
            self.high_shelf_filters[i].hard_reset();
            self.decays[i] = PolyF32::ZERO;
        }

        for lookup in &mut self.allpass_lookups {
            lookup.fill(0.0);
        }
        for memory in &mut self.feedback_memories {
            memory.fill(0.0);
        }
        self.memory.clear_all();
    }

    #[inline(always)]
    fn read_feedback(&self, container: usize, offset: PolyF32) -> PolyF32 {
        let write_offset = PolyF32::splat(self.write_index as f32) - offset;
        let floored_offset = write_offset.floor();
        let t = write_offset - floored_offset;
        let interpolation_matrix = polynomial_interpolation_matrix(t);
        let indices =
            floored_offset.to_i32_round() & PolyU32::splat(self.feedback_mask as u32);

        let row = |lane: usize| {
            let buffer = &self.feedback_memories[container * 4 + lane];
            // The lookup view starts one sample into the ring.
            let start = indices.lane(lane) as usize + 1;
            PolyF32::from_lanes([
                buffer[start],
                buffer[start + 1],
                buffer[start + 2],
                buffer[start + 3],
            ])
        };
        let mut value_matrix = Matrix::new(row(0), row(1), row(2), row(3));
        value_matrix.transpose();
        interpolation_matrix.multiply_and_sum_rows(&value_matrix)
    }

    #[inline(always)]
    fn read_allpass(&self, container: usize, offset: PolyU32) -> PolyF32 {
        let base = PolyU32::splat((self.write_index * 4) as u32);
        let indices = (base - offset) & PolyU32::splat(self.allpass_mask);
        let buffer = &self.allpass_lookups[container];
        PolyF32::from_lanes([
            buffer[indices.lane(0) as usize],
            buffer[indices.lane(1) as usize],
            buffer[indices.lane(2) as usize],
            buffer[indices.lane(3) as usize],
        ])
    }

    #[inline(always)]
    fn write_allpass(&mut self, container: usize, value: PolyF32) {
        let index = (self.write_index & self.poly_allpass_mask) * 4;
        self.allpass_lookups[container][index..index + 4].copy_from_slice(&value.to_lanes());
    }

    fn wrap_feedback_buffers(&mut self) {
        let max = self.max_feedback_size;
        for buffer in &mut self.feedback_memories {
            buffer[0] = buffer[max];
            buffer[max + 1] = buffer[1];
            buffer[max + 2] = buffer[2];
            buffer[max + 3] = buffer[3];
        }
    }

    pub fn process(&mut self, params: &ReverbParams, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let num_samples = audio_in.len();
        assert_eq!(audio_out.len(), num_samples);
        if num_samples == 0 {
            return;
        }

        self.wrap_feedback_buffers();

        let tick_increment = 1.0 / num_samples as f32;

        let mut current_dry = self.dry;
        let mut current_wet = self.wet;
        let current_low_pre_coefficient = self.low_pre_coefficient;
        let current_high_pre_coefficient = self.high_pre_coefficient;
        let current_low_coefficient = self.low_coefficient;
        let current_low_amplitude = self.low_amplitude;
        let mut current_high_coefficient = self.high_coefficient;
        let mut current_high_amplitude = self.high_amplitude;

        let wet_in = params.wet.clamp(0.0, 1.0);
        self.wet = math::equal_power_fade(wet_in);
        self.dry = math::equal_power_fade_inverse(wet_in);
        let delta_wet = (self.wet - current_wet) * tick_increment;
        let delta_dry = (self.dry - current_dry) * tick_increment;

        let sample_rate = self.sample_rate;
        let buffer_scale = self.buffer_scale(sample_rate);
        let sample_rate_ratio = self.sample_rate_ratio(sample_rate);

        let low_pre_cutoff_frequency =
            midi_note_to_frequency_precise(params.pre_low_cutoff.clamp(0.0, 130.0));
        self.low_pre_coefficient =
            OnePole::compute_coefficient(low_pre_cutoff_frequency, sample_rate);

        let high_pre_cutoff_frequency =
            midi_note_to_frequency_precise(params.pre_high_cutoff.clamp(0.0, 130.0));
        self.high_pre_coefficient =
            OnePole::compute_coefficient(high_pre_cutoff_frequency, sample_rate);

        let low_cutoff_frequency =
            midi_note_to_frequency_precise(params.low_cutoff.clamp(0.0, 130.0));
        self.low_coefficient = OnePole::compute_coefficient(low_cutoff_frequency, sample_rate);

        let high_cutoff_frequency =
            midi_note_to_frequency_precise(params.high_cutoff.clamp(0.0, 130.0));
        self.high_coefficient = OnePole::compute_coefficient(high_cutoff_frequency, sample_rate);
        let delta_high_coefficient =
            (self.high_coefficient - current_high_coefficient) * tick_increment;

        let low_gain = params.low_gain.clamp(-24.0, 0.0);
        self.low_amplitude = PolyF32::ONE - db_to_magnitude_precise(low_gain);
        let high_gain = params.high_gain.clamp(-24.0, 0.0);
        self.high_amplitude = db_to_magnitude_precise(high_gain);
        let delta_high_amplitude = (self.high_amplitude - current_high_amplitude) * tick_increment;

        let size = params.size.clamp(0.0, 1.0);
        let size_mult = math::pow(
            PolyF32::splat(2.0),
            size * SIZE_POWER_RANGE + MIN_SIZE_POWER,
        );

        let decay_samples =
            params.decay_time.clamp(MIN_DECAY_TIME, MAX_DECAY_TIME) * BASE_SAMPLE_RATE as f32;
        let decay_period = size_mult / decay_samples;
        let mut current_decay: [PolyF32; NETWORK_CONTAINERS] = self.decays;
        let mut delta_decay = [PolyF32::ZERO; NETWORK_CONTAINERS];
        let mut feedback_delays = [PolyF32::ZERO; NETWORK_CONTAINERS];
        for i in 0..NETWORK_CONTAINERS {
            feedback_delays[i] = PolyF32::from_lanes(FEEDBACK_DELAYS[i]);
            self.decays[i] = pow_exact(T60_AMPLITUDE, feedback_delays[i] * decay_period);
            delta_decay[i] = (self.decays[i] - current_decay[i]) * tick_increment;
        }

        let mut delay_offset = PolyU32::from_lanes([0, u32::MAX, u32::MAX - 1, u32::MAX - 2]);
        delay_offset += PolyU32::splat(4);

        let mut allpass_offsets = [PolyU32::ZERO; NETWORK_CONTAINERS];
        for i in 0..NETWORK_CONTAINERS {
            let delays = PolyU32::from_lanes(ALLPASS_DELAYS[i]);
            allpass_offsets[i] =
                (delays * PolyU32::splat((buffer_scale * 4) as u32) + delay_offset).swap_stereo();
        }

        let chorus_frequency = params.chorus_frequency.lane(0).clamp(0.0, MAX_CHORUS_FREQUENCY);
        let chorus_phase_increment = chorus_frequency / sample_rate;

        let network_offset = 2.0 * PI / NETWORK_SIZE as f32;
        let phase_offset = PolyF32::from_lanes([0.0, 1.0, 2.0, 3.0]) * network_offset;
        let container_phase = phase_offset + self.chorus_phase * 2.0 * PI;
        self.chorus_phase += num_samples as f32 * chorus_phase_increment;
        self.chorus_phase -= self.chorus_phase.floor();

        let chorus_increment_real =
            PolyF32::splat((chorus_phase_increment * 2.0 * PI).cos());
        let chorus_increment_imaginary =
            PolyF32::splat((chorus_phase_increment * 2.0 * PI).sin());
        let mut current_chorus_real = container_phase.map(f32::cos);
        let mut current_chorus_imaginary = container_phase.map(f32::sin);

        let mut delays = [PolyF32::ZERO; NETWORK_CONTAINERS];
        for i in 0..NETWORK_CONTAINERS {
            delays[i] = size_mult * feedback_delays[i] * sample_rate_ratio;
        }

        let mut current_chorus_amount = self.chorus_amount;
        self.chorus_amount = PolyF32::splat(
            params.chorus_amount.lane(0).clamp(0.0, 1.0) * MAX_CHORUS_DRIFT * sample_rate_ratio,
        );
        for delay in delays {
            self.chorus_amount = self.chorus_amount.min(delay - 32.0);
        }
        let delta_chorus_amount = (self.chorus_amount - current_chorus_amount) * tick_increment;
        current_chorus_amount *= size_mult;

        let mut current_sample_delay = self.sample_delay;
        let mut current_delay_increment = self.sample_delay_increment;
        let end_target = current_sample_delay + current_delay_increment * num_samples as f32;
        let mut target_delay =
            (params.delay * sample_rate).clamp(MIN_DELAY, MAX_SAMPLE_RATE as f32);
        target_delay = interpolate(
            self.sample_delay,
            target_delay,
            PolyF32::splat(SAMPLE_DELAY_MULTIPLIER),
        );
        let makeup_delay = target_delay - end_target;
        let delta_delay_increment = makeup_delay
            / (0.5 * num_samples as f32 * num_samples as f32)
            * SAMPLE_INCREMENT_MULTIPLIER;

        for i in 0..num_samples {
            current_chorus_amount += delta_chorus_amount;
            current_chorus_real = current_chorus_real * chorus_increment_real
                - current_chorus_imaginary * chorus_increment_imaginary;
            current_chorus_imaginary = current_chorus_imaginary * chorus_increment_real
                + current_chorus_real * chorus_increment_imaginary;

            let feedback_offsets = [
                delays[0] + current_chorus_real * current_chorus_amount,
                delays[1] - current_chorus_real * current_chorus_amount,
                delays[2] + current_chorus_imaginary * current_chorus_amount,
                delays[3] - current_chorus_imaginary * current_chorus_amount,
            ];
            let mut feedback_reads = [PolyF32::ZERO; NETWORK_CONTAINERS];
            for c in 0..NETWORK_CONTAINERS {
                feedback_reads[c] = self.read_feedback(c, feedback_offsets[c]);
            }

            let mut input = audio_in[i] & first_voice_mask();
            input += input.swap_voices();
            let high_filtered = self
                .high_pre_filter
                .tick_basic(input, current_high_pre_coefficient);
            let filtered_input = self
                .low_pre_filter
                .tick_basic(input, current_low_pre_coefficient)
                - high_filtered;
            let scaled_input = filtered_input * 0.25;

            let mut allpass_outputs = [PolyF32::ZERO; NETWORK_CONTAINERS];
            for c in 0..NETWORK_CONTAINERS {
                let allpass_read = self.read_allpass(c, allpass_offsets[c]);
                let allpass_delay_input = feedback_reads[c] - allpass_read * ALLPASS_FEEDBACK;
                self.write_allpass(c, scaled_input + allpass_delay_input);
                allpass_outputs[c] = allpass_read + allpass_delay_input * ALLPASS_FEEDBACK;
            }

            let total_rows = allpass_outputs[0]
                + allpass_outputs[1]
                + allpass_outputs[2]
                + allpass_outputs[3];
            let other_feedback = PolyF32::splat(total_rows.sum_lanes() * 0.25)
                .mul_add(total_rows, PolyF32::splat(-0.5));

            let mut writes = [PolyF32::ZERO; NETWORK_CONTAINERS];
            for c in 0..NETWORK_CONTAINERS {
                writes[c] = other_feedback + allpass_outputs[c];
            }

            let transposed = transpose4(allpass_outputs);
            let adjacent_feedback =
                (transposed[0] + transposed[1] + transposed[2] + transposed[3]) * -0.5;

            for (c, write) in writes.iter_mut().enumerate() {
                *write += adjacent_feedback.lane(c);
                let high_filtered =
                    self.high_shelf_filters[c].tick_basic(*write, current_high_coefficient);
                *write = high_filtered + current_high_amplitude * (*write - high_filtered);
                let low_filtered =
                    self.low_shelf_filters[c].tick_basic(*write, current_low_coefficient);
                *write -= low_filtered * current_low_amplitude;
            }

            let mut stores = [PolyF32::ZERO; NETWORK_CONTAINERS];
            for c in 0..NETWORK_CONTAINERS {
                current_decay[c] += delta_decay[c];
                stores[c] = current_decay[c] * writes[c];
                let lanes = stores[c].to_lanes();
                for (lane, &value) in lanes.iter().enumerate() {
                    self.feedback_memories[c * 4 + lane][self.write_index + 1] = value;
                }
            }

            self.write_index = (self.write_index + 1) & self.feedback_mask;

            let total_allpass = stores[0] + stores[1] + stores[2] + stores[3];
            let other_feedback_allpass = PolyF32::splat(total_allpass.sum_lanes() * 0.25)
                .mul_add(total_allpass, PolyF32::splat(-0.5));

            let mut feed_forwards = [PolyF32::ZERO; NETWORK_CONTAINERS];
            for c in 0..NETWORK_CONTAINERS {
                feed_forwards[c] = other_feedback_allpass + stores[c];
            }

            let transposed_stores = transpose4(stores);
            let adjacent_feedback_allpass = (transposed_stores[0]
                + transposed_stores[1]
                + transposed_stores[2]
                + transposed_stores[3])
                * -0.5;
            for (c, feed_forward) in feed_forwards.iter_mut().enumerate() {
                *feed_forward += adjacent_feedback_allpass.lane(c);
            }

            let mut total = writes[0] + writes[1] + writes[2] + writes[3];
            total += (feed_forwards[0] * current_decay[0]
                + feed_forwards[1] * current_decay[1]
                + feed_forwards[2] * current_decay[2]
                + feed_forwards[3] * current_decay[3])
                * 0.125;

            self.memory.push(total + total.swap_voices());
            audio_out[i] =
                current_wet * self.memory.get(current_sample_delay) + current_dry * input;

            current_delay_increment += delta_delay_increment;
            current_sample_delay += current_delay_increment;
            current_sample_delay =
                current_sample_delay.clamp(MIN_DELAY, MAX_SAMPLE_RATE as f32);
            current_dry += delta_dry;
            current_wet += delta_wet;
            current_high_coefficient += delta_high_coefficient;
            current_high_amplitude += delta_high_amplitude;
        }

        self.sample_delay_increment = current_delay_increment;
        self.sample_delay = current_sample_delay;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 44100.0;
    const BLOCK: usize = 128;

    #[test]
    fn impulse_response_decays_and_is_stereo() {
        let mut reverb = Reverb::new(SAMPLE_RATE);
        let params = ReverbParams {
            wet: PolyF32::ONE,
            decay_time: PolyF32::splat(0.8),
            ..ReverbParams::default()
        };

        let window = (SAMPLE_RATE * 0.5) as usize / BLOCK; // 0.5 s in blocks
        let mut window_rms = Vec::new();
        let mut stereo_difference = 0.0f32;
        let mut output = vec![PolyF32::ZERO; BLOCK];
        let mut input = vec![PolyF32::ZERO; BLOCK];

        // One silent block first: like the reference, the pre-filter
        // coefficients only take their real values after the first block.
        reverb.process(&params, &input, &mut output);
        input[0] = PolyF32::ONE;

        for w in 0..4 {
            let mut sum = 0.0f64;
            let mut count = 0usize;
            for _ in 0..window {
                reverb.process(&params, &input, &mut output);
                input[0] = PolyF32::ZERO;
                for sample in &output {
                    assert!(sample.is_finite());
                    let l = sample.lane(0);
                    let r = sample.lane(1);
                    sum += (l as f64) * (l as f64) + (r as f64) * (r as f64);
                    count += 2;
                    stereo_difference = stereo_difference.max((l - r).abs());
                }
            }
            let rms = ((sum / count as f64).sqrt()) as f32;
            if w > 0 {
                // Skip the onset window; the tail must strictly decay.
                window_rms.push(rms);
            }
        }

        assert!(window_rms[0] > 1e-6, "reverb produced no tail");
        for pair in window_rms.windows(2) {
            assert!(
                pair[1] < pair[0],
                "tail RMS not decreasing: {window_rms:?}"
            );
        }
        assert!(stereo_difference > 1e-5, "reverb output is not stereo");
    }

    #[test]
    fn sine_output_finite_with_chorus_and_shelves() {
        let mut reverb = Reverb::new(SAMPLE_RATE);
        let params = ReverbParams {
            wet: PolyF32::splat(0.5),
            decay_time: PolyF32::splat(2.0),
            chorus_amount: PolyF32::splat(0.6),
            chorus_frequency: PolyF32::splat(3.0),
            low_cutoff: PolyF32::splat(50.0),
            low_gain: PolyF32::splat(-6.0),
            high_cutoff: PolyF32::splat(90.0),
            high_gain: PolyF32::splat(-6.0),
            size: PolyF32::splat(0.7),
            delay: PolyF32::splat(0.01),
            ..ReverbParams::default()
        };
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..300 {
            let input: Vec<PolyF32> = (0..BLOCK)
                .map(|i| {
                    let t = (block * BLOCK + i) as f32 / SAMPLE_RATE;
                    PolyF32::splat((2.0 * core::f32::consts::PI * 220.0 * t).sin() * 0.5)
                })
                .collect();
            reverb.process(&params, &input, &mut output);
            for sample in &output {
                assert!(sample.is_finite());
                assert!(sample.abs().lane(0) < 16.0);
            }
        }
    }
}
