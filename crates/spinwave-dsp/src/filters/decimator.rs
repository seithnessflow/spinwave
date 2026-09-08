//! Halfband decimators and the multi-stage decimator stack (ports of
//! `fir_halfband_decimator.{h,cpp}`, `iir_halfband_decimator.{h,cpp}` and
//! `decimator.{h,cpp}`).
//!
//! These consolidate pairs of input samples into the vector lanes, so they
//! decimate the first stereo voice: output lanes 0/1 carry left/right, lanes
//! 2/3 mirror them (exactly like the reference, which uses them on the final
//! stereo output path).

use spinwave_poly::constants::{MAX_BUFFER_SIZE, MAX_OVERSAMPLE};
use spinwave_poly::utils::sum_split_audio;
use spinwave_poly::{PolyF32, PolyMask};

pub const FIR_NUM_TAPS: usize = 32;

/// 32-tap linear-phase FIR halfband decimator.
#[derive(Clone, Debug)]
pub struct FirHalfbandDecimator {
    memory: [PolyF32; FIR_NUM_TAPS / 2 - 1],
    taps: [PolyF32; FIR_NUM_TAPS / 2],
}

impl Default for FirHalfbandDecimator {
    fn default() -> FirHalfbandDecimator {
        FirHalfbandDecimator::new()
    }
}

impl FirHalfbandDecimator {
    pub fn new() -> FirHalfbandDecimator {
        #[allow(clippy::excessive_precision)]
        const COEFFICIENTS: [f32; FIR_NUM_TAPS] = [
            0.000088228877315364,
            0.000487010018128278,
            0.000852264975437944,
            -0.001283563593466774,
            -0.010130591831925894,
            -0.025688727779244691,
            -0.036346596505004387,
            -0.024088355516718698,
            0.012246773417129486,
            0.040021434054637831,
            0.017771298164062477,
            -0.046866403416502632,
            -0.075597513455990611,
            0.013331126342402619,
            0.202889888191404910,
            0.362615173769444080,
            0.362615173769444080,
            0.202889888191404910,
            0.013331126342402619,
            -0.075597513455990611,
            -0.046866403416502632,
            0.017771298164062477,
            0.040021434054637831,
            0.012246773417129486,
            -0.024088355516718698,
            -0.036346596505004387,
            -0.025688727779244691,
            -0.010130591831925894,
            -0.001283563593466774,
            0.000852264975437944,
            0.000487010018128278,
            0.000088228877315364,
        ];

        let mut taps = [PolyF32::ZERO; FIR_NUM_TAPS / 2];
        for (i, tap) in taps.iter_mut().enumerate() {
            *tap = PolyF32::stereo(COEFFICIENTS[2 * i], COEFFICIENTS[2 * i + 1]);
        }

        let mut decimator =
            FirHalfbandDecimator { memory: [PolyF32::ZERO; FIR_NUM_TAPS / 2 - 1], taps };
        decimator.reset(PolyMask::all_on());
        decimator
    }

    /// Note: like the reference, the mask is ignored and all lanes clear.
    pub fn reset(&mut self, _reset_mask: PolyMask) {
        self.memory = [PolyF32::ZERO; FIR_NUM_TAPS / 2 - 1];
    }

    fn save_memory(&mut self, audio_in: &[PolyF32], num_samples: usize) {
        let input_buffer_size = 2 * num_samples;
        let start_audio_index = input_buffer_size - FIR_NUM_TAPS + 2;
        for i in 0..FIR_NUM_TAPS / 2 - 1 {
            let audio_index = start_audio_index + 2 * i;
            self.memory[i] =
                PolyF32::consolidate_audio(audio_in[audio_index], audio_in[audio_index + 1]);
        }
    }

    /// Decimates `audio_in` (length `2 * audio_out.len()`) by two.
    pub fn process(&mut self, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let output_buffer_size = audio_out.len();
        assert!(output_buffer_size > FIR_NUM_TAPS / 2);
        assert!(audio_in.len() >= 2 * output_buffer_size);

        for (memory_start, out) in
            audio_out.iter_mut().take(FIR_NUM_TAPS / 2 - 1).enumerate()
        {
            let mut sum = PolyF32::ZERO;

            let num_memory = FIR_NUM_TAPS / 2 - memory_start - 1;
            let mut tap_index = 0;
            while tap_index < num_memory {
                sum = sum.mul_add(self.memory[tap_index + memory_start], self.taps[tap_index]);
                tap_index += 1;
            }

            let mut audio_index = 0;
            while tap_index < FIR_NUM_TAPS / 2 {
                let consolidated =
                    PolyF32::consolidate_audio(audio_in[audio_index], audio_in[audio_index + 1]);
                sum = sum.mul_add(consolidated, self.taps[tap_index]);
                audio_index += 2;
                tap_index += 1;
            }

            *out = sum_split_audio(sum);
        }

        let mut audio_start = 0;
        for out in audio_out.iter_mut().skip(FIR_NUM_TAPS / 2 - 1) {
            let mut sum = PolyF32::ZERO;
            let mut audio_index = audio_start;
            for tap in &self.taps {
                let consolidated =
                    PolyF32::consolidate_audio(audio_in[audio_index], audio_in[audio_index + 1]);
                sum = sum.mul_add(consolidated, *tap);
                audio_index += 2;
            }
            audio_start += 2;

            *out = sum_split_audio(sum);
        }

        self.save_memory(audio_in, output_buffer_size);
    }
}

pub const IIR_NUM_TAPS_9: usize = 2;
pub const IIR_NUM_TAPS_25: usize = 6;

#[allow(clippy::excessive_precision)]
const IIR_TAP_PAIRS_9: [[f32; 2]; IIR_NUM_TAPS_9] = [
    [0.167135116548925, 0.0413554705262319],
    [0.742130012538075, 0.3878932830211427],
];

#[allow(clippy::excessive_precision)]
const IIR_TAP_PAIRS_25: [[f32; 2]; IIR_NUM_TAPS_25] = [
    [0.093022421467960, 0.024388383731296],
    [0.312318050871736, 0.194029987625265],
    [0.548379093159427, 0.433855675727187],
    [0.737198546150414, 0.650124972769370],
    [0.872234992057129, 0.810418671775866],
    [0.975497791832324, 0.925979700943193],
];

/// Polyphase allpass IIR halfband decimator.
#[derive(Clone, Debug)]
pub struct IirHalfbandDecimator {
    sharp_cutoff: bool,
    taps9: [PolyF32; IIR_NUM_TAPS_9],
    taps25: [PolyF32; IIR_NUM_TAPS_25],
    in_memory: [PolyF32; IIR_NUM_TAPS_25],
    out_memory: [PolyF32; IIR_NUM_TAPS_25],
}

impl Default for IirHalfbandDecimator {
    fn default() -> IirHalfbandDecimator {
        IirHalfbandDecimator::new()
    }
}

impl IirHalfbandDecimator {
    pub fn new() -> IirHalfbandDecimator {
        let mut taps9 = [PolyF32::ZERO; IIR_NUM_TAPS_9];
        for (tap, pair) in taps9.iter_mut().zip(&IIR_TAP_PAIRS_9) {
            *tap = PolyF32::stereo(pair[0], pair[1]);
        }
        let mut taps25 = [PolyF32::ZERO; IIR_NUM_TAPS_25];
        for (tap, pair) in taps25.iter_mut().zip(&IIR_TAP_PAIRS_25) {
            *tap = PolyF32::stereo(pair[0], pair[1]);
        }

        let mut decimator = IirHalfbandDecimator {
            sharp_cutoff: false,
            taps9,
            taps25,
            in_memory: [PolyF32::ZERO; IIR_NUM_TAPS_25],
            out_memory: [PolyF32::ZERO; IIR_NUM_TAPS_25],
        };
        decimator.reset(PolyMask::all_on());
        decimator
    }

    pub fn set_sharp_cutoff(&mut self, sharp_cutoff: bool) {
        self.sharp_cutoff = sharp_cutoff;
    }

    /// Note: like the reference, the mask is ignored and all lanes clear.
    pub fn reset(&mut self, _reset_mask: PolyMask) {
        self.in_memory = [PolyF32::ZERO; IIR_NUM_TAPS_25];
        self.out_memory = [PolyF32::ZERO; IIR_NUM_TAPS_25];
    }

    /// Decimates `audio_in` (length `2 * audio_out.len()`) by two.
    pub fn process(&mut self, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let mut taps = [PolyF32::ZERO; IIR_NUM_TAPS_25];
        let num_taps = if self.sharp_cutoff {
            taps.copy_from_slice(&self.taps25);
            IIR_NUM_TAPS_25
        } else {
            taps[..IIR_NUM_TAPS_9].copy_from_slice(&self.taps9);
            IIR_NUM_TAPS_9
        };
        let taps = &taps[..num_taps];

        let output_buffer_size = audio_out.len();
        assert!(audio_in.len() >= 2 * output_buffer_size);

        for (i, out) in audio_out.iter_mut().enumerate() {
            let audio_in_index = 2 * i;
            let mut result = PolyF32::consolidate_audio(
                audio_in[audio_in_index],
                audio_in[audio_in_index + 1],
            );
            for (tap_index, tap) in taps.iter().enumerate() {
                let delta = result - self.out_memory[tap_index];
                let new_result = self.in_memory[tap_index].mul_add(*tap, delta);
                self.in_memory[tap_index] = result;
                self.out_memory[tap_index] = new_result;
                result = new_result;
            }

            *out = sum_split_audio(result) * 0.5;
        }
    }
}

/// Stack of IIR halfband stages selected from the input/output sample-rate
/// ratio (port of the `Decimator` router).
#[derive(Clone, Debug)]
pub struct Decimator {
    max_stages: usize,
    num_stages: Option<usize>,
    stages: Vec<IirHalfbandDecimator>,
    scratch_a: Vec<PolyF32>,
    scratch_b: Vec<PolyF32>,
}

impl Decimator {
    pub fn new(max_stages: usize) -> Decimator {
        let capacity = MAX_BUFFER_SIZE * MAX_OVERSAMPLE;
        Decimator {
            max_stages,
            num_stages: None,
            stages: (0..max_stages).map(|_| IirHalfbandDecimator::new()).collect(),
            scratch_a: vec![PolyF32::ZERO; capacity],
            scratch_b: vec![PolyF32::ZERO; capacity],
        }
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        for stage in &mut self.stages {
            stage.reset(reset_mask);
        }
    }

    /// Decimates `audio_in` at `input_sample_rate` down to `audio_out` at
    /// `output_sample_rate`. The input length must be
    /// `audio_out.len() << num_stages` where each stage halves the rate.
    pub fn process(
        &mut self,
        audio_in: &[PolyF32],
        input_sample_rate: u32,
        output_sample_rate: u32,
        audio_out: &mut [PolyF32],
    ) {
        let mut num_stages = 0usize;
        let mut rate = input_sample_rate;
        while rate > output_sample_rate {
            num_stages += 1;
            rate /= 2;
        }
        assert!(num_stages <= self.max_stages);
        assert_eq!(rate, output_sample_rate);
        assert_eq!(audio_in.len(), audio_out.len() << num_stages);

        if num_stages == 0 {
            audio_out.copy_from_slice(audio_in);
            return;
        }

        if self.num_stages != Some(num_stages) {
            for stage in self.stages.iter_mut().take(num_stages) {
                stage.reset(PolyMask::all_on());
            }
            self.num_stages = Some(num_stages);
            for (i, stage) in self.stages.iter_mut().enumerate().take(num_stages) {
                stage.set_sharp_cutoff(i == num_stages - 1);
            }
        }

        assert!(audio_in.len() <= self.scratch_a.len());
        let mut current_len = audio_in.len();
        self.scratch_a[..current_len].copy_from_slice(audio_in);

        for i in 0..num_stages {
            let out_len = current_len / 2;
            self.stages[i]
                .process(&self.scratch_a[..current_len], &mut self.scratch_b[..out_len]);
            std::mem::swap(&mut self.scratch_a, &mut self.scratch_b);
            current_len = out_len;
        }

        audio_out.copy_from_slice(&self.scratch_a[..current_len]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo_ramp_sine(len: usize, freq: f32, sample_rate: f32) -> Vec<PolyF32> {
        (0..len)
            .map(|i| {
                let phase = 2.0 * core::f32::consts::PI * freq * i as f32 / sample_rate;
                PolyF32::splat(phase.sin())
            })
            .collect()
    }

    #[test]
    fn fir_passes_dc_and_kills_nyquist() {
        let mut decimator = FirHalfbandDecimator::new();
        let out_len = 64;
        let dc = vec![PolyF32::ONE; 2 * out_len];
        let mut output = vec![PolyF32::ZERO; out_len];
        decimator.process(&dc, &mut output);
        // The reference taps sum to ~0.8606, so DC settles there.
        let settled = output[out_len - 1].lane(0);
        assert!((settled - 0.8606).abs() < 0.01, "DC settled at {settled}");

        let mut decimator = FirHalfbandDecimator::new();
        let nyquist: Vec<PolyF32> = (0..2 * out_len)
            .map(|i| PolyF32::splat(if i % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        decimator.process(&nyquist, &mut output);
        let residue = output[out_len - 1].lane(0).abs();
        assert!(residue < 0.01, "Nyquist residue {residue}");
    }

    #[test]
    fn fir_memory_carries_across_blocks() {
        // A continuous low-frequency sine decimated block by block should
        // stay smooth: no discontinuity at block borders.
        let sample_rate = 88200.0;
        let out_len = 64;
        let input = stereo_ramp_sine(4 * out_len, 200.0, sample_rate);
        let mut decimator = FirHalfbandDecimator::new();
        let mut out_a = vec![PolyF32::ZERO; out_len];
        let mut out_b = vec![PolyF32::ZERO; out_len];
        decimator.process(&input[..2 * out_len], &mut out_a);
        decimator.process(&input[2 * out_len..], &mut out_b);
        let step = (out_b[0].lane(0) - out_a[out_len - 1].lane(0)).abs();
        assert!(step < 0.05, "discontinuity {step} at block border");
    }

    #[test]
    fn iir_passes_dc_and_kills_nyquist() {
        for sharp in [false, true] {
            let mut decimator = IirHalfbandDecimator::new();
            decimator.set_sharp_cutoff(sharp);
            let out_len = 64;
            let dc = vec![PolyF32::ONE; 2 * out_len];
            let mut output = vec![PolyF32::ZERO; out_len];
            decimator.process(&dc, &mut output);
            let settled = output[out_len - 1].lane(0);
            assert!((settled - 1.0).abs() < 0.02, "sharp={sharp} DC settled at {settled}");

            let mut decimator = IirHalfbandDecimator::new();
            decimator.set_sharp_cutoff(sharp);
            let nyquist: Vec<PolyF32> = (0..2 * out_len)
                .map(|i| PolyF32::splat(if i % 2 == 0 { 1.0 } else { -1.0 }))
                .collect();
            decimator.process(&nyquist, &mut output);
            let residue = output[out_len - 1].lane(0).abs();
            assert!(residue < 0.05, "sharp={sharp} Nyquist residue {residue}");
        }
    }

    #[test]
    fn iir_reset_clears_state() {
        let mut decimator = IirHalfbandDecimator::new();
        let out_len = 32;
        let dc = vec![PolyF32::ONE; 2 * out_len];
        let mut output = vec![PolyF32::ZERO; out_len];
        decimator.process(&dc, &mut output);
        decimator.reset(PolyMask::all_on());
        let silence = vec![PolyF32::ZERO; 2 * out_len];
        decimator.process(&silence, &mut output);
        for value in &output {
            assert_eq!(value.lane(0), 0.0);
        }
    }

    #[test]
    fn decimator_stack_preserves_low_frequency_shape() {
        let mut decimator = Decimator::new(3);
        let out_len = 128;
        let num_stages = 2;
        let input_rate = 4 * 44100;
        let input = stereo_ramp_sine(out_len << num_stages, 440.0, input_rate as f32);
        let mut output = vec![PolyF32::ZERO; out_len];
        decimator.process(&input, input_rate, 44100, &mut output);

        // Second half (past the filter transient) should still be a sine of
        // roughly unity amplitude.
        let mut sum = 0.0f32;
        for value in &output[out_len / 2..] {
            assert!(value.is_finite());
            sum += value.lane(0) * value.lane(0);
        }
        let rms = (sum / (out_len / 2) as f32).sqrt();
        let expected = 1.0 / core::f32::consts::SQRT_2;
        assert!((rms - expected).abs() < 0.1 * expected, "rms {rms} vs {expected}");
    }

    #[test]
    fn decimator_stack_copies_at_equal_rates() {
        let mut decimator = Decimator::new(2);
        let input = stereo_ramp_sine(64, 1000.0, 44100.0);
        let mut output = vec![PolyF32::ZERO; 64];
        decimator.process(&input, 44100, 44100, &mut output);
        for i in 0..64 {
            assert_eq!(output[i].lane(0), input[i].lane(0));
        }
    }
}
