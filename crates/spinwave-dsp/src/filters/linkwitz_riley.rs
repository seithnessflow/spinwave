//! Linkwitz-Riley crossover filter (port of `linkwitz_riley_filter.{h,cpp}`).
//!
//! Splits the input into low and high bands with cascaded Butterworth
//! biquads; low + high recombine allpass-flat.

use spinwave_poly::constants::{PI, SQRT_2};
use spinwave_poly::{PolyF32, PolyMask};

const NUM_BANDS: usize = 2;
const LOW: usize = 0;
const HIGH: usize = 1;

#[derive(Clone, Debug)]
pub struct LinkwitzRileyFilter {
    cutoff: f32,
    sample_rate: f32,

    low_in_0: f32,
    low_in_1: f32,
    low_in_2: f32,
    low_out_1: f32,
    low_out_2: f32,
    high_in_0: f32,
    high_in_1: f32,
    high_in_2: f32,
    high_out_1: f32,
    high_out_2: f32,

    past_in_1a: [PolyF32; NUM_BANDS],
    past_in_2a: [PolyF32; NUM_BANDS],
    past_out_1a: [PolyF32; NUM_BANDS],
    past_out_2a: [PolyF32; NUM_BANDS],
    past_in_1b: [PolyF32; NUM_BANDS],
    past_in_2b: [PolyF32; NUM_BANDS],
    past_out_1b: [PolyF32; NUM_BANDS],
    past_out_2b: [PolyF32; NUM_BANDS],
}

impl LinkwitzRileyFilter {
    pub fn new(cutoff: f32, sample_rate: f32) -> LinkwitzRileyFilter {
        let mut filter = LinkwitzRileyFilter {
            cutoff,
            sample_rate,
            low_in_0: 0.0,
            low_in_1: 0.0,
            low_in_2: 0.0,
            low_out_1: 0.0,
            low_out_2: 0.0,
            high_in_0: 0.0,
            high_in_1: 0.0,
            high_in_2: 0.0,
            high_out_1: 0.0,
            high_out_2: 0.0,
            past_in_1a: [PolyF32::ZERO; NUM_BANDS],
            past_in_2a: [PolyF32::ZERO; NUM_BANDS],
            past_out_1a: [PolyF32::ZERO; NUM_BANDS],
            past_out_2a: [PolyF32::ZERO; NUM_BANDS],
            past_in_1b: [PolyF32::ZERO; NUM_BANDS],
            past_in_2b: [PolyF32::ZERO; NUM_BANDS],
            past_out_1b: [PolyF32::ZERO; NUM_BANDS],
            past_out_2b: [PolyF32::ZERO; NUM_BANDS],
        };
        filter.compute_coefficients();
        filter.reset(PolyMask::all_on());
        filter
    }

    pub fn compute_coefficients(&mut self) {
        let warp = 1.0 / (PI * self.cutoff / self.sample_rate).tan();
        let warp2 = warp * warp;
        let mult = 1.0 / (1.0 + SQRT_2 * warp + warp2);

        self.low_in_0 = mult;
        self.low_in_1 = 2.0 * mult;
        self.low_in_2 = mult;
        self.low_out_1 = -2.0 * (1.0 - warp2) * mult;
        self.low_out_2 = -(1.0 - SQRT_2 * warp + warp2) * mult;

        self.high_in_0 = warp2 * mult;
        self.high_in_1 = -2.0 * self.high_in_0;
        self.high_in_2 = self.high_in_0;
        self.high_out_1 = self.low_out_1;
        self.high_out_2 = self.low_out_2;
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.compute_coefficients();
    }

    /// Moves the crossover to `cutoff` Hz at `sample_rate`, recomputing
    /// only the coefficients: the filter memory is kept so a cutoff sweep
    /// stays continuous (re-creating the filter would zero the state and
    /// click). Allocation-free; safe to call from the audio thread.
    pub fn set_cutoff(&mut self, cutoff: f32, sample_rate: f32) {
        self.cutoff = cutoff;
        self.sample_rate = sample_rate;
        self.compute_coefficients();
    }

    /// Current crossover frequency in Hz.
    pub fn cutoff(&self) -> f32 {
        self.cutoff
    }

    /// Processes one block, writing the low band to `out_low` and the high
    /// band to `out_high`.
    pub fn process(&mut self, audio_in: &[PolyF32], out_low: &mut [PolyF32], out_high: &mut [PolyF32]) {
        let num_samples = audio_in.len();
        assert_eq!(num_samples, out_low.len());
        assert_eq!(num_samples, out_high.len());

        for i in 0..num_samples {
            let audio = audio_in[i];

            let low_in01 = (audio * self.low_in_0)
                .mul_add(self.past_in_1a[LOW], PolyF32::splat(self.low_in_1));
            let low_in = low_in01.mul_add(self.past_in_2a[LOW], PolyF32::splat(self.low_in_2));
            let low_in_out1 = low_in.mul_add(self.past_out_1a[LOW], PolyF32::splat(self.low_out_1));
            let low = low_in_out1.mul_add(self.past_out_2a[LOW], PolyF32::splat(self.low_out_2));

            self.past_in_2a[LOW] = self.past_in_1a[LOW];
            self.past_in_1a[LOW] = audio;
            self.past_out_2a[LOW] = self.past_out_1a[LOW];
            self.past_out_1a[LOW] = low;
            out_low[i] = low;
        }

        for value in out_low.iter_mut() {
            let audio = *value;

            let low_in01 = (audio * self.low_in_0)
                .mul_add(self.past_in_1b[LOW], PolyF32::splat(self.low_in_1));
            let low_in = low_in01.mul_add(self.past_in_2b[LOW], PolyF32::splat(self.low_in_2));
            let low_in_out1 = low_in.mul_add(self.past_out_1b[LOW], PolyF32::splat(self.low_out_1));
            let low = low_in_out1.mul_add(self.past_out_2b[LOW], PolyF32::splat(self.low_out_2));

            self.past_in_2b[LOW] = self.past_in_1b[LOW];
            self.past_in_1b[LOW] = audio;
            self.past_out_2b[LOW] = self.past_out_1b[LOW];
            self.past_out_1b[LOW] = low;
            *value = low;
        }

        for i in 0..num_samples {
            let audio = audio_in[i];
            let high_in01 = (audio * self.high_in_0)
                .mul_add(self.past_in_1a[HIGH], PolyF32::splat(self.high_in_1));
            let high_in = high_in01.mul_add(self.past_in_2a[HIGH], PolyF32::splat(self.high_in_2));
            let high_in_out1 =
                high_in.mul_add(self.past_out_1a[HIGH], PolyF32::splat(self.high_out_1));
            let high =
                high_in_out1.mul_add(self.past_out_2a[HIGH], PolyF32::splat(self.high_out_2));

            self.past_in_2a[HIGH] = self.past_in_1a[HIGH];
            self.past_in_1a[HIGH] = audio;
            self.past_out_2a[HIGH] = self.past_out_1a[HIGH];
            self.past_out_1a[HIGH] = high;
            out_high[i] = high;
        }

        for value in out_high.iter_mut() {
            let audio = *value;
            let high_in01 = (audio * self.high_in_0)
                .mul_add(self.past_in_1b[HIGH], PolyF32::splat(self.high_in_1));
            let high_in = high_in01.mul_add(self.past_in_2b[HIGH], PolyF32::splat(self.high_in_2));
            let high_in_out1 =
                high_in.mul_add(self.past_out_1b[HIGH], PolyF32::splat(self.high_out_1));
            let high =
                high_in_out1.mul_add(self.past_out_2b[HIGH], PolyF32::splat(self.high_out_2));

            self.past_in_2b[HIGH] = self.past_in_1b[HIGH];
            self.past_in_1b[HIGH] = audio;
            self.past_out_2b[HIGH] = self.past_out_1b[HIGH];
            self.past_out_1b[HIGH] = high;
            *value = high;
        }
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        for i in 0..NUM_BANDS {
            self.past_in_1a[i] = reset_mask.select(PolyF32::ZERO, self.past_in_1a[i]);
            self.past_in_2a[i] = reset_mask.select(PolyF32::ZERO, self.past_in_2a[i]);
            self.past_out_1a[i] = reset_mask.select(PolyF32::ZERO, self.past_out_1a[i]);
            self.past_out_2a[i] = reset_mask.select(PolyF32::ZERO, self.past_out_2a[i]);
            self.past_in_1b[i] = reset_mask.select(PolyF32::ZERO, self.past_in_1b[i]);
            self.past_in_2b[i] = reset_mask.select(PolyF32::ZERO, self.past_in_2b[i]);
            self.past_out_1b[i] = reset_mask.select(PolyF32::ZERO, self.past_out_1b[i]);
            self.past_out_2b[i] = reset_mask.select(PolyF32::ZERO, self.past_out_2b[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48000.0;

    fn run_split(filter: &mut LinkwitzRileyFilter, freq: f32, samples: usize) -> (f32, f32, f32) {
        let mut low_sum = 0.0f32;
        let mut high_sum = 0.0f32;
        let mut combined_sum = 0.0f32;
        let mut count = 0usize;
        let block = 128;
        let mut n = 0usize;
        let mut input = vec![PolyF32::ZERO; block];
        let mut out_low = vec![PolyF32::ZERO; block];
        let mut out_high = vec![PolyF32::ZERO; block];
        while n < samples {
            for value in input.iter_mut() {
                let phase = 2.0 * core::f32::consts::PI * freq * n as f32 / SAMPLE_RATE;
                *value = PolyF32::splat(phase.sin());
                n += 1;
            }
            filter.process(&input, &mut out_low, &mut out_high);
            if n > samples / 2 {
                for i in 0..block {
                    let low = out_low[i].lane(0);
                    let high = out_high[i].lane(0);
                    low_sum += low * low;
                    high_sum += high * high;
                    let combined = low + high;
                    combined_sum += combined * combined;
                    count += 1;
                }
            }
        }
        (
            (low_sum / count as f32).sqrt(),
            (high_sum / count as f32).sqrt(),
            (combined_sum / count as f32).sqrt(),
        )
    }

    #[test]
    fn splits_low_frequencies_to_low_band() {
        let mut filter = LinkwitzRileyFilter::new(1000.0, SAMPLE_RATE);
        let (low, high, _) = run_split(&mut filter, 100.0, 19200);
        assert!(low > 10.0 * high, "low {low} vs high {high}");
    }

    #[test]
    fn splits_high_frequencies_to_high_band() {
        let mut filter = LinkwitzRileyFilter::new(1000.0, SAMPLE_RATE);
        let (low, high, _) = run_split(&mut filter, 10000.0, 19200);
        assert!(high > 10.0 * low, "high {high} vs low {low}");
    }

    #[test]
    fn bands_recombine_flat() {
        for freq in [100.0, 1000.0, 5000.0] {
            let mut filter = LinkwitzRileyFilter::new(1000.0, SAMPLE_RATE);
            let (_, _, combined) = run_split(&mut filter, freq, 19200);
            let input_rms = 1.0 / core::f32::consts::SQRT_2;
            assert!(
                (combined - input_rms).abs() < 0.05 * input_rms,
                "freq {freq}: combined rms {combined} vs input {input_rms}"
            );
        }
    }

    #[test]
    fn set_cutoff_keeps_state_and_stays_continuous() {
        let mut filter = LinkwitzRileyFilter::new(1000.0, SAMPLE_RATE);
        const BLOCK: usize = 128;
        let mut low_trace = Vec::new();
        let run_block = |filter: &mut LinkwitzRileyFilter, index: usize| -> Vec<f32> {
            let input: Vec<PolyF32> = (0..BLOCK)
                .map(|i| {
                    let n = (index * BLOCK + i) as f32;
                    PolyF32::splat((2.0 * core::f32::consts::PI * 300.0 * n / SAMPLE_RATE).sin())
                })
                .collect();
            let mut out_low = vec![PolyF32::ZERO; BLOCK];
            let mut out_high = vec![PolyF32::ZERO; BLOCK];
            filter.process(&input, &mut out_low, &mut out_high);
            out_low.iter().map(|v| v.lane(0)).collect()
        };
        for index in 0..8 {
            low_trace.extend(run_block(&mut filter, index));
        }
        // Moving the crossover must not clear the memory: the first output
        // after the change continues the wave instead of restarting from 0.
        filter.set_cutoff(4000.0, SAMPLE_RATE);
        assert_eq!(filter.cutoff(), 4000.0);
        let before = low_trace.len();
        low_trace.extend(run_block(&mut filter, 8));
        let steady_step = low_trace[before - 128..before]
            .windows(2)
            .fold(0.0f32, |a, w| a.max((w[1] - w[0]).abs()));
        let boundary_step = (low_trace[before] - low_trace[before - 1]).abs();
        assert!(
            boundary_step < 3.0 * steady_step + 1e-3,
            "cutoff change clicked: step {boundary_step} vs steady {steady_step}"
        );
        assert!(low_trace[before].abs() > 1e-3 || low_trace[before + 1].abs() > 1e-3);

        // And the new cutoff is really in effect: a fresh filter at 4 kHz
        // matches once the old state has decayed.
        let mut fresh = LinkwitzRileyFilter::new(4000.0, SAMPLE_RATE);
        let (moved_low, _, _) = run_split(&mut filter, 300.0, 9600);
        let (fresh_low, _, _) = run_split(&mut fresh, 300.0, 9600);
        assert!((moved_low - fresh_low).abs() < 1e-3 * fresh_low.max(1e-3));
    }

    #[test]
    fn reset_clears_state() {
        let mut filter = LinkwitzRileyFilter::new(1000.0, SAMPLE_RATE);
        let _ = run_split(&mut filter, 440.0, 1920);
        filter.reset(PolyMask::all_on());
        let silence = vec![PolyF32::ZERO; 64];
        let mut out_low = vec![PolyF32::splat(1.0); 64];
        let mut out_high = vec![PolyF32::splat(1.0); 64];
        filter.process(&silence, &mut out_low, &mut out_high);
        for i in 0..64 {
            assert_eq!(out_low[i].lane(0), 0.0);
            assert_eq!(out_high[i].lane(0), 0.0);
        }
    }
}
