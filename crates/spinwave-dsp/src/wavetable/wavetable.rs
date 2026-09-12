//! Wavetable data model: a stack of frames with precomputed spectra.
//!
//! Rework of Vital's `wavetable.{h,cpp}`. Per frame the table stores the
//! time-domain cycle plus three padded float arrays the oscillator's
//! spectral morphs read directly: harmonic magnitudes (each duplicated so
//! a pair of floats maps onto one complex bin), unit-magnitude normalized
//! frequencies `(cos, sin)`, and raw phase angles. The atomic double
//! buffering of the reference is dropped; sharing across threads is the
//! engine's job in this rework.

use realfft::num_complex::Complex;

use super::wave_frame::{WaveFrame, NUM_REAL_COMPLEX, WAVEFORM_BITS, WAVEFORM_SIZE};

pub const FREQUENCY_BINS: usize = WAVEFORM_BITS;
pub const NUM_HARMONICS: usize = NUM_REAL_COMPLEX;
/// Float length of the padded per-frame frequency arrays
/// (`kPolyFrequencySize` poly vectors in the reference).
pub const POLY_FREQUENCY_FLOATS: usize = (2 * NUM_HARMONICS / 4 + 2) * 4;
/// Frame count of a full-size oscillator wavetable.
pub const NUM_OSCILLATOR_WAVE_FRAMES: usize = 257;

/// The audio-thread view of a wavetable: everything the oscillator reads.
pub struct WavetableData {
    num_frames: usize,
    pub frequency_ratio: f32,
    pub sample_rate: f32,
    version: u32,
    wave_data: Vec<Vec<f32>>,
    frequency_amplitudes: Vec<Vec<f32>>,
    normalized_frequencies: Vec<Vec<f32>>,
    phases: Vec<Vec<f32>>,
}

impl WavetableData {
    fn new(num_frames: usize, version: u32) -> WavetableData {
        WavetableData {
            num_frames,
            frequency_ratio: 1.0,
            sample_rate: super::wave_frame::DEFAULT_WAVE_SAMPLE_RATE,
            version,
            wave_data: vec![vec![0.0; WAVEFORM_SIZE]; num_frames],
            frequency_amplitudes: vec![vec![0.0; POLY_FREQUENCY_FLOATS]; num_frames],
            normalized_frequencies: vec![vec![0.0; POLY_FREQUENCY_FLOATS]; num_frames],
            phases: vec![vec![0.0; POLY_FREQUENCY_FLOATS]; num_frames],
        }
    }

    #[inline]
    pub fn num_frames(&self) -> usize {
        self.num_frames
    }

    #[inline]
    pub fn version(&self) -> u32 {
        self.version
    }

    #[inline]
    pub fn clamp_frame(&self, frame: usize) -> usize {
        frame.min(self.num_frames - 1)
    }

    /// Time-domain cycle of one frame.
    #[inline]
    pub fn wave_data(&self, frame: usize) -> &[f32] {
        &self.wave_data[self.clamp_frame(frame)]
    }

    /// Harmonic magnitudes, duplicated per float pair; padded with zeros.
    #[inline]
    pub fn frequency_amplitudes(&self, frame: usize) -> &[f32] {
        &self.frequency_amplitudes[self.clamp_frame(frame)]
    }

    /// Interleaved `(cos, sin)` of each harmonic's phase; padded.
    #[inline]
    pub fn normalized_frequencies(&self, frame: usize) -> &[f32] {
        &self.normalized_frequencies[self.clamp_frame(frame)]
    }

    /// Phase angles, duplicated per float pair; padded.
    #[inline]
    pub fn phases(&self, frame: usize) -> &[f32] {
        &self.phases[self.clamp_frame(frame)]
    }
}

/// A named stack of wave frames plus their oscillator-ready spectra.
pub struct Wavetable {
    pub name: String,
    pub author: String,
    max_frames: usize,
    shepard_table: bool,
    data: WavetableData,
}

impl Wavetable {
    /// Creates a table able to hold `max_frames`, loaded with one silent
    /// frame (the reference's default wavetable).
    pub fn new(max_frames: usize) -> Wavetable {
        let mut wavetable = Wavetable {
            name: String::new(),
            author: String::new(),
            max_frames: max_frames.max(1),
            shepard_table: false,
            data: WavetableData::new(1, 1),
        };
        wavetable.load_wave_frame(&WaveFrame::default());
        wavetable
    }

    #[inline]
    pub fn data(&self) -> &WavetableData {
        &self.data
    }

    #[inline]
    pub fn num_frames(&self) -> usize {
        self.data.num_frames
    }

    #[inline]
    pub fn version(&self) -> u32 {
        self.data.version
    }

    #[inline]
    pub fn set_frequency_ratio(&mut self, ratio: f32) {
        self.data.frequency_ratio = ratio;
    }

    #[inline]
    pub fn set_sample_rate(&mut self, rate: f32) {
        self.data.sample_rate = rate;
    }

    #[inline]
    pub fn set_shepard_table(&mut self, shepard: bool) {
        self.shepard_table = shepard;
    }

    #[inline]
    pub fn is_shepard_table(&self) -> bool {
        self.shepard_table
    }

    /// `log2` of how many table samples one output sample steps over —
    /// the fractional mip position for a normalized phase increment.
    #[inline]
    pub fn frequency_float_bin(phase_increment: f32) -> f32 {
        // futils::log2 in the reference, the polynomial: this bin is where
        // the oscillator's harmonic count and the morphs' band limit start.
        spinwave_poly::math::log2(spinwave_poly::PolyF32::splat(1.0 / phase_increment)).lane(0)
    }

    /// Integer mip level for a normalized phase increment, clamped to the
    /// available bins. Higher pitches produce lower bins.
    #[inline]
    pub fn frequency_bin(phase_increment: f32) -> usize {
        let num_waves = (1.0 / phase_increment) as i32;
        let log = 31 - (num_waves.max(1) as u32).leading_zeros();
        (log as usize).min(FREQUENCY_BINS - 1)
    }

    /// Resizes the table, keeping existing frames and repeating the last
    /// one into any new slots. Bumps the data version.
    pub fn set_num_frames(&mut self, num_frames: usize) {
        assert!(num_frames >= 1 && num_frames <= self.max_frames);
        if num_frames == self.data.num_frames {
            return;
        }

        let new_version = self.data.version + 1;
        let old = std::mem::replace(&mut self.data, WavetableData::new(num_frames, new_version));
        self.data.frequency_ratio = old.frequency_ratio;
        self.data.sample_rate = old.sample_rate;

        for i in 0..num_frames {
            let from = i.min(old.num_frames - 1);
            self.data.wave_data[i].copy_from_slice(&old.wave_data[from]);
            self.data.frequency_amplitudes[i].copy_from_slice(&old.frequency_amplitudes[from]);
            self.data.normalized_frequencies[i]
                .copy_from_slice(&old.normalized_frequencies[from]);
            self.data.phases[i].copy_from_slice(&old.phases[from]);
        }
    }

    /// Loads a frame at its own index.
    pub fn load_wave_frame(&mut self, wave_frame: &WaveFrame) {
        self.load_wave_frame_at(wave_frame, wave_frame.index);
    }

    /// Loads a frame at an explicit index; out-of-range indices are ignored.
    pub fn load_wave_frame_at(&mut self, wave_frame: &WaveFrame, to_index: usize) {
        if to_index >= self.data.num_frames {
            return;
        }

        let amplitudes = &mut self.data.frequency_amplitudes[to_index];
        let normalized = &mut self.data.normalized_frequencies[to_index];
        let phases = &mut self.data.phases[to_index];
        for (i, bin) in wave_frame.frequency_domain.iter().enumerate() {
            let amplitude = bin.norm();
            amplitudes[2 * i] = amplitude;
            amplitudes[2 * i + 1] = amplitude;
            let arg = bin.arg();
            normalized[2 * i] = arg.cos();
            normalized[2 * i + 1] = arg.sin();
            phases[2 * i] = arg;
            phases[2 * i + 1] = arg;
        }
        self.data.wave_data[to_index].copy_from_slice(&wave_frame.time_domain);
    }

    /// Post-load pass: optionally rescales to a target span and smooths the
    /// normalized phase of quiet harmonics across frames so interpolating
    /// between frames doesn't sweep through garbage phases.
    pub fn post_process(&mut self, max_span: f32) {
        const MIN_AMPLITUDE_PHASE: f32 = 0.1;

        if max_span > 0.0 {
            let scale = 2.0 / max_span;
            for w in 0..self.data.num_frames {
                for value in &mut self.data.frequency_amplitudes[w] {
                    *value *= scale;
                }
                for value in &mut self.data.wave_data[w] {
                    *value *= scale;
                }
            }
        }

        let num_frames = self.data.num_frames;
        for i in 0..NUM_HARMONICS {
            let amp_index = 2 * i;

            let mut last_min_amp_frame: isize = -1;
            let mut last_normalized = Complex::new(0.0f32, 1.0);
            for w in 0..num_frames {
                let amplitude = self.data.frequency_amplitudes[w][amp_index];
                let normalized = Complex::new(
                    self.data.normalized_frequencies[w][2 * i],
                    self.data.normalized_frequencies[w][2 * i + 1],
                );

                if amplitude > MIN_AMPLITUDE_PHASE {
                    if last_min_amp_frame < 0 {
                        last_min_amp_frame = 0;
                        last_normalized = normalized;
                    }

                    let delta = normalized - last_normalized;
                    for frame in (last_min_amp_frame as usize + 1)..w {
                        let t = (frame as isize - last_min_amp_frame) as f32
                            / (w as isize - last_min_amp_frame) as f32;
                        let value = delta * t + last_normalized;
                        self.data.normalized_frequencies[frame][2 * i] = value.re;
                        self.data.normalized_frequencies[frame][2 * i + 1] = value.im;
                    }
                    last_normalized = normalized;
                    last_min_amp_frame = w as isize;
                }
            }
            for frame in (last_min_amp_frame + 1).max(0) as usize..num_frames {
                self.data.normalized_frequencies[frame][2 * i] = last_normalized.re;
                self.data.normalized_frequencies[frame][2 * i + 1] = last_normalized.im;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::wave_frame::WaveShape;
    use super::*;

    #[test]
    fn frequency_bin_shrinks_with_pitch() {
        // Higher notes step faster through the table and pick lower bins
        // (fewer usable harmonics).
        let low = Wavetable::frequency_bin(55.0 / 44100.0);
        let mid = Wavetable::frequency_bin(440.0 / 44100.0);
        let high = Wavetable::frequency_bin(3520.0 / 44100.0);
        assert!(low > mid, "low={low} mid={mid}");
        assert!(mid > high, "mid={mid} high={high}");
        assert!(high < FREQUENCY_BINS);
    }

    #[test]
    fn loads_sine_amplitudes() {
        let mut wavetable = Wavetable::new(3);
        wavetable.load_wave_frame(&WaveFrame::predefined(WaveShape::Sin));
        let data = wavetable.data();
        let amps = data.frequency_amplitudes(0);
        // Bin 1 carries N/2, everything else is empty.
        assert!((amps[2] - (WAVEFORM_SIZE / 2) as f32).abs() < 1.0);
        assert!((amps[2] - amps[3]).abs() < 1e-6);
        assert!(amps[4].abs() < 1.0);
        // Normalized frequencies have unit magnitude.
        let norm = data.normalized_frequencies(0);
        let mag = norm[2] * norm[2] + norm[3] * norm[3];
        assert!((mag - 1.0).abs() < 1e-5);
    }

    #[test]
    fn resize_repeats_last_frame() {
        let mut wavetable = Wavetable::new(4);
        wavetable.load_wave_frame(&WaveFrame::predefined(WaveShape::Square));
        let version = wavetable.version();
        wavetable.set_num_frames(3);
        assert_eq!(wavetable.num_frames(), 3);
        assert!(wavetable.version() > version);
        let data = wavetable.data();
        assert_eq!(data.wave_data(0)[10], data.wave_data(2)[10]);
    }
}
