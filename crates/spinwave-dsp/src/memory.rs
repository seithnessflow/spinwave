//! Interpolated delay memory (rework of Vital's `lookups/memory.h`).
//!
//! Ring buffer with power-of-two size. Each sample is written twice,
//! `size` apart, so any 4 consecutive taps read for Catmull-Rom
//! interpolation stay in-bounds without per-tap masking.

use spinwave_poly::matrix::Matrix;
use spinwave_poly::simd::{PolyF32, PolyMask, PolyU32, LANES};
use spinwave_poly::utils::catmull_interpolation_matrix;

pub const MIN_PERIOD: f32 = 2.0;
const EXTRA_INTERPOLATION_VALUES: usize = 3;

#[derive(Clone, Debug)]
struct MemoryCore<const CHANNELS: usize> {
    buffers: [Vec<f32>; CHANNELS],
    size: usize,
    bitmask: usize,
    offset: usize,
    /// Samples pushed since the last full clear, saturating at `size`:
    /// how much of the ring is dirty, so a clear touches only that. A
    /// full clear of a four-second ring per offline render was memory
    /// bandwidth, and memory bandwidth is what many threads share.
    pushed: usize,
}

/// Smallest ring the core will build. Below this `max_period` (which is
/// `size - EXTRA_INTERPOLATION_VALUES`) would underflow and the 4-tap
/// Catmull-Rom reads in `interpolated_get` would run past the buffer.
const MIN_MEMORY_SIZE: usize = 8;

impl<const CHANNELS: usize> MemoryCore<CHANNELS> {
    fn new(min_size: usize) -> Self {
        let size = min_size.max(MIN_MEMORY_SIZE).next_power_of_two();
        MemoryCore {
            buffers: core::array::from_fn(|_| vec![0.0; 2 * size]),
            size,
            bitmask: size - 1,
            offset: 0,
            pushed: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, sample: PolyF32) {
        debug_assert!(sample.is_finite());
        self.offset = (self.offset + 1) & self.bitmask;
        self.pushed = (self.pushed + 1).min(self.size);
        let lanes = sample.to_lanes();
        for (channel, buffer) in self.buffers.iter_mut().enumerate() {
            let value = lanes[channel];
            buffer[self.offset] = value;
            buffer[self.offset + self.size] = value;
        }
    }

    fn clear_memory(&mut self, num: usize, clear_mask: PolyMask) {
        let start = self
            .offset
            .wrapping_sub(num + EXTRA_INTERPOLATION_VALUES)
            & self.bitmask;
        let end = (self.offset + EXTRA_INTERPOLATION_VALUES) & self.bitmask;

        for (channel, buffer) in self.buffers.iter_mut().enumerate() {
            if clear_mask.to_u32().lane(channel) != 0 {
                let mut i = start;
                while i != end {
                    buffer[i] = 0.0;
                    i = (i + 1) & self.bitmask;
                }
                buffer[end] = 0.0;
                for j in 0..EXTRA_INTERPOLATION_VALUES {
                    buffer[self.size + j] = 0.0;
                }
            }
        }
    }

    /// Zeroes the ring and rewinds it to its constructed state (offset
    /// 0). Only the samples pushed since the last clear are touched: they
    /// sit at `offset - pushed + 1 ..= offset` (mod size) and in the
    /// mirror `size` further on; a ring that has wrapped is cleared whole.
    fn clear_all(&mut self) {
        if self.pushed >= self.size {
            for buffer in &mut self.buffers {
                buffer.fill(0.0);
            }
        } else if self.pushed > 0 {
            let first = self.offset.wrapping_sub(self.pushed - 1) & self.bitmask;
            for buffer in &mut self.buffers {
                if first <= self.offset {
                    buffer[first..=self.offset].fill(0.0);
                    buffer[first + self.size..=self.offset + self.size].fill(0.0);
                } else {
                    // Wrapped within the window: two ranges, mirrored.
                    buffer[first..self.size].fill(0.0);
                    buffer[..=self.offset].fill(0.0);
                    buffer[first + self.size..].fill(0.0);
                    buffer[self.size..=self.offset + self.size].fill(0.0);
                }
            }
        }
        self.offset = 0;
        self.pushed = 0;
    }

    fn read_samples(&self, output: &mut [f32], offset: usize, channel: usize) {
        let buffer = &self.buffers[channel];
        let start_index = self
            .offset
            .wrapping_sub(output.len() + offset)
            & self.bitmask;
        for (i, out) in output.iter_mut().enumerate() {
            *out = buffer[(i + start_index) & self.bitmask];
        }
    }

    #[inline(always)]
    fn max_period(&self) -> usize {
        self.size - EXTRA_INTERPOLATION_VALUES
    }

    /// Catmull-Rom read `past` samples back, per lane, `taps` channel rows.
    #[inline(always)]
    fn interpolated_get(&self, past: PolyF32, rows: [usize; 4]) -> PolyF32 {
        let past_index = past.to_i32_round();
        let t = past_index.to_f32_signed() - past + 1.0;
        let interpolation_matrix = catmull_interpolation_matrix(t);

        let indices = (PolyU32::splat(self.offset as u32) - past_index - PolyU32::splat(2))
            & PolyU32::splat(self.bitmask as u32);

        let tap = |lane: usize| {
            let buffer = &self.buffers[rows[lane]];
            let start = indices.lane(lane) as usize;
            PolyF32::from_lanes([
                buffer[start],
                buffer[start + 1],
                buffer[start + 2],
                buffer[start + 3],
            ])
        };
        let mut value_matrix = Matrix::new(tap(0), tap(1), tap(2), tap(3));
        value_matrix.transpose();
        interpolation_matrix.multiply_and_sum_rows(&value_matrix)
    }
}

/// Per-lane delay memory: 4 independent channels (two stereo voices).
#[derive(Clone, Debug)]
pub struct Memory {
    core: MemoryCore<LANES>,
}

impl Memory {
    pub fn new(min_size: usize) -> Memory {
        Memory { core: MemoryCore::new(min_size) }
    }

    #[inline(always)]
    pub fn push(&mut self, sample: PolyF32) {
        self.core.push(sample);
    }

    /// Interpolated read `past` samples back (per lane, â‰¥ [`MIN_PERIOD`]).
    #[inline(always)]
    pub fn get(&self, past: PolyF32) -> PolyF32 {
        debug_assert!(!past.lt(PolyF32::splat(MIN_PERIOD)).any());
        debug_assert!(!past.gt(PolyF32::splat(self.max_period() as f32)).any());
        self.core.interpolated_get(past, [0, 1, 2, 3])
    }

    pub fn clear_memory(&mut self, num: usize, clear_mask: PolyMask) {
        self.core.clear_memory(num, clear_mask);
    }

    pub fn clear_all(&mut self) {
        self.core.clear_all();
    }

    pub fn max_period(&self) -> usize {
        self.core.max_period()
    }

    pub fn size(&self) -> usize {
        self.core.size
    }
}

/// Two-channel (global stereo) delay memory, e.g. for the EQ display and
/// mono effects. Lanes 2/3 of reads mirror lanes 0/1.
#[derive(Clone, Debug)]
pub struct StereoMemory {
    core: MemoryCore<2>,
}

impl StereoMemory {
    pub fn new(min_size: usize) -> StereoMemory {
        StereoMemory { core: MemoryCore::new(min_size) }
    }

    #[inline(always)]
    pub fn push(&mut self, sample: PolyF32) {
        self.core.push(sample);
    }

    #[inline(always)]
    pub fn get(&self, past: PolyF32) -> PolyF32 {
        debug_assert!(!past.lt(PolyF32::splat(MIN_PERIOD)).any());
        debug_assert!(!past.gt(PolyF32::splat(self.max_period() as f32)).any());
        self.core.interpolated_get(past, [0, 1, 0, 1])
    }

    pub fn read_samples(&self, output: &mut [f32], offset: usize, channel: usize) {
        self.core.read_samples(output, offset, channel);
    }

    pub fn clear_all(&mut self) {
        self.core.clear_all();
    }

    pub fn max_period(&self) -> usize {
        self.core.max_period()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_delay_recovers_pushed_samples() {
        let mut memory = Memory::new(64);
        for i in 0..32 {
            memory.push(PolyF32::splat(i as f32));
        }
        // 4 samples back from the last push (31) should read 27.
        let value = memory.get(PolyF32::splat(4.0));
        for lane in 0..LANES {
            assert!((value.lane(lane) - 27.0).abs() < 1e-4);
        }
    }

    #[test]
    fn fractional_delay_interpolates() {
        let mut memory = Memory::new(64);
        for i in 0..32 {
            memory.push(PolyF32::splat(i as f32));
        }
        // Linear ramp: Catmull-Rom reproduces it exactly at fractional taps.
        let value = memory.get(PolyF32::splat(4.5));
        assert!((value.lane(0) - 26.5).abs() < 1e-4);
    }

    #[test]
    fn per_lane_delays() {
        let mut memory = Memory::new(64);
        for i in 0..32 {
            memory.push(PolyF32::splat(i as f32));
        }
        let value = memory.get(PolyF32::from_lanes([2.0, 3.0, 4.0, 5.0]));
        assert!((value.lane(0) - 29.0).abs() < 1e-4);
        assert!((value.lane(1) - 28.0).abs() < 1e-4);
        assert!((value.lane(2) - 27.0).abs() < 1e-4);
        assert!((value.lane(3) - 26.0).abs() < 1e-4);
    }

    #[test]
    fn stereo_memory_mirrors_voices() {
        let mut memory = StereoMemory::new(64);
        for i in 0..32 {
            memory.push(PolyF32::from_lanes([i as f32, -(i as f32), 0.0, 0.0]));
        }
        let value = memory.get(PolyF32::splat(4.0));
        assert!((value.lane(0) - 27.0).abs() < 1e-4);
        assert!((value.lane(1) + 27.0).abs() < 1e-4);
        assert!((value.lane(2) - 27.0).abs() < 1e-4);
    }

    #[test]
    fn tiny_requested_size_is_padded_and_readable() {
        // Sizes below the interpolation guard used to underflow max_period
        // and read out of bounds; they must round up to a usable ring.
        for requested in [0, 1, 2, 3, 4] {
            let mut memory = Memory::new(requested);
            assert!(memory.max_period() >= 4, "requested {requested}");
            assert!(memory.size() >= 8);
            for i in 0..16 {
                memory.push(PolyF32::splat(i as f32));
            }
            let value = memory.get(PolyF32::splat(memory.max_period() as f32));
            assert!(value.is_finite());
            let stereo = StereoMemory::new(requested);
            assert!(stereo.max_period() >= 4);
        }
    }

    #[test]
    fn clear_all_after_a_partial_fill_equals_a_fresh_ring() {
        // Push fewer samples than the ring holds, clear, and the ring
        // must read as new everywhere — including the mirror half.
        for pushes in [1usize, 5, 300, 1023, 1024, 3000] {
            let mut m = Memory::new(1000);
            for i in 0..pushes {
                m.push(PolyF32::splat(1.0 + i as f32));
            }
            m.clear_all();
            let fresh = Memory::new(1000);
            assert_eq!(m.core.offset, 0, "{pushes}");
            for (a, b) in m.core.buffers[0].iter().zip(&fresh.core.buffers[0]) {
                assert_eq!(a, b, "after {pushes} pushes");
            }
        }
    }

    #[test]
    fn clear_memory_zeroes_recent_window() {
        let mut memory = Memory::new(64);
        for _ in 0..64 {
            memory.push(PolyF32::splat(1.0));
        }
        memory.clear_memory(16, PolyMask::all_on());
        let value = memory.get(PolyF32::splat(8.0));
        for lane in 0..LANES {
            assert_eq!(value.lane(lane), 0.0);
        }
    }
}
