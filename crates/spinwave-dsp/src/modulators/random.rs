//! Random number generation matching Vital's `utils::RandomGenerator`.
//!
//! The reference uses `std::mt19937` with `std::uniform_real_distribution`.
//! Both are ported bit-exactly (the distribution follows libstdc++'s
//! `generate_canonical`: one 32-bit draw scaled by `2^-32` in `f32`), so
//! sequences match a libstdc++ build of Vital for the same seed.

use std::sync::atomic::{AtomicU32, Ordering};

use spinwave_poly::{PolyF32, PolyMask, LANES};

/// Mirrors the C++ `static int next_seed_` used to seed each new generator.
static NEXT_SEED: AtomicU32 = AtomicU32::new(0);

const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;

/// MT19937 Mersenne Twister, identical to `std::mt19937`.
#[derive(Clone)]
struct Mt19937 {
    state: [u32; MT_N],
    index: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut engine = Mt19937 { state: [0; MT_N], index: MT_N };
        engine.seed(seed);
        engine
    }

    fn seed(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..MT_N {
            let prev = self.state[i - 1];
            self.state[i] =
                1_812_433_253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.index = MT_N;
    }

    fn generate(&mut self) {
        for i in 0..MT_N {
            let y =
                (self.state[i] & MT_UPPER_MASK) | (self.state[(i + 1) % MT_N] & MT_LOWER_MASK);
            let mut next = y >> 1;
            if y & 1 != 0 {
                next ^= MT_MATRIX_A;
            }
            self.state[i] = self.state[(i + MT_M) % MT_N] ^ next;
        }
        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            self.generate();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// Uniform random source in `[min, max)`, one value per call.
#[derive(Clone)]
pub struct RandomGenerator {
    engine: Mt19937,
    min: f32,
    max: f32,
}

impl RandomGenerator {
    /// Auto-seeded from a global counter, like the C++ `next_seed_++`.
    pub fn new(min: f32, max: f32) -> Self {
        let seed = NEXT_SEED.fetch_add(1, Ordering::Relaxed);
        Self::with_seed(min, max, seed)
    }

    /// Rewinds the global seed counter, so the next generators built get
    /// the seeds a fresh process would hand out. For offline renders that
    /// must be reproducible run to run: without it a render's random LFOs
    /// depend on how many generators the process built before, which made
    /// one golden case's residual change with the order the bench ran in.
    pub fn reset_seed_counter() {
        NEXT_SEED.store(0, Ordering::Relaxed);
    }

    /// Explicit seed for deterministic sequences.
    pub fn with_seed(min: f32, max: f32, seed: u32) -> Self {
        RandomGenerator { engine: Mt19937::new(seed), min, max }
    }

    /// Reseeds the engine, matching `RandomGenerator::seed`.
    pub fn seed(&mut self, seed: u32) {
        self.engine.seed(seed);
    }

    /// Next scalar value.
    #[inline]
    #[allow(clippy::should_implement_trait)] // mirrors the C++ API name
    pub fn next(&mut self) -> f32 {
        // libstdc++ generate_canonical<float, 24>: float(u32) / 2^32.
        let canonical = self.engine.next_u32() as f32 * (1.0 / 4_294_967_296.0);
        canonical * (self.max - self.min) + self.min
    }

    /// Independent value in every lane.
    #[inline]
    pub fn poly_next(&mut self) -> PolyF32 {
        let mut lanes = [0.0; LANES];
        for lane in &mut lanes {
            *lane = self.next();
        }
        PolyF32::from_lanes(lanes)
    }

    /// One value per voice, shared by that voice's stereo lanes.
    #[inline]
    pub fn poly_voice_next(&mut self) -> PolyF32 {
        let mut lanes = [0.0; LANES];
        for voice in 0..LANES / 2 {
            let value = self.next();
            lanes[voice * 2] = value;
            lanes[voice * 2 + 1] = value;
        }
        PolyF32::from_lanes(lanes)
    }

    /// Draws only for lanes set in `mask` (zero elsewhere). The number of
    /// engine draws depends on the mask, matching the reference.
    #[inline]
    pub fn poly_next_masked(&mut self, mask: PolyMask) -> PolyF32 {
        let mask_lanes = mask.to_u32().0;
        let mut lanes = [0.0; LANES];
        for i in 0..LANES {
            if mask_lanes[i] != 0 {
                lanes[i] = self.next();
            }
        }
        PolyF32::from_lanes(lanes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = RandomGenerator::with_seed(-1.0, 1.0, 42);
        let mut b = RandomGenerator::with_seed(-1.0, 1.0, 42);
        for _ in 0..1000 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn different_seed_different_sequence() {
        let mut a = RandomGenerator::with_seed(-1.0, 1.0, 1);
        let mut b = RandomGenerator::with_seed(-1.0, 1.0, 2);
        let same = (0..64).filter(|_| a.next() == b.next()).count();
        assert!(same < 8);
    }

    #[test]
    fn mt19937_reference_output() {
        // std::mt19937 seeded with 5489 (default) produces 3499211612 first.
        let mut engine = Mt19937::new(5489);
        assert_eq!(engine.next_u32(), 3_499_211_612);
        // 10000th output of mt19937(5489) is the classic 4123659995.
        let mut engine = Mt19937::new(5489);
        let mut last = 0;
        for _ in 0..10_000 {
            last = engine.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
    }

    #[test]
    fn range_respected() {
        let mut generator = RandomGenerator::with_seed(0.0, 1.0, 7);
        for _ in 0..10_000 {
            let value = generator.next();
            assert!((0.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn voice_next_duplicates_stereo_lanes() {
        let mut generator = RandomGenerator::with_seed(-1.0, 1.0, 3);
        let value = generator.poly_voice_next();
        assert_eq!(value.lane(0), value.lane(1));
        assert_eq!(value.lane(2), value.lane(3));
        assert_ne!(value.lane(0), value.lane(2));
    }
}
