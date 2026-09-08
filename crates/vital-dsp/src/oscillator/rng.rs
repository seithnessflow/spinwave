//! Tiny deterministic PRNG for phase randomization and noise tables.
//!
//! The reference uses `std::mt19937` + `uniform_real_distribution`, whose
//! exact stream is implementation defined; a xorshift32 keeps the same
//! statistical character with reproducible output.

#[derive(Clone, Debug)]
pub(crate) struct Xorshift32 {
    state: u32,
}

impl Xorshift32 {
    pub fn new(seed: u32) -> Xorshift32 {
        Xorshift32 { state: seed.max(1) }
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform value in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform value in `[min, max)`.
    #[inline]
    pub fn next_in(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}
