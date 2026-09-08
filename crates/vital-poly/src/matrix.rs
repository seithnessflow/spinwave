//! 4x4 interpolation matrix (port of Vital's `matrix.h`).
//!
//! Used for 4-point interpolation: one row per tap, transposed against a
//! value matrix gathered from a buffer, then multiplied and summed.

use crate::simd::PolyF32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Matrix {
    pub rows: [PolyF32; 4],
}

impl Matrix {
    #[inline(always)]
    pub fn new(r0: PolyF32, r1: PolyF32, r2: PolyF32, r3: PolyF32) -> Matrix {
        Matrix { rows: [r0, r1, r2, r3] }
    }

    #[inline(always)]
    pub fn transpose(&mut self) {
        let [r0, r1, r2, r3] = self.rows.map(PolyF32::to_lanes);
        self.rows = [
            PolyF32::from_lanes([r0[0], r1[0], r2[0], r3[0]]),
            PolyF32::from_lanes([r0[1], r1[1], r2[1], r3[1]]),
            PolyF32::from_lanes([r0[2], r1[2], r2[2], r3[2]]),
            PolyF32::from_lanes([r0[3], r1[3], r2[3], r3[3]]),
        ];
    }

    #[inline(always)]
    pub fn interpolate_columns(&mut self, other: &Matrix, t: PolyF32) {
        for i in 0..4 {
            self.rows[i] = self.rows[i].mul_add(other.rows[i] - self.rows[i], t);
        }
    }

    #[inline(always)]
    pub fn interpolate_rows(&mut self, other: &Matrix, t: PolyF32) {
        let lanes = t.to_lanes();
        for i in 0..4 {
            self.rows[i] =
                self.rows[i].mul_add(other.rows[i] - self.rows[i], PolyF32::splat(lanes[i]));
        }
    }

    #[inline(always)]
    pub fn sum_rows(&self) -> PolyF32 {
        self.rows[0] + self.rows[1] + self.rows[2] + self.rows[3]
    }

    #[inline(always)]
    pub fn multiply_and_sum_rows(&self, other: &Matrix) -> PolyF32 {
        let row01 = (self.rows[0] * other.rows[0]).mul_add(self.rows[1], other.rows[1]);
        let row012 = row01.mul_add(self.rows[2], other.rows[2]);
        row012.mul_add(self.rows[3], other.rows[3])
    }
}
