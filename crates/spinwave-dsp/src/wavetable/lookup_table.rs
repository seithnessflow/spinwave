//! Precomputed one-dimensional function table with cubic lookup.
//!
//! Rework of Vital's `lookup_table.h` (`OneDimLookup`). The function is
//! sampled once at construction; lookups read four neighboring taps with
//! Catmull-Rom interpolation, vectorized per lane.

use spinwave_poly::utils::{catmull_interpolation_matrix, value_matrix};
use spinwave_poly::PolyF32;

const EXTRA_VALUES: usize = 4;

/// A `RESOLUTION`-point table over the input range `[0, scale]`.
pub struct OneDimLookup<const RESOLUTION: usize> {
    lookup: Vec<f32>,
    scale: f32,
}

impl<const RESOLUTION: usize> OneDimLookup<RESOLUTION> {
    pub fn new(function: impl Fn(f32) -> f32, scale: f32) -> Self {
        let mut lookup = vec![0.0f32; RESOLUTION + EXTRA_VALUES];
        for (i, value) in lookup.iter_mut().enumerate() {
            let t = (i as f32 - 1.0) / (RESOLUTION as f32 - 1.0);
            *value = function(t * scale);
        }
        OneDimLookup { lookup, scale: RESOLUTION as f32 / scale }
    }

    /// Catmull-Rom interpolated read, per lane.
    #[inline]
    pub fn cubic_lookup(&self, value: PolyF32) -> PolyF32 {
        let boost = value * self.scale;
        let indices = boost.clamp(0.0, RESOLUTION as f32).to_i32_round();
        let t = boost - indices.to_f32_signed();

        let interpolation_matrix = catmull_interpolation_matrix(t);
        let mut values = value_matrix(&self.lookup, indices);
        values.transpose();

        interpolation_matrix.multiply_and_sum_rows(&values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_lookup_tracks_function() {
        let table = OneDimLookup::<2048>::new(|x| (x * std::f32::consts::TAU).sin(), 1.0);
        for &x in &[0.1f32, 0.25, 0.4, 0.6, 0.9] {
            let got = table.cubic_lookup(PolyF32::splat(x)).lane(0);
            let want = (x * std::f32::consts::TAU).sin();
            assert!((got - want).abs() < 1e-2, "x={x} got={got} want={want}");
        }
    }
}
