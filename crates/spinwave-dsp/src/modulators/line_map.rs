//! Maps a control value through a [`LineGenerator`] curve (port of `LineMap`).
//!
//! Cubic-Lagrange interpolation into the curve's guarded buffer; the result
//! is clamped to `[-1, 1]`.

use spinwave_poly::{utils, PolyF32, PolyU32};

use super::line_generator::LineGenerator;

/// Evaluates the curve at `phase` (clamped to `[0, 1]`).
pub fn process(source: &LineGenerator, phase: PolyF32) -> PolyF32 {
    let buffer = source.cubic_interpolation_buffer();
    let resolution = source.resolution();
    let resolution_f = resolution as f32;

    let boost = (phase * resolution_f).clamp(0.0, resolution_f);
    let indices = boost.to_i32_round().min(PolyU32::splat(resolution as u32 - 1));
    let t = boost - indices.to_f32_signed();

    let interpolation_matrix = utils::polynomial_interpolation_matrix(t);
    let mut values = utils::value_matrix(buffer, indices);
    values.transpose();

    interpolation_matrix.multiply_and_sum_rows(&values).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_map_is_identityish() {
        // The linear preset renders a 0..1 ramp: mapping x returns ~x.
        let source = LineGenerator::linear();
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let mapped = process(&source, PolyF32::splat(x)).lane(0);
            assert!((mapped - x).abs() < 2e-3, "map({x}) = {mapped}");
        }
    }

    #[test]
    fn clamps_out_of_range_phase() {
        let source = LineGenerator::linear();
        let below = process(&source, PolyF32::splat(-1.0)).lane(0);
        let above = process(&source, PolyF32::splat(2.0)).lane(0);
        assert!((below - 0.0).abs() < 1e-3);
        assert!((above - 1.0).abs() < 1e-3);
    }

    #[test]
    fn follows_flipped_curve() {
        let mut source = LineGenerator::linear();
        source.flip_vertical();
        // Flipping y turns the ramp into 1..0.
        let start = process(&source, PolyF32::splat(0.0)).lane(0);
        let end = process(&source, PolyF32::splat(1.0)).lane(0);
        assert!((start - 1.0).abs() < 1e-3);
        assert!((end - 0.0).abs() < 1e-3);
    }
}
