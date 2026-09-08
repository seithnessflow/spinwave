//! Drawable line/curve source backing LFO shapes and remap curves.
//!
//! A list of `(x, y)` points with per-segment power curving and optional
//! smoothing, rendered into a lookup buffer with one guard sample in front
//! and two behind so 4-point interpolation never goes out of bounds.

use vital_poly::{math, PolyF32};

pub const MAX_POINTS: usize = 100;
pub const DEFAULT_RESOLUTION: usize = 2048;
/// Guard samples around the rendered buffer for cubic interpolation.
pub const EXTRA_VALUES: usize = 3;

/// Scalar power-scale response, matching `futils::powerScale(mono_float)`.
/// Uses the same polynomial exp approximation as the reference.
fn power_scale(value: f32, power: f32) -> f32 {
    const MIN_POWER: f32 = 0.01;
    if power.abs() < MIN_POWER {
        return value;
    }
    let numerator = math::exp(PolyF32::splat(power * value)).lane(0) - 1.0;
    let denominator = math::exp(PolyF32::splat(power)).lane(0) - 1.0;
    numerator / denominator
}

/// Sine-eased transition used when smoothing is on.
#[inline]
pub fn smooth_transition(t: f32) -> f32 {
    0.5 * ((t - 0.5) * core::f32::consts::PI).sin() + 0.5
}

#[derive(Clone)]
pub struct LineGenerator {
    points: [(f32, f32); MAX_POINTS],
    powers: [f32; MAX_POINTS],
    num_points: usize,
    resolution: usize,
    buffer: Vec<f32>,
    looping: bool,
    smooth: bool,
    linear: bool,
    render_count: usize,
}

impl Default for LineGenerator {
    fn default() -> Self {
        LineGenerator::new(DEFAULT_RESOLUTION)
    }
}

impl LineGenerator {
    pub fn new(resolution: usize) -> Self {
        let mut generator = LineGenerator {
            points: [(0.0, 0.0); MAX_POINTS],
            powers: [0.0; MAX_POINTS],
            num_points: 2,
            resolution,
            buffer: vec![0.0; resolution + EXTRA_VALUES],
            looping: false,
            smooth: false,
            linear: true,
            render_count: 0,
        };
        generator.init_linear();
        generator
    }

    // -- Preset shapes -------------------------------------------------------

    pub fn linear() -> Self {
        Self::default()
    }

    pub fn triangle() -> Self {
        let mut generator = Self::default();
        generator.init_triangle();
        generator
    }

    pub fn square() -> Self {
        let mut generator = Self::default();
        generator.init_square();
        generator
    }

    pub fn sin() -> Self {
        let mut generator = Self::default();
        generator.init_sin();
        generator
    }

    pub fn saw_up() -> Self {
        let mut generator = Self::default();
        generator.init_saw_up();
        generator
    }

    pub fn saw_down() -> Self {
        let mut generator = Self::default();
        generator.init_saw_down();
        generator
    }

    pub fn init_linear(&mut self) {
        self.set_shape(&[(0.0, 1.0), (1.0, 0.0)], false);
        self.linear = true;
        self.render();
    }

    pub fn init_triangle(&mut self) {
        self.set_shape(&[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)], false);
        self.render();
    }

    pub fn init_square(&mut self) {
        self.set_shape(&[(0.0, 1.0), (0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (1.0, 1.0)], false);
        self.render();
    }

    pub fn init_sin(&mut self) {
        self.set_shape(&[(0.0, 1.0), (0.5, 0.0), (1.0, 1.0)], true);
        self.render();
    }

    pub fn init_saw_up(&mut self) {
        self.set_shape(&[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)], false);
        self.render();
    }

    pub fn init_saw_down(&mut self) {
        self.set_shape(&[(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)], false);
        self.render();
    }

    fn set_shape(&mut self, points: &[(f32, f32)], smooth: bool) {
        self.num_points = points.len();
        self.points[..points.len()].copy_from_slice(points);
        self.powers[..points.len()].fill(0.0);
        self.linear = false;
        self.smooth = smooth;
    }

    // -- Accessors -----------------------------------------------------------

    #[inline]
    pub fn resolution(&self) -> usize {
        self.resolution
    }

    #[inline]
    pub fn is_linear(&self) -> bool {
        self.linear
    }

    #[inline]
    pub fn smooth(&self) -> bool {
        self.smooth
    }

    #[inline]
    pub fn looping(&self) -> bool {
        self.looping
    }

    /// Rendered values, one per resolution step (excludes guard samples).
    #[inline]
    pub fn buffer(&self) -> &[f32] {
        &self.buffer[1..=self.resolution]
    }

    /// Full guarded buffer: index `i` holds the value one step before
    /// rendered sample `i`, for 4-tap interpolation.
    #[inline]
    pub fn cubic_interpolation_buffer(&self) -> &[f32] {
        &self.buffer
    }

    #[inline]
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    #[inline]
    pub fn point(&self, index: usize) -> (f32, f32) {
        self.points[index]
    }

    #[inline]
    pub fn power(&self, index: usize) -> f32 {
        self.powers[index]
    }

    pub fn last_point(&self) -> (f32, f32) {
        self.points[self.num_points - 1]
    }

    pub fn last_power(&self) -> f32 {
        self.powers[self.num_points - 1]
    }

    pub fn render_count(&self) -> usize {
        self.render_count
    }

    // -- Editing -------------------------------------------------------------

    pub fn set_loop(&mut self, looping: bool) {
        self.looping = looping;
        self.render();
    }

    pub fn set_smooth(&mut self, smooth: bool) {
        self.smooth = smooth;
        self.check_line_is_linear();
        self.render();
    }

    pub fn set_point(&mut self, index: usize, point: (f32, f32)) {
        self.points[index] = point;
        self.check_line_is_linear();
    }

    pub fn set_power(&mut self, index: usize, power: f32) {
        self.powers[index] = power;
        self.check_line_is_linear();
    }

    pub fn set_num_points(&mut self, num_points: usize) {
        debug_assert!(num_points <= MAX_POINTS);
        self.num_points = num_points;
        self.check_line_is_linear();
    }

    pub fn add_point(&mut self, index: usize, position: (f32, f32)) {
        debug_assert!(self.num_points < MAX_POINTS);
        let mut i = self.num_points;
        while i > index {
            self.points[i] = self.points[i - 1];
            self.powers[i] = self.powers[i - 1];
            i -= 1;
        }
        self.num_points += 1;
        self.points[index] = position;
        self.powers[index] = 0.0;
        self.check_line_is_linear();
    }

    pub fn add_middle_point(&mut self, index: usize) {
        debug_assert!(index > 0);
        let x = (self.points[index - 1].0 + self.points[index].0) * 0.5;
        let y = self.value_between_points(x, index - 1, index);
        self.add_point(index, (x, y));
    }

    pub fn remove_point(&mut self, index: usize) {
        self.num_points -= 1;
        for i in index..self.num_points {
            self.points[i] = self.points[i + 1];
            self.powers[i] = self.powers[i + 1];
        }
        self.check_line_is_linear();
    }

    pub fn flip_horizontal(&mut self) {
        let n = self.num_points;
        for i in 0..n.div_ceil(2) {
            let tmp_x = 1.0 - self.points[i].0;
            let tmp_y = self.points[i].1;
            self.points[i].0 = 1.0 - self.points[n - i - 1].0;
            self.points[i].1 = self.points[n - i - 1].1;
            self.points[n - i - 1].0 = tmp_x;
            self.points[n - i - 1].1 = tmp_y;
        }
        for i in 0..n / 2 {
            let tmp_power = self.powers[i];
            self.powers[i] = -self.powers[n - i - 2];
            self.powers[n - i - 2] = -tmp_power;
        }
        self.render();
        self.check_line_is_linear();
    }

    pub fn flip_vertical(&mut self) {
        for i in 0..self.num_points {
            self.points[i].1 = 1.0 - self.points[i].1;
        }
        self.render();
        self.check_line_is_linear();
    }

    fn check_line_is_linear(&mut self) {
        self.linear = !self.smooth
            && self.num_points == 2
            && self.powers[0] == 0.0
            && self.points[0] == (0.0, 1.0)
            && self.points[1] == (1.0, 0.0);
    }

    // -- Rendering and evaluation --------------------------------------------

    /// Rasterizes the point list into the lookup buffer. Rendered values are
    /// `1 - y` so the buffer reads top-of-editor as 0.
    pub fn render(&mut self) {
        self.render_count += 1;

        let mut point_index = 0usize;
        let mut last_point = self.points[0];
        let mut current_power = 0.0f32;
        let mut current_point = self.points[0];
        if self.looping {
            last_point = self.points[self.num_points - 1];
            last_point.0 -= 1.0;
            current_power = self.powers[self.num_points - 1];
        }

        let resolution = self.resolution;
        for i in 0..resolution {
            let x = i as f32 / (resolution as f32 - 1.0);
            let mut t = 1.0f32;
            if current_point.0 > last_point.0 {
                t = (x - last_point.0) / (current_point.0 - last_point.0);
            }

            if self.smooth {
                t = smooth_transition(t);
            }

            t = power_scale(t, current_power).clamp(0.0, 1.0);

            let y = last_point.1 + t * (current_point.1 - last_point.1);
            self.buffer[i + 1] = 1.0 - y;

            while x > current_point.0 && point_index < self.num_points {
                current_power = self.powers[point_index % self.num_points];
                point_index += 1;
                last_point = current_point;
                current_point = self.points[point_index % self.num_points];
                if point_index >= self.num_points {
                    current_point.0 += 1.0;
                    break;
                }
            }
        }

        if self.looping {
            self.buffer[0] = self.buffer[resolution];
            self.buffer[resolution + 1] = self.buffer[1];
            self.buffer[resolution + 2] = self.buffer[2];
        } else {
            self.buffer[0] = self.buffer[1];
            self.buffer[resolution + 1] = self.buffer[resolution];
            self.buffer[resolution + 2] = self.buffer[resolution];
        }
    }

    /// Linear interpolation into the rendered buffer.
    pub fn value_at_phase(&self, phase: f32) -> f32 {
        let scaled_phase = phase.clamp(0.0, 1.0) * self.resolution as f32;
        let index = scaled_phase as usize;
        let t = scaled_phase - index as f32;
        let from = self.buffer[index + 1];
        let to = self.buffer[index + 2];
        from + t * (to - from)
    }

    /// Exact curve evaluation between two points (editor precision, not the
    /// rasterized buffer).
    pub fn value_between_points(&self, x: f32, index_from: usize, index_to: usize) -> f32 {
        debug_assert!(index_to < self.num_points);

        let first = self.points[index_from];
        let second = self.points[index_to];
        let power = self.powers[index_from];

        let width = second.0 - first.0;
        if width <= 0.0 {
            return second.1;
        }

        let mut t = (x - first.0) / width;
        if self.smooth {
            t = smooth_transition(t);
        }

        t = power_scale(t, power).clamp(0.0, 1.0);
        t * (second.1 - first.1) + first.1
    }

    /// Exact curve evaluation at a phase, walking the point list.
    pub fn exact_value_at_phase(&self, phase: f32) -> f32 {
        for i in 0..self.num_points - 1 {
            if self.points[i].0 <= phase && self.points[i + 1].0 >= phase {
                return self.value_between_points(phase, i, i + 1);
            }
        }
        self.last_point().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_shape_renders_ramp() {
        let generator = LineGenerator::linear();
        assert!(generator.is_linear());
        // Points go (0,1) -> (1,0); buffer stores 1-y so it ramps 0 -> 1.
        assert!((generator.value_at_phase(0.0) - 0.0).abs() < 1e-3);
        assert!((generator.value_at_phase(0.5) - 0.5).abs() < 1e-3);
        assert!((generator.value_at_phase(1.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn triangle_shape() {
        let generator = LineGenerator::triangle();
        assert!((generator.value_at_phase(0.0) - 0.0).abs() < 1e-3);
        assert!((generator.value_at_phase(0.25) - 0.5).abs() < 1e-3);
        assert!((generator.value_at_phase(0.5) - 1.0).abs() < 1e-3);
        assert!((generator.value_at_phase(0.75) - 0.5).abs() < 1e-3);
        assert!((generator.value_at_phase(1.0) - 0.0).abs() < 1e-3);
    }

    #[test]
    fn square_shape() {
        let generator = LineGenerator::square();
        // First half high (1 - y with y=0), second half low.
        assert!((generator.value_at_phase(0.25) - 1.0).abs() < 1e-3);
        assert!((generator.value_at_phase(0.75) - 0.0).abs() < 1e-3);
    }

    #[test]
    fn sin_shape_is_smooth() {
        let generator = LineGenerator::sin();
        assert!(generator.smooth());
        // Sine eased: halfway between points hits 0.5, endpoints flat.
        assert!((generator.value_at_phase(0.25) - 0.5).abs() < 2e-3);
        assert!((generator.value_at_phase(0.5) - 1.0).abs() < 1e-3);
        // Slope near the ends is much flatter than linear.
        let start_slope = generator.value_at_phase(0.02) - generator.value_at_phase(0.0);
        assert!(start_slope < 0.02);
    }

    #[test]
    fn guard_samples_no_loop() {
        let generator = LineGenerator::linear();
        let buffer = generator.cubic_interpolation_buffer();
        let resolution = generator.resolution();
        assert_eq!(buffer.len(), resolution + EXTRA_VALUES);
        assert_eq!(buffer[0], buffer[1]);
        assert_eq!(buffer[resolution + 1], buffer[resolution]);
        assert_eq!(buffer[resolution + 2], buffer[resolution]);
    }

    #[test]
    fn guard_samples_loop() {
        let mut generator = LineGenerator::triangle();
        generator.set_loop(true);
        let buffer = generator.cubic_interpolation_buffer();
        let resolution = generator.resolution();
        assert_eq!(buffer[0], buffer[resolution]);
        assert_eq!(buffer[resolution + 1], buffer[1]);
        assert_eq!(buffer[resolution + 2], buffer[2]);
    }

    #[test]
    fn add_and_remove_points() {
        let mut generator = LineGenerator::linear();
        generator.add_middle_point(1);
        assert_eq!(generator.num_points(), 3);
        let (x, y) = generator.point(1);
        assert!((x - 0.5).abs() < 1e-6);
        assert!((y - 0.5).abs() < 1e-6);
        assert!(!generator.is_linear());
        generator.remove_point(1);
        assert_eq!(generator.num_points(), 2);
        assert!(generator.is_linear());
    }

    #[test]
    fn flips_are_involutions() {
        let mut generator = LineGenerator::saw_up();
        let before: Vec<f32> = generator.buffer().to_vec();
        generator.flip_vertical();
        generator.flip_vertical();
        let after: Vec<f32> = generator.buffer().to_vec();
        assert_eq!(before, after);

        generator.flip_horizontal();
        generator.flip_horizontal();
        let after: Vec<f32> = generator.buffer().to_vec();
        assert_eq!(before, after);
    }

    #[test]
    fn power_curves_bend_segments() {
        let mut generator = LineGenerator::linear();
        generator.set_power(0, 5.0);
        generator.render();
        // Positive power on a descending y segment: midpoint below linear.
        let mid = generator.value_at_phase(0.5);
        assert!(mid < 0.4, "expected bent curve, got {mid}");
    }
}
