//! Modulation connection transform (rework of Vital's
//! `modulation_connection_processor.cpp`).
//!
//! Each connection turns a source value in `[0, 1]` into a destination
//! offset: optional remap through a drawn curve, bipolar recentering,
//! power-curve morphing, stereo sign flip on right lanes, then scaling by
//! the connection amount and the destination's engine range.

use spinwave_poly::math::power_scale;
use spinwave_poly::utils::{catmull_interpolation_matrix, value_matrix};
use spinwave_poly::{PolyF32, PolyMask};

/// `[0, 1, 0, 1]`: selects right-channel lanes.
#[inline(always)]
fn right_one() -> PolyF32 {
    PolyF32::stereo(0.0, 1.0)
}

/// A drawn remap curve, sampled into a cubic-interpolation-ready buffer
/// (one guard value before, two after: Catmull-Rom reads 4 taps from
/// `index`). Provided by the LineGenerator in `spinwave-dsp`.
#[derive(Clone, Copy, Debug)]
pub struct ModRemap<'a> {
    pub buffer: &'a [f32],
    pub resolution: f32,
}

/// An owned remap curve (a preset's `line_mapping`), shared between the
/// kernels through an `Arc` so installing it on the audio thread is a
/// refcount bump, never a copy.
#[derive(Clone, Debug)]
pub struct RemapCurve {
    /// The LineGenerator's cubic interpolation buffer (`resolution + 3`
    /// values).
    pub buffer: Vec<f32>,
    pub resolution: f32,
}

impl RemapCurve {
    /// Snapshots a rendered LineGenerator.
    pub fn from_line_generator(generator: &spinwave_dsp::modulators::LineGenerator) -> RemapCurve {
        RemapCurve {
            buffer: generator.cubic_interpolation_buffer().to_vec(),
            resolution: generator.resolution() as f32,
        }
    }

    #[inline(always)]
    pub fn view(&self) -> ModRemap<'_> {
        ModRemap { buffer: &self.buffer, resolution: self.resolution }
    }
}

impl ModRemap<'_> {
    #[inline(always)]
    fn apply(&self, value: PolyF32) -> PolyF32 {
        let boost = (value * self.resolution).clamp(0.0, self.resolution);
        let indices = boost
            .min(PolyF32::splat(self.resolution - 1.0))
            .max(PolyF32::ZERO)
            .to_i32_round();
        let t = boost - indices.to_f32_signed();

        let interpolation_matrix = catmull_interpolation_matrix(t);
        let mut values = value_matrix(self.buffer, indices);
        values.transpose();
        interpolation_matrix.multiply_and_sum_rows(&values).clamp(-1.0, 1.0)
    }
}

/// Output of a control-rate connection evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModOutput {
    /// Before destination scaling (drives the modulation meters).
    pub raw: PolyF32,
    /// After destination scaling (added to the destination's base value).
    pub scaled: PolyF32,
}

/// One modulation connection's value transform, with the per-block
/// smoothing state for click-free audio-rate amount/power sweeps.
#[derive(Clone, Debug)]
pub struct ModulationTransform {
    /// Connection amount in `[-1, 1]` (itself modulatable).
    pub amount: PolyF32,
    /// Morph power; 0 = linear.
    pub power: PolyF32,
    pub bipolar: bool,
    pub stereo: bool,
    pub bypass: bool,
    /// Destination's engine-unit range (display scale of the target param).
    pub destination_scale: f32,
    /// Optional drawn remap of the source value (`line_mapping`).
    pub remap: Option<std::sync::Arc<RemapCurve>>,
    /// This connection's slot (`modulation_{slot+1}_*`), so a meta
    /// connection can target its amount or power.
    pub slot: usize,
    /// What meta-modulation adds to `amount` and `power` this block, set
    /// by the matrix before the connection is evaluated
    /// (notes/meta-modulation.md): the sum is clamped to [-1, 1] for the
    /// amount, not clamped for the power, both measured.
    pub amount_offset: PolyF32,
    pub power_offset: PolyF32,

    last_destination_scale: f32,
    current_amount: PolyF32,
    current_power: PolyF32,
}

impl Default for ModulationTransform {
    fn default() -> Self {
        ModulationTransform {
            amount: PolyF32::ZERO,
            power: PolyF32::ZERO,
            bipolar: false,
            stereo: false,
            bypass: false,
            slot: 0,
            amount_offset: PolyF32::ZERO,
            power_offset: PolyF32::ZERO,
            destination_scale: 1.0,
            remap: None,
            last_destination_scale: 0.0,
            current_amount: PolyF32::ZERO,
            current_power: PolyF32::ZERO,
        }
    }
}

impl ModulationTransform {
    /// A fresh connection with the given amount and destination scale.
    pub fn with_amount(amount: f32, destination_scale: f32) -> ModulationTransform {
        ModulationTransform {
            amount: PolyF32::splat(amount),
            destination_scale,
            ..Default::default()
        }
    }

    #[inline(always)]
    fn stereo_scale(&self) -> PolyF32 {
        let stereo = if self.stereo { 1.0 } else { 0.0 };
        PolyF32::ONE - right_one() * 2.0 * stereo
    }

    #[inline(always)]
    fn bipolar_value(&self) -> f32 {
        if self.bipolar {
            1.0
        } else {
            0.0
        }
    }

    /// Retargets smoothing when the connection is (re)pointed at a new
    /// destination; call once per block before processing.
    fn refresh_destination(&mut self) {
        if self.last_destination_scale != self.destination_scale {
            self.current_amount = PolyF32::ZERO;
        }
        self.last_destination_scale = self.destination_scale;
    }

    /// Control-rate evaluation: one value per block, through the
    /// connection's own remap curve when it has one.
    pub fn process_control(&mut self, source: PolyF32) -> ModOutput {
        // Move the curve out for the call (no refcount traffic) so the
        // borrow of `self` stays exclusive.
        let curve = self.remap.take();
        let output = self.process_control_with(source, curve.as_deref().map(RemapCurve::view).as_ref());
        self.remap = curve;
        output
    }

    /// Audio-rate evaluation through the connection's own remap curve.
    pub fn process_audio(&mut self, source: &[PolyF32], dest: &mut [PolyF32], reset_mask: PolyMask) {
        let curve = self.remap.take();
        self.process_audio_with(source, dest, reset_mask, curve.as_deref().map(RemapCurve::view).as_ref());
        self.remap = curve;
    }

    /// Control-rate evaluation with an explicit remap curve.
    pub fn process_control_with(&mut self, source: PolyF32, remap: Option<&ModRemap>) -> ModOutput {
        self.refresh_destination();
        if self.bypass {
            return ModOutput::default();
        }

        let mut modulation_input = source.clamp(0.0, 1.0);
        if let Some(remap) = remap {
            modulation_input = remap.apply(modulation_input);
        }

        let bipolar = PolyF32::splat(self.bipolar_value());
        let polarity_pre_scale = bipolar + 1.0;
        let polarity_post_scale = (bipolar * -0.5 + 1.0) * self.stereo_scale();

        let modulation_shift = modulation_input * polarity_pre_scale - bipolar;
        let modulation_abs = modulation_shift.abs();
        let sign_mask = modulation_shift.sign_mask();

        let power = -(self.power + self.power_offset);
        let shifted_modulation = power_scale(modulation_abs, power);
        let modulation_amount = (self.amount + self.amount_offset).clamp(-1.0, 1.0);
        let pre_modulation = modulation_amount * shifted_modulation;
        let raw = (pre_modulation ^ sign_mask) * polarity_post_scale;

        debug_assert!(raw.is_finite());
        ModOutput { raw, scaled: raw * self.destination_scale }
    }

    /// Audio-rate evaluation with per-sample smoothing of amount/power and
    /// an explicit remap curve. `reset_mask`: lanes whose voice restarted
    /// (smoothing jumps).
    pub fn process_audio_with(
        &mut self,
        source: &[PolyF32],
        dest: &mut [PolyF32],
        reset_mask: PolyMask,
        remap: Option<&ModRemap>,
    ) {
        self.refresh_destination();
        let num_samples = source.len();
        debug_assert_eq!(dest.len(), num_samples);

        if self.bypass {
            dest.fill(PolyF32::ZERO);
            return;
        }

        let power = -(self.power + self.power_offset);
        let using_power = power.ne(PolyF32::ZERO).any() || self.current_power.ne(PolyF32::ZERO).any();

        if using_power {
            self.process_audio_morphed(source, dest, reset_mask, remap, power);
        } else {
            self.process_audio_linear(source, dest, reset_mask, remap);
        }
        self.current_power = power;
    }

    fn process_audio_linear(
        &mut self,
        source: &[PolyF32],
        dest: &mut [PolyF32],
        reset_mask: PolyMask,
        remap: Option<&ModRemap>,
    ) {
        let bipolar_offset = PolyF32::splat(-self.bipolar_value() * 0.5);
        let modulation_amount = (self.amount + self.amount_offset).clamp(-1.0, 1.0) * self.stereo_scale();
        let target_amount = modulation_amount * self.destination_scale;

        let mut current_amount = reset_mask.select(target_amount, self.current_amount);
        self.current_amount = target_amount;
        let delta_amount = (target_amount - current_amount) * (1.0 / source.len() as f32);

        for (out, &value) in dest.iter_mut().zip(source) {
            current_amount += delta_amount;
            let value = match remap {
                Some(remap) => remap.apply(value),
                None => value,
            };
            *out = (value + bipolar_offset) * current_amount;
        }
    }

    fn process_audio_morphed(
        &mut self,
        source: &[PolyF32],
        dest: &mut [PolyF32],
        reset_mask: PolyMask,
        remap: Option<&ModRemap>,
        power: PolyF32,
    ) {
        let bipolar = PolyF32::splat(self.bipolar_value());
        let polarity_pre_scale = bipolar + 1.0;
        let polarity_post_scale = (bipolar * -0.5 + 1.0) * self.stereo_scale();

        let modulation_amount = (self.amount + self.amount_offset).clamp(-1.0, 1.0);
        let target_amount = modulation_amount * self.destination_scale;

        let mut current_amount = reset_mask.select(target_amount, self.current_amount);
        let mut current_power = reset_mask.select(power, self.current_power);
        self.current_amount = target_amount;

        let sample_inc = 1.0 / source.len() as f32;
        let delta_amount = (target_amount - current_amount) * sample_inc;
        let delta_power = (power - current_power) * sample_inc;

        for (out, &value) in dest.iter_mut().zip(source) {
            current_amount += delta_amount;
            current_power += delta_power;

            let value = match remap {
                Some(remap) => remap.apply(value),
                None => value,
            };
            let modulation_shift = value * polarity_pre_scale - bipolar;
            let modulation_abs = modulation_shift.abs();
            let sign_mask = modulation_shift.sign_mask();

            let shifted_modulation = power_scale(modulation_abs, current_power);
            let pre_modulation = current_amount * shifted_modulation;
            *out = (pre_modulation ^ sign_mask) * polarity_post_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(amount: f32) -> ModulationTransform {
        ModulationTransform { amount: PolyF32::splat(amount), ..Default::default() }
    }

    #[test]
    fn unipolar_identity() {
        let mut t = transform(1.0);
        let out = t.process_control(PolyF32::splat(0.75));
        assert!((out.scaled.lane(0) - 0.75).abs() < 1e-5);
        assert_eq!(out.raw.lane(0), out.scaled.lane(0));
    }

    #[test]
    fn bipolar_recenters() {
        let mut t = transform(1.0);
        t.bipolar = true;
        assert!((t.process_control(PolyF32::splat(0.5)).scaled.lane(0)).abs() < 1e-5);
        assert!((t.process_control(PolyF32::splat(1.0)).scaled.lane(0) - 0.5).abs() < 1e-5);
        assert!((t.process_control(PolyF32::splat(0.0)).scaled.lane(0) + 0.5).abs() < 1e-5);
    }

    #[test]
    fn stereo_flips_right_lanes() {
        let mut t = transform(1.0);
        t.stereo = true;
        let out = t.process_control(PolyF32::splat(0.6)).scaled;
        assert!((out.lane(0) - 0.6).abs() < 1e-5);
        assert!((out.lane(1) + 0.6).abs() < 1e-5);
    }

    #[test]
    fn bypass_outputs_zero() {
        let mut t = transform(1.0);
        t.bypass = true;
        let out = t.process_control(PolyF32::splat(0.9));
        assert_eq!(out.scaled.lane(0), 0.0);
    }

    #[test]
    fn destination_scale_applies_to_scaled_only() {
        let mut t = transform(1.0);
        t.destination_scale = 24.0;
        let out = t.process_control(PolyF32::splat(0.5));
        assert!((out.raw.lane(0) - 0.5).abs() < 1e-5);
        assert!((out.scaled.lane(0) - 12.0).abs() < 1e-4);
    }

    #[test]
    fn negative_amount_inverts() {
        let mut t = transform(-1.0);
        let out = t.process_control(PolyF32::splat(0.5));
        assert!((out.scaled.lane(0) + 0.5).abs() < 1e-5);
    }

    #[test]
    fn audio_rate_ramps_amount_smoothly() {
        let mut t = transform(1.0);
        let source = vec![PolyF32::splat(1.0); 64];
        let mut dest = vec![PolyF32::ZERO; 64];
        // First block starts from amount 0 (fresh connection) and ramps up.
        t.process_audio(&source, &mut dest, PolyMask::NONE);
        assert!(dest[0].lane(0) < dest[63].lane(0));
        assert!((dest[63].lane(0) - 1.0).abs() < 0.05);

        // Second block is steady.
        t.process_audio(&source, &mut dest, PolyMask::NONE);
        assert!((dest[0].lane(0) - 1.0).abs() < 0.05);
    }

    #[test]
    fn power_morph_bends_the_curve() {
        let mut t = transform(1.0);
        t.power = PolyF32::splat(5.0);
        let mid = t.process_control(PolyF32::splat(0.5)).scaled.lane(0);
        // The power input is negated internally: positive power bends
        // midpoints up (powerScale(0.5, -5) â‰ˆ 0.92); endpoints stay fixed.
        assert!(mid > 0.55, "mid was {mid}");
        let mut t_end = transform(1.0);
        t_end.power = PolyF32::splat(5.0);
        let end = t_end.process_control(PolyF32::splat(1.0)).scaled.lane(0);
        assert!((end - 1.0).abs() < 1e-4);
    }
}
