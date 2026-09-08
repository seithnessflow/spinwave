//! SIMD voice-pair primitives.
//!
//! The engine processes two stereo voices per vector, lane layout:
//! `[voice0_L, voice0_R, voice1_L, voice1_R]`. All per-voice state
//! (note-on resets, envelopes, modulation) is expressed through lane
//! masks so both voices advance in lock-step without branches.

use bytemuck::cast;
use wide::{f32x4, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, CmpNe};

/// Number of SIMD lanes: two stereo voices.
pub const LANES: usize = 4;

/// A packed pair of stereo voices.
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct PolyF32(pub f32x4);

/// Packed unsigned integers, one per lane (phase accumulators, bit tricks).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct PolyU32(pub [u32; LANES]);

/// Per-lane boolean mask (all bits set = true). Stored as float bits so it
/// composes with [`PolyF32`] bitwise ops without casts in the hot path.
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct PolyMask(pub f32x4);

// ---------------------------------------------------------------------------
// PolyF32
// ---------------------------------------------------------------------------

impl PolyF32 {
    pub const ZERO: PolyF32 = PolyF32(f32x4::ZERO);
    pub const ONE: PolyF32 = PolyF32(f32x4::ONE);

    /// `[l, r, l, r]` — the same stereo value for both voices.
    #[inline(always)]
    pub fn stereo(l: f32, r: f32) -> PolyF32 {
        PolyF32(f32x4::from([l, r, l, r]))
    }

    #[inline(always)]
    pub fn splat(value: f32) -> PolyF32 {
        PolyF32(f32x4::splat(value))
    }

    #[inline(always)]
    pub fn from_lanes(lanes: [f32; LANES]) -> PolyF32 {
        PolyF32(f32x4::from(lanes))
    }

    #[inline(always)]
    pub fn to_lanes(self) -> [f32; LANES] {
        self.0.to_array()
    }

    #[inline(always)]
    pub fn lane(self, i: usize) -> f32 {
        self.to_lanes()[i]
    }

    #[inline(always)]
    pub fn set_lane(&mut self, i: usize, value: f32) {
        let mut lanes = self.to_lanes();
        lanes[i] = value;
        *self = PolyF32::from_lanes(lanes);
    }

    /// Applies a scalar function to every lane. Escape hatch for the few
    /// cold paths with no vector form; never use it per-sample.
    #[inline(always)]
    pub fn map(self, f: impl Fn(f32) -> f32) -> PolyF32 {
        let l = self.to_lanes();
        PolyF32::from_lanes([f(l[0]), f(l[1]), f(l[2]), f(l[3])])
    }

    /// `self + b * c` (Vital's `mulAdd`).
    #[inline(always)]
    pub fn mul_add(self, b: PolyF32, c: PolyF32) -> PolyF32 {
        PolyF32(b.0.mul_add(c.0, self.0))
    }

    /// `self - b * c` (Vital's `mulSub`).
    #[inline(always)]
    pub fn mul_sub(self, b: PolyF32, c: PolyF32) -> PolyF32 {
        PolyF32((-b.0).mul_add(c.0, self.0))
    }

    #[inline(always)]
    pub fn abs(self) -> PolyF32 {
        PolyF32(self.0.abs())
    }

    #[inline(always)]
    pub fn min(self, other: PolyF32) -> PolyF32 {
        PolyF32(self.0.min(other.0))
    }

    #[inline(always)]
    pub fn max(self, other: PolyF32) -> PolyF32 {
        PolyF32(self.0.max(other.0))
    }

    #[inline(always)]
    pub fn clamp(self, min: f32, max: f32) -> PolyF32 {
        self.min(PolyF32::splat(max)).max(PolyF32::splat(min))
    }

    #[inline(always)]
    pub fn sqrt(self) -> PolyF32 {
        PolyF32(self.0.sqrt())
    }

    #[inline(always)]
    pub fn eq(self, other: PolyF32) -> PolyMask {
        PolyMask(self.0.cmp_eq(other.0))
    }

    #[inline(always)]
    pub fn ne(self, other: PolyF32) -> PolyMask {
        PolyMask(self.0.cmp_ne(other.0))
    }

    #[inline(always)]
    pub fn lt(self, other: PolyF32) -> PolyMask {
        PolyMask(self.0.cmp_lt(other.0))
    }

    #[inline(always)]
    pub fn le(self, other: PolyF32) -> PolyMask {
        PolyMask(self.0.cmp_le(other.0))
    }

    #[inline(always)]
    pub fn gt(self, other: PolyF32) -> PolyMask {
        PolyMask(self.0.cmp_gt(other.0))
    }

    #[inline(always)]
    pub fn ge(self, other: PolyF32) -> PolyMask {
        PolyMask(self.0.cmp_ge(other.0))
    }

    /// Sum of all four lanes.
    #[inline(always)]
    pub fn sum_lanes(self) -> f32 {
        let l = self.to_lanes();
        l[0] + l[1] + l[2] + l[3]
    }

    #[inline(always)]
    pub fn to_u32_bits(self) -> PolyU32 {
        cast(self)
    }

    #[inline(always)]
    pub fn from_u32_bits(bits: PolyU32) -> PolyF32 {
        cast(bits)
    }

    /// Truncation toward zero, per lane.
    #[inline(always)]
    pub fn trunc(self) -> PolyF32 {
        PolyF32(self.0.trunc_int().round_float())
    }

    #[inline(always)]
    pub fn floor(self) -> PolyF32 {
        PolyF32(self.0.floor())
    }

    #[inline(always)]
    pub fn ceil(self) -> PolyF32 {
        PolyF32(self.0.ceil())
    }

    /// Round half away from zero like Vital's `floor(x + 0.5)`.
    #[inline(always)]
    pub fn round(self) -> PolyF32 {
        (self + PolyF32::splat(0.5)).floor()
    }

    /// Fractional part in `[0, 1)` (Vital's `utils::mod`).
    #[inline(always)]
    pub fn fract(self) -> PolyF32 {
        self - self.floor()
    }

    /// Round-to-nearest-even conversion, matching Vital's SSE2 `toInt`.
    #[inline(always)]
    pub fn to_i32_round(self) -> PolyU32 {
        cast(self.0.round_int())
    }

    #[inline(always)]
    pub fn to_i32_floor(self) -> PolyU32 {
        cast(self.0.floor().trunc_int())
    }

    #[inline(always)]
    pub fn to_i32_trunc(self) -> PolyU32 {
        cast(self.0.trunc_int())
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.to_lanes().iter().all(|v| v.is_finite())
    }

    /// Per-lane sign bits as a mask (set where the lane is negative-signed).
    #[inline(always)]
    pub fn sign_mask(self) -> PolyMask {
        let sign_bits = PolyF32::from_u32_bits(PolyU32::splat(0x8000_0000));
        PolyMask(self.0 & sign_bits.0)
    }

    /// `[R0, L0, R1, L1]` — swaps left/right within each voice.
    #[inline(always)]
    pub fn swap_stereo(self) -> PolyF32 {
        let l = self.to_lanes();
        PolyF32::from_lanes([l[1], l[0], l[3], l[2]])
    }

    /// `[L1, R1, L0, R0]` — swaps the two voices.
    #[inline(always)]
    pub fn swap_voices(self) -> PolyF32 {
        let l = self.to_lanes();
        PolyF32::from_lanes([l[2], l[3], l[0], l[1]])
    }

    /// `[a0, a2, a1, a3]` — interleaves inner lanes (Vital's `swapInner`).
    #[inline(always)]
    pub fn swap_inner(self) -> PolyF32 {
        let l = self.to_lanes();
        PolyF32::from_lanes([l[0], l[2], l[1], l[3]])
    }

    #[inline(always)]
    pub fn reverse(self) -> PolyF32 {
        let l = self.to_lanes();
        PolyF32::from_lanes([l[3], l[2], l[1], l[0]])
    }

    /// `[a0, b0, a1, b1]` — packs two mono voice pairs into stereo lanes.
    #[inline(always)]
    pub fn consolidate_audio(a: PolyF32, b: PolyF32) -> PolyF32 {
        let (x, y) = (a.to_lanes(), b.to_lanes());
        PolyF32::from_lanes([x[0], y[0], x[1], y[1]])
    }

    /// `[a0, a1, b0, b1]` — first voice of each input side by side.
    #[inline(always)]
    pub fn compact_first_voices(a: PolyF32, b: PolyF32) -> PolyF32 {
        let (x, y) = (a.to_lanes(), b.to_lanes());
        PolyF32::from_lanes([x[0], x[1], y[0], y[1]])
    }
}

impl From<f32> for PolyF32 {
    #[inline(always)]
    fn from(value: f32) -> Self {
        PolyF32::splat(value)
    }
}

macro_rules! poly_f32_binop {
    ($trait:ident, $method:ident) => {
        impl core::ops::$trait for PolyF32 {
            type Output = PolyF32;
            #[inline(always)]
            fn $method(self, rhs: PolyF32) -> PolyF32 {
                PolyF32(core::ops::$trait::$method(self.0, rhs.0))
            }
        }
        impl core::ops::$trait<f32> for PolyF32 {
            type Output = PolyF32;
            #[inline(always)]
            fn $method(self, rhs: f32) -> PolyF32 {
                PolyF32(core::ops::$trait::$method(self.0, f32x4::splat(rhs)))
            }
        }
        impl core::ops::$trait<PolyF32> for f32 {
            type Output = PolyF32;
            #[inline(always)]
            fn $method(self, rhs: PolyF32) -> PolyF32 {
                PolyF32(core::ops::$trait::$method(f32x4::splat(self), rhs.0))
            }
        }
    };
}

poly_f32_binop!(Add, add);
poly_f32_binop!(Sub, sub);
poly_f32_binop!(Mul, mul);
poly_f32_binop!(Div, div);

macro_rules! poly_f32_assign {
    ($trait:ident, $method:ident, $op_trait:ident, $op_method:ident) => {
        impl core::ops::$trait for PolyF32 {
            #[inline(always)]
            fn $method(&mut self, rhs: PolyF32) {
                *self = core::ops::$op_trait::$op_method(*self, rhs);
            }
        }
        impl core::ops::$trait<f32> for PolyF32 {
            #[inline(always)]
            fn $method(&mut self, rhs: f32) {
                *self = core::ops::$op_trait::$op_method(*self, rhs);
            }
        }
    };
}

poly_f32_assign!(AddAssign, add_assign, Add, add);
poly_f32_assign!(SubAssign, sub_assign, Sub, sub);
poly_f32_assign!(MulAssign, mul_assign, Mul, mul);
poly_f32_assign!(DivAssign, div_assign, Div, div);

impl core::ops::Neg for PolyF32 {
    type Output = PolyF32;
    #[inline(always)]
    fn neg(self) -> PolyF32 {
        PolyF32(-self.0)
    }
}

impl core::ops::BitAnd<PolyMask> for PolyF32 {
    type Output = PolyF32;
    #[inline(always)]
    fn bitand(self, rhs: PolyMask) -> PolyF32 {
        PolyF32(self.0 & rhs.0)
    }
}

/// Applies/flips sign bits carried in a mask (pairs with [`PolyF32::sign_mask`]).
impl core::ops::BitXor<PolyMask> for PolyF32 {
    type Output = PolyF32;
    #[inline(always)]
    fn bitxor(self, rhs: PolyMask) -> PolyF32 {
        PolyF32(self.0 ^ rhs.0)
    }
}

// ---------------------------------------------------------------------------
// PolyMask
// ---------------------------------------------------------------------------

impl PolyMask {
    pub const NONE: PolyMask = PolyMask(f32x4::ZERO);

    #[inline(always)]
    pub fn all_on() -> PolyMask {
        PolyMask(f32x4::ZERO.cmp_eq(f32x4::ZERO))
    }

    /// Per-lane select: lane from `if_true` where the mask is set,
    /// else from `if_false` (Vital's `maskLoad`, arguments flipped
    /// to read naturally).
    #[inline(always)]
    pub fn select(self, if_true: PolyF32, if_false: PolyF32) -> PolyF32 {
        PolyF32(self.0.blend(if_true.0, if_false.0))
    }

    #[inline(always)]
    pub fn select_u32(self, if_true: PolyU32, if_false: PolyU32) -> PolyU32 {
        let m: PolyU32 = cast(self);
        (if_true & m) | (if_false & !m)
    }

    #[inline(always)]
    pub fn any(self) -> bool {
        self.to_u32().0.iter().any(|&l| l != 0)
    }

    #[inline(always)]
    pub fn all(self) -> bool {
        self.to_u32().0.iter().all(|&l| l == u32::MAX)
    }

    #[inline(always)]
    pub fn to_u32(self) -> PolyU32 {
        cast(self)
    }

    #[inline(always)]
    pub fn from_u32(bits: PolyU32) -> PolyMask {
        cast(bits)
    }

    /// Voice-level view: true if any lane of the corresponding voice is set.
    #[inline(always)]
    pub fn voice_any(self, voice: usize) -> bool {
        let l = self.to_u32().0;
        l[voice * 2] != 0 || l[voice * 2 + 1] != 0
    }
}

impl core::ops::BitAnd for PolyMask {
    type Output = PolyMask;
    #[inline(always)]
    fn bitand(self, rhs: PolyMask) -> PolyMask {
        PolyMask(self.0 & rhs.0)
    }
}

impl core::ops::BitOr for PolyMask {
    type Output = PolyMask;
    #[inline(always)]
    fn bitor(self, rhs: PolyMask) -> PolyMask {
        PolyMask(self.0 | rhs.0)
    }
}

impl core::ops::BitXor for PolyMask {
    type Output = PolyMask;
    #[inline(always)]
    fn bitxor(self, rhs: PolyMask) -> PolyMask {
        PolyMask(self.0 ^ rhs.0)
    }
}

impl core::ops::Not for PolyMask {
    type Output = PolyMask;
    #[inline(always)]
    fn not(self) -> PolyMask {
        self ^ PolyMask::all_on()
    }
}

impl core::ops::BitAndAssign for PolyMask {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: PolyMask) {
        *self = *self & rhs;
    }
}

impl core::ops::BitOrAssign for PolyMask {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: PolyMask) {
        *self = *self | rhs;
    }
}

// ---------------------------------------------------------------------------
// PolyU32
// ---------------------------------------------------------------------------

impl PolyU32 {
    pub const ZERO: PolyU32 = PolyU32([0; LANES]);

    #[inline(always)]
    pub fn splat(value: u32) -> PolyU32 {
        PolyU32([value; LANES])
    }

    #[inline(always)]
    pub fn from_lanes(lanes: [u32; LANES]) -> PolyU32 {
        PolyU32(lanes)
    }

    #[inline(always)]
    pub fn lane(self, i: usize) -> u32 {
        self.0[i]
    }

    #[inline(always)]
    pub fn set_lane(&mut self, i: usize, value: u32) {
        self.0[i] = value;
    }

    /// Per-lane signed-int → float conversion.
    #[inline(always)]
    pub fn to_f32_signed(self) -> PolyF32 {
        let ints: wide::i32x4 = cast(self);
        PolyF32(ints.round_float())
    }

    /// Per-lane `2^n` via exponent-field bit trick (Vital's `pow2ToFloat`).
    #[inline(always)]
    pub fn pow2_to_f32(self) -> PolyF32 {
        let bits = PolyU32([
            self.0[0].wrapping_add(127) << 23,
            self.0[1].wrapping_add(127) << 23,
            self.0[2].wrapping_add(127) << 23,
            self.0[3].wrapping_add(127) << 23,
        ]);
        PolyF32::from_u32_bits(bits)
    }

    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn shr(self, shift: u32) -> PolyU32 {
        PolyU32([self.0[0] >> shift, self.0[1] >> shift, self.0[2] >> shift, self.0[3] >> shift])
    }

    #[allow(clippy::should_implement_trait)]
    #[inline(always)]
    pub fn shl(self, shift: u32) -> PolyU32 {
        PolyU32([self.0[0] << shift, self.0[1] << shift, self.0[2] << shift, self.0[3] << shift])
    }

    #[inline(always)]
    pub fn eq(self, other: PolyU32) -> PolyMask {
        let m = |a: u32, b: u32| if a == b { u32::MAX } else { 0 };
        PolyMask::from_u32(PolyU32([
            m(self.0[0], other.0[0]),
            m(self.0[1], other.0[1]),
            m(self.0[2], other.0[2]),
            m(self.0[3], other.0[3]),
        ]))
    }

    #[inline(always)]
    pub fn min(self, other: PolyU32) -> PolyU32 {
        PolyU32([
            self.0[0].min(other.0[0]),
            self.0[1].min(other.0[1]),
            self.0[2].min(other.0[2]),
            self.0[3].min(other.0[3]),
        ])
    }

    #[inline(always)]
    pub fn max(self, other: PolyU32) -> PolyU32 {
        PolyU32([
            self.0[0].max(other.0[0]),
            self.0[1].max(other.0[1]),
            self.0[2].max(other.0[2]),
            self.0[3].max(other.0[3]),
        ])
    }

    #[inline(always)]
    pub fn swap_stereo(self) -> PolyU32 {
        PolyU32([self.0[1], self.0[0], self.0[3], self.0[2]])
    }

    #[inline(always)]
    pub fn swap_voices(self) -> PolyU32 {
        PolyU32([self.0[2], self.0[3], self.0[0], self.0[1]])
    }
}

impl From<u32> for PolyU32 {
    #[inline(always)]
    fn from(value: u32) -> Self {
        PolyU32::splat(value)
    }
}

macro_rules! poly_u32_lanewise {
    ($trait:ident, $method:ident, $op:ident) => {
        impl core::ops::$trait for PolyU32 {
            type Output = PolyU32;
            #[inline(always)]
            fn $method(self, rhs: PolyU32) -> PolyU32 {
                PolyU32([
                    self.0[0].$op(rhs.0[0]),
                    self.0[1].$op(rhs.0[1]),
                    self.0[2].$op(rhs.0[2]),
                    self.0[3].$op(rhs.0[3]),
                ])
            }
        }
    };
}

poly_u32_lanewise!(Add, add, wrapping_add);
poly_u32_lanewise!(Sub, sub, wrapping_sub);
poly_u32_lanewise!(Mul, mul, wrapping_mul);
poly_u32_lanewise!(BitAnd, bitand, bitand);
poly_u32_lanewise!(BitOr, bitor, bitor);
poly_u32_lanewise!(BitXor, bitxor, bitxor);

impl core::ops::Not for PolyU32 {
    type Output = PolyU32;
    #[inline(always)]
    fn not(self) -> PolyU32 {
        PolyU32([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
}

impl core::ops::AddAssign for PolyU32 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: PolyU32) {
        *self = *self + rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_layout_and_swaps() {
        let v = PolyF32::from_lanes([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.swap_stereo().to_lanes(), [2.0, 1.0, 4.0, 3.0]);
        assert_eq!(v.swap_voices().to_lanes(), [3.0, 4.0, 1.0, 2.0]);
        assert_eq!(v.swap_inner().to_lanes(), [1.0, 3.0, 2.0, 4.0]);
        assert_eq!(v.reverse().to_lanes(), [4.0, 3.0, 2.0, 1.0]);
        assert_eq!(PolyF32::stereo(0.5, -0.5).to_lanes(), [0.5, -0.5, 0.5, -0.5]);
    }

    #[test]
    fn mask_select() {
        let a = PolyF32::from_lanes([1.0, 2.0, 3.0, 4.0]);
        let b = PolyF32::from_lanes([-1.0, -2.0, -3.0, -4.0]);
        let mask = a.gt(PolyF32::splat(2.5));
        assert_eq!(mask.select(a, b).to_lanes(), [-1.0, -2.0, 3.0, 4.0]);
        assert!(mask.any());
        assert!(!mask.all());
        assert!(!mask.voice_any(0));
        assert!(mask.voice_any(1));
    }

    #[test]
    fn mul_add_matches_vital_semantics() {
        // Vital: mulAdd(a, b, c) == a + b * c
        let a = PolyF32::splat(1.0);
        let b = PolyF32::splat(2.0);
        let c = PolyF32::splat(3.0);
        assert_eq!(a.mul_add(b, c).to_lanes(), [7.0; 4]);
        assert_eq!(a.mul_sub(b, c).to_lanes(), [-5.0; 4]);
    }

    #[test]
    fn float_int_conversions() {
        let v = PolyF32::from_lanes([1.4, -1.4, 2.5, -2.5]);
        assert_eq!(v.floor().to_lanes(), [1.0, -2.0, 2.0, -3.0]);
        assert_eq!(v.ceil().to_lanes(), [2.0, -1.0, 3.0, -2.0]);
        // Vital rounds via floor(x + 0.5).
        assert_eq!(v.round().to_lanes(), [1.0, -1.0, 3.0, -2.0]);
        let fract = PolyF32::from_lanes([1.25, -0.25, 0.75, 2.0]).fract();
        assert_eq!(fract.to_lanes(), [0.25, 0.75, 0.75, 0.0]);
    }

    #[test]
    fn pow2_bit_trick() {
        let exps = PolyU32([0, 1, 3u32, (-2i32) as u32]);
        assert_eq!(exps.pow2_to_f32().to_lanes(), [1.0, 2.0, 8.0, 0.25]);
    }

    #[test]
    fn u32_wrapping_math() {
        let phase = PolyU32::splat(u32::MAX);
        let bumped = phase + PolyU32::splat(2);
        assert_eq!(bumped.0, [1; 4]);
    }
}
