//! Lane-mask helpers matching Vital's `constants::k*Mask` values.

use spinwave_poly::{PolyMask, PolyU32};

/// Mask of the left channels of both voices: lanes 0 and 2.
#[inline(always)]
pub fn left_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, 0, u32::MAX, 0]))
}

/// Mask of the right channels of both voices: lanes 1 and 3.
#[inline(always)]
pub fn right_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([0, u32::MAX, 0, u32::MAX]))
}

/// Mask of the first voice: lanes 0 and 1.
#[inline(always)]
pub fn first_voice_mask() -> PolyMask {
    PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, u32::MAX, 0, 0]))
}

/// Per-lane unsigned `a > b` (Vital's `poly_int::greaterThan`).
#[inline(always)]
pub fn u32_gt(a: PolyU32, b: PolyU32) -> PolyMask {
    let m = |x: u32, y: u32| if x > y { u32::MAX } else { 0 };
    PolyMask::from_u32(PolyU32::from_lanes([
        m(a.lane(0), b.lane(0)),
        m(a.lane(1), b.lane(1)),
        m(a.lane(2), b.lane(2)),
        m(a.lane(3), b.lane(3)),
    ]))
}
