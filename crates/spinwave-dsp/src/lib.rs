//! DSP building blocks for the spinwave synth engine.
//!
//! Everything here is framework-free: plain structs with explicit state,
//! processing `PolyF32` buffers (two stereo voices per vector). Voice
//! management, modulation routing and graph wiring live in `spinwave-engine`.

pub mod effects;
pub mod filters;
pub mod memory;
pub mod modulators;
pub mod oscillator;
pub mod utilities;
pub mod wavetable;

pub use spinwave_poly::{constants, math, utils, Matrix, PolyF32, PolyMask, PolyU32, LANES};

/// Forces every lazily built shared table in this crate (filter coefficient
/// lookups, the SVF lookup, the random-amplitude morph table, the shared
/// wave-frame FFT plans) so their one-time allocation happens here rather
/// than on the first `process` call. The engine calls this at construction;
/// it is idempotent and cheap after the first call.
pub fn warm_up() {
    let _ = filters::filter_state::coefficient_lookup();
    let _ = filters::filter_state::svf_coefficient_lookup();
    let _ = oscillator::synth_oscillator::random_amplitude_table();
    let _ = wavetable::wave_frame::wave_fft();
}

#[cfg(test)]
mod tests {
    #[test]
    fn warm_up_is_idempotent() {
        super::warm_up();
        super::warm_up();
        let table = super::oscillator::synth_oscillator::random_amplitude_table();
        assert!(!table.is_empty());
    }
}
