//! Formant filter bank (port of `formant_filter.{h,cpp}` and
//! `formant_manager.{h,cpp}`).
//!
//! Four parallel 12 dB SVFs tuned to vowel formants, bilinearly interpolated
//! over an X/Y vowel pad and summed. The reference's `vocal_tract.{h,cpp}` is
//! an empty stub (its `process` writes nothing), so it is ported as the
//! placeholder [`VocalTract`].

use spinwave_poly::constants::MAX_BUFFER_SIZE;
use spinwave_poly::utils::interpolate;
use spinwave_poly::{PolyF32, PolyMask};

use super::digital_svf::DigitalSvf;
use super::filter_state::{FilterState, FilterStyle};

pub const NUM_FORMANTS: usize = 4;
pub const CENTER_MIDI: f32 = 80.0;

/// Formant resonance bounds applied by the C++ `FormantManager`.
pub const MIN_FORMANT_RESONANCE: f32 = 4.0;
pub const MAX_FORMANT_RESONANCE: f32 = 30.0;

/// Gain (dB), resonance percent and MIDI cutoff of one formant peak.
#[derive(Clone, Copy, Debug)]
struct FormantValues {
    gain: f32,
    resonance: f32,
    midi_cutoff: f32,
}

const fn formant(gain: f32, resonance: f32, midi_cutoff: f32) -> FormantValues {
    FormantValues { gain, resonance, midi_cutoff }
}

#[allow(clippy::excessive_precision)]
const FORMANT_A: [FormantValues; NUM_FORMANTS] = [
    formant(-2.0, 0.66, 75.7552343327),
    formant(-8.0, 0.75, 84.5454706023),
    formant(-9.0, 1.0, 100.08500317),
    formant(-10.0, 1.0, 101.645729657),
];

#[allow(clippy::excessive_precision)]
const FORMANT_E: [FormantValues; NUM_FORMANTS] = [
    formant(0.0, 0.66, 67.349957715),
    formant(-14.0, 0.75, 92.39951181),
    formant(-4.0, 1.0, 99.7552343327),
    formant(-14.0, 1.0, 103.349957715),
];

#[allow(clippy::excessive_precision)]
const FORMANT_I: [FormantValues; NUM_FORMANTS] = [
    formant(0.0, 0.8, 61.7825925179),
    formant(-15.0, 0.75, 94.049554095),
    formant(-17.0, 1.0, 101.03821678),
    formant(-20.0, 1.0, 103.618371471),
];

#[allow(clippy::excessive_precision)]
const FORMANT_O: [FormantValues; NUM_FORMANTS] = [
    formant(-2.0, 0.7, 67.349957715),
    formant(-6.0, 0.75, 79.349957715),
    formant(-14.0, 1.0, 99.7552343327),
    formant(-14.0, 1.0, 101.03821678),
];

#[allow(clippy::excessive_precision)]
const FORMANT_U: [FormantValues; NUM_FORMANTS] = [
    formant(0.0, 0.7, 65.0382167797),
    formant(-20.0, 0.75, 74.3695077237),
    formant(-17.0, 1.0, 100.408607741),
    formant(-14.0, 1.0, 101.645729657),
];

// Pad corners in order: bottom-left, bottom-right, top-left, top-right.
// The style order matches the C++ `formant_styles` table (style 0 is the
// A/I/U/O layout, style 1 the A/O/I/E layout).
const FORMANT_STYLES: [[&[FormantValues; NUM_FORMANTS]; 4]; 2] = [
    [&FORMANT_A, &FORMANT_I, &FORMANT_U, &FORMANT_O],
    [&FORMANT_A, &FORMANT_O, &FORMANT_I, &FORMANT_E],
];

const BOTTOM_LEFT: usize = 0;
const BOTTOM_RIGHT: usize = 1;
const TOP_LEFT: usize = 2;
const TOP_RIGHT: usize = 3;

pub const NUM_FORMANT_STYLES: usize = 2;

#[inline]
fn bilinear_interpolate(
    top_left: PolyF32,
    top_right: PolyF32,
    bot_left: PolyF32,
    bot_right: PolyF32,
    x: PolyF32,
    y: PolyF32,
) -> PolyF32 {
    let top = interpolate(top_left, top_right, x);
    let bot = interpolate(bot_left, bot_right, x);
    interpolate(bot, top, y)
}

fn interpolate_formants(
    top_left: &FormantValues,
    top_right: &FormantValues,
    bot_left: &FormantValues,
    bot_right: &FormantValues,
    formant_x: PolyF32,
    formant_y: PolyF32,
) -> FilterState {
    FilterState {
        midi_cutoff: bilinear_interpolate(
            PolyF32::splat(top_left.midi_cutoff),
            PolyF32::splat(top_right.midi_cutoff),
            PolyF32::splat(bot_left.midi_cutoff),
            PolyF32::splat(bot_right.midi_cutoff),
            formant_x,
            formant_y,
        ),
        resonance_percent: bilinear_interpolate(
            PolyF32::splat(top_left.resonance),
            PolyF32::splat(top_right.resonance),
            PolyF32::splat(bot_left.resonance),
            PolyF32::splat(bot_right.resonance),
            formant_x,
            formant_y,
        ),
        gain: bilinear_interpolate(
            PolyF32::splat(top_left.gain),
            PolyF32::splat(top_right.gain),
            PolyF32::splat(bot_left.gain),
            PolyF32::splat(bot_right.gain),
            formant_x,
            formant_y,
        ),
        ..FilterState::default()
    }
}

/// Bank of formant SVFs summed to one output.
#[derive(Clone, Debug)]
pub struct FormantFilter {
    formants: [DigitalSvf; NUM_FORMANTS],
}

impl Default for FormantFilter {
    fn default() -> FormantFilter {
        FormantFilter::new()
    }
}

impl FormantFilter {
    pub fn new() -> FormantFilter {
        let mut formants: [DigitalSvf; NUM_FORMANTS] = Default::default();
        for formant in &mut formants {
            formant.set_resonance_bounds(MIN_FORMANT_RESONANCE, MAX_FORMANT_RESONANCE);
        }
        FormantFilter { formants }
    }

    /// Per-block parameter update.
    ///
    /// This follows the processor graph the reference actually runs
    /// (`FormantFilter::init` under a `FormantModule`), not the dead
    /// `FormantFilter::setupFilter` override — `SynthFilter::createFilter` is
    /// never called in Vital, so `setupFilter` is unreachable and its
    /// parameter mapping (`pass_blend` toward the centre, `transpose`,
    /// `resonance_percent`) is not what the plugin does. The live graph is:
    ///
    /// ```text
    /// formant_midi        = BilinearInterpolate(vowel table, formant_x, formant_y)
    /// formant_midi_spread = Interpolate(formant_midi -> kCenterMidi, formant_spread)
    /// formant_midi_adjust = formant_transpose + formant_midi_spread
    /// formant_q_adjust    = formant_resonance * BilinearInterpolate(vowel Q)
    /// svf: style = k12Db, pass_blend = 1, gain = BilinearInterpolate(vowel gain)
    /// ```
    ///
    /// So the formant model uses `{prefix}_formant_*` only: the filter's own
    /// `blend`, `blend_transpose` and `resonance` never reach it.
    #[allow(clippy::needless_range_loop)]
    pub fn setup(&mut self, filter_state: &FilterState, sample_rate: f32) {
        let style = (filter_state.style.index() as usize).min(NUM_FORMANT_STYLES - 1);
        let tables = &FORMANT_STYLES[style];

        for i in 0..NUM_FORMANTS {
            let mut formant_setting = interpolate_formants(
                &tables[TOP_LEFT][i],
                &tables[TOP_RIGHT][i],
                &tables[BOTTOM_LEFT][i],
                &tables[BOTTOM_RIGHT][i],
                filter_state.interpolate_x,
                filter_state.interpolate_y,
            );

            formant_setting.midi_cutoff = interpolate(
                formant_setting.midi_cutoff,
                PolyF32::splat(CENTER_MIDI),
                filter_state.formant_spread,
            );
            formant_setting.midi_cutoff += filter_state.formant_transpose;
            formant_setting.resonance_percent *= filter_state.formant_resonance;
            formant_setting.style = FilterStyle::TwelveDb;
            formant_setting.pass_blend = PolyF32::ONE;

            self.formants[i].setup(&formant_setting, sample_rate);
        }
    }

    pub fn reset(&mut self, reset_mask: PolyMask) {
        for formant in &mut self.formants {
            formant.reset(reset_mask);
        }
    }

    pub fn hard_reset(&mut self) {
        for formant in &mut self.formants {
            formant.hard_reset();
        }
    }

    /// Processes the block through all formants and sums their outputs
    /// (C++ `FormantManager`'s `VariableAdd`).
    /// Any block length is accepted: the scratch buffer is a fixed
    /// [`MAX_BUFFER_SIZE`] array, so longer blocks (the bus chains run
    /// oversampled, so they exceed it) are processed in chunks. Formants
    /// carry their state across chunks, so the result is identical.
    pub fn process(&mut self, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        assert_eq!(audio_in.len(), audio_out.len());
        for (chunk_in, chunk_out) in audio_in
            .chunks(MAX_BUFFER_SIZE)
            .zip(audio_out.chunks_mut(MAX_BUFFER_SIZE))
        {
            self.process_chunk(chunk_in, chunk_out);
        }
    }

    fn process_chunk(&mut self, audio_in: &[PolyF32], audio_out: &mut [PolyF32]) {
        let num_samples = audio_in.len();
        debug_assert!(num_samples <= MAX_BUFFER_SIZE);

        audio_out[..num_samples].fill(PolyF32::ZERO);
        let mut scratch = [PolyF32::ZERO; MAX_BUFFER_SIZE];
        for formant in &mut self.formants {
            formant.process(audio_in, &mut scratch[..num_samples]);
            for (out, &value) in audio_out.iter_mut().zip(&scratch[..num_samples]) {
                *out += value;
            }
        }
    }

    /// Audio-rate variant of [`Self::process`]: `midi_offset` is a
    /// per-sample MIDI offset added to every formant's block cutoff (the
    /// per-sample deviation of the modulated transpose from the value given
    /// to [`Self::setup`]). Each formant SVF then consumes a per-sample
    /// cutoff buffer, like the reference's audio-rate `formant_midi`
    /// interpolation (`formant_filter.cpp`).
    pub fn process_modulated(
        &mut self,
        audio_in: &[PolyF32],
        midi_offset: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        assert_eq!(audio_in.len(), audio_out.len());
        assert_eq!(audio_in.len(), midi_offset.len());
        for ((chunk_in, chunk_mod), chunk_out) in audio_in
            .chunks(MAX_BUFFER_SIZE)
            .zip(midi_offset.chunks(MAX_BUFFER_SIZE))
            .zip(audio_out.chunks_mut(MAX_BUFFER_SIZE))
        {
            self.process_modulated_chunk(chunk_in, chunk_mod, chunk_out);
        }
    }

    fn process_modulated_chunk(
        &mut self,
        audio_in: &[PolyF32],
        midi_offset: &[PolyF32],
        audio_out: &mut [PolyF32],
    ) {
        let num_samples = audio_in.len();
        debug_assert!(num_samples <= MAX_BUFFER_SIZE);

        audio_out[..num_samples].fill(PolyF32::ZERO);
        let mut scratch = [PolyF32::ZERO; MAX_BUFFER_SIZE];
        let mut cutoff = [PolyF32::ZERO; MAX_BUFFER_SIZE];
        for formant in &mut self.formants {
            let base = formant.midi_cutoff();
            for (dest, &offset) in cutoff[..num_samples].iter_mut().zip(midi_offset) {
                *dest = base + offset;
            }
            formant.process_modulated(audio_in, &cutoff[..num_samples], &mut scratch[..num_samples]);
            for (out, &value) in audio_out.iter_mut().zip(&scratch[..num_samples]) {
                *out += value;
            }
        }
    }

    pub fn formant(&self, index: usize) -> &DigitalSvf {
        &self.formants[index]
    }

    pub fn formant_mut(&mut self, index: usize) -> &mut DigitalSvf {
        &mut self.formants[index]
    }

    pub fn num_formants(&self) -> usize {
        NUM_FORMANTS
    }
}

/// Placeholder for the vocal tract model: the reference implementation is an
/// empty `ProcessorRouter` whose `processWithInput` does nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct VocalTract;

impl VocalTract {
    pub fn new() -> VocalTract {
        VocalTract
    }

    pub fn reset(&mut self, _reset_mask: PolyMask) {}

    pub fn hard_reset(&mut self) {}

    /// No-op, like the reference (the output buffer is left untouched).
    pub fn process(&mut self, _audio_in: &[PolyF32], _audio_out: &mut [PolyF32]) {}
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 48000.0;
    const BLOCK: usize = 128;

    fn formant_state() -> FilterState {
        let mut state = FilterState::default();
        state.formant_resonance = PolyF32::splat(0.7);
        state.interpolate_x = PolyF32::ZERO;
        state.interpolate_y = PolyF32::ZERO;
        state.formant_spread = PolyF32::ZERO;
        state.formant_transpose = PolyF32::ZERO;
        state.style = FilterStyle::TwelveDb;
        state
    }

    fn run_sine(filter: &mut FormantFilter, state: &FilterState, freq: f32, blocks: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        let mut n = 0usize;
        let mut input = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::ZERO; BLOCK];
        for block in 0..blocks {
            for value in input.iter_mut() {
                let phase = 2.0 * core::f32::consts::PI * freq * n as f32 / SAMPLE_RATE;
                *value = PolyF32::splat(0.5 * phase.sin());
                n += 1;
            }
            filter.setup(state, SAMPLE_RATE);
            filter.process(&input, &mut output);
            if block >= blocks / 2 {
                for value in &output {
                    assert!(value.is_finite());
                    sum += value.lane(0) * value.lane(0);
                    count += 1;
                }
            }
        }
        (sum / count as f32).sqrt()
    }

    #[test]
    fn passes_formant_frequencies_attenuates_highs() {
        // Bottom-left corner of style 0 is the "A" vowel; its first formant
        // sits at MIDI ~75.8 (~370 Hz).
        let state = formant_state();
        let first_formant_freq =
            spinwave_poly::constants::MIDI_0_FREQUENCY * (FORMANT_A[0].midi_cutoff / 12.0).exp2();

        let mut filter = FormantFilter::new();
        let on_formant = run_sine(&mut filter, &state, first_formant_freq, 40);
        let mut filter = FormantFilter::new();
        let far_above = run_sine(&mut filter, &state, 12000.0, 40);
        assert!(
            on_formant > 3.0 * far_above,
            "formant rms {on_formant} vs high rms {far_above}"
        );
    }

    #[test]
    fn vowel_positions_differ() {
        let mut state_a = formant_state();
        state_a.interpolate_x = PolyF32::ZERO;
        state_a.interpolate_y = PolyF32::ZERO;
        let mut state_u = formant_state();
        state_u.interpolate_x = PolyF32::ZERO;
        state_u.interpolate_y = PolyF32::ONE;

        let probe = 800.0;
        let mut filter = FormantFilter::new();
        let rms_a = run_sine(&mut filter, &state_a, probe, 40);
        let mut filter = FormantFilter::new();
        let rms_u = run_sine(&mut filter, &state_u, probe, 40);
        assert!(rms_a.is_finite() && rms_u.is_finite());
        assert!(
            (rms_a - rms_u).abs() > 0.05 * rms_a.max(rms_u),
            "vowel corners produced identical response: {rms_a} vs {rms_u}"
        );
    }

    /// The formant model reads `{prefix}_formant_*` only. The filter's own
    /// `resonance`, `blend` and `blend_transpose` are wired to the other
    /// models and must not reach the vowel peaks — `blend_transpose` in
    /// particular defaults to 42 semitones, which used to shift every peak.
    #[test]
    fn ignores_the_non_formant_filter_controls() {
        let base = formant_state();
        let mut decoys = base;
        decoys.resonance_percent = PolyF32::splat(0.1);
        decoys.pass_blend = PolyF32::splat(2.0);
        decoys.transpose = PolyF32::splat(42.0);

        let mut plain = FormantFilter::new();
        let mut with_decoys = FormantFilter::new();
        plain.setup(&base, SAMPLE_RATE);
        with_decoys.setup(&decoys, SAMPLE_RATE);
        for i in 0..NUM_FORMANTS {
            assert_eq!(
                plain.formant(i).midi_cutoff().lane(0),
                with_decoys.formant(i).midi_cutoff().lane(0),
                "formant {i} cutoff moved with a non-formant control"
            );
            assert_eq!(
                plain.formant(i).resonance().lane(0),
                with_decoys.formant(i).resonance().lane(0),
                "formant {i} Q moved with a non-formant control"
            );
        }
    }

    /// ...and the dedicated controls do reach them, the way the reference's
    /// `formant_midi_spread` / `formant_midi_adjust` / `formant_q_adjust`
    /// chain wires them.
    #[test]
    fn formant_controls_move_the_peaks() {
        let base = formant_state();
        let mut reference = FormantFilter::new();
        reference.setup(&base, SAMPLE_RATE);
        let cutoffs: Vec<f32> =
            (0..NUM_FORMANTS).map(|i| reference.formant(i).midi_cutoff().lane(0)).collect();

        // `formant_transpose` shifts every peak by that many semitones.
        let mut transposed_state = base;
        transposed_state.formant_transpose = PolyF32::splat(12.0);
        let mut transposed = FormantFilter::new();
        transposed.setup(&transposed_state, SAMPLE_RATE);
        for (i, &original) in cutoffs.iter().enumerate() {
            let moved = transposed.formant(i).midi_cutoff().lane(0);
            assert!((moved - original - 12.0).abs() < 1e-3, "formant {i}: {moved}");
        }

        // Full `formant_spread` collapses every peak onto the centre note.
        let mut spread_state = base;
        spread_state.formant_spread = PolyF32::ONE;
        let mut spread = FormantFilter::new();
        spread.setup(&spread_state, SAMPLE_RATE);
        for i in 0..NUM_FORMANTS {
            let collapsed = spread.formant(i).midi_cutoff().lane(0);
            assert!((collapsed - CENTER_MIDI).abs() < 1e-3, "formant {i}: {collapsed}");
        }

        // `formant_resonance` scales the vowel's own Q, so a lower value
        // widens (lowers `resonance()` is 1/Q in the SVF, so it rises).
        let mut low_q_state = base;
        low_q_state.formant_resonance = PolyF32::splat(0.3);
        let mut low_q = FormantFilter::new();
        low_q.setup(&low_q_state, SAMPLE_RATE);
        assert!(
            low_q.formant(0).resonance().lane(0) > reference.formant(0).resonance().lane(0),
            "formant_resonance did not change the peak Q"
        );
    }

    #[test]
    fn reset_clears_all_formants() {
        let state = formant_state();
        let mut filter = FormantFilter::new();
        let _ = run_sine(&mut filter, &state, 440.0, 4);
        filter.setup(&state, SAMPLE_RATE);
        filter.reset(PolyMask::all_on());
        let silence = vec![PolyF32::ZERO; BLOCK];
        let mut output = vec![PolyF32::splat(1.0); BLOCK];
        filter.process(&silence, &mut output);
        for value in &output {
            assert_eq!(value.lane(0), 0.0);
        }
    }

    #[test]
    fn process_modulated_with_zero_offset_matches_process() {
        let state = formant_state();
        let mut plain = FormantFilter::new();
        let mut modulated = FormantFilter::new();
        let input: Vec<PolyF32> = (0..BLOCK)
            .map(|n| {
                let phase = 2.0 * core::f32::consts::PI * 440.0 * n as f32 / SAMPLE_RATE;
                PolyF32::splat(0.5 * phase.sin())
            })
            .collect();
        let zeros = vec![PolyF32::ZERO; BLOCK];
        let mut out_plain = vec![PolyF32::ZERO; BLOCK];
        let mut out_modulated = vec![PolyF32::ZERO; BLOCK];
        for _ in 0..4 {
            plain.setup(&state, SAMPLE_RATE);
            modulated.setup(&state, SAMPLE_RATE);
            plain.process(&input, &mut out_plain);
            modulated.process_modulated(&input, &zeros, &mut out_modulated);
        }
        for (a, b) in out_plain.iter().zip(&out_modulated) {
            assert!((a.lane(0) - b.lane(0)).abs() < 1e-5);
        }

        // A large per-sample offset moves the formants: the output changes.
        let shifted = vec![PolyF32::splat(12.0); BLOCK];
        let mut out_shifted = vec![PolyF32::ZERO; BLOCK];
        modulated.setup(&state, SAMPLE_RATE);
        modulated.process_modulated(&input, &shifted, &mut out_shifted);
        let diff: f32 = out_shifted
            .iter()
            .zip(&out_modulated)
            .map(|(a, b)| (a.lane(0) - b.lane(0)).abs())
            .sum();
        assert!(diff > 1e-3, "per-sample formant offset had no effect");
    }

    #[test]
    fn vocal_tract_stub_leaves_output_untouched() {
        let mut tract = VocalTract::new();
        let input = vec![PolyF32::ONE; 8];
        let mut output = vec![PolyF32::splat(0.25); 8];
        tract.process(&input, &mut output);
        assert_eq!(output[0].lane(0), 0.25);
    }
}
