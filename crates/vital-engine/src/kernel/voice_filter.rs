//! One voice filter slot (rework of Vital's `FilterModule`): eight
//! switchable filter models sharing one [`FilterState`], with hard reset
//! on model change and a click-free dry/wet mix ramp.

use vital_dsp::effects::phaser_filter::{PhaserFilter, PhaserFilterParams};
use vital_dsp::filters::{
    CombFilter, DigitalSvf, DiodeFilter, DirtyFilter, FilterState, FormantFilter, LadderFilter,
    SallenKeyFilter,
};
use vital_poly::constants::MAX_BUFFER_SIZE;
use vital_poly::utils::interpolate;
use vital_poly::{PolyF32, PolyMask};

/// Reference `CombModule::kMaxFeedbackSamples`.
const MAX_COMB_FEEDBACK_SAMPLES: usize = 25000;
const MAX_OVERSAMPLED_BLOCK: usize = MAX_BUFFER_SIZE * 8;

/// Filter model, same order as the reference's `constants::FilterModel`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum FilterModel {
    #[default]
    Analog = 0,
    Dirty = 1,
    Ladder = 2,
    Digital = 3,
    Diode = 4,
    Formant = 5,
    Comb = 6,
    Phase = 7,
}

impl FilterModel {
    pub fn from_index(index: i32) -> FilterModel {
        match index {
            1 => FilterModel::Dirty,
            2 => FilterModel::Ladder,
            3 => FilterModel::Digital,
            4 => FilterModel::Diode,
            5 => FilterModel::Formant,
            6 => FilterModel::Comb,
            7 => FilterModel::Phase,
            _ => FilterModel::Analog,
        }
    }
}

/// Per-block settings for one voice filter slot.
#[derive(Clone, Copy, Debug)]
pub struct VoiceFilterParams {
    pub on: bool,
    pub model: FilterModel,
    /// Shared filter settings; `midi_cutoff` must already include keytrack
    /// and modulation.
    pub state: FilterState,
    /// Dry/wet in `[0, 1]`.
    pub mix: PolyF32,
}

impl Default for VoiceFilterParams {
    fn default() -> Self {
        VoiceFilterParams {
            on: false,
            model: FilterModel::Analog,
            state: FilterState::default(),
            mix: PolyF32::ONE,
        }
    }
}

pub struct VoiceFilter {
    sample_rate: f32,
    last_model: Option<FilterModel>,
    mix: PolyF32,

    sallen_key: SallenKeyFilter,
    dirty: DirtyFilter,
    ladder: LadderFilter,
    svf: DigitalSvf,
    diode: DiodeFilter,
    formant: FormantFilter,
    comb: CombFilter,
    phaser: PhaserFilter,

    cutoff_scratch: Vec<PolyF32>,
}

impl VoiceFilter {
    pub fn new(sample_rate: f32) -> VoiceFilter {
        VoiceFilter {
            sample_rate,
            last_model: None,
            mix: PolyF32::ZERO,
            sallen_key: SallenKeyFilter::new(),
            dirty: DirtyFilter::new(),
            ladder: LadderFilter::new(),
            svf: DigitalSvf::new(),
            diode: DiodeFilter::new(),
            formant: FormantFilter::new(),
            comb: CombFilter::new(MAX_COMB_FEEDBACK_SAMPLES),
            phaser: PhaserFilter::new(false, sample_rate),
            cutoff_scratch: vec![PolyF32::ZERO; MAX_OVERSAMPLED_BLOCK],
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.phaser = PhaserFilter::new(false, sample_rate);
    }

    pub fn hard_reset(&mut self) {
        self.sallen_key.hard_reset();
        self.dirty.hard_reset();
        self.ladder.hard_reset();
        self.svf.hard_reset();
        self.diode.hard_reset();
        self.formant.hard_reset();
        self.comb.hard_reset();
        self.phaser.hard_reset();
        self.last_model = None;
    }

    fn switch_model(&mut self, model: FilterModel) {
        if self.last_model == Some(model) {
            return;
        }
        match model {
            FilterModel::Analog => self.sallen_key.hard_reset(),
            FilterModel::Dirty => self.dirty.hard_reset(),
            FilterModel::Ladder => self.ladder.hard_reset(),
            FilterModel::Digital => self.svf.hard_reset(),
            FilterModel::Diode => self.diode.hard_reset(),
            FilterModel::Formant => self.formant.hard_reset(),
            FilterModel::Comb => self.comb.hard_reset(),
            FilterModel::Phase => self.phaser.hard_reset(),
        }
        self.last_model = Some(model);
    }

    /// Renders one block. `reset_mask` marks lanes whose voice restarted at
    /// the beginning of this block (the kernel splits blocks at trigger
    /// offsets so block-start resets are sample-accurate).
    pub fn process(
        &mut self,
        params: &VoiceFilterParams,
        audio_in: &[PolyF32],
        audio_out: &mut [PolyF32],
        reset_mask: PolyMask,
    ) {
        let num_samples = audio_in.len();
        debug_assert!(num_samples <= self.cutoff_scratch.len());

        if !params.on {
            audio_out[..num_samples].fill(PolyF32::ZERO);
            return;
        }

        self.switch_model(params.model);
        let state = &params.state;
        let sample_rate = self.sample_rate;

        match params.model {
            FilterModel::Analog => {
                self.sallen_key.setup(state, sample_rate);
                if reset_mask.any() {
                    self.sallen_key.reset(reset_mask);
                }
                self.sallen_key.process(audio_in, audio_out);
            }
            FilterModel::Dirty => {
                self.dirty.setup(state, sample_rate);
                if reset_mask.any() {
                    self.dirty.reset(reset_mask);
                }
                self.dirty.process(audio_in, audio_out);
            }
            FilterModel::Ladder => {
                self.ladder.setup(state, sample_rate);
                if reset_mask.any() {
                    self.ladder.reset(reset_mask);
                }
                self.ladder.process(audio_in, audio_out);
            }
            FilterModel::Digital => {
                self.svf.setup(state, sample_rate);
                if reset_mask.any() {
                    self.svf.reset(reset_mask);
                }
                self.svf.process(audio_in, audio_out);
            }
            FilterModel::Diode => {
                self.diode.setup(state, sample_rate);
                if reset_mask.any() {
                    self.diode.reset(reset_mask);
                }
                self.diode.process(audio_in, audio_out);
            }
            FilterModel::Formant => {
                self.formant.setup(state, sample_rate);
                if reset_mask.any() {
                    self.formant.reset(reset_mask);
                }
                self.formant.process(audio_in, audio_out);
            }
            FilterModel::Comb => {
                self.comb.setup(state, sample_rate);
                if reset_mask.any() {
                    self.comb.reset(reset_mask);
                }
                self.comb.process(audio_in, audio_out);
            }
            FilterModel::Phase => {
                // The voice phaser reads its sweep from a per-sample cutoff
                // buffer; constant within the block, transpose folded in.
                let cutoff = state.midi_cutoff + state.transpose;
                self.cutoff_scratch[..num_samples].fill(cutoff);
                let phaser_params = PhaserFilterParams {
                    resonance_percent: state.resonance_percent,
                    drive: state.drive,
                    pass_blend: state.pass_blend,
                    invert: false,
                };
                if reset_mask.any() {
                    self.phaser.reset(reset_mask);
                }
                self.phaser.process(
                    &phaser_params,
                    &self.cutoff_scratch[..num_samples],
                    audio_in,
                    audio_out,
                );
            }
        }

        // Dry/wet ramp, exactly like FilterModule::process.
        let target_mix = params.mix.clamp(0.0, 1.0);
        let mut current_mix = reset_mask.select(target_mix, self.mix);
        self.mix = target_mix;
        let delta_mix = (target_mix - current_mix) * (1.0 / num_samples as f32);
        for (out, &dry) in audio_out[..num_samples].iter_mut().zip(audio_in) {
            current_mix += delta_mix;
            *out = interpolate(dry, *out, current_mix);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vital_dsp::filters::FilterStyle;

    fn sine_block(frequency: f32, sample_rate: f32, num_samples: usize) -> Vec<PolyF32> {
        (0..num_samples)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate;
                PolyF32::splat(phase.sin())
            })
            .collect()
    }

    fn rms(buffer: &[PolyF32]) -> f32 {
        let sum: f32 = buffer.iter().map(|v| v.lane(0) * v.lane(0)).sum();
        (sum / buffer.len() as f32).sqrt()
    }

    fn low_pass_params(model: FilterModel, cutoff_midi: f32) -> VoiceFilterParams {
        let mut state = FilterState {
            midi_cutoff: PolyF32::splat(cutoff_midi),
            style: FilterStyle::TwelveDb,
            ..FilterState::default()
        };
        state.set_pass_blend(PolyF32::ZERO); // full low-pass
        state.set_drive_db(PolyF32::ZERO);
        VoiceFilterParams { on: true, model, state, mix: PolyF32::ONE }
    }

    #[test]
    fn svf_low_pass_attenuates_highs() {
        let sample_rate = 44100.0;
        let mut filter = VoiceFilter::new(sample_rate);
        // Cutoff around 500 Hz (midi ~71): pass 100 Hz, cut 8 kHz.
        let params = low_pass_params(FilterModel::Digital, 71.0);

        let process_chunked = |filter: &mut VoiceFilter, input: &[PolyF32]| {
            let mut output = vec![PolyF32::ZERO; input.len()];
            let mut first = true;
            for (in_chunk, out_chunk) in input.chunks(128).zip(output.chunks_mut(128)) {
                let reset = if first { PolyMask::all_on() } else { PolyMask::NONE };
                filter.process(&params, in_chunk, out_chunk, reset);
                first = false;
            }
            output
        };

        let low_in = sine_block(100.0, sample_rate, 4096);
        let low_out = process_chunked(&mut filter, &low_in);

        let mut filter = VoiceFilter::new(sample_rate);
        let high_in = sine_block(8000.0, sample_rate, 4096);
        let high_out = process_chunked(&mut filter, &high_in);

        // Skip the transient at the start.
        let low_rms = rms(&low_out[1024..]);
        let high_rms = rms(&high_out[1024..]);
        assert!(
            low_rms > high_rms * 4.0,
            "low {low_rms} vs high {high_rms}"
        );
    }

    #[test]
    fn mix_zero_is_transparent() {
        let sample_rate = 44100.0;
        let mut filter = VoiceFilter::new(sample_rate);
        let mut params = low_pass_params(FilterModel::Digital, 40.0);
        params.mix = PolyF32::ZERO;

        let input = sine_block(2000.0, sample_rate, 512);
        let mut output = vec![PolyF32::ZERO; 512];
        filter.process(&params, &input, &mut output, PolyMask::all_on());
        // After the (instant, reset) ramp the dry signal passes through.
        for i in 8..512 {
            assert!((output[i].lane(0) - input[i].lane(0)).abs() < 1e-4);
        }
    }

    #[test]
    fn off_outputs_silence() {
        let mut filter = VoiceFilter::new(44100.0);
        let params = VoiceFilterParams::default();
        let input = sine_block(440.0, 44100.0, 64);
        let mut output = vec![PolyF32::splat(1.0); 64];
        filter.process(&params, &input, &mut output, PolyMask::NONE);
        assert!(output.iter().all(|v| v.lane(0) == 0.0));
    }

    #[test]
    fn model_switching_does_not_panic_and_stays_finite() {
        let sample_rate = 44100.0;
        let mut filter = VoiceFilter::new(sample_rate);
        let input = sine_block(440.0, sample_rate, 128);
        let mut output = vec![PolyF32::ZERO; 128];

        for model in [
            FilterModel::Analog,
            FilterModel::Dirty,
            FilterModel::Ladder,
            FilterModel::Digital,
            FilterModel::Diode,
            FilterModel::Formant,
            FilterModel::Comb,
            FilterModel::Phase,
        ] {
            let params = low_pass_params(model, 60.0);
            filter.process(&params, &input, &mut output, PolyMask::all_on());
            filter.process(&params, &input, &mut output, PolyMask::NONE);
            for value in &output {
                assert!(value.is_finite(), "{model:?} produced non-finite output");
            }
        }
    }
}
