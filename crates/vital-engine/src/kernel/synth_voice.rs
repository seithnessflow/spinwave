//! The complete synth voice kernel: 3 wavetable oscillators + sampler,
//! two switchable filters, 6 envelopes, 8 LFOs, 4 random LFOs and the
//! modulation matrix, statically wired (rework of `SynthVoiceHandler` +
//! `ProducersModule` + `FiltersModule`).

use std::sync::Arc;

use vital_dsp::modulators::{
    Envelope, EnvelopeParams, LineGenerator, RandomLfo, RandomLfoParams, SynthLfo, SynthLfoParams,
    TriggerRandom,
};
use vital_dsp::oscillator::{
    SampleSource, SampleSourceParams, SynthOscillator, SynthOscillatorParams,
};
use vital_dsp::wavetable::Wavetable;
use vital_poly::constants::{MAX_BUFFER_SIZE, VoiceEvent};
use vital_poly::{PolyF32, PolyMask};

use crate::allocator::VoiceKernel;
use crate::kernel::mod_matrix::{
    ModMatrix, ModOffsets, SourceValues, NUM_ENVELOPES, NUM_LFOS, NUM_OSCILLATORS,
    NUM_RANDOM_LFOS,
};
use crate::kernel::voice_filter::{VoiceFilter, VoiceFilterParams};
use crate::voice::{Trigger, VoiceControls};

const MAX_BLOCK: usize = MAX_BUFFER_SIZE * 8;

/// Where a producer's signal goes (reference `constants::SourceDestination`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum ProducerDestination {
    #[default]
    Filter1 = 0,
    Filter2 = 1,
    DualFilters = 2,
    Effects = 3,
    DirectOut = 4,
}

impl ProducerDestination {
    pub fn from_index(index: i32) -> ProducerDestination {
        match index {
            1 => ProducerDestination::Filter2,
            2 => ProducerDestination::DualFilters,
            3 => ProducerDestination::Effects,
            4 => ProducerDestination::DirectOut,
            _ => ProducerDestination::Filter1,
        }
    }

    fn feeds_filter_1(self) -> bool {
        matches!(self, ProducerDestination::Filter1 | ProducerDestination::DualFilters)
    }

    fn feeds_filter_2(self) -> bool {
        matches!(self, ProducerDestination::Filter2 | ProducerDestination::DualFilters)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterRouting {
    #[default]
    Parallel,
    SerialForward,
    SerialBackward,
}

#[derive(Clone, Debug)]
pub struct OscSection {
    pub on: bool,
    pub destination: ProducerDestination,
    pub params: SynthOscillatorParams,
}

impl Default for OscSection {
    fn default() -> Self {
        OscSection {
            on: false,
            destination: ProducerDestination::Filter1,
            params: SynthOscillatorParams::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SampleSection {
    pub on: bool,
    pub destination: ProducerDestination,
    pub params: SampleSourceParams,
}

#[derive(Clone, Debug)]
pub struct FilterSection {
    pub params: VoiceFilterParams,
    /// Keytrack amount in `[-1, 1]`: cutoff follows `(note - 60) * amount`.
    pub keytrack: f32,
}

impl Default for FilterSection {
    fn default() -> Self {
        FilterSection { params: VoiceFilterParams::default(), keytrack: 0.0 }
    }
}

#[derive(Clone)]
pub struct LfoSection {
    pub params: SynthLfoParams,
    pub shape: LineGenerator,
}

impl Default for LfoSection {
    fn default() -> Self {
        LfoSection { params: SynthLfoParams::default(), shape: LineGenerator::triangle() }
    }
}

/// All base (unmodulated) voice parameters, set from the parameter layer.
#[derive(Clone)]
pub struct KernelParams {
    pub oscillators: [OscSection; NUM_OSCILLATORS],
    pub sample: SampleSection,
    pub filters: [FilterSection; 2],
    pub filter_routing: FilterRouting,
    pub envelopes: [EnvelopeParams; NUM_ENVELOPES],
    pub lfos: [LfoSection; NUM_LFOS],
    pub random_lfos: [RandomLfoParams; NUM_RANDOM_LFOS],
    /// How much velocity scales the voice amplitude, `[0, 1]`.
    pub velocity_track: f32,
    /// Pitch wheel range in semitones.
    pub pitch_bend_range: f32,
    pub macros: [f32; 4],
}

impl Default for KernelParams {
    fn default() -> Self {
        let mut oscillators: [OscSection; NUM_OSCILLATORS] = Default::default();
        oscillators[0].on = true;
        KernelParams {
            oscillators,
            sample: SampleSection::default(),
            filters: Default::default(),
            filter_routing: FilterRouting::Parallel,
            envelopes: Default::default(),
            lfos: Default::default(),
            random_lfos: Default::default(),
            velocity_track: 0.6,
            pitch_bend_range: 2.0,
            macros: [0.0; 4],
        }
    }
}

/// The per-pair voice kernel. Envelope 0 is the amplitude envelope.
pub struct SynthVoiceKernel {
    sample_rate: u32,
    pub params: KernelParams,
    pub matrix: ModMatrix,
    wavetables: [Arc<Wavetable>; NUM_OSCILLATORS],

    oscillators: [SynthOscillator; NUM_OSCILLATORS],
    sampler: SampleSource,
    filters: [VoiceFilter; 2],
    envelopes: [Envelope; NUM_ENVELOPES],
    lfos: [SynthLfo; NUM_LFOS],
    random_lfos: [RandomLfo; NUM_RANDOM_LFOS],
    trigger_random: TriggerRandom,

    offsets: ModOffsets,
    sources: SourceValues,

    // Scratch buffers (no allocation in process).
    raw: [Vec<PolyF32>; NUM_OSCILLATORS],
    leveled: Vec<PolyF32>,
    filter1_bus: Vec<PolyF32>,
    filter2_bus: Vec<PolyF32>,
    effects_bus: Vec<PolyF32>,
    filter1_out: Vec<PolyF32>,
    filter2_out: Vec<PolyF32>,
    serial_bus: Vec<PolyF32>,
    amp_env: Vec<PolyF32>,
    output: Vec<PolyF32>,
}

impl SynthVoiceKernel {
    pub fn new(sample_rate: u32) -> SynthVoiceKernel {
        let sr = sample_rate as f32;
        SynthVoiceKernel {
            sample_rate,
            params: KernelParams::default(),
            matrix: ModMatrix::default(),
            wavetables: core::array::from_fn(|_| Arc::new(default_wavetable())),
            oscillators: core::array::from_fn(|_| SynthOscillator::new()),
            sampler: SampleSource::new(),
            filters: core::array::from_fn(|_| VoiceFilter::new(sr)),
            envelopes: core::array::from_fn(|_| Envelope::new(sr)),
            lfos: core::array::from_fn(|_| SynthLfo::new(sr)),
            random_lfos: core::array::from_fn(|_| RandomLfo::new(sr)),
            trigger_random: TriggerRandom::new(),
            offsets: ModOffsets::default(),
            sources: SourceValues::default(),
            raw: core::array::from_fn(|_| vec![PolyF32::ZERO; MAX_BLOCK]),
            leveled: vec![PolyF32::ZERO; MAX_BLOCK],
            filter1_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            filter2_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            effects_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            filter1_out: vec![PolyF32::ZERO; MAX_BLOCK],
            filter2_out: vec![PolyF32::ZERO; MAX_BLOCK],
            serial_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            amp_env: vec![PolyF32::ZERO; MAX_BLOCK],
            output: vec![PolyF32::ZERO; MAX_BLOCK],
        }
    }

    pub fn set_wavetable(&mut self, index: usize, wavetable: Arc<Wavetable>) {
        self.wavetables[index] = wavetable;
    }

    pub fn sampler_mut(&mut self) -> &mut SampleSource {
        &mut self.sampler
    }

    fn dispatch_triggers(&mut self, controls: &VoiceControls) {
        let retrigger = &controls.retrigger;
        if !retrigger.mask.any() {
            return;
        }

        let event_value = retrigger.value;
        let offset = first_offset(retrigger);
        let on_mask = retrigger.mask
            & event_value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));

        for envelope in &mut self.envelopes {
            envelope.trigger(retrigger.mask, event_value, offset);
        }
        for lfo in &mut self.lfos {
            lfo.trigger(retrigger.mask, event_value, offset);
        }
        for random in &mut self.random_lfos {
            random.trigger(retrigger.mask, event_value, offset);
        }
        self.trigger_random.trigger(retrigger.mask, event_value, offset);

        if on_mask.any() {
            for oscillator in &mut self.oscillators {
                oscillator.note_on(on_mask, retrigger.offset);
            }
            self.sampler.note_on(on_mask, retrigger.offset);
        }
    }

    /// Computes all control-rate modulator values for the block.
    fn update_modulators(&mut self, controls: &VoiceControls, num_samples: usize) {
        // Envelope 0 runs at audio rate (amplitude + voice killer); the
        // others at control rate.
        self.envelopes[0]
            .process_audio(&self.resolved_env_params(0), &mut self.amp_env[..num_samples]);
        self.sources.envelopes[0] = self.envelopes[0].value();
        for i in 1..NUM_ENVELOPES {
            let params = self.resolved_env_params(i);
            self.sources.envelopes[i] = self.envelopes[i].process_control(&params, num_samples);
        }

        for i in 0..NUM_LFOS {
            let mut params = self.params.lfos[i].params.clone();
            params.frequency = params.frequency + self.offsets.lfo_frequency[i];
            self.sources.lfos[i] =
                self.lfos[i].process_control(&self.params.lfos[i].shape, &params, num_samples)
                    * 0.5
                    + 0.5;
        }

        for i in 0..NUM_RANDOM_LFOS {
            self.sources.random_lfos[i] = self.random_lfos[i]
                .process_control(&self.params.random_lfos[i], num_samples)
                * 0.5
                + 0.5;
        }

        self.sources.macros = self.params.macros.map(PolyF32::splat);
        self.sources.note = controls.note.value * (1.0 / 127.0);
        self.sources.note_in_octave = controls.note_in_octave;
        self.sources.velocity = controls.velocity.value;
        self.sources.lift = controls.lift.value;
        self.sources.mod_wheel = controls.mod_wheel;
        self.sources.pitch_wheel = controls.pitch_wheel_percent;
        self.sources.aftertouch = controls.aftertouch.value;
        self.sources.slide = controls.slide.value;
        self.sources.random = self.trigger_random.value() * 0.5 + 0.5;
        self.sources.stereo = PolyF32::stereo(0.0, 1.0);
    }

    fn resolved_env_params(&self, i: usize) -> EnvelopeParams {
        let mut params = self.params.envelopes[i].clone();
        params.attack = (params.attack + self.offsets.env_attack[i]).max(PolyF32::ZERO);
        params.decay = (params.decay + self.offsets.env_decay[i]).max(PolyF32::ZERO);
        params.sustain = (params.sustain + self.offsets.env_sustain[i]).clamp(0.0, 1.0);
        params.release = (params.release + self.offsets.env_release[i]).max(PolyF32::ZERO);
        params
    }

    fn bent_midi(&self, controls: &VoiceControls) -> PolyF32 {
        controls.note.value
            + controls.local_pitch_bend
            + controls.pitch_wheel * self.params.pitch_bend_range
            + self.offsets.pitch_bend
    }

    fn run_producers(&mut self, controls: &VoiceControls, num_samples: usize) {
        self.filter1_bus[..num_samples].fill(PolyF32::ZERO);
        self.filter2_bus[..num_samples].fill(PolyF32::ZERO);
        self.effects_bus[..num_samples].fill(PolyF32::ZERO);

        let midi = self.bent_midi(controls);

        // Reverse order so FM modulators are fresh: osc i is modulated by
        // osc i+1's raw output (v1 wiring; the reference's selectable pair
        // routing comes later).
        for i in (0..NUM_OSCILLATORS).rev() {
            let section = &self.params.oscillators[i];
            if !section.on {
                self.raw[i][..num_samples].fill(PolyF32::ZERO);
                continue;
            }
            let mut params = section.params.clone();
            params.midi_note = midi;
            params.amplitude =
                (params.amplitude + self.offsets.osc_level[i]).clamp(0.0, 1.0);
            params.transpose = params.transpose + self.offsets.osc_transpose[i];
            params.tune = params.tune + self.offsets.osc_tune[i];
            params.wave_frame = params.wave_frame + self.offsets.osc_frame[i];
            params.pan = (params.pan + self.offsets.osc_pan[i]).clamp(-1.0, 1.0);
            params.unison_detune =
                (params.unison_detune + self.offsets.osc_unison_detune[i]).clamp(0.0, 1.0);
            params.distortion_amount = (params.distortion_amount
                + self.offsets.osc_distortion_amount[i])
                .clamp(0.0, 1.0);
            params.spectral_morph_amount = (params.spectral_morph_amount
                + self.offsets.osc_spectral_morph_amount[i])
                .clamp(0.0, 1.0);
            params.phase = (params.phase + self.offsets.osc_phase[i]).fract();

            let (before, current_and_after) = self.raw.split_at_mut(i + 1);
            let raw_out = &mut before[i];
            let modulation: Option<&[PolyF32]> =
                current_and_after.first().map(|m| &m[..num_samples]);

            let wavetable = &self.wavetables[i];
            self.oscillators[i].process(
                &params,
                wavetable,
                modulation,
                num_samples,
                &mut raw_out[..num_samples],
                &mut self.leveled[..num_samples],
            );

            route(
                section.destination,
                &self.leveled[..num_samples],
                &mut self.filter1_bus,
                &mut self.filter2_bus,
                &mut self.effects_bus,
            );
        }

        if self.params.sample.on {
            let mut params = self.params.sample.params.clone();
            params.midi = midi;
            params.level = (params.level + self.offsets.sample_level).clamp(0.0, 1.0);
            let raw = &mut self.serial_bus; // reuse as sampler raw scratch
            self.sampler.process(
                &params,
                num_samples,
                &mut raw[..num_samples],
                &mut self.leveled[..num_samples],
            );
            route(
                self.params.sample.destination,
                &self.leveled[..num_samples],
                &mut self.filter1_bus,
                &mut self.filter2_bus,
                &mut self.effects_bus,
            );
        }
    }

    fn run_filters(&mut self, controls: &VoiceControls, num_samples: usize, reset_mask: PolyMask) {
        let note = controls.note.value;
        let mut filter_params: [VoiceFilterParams; 2] = [
            self.params.filters[0].params,
            self.params.filters[1].params,
        ];
        for (i, params) in filter_params.iter_mut().enumerate() {
            let keytrack = (note - 60.0) * self.params.filters[i].keytrack;
            params.state.midi_cutoff =
                params.state.midi_cutoff + keytrack + self.offsets.filter_cutoff[i];
            params.state.resonance_percent = (params.state.resonance_percent
                + self.offsets.filter_resonance[i])
                .clamp(0.0, 1.0);
            params.state.set_pass_blend(
                params.state.pass_blend + self.offsets.filter_blend[i],
            );
            params.mix = (params.mix + self.offsets.filter_mix[i]).clamp(0.0, 1.0);
        }

        match self.params.filter_routing {
            FilterRouting::Parallel => {
                self.filters[0].process(
                    &filter_params[0],
                    &self.filter1_bus[..num_samples],
                    &mut self.filter1_out[..num_samples],
                    reset_mask,
                );
                self.filters[1].process(
                    &filter_params[1],
                    &self.filter2_bus[..num_samples],
                    &mut self.filter2_out[..num_samples],
                    reset_mask,
                );
            }
            FilterRouting::SerialForward => {
                self.filters[0].process(
                    &filter_params[0],
                    &self.filter1_bus[..num_samples],
                    &mut self.filter1_out[..num_samples],
                    reset_mask,
                );
                for i in 0..num_samples {
                    self.serial_bus[i] = self.filter2_bus[i] + self.filter1_out[i];
                }
                self.filter1_out[..num_samples].fill(PolyF32::ZERO);
                self.filters[1].process(
                    &filter_params[1],
                    &self.serial_bus[..num_samples],
                    &mut self.filter2_out[..num_samples],
                    reset_mask,
                );
            }
            FilterRouting::SerialBackward => {
                self.filters[1].process(
                    &filter_params[1],
                    &self.filter2_bus[..num_samples],
                    &mut self.filter2_out[..num_samples],
                    reset_mask,
                );
                for i in 0..num_samples {
                    self.serial_bus[i] = self.filter1_bus[i] + self.filter2_out[i];
                }
                self.filter2_out[..num_samples].fill(PolyF32::ZERO);
                self.filters[0].process(
                    &filter_params[0],
                    &self.serial_bus[..num_samples],
                    &mut self.filter1_out[..num_samples],
                    reset_mask,
                );
            }
        }

        // Filters that are off pass their bus through dry (reference: an
        // off FilterModule outputs silence, but its input bus is routed
        // straight to output by the producers wiring; net effect is dry).
        if !filter_params[0].on {
            self.filter1_out[..num_samples].copy_from_slice(&self.filter1_bus[..num_samples]);
        }
        if !filter_params[1].on
            && self.params.filter_routing == FilterRouting::Parallel
        {
            self.filter2_out[..num_samples].copy_from_slice(&self.filter2_bus[..num_samples]);
        }
    }
}

/// A single saw frame: the audible default until a table is loaded.
fn default_wavetable() -> Wavetable {
    use vital_dsp::wavetable::{WaveFrame, WaveShape};
    let mut wavetable = Wavetable::new(1);
    wavetable.load_wave_frame(&WaveFrame::predefined(WaveShape::Saw));
    wavetable.post_process(0.0);
    wavetable
}

#[inline]
fn first_offset(trigger: &Trigger) -> usize {
    let mask = trigger.mask.to_u32();
    let offsets = trigger.offset;
    let mut result = u32::MAX;
    for lane in 0..vital_poly::LANES {
        if mask.lane(lane) != 0 {
            result = result.min(offsets.lane(lane));
        }
    }
    if result == u32::MAX {
        0
    } else {
        result as usize
    }
}

fn route(
    destination: ProducerDestination,
    leveled: &[PolyF32],
    filter1_bus: &mut [PolyF32],
    filter2_bus: &mut [PolyF32],
    effects_bus: &mut [PolyF32],
) {
    let num_samples = leveled.len();
    if destination.feeds_filter_1() {
        for i in 0..num_samples {
            filter1_bus[i] += leveled[i];
        }
    }
    if destination.feeds_filter_2() {
        for i in 0..num_samples {
            filter2_bus[i] += leveled[i];
        }
    }
    if matches!(destination, ProducerDestination::Effects | ProducerDestination::DirectOut) {
        for i in 0..num_samples {
            effects_bus[i] += leveled[i];
        }
    }
}

impl VoiceKernel for SynthVoiceKernel {
    fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
        let sr = sample_rate as f32;
        for oscillator in &mut self.oscillators {
            oscillator.set_sample_rate(sr);
        }
        for filter in &mut self.filters {
            filter.set_sample_rate(sr);
        }
        self.envelopes = core::array::from_fn(|_| Envelope::new(sr));
        self.lfos = core::array::from_fn(|_| SynthLfo::new(sr));
        self.random_lfos = core::array::from_fn(|_| RandomLfo::new(sr));
    }

    fn process(&mut self, controls: &VoiceControls, num_samples: usize) {
        debug_assert!(num_samples <= MAX_BLOCK);

        let reset_mask = controls.reset.mask;
        self.dispatch_triggers(controls);
        self.update_modulators(controls, num_samples);

        // Matrix uses last tick's modulator values for its own params â€”
        // resolve after updating modulators, before building params.
        let sources = self.sources.clone();
        let mut offsets = std::mem::take(&mut self.offsets);
        self.matrix.resolve(&sources, &mut offsets, reset_mask);
        self.offsets = offsets;

        self.run_producers(controls, num_samples);
        self.run_filters(controls, num_samples, reset_mask);

        // Amplitude: squared amp envelope with velocity tracking.
        let velocity_scale = vital_poly::utils::interpolate(
            PolyF32::ONE,
            controls.velocity.value,
            PolyF32::splat(self.params.velocity_track),
        );
        let amp_offset = self.offsets.volume_amp;
        for i in 0..num_samples {
            let env = self.amp_env[i];
            let amplitude = (env * env + amp_offset).max(PolyF32::ZERO) * velocity_scale;
            self.output[i] = (self.filter1_out[i] + self.filter2_out[i] + self.effects_bus[i])
                * amplitude
                * controls.active_mask;
        }
    }

    fn output(&self) -> &[PolyF32] {
        &self.output
    }

    fn voice_killer(&self) -> Option<&[PolyF32]> {
        Some(&self.amp_env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::VoiceAllocator;
    use crate::kernel::mod_matrix::{Connection, ModDest, ModSource};
    use crate::modulation::ModulationTransform;

    fn make_allocator() -> VoiceAllocator<SynthVoiceKernel> {
        let mut allocator = VoiceAllocator::new(8, || {
            let mut kernel = SynthVoiceKernel::new(44100);
            // Fast envelope so tests are short.
            kernel.params.envelopes[0] = EnvelopeParams {
                attack: PolyF32::splat(0.001),
                release: PolyF32::splat(0.02),
                sustain: PolyF32::ONE,
                ..Default::default()
            };
            kernel
        });
        allocator.set_sample_rate(44100);
        allocator
    }

    fn render_blocks(allocator: &mut VoiceAllocator<SynthVoiceKernel>, blocks: usize) -> Vec<f32> {
        let mut rendered = Vec::new();
        for _ in 0..blocks {
            let mut mix = vec![PolyF32::ZERO; MAX_BUFFER_SIZE];
            allocator.process(MAX_BUFFER_SIZE, |out| {
                for (dest, src) in mix.iter_mut().zip(out) {
                    *dest += *src;
                }
            });
            for value in &mix {
                let folded = *value + value.swap_voices();
                rendered.push(folded.lane(0));
            }
        }
        rendered
    }

    #[test]
    fn note_produces_audio_and_release_silences() {
        let mut allocator = make_allocator();
        allocator.note_on(60, 1.0, 0, 0);
        let sustain = render_blocks(&mut allocator, 8);
        let sustain_peak = sustain.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(sustain_peak > 0.01, "no audio produced: peak {sustain_peak}");
        assert!(sustain.iter().all(|v| v.is_finite()));

        allocator.note_off(60, 0.5, 0, 0);
        // Enough blocks for the 20 ms release to finish.
        let tail = render_blocks(&mut allocator, 12);
        let tail_end_peak = tail[tail.len() - 256..]
            .iter()
            .fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(tail_end_peak < 1e-4, "voice did not silence: {tail_end_peak}");
        assert_eq!(allocator.num_active_voices(), 0, "voice was not retired");
    }

    #[test]
    fn lfo_to_cutoff_modulation_changes_spectrum_over_time() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.filters[0].params.on = true;
            kernel.params.filters[0].params.state.midi_cutoff = PolyF32::splat(60.0);
            kernel.params.lfos[0].params.frequency = PolyF32::splat(8.0);
            kernel.matrix.connections.push(Connection {
                source: ModSource::Lfo(0),
                dest: ModDest::FilterCutoff(0),
                transform: ModulationTransform::with_amount(1.0, 60.0),
            });
        }
        allocator.note_on(48, 1.0, 0, 0);
        let audio = render_blocks(&mut allocator, 40);

        // Compare block RMS across time: a swept filter makes them vary.
        let block_rms: Vec<f32> = audio
            .chunks(512)
            .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        let max = block_rms[2..].iter().cloned().fold(0.0f32, f32::max);
        let min = block_rms[2..].iter().cloned().fold(f32::MAX, f32::min);
        assert!(max > 0.0);
        assert!(
            max / min.max(1e-9) > 1.05,
            "cutoff modulation had no audible effect: {min}..{max}"
        );
    }

    #[test]
    fn two_oscillators_detuned_beat() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.oscillators[1].on = true;
            kernel.params.oscillators[1].params.tune = PolyF32::splat(0.1);
            kernel.params.oscillators[1].destination = ProducerDestination::Filter1;
        }
        allocator.note_on(60, 1.0, 0, 0);
        let audio = render_blocks(&mut allocator, 20);
        assert!(audio.iter().all(|v| v.is_finite()));
        let peak = audio.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        assert!(peak > 0.01);
    }
}

