//! The complete synth voice kernel: 4 switchable-engine oscillator slots
//! (wavetable / sample / granular / multisample, Serum-2 style) + legacy
//! sampler + noise source, two switchable filters, 8 envelopes, 12 LFOs,
//! 4 random LFOs and the modulation matrix, statically wired (rework of
//! `SynthVoiceHandler` + `ProducersModule` + `FiltersModule`).

use std::sync::Arc;

use spinwave_dsp::filters::filter_state;
use spinwave_dsp::filters::DcFilter;
use spinwave_dsp::modulators::{
    Envelope, EnvelopeParams, LineGenerator, RandomLfo, RandomLfoParams, SynthLfo, SynthLfoParams,
    TriggerRandom,
};
use spinwave_dsp::oscillator::noise::{NoiseParams, NoiseSource};
use spinwave_dsp::oscillator::sample_source::BUFFER_SAMPLES;
use spinwave_dsp::oscillator::{
    Granular, GranularParams, Multisample, MultisampleSource, Sample, SampleSource,
    SampleSourceParams, SynthOscillator, SynthOscillatorParams,
};
use spinwave_dsp::wavetable::Wavetable;
use spinwave_poly::constants::{MAX_BUFFER_SIZE, VoiceEvent};
use spinwave_poly::{PolyF32, PolyMask, LANES};

use crate::allocator::VoiceKernel;
use crate::kernel::mod_matrix::{
    ModMatrix, ModOffsets, SourceValues, NUM_ENVELOPES, NUM_LFOS, NUM_MACROS, NUM_OSCILLATORS,
    NUM_RANDOM_LFOS,
};
use crate::kernel::voice_filter::{VoiceFilter, VoiceFilterParams};
use crate::tempo::LfoSync;
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
    /// Spinwave extension: hard-route this producer into effect bus A.
    BusA = 5,
    /// Spinwave extension: hard-route this producer into effect bus B.
    BusB = 6,
}

impl ProducerDestination {
    pub fn from_index(index: i32) -> ProducerDestination {
        match index {
            1 => ProducerDestination::Filter2,
            2 => ProducerDestination::DualFilters,
            3 => ProducerDestination::Effects,
            4 => ProducerDestination::DirectOut,
            5 => ProducerDestination::BusA,
            6 => ProducerDestination::BusB,
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

/// Which sound engine an oscillator slot runs (Serum-2 style switchable
/// engines). Every slot holds all engine instances, so switching never
/// allocates; the selection takes effect at the next processed block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OscEngineKind {
    #[default]
    Wavetable,
    Sample,
    Granular,
    Multisample,
}

/// One oscillator slot: engine selection plus the per-engine parameters.
///
/// Modulation offsets apply to the active engine's common fields:
/// - Wavetable: every `osc_*` offset (level, transpose, tune, frame, ...).
/// - Sample / Multisample: `osc_level` → level, `osc_transpose` → transpose,
///   `osc_tune` → tune, `osc_pan` → pan; the rest are ignored.
/// - Granular: `osc_level` → level, `osc_transpose` → transpose,
///   `osc_tune` → tune (granular has no pan); the rest are ignored.
#[derive(Clone, Debug)]
pub struct OscSection {
    pub on: bool,
    pub engine: OscEngineKind,
    pub destination: ProducerDestination,
    /// Wavetable engine parameters.
    pub params: SynthOscillatorParams,
    /// Sample AND Multisample engine parameters (pitch/level/loop; the
    /// multisample engine overrides keytrack/loop per zone).
    pub sample_params: SampleSourceParams,
    /// Granular engine parameters.
    pub granular_params: GranularParams,
}

impl Default for OscSection {
    fn default() -> Self {
        OscSection {
            on: false,
            engine: OscEngineKind::Wavetable,
            destination: ProducerDestination::Filter1,
            params: SynthOscillatorParams::default(),
            sample_params: SampleSourceParams::default(),
            granular_params: GranularParams::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SampleSection {
    pub on: bool,
    pub destination: ProducerDestination,
    pub params: SampleSourceParams,
}

/// The dedicated noise source (white/pink blend with tilt), routed like
/// every other producer. No modulation offsets target it yet.
#[derive(Clone, Debug)]
pub struct NoiseSection {
    pub on: bool,
    pub destination: ProducerDestination,
    pub params: NoiseParams,
}

impl Default for NoiseSection {
    fn default() -> Self {
        NoiseSection {
            on: false,
            destination: ProducerDestination::Filter1,
            params: NoiseParams::default(),
        }
    }
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
    /// Tempo sync: free mode uses `params.frequency`, the tempo modes
    /// resolve the ratio table against [`KernelParams::beats_per_second`].
    pub sync: LfoSync,
}

impl Default for LfoSection {
    fn default() -> Self {
        LfoSection {
            params: SynthLfoParams::default(),
            shape: LineGenerator::triangle(),
            sync: LfoSync::default(),
        }
    }
}

/// A random LFO with its tempo sync selection. `params.sync` stays the
/// transport-follow flag; `sync` here is the tempo-ratio resolution for
/// `params.frequency`, mirroring [`LfoSection::sync`].
#[derive(Clone, Debug, Default)]
pub struct RandomLfoSection {
    pub params: RandomLfoParams,
    pub sync: LfoSync,
}

/// All base (unmodulated) voice parameters, set from the parameter layer.
#[derive(Clone)]
pub struct KernelParams {
    pub oscillators: [OscSection; NUM_OSCILLATORS],
    pub sample: SampleSection,
    pub noise: NoiseSection,
    pub filters: [FilterSection; 2],
    pub filter_routing: FilterRouting,
    pub envelopes: [EnvelopeParams; NUM_ENVELOPES],
    pub lfos: [LfoSection; NUM_LFOS],
    pub random_lfos: [RandomLfoSection; NUM_RANDOM_LFOS],
    /// How much velocity scales the voice amplitude, `[0, 1]`.
    pub velocity_track: f32,
    /// Pitch wheel range in semitones.
    pub pitch_bend_range: f32,
    pub macros: [f32; NUM_MACROS],
    /// Host tempo in beats per second, fed by `SoundEngine::set_bpm`
    /// (default 2.0 = 120 bpm). Tempo-synced LFOs resolve against it.
    pub beats_per_second: f32,
}

impl Default for KernelParams {
    fn default() -> Self {
        let mut oscillators: [OscSection; NUM_OSCILLATORS] = Default::default();
        oscillators[0].on = true;
        KernelParams {
            oscillators,
            sample: SampleSection::default(),
            noise: NoiseSection::default(),
            filters: Default::default(),
            filter_routing: FilterRouting::Parallel,
            envelopes: Default::default(),
            lfos: Default::default(),
            random_lfos: Default::default(),
            velocity_track: 0.6,
            pitch_bend_range: 2.0,
            macros: [0.0; NUM_MACROS],
            beats_per_second: 2.0,
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
    /// Per-slot Sample engines. Each owns a private copy of the slot's
    /// sample material, rebuilt by [`Self::set_sample`].
    slot_samplers: [SampleSource; NUM_OSCILLATORS],
    /// Per-slot Granular engines, reading from `slot_samples`.
    slot_granulars: [Granular; NUM_OSCILLATORS],
    /// Per-slot Multisample engines (empty by default: silent until
    /// [`Self::set_multisample`] installs zones).
    slot_multisamples: [MultisampleSource; NUM_OSCILLATORS],
    /// Per-slot sample material shared by the Sample and Granular engines
    /// (the granular engine reads it directly; the Sample engine keeps a
    /// rebuilt private copy because `SampleSource` owns its sample).
    slot_samples: [Arc<Sample>; NUM_OSCILLATORS],
    sampler: SampleSource,
    noise: NoiseSource,
    filters: [VoiceFilter; 2],
    envelopes: [Envelope; NUM_ENVELOPES],
    lfos: [SynthLfo; NUM_LFOS],
    random_lfos: [RandomLfo; NUM_RANDOM_LFOS],
    trigger_random: TriggerRandom,

    /// DC blockers on the two voice output buses (`dc_filter.{h,cpp}`).
    dc_filter: DcFilter,
    direct_dc_filter: DcFilter,

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
    direct_bus: Vec<PolyF32>,
    direct_out: Vec<PolyF32>,
    bus_a_bus: Vec<PolyF32>,
    bus_a_out: Vec<PolyF32>,
    bus_b_bus: Vec<PolyF32>,
    bus_b_out: Vec<PolyF32>,
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
            slot_samplers: core::array::from_fn(|_| {
                let mut source = SampleSource::with_sample(empty_sample());
                source.set_sample_rate(sr);
                source
            }),
            slot_granulars: core::array::from_fn(|_| {
                let mut granular = Granular::new();
                granular.set_sample_rate(sr);
                granular
            }),
            slot_multisamples: core::array::from_fn(|_| {
                let mut source = MultisampleSource::new(empty_multisample());
                source.set_sample_rate(sr);
                source
            }),
            slot_samples: core::array::from_fn(|_| Arc::new(empty_sample())),
            sampler: SampleSource::new(),
            noise: NoiseSource::new(),
            filters: core::array::from_fn(|_| VoiceFilter::new(sr)),
            envelopes: core::array::from_fn(|_| Envelope::new(sr)),
            lfos: core::array::from_fn(|_| SynthLfo::new(sr)),
            random_lfos: core::array::from_fn(|_| RandomLfo::new(sr)),
            trigger_random: TriggerRandom::new(),
            dc_filter: DcFilter::new(sr),
            direct_dc_filter: DcFilter::new(sr),
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
            direct_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            direct_out: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_a_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_a_out: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_b_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_b_out: vec![PolyF32::ZERO; MAX_BLOCK],
        }
    }

    pub fn set_wavetable(&mut self, index: usize, wavetable: Arc<Wavetable>) {
        self.wavetables[index] = wavetable;
    }

    /// Installs the sample material for one oscillator slot, shared by the
    /// Sample and Granular engines. The Granular engine reads the `Arc`
    /// directly; the Sample engine rebuilds its private band-limited copy
    /// (`SampleSource` owns its sample), which recomputes the tier pyramid —
    /// call this at patch-load time, not per block.
    pub fn set_sample(&mut self, slot: usize, sample: Arc<Sample>) {
        let mut copy = duplicate_sample(&sample);
        copy.set_slices(sample.slices().to_vec());
        *self.slot_samplers[slot].sample_mut() = copy;
        self.slot_samples[slot] = sample;
    }

    /// Sample material of one slot (as installed by [`Self::set_sample`]).
    pub fn slot_sample(&self, slot: usize) -> &Arc<Sample> {
        &self.slot_samples[slot]
    }

    /// Installs the multisample instrument for one oscillator slot.
    /// `Multisample` owns its zone samples (it is not `Clone`), so each
    /// kernel needs its own instance — build one per kernel from the SFZ
    /// source at patch-load time.
    pub fn set_multisample(&mut self, slot: usize, multisample: Multisample) {
        let mut source = MultisampleSource::new(multisample);
        source.set_sample_rate(self.sample_rate as f32);
        self.slot_multisamples[slot] = source;
    }

    pub fn sampler_mut(&mut self) -> &mut SampleSource {
        &mut self.sampler
    }

    /// Modulation source values of the last processed block. The engine
    /// reads these from the most recently active kernel to drive the mono
    /// (bus-effect) modulation matrix, like Vital's mono modulations.
    pub fn last_source_values(&self) -> &SourceValues {
        &self.sources
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
            // Every engine of every slot is notified (cheap): a slot whose
            // engine switches mid-life starts the next note cleanly.
            for oscillator in &mut self.oscillators {
                oscillator.note_on(on_mask, retrigger.offset);
            }
            for sampler in &mut self.slot_samplers {
                sampler.note_on(on_mask, retrigger.offset);
            }
            for granular in &mut self.slot_granulars {
                granular.note_on(on_mask, retrigger.offset);
            }
            // Multisample zone selection needs the per-voice note and
            // velocity, so its note-on dispatches per voice.
            for voice in 0..LANES / 2 {
                let mask = on_mask & voice_lane_mask(voice);
                if !mask.any() {
                    continue;
                }
                let note = controls.note.value.lane(voice * 2).round().clamp(0.0, 127.0) as u8;
                let velocity =
                    (controls.velocity.value.lane(voice * 2) * 127.0).round().clamp(0.0, 127.0)
                        as u8;
                for multisample in &mut self.slot_multisamples {
                    multisample.note_on(mask, retrigger.offset, note, velocity);
                }
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

        let beats_per_second = self.params.beats_per_second;
        for i in 0..NUM_LFOS {
            let section = &self.params.lfos[i];
            let mut params = section.params;
            params.frequency = section.sync.resolve(
                params.frequency + self.offsets.lfo_frequency[i],
                beats_per_second,
            );
            // The LFO wraps its phase internally, so the offset adds raw.
            params.phase += self.offsets.lfo_phase[i];
            self.sources.lfos[i] =
                self.lfos[i].process_control(&section.shape, &params, num_samples) * 0.5 + 0.5;
        }

        for i in 0..NUM_RANDOM_LFOS {
            let section = &self.params.random_lfos[i];
            let mut params = section.params;
            params.frequency = section.sync.resolve(
                params.frequency + self.offsets.random_lfo_frequency[i],
                beats_per_second,
            );
            self.sources.random_lfos[i] =
                self.random_lfos[i].process_control(&params, num_samples) * 0.5 + 0.5;
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
        let mut params = self.params.envelopes[i];
        params.delay = (params.delay + self.offsets.env_delay[i]).max(PolyF32::ZERO);
        params.attack = (params.attack + self.offsets.env_attack[i]).max(PolyF32::ZERO);
        params.attack_power =
            (params.attack_power + self.offsets.env_attack_power[i]).clamp(-20.0, 20.0);
        params.hold = (params.hold + self.offsets.env_hold[i]).max(PolyF32::ZERO);
        params.decay = (params.decay + self.offsets.env_decay[i]).max(PolyF32::ZERO);
        params.decay_power =
            (params.decay_power + self.offsets.env_decay_power[i]).clamp(-20.0, 20.0);
        params.sustain = (params.sustain + self.offsets.env_sustain[i]).clamp(0.0, 1.0);
        params.release = (params.release + self.offsets.env_release[i]).max(PolyF32::ZERO);
        params.release_power =
            (params.release_power + self.offsets.env_release_power[i]).clamp(-20.0, 20.0);
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
        self.direct_bus[..num_samples].fill(PolyF32::ZERO);
        self.bus_a_bus[..num_samples].fill(PolyF32::ZERO);
        self.bus_b_bus[..num_samples].fill(PolyF32::ZERO);

        let midi = self.bent_midi(controls);

        // Reverse order so FM modulators are fresh: wavetable osc i is
        // modulated by osc i+1's raw output (v1 wiring; the reference's
        // selectable pair routing comes later). Each slot dispatches on its
        // engine; the engine renders into `leveled` which is then routed.
        for i in (0..NUM_OSCILLATORS).rev() {
            let section = &self.params.oscillators[i];
            if !section.on {
                self.raw[i][..num_samples].fill(PolyF32::ZERO);
                continue;
            }
            let destination = section.destination;
            match section.engine {
                OscEngineKind::Wavetable => {
                    let mut params = section.params.clone();
                    params.midi_note = midi;
                    params.amplitude =
                        (params.amplitude + self.offsets.osc_level[i]).clamp(0.0, 1.0);
                    params.transpose += self.offsets.osc_transpose[i];
                    params.tune += self.offsets.osc_tune[i];
                    params.wave_frame += self.offsets.osc_frame[i];
                    params.frame_spread += self.offsets.osc_frame_spread[i];
                    params.pan = (params.pan + self.offsets.osc_pan[i]).clamp(-1.0, 1.0);
                    params.unison_detune = (params.unison_detune
                        + self.offsets.osc_unison_detune[i])
                        .clamp(0.0, 1.0);
                    params.blend =
                        (params.blend + self.offsets.osc_unison_blend[i]).clamp(0.0, 1.0);
                    params.stereo_spread = (params.stereo_spread
                        + self.offsets.osc_stereo_spread[i])
                        .clamp(0.0, 1.0);
                    params.distortion_amount = (params.distortion_amount
                        + self.offsets.osc_distortion_amount[i])
                        .clamp(0.0, 1.0);
                    params.distortion_phase = (params.distortion_phase
                        + self.offsets.osc_distortion_phase[i])
                        .clamp(0.0, 1.0);
                    params.spectral_morph_amount = (params.spectral_morph_amount
                        + self.offsets.osc_spectral_morph_amount[i])
                        .clamp(0.0, 1.0);
                    params.phase = (params.phase + self.offsets.osc_phase[i]).fract();

                    // FM stays wavetable-only: the modulation input comes
                    // from the next slot's raw output only when that slot
                    // is an active Wavetable engine.
                    let next_is_wavetable = i + 1 < NUM_OSCILLATORS && {
                        let next = &self.params.oscillators[i + 1];
                        next.on && next.engine == OscEngineKind::Wavetable
                    };
                    let (before, current_and_after) = self.raw.split_at_mut(i + 1);
                    let raw_out = &mut before[i];
                    let modulation: Option<&[PolyF32]> = if next_is_wavetable {
                        current_and_after.first().map(|m| &m[..num_samples])
                    } else {
                        None
                    };

                    let wavetable = &self.wavetables[i];
                    self.oscillators[i].process(
                        &params,
                        wavetable,
                        modulation,
                        num_samples,
                        &mut raw_out[..num_samples],
                        &mut self.leveled[..num_samples],
                    );
                }
                OscEngineKind::Sample => {
                    let mut params = section.sample_params.clone();
                    params.midi = midi;
                    params.level = (params.level + self.offsets.osc_level[i]).clamp(0.0, 1.0);
                    params.transpose += self.offsets.osc_transpose[i];
                    params.tune += self.offsets.osc_tune[i];
                    params.pan = (params.pan + self.offsets.osc_pan[i]).clamp(-1.0, 1.0);
                    self.slot_samplers[i].process(
                        &params,
                        num_samples,
                        &mut self.raw[i][..num_samples],
                        &mut self.leveled[..num_samples],
                    );
                }
                OscEngineKind::Granular => {
                    let mut params = section.granular_params.clone();
                    params.midi = midi;
                    params.level = (params.level + self.offsets.osc_level[i]).clamp(0.0, 1.0);
                    params.transpose += self.offsets.osc_transpose[i];
                    params.tune += self.offsets.osc_tune[i];
                    // Granular renders leveled output only; its raw buffer
                    // stays silent (it never feeds FM).
                    self.raw[i][..num_samples].fill(PolyF32::ZERO);
                    self.slot_granulars[i].process(
                        &params,
                        &self.slot_samples[i],
                        num_samples,
                        &mut self.leveled[..num_samples],
                    );
                }
                OscEngineKind::Multisample => {
                    let mut params = section.sample_params.clone();
                    params.midi = midi;
                    params.level = (params.level + self.offsets.osc_level[i]).clamp(0.0, 1.0);
                    params.transpose += self.offsets.osc_transpose[i];
                    params.tune += self.offsets.osc_tune[i];
                    params.pan = (params.pan + self.offsets.osc_pan[i]).clamp(-1.0, 1.0);
                    // MultisampleSource caps its blocks at MAX_BUFFER_SIZE;
                    // oversampled kernel blocks are processed in chunks.
                    let mut start = 0;
                    while start < num_samples {
                        let chunk = (num_samples - start).min(MAX_BUFFER_SIZE);
                        self.slot_multisamples[i].process(
                            &params,
                            chunk,
                            &mut self.raw[i][start..start + chunk],
                            &mut self.leveled[start..start + chunk],
                        );
                        start += chunk;
                    }
                }
            }

            route(
                destination,
                &self.leveled[..num_samples],
                &mut ProducerBuses {
                    filter1: &mut self.filter1_bus,
                    filter2: &mut self.filter2_bus,
                    effects: &mut self.effects_bus,
                    direct: &mut self.direct_bus,
                    bus_a: &mut self.bus_a_bus,
                    bus_b: &mut self.bus_b_bus,
                },
            );
        }

        if self.params.sample.on {
            let mut params = self.params.sample.params.clone();
            params.midi = midi;
            params.level = (params.level + self.offsets.sample_level).clamp(0.0, 1.0);
            params.transpose += self.offsets.sample_transpose;
            params.tune += self.offsets.sample_tune;
            params.pan = (params.pan + self.offsets.sample_pan).clamp(-1.0, 1.0);
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
                &mut ProducerBuses {
                    filter1: &mut self.filter1_bus,
                    filter2: &mut self.filter2_bus,
                    effects: &mut self.effects_bus,
                    direct: &mut self.direct_bus,
                    bus_a: &mut self.bus_a_bus,
                    bus_b: &mut self.bus_b_bus,
                },
            );
        }

        if self.params.noise.on {
            let params = self.params.noise.params;
            let destination = self.params.noise.destination;
            self.noise.process(&params, num_samples, &mut self.leveled[..num_samples]);
            route(
                destination,
                &self.leveled[..num_samples],
                &mut ProducerBuses {
                    filter1: &mut self.filter1_bus,
                    filter2: &mut self.filter2_bus,
                    effects: &mut self.effects_bus,
                    direct: &mut self.direct_bus,
                    bus_a: &mut self.bus_a_bus,
                    bus_b: &mut self.bus_b_bus,
                },
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
            let keytrack_amount = (PolyF32::splat(self.params.filters[i].keytrack)
                + self.offsets.filter_keytrack[i])
                .clamp(-1.0, 1.0);
            let keytrack = (note - 60.0) * keytrack_amount;
            params.state.midi_cutoff =
                params.state.midi_cutoff + keytrack + self.offsets.filter_cutoff[i];
            params.state.resonance_percent = (params.state.resonance_percent
                + self.offsets.filter_resonance[i])
                .clamp(0.0, 1.0);
            // Drive is stored as (magnitude, percent of the dB range); rebuild
            // the base dB from the percent and re-map with the offset applied.
            let base_drive_db = params.state.drive_percent
                * (filter_state::MAX_DRIVE_GAIN - filter_state::MIN_DRIVE_GAIN)
                + filter_state::MIN_DRIVE_GAIN;
            params
                .state
                .set_drive_db(base_drive_db + self.offsets.filter_drive[i]);
            params.state.set_pass_blend(
                params.state.pass_blend + self.offsets.filter_blend[i],
            );
            params.state.transpose += self.offsets.filter_blend_transpose[i];
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

/// Default table: the factory basic-shapes morph (sin → triangle → saw →
/// square → pulse), so wave-frame modulation works out of the box.
fn default_wavetable() -> Wavetable {
    spinwave_dsp::wavetable::factory::basic_shapes()
}

/// Zero-length sample: slot engines are silent until material is loaded
/// (and cost no memory per slot, unlike the default white-noise pyramid).
fn empty_sample() -> Sample {
    Sample::from_mono("empty", &[], spinwave_poly::constants::DEFAULT_SAMPLE_RATE)
}

/// A multisample with no zones: every note-on is silently ignored.
fn empty_multisample() -> Multisample {
    Multisample { zones: Vec::new(), warnings: Vec::new() }
}

/// Rebuilds an owned copy of a sample from its original-rate frames
/// (tier 1 of the pyramid, guard samples stripped). `Sample` is not
/// `Clone`, so sharing material with an engine that owns its sample
/// (`SampleSource`) means recomputing the band-limited tiers once.
fn duplicate_sample(sample: &Sample) -> Sample {
    let length = sample.original_length();
    let range = BUFFER_SAMPLES..BUFFER_SAMPLES + length;
    let left = &sample.left_buffer(1)[range.clone()];
    if sample.stereo() {
        let right = &sample.right_buffer(1)[range];
        Sample::from_stereo(&sample.name, left, right, sample.sample_rate())
    } else {
        Sample::from_mono(&sample.name, left, sample.sample_rate())
    }
}

/// Mask covering the two stereo lanes of one voice (voice 0 = lanes 0/1,
/// voice 1 = lanes 2/3), mirroring `Voice::mask`.
#[inline]
fn voice_lane_mask(voice: usize) -> PolyMask {
    PolyF32::from_lanes([0.0, 0.0, 1.0, 1.0]).eq(PolyF32::splat(voice as f32))
}

#[inline]
fn first_offset(trigger: &Trigger) -> usize {
    let mask = trigger.mask.to_u32();
    let offsets = trigger.offset;
    let mut result = u32::MAX;
    for lane in 0..spinwave_poly::LANES {
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

struct ProducerBuses<'a> {
    filter1: &'a mut [PolyF32],
    filter2: &'a mut [PolyF32],
    effects: &'a mut [PolyF32],
    direct: &'a mut [PolyF32],
    bus_a: &'a mut [PolyF32],
    bus_b: &'a mut [PolyF32],
}

fn route(destination: ProducerDestination, leveled: &[PolyF32], buses: &mut ProducerBuses) {
    let num_samples = leveled.len();
    let add_into = |bus: &mut [PolyF32]| {
        for i in 0..num_samples {
            bus[i] += leveled[i];
        }
    };
    if destination.feeds_filter_1() {
        add_into(buses.filter1);
    }
    if destination.feeds_filter_2() {
        add_into(buses.filter2);
    }
    match destination {
        ProducerDestination::Effects => add_into(buses.effects),
        ProducerDestination::DirectOut => add_into(buses.direct),
        ProducerDestination::BusA => add_into(buses.bus_a),
        ProducerDestination::BusB => add_into(buses.bus_b),
        _ => {}
    }
}

impl VoiceKernel for SynthVoiceKernel {
    fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
        let sr = sample_rate as f32;
        for oscillator in &mut self.oscillators {
            oscillator.set_sample_rate(sr);
        }
        for sampler in &mut self.slot_samplers {
            sampler.set_sample_rate(sr);
        }
        for granular in &mut self.slot_granulars {
            granular.set_sample_rate(sr);
        }
        for multisample in &mut self.slot_multisamples {
            multisample.set_sample_rate(sr);
        }
        for filter in &mut self.filters {
            filter.set_sample_rate(sr);
        }
        self.envelopes = core::array::from_fn(|_| Envelope::new(sr));
        self.lfos = core::array::from_fn(|_| SynthLfo::new(sr));
        self.random_lfos = core::array::from_fn(|_| RandomLfo::new(sr));
        self.dc_filter.set_sample_rate(sr);
        self.direct_dc_filter.set_sample_rate(sr);
    }

    fn process(&mut self, controls: &VoiceControls, num_samples: usize) {
        debug_assert!(num_samples <= MAX_BLOCK);

        let reset_mask = controls.reset.mask;
        if reset_mask.any() {
            self.dc_filter.reset(reset_mask);
            self.direct_dc_filter.reset(reset_mask);
            self.noise.reset(reset_mask);
        }
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

        // Amplitude: squared amp envelope with velocity tracking. The
        // direct-out bus is gated by the same voice amplitude (reference:
        // `direct_output_` multiplies the producers' direct bus by
        // `amplitude_`), and both buses pass a DC blocker.
        let velocity_scale = spinwave_poly::utils::interpolate(
            PolyF32::ONE,
            controls.velocity.value,
            PolyF32::splat(self.params.velocity_track),
        );
        let amp_offset = self.offsets.volume_amp;
        for i in 0..num_samples {
            let env = self.amp_env[i];
            let amplitude = (env * env + amp_offset).max(PolyF32::ZERO)
                * velocity_scale
                * controls.active_mask;
            self.output[i] = self.dc_filter.tick(
                (self.filter1_out[i] + self.filter2_out[i] + self.effects_bus[i]) * amplitude,
            );
            self.direct_out[i] = self.direct_dc_filter.tick(self.direct_bus[i] * amplitude);
            // Hard-routed effect buses share the voice amplitude gate; DC
            // blocking happens once in the bus chains' distortion staging,
            // so a plain gate is enough here.
            self.bus_a_out[i] = self.bus_a_bus[i] * amplitude;
            self.bus_b_out[i] = self.bus_b_bus[i] * amplitude;
        }
    }

    fn output(&self) -> &[PolyF32] {
        &self.output
    }

    fn direct_output(&self) -> Option<&[PolyF32]> {
        Some(&self.direct_out)
    }

    fn bus_outputs(&self) -> (Option<&[PolyF32]>, Option<&[PolyF32]>) {
        (Some(&self.bus_a_out), Some(&self.bus_b_out))
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
    use crate::tempo::SyncMode;

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
            allocator.process(MAX_BUFFER_SIZE, |outputs| {
                for (dest, src) in mix.iter_mut().zip(outputs.main) {
                    *dest += *src;
                }
                if let Some(direct) = outputs.direct {
                    for (dest, src) in mix.iter_mut().zip(direct) {
                        *dest += *src;
                    }
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
            // Frame 128 of the factory table is the saw anchor — rich in
            // harmonics so the filter sweep is measurable.
            kernel.params.oscillators[0].params.wave_frame = PolyF32::splat(128.0);
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
    fn dc_filter_removes_forced_offset() {
        // An 8 kHz kernel keeps the DC blocker's ~1 s time constant within
        // a short render; a constant looped sample is pure DC on the bus.
        let mut allocator = VoiceAllocator::new(2, || {
            let mut kernel = SynthVoiceKernel::new(8000);
            kernel.params.envelopes[0] = EnvelopeParams {
                attack: PolyF32::splat(0.001),
                sustain: PolyF32::ONE,
                release: PolyF32::splat(0.02),
                ..Default::default()
            };
            kernel.params.oscillators[0].on = false;
            kernel.params.sample.on = true;
            kernel.params.sample.destination = ProducerDestination::Effects;
            kernel.params.sample.params.loop_sample = true;
            kernel.sampler_mut().sample_mut().load_sample(&[0.8; 8000], 8000);
            kernel
        });
        allocator.set_sample_rate(8000);
        allocator.note_on(60, 1.0, 0, 0);

        // 300 blocks = 38400 samples = 4.8 s at 8 kHz, several times the
        // DC blocker's time constant.
        let audio = render_blocks(&mut allocator, 300);
        let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
        let early = mean(&audio[256..2048]);
        let late = mean(&audio[audio.len() - 2048..]);
        assert!(early.abs() > 0.1, "offset never reached the output: {early}");
        assert!(
            late.abs() < 0.1 * early.abs(),
            "DC offset was not removed: early {early}, late {late}"
        );
    }

    #[test]
    fn lfo_tempo_sync_matches_equivalent_free_frequency() {
        let render = |sync: LfoSync, free_hz: f32| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                // 150 bpm.
                kernel.params.beats_per_second = 2.5;
                kernel.params.filters[0].params.on = true;
                kernel.params.filters[0].params.state.midi_cutoff = PolyF32::splat(60.0);
                kernel.params.lfos[0].params.frequency = PolyF32::splat(free_hz);
                kernel.params.lfos[0].sync = sync;
                kernel.matrix.connections.push(Connection {
                    source: ModSource::Lfo(0),
                    dest: ModDest::FilterCutoff(0),
                    transform: ModulationTransform::with_amount(1.0, 60.0),
                });
            }
            allocator.note_on(48, 1.0, 0, 0);
            render_blocks(&mut allocator, 20)
        };

        // Ratio index 9 is 2/1: 2.0 * 2.5 bps = 5 Hz; the (bogus) free
        // frequency must be ignored in tempo mode.
        let synced = render(LfoSync { mode: SyncMode::Tempo, tempo_index: 9.0 }, 123.0);
        let free = render(LfoSync::default(), 5.0);
        assert!(synced.iter().any(|v| v.abs() > 0.01));
        for (a, b) in synced.iter().zip(&free) {
            assert!(
                (a - b).abs() < 1e-6,
                "tempo-synced LFO diverged from the 5 Hz free render"
            );
        }
    }

    // Random LFOs auto-seed from a global counter, so two allocators never
    // render bit-identically; verify the tempo resolution behaviorally
    // instead: a DC carrier with the random LFO stepping the sample level
    // must hold still under Freeze (ratio 0) even though the free
    // frequency says 20 Hz.
    #[test]
    fn random_lfo_tempo_sync_overrides_free_frequency() {
        use spinwave_dsp::modulators::RandomLfoStyle;

        let max_adjacent_block_ratio = |sync: LfoSync| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                kernel.params.oscillators[0].on = false;
                kernel.params.sample.on = true;
                kernel.params.sample.destination = ProducerDestination::Effects;
                kernel.params.sample.params.loop_sample = true;
                kernel.params.sample.params.level = PolyF32::splat(0.1);
                kernel.sampler_mut().sample_mut().load_sample(&[0.8; 44100], 44100);
                kernel.params.random_lfos[0].params.frequency = PolyF32::splat(20.0);
                kernel.params.random_lfos[0].params.style = RandomLfoStyle::SampleAndHold;
                kernel.params.random_lfos[0].sync = sync;
                kernel.matrix.connections.push(Connection {
                    source: ModSource::RandomLfo(0),
                    dest: ModDest::SampleLevel,
                    transform: ModulationTransform::with_amount(1.0, 0.8),
                });
            }
            allocator.note_on(60, 1.0, 0, 0);
            let audio = render_blocks(&mut allocator, 200);

            // The level offset is applied once per control block, so holds
            // land on block boundaries; sample-and-hold steps show up as
            // jumps in the per-block mean amplitude.
            let block_mean: Vec<f32> = audio[MAX_BUFFER_SIZE * 4..]
                .chunks(MAX_BUFFER_SIZE)
                .map(|c| c.iter().map(|v| v.abs()).sum::<f32>() / c.len() as f32)
                .collect();
            let mut max_ratio = 1.0f32;
            for pair in block_mean.windows(2) {
                let (a, b) = (pair[0].max(1e-6), pair[1].max(1e-6));
                max_ratio = max_ratio.max((a / b).max(b / a));
            }
            max_ratio
        };

        // Free-running 20 Hz sample-and-hold: the level steps every ~17
        // blocks (~11 steps over the render).
        let free = max_adjacent_block_ratio(LfoSync::default());
        assert!(free > 1.2, "free random level modulation shows no steps: {free}");
        // Tempo index 0 is Freeze (0 Hz): the free 20 Hz must be ignored.
        let frozen = max_adjacent_block_ratio(LfoSync { mode: SyncMode::Tempo, tempo_index: 0.0 });
        assert!(frozen < 1.05, "frozen random LFO still modulates: {frozen}");
    }

    #[test]
    fn macro_to_filter_drive_changes_output() {
        let render = |macro_value: f32| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                kernel.params.oscillators[0].params.wave_frame = PolyF32::splat(128.0);
                kernel.params.filters[0].params.on = true;
                kernel.params.filters[0].params.state.midi_cutoff = PolyF32::splat(100.0);
                kernel.params.macros[0] = macro_value;
                kernel.matrix.connections.push(Connection {
                    source: ModSource::Macro(0),
                    dest: ModDest::FilterDrive(0),
                    transform: ModulationTransform::with_amount(1.0, 20.0),
                });
            }
            allocator.note_on(60, 1.0, 0, 0);
            render_blocks(&mut allocator, 12)
        };

        let plain = render(0.0);
        let driven = render(1.0);
        let rms = |audio: &[f32]| {
            (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt()
        };
        // +20 dB of filter drive must audibly change the level (the drive
        // offset was previously declared but never applied). The Analog
        // model's drive saturates, so the level moves either way.
        let (plain_rms, driven_rms) = (rms(&plain[512..]), rms(&driven[512..]));
        assert!(plain_rms > 0.0 && driven_rms > 0.0);
        let ratio = (plain_rms / driven_rms).max(driven_rms / plain_rms);
        assert!(
            ratio > 1.2,
            "filter drive modulation had no effect: {plain_rms} vs {driven_rms}"
        );
    }

    #[test]
    fn macro_to_env_attack_power_changes_attack_shape() {
        let render = |macro_value: f32| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                // Slow attack so the curve shape is visible mid-attack.
                kernel.params.envelopes[0].attack = PolyF32::splat(0.5);
                kernel.params.macros[0] = macro_value;
                kernel.matrix.connections.push(Connection {
                    source: ModSource::Macro(0),
                    dest: ModDest::EnvAttackPower(0),
                    transform: ModulationTransform::with_amount(1.0, 10.0),
                });
            }
            allocator.note_on(60, 1.0, 0, 0);
            render_blocks(&mut allocator, 40)
        };

        let linear = render(0.0);
        let curved = render(10.0 / 10.0);
        let mid_rms = |audio: &[f32]| {
            let mid = &audio[audio.len() / 3..audio.len() / 2];
            (mid.iter().map(|v| v * v).sum::<f32>() / mid.len() as f32).sqrt()
        };
        let (a, b) = (mid_rms(&linear), mid_rms(&curved));
        assert!(a > 0.0 && b > 0.0);
        assert!(
            (a / b).max(b / a) > 1.1,
            "attack power modulation had no effect mid-attack: {a} vs {b}"
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

    // -- KERNEL V2: switchable engines, noise, raised limits ----------------

    fn constant_sample_arc(value: f32, length: usize) -> Arc<Sample> {
        Arc::new(Sample::from_mono("const", &vec![value; length], 44100))
    }

    fn peak(audio: &[f32]) -> f32 {
        audio.iter().fold(0.0f32, |a, &v| a.max(v.abs()))
    }

    /// Mean over an early window: DC from a constant sample survives the
    /// ~1 s DC blocker there, while a wavetable render stays zero-mean.
    fn early_mean(audio: &[f32]) -> f32 {
        let window = &audio[256..2048.min(audio.len())];
        window.iter().sum::<f32>() / window.len() as f32
    }

    /// Wavetable vs Sample vs Granular on slot 0 with a loaded constant
    /// sample: all three sound, and the sample-reading engines carry the
    /// source's DC character that the wavetable does not.
    #[test]
    fn slot_engines_produce_sound_with_distinct_character() {
        let sample = constant_sample_arc(0.8, 44100);
        let render_engine = |engine: OscEngineKind| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                kernel.set_sample(0, sample.clone());
                let slot = &mut kernel.params.oscillators[0];
                slot.engine = engine;
                slot.sample_params.loop_sample = true;
                slot.granular_params.position = PolyF32::splat(0.25);
                slot.granular_params.density = PolyF32::splat(110.0);
            }
            allocator.note_on(60, 1.0, 0, 0);
            render_blocks(&mut allocator, 20)
        };

        let wavetable = render_engine(OscEngineKind::Wavetable);
        let sampled = render_engine(OscEngineKind::Sample);
        let granular = render_engine(OscEngineKind::Granular);
        for audio in [&wavetable, &sampled, &granular] {
            assert!(audio.iter().all(|v| v.is_finite()));
        }

        assert!(peak(&wavetable) > 0.01, "wavetable engine silent");
        assert!(peak(&sampled) > 0.01, "sample engine silent");
        assert!(peak(&granular) > 0.01, "granular engine silent");

        let wavetable_mean = early_mean(&wavetable);
        let sample_mean = early_mean(&sampled);
        let granular_mean = early_mean(&granular);
        assert!(
            wavetable_mean.abs() < 0.05,
            "wavetable render should be zero-mean: {wavetable_mean}"
        );
        assert!(sample_mean > 0.1, "sample engine lost the source DC: {sample_mean}");
        assert!(granular_mean > 0.02, "granular engine lost the source DC: {granular_mean}");
    }

    #[test]
    fn engine_switch_mid_note_is_safe_and_next_note_uses_new_engine() {
        let sample = constant_sample_arc(0.8, 44100);
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.set_sample(0, sample.clone());
            kernel.params.oscillators[0].sample_params.loop_sample = true;
        }

        allocator.note_on(60, 1.0, 0, 0);
        let wavetable_audio = render_blocks(&mut allocator, 8);
        assert!(peak(&wavetable_audio) > 0.01);

        // Switch the sounding slot to the Sample engine mid-note: no panic,
        // finite output, and the engine change is effective immediately.
        for kernel in allocator.kernels_mut() {
            kernel.params.oscillators[0].engine = OscEngineKind::Sample;
        }
        let switched = render_blocks(&mut allocator, 8);
        assert!(switched.iter().all(|v| v.is_finite()));

        allocator.note_off(60, 0.5, 0, 0);
        let _ = render_blocks(&mut allocator, 12);
        assert_eq!(allocator.num_active_voices(), 0);

        // The next note renders through the Sample engine: the constant
        // source's DC shows up where the wavetable was zero-mean.
        allocator.note_on(60, 1.0, 0, 0);
        let sample_audio = render_blocks(&mut allocator, 8);
        assert!(peak(&sample_audio) > 0.01);
        assert!(early_mean(&wavetable_audio).abs() < 0.05);
        assert!(
            early_mean(&sample_audio) > 0.1,
            "next note did not use the sample engine: mean {}",
            early_mean(&sample_audio)
        );
    }

    #[test]
    fn multisample_engine_plays_zones() {
        let sfz = "<region> sample=const.wav lokey=0 hikey=127 pitch_keycenter=60 \
                   loop_mode=loop_continuous";
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            let multisample = Multisample::from_sfz(sfz, |_| {
                Some(Sample::from_mono("const", &vec![0.6; 8192], 44100))
            })
            .unwrap();
            kernel.set_multisample(0, multisample);
            kernel.params.oscillators[0].engine = OscEngineKind::Multisample;
        }
        allocator.note_on(60, 1.0, 0, 0);
        let audio = render_blocks(&mut allocator, 12);
        assert!(audio.iter().all(|v| v.is_finite()));
        assert!(
            early_mean(&audio) > 0.05,
            "multisample zone did not sound: mean {}",
            early_mean(&audio)
        );
    }

    #[test]
    fn noise_section_routes_and_sounds() {
        let render_noise = |on: bool| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                kernel.params.oscillators[0].on = false;
                kernel.params.noise.on = on;
                kernel.params.noise.destination = ProducerDestination::Effects;
            }
            allocator.note_on(60, 1.0, 0, 0);
            render_blocks(&mut allocator, 8)
        };

        let noisy = render_noise(true);
        assert!(noisy.iter().all(|v| v.is_finite()));
        let rms = (noisy[512..].iter().map(|v| v * v).sum::<f32>()
            / (noisy.len() - 512) as f32)
            .sqrt();
        assert!(rms > 0.02, "noise section is silent: rms {rms}");

        let silent = render_noise(false);
        assert!(peak(&silent) < 1e-6, "noise leaked while off: {}", peak(&silent));
    }

    #[test]
    fn fourth_oscillator_slot_works_like_the_others() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.oscillators[0].on = false;
            kernel.params.oscillators[NUM_OSCILLATORS - 1].on = true;
        }
        allocator.note_on(60, 1.0, 0, 0);
        let audio = render_blocks(&mut allocator, 8);
        assert!(audio.iter().all(|v| v.is_finite()));
        assert!(peak(&audio) > 0.01, "4th oscillator slot is silent");
    }

    #[test]
    fn twelfth_lfo_modulates_cutoff_through_the_matrix() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.oscillators[0].params.wave_frame = PolyF32::splat(128.0);
            kernel.params.filters[0].params.on = true;
            kernel.params.filters[0].params.state.midi_cutoff = PolyF32::splat(60.0);
            kernel.params.lfos[NUM_LFOS - 1].params.frequency = PolyF32::splat(8.0);
            kernel.matrix.connections.push(Connection {
                source: ModSource::Lfo(NUM_LFOS - 1),
                dest: ModDest::FilterCutoff(0),
                transform: ModulationTransform::with_amount(1.0, 60.0),
            });
        }
        allocator.note_on(48, 1.0, 0, 0);
        let audio = render_blocks(&mut allocator, 40);

        let block_rms: Vec<f32> = audio
            .chunks(512)
            .map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt())
            .collect();
        let max = block_rms[2..].iter().cloned().fold(0.0f32, f32::max);
        let min = block_rms[2..].iter().cloned().fold(f32::MAX, f32::min);
        assert!(max > 0.0);
        assert!(
            max / min.max(1e-9) > 1.05,
            "12th LFO cutoff modulation had no audible effect: {min}..{max}"
        );
    }

    #[test]
    fn eighth_envelope_modulates_osc_level_through_the_matrix() {
        let render_with = |connect: bool| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                kernel.params.envelopes[NUM_ENVELOPES - 1] = EnvelopeParams {
                    attack: PolyF32::splat(0.001),
                    sustain: PolyF32::ONE,
                    ..Default::default()
                };
                if connect {
                    kernel.matrix.connections.push(Connection {
                        source: ModSource::Envelope(NUM_ENVELOPES - 1),
                        dest: ModDest::OscLevel(0),
                        transform: ModulationTransform::with_amount(1.0, -1.0),
                    });
                }
            }
            allocator.note_on(60, 1.0, 0, 0);
            render_blocks(&mut allocator, 8)
        };

        let plain = render_with(false);
        assert!(peak(&plain) > 0.01);
        // The 8th envelope at full sustain drives the level offset to -1,
        // clamping the oscillator amplitude to zero.
        let muted = render_with(true);
        assert!(
            peak(&muted[512..]) < 1e-3,
            "8th envelope level modulation had no effect: {}",
            peak(&muted[512..])
        );
    }
}

