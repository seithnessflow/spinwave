//! The complete synth voice kernel: 4 switchable-engine oscillator slots
//! (wavetable / sample / granular / multisample, Serum-2 style) + legacy
//! sampler + noise source, two switchable filters, 8 envelopes, 12 LFOs,
//! 4 random LFOs and the modulation matrix, statically wired (rework of
//! `SynthVoiceHandler` + `ProducersModule` + `FiltersModule`).

use std::sync::{Arc, OnceLock};

use spinwave_dsp::filters::filter_state;
use spinwave_dsp::filters::DcFilter;
use spinwave_dsp::modulators::{
    Envelope, EnvelopeParams, LfoSyncType, LineGenerator, RandomLfo, RandomLfoParams, SynthLfo,
    SynthLfoParams, TriggerRandom,
};
use spinwave_dsp::oscillator::noise::{NoiseParams, NoiseSource};
use spinwave_dsp::oscillator::{
    Granular, GranularParams, Multisample, MultisampleSource, Sample, SampleSource,
    SampleSourceParams, SynthOscillator, SynthOscillatorParams,
};
use spinwave_dsp::utilities::{PortamentoParams, PortamentoSlope};
use spinwave_dsp::wavetable::Wavetable;
use spinwave_poly::constants::{MAX_BUFFER_SIZE, VoiceEvent};
use spinwave_poly::{PolyF32, PolyMask, LANES};

use crate::allocator::VoiceKernel;
use crate::kernel::mod_matrix::{
    AudioRateSources, AudioSourceBuffers, ModMatrix, ModOffsets, SourceValues, NUM_ENVELOPES,
    NUM_LFOS, NUM_MACROS, NUM_OSCILLATORS, NUM_RANDOM_LFOS,
};
use crate::kernel::voice_filter::{VoiceFilter, VoiceFilterParams};
use crate::tempo::LfoSync;
use crate::voice::VoiceControls;

const MAX_BLOCK: usize = MAX_BUFFER_SIZE * 8;
/// MIDI note the filter keytrack is centred on (`kMidiTrackCenter`).
const MIDI_TRACK_CENTER: f32 = 60.0;
/// Default `portamento_time` (2^-10 s, the table's minimum): below the
/// slope's 1 ms threshold, so glide is off.
const DEFAULT_PORTAMENTO_TIME: f32 = 1.0 / 1024.0;

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
    /// Per-voice amplitude multiplier `[0, 1]` (table `voice_amplitude`,
    /// default 1). The amplitude law is
    /// `(env × interp(1, velocity, velocity_track) × voice_amplitude)²`
    /// with the control part smoothed across the block (reference
    /// `SmoothMultiply` + `Square`).
    pub voice_amplitude: f32,
    /// Pitch wheel range in semitones.
    pub pitch_bend_range: f32,
    /// Portamento glide time in SECONDS (table `portamento_time` stores
    /// log2 seconds: engine value = `2^stored`); at or below 1 ms the glide
    /// is off. Default 2^-10.
    pub portamento_time: f32,
    /// Glide curve power (table `portamento_slope`, default 0 = linear).
    pub portamento_slope: f32,
    /// Always glide (`portamento_force`); when false only glide while other
    /// notes are held (auto mode). Default false.
    pub portamento_force: bool,
    /// Scale the glide time by the interval size (`portamento_scale`).
    pub portamento_scale: bool,
    /// Global transpose in semitones (table `voice_transpose`, default 0).
    pub voice_transpose: f32,
    /// Global fine tune in semitones (table `voice_tune` stores `[-1, 1]`
    /// semitones, displayed ×100 as cents; default 0).
    pub voice_tune: f32,
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
            voice_amplitude: 1.0,
            pitch_bend_range: 2.0,
            portamento_time: DEFAULT_PORTAMENTO_TIME,
            portamento_slope: 0.0,
            portamento_force: false,
            portamento_scale: false,
            voice_transpose: 0.0,
            voice_tune: 0.0,
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
    /// Note glide between the previous and the current note
    /// (`PortamentoSlope`), reset by every voice event.
    portamento: PortamentoSlope,
    /// Bent MIDI note of the current block (portamento + bends + voice
    /// tune/transpose): pitch of every producer, `note` source, keytrack.
    bent_midi: PolyF32,
    /// Smoothed control part of the amplitude law (`SmoothMultiply` state).
    amp_control: PolyF32,
    /// Last block's control-rate cutoff target per filter, start of the
    /// per-sample ramp.
    cutoff_state: [PolyF32; 2],
    /// Host transport position in seconds (`correct_to_time`).
    transport_seconds: f64,
    /// Transport-synced random LFO values shared by every voice (the
    /// reference's `shared_state_`), pushed by the engine each block.
    shared_random: [PolyF32; NUM_RANDOM_LFOS],
    shared_random_valid: bool,
    /// Which envelopes / LFOs feed audio-rate destinations this block.
    audio_rate: AudioRateSources,

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

    /// DC blockers on the two voice output buses.
    ///
    /// A Spinwave addition: the reference ships `dc_filter.{h,cpp}` and
    /// wires it NOWHERE, so its voices pass whatever offset an asymmetric
    /// waveform or a phase distortion leaves behind. Blocking it is the
    /// better synth, and the golden bench measured what it costs in
    /// fidelity: with these on, every filter case carries a slowly
    /// decaying offset the reference keeps, which is 99.7% of the residual
    /// on those cases. The bench pins them off (see `set_dc_blockers`) so
    /// it compares the DSP the two engines share.
    dc_filter: DcFilter,
    direct_dc_filter: DcFilter,
    /// Whether the two above run. True everywhere except the bench.
    dc_blockers_enabled: bool,

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
    /// Audio-rate envelope outputs; index 0 is the amplitude envelope
    /// (always audio rate), the others only when flagged by `audio_rate`.
    env_audio: [Vec<PolyF32>; NUM_ENVELOPES],
    /// Audio-rate LFO outputs, only valid when flagged by `audio_rate`.
    lfo_audio: [Vec<PolyF32>; NUM_LFOS],
    /// Audio-rate modulation contributions to each filter cutoff.
    cutoff_audio: [Vec<PolyF32>; 2],
    /// Final per-sample MIDI cutoff handed to each filter.
    pub(crate) cutoff_buffer: [Vec<PolyF32>; 2],
    mod_scratch: Vec<PolyF32>,
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
            wavetables: core::array::from_fn(|_| default_wavetable()),
            portamento: PortamentoSlope::new(sr),
            bent_midi: PolyF32::ZERO,
            amp_control: PolyF32::ZERO,
            cutoff_state: [PolyF32::ZERO; 2],
            transport_seconds: 0.0,
            shared_random: [PolyF32::ZERO; NUM_RANDOM_LFOS],
            shared_random_valid: false,
            audio_rate: AudioRateSources::default(),
            oscillators: core::array::from_fn(|_| SynthOscillator::new()),
            slot_samplers: core::array::from_fn(|_| {
                let mut source = SampleSource::with_sample(Arc::new(empty_sample()));
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
            dc_blockers_enabled: true,
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
            env_audio: core::array::from_fn(|_| vec![PolyF32::ZERO; MAX_BLOCK]),
            lfo_audio: core::array::from_fn(|_| vec![PolyF32::ZERO; MAX_BLOCK]),
            cutoff_audio: core::array::from_fn(|_| vec![PolyF32::ZERO; MAX_BLOCK]),
            cutoff_buffer: core::array::from_fn(|_| vec![PolyF32::ZERO; MAX_BLOCK]),
            mod_scratch: vec![PolyF32::ZERO; MAX_BLOCK],
            output: vec![PolyF32::ZERO; MAX_BLOCK],
            direct_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            direct_out: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_a_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_a_out: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_b_bus: vec![PolyF32::ZERO; MAX_BLOCK],
            bus_b_out: vec![PolyF32::ZERO; MAX_BLOCK],
        }
    }

    /// Installs a wavetable and returns the previous handle so its last
    /// drop can happen off the audio thread. RT-safe.
    pub fn set_wavetable(&mut self, index: usize, wavetable: Arc<Wavetable>) -> Arc<Wavetable> {
        std::mem::replace(&mut self.wavetables[index], wavetable)
    }

    /// Installs the sample material for one oscillator slot, shared by the
    /// Sample and Granular engines: both read the same `Arc` (the
    /// band-limited pyramid is built once, by whoever created the sample).
    /// Returns the previous handle so its last drop can happen off the
    /// audio thread. RT-safe.
    pub fn set_sample(&mut self, slot: usize, sample: Arc<Sample>) -> Arc<Sample> {
        let previous_source = self.slot_samplers[slot].set_sample(sample.clone());
        let previous = std::mem::replace(&mut self.slot_samples[slot], sample);
        // Both handles pointed at the same material; returning one keeps
        // the pyramid alive until the caller drops it.
        drop(previous_source);
        previous
    }

    /// Sample material of one slot (as installed by [`Self::set_sample`]).
    pub fn slot_sample(&self, slot: usize) -> &Arc<Sample> {
        &self.slot_samples[slot]
    }

    /// Installs the multisample instrument for one oscillator slot. The
    /// per-zone playback state is private to the kernel, so a
    /// `MultisampleSource` is built here from a (cheaply cloned)
    /// `Multisample`; the zone material stays shared. This allocates the
    /// zone list: call it at patch-load time, off the audio thread when
    /// possible, and drop the returned previous source off-thread.
    pub fn set_multisample(&mut self, slot: usize, multisample: Multisample) -> MultisampleSource {
        let mut source = MultisampleSource::new(multisample);
        source.set_sample_rate(self.sample_rate as f32);
        std::mem::replace(&mut self.slot_multisamples[slot], source)
    }

    /// Installs a prebuilt multisample source (built off the audio thread
    /// with [`SynthVoiceKernel::build_multisample_source`]) and returns the
    /// previous one. RT-safe.
    pub fn install_multisample_source(
        &mut self,
        slot: usize,
        mut source: MultisampleSource,
    ) -> MultisampleSource {
        // The builder may not know this kernel's (oversampled) rate.
        source.set_sample_rate(self.sample_rate as f32);
        std::mem::replace(&mut self.slot_multisamples[slot], source)
    }

    /// Builds a multisample source at this kernel's sample rate without
    /// installing it (for off-thread preparation).
    pub fn build_multisample_source(&self, multisample: Multisample) -> MultisampleSource {
        let mut source = MultisampleSource::new(multisample);
        source.set_sample_rate(self.sample_rate as f32);
        source
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

    /// Sample rate this kernel runs at (the engine rate).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Control-rate modulation offsets of the last processed block.
    pub fn last_offsets(&self) -> &ModOffsets {
        &self.offsets
    }

    /// Phase the given LFO output last (per lane), for transport-sync
    /// checks and displays.
    pub fn lfo_phase(&self, index: usize) -> PolyF32 {
        self.lfos[index].phase()
    }

    /// Bent MIDI note of the last processed block per lane: portamento
    /// glide, pitch bends, `voice_tune` and `voice_transpose` applied.
    pub fn current_midi(&self) -> PolyF32 {
        self.bent_midi
    }

    /// Host transport position in seconds for the next block
    /// (`SynthVoiceHandler::correctToTime`): transport-synced LFOs snap
    /// their phase to it on trigger.
    /// Turns the per-voice DC blockers off. They are a Spinwave addition
    /// the reference does not have (see the field), so the golden bench
    /// pins them off to compare the shared DSP path. Nothing else should
    /// call this: a synth that lets DC through is worse.
    pub fn set_dc_blockers(&mut self, enabled: bool) {
        self.dc_blockers_enabled = enabled;
        self.dc_filter.hard_reset();
        self.direct_dc_filter.hard_reset();
    }

    pub fn set_transport(&mut self, seconds: f64) {
        self.transport_seconds = seconds;
    }

    /// Installs the transport-synced random LFO values every voice must
    /// share this block (the reference's `shared_state_`: synced random
    /// LFOs output one value for all voices). Random LFOs whose
    /// `params.sync` is on read these instead of their own instance.
    pub fn set_shared_random_values(&mut self, values: [PolyF32; NUM_RANDOM_LFOS]) {
        self.shared_random = values;
        self.shared_random_valid = true;
    }

    fn dispatch_triggers(&mut self, controls: &VoiceControls) {
        // Every voice event (including legato-suppressed retriggers) resets
        // the glide, like the reference plugging `voice_event` into the
        // slope's reset.
        let event = &controls.voice_event;
        if event.mask.any() {
            self.portamento.trigger(event.mask, event.value, 0);
        }

        let retrigger = &controls.retrigger;
        if !retrigger.mask.any() {
            return;
        }

        let event_value = retrigger.value;
        let offsets = retrigger.offset;
        let on_mask = retrigger.mask
            & event_value.eq(PolyF32::splat(VoiceEvent::On.as_f32()));

        // Per-lane offsets: the two voices of the pair may start at
        // different samples of this block.
        for envelope in &mut self.envelopes {
            envelope.trigger_at(retrigger.mask, event_value, offsets);
        }
        for lfo in &mut self.lfos {
            lfo.trigger_at(retrigger.mask, event_value, offsets);
        }
        for random in &mut self.random_lfos {
            random.trigger_at(retrigger.mask, event_value, offsets);
        }
        self.trigger_random.trigger_at(retrigger.mask, event_value, offsets);

        if on_mask.any() {
            // Producers only restart on `reset` (voice rising from Dead);
            // a stolen / legato-retriggered voice keeps its phase and only
            // snaps its pitch ramps (`SynthOscillator::retrigger`), like
            // the reference's kReset / kRetrigger inputs. The sampler
            // likewise resets only from Dead (SampleModule::kReset).
            let reset_on = controls.reset.mask & on_mask;
            let retrigger_on = on_mask & !controls.reset.mask;
            if reset_on.any() {
                for oscillator in &mut self.oscillators {
                    oscillator.note_on(reset_on, offsets);
                }
                for sampler in &mut self.slot_samplers {
                    sampler.note_on(reset_on, offsets);
                }
                self.sampler.note_on(reset_on, offsets);
            }
            if retrigger_on.any() {
                for oscillator in &mut self.oscillators {
                    oscillator.retrigger(retrigger_on);
                }
            }
            // Spinwave extensions: grains restart and multisample zones
            // are reselected on every note-on (zone selection needs the
            // new per-voice note and velocity).
            for granular in &mut self.slot_granulars {
                granular.note_on(on_mask, offsets);
            }
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
                    multisample.note_on(mask, offsets, note, velocity);
                }
            }
        }
    }

    /// Bent MIDI note for the block (`synth_voice_handler.cpp`
    /// `createNoteArticulation`): portamento glide from the last note to
    /// the current one, then pitch wheel × range, local (MPE) bend, the
    /// `pitch_wheel` modulation offset, `voice_tune` and `voice_transpose`.
    fn compute_bent_midi(&mut self, controls: &VoiceControls, num_samples: usize) -> PolyF32 {
        let params = &self.params;
        let glide = PortamentoParams {
            target: controls.note.value,
            source: controls.last_note.value,
            run_seconds: PolyF32::splat(params.portamento_time),
            slope_power: PolyF32::splat(params.portamento_slope),
            num_notes_pressed: controls.note_pressed,
            force: params.portamento_force,
            scale: params.portamento_scale,
        };
        let glided = self.portamento.process(&glide, num_samples);
        glided
            + controls.local_pitch_bend
            + controls.pitch_wheel * params.pitch_bend_range
            + self.offsets.pitch_bend
            + params.voice_tune
            + params.voice_transpose
    }

    /// Computes all modulator values for the block: envelopes / LFOs that
    /// feed an audio-rate destination render sample by sample into their
    /// buffers (their last sample is the control value), the rest tick at
    /// control rate.
    fn update_modulators(&mut self, controls: &VoiceControls, num_samples: usize) {
        let audio_rate = self.audio_rate;

        // Envelope 0 always runs at audio rate (amplitude + voice killer).
        let params = self.resolved_env_params(0);
        self.envelopes[0].process_audio(&params, &mut self.env_audio[0][..num_samples]);
        self.sources.envelopes[0] = self.envelopes[0].value();
        for i in 1..NUM_ENVELOPES {
            let params = self.resolved_env_params(i);
            if audio_rate.envelope(i) {
                self.envelopes[i].process_audio(&params, &mut self.env_audio[i][..num_samples]);
                self.sources.envelopes[i] = self.envelopes[i].value();
            } else {
                self.sources.envelopes[i] =
                    self.envelopes[i].process_control(&params, num_samples);
            }
        }

        let beats_per_second = self.params.beats_per_second;
        let transport_seconds = self.transport_seconds;
        for i in 0..NUM_LFOS {
            let section = &self.params.lfos[i];
            let mut params = section.params;
            params.frequency = section.sync.resolve(
                params.frequency + self.offsets.lfo_frequency[i],
                beats_per_second,
            );
            // The LFO wraps its phase internally, so the offset adds raw.
            params.phase += self.offsets.lfo_phase[i];
            if params.sync_type == LfoSyncType::Sync {
                self.lfos[i].correct_to_time(transport_seconds);
            }
            // The LFO already outputs the unipolar shape value in [0, 1]
            // (the matrix recentres bipolar connections itself).
            self.sources.lfos[i] = if audio_rate.lfo(i) {
                self.lfos[i].process_audio(
                    &section.shape,
                    &params,
                    &mut self.lfo_audio[i][..num_samples],
                );
                self.lfos[i].value()
            } else {
                self.lfos[i].process_control(&section.shape, &params, num_samples)
            };
        }

        for i in 0..NUM_RANDOM_LFOS {
            let section = &self.params.random_lfos[i];
            let mut params = section.params;
            params.frequency = section.sync.resolve(
                params.frequency + self.offsets.random_lfo_frequency[i],
                beats_per_second,
            );
            self.random_lfos[i].correct_to_time(transport_seconds);
            // Unipolar [0, 1] already. The per-voice instance always ticks
            // (consumes its trigger, keeps state coherent); in sync mode
            // the engine's shared value wins so every voice agrees.
            let own = self.random_lfos[i].process_control(&params, num_samples);
            self.sources.random_lfos[i] = if params.sync && self.shared_random_valid {
                self.shared_random[i]
            } else {
                own
            };
        }

        self.sources.macros = self.params.macros.map(PolyF32::splat);
        // `note` is the BENT midi (`note_percentage_` reads `bent_midi_`).
        self.sources.note = self.bent_midi * (1.0 / 127.0);
        self.sources.note_in_octave = controls.note_in_octave;
        self.sources.velocity = controls.velocity.value;
        self.sources.lift = controls.lift.value;
        self.sources.mod_wheel = controls.mod_wheel;
        self.sources.pitch_wheel = controls.pitch_wheel_percent;
        self.sources.aftertouch = controls.aftertouch.value;
        self.sources.slide = controls.slide.value;
        // TriggerRandom draws in [0, 1) already.
        self.sources.random = self.trigger_random.value();
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

    fn run_producers(&mut self, num_samples: usize) {
        self.filter1_bus[..num_samples].fill(PolyF32::ZERO);
        self.filter2_bus[..num_samples].fill(PolyF32::ZERO);
        self.effects_bus[..num_samples].fill(PolyF32::ZERO);
        self.direct_bus[..num_samples].fill(PolyF32::ZERO);
        self.bus_a_bus[..num_samples].fill(PolyF32::ZERO);
        self.bus_b_bus[..num_samples].fill(PolyF32::ZERO);

        let midi = self.bent_midi;

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
            let common = self.slot_offsets(i);
            match section.engine {
                OscEngineKind::Wavetable => {
                    let params = self.modulated_wavetable_params(i, midi, &common);

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
                    let params = modulated_sample_params(&section.sample_params, midi, &common);
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
                    params.level = (params.level + common.level).clamp(0.0, 1.0);
                    params.transpose += common.transpose;
                    params.tune += common.tune;
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
                    let params = modulated_sample_params(&section.sample_params, midi, &common);
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
            self.route_leveled(destination, num_samples);
        }

        if self.params.sample.on {
            let common = CommonOffsets {
                level: self.offsets.sample_level,
                transpose: self.offsets.sample_transpose,
                tune: self.offsets.sample_tune,
                pan: self.offsets.sample_pan,
            };
            let params = modulated_sample_params(&self.params.sample.params, midi, &common);
            let raw = &mut self.serial_bus; // reuse as sampler raw scratch
            self.sampler.process(
                &params,
                num_samples,
                &mut raw[..num_samples],
                &mut self.leveled[..num_samples],
            );
            self.route_leveled(self.params.sample.destination, num_samples);
        }

        if self.params.noise.on {
            let params = self.params.noise.params;
            let destination = self.params.noise.destination;
            self.noise.process(&params, num_samples, &mut self.leveled[..num_samples]);
            self.route_leveled(destination, num_samples);
        }
    }

    /// The modulation offsets every slot engine shares (level / pitch /
    /// pan), for oscillator slot `i`.
    fn slot_offsets(&self, i: usize) -> CommonOffsets {
        CommonOffsets {
            level: self.offsets.osc_level[i],
            transpose: self.offsets.osc_transpose[i],
            tune: self.offsets.osc_tune[i],
            pan: self.offsets.osc_pan[i],
        }
    }

    /// Wavetable engine params of slot `i` with every offset applied.
    fn modulated_wavetable_params(
        &self,
        i: usize,
        midi: PolyF32,
        common: &CommonOffsets,
    ) -> SynthOscillatorParams {
        let offsets = &self.offsets;
        let mut params = self.params.oscillators[i].params.clone();
        params.midi_note = midi;
        params.amplitude = (params.amplitude + common.level).clamp(0.0, 1.0);
        params.transpose += common.transpose;
        params.tune += common.tune;
        params.pan = (params.pan + common.pan).clamp(-1.0, 1.0);
        params.wave_frame += offsets.osc_frame[i];
        params.frame_spread += offsets.osc_frame_spread[i];
        // `unison_detune` is a Quadratic parameter (stored 0..10): the
        // reference squares the modulated sum (`cr::Square` after the
        // modulation total) before `cents = range * detune`.
        let detune = (params.unison_detune + offsets.osc_unison_detune[i]).clamp(0.0, 10.0);
        params.unison_detune = detune * detune;
        params.blend = (params.blend + offsets.osc_unison_blend[i]).clamp(0.0, 1.0);
        params.stereo_spread = (params.stereo_spread + offsets.osc_stereo_spread[i]).clamp(0.0, 1.0);
        params.distortion_amount =
            (params.distortion_amount + offsets.osc_distortion_amount[i]).clamp(0.0, 1.0);
        params.distortion_phase =
            (params.distortion_phase + offsets.osc_distortion_phase[i]).clamp(0.0, 1.0);
        params.spectral_morph_amount =
            (params.spectral_morph_amount + offsets.osc_spectral_morph_amount[i]).clamp(0.0, 1.0);
        params.phase = (params.phase + offsets.osc_phase[i]).fract();
        params
    }

    /// Routes the `leveled` scratch (the producer just rendered) into the
    /// buses per its destination and the filters' on/off state.
    fn route_leveled(&mut self, destination: ProducerDestination, num_samples: usize) {
        // Producers whose destination filters are all off bypass to the
        // raw (effects) bus, per producer (producers_module.cpp).
        let filters_on = [self.params.filters[0].params.on, self.params.filters[1].params.on];
        route(
            destination,
            &self.leveled[..num_samples],
            filters_on,
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

    fn run_filters(&mut self, num_samples: usize, reset_mask: PolyMask) {
        // Keytrack follows the BENT midi (FiltersModule::kMidi ← bent_midi_).
        let note = self.bent_midi;
        let mut filter_params: [VoiceFilterParams; 2] = [
            self.params.filters[0].params,
            self.params.filters[1].params,
        ];
        for (i, params) in filter_params.iter_mut().enumerate() {
            let keytrack_amount = (PolyF32::splat(self.params.filters[i].keytrack)
                + self.offsets.filter_keytrack[i])
                .clamp(-1.0, 1.0);
            let keytrack = (note - MIDI_TRACK_CENTER) * keytrack_amount;
            // Block target of the cutoff (control-rate part); the filters
            // consume a per-sample buffer ramping from last block's target
            // to this one, plus the audio-rate modulation contributions
            // (reference: audio-rate `midi_cutoff` control + SmoothValue).
            let target = params.state.midi_cutoff + keytrack + self.offsets.filter_cutoff[i];
            params.state.midi_cutoff = target;
            let start = reset_mask.select(target, self.cutoff_state[i]);
            self.cutoff_state[i] = target;
            ramp_with_audio(
                start,
                target,
                &self.cutoff_audio[i][..num_samples],
                &mut self.cutoff_buffer[i][..num_samples],
            );
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

        // FiltersModule::process: serial only when the TARGET filter is on
        // (backward = filter 1 fed by filter 2's output, forward = filter 2
        // fed by filter 1's output); otherwise parallel. An off filter
        // outputs silence (its producers already bypassed to the raw bus).
        let routing = self.params.filter_routing;
        if routing == FilterRouting::SerialBackward && filter_params[0].on {
            self.filters[1].process_modulated(
                &filter_params[1],
                &self.cutoff_buffer[1][..num_samples],
                &self.filter2_bus[..num_samples],
                &mut self.filter2_out[..num_samples],
                reset_mask,
            );
            for i in 0..num_samples {
                self.serial_bus[i] = self.filter1_bus[i] + self.filter2_out[i];
            }
            self.filter2_out[..num_samples].fill(PolyF32::ZERO);
            self.filters[0].process_modulated(
                &filter_params[0],
                &self.cutoff_buffer[0][..num_samples],
                &self.serial_bus[..num_samples],
                &mut self.filter1_out[..num_samples],
                reset_mask,
            );
        } else if routing == FilterRouting::SerialForward && filter_params[1].on {
            self.filters[0].process_modulated(
                &filter_params[0],
                &self.cutoff_buffer[0][..num_samples],
                &self.filter1_bus[..num_samples],
                &mut self.filter1_out[..num_samples],
                reset_mask,
            );
            for i in 0..num_samples {
                self.serial_bus[i] = self.filter2_bus[i] + self.filter1_out[i];
            }
            self.filter1_out[..num_samples].fill(PolyF32::ZERO);
            self.filters[1].process_modulated(
                &filter_params[1],
                &self.cutoff_buffer[1][..num_samples],
                &self.serial_bus[..num_samples],
                &mut self.filter2_out[..num_samples],
                reset_mask,
            );
        } else {
            self.filters[0].process_modulated(
                &filter_params[0],
                &self.cutoff_buffer[0][..num_samples],
                &self.filter1_bus[..num_samples],
                &mut self.filter1_out[..num_samples],
                reset_mask,
            );
            self.filters[1].process_modulated(
                &filter_params[1],
                &self.cutoff_buffer[1][..num_samples],
                &self.filter2_bus[..num_samples],
                &mut self.filter2_out[..num_samples],
                reset_mask,
            );
        }
    }
}

/// Default table: the factory basic-shapes morph (sin → triangle → saw →
/// square → pulse), so wave-frame modulation works out of the box. Built
/// once and shared by every slot of every kernel.
fn default_wavetable() -> Arc<Wavetable> {
    static DEFAULT_WAVETABLE: OnceLock<Arc<Wavetable>> = OnceLock::new();
    DEFAULT_WAVETABLE
        .get_or_init(|| Arc::new(spinwave_dsp::wavetable::factory::basic_shapes()))
        .clone()
}

/// Per-sample control ramp:
/// `out[k] = start + (target - start) * (k + 1) / n + audio[k]`,
/// so the last sample lands exactly on `target` and the next block
/// continues from it without a step (the audio-rate contribution is added
/// on top).
fn ramp_with_audio(start: PolyF32, target: PolyF32, audio: &[PolyF32], out: &mut [PolyF32]) {
    let num_samples = out.len();
    debug_assert_eq!(num_samples, audio.len());
    let delta = (target - start) * (1.0 / num_samples as f32);
    let mut current = start;
    for (dest, &extra) in out.iter_mut().zip(audio) {
        current += delta;
        *dest = current + extra;
    }
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

/// Mask covering the two stereo lanes of one voice (voice 0 = lanes 0/1,
/// voice 1 = lanes 2/3), mirroring `Voice::mask`.
#[inline]
fn voice_lane_mask(voice: usize) -> PolyMask {
    PolyF32::from_lanes([0.0, 0.0, 1.0, 1.0]).eq(PolyF32::splat(voice as f32))
}

/// Modulation offsets shared by every producer engine: level, pitch and
/// pan (the fields each engine's own params expose).
#[derive(Default)]
struct CommonOffsets {
    level: PolyF32,
    transpose: PolyF32,
    tune: PolyF32,
    pan: PolyF32,
}

/// Sample-engine params (Sample, Multisample and the legacy sampler) with
/// the pitch and the common offsets applied.
fn modulated_sample_params(
    base: &SampleSourceParams,
    midi: PolyF32,
    common: &CommonOffsets,
) -> SampleSourceParams {
    let mut params = base.clone();
    params.midi = midi;
    params.level = (params.level + common.level).clamp(0.0, 1.0);
    params.transpose += common.transpose;
    params.tune += common.tune;
    params.pan = (params.pan + common.pan).clamp(-1.0, 1.0);
    params
}

struct ProducerBuses<'a> {
    filter1: &'a mut [PolyF32],
    filter2: &'a mut [PolyF32],
    effects: &'a mut [PolyF32],
    direct: &'a mut [PolyF32],
    bus_a: &'a mut [PolyF32],
    bus_b: &'a mut [PolyF32],
}

/// Routes one producer's levelled output (`ProducersModule::process`): the
/// signal goes raw (straight to the effects bus) when it targets the
/// effects, or when EVERY filter it targets is off — per producer, so a
/// producer aimed at an off filter is heard dry while another aimed at the
/// on filter is filtered.
fn route(
    destination: ProducerDestination,
    leveled: &[PolyF32],
    filters_on: [bool; 2],
    buses: &mut ProducerBuses,
) {
    let num_samples = leveled.len();
    let add_into = |bus: &mut [PolyF32]| {
        for i in 0..num_samples {
            bus[i] += leveled[i];
        }
    };
    let filter1 = destination.feeds_filter_1();
    let filter2 = destination.feeds_filter_2();
    let raw = destination == ProducerDestination::Effects
        || (filter1 && !filter2 && !filters_on[0])
        || (filter2 && !filter1 && !filters_on[1])
        || (filter1 && filter2 && !filters_on[0] && !filters_on[1]);
    if raw {
        add_into(buses.effects);
    }
    if filter1 {
        add_into(buses.filter1);
    }
    if filter2 {
        add_into(buses.filter2);
    }
    match destination {
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
        self.portamento.set_sample_rate(sr);
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
        // Pitch first (note articulation precedes the modulators in the
        // reference voice graph): the `note` source reads this block's
        // bent midi; the `pitch_wheel` modulation offset lags one block.
        self.bent_midi = self.compute_bent_midi(controls, num_samples);
        self.audio_rate = self.matrix.audio_rate_sources();
        self.update_modulators(controls, num_samples);

        // Control-rate connections resolve into per-block offsets (ramped
        // across the block where the destination is consumed per sample);
        // audio-rate connections render into the cutoff buffers.
        let sources = self.sources.clone();
        let mut offsets = std::mem::take(&mut self.offsets);
        self.matrix.resolve(&sources, &mut offsets, reset_mask);
        self.offsets = offsets;
        {
            let [cutoff_a, cutoff_b] = &mut self.cutoff_audio;
            self.matrix.resolve_audio(
                &AudioSourceBuffers { envelopes: &self.env_audio, lfos: &self.lfo_audio },
                num_samples,
                reset_mask,
                &mut self.mod_scratch,
                &mut [&mut cutoff_a[..], &mut cutoff_b[..]],
            );
        }

        self.run_producers(num_samples);
        self.run_filters(num_samples, reset_mask);

        // Amplitude law (createVoiceOutput): `Square(SmoothMultiply(env,
        // interp(1, velocity, velocity_track) × voice_amplitude))`, the
        // control part ramped linearly across the block (jumping on reset).
        // The modulatable `voice_amplitude` offset adds before squaring.
        // The direct-out bus is gated by the same voice amplitude
        // (`direct_output_`), and both buses pass a DC blocker.
        let velocity_scale = spinwave_poly::utils::interpolate(
            PolyF32::ONE,
            controls.velocity.value,
            PolyF32::splat(self.params.velocity_track),
        );
        let control_target = velocity_scale
            * (PolyF32::splat(self.params.voice_amplitude) + self.offsets.volume_amp)
                .max(PolyF32::ZERO);
        let mut control = reset_mask.select(control_target, self.amp_control);
        self.amp_control = control_target;
        let control_delta = (control_target - control) * (1.0 / num_samples as f32);
        for i in 0..num_samples {
            control += control_delta;
            let amp = self.env_audio[0][i] * control;
            let amplitude = amp * amp * controls.active_mask;
            let mixed = (self.filter1_out[i] + self.filter2_out[i] + self.effects_bus[i])
                * amplitude;
            let direct = self.direct_bus[i] * amplitude;
            if self.dc_blockers_enabled {
                self.output[i] = self.dc_filter.tick(mixed);
                self.direct_out[i] = self.direct_dc_filter.tick(direct);
            } else {
                self.output[i] = mixed;
                self.direct_out[i] = direct;
            }
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
        Some(&self.env_audio[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocator::{VoiceAllocator, VoiceOverride};
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
            kernel.sampler_mut().set_sample(constant_sample_arc(0.8, 8000));
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
                kernel.sampler_mut().set_sample(constant_sample_arc(0.8, 44100));
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

    // -- Review fixes -------------------------------------------------------

    fn rms(audio: &[f32]) -> f32 {
        (audio.iter().map(|v| v * v).sum::<f32>() / audio.len() as f32).sqrt()
    }

    /// Finding 1: the LFO source is unipolar [0, 1] as produced by the DSP
    /// (no extra remap), so an amount-1 connection spans the destination's
    /// full range and a bipolar one is symmetric around zero.
    #[test]
    fn lfo_source_spans_full_range_and_bipolar_is_symmetric() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.lfos[0].params.frequency = PolyF32::splat(20.0);
            kernel.matrix.connections.push(Connection {
                source: ModSource::Lfo(0),
                dest: ModDest::OscTune(0),
                transform: ModulationTransform::with_amount(1.0, 12.0),
            });
            let mut bipolar = ModulationTransform::with_amount(1.0, 12.0);
            bipolar.bipolar = true;
            kernel.matrix.connections.push(Connection {
                source: ModSource::Lfo(0),
                dest: ModDest::OscTune(1),
                transform: bipolar,
            });
        }
        allocator.note_on(60, 1.0, 0, 0);

        let (mut src_min, mut src_max) = (f32::MAX, f32::MIN);
        let (mut uni_min, mut uni_max) = (f32::MAX, f32::MIN);
        let (mut bi_min, mut bi_max) = (f32::MAX, f32::MIN);
        for _ in 0..200 {
            let _ = render_blocks(&mut allocator, 1);
            let kernel = &allocator.kernels()[0];
            let source = kernel.last_source_values().lfos[0].lane(0);
            src_min = src_min.min(source);
            src_max = src_max.max(source);
            let offsets = kernel.last_offsets();
            uni_min = uni_min.min(offsets.osc_tune[0].lane(0));
            uni_max = uni_max.max(offsets.osc_tune[0].lane(0));
            bi_min = bi_min.min(offsets.osc_tune[1].lane(0));
            bi_max = bi_max.max(offsets.osc_tune[1].lane(0));
        }
        assert!(src_min < 0.05 && src_max > 0.95, "LFO source range {src_min}..{src_max}");
        assert!(uni_min < 0.6 && uni_max > 11.4, "unipolar offset range {uni_min}..{uni_max}");
        assert!(bi_min < -5.4 && bi_max > 5.4, "bipolar offset range {bi_min}..{bi_max}");
        assert!(
            (bi_min + bi_max).abs() < 0.6,
            "bipolar offsets not symmetric: {bi_min}..{bi_max}"
        );
    }

    /// Finding 2: filter routing decisions follow `FiltersModule::process`
    /// and the producers' per-destination raw bypass.
    fn render_routing(
        routing: FilterRouting,
        destination: ProducerDestination,
        filter1_on: bool,
        filter2_on: bool,
    ) -> Vec<f32> {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.oscillators[0].params.wave_frame = PolyF32::splat(128.0);
            kernel.params.oscillators[0].destination = destination;
            kernel.params.filter_routing = routing;
            for (i, on) in [filter1_on, filter2_on].into_iter().enumerate() {
                let filter = &mut kernel.params.filters[i].params;
                filter.on = on;
                filter.state.midi_cutoff = PolyF32::splat(50.0);
                filter.state.set_pass_blend(PolyF32::ZERO);
            }
        }
        allocator.note_on(48, 1.0, 0, 0);
        render_blocks(&mut allocator, 12)
    }

    #[test]
    fn serial_forward_with_filter_2_off_still_passes_audio() {
        let serial = render_routing(FilterRouting::SerialForward, ProducerDestination::Filter1, true, false);
        let parallel = render_routing(FilterRouting::Parallel, ProducerDestination::Filter1, true, false);
        assert!(rms(&serial[512..]) > 0.01, "serial forward with f2 off is silent");
        // Vital falls back to parallel when the target filter is off.
        for (a, b) in serial.iter().zip(&parallel) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn serial_backward_with_filter_1_off_still_passes_audio() {
        let serial = render_routing(FilterRouting::SerialBackward, ProducerDestination::Filter2, false, true);
        let parallel = render_routing(FilterRouting::Parallel, ProducerDestination::Filter2, false, true);
        assert!(rms(&serial[512..]) > 0.01, "serial backward with f1 off is silent");
        for (a, b) in serial.iter().zip(&parallel) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn dual_filters_with_one_filter_off_does_not_double_the_dry_signal() {
        // Dual destination with only filter 1 on: the producer is NOT raw,
        // so the output equals the filter-1-only routing (no dry copy added).
        let dual = render_routing(FilterRouting::Parallel, ProducerDestination::DualFilters, true, false);
        let single = render_routing(FilterRouting::Parallel, ProducerDestination::Filter1, true, false);
        assert!(rms(&dual[512..]) > 0.01);
        for (a, b) in dual.iter().zip(&single) {
            assert!((a - b).abs() < 1e-6, "dual routing added the dry bus: {a} vs {b}");
        }
        // Both off: the producer bypasses raw exactly once.
        let bypass = render_routing(FilterRouting::Parallel, ProducerDestination::DualFilters, false, false);
        let raw = render_routing(FilterRouting::Parallel, ProducerDestination::Effects, false, false);
        for (a, b) in bypass.iter().zip(&raw) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn serial_routing_with_both_filters_on_differs_from_parallel() {
        let serial = render_routing(FilterRouting::SerialForward, ProducerDestination::Filter1, true, true);
        let parallel = render_routing(FilterRouting::Parallel, ProducerDestination::Filter1, true, true);
        assert!(rms(&serial[512..]) > 0.001);
        // Two low-passes in series attenuate more than one.
        assert!(
            rms(&serial[512..]) < rms(&parallel[512..]) * 0.9,
            "serial {} vs parallel {}",
            rms(&serial[512..]),
            rms(&parallel[512..])
        );
        let backward = render_routing(FilterRouting::SerialBackward, ProducerDestination::Filter2, true, true);
        assert!(rms(&backward[512..]) < rms(&parallel[512..]) * 0.9);
    }

    /// Finding 4: `(env × vel_scale × voice_amplitude)²`.
    #[test]
    fn amplitude_law_squares_voice_amplitude_and_velocity() {
        let render_level = |voice_amplitude: f32, velocity: f32, velocity_track: f32| {
            let mut allocator = make_allocator();
            for kernel in allocator.kernels_mut() {
                kernel.params.voice_amplitude = voice_amplitude;
                kernel.params.velocity_track = velocity_track;
            }
            allocator.note_on(60, velocity, 0, 0);
            let audio = render_blocks(&mut allocator, 12);
            rms(&audio[1024..])
        };
        let full = render_level(1.0, 1.0, 0.0);
        let half_amp = render_level(0.5, 1.0, 0.0);
        let half_vel = render_level(1.0, 0.5, 1.0);
        assert!(full > 0.01);
        assert!((full / half_amp - 4.0).abs() < 0.05, "voice_amplitude ratio {}", full / half_amp);
        assert!((full / half_vel - 4.0).abs() < 0.05, "velocity ratio {}", full / half_vel);
    }

    /// Finding 5: portamento glides the second voice from the last note.
    #[test]
    fn portamento_glides_from_last_note_over_the_configured_time() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.portamento_time = 0.1;
            kernel.params.portamento_force = true;
        }
        allocator.note_on(60, 1.0, 0, 0);
        let _ = render_blocks(&mut allocator, 2);
        allocator.note_on(72, 1.0, 0, 0);
        // Both notes share pair 0; the new note is voice slot 1 (lane 2).
        let _ = render_blocks(&mut allocator, 1);
        let early = allocator.kernels()[0].current_midi().lane(2);
        assert!(early > 59.9 && early < 62.0, "glide did not start at 60: {early}");
        let _ = render_blocks(&mut allocator, 16); // ~17 blocks ≈ 49 ms
        let mid = allocator.kernels()[0].current_midi().lane(2);
        assert!(mid > 63.0 && mid < 69.0, "glide midpoint {mid}");
        let _ = render_blocks(&mut allocator, 30); // well past 100 ms
        let end = allocator.kernels()[0].current_midi().lane(2);
        assert!((end - 72.0).abs() < 1e-3, "glide did not reach 72: {end}");
        // The first voice never glided.
        assert!((allocator.kernels()[0].current_midi().lane(0) - 60.0).abs() < 1e-3);
    }

    #[test]
    fn voice_transpose_and_tune_shift_the_bent_midi_exactly() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.voice_transpose = 7.0;
            kernel.params.voice_tune = 0.25;
        }
        allocator.note_on(60, 1.0, 0, 0);
        let _ = render_blocks(&mut allocator, 2);
        let kernel = &allocator.kernels()[0];
        assert!((kernel.current_midi().lane(0) - 67.25).abs() < 1e-4);
        // The `note` source reads the bent midi.
        assert!((kernel.last_source_values().note.lane(0) - 67.25 / 127.0).abs() < 1e-5);
    }

    /// Finding 6: a stolen voice retriggers without a phase reset.
    #[test]
    fn stolen_voice_keeps_phase_continuity() {
        let render = |steal: bool| {
            let mut allocator = VoiceAllocator::new(1, || {
                let mut kernel = SynthVoiceKernel::new(44100);
                kernel.params.envelopes[0] = EnvelopeParams {
                    attack: PolyF32::splat(0.001),
                    sustain: PolyF32::ONE,
                    release: PolyF32::splat(0.02),
                    ..Default::default()
                };
                kernel.params.oscillators[0].params.unison_voices = 1;
                kernel.params.oscillators[0].params.random_phase = PolyF32::ZERO;
                kernel
            });
            allocator.set_sample_rate(44100);
            allocator.set_override(VoiceOverride::Steal);
            allocator.note_on(60, 1.0, 0, 0);
            let mut audio = render_blocks(&mut allocator, 4);
            if steal {
                // Same note stolen mid-flight: Held → Triggering, no reset.
                allocator.note_on(60, 1.0, 37, 0);
            }
            audio.extend(render_blocks(&mut allocator, 4));
            audio
        };
        let continuous = render(false);
        let stolen = render(true);
        let max_diff = continuous
            .iter()
            .zip(&stolen)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(peak(&continuous) > 0.05);
        assert!(max_diff < 0.02, "stolen voice reset its phase: max diff {max_diff}");
    }

    /// Finding 7: per-lane trigger offsets inside one block.
    #[test]
    fn pair_voices_start_their_envelopes_at_their_own_offsets() {
        let mut allocator = make_allocator();
        allocator.note_on(60, 1.0, 0, 0);
        allocator.note_on(64, 1.0, 100, 0);
        assert_eq!(allocator.last_active_voice(), Some((0, 1)));
        let _ = render_blocks(&mut allocator, 1);
        let env = allocator.kernels()[0].voice_killer().unwrap();
        for (i, value) in env.iter().enumerate().take(100) {
            assert_eq!(value.lane(2), 0.0, "voice 1 started early at {i}");
        }
        assert!(env[10].lane(0) > 0.0);
        for i in 0..28 {
            assert!(
                (env[i].lane(0) - env[i + 100].lane(2)).abs() < 1e-5,
                "envelope shapes differ at {i}: {} vs {}",
                env[i].lane(0),
                env[i + 100].lane(2)
            );
        }
    }

    /// Finding 9/10: the per-sample cutoff ramp has no step at block
    /// boundaries: consecutive blocks continue from the previous target.
    #[test]
    fn cutoff_ramp_is_continuous_across_blocks() {
        let targets = [60.0f32, 70.0, 80.0, 90.0];
        let audio = vec![PolyF32::ZERO; 128];
        let mut out = vec![PolyF32::ZERO; 128];
        let mut sweep = Vec::new();
        let mut previous = PolyF32::splat(60.0);
        for &target in &targets {
            let target = PolyF32::splat(target);
            ramp_with_audio(previous, target, &audio, &mut out);
            previous = target;
            sweep.extend(out.iter().map(|v| v.lane(0)));
        }
        // Reference: a straight line from 60 to 90 over 3 blocks after the
        // first (flat) block. Every step is the same size (no jump).
        let step = 10.0 / 128.0;
        for i in 129..sweep.len() {
            let delta = sweep[i] - sweep[i - 1];
            assert!((delta - step).abs() < 1e-4, "step {delta} at sample {i}");
        }
        assert!((sweep[127] - 60.0).abs() < 1e-5);
        assert!((sweep[255] - 70.0).abs() < 1e-4);
        assert!((sweep[511] - 90.0).abs() < 1e-4);
    }

    /// Finding 10: an envelope into the cutoff is evaluated at audio rate
    /// (the cutoff buffer follows the envelope inside the block).
    #[test]
    fn envelope_to_cutoff_is_evaluated_per_sample() {
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.envelopes[1] = EnvelopeParams {
                attack: PolyF32::splat(0.05),
                sustain: PolyF32::ONE,
                ..Default::default()
            };
            kernel.params.filters[0].params.on = true;
            kernel.matrix.connections.push(Connection {
                source: ModSource::Envelope(1),
                dest: ModDest::FilterCutoff(0),
                transform: ModulationTransform::with_amount(1.0, 60.0),
            });
        }
        allocator.note_on(60, 1.0, 0, 0);
        let _ = render_blocks(&mut allocator, 2);
        let kernel = &allocator.kernels()[0];
        // The control-rate offset stays zero: the connection is audio rate.
        assert_eq!(kernel.last_offsets().filter_cutoff[0].lane(0), 0.0);
        let cutoff = &kernel.cutoff_buffer[0][..MAX_BUFFER_SIZE];
        assert!(cutoff[127].lane(0) > cutoff[0].lane(0) + 0.5, "cutoff not rising in-block");
        for window in cutoff.windows(2) {
            assert!(window[1].lane(0) >= window[0].lane(0) - 1e-4);
        }
    }

    /// Finding 14: transport-synced LFOs snap to the song position.
    #[test]
    fn synced_lfo_phase_follows_transport_seconds() {
        use spinwave_dsp::modulators::LfoSyncType;
        let mut allocator = make_allocator();
        for kernel in allocator.kernels_mut() {
            kernel.params.lfos[0].params.frequency = PolyF32::splat(2.0);
            kernel.params.lfos[0].params.sync_type = LfoSyncType::Sync;
            kernel.set_transport(1.3);
        }
        allocator.note_on(60, 1.0, 0, 0);
        let _ = render_blocks(&mut allocator, 1);
        let expected =
            spinwave_poly::utils::cycle_offset_from_seconds(1.3, PolyF32::splat(2.0)).lane(0);
        let phase = allocator.kernels()[0].lfo_phase(0).lane(0);
        assert!((phase - expected).abs() < 1e-4, "phase {phase} vs expected {expected}");
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

    /// `unison_detune` is a Quadratic parameter: the reference squares the
    /// modulated total (`cr::Square` after the modulation sum), so the
    /// preset stores 4.472 for 20 cents of range.
    #[test]
    fn unison_detune_squares_the_modulated_total() {
        let mut kernel = SynthVoiceKernel::new(44100);
        kernel.params.oscillators[0].params.unison_detune = PolyF32::splat(4.472_136);
        let common = CommonOffsets::default();

        let plain = kernel.modulated_wavetable_params(0, PolyF32::splat(60.0), &common);
        assert!(
            (plain.unison_detune.lane(0) - 20.0).abs() < 1e-3,
            "stored 4.472 should square to 20, got {}",
            plain.unison_detune.lane(0)
        );

        // The offset is added in the STORED domain, before the square.
        kernel.offsets.osc_unison_detune[0] = PolyF32::splat(0.527_864);
        let modulated = kernel.modulated_wavetable_params(0, PolyF32::splat(60.0), &common);
        assert!(
            (modulated.unison_detune.lane(0) - 25.0).abs() < 1e-3,
            "(4.472 + 0.528)^2 should be 25, got {}",
            modulated.unison_detune.lane(0)
        );

        // Clamped in the stored domain too (0..10 -> 0..100 cents scale).
        kernel.offsets.osc_unison_detune[0] = PolyF32::splat(50.0);
        let clamped = kernel.modulated_wavetable_params(0, PolyF32::splat(60.0), &common);
        assert!((clamped.unison_detune.lane(0) - 100.0).abs() < 1e-3);
    }

    /// Material is shared: installing a sample hands every kernel the same
    /// `Arc` (one band-limited pyramid) and returns the previous handle so
    /// its last drop can happen off the audio thread.
    #[test]
    fn installing_material_shares_one_handle() {
        let mut kernel = SynthVoiceKernel::new(44100);
        let sample = constant_sample_arc(0.5, 4096);
        let strong_before = Arc::strong_count(&sample);

        let previous = kernel.set_sample(0, sample.clone());
        assert!(!Arc::ptr_eq(&previous, &sample), "the default sample came back");
        // Slot handle + sampler handle + our two: no deep copy anywhere.
        assert_eq!(Arc::strong_count(&sample), strong_before + 2);
        assert!(Arc::ptr_eq(kernel.slot_sample(0), &sample));

        let replacement = constant_sample_arc(0.25, 4096);
        let returned = kernel.set_sample(0, replacement);
        assert!(Arc::ptr_eq(&returned, &sample), "the replaced handle must come back");

        let table = kernel.wavetables[0].clone();
        let old_table = kernel.set_wavetable(0, table.clone());
        assert!(Arc::ptr_eq(&old_table, &table));
    }

    /// A connection's drawn curve remaps the source value before the
    /// amount is applied (`line_mapping` in a preset).
    #[test]
    fn connection_remap_curve_is_applied() {
        use crate::modulation::RemapCurve;
        use spinwave_dsp::modulators::LineGenerator;

        // A flat curve at 1: any source value maps to full modulation.
        let mut generator = LineGenerator::new(2048);
        generator.set_num_points(2);
        generator.set_point(0, (0.0, 0.0));
        generator.set_point(1, (1.0, 0.0));
        generator.render();
        let curve = Arc::new(RemapCurve::from_line_generator(&generator));

        let mut transform = ModulationTransform::with_amount(1.0, 1.0);
        let plain = transform.process_control(PolyF32::splat(0.25)).scaled.lane(0);
        assert!((plain - 0.25).abs() < 1e-4);

        transform.remap = Some(curve);
        let remapped = transform.process_control(PolyF32::splat(0.25)).scaled.lane(0);
        assert!(
            (remapped - 1.0).abs() < 1e-3,
            "the flat curve should force full modulation, got {remapped}"
        );
    }
}

