//! Spinwave plugin shell (CLAP + VST3 + standalone) built on nih-plug.
//!
//! Drives the full synth voice kernel (wavetable oscillators, filters,
//! envelopes, modulation matrix) through the voice allocator.
//!
//! Threading model: the audio thread only swaps prebuilt structures in
//! (`LiveCommand::ApplyBuilt`) and hands everything it replaces to the
//! garbage collector thread; patches are built on the live network thread
//! or nih-plug's background executor (DAW state restore).

pub mod garbage;
pub mod live;
pub mod materials;
pub mod note_sequencer;
pub mod patch;

use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread::JoinHandle;

use nih_plug::prelude::*;
use spinwave_dsp::modulators::EnvelopeParams;
use spinwave_engine::engine::SoundEngine;
use spinwave_params::preset::LoadReport;
use spinwave_poly::constants::MAX_BUFFER_SIZE;
use spinwave_poly::PolyF32;

use garbage::{Garbage, GarbageChute};
use live::{LiveCommand, LiveHandle, LiveShared, PersistedPreset, PresetStore};
use patch::BuiltPatch;

/// Applies a complete `.vital` preset to the engine (offline use: builds
/// and drops on the calling thread). Returns what could not be mapped.
pub fn apply_preset(preset: &spinwave_params::Preset, engine: &mut SoundEngine) -> LoadReport {
    apply_preset_with(preset, engine, &mut materials::decode_wav_zone)
}

/// [`apply_preset`] with a custom SFZ zone decoder (any audio format).
pub fn apply_preset_with(
    preset: &spinwave_params::Preset,
    engine: &mut SoundEngine,
    decode: &mut dyn FnMut(&std::path::Path) -> Option<materials::ZoneFrames>,
) -> LoadReport {
    let mut report = LoadReport::default();
    patch::connections_report(preset, &mut report);
    let master = patch::master_from_preset(preset);
    let kernel_count = BuiltPatch::kernel_count(engine.allocator().kernels().len(), master.polyphony);
    // Built for the rate the preset will run at, not the one the engine
    // is at now (the convolution impulse is rendered at that rate).
    let engine_rate = engine_rate_for(engine.sample_rate(), master.oversampling);
    let built = BuiltPatch::build_with(preset, kernel_count, engine_rate, &mut report, decode);
    apply_built(engine, Box::new(built), &mut |_| {});
    report
}

/// The engine rate a preset runs at on a host at `sample_rate`: the
/// preset's oversampling through the same sample-rate rule the engine
/// applies (`effective_oversample`).
pub fn engine_rate_for(sample_rate: u32, oversampling: usize) -> u32 {
    sample_rate * spinwave_engine::engine::effective_oversample(oversampling, sample_rate) as u32
}

/// Swaps a prebuilt patch into the engine. Everything replaced goes to
/// `discard` instead of being dropped here, so the audio thread frees no
/// memory.
pub fn apply_built(
    engine: &mut SoundEngine,
    mut patch: Box<BuiltPatch>,
    discard: &mut impl FnMut(Garbage),
) {
    use spinwave_engine::engine::ChainId;

    let master = patch.master;
    // The preset's oversampling, as in the reference: a change reconfigures
    // the engine at its new rate with every voice cut, the way
    // `SynthBase::notifyOversamplingChanged` pauses, silences and rebuilds.
    // This is the one place the audio thread allocates, and it happens on
    // a preset load that changes the factor — never per block.
    if engine.requested_oversampling() != master.oversampling {
        engine.all_sounds_off();
        engine.set_oversampling(master.oversampling);
    }
    engine.master.volume_db = master.volume_db;
    engine.master.stereo_routing = master.stereo_routing;
    engine.master.stereo_mode = master.stereo_mode;
    engine.mixer = master.mixer;
    engine.set_polyphony(master.polyphony);

    let kernels = engine.allocator_mut().kernels_mut();
    for (index, kernel) in kernels.iter_mut().enumerate() {
        match patch.kernels.get_mut(index) {
            Some(new_params) => {
                // Keep the tempo the engine last received.
                new_params.beats_per_second = kernel.params.beats_per_second;
                std::mem::swap(&mut kernel.params, new_params);
            }
            None => {
                // More kernels than prebuilt (the pool grew past the
                // published count): clone as a fallback.
                if let Some(template) = patch.kernels.last() {
                    let mut params = template.clone();
                    params.beats_per_second = kernel.params.beats_per_second;
                    discard(Garbage::Kernel(Box::new(std::mem::replace(&mut kernel.params, params))));
                }
            }
        }
        // The matrix keeps a fixed-capacity list: copying into it clones
        // plain values and bumps the remap-curve refcounts, no allocation.
        kernel.matrix.set_connections(&patch.connections);
    }

    engine.effects_matrix.set_connections(&patch.effects_connections);
    std::mem::swap(engine.params_mut(), &mut *patch.effects);
    std::mem::swap(engine.chain_params_mut(ChainId::BusA), &mut *patch.bus_a);
    std::mem::swap(engine.chain_params_mut(ChainId::BusB), &mut *patch.bus_b);
    engine.allocator_mut().set_legato(master.legato);
    engine.allocator_mut().set_priority(master.voice_priority);
    engine.allocator_mut().set_override(master.voice_override);

    // Materials. Every kernel shares the same `Arc`s (the patch keeps its
    // own clones, which travel to the collector); the previous handle of
    // the first kernel is kept alive through `discard` so that the last
    // drop of the old material never happens here.
    for (slot, table) in &patch.wavetables {
        install_slot_wavetable(engine, *slot, table, discard);
    }
    for (slot, sample) in &patch.samples {
        install_slot_sample(engine, *slot, sample, discard);
    }
    if let Some(sample) = &patch.global_sample {
        install_global_sample(engine, sample, discard);
    }
    for (slot, sources) in patch.multisamples.iter_mut() {
        install_multisample_sources(engine, *slot, sources, discard);
    }
    // The impulse response was rendered and transformed off this thread;
    // swapping it in is a move, and the replaced engine leaves through the
    // chute (its spectra are megabytes).
    for (chain, prebuilt) in patch.convolutions.drain(..) {
        let previous = engine.set_convolution_engine(chain, prebuilt);
        discard(Garbage::Convolution(Box::new(previous)));
    }

    discard(Garbage::Patch(patch));
}

/// Installs one slot's wavetable on every kernel (refcount bumps only).
fn install_slot_wavetable(
    engine: &mut SoundEngine,
    slot: usize,
    table: &Arc<spinwave_dsp::wavetable::Wavetable>,
    discard: &mut impl FnMut(Garbage),
) {
    let mut previous = None;
    for kernel in engine.allocator_mut().kernels_mut() {
        let old = kernel.set_wavetable(slot, table.clone());
        if previous.is_none() {
            previous = Some(old);
        }
    }
    if let Some(previous) = previous {
        discard(Garbage::Wavetable(previous));
    }
}

/// Installs one slot's sample on every kernel (Sample + Granular engines
/// read the same shared pyramid; refcount bumps only).
fn install_slot_sample(
    engine: &mut SoundEngine,
    slot: usize,
    sample: &Arc<spinwave_dsp::oscillator::Sample>,
    discard: &mut impl FnMut(Garbage),
) {
    let mut previous = None;
    for kernel in engine.allocator_mut().kernels_mut() {
        let old = kernel.set_sample(slot, sample.clone());
        if previous.is_none() {
            previous = Some(old);
        }
    }
    if let Some(previous) = previous {
        discard(Garbage::Sample(previous));
    }
}

/// Installs the global (`settings.sample`, SMP section) sample on every
/// kernel's sampler.
fn install_global_sample(
    engine: &mut SoundEngine,
    sample: &Arc<spinwave_dsp::oscillator::Sample>,
    discard: &mut impl FnMut(Garbage),
) {
    let mut previous = None;
    for kernel in engine.allocator_mut().kernels_mut() {
        let old = kernel.sampler_mut().set_sample(sample.clone());
        if previous.is_none() {
            previous = Some(old);
        }
    }
    if let Some(previous) = previous {
        discard(Garbage::Sample(previous));
    }
}

/// Installs prebuilt multisample sources (one per kernel, built on the
/// network thread, popped in place: no allocation); the replaced sources
/// go to the collector.
fn install_multisample_sources(
    engine: &mut SoundEngine,
    slot: usize,
    sources: &mut Vec<spinwave_dsp::oscillator::MultisampleSource>,
    discard: &mut impl FnMut(Garbage),
) {
    for kernel in engine.allocator_mut().kernels_mut() {
        let Some(source) = sources.pop() else { break };
        let old = kernel.install_multisample_source(slot, source);
        discard(Garbage::MultisampleSource(old));
    }
}

/// Default playable patch until a preset is applied: saw oscillator into a
/// soft ADSR.
fn apply_default_patch(engine: &mut SoundEngine) {
    engine.kernel_params_mut(|params| {
        params.oscillators[0].on = true;
        params.oscillators[0].params.amplitude = PolyF32::splat(0.7);
        params.envelopes[0] = EnvelopeParams {
            attack: PolyF32::splat(0.005),
            decay: PolyF32::splat(0.3),
            sustain: PolyF32::splat(0.8),
            release: PolyF32::splat(0.15),
            ..Default::default()
        };
    });
}

// -- MIDI event mapping -------------------------------------------------------

/// What an incoming host event asks of the engine, decoupled from nih-plug
/// so the mapping is unit-testable. Notes go through the sequencer first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EngineCall {
    NoteOn { note: i32, velocity: f32, channel: usize },
    NoteOff { note: i32, velocity: f32, channel: usize },
    /// Bipolar wheel position in `[-1, 1]`.
    PitchWheel { value: f32, channel: usize },
    ModWheel { value: f32, channel: usize },
    SustainOn { channel: usize },
    SustainOff { channel: usize },
    SostenutoOn { channel: usize },
    SostenutoOff { channel: usize },
    /// Per-note aftertouch (polyphonic key pressure).
    PolyPressure { note: i32, pressure: f32, channel: usize },
    ChannelPressure { pressure: f32, channel: usize },
    /// MPE slide (CC74).
    Slide { value: f32, channel: usize },
    AllNotesOff,
    AllSoundsOff,
}

/// Maps a host note/CC event to an engine call; `None` for events the
/// engine has no use for (expressions, program changes, SysEx).
#[must_use]
pub fn map_note_event(event: &NoteEvent<()>) -> Option<EngineCall> {
    Some(match *event {
        NoteEvent::NoteOn { note, velocity, channel, .. } => {
            EngineCall::NoteOn { note: note as i32, velocity, channel: channel as usize }
        }
        NoteEvent::NoteOff { note, velocity, channel, .. } => {
            EngineCall::NoteOff { note: note as i32, velocity, channel: channel as usize }
        }
        NoteEvent::Choke { note, channel, .. } => {
            EngineCall::NoteOff { note: note as i32, velocity: 0.0, channel: channel as usize }
        }
        NoteEvent::PolyPressure { note, pressure, channel, .. } => {
            EngineCall::PolyPressure { note: note as i32, pressure, channel: channel as usize }
        }
        NoteEvent::MidiPitchBend { value, channel, .. } => {
            EngineCall::PitchWheel { value: (value * 2.0 - 1.0).clamp(-1.0, 1.0), channel: channel as usize }
        }
        NoteEvent::MidiChannelPressure { pressure, channel, .. } => {
            EngineCall::ChannelPressure { pressure, channel: channel as usize }
        }
        NoteEvent::MidiCC { cc, value, channel, .. } => {
            let channel = channel as usize;
            match cc {
                1 => EngineCall::ModWheel { value, channel },
                64 => {
                    if value >= 0.5 {
                        EngineCall::SustainOn { channel }
                    } else {
                        EngineCall::SustainOff { channel }
                    }
                }
                66 => {
                    if value >= 0.5 {
                        EngineCall::SostenutoOn { channel }
                    } else {
                        EngineCall::SostenutoOff { channel }
                    }
                }
                74 => EngineCall::Slide { value, channel },
                120 => EngineCall::AllSoundsOff,
                123 => EngineCall::AllNotesOff,
                _ => return None,
            }
        }
        _ => return None,
    })
}

// -- Plugin -------------------------------------------------------------------

/// Background work run through nih-plug's task executor (never on the
/// audio thread). Carries no heap data.
#[derive(Clone, Copy, Debug)]
pub enum SpinwaveTask {
    /// Build the store's current preset and send it to the audio thread
    /// (DAW state restore, first activation).
    RebuildFromStore,
}

#[derive(Params)]
struct SpinwaveParams {
    #[id = "gain"]
    pub gain: FloatParam,
    /// The whole patch as `.vital` JSON, saved with the DAW project. Backed
    /// by the live channel's preset store (one source of truth).
    #[persist = "preset"]
    pub preset: PersistedPreset,
}

impl SpinwaveParams {
    fn new(store: Arc<PresetStore>) -> Self {
        SpinwaveParams {
            gain: FloatParam::new(
                "Gain",
                util::db_to_gain(0.0),
                FloatRange::Skewed {
                    min: util::db_to_gain(-60.0),
                    max: util::db_to_gain(6.0),
                    factor: FloatRange::gain_skew_factor(-60.0, 6.0),
                },
            )
            .with_smoother(SmoothingStyle::Logarithmic(50.0))
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
            preset: PersistedPreset(store),
        }
    }
}

pub struct Spinwave {
    params: Arc<SpinwaveParams>,
    engine: SoundEngine,
    /// Note processor between incoming MIDI and the engine (arp/step seq).
    sequencer: note_sequencer::NoteSequencer,
    sample_rate: f32,
    scratch_left: Vec<f32>,
    scratch_right: Vec<f32>,
    /// Shared with the live network thread and the task executor.
    shared: Arc<LiveShared>,
    /// Audio-thread end of the command channel.
    live_rx: Receiver<LiveCommand>,
    /// The listener, once `initialize` started it (`None` when disabled).
    live: Option<LiveHandle>,
    garbage: GarbageChute,
    collector: Option<JoinHandle<()>>,
    /// Last latency announced to the host (host samples).
    reported_latency: u32,
}

/// Tempo assumed when the host reports none (standalone, some hosts).
const DEFAULT_BPM: f32 = 120.0;

impl Default for Spinwave {
    fn default() -> Self {
        let mut engine = SoundEngine::new(44100);
        apply_default_patch(&mut engine);
        let store = Arc::new(PresetStore::default());
        let (shared, live_rx) = LiveShared::new(store.clone());
        let (garbage, garbage_rx) = GarbageChute::new();
        let collector = garbage::spawn_collector(garbage_rx);
        Spinwave {
            params: Arc::new(SpinwaveParams::new(store)),
            engine,
            sequencer: note_sequencer::NoteSequencer::default(),
            sample_rate: 44100.0,
            scratch_left: vec![0.0; MAX_BUFFER_SIZE],
            scratch_right: vec![0.0; MAX_BUFFER_SIZE],
            shared,
            live_rx,
            live: None,
            garbage,
            collector,
            reported_latency: 0,
        }
    }
}

impl Drop for Spinwave {
    fn drop(&mut self) {
        // Stops the listener thread and unregisters the instance.
        self.live.take();
        // The collector exits once every chute is gone.
        let (empty, _) = GarbageChute::new();
        let chute = std::mem::replace(&mut self.garbage, empty);
        drop(chute);
        if let Some(collector) = self.collector.take() {
            let _ = collector.join();
        }
    }
}

impl Spinwave {
    /// Routes one engine call, notes through the sequencer.
    fn apply_engine_call(&mut self, call: EngineCall, offset: usize) {
        match call {
            EngineCall::NoteOn { note, velocity, channel } => {
                if let Some(out) = self.sequencer.note_on(note, velocity, channel) {
                    self.engine.note_on(out.note, out.velocity, offset, out.channel);
                }
            }
            EngineCall::NoteOff { note, velocity, channel } => {
                if let Some(out) = self.sequencer.note_off(note, velocity, channel) {
                    self.engine.note_off(out.note, out.velocity, offset, out.channel);
                }
            }
            EngineCall::PitchWheel { value, channel } => self.engine.set_pitch_wheel(value, channel),
            EngineCall::ModWheel { value, channel } => self.engine.set_mod_wheel(value, channel),
            EngineCall::SustainOn { channel } => self.engine.sustain_on(channel),
            EngineCall::SustainOff { channel } => self.engine.sustain_off(offset, channel),
            EngineCall::SostenutoOn { channel } => self.engine.sostenuto_on(channel),
            EngineCall::SostenutoOff { channel } => self.engine.sostenuto_off(offset, channel),
            EngineCall::PolyPressure { note, pressure, channel } => {
                self.engine.set_aftertouch(note, pressure, offset, channel)
            }
            EngineCall::ChannelPressure { pressure, channel } => {
                self.engine.set_channel_aftertouch(channel, pressure, offset)
            }
            EngineCall::Slide { value, channel } => {
                self.engine.set_channel_slide(channel, value, offset)
            }
            EngineCall::AllNotesOff => {
                let engine = &mut self.engine;
                self.sequencer.flush(|event| {
                    if !event.on {
                        engine.note_off(event.note, event.velocity, offset, event.channel);
                    }
                });
                self.engine.all_notes_off(offset);
            }
            EngineCall::AllSoundsOff => {
                self.sequencer.reset();
                self.engine.all_sounds_off();
            }
        }
    }

    /// Applies pending live commands at a block boundary. Patch structures
    /// arrive prebuilt from the network thread; only the swap happens here
    /// and everything replaced leaves through the garbage chute.
    fn drain_live_commands(&mut self) {
        self.garbage.flush();
        while let Ok(command) = self.live_rx.try_recv() {
            match command {
                LiveCommand::ApplyBuilt(patch) => {
                    apply_built(&mut self.engine, patch, &mut |item| self.garbage.discard(item));
                }
                LiveCommand::SetSample { slot, sample } => {
                    install_slot_sample(&mut self.engine, slot, &sample, &mut |item| {
                        self.garbage.discard(item)
                    });
                    self.garbage.discard(Garbage::Sample(sample));
                }
                LiveCommand::SetWavetable { slot, table } => {
                    for voice_kernel in self.engine.allocator_mut().kernels_mut() {
                        voice_kernel.set_wavetable(slot, table.clone());
                    }
                    self.garbage.discard(Garbage::Wavetable(table));
                }
                LiveCommand::SetMultisample { slot, mut sources } => {
                    let garbage = &mut self.garbage;
                    install_multisample_sources(&mut self.engine, slot, &mut sources, &mut |item| {
                        garbage.discard(item)
                    });
                    self.garbage.discard(Garbage::MultisampleSources(sources));
                }
                LiveCommand::NoteOn { note, velocity, channel } => {
                    self.apply_engine_call(EngineCall::NoteOn { note, velocity, channel }, 0);
                }
                LiveCommand::NoteOff { note, channel } => {
                    self.apply_engine_call(
                        EngineCall::NoteOff { note, velocity: 0.5, channel },
                        0,
                    );
                }
                LiveCommand::Seq(config) => {
                    let engine = &mut self.engine;
                    self.sequencer.set_config(*config, |event| {
                        // A config/mode change only ever releases notes.
                        if !event.on {
                            engine.note_off(event.note, event.velocity, 0, event.channel);
                        }
                    });
                }
                LiveCommand::Panic => self.apply_engine_call(EngineCall::AllSoundsOff, 0),
            }
        }
        self.shared
            .kernel_count
            .store(self.engine.allocator().kernels().len(), Ordering::Relaxed);
    }
}

impl Plugin for Spinwave {
    const NAME: &'static str = "Spinwave";
    const VENDOR: &'static str = "Spinwave";
    const URL: &'static str = "https://github.com/spinwave";
    const EMAIL: &'static str = "mb6684527@gmail.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: None,
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    /// `MidiCCs`: pitch bend, CCs (mod wheel, sustain, sostenuto, slide,
    /// all notes/sounds off) and channel pressure reach `process`.
    const MIDI_INPUT: MidiConfig = MidiConfig::MidiCCs;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = SpinwaveTask;

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let shared = self.shared.clone();
        Box::new(move |task| match task {
            SpinwaveTask::RebuildFromStore => match shared.build_and_send() {
                Ok(report) if !report.is_clean() => {
                    eprintln!("spinwave: patch rebuilt: {}", report.summary())
                }
                Ok(_) => {}
                Err(e) => eprintln!("spinwave: patch rebuild failed: {e}"),
            },
        })
    }

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        self.engine.set_sample_rate(buffer_config.sample_rate as u32);
        self.sample_rate = buffer_config.sample_rate;
        self.reported_latency = self.engine.latency_samples() as u32;
        context.set_latency_samples(self.reported_latency);
        // The network thread builds patches for the rate the preset's
        // oversampling gives at this host rate.
        self.shared.sample_rate.store(self.engine.sample_rate(), Ordering::Relaxed);
        self.shared
            .kernel_count
            .store(self.engine.allocator().kernels().len(), Ordering::Relaxed);
        // First activation: open the live channel (not in `Default`, so a
        // host scanning plugins opens no port).
        if self.live.is_none() && live::enabled() {
            self.live = live::start(self.shared.clone());
        }
        // nih-plug calls `initialize` again after a state restore (the
        // persisted preset has been written to the store by then): rebuild
        // the patch off the audio thread only when the store changed.
        if !self.shared.store.is_applied() {
            context.execute(SpinwaveTask::RebuildFromStore);
        }
        true
    }

    fn reset(&mut self) {
        self.sequencer.reset();
        self.engine.all_sounds_off();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.drain_live_commands();
        self.shared.blocks.fetch_add(1, Ordering::Relaxed);

        let num_samples = buffer.samples();
        let mut block_start = 0usize;

        let transport = context.transport();
        let bpm = transport.tempo.unwrap_or(DEFAULT_BPM as f64) as f32;
        if transport.tempo.is_some() {
            self.sequencer.set_bpm(bpm);
        }
        self.sequencer.set_transport(transport.playing, transport.pos_beats());
        // Transport seconds drive the synced LFOs / random LFOs
        // (`correct_to_time`). Without a host position (standalone) the
        // engine's own clock free-runs: it advances across each block while
        // "playing", so feed its last value back and keep it running.
        match transport.pos_seconds() {
            Some(seconds) => self.engine.set_transport(seconds, bpm, transport.playing),
            None => {
                let seconds = self.engine.transport_seconds();
                self.engine.set_transport(seconds, bpm, true);
            }
        }

        // The wet convolution path adds latency only while it is active;
        // tell the host whenever that changes.
        let latency = self.engine.latency_samples() as u32;
        if latency != self.reported_latency {
            self.reported_latency = latency;
            context.set_latency_samples(latency);
        }

        let mut next_event = context.next_event();
        while block_start < num_samples {
            let block_end = (block_start + MAX_BUFFER_SIZE).min(num_samples);

            // Route events landing in this block, with in-block offsets.
            while let Some(event) = next_event {
                let timing = event.timing() as usize;
                if timing >= block_end {
                    break;
                }
                let offset = timing.saturating_sub(block_start);
                if let Some(call) = map_note_event(&event) {
                    self.apply_engine_call(call, offset);
                }
                next_event = context.next_event();
            }

            let block_len = block_end - block_start;

            // The sequencer clock emits its own engine note events for
            // this block (no-op in Off mode).
            let sample_rate = self.sample_rate;
            let engine = &mut self.engine;
            self.sequencer.process(block_len, sample_rate, |event| {
                if event.on {
                    engine.note_on(event.note, event.velocity, event.offset, event.channel);
                } else {
                    engine.note_off(event.note, event.velocity, event.offset, event.channel);
                }
            });

            self.engine.process(
                block_len,
                &mut self.scratch_left[..block_len],
                &mut self.scratch_right[..block_len],
            );

            let output = buffer.as_slice();
            for i in 0..block_len {
                let gain = self.params.gain.smoothed.next();
                output[0][block_start + i] = self.scratch_left[i] * gain;
                output[1][block_start + i] = self.scratch_right[i] * gain;
            }

            block_start = block_end;
        }

        ProcessStatus::KeepAlive
    }
}

impl ClapPlugin for Spinwave {
    const CLAP_ID: &'static str = "org.spinwave.synth";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Spinwave: a wavetable synthesizer (Rust rework of Vital)");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::Instrument,
        ClapFeature::Synthesizer,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for Spinwave {
    const VST3_CLASS_ID: [u8; 16] = *b"SpinwaveSynth001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Instrument, Vst3SubCategory::Synth];
}

nih_export_clap!(Spinwave);
nih_export_vst3!(Spinwave);

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(cc: u8, value: f32, channel: u8) -> NoteEvent<()> {
        NoteEvent::MidiCC { timing: 0, channel, cc, value }
    }

    #[test]
    fn host_events_map_to_engine_calls() {
        let on = NoteEvent::NoteOn { timing: 0, voice_id: None, channel: 2, note: 60, velocity: 0.7 };
        assert_eq!(
            map_note_event(&on),
            Some(EngineCall::NoteOn { note: 60, velocity: 0.7, channel: 2 })
        );
        let off = NoteEvent::NoteOff { timing: 0, voice_id: None, channel: 2, note: 60, velocity: 0.3 };
        assert_eq!(
            map_note_event(&off),
            Some(EngineCall::NoteOff { note: 60, velocity: 0.3, channel: 2 })
        );
        let bend = NoteEvent::MidiPitchBend { timing: 0, channel: 0, value: 1.0 };
        assert_eq!(map_note_event(&bend), Some(EngineCall::PitchWheel { value: 1.0, channel: 0 }));
        let center = NoteEvent::MidiPitchBend { timing: 0, channel: 0, value: 0.5 };
        assert_eq!(map_note_event(&center), Some(EngineCall::PitchWheel { value: 0.0, channel: 0 }));
        let pressure = NoteEvent::PolyPressure { timing: 0, voice_id: None, channel: 1, note: 64, pressure: 0.4 };
        assert_eq!(
            map_note_event(&pressure),
            Some(EngineCall::PolyPressure { note: 64, pressure: 0.4, channel: 1 })
        );
        let channel_pressure = NoteEvent::MidiChannelPressure { timing: 0, channel: 3, pressure: 0.9 };
        assert_eq!(
            map_note_event(&channel_pressure),
            Some(EngineCall::ChannelPressure { pressure: 0.9, channel: 3 })
        );
        assert_eq!(map_note_event(&cc(1, 0.6, 0)), Some(EngineCall::ModWheel { value: 0.6, channel: 0 }));
        assert_eq!(map_note_event(&cc(64, 1.0, 0)), Some(EngineCall::SustainOn { channel: 0 }));
        assert_eq!(map_note_event(&cc(64, 0.0, 5)), Some(EngineCall::SustainOff { channel: 5 }));
        assert_eq!(map_note_event(&cc(66, 1.0, 0)), Some(EngineCall::SostenutoOn { channel: 0 }));
        assert_eq!(map_note_event(&cc(66, 0.2, 0)), Some(EngineCall::SostenutoOff { channel: 0 }));
        assert_eq!(map_note_event(&cc(74, 0.25, 1)), Some(EngineCall::Slide { value: 0.25, channel: 1 }));
        assert_eq!(map_note_event(&cc(120, 0.0, 0)), Some(EngineCall::AllSoundsOff));
        assert_eq!(map_note_event(&cc(123, 0.0, 0)), Some(EngineCall::AllNotesOff));
        assert_eq!(map_note_event(&cc(7, 0.5, 0)), None);
        let program = NoteEvent::MidiProgramChange { timing: 0, channel: 0, program: 3 };
        assert_eq!(map_note_event(&program), None);
    }

    #[test]
    fn midi_config_receives_ccs() {
        assert!(Spinwave::MIDI_INPUT >= MidiConfig::MidiCCs);
    }

    #[test]
    fn prebuilt_patch_swaps_in_and_previous_structures_leave_through_the_chute() {
        let mut engine = SoundEngine::new(44100);
        let preset = spinwave_params::Preset::from_json(
            r#"{"synth_version":"1.0.7","preset_name":"t",
                "settings":{"polyphony": 6.0, "filter_1_cutoff": 90.0,
                "modulations":[{"source":"lfo_1","destination":"filter_1_cutoff"}]}}"#,
        )
        .unwrap();
        let mut report = LoadReport::default();
        let kernel_count = BuiltPatch::kernel_count(engine.allocator().kernels().len(), 6);
        let built = BuiltPatch::build(&preset, kernel_count, engine.engine_rate(), &mut report);
        let mut discarded = Vec::new();
        apply_built(&mut engine, Box::new(built), &mut |item| discarded.push(item));
        assert_eq!(engine.allocator().polyphony(), 6);
        for kernel in engine.allocator().kernels() {
            assert_eq!(kernel.params.filters[0].params.state.midi_cutoff.lane(0), 90.0);
            assert_eq!(kernel.matrix.connections.len(), 1);
        }
        // The previous params travelled out inside the patch.
        assert!(discarded.iter().any(|g| matches!(g, Garbage::Patch(_))));
        let Some(Garbage::Patch(old)) = discarded.iter().find(|g| matches!(g, Garbage::Patch(_))) else {
            unreachable!()
        };
        assert_eq!(old.kernels.len(), kernel_count);
        // Offline helper reports cleanly too.
        let report = apply_preset(&preset, &mut engine);
        assert!(report.is_clean());
    }

    #[test]
    fn plugin_default_opens_no_port_and_drops_cleanly() {
        let plugin = Spinwave::default();
        assert!(plugin.live.is_none());
        assert!(!plugin.shared.store.is_applied());
        drop(plugin);
    }
}

#[cfg(test)]
mod oversampling_follows_the_preset {
    use super::*;

    /// The reference reconfigures its engine from the preset's
    /// `oversampling` on every load; the plugin path did not, so the same
    /// patch ran at 2x in the plugin and at the preset's factor offline.
    #[test]
    fn a_preset_at_4x_runs_the_engine_at_4x_and_back() {
        let mut engine = SoundEngine::with_pool(44100, 2);
        assert_eq!(engine.oversampling(), 2, "the default");
        let four = spinwave_params::Preset::from_json(
            r#"{"synth_version":"1.0.7","preset_name":"t","settings":{"oversampling": 2.0}}"#,
        )
        .unwrap();
        apply_preset(&four, &mut engine);
        assert_eq!(engine.oversampling(), 4);
        assert_eq!(engine.engine_rate(), 176_400);
        let one = spinwave_params::Preset::from_json(
            r#"{"synth_version":"1.0.7","preset_name":"t","settings":{"oversampling": 0.0}}"#,
        )
        .unwrap();
        apply_preset(&one, &mut engine);
        assert_eq!(engine.oversampling(), 1);
        // The sample-rate rule still applies: 2x asked at 96 kHz runs 1x.
        assert_eq!(engine_rate_for(96_000, 2), 96_000);
        assert_eq!(engine_rate_for(44_100, 8), 176_400, "the engine caps at 4x");
    }
}
