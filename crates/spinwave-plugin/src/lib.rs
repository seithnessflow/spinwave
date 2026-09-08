//! Spinwave plugin shell (CLAP + VST3 + standalone) built on nih-plug.
//!
//! Drives the full synth voice kernel (wavetable oscillators, filters,
//! envelopes, modulation matrix) through the voice allocator.

pub mod live;
pub mod note_sequencer;
pub mod patch;

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use nih_plug::prelude::*;
use spinwave_dsp::modulators::EnvelopeParams;
use spinwave_engine::engine::SoundEngine;
use spinwave_poly::constants::MAX_BUFFER_SIZE;
use spinwave_poly::PolyF32;

/// Applies a complete `.vital` preset to the engine: voice kernel,
/// modulation matrix, bus effects (main + both send-bus chains), mixer and
/// master settings.
pub fn apply_preset(preset: &spinwave_params::Preset, engine: &mut SoundEngine) {
    let kernel_params = patch::kernel_params_from_preset(preset);
    let connections = patch::connections_from_preset(preset);
    let effects_connections = patch::effects_connections_from_preset(preset);
    let effects = patch::effects_params_from_preset(preset);
    let bus_a = patch::effects_params_from_preset_prefixed(preset, "bus_a_");
    let bus_b = patch::effects_params_from_preset_prefixed(preset, "bus_b_");
    let master = patch::master_from_preset(preset);
    apply_built(
        engine,
        &kernel_params,
        &connections,
        &effects_connections,
        effects,
        bus_a,
        bus_b,
        &master,
    );
    for (index, table) in patch::wavetables_from_preset(preset) {
        for kernel in engine.allocator_mut().kernels_mut() {
            kernel.set_wavetable(index, table.clone());
        }
    }
}

/// Applies prebuilt patch structures (the live channel builds them on the
/// network thread so the audio thread only swaps them in).
#[allow(clippy::too_many_arguments)]
pub fn apply_built(
    engine: &mut SoundEngine,
    kernel_params: &spinwave_engine::kernel::KernelParams,
    connections: &[spinwave_engine::kernel::mod_matrix::Connection],
    effects_connections: &[spinwave_engine::engine::EffectsConnection],
    effects: spinwave_engine::engine::EffectsParams,
    bus_a: spinwave_engine::engine::EffectsParams,
    bus_b: spinwave_engine::engine::EffectsParams,
    master: &patch::MasterFromPreset,
) {
    use spinwave_engine::engine::ChainId;

    engine.master.volume_db = master.volume_db;
    engine.master.stereo_routing = master.stereo_routing;
    engine.master.stereo_mode = master.stereo_mode;
    engine.mixer = master.mixer;
    engine.set_polyphony(master.polyphony);
    engine.kernel_params_mut(|params| *params = kernel_params.clone());
    for kernel in engine.allocator_mut().kernels_mut() {
        kernel.matrix.connections = connections.to_vec();
    }
    engine.effects_matrix.connections = effects_connections.to_vec();
    *engine.params_mut() = effects;
    *engine.chain_params_mut(ChainId::BusA) = bus_a;
    *engine.chain_params_mut(ChainId::BusB) = bus_b;
    engine.allocator_mut().set_legato(master.legato);
    engine.allocator_mut().set_priority(master.voice_priority);
    engine.allocator_mut().set_override(master.voice_override);
}

/// Default playable patch until a preset is loaded: saw oscillator into a
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

#[derive(Params)]
struct SpinwaveParams {
    #[id = "gain"]
    pub gain: FloatParam,
}

impl Default for SpinwaveParams {
    fn default() -> Self {
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
    /// Live control commands (always on, standalone and DAW-hosted alike;
    /// disable with `SPINWAVE_LIVE=0`).
    live_rx: Option<Receiver<live::LiveCommand>>,
    /// Rendered-block counter, reported by the live ping as proof of life.
    live_blocks: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl Default for Spinwave {
    fn default() -> Self {
        let mut engine = SoundEngine::new(44100);
        apply_default_patch(&mut engine);
        let live_blocks = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let live_rx = live::start(live_blocks.clone()).map(|state| state.receiver);
        Spinwave {
            params: Arc::new(SpinwaveParams::default()),
            engine,
            sequencer: note_sequencer::NoteSequencer::default(),
            sample_rate: 44100.0,
            scratch_left: vec![0.0; MAX_BUFFER_SIZE],
            scratch_right: vec![0.0; MAX_BUFFER_SIZE],
            live_rx,
            live_blocks,
        }
    }
}

impl Spinwave {
    /// Applies pending live commands at a block boundary. Patch structures
    /// arrive prebuilt from the network thread; only the swap (and the drop
    /// of the previous structs) happens here.
    fn drain_live_commands(&mut self) {
        let Some(receiver) = &self.live_rx else { return };
        while let Ok(command) = receiver.try_recv() {
            match command {
                live::LiveCommand::ApplyBuilt {
                    kernel,
                    connections,
                    effects_connections,
                    effects,
                    bus_a,
                    bus_b,
                    master,
                    wavetables,
                } => {
                    apply_built(
                        &mut self.engine,
                        &kernel,
                        &connections,
                        &effects_connections,
                        *effects,
                        *bus_a,
                        *bus_b,
                        &master,
                    );
                    for (index, table) in wavetables {
                        for voice_kernel in self.engine.allocator_mut().kernels_mut() {
                            voice_kernel.set_wavetable(index, table.clone());
                        }
                    }
                }
                live::LiveCommand::SetSample { slot, sample } => {
                    // set_sample rebuilds each kernel's private band-limited
                    // pyramid — heavy, but acceptable at patch-load time.
                    for voice_kernel in self.engine.allocator_mut().kernels_mut() {
                        voice_kernel.set_sample(slot, sample.clone());
                    }
                }
                live::LiveCommand::SetWavetable { slot, table } => {
                    for voice_kernel in self.engine.allocator_mut().kernels_mut() {
                        voice_kernel.set_wavetable(slot, table.clone());
                    }
                }
                live::LiveCommand::SetMultisample { slot, mut instruments } => {
                    // One prebuilt Multisample per kernel (they are not
                    // Clone); the network thread sends enough for the
                    // maximum kernel count.
                    for voice_kernel in self.engine.allocator_mut().kernels_mut() {
                        let Some(instrument) = instruments.pop() else { break };
                        voice_kernel.set_multisample(slot, instrument);
                    }
                }
                live::LiveCommand::NoteOn { note, velocity, channel } => {
                    if let Some(event) = self.sequencer.note_on(note, velocity, channel) {
                        self.engine.note_on(event.note, event.velocity, 0, event.channel);
                    }
                }
                live::LiveCommand::NoteOff { note, channel } => {
                    if let Some(event) = self.sequencer.note_off(note, 0.5, channel) {
                        self.engine.note_off(event.note, event.velocity, 0, event.channel);
                    }
                }
                live::LiveCommand::Seq(config) => {
                    let engine = &mut self.engine;
                    self.sequencer.set_config(*config, |event| {
                        // A config/mode change only ever releases notes.
                        if !event.on {
                            engine.note_off(event.note, event.velocity, 0, event.channel);
                        }
                    });
                }
                live::LiveCommand::Panic => {
                    self.sequencer.reset();
                    self.engine.all_sounds_off()
                }
            }
        }
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

    const MIDI_INPUT: MidiConfig = MidiConfig::Basic;
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.engine.set_sample_rate(buffer_config.sample_rate as u32);
        self.sample_rate = buffer_config.sample_rate;
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
        self.live_blocks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let num_samples = buffer.samples();
        let mut block_start = 0usize;

        if let Some(tempo) = context.transport().tempo {
            self.engine.set_bpm(tempo as f32);
            self.sequencer.set_bpm(tempo as f32);
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
                match event {
                    NoteEvent::NoteOn { note, velocity, channel, .. } => {
                        // The sequencer consumes notes unless its mode is Off.
                        if let Some(out) =
                            self.sequencer.note_on(note as i32, velocity, channel as usize)
                        {
                            self.engine.note_on(out.note, out.velocity, offset, out.channel)
                        }
                    }
                    NoteEvent::NoteOff { note, velocity, channel, .. } => {
                        if let Some(out) =
                            self.sequencer.note_off(note as i32, velocity, channel as usize)
                        {
                            self.engine.note_off(out.note, out.velocity, offset, out.channel)
                        }
                    }
                    NoteEvent::MidiPitchBend { value, channel, .. } => {
                        self.engine.set_pitch_wheel(value * 2.0 - 1.0, channel as usize)
                    }
                    NoteEvent::MidiCC { cc, value, channel, .. } => match cc {
                        1 => self.engine.set_mod_wheel(value, channel as usize),
                        64 => {
                            if value >= 0.5 {
                                self.engine.sustain_on(channel as usize);
                            } else {
                                self.engine.sustain_off(offset, channel as usize);
                            }
                        }
                        _ => {}
                    },
                    NoteEvent::MidiChannelPressure { pressure, channel, .. } => self
                        .engine
                        .set_channel_aftertouch(channel as usize, pressure, offset),
                    _ => {}
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
