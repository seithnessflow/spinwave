//! Spinwave plugin shell (CLAP + VST3 + standalone) built on nih-plug.
//!
//! Drives the full synth voice kernel (wavetable oscillators, filters,
//! envelopes, modulation matrix) through the voice allocator.

use std::sync::Arc;

use nih_plug::prelude::*;
use spinwave_dsp::modulators::EnvelopeParams;
use spinwave_engine::kernel::SynthVoiceKernel;
use spinwave_engine::VoiceAllocator;
use spinwave_poly::constants::MAX_BUFFER_SIZE;
use spinwave_poly::PolyF32;

/// Default playable patch until preset loading is wired: saw oscillator
/// into a soft ADSR.
fn make_kernel() -> SynthVoiceKernel {
    let mut kernel = SynthVoiceKernel::new(44100);
    kernel.params.oscillators[0].on = true;
    kernel.params.oscillators[0].params.amplitude = PolyF32::splat(0.7);
    kernel.params.envelopes[0] = EnvelopeParams {
        attack: PolyF32::splat(0.005),
        decay: PolyF32::splat(0.3),
        sustain: PolyF32::splat(0.8),
        release: PolyF32::splat(0.15),
        ..Default::default()
    };
    kernel
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
    allocator: VoiceAllocator<SynthVoiceKernel>,
    mix: Vec<PolyF32>,
}

impl Default for Spinwave {
    fn default() -> Self {
        Spinwave {
            params: Arc::new(SpinwaveParams::default()),
            allocator: VoiceAllocator::new(16, make_kernel),
            mix: vec![PolyF32::ZERO; MAX_BUFFER_SIZE],
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
        self.allocator.set_sample_rate(buffer_config.sample_rate as u32);
        true
    }

    fn reset(&mut self) {
        self.allocator.all_sounds_off();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let num_samples = buffer.samples();
        let mut block_start = 0usize;

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
                        self.allocator
                            .note_on(note as i32, velocity, offset, channel as usize)
                    }
                    NoteEvent::NoteOff { note, velocity, channel, .. } => {
                        self.allocator
                            .note_off(note as i32, velocity, offset, channel as usize)
                    }
                    NoteEvent::MidiPitchBend { value, channel, .. } => {
                        self.allocator.set_pitch_wheel(value * 2.0 - 1.0, channel as usize)
                    }
                    NoteEvent::MidiCC { cc, value, channel, .. } => match cc {
                        1 => self.allocator.set_mod_wheel(value, channel as usize),
                        64 => {
                            if value >= 0.5 {
                                self.allocator.sustain_on(channel as usize);
                            } else {
                                self.allocator.sustain_off(offset, channel as usize);
                            }
                        }
                        _ => {}
                    },
                    NoteEvent::MidiChannelPressure { pressure, channel, .. } => self
                        .allocator
                        .set_channel_aftertouch(channel as usize, pressure, offset),
                    _ => {}
                }
                next_event = context.next_event();
            }

            let block_len = block_end - block_start;
            self.mix[..block_len].fill(PolyF32::ZERO);
            let mix = &mut self.mix;
            self.allocator.process(block_len, |kernel_out| {
                for (dest, src) in mix[..block_len].iter_mut().zip(kernel_out) {
                    *dest += *src;
                }
            });

            let output = buffer.as_slice();
            for i in 0..block_len {
                // Fold the two voices per vector into one stereo frame.
                let folded = self.mix[i] + self.mix[i].swap_voices();
                let gain = self.params.gain.smoothed.next();
                output[0][block_start + i] = folded.lane(0) * gain;
                output[1][block_start + i] = folded.lane(1) * gain;
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
