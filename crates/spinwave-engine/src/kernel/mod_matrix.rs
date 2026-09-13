//! The per-voice modulation matrix: routes modulator outputs into
//! parameter offsets through [`ModulationTransform`]s.
//!
//! Rework of Vital's dynamic modulation plumbing: sources and destinations
//! are enums, offsets accumulate into a plain struct the kernel adds to its
//! base parameter values each block.

use crate::modulation::ModulationTransform;
use spinwave_poly::{PolyF32, PolyMask};

// Engine limits — the single source of truth for the whole workspace
// (re-exported from the crate root; the parameter table is aligned to
// these). They are Spinwave's raised limits, not Vital's (3 / 6 / 8 / 4).
/// Oscillator slots per voice.
pub const NUM_OSCILLATORS: usize = 4;
/// Envelopes per voice (envelope 0 is the amplitude envelope).
pub const NUM_ENVELOPES: usize = 8;
/// LFOs per voice.
pub const NUM_LFOS: usize = 12;
/// Random LFOs per voice.
pub const NUM_RANDOM_LFOS: usize = 4;
/// Macro controls.
pub const NUM_MACROS: usize = 8;
/// Fixed capacity of every modulation connection list (reference
/// `kMaxModulationConnections`): [`ModMatrix::set_connections`] truncates
/// beyond it and never reallocates.
pub const MAX_MODULATION_CONNECTIONS: usize = 64;

/// Modulation sources readable each control tick. All values in `[0, 1]`
/// (bipolar handling happens inside the transform).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModSource {
    Envelope(usize),
    Lfo(usize),
    RandomLfo(usize),
    Macro(usize),
    Note,
    NoteInOctave,
    Velocity,
    Lift,
    ModWheel,
    PitchWheel,
    Aftertouch,
    Slide,
    Random,
    Stereo,
}

impl ModSource {
    /// Sources the kernel can render sample by sample (envelopes and LFOs;
    /// the reference switches them to audio rate when they feed an
    /// audio-rate destination).
    pub fn is_audio_rate_capable(self) -> bool {
        matches!(self, ModSource::Envelope(_) | ModSource::Lfo(_))
    }

    /// A source that is one value for every voice — the reference's
    /// `createMonoModControl` sources. A connection from one of these is
    /// evaluated before the voices there, so it reads a meta-modulated
    /// amount one block late (measured on the macro:
    /// `meta_step_timing`, `meta_ramp_on_mono_source_target`; the wheels
    /// are the same kind of control and are ASSUMED to behave the same,
    /// no case yet).
    pub fn is_mono(self) -> bool {
        matches!(self, ModSource::Macro(_) | ModSource::ModWheel | ModSource::PitchWheel)
    }
}

/// Curated destination set for the first kernel iteration; grows toward
/// the full ~400-destination table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModDest {
    OscLevel(usize),
    OscTranspose(usize),
    OscTune(usize),
    OscFrame(usize),
    OscFrameSpread(usize),
    OscPan(usize),
    OscUnisonDetune(usize),
    OscUnisonBlend(usize),
    OscStereoSpread(usize),
    OscDistortionAmount(usize),
    OscDistortionPhase(usize),
    OscSpectralMorphAmount(usize),
    OscPhase(usize),
    SampleLevel,
    SampleTranspose,
    SampleTune,
    SamplePan,
    FilterCutoff(usize),
    FilterResonance(usize),
    FilterDrive(usize),
    FilterBlend(usize),
    FilterBlendTranspose(usize),
    FilterKeytrack(usize),
    FilterMix(usize),
    EnvDelay(usize),
    EnvAttack(usize),
    EnvAttackPower(usize),
    EnvHold(usize),
    EnvDecay(usize),
    EnvDecayPower(usize),
    EnvSustain(usize),
    EnvRelease(usize),
    EnvReleasePower(usize),
    /// `lfo_N_frequency`: offset in the stored log2 domain (Exponential
    /// scale), applied before the scale like the reference's
    /// ExponentialScale — an offset in Hz was wrong for as long as no
    /// case modulated an LFO rate.
    LfoFrequency(usize),
    LfoPhase(usize),
    /// `lfo_N_tempo`: an offset on the sync ratio index.
    LfoTempo(usize),
    /// `lfo_N_smooth_time`, stored log2 domain (Exponential scale).
    LfoSmoothTime(usize),
    LfoDelayTime(usize),
    LfoFadeTime(usize),
    LfoStereo(usize),
    /// `lfo_N_keytrack_transpose` (semitones, in the keytrack sync mode).
    LfoKeytrackTranspose(usize),
    RandomLfoFrequency(usize),
    RandomLfoTempo(usize),
    RandomLfoKeytrackTranspose(usize),
    OscDetuneRange(usize),
    OscDetunePower(usize),
    /// `osc_N_unison_voices`: rounded to the nearest count, as the
    /// reference's `roundf`.
    OscUnisonVoices(usize),
    OscSpectralMorphSpread(usize),
    OscDistortionSpread(usize),
    FilterFormantX(usize),
    FilterFormantY(usize),
    FilterFormantTranspose(usize),
    FilterFormantSpread(usize),
    VoiceTune,
    VoiceTranspose,
    /// `portamento_time`, stored log2 domain (Exponential scale).
    PortamentoTime,
    /// Per-voice amplitude offset (the reference's modulatable
    /// `voice_amplitude`): added to `KernelParams::voice_amplitude` BEFORE
    /// the amplitude law squares it, destination scale 1.0. This is NOT the
    /// master `volume` parameter — a preset's "volume" destination belongs
    /// to the master path.
    VolumeAmp,
    PitchBend,
    /// The amount of the connection in slot `n` (`modulation_{n+1}_amount`):
    /// meta-modulation. Range 2, clamped with the base amount to [-1, 1],
    /// read in the block the source moves (notes/meta-modulation.md).
    ModulationAmount(usize),
    /// The power of the connection in slot `n`. Range 20, not clamped.
    ModulationPower(usize),
}

impl ModDest {
    /// Every destination, one per index where indexed (tests and the
    /// bench's bounds check).
    pub fn every() -> Vec<ModDest> {
        let mut all = Vec::new();
        for i in 0..NUM_OSCILLATORS {
            all.extend([
                ModDest::OscLevel(i),
                ModDest::OscTranspose(i),
                ModDest::OscTune(i),
                ModDest::OscFrame(i),
                ModDest::OscFrameSpread(i),
                ModDest::OscPan(i),
                ModDest::OscUnisonDetune(i),
                ModDest::OscUnisonBlend(i),
                ModDest::OscStereoSpread(i),
                ModDest::OscDistortionAmount(i),
                ModDest::OscDistortionPhase(i),
                ModDest::OscSpectralMorphAmount(i),
                ModDest::OscPhase(i),
            ]);
        }
        all.extend([
            ModDest::SampleLevel,
            ModDest::SampleTranspose,
            ModDest::SampleTune,
            ModDest::SamplePan,
        ]);
        for i in 0..2 {
            all.extend([
                ModDest::FilterCutoff(i),
                ModDest::FilterResonance(i),
                ModDest::FilterDrive(i),
                ModDest::FilterBlend(i),
                ModDest::FilterBlendTranspose(i),
                ModDest::FilterKeytrack(i),
                ModDest::FilterMix(i),
            ]);
        }
        for i in 0..NUM_ENVELOPES {
            all.extend([
                ModDest::EnvDelay(i),
                ModDest::EnvAttack(i),
                ModDest::EnvAttackPower(i),
                ModDest::EnvHold(i),
                ModDest::EnvDecay(i),
                ModDest::EnvDecayPower(i),
                ModDest::EnvSustain(i),
                ModDest::EnvRelease(i),
                ModDest::EnvReleasePower(i),
            ]);
        }
        for i in 0..NUM_LFOS {
            all.extend([
                ModDest::LfoFrequency(i),
                ModDest::LfoPhase(i),
                ModDest::LfoTempo(i),
                ModDest::LfoSmoothTime(i),
                ModDest::LfoDelayTime(i),
                ModDest::LfoFadeTime(i),
                ModDest::LfoStereo(i),
                ModDest::LfoKeytrackTranspose(i),
            ]);
        }
        for i in 0..NUM_RANDOM_LFOS {
            all.extend([
                ModDest::RandomLfoFrequency(i),
                ModDest::RandomLfoTempo(i),
                ModDest::RandomLfoKeytrackTranspose(i),
            ]);
        }
        for i in 0..NUM_OSCILLATORS {
            all.extend([
                ModDest::OscDetuneRange(i),
                ModDest::OscDetunePower(i),
                ModDest::OscUnisonVoices(i),
                ModDest::OscSpectralMorphSpread(i),
                ModDest::OscDistortionSpread(i),
            ]);
        }
        for i in 0..2 {
            all.extend([
                ModDest::FilterFormantX(i),
                ModDest::FilterFormantY(i),
                ModDest::FilterFormantTranspose(i),
                ModDest::FilterFormantSpread(i),
            ]);
        }
        all.extend([
            ModDest::VoiceTune,
            ModDest::VoiceTranspose,
            ModDest::PortamentoTime,
            ModDest::VolumeAmp,
            ModDest::PitchBend,
        ]);
        for slot in 0..MAX_MODULATION_CONNECTIONS {
            all.extend([ModDest::ModulationAmount(slot), ModDest::ModulationPower(slot)]);
        }
        all
    }

    /// Destinations the kernel consumes sample by sample. Connections from
    /// an audio-rate-capable source into one of these are evaluated at
    /// audio rate ([`ModMatrix::resolve_audio`]); every other connection is
    /// control rate, with its offset ramped across the block by the kernel.
    pub fn is_audio_rate(self) -> bool {
        // The reference's `createPolyModControl(..., audio_rate = true)`
        // controls: the filter cutoff and, on each oscillator, level,
        // transpose, tune and phase (`OscillatorModule::init`).
        matches!(
            self,
            ModDest::FilterCutoff(_)
                | ModDest::OscLevel(_)
                | ModDest::OscTranspose(_)
                | ModDest::OscTune(_)
                | ModDest::OscPhase(_)
        )
    }
}

/// Accumulated per-lane offsets for one control tick.
#[derive(Clone, Debug, Default)]
pub struct ModOffsets {
    pub osc_level: [PolyF32; NUM_OSCILLATORS],
    pub osc_transpose: [PolyF32; NUM_OSCILLATORS],
    pub osc_tune: [PolyF32; NUM_OSCILLATORS],
    pub osc_frame: [PolyF32; NUM_OSCILLATORS],
    pub osc_frame_spread: [PolyF32; NUM_OSCILLATORS],
    pub osc_pan: [PolyF32; NUM_OSCILLATORS],
    pub osc_unison_detune: [PolyF32; NUM_OSCILLATORS],
    pub osc_unison_blend: [PolyF32; NUM_OSCILLATORS],
    pub osc_stereo_spread: [PolyF32; NUM_OSCILLATORS],
    pub osc_distortion_amount: [PolyF32; NUM_OSCILLATORS],
    pub osc_distortion_phase: [PolyF32; NUM_OSCILLATORS],
    pub osc_spectral_morph_amount: [PolyF32; NUM_OSCILLATORS],
    pub osc_phase: [PolyF32; NUM_OSCILLATORS],
    pub sample_level: PolyF32,
    pub sample_transpose: PolyF32,
    pub sample_tune: PolyF32,
    pub sample_pan: PolyF32,
    pub filter_cutoff: [PolyF32; 2],
    pub filter_resonance: [PolyF32; 2],
    pub filter_drive: [PolyF32; 2],
    pub filter_blend: [PolyF32; 2],
    pub filter_blend_transpose: [PolyF32; 2],
    pub filter_keytrack: [PolyF32; 2],
    pub filter_mix: [PolyF32; 2],
    pub env_delay: [PolyF32; NUM_ENVELOPES],
    pub env_attack: [PolyF32; NUM_ENVELOPES],
    pub env_attack_power: [PolyF32; NUM_ENVELOPES],
    pub env_hold: [PolyF32; NUM_ENVELOPES],
    pub env_decay: [PolyF32; NUM_ENVELOPES],
    pub env_decay_power: [PolyF32; NUM_ENVELOPES],
    pub env_sustain: [PolyF32; NUM_ENVELOPES],
    pub env_release: [PolyF32; NUM_ENVELOPES],
    pub env_release_power: [PolyF32; NUM_ENVELOPES],
    pub lfo_frequency: [PolyF32; NUM_LFOS],
    pub lfo_phase: [PolyF32; NUM_LFOS],
    pub lfo_tempo: [PolyF32; NUM_LFOS],
    pub lfo_smooth_time: [PolyF32; NUM_LFOS],
    pub lfo_delay_time: [PolyF32; NUM_LFOS],
    pub lfo_fade_time: [PolyF32; NUM_LFOS],
    pub lfo_stereo: [PolyF32; NUM_LFOS],
    pub lfo_keytrack_transpose: [PolyF32; NUM_LFOS],
    pub random_lfo_frequency: [PolyF32; NUM_RANDOM_LFOS],
    pub random_lfo_tempo: [PolyF32; NUM_RANDOM_LFOS],
    pub random_lfo_keytrack_transpose: [PolyF32; NUM_RANDOM_LFOS],
    pub osc_detune_range: [PolyF32; NUM_OSCILLATORS],
    pub osc_detune_power: [PolyF32; NUM_OSCILLATORS],
    pub osc_unison_voices: [PolyF32; NUM_OSCILLATORS],
    pub osc_spectral_morph_spread: [PolyF32; NUM_OSCILLATORS],
    pub osc_distortion_spread: [PolyF32; NUM_OSCILLATORS],
    pub filter_formant_x: [PolyF32; 2],
    pub filter_formant_y: [PolyF32; 2],
    pub filter_formant_transpose: [PolyF32; 2],
    pub filter_formant_spread: [PolyF32; 2],
    pub voice_tune: PolyF32,
    pub voice_transpose: PolyF32,
    pub portamento_time: PolyF32,
    pub volume_amp: PolyF32,
    pub pitch_bend: PolyF32,
}

impl ModOffsets {
    pub fn clear(&mut self) {
        *self = ModOffsets::default();
    }

    #[inline]
    fn add(&mut self, dest: ModDest, value: PolyF32) {
        match dest {
            ModDest::OscLevel(i) => self.osc_level[i] += value,
            ModDest::OscTranspose(i) => self.osc_transpose[i] += value,
            ModDest::OscTune(i) => self.osc_tune[i] += value,
            ModDest::OscFrame(i) => self.osc_frame[i] += value,
            ModDest::OscFrameSpread(i) => self.osc_frame_spread[i] += value,
            ModDest::OscPan(i) => self.osc_pan[i] += value,
            ModDest::OscUnisonDetune(i) => self.osc_unison_detune[i] += value,
            ModDest::OscUnisonBlend(i) => self.osc_unison_blend[i] += value,
            ModDest::OscStereoSpread(i) => self.osc_stereo_spread[i] += value,
            ModDest::OscDistortionAmount(i) => self.osc_distortion_amount[i] += value,
            ModDest::OscDistortionPhase(i) => self.osc_distortion_phase[i] += value,
            ModDest::OscSpectralMorphAmount(i) => self.osc_spectral_morph_amount[i] += value,
            ModDest::OscPhase(i) => self.osc_phase[i] += value,
            ModDest::SampleLevel => self.sample_level += value,
            ModDest::SampleTranspose => self.sample_transpose += value,
            ModDest::SampleTune => self.sample_tune += value,
            ModDest::SamplePan => self.sample_pan += value,
            ModDest::FilterCutoff(i) => self.filter_cutoff[i] += value,
            ModDest::FilterResonance(i) => self.filter_resonance[i] += value,
            ModDest::FilterDrive(i) => self.filter_drive[i] += value,
            ModDest::FilterBlend(i) => self.filter_blend[i] += value,
            ModDest::FilterBlendTranspose(i) => self.filter_blend_transpose[i] += value,
            ModDest::FilterKeytrack(i) => self.filter_keytrack[i] += value,
            ModDest::FilterMix(i) => self.filter_mix[i] += value,
            ModDest::EnvDelay(i) => self.env_delay[i] += value,
            ModDest::EnvAttack(i) => self.env_attack[i] += value,
            ModDest::EnvAttackPower(i) => self.env_attack_power[i] += value,
            ModDest::EnvHold(i) => self.env_hold[i] += value,
            ModDest::EnvDecay(i) => self.env_decay[i] += value,
            ModDest::EnvDecayPower(i) => self.env_decay_power[i] += value,
            ModDest::EnvSustain(i) => self.env_sustain[i] += value,
            ModDest::EnvRelease(i) => self.env_release[i] += value,
            ModDest::EnvReleasePower(i) => self.env_release_power[i] += value,
            ModDest::LfoFrequency(i) => self.lfo_frequency[i] += value,
            ModDest::LfoPhase(i) => self.lfo_phase[i] += value,
            ModDest::LfoTempo(i) => self.lfo_tempo[i] += value,
            ModDest::LfoSmoothTime(i) => self.lfo_smooth_time[i] += value,
            ModDest::LfoDelayTime(i) => self.lfo_delay_time[i] += value,
            ModDest::LfoFadeTime(i) => self.lfo_fade_time[i] += value,
            ModDest::LfoStereo(i) => self.lfo_stereo[i] += value,
            ModDest::LfoKeytrackTranspose(i) => self.lfo_keytrack_transpose[i] += value,
            ModDest::RandomLfoFrequency(i) => self.random_lfo_frequency[i] += value,
            ModDest::RandomLfoTempo(i) => self.random_lfo_tempo[i] += value,
            ModDest::RandomLfoKeytrackTranspose(i) => self.random_lfo_keytrack_transpose[i] += value,
            ModDest::OscDetuneRange(i) => self.osc_detune_range[i] += value,
            ModDest::OscDetunePower(i) => self.osc_detune_power[i] += value,
            ModDest::OscUnisonVoices(i) => self.osc_unison_voices[i] += value,
            ModDest::OscSpectralMorphSpread(i) => self.osc_spectral_morph_spread[i] += value,
            ModDest::OscDistortionSpread(i) => self.osc_distortion_spread[i] += value,
            ModDest::FilterFormantX(i) => self.filter_formant_x[i] += value,
            ModDest::FilterFormantY(i) => self.filter_formant_y[i] += value,
            ModDest::FilterFormantTranspose(i) => self.filter_formant_transpose[i] += value,
            ModDest::FilterFormantSpread(i) => self.filter_formant_spread[i] += value,
            ModDest::VoiceTune => self.voice_tune += value,
            ModDest::VoiceTranspose => self.voice_transpose += value,
            ModDest::PortamentoTime => self.portamento_time += value,
            ModDest::VolumeAmp => self.volume_amp += value,
            ModDest::PitchBend => self.pitch_bend += value,
            // Resolved into the matrix's own offset arrays, never here.
            ModDest::ModulationAmount(_) | ModDest::ModulationPower(_) => {}
        }
    }

    /// The control-rate offset summed into `dest` this block — the read
    /// side of `add`, for the bench's bounds check (a case whose modulated
    /// value leaves its range measures the clamp, not the connection).
    /// Zero for the two meta destinations, which live in the matrix.
    pub fn control(&self, dest: ModDest) -> PolyF32 {
        match dest {
            ModDest::OscLevel(i) => self.osc_level[i],
            ModDest::OscTranspose(i) => self.osc_transpose[i],
            ModDest::OscTune(i) => self.osc_tune[i],
            ModDest::OscFrame(i) => self.osc_frame[i],
            ModDest::OscFrameSpread(i) => self.osc_frame_spread[i],
            ModDest::OscPan(i) => self.osc_pan[i],
            ModDest::OscUnisonDetune(i) => self.osc_unison_detune[i],
            ModDest::OscUnisonBlend(i) => self.osc_unison_blend[i],
            ModDest::OscStereoSpread(i) => self.osc_stereo_spread[i],
            ModDest::OscDistortionAmount(i) => self.osc_distortion_amount[i],
            ModDest::OscDistortionPhase(i) => self.osc_distortion_phase[i],
            ModDest::OscSpectralMorphAmount(i) => self.osc_spectral_morph_amount[i],
            ModDest::OscPhase(i) => self.osc_phase[i],
            ModDest::SampleLevel => self.sample_level,
            ModDest::SampleTranspose => self.sample_transpose,
            ModDest::SampleTune => self.sample_tune,
            ModDest::SamplePan => self.sample_pan,
            ModDest::FilterCutoff(i) => self.filter_cutoff[i],
            ModDest::FilterResonance(i) => self.filter_resonance[i],
            ModDest::FilterDrive(i) => self.filter_drive[i],
            ModDest::FilterBlend(i) => self.filter_blend[i],
            ModDest::FilterBlendTranspose(i) => self.filter_blend_transpose[i],
            ModDest::FilterKeytrack(i) => self.filter_keytrack[i],
            ModDest::FilterMix(i) => self.filter_mix[i],
            ModDest::EnvDelay(i) => self.env_delay[i],
            ModDest::EnvAttack(i) => self.env_attack[i],
            ModDest::EnvAttackPower(i) => self.env_attack_power[i],
            ModDest::EnvHold(i) => self.env_hold[i],
            ModDest::EnvDecay(i) => self.env_decay[i],
            ModDest::EnvDecayPower(i) => self.env_decay_power[i],
            ModDest::EnvSustain(i) => self.env_sustain[i],
            ModDest::EnvRelease(i) => self.env_release[i],
            ModDest::EnvReleasePower(i) => self.env_release_power[i],
            ModDest::LfoFrequency(i) => self.lfo_frequency[i],
            ModDest::LfoPhase(i) => self.lfo_phase[i],
            ModDest::LfoTempo(i) => self.lfo_tempo[i],
            ModDest::LfoSmoothTime(i) => self.lfo_smooth_time[i],
            ModDest::LfoDelayTime(i) => self.lfo_delay_time[i],
            ModDest::LfoFadeTime(i) => self.lfo_fade_time[i],
            ModDest::LfoStereo(i) => self.lfo_stereo[i],
            ModDest::LfoKeytrackTranspose(i) => self.lfo_keytrack_transpose[i],
            ModDest::RandomLfoFrequency(i) => self.random_lfo_frequency[i],
            ModDest::RandomLfoTempo(i) => self.random_lfo_tempo[i],
            ModDest::RandomLfoKeytrackTranspose(i) => self.random_lfo_keytrack_transpose[i],
            ModDest::OscDetuneRange(i) => self.osc_detune_range[i],
            ModDest::OscDetunePower(i) => self.osc_detune_power[i],
            ModDest::OscUnisonVoices(i) => self.osc_unison_voices[i],
            ModDest::OscSpectralMorphSpread(i) => self.osc_spectral_morph_spread[i],
            ModDest::OscDistortionSpread(i) => self.osc_distortion_spread[i],
            ModDest::FilterFormantX(i) => self.filter_formant_x[i],
            ModDest::FilterFormantY(i) => self.filter_formant_y[i],
            ModDest::FilterFormantTranspose(i) => self.filter_formant_transpose[i],
            ModDest::FilterFormantSpread(i) => self.filter_formant_spread[i],
            ModDest::VoiceTune => self.voice_tune,
            ModDest::VoiceTranspose => self.voice_transpose,
            ModDest::PortamentoTime => self.portamento_time,
            ModDest::VolumeAmp => self.volume_amp,
            ModDest::PitchBend => self.pitch_bend,
            ModDest::ModulationAmount(_) | ModDest::ModulationPower(_) => PolyF32::ZERO,
        }
    }
}

/// One active modulation connection.
#[derive(Clone, Debug)]
pub struct Connection {
    pub source: ModSource,
    pub dest: ModDest,
    pub transform: ModulationTransform,
}

impl Connection {
    /// The slot this connection modulates the amount or power of, when it
    /// is a meta connection.
    pub fn meta_target_slot(&self) -> Option<usize> {
        match self.dest {
            ModDest::ModulationAmount(slot) | ModDest::ModulationPower(slot) => Some(slot),
            _ => None,
        }
    }

    /// True when this connection is evaluated sample by sample.
    pub fn is_audio_rate(&self) -> bool {
        self.source.is_audio_rate_capable() && self.dest.is_audio_rate()
    }
}

/// Which modulators must be rendered at audio rate this block, as bitmasks
/// over the envelope / LFO indices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AudioRateSources {
    pub envelopes: u32,
    pub lfos: u32,
}

impl AudioRateSources {
    pub fn envelope(&self, index: usize) -> bool {
        self.envelopes & (1 << index) != 0
    }

    pub fn lfo(&self, index: usize) -> bool {
        self.lfos & (1 << index) != 0
    }
}

/// Per-sample buffers for the audio-rate destinations, one per lane of each
/// family, filled by [`ModMatrix::resolve_audio`] with the sum of the
/// audio-rate connections into that destination.
pub struct AudioDestBuffers {
    pub filter_cutoff: [Vec<PolyF32>; 2],
    pub osc_level: [Vec<PolyF32>; NUM_OSCILLATORS],
    pub osc_transpose: [Vec<PolyF32>; NUM_OSCILLATORS],
    pub osc_tune: [Vec<PolyF32>; NUM_OSCILLATORS],
    pub osc_phase: [Vec<PolyF32>; NUM_OSCILLATORS],
}

impl AudioDestBuffers {
    pub fn new(capacity: usize) -> Self {
        let make = |_| vec![PolyF32::ZERO; capacity];
        Self {
            filter_cutoff: core::array::from_fn(make),
            osc_level: core::array::from_fn(make),
            osc_transpose: core::array::from_fn(make),
            osc_tune: core::array::from_fn(make),
            osc_phase: core::array::from_fn(make),
        }
    }

    /// The buffer of an audio-rate destination; `None` for a control-rate
    /// one.
    pub fn get(&self, dest: ModDest) -> Option<&[PolyF32]> {
        Some(match dest {
            ModDest::FilterCutoff(i) => &self.filter_cutoff[i][..],
            ModDest::OscLevel(i) => &self.osc_level[i][..],
            ModDest::OscTranspose(i) => &self.osc_transpose[i][..],
            ModDest::OscTune(i) => &self.osc_tune[i][..],
            ModDest::OscPhase(i) => &self.osc_phase[i][..],
            _ => return None,
        })
    }

    fn get_mut(&mut self, dest: ModDest) -> Option<&mut [PolyF32]> {
        Some(match dest {
            ModDest::FilterCutoff(i) => &mut self.filter_cutoff[i][..],
            ModDest::OscLevel(i) => &mut self.osc_level[i][..],
            ModDest::OscTranspose(i) => &mut self.osc_transpose[i][..],
            ModDest::OscTune(i) => &mut self.osc_tune[i][..],
            ModDest::OscPhase(i) => &mut self.osc_phase[i][..],
            _ => return None,
        })
    }

    fn clear(&mut self, num_samples: usize) {
        let families: [&mut [Vec<PolyF32>]; 5] = [
            &mut self.filter_cutoff,
            &mut self.osc_level,
            &mut self.osc_transpose,
            &mut self.osc_tune,
            &mut self.osc_phase,
        ];
        for family in families {
            for buffer in family.iter_mut() {
                buffer[..num_samples].fill(PolyF32::ZERO);
            }
        }
    }
}

/// Audio-rate source buffers for one block (one buffer per envelope / LFO,
/// only those flagged by [`ModMatrix::audio_rate_sources`] need to hold
/// valid data).
pub struct AudioSourceBuffers<'a> {
    pub envelopes: &'a [Vec<PolyF32>; NUM_ENVELOPES],
    pub lfos: &'a [Vec<PolyF32>; NUM_LFOS],
}

impl AudioSourceBuffers<'_> {
    fn get(&self, source: ModSource, num_samples: usize) -> Option<&[PolyF32]> {
        match source {
            ModSource::Envelope(i) => Some(&self.envelopes[i][..num_samples]),
            ModSource::Lfo(i) => Some(&self.lfos[i][..num_samples]),
            _ => None,
        }
    }
}

/// Values of every modulation source for one control tick.
#[derive(Clone, Debug, Default)]
pub struct SourceValues {
    pub envelopes: [PolyF32; NUM_ENVELOPES],
    pub lfos: [PolyF32; NUM_LFOS],
    pub random_lfos: [PolyF32; NUM_RANDOM_LFOS],
    pub macros: [PolyF32; NUM_MACROS],
    pub note: PolyF32,
    pub note_in_octave: PolyF32,
    pub velocity: PolyF32,
    pub lift: PolyF32,
    pub mod_wheel: PolyF32,
    pub pitch_wheel: PolyF32,
    pub aftertouch: PolyF32,
    pub slide: PolyF32,
    pub random: PolyF32,
    pub stereo: PolyF32,
}

impl SourceValues {
    #[inline]
    pub fn get(&self, source: ModSource) -> PolyF32 {
        match source {
            ModSource::Envelope(i) => self.envelopes[i],
            ModSource::Lfo(i) => self.lfos[i],
            ModSource::RandomLfo(i) => self.random_lfos[i],
            ModSource::Macro(i) => self.macros[i],
            ModSource::Note => self.note,
            ModSource::NoteInOctave => self.note_in_octave,
            ModSource::Velocity => self.velocity,
            ModSource::Lift => self.lift,
            ModSource::ModWheel => self.mod_wheel,
            ModSource::PitchWheel => self.pitch_wheel,
            ModSource::Aftertouch => self.aftertouch,
            ModSource::Slide => self.slide,
            ModSource::Random => self.random,
            ModSource::Stereo => self.stereo,
        }
    }
}

/// The matrix itself: a list of connections resolved each control tick.
///
/// `connections` is preallocated to [`MAX_MODULATION_CONNECTIONS`]; fill it
/// through [`ModMatrix::set_connections`] on the audio thread (no
/// reallocation). Assigning a fresh `Vec` still works but allocates.
///
/// A connection can target another connection's amount or power
/// ([`ModDest::ModulationAmount`], [`ModDest::ModulationPower`]) — the
/// reference's meta-modulation, measured in notes/meta-modulation.md.
/// Those are resolved in dependency order: a connection whose amount is
/// modulated is evaluated after the connections that modulate it, whatever
/// their slot numbers (the reference's ProcessorRouter orders by
/// dependency too; a chain wired in either slot order renders
/// byte-identically). Connections on a cycle keep the previous block's
/// offsets: bounded, deterministic, one block of lag inside the cycle.
#[derive(Clone, Debug)]
pub struct ModMatrix {
    pub connections: Vec<Connection>,
    /// Per slot, what meta connections added to that slot's amount and
    /// power this block (lanes = voices), and the previous block's for the
    /// connections on a cycle.
    amount_offsets: [PolyF32; MAX_MODULATION_CONNECTIONS],
    power_offsets: [PolyF32; MAX_MODULATION_CONNECTIONS],
    previous_amount_offsets: [PolyF32; MAX_MODULATION_CONNECTIONS],
    previous_power_offsets: [PolyF32; MAX_MODULATION_CONNECTIONS],
    /// Evaluation order (indices into `connections`) and how many of them
    /// are acyclic; recomputed each block, no allocation.
    order: [usize; MAX_MODULATION_CONNECTIONS],
    acyclic: usize,
}

impl Default for ModMatrix {
    fn default() -> ModMatrix {
        ModMatrix {
            connections: Vec::with_capacity(MAX_MODULATION_CONNECTIONS),
            amount_offsets: [PolyF32::ZERO; MAX_MODULATION_CONNECTIONS],
            power_offsets: [PolyF32::ZERO; MAX_MODULATION_CONNECTIONS],
            previous_amount_offsets: [PolyF32::ZERO; MAX_MODULATION_CONNECTIONS],
            previous_power_offsets: [PolyF32::ZERO; MAX_MODULATION_CONNECTIONS],
            order: [0; MAX_MODULATION_CONNECTIONS],
            acyclic: 0,
        }
    }
}

impl ModMatrix {
    /// Replaces the connection list without reallocating: clears, then
    /// copies at most [`MAX_MODULATION_CONNECTIONS`] entries (the rest are
    /// dropped). Safe on the audio thread as long as the list's capacity
    /// was reserved (it is, by `Default`).
    pub fn set_connections(&mut self, connections: &[Connection]) {
        self.connections.clear();
        if self.connections.capacity() < MAX_MODULATION_CONNECTIONS {
            // Only reached when a caller assigned a smaller Vec directly.
            self.connections.reserve_exact(MAX_MODULATION_CONNECTIONS);
        }
        let count = connections.len().min(MAX_MODULATION_CONNECTIONS);
        self.connections.extend_from_slice(&connections[..count]);
    }

    /// Which envelopes / LFOs feed an audio-rate destination and therefore
    /// must be rendered sample by sample this block.
    pub fn audio_rate_sources(&self) -> AudioRateSources {
        let mut sources = AudioRateSources::default();
        for connection in &self.connections {
            if !connection.is_audio_rate() {
                continue;
            }
            match connection.source {
                ModSource::Envelope(i) => sources.envelopes |= 1 << i,
                ModSource::Lfo(i) => sources.lfos |= 1 << i,
                _ => {}
            }
        }
        sources
    }

    /// The meta-modulation offsets on every slot's amount, as resolved
    /// this block (lanes = voices). The effects matrix reads its
    /// connections' slots from here.
    pub fn amount_offsets(&self) -> &[PolyF32; MAX_MODULATION_CONNECTIONS] {
        &self.amount_offsets
    }

    pub fn power_offsets(&self) -> &[PolyF32; MAX_MODULATION_CONNECTIONS] {
        &self.power_offsets
    }

    /// Orders the connections so that every connection whose amount or
    /// power is modulated comes after the connections modulating it
    /// (Kahn's algorithm on "meta connection -> target slot"). What
    /// cannot be ordered — a cycle — is appended in slot order and
    /// counted separately. No allocation: fixed arrays, n <= 64.
    #[allow(clippy::needless_range_loop)] // indices are the graph's nodes
    fn compute_order(&mut self) {
        let n = self.connections.len().min(MAX_MODULATION_CONNECTIONS);
        let mut indegree = [0u8; MAX_MODULATION_CONNECTIONS];
        let mut placed = [false; MAX_MODULATION_CONNECTIONS];
        // A meta connection j raises the in-degree of every connection
        // whose slot it targets.
        for j in 0..n {
            if let Some(target) = self.connections[j].meta_target_slot() {
                for k in 0..n {
                    if self.connections[k].transform.slot == target {
                        indegree[k] = indegree[k].saturating_add(1);
                    }
                }
            }
        }
        let mut count = 0;
        loop {
            let mut progressed = false;
            for k in 0..n {
                if !placed[k] && indegree[k] == 0 {
                    placed[k] = true;
                    self.order[count] = k;
                    count += 1;
                    progressed = true;
                    if let Some(target) = self.connections[k].meta_target_slot() {
                        for m in 0..n {
                            if self.connections[m].transform.slot == target {
                                indegree[m] = indegree[m].saturating_sub(1);
                            }
                        }
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        self.acyclic = count;
        for k in 0..n {
            if !placed[k] {
                self.order[count] = k;
                count += 1;
            }
        }
    }

    /// Resolves every control-rate connection into `offsets` (cleared
    /// first), meta connections first (see the type's docs). Audio-rate
    /// connections (see [`Connection::is_audio_rate`]) are skipped here
    /// and evaluated by [`Self::resolve_audio`], with the amount and power
    /// offsets this pass computed for them.
    pub fn resolve(
        &mut self,
        sources: &SourceValues,
        offsets: &mut ModOffsets,
        _reset_mask: PolyMask,
    ) {
        offsets.clear();
        self.compute_order();
        self.previous_amount_offsets = self.amount_offsets;
        self.previous_power_offsets = self.power_offsets;
        self.amount_offsets = [PolyF32::ZERO; MAX_MODULATION_CONNECTIONS];
        self.power_offsets = [PolyF32::ZERO; MAX_MODULATION_CONNECTIONS];
        let n = self.connections.len().min(MAX_MODULATION_CONNECTIONS);
        for position in 0..n {
            let index = self.order[position];
            let slot = self.connections[index].transform.slot.min(MAX_MODULATION_CONNECTIONS - 1);
            // On a cycle the offsets of this block are incomplete by
            // construction; the previous block's are whole. A connection
            // from a mono source reads the previous block's too — that is
            // where the reference evaluates it (`ModSource::is_mono`).
            let lagged = position >= self.acyclic || self.connections[index].source.is_mono();
            let (amount_offset, power_offset) = if lagged {
                (self.previous_amount_offsets[slot], self.previous_power_offsets[slot])
            } else {
                (self.amount_offsets[slot], self.power_offsets[slot])
            };
            let connection = &mut self.connections[index];
            connection.transform.amount_offset = amount_offset;
            connection.transform.power_offset = power_offset;
            if connection.is_audio_rate() {
                continue;
            }
            let value = sources.get(connection.source);
            let output = connection.transform.process_control(value);
            match connection.dest {
                ModDest::ModulationAmount(target) => {
                    if target < MAX_MODULATION_CONNECTIONS {
                        self.amount_offsets[target] += output.scaled;
                    }
                }
                ModDest::ModulationPower(target) => {
                    if target < MAX_MODULATION_CONNECTIONS {
                        self.power_offsets[target] += output.scaled;
                    }
                }
                dest => offsets.add(dest, output.scaled),
            }
        }
    }

    /// Evaluates the audio-rate connections sample by sample: every buffer
    /// in `dests` is cleared, then receives the sum of the audio-rate
    /// connections targeting it (amount / power smoothing per sample inside
    /// the transform, jumping on `reset_mask` lanes). `scratch` must hold at
    /// least `num_samples` entries.
    pub fn resolve_audio(
        &mut self,
        sources: &AudioSourceBuffers,
        num_samples: usize,
        reset_mask: PolyMask,
        scratch: &mut [PolyF32],
        dests: &mut AudioDestBuffers,
    ) {
        dests.clear(num_samples);
        let scratch = &mut scratch[..num_samples];
        for connection in &mut self.connections {
            if !connection.is_audio_rate() {
                continue;
            }
            let Some(source) = sources.get(connection.source, num_samples) else { continue };
            connection.transform.process_audio(source, scratch, reset_mask);
            let buffer =
                dests.get_mut(connection.dest).expect("audio-rate destination without a buffer");
            for (dest, &value) in buffer[..num_samples].iter_mut().zip(&*scratch) {
                *dest += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_routes_source_to_destination() {
        let mut matrix = ModMatrix::default();
        // A macro is control rate even into the (audio-rate) cutoff.
        matrix.connections.push(Connection {
            source: ModSource::Macro(0),
            dest: ModDest::FilterCutoff(0),
            transform: ModulationTransform::with_amount(1.0, 128.0),
        });

        let mut sources = SourceValues::default();
        sources.macros[0] = PolyF32::splat(0.5);
        let mut offsets = ModOffsets::default();
        matrix.resolve(&sources, &mut offsets, PolyMask::NONE);

        assert!((offsets.filter_cutoff[0].lane(0) - 64.0).abs() < 1e-3);
        assert_eq!(offsets.filter_cutoff[1].lane(0), 0.0);
    }

    #[test]
    fn expanded_destinations_route_to_their_offsets() {
        let mut matrix = ModMatrix::default();
        for dest in [
            ModDest::EnvAttackPower(2),
            ModDest::OscStereoSpread(1),
            ModDest::FilterBlendTranspose(0),
            ModDest::SampleTranspose,
            ModDest::LfoPhase(3),
            ModDest::RandomLfoFrequency(1),
        ] {
            matrix.connections.push(Connection {
                source: ModSource::Macro(0),
                dest,
                transform: ModulationTransform::with_amount(1.0, 2.0),
            });
        }

        let mut sources = SourceValues::default();
        sources.macros[0] = PolyF32::splat(0.5);
        let mut offsets = ModOffsets::default();
        matrix.resolve(&sources, &mut offsets, PolyMask::NONE);

        for (value, untouched) in [
            (offsets.env_attack_power[2], offsets.env_attack_power[0]),
            (offsets.osc_stereo_spread[1], offsets.osc_stereo_spread[0]),
            (offsets.filter_blend_transpose[0], offsets.filter_blend_transpose[1]),
            (offsets.sample_transpose, offsets.sample_tune),
            (offsets.lfo_phase[3], offsets.lfo_phase[0]),
            (offsets.random_lfo_frequency[1], offsets.random_lfo_frequency[0]),
        ] {
            assert!((value.lane(0) - 1.0).abs() < 1e-5, "offset missing: {value:?}");
            assert_eq!(untouched.lane(0), 0.0);
        }
    }

    /// The raised limits: a 12th LFO, an 8th envelope, an 8th macro and the
    /// 4th oscillator slot all route through the matrix.
    #[test]
    fn raised_limit_indices_route_through_the_matrix() {
        let mut matrix = ModMatrix::default();
        // An LFO into a level would go audio-rate (the level is a
        // per-sample destination, like the cutoff); pan stays control-rate.
        matrix.connections.push(Connection {
            source: ModSource::Lfo(NUM_LFOS - 1),
            dest: ModDest::OscPan(NUM_OSCILLATORS - 1),
            transform: ModulationTransform::with_amount(1.0, 2.0),
        });
        matrix.connections.push(Connection {
            source: ModSource::Envelope(NUM_ENVELOPES - 1),
            dest: ModDest::LfoFrequency(NUM_LFOS - 1),
            transform: ModulationTransform::with_amount(1.0, 2.0),
        });
        matrix.connections.push(Connection {
            source: ModSource::Macro(NUM_MACROS - 1),
            dest: ModDest::EnvAttack(NUM_ENVELOPES - 1),
            transform: ModulationTransform::with_amount(1.0, 2.0),
        });

        let mut sources = SourceValues::default();
        sources.lfos[NUM_LFOS - 1] = PolyF32::splat(0.5);
        sources.envelopes[NUM_ENVELOPES - 1] = PolyF32::splat(0.5);
        sources.macros[NUM_MACROS - 1] = PolyF32::splat(0.5);
        let mut offsets = ModOffsets::default();
        matrix.resolve(&sources, &mut offsets, PolyMask::NONE);

        assert!((offsets.osc_pan[NUM_OSCILLATORS - 1].lane(0) - 1.0).abs() < 1e-5);
        assert!((offsets.lfo_frequency[NUM_LFOS - 1].lane(0) - 1.0).abs() < 1e-5);
        assert!((offsets.env_attack[NUM_ENVELOPES - 1].lane(0) - 1.0).abs() < 1e-5);
        assert_eq!(offsets.osc_pan[0].lane(0), 0.0);
        assert_eq!(offsets.lfo_frequency[0].lane(0), 0.0);
        assert_eq!(offsets.env_attack[0].lane(0), 0.0);
    }

    #[test]
    fn set_connections_never_reallocates_and_truncates_at_capacity() {
        let mut matrix = ModMatrix::default();
        let capacity = matrix.connections.capacity();
        assert!(capacity >= MAX_MODULATION_CONNECTIONS);
        let connection = Connection {
            source: ModSource::Macro(0),
            dest: ModDest::OscLevel(0),
            transform: ModulationTransform::with_amount(1.0, 1.0),
        };
        let too_many = vec![connection; MAX_MODULATION_CONNECTIONS + 10];
        matrix.set_connections(&too_many);
        assert_eq!(matrix.connections.len(), MAX_MODULATION_CONNECTIONS);
        assert_eq!(matrix.connections.capacity(), capacity);
        matrix.set_connections(&too_many[..3]);
        assert_eq!(matrix.connections.len(), 3);
        assert_eq!(matrix.connections.capacity(), capacity);
    }

    #[test]
    fn audio_rate_connections_are_split_from_control_rate() {
        let mut matrix = ModMatrix::default();
        matrix.connections.push(Connection {
            source: ModSource::Envelope(1),
            dest: ModDest::FilterCutoff(0),
            transform: ModulationTransform::with_amount(1.0, 10.0),
        });
        matrix.connections.push(Connection {
            source: ModSource::Macro(0),
            dest: ModDest::FilterCutoff(0),
            transform: ModulationTransform::with_amount(1.0, 10.0),
        });
        let flagged = matrix.audio_rate_sources();
        assert!(flagged.envelope(1));
        assert!(!flagged.envelope(0));
        assert_eq!(flagged.lfos, 0);

        // Control-rate pass: only the macro contributes.
        let mut sources = SourceValues::default();
        sources.envelopes[1] = PolyF32::ONE;
        sources.macros[0] = PolyF32::splat(0.5);
        let mut offsets = ModOffsets::default();
        matrix.resolve(&sources, &mut offsets, PolyMask::NONE);
        assert!((offsets.filter_cutoff[0].lane(0) - 5.0).abs() < 1e-5);

        // Audio-rate pass: the envelope buffer drives the cutoff buffer.
        let envelopes: [Vec<PolyF32>; NUM_ENVELOPES] =
            core::array::from_fn(|i| vec![PolyF32::splat(if i == 1 { 0.5 } else { 9.0 }); 16]);
        let lfos: [Vec<PolyF32>; NUM_LFOS] = core::array::from_fn(|_| vec![PolyF32::ZERO; 16]);
        let mut scratch = vec![PolyF32::ZERO; 16];
        let mut dests = AudioDestBuffers::new(16);
        dests.filter_cutoff[0].fill(PolyF32::splat(123.0));
        dests.filter_cutoff[1].fill(PolyF32::splat(123.0));
        matrix.resolve_audio(
            &AudioSourceBuffers { envelopes: &envelopes, lfos: &lfos },
            16,
            PolyMask::all_on(),
            &mut scratch,
            &mut dests,
        );
        // Reset lanes jump straight to the target amount: 0.5 * 10.
        let cutoff0 = &dests.filter_cutoff[0];
        assert!((cutoff0[0].lane(0) - 5.0).abs() < 1e-4);
        assert!((cutoff0[15].lane(0) - 5.0).abs() < 1e-4);
        assert_eq!(dests.filter_cutoff[1][7].lane(0), 0.0, "untargeted buffer must be cleared");
    }

    #[test]
    fn control_reads_back_what_add_summed() {
        // `control` is a second match over the destinations; a variant
        // added to `add` and forgotten here would read zero forever.
        let mut offsets = ModOffsets::default();
        let mut expected = 0.0;
        for (n, dest) in ModDest::every().into_iter().enumerate() {
            let value = n as f32 + 1.0;
            offsets.add(dest, PolyF32::splat(value));
            match dest {
                ModDest::ModulationAmount(_) | ModDest::ModulationPower(_) => {
                    assert_eq!(offsets.control(dest).lane(0), 0.0, "{dest:?}");
                }
                _ => {
                    assert_eq!(offsets.control(dest).lane(0), value, "{dest:?}");
                    expected += value;
                }
            }
        }
        assert!(expected > 0.0);
    }

    #[test]
    fn every_audio_rate_destination_has_a_buffer() {
        // The audio-rate pass indexes `AudioDestBuffers` by destination; a
        // destination flagged audio rate without a buffer would panic there.
        let dests = AudioDestBuffers::new(4);
        for i in 0..NUM_OSCILLATORS {
            for dest in [
                ModDest::OscLevel(i),
                ModDest::OscTranspose(i),
                ModDest::OscTune(i),
                ModDest::OscPhase(i),
            ] {
                assert!(dest.is_audio_rate(), "{dest:?}");
                assert!(dests.get(dest).is_some(), "{dest:?} has no buffer");
            }
            assert!(!ModDest::OscPan(i).is_audio_rate());
            assert!(dests.get(ModDest::OscPan(i)).is_none());
        }
        for i in 0..2 {
            assert!(ModDest::FilterCutoff(i).is_audio_rate());
            assert!(dests.get(ModDest::FilterCutoff(i)).is_some());
        }
    }

    #[test]
    fn connections_sum_on_shared_destination() {
        // Two control-rate connections into one control-rate destination
        // (pan; a level would send the envelope's share to audio rate).
        let mut matrix = ModMatrix::default();
        for source in [ModSource::Macro(1), ModSource::Macro(0)] {
            matrix.connections.push(Connection {
                source,
                dest: ModDest::OscPan(0),
                transform: ModulationTransform::with_amount(1.0, 1.0),
            });
        }
        let mut sources = SourceValues::default();
        sources.macros[1] = PolyF32::splat(0.25);
        sources.macros[0] = PolyF32::splat(0.5);
        let mut offsets = ModOffsets::default();
        matrix.resolve(&sources, &mut offsets, PolyMask::NONE);
        assert!((offsets.osc_pan[0].lane(0) - 0.75).abs() < 1e-4);
    }

    #[test]
    fn an_envelope_into_a_level_is_an_audio_rate_connection() {
        let connection = Connection {
            source: ModSource::Envelope(1),
            dest: ModDest::OscLevel(0),
            transform: ModulationTransform::with_amount(1.0, 1.0),
        };
        assert!(connection.is_audio_rate());
        let macro_connection = Connection { source: ModSource::Macro(0), ..connection };
        assert!(!macro_connection.is_audio_rate(), "a macro is control rate into anything");
    }
}

