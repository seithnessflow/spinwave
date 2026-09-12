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
    LfoFrequency(usize),
    LfoPhase(usize),
    RandomLfoFrequency(usize),
    /// Per-voice amplitude offset (the reference's modulatable
    /// `voice_amplitude`): added to `KernelParams::voice_amplitude` BEFORE
    /// the amplitude law squares it, destination scale 1.0. This is NOT the
    /// master `volume` parameter — a preset's "volume" destination belongs
    /// to the master path.
    VolumeAmp,
    PitchBend,
}

impl ModDest {
    /// Destinations the kernel consumes sample by sample. Connections from
    /// an audio-rate-capable source into one of these are evaluated at
    /// audio rate ([`ModMatrix::resolve_audio`]); every other connection is
    /// control rate, with its offset ramped across the block by the kernel.
    pub fn is_audio_rate(self) -> bool {
        // Both are `createPolyModControl(..., audio_rate = true)` in the
        // reference: the filter cutoff and the oscillator level.
        matches!(self, ModDest::FilterCutoff(_) | ModDest::OscLevel(_))
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
    pub random_lfo_frequency: [PolyF32; NUM_RANDOM_LFOS],
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
            ModDest::RandomLfoFrequency(i) => self.random_lfo_frequency[i] += value,
            ModDest::VolumeAmp => self.volume_amp += value,
            ModDest::PitchBend => self.pitch_bend += value,
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
#[derive(Clone, Debug)]
pub struct ModMatrix {
    pub connections: Vec<Connection>,
}

impl Default for ModMatrix {
    fn default() -> ModMatrix {
        ModMatrix { connections: Vec::with_capacity(MAX_MODULATION_CONNECTIONS) }
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

    /// Resolves every control-rate connection into `offsets` (cleared
    /// first). Audio-rate connections (see [`Connection::is_audio_rate`])
    /// are skipped here and evaluated by [`Self::resolve_audio`].
    pub fn resolve(
        &mut self,
        sources: &SourceValues,
        offsets: &mut ModOffsets,
        _reset_mask: PolyMask,
    ) {
        offsets.clear();
        for connection in &mut self.connections {
            if connection.is_audio_rate() {
                continue;
            }
            let value = sources.get(connection.source);
            let output = connection.transform.process_control(value);
            offsets.add(connection.dest, output.scaled);
        }
    }

    /// Evaluates the audio-rate connections sample by sample: each filter
    /// cutoff buffer in `filter_cutoff` is cleared, then receives the sum of
    /// every audio-rate connection targeting it (amount / power smoothing
    /// per sample inside the transform, jumping on `reset_mask` lanes).
    /// `scratch` must hold at least `num_samples` entries.
    pub fn resolve_audio(
        &mut self,
        sources: &AudioSourceBuffers,
        num_samples: usize,
        reset_mask: PolyMask,
        scratch: &mut [PolyF32],
        filter_cutoff: &mut [&mut [PolyF32]; 2],
        osc_level: &mut [&mut [PolyF32]; NUM_OSCILLATORS],
    ) {
        for buffer in filter_cutoff.iter_mut() {
            buffer[..num_samples].fill(PolyF32::ZERO);
        }
        for buffer in osc_level.iter_mut() {
            buffer[..num_samples].fill(PolyF32::ZERO);
        }
        let scratch = &mut scratch[..num_samples];
        for connection in &mut self.connections {
            if !connection.is_audio_rate() {
                continue;
            }
            let Some(source) = sources.get(connection.source, num_samples) else { continue };
            connection.transform.process_audio(source, scratch, reset_mask);
            match connection.dest {
                ModDest::FilterCutoff(i) => {
                    for (dest, &value) in filter_cutoff[i][..num_samples].iter_mut().zip(&*scratch) {
                        *dest += value;
                    }
                }
                ModDest::OscLevel(i) => {
                    for (dest, &value) in osc_level[i][..num_samples].iter_mut().zip(&*scratch) {
                        *dest += value;
                    }
                }
                _ => unreachable!("audio-rate destination without a buffer"),
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
        let mut cutoff0 = vec![PolyF32::splat(123.0); 16];
        let mut cutoff1 = vec![PolyF32::splat(123.0); 16];
        let mut levels: [Vec<PolyF32>; NUM_OSCILLATORS] = core::array::from_fn(|_| vec![PolyF32::ZERO; 16]);
        let [l0, l1, l2, l3] = &mut levels;
        matrix.resolve_audio(
            &AudioSourceBuffers { envelopes: &envelopes, lfos: &lfos },
            16,
            PolyMask::all_on(),
            &mut scratch,
            &mut [&mut cutoff0, &mut cutoff1],
            &mut [&mut l0[..], &mut l1[..], &mut l2[..], &mut l3[..]],
        );
        // Reset lanes jump straight to the target amount: 0.5 * 10.
        assert!((cutoff0[0].lane(0) - 5.0).abs() < 1e-4);
        assert!((cutoff0[15].lane(0) - 5.0).abs() < 1e-4);
        assert_eq!(cutoff1[7].lane(0), 0.0, "untargeted buffer must be cleared");
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

