//! The per-voice modulation matrix: routes modulator outputs into
//! parameter offsets through [`ModulationTransform`]s.
//!
//! Rework of Vital's dynamic modulation plumbing: sources and destinations
//! are enums, offsets accumulate into a plain struct the kernel adds to its
//! base parameter values each block.

use crate::modulation::ModulationTransform;
use vital_poly::{PolyF32, PolyMask};

pub const NUM_OSCILLATORS: usize = 3;
pub const NUM_ENVELOPES: usize = 6;
pub const NUM_LFOS: usize = 8;
pub const NUM_RANDOM_LFOS: usize = 4;
pub const NUM_MACROS: usize = 4;

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

/// Curated destination set for the first kernel iteration; grows toward
/// the full ~400-destination table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModDest {
    OscLevel(usize),
    OscTranspose(usize),
    OscTune(usize),
    OscFrame(usize),
    OscPan(usize),
    OscUnisonDetune(usize),
    OscDistortionAmount(usize),
    OscSpectralMorphAmount(usize),
    OscPhase(usize),
    SampleLevel,
    FilterCutoff(usize),
    FilterResonance(usize),
    FilterDrive(usize),
    FilterBlend(usize),
    FilterMix(usize),
    EnvAttack(usize),
    EnvDecay(usize),
    EnvSustain(usize),
    EnvRelease(usize),
    LfoFrequency(usize),
    VolumeAmp,
    PitchBend,
}

/// Accumulated per-lane offsets for one control tick.
#[derive(Clone, Debug, Default)]
pub struct ModOffsets {
    pub osc_level: [PolyF32; NUM_OSCILLATORS],
    pub osc_transpose: [PolyF32; NUM_OSCILLATORS],
    pub osc_tune: [PolyF32; NUM_OSCILLATORS],
    pub osc_frame: [PolyF32; NUM_OSCILLATORS],
    pub osc_pan: [PolyF32; NUM_OSCILLATORS],
    pub osc_unison_detune: [PolyF32; NUM_OSCILLATORS],
    pub osc_distortion_amount: [PolyF32; NUM_OSCILLATORS],
    pub osc_spectral_morph_amount: [PolyF32; NUM_OSCILLATORS],
    pub osc_phase: [PolyF32; NUM_OSCILLATORS],
    pub sample_level: PolyF32,
    pub filter_cutoff: [PolyF32; 2],
    pub filter_resonance: [PolyF32; 2],
    pub filter_drive: [PolyF32; 2],
    pub filter_blend: [PolyF32; 2],
    pub filter_mix: [PolyF32; 2],
    pub env_attack: [PolyF32; NUM_ENVELOPES],
    pub env_decay: [PolyF32; NUM_ENVELOPES],
    pub env_sustain: [PolyF32; NUM_ENVELOPES],
    pub env_release: [PolyF32; NUM_ENVELOPES],
    pub lfo_frequency: [PolyF32; NUM_LFOS],
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
            ModDest::OscPan(i) => self.osc_pan[i] += value,
            ModDest::OscUnisonDetune(i) => self.osc_unison_detune[i] += value,
            ModDest::OscDistortionAmount(i) => self.osc_distortion_amount[i] += value,
            ModDest::OscSpectralMorphAmount(i) => self.osc_spectral_morph_amount[i] += value,
            ModDest::OscPhase(i) => self.osc_phase[i] += value,
            ModDest::SampleLevel => self.sample_level += value,
            ModDest::FilterCutoff(i) => self.filter_cutoff[i] += value,
            ModDest::FilterResonance(i) => self.filter_resonance[i] += value,
            ModDest::FilterDrive(i) => self.filter_drive[i] += value,
            ModDest::FilterBlend(i) => self.filter_blend[i] += value,
            ModDest::FilterMix(i) => self.filter_mix[i] += value,
            ModDest::EnvAttack(i) => self.env_attack[i] += value,
            ModDest::EnvDecay(i) => self.env_decay[i] += value,
            ModDest::EnvSustain(i) => self.env_sustain[i] += value,
            ModDest::EnvRelease(i) => self.env_release[i] += value,
            ModDest::LfoFrequency(i) => self.lfo_frequency[i] += value,
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
#[derive(Clone, Debug, Default)]
pub struct ModMatrix {
    pub connections: Vec<Connection>,
}

impl ModMatrix {
    /// Resolves every connection into `offsets` (cleared first).
    pub fn resolve(
        &mut self,
        sources: &SourceValues,
        offsets: &mut ModOffsets,
        _reset_mask: PolyMask,
    ) {
        offsets.clear();
        for connection in &mut self.connections {
            let value = sources.get(connection.source);
            let output = connection.transform.process_control(value, None);
            offsets.add(connection.dest, output.scaled);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_routes_source_to_destination() {
        let mut matrix = ModMatrix::default();
        matrix.connections.push(Connection {
            source: ModSource::Lfo(0),
            dest: ModDest::FilterCutoff(0),
            transform: ModulationTransform::with_amount(1.0, 128.0),
        });

        let mut sources = SourceValues::default();
        sources.lfos[0] = PolyF32::splat(0.5);
        let mut offsets = ModOffsets::default();
        matrix.resolve(&sources, &mut offsets, PolyMask::NONE);

        assert!((offsets.filter_cutoff[0].lane(0) - 64.0).abs() < 1e-3);
        assert_eq!(offsets.filter_cutoff[1].lane(0), 0.0);
    }

    #[test]
    fn connections_sum_on_shared_destination() {
        let mut matrix = ModMatrix::default();
        for source in [ModSource::Envelope(1), ModSource::Macro(0)] {
            matrix.connections.push(Connection {
                source,
                dest: ModDest::OscLevel(0),
                transform: ModulationTransform::with_amount(1.0, 1.0),
            });
        }
        let mut sources = SourceValues::default();
        sources.envelopes[1] = PolyF32::splat(0.25);
        sources.macros[0] = PolyF32::splat(0.5);
        let mut offsets = ModOffsets::default();
        matrix.resolve(&sources, &mut offsets, PolyMask::NONE);
        assert!((offsets.osc_level[0].lane(0) - 0.75).abs() < 1e-4);
    }
}

