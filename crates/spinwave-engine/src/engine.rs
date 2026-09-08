//! Top-level sound engine (rework of Vital's `SoundEngine` +
//! `ReorderableEffectChain`): voice allocator → reorderable bus effect
//! chain → stereo encoder → smoothed master volume → peak meter → clamp.
//!
//! Voices and effects run 2x oversampled; the decimator brings the signal
//! back to the host rate before the master path. The folded voice signal
//! uses lanes `[L, R, L, R]`.

use spinwave_dsp::effects::{
    Chorus, ChorusParams, DelayParams, DelayStyle, Distortion, DistortionType, Equalizer,
    EqualizerParams, Flanger, FlangerParams, MultibandCompressor, MultibandCompressorParams,
    Phaser, PhaserParams, Reverb, ReverbParams, StereoDelay,
};
use spinwave_dsp::filters::Decimator;
use spinwave_dsp::utilities::PeakMeter;
use spinwave_poly::constants::{MAX_BUFFER_SIZE, PI};
use spinwave_poly::utils::interpolate;
use spinwave_poly::{math, PolyF32, PolyMask};

use crate::allocator::VoiceAllocator;
use crate::kernel::mod_matrix::{ModSource, SourceValues};
use crate::kernel::voice_filter::{VoiceFilter, VoiceFilterParams};
use crate::kernel::{KernelParams, SynthVoiceKernel};
use crate::modulation::ModulationTransform;

/// Bus effects, in the reference declaration order
/// (`vital::constants::Effect`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Effect {
    Chorus = 0,
    Compressor = 1,
    Delay = 2,
    Distortion = 3,
    Eq = 4,
    FilterFx = 5,
    Flanger = 6,
    Phaser = 7,
    Reverb = 8,
}

pub const NUM_EFFECTS: usize = 9;

/// The default (identity) chain order, `effect_chain_order == 0`.
pub const DEFAULT_ORDER: [Effect; NUM_EFFECTS] = [
    Effect::Chorus,
    Effect::Compressor,
    Effect::Delay,
    Effect::Distortion,
    Effect::Eq,
    Effect::FilterFx,
    Effect::Flanger,
    Effect::Phaser,
    Effect::Reverb,
];

impl Effect {
    pub fn from_index(index: usize) -> Effect {
        match index {
            1 => Effect::Compressor,
            2 => Effect::Delay,
            3 => Effect::Distortion,
            4 => Effect::Eq,
            5 => Effect::FilterFx,
            6 => Effect::Flanger,
            7 => Effect::Phaser,
            8 => Effect::Reverb,
            _ => Effect::Chorus,
        }
    }
}

/// Decodes the `effect_chain_order` value into a chain order (port of
/// `vital::utils::decodeFloatToOrder`): a factorial number system where the
/// digit for position `i` (walking from the last position down) is the
/// number of inversions to apply at that position.
pub fn decode_order(code: u32) -> [Effect; NUM_EFFECTS] {
    let mut order: [usize; NUM_EFFECTS] = [0; NUM_EFFECTS];
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }

    let mut code = code as usize;
    for i in 0..NUM_EFFECTS {
        let remaining = NUM_EFFECTS - i;
        let index = remaining - 1;
        let inversions = code % remaining;
        code /= remaining;

        let placement = order[index - inversions];
        order.copy_within(index - inversions + 1..index + 1, index - inversions);
        order[index] = placement;
    }
    order.map(Effect::from_index)
}

/// Inverse of [`decode_order`] (port of `vital::utils::encodeOrderToFloat`).
pub fn encode_order(order: &[Effect; NUM_EFFECTS]) -> u32 {
    let mut code: u32 = 0;
    for i in 1..NUM_EFFECTS {
        let inversions = order[..i]
            .iter()
            .filter(|&&earlier| (order[i] as usize) < (earlier as usize))
            .count() as u32;
        code = code * (i as u32 + 1) + inversions;
    }
    code
}

// Moved to `crate::tempo` so the kernel can share them without an
// engine ← kernel cycle; re-exported here for compatibility.
pub use crate::tempo::{
    SyncMode, SyncedFrequency, NUM_SYNCED_FREQUENCY_RATIOS, SYNCED_FREQUENCY_RATIOS,
};

/// Stereo encoding mode (`StereoEncoder`): `Spread` narrows/widens through a
/// mid-side-like crossfade, `Rotate` rotates the stereo field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StereoMode {
    #[default]
    Spread,
    Rotate,
}

/// All bus effect parameters plus the chain order. The `frequency` /
/// `rate` / `period_samples` fields of the tempo-syncable dsp param structs
/// are overwritten from the corresponding [`SyncedFrequency`] every block.
#[derive(Clone, Debug)]
pub struct EffectsParams {
    pub order: [Effect; NUM_EFFECTS],

    pub chorus_on: bool,
    pub chorus: ChorusParams,
    pub chorus_sync: SyncedFrequency,

    pub compressor_on: bool,
    pub compressor: MultibandCompressorParams,

    pub delay_on: bool,
    pub delay: DelayParams,
    pub delay_sync: SyncedFrequency,
    pub delay_aux_sync: SyncedFrequency,

    pub distortion_on: bool,
    pub distortion_type: DistortionType,
    /// Drive in dB (the dsp layer maps it per distortion type).
    pub distortion_drive_db: f32,
    /// Dry/wet in `[0, 1]` (`DistortionModule` mix ramp).
    pub distortion_mix: f32,

    pub eq_on: bool,
    pub eq: EqualizerParams,

    pub filter_fx_on: bool,
    /// The `on` field is ignored; `filter_fx_on` gates the chain slot.
    // TODO(fidelity): the reference FilterFxModule also wires the last
    // played note as keytrack; feed it into `state.midi_cutoff` when the
    // parameter layer lands.
    pub filter_fx: VoiceFilterParams,

    pub flanger_on: bool,
    pub flanger: FlangerParams,
    pub flanger_sync: SyncedFrequency,

    pub phaser_on: bool,
    pub phaser: PhaserParams,
    pub phaser_sync: SyncedFrequency,

    pub reverb_on: bool,
    pub reverb: ReverbParams,
}

impl Default for EffectsParams {
    fn default() -> EffectsParams {
        EffectsParams {
            order: DEFAULT_ORDER,
            chorus_on: false,
            chorus: ChorusParams::default(),
            chorus_sync: SyncedFrequency::free(0.5),
            compressor_on: false,
            compressor: MultibandCompressorParams::default(),
            delay_on: false,
            delay: DelayParams::default(),
            delay_sync: SyncedFrequency::free(2.0),
            delay_aux_sync: SyncedFrequency::free(2.0),
            distortion_on: false,
            distortion_type: DistortionType::SoftClip,
            distortion_drive_db: 0.0,
            distortion_mix: 1.0,
            eq_on: false,
            eq: EqualizerParams::default(),
            filter_fx_on: false,
            filter_fx: VoiceFilterParams::default(),
            flanger_on: false,
            flanger: FlangerParams::default(),
            flanger_sync: SyncedFrequency::free(2.0),
            phaser_on: false,
            phaser: PhaserParams::default(),
            phaser_sync: SyncedFrequency::free(1.0),
            reverb_on: false,
            reverb: ReverbParams::default(),
        }
    }
}

impl EffectsParams {
    fn is_on(&self, effect: Effect) -> bool {
        match effect {
            Effect::Chorus => self.chorus_on,
            Effect::Compressor => self.compressor_on,
            Effect::Delay => self.delay_on,
            Effect::Distortion => self.distortion_on,
            Effect::Eq => self.eq_on,
            Effect::FilterFx => self.filter_fx_on,
            Effect::Flanger => self.flanger_on,
            Effect::Phaser => self.phaser_on,
            Effect::Reverb => self.reverb_on,
        }
    }
}

/// Mono modulation destination on the bus effect chain. Offsets are in the
/// destination's engine unit, except the `*Frequency` / `ReverbDecayTime`
/// destinations whose offsets are in the stored log2 domain (matching the
/// table's `Exponential` scale) and apply as a `exp2(offset)` multiplier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectsModDest {
    DelayFeedback,
    DelayDryWet,
    DelayFrequency,
    DelayAuxFrequency,
    ReverbDryWet,
    ReverbDecayTime,
    ReverbSize,
    ChorusDryWet,
    ChorusFeedback,
    ChorusModDepth,
    ChorusFrequency,
    FlangerDryWet,
    FlangerFeedback,
    FlangerModDepth,
    FlangerFrequency,
    FlangerPhaseOffset,
    PhaserDryWet,
    PhaserFeedback,
    PhaserModDepth,
    PhaserFrequency,
    PhaserBlend,
    DistortionDrive,
    DistortionMix,
    FilterFxCutoff,
    FilterFxResonance,
    FilterFxBlend,
    EqLowCutoff,
    EqBandCutoff,
    EqHighCutoff,
    EqLowGain,
    EqBandGain,
    EqHighGain,
    CompressorMix,
    CompressorLowGain,
    CompressorBandGain,
    CompressorHighGain,
}

/// One active mono modulation connection into the bus effect chain.
#[derive(Clone, Debug)]
pub struct EffectsConnection {
    pub source: ModSource,
    pub dest: EffectsModDest,
    pub transform: ModulationTransform,
}

/// Accumulated mono offsets for one block, in engine units (log2 for the
/// exponential destinations, see [`EffectsModDest`]).
#[derive(Clone, Debug, Default)]
pub struct EffectsModOffsets {
    pub delay_feedback: f32,
    pub delay_dry_wet: f32,
    pub delay_frequency: f32,
    pub delay_aux_frequency: f32,
    pub reverb_dry_wet: f32,
    pub reverb_decay_time: f32,
    pub reverb_size: f32,
    pub chorus_dry_wet: f32,
    pub chorus_feedback: f32,
    pub chorus_mod_depth: f32,
    pub chorus_frequency: f32,
    pub flanger_dry_wet: f32,
    pub flanger_feedback: f32,
    pub flanger_mod_depth: f32,
    pub flanger_frequency: f32,
    pub flanger_phase_offset: f32,
    pub phaser_dry_wet: f32,
    pub phaser_feedback: f32,
    pub phaser_mod_depth: f32,
    pub phaser_frequency: f32,
    pub phaser_blend: f32,
    pub distortion_drive_db: f32,
    pub distortion_mix: f32,
    pub filter_fx_cutoff: f32,
    pub filter_fx_resonance: f32,
    pub filter_fx_blend: f32,
    pub eq_low_cutoff: f32,
    pub eq_band_cutoff: f32,
    pub eq_high_cutoff: f32,
    pub eq_low_gain: f32,
    pub eq_band_gain: f32,
    pub eq_high_gain: f32,
    pub compressor_mix: f32,
    pub compressor_low_gain: f32,
    pub compressor_band_gain: f32,
    pub compressor_high_gain: f32,
}

impl EffectsModOffsets {
    pub fn clear(&mut self) {
        *self = EffectsModOffsets::default();
    }

    #[inline]
    fn add(&mut self, dest: EffectsModDest, value: f32) {
        match dest {
            EffectsModDest::DelayFeedback => self.delay_feedback += value,
            EffectsModDest::DelayDryWet => self.delay_dry_wet += value,
            EffectsModDest::DelayFrequency => self.delay_frequency += value,
            EffectsModDest::DelayAuxFrequency => self.delay_aux_frequency += value,
            EffectsModDest::ReverbDryWet => self.reverb_dry_wet += value,
            EffectsModDest::ReverbDecayTime => self.reverb_decay_time += value,
            EffectsModDest::ReverbSize => self.reverb_size += value,
            EffectsModDest::ChorusDryWet => self.chorus_dry_wet += value,
            EffectsModDest::ChorusFeedback => self.chorus_feedback += value,
            EffectsModDest::ChorusModDepth => self.chorus_mod_depth += value,
            EffectsModDest::ChorusFrequency => self.chorus_frequency += value,
            EffectsModDest::FlangerDryWet => self.flanger_dry_wet += value,
            EffectsModDest::FlangerFeedback => self.flanger_feedback += value,
            EffectsModDest::FlangerModDepth => self.flanger_mod_depth += value,
            EffectsModDest::FlangerFrequency => self.flanger_frequency += value,
            EffectsModDest::FlangerPhaseOffset => self.flanger_phase_offset += value,
            EffectsModDest::PhaserDryWet => self.phaser_dry_wet += value,
            EffectsModDest::PhaserFeedback => self.phaser_feedback += value,
            EffectsModDest::PhaserModDepth => self.phaser_mod_depth += value,
            EffectsModDest::PhaserFrequency => self.phaser_frequency += value,
            EffectsModDest::PhaserBlend => self.phaser_blend += value,
            EffectsModDest::DistortionDrive => self.distortion_drive_db += value,
            EffectsModDest::DistortionMix => self.distortion_mix += value,
            EffectsModDest::FilterFxCutoff => self.filter_fx_cutoff += value,
            EffectsModDest::FilterFxResonance => self.filter_fx_resonance += value,
            EffectsModDest::FilterFxBlend => self.filter_fx_blend += value,
            EffectsModDest::EqLowCutoff => self.eq_low_cutoff += value,
            EffectsModDest::EqBandCutoff => self.eq_band_cutoff += value,
            EffectsModDest::EqHighCutoff => self.eq_high_cutoff += value,
            EffectsModDest::EqLowGain => self.eq_low_gain += value,
            EffectsModDest::EqBandGain => self.eq_band_gain += value,
            EffectsModDest::EqHighGain => self.eq_high_gain += value,
            EffectsModDest::CompressorMix => self.compressor_mix += value,
            EffectsModDest::CompressorLowGain => self.compressor_low_gain += value,
            EffectsModDest::CompressorBandGain => self.compressor_band_gain += value,
            EffectsModDest::CompressorHighGain => self.compressor_high_gain += value,
        }
    }
}

/// The mono (control-rate) modulation matrix for the bus effects: sources
/// come from the most recently active voice kernel, reduced to a single
/// value (lane 0), like Vital's mono modulations.
#[derive(Clone, Debug, Default)]
pub struct EffectsModMatrix {
    pub connections: Vec<EffectsConnection>,
}

impl EffectsModMatrix {
    /// Resolves every connection into `offsets` (cleared first).
    pub fn resolve(&mut self, sources: &SourceValues, offsets: &mut EffectsModOffsets) {
        offsets.clear();
        for connection in &mut self.connections {
            let value = sources.get(connection.source);
            let output = connection.transform.process_control(value, None);
            offsets.add(connection.dest, output.scaled.lane(0));
        }
    }
}

/// Master output parameters.
#[derive(Clone, Copy, Debug)]
pub struct MasterParams {
    /// Master volume in dB, clamped to `[-80, 12.2]` like `SmoothVolume`
    /// (-80 dB is treated as silence).
    pub volume_db: f32,
    /// `stereo_routing` in `[0, 1]`; 1.0 is transparent in [`StereoMode::Spread`].
    pub stereo_routing: f32,
    pub stereo_mode: StereoMode,
}

impl Default for MasterParams {
    fn default() -> MasterParams {
        MasterParams { volume_db: 0.0, stereo_routing: 1.0, stereo_mode: StereoMode::Spread }
    }
}

const SMOOTH_VOLUME_MIN_DB: f32 = -80.0;
const SMOOTH_VOLUME_MAX_DB: f32 = 12.2;
const OUTPUT_CLAMP: f32 = 2.1;
/// `DelayModule::kMaxDelayTime` in seconds.
const MAX_DELAY_TIME: f32 = 4.0;
/// Default polyphony, matching the reference's `polyphony` default.
const DEFAULT_POLYPHONY: usize = 8;
/// Voices and bus effects run this many times oversampled; the decimator
/// brings the signal back before the master path (reference
/// `kDefaultOversamplingAmount`).
const OVERSAMPLE: usize = 2;

/// The complete synthesizer: voices, bus effects and the master path.
pub struct SoundEngine {
    sample_rate: u32,
    beats_per_second: f32,

    allocator: VoiceAllocator<SynthVoiceKernel>,
    effects: EffectsParams,
    /// Mono modulation connections into the bus effect parameters.
    pub effects_matrix: EffectsModMatrix,
    effects_offsets: EffectsModOffsets,
    pub master: MasterParams,

    chorus: Chorus,
    compressor: MultibandCompressor,
    delay: StereoDelay,
    distortion: Distortion,
    equalizer: Equalizer,
    filter_fx: VoiceFilter,
    flanger: Flanger,
    phaser: Phaser,
    reverb: Reverb,
    was_on: [bool; NUM_EFFECTS],

    // Master path state (ported ramp state of the reference processors).
    distortion_mix: PolyF32,
    volume_mult: PolyF32,
    encoder_cos: PolyF32,
    encoder_sin: PolyF32,
    peak_meter: PeakMeter,

    decimator: Decimator,

    // Preallocated block buffers (no allocation in `process`).
    mix_bus: Vec<PolyF32>,
    direct_bus: Vec<PolyF32>,
    chain_a: Vec<PolyF32>,
    chain_b: Vec<PolyF32>,
    drive_scratch: Vec<PolyF32>,
    decimated: Vec<PolyF32>,
}

impl SoundEngine {
    pub fn new(sample_rate: u32) -> SoundEngine {
        // Voices and effects run oversampled; only the master path (after
        // the decimator) sees the host rate.
        let engine_rate = sample_rate * OVERSAMPLE as u32;
        let er = engine_rate as f32;
        let mut allocator =
            VoiceAllocator::new(DEFAULT_POLYPHONY, || SynthVoiceKernel::new(engine_rate));
        allocator.set_sample_rate(engine_rate);
        allocator.set_oversample(OVERSAMPLE);
        let oversampled_len = MAX_BUFFER_SIZE * OVERSAMPLE;
        SoundEngine {
            sample_rate,
            beats_per_second: 2.0,
            allocator,
            effects: EffectsParams::default(),
            effects_matrix: EffectsModMatrix::default(),
            effects_offsets: EffectsModOffsets::default(),
            master: MasterParams::default(),
            chorus: Chorus::new(er),
            compressor: MultibandCompressor::new(er),
            delay: StereoDelay::new(delay_max_samples(er), er),
            distortion: Distortion::new(er),
            equalizer: Equalizer::new(er),
            filter_fx: VoiceFilter::new(er),
            flanger: Flanger::new(er),
            phaser: Phaser::new(er),
            reverb: Reverb::new(er),
            was_on: [false; NUM_EFFECTS],
            distortion_mix: PolyF32::ZERO,
            volume_mult: PolyF32::ZERO,
            encoder_cos: PolyF32::ZERO,
            encoder_sin: PolyF32::ZERO,
            peak_meter: PeakMeter::new(),
            decimator: Decimator::new(3),
            mix_bus: vec![PolyF32::ZERO; oversampled_len],
            direct_bus: vec![PolyF32::ZERO; oversampled_len],
            chain_a: vec![PolyF32::ZERO; oversampled_len],
            chain_b: vec![PolyF32::ZERO; oversampled_len],
            drive_scratch: vec![PolyF32::ZERO; oversampled_len],
            decimated: vec![PolyF32::ZERO; MAX_BUFFER_SIZE],
        }
    }

    /// Sample rate the voices and effects actually run at.
    fn engine_rate(&self) -> u32 {
        self.sample_rate * OVERSAMPLE as u32
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
        let engine_rate = self.engine_rate();
        let er = engine_rate as f32;
        self.allocator.set_sample_rate(engine_rate);
        self.allocator.set_oversample(OVERSAMPLE);
        self.chorus.set_sample_rate(er);
        self.compressor.set_sample_rate(er);
        // The delay ring is sized for the sample rate, like DelayModule.
        self.delay = StereoDelay::new(delay_max_samples(er), er);
        self.distortion.set_sample_rate(er);
        self.equalizer.set_sample_rate(er);
        self.filter_fx.set_sample_rate(er);
        self.flanger.set_sample_rate(er);
        self.phaser.set_sample_rate(er);
        self.reverb.set_sample_rate(er);
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Stores the host tempo; tempo-synced parameters resolve against it
    /// every block (`SoundEngine::setBpm` + the lower bound at 0). The
    /// value is also propagated to every voice kernel so the kernel LFOs
    /// can resolve their own tempo sync.
    pub fn set_bpm(&mut self, bpm: f32) {
        self.beats_per_second = (bpm / 60.0).max(0.0);
        let bps = self.beats_per_second;
        for kernel in self.allocator.kernels_mut() {
            kernel.params.beats_per_second = bps;
        }
    }

    // -- Parameter access ----------------------------------------------------

    pub fn params(&self) -> &EffectsParams {
        &self.effects
    }

    pub fn params_mut(&mut self) -> &mut EffectsParams {
        &mut self.effects
    }

    /// Applies `apply` to every voice-pair kernel's parameters (all kernels
    /// share one patch).
    pub fn kernel_params_mut(&mut self, mut apply: impl FnMut(&mut KernelParams)) {
        for kernel in self.allocator.kernels_mut() {
            apply(&mut kernel.params);
        }
    }

    pub fn allocator(&self) -> &VoiceAllocator<SynthVoiceKernel> {
        &self.allocator
    }

    pub fn allocator_mut(&mut self) -> &mut VoiceAllocator<SynthVoiceKernel> {
        &mut self.allocator
    }

    /// Grows/shrinks the voice pool. New kernels start with default params;
    /// reapply the patch through [`Self::kernel_params_mut`] afterwards.
    pub fn set_polyphony(&mut self, polyphony: usize) {
        let sample_rate = self.sample_rate;
        self.allocator
            .set_polyphony_with(polyphony, &mut || SynthVoiceKernel::new(sample_rate));
    }

    /// Post-volume peak/RMS meter (`peak_meter` status output).
    pub fn peak_meter(&self) -> &PeakMeter {
        &self.peak_meter
    }

    // -- Voice event passthroughs -------------------------------------------

    pub fn note_on(&mut self, note: i32, velocity: f32, sample: usize, channel: usize) {
        self.allocator.note_on(note, velocity, sample, channel);
    }

    pub fn note_off(&mut self, note: i32, lift: f32, sample: usize, channel: usize) {
        self.allocator.note_off(note, lift, sample, channel);
    }

    pub fn all_notes_off(&mut self, sample: usize) {
        self.allocator.all_notes_off(sample);
    }

    /// Kills all voices and hard-resets the effect chain
    /// (`SoundEngine::allSoundsOff`).
    pub fn all_sounds_off(&mut self) {
        self.allocator.all_sounds_off();
        self.chorus.hard_reset();
        self.compressor.reset();
        self.delay.hard_reset();
        self.distortion.hard_reset();
        self.equalizer.hard_reset();
        self.filter_fx.hard_reset();
        self.flanger.hard_reset();
        self.phaser.hard_reset(&self.effects.phaser);
        self.reverb.hard_reset();
    }

    pub fn set_pitch_wheel(&mut self, value: f32, channel: usize) {
        self.allocator.set_pitch_wheel(value, channel);
    }

    pub fn set_zoned_pitch_wheel(&mut self, value: f32, from_channel: usize, to_channel: usize) {
        self.allocator.set_zoned_pitch_wheel(value, from_channel, to_channel);
    }

    pub fn set_mod_wheel(&mut self, value: f32, channel: usize) {
        self.allocator.set_mod_wheel(value, channel);
    }

    pub fn set_mod_wheel_all_channels(&mut self, value: f32) {
        self.allocator.set_mod_wheel_all_channels(value);
    }

    pub fn sustain_on(&mut self, channel: usize) {
        self.allocator.sustain_on(channel);
    }

    pub fn sustain_off(&mut self, sample: usize, channel: usize) {
        self.allocator.sustain_off(sample, channel);
    }

    pub fn sostenuto_on(&mut self, channel: usize) {
        self.allocator.sostenuto_on(channel);
    }

    pub fn sostenuto_off(&mut self, sample: usize, channel: usize) {
        self.allocator.sostenuto_off(sample, channel);
    }

    pub fn set_aftertouch(&mut self, note: i32, value: f32, sample: usize, channel: usize) {
        self.allocator.set_aftertouch(note, value, sample, channel);
    }

    pub fn set_channel_aftertouch(&mut self, channel: usize, value: f32, sample: usize) {
        self.allocator.set_channel_aftertouch(channel, value, sample);
    }

    pub fn set_channel_slide(&mut self, channel: usize, value: f32, sample: usize) {
        self.allocator.set_channel_slide(channel, value, sample);
    }

    pub fn num_active_voices(&self) -> usize {
        self.allocator.num_active_voices()
    }

    // -- Processing ----------------------------------------------------------

    /// Renders `num_samples` samples of stereo output. Blocks larger than
    /// [`MAX_BUFFER_SIZE`] are processed in chunks.
    pub fn process(&mut self, num_samples: usize, out_left: &mut [f32], out_right: &mut [f32]) {
        assert!(out_left.len() >= num_samples && out_right.len() >= num_samples);

        let mut start = 0;
        while start < num_samples {
            let block = (num_samples - start).min(MAX_BUFFER_SIZE);
            self.process_block(
                &mut out_left[start..start + block],
                &mut out_right[start..start + block],
            );
            start += block;
        }
    }

    fn process_block(&mut self, out_left: &mut [f32], out_right: &mut [f32]) {
        let num_samples = out_left.len();
        debug_assert!(num_samples <= MAX_BUFFER_SIZE);
        debug_assert_eq!(num_samples, out_right.len());
        if num_samples == 0 {
            return;
        }
        // Voices and effects render this many samples at the engine rate.
        let os_samples = num_samples * OVERSAMPLE;

        // Run the voices and fold the two voice slots into one stereo
        // signal replicated in both vector halves: [L, R, L, R]. The main
        // bus feeds the effect chain; the direct-out bus is kept aside and
        // added after the chain (reference `output_total_`). Voices render
        // before the effect params resolve so the mono effects modulation
        // reads this block's source values.
        let mut mix = std::mem::take(&mut self.mix_bus);
        let mut direct = std::mem::take(&mut self.direct_bus);
        let mut input = std::mem::take(&mut self.chain_a);
        let mut output = std::mem::take(&mut self.chain_b);
        let mut drive = std::mem::take(&mut self.drive_scratch);

        mix[..os_samples].fill(PolyF32::ZERO);
        direct[..os_samples].fill(PolyF32::ZERO);
        self.allocator.process(os_samples, |kernel_out, kernel_direct| {
            for (dest, &src) in mix.iter_mut().zip(kernel_out) {
                *dest += src;
            }
            if let Some(kernel_direct) = kernel_direct {
                for (dest, &src) in direct.iter_mut().zip(kernel_direct) {
                    *dest += src;
                }
            }
        });
        for (folded, &sum) in input[..os_samples].iter_mut().zip(&mix[..os_samples]) {
            *folded = sum + sum.swap_voices();
        }

        // Mono modulation offsets for the bus effects: sources come from
        // the most recently active voice kernel, reduced to lane 0 (Vital's
        // mono modulations). Offsets hold their last value when every voice
        // has died, like the reference control-rate readouts.
        if self.effects_matrix.connections.is_empty() {
            self.effects_offsets.clear();
        } else if let Some(pair) = self.allocator.last_active_pair() {
            self.effects_matrix.resolve(
                self.allocator.kernels()[pair].last_source_values(),
                &mut self.effects_offsets,
            );
        }

        // Resolve tempo-synced parameters once per block; the mono offsets
        // apply to per-block copies of the effect params (the stored params
        // stay unmodulated). Exponential-scale destinations (frequencies,
        // reverb decay) get their offsets in the log2 domain.
        let bps = self.beats_per_second;
        let mods = self.effects_offsets.clone();

        let mut chorus_params = self.effects.chorus;
        chorus_params.wet = (chorus_params.wet + mods.chorus_dry_wet).clamp(0.0, 1.0);
        chorus_params.feedback =
            (chorus_params.feedback + mods.chorus_feedback).clamp(-0.95, 0.95);
        chorus_params.mod_depth =
            (chorus_params.mod_depth + mods.chorus_mod_depth).clamp(0.0, 1.0);
        chorus_params.frequency = PolyF32::splat(
            self.effects.chorus_sync.frequency_hz(bps) * mods.chorus_frequency.exp2(),
        );

        let mut flanger_params = self.effects.flanger;
        flanger_params.wet = (flanger_params.wet + mods.flanger_dry_wet).clamp(0.0, 0.5);
        flanger_params.feedback =
            (flanger_params.feedback + mods.flanger_feedback).clamp(-1.0, 1.0);
        flanger_params.mod_depth =
            (flanger_params.mod_depth + mods.flanger_mod_depth).clamp(0.0, 1.0);
        flanger_params.phase_offset =
            (flanger_params.phase_offset + mods.flanger_phase_offset).clamp(0.0, 1.0);
        flanger_params.frequency = PolyF32::splat(
            self.effects.flanger_sync.frequency_hz(bps) * mods.flanger_frequency.exp2(),
        );

        let mut phaser_params = self.effects.phaser;
        phaser_params.mix = (phaser_params.mix + mods.phaser_dry_wet).clamp(0.0, 1.0);
        phaser_params.feedback_gain =
            (phaser_params.feedback_gain + mods.phaser_feedback).clamp(0.0, 1.0);
        phaser_params.mod_depth =
            (phaser_params.mod_depth + mods.phaser_mod_depth).clamp(0.0, 48.0);
        phaser_params.blend = (phaser_params.blend + mods.phaser_blend).clamp(0.0, 2.0);
        phaser_params.rate = PolyF32::splat(
            self.effects.phaser_sync.frequency_hz(bps) * mods.phaser_frequency.exp2(),
        );

        let mut delay_params = self.resolve_delay_params(bps, &mods);
        delay_params.feedback = (delay_params.feedback + mods.delay_feedback).clamp(-1.0, 1.0);
        delay_params.wet = (delay_params.wet + mods.delay_dry_wet).clamp(0.0, 1.0);

        let mut compressor_params = self.effects.compressor;
        compressor_params.mix = (compressor_params.mix + mods.compressor_mix).clamp(0.0, 1.0);
        compressor_params.low_output_gain_db =
            (compressor_params.low_output_gain_db + mods.compressor_low_gain).clamp(-30.0, 30.0);
        compressor_params.band_output_gain_db = (compressor_params.band_output_gain_db
            + mods.compressor_band_gain)
            .clamp(-30.0, 30.0);
        compressor_params.high_output_gain_db = (compressor_params.high_output_gain_db
            + mods.compressor_high_gain)
            .clamp(-30.0, 30.0);

        let mut eq_params = self.effects.eq;
        eq_params.low_cutoff_midi += PolyF32::splat(mods.eq_low_cutoff);
        eq_params.band_cutoff_midi += PolyF32::splat(mods.eq_band_cutoff);
        eq_params.high_cutoff_midi += PolyF32::splat(mods.eq_high_cutoff);
        eq_params.low_gain_db = (eq_params.low_gain_db + mods.eq_low_gain).clamp(-15.0, 15.0);
        eq_params.band_gain_db = (eq_params.band_gain_db + mods.eq_band_gain).clamp(-15.0, 15.0);
        eq_params.high_gain_db =
            (eq_params.high_gain_db + mods.eq_high_gain).clamp(-15.0, 15.0);

        let mut reverb_params = self.effects.reverb;
        reverb_params.wet = (reverb_params.wet + mods.reverb_dry_wet).clamp(0.0, 1.0);
        reverb_params.decay_time *= mods.reverb_decay_time.exp2();
        reverb_params.size = (reverb_params.size + mods.reverb_size).clamp(0.0, 1.0);

        let mut filter_fx_params = self.effects.filter_fx;
        filter_fx_params.state.midi_cutoff += PolyF32::splat(mods.filter_fx_cutoff);
        filter_fx_params.state.resonance_percent = (filter_fx_params.state.resonance_percent
            + mods.filter_fx_resonance)
            .clamp(0.0, 1.0);
        filter_fx_params
            .state
            .set_pass_blend(filter_fx_params.state.pass_blend + mods.filter_fx_blend);

        let distortion_drive_db =
            (self.effects.distortion_drive_db + mods.distortion_drive_db).clamp(-30.0, 30.0);
        let distortion_mix =
            (self.effects.distortion_mix + mods.distortion_mix).clamp(0.0, 1.0);

        self.update_effect_switches(&phaser_params);

        // The bus effect chain, in the decoded order (main bus only; the
        // direct-out bus bypasses it entirely).
        for effect in self.effects.order {
            if !self.effects.is_on(effect) {
                continue;
            }
            match effect {
                Effect::Chorus => {
                    self.chorus
                        .process(&chorus_params, &input[..os_samples], &mut output[..os_samples]);
                }
                Effect::Compressor => {
                    self.compressor.process(
                        &compressor_params,
                        &input[..os_samples],
                        &mut output[..os_samples],
                    );
                }
                Effect::Delay => {
                    self.delay
                        .process(&delay_params, &input[..os_samples], &mut output[..os_samples]);
                }
                Effect::Distortion => {
                    // TODO(fidelity): DistortionModule's optional pre/post
                    // filter (distortion_filter_order) is not ported yet.
                    output[..os_samples].copy_from_slice(&input[..os_samples]);
                    drive[..os_samples].fill(PolyF32::splat(distortion_drive_db));
                    self.distortion.process(
                        self.effects.distortion_type,
                        &drive[..os_samples],
                        &mut output[..os_samples],
                    );

                    // Dry/wet ramp exactly like DistortionModule::processWithInput.
                    let mut current_mix = self.distortion_mix;
                    self.distortion_mix = PolyF32::splat(distortion_mix);
                    let delta_mix =
                        (self.distortion_mix - current_mix) * (1.0 / os_samples as f32);
                    for (wet, &dry) in output[..os_samples].iter_mut().zip(&input[..os_samples])
                    {
                        current_mix += delta_mix;
                        *wet = interpolate(dry, *wet, current_mix);
                    }
                }
                Effect::Eq => {
                    self.equalizer.process(
                        &eq_params,
                        &input[..os_samples],
                        &mut output[..os_samples],
                    );
                }
                Effect::FilterFx => {
                    let mut params = filter_fx_params;
                    params.on = true; // gated by `filter_fx_on` instead
                    self.filter_fx.process(
                        &params,
                        &input[..os_samples],
                        &mut output[..os_samples],
                        PolyMask::NONE,
                    );
                }
                Effect::Flanger => {
                    self.flanger.process(
                        &flanger_params,
                        &input[..os_samples],
                        &mut output[..os_samples],
                    );
                }
                Effect::Phaser => {
                    self.phaser
                        .process(&phaser_params, &input[..os_samples], &mut output[..os_samples]);
                }
                Effect::Reverb => {
                    self.reverb.process(
                        &reverb_params,
                        &input[..os_samples],
                        &mut output[..os_samples],
                    );
                }
            }
            std::mem::swap(&mut input, &mut output);
        }

        // Add the folded direct-out bus after the chain, like the reference
        // `output_total_ = effect_chain_ + voice_handler_->getDirectOutput()`.
        for (out, &sum) in input[..os_samples].iter_mut().zip(&direct[..os_samples]) {
            *out += sum + sum.swap_voices();
        }

        // Decimate back to the host rate, then the master path:
        // stereo encoder → smoothed volume → meter → clamp.
        let mut decimated = std::mem::take(&mut self.decimated);
        self.decimator.process(
            &input[..os_samples],
            self.engine_rate(),
            self.sample_rate,
            &mut decimated[..num_samples],
        );

        self.apply_stereo_encoding(&mut decimated[..num_samples]);
        self.apply_master_volume(&mut decimated[..num_samples]);
        self.peak_meter.process(&decimated[..num_samples]);

        for ((&sample, left), right) in decimated[..num_samples]
            .iter()
            .zip(out_left.iter_mut())
            .zip(out_right.iter_mut())
        {
            let clamped = sample.clamp(-OUTPUT_CLAMP, OUTPUT_CLAMP);
            *left = clamped.lane(0);
            *right = clamped.lane(1);
        }

        self.mix_bus = mix;
        self.direct_bus = direct;
        self.chain_a = input;
        self.chain_b = output;
        self.drive_scratch = drive;
        self.decimated = decimated;
    }

    /// Resolves the delay tempo sync into per-lane periods: the main line
    /// feeds the left lanes and the aux line the right lanes for the stereo
    /// styles, matching `Delay::processWithInput`'s `kFrequencyAux` load.
    fn resolve_delay_params(&self, beats_per_second: f32, mods: &EffectsModOffsets) -> DelayParams {
        // A tiny floor keeps `Freeze` (ratio 0) finite; the delay clamps the
        // resulting period to its memory size, like the reference clamp.
        // Frequency modulation offsets are in log2 Hz (the stored domain).
        const MIN_HZ: f32 = 1.0e-4;
        let sr = self.engine_rate() as f32;
        let mut params = self.effects.delay;
        let main_hz = self.effects.delay_sync.frequency_hz(beats_per_second)
            * mods.delay_frequency.exp2();
        let main_period = sr / main_hz.max(MIN_HZ);
        let uses_aux = matches!(
            params.style,
            DelayStyle::Stereo | DelayStyle::PingPong | DelayStyle::MidPingPong
        );
        params.period_samples = if uses_aux {
            let aux_hz = self.effects.delay_aux_sync.frequency_hz(beats_per_second)
                * mods.delay_aux_frequency.exp2();
            let aux_period = sr / aux_hz.max(MIN_HZ);
            PolyF32::stereo(main_period, aux_period)
        } else {
            PolyF32::splat(main_period)
        };
        params
    }

    /// Mirrors each effect module's `enable` override: some reset when
    /// switched on, some when switched off (`ReorderableEffectChain`'s
    /// on/enabled bookkeeping).
    fn update_effect_switches(&mut self, phaser_params: &PhaserParams) {
        for index in 0..NUM_EFFECTS {
            let effect = Effect::from_index(index);
            let on = self.effects.is_on(effect);
            if on == self.was_on[index] {
                continue;
            }
            match effect {
                Effect::Chorus => {
                    if on {
                        self.chorus.hard_reset();
                    }
                }
                Effect::Compressor => {
                    if !on {
                        self.compressor.reset();
                    }
                }
                Effect::Delay => {
                    if !on {
                        self.delay.hard_reset();
                    }
                }
                Effect::Distortion | Effect::FilterFx => {}
                Effect::Eq => {
                    if on {
                        self.equalizer.hard_reset();
                    }
                }
                Effect::Flanger => {
                    if !on {
                        self.flanger.hard_reset();
                    }
                }
                Effect::Phaser => {
                    if on {
                        self.phaser.hard_reset(phaser_params);
                    }
                }
                Effect::Reverb => {
                    if !on {
                        self.reverb.hard_reset();
                    }
                }
            }
            self.was_on[index] = on;
        }
    }

    /// Port of `StereoEncoder` as wired in the reference `SoundEngine`
    /// (decoding = true). Coefficients ramp linearly across the block.
    fn apply_stereo_encoding(&mut self, buffer: &mut [PolyF32]) {
        const DECODING_MULT: f32 = -1.0;
        let routing = self.master.stereo_routing.clamp(0.0, 1.0);
        let (target_cos, target_sin, sign) = match self.master.stereo_mode {
            StereoMode::Rotate => {
                let encoding = routing * DECODING_MULT * 2.0 * PI;
                (
                    PolyF32::splat(encoding.cos()),
                    PolyF32::splat(encoding.sin()),
                    PolyF32::stereo(1.0, -1.0),
                )
            }
            StereoMode::Spread => {
                let phase = (1.0 - routing) * 0.25 * PI;
                (PolyF32::splat(phase.cos()), PolyF32::splat(phase.sin()), PolyF32::ONE)
            }
        };

        let mut current_cos = self.encoder_cos;
        let mut current_sin = self.encoder_sin;
        self.encoder_cos = target_cos;
        self.encoder_sin = target_sin;
        let delta_tick = 1.0 / buffer.len() as f32;
        let delta_cos = (target_cos - current_cos) * delta_tick;
        let delta_sin = (target_sin - current_sin) * delta_tick;

        for sample in buffer.iter_mut() {
            current_cos += delta_cos;
            current_sin += delta_sin;
            let swap = sign * sample.swap_stereo();
            *sample = *sample * current_cos + swap * current_sin;
        }
    }

    /// Port of `SmoothVolume::process`: dB clamped to `[-80, 12.2]`, mapped
    /// to magnitude (zero at the floor) and ramped linearly over the block.
    fn apply_master_volume(&mut self, buffer: &mut [PolyF32]) {
        let db = self.master.volume_db.clamp(SMOOTH_VOLUME_MIN_DB, SMOOTH_VOLUME_MAX_DB);
        let target = if db <= SMOOTH_VOLUME_MIN_DB {
            PolyF32::ZERO
        } else {
            math::db_to_magnitude(PolyF32::splat(db))
        };

        let mut current = self.volume_mult;
        self.volume_mult = target;
        let delta = (target - current) * (1.0 / buffer.len() as f32);
        for sample in buffer.iter_mut() {
            current += delta;
            *sample *= current;
        }
    }
}

fn delay_max_samples(sample_rate: f32) -> usize {
    (MAX_DELAY_TIME * sample_rate) as usize + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ProducerDestination;
    use spinwave_dsp::modulators::EnvelopeParams;

    // -- Effect order codec --------------------------------------------------

    #[test]
    fn effect_order_zero_decodes_to_default() {
        assert_eq!(decode_order(0), DEFAULT_ORDER);
        assert_eq!(encode_order(&DEFAULT_ORDER), 0);
    }

    #[test]
    fn effect_order_swapping_last_two_encodes_to_one() {
        // A single inversion at the last position is the lowest non-zero code.
        let mut order = DEFAULT_ORDER;
        order.swap(7, 8);
        assert_eq!(encode_order(&order), 1);
        assert_eq!(decode_order(1), order);
    }

    #[test]
    fn effect_order_roundtrip() {
        let reversed = [
            Effect::Reverb,
            Effect::Phaser,
            Effect::Flanger,
            Effect::FilterFx,
            Effect::Eq,
            Effect::Distortion,
            Effect::Delay,
            Effect::Compressor,
            Effect::Chorus,
        ];
        assert_eq!(decode_order(encode_order(&reversed)), reversed);

        // 9! - 1 is the largest valid code.
        for code in [1u32, 2, 100, 5040, 362_879] {
            assert_eq!(encode_order(&decode_order(code)), code);
        }
    }

    // -- Tempo sync ----------------------------------------------------------

    #[test]
    fn synced_frequency_resolves_tempo_modes() {
        let bps = 2.0; // 120 bpm
        let free = SyncedFrequency::free(3.5);
        assert_eq!(free.frequency_hz(bps), 3.5);

        // Index 8 is the 1/1 ratio.
        let synced = SyncedFrequency { sync: SyncMode::Tempo, frequency_hz: 0.0, tempo_index: 8.0 };
        assert!((synced.frequency_hz(bps) - 2.0).abs() < 1e-6);

        let dotted = SyncedFrequency { sync: SyncMode::DottedTempo, ..synced };
        assert!((dotted.frequency_hz(bps) - 2.0 * 2.0 / 3.0).abs() < 1e-6);

        let triplet = SyncedFrequency { sync: SyncMode::TripletTempo, ..synced };
        assert!((triplet.frequency_hz(bps) - 3.0).abs() < 1e-6);
    }

    // -- Engine --------------------------------------------------------------

    fn make_engine() -> SoundEngine {
        let mut engine = SoundEngine::new(44100);
        engine.kernel_params_mut(|params| {
            params.envelopes[0] = EnvelopeParams {
                attack: PolyF32::splat(0.001),
                sustain: PolyF32::ONE,
                release: PolyF32::splat(0.02),
                ..Default::default()
            };
        });
        engine
    }

    fn render(engine: &mut SoundEngine, blocks: usize) -> (Vec<f32>, Vec<f32>) {
        let mut left = vec![0.0f32; blocks * MAX_BUFFER_SIZE];
        let mut right = vec![0.0f32; blocks * MAX_BUFFER_SIZE];
        engine.process(blocks * MAX_BUFFER_SIZE, &mut left, &mut right);
        (left, right)
    }

    fn peak(buffer: &[f32]) -> f32 {
        buffer.iter().fold(0.0f32, |a, &v| a.max(v.abs()))
    }

    fn rms(buffer: &[f32]) -> f32 {
        (buffer.iter().map(|v| v * v).sum::<f32>() / buffer.len() as f32).sqrt()
    }

    #[test]
    fn silence_when_no_notes() {
        let mut engine = make_engine();
        let (left, right) = render(&mut engine, 4);
        assert!(left.iter().chain(&right).all(|&v| v == 0.0));
    }

    #[test]
    fn note_produces_audio_and_reverb_tail_survives_voice_death() {
        let mut engine = make_engine();
        {
            let effects = engine.params_mut();
            effects.reverb_on = true;
            effects.reverb.wet = PolyF32::splat(0.8);
            effects.reverb.decay_time = PolyF32::splat(2.0);
        }

        engine.note_on(60, 1.0, 0, 0);
        let (left, right) = render(&mut engine, 8);
        assert!(peak(&left) > 0.01, "no audio produced: peak {}", peak(&left));
        assert!(peak(&right) > 0.01);
        assert!(left.iter().chain(&right).all(|v| v.is_finite()));

        engine.note_off(60, 0.5, 0, 0);
        // Enough blocks for the 20 ms release; the voice dies but the
        // reverb keeps ringing through the chain.
        let _ = render(&mut engine, 20);
        assert_eq!(engine.num_active_voices(), 0, "voice was not retired");
        let (tail, _) = render(&mut engine, 4);
        assert!(peak(&tail) > 1e-5, "reverb tail is silent");
    }

    #[test]
    fn master_volume_scales_output() {
        let render_rms = |volume_db: f32| {
            let mut engine = make_engine();
            engine.master.volume_db = volume_db;
            engine.note_on(60, 1.0, 0, 0);
            let _ = render(&mut engine, 16); // attack + ramp warmup
            let (left, _) = render(&mut engine, 16);
            rms(&left)
        };

        let loud = render_rms(0.0);
        let quiet = render_rms(-20.0);
        assert!(loud > 0.0 && quiet > 0.0);
        let ratio = loud / quiet;
        assert!(
            (8.0..12.0).contains(&ratio),
            "-20 dB should be ~10x quieter, got ratio {ratio}"
        );
    }

    #[test]
    fn output_finite_and_clamped_at_max_volume() {
        let mut engine = make_engine();
        engine.master.volume_db = 30.0; // clamps to +12.2 dB inside
        {
            let effects = engine.params_mut();
            effects.distortion_on = true;
            effects.distortion_drive_db = 30.0;
        }
        for note in [48, 55, 60, 64, 67, 72] {
            engine.note_on(note, 1.0, 0, 0);
        }
        let (left, right) = render(&mut engine, 24);
        assert!(peak(&left) > 0.1);
        for value in left.iter().chain(&right) {
            assert!(value.is_finite());
            assert!(value.abs() <= OUTPUT_CLAMP, "sample beyond clamp: {value}");
        }
    }

    #[test]
    fn set_bpm_propagates_to_kernels() {
        let mut engine = make_engine();
        engine.set_bpm(90.0);
        for kernel in engine.allocator().kernels() {
            assert!((kernel.params.beats_per_second - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn direct_out_bypasses_bus_distortion() {
        let render_left = |destination: ProducerDestination, distortion_on: bool| {
            let mut engine = make_engine();
            engine.kernel_params_mut(|params| {
                params.oscillators[0].destination = destination;
            });
            if distortion_on {
                let effects = engine.params_mut();
                effects.distortion_on = true;
                effects.distortion_drive_db = 30.0;
            }
            engine.note_on(60, 1.0, 0, 0);
            let _ = render(&mut engine, 4);
            let (left, _) = render(&mut engine, 8);
            left
        };

        let clean = render_left(ProducerDestination::Effects, false);
        let direct = render_left(ProducerDestination::DirectOut, true);
        let distorted = render_left(ProducerDestination::Effects, true);
        assert!(peak(&clean) > 0.01);
        assert!(peak(&direct) > 0.01);

        let diff = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
        };
        // Direct-out skips the chain: the heavy drive changes nothing.
        assert!(
            diff(&direct, &clean) < 1e-6,
            "direct-out went through the distortion: diff {}",
            diff(&direct, &clean)
        );
        // The main (effects) bus does run through the distortion.
        assert!(
            diff(&distorted, &clean) > 1e-3,
            "main-bus distortion had no effect: diff {}",
            diff(&distorted, &clean)
        );
    }

    #[test]
    fn lfo_modulates_delay_wet_over_blocks() {
        let render_left = |modulate: bool| {
            let mut engine = make_engine();
            {
                let effects = engine.params_mut();
                effects.delay_on = true;
                effects.delay.wet = PolyF32::ZERO;
                effects.delay.feedback = PolyF32::splat(0.5);
                // 20 Hz free line: the echo returns within the render.
                effects.delay_sync = SyncedFrequency::free(20.0);
            }
            engine.kernel_params_mut(|params| {
                params.lfos[0].params.frequency = PolyF32::splat(3.0);
            });
            if modulate {
                engine.effects_matrix.connections.push(EffectsConnection {
                    source: ModSource::Lfo(0),
                    dest: EffectsModDest::DelayDryWet,
                    transform: ModulationTransform::with_amount(1.0, 1.0),
                });
            }
            engine.note_on(60, 1.0, 0, 0);
            let _ = render(&mut engine, 4);
            let (left, _) = render(&mut engine, 24);
            left
        };

        let dry = render_left(false);
        let modulated = render_left(true);
        assert!(peak(&dry) > 0.01);
        let diff: f32 = dry
            .iter()
            .zip(&modulated)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / dry.len() as f32;
        // The base wet is zero: only the LFO connection can move the wet
        // level, so a difference proves the mono matrix reached the delay.
        assert!(diff > 1e-4, "delay wet modulation had no effect: diff {diff}");
    }

    #[test]
    fn macro_offsets_distortion_drive() {
        let render_left = |macro_value: f32| {
            let mut engine = make_engine();
            {
                let effects = engine.params_mut();
                effects.distortion_on = true;
                effects.distortion_drive_db = 0.0;
            }
            engine.kernel_params_mut(|params| params.macros[0] = macro_value);
            engine.effects_matrix.connections.push(EffectsConnection {
                source: ModSource::Macro(0),
                dest: EffectsModDest::DistortionDrive,
                transform: ModulationTransform::with_amount(0.5, 60.0),
            });
            engine.note_on(60, 1.0, 0, 0);
            let _ = render(&mut engine, 4);
            let (left, _) = render(&mut engine, 8);
            left
        };

        // Same connection in both configs; only the macro moves. At 1.0 the
        // offset is 0.5 * 60 = +30 dB of drive into the soft clip.
        let clean = render_left(0.0);
        let driven = render_left(1.0);
        assert!(peak(&clean) > 0.01 && peak(&driven) > 0.01);
        let diff: f32 = clean
            .iter()
            .zip(&driven)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / clean.len() as f32;
        assert!(diff > 1e-3, "distortion drive offset had no effect: diff {diff}");
    }

    #[test]
    fn effect_chain_runs_in_decoded_order() {
        // Sanity: enabling an effect changes the output vs. bypass.
        let base = {
            let mut engine = make_engine();
            engine.note_on(60, 1.0, 0, 0);
            let _ = render(&mut engine, 8);
            let (left, _) = render(&mut engine, 8);
            left
        };
        let flanged = {
            let mut engine = make_engine();
            {
                let effects = engine.params_mut();
                effects.flanger_on = true;
                effects.flanger.wet = PolyF32::splat(0.5);
            }
            engine.note_on(60, 1.0, 0, 0);
            let _ = render(&mut engine, 8);
            let (left, _) = render(&mut engine, 8);
            left
        };
        let difference: f32 = base
            .iter()
            .zip(&flanged)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(difference > 1e-3, "flanger had no effect on the output");
    }
}
