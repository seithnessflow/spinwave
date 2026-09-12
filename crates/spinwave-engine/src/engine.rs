//! Top-level sound engine (rework of Vital's `SoundEngine` +
//! `ReorderableEffectChain`): voice allocator → three reorderable bus effect
//! chains (main + two send buses, Serum-2-mixer style) → stereo encoder →
//! smoothed master volume → peak meter → clamp.
//!
//! Voices and effects run oversampled (2x by default, see
//! [`SoundEngine::set_oversampling`]); the decimator brings the signal back
//! to the host rate before the master path. The folded voice signal uses
//! lanes `[L, R, L, R]`.

use spinwave_dsp::effects::ConvolutionReverb;
use spinwave_dsp::filters::{DcFilter, Decimator};
use spinwave_dsp::modulators::RandomLfo;
use spinwave_dsp::utilities::PeakMeter;
use spinwave_poly::constants::{MAX_BUFFER_SIZE, PI};
use spinwave_poly::{math, PolyF32, PolyMask};

use crate::allocator::{VoiceAllocator, MAX_POLYPHONY};
use crate::kernel::mod_matrix::{
    ModDest, ModSource, SourceValues, MAX_MODULATION_CONNECTIONS, NUM_RANDOM_LFOS,
};
use crate::kernel::{KernelParams, SynthVoiceKernel};
use crate::modulation::ModulationTransform;

// The effect chain moved to `crate::effect_chain`; re-exported here so the
// existing `spinwave_engine::engine::*` paths keep working.
pub use crate::effect_chain::{
    decode_order, encode_order, DistortionFilterOrder, Effect, EffectChain, EffectSplit,
    EffectsParams, ResolvedEffectsParams, SplitMode, DEFAULT_ORDER, DEFAULT_SPLIT_CROSSOVER_HZ,
    NUM_EFFECTS, NUM_LEGACY_EFFECTS,
};

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

/// Selects one of the engine's three effect chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainId {
    /// The historical bus chain: the dry voice signal runs through it, and
    /// the effect mod matrix targets it.
    Main,
    BusA,
    BusB,
}

/// Where a bus chain's output goes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BusOutput {
    /// Parallel: sums with the main chain output (post-main).
    #[default]
    Master,
    /// Serial: feeds the main chain input (the bus return passes through
    /// the main chain's effects).
    MainChain,
}

/// One send bus of the effects mixer. A bus receives BOTH the hard-routed
/// producer signal (`ProducerDestination::BusA`/`BusB`) and `send × folded`
/// tapped before the main chain.
#[derive(Clone, Copy, Debug)]
pub struct BusParams {
    /// Send level in `[0, 1]`, tapping the folded voice signal before the
    /// main chain (in addition to any hard-routed producers).
    pub send: f32,
    /// Return gain applied to the bus chain output.
    pub return_gain_db: f32,
    pub on: bool,
    /// Serial (into the main chain) or parallel (to the master) return.
    pub output: BusOutput,
}

impl Default for BusParams {
    fn default() -> BusParams {
        BusParams { send: 0.0, return_gain_db: 0.0, on: false, output: BusOutput::Master }
    }
}

/// Send/return levels for the two effect bus chains.
#[derive(Clone, Copy, Debug, Default)]
pub struct MixerParams {
    pub bus_a: BusParams,
    pub bus_b: BusParams,
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
    /// `phaser_center`, the sweep's centre in MIDI (audio-rate in the
    /// reference; consumed per block here, see notes/audio-rate-audit.md).
    PhaserCenter,
    DistortionDrive,
    DistortionMix,
    /// `distortion_filter_cutoff`, MIDI (audio-rate in the reference).
    DistortionFilterCutoff,
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
    pub phaser_center: f32,
    pub distortion_drive_db: f32,
    pub distortion_mix: f32,
    pub distortion_filter_cutoff: f32,
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
            EffectsModDest::PhaserCenter => self.phaser_center += value,
            EffectsModDest::DistortionFilterCutoff => self.distortion_filter_cutoff += value,
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
/// come from the most recently activated VOICE, reduced to that voice's
/// left lane, like Vital's mono modulations.
///
/// `connections` is preallocated to [`MAX_MODULATION_CONNECTIONS`]; fill it
/// through [`EffectsModMatrix::set_connections`] on the audio thread.
#[derive(Clone, Debug)]
pub struct EffectsModMatrix {
    pub connections: Vec<EffectsConnection>,
}

impl Default for EffectsModMatrix {
    fn default() -> EffectsModMatrix {
        EffectsModMatrix { connections: Vec::with_capacity(MAX_MODULATION_CONNECTIONS) }
    }
}

impl EffectsModMatrix {
    /// Replaces the connection list without reallocating (clear + copy of
    /// at most [`MAX_MODULATION_CONNECTIONS`] entries).
    pub fn set_connections(&mut self, connections: &[EffectsConnection]) {
        self.connections.clear();
        if self.connections.capacity() < MAX_MODULATION_CONNECTIONS {
            self.connections.reserve_exact(MAX_MODULATION_CONNECTIONS);
        }
        let count = connections.len().min(MAX_MODULATION_CONNECTIONS);
        self.connections.extend_from_slice(&connections[..count]);
    }

    /// Resolves every connection into `offsets` (cleared first), reading
    /// lane `lane` of every source (the last active voice's left lane),
    /// and that voice's meta-modulation offsets on each connection's slot.
    pub fn resolve(
        &mut self,
        sources: &SourceValues,
        lane: usize,
        amount_offsets: &[PolyF32],
        power_offsets: &[PolyF32],
        offsets: &mut EffectsModOffsets,
    ) {
        offsets.clear();
        for connection in &mut self.connections {
            let slot = connection.transform.slot;
            connection.transform.amount_offset =
                PolyF32::splat(amount_offsets.get(slot).map_or(0.0, |o| o.lane(lane)));
            connection.transform.power_offset =
                PolyF32::splat(power_offsets.get(slot).map_or(0.0, |o| o.lane(lane)));
            let value = sources.get(connection.source);
            let output = connection.transform.process_control(value);
            offsets.add(connection.dest, output.scaled.lane(lane));
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
/// Default polyphony, matching the reference's `polyphony` default.
pub const DEFAULT_POLYPHONY: usize = 8;
/// Default oversampling factor (reference `kDefaultOversamplingAmount`).
const DEFAULT_OVERSAMPLE: usize = 2;
/// Largest oversampling factor the engine buffers are sized for.
pub const MAX_OVERSAMPLE: usize = 4;
/// Sample rate the oversampling factor is specified at
/// (`SoundEngine::setOversamplingAmount`'s `kBaseSampleRate`).
/// Corner of the master DC blocker. Well below the lowest musical
/// fundamental, but high enough that the output settles to true silence
/// within tens of milliseconds after the last voice dies (the voice-level
/// blockers keep the reference's much lower corner).
const MASTER_DC_CUTOFF_HZ: f32 = 5.0;

const BASE_SAMPLE_RATE: u32 = 44100;

/// Halves the requested oversampling for every doubling of the host rate
/// above 44.1 kHz (`sound_engine.cpp` `setOversamplingAmount`): 2x at
/// 96 kHz runs 1x, 4x at 96 kHz runs 2x.
pub fn effective_oversample(requested: usize, sample_rate: u32) -> usize {
    let mut oversample = requested.clamp(1, MAX_OVERSAMPLE).next_power_of_two();
    if oversample > MAX_OVERSAMPLE {
        oversample = MAX_OVERSAMPLE;
    }
    let mut sample_rate_mult = sample_rate / BASE_SAMPLE_RATE;
    while sample_rate_mult > 1 && oversample > 1 {
        sample_rate_mult >>= 1;
        oversample >>= 1;
    }
    oversample
}

/// The complete synthesizer: voices, the effects mixer (main chain plus two
/// send buses) and the master path.
pub struct SoundEngine {
    sample_rate: u32,
    beats_per_second: f32,
    /// Oversampling factor asked for (the `oversampling` parameter as a
    /// factor); `oversample` is what actually runs after the sample-rate
    /// rule.
    requested_oversample: usize,
    oversample: usize,
    /// Host transport position in seconds at the start of the next block.
    transport_seconds: f64,
    transport_playing: bool,

    allocator: VoiceAllocator<SynthVoiceKernel>,
    /// Transport-synced random LFO generators shared by every voice
    /// (reference `random_lfo.h` `shared_state_`): advanced once per block
    /// and pushed to the kernels.
    sync_random_lfos: [RandomLfo; NUM_RANDOM_LFOS],
    /// Mono modulation connections into the MAIN chain's effect parameters.
    pub effects_matrix: EffectsModMatrix,
    effects_offsets: EffectsModOffsets,
    pub master: MasterParams,
    /// Send/return levels for the two bus chains.
    pub mixer: MixerParams,

    main: EffectChain,
    bus_a: EffectChain,
    bus_b: EffectChain,

    // Master path state (ported ramp state of the reference processors).
    volume_mult: PolyF32,
    encoder_cos: PolyF32,
    encoder_sin: PolyF32,
    peak_meter: PeakMeter,
    /// DC blocker on the master output (the effect chain can add DC that
    /// the per-voice blockers never see).
    master_dc_filter: DcFilter,
    /// Whether the master DC blocker runs. On everywhere except the golden
    /// bench, which pins it off so the comparison sees the DSP path the
    /// reference has rather than an output stage the reference lacks.
    master_dc_enabled: bool,

    decimator: Decimator,

    // Preallocated block buffers (no allocation in `process`).
    mix_bus: Vec<PolyF32>,
    direct_bus: Vec<PolyF32>,
    folded_bus: Vec<PolyF32>,
    bus_a_scratch: Vec<PolyF32>,
    bus_b_scratch: Vec<PolyF32>,
    decimated: Vec<PolyF32>,
}

impl SoundEngine {
    /// Builds the engine with the FULL voice pool ([`MAX_POLYPHONY`] voices,
    /// `MAX_POLYPHONY / 2` kernels) preallocated, like the reference's
    /// `setPolyphony(kMaxPolyphony)` at init; the active polyphony starts
    /// at 8 and [`Self::set_polyphony`] never allocates afterwards.
    pub fn new(sample_rate: u32) -> SoundEngine {
        Self::with_pool(sample_rate, MAX_POLYPHONY)
    }

    /// Builds the engine with a voice pool of `pool_voices` (clamped to
    /// 1..=[`MAX_POLYPHONY`], rounded up to a whole kernel). For offline
    /// renders that know their polyphony: the full pool costs ~120 ms to
    /// build, a one-voice pool ~20 ms, and a search builds one per render.
    /// `set_polyphony` later clamps to this pool.
    pub fn with_pool(sample_rate: u32, pool_voices: usize) -> SoundEngine {
        let pool_voices = pool_voices.clamp(1, MAX_POLYPHONY);
        // Build every lazy lookup table now so none is first touched on
        // the audio thread.
        spinwave_dsp::warm_up();
        // Voices and effects run oversampled; only the master path (after
        // the decimator) sees the host rate.
        let oversample = effective_oversample(DEFAULT_OVERSAMPLE, sample_rate);
        let engine_rate = sample_rate * oversample as u32;
        let er = engine_rate as f32;
        let mut allocator =
            VoiceAllocator::new(pool_voices, || SynthVoiceKernel::new(engine_rate));
        allocator.set_sample_rate(engine_rate);
        allocator.set_oversample(oversample);
        allocator.set_polyphony(DEFAULT_POLYPHONY.min(pool_voices));
        let max_block = MAX_BUFFER_SIZE * MAX_OVERSAMPLE;
        SoundEngine {
            sample_rate,
            beats_per_second: 2.0,
            requested_oversample: DEFAULT_OVERSAMPLE,
            oversample,
            transport_seconds: 0.0,
            transport_playing: false,
            allocator,
            sync_random_lfos: core::array::from_fn(|_| RandomLfo::new(er)),
            effects_matrix: EffectsModMatrix::default(),
            effects_offsets: EffectsModOffsets::default(),
            master: MasterParams::default(),
            mixer: MixerParams::default(),
            main: EffectChain::new(er, max_block),
            bus_a: EffectChain::new(er, max_block),
            bus_b: EffectChain::new(er, max_block),
            volume_mult: PolyF32::ZERO,
            encoder_cos: PolyF32::ZERO,
            encoder_sin: PolyF32::ZERO,
            peak_meter: PeakMeter::new(),
            master_dc_filter: DcFilter::with_cutoff(MASTER_DC_CUTOFF_HZ, sample_rate as f32),
            master_dc_enabled: true,
            decimator: Decimator::new(3),
            mix_bus: vec![PolyF32::ZERO; max_block],
            direct_bus: vec![PolyF32::ZERO; max_block],
            folded_bus: vec![PolyF32::ZERO; max_block],
            bus_a_scratch: vec![PolyF32::ZERO; max_block],
            bus_b_scratch: vec![PolyF32::ZERO; max_block],
            decimated: vec![PolyF32::ZERO; MAX_BUFFER_SIZE],
        }
    }

    /// Returns the engine to the state [`Self::with_pool`] would give at
    /// the same sample rate and oversampling — fresh voice kernels for
    /// `pool_voices` (rebuilt: they are cheap), the effect chains reset in
    /// place (they are not: their delay and reverb memories are most of a
    /// build), everything else back at its constructor value. For offline
    /// renders that would otherwise build an engine per render; the
    /// bit-identity with a fresh engine is asserted by
    /// `spinwave_control::ops::tests::a_recycled_engine_renders_the_same_bytes_as_a_fresh_one`.
    pub fn recycle(&mut self, pool_voices: usize) {
        let pool_voices = pool_voices.clamp(1, MAX_POLYPHONY);
        let engine_rate = self.engine_rate();
        let er = engine_rate as f32;
        let mut allocator = VoiceAllocator::new(pool_voices, || SynthVoiceKernel::new(engine_rate));
        allocator.set_sample_rate(engine_rate);
        allocator.set_oversample(self.oversample);
        allocator.set_polyphony(DEFAULT_POLYPHONY.min(pool_voices));
        self.allocator = allocator;
        self.beats_per_second = 2.0;
        self.transport_seconds = 0.0;
        self.transport_playing = false;
        self.sync_random_lfos = core::array::from_fn(|_| RandomLfo::new(er));
        self.effects_matrix = EffectsModMatrix::default();
        self.effects_offsets = EffectsModOffsets::default();
        self.master = MasterParams::default();
        self.mixer = MixerParams::default();
        self.main.reset_for_reuse();
        self.bus_a.reset_for_reuse();
        self.bus_b.reset_for_reuse();
        self.volume_mult = PolyF32::ZERO;
        self.encoder_cos = PolyF32::ZERO;
        self.encoder_sin = PolyF32::ZERO;
        self.peak_meter = PeakMeter::new();
        self.master_dc_filter = DcFilter::with_cutoff(MASTER_DC_CUTOFF_HZ, self.sample_rate as f32);
        self.master_dc_enabled = true;
        self.decimator = Decimator::new(3);
        for buffer in [&mut self.mix_bus, &mut self.direct_bus, &mut self.folded_bus, &mut self.bus_a_scratch, &mut self.bus_b_scratch, &mut self.decimated] {
            buffer.fill(PolyF32::ZERO);
        }
    }

    /// Voices the pool holds.
    pub fn pool_voices(&self) -> usize {
        self.allocator.pool_size()
    }

    /// Sample rate the voices and effects actually run at.
    pub fn engine_rate(&self) -> u32 {
        self.sample_rate * self.oversample as u32
    }

    /// Oversampling factor actually running (after the sample-rate rule).
    /// The factor asked for, before the sample-rate rule.
    pub fn requested_oversampling(&self) -> usize {
        self.requested_oversample
    }

    pub fn oversampling(&self) -> usize {
        self.oversample
    }

    /// Not RT-safe (the effect chains' delay rings are reallocated): call
    /// from the host's prepare / initialize path.
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
        self.apply_engine_rate();
    }

    /// Sets the oversampling factor (1, 2 or 4, the `oversampling`
    /// parameter's `1 << index`), subject to the reference's rule that
    /// halves it for every doubling of the host rate above 44.1 kHz. Like
    /// [`Self::set_sample_rate`] this is not RT-safe when the effective
    /// factor changes (chains are re-prepared for the new engine rate).
    pub fn set_oversampling(&mut self, factor: usize) {
        self.requested_oversample = factor.clamp(1, MAX_OVERSAMPLE);
        if effective_oversample(self.requested_oversample, self.sample_rate) != self.oversample {
            self.apply_engine_rate();
        }
    }

    fn apply_engine_rate(&mut self) {
        self.oversample = effective_oversample(self.requested_oversample, self.sample_rate);
        let engine_rate = self.engine_rate();
        let er = engine_rate as f32;
        self.allocator.set_sample_rate(engine_rate);
        self.allocator.set_oversample(self.oversample);
        for lfo in &mut self.sync_random_lfos {
            lfo.set_sample_rate(er);
        }
        self.main.set_sample_rate(er);
        self.bus_a.set_sample_rate(er);
        self.bus_b.set_sample_rate(er);
        self.decimator.reset(PolyMask::all_on());
        self.master_dc_filter.set_cutoff(MASTER_DC_CUTOFF_HZ, self.sample_rate as f32);
        self.master_dc_filter.hard_reset();
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

    /// Host transport for the next block (`SoundEngine::correctToTime` +
    /// `setBpm`): `seconds` is the song position, `bpm` the tempo,
    /// `playing` whether the transport runs. Transport-synced LFOs snap to
    /// `seconds` on trigger and transport-synced random LFOs follow it
    /// (sharing one value across all voices). While `playing`, the engine
    /// advances `seconds` itself across the sub-blocks of one
    /// [`Self::process`] call; call this once per host block.
    pub fn set_transport(&mut self, seconds: f64, bpm: f32, playing: bool) {
        self.set_bpm(bpm);
        self.transport_seconds = seconds;
        self.transport_playing = playing;
    }

    /// Turns the master DC blocker off. It is a Spinwave addition the
    /// reference does not have (see the comment at the blocker), so the
    /// golden bench pins it off to compare the shared DSP path. Nothing
    /// else should call this: a synth that lets DC through is worse.
    pub fn set_master_dc_blocker(&mut self, enabled: bool) {
        self.master_dc_enabled = enabled;
        self.master_dc_filter.hard_reset();
    }

    /// Turns the per-voice DC blockers off on every kernel. Same reason as
    /// the master one: the reference wires none of them, so the bench pins
    /// them off. Nothing else should call this.
    pub fn set_voice_dc_blockers(&mut self, enabled: bool) {
        for kernel in self.allocator.kernels_mut() {
            kernel.set_dc_blockers(enabled);
        }
    }

    /// Transport position the next block will start at.
    pub fn transport_seconds(&self) -> f64 {
        self.transport_seconds
    }

    /// Latency the plugin must report to the host, in HOST samples: the
    /// largest convolution latency among the chains whose convolution is
    /// on and loaded (engine-rate samples divided by the oversampling),
    /// else 0.
    pub fn latency_samples(&self) -> usize {
        let engine_latency = self
            .main
            .latency_samples()
            .max(self.bus_a.latency_samples())
            .max(self.bus_b.latency_samples());
        engine_latency / self.oversample
    }

    /// Swaps a prebuilt convolution reverb into one chain (RT-safe) and
    /// returns the previous one. Build the replacement OFF the audio
    /// thread: `ConvolutionReverb::new()` + `set_impulse_response(left,
    /// right, ir_rate, engine.engine_rate())` (the IR must be prepared at
    /// the engine rate, see [`Self::engine_rate`]); drop the returned
    /// instance off-thread as well.
    pub fn set_convolution_engine(
        &mut self,
        chain: ChainId,
        prebuilt: ConvolutionReverb,
    ) -> ConvolutionReverb {
        self.chain_mut(chain).set_convolution_engine(prebuilt)
    }

    /// Mono modulation offsets applied to the main chain in the last block.
    pub fn effects_offsets(&self) -> &EffectsModOffsets {
        &self.effects_offsets
    }

    // -- Parameter access ----------------------------------------------------

    /// Main-chain effect parameters (source compatibility for the historical
    /// single-chain API; see [`Self::chain_params`] for the buses).
    pub fn params(&self) -> &EffectsParams {
        self.main.params()
    }

    /// Main-chain effect parameters, mutable.
    pub fn params_mut(&mut self) -> &mut EffectsParams {
        self.main.params_mut()
    }

    fn chain(&self, chain: ChainId) -> &EffectChain {
        match chain {
            ChainId::Main => &self.main,
            ChainId::BusA => &self.bus_a,
            ChainId::BusB => &self.bus_b,
        }
    }

    fn chain_mut(&mut self, chain: ChainId) -> &mut EffectChain {
        match chain {
            ChainId::Main => &mut self.main,
            ChainId::BusA => &mut self.bus_a,
            ChainId::BusB => &mut self.bus_b,
        }
    }

    pub fn chain_params(&self, chain: ChainId) -> &EffectsParams {
        self.chain(chain).params()
    }

    pub fn chain_params_mut(&mut self, chain: ChainId) -> &mut EffectsParams {
        self.chain_mut(chain).params_mut()
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

    /// Bounds the active polyphony within the preallocated pool (clamped
    /// to `1..=MAX_ACTIVE_POLYPHONY`). Never allocates: every kernel exists
    /// since construction and already carries the patch, the engine rate
    /// and any installed samples / wavetables.
    pub fn set_polyphony(&mut self, polyphony: usize) {
        self.allocator.set_polyphony(polyphony);
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

    /// Kills all voices and hard-resets every effect chain and the
    /// decimator (`SoundEngine::allSoundsOff`).
    pub fn all_sounds_off(&mut self) {
        self.allocator.all_sounds_off();
        self.main.hard_reset();
        self.bus_a.hard_reset();
        self.bus_b.hard_reset();
        self.decimator.reset(PolyMask::all_on());
        self.master_dc_filter.hard_reset();
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

    /// The control-rate value of one modulation source, read the way the
    /// mono modulation readout reads it: from the most recently activated
    /// voice, reduced to that voice's left lane.
    ///
    /// This exists for the golden bench's `--probe`, which compares the
    /// modulation curve itself against the reference instead of inferring
    /// it from the audio. It reads state the audio path has already
    /// computed and changes nothing; nothing in the engine calls it.
    ///
    /// Zero when no voice has ever played: there is no source value yet.
    pub fn probe_source(&self, source: ModSource) -> f32 {
        match self.allocator.last_active_voice() {
            Some((pair, slot)) => {
                self.allocator.kernels()[pair].last_source_values().get(source).lane(2 * slot)
            }
            None => 0.0,
        }
    }

    /// Reseeds every generator in the engine from one seed, so a render's
    /// random draws depend on the seed and on nothing else — not on how
    /// many generators the process built before, not on which thread runs
    /// it. Voice kernel `k` gets `seed + 64k` (see
    /// [`SynthVoiceKernel::reseed`] for the layout inside a kernel), the
    /// transport-synced random LFOs `seed + 64 × 32 + i`.
    pub fn reseed(&mut self, seed: u32) {
        for (k, kernel) in self.allocator.kernels_mut().iter_mut().enumerate() {
            kernel.reseed(seed.wrapping_add(64 * k as u32));
        }
        for (i, random) in self.sync_random_lfos.iter_mut().enumerate() {
            random.reseed(seed.wrapping_add(64 * 32 + i as u32));
        }
    }

    /// The resolved modulation offset on filter 1's cutoff at the start of
    /// the last block — the control-rate offset plus the first sample of
    /// the audio-rate sum — per lane of the voice pair, read from the most
    /// recently activated voice.
    ///
    /// The companion to [`probe_source`](Self::probe_source): that one
    /// says whether the two engines agree about what a source is DOING,
    /// this one whether they agree about what reaches the destination.
    /// Per lane rather than folded, because a fold across voice lanes
    /// counted twice or dropped is invisible in a single scalar.
    pub fn probe_cutoff_offset(&self) -> [f32; 4] {
        self.probe_offset(ModDest::FilterCutoff(0), |offsets| offsets.filter_cutoff[0])
    }

    /// The same reading for oscillator 1's level. Both destinations are
    /// consumed per sample, so a connection from an envelope or LFO lands
    /// in the audio-rate sum and one from a macro, note or velocity in the
    /// control-rate offset; the probe adds the two, and a zero here means
    /// nothing reached the destination.
    pub fn probe_osc_level_offset(&self) -> [f32; 4] {
        self.probe_offset(ModDest::OscLevel(0), |offsets| offsets.osc_level[0])
    }

    /// The active voice's own lanes come first (left, right), then the
    /// other slot of the pair. Reading lanes 0 and 1 unconditionally was a
    /// bug that made a retriggered note look unmodulated: the second note
    /// of a case can land on slot 1, and lanes 0 and 1 then belong to the
    /// voice that just died.
    fn probe_offset(
        &self,
        dest: ModDest,
        control: impl Fn(&crate::kernel::ModOffsets) -> PolyF32,
    ) -> [f32; 4] {
        match self.allocator.last_active_voice() {
            Some((pair, slot)) => {
                let kernel = &self.allocator.kernels()[pair];
                let offset = control(kernel.last_offsets()) + kernel.audio_offset_at(dest, 0);
                let (own, other) = (2 * slot, 2 * (1 - slot));
                [offset.lane(own), offset.lane(own + 1), offset.lane(other), offset.lane(other + 1)]
            }
            None => [0.0; 4],
        }
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
            if self.transport_playing {
                self.transport_seconds += block as f64 / self.sample_rate as f64;
            }
            start += block;
        }
    }

    /// Advances the shared transport-synced random LFOs once for the block
    /// and hands the transport and their values to every kernel.
    fn update_transport(&mut self, os_samples: usize) {
        let seconds = self.transport_seconds;
        let bps = self.beats_per_second;
        let mut values = [PolyF32::ZERO; NUM_RANDOM_LFOS];
        if let Some(reference) = self.allocator.kernels().first() {
            for (i, lfo) in self.sync_random_lfos.iter_mut().enumerate() {
                let section = &reference.params.random_lfos[i];
                if !section.params.sync {
                    continue;
                }
                let mut params = section.params;
                params.frequency = section.sync.resolve(params.frequency, bps);
                lfo.correct_to_time(seconds);
                values[i] = lfo.process_control(&params, os_samples);
            }
        }
        for kernel in self.allocator.kernels_mut() {
            kernel.set_transport(seconds);
            kernel.set_shared_random_values(values);
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
        let os_samples = num_samples * self.oversample;

        self.update_transport(os_samples);

        // Run the voices and fold the two voice slots into one stereo
        // signal replicated in both vector halves: [L, R, L, R]. The main
        // bus feeds the effect chains; the direct-out bus is kept aside and
        // added after them (reference `output_total_`). Voices render
        // before the effect params resolve so the mono effects modulation
        // reads this block's source values.
        let mut mix = std::mem::take(&mut self.mix_bus);
        let mut direct = std::mem::take(&mut self.direct_bus);
        let mut folded = std::mem::take(&mut self.folded_bus);
        let mut bus_a_buffer = std::mem::take(&mut self.bus_a_scratch);
        let mut bus_b_buffer = std::mem::take(&mut self.bus_b_scratch);

        mix[..os_samples].fill(PolyF32::ZERO);
        direct[..os_samples].fill(PolyF32::ZERO);
        bus_a_buffer[..os_samples].fill(PolyF32::ZERO);
        bus_b_buffer[..os_samples].fill(PolyF32::ZERO);
        self.allocator.process(os_samples, |outputs: crate::allocator::KernelOutputs| {
            for (dest, &src) in mix.iter_mut().zip(outputs.main) {
                *dest += src;
            }
            if let Some(kernel_direct) = outputs.direct {
                for (dest, &src) in direct.iter_mut().zip(kernel_direct) {
                    *dest += src;
                }
            }
            if let Some(kernel_bus) = outputs.bus_a {
                for (dest, &src) in bus_a_buffer.iter_mut().zip(kernel_bus) {
                    *dest += src;
                }
            }
            if let Some(kernel_bus) = outputs.bus_b {
                for (dest, &src) in bus_b_buffer.iter_mut().zip(kernel_bus) {
                    *dest += src;
                }
            }
        });
        for (fold, &sum) in folded[..os_samples].iter_mut().zip(&mix[..os_samples]) {
            *fold = sum + sum.swap_voices();
        }
        // Fold the hard-routed bus voice signals in place; the send taps
        // add on top inside process_bus.
        for value in bus_a_buffer[..os_samples].iter_mut() {
            *value += value.swap_voices();
        }
        for value in bus_b_buffer[..os_samples].iter_mut() {
            *value += value.swap_voices();
        }

        // Mono modulation offsets for the MAIN chain's bus effects: sources
        // come from the most recently activated VOICE, reduced to its left
        // lane (Vital's mono modulations). Offsets hold their last value
        // when every voice has died, like the reference control-rate
        // readouts.
        if self.effects_matrix.connections.is_empty() {
            self.effects_offsets.clear();
        } else if let Some((pair, slot)) = self.allocator.last_active_voice() {
            let kernel = &self.allocator.kernels()[pair];
            self.effects_matrix.resolve(
                kernel.last_source_values(),
                2 * slot,
                kernel.matrix.amount_offsets(),
                kernel.matrix.power_offsets(),
                &mut self.effects_offsets,
            );
        }

        // Bus filter keytrack follows the last played note.
        let keytrack_note = self.allocator.last_played_note();
        self.main.set_keytrack_note(keytrack_note);
        self.bus_a.set_keytrack_note(keytrack_note);
        self.bus_b.set_keytrack_note(keytrack_note);

        let bps = self.beats_per_second;
        let resolved_main = self.main.resolve(bps, &self.effects_offsets);

        // The send buses tap the folded voice signal before the main chain
        // and run their own (unmodulated) chains on `send * folded`.
        Self::process_bus(
            &mut self.bus_a,
            &self.mixer.bus_a,
            bps,
            &folded[..os_samples],
            &mut bus_a_buffer[..os_samples],
        );
        Self::process_bus(
            &mut self.bus_b,
            &self.mixer.bus_b,
            bps,
            &folded[..os_samples],
            &mut bus_b_buffer[..os_samples],
        );

        // Serial buses feed the main chain input; parallel buses sum with
        // its output.
        if self.mixer.bus_a.output == BusOutput::MainChain {
            Self::add_bus_return(&self.mixer.bus_a, &bus_a_buffer[..os_samples], &mut folded[..os_samples]);
        }
        if self.mixer.bus_b.output == BusOutput::MainChain {
            Self::add_bus_return(&self.mixer.bus_b, &bus_b_buffer[..os_samples], &mut folded[..os_samples]);
        }

        // The main chain processes the dry bus (plus serial returns) in place.
        self.main.process(&resolved_main, &mut folded[..os_samples]);

        if self.mixer.bus_a.output == BusOutput::Master {
            Self::add_bus_return(&self.mixer.bus_a, &bus_a_buffer[..os_samples], &mut folded[..os_samples]);
        }
        if self.mixer.bus_b.output == BusOutput::Master {
            Self::add_bus_return(&self.mixer.bus_b, &bus_b_buffer[..os_samples], &mut folded[..os_samples]);
        }

        // Add the folded direct-out bus after the chains, like the reference
        // `output_total_ = effect_chain_ + voice_handler_->getDirectOutput()`.
        for (out, &sum) in folded[..os_samples].iter_mut().zip(&direct[..os_samples]) {
            *out += sum + sum.swap_voices();
        }

        // Decimate back to the host rate, then the master path:
        // DC blocker → stereo encoder → smoothed volume → meter → clamp.
        let mut decimated = std::mem::take(&mut self.decimated);
        self.decimator.process(
            &folded[..os_samples],
            self.engine_rate(),
            self.sample_rate,
            &mut decimated[..num_samples],
        );

        // The voices block their own DC, but the effect chain adds more:
        // asymmetric waveshaping is the usual source, and heavy distortion
        // patches drift several percent off zero, which costs headroom and
        // thumps on note transitions. The reference leaves this unfiltered
        // (`DcFilter` exists there but is wired nowhere); one blocker on
        // the way out costs a one-pole per channel and cannot be heard.
        if self.master_dc_enabled {
            let mut master_dc = self.master_dc_filter;
            master_dc.process_in_place(&mut decimated[..num_samples]);
            self.master_dc_filter = master_dc;
        }

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
        self.folded_bus = folded;
        self.bus_a_scratch = bus_a_buffer;
        self.bus_b_scratch = bus_b_buffer;
        self.decimated = decimated;
    }

    /// Runs a bus chain over `voice_bus + send * folded` (the hard-routed
    /// producer signal plus the pre-main send tap). `buffer` holds the
    /// folded voice bus on entry. Skipped entirely when the bus is off.
    fn process_bus(
        chain: &mut EffectChain,
        bus: &BusParams,
        beats_per_second: f32,
        folded: &[PolyF32],
        buffer: &mut [PolyF32],
    ) {
        if !bus.on {
            return;
        }
        let send = PolyF32::splat(bus.send.clamp(0.0, 1.0));
        for (dest, &src) in buffer.iter_mut().zip(folded) {
            *dest += src * send;
        }
        let resolved = chain.resolve(beats_per_second, &EffectsModOffsets::default());
        chain.process(&resolved, buffer);
    }

    /// Sums a processed bus back into the main output with its return gain.
    fn add_bus_return(bus: &BusParams, bus_buffer: &[PolyF32], out: &mut [PolyF32]) {
        if !bus.on {
            return;
        }
        let db = bus.return_gain_db.clamp(SMOOTH_VOLUME_MIN_DB, SMOOTH_VOLUME_MAX_DB);
        let gain = if db <= SMOOTH_VOLUME_MIN_DB {
            PolyF32::ZERO
        } else {
            math::db_to_magnitude(PolyF32::splat(db))
        };
        for (dest, &src) in out.iter_mut().zip(bus_buffer) {
            *dest += src * gain;
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
    fn effect_order_swapping_last_two_legacy_effects_encodes_to_one() {
        // A single inversion at the last legacy position (Phaser/Reverb) is
        // the lowest non-zero code; the extras keep their anchored places
        // (shifter after flanger, convolution after reverb).
        let order = [
            Effect::Chorus,
            Effect::Compressor,
            Effect::Delay,
            Effect::Distortion,
            Effect::Eq,
            Effect::FilterFx,
            Effect::Flanger,
            Effect::FrequencyShifter,
            Effect::Reverb,
            Effect::Convolution,
            Effect::Phaser,
        ];
        assert_eq!(encode_order(&order), 1);
        assert_eq!(decode_order(1), order);
    }

    #[test]
    fn effect_order_roundtrip() {
        let reversed = [
            Effect::Convolution,
            Effect::Reverb,
            Effect::Phaser,
            Effect::FrequencyShifter,
            Effect::Flanger,
            Effect::FilterFx,
            Effect::Eq,
            Effect::Distortion,
            Effect::Delay,
            Effect::Compressor,
            Effect::Chorus,
        ];
        assert_eq!(decode_order(encode_order(&reversed)), reversed);

        // Legacy codes (9! - 1 is the largest) decode to the reference
        // relative order plus the anchored extras, and roundtrip.
        for code in [1u32, 2, 100, 5040, 362_879] {
            let order = decode_order(code);
            assert_eq!(encode_order(&order), code);
            let shifter = order.iter().position(|&e| e == Effect::FrequencyShifter).unwrap();
            let flanger = order.iter().position(|&e| e == Effect::Flanger).unwrap();
            let convolution = order.iter().position(|&e| e == Effect::Convolution).unwrap();
            let reverb = order.iter().position(|&e| e == Effect::Reverb).unwrap();
            assert_eq!(shifter, flanger + 1, "code {code}: shifter not after flanger");
            assert_eq!(convolution, reverb + 1, "code {code}: convolution not after reverb");
        }
        // Reference check against the 9-effect codec: reversed legacy order.
        let legacy_reversed = decode_order(362_879);
        let legacy_only: Vec<Effect> = legacy_reversed
            .iter()
            .copied()
            .filter(|e| (*e as usize) < NUM_LEGACY_EFFECTS)
            .collect();
        assert_eq!(
            legacy_only,
            vec![
                Effect::Reverb,
                Effect::Phaser,
                Effect::Flanger,
                Effect::FilterFx,
                Effect::Eq,
                Effect::Distortion,
                Effect::Delay,
                Effect::Compressor,
                Effect::Chorus,
            ]
        );
        // Explicit extra placements roundtrip too.
        let mut moved = DEFAULT_ORDER;
        moved.swap(0, 10); // convolution first, chorus last
        assert_eq!(decode_order(encode_order(&moved)), moved);
        assert!(encode_order(&moved) >= 362_880);
    }

    // -- Review fixes -------------------------------------------------------

    #[test]
    fn set_polyphony_never_allocates_and_kernels_run_at_engine_rate() {
        let mut engine = make_engine();
        let pairs = engine.allocator().kernels().len();
        assert_eq!(pairs, MAX_POLYPHONY / 2);
        assert_eq!(engine.allocator().polyphony(), DEFAULT_POLYPHONY);
        engine.set_polyphony(48);
        assert_eq!(engine.allocator().polyphony(), 48);
        assert_eq!(engine.allocator().kernels().len(), pairs);
        let engine_rate = engine.engine_rate();
        assert_eq!(engine_rate, 44100 * 2);
        for kernel in engine.allocator().kernels() {
            assert_eq!(kernel.sample_rate(), engine_rate);
        }
        // 48 simultaneous notes all sound (no pool growth needed).
        for note in 0..48 {
            engine.note_on(30 + note, 0.8, 0, 0);
        }
        assert_eq!(engine.num_active_voices(), 48);
        let (left, _) = render(&mut engine, 2);
        assert!(peak(&left) > 0.1);
    }

    #[test]
    fn mono_modulation_reads_the_last_voice_not_the_last_pair() {
        let mut engine = make_engine();
        engine.params_mut().distortion_on = true;
        engine.effects_matrix.connections.push(EffectsConnection {
            source: ModSource::Velocity,
            dest: EffectsModDest::DistortionDrive,
            transform: ModulationTransform::with_amount(1.0, 60.0),
        });
        // Both notes land on pair 0: slot 0 at velocity 1, slot 1 at 0.2.
        engine.note_on(60, 1.0, 0, 0);
        engine.note_on(64, 0.2, 0, 0);
        let _ = render(&mut engine, 1);
        let drive = engine.effects_offsets().distortion_drive_db;
        assert!((drive - 12.0).abs() < 1e-3, "expected the last voice's 0.2 × 60, got {drive}");
    }

    #[test]
    fn oversampling_halves_above_44100() {
        assert_eq!(effective_oversample(2, 44100), 2);
        assert_eq!(effective_oversample(2, 48000), 2);
        assert_eq!(effective_oversample(2, 88200), 1);
        assert_eq!(effective_oversample(2, 96000), 1);
        assert_eq!(effective_oversample(4, 96000), 2);
        assert_eq!(effective_oversample(4, 192000), 1);
        assert_eq!(effective_oversample(1, 44100), 1);
        assert_eq!(effective_oversample(8, 44100), 4);

        let mut engine = make_engine();
        engine.set_oversampling(4);
        assert_eq!(engine.oversampling(), 4);
        assert_eq!(engine.engine_rate(), 44100 * 4);
        for kernel in engine.allocator().kernels() {
            assert_eq!(kernel.sample_rate(), 44100 * 4);
        }
        engine.note_on(60, 1.0, 0, 0);
        let (left, _) = render(&mut engine, 4);
        assert!(left.iter().all(|v| v.is_finite()));
        assert!(peak(&left) > 0.01);

        engine.set_sample_rate(96000);
        assert_eq!(engine.oversampling(), 2);
        engine.set_oversampling(2);
        assert_eq!(engine.oversampling(), 1);
        assert_eq!(engine.engine_rate(), 96000);
        engine.note_on(64, 1.0, 0, 0);
        let (left, _) = render(&mut engine, 4);
        assert!(left.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn all_sounds_off_silences_immediately_including_the_decimator() {
        let mut engine = make_engine();
        engine.note_on(60, 1.0, 0, 0);
        let (left, _) = render(&mut engine, 4);
        assert!(peak(&left) > 0.01);
        engine.all_sounds_off();
        let (left, right) = render(&mut engine, 2);
        assert!(left.iter().chain(&right).all(|&v| v == 0.0), "state survived all_sounds_off");
    }

    #[test]
    fn synced_random_lfo_is_shared_by_all_voices() {
        let mut engine = make_engine();
        engine.kernel_params_mut(|params| {
            params.random_lfos[0].params.sync = true;
            // Fast enough to wrap several cycles: like the reference, a
            // synced random LFO only draws a new target at a cycle wrap.
            params.random_lfos[0].params.frequency = PolyF32::splat(40.0);
            params.random_lfos[0].params.stereo = true;
        });
        engine.set_transport(0.0, 120.0, true);
        engine.note_on(60, 1.0, 0, 0);
        engine.note_on(64, 1.0, 0, 0);
        engine.note_on(67, 1.0, 0, 0);
        let mut values = Vec::new();
        for _ in 0..80 {
            let _ = render(&mut engine, 1);
            let kernels = engine.allocator().kernels();
            let first = kernels[0].last_source_values().random_lfos[0];
            let second = kernels[1].last_source_values().random_lfos[0];
            // Both voices of a pair agree (stereo lanes may differ: the
            // LFO is in stereo mode), and every pair agrees with pair 0.
            assert_eq!(first.lane(0), first.lane(2), "voices differ within pair 0");
            assert_eq!(first.lane(1), first.lane(3), "voices differ within pair 0");
            for lane in 0..4 {
                assert_eq!(second.lane(lane), first.lane(lane), "pair 1 disagrees with pair 0");
            }
            values.push(first.lane(0));
        }
        let distinct = values.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(distinct > 3, "synced random LFO never moved with the transport");
        assert!(engine.transport_seconds() > 0.1);
    }

    #[test]
    fn latency_reports_the_loaded_convolution() {
        use spinwave_dsp::effects::{convolution::LATENCY_SAMPLES, ir_plate, ConvolutionReverb};
        let mut engine = make_engine();
        assert_eq!(engine.latency_samples(), 0);
        engine.params_mut().convolution_on = true;
        assert_eq!(engine.latency_samples(), 0, "no IR loaded yet");

        let engine_rate = engine.engine_rate();
        let (left, right) = ir_plate(0.3, engine_rate);
        let mut prebuilt = ConvolutionReverb::new();
        prebuilt
            .set_impulse_response(&left, &right, engine_rate, engine_rate)
            .expect("valid IR");
        let _old = engine.set_convolution_engine(ChainId::Main, prebuilt);
        assert_eq!(engine.latency_samples(), LATENCY_SAMPLES / engine.oversampling());

        // The convolution rings after the voice dies.
        engine.params_mut().convolution.dry_wet = 0.8;
        engine.note_on(60, 1.0, 0, 0);
        let _ = render(&mut engine, 8);
        engine.note_off(60, 0.5, 0, 0);
        let _ = render(&mut engine, 20);
        assert_eq!(engine.num_active_voices(), 0);
        let (tail, _) = render(&mut engine, 8);
        assert!(peak(&tail) > 1e-5, "convolution tail is silent");
        engine.params_mut().convolution_on = false;
        assert_eq!(engine.latency_samples(), 0);
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

    // -- Effects mixer (send buses) ------------------------------------------

    /// Configures bus A as a reverb send with the given send level, plays a
    /// short note, waits out the voice, and returns the tail peak.
    fn bus_reverb_tail(send: f32, bus_on: bool) -> f32 {
        let mut engine = make_engine();
        engine.mixer.bus_a.on = bus_on;
        engine.mixer.bus_a.send = send;
        {
            let effects = engine.chain_params_mut(ChainId::BusA);
            effects.reverb_on = true;
            effects.reverb.wet = PolyF32::splat(0.8);
            effects.reverb.decay_time = PolyF32::splat(2.0);
        }
        engine.note_on(60, 1.0, 0, 0);
        let _ = render(&mut engine, 8);
        engine.note_off(60, 0.5, 0, 0);
        let _ = render(&mut engine, 20);
        assert_eq!(engine.num_active_voices(), 0, "voice was not retired");
        let (tail, _) = render(&mut engine, 4);
        peak(&tail)
    }

    #[test]
    fn hard_routed_producer_feeds_bus_a_without_send() {
        // Osc routed to BusA by destination, send at ZERO: audio must still
        // reach the bus chain (reverb tail), proving hard routing works.
        let mut engine = make_engine();
        engine.mixer.bus_a.on = true;
        engine.mixer.bus_a.send = 0.0;
        {
            let effects = engine.chain_params_mut(ChainId::BusA);
            effects.reverb_on = true;
            effects.reverb.wet = PolyF32::splat(0.8);
            effects.reverb.decay_time = PolyF32::splat(2.0);
        }
        engine.kernel_params_mut(|params| {
            params.oscillators[0].destination =
                crate::kernel::ProducerDestination::BusA;
        });
        engine.note_on(60, 1.0, 0, 0);
        let _ = render(&mut engine, 8);
        engine.note_off(60, 0.5, 0, 0);
        let _ = render(&mut engine, 20);
        let (tail, _) = render(&mut engine, 4);
        assert!(peak(&tail) > 1e-5, "hard-routed bus reverb tail is silent");

        // Same patch with the bus off: nothing reaches the output at all
        // (the producer is routed only to the muted bus).
        let mut muted = make_engine();
        muted.kernel_params_mut(|params| {
            params.oscillators[0].destination =
                crate::kernel::ProducerDestination::BusA;
        });
        muted.note_on(60, 1.0, 0, 0);
        let (during, _) = render(&mut muted, 8);
        assert!(peak(&during) < 1e-6, "muted bus leaked: {}", peak(&during));
    }

    #[test]
    fn serial_bus_feeds_the_main_chain() {
        // Bus A (plain pass-through chain) returns into the main chain,
        // whose distortion is cranked: the bus signal must come out
        // distorted, differing from the parallel configuration.
        let render_config = |output: BusOutput| {
            let mut engine = make_engine();
            engine.mixer.bus_a.on = true;
            engine.mixer.bus_a.send = 1.0;
            engine.mixer.bus_a.output = output;
            {
                let effects = engine.params_mut();
                effects.distortion_on = true;
                effects.distortion_type = spinwave_dsp::effects::DistortionType::HardClip;
                effects.distortion_drive_db = 30.0;
                effects.distortion_mix = 1.0;
            }
            engine.note_on(48, 1.0, 0, 0);
            let _ = render(&mut engine, 8);
            let (left, _) = render(&mut engine, 8);
            left
        };
        let serial = render_config(BusOutput::MainChain);
        let parallel = render_config(BusOutput::Master);
        let difference: f32 = serial
            .iter()
            .zip(&parallel)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(difference > 1e-2, "serial vs parallel bus made no difference");
    }

    #[test]
    fn bus_send_routes_through_bus_reverb_and_main_stays_dry() {
        // With the send up, the bus reverb tail rings on after the voice
        // dies even though the MAIN chain has no effects at all.
        assert!(bus_reverb_tail(1.0, true) > 1e-5, "bus reverb tail is silent");
        // With the send at zero the bus gets no signal: what remains is
        // the master DC blocker settling, far below the live tail.
        assert!(
            bus_reverb_tail(0.0, true) < bus_reverb_tail(1.0, true) * 0.02,
            "tail present with send 0"
        );
    }

    #[test]
    fn bus_off_ignores_send_and_return() {
        // The bus contributes nothing: what is left is the master DC
        // blocker settling after the note, orders below the live tail.
        assert!(bus_reverb_tail(1.0, false) < bus_reverb_tail(1.0, true) * 0.02);
    }

    #[test]
    fn bus_return_gain_scales_bus_output() {
        let render_bus_rms = |return_gain_db: f32| {
            let mut engine = make_engine();
            engine.mixer.bus_b.on = true;
            engine.mixer.bus_b.send = 1.0;
            engine.mixer.bus_b.return_gain_db = return_gain_db;
            {
                // Distortion at 0 dB drive: an audible, deterministic bus.
                let effects = engine.chain_params_mut(ChainId::BusB);
                effects.distortion_on = true;
            }
            engine.note_on(60, 1.0, 0, 0);
            let _ = render(&mut engine, 16);
            let (left, _) = render(&mut engine, 16);
            rms(&left)
        };

        let loud = render_bus_rms(0.0);
        let quiet = render_bus_rms(-20.0);
        // The dry main signal is present in both renders; the bus return
        // shrinking by -20 dB must still change the sum audibly.
        assert!(loud > quiet * 1.05, "return gain had no effect: {loud} vs {quiet}");
    }

    #[test]
    fn chain_params_mut_main_aliases_params_mut() {
        let mut engine = make_engine();
        engine.params_mut().distortion_drive_db = 17.5;
        assert_eq!(engine.chain_params(ChainId::Main).distortion_drive_db, 17.5);
        engine.chain_params_mut(ChainId::Main).distortion_drive_db = -3.0;
        assert_eq!(engine.params().distortion_drive_db, -3.0);
    }
}
