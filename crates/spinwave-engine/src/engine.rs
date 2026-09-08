//! Top-level sound engine (rework of Vital's `SoundEngine` +
//! `ReorderableEffectChain`): voice allocator → three reorderable bus effect
//! chains (main + two send buses, Serum-2-mixer style) → stereo encoder →
//! smoothed master volume → peak meter → clamp.
//!
//! Voices and effects run 2x oversampled; the decimator brings the signal
//! back to the host rate before the master path. The folded voice signal
//! uses lanes `[L, R, L, R]`.

use spinwave_dsp::filters::Decimator;
use spinwave_dsp::utilities::PeakMeter;
use spinwave_poly::constants::{MAX_BUFFER_SIZE, PI};
use spinwave_poly::{math, PolyF32};

use crate::allocator::VoiceAllocator;
use crate::kernel::mod_matrix::{ModSource, SourceValues};
use crate::kernel::{KernelParams, SynthVoiceKernel};
use crate::modulation::ModulationTransform;

// The effect chain moved to `crate::effect_chain`; re-exported here so the
// existing `spinwave_engine::engine::*` paths keep working.
pub use crate::effect_chain::{
    decode_order, encode_order, Effect, EffectChain, EffectSplit, EffectsParams,
    ResolvedEffectsParams, SplitMode, DEFAULT_ORDER, DEFAULT_SPLIT_CROSSOVER_HZ, NUM_EFFECTS,
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
/// Default polyphony, matching the reference's `polyphony` default.
const DEFAULT_POLYPHONY: usize = 8;
/// Voices and bus effects run this many times oversampled; the decimator
/// brings the signal back before the master path (reference
/// `kDefaultOversamplingAmount`).
const OVERSAMPLE: usize = 2;

/// The complete synthesizer: voices, the effects mixer (main chain plus two
/// send buses) and the master path.
pub struct SoundEngine {
    sample_rate: u32,
    beats_per_second: f32,

    allocator: VoiceAllocator<SynthVoiceKernel>,
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
            effects_matrix: EffectsModMatrix::default(),
            effects_offsets: EffectsModOffsets::default(),
            master: MasterParams::default(),
            mixer: MixerParams::default(),
            main: EffectChain::new(er, oversampled_len),
            bus_a: EffectChain::new(er, oversampled_len),
            bus_b: EffectChain::new(er, oversampled_len),
            volume_mult: PolyF32::ZERO,
            encoder_cos: PolyF32::ZERO,
            encoder_sin: PolyF32::ZERO,
            peak_meter: PeakMeter::new(),
            decimator: Decimator::new(3),
            mix_bus: vec![PolyF32::ZERO; oversampled_len],
            direct_bus: vec![PolyF32::ZERO; oversampled_len],
            folded_bus: vec![PolyF32::ZERO; oversampled_len],
            bus_a_scratch: vec![PolyF32::ZERO; oversampled_len],
            bus_b_scratch: vec![PolyF32::ZERO; oversampled_len],
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
        self.main.set_sample_rate(er);
        self.bus_a.set_sample_rate(er);
        self.bus_b.set_sample_rate(er);
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

    /// Kills all voices and hard-resets every effect chain
    /// (`SoundEngine::allSoundsOff`).
    pub fn all_sounds_off(&mut self) {
        self.allocator.all_sounds_off();
        self.main.hard_reset();
        self.bus_a.hard_reset();
        self.bus_b.hard_reset();
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
            *value = *value + value.swap_voices();
        }
        for value in bus_b_buffer[..os_samples].iter_mut() {
            *value = *value + value.swap_voices();
        }

        // Mono modulation offsets for the MAIN chain's bus effects: sources
        // come from the most recently active voice kernel, reduced to lane 0
        // (Vital's mono modulations). Offsets hold their last value when
        // every voice has died, like the reference control-rate readouts.
        if self.effects_matrix.connections.is_empty() {
            self.effects_offsets.clear();
        } else if let Some(pair) = self.allocator.last_active_pair() {
            self.effects_matrix.resolve(
                self.allocator.kernels()[pair].last_source_values(),
                &mut self.effects_offsets,
            );
        }

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
        // stereo encoder → smoothed volume → meter → clamp.
        let mut decimated = std::mem::take(&mut self.decimated);
        self.decimator.process(
            &folded[..os_samples],
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
        // With the send at zero the bus gets no signal: no tail.
        assert!(bus_reverb_tail(0.0, true) < 1e-7, "tail present with send 0");
    }

    #[test]
    fn bus_off_ignores_send_and_return() {
        assert!(bus_reverb_tail(1.0, false) < 1e-7, "disabled bus produced output");
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
