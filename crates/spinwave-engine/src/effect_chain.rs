//! Reusable bus effect chain (extracted from `SoundEngine`): the nine
//! reorderable bus effects, their parameters, per-effect enable-edge resets
//! and per-effect signal splitting (full / mid / side / low / high).
//!
//! The engine owns three chains ([`crate::engine::ChainId`]): the main chain
//! plus two send buses, Serum-2-mixer style.

use spinwave_dsp::effects::{
    Chorus, ChorusParams, DelayParams, DelayStyle, Distortion, DistortionType, Equalizer,
    EqualizerParams, Flanger, FlangerParams, MultibandCompressor, MultibandCompressorParams,
    Phaser, PhaserParams, Reverb, ReverbParams, StereoDelay,
};
use spinwave_dsp::filters::LinkwitzRileyFilter;
use spinwave_poly::utils::{decode_mid_side, encode_mid_side, interpolate};
use spinwave_poly::{PolyF32, PolyMask};

use crate::engine::EffectsModOffsets;
use crate::kernel::voice_filter::{VoiceFilter, VoiceFilterParams};
use crate::tempo::SyncedFrequency;

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

/// Which part of the signal an effect slot processes; the untouched part is
/// recombined after the effect (Serum-2-style per-effect splitting).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SplitMode {
    #[default]
    Full,
    Mid,
    Side,
    Low,
    High,
}

/// Default crossover for the [`SplitMode::Low`]/[`SplitMode::High`] split.
pub const DEFAULT_SPLIT_CROSSOVER_HZ: f32 = 1000.0;

/// Per-effect signal split settings. `crossover_hz` only matters for the
/// `Low`/`High` modes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectSplit {
    pub mode: SplitMode,
    pub crossover_hz: f32,
}

impl Default for EffectSplit {
    fn default() -> EffectSplit {
        EffectSplit { mode: SplitMode::Full, crossover_hz: DEFAULT_SPLIT_CROSSOVER_HZ }
    }
}

/// All bus effect parameters plus the chain order. The `frequency` /
/// `rate` / `period_samples` fields of the tempo-syncable dsp param structs
/// are overwritten from the corresponding [`SyncedFrequency`] every block.
#[derive(Clone, Debug)]
pub struct EffectsParams {
    pub order: [Effect; NUM_EFFECTS],

    /// Per-effect signal splitting, indexed by `Effect as usize` (not by
    /// chain position).
    pub split: [EffectSplit; NUM_EFFECTS],

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
            split: [EffectSplit::default(); NUM_EFFECTS],
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

/// `DelayModule::kMaxDelayTime` in seconds.
const MAX_DELAY_TIME: f32 = 4.0;

pub(crate) fn delay_max_samples(sample_rate: f32) -> usize {
    (MAX_DELAY_TIME * sample_rate) as usize + 1
}

/// Per-block effect parameters after tempo-sync resolution and mono
/// modulation offsets: what [`EffectChain::process`] actually consumes.
#[derive(Clone, Debug)]
pub struct ResolvedEffectsParams {
    pub chorus: ChorusParams,
    pub compressor: MultibandCompressorParams,
    pub delay: DelayParams,
    pub distortion_type: DistortionType,
    pub distortion_drive_db: f32,
    pub distortion_mix: f32,
    pub eq: EqualizerParams,
    pub filter_fx: VoiceFilterParams,
    pub flanger: FlangerParams,
    pub phaser: PhaserParams,
    pub reverb: ReverbParams,
}

/// One reorderable bus effect chain: nine effect instances, per-effect
/// on/off with enable-edge resets, and per-effect signal splitting.
pub struct EffectChain {
    engine_rate: f32,
    params: EffectsParams,

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

    /// Distortion dry/wet ramp state (`DistortionModule::processWithInput`).
    distortion_mix: PolyF32,

    /// Per-slot crossover filters for the `Low`/`High` split modes,
    /// recreated when the slot's crossover changes materially.
    split_filters: [LinkwitzRileyFilter; NUM_EFFECTS],
    split_crossover_hz: [f32; NUM_EFFECTS],

    // Preallocated scratch buffers (no allocation in `process`).
    out_scratch: Vec<PolyF32>,
    split_a: Vec<PolyF32>,
    split_b: Vec<PolyF32>,
    drive_scratch: Vec<PolyF32>,
}

impl EffectChain {
    /// `engine_rate` is the (oversampled) rate the chain runs at;
    /// `max_block` the largest block `process` will see, in samples at that
    /// rate.
    pub fn new(engine_rate: f32, max_block: usize) -> EffectChain {
        let er = engine_rate;
        EffectChain {
            engine_rate: er,
            params: EffectsParams::default(),
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
            split_filters: std::array::from_fn(|_| {
                LinkwitzRileyFilter::new(DEFAULT_SPLIT_CROSSOVER_HZ, er)
            }),
            split_crossover_hz: [DEFAULT_SPLIT_CROSSOVER_HZ; NUM_EFFECTS],
            out_scratch: vec![PolyF32::ZERO; max_block],
            split_a: vec![PolyF32::ZERO; max_block],
            split_b: vec![PolyF32::ZERO; max_block],
            drive_scratch: vec![PolyF32::ZERO; max_block],
        }
    }

    pub fn set_sample_rate(&mut self, engine_rate: f32) {
        let er = engine_rate;
        self.engine_rate = er;
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
        for (filter, &hz) in self.split_filters.iter_mut().zip(&self.split_crossover_hz) {
            *filter = LinkwitzRileyFilter::new(hz, er);
        }
    }

    pub fn params(&self) -> &EffectsParams {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut EffectsParams {
        &mut self.params
    }

    /// Hard-resets every effect's state (`SoundEngine::allSoundsOff`).
    pub fn hard_reset(&mut self) {
        self.chorus.hard_reset();
        self.compressor.reset();
        self.delay.hard_reset();
        self.distortion.hard_reset();
        self.equalizer.hard_reset();
        self.filter_fx.hard_reset();
        self.flanger.hard_reset();
        self.phaser.hard_reset(&self.params.phaser);
        self.reverb.hard_reset();
        for filter in self.split_filters.iter_mut() {
            filter.reset(PolyMask::all_on());
        }
    }

    /// Resolves the stored params for one block: tempo sync against
    /// `beats_per_second`, plus the mono modulation offsets (pass a default
    /// [`EffectsModOffsets`] for an unmodulated chain). The stored params
    /// stay unmodulated. Exponential-scale destinations (frequencies, reverb
    /// decay) get their offsets in the log2 domain.
    pub fn resolve(&self, beats_per_second: f32, mods: &EffectsModOffsets) -> ResolvedEffectsParams {
        let bps = beats_per_second;
        let params = &self.params;

        let mut chorus_params = params.chorus;
        chorus_params.wet = (chorus_params.wet + mods.chorus_dry_wet).clamp(0.0, 1.0);
        chorus_params.feedback =
            (chorus_params.feedback + mods.chorus_feedback).clamp(-0.95, 0.95);
        chorus_params.mod_depth =
            (chorus_params.mod_depth + mods.chorus_mod_depth).clamp(0.0, 1.0);
        chorus_params.frequency = PolyF32::splat(
            params.chorus_sync.frequency_hz(bps) * mods.chorus_frequency.exp2(),
        );

        let mut flanger_params = params.flanger;
        flanger_params.wet = (flanger_params.wet + mods.flanger_dry_wet).clamp(0.0, 0.5);
        flanger_params.feedback =
            (flanger_params.feedback + mods.flanger_feedback).clamp(-1.0, 1.0);
        flanger_params.mod_depth =
            (flanger_params.mod_depth + mods.flanger_mod_depth).clamp(0.0, 1.0);
        flanger_params.phase_offset =
            (flanger_params.phase_offset + mods.flanger_phase_offset).clamp(0.0, 1.0);
        flanger_params.frequency = PolyF32::splat(
            params.flanger_sync.frequency_hz(bps) * mods.flanger_frequency.exp2(),
        );

        let mut phaser_params = params.phaser;
        phaser_params.mix = (phaser_params.mix + mods.phaser_dry_wet).clamp(0.0, 1.0);
        phaser_params.feedback_gain =
            (phaser_params.feedback_gain + mods.phaser_feedback).clamp(0.0, 1.0);
        phaser_params.mod_depth =
            (phaser_params.mod_depth + mods.phaser_mod_depth).clamp(0.0, 48.0);
        phaser_params.blend = (phaser_params.blend + mods.phaser_blend).clamp(0.0, 2.0);
        phaser_params.rate = PolyF32::splat(
            params.phaser_sync.frequency_hz(bps) * mods.phaser_frequency.exp2(),
        );

        let mut delay_params = self.resolve_delay_params(bps, mods);
        delay_params.feedback = (delay_params.feedback + mods.delay_feedback).clamp(-1.0, 1.0);
        delay_params.wet = (delay_params.wet + mods.delay_dry_wet).clamp(0.0, 1.0);

        let mut compressor_params = params.compressor;
        compressor_params.mix = (compressor_params.mix + mods.compressor_mix).clamp(0.0, 1.0);
        compressor_params.low_output_gain_db =
            (compressor_params.low_output_gain_db + mods.compressor_low_gain).clamp(-30.0, 30.0);
        compressor_params.band_output_gain_db = (compressor_params.band_output_gain_db
            + mods.compressor_band_gain)
            .clamp(-30.0, 30.0);
        compressor_params.high_output_gain_db = (compressor_params.high_output_gain_db
            + mods.compressor_high_gain)
            .clamp(-30.0, 30.0);

        let mut eq_params = params.eq;
        eq_params.low_cutoff_midi += PolyF32::splat(mods.eq_low_cutoff);
        eq_params.band_cutoff_midi += PolyF32::splat(mods.eq_band_cutoff);
        eq_params.high_cutoff_midi += PolyF32::splat(mods.eq_high_cutoff);
        eq_params.low_gain_db = (eq_params.low_gain_db + mods.eq_low_gain).clamp(-15.0, 15.0);
        eq_params.band_gain_db = (eq_params.band_gain_db + mods.eq_band_gain).clamp(-15.0, 15.0);
        eq_params.high_gain_db =
            (eq_params.high_gain_db + mods.eq_high_gain).clamp(-15.0, 15.0);

        let mut reverb_params = params.reverb;
        reverb_params.wet = (reverb_params.wet + mods.reverb_dry_wet).clamp(0.0, 1.0);
        reverb_params.decay_time *= mods.reverb_decay_time.exp2();
        reverb_params.size = (reverb_params.size + mods.reverb_size).clamp(0.0, 1.0);

        let mut filter_fx_params = params.filter_fx;
        filter_fx_params.state.midi_cutoff += PolyF32::splat(mods.filter_fx_cutoff);
        filter_fx_params.state.resonance_percent = (filter_fx_params.state.resonance_percent
            + mods.filter_fx_resonance)
            .clamp(0.0, 1.0);
        filter_fx_params
            .state
            .set_pass_blend(filter_fx_params.state.pass_blend + mods.filter_fx_blend);

        let distortion_drive_db =
            (params.distortion_drive_db + mods.distortion_drive_db).clamp(-30.0, 30.0);
        let distortion_mix = (params.distortion_mix + mods.distortion_mix).clamp(0.0, 1.0);

        ResolvedEffectsParams {
            chorus: chorus_params,
            compressor: compressor_params,
            delay: delay_params,
            distortion_type: params.distortion_type,
            distortion_drive_db,
            distortion_mix,
            eq: eq_params,
            filter_fx: filter_fx_params,
            flanger: flanger_params,
            phaser: phaser_params,
            reverb: reverb_params,
        }
    }

    /// Resolves the delay tempo sync into per-lane periods: the main line
    /// feeds the left lanes and the aux line the right lanes for the stereo
    /// styles, matching `Delay::processWithInput`'s `kFrequencyAux` load.
    fn resolve_delay_params(&self, beats_per_second: f32, mods: &EffectsModOffsets) -> DelayParams {
        // A tiny floor keeps `Freeze` (ratio 0) finite; the delay clamps the
        // resulting period to its memory size, like the reference clamp.
        // Frequency modulation offsets are in log2 Hz (the stored domain).
        const MIN_HZ: f32 = 1.0e-4;
        let sr = self.engine_rate;
        let mut params = self.params.delay;
        let main_hz =
            self.params.delay_sync.frequency_hz(beats_per_second) * mods.delay_frequency.exp2();
        let main_period = sr / main_hz.max(MIN_HZ);
        let uses_aux = matches!(
            params.style,
            DelayStyle::Stereo | DelayStyle::PingPong | DelayStyle::MidPingPong
        );
        params.period_samples = if uses_aux {
            let aux_hz = self.params.delay_aux_sync.frequency_hz(beats_per_second)
                * mods.delay_aux_frequency.exp2();
            let aux_period = sr / aux_hz.max(MIN_HZ);
            PolyF32::stereo(main_period, aux_period)
        } else {
            PolyF32::splat(main_period)
        };
        params
    }

    /// Processes one block in place, running every enabled effect in the
    /// decoded order, each wrapped in its slot's signal split.
    pub fn process(&mut self, resolved: &ResolvedEffectsParams, buffer: &mut [PolyF32]) {
        let num_samples = buffer.len();
        debug_assert!(num_samples <= self.out_scratch.len());
        if num_samples == 0 {
            return;
        }

        self.update_effect_switches(&resolved.phaser);

        for effect in self.params.order {
            if !self.params.is_on(effect) {
                continue;
            }
            let slot = effect as usize;
            let split = self.params.split[slot];

            let mut out = std::mem::take(&mut self.out_scratch);
            let mut split_a = std::mem::take(&mut self.split_a);
            let mut split_b = std::mem::take(&mut self.split_b);

            match split.mode {
                SplitMode::Full => {
                    self.process_one(effect, resolved, buffer, &mut out[..num_samples]);
                    buffer.copy_from_slice(&out[..num_samples]);
                }
                SplitMode::Mid | SplitMode::Side => {
                    // Mid lives in the L lanes and side in the R lanes of the
                    // encoded signal; the effect sees the selected component
                    // with the other zeroed, and the untouched component is
                    // recombined before decoding back to L/R.
                    let (process_mask, keep_mask) = if split.mode == SplitMode::Mid {
                        (PolyF32::stereo(1.0, 0.0), PolyF32::stereo(0.0, 1.0))
                    } else {
                        (PolyF32::stereo(0.0, 1.0), PolyF32::stereo(1.0, 0.0))
                    };
                    for i in 0..num_samples {
                        let encoded = encode_mid_side(buffer[i]);
                        split_a[i] = encoded;
                        split_b[i] = encoded * process_mask;
                    }
                    self.process_one(
                        effect,
                        resolved,
                        &split_b[..num_samples],
                        &mut out[..num_samples],
                    );
                    for i in 0..num_samples {
                        buffer[i] =
                            decode_mid_side(out[i] * process_mask + split_a[i] * keep_mask);
                    }
                }
                SplitMode::Low | SplitMode::High => {
                    self.update_split_crossover(slot, split.crossover_hz);
                    self.split_filters[slot].process(
                        buffer,
                        &mut split_a[..num_samples],
                        &mut split_b[..num_samples],
                    );
                    let (selected, kept) = if split.mode == SplitMode::Low {
                        (&split_a, &split_b)
                    } else {
                        (&split_b, &split_a)
                    };
                    self.process_one(
                        effect,
                        resolved,
                        &selected[..num_samples],
                        &mut out[..num_samples],
                    );
                    for (dest, (&wet, &dry)) in buffer
                        .iter_mut()
                        .zip(out[..num_samples].iter().zip(&kept[..num_samples]))
                    {
                        *dest = wet + dry;
                    }
                }
            }

            self.out_scratch = out;
            self.split_a = split_a;
            self.split_b = split_b;
        }
    }

    /// Runs one effect instance `input -> output` with the resolved params.
    fn process_one(
        &mut self,
        effect: Effect,
        resolved: &ResolvedEffectsParams,
        input: &[PolyF32],
        output: &mut [PolyF32],
    ) {
        let num_samples = input.len();
        debug_assert_eq!(num_samples, output.len());
        match effect {
            Effect::Chorus => {
                self.chorus.process(&resolved.chorus, input, output);
            }
            Effect::Compressor => {
                self.compressor.process(&resolved.compressor, input, output);
            }
            Effect::Delay => {
                self.delay.process(&resolved.delay, input, output);
            }
            Effect::Distortion => {
                // TODO(fidelity): DistortionModule's optional pre/post
                // filter (distortion_filter_order) is not ported yet.
                output.copy_from_slice(input);
                self.drive_scratch[..num_samples]
                    .fill(PolyF32::splat(resolved.distortion_drive_db));
                self.distortion.process(
                    resolved.distortion_type,
                    &self.drive_scratch[..num_samples],
                    output,
                );

                // Dry/wet ramp exactly like DistortionModule::processWithInput.
                let mut current_mix = self.distortion_mix;
                self.distortion_mix = PolyF32::splat(resolved.distortion_mix);
                let delta_mix = (self.distortion_mix - current_mix) * (1.0 / num_samples as f32);
                for (wet, &dry) in output.iter_mut().zip(input) {
                    current_mix += delta_mix;
                    *wet = interpolate(dry, *wet, current_mix);
                }
            }
            Effect::Eq => {
                self.equalizer.process(&resolved.eq, input, output);
            }
            Effect::FilterFx => {
                let mut params = resolved.filter_fx;
                params.on = true; // gated by `filter_fx_on` instead
                self.filter_fx.process(&params, input, output, PolyMask::NONE);
            }
            Effect::Flanger => {
                self.flanger.process(&resolved.flanger, input, output);
            }
            Effect::Phaser => {
                self.phaser.process(&resolved.phaser, input, output);
            }
            Effect::Reverb => {
                self.reverb.process(&resolved.reverb, input, output);
            }
        }
    }

    /// Recreates a slot's crossover filter when its cutoff moved materially
    /// (the filter has no cutoff setter; recreation is allocation-free).
    fn update_split_crossover(&mut self, slot: usize, crossover_hz: f32) {
        let hz = crossover_hz.clamp(20.0, 20_000.0).min(self.engine_rate * 0.49);
        let stored = self.split_crossover_hz[slot];
        if (hz - stored).abs() > stored * 1.0e-3 {
            self.split_filters[slot] = LinkwitzRileyFilter::new(hz, self.engine_rate);
            self.split_crossover_hz[slot] = hz;
        }
    }

    /// Mirrors each effect module's `enable` override: some reset when
    /// switched on, some when switched off (`ReorderableEffectChain`'s
    /// on/enabled bookkeeping).
    fn update_effect_switches(&mut self, phaser_params: &PhaserParams) {
        for index in 0..NUM_EFFECTS {
            let effect = Effect::from_index(index);
            let on = self.params.is_on(effect);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_RATE: f32 = 88200.0;
    const BLOCK: usize = 128;

    /// Renders `buffer` through `chain` in `BLOCK`-sample chunks, in place.
    fn run_chain(chain: &mut EffectChain, buffer: &mut [PolyF32]) {
        let resolved = chain.resolve(2.0, &EffectsModOffsets::default());
        for chunk in buffer.chunks_mut(BLOCK) {
            chain.process(&resolved, chunk);
        }
    }

    /// `amp_low * sin(f_low) + amp_high * sin(f_high)`, mono (all lanes equal).
    fn two_tone(f_low: f32, f_high: f32, num_samples: usize) -> Vec<PolyF32> {
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE;
                let value = 0.5 * (2.0 * std::f32::consts::PI * f_low * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * f_high * t).sin();
                PolyF32::splat(value)
            })
            .collect()
    }

    /// Splits `signal` at `crossover` and returns (low_rms, high_rms) of
    /// lane 0, skipping the first half (filter transients).
    fn band_rms(signal: &[PolyF32], crossover: f32) -> (f32, f32) {
        let mut filter = LinkwitzRileyFilter::new(crossover, SAMPLE_RATE);
        let mut low = vec![PolyF32::ZERO; signal.len()];
        let mut high = vec![PolyF32::ZERO; signal.len()];
        filter.process(signal, &mut low, &mut high);
        let start = signal.len() / 2;
        let rms = |buffer: &[PolyF32]| {
            let sum: f32 = buffer[start..].iter().map(|v| v.lane(0) * v.lane(0)).sum();
            (sum / (buffer.len() - start) as f32).sqrt()
        };
        (rms(&low), rms(&high))
    }

    fn distortion_chain(on: bool, split: EffectSplit) -> EffectChain {
        let mut chain = EffectChain::new(SAMPLE_RATE, BLOCK);
        let params = chain.params_mut();
        params.distortion_on = on;
        params.distortion_drive_db = 12.0;
        params.split[Effect::Distortion as usize] = split;
        chain
    }

    #[test]
    fn low_split_distortion_distorts_only_below_crossover() {
        const CROSSOVER: f32 = 2000.0;
        let num_samples = 8192;
        let render = |on: bool, split: EffectSplit| {
            let mut chain = distortion_chain(on, split);
            let mut buffer = two_tone(100.0, 8000.0, num_samples);
            run_chain(&mut chain, &mut buffer);
            buffer
        };

        let clean = render(false, EffectSplit::default());
        let full = render(true, EffectSplit::default());
        let low_split =
            render(true, EffectSplit { mode: SplitMode::Low, crossover_hz: CROSSOVER });

        let (clean_low, clean_high) = band_rms(&clean, CROSSOVER);
        let (full_low, full_high) = band_rms(&full, CROSSOVER);
        let (split_low, split_high) = band_rms(&low_split, CROSSOVER);

        // The low band is distorted in both renders.
        assert!(
            (split_low - clean_low).abs() > 0.2 * clean_low,
            "low-split left the low band clean: {split_low} vs {clean_low}"
        );
        assert!((full_low - clean_low).abs() > 0.2 * clean_low);
        // Full-split distortion mangles the high band too...
        assert!(
            (full_high - clean_high).abs() > 0.2 * clean_high,
            "full-split left the high band clean: {full_high} vs {clean_high}"
        );
        // ...but the low split leaves it untouched (small tolerance for the
        // distorted low band's harmonics leaking over the crossover).
        assert!(
            (split_high - clean_high).abs() < 0.1 * clean_high,
            "low-split distorted the high band: {split_high} vs {clean_high}"
        );
    }

    #[test]
    fn side_split_on_mono_signal_is_noop() {
        let num_samples = 1024;
        let mut chain = distortion_chain(
            true,
            EffectSplit { mode: SplitMode::Side, crossover_hz: DEFAULT_SPLIT_CROSSOVER_HZ },
        );
        chain.params_mut().distortion_drive_db = 30.0;
        let input = two_tone(100.0, 8000.0, num_samples);
        let mut buffer = input.clone();
        run_chain(&mut chain, &mut buffer);
        for (i, (out, expected)) in buffer.iter().zip(&input).enumerate() {
            for lane in 0..4 {
                assert!(
                    (out.lane(lane) - expected.lane(lane)).abs() < 1e-5,
                    "sample {i} lane {lane}: {} vs {}",
                    out.lane(lane),
                    expected.lane(lane)
                );
            }
        }
    }

    #[test]
    fn mid_split_distortion_changes_mono_signal() {
        // Complement of the side no-op: a mono signal is all mid, so the mid
        // split must distort it.
        let num_samples = 2048;
        let mut chain = distortion_chain(
            true,
            EffectSplit { mode: SplitMode::Mid, crossover_hz: DEFAULT_SPLIT_CROSSOVER_HZ },
        );
        chain.params_mut().distortion_drive_db = 30.0;
        let input = two_tone(100.0, 8000.0, num_samples);
        let mut buffer = input.clone();
        run_chain(&mut chain, &mut buffer);
        let diff: f32 = buffer
            .iter()
            .zip(&input)
            .map(|(a, b)| (a.lane(0) - b.lane(0)).abs())
            .sum::<f32>()
            / num_samples as f32;
        assert!(diff > 1e-3, "mid split did not process a mono signal: diff {diff}");
    }

    #[test]
    fn full_split_matches_pre_split_processing() {
        // Regression for the chain extraction: with the default (Full)
        // split, the wrapped runner must behave exactly like a plain
        // input -> effect -> output pass (same build, split machinery
        // bypassed). Here: distortion output must be deterministic and
        // identical whether the split array is left defaulted or set to
        // Full explicitly.
        let num_samples = 2048;
        let render = |split: EffectSplit| {
            let mut chain = distortion_chain(true, split);
            let mut buffer = two_tone(100.0, 8000.0, num_samples);
            run_chain(&mut chain, &mut buffer);
            buffer
        };
        let defaulted = render(EffectSplit::default());
        let explicit = render(EffectSplit { mode: SplitMode::Full, crossover_hz: 123.0 });
        for (a, b) in defaulted.iter().zip(&explicit) {
            for lane in 0..4 {
                assert_eq!(a.lane(lane), b.lane(lane));
            }
        }
    }
}
