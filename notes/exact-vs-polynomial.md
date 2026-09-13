# Exact vs polynomial: every call site of the reference, and what Spinwave calls (2026-09-13)

**The rule.** Vital is not self-consistent: `utils::` is exact (`powf`,
`sinf`, `log2f` on scalars, mapped lane by lane) and `futils::` is
polynomial (`exp2` a 5th-order polynomial on the mantissa, `log2` the
same on the bits, `pow` = `exp2(log2(b) · x)`, `sin` a parabola with a
correction, `tanh`/`quickTanh` rationals). Which one a site uses is
not a convention, it is a choice made call by call — the same file
calls both in adjacent lines (`digital_svf.cpp:81` exact base
frequency, `:85` polynomial offset). So no rule of thumb ("controls
exact, per-sample polynomial") is trusted; **each site is checked
individually** against the reference's line, and the port calls what
the reference calls. Where Spinwave's own helper names encode this:
`filter_state::*_precise` are the exact `utils::` maps, `math::*`
(spinwave-poly) the polynomial `futils::` ones.

**Why it matters at the level the bench measures.** The polynomial
`exp2` has a relative error of ~1e-6; the exact one ~6e-8. A base
frequency through the wrong one was the bench's floor for months
(every oscillator case at 4e-4 RMS, read as "phase-accumulator
timing"); the unison ratio, the delay's filter frequencies and one bit
of operation order in the exact conversion (`(n·100)/1200` vs
`n·(1/12)`) were each a tracked case. Same function, wrong twin =
a residual between 1e-4 and 1e-3, every time so far.

Method: the reference's `synthesis/` and `common/` grepped for every
`utils::` / `futils::` call and every libm call outside the framework
headers (204 lines, 128 in the DSP), each mapped to the Spinwave line
that computes the same thing. Framework definitions, the preset
migration (`load_save.cpp`, ported in `spinwave-params/migrate.rs` with
the exact `powf`/`sqrt` it uses) and the wavetable *editor* sources
(`common/wavetable/`, offline construction, not the DSP) are listed at
the end, not row by row.

Status column: **same** = same twin, same call order; **fixed** =
was the other twin, corrected in this pass; **Spinwave-only** = no
reference site. Residual = the golden case that exercises the site, its
RMS after this pass.

## Oscillator (`producers/synth_oscillator.cpp`)

| Ref line | Reference call | Spinwave | Status | Case, residual |
|---|---|---|---|---|
| 1315 | `utils::midiNoteToFrequency(base_midi)` (exact) | `synth_oscillator.rs:1236` `midi_note_to_frequency_precise` | same (fixed 2026-09-11, op order 2026-09-12) | osc_saw_dry 4.4e-8 |
| 1331 | `futils::midiOffsetToRatio(midi - base_midi)` per sample | `:1248` `math::midi_offset_to_ratio` | same | osc_saw_dry, all mod_env_to_pitch* ≤ 7.6e-8 |
| 627 | `utils::centsToRatio(oscillator_cents)` unison (exact) | `:1154` `cents_to_ratio` exact `exp2(c/1200)` | same (fixed 2026-09-12) | osc_unison 5e-8 |
| 624 | `futils::powerScale(t, power)` unison spread | `:1151` `math::power_scale` | same | osc_unison |
| 643 | `futils::exp2(-spectral_morph_values_)` | `:1484` `math::exp2` | same | osc_morph_* |
| 790 | `futils::exp2(-bin_shift)` → `last_harmonic` | `:251` `math::exp2` | **fixed** (was exact `f32::exp2`; a truncated count can land one off at a bin edge) | no case moved (see below) |
| 494, 497 | `futils::pow(2, (v-0.5)·2·exponent)` phase distortion | `phase.rs:279, 282` `math::pow` | same | osc_warp_* |
| 1044, 1051 | `futils::pow(2, distortion·kDistortBits+1)` quantize | `phase.rs:313, 319` `math::pow` | same | osc_warp_quantize 5e-8 |
| 151 | `futils::sin(phase + 0.25)` half-sine window | `phase.rs:221` `math::sin` | same | osc_warp_formant 5e-8 |
| `lookups/wavetable.h:64` | `futils::log2(1/phase_inc)` frequency float bin | `wavetable.rs:151` `math::log2` | **fixed** (was exact) | no case moved |
| `wavetable.h:69` | `ilog2(int)` | `wavetable.rs:158` integer | same | — |

## Spectral morph (`producers/spectral_morph.h`)

| Ref line | Reference call | Spinwave | Status | Case, residual |
|---|---|---|---|---|
| 115, 116 | `futils::sin(mod(phase+0.75)-0.5)` shepard | `spectral_morph.rs:309, 310` `math::sin` | same | osc_morph_shepard 6.2e-6 |
| 152 | `futils::log2(i) / kFrequencyBins` skew | `:346` `math::log2` | **fixed** (was exact) | osc_morph_spectral_time_skew < 1e-6 either way |
| 250, 280 | `futils::pow(2, (bins-1)·cutoff_t)` low / high pass | `:440, 469` `poly_pow2` (= `math::pow(2, x)`) | **fixed** (was exact `exp2`) | osc_morph_low_pass, high_pass < 1e-6 either way |
| 397, 399 | `futils::log2(index)`, `futils::pow(mult, power)` inharmonic | `:589, 591` `math::log2`, `math::pow` | same | osc_morph_inharmonic_stretch 1.7e-6 |

The three "fixed" sites changed no case's residual to three digits:
their values fall where the polynomial and the exact agree after
truncation. They are fixed because the rule is call for call, not
because a case asked.

## Filters

Every filter takes its base frequency with the exact
`utils::midiNoteToFrequency` and its per-sample offset with the
polynomial `futils::midiOffsetToRatio` — the pattern that was wrong in
the oscillator and is right in every filter since the first port.

| Ref line | Reference call | Spinwave | Status | Case, residual |
|---|---|---|---|---|
| `digital_svf.cpp:81,111,141,172,205` | exact base · `(1/rate)` | `digital_svf.rs:370,411,453,497` `_precise` | same | filter_digital_* 5e-8 |
| `:85,115,145,176,209` | `futils::midiOffsetToRatio(delta)`, `min(…, 1)` | `:376,417,459,503` `math::midi_offset_to_ratio` | same | filter_digital_* |
| `:226` | `utils::midiNoteToFrequency(midi_cutoff)` (exact) | `:159` `_precise` | same | fx_eq |
| `:231` | `utils::dbToMagnitude(gain)` (exact) | `:164` `db_to_magnitude_precise` | same | fx_eq 5e-8 |
| `:284` | `futils::pow(amplitude_quartic, blend)` | `:223` `math::pow` | same | filter_digital_* |
| `sallen_key_filter.cpp:103,155,208` / `:109,161,214` | exact base / poly offset | `sallen_key.rs:209` / `:216` | same | filter_analog_* 7e-8 |
| `:244` | exact `midiNoteToFrequency(midi_cutoff)` | `:104` `_precise` | same | filter_analog_* |
| `:326` | `futils::tanh(...)` | `:362` `math::tanh` | same | filter_analog_high_q |
| `ladder_filter.cpp:79` / `:84` | exact base / poly offset, `min(…, max_frequency)` | `ladder.rs:239` / `:246` | same | filter_ladder_* |
| `:202` | `futils::tanh(filter_input)` | `:289` `math::tanh` | same | filter_ladder_* |
| `dirty_filter.cpp:115,169,228` / `:122,176,235` | exact base / poly offset | `dirty.rs:218` / `:228` | same | filter_dirty_* |
| `:268` | exact `midiNoteToFrequency(cutoff)` (one-pole setup) | `:122` `_precise` | same | filter_dirty_* |
| `:372` | `futils::tanh(mulAdd(...))` | `:435` `math::tanh` | same | filter_dirty_* |
| `dirty_filter.h:120,121` | `OnePoleFilter<futils::quickTanh>` stages 3-4 | `one_pole.rs:52` `math::quick_tanh` | same | filter_dirty_* |
| `diode_filter.cpp:77` / `:83` | exact base / poly offset | `diode.rs:161` / `:170` | same | filter_diode_* (tracked 2.3e-3 / 1.9e-2, not this) |
| `:112, 116` | `futils::exp2(high-pass blend)` | `:107, 111` `math::exp2` | same | filter_diode_* |
| `:145`, `diode_filter.h:64` | `futils::tanh` | `:231`, `one_pole.rs:42` `math::tanh` | same | filter_diode_* |
| `comb_filter.cpp:170,171,191,192,231,264` | exact `midiNoteToFrequency` | `comb.rs:209,210,231,232,264,309` `_precise` | same | filter_comb_* 6e-8 |
| `:175,176` | exact `frequencyToMidiNote` | `:214,215` `frequency_to_midi_note_precise` | same | filter_comb_* |
| `:271` | `futils::midiOffsetToRatio(midi_offset)` | `:322` `math::midi_offset_to_ratio` | same | filter_comb_* |
| `phaser_filter.h:89` / `:93` | exact base / poly offset | `phaser_filter.rs:183` / `:188` | same | filter_phaser_*, fx_phaser 6e-8 |
| `phaser_filter.cpp:54` | `process<futils::tanh>` | `:142` `math::tanh` | same | fx_phaser |
| `phaser_filter` coefficient | `OneDimLookup` cubic table | `coefficient_lookup().cubic_lookup` | same (fixed 2026-09-12) | fx_phaser 1.8e-3 → 6e-8 |
| `synth_filter.cpp:43` | `futils::dbToMagnitude(input_drive)` | `filter_state.rs:122` `math::db_to_magnitude` | same | every filter case |
| `formant` | table of cutoffs (no conversion at run time) | `formant.rs:380` exact `exp2` in a **test** only | — | filter_formant_* 5e-8 |

## Effects

| Ref line | Reference call | Spinwave | Status | Case, residual |
|---|---|---|---|---|
| `delay.cpp:97,102,109` | exact `midiNoteToFrequency` (filter, damping) | `delay.rs:270,274,284` `_precise` | same (fixed 2026-09-12) | fx_delay tracked 1.4e-2, unchanged by it |
| `delay.cpp:75` | `futils::exp_half(n / (half_life·rate))` = poly `exp2(-x)` | `delay.rs:241` `math::exp2(-x)` | same | fx_delay |
| `reverb.cpp:119,123,129,133` | exact `midiNoteToFrequency` | `reverb.rs:391-405` `_precise` | same | fx_reverb tracked 6.4e-3 |
| `:139,141` | exact `utils::dbToMagnitude` | `:411,413` `db_to_magnitude_precise` | same | fx_reverb |
| `:160` | `futils::pow(2, size…)` | `:417` `math::pow` | same | fx_reverb |
| `:168-171` | exact `utils::pow(kT60Amplitude, …)` | `:430` `pow_exact` (`powf`) | same | fx_reverb |
| `compressor.cpp:69,70` | `futils::exp(exponent)` | `compressor.rs:154,156` `math::exp` | same | fx_compressor 9.6e-6 |
| `:78,81,128` | `futils::dbToMagnitude` | `:161,165,217` `math::db_to_magnitude` | same | fx_compressor |
| `:102,112` | `futils::pow(delta, ratio)` | `:189,201` `math::pow` | same | fx_compressor |
| `distortion.cpp:39` | `futils::tanh(value·drive)` | `distortion.rs:46` `math::tanh` | same | fx_distortion 5e-8 |
| `distortion.h:55` | `futils::dbToMagnitude(clamp(db))` | `:62` `math::db_to_magnitude` | same | fx_distortion |
| `flanger_module.cpp:71` | `1 / utils::midiNoteToFrequency(center)` (exact) | `flanger.rs:115` `_precise` | same | fx_flanger 5e-8 |
| `chorus_module.cpp:116` | `utils::sin(phase·2π)` (**exact** `sinf`) | `chorus.rs:179` `f32::sin` | same | fx_chorus tracked 1.08e-4 — not this |
| `:133,134` | `futils::equalPowerFade(Inverse)` | `:205,206` `math::equal_power_fade*` | same | fx_chorus |
| `peak_meter.cpp:64` | `utils::sqrt` | sqrt | same | — |

## Modulators, voice, control-rate operators

| Ref line | Reference call | Spinwave | Status | Case, residual |
|---|---|---|---|---|
| `envelope.cpp:86,131` | `futils::powerScale` | `envelope.rs:163,219` `math::power_scale` | same | env_fast 3e-9, env_slow 3e-8 |
| `synth_lfo.cpp:124,162,211,266,318,367` | `futils::exp2(exponent)` smoothing | `synth_lfo.rs:450,510,848,973` `math::exp2` | same | mod_lfo_* ≤ 5e-7 |
| `modulation_connection_processor.cpp:146,201,279` | `futils::powerScale(abs, power)` | `modulation.rs:203,302` `power_scale` | same | meta_power 3e-7 |
| `portamento_slope.cpp:62` | `futils::powerScale` | `portamento.rs:85` | same | no case (portamento) |
| `smooth_value.cpp:34,71` | `futils::exp(-2π·cutoff/rate)` | `smooth_value.rs:59,130` `math::exp` | same | every case (master volume) |
| `operators.cpp:132` | `SmoothVolume`: `futils::dbToMagnitude(db)` | `engine.rs:1243,1293` `math::db_to_magnitude` | same | every case |
| `operators.cpp:341` | `cr::TempoChooser` keytrack: exact `midiNoteToFrequency` | keytracked LFO rate not read (read audit) | **no site** | no case; `lfo_N_keytrack_transpose` is in the destinations table |
| `operators.h:618` | `cr::ExponentialScale`: `futils::pow(2, x)` = poly `exp2(log2(2)·x)`, the poly `log2(2)` being 1 to a few ulp | `tempo::exponential_scale` = `math::pow(2, clamp(x))` on every Exponential control, base and offsets summed first | **fixed** — was `math::exp2(x)`, "the same to 1e-7", and those ulp on a chorus delay of 2^-9 s were fx_chorus's 1.1e-4 (→ 3.7e-6) | fx_chorus 3.7e-6; `mono_macro_to_chorus_delay_1` 2.3e-6, `poly_macro_to_lfo_1_frequency` 5.0e-8 |
| — | the same scale on a *modulated* control: `pow(2, clamp(base + offset))` once | the same call, offsets on the stored value (`effect_chain.rs` resolve, `synth_voice.rs` LFO / portamento) | **fixed** — was `hz(base) · offset.exp2()`, a product of two, and the LFO rate added an offset in Hz | `poly_macro_to_lfo_1_frequency`, `mono_macro_to_chorus_frequency` 2.0e-6, `reverb_decay_time` 3.7e-8 |
| `operators.h:710` | `cr::MagnitudeScale` `futils::dbToMagnitude` | — (no such control is read through it; volume goes through SmoothVolume) | — | — |
| `operators.h:726` | `cr::MidiScale` exact `midiCentsToFrequency` | — (unused by any module the reference builds) | — | — |

## Sample source (`producers/sample_source.cpp`)

| Ref line | Reference call | Spinwave | Status | Case, residual |
|---|---|---|---|---|
| 383 | `utils::centsToRatio(transpose · 100)` (**exact**) | `sample_source.rs` `cents_to_ratio_exact` | **fixed** (was `math::midi_offset_to_ratio`, polynomial) | **no case**: the reference harness loads no sample |
| 520 | `utils::noteOffsetToRatio(snapped − last)` (exact) | `note_offset_to_ratio_exact` | **fixed** (was polynomial) | no case |
| `sample_source.h:68` | `ilog2(delta)` | integer | same | — |

Spinwave-only on this path: `params.rate` (tape-style speed) multiplies
the exact ratio; the reference has no such control.

## Sites with no case

- The sample source's two ratios (above). A case needs the reference
  harness to load a sample; `main.cpp` loads wavetables only.
- Portamento's `powerScale` (no glide case).
- The keytracked LFO rate (destination not read; table in the
  destinations note).
- The modulated exponential effect controls (frequency/time of chorus,
  flanger, phaser, delay, reverb): factorised differently, ≤ 1e-6
  relative expected; a case goes with the destinations work.

## Where the maximum residual sits, and the threshold

After this pass (112 cases) the corpus was: **97 cases at or below 1e-6**
(float noise), **5 between 1e-6 and 1e-5**, **nothing between 1e-5 and
1e-4**, 10 tracked above 1e-4. The destinations pass that followed
(329 cases, notes/destinations.md) delisted fx_chorus, fx_delay and
fx_reverb and added eleven cases in (1e-5, 1e-4] — every compressor
case at ~1e-5, the chorus at two settings (1.6e-5, 3.2e-5) and
`macro_dest_lfo` (4.5e-5) — all located, none an exact/polynomial twin.
The five of the first pass:

| case | RMS | where it sits (measured) |
|---|---|---|
| fx_compressor | 9.6e-6 | steady over the note, at 30-40 Hz (the envelope follower), −42 dB relative; every math call is the same twin — the follower's state, not a conversion |
| osc_morph_shepard | 6.2e-6 | **only the first 100 ms after note-on** (2e-5 there, 5e-8 after): an onset event, not the morph's steady spectrum |
| osc_morph_harmonic_stretch | 2.3e-6 | steady, a single component near 14.9 kHz: one harmonic near the band edge |
| mod_random_to_cutoff | 2.1e-6 | steady around the cutoff; the random LFO's value path (Perlin, same twin) |
| osc_morph_inharmonic_stretch | 1.7e-6 | steady, near Nyquist at −105 dB relative: the top harmonics |

None of the five is an exact/polynomial twin (all their sites are
`same` above). **The chorus was diagnosed and fixed the same day**, and
it was the same family of error one step removed: the reference's
`Delay` takes a *frequency* (`kFrequency`), smooths it, and divides the
sample rate by it; Spinwave's took a *period* the caller had computed as
`sample_rate / frequency`, then divided the sample rate by that — the
same quantity through two divisions, off by ulps, and the fractional
delay read picked the neighbouring sample pair on rare instants (the
spikes: 5935 samples above 1e-6, peak 5e-3, RMS 3.2e-5). `DelayParams`
now carries the frequency: fx_chorus 3.7e-6 → 8.9e-8, every chorus
case at noise, the delay and flanger unchanged. The rule generalises:
not just which function, but **which association of operations** — the
reference's, call for call.

What remains above 1e-5 is the compressor family (~1e-5, steady at the
envelope follower's frequency, every math call the same twin) and
`macro_dest_lfo` (4.5e-5, the value held across the silence). The
threshold (RMS 1e-4) cannot drop to 1e-5 until the compressor is
diagnosed: it would sit against it.
