# The bank's modulation destinations: route, regime, case, residual (2026-09-13)

The standard is the real bank: 75 `.vital` presets under
`~/Documents/Vital` (Factory and Afro, versions 0.6.1 to 1.0.0), 1616
connections into 110 destination families. `spinwave-cli bank <dir>`
loads them all and prints what the engine could not route. Before this
pass: 62 of 75 refused, 356 connections into 48 unrouted families. After
meta-modulation: 56 refused, 177 into 46. After this pass: **75 of 75
load with zero ignored connection.**

Every family below has a golden case unless the last column says why
not. The method is the same for every one added in this pass: a **macro
(a constant) into the destination, and a static twin** set to the value
the macro should produce — `stored + macro × amount × range`, the offset
on the STORED value, before the square of a Quadratic control, the
fourth power of a Quartic one, the `pow(2, ·)` of an Exponential one.
Two references that are byte-identical say the route and the scale are
right with no reading of the code; Spinwave is then judged against both.
The regime (per block, per sample) is the mechanism the earlier cases
prove (`mod_lfo_to_*`); a macro cannot show it. Every value is interior
to its range (the bounds check runs on every render).

What the pass found beyond the routing, each fixed and measured:

- **The tempo index rounds.** The reference's `TempoChooser` does
  `toInt(index + 0.3)` and `toInt` is `_mm_cvtps_epi32`, round to
  nearest; Spinwave truncated. Integer indices never showed it; a
  modulated 6.5 did (`mono_macro_to_chorus_tempo` 3.8e-1 → 9.6e-7).
- **`ExponentialScale` is `pow(2, x)`, not `exp2(x)`.** The polynomial
  `pow` is `exp2(log2(2) · x)` with the polynomial `log2(2)`, one to a
  few ulp; on a chorus delay of 2^-9 s those ulp moved fx_chorus from
  1.1e-4 to 3.7e-6. Every Exponential control now goes through
  `tempo::exponential_scale`, base and offsets summed before it.
- **Envelope times are Quartic**: the offset adds to the stored root
  and the sum is raised to the fourth (`cr::Quart`); Spinwave added the
  offset (scaled by the stored range) to the seconds
  (`poly_macro_to_env_2_attack` 6.9e-2 → 4.8e-8).
- **LFO rates are Exponential**: same class, an offset in Hz added to a
  Hz value; 21 connections in the bank did that wrong silently
  (`poly_macro_to_lfo_1_frequency` now 5.0e-8).
- **`unison_voices` rounds** (`roundf`), the reader truncated: 3.5
  voices is 4 (`poly_macro_to_osc_1_unison_voices_static` 1.2e-1 →
  4.4e-8).
- **The primer's echo.** fx_delay (1.4e-2) and fx_reverb (6.4e-3) were
  never the effects: the primer note differs between the engines by
  construction and `skip` hides it, but not its echo through a delay
  line or a reverb tail. Cases with either skip 5 s: 2.2e-7 and 4.0e-7.
- **A macro as a destination** reaches the connections reading it two
  blocks after the source moves (`macro_dest_step` against its static
  twins: 2.8e-3 with one block of lag, 3.3e-7 with two): the reference's
  mono chain runs before the voices on the previous block's voice
  outputs, and a connection from a macro is itself a mono processor
  reading the macro's sum of the block before. `macro_dest_lfo` sits at
  4.5e-5: its first two blocks after the note carry the value held
  across the silence (the reference and Spinwave kill the voice at
  different instants, so hold a different LFO value — measured 7.7e-4
  and 2.8e-4 on those blocks, ~1e-5 after; the steady part is not
  diagnosed).
- **The keytracked LFO rate** (sync mode 4) was not implemented: the
  frequency of `bent midi + keytrack_transpose + keytrack_tune`
  through the exact conversion, for LFOs and random LFOs, with the
  transpose as a destination (`mod_lfo_keytrack` 4.8e-8).
- A modulated envelope **delay or hold** is 0 in the trigger block in
  the reference (the twins differ by 1.4e-2 / 1.7e-3 on the reference
  side, from exactly the delay time on); Spinwave does the same (both
  the case and the twin at 4.7e-8), so it is recorded, not chased.

Not covered, and why:

- `osc_N_wave_frame` (84 connections): the bench loads a single-cycle
  table, so a frame offset is inert on both sides. The route is the
  same `ModOffsets` path as the other oscillator controls; a case needs
  the harness to load a multi-frame table.
- `sample_level`, `sample_transpose`: the harness loads no sample.
  `sample_level` is audio-rate in the reference and control-rate here
  (notes/audio-rate-audit.md).
- `volume` (13 connections): the master volume as a destination — the
  reader routes it to the master path; no case yet.
- The five mono audio-rate destinations (`filter_fx_cutoff`,
  `distortion_drive`, `distortion_filter_cutoff`, `eq_*_cutoff`,
  `phaser_center`) and `filter_N_formant_x/y/transpose` are per block
  here; the tracked residuals (1e-3 to 6e-3) are the rate, and the
  plumbing to make them per sample is described in the audio-rate note.

Residuals are from the run of 2026-09-13 (331 cases: 301 at or below
1e-6, 16 in (1e-6, 1e-5], 7 in (1e-5, 1e-4] — the compressor cases at
~1e-5 and macro_dest_lfo 4.5e-5 — and 7 tracked above; the chorus
family fell to noise once the delay took a frequency instead of a
period, see the exact/polynomial note). `presets` counts the presets
carrying at least one such connection. The residual column below is
the run before that fix for the chorus rows (2e-6 to 3e-5 there; all
≤ 1e-7 after).

| destination | conn. | presets | route | regime | case | residual (RMS) |
|---|---|---|---|---|---|---|
| `modulation_N_amount` | 174 | 42 | poly (voice matrix) | per block | `meta_macro_to_amount` | 4.3e-08 |
| `filter_N_cutoff` | 134 | 55 | poly (voice matrix) | audio-rate in the reference; per sample | `mod_lfo_to_cutoff` | 4.1e-08 |
| `osc_N_level` | 128 | 52 | poly (voice matrix) | audio-rate in the reference; per sample | `mod_env_to_level` | 5.7e-08 |
| `osc_N_spectral_morph_amount` | 101 | 45 | poly (voice matrix) | per block | `poly_macro_to_osc_1_spectral_morph_amount` | 3.7e-08 |
| `osc_N_wave_frame` | 84 | 41 | poly (voice matrix) | per block | — | — |
| `osc_N_distortion_amount` | 70 | 36 | poly (voice matrix) | per block | `poly_macro_to_osc_1_distortion_amount` | 4.8e-08 |
| `osc_N_transpose` | 67 | 30 | poly (voice matrix) | audio-rate in the reference; per sample | `mod_env_to_pitch` | 7.4e-08 |
| `reverb_dry_wet` | 48 | 42 | mono (bus matrix) | per block | `mono_macro_to_reverb_dry_wet` | 4.1e-08 |
| `distortion_drive` | 33 | 24 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mono_macro_to_distortion_drive` | 6.6e-08 |
| `sample_level` | 33 | 23 | poly (voice matrix) | audio-rate in the reference; per sample | — | — |
| `filter_fx_cutoff` | 30 | 20 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mod_lfo_to_filter_fx_cutoff` | 2.2e-03 |
| `filter_N_blend` | 30 | 21 | poly (voice matrix) | per block | `poly_macro_to_filter_1_blend` | 6.3e-08 |
| `osc_N_tune` | 29 | 12 | poly (voice matrix) | audio-rate in the reference; per sample | `mod_env_to_tune` | 4.4e-08 |
| `chorus_dry_wet` | 28 | 26 | mono (bus matrix) | per block | `mono_macro_to_chorus_dry_wet` | 2.3e-06 |
| `delay_dry_wet` | 28 | 26 | mono (bus matrix) | per block | `mono_macro_to_delay_dry_wet` | 1.4e-07 |
| `filter_N_resonance` | 27 | 15 | poly (voice matrix) | per block | `poly_macro_to_filter_1_resonance` | 7.5e-08 |
| `lfo_N_tempo` | 26 | 15 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_tempo` | 5.2e-08 |
| `distortion_filter_cutoff` | 24 | 17 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mod_lfo_to_distortion_filter_cutoff` | 5.3e-03 |
| `env_N_attack` | 23 | 16 | poly (voice matrix) | per block | `poly_macro_to_env_2_attack` | 4.8e-08 |
| `distortion_mix` | 22 | 19 | mono (bus matrix) | per block | `mod_lfo_to_distortion_mix` | 4.7e-08 |
| `lfo_N_frequency` | 21 | 14 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_frequency` | 5.0e-08 |
| `filter_N_mix` | 20 | 12 | poly (voice matrix) | per block | `poly_macro_to_filter_1_mix` | 5.7e-08 |
| `env_N_sustain` | 17 | 12 | poly (voice matrix) | per block | `poly_macro_to_env_2_sustain` | 4.5e-08 |
| `osc_N_unison_detune` | 16 | 14 | poly (voice matrix) | per block | `poly_macro_to_osc_1_unison_detune` | 4.6e-08 |
| `filter_N_blend_transpose` | 15 | 10 | poly (voice matrix) | per block | `poly_macro_to_filter_1_blend_transpose` | 4.5e-08 |
| `reverb_decay_time` | 15 | 15 | mono (bus matrix) | per block | `mono_macro_to_reverb_decay_time` | 3.7e-08 |
| `osc_N_detune_range` | 15 | 6 | poly (voice matrix) | per block | `poly_macro_to_osc_1_detune_range` | 5.8e-08 |
| `filter_fx_mix` | 14 | 12 | mono (bus matrix) | per block | `mono_macro_to_filter_fx_mix` | 5.6e-08 |
| `env_N_decay` | 14 | 11 | poly (voice matrix) | per block | `poly_macro_to_env_2_decay` | 4.7e-08 |
| `env_N_release` | 13 | 9 | poly (voice matrix) | per block | `poly_macro_to_env_2_release` | 4.9e-08 |
| `volume` | 13 | 11 | master | per block | — | — |
| `phaser_center` | 12 | 8 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mod_lfo_to_phaser_center` | 6.0e-03 |
| `phaser_dry_wet` | 12 | 12 | mono (bus matrix) | per block | `mono_macro_to_phaser_dry_wet` | 1.1e-07 |
| `filter_N_drive` | 12 | 11 | poly (voice matrix) | per block | `poly_macro_to_filter_1_drive` | 5.7e-08 |
| `osc_N_pan` | 12 | 5 | poly (voice matrix) | per block | `poly_macro_to_osc_1_pan` | 4.5e-08 |
| `random_N_tempo` | 11 | 5 | poly (voice matrix) | per block | `poly_macro_to_random_1_tempo` | 5.9e-06 |
| `macro_control_N` | 10 | 6 | mono (bus matrix) | per block | `macro_dest_constant` | 4.8e-08 |
| `eq_band_cutoff` | 9 | 9 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mono_macro_to_eq_band_cutoff` | 5.9e-08 |
| `delay_feedback` | 9 | 8 | mono (bus matrix) | per block | `mono_macro_to_delay_feedback` | 3.9e-08 |
| `flanger_dry_wet` | 8 | 8 | mono (bus matrix) | per block | `mono_macro_to_flanger_dry_wet` | 5.5e-08 |
| `eq_band_gain` | 7 | 7 | mono (bus matrix) | per block | `mono_macro_to_eq_band_gain` | 5.9e-08 |
| `flanger_center` | 7 | 6 | mono (bus matrix) | per block | `mono_macro_to_flanger_center` | 4.7e-08 |
| `sample_transpose` | 7 | 6 | poly (voice matrix) | per block | — | — |
| `chorus_delay_N` | 7 | 3 | mono (bus matrix) | per block | `mono_macro_to_chorus_delay_1` | 2.3e-06 |
| `eq_low_cutoff` | 6 | 6 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mod_lfo_to_eq_low_cutoff` | 1.0e-03 |
| `filter_fx_blend` | 6 | 4 | mono (bus matrix) | per block | `mono_macro_to_filter_fx_blend` | 6.1e-08 |
| `osc_N_spectral_morph_spread` | 6 | 5 | poly (voice matrix) | per block | `poly_macro_to_osc_1_spectral_morph_spread` | 3.7e-08 |
| `osc_N_unison_blend` | 6 | 5 | poly (voice matrix) | per block | `poly_macro_to_osc_1_unison_blend` | 4.4e-08 |
| `osc_N_unison_voices` | 5 | 5 | poly (voice matrix) | per block | `poly_macro_to_osc_1_unison_voices` | 4.4e-08 |
| `modulation_N_power` | 5 | 3 | poly (voice matrix) | per block | `meta_power` | 4.2e-08 |
| `compressor_high_gain` | 5 | 5 | mono (bus matrix) | per block | `mono_macro_to_compressor_high_gain` | 9.6e-06 |
| `chorus_mod_depth` | 5 | 4 | mono (bus matrix) | per block | `mono_macro_to_chorus_mod_depth` | 1.6e-05 |
| `voice_tune` | 5 | 4 | poly (voice matrix) | per block | `poly_macro_to_voice_tune` | 4.4e-08 |
| `delay_frequency` | 5 | 5 | mono (bus matrix) | per block | `mono_macro_to_delay_frequency` | 4.9e-08 |
| `lfo_N_smooth_time` | 5 | 3 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_smooth_time` | 4.9e-08 |
| `eq_high_gain` | 4 | 4 | mono (bus matrix) | per block | `mono_macro_to_eq_high_gain` | 6.0e-08 |
| `chorus_feedback` | 4 | 4 | mono (bus matrix) | per block | `mono_macro_to_chorus_feedback` | 5.6e-06 |
| `phaser_blend` | 4 | 3 | mono (bus matrix) | per block | `mono_macro_to_phaser_blend` | 9.9e-08 |
| `osc_N_distortion_phase` | 4 | 3 | poly (voice matrix) | per block | `poly_macro_to_osc_1_distortion_phase` | 4.2e-08 |
| `eq_high_cutoff` | 4 | 4 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mono_macro_to_eq_high_cutoff` | 5.8e-08 |
| `filter_fx_resonance` | 4 | 4 | mono (bus matrix) | per block | `mono_macro_to_filter_fx_resonance` | 7.5e-08 |
| `compressor_mix` | 4 | 4 | mono (bus matrix) | per block | `mono_macro_to_compressor_mix` | 1.0e-05 |
| `voice_transpose` | 4 | 4 | poly (voice matrix) | per block | `poly_macro_to_voice_transpose` | 4.3e-08 |
| `lfo_N_stereo` | 4 | 3 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_stereo` | 4.9e-08 |
| `env_N_delay` | 4 | 4 | poly (voice matrix) | per block | `poly_macro_to_env_2_delay` | 4.7e-08 |
| `lfo_N_delay_time` | 4 | 1 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_delay_time` | 6.0e-08 |
| `filter_fx_blend_transpose` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_filter_fx_blend_transpose` | 4.5e-08 |
| `eq_low_gain` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_eq_low_gain` | 4.7e-08 |
| `delay_filter_cutoff` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_delay_filter_cutoff` | 1.7e-07 |
| `chorus_cutoff` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_chorus_cutoff` | 2.7e-06 |
| `distortion_filter_blend` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_distortion_filter_blend` | 9.3e-08 |
| `eq_band_resonance` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_eq_band_resonance` | 5.9e-08 |
| `compressor_band_gain` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_compressor_band_gain` | 8.6e-06 |
| `distortion_filter_resonance` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_distortion_filter_resonance` | 1.5e-07 |
| `lfo_N_phase` | 3 | 3 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_phase` | 4.5e-08 |
| `stereo_routing` | 3 | 3 | mono (bus matrix) | per block | `mono_macro_to_stereo_routing` | 4.9e-08 |
| `eq_low_resonance` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_eq_low_resonance` | 5.8e-08 |
| `compressor_low_gain` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_compressor_low_gain` | 5.9e-06 |
| `osc_N_frame_spread` | 2 | 2 | poly (voice matrix) | per block | `poly_macro_to_osc_1_frame_spread` | 4.4e-08 |
| `osc_N_distortion_spread` | 2 | 2 | poly (voice matrix) | per block | `poly_macro_to_osc_1_distortion_spread` | 4.8e-08 |
| `reverb_delay` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_reverb_delay` | 3.9e-08 |
| `osc_N_detune_power` | 2 | 2 | poly (voice matrix) | per block | `poly_macro_to_osc_1_detune_power` | 4.4e-08 |
| `delay_aux_frequency` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_delay_aux_frequency` | 1.2e-07 |
| `compressor_release` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_compressor_release` | 1.0e-05 |
| `filter_N_formant_y` | 2 | 1 | poly (voice matrix) | audio-rate in the reference; per block here (tracked) | `poly_macro_to_filter_1_formant_y` | 1.9e-07 |
| `phaser_mod_depth` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_phaser_mod_depth` | 1.4e-07 |
| `phaser_feedback` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_phaser_feedback` | 1.3e-07 |
| `phaser_phase_offset` | 2 | 1 | mono (bus matrix) | per block | `mono_macro_to_phaser_phase_offset` | 1.2e-07 |
| `reverb_low_shelf_gain` | 2 | 2 | mono (bus matrix) | per block | `mono_macro_to_reverb_low_shelf_gain` | 4.0e-08 |
| `env_N_hold` | 2 | 2 | poly (voice matrix) | per block | `poly_macro_to_env_2_hold` | 4.8e-08 |
| `lfo_N_fade_time` | 1 | 1 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_fade_time` | 5.0e-08 |
| `chorus_spread` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_chorus_spread` | 2.2e-06 |
| `portamento_time` | 1 | 1 | poly (voice matrix) | per block | `poly_macro_to_portamento_time` | 4.4e-08 |
| `delay_tempo` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_delay_tempo` | 4.3e-08 |
| `delay_aux_tempo` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_delay_aux_tempo` | 1.2e-07 |
| `lfo_N_keytrack_transpose` | 1 | 1 | poly (voice matrix) | per block | `poly_macro_to_lfo_1_keytrack_transpose` | 4.8e-08 |
| `delay_filter_spread` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_delay_filter_spread` | 1.7e-07 |
| `reverb_size` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_reverb_size` | 3.9e-08 |
| `reverb_chorus_amount` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_reverb_chorus_amount` | 4.0e-08 |
| `compressor_attack` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_compressor_attack` | 1.0e-05 |
| `filter_N_formant_x` | 1 | 1 | poly (voice matrix) | audio-rate in the reference; per block here (tracked) | `poly_macro_to_filter_1_formant_x` | 1.6e-07 |
| `filter_N_formant_spread` | 1 | 1 | poly (voice matrix) | per block | `poly_macro_to_filter_1_formant_spread` | 1.9e-07 |
| `filter_fx_drive` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_filter_fx_drive` | 5.7e-08 |
| `eq_high_resonance` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_eq_high_resonance` | 5.9e-08 |
| `filter_fx_formant_transpose` | 1 | 1 | mono (bus matrix) | audio-rate in the reference; per block here (tracked) | `mono_macro_to_filter_fx_formant_transpose` | 1.8e-07 |
| `phaser_tempo` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_phaser_tempo` | 1.3e-07 |
| `osc_N_phase` | 1 | 1 | poly (voice matrix) | audio-rate in the reference; per sample | `mod_lfo_to_phase` | 4.4e-08 |
| `reverb_high_shelf_gain` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_reverb_high_shelf_gain` | 3.9e-08 |
| `reverb_low_shelf_cutoff` | 1 | 1 | mono (bus matrix) | per block | `mono_macro_to_reverb_low_shelf_cutoff` | 4.0e-08 |
| `osc_N_stereo_spread` | 1 | 1 | poly (voice matrix) | per block | `poly_macro_to_osc_1_stereo_spread` | 4.5e-08 |
