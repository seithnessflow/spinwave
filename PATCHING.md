# Spinwave patching guide (for LLM agents)

How to sound-design on Spinwave through the MCP tools. Values are ENGINE
values (`describe_params` gives each range/default).

## Signal flow

```
osc 1/2/3/4 ─destination─▶ filter 1 ─┐
sample ────destination──▶ filter 2 ─┼─▶ amp env ─▶ main effect chain ─▶ master
noise ─────destination──▶ effects ──┘              (reorderable, 9 slots)
                        ▶ direct out (bypasses the chain)
                        ▶ bus A / bus B (send buses with their own chains,
                          returning to the master or into the main chain)
```

- `osc_N_destination` / `noise_destination`: 0=filter1, 1=filter2, 2=both,
  3=effects (skip voice filters), 4=direct out (skip the whole chain —
  clean subs live here), 5=bus A, 6=bus B.
- Filter routing: `filter_2_filter_input=1` chains filter1→filter2.
- Effect chain order: `effect_chain_order` (factorial code; default 0 =
  chorus, comp, delay, disto, eq, filterfx, flanger, phaser, reverb).
- Send buses: `bus_a_on`, `bus_a_send` (0..1, tapped before the main
  chain), `bus_a_return_db`, `bus_a_output` (0 = master, 1 = into the main
  chain), then every effect parameter with the `bus_a_` prefix
  (`bus_a_reverb_on`, `bus_a_delay_feedback`, `bus_a_effect_chain_order`,
  `bus_a_filter_fx_cutoff`...). Same with `bus_b_`.
- Per-effect splits: `fx_split_<effect>` (0 full, 1 mid, 2 side, 3 low,
  4 high) and `fx_split_<effect>_crossover` (Hz), also under `bus_a_` /
  `bus_b_`. Convolution and frequency-shifter slots arrive with the
  engine merge.

## Spinwave-only parameters

Everything Vital has, plus (all listed by `describe_params`, flagged
`spinwave_only`; `save_preset` omits them at their default so Vital can
still open the file):

- Extra slots with Vital's scales: `osc_4_*`, `env_7_*`, `env_8_*`,
  `lfo_9_*` .. `lfo_12_*`, `macro_control_5..8`; `polyphony` up to 64.
- `osc_N_engine`: 0 wavetable, 1 sample, 2 granular, 3 multisample.
  Material comes from `load_sample` (sample + granular), `import_wavetable`
  (wavetable), `load_sfz` (multisample) and is embedded in the preset
  (`settings.wavetables[slot]`, `settings.spinwave_materials`), so
  `get_patch`, `save_preset`, `live_apply` and the DAW's saved state carry it.
- Sample engine: `osc_N_smp_rate` (0.25..4), `osc_N_smp_loop`,
  `osc_N_smp_slice` (-1 = whole), `osc_N_smp_offset` (frames). Level /
  transpose / tune / pan / keytrack come from the common `osc_N_*` keys.
- Granular engine: `osc_N_gran_position`, `_position_spray`, `_size`
  (seconds), `_size_spray`, `_density` (grains/s), `_pitch_spray`
  (semitones), `_window` (0 hann, 1 triangle, 2 expo, 3 tukey, 4 rect),
  `_direction` (0 fwd, 1 rev, 2 bidir), `_stereo_spray`.
- Noise source: `noise_on`, `noise_destination`, `noise_level`,
  `noise_pink` (white→pink), `noise_tilt` (-1 dark .. 1 bright),
  `noise_pan`, `noise_stereo` (0 mono .. 1 decorrelated).
- LFO generators: `lfo_N_generator` (0 shape, 1 sample&hold, 2 chaos
  Lorenz, 3 chaos Rössler), `lfo_N_sh_glide`, `lfo_N_chaos_speed`.
- Vital's global `settings.sample` (the SMP section) is decoded too.

## Scales to never get wrong

| Parameters | Stored as | Example |
|---|---|---|
| `env_N_attack/decay/release/hold/delay` | quartic root of seconds | 0.35 ≈ 15 ms, 0.75 ≈ 0.32 s, 1.0 = 1 s |
| `lfo_N_frequency`, `random_N_frequency`, effect `*_frequency` | log2(Hz) | −2 = 0.25 Hz, 0 = 1 Hz, 3 = 8 Hz |
| `chorus_delay_1/2`, `reverb_decay_time`, `lfo_N_smooth_time`, `portamento_time` | log2(seconds) | 0.5 ≈ 1.4 s, −7.5 ≈ 5.5 ms |
| `filter_*_cutoff`, `eq_*_cutoff` | MIDI note | 60 ≈ 261 Hz, 72 ≈ 523 Hz |
| `volume` | (dB + 80)² | 5473 ≈ −6 dB, 6400 = 0 dB |
| `osc_N_unison_detune` | square root of the detune amount (× `osc_N_detune_range` = cents) | default 4.47 → 20 → 40 cents |
| `osc_N_level`, `sample_level`, `eq_*_resonance`, `reverb_chorus_amount` | square root of value | 0.707 → 0.5 |

`describe_params` reports each parameter's `scale` (`Quadratic` = the
stored value is squared before use, `Exponential` = 2^stored, `Quartic` =
stored⁴).

## Modulation

`add_modulation(source, destination, amount)`. Sources: `lfo_1..12`,
`env_1..8` (env_1 = amp), `random_1..4`, `macro_control_1..8`, `note`,
`note_in_octave`, `velocity`, `aftertouch`, `slide`, `lift`, `mod_wheel`,
`pitch_wheel`, `stereo`, `random`. Options: `bipolar` (center the source),
`stereo` (invert on right channel), `power` (curve). Destinations: every
`osc_N_*`, `filter_N_*`, `env_N_*`, `lfo_N_*`, `random_N_frequency`,
`sample_*`, `volume`, `pitch_wheel` the engine exposes, plus the bus
effect parameters of the main chain. A connection the engine cannot route
is kept in the preset and listed in the load report
(`ignored_connections`).

## Load report

`set_patch`, `load_preset` and the live `preset` command answer with a
report: `migrated_from` + `migrations` (old Vital versions are converted
like `LoadSave::updateFromOldVersion`), `ignored_connections`,
`unknown_params` (numeric settings keys the table does not know), `notes`
(remap curves not applied yet, SFZ files that failed). `is_clean` means
nothing was dropped.

LFO shapes come from the preset's `lfos` array (points/powers/smooth);
default is a triangle. A 3-point smooth shape `[0,1, 0.5,0, 1,1]` is a
sine-like wave.

## Recipes that work

- **Round bass**: saw → 24 dB low-pass (cutoff ~55–60 + keytrack 0.7),
  drive 6–9 dB, attack ~30 ms, small env→cutoff bloom (amount 0.2).
- **Neuro bass**: osc FM'd by the sub (`osc_1_distortion_type=7`), band or
  dual-notch filter (style 3) resonance 0.6–0.8, 2–3 desynced LFOs on
  cutoff/blend/phase-distortion, sub at transpose 0 routed direct-out,
  hard clip 24–28 dB, multiband comp crushing. Avoid big unison detune
  (chorus-y "tonal" blur) — keep 1–2 voices, get fat from drive.
- **Pad**: unison 6–8 detune ~1.5 spread 0.9, slow attack (stored ≥0.9),
  fifth/octave layer, Wide & Wet rack.
- **Pluck**: fast decay to low sustain + env→cutoff (amount 0.4–0.5,
  fast decay env), keytrack ~0.7, synced delay.
- **Loudness**: measure first (`play` returns peak/RMS). Compress →
  clip → conservative EQ after the clip → trim `volume` so peak ≈ 0.95.
  Boosting EQ after the clipper re-creates peaks.

## Listening to references

`analyze_file(path, start, duration)` hears any WAV/MP3/FLAC/OGG — use it
on the user's reference tracks (target the drop with start/duration).
Key readings: `movement.mod_rates_hz` = wobble/LFO rates to reproduce,
`centroid_trajectory_hz` = brightness motion, `bands_db` = spectral
balance, `texture.spectral_flatness` = dirtiness, `onset_density` =
rhythm. Then `compare(reference_path)` against your last render tells
you what still differs, in sound-design terms. Loop: analyze reference →
patch → play → compare → adjust.

## Workflow

1. `describe_params` to find names; `list_racks`/`apply_rack` for
   ready-made effect chains.
2. Edit with `set_params` / `add_modulation`.
3. `play` — read the analysis (bands, centroid=brightness, envelope
   attack, pitch) instead of guessing. `rms_db` is dBFS of the mono sum
   (not LUFS); band levels are relative to the loudest band; relative
   `out_path`s land in the server's output directory and get a `-1`,
   `-2`... suffix instead of overwriting (`overwrite: true` to replace).
4. Iterate; `save_preset` when done (materials embedded).
5. Live: `live_instances` → `live_attach` → `live_set_params` /
   `live_sequence` while the user listens. Always pass the user's audio
   device to `live_start` (see `list_audio_devices`). `live_stop` only
   stops a standalone this server spawned; an attached DAW instance is
   just detached. The `sequencer` locks to the host transport's beat grid
   while the DAW plays.
