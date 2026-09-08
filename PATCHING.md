# Spinwave patching guide (for LLM agents)

How to sound-design on Spinwave through the MCP tools. Values are ENGINE
values (`describe_params` gives each range/default).

## Signal flow

```
osc 1/2/3 ──destination──▶ filter 1 ─┐
sample ────destination──▶ filter 2 ─┼─▶ amp env ─▶ bus effect chain ─▶ master
                        ▶ effects ──┘              (reorderable, 9 slots)
                        ▶ direct out (bypasses the chain)
```

- `osc_N_destination`: 0=filter1, 1=filter2, 2=both, 3=effects (skip voice
  filters), 4=direct out (skip the whole chain — clean subs live here).
- Filter routing: `filter_2_filter_input=1` chains filter1→filter2.
- Effect chain order: `effect_chain_order` (factorial code; default 0 =
  chorus, comp, delay, disto, eq, filterfx, flanger, phaser, reverb).

## Scales to never get wrong

| Parameters | Stored as | Example |
|---|---|---|
| `env_N_attack/decay/release/hold/delay` | quartic root of seconds | 0.35 ≈ 15 ms, 0.75 ≈ 0.32 s, 1.0 = 1 s |
| `lfo_N_frequency`, `random_N_frequency`, effect `*_frequency` | log2(Hz) | −2 = 0.25 Hz, 0 = 1 Hz, 3 = 8 Hz |
| `chorus_delay_1/2`, `reverb_decay_time` | log2(seconds) | 0.5 ≈ 1.4 s |
| `filter_*_cutoff`, `eq_*_cutoff` | MIDI note | 60 ≈ 261 Hz, 72 ≈ 523 Hz |
| `volume` | (dB + 80)² | 5473 ≈ −6 dB, 6400 = 0 dB |
| `eq_*_resonance`, `reverb_chorus_amount` | square root of value | — |

## Modulation

`add_modulation(source, destination, amount)`. Sources: `lfo_1..8`,
`env_1..6` (env_1 = amp), `random_1..4`, `macro_control_1..4`, `note`,
`velocity`, `aftertouch`, `slide`, `lift`, `mod_wheel`, `pitch_wheel`,
`stereo`, `random`. Options: `bipolar` (center the source), `stereo`
(invert on right channel), `power` (curve).

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

## Workflow

1. `describe_params` to find names; `list_racks`/`apply_rack` for
   ready-made effect chains.
2. Edit with `set_params` / `add_modulation`.
3. `play` — read the analysis (bands, centroid=brightness, envelope
   attack, pitch) instead of guessing.
4. Iterate; `save_preset` when done.
5. Live: `live_instances` → `live_attach` → `live_set_params` /
   `live_sequence` while the user listens. Always pass the user's audio
   device to `live_start` (see `list_audio_devices`).
