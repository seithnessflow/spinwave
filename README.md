# Spinwave

A wavetable synthesizer in Rust — a ground-up rework of the
[Vital](https://github.com/mtytel/vital) engine (GPLv3). Not a
transliteration: the DSP is ported faithfully (same approximations, same
sound), the architecture is redesigned.

## What changed from the C++ reference

- **Static voice graph** instead of the dynamic pointer-based `Processor`
  graph — the patch topology is fixed at compile time; only modulation
  connections are dynamic.
- **Portable SIMD** (`wide`) instead of per-platform intrinsics; same
  two-stereo-voices-per-vector model (`[L0, R0, L1, R1]`).
- **nih-plug** (CLAP + VST3 + standalone) instead of JUCE.
- **serde** presets, aiming for `.vital` (JSON) compatibility.
- No account/auth, no cloud features.

## Workspace

- `crates/spinwave-poly` — SIMD voice-pair primitives, fast math.
- `crates/spinwave-dsp` — oscillators, filters, modulators, effects.
- `crates/spinwave-engine` — voice allocation, modulation matrix, the
  synth voice kernel, microtuning.
- `crates/spinwave-params` — parameter table (Vital's 794 params plus the
  Spinwave-only namespace, flagged `spinwave_only`), the `.vital` preset
  model, and the version migrations ported from `LoadSave::updateFromOldVersion`
  (`Preset::upgrade`, presets from Vital 0.2.x onwards load like in Vital).
- `crates/spinwave-plugin` — CLAP/VST3/standalone shell (`spinwave.exe`):
  MIDI CCs / pitch bend / pressure (`MidiConfig::MidiCCs`), the patch saved
  with the DAW project (`#[persist = "preset"]`, rebuilt off the audio thread
  on restore), the live TCP control channel (opened at activation, never at
  scan), an arp/step sequencer phase-locked to the host transport, and a
  garbage-collector thread so the audio thread never frees memory.
- `crates/spinwave-control` — MCP server (`spinwave-mcp.exe`) exposing the
  engine to LLM agents: patch editing, modulation routing, note rendering
  with audio analysis (levels, spectrum, envelope, pitch), live control of
  standalone or DAW-hosted instances.
- `tools/` — Python analysis/golden-test scripts, PowerShell live helpers.

## Spinwave beyond Vital

The engine is larger than Vital's: 4 oscillators, 8 envelopes, 12 LFOs,
8 macros, 64 voices, a dedicated noise source, two effect send buses with
their own chains, per-effect signal splits, sample / granular /
multisample oscillator engines. The extra parameters share Vital's names
and scales (`osc_4_level`, `env_7_attack`, `lfo_9_frequency`,
`macro_control_5`) and add a Spinwave namespace (`osc_N_engine`,
`osc_N_smp_*`, `osc_N_gran_*`, `noise_*`, `bus_a_*` / `bus_b_*`,
`fx_split_*`, `lfo_N_generator`, ...). See `PATCHING.md` for the list.
`save_preset` omits the Spinwave-only parameters at their default so the
`.vital` file still opens in Vital. Convolution and frequency-shifter
effect slots arrive with the engine merge.

Loading a preset returns a **load report**: migrations applied, modulation
connections the engine cannot route, unknown parameter names, remap curves
not applied yet. The MCP `set_patch` / `load_preset` tools and the live
`preset` command all return it.

## Try it

```sh
cargo run -p spinwave-engine --example render_demo   # writes spinwave-demo.wav
cargo run -p spinwave-plugin --release               # standalone with MIDI
cargo test                                           # full suite

# LLM control (MCP): build, then register with Claude Code
cargo build -p spinwave-control --release
claude mcp add spinwave -- <repo>/target/release/spinwave-mcp.exe --out-dir <renders>
```

Renders with a relative `out_path` land in `--out-dir` (or
`SPINWAVE_OUT_DIR`, else the repository root, else the current directory)
and never overwrite an earlier take unless asked (`overwrite: true`).

Live control: every instance (standalone or DAW plugin) listens on
`127.0.0.1:41929..41979` once the host activates it and registers in
`%TEMP%/spinwave-instances.json` (dead entries are purged by every
reader). `SPINWAVE_LIVE=0` disables it, `SPINWAVE_LIVE_PORT` forces a port.

## License

GPLv3 (derivative of Vital by Matt Tytel). "Vital" is a trademark of its
author; Spinwave ships under its own name and branding.
