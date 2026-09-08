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
- `crates/spinwave-params` — parameter table (794 params) + `.vital` preset model.
- `crates/spinwave-plugin` — CLAP/VST3/standalone shell (`spinwave.exe`).
- `tools/` — Python analysis/golden-test scripts.

## Try it

```sh
cargo run -p spinwave-engine --example render_demo   # writes spinwave-demo.wav
cargo run -p spinwave-plugin --release               # standalone with MIDI
cargo test                                           # full suite
```

## License

GPLv3 (derivative of Vital by Matt Tytel). "Vital" is a trademark of its
author; Spinwave ships under its own name and branding.
