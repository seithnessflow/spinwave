# vital-rs (working name)

A ground-up Rust rework of the [Vital](https://github.com/mtytel/vital) wavetable
synthesizer engine (GPLv3). Not a transliteration: the DSP is ported faithfully
(same approximations, same sound), the architecture is redesigned.

## What changed from the C++ reference

- **Static voice graph** instead of the dynamic pointer-based `Processor`
  graph — the patch topology is fixed at compile time; only modulation
  connections are dynamic.
- **Portable SIMD** (`wide`) instead of per-platform intrinsics; same
  two-stereo-voices-per-vector model (`[L0, R0, L1, R1]`).
- **nih-plug** (CLAP + VST3) instead of JUCE for plugin hosting.
- **serde** for presets, aiming for `.vital` (JSON) compatibility.
- No account/auth, no cloud features.

## Workspace

- `crates/vital-poly` — SIMD voice-pair primitives, fast math (`futils` port).
- `crates/vital-dsp` — filters, oscillators, modulators, effects. Framework-free.
- `tools/` — Python analysis/golden-test scripts.

## License

GPLv3 (derivative of Vital by Matt Tytel). "Vital" is a trademark of its
author; this project will ship under its own name and branding.
