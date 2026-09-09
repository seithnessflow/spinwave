# Golden bench: the reference half

Renders test cases through **Vital's own DSP core**, so Spinwave can be
compared against the thing it claims to reproduce.

This is not part of Spinwave. No crate depends on it, and it is built once
on one machine to generate reference audio that gets committed. After that
the comparison is pure Rust and nobody needs a C++ compiler again.

## Why not write the reference in Rust

Because then it would compare Spinwave against Spinwave. The value comes
precisely from the reference being foreign code we did not write: a
reimplementation would carry over the same misreadings of the C++ that
produced the bugs this bench exists to catch.

## Building

Needs a C++ compiler, CMake and Ninja, plus a clone of Vital's sources.

```sh
cmake -S tools/golden -B tools/golden/build -G Ninja \
      -DVITAL_SRC=<path-to>/vital-reference/src
cmake --build tools/golden/build
```

On Windows, run those inside a `vcvars64.bat` shell.

Vital's DSP core barely touches JUCE: two of its 68 sources include
`JuceHeader.h`, and between them they use `String` and a couple of file
helpers. `stub/` replaces the framework, so no Projucer run and no JUCE
build are involved. `stub/kissfft/` is Vital's own bundled FFT with one
variable-length array replaced, because MSVC has no VLAs.

## Running a case

```sh
tools/golden/build/vital_golden cases/osc_saw.txt out.raw
```

A case file is one directive per line:

```
rate 44100            # sample rate
seconds 1.0           # render length
wave saw              # single-cycle shape loaded into every oscillator
note 45 0.9 0.0 0.6   # midi note, velocity, start, hold (seconds)
set osc_1_level 0.7   # any control, by its Vital parameter name
```

Output is raw little-endian f32, interleaved stereo.

## What the bench does and does not prove

Both halves load **one predefined single-cycle shape** rather than building
a morphing wavetable, so a difference in the audio is a difference in the
DSP rather than in two different wavetable builders. Wavetable
construction needs its own comparison, and does not have one yet.
