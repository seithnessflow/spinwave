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
skip 1.0              # seconds rendered but not written (the primer note)
modulate lfo_1 filter_1_cutoff 0.5   # a connection: source, destination, amount
lfo_shape flat 0.3    # every LFO drawn as a horizontal line at 0.3
random_seed 18        # the seed the Spinwave side gives its random LFOs;
                      # accepted and ignored here, see random_seed.py
```

Output is raw little-endian f32, interleaved stereo. `--probe <source>`
also writes `<out.raw>.probe.csv`, the source's control-rate value per
block, after a self-test on `env_1`.

`random_seed.py` recovers the seed behind a Perlin random LFO from such a
probe CSV: both engines seed generators from a process-global counter, so
the reference's `random_1` holds whatever seed its construction order
gave it (18 in this build; random_2..4 hold 17, 16, 15 and the per-note
`random` 19 — the counter runs down the construction order), and a case
with a random source must pin the Spinwave side to the same one or it
compares noise.

## Real presets

`vital_golden --preset <file.vital> <out.raw> [--fixed-phase] [--skip S]
[--dump-tables DIR]` renders a real preset: the reference's own preset
migration (`LoadSave::updateFromOldVersion`, copied verbatim into
`reference_migration.inc` by `extract_migration.py`), its wavetable
creator, its sample decoder; a primer note hidden by the skip, then C3.
`--dump-tables` writes each oscillator's built table (`osc_N.raw`) and
the SMP sample (`sample.raw`). `bank_compare.py` runs a whole folder of
presets through both engines and `bank_bisect.py` takes one preset
apart; the results live in `notes/bank-compare.md`.

`VITAL_GOLDEN_DUMP_DEST=<name>` makes any render write `<out>.dest.raw`:
the named mono destination's total per sample (a control-rate total
repeated over its block) and, for a voice-level name, the poly total's
value per block, four lanes.

`--diode <cutoff> <res> <drive_db> <rate> out.raw` and
`--distortion <type> <drive_db> <rate> out.raw` run one filter or one
distortion alone on a fixed saw, the twins of the `diode_probe` and
`distortion_probe` examples in `spinwave-dsp`: a unit compared apart
from the voice.

## What the bench does and does not prove

The case corpus loads **one predefined single-cycle shape** rather than
building a morphing wavetable, so a difference in a case's audio is a
difference in the DSP rather than in two different wavetable builders.
Wavetable construction is compared through the presets: `--dump-tables`
on both sides, frame by frame (the `tables` column of the bank table).
