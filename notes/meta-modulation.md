# Meta-modulation: what the reference does, established by cases (2026-09-12)

A connection's amount is itself a modulation destination in the
reference (`modulation_N_amount` is a poly mod control,
`synth_voice_handler.cpp:86`). Half of the connections the real presets
lose on load go there. Before porting, each behaviour below was settled
by a golden case and a twin differing by one setting, and read off the
reference's own bytes. Cases: `tools/golden/cases/meta_*`.

The method for every row: two references that should be identical
under one hypothesis and different under the other; `cmp` the files, or
the first differing frame and the RMS per 100 ms window.

## 1. Scale and route — `meta_macro_to_amount`

A macro at 0.6 into the amount of `lfo_1 → filter_1_cutoff` (base amount
0.3) with amount 0.5. The twin without the meta connection differs
(9.2e-2 RMS: the connection arrives). The twin with the LFO's amount set
statically to **0.9** matches at **2.5e-8** — float noise. So the
effective amount is

    base + source × amount × range(modulation_N_amount) = 0.3 + 0.6 × 0.5 × 2 = 0.9

the destination range of an amount being 2 (`[−1, 1]`), like any other
destination. Unipolar source, no surprise.

## 2. Bounds — `meta_bounds`

Base 0.5, macro 1.0 × amount 1.0 × 2 = +2.0 → 2.5 if nothing clamps.
Three references are **byte-identical**: the meta case, the twin with a
static amount of 1.0, and the twin with a static amount of 2.5. So the
amount is clamped to `[−1, 1]` after the sum — and a static amount past
1 is clamped the same way (which Spinwave already does: both static
twins pass at 5e-8).

## 3. Timing — `meta_step_timing`

An envelope with zero attack (a step at note-on) on the amount of a
constant `macro → cutoff` connection (0.2, stepping to 0.8). Against the
static-0.8 twin, the RMS of the difference per block after note-on:
**1.6e-2 in the note-on block, 1.6e-4 in the next, 4.6e-6, then 8.8e-9**.
Against the static-0.2 twin it stays at 1.2e-1. So the stepped amount
is read close to the block the source steps and **ramped linearly
across a block** from the previous block's amount (the reference's
`ModulationConnectionProcessor` ramps `current_amount` by
`delta_amount` per sample).

**Corrected by the port.** "Read in the same block, no lag" was what
these numbers looked like before a port existed to test it; the port
read this case at 9.9e-4 with no lag and at 5.4e-8 with **one block of
lag on the target connection**. The lag is specific to the target: the
target here is a connection from a **mono** source (the macro), which
the reference evaluates before the voices — so it sees the meta
offsets of the previous block. Cross-checked both ways:
`meta_ramp_on_mono_source_target` (envelope ramp on the amount of a
macro connection) 3.5e-3 without the lag → 5.4e-8 with;
`meta_step_on_poly_source_target` (the same step on the amount of a
velocity connection) 4.2e-8 either way. Rule in the port: a connection
whose source `is_mono()` (macros, the wheels — the wheels are assumed
from the macro measurement, not measured) reads its amount and power
offsets from the previous block; every other connection reads the
current block's.

## 4. Chaining and slot order — `meta_chain_forward` / `_backward` / `_no_macro`

The same graph, `macro → amount of (env_2 → amount of (lfo_1 → cutoff))`,
wired in slot order 1-2-3 and in the reverse order. Amounts kept small
(0.1 each, sources at most 0.8) so nothing clamps — a first attempt with
0.5s saturated every link at 1 and hid the chain behind the clamp
(differences only during the envelope's attack). The two orders are
**byte-identical**, and both differ from the chain without its macro
link by 7.8e-2: the chain is live, and **slot order does not matter**.
Combined with §3 (no lag on a direct link) the reference resolves a
chain within the block regardless of slot numbering — consistent with
its `ProcessorRouter` ordering processors by dependency, which the port
must do too (resolve amounts in dependency order, not slot order).

## 5. Three links, and a true cycle — `meta_chain_three_links`, `meta_true_cycle`

The case first named `meta_cycle` (`env_2` on the LFO connection's
amount, `lfo_1` on the envelope connection's amount) is **not a graph
cycle**: the source `lfo_1` is a modulator, not the connection
`lfo_1 → cutoff`; the graph is a three-link chain and the port renders
it at 4.4e-8 in dependency order. Renamed.

`meta_true_cycle` is one: two meta connections targeting each other's
amounts (`env_2 → amount of slot 3`, `lfo_1 → amount of slot 2`, slot 1
being `lfo_1 → cutoff`). The reference renders it (no hang, no NaN); the
port keeps the previous block's offsets for the connections on a cycle
(Kahn's algorithm leaves them unplaced) and matches at 4.1e-8. Honest
limit of that case: by construction the two meta connections'
outputs never reach the audio (each modulates only the other), so the
bytes prove the cycle is bounded, deterministic and NaN-free — the
gate's minimum requirement — and not the lag policy. A cycle whose
members also feed an audible destination would; nothing real does.

The reference's `ProcessorRouter` has no explicit cycle handling for
modulation connections; the order it falls into is whichever its
dependency sort yields, and the case cannot distinguish it from ours.

## 6. Regime — `meta_lfo_on_audio_rate_amount`

An 8 Hz LFO on the amount of an audio-rate connection (`env_2 →
cutoff`). The reference reads the amount once per block and ramps it
across the block, for control-rate and audio-rate connections alike;
the port does the same and matches at 4.7e-8.

Related, found by `meta_chain_three_links` (7.8e-4 before, 4.4e-8
after): the **control value of an audio-rate source** (an envelope or
LFO that is also rendered per sample) is its buffer's **first** sample
of the block — the reference's `at(0)` — not its last. Spinwave read
the end-of-block value. This is not specific to meta-modulation; it
affects every control-rate connection from such a source and was only
invisible because the sources are smooth.

## 7. Power — `meta_power`, `meta_power_bounds`

The power (`modulation_N_power`, `[−10, 10]`) is its own destination
with range **20** and is **not clamped**: `meta_power` (macro 0.5 ×
amount 0.2 × 20 = +2) is byte-identical to its static twin at power 2;
`meta_power_bounds` (+20) matches the static twin at **20**, not the
one at 10. The port adds the offset to the power unclamped.

## 8. Interactions — bipolar target, LFO meta source, poly source, bypass

- **Bipolar target** (`meta_on_bipolar_target`): the modulated amount
  multiplies after the polarity branch — the meta case equals the
  static twin with amount 0.9 and the bipolar flag (5e-8).
- **Polarity of the meta connection** (`meta_lfo_source_unipolar` /
  `_bipolar`): both render, both match (4e-8, 5e-8); the difference
  between them is what the polarity flag does to any connection. Since
  this pass **every case writes every slot's polarity explicitly**
  (the reference makes a fresh connection from an lfo / random /
  stereo / pitch source bipolar; `make_cases.py` no longer relies on
  that default). All 96 earlier references re-rendered byte-identical.
- **Poly meta source on two voices** (`meta_poly_source_two_voices`,
  velocity on the amount, two notes of different velocities): the
  amount is per voice (5e-8 with lanes; a single value would not
  match).
- **Bypassed target** (`meta_bypassed_target`): identical to the twin
  with no connection at all — bypass wins over the modulated amount.

## What the port is (2026-09-12)

- `ModDest::ModulationAmount(slot)` (range 2, summed with the base then
  clamped to `[−1, 1]`) and `ModDest::ModulationPower(slot)` (range 20,
  not clamped), poly, per voice, per block.
- Resolved in **dependency order** (Kahn on "meta connection → target
  slot", fixed arrays, no allocation); connections on a cycle read the
  previous block's offsets.
- Connections from a **mono source** read the previous block's offsets
  (§3); all others the current block's.
- An audio-rate source's control value is its buffer's sample 0 (§6).
- Bypass, polarity and stereo of the target are untouched: the offset
  enters only where the amount and the power enter.

All 23 meta cases (including twins) match at 4e-8..6e-8. Both
`meta_bounds` twins and `meta_power_bounds_twin_overflow` sit
deliberately against a bound (that is their point) and are the origin
of the rule that every other case's values must stay interior to their
range over the compared window.

One method note from this pass: the probes' status outputs read one
block late (the reference posts them after the block), which had been
read earlier as a one-block *lead* of Spinwave's sources. It was the
probe.

