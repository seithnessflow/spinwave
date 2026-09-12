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
is read **in the block the source steps**, not one block later, and
**ramped linearly across that block** from the previous block's amount
(the residual in the note-on block is the ramp; the reference's
`ModulationConnectionProcessor` ramps `current_amount` by
`delta_amount` per sample). No lag to model.

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

## 5. A cycle — `meta_cycle`

`env_2` on the LFO connection's amount and `lfo_1` on the envelope
connection's amount. The reference renders it (no hang, no NaN) and it
differs from the acyclic chain by 6.2e-2: the back edge acts. What order
it uses inside the cycle is not readable from the bytes alone; the port
will try slot order with a one-block lag on the back edge, and the case
judges it. Nothing real does this; a generated patch will.

## 6. Regime — `meta_lfo_on_audio_rate_amount`

An 8 Hz LFO on the amount of an audio-rate connection (`env_2 →
cutoff`). The reference reads the amount once per block (`at(0)`) and
ramps it across the block, for control-rate and audio-rate connections
alike — that is what the code says; the case is what will say whether
the port reads it right. Not readable from the bytes before the port.

## What the port has to be, then

- `ModDest::ModulationAmount(slot)` (and, the same way, `power`) as a
  poly destination with range 2, resolved per voice per block.
- Amounts resolved in **dependency order**: a connection's amount is
  computed from the amounts of the connections that target it, which
  are computed first. Cycles fall back to slot order with the previous
  block's value on the back edge (hypothesis; `meta_cycle` judges).
- The summed amount **clamped to [−1, 1]**, then ramped across the block
  as the transform already ramps a changed amount.
- Read in the same block as the source, no lag.

Spinwave today refuses all eight meta cases (`modulation_N_amount` is
not a destination) and passes their six plain twins at float noise, so
the bench is ready to judge the port the moment it exists.
