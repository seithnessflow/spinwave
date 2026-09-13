# Knowledge base for sound design — the design (2026-09-13)

A memory of what works, in three stores that never mix: what this engine
measurably does (`measured/`), what real patches are shaped like
(`corpus/`), what the trade says (`declared/`), and later a fourth of
weights over the first three. This note is the structuring decision the
task asked to stop at: the schema of the stores, the format of an entry
with its provenance, the regeneration command, and how `explore`
consumes `measured/`. Nothing below is implemented.

## What exists that this builds on

- `ops::explore` already weights its mutations by a sensitivity measured
  **on the patch itself**: one Lite render per active parameter, a
  quarter-range step, the band distance to the origin, normalised to the
  strongest (`explore.rs::sensitivity_weights`). So `explore` with 40
  active parameters spends 41 renders before its first variant. That is
  the cost `measured/` removes, and the number it must beat.
- `ops::explain::suggest` moves every active parameter a step each way
  and re-renders for a named quality (`Quality`: level, brightness,
  harshness, warmth, width, attack, sustain, noise, movement, a band,
  aliasing). Same shape: the answer is recomputed at the question.
- `sensitivity::context_for` is the dependency table: which switches,
  models and connections a parameter needs before it can do anything
  (a filter's formant controls need the filter on AND the formant
  model; an LFO's shape needs the LFO connected). `explain::active_parameters`
  is its inverse: which parameters are live in THIS patch.
- `Descriptors` (`ops::descriptors`) is the vocabulary every store
  speaks — 30-odd named, unit-bearing measures; `f0` is conditional
  already (`describe_without_pitch`).
- Engines are recycled per session and proven bit-identical to fresh
  ones; kernels are not, and that is the throughput ceiling (43 → 94
  renders/s from 1 to 4 workers, then down: allocation, not compute).
- The 75 factory presets, loaded and rendered, with a bank table against
  the reference (`notes/bank-compare.md`): the first corpus, already
  trusted to the extent that note says.

## The engine version, so a measurement knows when it died

Every derived entry carries `engine`: the **source hash** of the engine —
SHA-256 over the sorted contents of `crates/spinwave-poly/src`,
`spinwave-dsp/src`, `spinwave-engine/src`, `spinwave-params/src` and
`spinwave-control/src/{analysis,ops}` (the descriptors are part of the
measurement), computed by a `build.rs` in `spinwave-control` and exposed
as `spinwave_control::ENGINE_FINGERPRINT`. Not the git commit: a commit
that touches only a note must not stale ten thousand renders, and a
dirty tree must still be able to measure. The git commit (and a dirty
flag) is stored beside it as provenance, never as the validity key.

An entry whose fingerprint differs from the running engine's is **stale**:
readable, reported as such by every consumer, never used for a weight.
`knowledge status` lists them; `knowledge measure` regenerates them.

## The three stores

All under `knowledge/` at the repo root, JSON, one file per unit small
enough to diff (a parameter, a term, a corpus statistic), committed. No
third-party preset, text or sample enters the repo; the corpus path is a
local configuration (`SPINWAVE_CORPUS`, or `--corpus`).

Every entry, in every store, starts with the same header:

```json
{
  "kind": "measured.parameter | corpus.structure | declared.term | weight.rule",
  "source": "knowledge measure --contexts factory | knowledge corpus <dir> | hand | knowledge weigh",
  "date": "2026-09-13",
  "engine": { "fingerprint": "sha256:…", "git": "a856e78", "dirty": false },
  "status": "fresh | stale | unvalidated | validated | refuted"
}
```

`status` is derived for measured and corpus entries (fresh/stale from the
fingerprint) and declared for `declared/` (unvalidated / validated /
refuted, by hand or by `knowledge validate`).

### `measured/` — the causal map

One file per parameter: `knowledge/measured/params/<name>.json`. It holds
a list of **observations**, each made in one **context** with one
**scenario** and one **step**:

```json
{
  "kind": "measured.parameter",
  "name": "filter_1_cutoff",
  "observations": [
    {
      "context": {
        "key": "filter_1_on=1;filter_1_model=3;osc_1_destination=0",
        "origin": "factory:Analog Pad",
        "preset_hash": "sha256:…"
      },
      "scenario": "lite",
      "step": { "from": 60.0, "to": 92.0, "fraction_of_range": 0.25 },
      "effect": {
        "distance_db": 6.4,
        "deltas": { "brightness_st": 9.1, "warmth_db": -2.3, "level_db": 0.8, "width": 0.0, "attack_s": 0.0, "sustain_s": 0.0, "noise": 0.01, "movement_db": 0.0, "harshness_db": 4.0 },
        "bands_db": [0.0, -0.4, 1.2, 3.8, 5.1, 6.0, 4.4, 1.0]
      },
      "renders": 2,
      "engine": { … }, "date": "…"
    }
  ]
}
```

The **context key** is the parameter's activation conditions as
`context_for` names them — the switches, models and connections that make
it live — evaluated on the patch it was measured in, in a canonical order.
Two patches with the same key are the same context *for that parameter*
as far as the dependency table knows; everything else about them is
noise the store must not pretend to average away. That is the honest unit
of generalisation: "cutoff of a 24 dB digital filter fed by osc 1", not
"cutoff". `origin` says which patch; the observation is reproducible from
the preset hash and the scenario.

Deltas are the `Quality` measures (`explain.rs`), signed, in their units,
so `suggest` reads them directly; `distance_db` is what `explore` weights
by; `bands_db` keeps the per-band shape for a later consumer.

Contexts measured, in this order: the parameter's **canonical context**
(what the sensitivity sweep builds: a base patch plus `context_for`'s
settings — one observation per parameter, always present), then the
**factory presets** where the parameter is active (up to 75 observations,
each with its own key), then the generated corpus when it exists.

What the store does not know, it says: a consumer asking for
`filter_1_cutoff` in context key K gets the observations with key K, or
`None` and the nearest keys with their observation counts — never a
value from another context passed off as an answer.

### `corpus/` — the structure of real patches

`knowledge/corpus/<corpus-id>/structure.json`, one per corpus (`factory`,
`public-<name>`, `generated-<seed>`), plus `catalogue.json` (per patch:
name hash, measured descriptors at the Lite scenario, quality flags: clips,
silent, fraction of inert modules). Structure only, no values:

```json
{
  "kind": "corpus.structure",
  "corpus": { "id": "factory", "patches": 75, "path_local": true },
  "engine": { … }, "date": "…",
  "modules_on": { "filter_1": 0.87, "filter_2": 0.31, "distortion": 0.44, "sample": 0.35, … },
  "cooccurrence": { "distortion&compressor": 0.28, … },
  "destinations_modulated": { "filter_1_cutoff": 0.72, "osc_1_wave_frame": 0.51, "osc_1_level": 0.37, … },
  "source_to_destination": { "env_2->filter_1_cutoff": 0.33, "lfo_1->osc_1_wave_frame": 0.28, … },
  "connections_per_patch": { "median": 9, "p90": 24 },
  "value_ranges_used": { "filter_1_cutoff": { "p10": 38.0, "p50": 71.0, "p90": 110.0 } }
}
```

`value_ranges_used` is the one concession to values: a *range*, not a
recommendation, so a mutation can stay where real patches live without
copying anyone's taste. Conditioning by measured class ("patches that
measure as a bass") arrives with the dictionary, which defines the
classes; until then the statistics are unconditioned and say so.

The quality filter for public banks: a patch that clips at the Lite
scenario, renders silent, or has more than half its switched-on modules
inert (the sensitivity sweep's test, per module) is excluded and counted.

### `declared/` — the dictionary

`knowledge/declared/terms/<term>.json`, written by hand:

```json
{
  "kind": "declared.term",
  "term": "reese",
  "says": "Two detuned saws beating slowly, low, thick, often through a low-pass with little resonance.",
  "sources": [ { "title": "Synth Secrets", "author": "Gordon Reid", "part": 12 } ],
  "expects": {
    "descriptors": { "f0_hz": [30, 120], "movement_db": [1.0, 6.0], "brightness_st": [80, 105], "harmonicity": [0.6, 1.0] },
    "structure": { "modules_on": ["osc_1", "osc_2"], "any_of": [["osc_1_unison_voices>=2"], ["osc_2_on=1"]], "destinations_modulated": [] }
  },
  "validation": {
    "status": "unvalidated",
    "patch": null, "measured": null, "engine": null, "date": null,
    "refuted_by": null
  }
}
```

`knowledge validate <term>` builds the patch the entry describes (or takes
the one given), renders it at the term's scenario, and writes `measured`
against `expects`: every descriptor inside its range → `validated`, any
outside → `refuted` with the measured value kept beside the claim. The
claim is never edited to fit; a refuted entry stays refuted until someone
rewrites the claim and validates again. Sources are cited, not copied.

Measured beats declared: a consumer that finds a `declared` expectation
contradicted by a `measured` observation in the same context uses the
measurement and reports the contradiction.

### `weights/` — later

`knowledge/weights/<consumer>.json`: per rule or association a `score`,
the `observations` it rests on, the `share_explored` it was given, the
signal (`measured` only at first: distance-to-target decreased after the
rule was applied). A weight only orders measured-valid options; it never
overrides a measurement. Not designed further here — it has nothing to
weigh before layers 1–3 exist.

## Regeneration

One CLI family, `spinwave-cli knowledge …`, every subcommand idempotent
and self-testing (a measurement of a parameter known inert must come out
zero, and a known-strong one non-zero, before a run writes anything):

| command | writes | notes |
| --- | --- | --- |
| `knowledge measure [--params a,b] [--contexts canonical\|factory\|corpus <id>] [--stale-only]` | `measured/params/*.json` | the sensitivity sweep, stored; `--stale-only` refreshes entries whose fingerprint differs |
| `knowledge corpus <dir> --id <id> [--min-quality]` | `corpus/<id>/*.json` | structure and catalogue; sources stay outside the repo |
| `knowledge validate <term\|--all>` | `declared/terms/*.json` (validation block only) | builds, renders, compares to `expects` |
| `knowledge status` | nothing | counts per store, stale entries, unvalidated terms, contradictions |
| `knowledge weigh` (layer 4) | `weights/*.json` | from logged applications |

`cargo test` carries a test that `knowledge status` on the committed
files reports no entry with a *malformed* header and that the schema
version matches; staleness is allowed in the repo (it is information),
malformation is not.

## How `explore` consumes `measured/`

`ExploreSpec` gains `prior: Prior` — `Live` (today's behaviour),
`Measured` (the store only), `MeasuredThenLive` (the default once the
store exists). With `Measured`, `sensitivity_weights` becomes:

1. For each active parameter, compute its context key on THIS patch.
2. Look up fresh observations with that key. `n ≥ 3` → weight = median
   `distance_db` over them, normalised as today. `1 ≤ n < 3` → the same,
   flagged low-confidence. `n = 0` → under `MeasuredThenLive`, one live
   render as today; under `Measured`, weight 0 and the parameter is
   reported as unknown.
3. The exploration's report lists, per parameter, where its weight came
   from (`store:n=12`, `store:n=1`, `live`, `unknown`), so a variation
   can be read back to its evidence.

Mutation amplitude also reads `corpus/value_ranges_used` when present: a
continuous parameter's triangular draw is clipped to the p10–p90 range
of the corpus, unless the caller says `free_ranges`.

**The gain, measured before the layer counts as delivered:**

- *Cost*: renders per exploration on the 75 factory presets, `Live`
  against `MeasuredThenLive`. Expected: 41 → 1 + (unknown parameters).
- *Agreement*: over the 75 presets, the Spearman rank correlation between
  the live weights and the store's weights for the same parameters. This
  is the number that says whether a context key generalises. Below ~0.7
  the key is too coarse and the layer is not delivered; the note records
  the value either way.
- *"More musical"* cannot be measured without the dictionary (what is a
  musical mutation of a bass, if not one that stays a bass?). Until layer
  3 exists the claim is not made; the two numbers above are what layer 1
  promises.

`suggest` consumes the same observations one step later: with the
`deltas` per quality it can rank alternatives from the store and render
only the top few to confirm — the same `MeasuredThenLive` shape.

## Efficiency, in the order the task gives it

1. Engine recycling: exists, proven.
2. Lite mode with the scenario's length as a parameter (`Scenario.seconds`
   is already a field; the store records the scenario id, and a term can
   ask for a longer one).
3. Conditional descriptors: `describe_without_pitch` exists; the measure
   command asks for the descriptor set each store needs and no more.
4. **Kernel recycling** (`SoundEngine::recycle` for voices): before the
   corpus volume, as the task says — the allocation ceiling is measured.
5. **Content-addressed render cache**: key = SHA-256 of (canonical preset
   JSON, scenario, descriptor set, engine fingerprint); value = the
   descriptors (never the audio: 30 floats, not 100 k), in
   `%LOCALAPPDATA%/spinwave/render-cache/` or `SPINWAVE_RENDER_CACHE`,
   never in the repo. A sensitivity sweep over neighbouring patches hits
   it constantly (the origin render of every parameter's step is the same
   patch).
6. Parallelism as today: a seed per render, the permutation-and-threads
   test as the proof.

Cost is published relative, by alternating commits, as before.

## Order of construction, and what each step must show

1. `measured/` on the canonical contexts and the 75 factory presets,
   `explore` reading it: the two numbers above (renders saved, rank
   agreement).
2. Kernel recycling and the render cache: renders/s before and after,
   alternated; then the sweep of step 1 re-run to show the cache's hit
   rate.
3. `corpus/factory`, structure only; `explore` clipping to its ranges
   (measured: fraction of variants inside the corpus's ranges, and the
   distance distribution before/after). Then a public bank with the
   quality filter, counted.
4. Five or six terms (`sub_bass`, `pluck`, `reese`, `wub`, `pad`,
   `supersaw`), each validated by a built and measured patch; the
   ten-sounds targets become terms with the judge's checks as `expects`.
5. Weights on the measured signal.

## Open questions, to decide when they bite

- The context key's granularity: `context_for`'s conditions are what the
  sweep needed to make a parameter live, not necessarily what makes its
  *effect* comparable (a cutoff's effect depends on the oscillator's
  brightness, which no switch names). The rank-agreement number will say
  whether the key needs the corpus's measured descriptors added to it.
- Where a measurement made in `Lite` mode misleads: polyphony and
  oversampling change little for a sensitivity, but a term's validation
  (a supersaw's width) may need `Faithful`. The scenario id in every entry
  keeps the two apart.

## Corrected on the way (the session's `lfo_N_generator`)

The guide was right and the table has the parameter; the MCP binary the
desktop app was running dated from 2026-09-08, before the Spinwave
namespace entered the table (its `describe_params` reported 794
parameters and no `spinwave_only` count — the current one reports the
extension too). Fixed by rebuilding; the stale binary is renamed beside
it (`spinwave-mcp.stale-2026-09-08.exe`) because the running process held
the file. **The MCP server must be restarted to pick up the new binary.**
So that the guide and the table cannot drift again unnoticed,
`session::guide_tests::every_parameter_the_guide_names_is_in_the_table`
expands every backticked parameter-looking token of `PATCHING.md`
(`osc_N_*` families, `env_1..8` ranges, `a/b` alternatives) and looks it
up in the table or the modulation sources; it passes on the current
guide, which names nothing the table lacks.
