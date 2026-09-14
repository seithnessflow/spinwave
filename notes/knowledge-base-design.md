# Knowledge base for sound design — the design (2026-09-13)

A memory of what works, in three stores that never mix: what this engine
measurably does (`measured/`), what real patches are shaped like
(`corpus/`), what the trade says (`declared/`), and later a fourth of
weights over the first three. This note is the structuring decision the
task asked to stop at: the schema of the stores, the format of an entry
with its provenance, the regeneration command, and how `explore`
consumes `measured/`. Layers 1 to 4 are implemented and measured (their
sections below); layer 5 is deliberately not.

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

Every derived entry carries two fingerprints, computed by a `build.rs`
in `spinwave-control` and exposed as constants:

- `engine`: SHA-256 over the sorted contents of `crates/spinwave-poly/src`,
  `spinwave-dsp/src`, `spinwave-engine/src`, `spinwave-params/src` —
  code AND data: the parameter table (a scale that changes moves every
  measurement without a line of DSP) and the factory wavetable
  (`spinwave-dsp`'s built-in frames) live in those trees and are hashed
  with them — plus the **toolchain**: `rustc -vV` (version, host, commit)
  and the build target triple. Two machines with different compilers
  can differ in the ulps, and this project has paid to learn that ulps
  count; the price is a fingerprint that stales more often, accepted.
- `descriptors`: SHA-256 over `spinwave-control/src/{analysis,ops/descriptors,ops/distance}.rs`,
  kept **apart** so a new descriptor does not stale the engine's renders
  (see the cache below). Each stored descriptor value also carries the
  version of the descriptor set that computed it.

Not the git commit: a commit that touches only a note must not stale
ten thousand renders, and a dirty tree must still be able to measure.
The git commit (and a dirty flag) is stored beside it as provenance,
never as the validity key.

An entry whose engine fingerprint differs from the running engine's is
**stale**: readable, reported as such by every consumer, never used for
a weight. `knowledge status` prints the stale **proportion per store**
(the number to watch: the base degrades silently with every engine fix
otherwise, and one notices only when `explore` has gone dumb again);
`knowledge measure --stale-only` regenerates exactly those entries.

The MCP server announces the engine fingerprint of the binary it runs
in every `describe_params` and `patching_guide` response, and the CLI
`spinwave-cli fingerprint` prints the repo's; a client that sees them
differ warns before its first `set_params`. The 2026-09-08 binary that
cost a turn in the sound-design session is the case: the handoff had the
rule written down and it was not enough — a warning that fires beats a
rule that is reread.

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
2. Look up fresh observations with that key. The weight is the median
   `distance_db` over them, normalised as today, **shrunk by its
   evidence**: `w · n / (n + k)` with `k = 5`, so an observation count of
   3 weighs 3/8 of what the same median would at n = 50 (50/55) — a
   stored weight on three patches must not rank as confidently as one
   on fifty. `n = 0` → under `MeasuredThenLive`, one live render as
   today; under `Measured`, weight 0 and the parameter is reported as
   unknown. `k` is a constant to tune against the rank-agreement
   number, not a rule.
3. The exploration's report lists, per parameter, where its weight came
   from and on how much (`store:n=12`, `store:n=1`, `live`, `unknown`),
   so a variation can be read back to its evidence.

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

## Layer 1, delivered and measured (2026-09-13, evening)

Implemented as designed: `crates/spinwave-control/build.rs` (the two
fingerprints, git provenance), `knowledge.rs` (`Store`, `context_key`,
`measure`, `status`, `agreement`), `spinwave-cli knowledge measure |
status | agreement` and `spinwave-cli fingerprint`, `ExploreSpec.prior`
with `weight_sources` in the report, and the MCP server announcing its
engine fingerprint in `describe_params` and at the head of the guide.
The store is under `knowledge/measured/params/`, 962 parameters, 19 433
observations: the canonical context of every parameter the sweep can
activate (944; the 40 sample/granular slot controls fail silent with no
sample loaded and are listed in `measure`'s `failed`), plus the 75
factory presets (18 489 observations, 6 m 37 s — after the global
sample was memoized like the slot samples: one preset had cost 3 m 48 s
rebuilding its 42 s sample's pyramid per render, 8.5 s after).

The two numbers, `knowledge agreement --patches ~/Documents/Vital`,
each preset's own observations left out of its prior:

| | value |
| --- | --- |
| renders, live weights | 18 564 |
| renders, `MeasuredThenLive` | 3 731 (**80 % saved**) |
| parameters the store knew in the patch's context | 80.2 % |
| Spearman(live, prior), mean over 74 | **0.717** (median 0.740) |
| the same without the five presets whose output is the NaN clamp | **0.747** (median 0.750, min 0.519) |
| under 0.5 | the four NaN presets only (0.18–0.20); Metal Head n/a |

The bar was ~0.7: cleared, narrowly on the full set and clearly once the
presets that render a constant are set aside (their live weights are
noise). The context key generalises well enough to spare four renders in
five; the lowest honest agreements (Disrupt 0.52, Staggered Phrases
0.52, Memory Leak 0.54) are the presets whose sound depends on things
the key does not name — the wavetable's content, a sample — which is the
open question below, now with a number to move.

One finding of the measurement itself: `delay_frequency`,
`chorus_frequency` and every `lfo_N_frequency` read 0 in their canonical
context because their `*_sync` switch is tempo-synced there; the key
now carries the switch, so a preset with a free-running delay is a
different context, not a contradiction.

## Layers 2, 3 and 4, delivered and measured (2026-09-13, night)

**The effect cache** (`knowledge::EffectCache`): a patch, a parameter
moved to a value, a scenario, the engine fingerprint and the descriptors'
fingerprint give the same effect every time, so the effect is kept
under the SHA-256 of all of those, outside the repo
(`%LOCALAPPDATA%/spinwave/render-cache`, `SPINWAVE_RENDER_CACHE`, `off`
to disable). The value is the effect — thirty floats — never the audio.
The design above said a new descriptor would stale no render; that
needs the audio kept (100 k floats an entry), so it is not done: a
descriptor change recomputes, and the note is corrected here.
Measured on the 75 factory presets: cold 18 316 renders in 5 m 44 s;
warm **0 renders, 18 489 hits, 5.2 s**.

**Kernel recycling is not where the time is.** `examples/alloc_cost.rs`
times the pieces: a Lite render 12 ms, of which the engine's `recycle`
2.5–3.5 ms — the voice allocator's rebuild 0.3 ms (one kernel), the
three effect chains' reset 0.6 ms, the remaining ~1.5 ms not located;
building an engine from nothing 13 ms. The lever the operations note
named ("reusing kernels") would save a third of a millisecond per
render. Not built; the cache above and the SMP sample's memoization
(one preset cost 3 m 48 s per measurement rebuilding a 42 s sample's
pyramid per render, 8.5 s after) were the real gains.

**The corpus** (`knowledge/corpus/factory/`): structure and catalogue of
the 75 factory presets, structure only. The quality filter (clips above
1 % or silent at the Lite scenario) excludes exactly the five presets
whose output is the NaN clamp. What the factory bank is shaped like:
osc_1 on in 96 %, filter_1 86 %, distortion 81 %, compressor 80 %,
osc_2 76 %, reverb 73 %; the most modulated destinations
`filter_1_cutoff` (70 %), `osc_1_level` (57 %), `reverb_dry_wet` (54 %),
`osc_2_level` (53 %), `osc_1_spectral_morph_amount` (51 %),
`osc_1_wave_frame` (50 %); the commonest connections
`lfo_1→osc_1_wave_frame` (27 %), `macro_4→reverb_dry_wet` (21 %),
`lfo_1→filter_1_cutoff` (20 %); 8–41 connections per patch (p10–p90),
16 at the median. No public bank is on this machine; the filter and the
command are ready for one (`knowledge corpus --patches DIR --id NAME`).

`explore` holds a continuous parameter's draw to the corpus's p10..p90
(`free_ranges` lifts it; a value already outside is not pulled in).
Measured on twelve factory presets, eight variants each: the share of
moved parameters landing inside the corpus's ranges **61 % → 91 %**, the
mean distance from the origin 5.0 → 3.9 dB, 11 renders per exploration
where the live weights alone cost 200–290.

**The dictionary** (`knowledge/declared/terms/`, `knowledge validate`):
six terms — sub_bass, pluck, pad (Reid, high trust), supersaw, reese,
wub (genre tutorials, low trust) — each with a claim in our words, its
sources cited, `expects` in the engine's descriptors and structure, the
patch the claim describes, and the verdict `validate` wrote after
building, rendering (Faithful, the entry's note) and measuring it. All
six are `validated`; what validating taught:

- A patch that "puts two saws through the low-pass" must say
  `osc_2_destination: 0`: the table's default routes oscillator 2 to
  filter 2 (Vital's default too), which was off, so the second saw
  bypassed the filter and the reese measured a centroid of 3.4 kHz
  under a cutoff of 370 Hz — refuted on the first pass, the patch
  corrected (not the claim), validated on the second.
- `movement_db` over the whole render counts the release and the
  silence, so any decaying sound "moves" by tens of dB; the dictionary
  measures `movement_held_db` and `brightness_movement_held_st` over
  the held note only, and `movement_rate_hz` (the strongest periodic
  rate the analysis finds).
- The wub's claim of "a large level movement" was refuted: with the
  clipper and the compressor the level moves 0.3 dB over the held note
  while the centroid swings by 5.7 semitones at 2.69 Hz (the LFO's
  2.67). The entry now says brightness, records the refutation, and
  expects a movement rate in 2–8 Hz — which is what tells a wub from a
  supersaw (5.3 semitones of brightness movement too, from its unison
  beating, at 0.34 Hz).

**Dictionary session, 2026-09-14 (growl, stab).** Two entries added
and validated (`knowledge validate --wav DIR` now keeps each render to
listen to). The stab (Reid's brass "blip" envelope with the sustain
cut; medium trust) validated on the second pass: its first patch fell
20 dB in 0.10 s — a stab is longer than a pluck, the amp decay went to
0.66 s and the filter envelope keeps a sustain of 0.4. The growl
(neuro recipe: osc 1 FM'd by the sine sub, dual-notch resonance swept
at 4 Hz, a slower LFO on the frame, hard clip) exposed the pitch
detector: YIN on the loudest 100 ms with an absolute threshold declared
it unvoiced — a swept resonance rings louder than the fundamental and
is periodic at no lag (CMND minimum 0.35 raw, at the right lag). The
detector (`ops::descriptors::yin`, now also behind `analyze`'s pitch)
reads up to five windows over the loud part on a copy low-passed at
1 kHz, takes YIN's global minimum below 0.5 when nothing crosses 0.15,
and reports the median when half the windows agree within a semitone.
Effects on the other entries, all still validated: the pad now reads
its real fundamental (131 Hz, the octave layer) instead of the note;
the pluck is voiced (248 Hz, 5 % under the note: the closing filter's
phase); the growl's period is 90.3 Hz where its sub's peak is 87.5 (the
FM'd harmonics carry the LFO's phase modulation) — `f0_hz` is the
period, not the strongest partial, and the entries' ranges are wide
enough for that. Cost: five windows with the direct O(W·lag) loop made
`analyze` 61 → 122 ms on a 2.5 s file; the difference function by FFT
(e₀ + e_τ − 2r(τ), one planner per call, equal to the loop within 1e-3)
brings it to 62 ms, and the searches never ask for the pitch
(`analyze_with(…, false)` — the detector was also being run twice per
measurement, once in `analyze` and once in `describe`; once now). The
descriptors' fingerprint changed, so the effect cache recomputes on the
next measure (5 m 44 s cold, measured before).

Second pair, same session: **vocal_formant** (Reid's voice parts, high
trust) — a saw into the formant bank (model 5, a/i/u/o pad, an LFO at
0.5 Hz on `formant_x`, a vibrato) was refuted on harshness (−5.9 dB):
a saw falls 6 dB per octave where a glottal pulse falls 12, so the
bank's upper formants at 2.6–3.2 kHz got too much; a 12 dB low-pass at
1.3 kHz in series stands in for the tilt (−25.5 dB, centroid 650 Hz,
4.1 semitones of movement at the morph). **fm_bell** (Chowning 1973,
ratio 1:1.4; high trust) was refuted on inharmonicity (0.002): 7:5 is
rational, every partial sits on a grid five times finer than the
note's harmonics and the render is periodic at the note's fifth
(104.7 Hz under C5) — what the ear calls inharmonic is a missing
fundamental, not partials off a grid. The claim now says so and
expects the period at most a quarter of the note, harmonic against it;
the judge's `fm_bell` target still says "inharmonic" in words, which
is fine for a listener and wrong for the descriptors.

Third batch, on my own: **brass** (Reid, parts 24–25: the overshooting
filter envelope; attack 40 ms, brightness moving 4.2 semitones as it
settles), **organ** (Reid, tonewheel parts: sines at 16', 8', 5⅓' —
attack 5 ms, level steady to 0.14 dB over the held note, flatness 0),
**kick** (Reid, part 34: a sine whose pitch falls from ~160 Hz to the
body within 20 ms — refuted once because the entry's note was released
at 50 ms and the release, not the decay, ended it; held through its
decay it reads f0 55 Hz, −20 dB at 0.16 s), **lead** and
**noise_riser** (the judge's targets, low trust: centroid 1.55 kHz and
harshness +2 dB for the lead; flatness 0.66 and 27 semitones of
brightness rise for the riser).

Fourth batch: **strings**, **snare**, **hihat**, **flute** (Reid, high
trust), **acid**, **e_piano** (tutorials, low). Three refutations kept:
the strings' "steady level" — five copies detuned by a few cents beat
by 2.5 dB over the held note, and the beating is the ensemble (the
entry expects 0.5–5 dB now); the acid's "instant attack" — the level
peaks 0.2 s in, when the closing resonance sweeps past the low
harmonics, so `attack_s` (time to peak) reads the squelch, not the
onset (expects 50–500 ms now); the flute's first patch put a sine
through a band-pass three octaves above it and the breath noise won
(peak −40 dB, harmonicity 0.54) — a low-pass, and the noise at 0.03.
Twenty-one entries, all validated; every one has its render under
`knowledge validate --wav`.

**The first consumer (2026-09-14):** the MCP tool `dictionary_term`
lists the entries, serves one with its verdict (and `stale` against the
running engine), and with `apply: true` makes the entry's patch the
session's. What it does not do yet is close the loop a weight needs —
"did starting from this entry bring the judge's target closer than
starting from nothing" — which is the ten-sounds test's question, so
the weights still wait for it.

**Layer 5, the weights, is not built.** Nothing applies a rule yet:
`explore` reads measurements, `suggest` renders, the dictionary is
consumed by nobody so far (the copilot and the judge are its readers to
come). A weight adjusts by whether applying a rule reduced the distance
to a target; with no applications there is nothing to weigh, and a store
of untouched weights would arrive too early — the end criterion of the
task. It starts the day a consumer applies dictionary entries.

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
   JSON, scenario, **engine** fingerprint) — the engine's alone; value =
   a map of descriptor name → (value, descriptor-set version), never the
   audio (30 floats, not 100 k). A new descriptor is computed on the
   cached entry's render (re-rendered once, the same bytes by the
   engine's determinism) and added to the map; the other descriptors
   stay. Adding a descriptor therefore stales nothing. Lives in
   `%LOCALAPPDATA%/spinwave/render-cache/` or `SPINWAVE_RENDER_CACHE`,
   never in the repo. A sensitivity sweep over neighbouring patches hits
   it constantly (the origin render of every parameter's step is the
   same patch); it is the largest throughput gain of the layer.
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

- The resources the dictionary reads from are listed in
  `notes/knowledge-base-resources.md`, with what each yields and how
  far to trust it; none enters the repo.
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
