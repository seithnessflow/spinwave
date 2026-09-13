# The ten-sounds test

Does the `.spinwave` format, and the render-measure-correct loop built on
it, help a model turn a sentence into a sound? Written 2026-09-12; the
judge is `crates/spinwave-control/src/judge.rs`, the runner
`tools/ten-sounds/run.py`. **Not yet run** — this machine has no API
credentials, and the person who built the judge cannot be the subject.

## What is being measured, and what is not

A model asked for a sub bass that produces one proves it can design a sub
bass. That is interesting but it is not the question. Vital's presets are
public, so the model already knows the parameter names; a good result in
isolation could be the prior alone. The question is whether the format
and the measurement add anything, and only a comparison answers it.

Three conditions, same targets, same model, same sampling:

| | Condition | What the difference means |
| --- | --- | --- |
| A | raw `.vital` JSON, one shot, no render | the model's prior alone |
| B | `.spinwave`, one shot, no render | **B − A: what the format adds** |
| C | `.spinwave`, then render → analyse → correct, up to 5 rounds | **C − B: what measuring adds** |

Conditions B and C are given the format's design notes and the three
committed examples (`presets/text/`), which is what any agent would read.
Condition C is fed, after each attempt, either the load report (errors
with lines and suggestions) or the **analysis** of its patch rendered —
levels, spectrum, envelope, pitch, width, movement. It is never fed the
judge's checks or thresholds. A model that is told "the centroid must be
under 150 Hz" is being tested on hitting a number; a model that is told
"a deep fundamental you feel more than hear" and shown a centroid of
900 Hz is being tested on understanding.

Three samples per target per condition. The API has no seed; each call's
`request_id` and token usage are logged instead, which is the
reproducibility handle available.

## The targets

Ten sounds, each carrying a criterion a program can measure, and two
controls that are worth as much as the ten. The description is what the
model receives; the criterion lives only in the judge.

| id | description the model sees | what the judge measures |
| --- | --- | --- |
| `sub_bass` | A clean sub bass: a deep fundamental you feel more than hear, with nothing bright on top. | rendered at C2: spectral centroid < 150 Hz, rolloff < 2 kHz |
| `pluck` | A short plucked sound that dies away quickly after each note. | level 300 ms after the peak ≤ −20 dB |
| `pad` | A slow, wide pad that swells in and keeps its body in mono. | attack > 500 ms, stereo width > 0.3, mono sum within 6 dB of stereo |
| `fm_bell` | A bell made with FM: inharmonic, fast strike, ringing decay. | harmonicity < 0.5, attack < 50 ms, still > −40 dB one second after the peak, gone by the end |
| `lead` | A cutting lead whose energy sits in the upper mids. | the 1–4 kHz band within 3 dB of the loudest band |
| `filter_sweep_up` | A filter that opens over a held note, dark to bright. | centroid at the end > 2× centroid at the start |
| `velocity_dark_soft` | Soft notes darker, hard notes brighter. | two renders: centroid(vel 0.3) < 0.7 × centroid(vel 1.0) |
| `keytrack_bright_high` | Notes high up noticeably brighter than notes low down. | two renders three octaves apart: centroid ratio > 4.5 (more than the pitch alone gives) |
| `noise_riser` | Filtered noise, not a tone, sweeping upward over a few seconds. | harmonicity < 0.3, flatness > 0.2, centroid end > 1.5× start |
| `tempo_wobble` | A timbre cycling once every eight beats at 120 BPM. | a modulation rate within 8 % of 0.25 Hz, strength > 0.3 |
| `reconstruct` | A prose description of `lush-pad.vital`, written with no numbers. | a scale-free distance between the two analyses < 1.0; the parameter diff is reported |
| `edit` | "Make this patch darker, with a softer attack. Change as little as possible." on `neuro-trinity` | centroid ratio < 0.85, attack longer by > 20 ms, **≤ 8 parameters changed** |

The two controls are chosen from presets the model is **not** shown as
examples. The reconstruct control has a ground truth, so it yields a
distance rather than a verdict. The edit control is probably the most
useful thing in practice, and a model that rewrites the whole patch to
make it a little darker is a bad sign even when the sound goes the right
way.

Every target has a known positive and a known negative in the judge's
tests — a synthetic patch that meets it and one that does not — so a
criterion that could not be met, or could not fail, would not have got
in. Two things those tests turned up: the autocorrelation pitch detector
returns nothing on a pure low sine, so the sub bass is judged by where its
energy sits rather than by a detected pitch; and an LFO is tempo-synced
**by default**, so its frequency knob is ignored until `sync` is changed —
a fact a model writing a free-running LFO will run into.

The judge refuses before it scores: a render peaking under −60 dBFS is an
error, not a verdict, because two silent renders would satisfy half these
criteria for the wrong reason.

## What to read off the results

Per condition, the runner reports:

- **pass rate** overall and per target;
- **first-load rate** — the share of first attempts that load at all;
- **mean rounds** to the final attempt (condition C);
- **the ranking of load-report error codes**, which is the most actionable
  number of the whole test: each frequent code names an alias to add, a
  unit to tolerate, or a sentence in the format's documentation to fix.

Three samples per cell separate a tendency from noise; they do not
separate 60 % from 70 %. Read the differences between conditions, not the
absolute rates.

## Two things the numbers cannot do

The judge does not know whether a sound is musical. Keep a **blind
listening pass** on the final renders, without knowing which condition
each came from, and record it beside the numbers.

And the criteria must stay out of the prompt. The descriptions above are
the whole of what the model sees about a target; the judge's tests assert
they carry no `dB`, `Hz` or `ms`.

## Running it

```sh
cargo build --release -p spinwave-control --bin spinwave-cli
pip install anthropic          # and ANTHROPIC_API_KEY, or `ant auth login`
python tools/ten-sounds/run.py --samples 3
```

Rough cost: 12 targets × 3 conditions × 3 samples is 108 cells; condition C
runs up to five calls per cell. Around 250 calls of a few thousand input
tokens each, at Opus 5 pricing, on the order of twenty dollars. Results
land in `tools/ten-sounds/results/<stamp>/`: every patch the model wrote,
`runs.jsonl` with each attempt's load report and checks, and
`summary.json`.

## Dry run without a model (2026-09-13)

Nothing above had run: no credentials, and the person who wrote the
judge cannot be the subject. Three things now exist that need no model,
so the harness is checked before the first dollar is spent.

**The ceiling.** `tools/ten-sounds/make_ceiling.py` writes a hand-written
patch per target in both formats (`tools/ten-sounds/ceiling/<id>.vital`
and `.spinwave`): the judge's own known positives for the ten sounds,
the truth itself for `reconstruct` (distance 0), and for `edit` a six-
parameter darkening of neuro-trinity — the hard clip's drive, the EQ's
high shelf, the noise, the filter, the attack. `run.py --ceiling` judges
them: **24 of 24 pass** (every target, both formats). That is the number
every condition is read against; a model at 60 % is 60 % of what a
person did in a minute. The first edit ceiling, lowering the cutoff by
two octaves, did not darken the patch at all (centroid ratio 1.08): the
brightness of that patch is the clipper and the noise, not the filter —
a fact the edit target will test a model on too.

**The stub model.** `run.py --stub ceiling` runs the three conditions
with a model that answers every prompt with the ceiling patch and says
DONE on the second round of C; `--stub init` answers with the init patch
and never says DONE. The first exercises the whole loop end to end (36
cells, pass rate 1.0, mean rounds 2.0 in C, no error codes); the second
exercises every failure path (5 rounds in C, fail, the summary still
written). Records carry `stub` where the API's request id and usage
would be, so a summary with `stub` in it cannot be mistaken for a
result.

**Pre-flight of the targets against the descriptors' known limits.**

- `sub_bass` **is passed by the init patch** (`--stub init`: A, B and C
  all PASS on it). The engine's default is a sine at the played note
  with nothing on top; the criterion (centroid < 150 Hz, rolloff < 2 kHz
  at C2) is met by doing nothing. A target every condition passes for
  free measures nothing — the same rule as a case whose value rests
  against a bound. It needs either a criterion the default does not meet
  (a body: loudness above a floor, or a second oscillator an octave up
  for weight, or a note-off release the default lacks) or to be dropped
  from the count. Left as is, flagged: changing a criterion is a
  protocol decision.
- The ops descriptors' `movement_rates_hz` does not find the
  `tempo_wobble` ceiling's 0.25 Hz (it reports 0.5 / 1.5 Hz on the
  default 2.5 s render and 1.1 Hz on a 6 s one): a period longer than
  half the analysis window is invisible to it. The judge's own
  `mod_rates_hz` (analysis.rs) does find it, and that is what condition
  C is fed; if the harness ever moves to the ops descriptors, the wobble
  needs an 8 s render or a different measure.
- The aliasing measure reads the `pad` ceiling at 0.033 (unison detune
  beating, not aliasing) and the `noise_riser` at 0.0; a model told its
  pad "aliases" would be misled. Condition C is not fed the aliasing
  measure; keep it that way unless the measure learns unison.
- YIN (`f0_hz`) finds the sub's 65.4 Hz at C2 — the older
  autocorrelation detector did not, which is why the sub is judged by
  where its energy sits. Either detector is fine for the judge as it
  stands.
- `reconstruct`'s ceiling is the truth: distance 0 by construction. A
  distance below 1.0 is the pass; what a model's typical distance is has
  no ceiling to compare against other than 0.

Running it for real is unchanged: `pip install anthropic`, credentials,
`python tools/ten-sounds/run.py --samples 3` — about $20 at Opus 5 — and
the API is still off until told otherwise.
