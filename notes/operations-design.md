# Operations in `spinwave-control` — design note (2026-09-12)

Five primitives that make the synth answerable: measure, compare,
explain, apply, explore. Library first; CLI, MCP, GUI and any hosted API
are clients. Nothing here needs a model; a model calls these.

What already exists and is reused, not rewritten: `analysis::analyze`
(levels, six bands, centroid, rolloff, envelope, autocorrelation pitch,
width, movement, texture), `sensitivity::context_for` (what a parameter
needs switched on to be audible), `judge::analysis_distance` (a first
descriptor distance), `fuzz::Rng` (seeded xorshift), `text_preset`
(exact-or-refused parameter spelling, per-unit), `Session::render_samples`
(deterministic: seed counter rewound per render, random LFOs reseedable).

## 1. Descriptors

All computed on the render of a *scenario* (notes, velocities, hold, total
length, tempo). The mono sum `(L+R)/2` unless stated. Frames: 2048-sample
Hann, hop 512, at the render rate. Every descriptor carries its unit in
the field name. Reference given where a definition is taken from the
literature; the rest is arithmetic.

| descriptor | definition | unit |
|---|---|---|
| `peak_dbfs` | max \|x\| over both channels | dBFS |
| `rms_dbfs` | RMS of the mono sum over the whole render | dBFS |
| `loudness_lufs` | integrated loudness, ITU-R BS.1770-4: K-weighting (the two specified biquads), 400 ms blocks with 75 % overlap, absolute gate −70 LKFS, relative gate −10 LU | LUFS |
| `dc_offset_dbfs` | mean of the mono sum, in dB (−∞ → floored at −120) | dBFS |
| `clipping` | samples with \|x\| ≥ 0.999 that sit in a run of ≥ 3 consecutive such samples (a flat top, not a single peak); count and fraction of all samples | count, ratio |
| `bands_dbfs[8]` | mean power per band, bands at octave edges 40·2ⁿ Hz (40–80, 80–160, …, 5120–10240, 10240–20000) | dBFS |
| `centroid_hz`, `centroid_trajectory_hz[]` | power-weighted mean frequency (Peeters, *A large set of audio features for sound description*, CUIDADO 2004, §spectral centroid), whole render and per 50 ms | Hz |
| `rolloff_hz` | frequency below which 85 % of the power sits (same source) | Hz |
| `attack_seconds` | RMS envelope (5 ms hop) rising from 10 % to 90 % of its maximum — the MPEG-7 *LogAttackTime* thresholds, reported linear | s |
| `decay_20db_seconds` | time from the envelope maximum to the first point 20 dB below it; `None` if it never gets there within the hold | s |
| `tail_dbfs` | RMS of the last 10 % of the render | dBFS |
| `f0_hz` | YIN (de Cheveigné & Kawahara, *YIN, a fundamental frequency estimator for speech and music*, JASA 2002): cumulative-mean-normalised difference, threshold 0.15, parabolic refinement, on the loudest 100 ms; `None` when the CMND minimum exceeds the threshold (unvoiced) | Hz |
| `harmonicity` | power at peaks within ±3 % of `n·f0` (n ≤ 40) over total power, on the same window; `None` without `f0` | ratio |
| `inharmonicity` | Peeters 2004 §inharmonicity: power-weighted mean of \|f_k − n_k·f0\| / f0 over the 20 strongest peaks, n_k the nearest harmonic index; `None` without `f0` | ratio (0 = harmonic) |
| `spectral_flatness` | geometric / arithmetic mean of the power spectrum (existing) | ratio |
| `odd_even_ratio` | odd-harmonic power over even (existing) | ratio |
| `stereo_width` | 1 − \|corr(L, R)\| (existing) | ratio |
| `mono_compatibility_db` | RMS(mono sum) − RMS(stereo), the judge's `mono_collapse_db` | dB |
| `aliasing_proxy` | power at spectral peaks above 4 kHz that are NOT within ±3 % of a harmonic of `f0`, over total power above 4 kHz; `None` without `f0`. **A proxy, labelled as such**: it also counts legitimate inharmonic content (FM, noise). The honest aliasing test — the same note at two pitches, partials moving the wrong way — costs two renders and is left to `explain` as a named quality, not to every measurement | ratio |
| `movement` | existing: dominant modulation rates 0.2–16 Hz, RMS trajectory per 50 ms, onset density | Hz, dBFS, /s |

Not included, and why: no LUFS short-term / momentary (nothing here is
long enough); no MFCCs (a distance basis, not a readable descriptor; the
band spectrogram below does that job); no "warmth"/"punch" words as
numbers — those are `Quality` names in `explain`, mapped to bands and
descriptors, never descriptors themselves.

## 2. The perceptual distance

**Multi-resolution log-band spectrogram distance.** For each of two STFT
resolutions (4096/1024 and 1024/256), the power spectrogram is folded
into 24 bands, half-octave from 40 Hz to 16 kHz (an approximation of the
ERB scale coarse enough to be stable on short renders; Moore & Glasberg,
*Suggested formulae for calculating auditory-filter bandwidths and
excitation patterns*, JASA 1983, for why log-spaced bands are the right
axis), each band expressed in dB with a floor at −90 dBFS. The distance is
the mean over resolutions of the mean absolute dB difference over (frame,
band). This is the log-STFT-magnitude term of the multi-resolution STFT
loss (Yamamoto, Song & Kim, *Parallel WaveGAN*, ICASSP 2020) on a log
frequency axis instead of linear bins, and a log-spectral distance (Gray
& Markel, *Distance measures for speech processing*, IEEE TASSP 1976) in
its per-frame form.

Why not RMS on the waveform: a phase shift the ear cannot hear (a
different random phase, a one-block lead) gives a waveform error as large
as a timbre change. The bench needs waveform RMS because it asks "is this
the same DSP"; these operations ask "does this sound the same", and dB per
band per frame is the coarsest measure that still says *where*.

The two renders share the scenario, so they are aligned by construction;
no time warping. The result is the total plus two decompositions: per
band (time-averaged) and per time (band-averaged), so the answer to "how
different" comes with "in the low mids, during the attack".

Scale, measured on the way (it goes in the note once measured): the
distance between a patch and itself with a different random seed (the
noise floor), between a patch and its −1 dB copy, between a saw and a
square at the same pitch. Thresholds for "the same" and "different" are
read off those, not chosen.

## 3. The five operations

Rust signatures; every result is `Serialize`, every error is a code.

```rust
pub struct Scenario { pub notes: Vec<NoteSpec>, pub seconds: f32, pub bpm: f32, pub mode: RenderMode }
pub enum RenderMode {
    /// Polyphony as the patch says, oversampling as the patch says, stereo.
    Faithful,
    /// Polyphony 1, oversampling 1×, one 0.6 s note, stereo kept (width is a descriptor).
    Lite,
}
pub struct Measurement { pub seed: u64, pub scenario: Scenario, pub descriptors: Descriptors,
                         pub self_test: SelfTest, pub cost_ms: f32 }

pub fn measure(preset: &Preset, scenario: &Scenario, seed: u64) -> Result<Measurement, OpError>;

pub struct Comparison { pub parameters: Vec<ParamChange>,  // name, from, to, unit, delta — from the text format's spelling
                        pub distance: Distance,             // total_db, per_band: [(band_hz, db)], per_time: [(t_s, db)]
                        pub a: Descriptors, pub b: Descriptors }
pub fn compare(a: &Preset, b: &Preset, scenario: &Scenario, seed: u64) -> Result<Comparison, OpError>;

pub enum Quality { Brightness, Harshness, Warmth, Width, Attack, Sustain, Noise, Movement, Level,
                   Band(u8) }      // each maps to a band set and/or a descriptor, documented in code
pub struct Contribution { pub name: String, pub neutral_value: f32, pub effect_db: f32, pub direction: Sign }
pub struct Explanation { pub quality: Quality, pub measured: f32, pub ranked: Vec<Contribution>, pub skipped_inert: usize, pub renders: usize }
pub fn explain(preset: &Preset, scenario: &Scenario, quality: Quality, budget: Budget) -> Result<Explanation, OpError>;

pub enum Direction { More, Less }
pub struct Move { pub name: String, pub from: f32, pub to: f32, pub unit: String, pub effect: f32, pub side_effects_db: f32 }
pub fn suggest(preset: &Preset, scenario: &Scenario, quality: Quality, direction: Direction, budget: Budget) -> Result<Vec<Move>, OpError>;

pub struct Applied { pub preset: Preset, pub report: LoadReport,        // corrections and errors from the format
                     pub before: Descriptors, pub after: Descriptors, pub distance: Distance,
                     pub goal: Option<GoalCheck> }                        // measured vs requested direction
pub fn apply(preset: &Preset, diff: &Diff, scenario: &Scenario, goal: Option<(Quality, Direction)>, seed: u64) -> Result<Applied, OpError>;

pub struct ExploreSpec { pub count: usize, pub amplitude: f32, pub seed: u64, pub budget: Budget }
pub struct Variant { pub preset: Preset, pub diff: Vec<ParamChange>, pub distance_from_origin: f32, pub descriptors: Descriptors }
pub fn explore(preset: &Preset, scenario: &Scenario, spec: &ExploreSpec) -> Result<Vec<Variant>, OpError>;
pub fn interpolate(a: &Preset, b: &Preset, t: &[f32]) -> Result<Vec<Preset>, OpError>;
```

`Diff` is a list of `ParamChange` or a `.spinwave` fragment (a TOML table
with only the keys to change), read by the same parser with the same
report — no second spelling of a parameter.

`Budget { max_renders, max_seconds }`: an operation that would exceed it
returns what it has with `truncated: true`, never blocks unbounded.

**Explain, concretely.** For the patch's *active* parameters (the module
is on, the source is connected — the inverse of `context_for`), each is
neutralised (set to its table default, or to the value that removes its
effect: a modulation amount to 0, an effect to off) and the patch
re-rendered in `Lite`; the contribution is the change of the quality's
measure (a band-set level in dB, or a descriptor) between the original
and the neutralised render. Ranked by \|effect\|. Parameters the
sensitivity sweep's context rules declare inert in this patch are skipped
and counted. `suggest` is the same loop with each parameter moved a
measured step in each direction instead of neutralised, ranked by effect
in the requested direction, with the band-distance outside the target
quality reported as `side_effects_db`.

## 4. Explore: mutation and interpolation

**Mutation.** Only parameters of active modules (same rule as explain).
Amplitude per parameter = `spec.amplitude × range × weight`, the weight
being that parameter's sensitivity measured on *this* patch by `explain`
(one Lite pass, cached in the result) normalised so the most sensitive
parameter gets 1 and the inert get 0 — a parameter that does nothing is
not mutated, one that does a lot is mutated gently. Continuous values are
drawn from a triangular distribution centred on the current value and
clamped to the range. Booleans flip with probability `amplitude`. The
fuzzer's lesson stands: uniform over the whole table is noise.

**Indexed parameters** (filter model, distortion type, wave source, LFO
generator…) do not interpolate. Decision: **frozen in interpolation,
switched by threshold** — `interpolate(a, b, t)` keeps `a`'s index for
`t < 0.5` and takes `b`'s from `t ≥ 0.5`; booleans likewise. In
`explore`, an indexed parameter changes with probability `amplitude / 2`
to a uniformly chosen other index. Both rules are in the result's
documentation string, and a variant whose index changed says so in its
diff (it is a `ParamChange` like any other).

**Modulation connections** interpolate by amount: the union of both
patches' connections, an absent connection counting as amount 0, and a
connection whose amount interpolates to \|amount\| < 1e-3 is dropped from
the result.

Every variant is validated through the format's reader before it is
returned: a mutation the reader would refuse is regenerated (up to 8
tries, then the variant is dropped and counted).

## 5. Transverse

**Self-test before numbers.** `measure` refuses (`OpError::Silent`) when
the peak is below −60 dBFS while any oscillator is on and any envelope
can open; refuses (`NotFinite`) on NaN/inf; refuses (`Empty`) when the
scenario renders fewer than two analysis frames; refuses (`Rejected`)
when the preset does not load cleanly — the format's report is attached.
Every result carries `self_test: { peak_dbfs, frames, rejected: [] }` so
a caller can see the check happened.

**Determinism.** `seed` is an input of every rendering operation and is
echoed in the result. It reseeds the random LFOs and the exploration RNG;
the process-global generator counter is rewound per render (already).
Test: the same operation, same seed, run on a permuted list of patches,
byte-identical descriptors — the rule the bench learned.

**Error codes.** `OpError { Silent, NotFinite, Empty, Rejected(LoadError),
UnknownParameter(name), NotInterpolable(name), BudgetExceeded, … }`,
serialised with a `code` string and the offending name, so an agent can
fix its own call.

**Budget.** Lite renders are what explain / suggest / explore spend.
Measured on the way and published as a *relative* figure (the CPU rule
from the review): Lite cost as a fraction of Faithful, and renders per
second per core, interleaved runs only. Parallelism with
`std::thread::scope` over `available_parallelism()`, each thread its own
`Session` (no shared engine state; the seed counter is per process and
rewound per render — a data race on it would break determinism, so the
counter becomes thread-local or per-render, measured by the permutation
test run multi-threaded).

**No audio thread.** Everything here builds its own `SoundEngine`
offline. Nothing is reachable from the plugin's process callback.

## 6. Exposure

CLI: `spinwave-cli measure <patch> [--lite] [--seed N] [--json]`,
`compare <a> <b>`, `explain <patch> --quality Q`, `suggest <patch>
--quality Q --more|--less`, `apply <patch> <diff.spinwave|--set name=v…>
[--goal Q:more]`, `explore <patch> --count N --amplitude A --seed S --out
DIR`, `interpolate <a> <b> --steps N --out DIR`. JSON on `--json`, the
same struct the MCP returns. MCP: one tool per operation, same names,
same JSON; the existing `analyze` / `compare` tools become thin calls
into `measure` / `compare`.

## 7. The two maintenance points

**Tolerance floor, measured (2026-09-12, 73 passing cases).** The RMS
residuals are bimodal: 64 cases at ≤ 1e-5 (59 of them ≤ 1e-6 — float
noise), 9 cases between 1e-4 and 1e-3, **none between 1e-5 and 1e-4**.
The nine: `mod_lfo_to_distortion_drive` 9.1e-4 (rate, known),
`fx_phaser` 4.5e-4 / `filter_phaser_high_q` 1.9e-4 / `filter_phaser_low_q`
1.6e-4 (the phaser filter's `tan()` instead of the reference's coefficient
lookup — listed as known and unfixed in the handoff, now measurable),
`mod_env_to_pitch_snapped` 4.4e-4 / `mod_env_to_pitch` 3.6e-4 /
`mod_env_to_tune` 1.2e-4 (a pitch ramp with the one-block lead is the
likely remainder), `osc_unison` 3.2e-4, `fx_chorus` 1.1e-4. Recommendation:
a threshold at **1e-4 RMS** would split the corpus exactly at the gap and
promote those nine to tracked; the peak bound should follow to
`2e-3 × max(peak, 1)` (the worst passing peak ratio is 1.2e-2 on the
snapped pitch case, all nine are above 4e-4, every other case is below
1e-4). Not tightened in this pass, as asked; the numbers are in the
handoff.

**Static audit of read parameters.** Not a source scan: the preset
reader (`patch.rs::Reader`) gets a recording mode; a test applies a preset
that sets *every* table parameter to a non-default value, records the set
of names the reader was asked for, and diffs it against the table. A name
never asked for is a parameter the engine cannot receive — the formant
and distortion-filter class — with no context false positives, because
being *read* does not depend on being audible. Names that are legitimately
never read (GUI-only, e.g. `osc_N_view_2d`) go on an allowlist with a
reason each, like the sweep's `EXPECTED_INERT`.

## Decisions to ratify

1. Descriptor set and definitions above (LUFS per BS.1770-4 added; aliasing as a labelled proxy).
2. Distance = multi-resolution half-octave band spectrogram, mean |dB|, with per-band and per-time decompositions.
3. Signatures of the five operations; `Diff` accepts a `.spinwave` fragment or a change list, nothing else.
4. Explain = neutralise-and-remeasure on active parameters, weight-cached for explore.
5. Indexed parameters: frozen and switched at t = 0.5 in interpolation; switched with probability amplitude/2 in exploration.
6. Lite mode = polyphony 1, oversampling 1×, one 0.6 s note.
7. Tolerance: measured, recommendation 1e-4 / 2e-3, not applied.
8. Read-parameter audit as a recording reader, not a source scan.

## What the implementation settled, and measured (2026-09-12)

Ratified with six amendments, all in: the distance weights every cell by
its level relative to the loudest cell of either render (a silent tail
weighs nothing); two more scale measurements; `switch_indexed` explicit
and zero by default, switches included; the seed derived per render from
the operation's seed and the render's index, never thread-local;
loudness normalisation as an explicit `DistanceOptions` flag; the phaser
fixed before the tightening.

**The distance's scale, measured** (`ops::tests::distance_scale`,
Faithful scenario, in dB): identical renders 0.000; the same patch with
`random_phase` on at two seeds, waveforms 16.5 % apart sample by sample,
**0.002** — the control that says this is a perceptual distance and not
a waveform error in disguise; a −1 dB copy 0.83 (a pure gain reads
1.00; the filter's saturation eats the rest); two seeds of a 30 % random
LFO on the cutoff 2.5; saw against square **17.87**, and 17.86 twenty dB
down — the floor holds. Read: under ~0.1 is "the same", a few dB is a
knob moved, ten and more is a different sound.

**Cost, as ratios** (`examples/render_cost.rs`, one machine, interleaved
runs): a Lite render plus descriptors is 0.73 of a Faithful one; the
render alone is ~25 ms of which the block loop is 6 ms and the
descriptors 10 ms; the rest is rebuilding voice kernels and reading the
patch. Building an engine was 120 ms with the full pool; the pool is now
the patch's polyphony, and engines are recycled between renders (voices
rebuilt, chains reset in place, rings clearing only what was written) —
bit-identical to fresh, by test. Parallel throughput of an exploration:
43 renders/s on one thread, **94 on four**, 80 on eight, 52 on sixteen,
31 on thirty-two. The block loop scales (6× at 16 threads); what does not
is rebuilding kernels — allocation and first-touch page faults through
one memory system. Default four workers (`SPINWAVE_THREADS` overrides);
the next lever is reusing kernels the way the chains are reused.

**Determinism, tested**: the same patches measured forward and backward
give byte-identical descriptor JSON; the same exploration on one thread
and on four gives byte-identical variants.

**The read-parameter audit found more than the design expected.** Beyond
the formant filter: `lfo_N_sync_type` (the envelope / loop-point / sync
modes — the DSP had them all, the reader never asked), `random_N_sync_type`,
`sample_pan`, `sample_transpose_quantize`, `filter_fx_keytrack`, and the
filter-input priority reversed against the reference. All wired, two
golden cases added for the first two families (7.9e-7 and 5.3e-8). Not
implemented and now listed as findings: the reference's **sub
oscillator** (eight controls, dropped by every preset that uses it),
`osc_N_smooth_interpolation`, and the keytracked LFO rates.

**Tolerances tightened** to RMS 1e-4 / peak 2e-3 after the phaser fix,
at the measured gap; six cases promoted to tracked with their residuals
(three pitch ramps, the unison, the chorus, the drive's rate).

Not done, named: kernel reuse (above); the `Lite` descriptors still
compute YIN on a 4096 window, the largest single cost; `explain` on a
quality mapped to bands cannot yet exclude the aliasing proxy's
limitation (a real aliasing test needs two pitches).
