# Spinwave — handoff for an agent arriving cold

State at 2026-09-13. 597 tests green, clippy silent, working tree clean.
The pass of 2026-09-12/13 ("fermer la compatibilité .vital") is
summarised at the end of the golden section and in three notes:
`notes/meta-modulation.md`, `notes/destinations.md`,
`notes/exact-vs-polynomial.md`.
Read `README.md` for what the project *is*; this file is what a review or
a fix pass needs to know before touching anything.

## What it is, in one paragraph

A wavetable synthesizer in Rust: a ground-up rework of Vital (GPLv3), not
a transliteration. The DSP is ported faithfully — same approximations,
same sound — while the architecture is redesigned around a static voice
graph and portable SIMD. It builds as a CLAP/VST3/standalone plugin, and
is currently used **standalone only**; there is no GUI. Roughly 50k lines
of Rust across six crates.

## Workspace map

| crate | lines | what lives there |
| --- | --- | --- |
| `spinwave-poly` | 1.5k | SIMD voice-pair primitives, fast math. One vector holds two stereo voices as `[L0, R0, L1, R1]`. |
| `spinwave-dsp` | 27k | Oscillators, filters, modulators, 11 effects. The bulk of the port. |
| `spinwave-engine` | 8.8k | Voice allocation, modulation matrix, the synth voice kernel, microtuning. |
| `spinwave-params` | 3.5k | The parameter table (Vital's 794 + a `spinwave_only` namespace), the `.vital` preset model, version migrations from 0.2.x. |
| `spinwave-plugin` | 5.3k | nih-plug shell, MIDI, DAW state persistence, live TCP control, the audio-thread garbage chute. |
| `spinwave-control` | 8k | Library with two shells: `spinwave-mcp` (MCP server) and `spinwave-cli`. The **operations** (`ops/`: measure, compare, explain, apply, explore), the `.spinwave` text format, the golden bench, the fuzzer, the judge. |

Spinwave's engine is deliberately **larger** than Vital's: 4 oscillators,
8 envelopes, 12 LFOs, 8 macros, 64 voices, a noise source, two effect send
buses, sample/granular/multisample oscillator engines. Extra parameters
follow Vital's naming and scales. `save_preset` omits Spinwave-only
parameters sitting at their default, so the `.vital` file still opens in
Vital.

## How to run and judge things

```sh
cargo test --workspace                      # 617 tests, ~2 min (the DSP crates are opt-level 3 under dev: 45 min before); the bank order test only under SPINWAVE_FULL_BANK=1
cargo run -p spinwave-plugin --release      # standalone with MIDI
cargo run -p spinwave-control --bin spinwave-cli -- render <preset> out.wav --notes 60,64 --seconds 4 --hold 2
cargo run -p spinwave-control --bin spinwave-cli -- analyze out.wav --start 0.5 --duration 1.0
cargo run -p spinwave-control --bin spinwave-cli -- fuzz --count 200 --wildness full
cargo run -p spinwave-control --bin spinwave-cli -- golden
cargo run -p spinwave-control --bin spinwave-cli -- sensitivity --only filter_1_
cargo run -p spinwave-control --bin spinwave-cli -- to-text in.vital out.spinwave   # and from-text, check
cargo run --release -p spinwave-control --bin spinwave-cli -- measure p.spinwave --lite
cargo run --release -p spinwave-control --bin spinwave-cli -- explain p.spinwave --quality brightness
cargo run --release -p spinwave-control --bin spinwave-cli -- suggest p.spinwave --quality warmth --more
cargo run --release -p spinwave-control --bin spinwave-cli -- apply p.spinwave --set filter_1_cutoff="800 Hz" --goal brightness:less --out q.spinwave
cargo run --release -p spinwave-control --bin spinwave-cli -- explore p.spinwave --count 8 --out variants/
cargo run --release -p spinwave-engine --example bench_voices
cargo run --release -p spinwave-control --example render_cost   # the operations' cost, as ratios
```

The operations are the product surface — see "The operations" below and
`notes/operations-design.md`. Every one prints one JSON document; a
refusal is JSON with a `code`. Always `--release` for anything that
renders more than once.

**Use the CLI, never the MCP server, to judge a change made in the same
session.** The MCP server is a long-lived process holding its own binary
open; a rebuilt engine never reaches it until the session restarts, so
every render through it is silently stale. This has misled us before.

## The golden bench — the thing that matters most

`tools/golden/` compiles **Vital's own DSP core** with MSVC and a stub
`JuceHeader.h`, driving `vital::SoundEngine` directly. No JUCE, no
Projucer: only 2 of 68 files under `src/synthesis` include JuceHeader.
`crates/spinwave-control/src/golden.rs` is the Rust half — it renders the
same case through Spinwave and compares sample for sample.

- **433 cases, 424 match, 9 tracked** in `KNOWN_DIVERGENCES`, at the
  tightened bounds (2026-09-13, after the consolidation pass). Above
  1e-5 and under the bound: the compressor family at ~1e-5,
  `macro_dest_lfo` 4.5e-5 (`notes/exact-vs-polynomial.md`), three
  phaser control-rate destinations from an LFO (1.2e-5 … 5.5e-5). The
  9 tracked, each with its diagnosis in the list: the downsample
  distortion's hold grid, four macro-as-destination-into-a-pitch cases,
  a fast LFO sweep of the unison count, an LFO into the delay time, an
  LFO into the filter fx's blend and resonance, a meta amount from the
  velocity on a mono destination. The five mono audio-rate destinations
  and the diode of the previous list are at the floor. The threshold
  cannot drop to 1e-5 until the compressor is diagnosed: it would sit
  against it.
- **The bank** (75 real `.vital`) is measured against the reference:
  `notes/bank-compare.md`, the table and every class above 1e-4 with
  the case that reproduces it. What that pass fixed in the engine (the
  global sampler's rate, the random seeds, random LFOs per sample, the
  `stereo` source, the FM/RM wiring, a mono plug into a modulator, the
  envelope into voice_transpose, the comb's cutoff, the stereo sample
  cap) is listed there with before/after numbers.
- **Every case's values stay interior to their ranges** (`bounds.rs`,
  run on every golden render; `BOUNDED_BY_DESIGN` lists the six cases
  that measure a bound on purpose). Rule of 2026-09-12: a value against
  a bound measures the clamp, not the case. The first automatic pass
  flagged 28 cases, among them the last gap occupant.
- **Twins.** Most cases added in the pass come in pairs: the modulated
  case and a static twin at the value the modulation should produce.
  Byte-identical references prove the route and the scale on the
  reference's own bytes before Spinwave is judged; a pair whose
  references differ is a reference behaviour (a modulated envelope
  delay is 0 in the trigger block; a chorus delay 2 differs in the
  ulps) and is recorded as such.
- **The primer's echo.** The primer note differs between the engines by
  construction and `skip` hides it, but not its echo: a delay at 250 ms
  with feedback 0.5 brings it back at 0.5^4 and a reverb tail lasts
  seconds. Cases with a delay line or a reverb skip 5 s (`LONG_SKIP`).
  fx_delay's 1.4e-2 and fx_reverb's 6.4e-3 were that.
- The test asserts that a listed case **still diverges**, so fixing one
  fails the suite until it is delisted. That is deliberate.
- Cases are generated by `tools/golden/make_cases.py`, not typed.
- Every report also carries the residual **relative to the reference**, in
  dB. The tolerances stay absolute, which is right for the bound, but the
  absolute number hides something: 1e-3 against a full-scale render is
  -60 dB and inaudible, while the same 1e-3 against a reference at
  -40 dBFS is -20 dB relative and plainly wrong. Read the relative column
  to spot quiet cases passing too easily. It is not a pass/fail input.
- Tolerance is RMS `1e-4`, peak `2e-3 x max(reference_peak, 1)`, since
  2026-09-12 (it was 1e-3 / 2e-2). **Never loosen these to make a case
  pass.** And never accept a floor either: for months the passing cases
  sat at ~4e-4 RMS (−67 dB) with the peak on the saw's edges, explained
  as "0.015 samples of timing between two phase accumulators" — it was
  one wrong call (the oscillator's base frequency through the approximate
  `exp2` where the reference uses the exact one; see
  `notes/audio-rate-audit.md`). With it gone the residuals split in two
  with nothing between 1e-5 and 1e-4: 64 cases at or below 1e-5 (59 at
  float noise, `osc_saw_dry` 4.4e-8), a handful between 1e-4 and 1e-3 —
  each a real small error (the phaser's `tan()` instead of the lookup:
  fixed, to 2e-7; a per-block rate; a ramp with the one-block lead). The
  bounds sit in the gap. A passing case above 1e-5 is telling you
  something.
- **The corpus must render the same bytes twice, in any order**
  (`every_case_renders_the_same_bytes_in_any_order`): the second pass runs
  the cases backwards. A residual that moves with the run order supports
  no conclusion; `mod_random_to_cutoff` did exactly that for as long as
  the random seed counter was process-global, and the test fails on it
  (worst sample 0.7) if the rewind or the seed pin is removed.
- **A case whose connection the engine ignores is refused.** Two mono
  destinations were missing for as long as it was not.

Four rules a case must obey, all learned the hard way:

1. `random_phase 0` is mandatory (it defaults to 1). Two engines cannot
   agree sample for sample on a waveform that starts somewhere random.
2. Every case primes with a **whole first note** that `skip` excludes.
   Vital scoops its very first note up from MIDI note 0 (`last_played_note_`
   starts zeroed); Spinwave deliberately does not — the deviation is
   recorded in `crates/spinwave-engine/src/allocator.rs::note_on`.
   Skipping only the glide is not enough, it leaves a permanent phase
   offset between the two oscillators.
3. **The bench runs with the DC blockers OFF** (`session.set_dc_blockers(false)`).
   Spinwave blocks DC per voice and on the master; the reference wires
   `DcFilter` nowhere despite shipping the class. That slowly decaying
   offset was **99.7 % of the residual on every filter case** and made
   `osc_wave_pulse` look like a wrong waveform. Fourteen cases went green
   from pinning it off alone. The blockers stay in the product — a synth
   that passes DC is worse — they are simply not part of the comparison.
4. **A case with a random source names its seed** (`random_seed 18`).
   Both engines seed every `RandomGenerator` from a process-global
   counter (`next_seed_++`), so the seed a voice's `random_1` holds is a
   fact about construction order: the reference runs one process per
   case and gets 18; Spinwave got 684, and before the counter was rewound
   per render (`RandomGenerator::reset_seed_counter`, in
   `Session::render_samples_probed`) it got whatever the cases before it
   left — the residual moved with the run order. The reference cannot be
   told its seed; `tools/golden/random_seed.py` recovers it from a
   `--probe random_1` curve (least-squares fit of the Perlin gradient
   model, residual 1e-9, then a seed search; it self-tests on a synthetic
   curve first), and the case pins Spinwave's to it. The harness accepts
   the directive and ignores it, byte-identical output verified.

Both halves load **one predefined single-cycle shape**, so the comparison
tests DSP, not wavetable builders.

### Bugs the bench found and we fixed

- The **phaser voice filter** read `blend_transpose`, which the reference
  never reads: its allpass comb ran ~4 octaves too high.
- The **formant filter** was a port of **dead C++** (`SynthFilter::createFilter`
  is called nowhere in Vital), driven by the wrong controls. Proof: the
  low_q and high_q reference renders are bit-identical, so resonance never
  reaches that model.
- The **flanger** clamped its wet mix to `[0, 0.5]`; the reference allows
  `[0, 1]`.
- The **random-amplitudes morph** divided before multiplying in its stage
  index — with 1025 harmonics a stage is not a whole number of SIMD quads.

## The tracked divergences (at RMS 1e-4 / peak 2e-3)

(The current list is 9, all from the bank's second round; see the
bench bullet above and `notes/bank-compare.md`. The history below is
kept as written.)

(The list below was written at 13; the delistings since are in
`KNOWN_DIVERGENCES` with their diagnoses: the three pitch ramps were
the exact conversion's operation order; fx_chorus was the reference's
`ExponentialScale` being `pow(2, x)` and not `exp2(x)`; fx_delay and
fx_reverb were the primer's echo; meta-modulation is ported.)

Each carries its diagnosis in `KNOWN_DIVERGENCES`, not just its first
measurement. The bounds tightened tenfold on 2026-09-12 after the
residual floor went (see the tolerance note above); six cases sitting in
the old gap were promoted with their residuals: three pitch ramps
(`mod_env_to_pitch` 3.6e-4, `_snapped` 4.4e-4, `mod_env_to_tune` 1.2e-4
— hypothesis: the source's one-block lead, visible on a ramp),
`fx_chorus` 1.1e-4 (undiagnosed; ruled out: the exponential-scale
conversion, the delay's filter conversions, the LFO phase arithmetic),
and `mod_lfo_to_distortion_drive` 9.1e-4 (the rate). `osc_unison`
(3.2e-4) was promoted and fixed the same day: the detune ratio through
the polynomial exp2 where the reference is exact — the residual-floor
bug one function over. The phaser's three cases went to float noise once
the coefficient came from the reference's lookup table instead of
`tan()`.

1. **The `mod_*` cases — mostly SOLVED, and it was the bench.**

   Vital's own `ModulationConnectionBank::createConnection` gives a **new**
   connection its bipolar flag from the source's prefix — `lfo`, `random`,
   `stereo`, `pitch` are born bipolar (`kBipolarModulationSourcePrefixes`
   in `synth_types.cpp`). A loaded `.vital` overrides that with its stored
   flag; the harness created connections fresh and never wrote the flag,
   so it inherited the creation default on exactly those sources. Measured
   under the old harness: lfo and random centred, macro / note / velocity /
   **envelope** not. So the constant-source cases were right all along,
   "the poly route is sound" stands, and the envelope cases that still
   fail are **real**.

   It hid because setting the control did nothing either. The proof is the
   one that caught the dead formant filter: the reference rendered
   `mod_lfo_bipolar_low` **byte for byte identically** to
   `mod_lfo_to_cutoff`, two cases differing by exactly that line. Both
   "bipolar" cases were testing the unipolar path, and the whole
   difference between them and their twins was Spinwave alone — which is
   what made Spinwave look wrong.

   Six references changed. `mod_lfo_to_cutoff` went from rms 1.6e-1 to
   **2.8e-4**, with `mod_lfo_to_cutoff_high` and
   `mod_two_sources_one_dest`; all three now pass.
   `mod_two_voices_one_lfo` went 2.5e-1 → 2.5e-3, `mod_random_to_cutoff`
   2.2e-1 → 6.7e-2.

   **The level cases were then two engine bugs, found with the destination
   probe** (`--probe` also prints Spinwave's level offset per lane, read
   from the active slot — the first version read lanes 0/1 and showed a
   retriggered note as silent). First, Spinwave clamped a modulated level
   to `[0, 1]`; the reference does `amp = max(amplitude, 0)` then
   `raw * amp * amp`, with no ceiling (commit 45d3cd1: `mod_lfo_to_level`
   2.2e-1 → 1.7e-3, `mod_env_to_level` 7.3e-2 → 1.25e-2). Second,
   `osc_N_level` is `createPolyModControl(..., audio_rate = true)` like the
   cutoff, so an envelope into it is heard sample by sample, not as a ramp
   once per block. `ModDest::OscLevel` is now audio-rate: the matrix sums
   those connections into a per-oscillator buffer that
   `SynthOscillator::set_amplitude_offset` adds before the square.
   `mod_env_to_level` → 6.99e-4, delisted. `mod_lfo_to_level` sits at
   1.07e-3 against 1e-3, still listed; what remains there is the one-block
   lead below.

   **`mod_env_to_pitch` (2.6e-1) was the same bug on the other per-sample
   inputs.** The reference's oscillator evaluates level, transpose, tune
   and phase per sample (`createPolyModControl(..., audio_rate = true)`
   for all four, `OscillatorModule::init`), reading the transpose buffer
   inside its phase-increment loop; Spinwave ramped the whole pitch once
   per block. Measured before the fix: the sustain pitch matched to
   0.01 st, the divergence was all phase — a 48-semitone chirp over 25 ms
   is a dozen blocks, and the per-block ramp lands the phase somewhere
   else. `ModDest::{OscLevel, OscTranspose, OscTune, OscPhase}` are now
   audio-rate; the matrix sums them into `AudioDestBuffers` and the
   oscillator takes them through `set_audio_offset(AudioOffset::*)`.
   2.6e-1 → 5.3e-4, delisted. Three cases were added for the inputs that
   had none (`mod_env_to_tune` 1.3e-4, `mod_lfo_to_phase` 3.3e-4,
   `mod_env_to_pitch_snapped` 5.4e-4) — the snapped one only after
   replacing the snap: the oscillator rounds to the nearest note THEN
   looks it up in a table (`fillSnapBuffer`, ported as
   `spinwave_poly::utils::SnapBuffer`; ties go down), which the sample
   source's nearest-by-distance `snapTranspose` — what Spinwave used for
   both — does not reproduce (3.1e-1 with the wrong snap).

   **`mod_random_to_cutoff` (6.7e-2) was the seed** — rule 4 above. Same
   draw indices on both sides for both notes (0/2 then 5/7: the second
   note takes slot 1 of the same generator); only the seed differed.
   1.6e-4, delisted. The "unipolar contribution DC offset" reading of it
   was a diagnosis of noise.

   Still failing: the two near misses, `mod_lfo_to_level` (1.07e-3) and
   `mod_two_voices_one_lfo` (2.5e-3).

   **The harness now audits itself.** After wiring, every control the case
   does not name must hold its table default, or the run aborts naming the
   offenders. This is the third initialisation SynthBase does that the
   harness had skipped (the wavetable, `initTriangle()`, now this); the
   audit turns the class of bug into an assertion.

   **Five diagnoses died getting here** — an onset ramp, the poly route,
   the polarity branch, "unipolar and varying", and "the reference centres
   its oscillating sources". The last was carefully measured and still
   wrong, because the instrument was. The cheap rule that would have
   caught it on day one: **when two cases differ by ONE setting, check
   their two references differ.** Identical bytes mean the setting never
   arrived.

   Still open, and now known NOT to be the cause of anything tracked:
   `--probe` found Spinwave one control block (2.9 ms) **ahead** of the
   reference on every source. Two fixes were tried and reverted
   (resolving the matrix before `update_modulators` broke the
   note/velocity cases; a surgical "read before advance" moved the env
   cases by a few percent). `mod_lfo_to_level`, once blamed on it, is at
   1.2e-7 since the base-frequency fix. If it is ever fixed, it is per
   source category, not globally — the reference applies constants at
   once and advances the modulators at a fixed point.

   **All the oscillator near misses were one line** — see the tolerance
   note above and `notes/audio-rate-audit.md` §2: `mod_lfo_to_level`,
   `mod_two_voices_one_lfo`, `osc_morph_inharmonic_stretch`,
   `osc_warp_sync`, `osc_warp_pulse_width`, `osc_warp_quantize` and
   `mod_lfo_to_distortion_drive` all fell to float noise when the
   oscillator's base frequency went through the exact conversion.

2. `filter_diode_high_q` (1.9e-2) / `filter_diode_low_q` (2.3e-3) — state
   persisting between notes.
3. `fx_delay` (1.4e-2) / `fx_reverb` (6.4e-3) — diverge during the skipped
   window, then decay. The delay's three filter frequencies used the
   approximate conversion where the reference is exact; fixing that did
   not move it, so the cause is elsewhere.
4. Four mono effect destinations the reference evaluates per sample and
   Spinwave per block: `mod_lfo_to_distortion_filter_cutoff` (5.3e-3),
   `mod_lfo_to_phaser_center` (5.8e-3), `mod_lfo_to_filter_fx_cutoff`
   (2.2e-3), `mod_lfo_to_eq_low_cutoff` (1.0e-3). The full table of the
   reference's 31 audio-rate controls, what each side does, and what
   making these four per sample would take (half a day, plumbing; the
   consumers are mostly ready) is in `notes/audio-rate-audit.md` §1.

## Known and unfixed, outside the bench

- **Transport-synced sources are unmeasured**: neither harness runs a
  transport (`correctToTime` is never called in main.cpp, the session's
  transport stays stopped), so an LFO or random in sync type "sync"
  renders as 0 on both sides. Two bank presets use it (Ah Eh Ee Oh,
  corrupted_…). Adding a transport to both harnesses is the next step
  for that class.
- The read-parameter audit (`patch::read_audit`) lists what the engine
  does not implement, with reasons: `osc_N_smooth_interpolation` and the
  keytracked LFO rates. A name leaves that list when the control is
  wired (the formant filter's five controls and the LFOs' `sync_type`
  were on it for a day). It first listed the `sub_*` controls as a
  missing sub oscillator — wrong: the sub left Vital's voice graph in
  0.5.0 and the migration converts it to `osc_3`, which `migrate.rs`
  ports; measured on 75 real presets (`Documents/Vital`, 0.6.1–1.0.0),
  none carries a `sub_*` key.
- **A real preset bank exists on this machine**: `~/Documents/Vital/`
  (Factory + Afro, 75 `.vital`, versions 0.6.1 to 1.0.0) — the corpus for
  the ten-sounds targets, the format's round trip and any prevalence
  question. The five `presets/packs/` are the project's own.
- **All 75 load with zero ignored connection and all 75 render**
  (`spinwave-cli bank ~/Documents/Vital --render`: peak and loudness
  per preset, the wavetable creator's warnings — none across the bank —
  and any render failure). Before the pass 62 were refused for 356
  connections; meta-modulation (`notes/meta-modulation.md`) took 179,
  then 46 destination families (`notes/destinations.md`, a case each
  with its residual), a macro as a destination, the keytracked LFO rate,
  and the master `volume` — which had been routed to the per-voice
  amplitude with scale 1 and silenced "E4 One Note Metallophone".
- **What the 75 sound like against the reference IS measured** since
  the pass of 2026-09-13: the harness loads a `.vital` (`--preset`, the
  reference's own migration copied verbatim into
  `reference_migration.inc` by `extract_migration.py`, the wavetable
  creator compiled from `common/wavetable`, the sample's base64), and
  `tools/golden/bank_compare.py` / `bank_bisect.py` do the rest. See
  `notes/bank-compare.md`.
- **No GUI.** For standalone use this is the real gap, and it is a large
  enough chantier to scope before building.

## The operations (`spinwave_control::ops`)

The layer that makes the synth answerable: `measure`, `compare`,
`explain` / `suggest`, `apply`, `explore` / `interpolate`. Design, with
every descriptor's definition and reference, the distance and its
measured scale, and what was settled: `notes/operations-design.md`.
The rules, each learned the hard way and each with a test:

- **Self-test before numbers.** A render that is silent while a source is
  on (`code: silent`), non-finite, too short, or from a patch that did
  not load cleanly is refused; every result carries what the check saw.
- **A seed in, the seed out.** `SoundEngine::reseed` covers every
  generator; each render's seed is derived from the operation's seed and
  the render's index (`ops::render_seed`), never from the process or the
  thread. `the_same_operation_gives_the_same_bytes_in_any_order_on_any_thread_count`
  is the test.
- **Engines are recycled, not rebuilt** (`SoundEngine::recycle`: voices
  rebuilt, chains reset in place, rings clearing only what was written),
  and `a_recycled_engine_renders_the_same_bytes_as_a_fresh_one` says so
  on light-after-heavy, heavy-after-light, heavy-after-heavy. Building
  the full engine cost 120 ms; a Lite render is ~25 ms now.
- **The distance is perceptual**, and `distance_scale` proves it: two
  renders 16 % apart sample by sample from a random phase read 0.002 dB;
  saw against square reads 17.9 at nominal and 17.9 twenty dB down.
- **Throughput was measured and the default follows it**: four workers
  (`ops::DEFAULT_THREADS`, `SPINWAVE_THREADS` overrides). More were
  slower. Kernel reuse was measured (2026-09-13) at 0.3 ms of a 12 ms
  Lite render and is not the lever; the knowledge base's effect cache
  is (a repeated measurement renders nothing).
- **Explain and suggest are honest about switches**: suggest tries
  continuous moves only unless `--switches`; explore holds indexed
  parameters unless `--switch-indexed P`. A topology change is a jump.

The CLI commands and the seven MCP tools (`measure_patch`,
`compare_patches`, `explain_patch`, `suggest_moves`, `apply_diff`,
`explore_patch`, `interpolate_patches`) are thin: the same structs, the
same JSON. The ten-sounds runner should call these, not the judge's
analysis alone, when it runs.

## The `.spinwave` text preset

A readable, editable, diffable view of a `.vital`, in strict bijection with
it: TOML, only what departs from the default, values in their real unit,
modules as tables with each modulation under its destination. Design and
the six decisions behind it: `notes/preset-text-format.md`. Code:
`crates/spinwave-control/src/text_preset/`. Examples with commentary:
`presets/text/`. CLI: `to-text`, `from-text`, `check` (the report as JSON,
built for a program to fix its own file).

Three properties the tests hold, over the five packs and 24 fuzzed patches:
`.vital -> text -> .vital` equal value for value; `text -> .vital -> text`
equal byte for byte; **the two `.vital`s render bit-identically**. The
render comparison self-tests: a pack rendering below -60 dBFS is refused,
and the fuzz test requires at least half its patches audible so that
silence cannot pass for agreement.

**Exact or refused.** A value is written with the fewest digits from which
the inverse recovers the same f32, else as `raw:<engine>`. The digit count
is intrinsic (a half-ulp of 0.7 is 7e-7 dB), so where two exact spellings
exist the shorter wins: cutoffs come out in semitones on the packs, levels
as the knob value with dB in the comment (`level = 0.7   # -6.2 dB`).
Times stay in ms/s and are long only when authored in engine units
(`0.5476` is `"89.91946 ms"`); a typed `"90 ms"` stays. Measured: `raw:`
fires on 0.04 % of continuous values on fuzzed patches, never on the
packs.

**Vital creates lfo/random/stereo/pitch connections bipolar by default**
(`kBipolarModulationSourcePrefixes`), so `bipolar` is always written for
those sources even when false: the one place a default is not omitted.

**Fixed in its own pass:** the table named `osc_N_destination` with
Vital's fourteen-entry list (index 5 = "chorus") while the engine routes 5
-> bus A. Vital's nine effect entries are dead in Vital itself (popup of
five, arrows modulo five, DSP tests five), so the buses break no real
preset; the table now names the engine's seven destinations. Still open in
the table: `style` runs to 9 with five names, and a few other indexed
ranges outrun their name lists — the writer falls back to `raw:` there.

## The ten-sounds test (built, dry-run without a model, not yet run)

Since 2026-09-13 the harness runs without credentials: `run.py
--ceiling` judges the hand-written ceiling patch of every target
(`tools/ten-sounds/ceiling/`, 24 of 24 pass), `run.py --stub ceiling`
and `--stub init` run the three conditions with a stub model (every
path exercised, no API). The pre-flight in the protocol note found that
`sub_bass` is passed by the init patch — a target every condition passes
for free — and two descriptor limits (the ops movement measure does not
find a 0.25 Hz wobble; the aliasing measure reads unison as aliasing).

Does the format and the measure loop help a model turn a sentence into a
sound? Three conditions on the same twelve targets — raw `.vital` one
shot, `.spinwave` one shot, `.spinwave` with a five-round
render-analyse-correct loop — so that B−A is what the format adds and C−B
what measuring adds. Protocol: `notes/ten-sounds-protocol.md`. Judge:
`crates/spinwave-control/src/judge.rs` (`spinwave-cli judge <patch>
--target <id>`, `spinwave-cli targets`), every target with a known
positive and negative in its tests, refusing silent renders. Runner:
`tools/ten-sounds/run.py` (needs `anthropic` and credentials; ~$20 for
the full grid at Opus 5). **The thresholds never reach the model**, and
the person who wrote the judge is not a valid subject.

## The knowledge base (`knowledge/`, `spinwave-cli knowledge …`)

A memory of what works, in three stores that never mix, designed and
measured in `notes/knowledge-base-design.md` (read it first; the
reading list is `notes/knowledge-base-resources.md`):

- `measured/params/<name>.json` — what each parameter does in THIS
  engine: one observation per (patch, scenario), the band distance of
  a quarter-range step and the signed change of nine qualities, with
  the CONTEXT it was made in (`knowledge::context_key`: every switch,
  model, engine, routing, connection and sync mode that gates the
  parameter, on that patch) and the engine fingerprint (`build.rs`:
  sources + data + toolchain). 962 parameters, 19 433 observations
  (the canonical contexts and the 75 factory presets). `explore` reads
  it (`prior`, default `measured_then_live`): 80 % of the renders
  saved, rank agreement with live weights 0.72 (0.75 without the five
  NaN presets), each preset left out of its own prior.
- `corpus/factory/` — the STRUCTURE of the factory bank (modules on,
  co-occurrence, destinations modulated, connections, p10/p50/p90 per
  parameter); no preset in the repo. `explore` holds its draws to
  those ranges (61 % → 91 % inside, `free_ranges` lifts it).
- `declared/terms/<term>.json` — the dictionary: twenty-one terms validated
  by `knowledge validate [--wav DIR]` (builds the entry's patch,
  measures it against `expects`; a refuted claim keeps its
  measurement; `--wav` keeps the renders to listen to). The growl
  entry changed the pitch detector (several windows, low-passed, YIN's
  global-minimum fallback): `f0_hz` is the period, and on a sound whose
  harmonics carry an LFO's phase modulation it sits a few percent from
  the sub's spectral peak.

Staleness: an entry whose fingerprint is not the running engine's is
stale, counted by `knowledge status` (watch the fraction), regenerated
by `knowledge measure --stale-only`. Every engine fix stales the store;
regenerate before trusting `explore` again. The effect cache
(`%LOCALAPPDATA%/spinwave/render-cache`) makes a re-measurement of
unchanged patches free. The MCP server announces its engine
fingerprint in `describe_params`; `spinwave-cli fingerprint` prints the
repo's; a difference means a stale server binary (it cost a turn once).

Layer 5, the weights, is deliberately not built: nothing applies a
rule yet.

## The probe: asking a divergence WHERE

`vital_golden case out.raw --probe lfo_1` and `spinwave-cli golden --case
X --probe lfo_1` each write the control-rate value of a modulation source,
one row per block; the Spinwave side also prints the offset that reached
filter 1's cutoff and oscillator 1's level, per lane (control-rate offset
plus the first sample of the audio-rate sum, from the ACTIVE slot's
lanes). Diff the two curves and the shape names the mechanism:
an exponential is a one-pole smoother whose coefficient you can read off,
a linear ramp per block is a buffer interpolation, a step one block late
is a reset firing at the wrong time. It is a diagnostic — no committed
case uses it, and it does not touch the audio (verified: the harness
renders byte-identical output with it compiled in).

**The probe self-tests before it reports.** `env_1` runs on every voice
and is non-zero through any sounding note, so the harness checks that
witness channel and REFUSES to write probe output if it stayed flat. An
instrument that returns a plausible but wrong curve is the worst
outcome, and this one did exactly that once:

**the trap, which cost an hour.** On the reference side, read the
**status output** (`engine.getStatusOutput(name)`), never
`getModulationSource(name)->buffer`. Poly sources live in the voice graph,
which the voice handler CLONES per aggregate voice: the Output the source
map hands out belongs to the template processor, which no voice ever
writes. It yields a plausible-looking curve that is not the one driving
the audio. What caught it was probing `env_1` — always running, always
audible — and reading a flat zero straight through a sounding note.

## The parameter sensitivity sweep

`spinwave-cli sensitivity [--only <substring>]` moves every parameter in
the table and checks the sound moves too. It exists because the formant
filter's five controls were wired to nothing for months and no test could
see it: every test was written by someone who already believed the
parameter worked.

It reproduces those five from a cold start. **It is not yet a clean
signal**: 393 of 1313 parameters report inert, and that is a worklist, not
393 bugs — which is why it exits 0. Almost all of the remainder is missing
CONTEXT, and that context is the whole difficulty: an LFO's rate does
nothing until the LFO is connected AND its sync type is free-running
rather than tempo-synced; a granular control does nothing with no sample
loaded; a decay does nothing at full sustain. The count has come down
964 -> 595 -> 401 -> 393 as rules were added, and the module docs list
what is left and what each family needs. Do not read a name in the output
as a bug until its context rule exists.

## The fuzzer

`spinwave-cli fuzz` builds a patch per seed from the whole parameter table,
renders it, and judges it (`crates/spinwave-control/src/fuzz.rs`). It
crashed the engine on its first 40 patches: the Formant model on the
Filter FX panicked because bus chains run oversampled and exceed its fixed
scratch buffer. 1200 patches now clean.

Its criteria took three rounds of triage, and the lesson generalises: a
random patch may legitimately be loud, ugly, and ring forever. **Only a
NaN (or a panic) is fatal.** Telling self-oscillation from a long reverb
needs the *slope across the tail*, not a comparison against the peak — a
resonant Filter FX past threshold drones forever at a saturator-fixed
amplitude, and that is correct behaviour.

## Presets

Four of the five packs under `presets/packs/` never set `osc_N_wave_frame`,
so every oscillator sat at frame 0 of the factory table — **a pure sine**.
`reese-classic` described itself as "two wide-beating saw layers" in its own
comments and was three sines. The factory table morphs sin/triangle/saw/
square/pulse over 256 frames, so saw = 128, square = 191. Fixed, along with
levels (three packs clipped). **Do not trust any level or brightness target
written before 2026-09-09.**

## Environment notes for this machine

Things that look absent and are not:

- **MSVC 14.44** build tools are installed under
  `AppData/Local/Microsoft/VisualStudio/BuildTools` — a Program Files
  search misses them. Ninja and CMake too (scoop).
- **Python 3.14** is at `AppData/Local/Programs/Python/Python314`; the bare
  `python` command hits the Microsoft Store stub and looks like an absence.
- **Vital is installed** (standalone, VST3, CLAP, VST2), but the standalone
  is the GUI build and ignores `--headless`, writing nothing. Hence the
  golden harness driving the DSP core directly.
- Golden build gotchas: MSVC needs `-D__SSE2__=1`; exclude
  `synthesis/effects_engine/` (a separate product with its own SoundEngine);
  the harness must repeat `SynthBase`'s startup (load a wavetable and
  `initTriangle()` the LFOs) or the engine renders **silence**.

## Measured CPU

`cargo run --release -p spinwave-engine --example bench_voices`. Worst
case: 4 osc x 7 unison, both filters, every envelope and LFO.

| voices | realtime | p99 | worst | onset | tail |
| --- | --- | --- | --- | --- | --- |
| 1 | 18.4x | 7% | 13% | 9% | 13% |
| 4 | 9.2x | 16% | 19% | 14% | 19% |
| 8 | 4.5x | 30% | 36% | 25% | 34% |
| 16 | 2.2x | 70% | 76% | 54% | 63% |
| 32 | 1.1x | 144% | 166% | 110% | 149% |
| 64 | 0.5x | 292% | 316% | 195% | 313% |

**Read the worst-block column, not the realtime one.** The mean is the
reassuring number and the wrong one: a host hands the engine one block and
a deadline, and missing it once is an audible click however comfortable
the average was. At 16 voices "2.2x realtime" sounds fine while the worst
block is already at 76% of its deadline; at 32, "1.1x" is a 166% overrun.
Each percentage is one block's cost as a share of ITS OWN deadline. In a
DAW the budget is shared with every other track, so aim well under two
thirds.

Two columns answer questions worth asking that came back **negative**:

* `onset` is the block every voice starts in. It is consistently CHEAPER
  than the worst sustained block, so voice allocation is not a spike.
* `tail` is the worst block of a four-second release. **Nothing in this
  engine sets flush-to-zero** (verified: no FTZ/DAZ anywhere), so
  denormals were a live worry — but the tail costs no more than a held
  note (34% against 36% at 8 voices), because voices are killed once
  silent and never linger in the denormal range. Re-check if voice
  killing ever changes.

Default polyphony is **8** (worst block 36%), which is both Vital's
default and the right one. The 64 is a maximum, reachable only on light
patches; audio-rate modulation into the cutoff costs nothing measurable.

The consolidation pass of 2026-09-13 (random LFOs rendered per sample,
the FM modulator copied out per slot, the SMP sampler keeping its raw
output, the bent midi recomputed after the modulators) cost about
**4 %**, by the same alternation on the same day (old = 2ca70b0, three
interleaved rounds, this machine): 8 voices 3.1–3.2× → 3.0–3.1× real
time, 16 voices p99 102–104 % → 106–108 %. The `worst` column stayed
noise-dominated (233 % once on the new build, 89 % once on the old).

The four per-sample oscillator inputs (2026-09-12) cost about **5 %**:
measured by alternating the commit before and after on the same machine,
the real-time factor at 16 voices went 1.5× → 1.4× and p99 90–96 % →
97 %. The `worst` column is noise-dominated on this machine (119 % once on
the OLD commit); the absolute numbers that day were all ~1.5× the ones
above, so compare only interleaved runs, never against this table.
