# `.spinwave` — the text preset format

Design document, 2026-09-12. Status: **implemented** —
`crates/spinwave-control/src/text_preset/`, `spinwave-cli to-text |
from-text | check`, examples under `presets/text/`. Ratified with two
amendments (decision 2: unit-less quadratic values in dB; decision 4:
percentage of range confirmed, unit input accepted on linear destinations,
conversions reported). Six decisions below, each with the alternative it
beat; the section at the end records where the implementation had to
depart from the text above, and what it measured.

## What it is for

A patch a human or an LLM can read, edit and diff, in strict bijection with
the `.vital` JSON the engine already loads. The `.vital` stays the source of
truth; this is a **view** of it, not a second model. Every key in the text
maps to something the engine reads; nothing exists only in the text.

Three principles, given:

1. Only what departs from the default is written. A module at its defaults
   does not appear. Load-then-save adds no lines.
2. Values in their real unit, never internal: `attack = "90 ms"`, not
   `0.5476`. Indexed parameters by name: `model = "ladder"`, not `2`.
3. Grouped by module, with each modulation written next to the destination
   it modulates.

## A patch, to fix ideas

```toml
# Reese Classic — two wide-beating saw layers over a sub
format = 1
synth_version = "1.0.7"
requires = "vital"

[preset]
name = "Reese Classic"
author = "spinwave"
style = "Bass"

[osc_1]
on = true
level = 0.7                  # -6.2 dB: the gain is the square of the knob
frame = 128                  # saw
unison_voices = 5
unison_detune = "12.0 st"
stereo_spread = "80%"

[osc_2]
on = true
level = 0.31                 # -10.2 dB
frame = 191                  # square
transpose = "-12 st"

[filter_1]
on = true
model = "ladder"
style = "24 dB"
cutoff = "69 st"             # 440 Hz; the shorter exact form wins
resonance = "50%"
drive = "6.0 dB"

[[filter_1.mod]]
to = "cutoff"
from = "lfo_1"
amount = "+70%"              # of 128 st: reaches 440 Hz → 77.8 kHz, clamped by the filter
rate = "audio"               # envelope or LFO into a cutoff runs per sample

[[filter_1.mod]]
to = "cutoff"
from = "env_2"
amount = "+40%"
bipolar = true
power = 2.0
rate = "audio"

[env_1]
attack = "4 ms"
decay = "320 ms"
sustain = 0.78               # unit-less in the table, so bare
release = "90 ms"

[lfo_1]
frequency = "2 Hz"
sync = "tempo"
tempo = "1/4"
shape = "triangle"           # a factory shape by name; drawn shapes give their points

[compressor]
on = true
mix = "100%"
```

## Decision 1 — TOML, not YAML

**TOML.** Four reasons, in order of weight.

- **No indentation semantics.** The one error class an LLM and a tired human
  both produce is a wrong indent; in YAML that silently changes meaning, in
  TOML it changes nothing. A `[filter_1]` header cannot be misplaced by
  whitespace.
- **No implicit typing.** YAML reads `no` as `false`, `1e3` as a float and
  `8000` as an int while `8kHz` is a string — three ways for a value to be
  parsed as something other than what was written. TOML has one string
  syntax, one number syntax, and units are always in a string. The parser
  decides what a string means, and the report says what it decided.
- **The structure fits.** A module is a table, a modulated destination is an
  array of tables under it (`[[filter_1.mod]]`). Nesting never exceeds two
  levels, which is where TOML is at its best and YAML's advantage does not
  apply.
- **The Rust side is solid.** `toml_edit` (already in the lockfile, TOML 1.1)
  parses with line numbers, which the error report needs. `serde_yaml` is
  deprecated and its successors are less settled.

One-line-per-change follows from a **canonical serializer** — stable key
order, stable formatting — rather than from preserving the user's own
formatting. Text → `.vital` → text is therefore stable "up to
normalisation", which is what was asked. Comments are regenerated, not
preserved: a comment the serializer writes is derived from the data (the
reach of a modulation, the semitone value behind a Hz), never authored.

File extension: **`.spinwave`**. It says what it is; the first line
`# spinwave preset (TOML)` tells an editor how to highlight it. Rejected:
`.swp` (vim swap files), `.spinwave.toml` (two extensions, and every tool
that splits on the last dot sees a generic TOML file).

## Decision 2 — units and precision, per scale

The parameter table already carries the display law (`skew`, `post_offset`,
`display_multiply`, `display_units`, `string_lookup`), so the conversion is
derived from it, not re-declared. What the format decides is the unit
spelling and the precision.

| Scale | Written as | Example | Notes |
| --- | --- | --- | --- |
| Indexed | the option's name, lowercase | `model = "ladder"` | booleans as `true`/`false`; names come from `string_lookup`, aliases accepted (see tolerance) |
| Linear, semitones | Hz for cutoffs, st otherwise | `cutoff = "440 Hz"`, `transpose = "-12 st"` | cutoff's engine unit is MIDI semitones; Hz is the readable view, the semitone value is written as a trailing comment |
| Linear, dB / % / plain | as the table displays it | `drive = "6.0 dB"`, `sustain = 0.78               # unit-less in the table, so bare` | |
| Quadratic, with a unit | the squared value in that unit | `unison_detune = "16.0 st"` | the square IS the musical quantity |
| Quadratic, unit-less (levels) | **the knob value, bare, or dB — whichever exact spelling is shorter** | `level = 0.7  # -6.2 dB` | A bare number is the knob value Vital shows and stores; dB always carries its unit, so nothing is ambiguous. The dB form is accepted on input and kept when it is the shorter exact one (`"-6.2 dB"` typed stays `"-6.2 dB"`). See the measurement below for why dB alone was the wrong default |
| Cubic / Quartic | seconds, written as ms below 1 s | `attack = "90 ms"`, `release = "1.6 s"` | |
| Exponential | Hz, or a tempo fraction when synced | `frequency = "2 Hz"`, `frequency = "1/4"` | `display_invert` params (delay times) in seconds |
| SquareRoot | as the table displays it | `volume = "-6.0 dB"` | the C++ `unskew` is identity here, reproduced |

**Precision: exact by construction, or refused.** The serializer writes a
value with the fewest digits such that parsing it back and inverting the
scale returns the **same f32 bit pattern**. It starts at three significant
digits, adds digits until the inverse is exact, and if nine digits still
miss (an inverse `x^(1/4)` in f32 can), it writes the engine value
verbatim with an explicit marker: `attack = "raw:0.5476"`. That marker is
the honest exception: it is rare (the round-trip test on the packs and on
fuzzed patches counts how rare and the count is committed), it round-trips
by definition, and it never guesses. Inverses run in f64 and round once to
f32, which is what makes exactness reachable.

So the "rounding of writing" the round-trip property tolerates is **zero**:
either the display form recovers the value bit for bit, or the raw form is
used.

## Decision 3 — bulky material

Three kinds, three treatments.

- **Drawn LFO shapes and remap curves stay in the text.** They are a list of
  points and powers, rarely more than a dozen, and reading them is the
  point of a text format. A shape equal to a factory shape is written by
  name (`shape = "triangle"`) and regenerated on load; the test asserts the
  regenerated points equal the original exactly, else the points are
  written out. Drawn: `shape = { points = [[0.0, 1.0], [0.5, 0.0], [1.0, 1.0]], powers = [0, 0, 0], smooth = false }`.
- **Wavetables, samples, SFZ zones and the Spinwave material block go to a
  content-addressed sidecar**, `name.spinwave.d/<sha256>.json`, holding the
  JSON value **verbatim** as it sat in the `.vital`. The text references it:
  `[osc_1] wavetable = "blob:sha256:3f9a…"`. Verbatim bytes make the
  bijection exact without the format knowing anything about wavetable
  internals; content addressing means two presets sharing a table share the
  file, and changing a table shows in git as one new file plus one changed
  line — never a megabyte of base64 in a diff.
- **Unknown fields** the `.vital` carries (the `extra` maps on the preset
  and on each connection) go under `[vital.extra]` as raw JSON strings, so
  a preset from a newer Vital survives the round trip untouched.

Rejected: base64 blocks at the end of the file (pollute every diff and the
context window for nothing readable), and leaving material only in the
`.vital` (then the text is not a complete view and cannot be the agent's
sole interface). A single sidecar file instead of a directory is the
nearest alternative; the directory wins on diffs and sharing, and a
preset with no material — all five packs — has no sidecar at all.

## Decision 4 — modulation connections

A connection is written **inside the module of its destination**, as an
array-of-tables entry naming the destination parameter:

```toml
[[filter_1.mod]]
to = "cutoff"
from = "lfo_1"
amount = "+70%"
bipolar = false
power = 0.0
stereo = false
rate = "audio"
```

`to` is required because TOML cannot hang a table under a key that already
holds a value; the alternative (`cutoff = { value = "440 Hz", mod = [...] }`
inline) puts a whole modulation on one line and defeats the one-line diff.

**Amount is a signed percentage of the destination's range** — one canonical
form, the same rule everywhere, so an agent never has to know which form
applies where. It is what the engine adds before the parameter's scale and
what Vital's own modulation ring shows; for a scaled destination there is
no honest unit for "what is added before squaring". Two requirements come
with it: on a **Linear destination the unit form is accepted on input**
(`amount = "+2 oct"`, `"+24 st"`, `"+6 dB"`), because someone will think
"a two-octave sweep" and not "18.75 %"; and every such conversion is
returned in the report as a correction (`amount "+2 oct" read as +18.75%`),
so the writer sees what was actually written. The serializer always writes
the percentage, with the reach in the destination's unit as a comment.

**Polarity has a creation default in Vital that the format must not hide.**
`ModulationConnectionBank::createConnection` makes a new connection
bipolar when its source prefix is `lfo`, `random`, `stereo` or `pitch`
(`kBipolarModulationSourcePrefixes`, `synth_types.cpp`) — a UI-time
default, overridden by whatever a loaded file says. Anyone who has patched
in Vital therefore expects an LFO connection to be bipolar unless told
otherwise, which is the opposite of the table default. So `bipolar` is
**always written for connections from those four source families**, true
or false, and omitted only where its absence is unsurprising. Omitting a
default is a readability rule; it yields where the reader's prior points
the other way.

**Regime is written, derived, and checked.** `rate = "audio"` appears on
every connection the engine evaluates per sample — an envelope or LFO into a
filter cutoff, `Connection::is_audio_rate()` — and is omitted on every
other. It is not a free field: writing `rate = "audio"` on a pair the
engine runs at control rate is a load **error** ("`macro_control_1 →
filter_1_cutoff` runs at control rate; audio rate needs an envelope or LFO
source into a filter cutoff"). A comment would carry the same information
but a program cannot read a comment, and an agent choosing a source needs
to know that this pair will sweep per sample and that one will step per
block.

Other defaults (`power = 0`, `stereo = false`, `bypass = false`) are
omitted, per principle 1. A remap curve is inline under the connection
as `curve = { ... }`, same shape syntax as an LFO.

**Slots.** The `.vital` stores amount, bipolar, power, stereo and bypass as
`modulation_N_*` keyed by slot, and the connection list is positional. The
text has no slots: on save, connections are numbered in their text order.
For strict bijection with a `.vital` whose slots are not already in that
order (or that has gaps), the serializer writes `slot = N` on the affected
connections — only then, so a hand-written patch never sees it. This is the
one normalisation the round-trip test applies: a `.vital` whose slot order
is canonical round-trips byte-identically; one that is not round-trips to a
canonical order with `slot` markers, and the rendered audio is compared to
prove the renumbering is silent.

## Decision 5 — the `spinwave_only` namespace

The header carries `requires = "vital"` or `requires = "spinwave"`. It is
**derived** from the parameters actually present — any key the table marks
`spinwave_only` at a non-default value, or a module beyond Vital's sizes
(`osc_4`, `env_7`, `lfo_9`, `macro_5`, `bus_a`, `noise`) — **written by the
serializer, and checked by the parser**: a file that says `vital` but uses
`osc_4` is a load error, and a file that says `spinwave` while using
nothing Spinwave-specific gets a note. The load report names what makes
the patch Spinwave-only (`requires spinwave: osc_4, bus_a_on,
lfo_1_generator`), so an agent asked for a Vital-compatible patch can see
exactly what to drop.

Nothing else changes: Spinwave-only keys are written in their module like
any other, in the same units, because they follow Vital's naming and
scales. The format does not hide them or segregate them; it just refuses
to let a file claim a compatibility it does not have.

## Decision 6 — versioning the format

`format = 1` is the first key of every file. The parser accepts `format`
values it knows and refuses others with the version it expected; a missing
`format` is an error, not a default. Rules from v1 on:

- Adding an optional key or a new unit alias is not a version bump.
- Changing the meaning of an existing key, a unit, a default, or the slot
  rule is a bump, and comes with a migration in a `format_migrations` table
  mirroring `Preset::upgrade`, so any older file still loads and the report
  says what was migrated.
- `synth_version` is carried verbatim from the `.vital` and written back,
  so the `.vital` side's own migrations keep working.

## The round trip, as tests

1. For every `.vital` under `presets/packs/` and for fuzzed patches
   (`--wildness full`, a fixed seed set): `.vital → text → .vital` equal
   value for value on the parsed JSON, modulo the slot normalisation above
   and nothing else. `text → .vital → text` equal byte for byte. A failure
   is a failed test.
2. **Sound equality, bit for bit.** Render the original `.vital` and the
   reconstructed one through `Session` and compare the samples exactly.
   This catches what value equality cannot (a slot renumbering that was not
   silent, a shape regenerated one ulp off). The measurement **self-tests
   first**: a render whose peak is below −60 dBFS for a patch with an
   oscillator on is refused, not compared — an instrument that reads
   silence where there cannot be any must fail loudly, not produce a
   plausible "identical".
3. Precision: the count of `raw:` fallbacks over the corpus is asserted
   (expected zero on the packs; the fuzz figure is measured and committed).
4. Every unit alias, every name alias, every refusal in the tolerance
   section has a test that shows the report entry it produces.

## Tolerance to imperfect input

The parser is generous about spelling and strict about meaning.

- **Units:** `8kHz`, `8 kHz`, `8000Hz`, `8000 Hz`; `90ms`, `90 ms`,
  `0.09s`, `0.09 s`; `st`, `semi`, `semitones`; `%` or a bare fraction
  where the table's unit is `%`; `dB`/`db`. A value with no unit where one
  is expected is **an error naming the expected unit** — `cutoff = 440` is
  refused ("did you mean 440 Hz or 440 st?"), because guessing here is
  exactly the silent tolerance that was ruled out.
- **Names:** every parameter's machine name and `local_description`
  (`"Level"`) and Vital display name; a near miss is refused with a
  suggestion by edit distance over the module's keys (`unknown key freq in
  [filter_1]; did you mean cutoff?`). Two candidates at the same distance
  → refused with both, no pick.
- **Range:** out of range is an error carrying the valid range in the same
  unit as the input.
- **Indexed:** option names case-insensitively, with the table's display
  string and a few spellings (`24db`, `24 dB`).

Everything accepted with a normalisation is reported as a **correction**;
nothing is corrected silently.

## The report

The existing `LoadReport` gains two fields and stays one type:

```rust
pub struct Correction { pub line: u32, pub key: String, pub written: String, pub read_as: String, pub kind: CorrectionKind }
pub struct LoadError  { pub line: u32, pub key: String, pub code: ErrorCode, pub message: String, pub expected: Option<String>, pub suggestion: Option<String> }
```

with `CorrectionKind` (`UnitNormalised`, `AliasResolved`, `FactoryShape`,
`SlotRenumbered`) and `ErrorCode` (`UnknownKey`, `MissingUnit`,
`OutOfRange`, `AmbiguousName`, `BadRegime`, `RequiresMismatch`,
`UnsupportedFormatVersion`, `BlobMissing`). It serialises to JSON like the
rest of the report, so an agent reads `errors[0].suggestion` and fixes its
own file. The existing fields (migrations, ignored connections, unknown
params, notes) are reused, not duplicated.

## CLI

```
spinwave-cli to-text   in.vital     out.spinwave      # writes out.spinwave.d/ only if material exists
spinwave-cli from-text in.spinwave  out.vital
spinwave-cli check     in.spinwave                    # parse only, print the report as JSON
```

## Out of scope, on purpose

No MCP tool, no UI, no LLM anywhere in the parser, no change to the DSP or
to the parameter table. The format adapts to the engine.


## What the implementation settled, and measured

Things the text above left open or got slightly wrong, decided by building
it and running the round trip over the five packs and 24 fuzzed patches.

- **Cutoff: the shorter exact spelling wins.** A cutoff authored in Vital
  sits on a whole semitone, whose Hz value needs seven digits to invert
  exactly (`"369.9944 Hz"` for 66 st). Both forms are accepted on input; the
  writer picks the shorter exact one and puts the other in the comment.
  On the packs that is semitones every time; a cutoff typed as `"1 kHz"`
  stays `"1 kHz"`.
- **Unit-less table entries are bare numbers**, not strings: `sustain =
  0.78`, `wave_frame = 128`, `tune = 5`. A unit was never there to invent.
- **Exactness costs digits on values authored in engine units, and the
  cost is intrinsic, not a conversion loss.** Checked before accepting it:
  a half-ulp of `0.7` in f32 is 7.4e-7 dB, so a dB spelling must carry
  seven digits to pin the same float (`"-6.196079 dB"` is the shortest
  that does); a quartic time needs 2e-5 ms of resolution (`"89.91946 ms"`
  for Vital's default attack `0.5476`). Inverses run in f64 and round once;
  nothing shorter exists. So levels take the **knob value** where it is
  shorter (`0.7`, one digit, exact, and what Vital's files and interface
  show), with dB in the comment — the same shortest-exact rule as the
  cutoff. Times stay in ms/s: their knob value is a quartic position with
  no meaning to anyone, and the long form is only ever seen on values
  authored in engine units. A value typed as `"90 ms"` stays `"90 ms"`;
  rounding one never produces `raw:`, only a slightly different value.
  Measured over 24 fuzzed patches, 24 313 lines: the `raw:` fallback on a
  **continuous** value fired **10 times (0.04 %)**. 104 more `raw:` lines
  are indices past the end of a name list (`style` runs to 9 with five
  names, `view_2d`, `spectrogram`): a table gap, recorded, not the
  format's. The fuzzer used to add 404 fractions where the engine reads an
  integer — patches no preset loader could produce — and now draws
  integers for every indexed parameter. On the packs: zero.
- **Vital reuses a display name for two options** ("FM <- Osc" is both
  oscillator A and B). Such a name carries its index: `"fm <- osc [7]"`.
  A name that is unique reads without it; an ambiguous one is refused
  naming both.
- **Keys inside a module are alphabetical, `on` first.** The table's own
  order is Vital's panel layout and means nothing away from it.
- **An absent LFO shape is the engine's triangle**, so a triangle is never
  written and the round-trip test treats a trailing run of triangles as
  absent. Any other shape, including a triangle that is smoothed or
  renamed, is written out.
- **Modulation slots** are assigned in the writer's module order, which the
  reader reproduces (the TOML map does not keep document order, and a
  hand-written file may put modules anywhere). `slot = N` is written only
  where the source `.vital` is not already in that order; on the packs
  that is two lines in one preset and none elsewhere.
- **`[material]`** holds the blob references (`wavetables`, `sample`,
  `spinwave`); **`[vital.settings]`** carries numeric settings the table
  does not know and **`[vital.extra]`** unknown top-level fields, both as
  raw JSON strings, so a preset from a newer Vital survives untouched.
- **A table bug the format made visible, fixed in its own pass:** the
  table named `osc_N_destination` and `sample_destination` with Vital's
  fourteen-entry list, where index 5 is "chorus", while the engine routes
  5 to bus A and 6 to bus B. Read at the source, Vital's nine effect
  entries are dead — its oscillator popup offers five, the arrows wrap
  modulo five, the producers module tests only the first five constants —
  so no `.vital` written by Vital carries a destination past 4 and the
  buses break nothing. The table now names the engine's seven
  destinations, and `destination = "bus a"` loads.

**Measured on 2026-09-12:** all five packs and 24 fuzzed patches
round-trip value for value and render **bit-identically** through
`Session`; 13 of 24 fuzzed patches are audible (the test requires at least
half, so silence cannot pass for agreement); the golden bench stands at
57 of 70, unchanged.
