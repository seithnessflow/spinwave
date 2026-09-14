# The bank against the reference — gate 3 (2026-09-13)

The 75 `.vital` of `~/Documents/Vital` (Factory + Afro, 0.6.1–1.0.0)
rendered by both engines and compared sample for sample, with the
wavetable construction and the SMP sample compared apart from the DSP.
Everything here was measured on this day with the Debug harness
(`tools/golden/build/vital_golden.exe --preset`) as the judge; the
Release build (`scratchpad/golden_release`, 60× faster) drove the
bisections only.

## How

`tools/golden/bank_compare.py ~/Documents/Vital --work <dir>` renders
each preset both ways — a primer note hidden by a 5 s skip, then C3 at
0.8 for 1.5 s, 2.5 s rendered, random phases forced off on both sides
(`--fixed-phase`), the host tempo set after the preset loads (40 of the
75 store another tempo) — dumps every oscillator's built table and the
SMP sample (`--dump-tables`), and writes the table below. The reference
side is the slow half (about two minutes a preset in Debug);
`--reuse-reference` keeps it and re-renders the Spinwave side, which is
how the table was refreshed after every fix.

`tools/golden/bank_bisect.py <preset>` renders one preset in variants
— each effect off, each producer off, each filter off, unison 1, LFO
sync types 0, portamento off, no modulations, no connection from each
source, no single connection, LFO smooth mode 0, tempo sync off — and
prints the distance of each. Two of its bugs cost a round each and are
fixed: dropping a connection must renumber the positional
`modulation_N_*` settings (round 1), and must re-point or drop the meta
connections that name a slot (round 2: every "no X" row on a preset with
meta connections had measured a meta link swung onto a neighbour; VLT
Future Gun's "no stereo" read as a fix). Its reference renders are
cached per variant.

The threshold: a preset is settled at RMS 1e-4 (the golden bench's own
bound); above it, the class is named below with the case that reproduces
it, or the preset is listed as open. The band distance (dB over the
spectrum) is the audible measure; 0.01 dB is inaudible.

## Where it stands

75 presets, none failing to load or render. RMS distribution: 25 at or
under 1e-5, 14 in (1e-5, 1e-4], 17 in (1e-4, 1e-3], 10 in (1e-3, 1e-2],
9 above 1e-2. Band distance: 50 at or under 0.01 dB, 63 at or under
0.1 dB. The first table of the day had its top at 21, 15, 15, 13, 10 dB
with 20 presets above 1 dB; it now has one at 21 dB (a tracked class),
four between 1 and 2 dB, and the rest under 1 dB.

Five presets render as a constant +2.1 on BOTH engines — Cinema Bells,
Feeder, Boot Scre3n, Simple Weoum, Metal Head: an inharmonic-stretch
spectral morph with a spectral unison of three or more produces NaN in
the reference (its frame buffer aliases its own guard, reproduced in
`spectral_morph.rs`), clamped to 2.1 by the output stage. Byte-identical,
and an open question against the real plugin, which we could not run.
The order test on the bank (`the_bank_renders_independently_of_what_was_
loaded_before`) leaves these five out by name: NaN compares equal to
nothing, and under a Debug build the DSP's guard on a NaN pushed into a
delay fires on them (found 2026-09-14 on a Debug run of the workspace;
why the earlier runs were green is not recorded). The test itself now
runs only under `SPINWAVE_FULL_BANK=1` (two renders of seventy presets
are 43 minutes of every core in Debug); run it in Release, the build
byte identity is about.

## What the pass found and fixed

Each was reproduced by a case (in `tools/golden/cases`, named after the
class or the preset), measured before, fixed in the engine, measured at
the floor after. The bank's number is the Debug reading after the fix.

| class (preset) | before | after | what it was |
|---|---|---|---|
| global sampler at 44.1 kHz (Plucked String, Float Chords) | 7.1 dB / 6.6 dB | 9.4e-7 / 8.2e-5 | `SynthVoiceKernel::set_sample_rate` never reached the SMP sampler: every sample played an octave up at 2× oversampling |
| random_2..4 seeds (A Night in Kalyan, Special Glitch Thing) | 9.1 dB | 7.6e-6 | the reference's counter runs DOWN the construction order: random_N holds seed 19 − N, the per-note `random` 19 (`tools/golden/random_seed.py` on `--probe random_N`) |
| the per-note `random` (Satellites, DIY, Salomon) | 8.1 dB | 2.2e-5 | same counter, seed 19 |
| random LFOs per sample (Fun Pulse, 14 presets on sample-and-hold) | 4.4e-4 (case) | floor | the reference's RandomLfo never runs control rate: steps land mid-block, a reset block ramps from the fresh draw, a control-rate consumer reads the buffer's first sample |
| the `stereo` source (DIY, Salomon, Metal Head, Destruction, 41 connections) | 1.5e-1 (case) | floor | `cr::Value(kLeftOne)` is [1, 0]; Spinwave held [0, 1] |
| FM / RM wiring (Strings Section, Banana Wob) | 7.4 dB / 5.9 dB | 1.3e-4 / 6.6e-6 | oscillator A of slot 1 is slot 2, of slots 2 and 3 it is slot 1; B is slot 3 for slots 1 and 2, slot 2 for slot 3; the sample is the third source; slots render in the reference's dependency order and a cycle renders nothing. Spinwave wired every slot to the next |
| mono plug into a modulator parameter (VLT Future Gun) | 2.5 dB | 4.6e-4 | a macro's connection plugs the destination's MONO total, whose dependencies exclude the poly connections: the router replay had pulled the envelope's lagged edge back to the front |
| envelope into voice_transpose (Oolacile Evil Dubstep Bass) | 1e-1 | 1.6e-5 | read the same block (the plug moves the envelope ahead of `bent_midi_`); an LFO or random stays a block late (it reads the bent midi for its keytrack: a cycle) — Memory Leak and THUNK regressed to 1.3e-1 / 4.9e-2 under the first, blanket version and returned to 2.3e-6 / 9.6e-6 with the split |
| the comb filter under a moving cutoff (Crescendo Bells, Fun Pulse, Railgun, Space Station, Big Stomp, Digital roller, Fleet…) | 5.4e-2 (case) | floor | a filter's setup reads its cutoff buffer's FIRST sample (`FilterState::loadSettings`), not the block target; only the comb's internal one-poles hear it |
| stereo sample cap (Float Chords) | length 1 764 000 vs 1 860 924 | whole | the reference caps a MONO load at 40 s and a stereo one not at all; reproduced |

Beside the bank: the bisection tool's two renumbering bugs above, and
the destination dump of the reference harness (`VITAL_GOLDEN_DUMP_DEST`)
that wrote garbage past a control-rate total's one-sample buffer.

## What is diagnosed and open

Each has a case in the corpus and an entry in `KNOWN_DIVERGENCES` with
the measurements; the bank's number is the Debug reading.

| preset | rms | band | class | where it stands |
|---|---|---|---|---|
| Staggered Phrases | 9.0e-2 | 20.98 dB | downsample distortion with an LFO on its drive (`mod_lfo_to_downsample_drive`, tracked since the previous pass) | the hold grid's phase from the render's first block; the rest of the preset reads 1.4e-5 with the distortion off |
| Special Glitch Thing | 1.7e-1 | 1.88 dB | the WAVETABLE: an Audio File Source with frequency interpolation between cycles 5 and 6 of its file, where cycle 6 has exactly-zero harmonics; the interpolated phase is `arg` of the FFT's rounding residue, and the two FFTs round differently (frames 54–63 differ at 2.5e-3, the keyframes at 6e-7) | ill-conditioned by construction; the preset sweeps the frame through it. Not an engine difference |
| Remedial Shikari | 1.4e-1 | 1.45 dB | an LFO sweeping osc_1_unison_voices 1→16 in a quarter second (`bank_remedial_lfo_to_unison_voices` 2.7e-2, confined to counts 3–5; narrower sweeps and an envelope at the floor) | not located |
| Cursed Steps | 4.6e-2 | 1.30 dB | a random into a macro into a pitch (`bank_cursed_*`, 2.2e-1 / 3.7e-2 with an LFO): the macro chain's timing around a note-on, invisible on a cutoff (`macro_dest_lfo` 4.5e-5), integrated by a pitch | the two-block macro lag was measured on static steps only |
| Smoker's Lounge | 2.4e-2 | 0.31 dB | velocity into a macro into osc_1_wave_frame: the same macro-destination class (3.0e-4 without that connection) | as above |
| Squish Clicker | 2.8e-2 | 0.92 dB | an LFO into the delay time (`bank_squish_lfo_to_delay_frequency` 1.2e-1): alive the engines agree to 1e-3, after the voice dies the held modulation differs | not established |
| Phaser Entropy | 2.5e-2 | 0.23 dB | an LFO into filter_fx_blend + resonance (`bank_phaser_entropy_lfo_to_filter_fx_blend` 8.4e-4) and env_1 into the effects; the phaser's four control-rate destinations read ≤ 5.5e-5 | not located |
| Piano from the yard sale | 1.9e-2 | 0.44 dB | a hard-clipped burst (both engines hit the 2.1 clamp at the same samples but one) under `note → env_2_decay` with decay power −8.84; the case at note 45 reads the floor | one sample of a clipped burst |
| Dispersed Grit | 1.8e-2 | 0.16 dB | a poly envelope into the mono filter_fx_cutoff with its amount from the velocity (`bank_dispersed_env_to_filter_fx_meta_velocity` 2.9e-4 with two voices) | which voice's velocity, and when, is not pinned |
| Complex Electro 1, Phaser Man, Space Station, Super Nice Pluck, Digital roller, Fleet, Railgun (3.3e-3 … 2.0e-3) | | ≤ 0.15 dB | the compressor's ~1e-5 class through a hot chain, `lfo_6 → volume` with a meta amount (1.3e-4), the wavetable columns (Railgun's tables at 2e-3) | under the audible line; not pursued this pass |

The `sample` column of the table reads 1.8e-5 … 3.1e-5 on every preset
with a sample: one PCM16 step, the decode's rounding (`pcmToFloatData`
against `pcm_bytes_to_f32`), −90 dB. Not pursued.

Transport-synced sources (LFO sync type "sync", random sync type 1: Ah
Eh Ee Oh, corrupted_…) are UNMEASURED: neither harness runs a
transport, so both sides render the source at 0
(`mod_random_*_sync_two_voices` pin exactly that).

## The table

Sorted by RMS, descending. `tables` is the max |Δ| of each oscillator's
built wavetable over frames and samples; `sample` the same for the SMP
sample (`missing` when the preset has none).

| preset | RMS | band dB | peak | ref peak | tables (osc 1 / 2 / 3) | sample |
|---|---|---|---|---|---|---|
| Special Glitch Thing | 1.7e-01 | 1.88 | 9.0e-01 | 0.691 | 2.5e-03 / 0.0e+00 / 0.0e+00 | missing |
| Remedial Shikari | 1.4e-01 | 1.45 | 1.2e+00 | 0.687 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Staggered Phrases | 9.0e-02 | 20.98 | 1.1e+00 | 0.767 | 2.4e-04 / 0.0e+00 / 0.0e+00 | missing |
| Cursed Steps | 4.6e-02 | 1.30 | 8.2e-01 | 0.808 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Squish Clicker | 2.8e-02 | 0.92 | 6.9e-01 | 0.723 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Phaser Entropy  | 2.5e-02 | 0.23 | 9.0e-01 | 0.616 | 3.0e-04 / 0.0e+00 / 0.0e+00 | missing |
| Smoker's Lounge | 2.4e-02 | 0.31 | 5.7e-01 | 0.609 | 7.1e-05 / 1.5e-04 / 9.7e-04 | 3.1e-05 |
| Piano from the yard sale | 1.9e-02 | 0.44 | 3.6e-01 | 2.100 | 0.0e+00 / 0.0e+00 / 1.5e-05 | missing |
| Dispersed Grit | 1.8e-02 | 0.16 | 2.9e-01 | 0.511 | 5.3e-05 / 5.3e-05 / 0.0e+00 | missing |
| Space Station | 6.4e-03 | 0.12 | 7.7e-02 | 0.575 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 2.7e-05 |
| Fleet | 5.6e-03 | 0.08 | 7.2e-02 | 0.561 | 2.4e-04 / 4.8e-07 / 0.0e+00 | 3.1e-05 |
| Digital roller | 4.3e-03 | 0.09 | 2.0e-01 | 0.735 | 4.8e-07 / 9.8e-04 / 0.0e+00 | missing |
| Complex Electro 1 | 3.3e-03 | 0.15 | 4.7e-02 | 0.834 | 9.0e-05 / 0.0e+00 / 1.4e-06 | 1.9e-05 |
| Super Nice Pluck | 2.8e-03 | 0.09 | 3.0e-02 | 1.281 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 3.0e-05 |
| Railgun | 2.0e-03 | 0.04 | 1.9e-01 | 0.682 | 2.0e-03 / 2.0e-03 / 0.0e+00 | 2.7e-05 |
| corrupted_...+=boot-_ | 1.6e-03 | 0.02 | 1.7e-02 | 0.135 | 1.0e-02 / 1.0e-02 / 1.0e-02 | missing |
| Phaser Man | 1.5e-03 | 0.14 | 7.9e-02 | 0.589 | 9.1e-05 / 2.7e-04 / 0.0e+00 | missing |
| Destruction | 1.4e-03 | 0.03 | 5.3e-02 | 1.277 | 3.0e-07 / 0.0e+00 / 1.2e-07 | 2.6e-05 |
| Honk Wub | 1.3e-03 | 0.03 | 3.3e-02 | 0.962 | 8.3e-07 / 0.0e+00 / 2.1e-04 | 1.8e-05 |
| Disrupt | 9.9e-04 | 0.02 | 1.2e-02 | 0.675 | 3.9e-02 / 7.4e-04 / 3.0e-04 | missing |
| LORN Style Lead | 8.9e-04 | 0.01 | 7.6e-03 | 0.615 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| FM Drum Circle | 8.7e-04 | 0.03 | 9.7e-03 | 0.458 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| VLT Future Gun | 6.9e-04 | 0.00 | 1.1e-02 | 0.526 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 2.7e-05 |
| Digestive Trauma | 6.9e-04 | 0.00 | 4.1e-02 | 0.573 | 1.2e-06 / 0.0e+00 / 4.6e-06 | missing |
| FM Mode | 5.7e-04 | 0.02 | 2.0e-02 | 0.346 | 0.0e+00 / 4.8e-05 / 0.0e+00 | missing |
| Ah Eh Ee Oh | 5.3e-04 | 0.00 | 2.9e-02 | 0.660 | 4.2e-05 / 0.0e+00 / 0.0e+00 | missing |
| Gorgled | 3.9e-04 | 0.00 | 1.0e-02 | 2.067 | 3.7e-04 / 8.5e-04 / 0.0e+00 | missing |
| Keystation | 3.3e-04 | 0.01 | 8.6e-04 | 0.403 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Drowning Machine | 3.2e-04 | 0.01 | 2.0e-03 | 0.487 | 4.5e-03 / 4.5e-03 / 4.5e-03 | missing |
| Shepard Tone Template | 3.0e-04 | 0.00 | 8.2e-03 | 0.492 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Damped Horn | 2.5e-04 | 0.01 | 2.7e-03 | 0.403 | 3.9e-06 / 0.0e+00 / 0.0e+00 | missing |
| Big Stomp | 2.4e-04 | 0.00 | 2.7e-02 | 1.008 | 5.9e-03 / 1.5e-05 / 0.0e+00 | missing |
| Chorusy Keys | 1.5e-04 | 0.04 | 2.2e-03 | 0.344 | 7.0e-05 / 3.1e-04 / 0.0e+00 | missing |
| E4 One Note Metallophone | 1.4e-04 | 0.00 | 4.3e-03 | 0.573 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 3.1e-05 |
| Strings Section | 1.3e-04 | 0.01 | 4.4e-03 | 0.244 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Distant Majestic Lead | 1.1e-04 | 0.02 | 6.6e-04 | 0.284 | 1.4e-04 / 0.0e+00 / 0.0e+00 | missing |
| Real Squarepusher Hours | 9.2e-05 | 0.00 | 7.7e-04 | 0.582 | 1.9e-06 / 5.3e-05 / 0.0e+00 | missing |
| Float Chords | 8.2e-05 | 0.00 | 5.0e-04 | 0.616 | 1.9e-06 / 0.0e+00 / 0.0e+00 | 3.1e-05 |
| Horror of Melbourne 1 | 7.9e-05 | 0.01 | 1.1e-03 | 0.618 | 0.0e+00 / 4.0e-03 / 0.0e+00 | missing |
| Fun Pulse | 7.2e-05 | 0.00 | 6.7e-03 | 0.583 | 1.9e-03 / 0.0e+00 / 0.0e+00 | missing |
| Oolacile Evil Dubstep Bass | 6.6e-05 | 0.00 | 5.6e-04 | 0.734 | 3.0e-04 / 1.1e-06 / 9.5e-07 | 3.1e-05 |
| Skew Resandal | 3.5e-05 | 0.00 | 1.9e-03 | 0.540 | 4.7e-04 / 0.0e+00 / 0.0e+00 | missing |
| Salomon | 3.3e-05 | 0.00 | 3.4e-04 | 0.684 | 1.8e-05 / 8.3e-03 / 5.3e-05 | missing |
| Thumpus | 2.5e-05 | 0.00 | 1.3e-03 | 0.823 | 3.4e-04 / 3.4e-04 / 2.1e-03 | missing |
| Satellites | 2.2e-05 | 0.00 | 1.9e-04 | 0.422 | 0.0e+00 / 0.0e+00 / 2.0e-03 | 2.8e-05 |
| Jupiter Bass | 2.0e-05 | 0.00 | 2.4e-04 | 0.755 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 2.7e-05 |
| Random Amp Growl | 1.8e-05 | 0.00 | 2.0e-04 | 0.386 | 3.8e-03 / 3.7e-02 / 0.0e+00 | missing |
| Super Pluck | 1.6e-05 | 0.00 | 6.8e-04 | 0.625 | 1.1e-06 / 1.1e-06 / 0.0e+00 | missing |
| Snowcrash | 1.3e-05 | 0.00 | 7.1e-05 | 0.468 | 8.1e-03 / 3.9e-02 / 3.9e-02 | missing |
| Swiss Army Knife | 1.1e-05 | 0.00 | 5.8e-05 | 0.891 | 5.3e-05 / 0.0e+00 / 0.0e+00 | 2.9e-05 |
| THUNK | 9.6e-06 | 0.00 | 2.0e-04 | 0.574 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 3.1e-05 |
| Synthetic Quartet | 9.0e-06 | 0.00 | 4.8e-05 | 0.886 | 0.0e+00 / 4.9e-04 / 0.0e+00 | missing |
| Ceramic | 8.7e-06 | 0.00 | 5.0e-05 | 0.450 | 3.9e-02 / 3.9e-02 / 3.9e-02 | 2.7e-05 |
| A Night in Kalyan | 7.6e-06 | 0.00 | 1.4e-04 | 0.505 | 2.5e-04 / 4.4e-04 / 0.0e+00 | 2.7e-05 |
| Banana Wob | 6.6e-06 | 0.00 | 4.7e-05 | 0.467 | 2.8e-03 / 2.4e-07 / 0.0e+00 | missing |
| Easy Mallet | 5.6e-06 | 0.00 | 5.2e-05 | 0.462 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 2.7e-05 |
| Crescendo Bells | 4.0e-06 | 0.00 | 3.7e-05 | 0.452 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 2.9e-05 |
| A happy ending of the world | 3.1e-06 | 0.00 | 1.1e-04 | 0.579 | 1.2e-06 / 3.0e-04 / 0.0e+00 | missing |
| Memory Leak | 2.3e-06 | 0.00 | 1.0e-04 | 0.396 | 6.1e-06 / 0.0e+00 / 6.1e-06 | 3.1e-05 |
| Analog Pad | 1.7e-06 | 0.00 | 5.4e-06 | 0.354 | 3.6e-04 / 0.0e+00 / 0.0e+00 | missing |
| Moving Harmonics | 1.6e-06 | 0.00 | 5.3e-05 | 0.316 | 7.8e-03 / 0.0e+00 / 0.0e+00 | missing |
| Physical Tension | 1.2e-06 | 0.00 | 2.1e-05 | 0.375 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Plucked String | 9.4e-07 | 0.00 | 8.7e-06 | 0.410 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 2.6e-05 |
| Text To Wavetable Template | 7.1e-07 | 0.00 | 1.1e-05 | 0.241 | 1.9e-04 / 0.0e+00 / 0.0e+00 | missing |
| Abbysun | 2.5e-07 | 0.00 | 1.1e-06 | 0.097 | 0.0e+00 / 5.3e-05 / 1.3e-06 | missing |
| DIY | 2.3e-07 | 0.00 | 5.4e-06 | 0.629 | 1.5e-03 / 0.0e+00 / 0.0e+00 | 2.5e-05 |
| Kick Drum 1 | 1.1e-07 | 0.00 | 6.4e-06 | 0.615 | 0.0e+00 / 0.0e+00 / 0.0e+00 | 2.7e-05 |
| Moog Pluck | 4.9e-08 | 0.00 | 3.6e-07 | 0.661 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Touch Tone | 4.4e-08 | 0.00 | 2.4e-07 | 0.623 | 0.0e+00 / 0.0e+00 / 0.0e+00 | missing |
| Growl Bass Sidechain | 4.3e-08 | 0.00 | 3.3e-07 | 0.522 | 6.1e-06 / 0.0e+00 / 0.0e+00 | missing |
| Cinema Bells | 0.0e+00 | 0.00 | 0.0e+00 | 2.100 | 1.1e-06 / 1.1e-06 / 0.0e+00 | missing |
| Feeder | 0.0e+00 | 0.00 | 0.0e+00 | 2.100 | 1.3e-03 / 4.3e-04 / 1.3e-03 | 3.1e-05 |
| Boot Scre3n | 0.0e+00 | 0.00 | 0.0e+00 | 2.100 | 5.3e-05 / 0.0e+00 / 0.0e+00 | missing |
| Simple Weoum | 0.0e+00 | 0.00 | 0.0e+00 | 2.100 | 2.0e-05 / 2.0e-05 / 2.0e-05 | missing |
| Metal Head | 0.0e+00 | 0.00 | 0.0e+00 | 2.100 | 2.9e-06 / 0.0e+00 / 0.0e+00 | 3.1e-05 |
