# Two audits of the port against the reference (2026-09-12)

Both came out of one review remark: after four oscillator inputs turned
out to be audio-rate in Vital, stop discovering them one failing case at
a time and enumerate. The same move — list what the reference does, check
each entry — had already paid twice (the harness's control defaults, the
level ceiling). It paid again, twice, in the same afternoon.

## 1. Every audio-rate modulation destination

Method: `grep createPolyModControl\|createMonoModControl` over
`src/synthesis` (124 calls), keep those with `audio_rate = true`, plus the
`FilterModule::createModControl` / `FormantModule::createModControl`
wrappers that forward the flag. 31 controls in 8 families. For each: what
the reference's consumer does with the per-sample buffer, what Spinwave
does, and a golden case that measures the difference where one can exist.

An audio-rate destination in the reference means its `ModulationSum`
output is a buffer: control-rate sources are ramped linearly across the
block, audio-rate sources (envelopes, LFOs flagged audio-rate) are added
per sample, and the consumer reads the buffer per sample. A control-rate
destination is a `cr::VariableAdd`: one value per block.

| control | scope | reference consumer | Spinwave | case | residual |
|---|---|---|---|---|---|
| `osc_N_transpose` | poly ×3 | per sample in `setPhaseIncBuffer` | audio-rate (`AudioDestBuffers`) | `mod_env_to_pitch` | 2.6e-1 → 5.3e-4 |
| `osc_N_tune` | poly ×3 | same loop | audio-rate | `mod_env_to_tune` | 1.3e-4 |
| `osc_N_phase` | poly ×3 | per sample, `mod` then shift | audio-rate | `mod_lfo_to_phase` | 3.3e-4 |
| `osc_N_level` | poly ×3 | per sample before the square | audio-rate | `mod_env_to_level`, `mod_lfo_to_level` | 7.0e-4, 1.2e-7 |
| `sample_level` | poly | per sample in `SampleSource` | **control-rate** | none: the default sample is noise from a seeded generator, and the bench cannot yet pin that seed | unmeasured — hypothesis: same fix as `osc_N_level` |
| `filter_N_cutoff` | poly ×2 | per-sample cutoff buffer | audio-rate | `mod_env_to_cutoff`, `mod_lfo_to_cutoff` | pass |
| `filter_N_formant_x/y/transpose` | poly ×2 | per-sample formant position | **the formant filter's controls are inert** (`notes/handoff.md`, known and unfixed) | none until the formant is wired | — |
| `filter_fx_cutoff` | mono | per-sample cutoff buffer | control-rate, resolved per block | `mod_lfo_to_filter_fx_cutoff` | 2.2e-3, tracked |
| `filter_fx_formant_*` | mono | as above | inert | — | — |
| `distortion_drive` | mono | `processTimeInvariant(audio, drive[i])` | control-rate | `mod_lfo_to_distortion_drive` | 9.1e-4, passes (barely) |
| `distortion_filter_cutoff` | mono | per-sample SVF cutoff | **was not a destination at all**; now control-rate | `mod_lfo_to_distortion_filter_cutoff` | 1.3e-1 → 5.3e-3, tracked |
| `eq_low/band/high_cutoff` | mono ×3 | per-sample SVF cutoffs | control-rate | `mod_lfo_to_eq_low_cutoff` | 1.0e-3, tracked |
| `phaser_center` | mono | `cutoff[i] = center[i] + sweep` | **was not a destination at all**; now control-rate | `mod_lfo_to_phaser_center` | 2.0e-1 → 5.8e-3, tracked |

The control for the mono rows is `mod_lfo_to_distortion_mix`: a
control-rate destination on the same route, 3.2e-4. So the route is right
and the rate is what the tracked rows measure. Two rows were not rate at
all — two destinations were missing, and the connection was reported as
ignored by the load report and then dropped. The bench now refuses a case
whose connection the engine ignores.

While writing the distortion cases: the distortion's own filter
(`distortion_filter_order/cutoff/resonance/blend`) was never read from
the preset. The engine had the fields, the reader never filled them.
`fx_distortion_filter_pre/post`, unmodulated: 2.3e-1 → 7e-5. Same class
as the formant filter, found the same way — a case that asks for it.

### What making the mono rows audio-rate would take

Not done; proposed. The consumers are mostly ready: the distortion already
takes a per-sample drive buffer (`drive_scratch`), the phaser builds a
per-sample cutoff buffer, the SVF has a per-sample cutoff path used by the
voice filters. What is missing is the plumbing: (1) the kernels must flag
a source audio-rate when an effects connection needs it, not only a voice
connection; (2) an `EffectsAudioBuffers` (7 destinations) filled from the
last active voice's audio-rate source buffers, the control part ramped
across the block as `ModulationSum` does; (3) `EffectChain::resolve`
handing those buffers to the five consumers. Mono, so the CPU cost is one
instance per destination. Estimated at half a day. Worth doing when a
preset that modulates an effect cutoff at audio rate matters more than
the next item on the list; the residuals are 1e-3 to 6e-3.

## 2. Two reference functions, one port

The oscillator's transpose snap (`fillSnapBuffer`: round to the nearest
note, then a table) and the sample source's (`utils::snapTranspose`:
nearest enabled note by distance) are different algorithms with the same
name. Spinwave used the second for both. That is the kind of error a port
produces naturally, so: where else?

Method: list every name the reference defines more than once
(`force_inline ... name(` over `src/synthesis`, grouped by name), keep the
pairs whose bodies differ in semantics rather than in lane width, then
check each call site of the *precise* variant (`utils::`) against the
Spinwave line for that site, and each call site of the *approximate*
variant (`futils::`) likewise.

The pairs that matter are all `utils::X` (exact: `powf`, `log10f`,
`sinf`) against `futils::X` (polynomial approximations): `midiNoteToFrequency`,
`dbToMagnitude`, `magnitudeToDb`, `sin`, `pow`, `exp2`. The port had
already distinguished them in most places (`midi_note_to_frequency_precise`
and friends in `filters/filter_state.rs`, `map(f32::sin)` in the ladder
and the chorus, `pow_exact` in the reverb). Two sites had not:

- **`SynthOscillator::setPhaseIncBuffer`**: the reference converts the
  block's base note with the exact function and scales it per sample by
  the polynomial ratio of the small offset. Spinwave converted the note
  with the polynomial. This was the residual floor of the whole bench:
  every oscillator case sat at ~4e-4 RMS (−67 dB) with the peak on the
  saw's edges, read as "0.015 samples of timing between two phase
  accumulators" and accepted. `osc_saw_dry` 3.3e-4 → 4.4e-8. Sixty of
  seventy cases improved more than tenfold, none got worse, and seven
  tracked cases fell to float noise at once: `mod_lfo_to_level` (so the
  "LFO one-block lead" was never what it was made of),
  `mod_two_voices_one_lfo`, `osc_morph_inharmonic_stretch` (the "term
  near Nyquist growing across the note" was the base pitch error
  integrating), `osc_warp_sync`, `osc_warp_pulse_width`,
  `osc_warp_quantize`, `mod_lfo_to_distortion_drive`.
- **`Delay::setup`**: the three filter frequencies (low, high, damping),
  exact in the reference. Fixed; `fx_delay` did not move (1.4e-2), so its
  cause is elsewhere.

Two more found on the next pass (2026-09-12, later): the unison detune
ratio — `setPhaseIncMults` uses the exact `utils::centsToRatio`, Spinwave
had the polynomial; `osc_unison` 3.2e-4 → 5.5e-8 — and, in the opposite
direction, the exponential-scale control conversion: the reference's
`cr::ExponentialScale` runs `futils::pow` (the polynomial) on every
frequency and delay-time control, Spinwave used the exact `exp2`; aligned,
which moved nothing above float noise.

Not yet checked line by line, listed so the next pass starts here:
`futils::dbToMagnitude` in the compressor thresholds and output gain, the
distortion's drive (`Distortion::scale`), `SynthFilter`'s drive;
`futils::pow` in the compressor ratios, the SVF blend adjust
(`amplitude_quartic`), the reverb size; `futils::exp2` in the diode's
high-pass ratio and the LFO smoothing; `futils::log2` in the wavetable's
frequency-bin lookup and the spectral morphs. Each is "Spinwave must use
the approximation here, not the exact function" — the opposite direction
from the two found, and lower stakes, since the approximations are within
1e-6 of exact and the golden cases that exercise them pass.

## The rule that comes out of both

Before diagnosing a divergence, ask which of these it could be:

1. the case (does the setting arrive? do two cases differing by one
   setting have different references? does the case render the same bytes
   twice, in any order?);
2. a control the reader never fills (the formant, the distortion filter);
3. a destination that does not exist (phaser_center);
4. a rate (per block where the reference is per sample);
5. two reference functions behind one port (the snap, the base frequency);
6. and only then, the DSP.

Items 1–5 have now each been found at least once, and every one of them
was first misread as item 6.
