# Resources for `declared/` — sound design and synthesis

The reading list behind the dictionary (`notes/knowledge-base-design.md`,
layer 3). None of these documents enters the repo or ships with the
plugin: rules are extracted in our own words and our own parameter
names, each validated by a patch built and measured. A rule not yet
validated stays labelled a hypothesis. Each entry says what it yields
and how far to trust it.

## 1. The core — patch recipes and the teaching of synthesis

**Synth Secrets — Gordon Reid, Sound On Sound.**
https://www.soundonsound.com/series/synth-secrets-sound-sound — 63
parts, free, 1999–2004, still the reading of reference in schools.
Subtractive synthesis in depth, then additive, FM, physical modelling,
effects, sampling, wavetables, granular.
*Yields:* the chain "which sound → which patch structure"; Reid reasons
from the physical sound to the modules, which is what a dictionary
entry needs. The instrument-imitation parts (brass, strings,
percussion, bells) give a structure and an order of magnitude per
parameter.
*Trust:* high, but classic analogue; what it says of a filter or an
envelope transposes, what it does not cover is the modern wavetable and
the warps.

**Welsh's Synthesizer Cookbook — Fred Welsh.** A book of patch recipes,
instrument by instrument, with settings — the very format the dictionary
takes.
*Yields:* named patch structures, directly testable.
*Trust:* medium, calibrated on a generic subtractive synth: the values do
not transpose, the structure does. Paid.

**The Vital, Serum and Phase Plant manuals.** Underrated. Vital's
describes the exact parameters we port, with their intent; Serum's and
Phase Plant's describe modules we do not have but whose equivalents we
do.
*Yields:* the intended semantics of every control, and the vocabulary
users actually use.
*Trust:* high for Vital — it is the source.

## 2. The technical side — DSP and spectral analysis

**Julius O. Smith's four books (CCRMA, Stanford).**
https://ccrma.stanford.edu/~jos/ — free, online: *Mathematics of the
DFT with Audio Applications*, *Introduction to Digital Filters*,
*Physical Audio Signal Processing*, *Spectral Audio Signal Processing*.
*Yields:* the grounding of the descriptors and the distance rather than
the dictionary; *Spectral Audio Signal Processing* for the spectral
morph and the analysis, *Introduction to Digital Filters* for what each
filter model does to a spectrum — which feeds `measured/`.
*Trust:* very high, academic.

**The Computer Music Tutorial — Curtis Roads.** The encyclopaedic
reference on synthesis techniques. Paid.
*Yields:* the taxonomy of techniques — to structure the dictionary more
than to fill it.

**Sound Synthesis and Sampling — Martin Russ.** A full, readable,
non-mathematical introduction to the principles of electronic
instruments. Paid.
*Yields:* the bridge between everyday vocabulary and mechanisms.

**Designing Sound — Andy Farnell.** Procedural audio: a sound built from
its physical model rather than from a preset. Paid.
*Yields:* the method "which physical phenomenon produces this sound,
hence which components" — the reasoning we want a model to hold for
non-musical sounds (impacts, breath, textures).

## 3. Descriptors and measurement — already partly in use

Cited in the operations design note already; grouped here so they are
not lost.

- Peeters 2004, *A large set of audio features for sound description*
  (Ircam) — centroid, rolloff, harmonicity, inharmonicity; the base of
  most descriptors.
- de Cheveigné & Kawahara 2002, *YIN, a fundamental frequency
  estimator* — the f0 detector in place.
- ITU-R BS.1770-4 — loudness; the standard, from the ITU.
- Moore & Glasberg — critical bands, the frequency axis of the distance.
- Yamamoto 2020, *Parallel WaveGAN* — the log-STFT term of the
  multi-resolution loss the distance derives from.

## 4. For the Genopatch, later

To read before opening that chantier, not now.

- DDSP (Engel et al., Google Magenta, 2020) — differentiable synthesis;
  the gradient approach on synthesis parameters.
- InverSynth (Barkan et al.) — synthesizer parameter estimation by a
  neural network from the sound: the Genopatch's problem, treated
  academically.
- The "synthesizer sound matching" literature in general — several
  approaches compared and, above all, their documented limits.
- CMA-ES — the optimiser most suited to a continuous, rugged,
  gradient-free parameter space.

## 5. Low-trust sources, treated as such

YouTube tutorials and forums (KVR, r/synthesizers, sound-design
Discords) are the only source on recent genre techniques — the dubstep
growl, the hardstyle supersaw, the modern reese. None of that is in the
books.

Rule: what comes from them enters `declared/` at the lowest trust and
leaves that state only once validated by a measured patch. If a tutorial
says a growl comes from `wave_frame` modulated by a stepped LFO, build
it, measure it, keep the measurement — not the claim. This holds
especially for the techniques the test session showed missing: growl,
resampling, vocal formant, per-note variation.

## 6. Patch corpora — not documentation, the same harvest

- Free `.vital` banks online, large volume, variable quality.
- Vital's factory presets, already in the corpus.

For `corpus/`, structure only, not values. The files stay outside the
repo; only derived statistics enter.

## How to use all of this

1. Read aiming at one dictionary entry per term, not a summary of the
   document.
2. Rephrase in our own words and our own parameters. An entry that
   quotes a text is not a usable entry.
3. Build the patch the rule describes, render it, measure it.
4. Record the measurement as fact, the original rule as provenance, and
   mark the entry validated or refuted.
5. Start small: five or six validated terms are worth more than fifty
   unchecked ones.
