// Force-included before every Vital source (see CMakeLists).
//
// Vital's own build resolves several headers transitively, through its
// include order and its JUCE unity build. Compiling the DSP core on its
// own puts headers in a different order, so a handful of types end up used
// before they are declared: `Envelope` in envelope_module.h, `StereoMemory`
// in equalizer_module.h, the `futils` namespace in synth_lfo.cpp.
//
// Pulling them in up front fixes that without editing the reference, which
// has to stay byte-identical to the thing it is the reference for. Every
// header here is `#pragma once`, so arriving early is harmless.

#pragma once

#include "common.h"
#include "futils.h"
#include "memory.h"
#include "envelope.h"
// sound_engine.cpp sizes its chorus memory from ChorusModule::kMaxDelayPairs.
#include "chorus_module.h"
// wave_line_source.h (the wavetable creator) uses LineGenerator without
// including it; Vital's unity build had it in scope.
#include "line_generator.h"
