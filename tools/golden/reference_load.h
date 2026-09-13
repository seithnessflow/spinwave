// Loading a real .vital into the reference engine, without JUCE.
//
// LoadSave::jsonToState (the reference's loader) needs JUCE's File,
// MemoryBlock, Base64 and SynthBase, so it cannot be compiled here. Its
// six steps are re-done in ReferenceLoad with the reference's own parts:
// the version migration copied verbatim (reference_migration.inc, see
// extract_migration.py), the controls set from the settings or their
// table defaults, the modulations connected as jsonToState connects
// them (line mappings included), the sample decoded by Sample::
// jsonToState, the wavetables built by WavetableCreator::jsonToState +
// render, the LFO shapes by LineGenerator::jsonToState, and the
// oversampling taken from the preset as checkOversampling does.
#pragma once

#include <string>
#include <vector>

#include "json/json.h"
#include "load_save.h"
#include "sound_engine.h"

namespace ReferenceLoad {

// Applies a parsed .vital to the engine. Returns false with `error` set
// when the file is newer than the reference or a step fails. Fills
// `ignored` with the connections the engine could not route.
bool applyPreset(vital::SoundEngine& engine, json data, std::string& error,
                 std::vector<std::string>& ignored);

}  // namespace ReferenceLoad
