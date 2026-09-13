// Stub for the reference's load_save.h: the DSP-free part of LoadSave the
// harness needs — the preset migration (reference_migration.inc, copied
// verbatim from the reference), the PCM/base64 converters the wavetable
// creator's own migration calls, and the version comparators
// (reimplemented, reference_load.cpp). The real header pulls in JUCE's
// File and MemoryBlock and SynthBase.
#pragma once

#include <string>

#include "JuceHeader.h"
#include "json/json.h"

using json = nlohmann::json;

class LoadSave {
  public:
    static json updateFromOldVersion(json state);
    static void convertBufferToPcm(json& data, const std::string& field);
    static void convertPcmToFloatBuffer(json& data, const std::string& field);
    static int compareVersionStrings(String a, String b);
    static int compareFeatureVersionStrings(String a, String b);
};
