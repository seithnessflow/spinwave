#include "reference_load.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "line_generator.h"
#include "modulation_connection_processor.h"
#include "sample_source.h"
#include "synth_constants.h"
#include "synth_parameters.h"
#include "synth_types.h"
#include "wavetable.h"
#include "wavetable_creator.h"
#include "synth_oscillator.h"

namespace ReferenceLoad {

// The reference's ProjectInfo::versionString (JuceHeader).
static const char* kReferenceVersion = "1.0.7";

std::string trim(std::string s) {
  auto blank = [](char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; };
  while (!s.empty() && blank(s.back()))
    s.pop_back();
  size_t start = 0;
  while (start < s.size() && blank(s[start]))
    ++start;
  return s.substr(start);
}

std::string upToFirst(const std::string& s, char c) {
  size_t i = s.find(c);
  return i == std::string::npos ? s : s.substr(0, i);
}

std::string fromFirst(const std::string& s, char c) {
  size_t i = s.find(c);
  return i == std::string::npos ? "" : s.substr(i + 1);
}

std::string upToLast(const std::string& s, char c) {
  size_t i = s.rfind(c);
  return i == std::string::npos ? s : s.substr(0, i);
}

int numeric(const std::string& s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
    return 0;
  return std::atoi(s.c_str());
}

}  // namespace ReferenceLoad

// LoadSave::compareVersionStrings: major, then minor, then patch, each
// compared as an integer (a non-numeric token counts as 0). The
// reference's is JUCE String code; this is the same on std::string.
int LoadSave::compareVersionStrings(String a_in, String b_in) {
  using namespace ReferenceLoad;
  std::string a = trim(a_in.toStdString()), b = trim(b_in.toStdString());
  if (a.empty() && b.empty())
    return 0;
  for (int level = 0; level < 3; ++level) {
    int va = numeric(upToFirst(a, '.'));
    int vb = numeric(upToFirst(b, '.'));
    if (va > vb)
      return 1;
    if (va < vb)
      return -1;
    a = fromFirst(a, '.');
    b = fromFirst(b, '.');
  }
  return 0;
}

int LoadSave::compareFeatureVersionStrings(String a, String b) {
  using namespace ReferenceLoad;
  return compareVersionStrings(String(upToLast(trim(a.toStdString()), '.')),
                               String(upToLast(trim(b.toStdString()), '.')));
}

#include "reference_migration.inc"

namespace ReferenceLoad {

bool applyPreset(vital::SoundEngine& engine, json data, std::string& error,
                 std::vector<std::string>& ignored) {
  std::string version = data["synth_version"];
  if (LoadSave::compareFeatureVersionStrings(version, kReferenceVersion) > 0) {
    error = "preset is newer than the reference: " + version;
    return false;
  }
  if (LoadSave::compareVersionStrings(version, kReferenceVersion) < 0 || data["settings"].count("sub_octave"))
    data = LoadSave::updateFromOldVersion(data);

  json settings = data["settings"];
  json modulations = settings["modulations"];
  json sample = settings["sample"];
  json wavetables = settings["wavetables"];
  json lfos = settings["lfos"];

  // loadControls: every control from the settings, else its default.
  vital::control_map controls = engine.getControls();
  for (auto& control : controls) {
    std::string name = control.first;
    if (settings.count(name)) {
      vital::mono_float value = settings[name];
      control.second->set(value);
    }
    else {
      vital::ValueDetails details = vital::Parameters::getDetails(name);
      control.second->set(details.default_value);
    }
  }

  // loadModulations: the bank's own processors in slot order, connected
  // the way SynthBase::connectModulation does it, the stored line
  // mapping applied.
  vital::ModulationConnectionBank& bank = engine.getModulationBank();
  int index = 0;
  for (const json& modulation : modulations) {
    std::string source = modulation["source"];
    std::string destination = modulation["destination"];
    vital::ModulationConnection* connection = bank.atIndex(index);
    index++;
    if (engine.getModulationSource(source) == nullptr ||
        engine.getMonoModulationDestination(destination) == nullptr) {
      // Presets carry every slot of the bank; the empty ones are not
      // connections.
      if (!source.empty() || !destination.empty())
        ignored.push_back(source + " -> " + destination);
      continue;
    }
    if (source.length() && destination.length()) {
      connection->source_name = source;
      connection->destination_name = destination;
      vital::modulation_change change;
      change.source = engine.getModulationSource(source);
      change.mono_destination = engine.getMonoModulationDestination(destination);
      change.mono_modulation_switch = engine.getMonoModulationSwitch(destination);
      change.destination_scale = vital::Parameters::getParameterRange(destination);
      change.poly_modulation_switch = engine.getPolyModulationSwitch(destination);
      change.poly_destination = engine.getPolyModulationDestination(destination);
      change.modulation_processor = connection->modulation_processor.get();
      change.disconnecting = false;
      change.num_audio_rate = 0;
      engine.connectModulation(change);
    }
    if (modulation.count("line_mapping"))
      connection->modulation_processor->lineMapGenerator()->jsonToState(modulation["line_mapping"]);
    else
      connection->modulation_processor->lineMapGenerator()->initLinear();
  }

  // loadSample.
  vital::Sample* sample_source = engine.getSample();
  if (sample_source && !sample.is_null())
    sample_source->jsonToState(sample);

  // loadWavetables: the creator builds each table into the engine's own.
  int table_index = 0;
  for (const json& wavetable : wavetables) {
    vital::Wavetable* table = engine.getWavetable(table_index);
    if (table == nullptr)
      break;
    WavetableCreator creator(table);
    creator.jsonToState(wavetable);
    creator.render();
    table_index++;
  }

  // loadLfos.
  int lfo_index = 0;
  for (const json& lfo : lfos) {
    LineGenerator* source = engine.getLfoSource(lfo_index);
    if (source == nullptr)
      break;
    source->jsonToState(lfo);
    source->render();
    lfo_index++;
  }
  return true;
}

}  // namespace ReferenceLoad
