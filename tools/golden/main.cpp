// Renders a test case through Vital's own DSP core, so Spinwave can be
// compared against it sample for sample.
//
// This is the reference half of the golden bench. It drives
// `vital::SoundEngine` directly rather than loading a `.vital` preset,
// because preset loading lives in JUCE-heavy code and because setting
// parameters by name is exactly what the Rust side does too: both halves
// read the same case file, so a difference in the audio is a difference in
// the DSP and nothing else.
//
//     vital_golden <case-file> <out.raw>
//
// The case file is one directive per line:
//     rate 44100          sample rate (default 44100)
//     seconds 1.0         render length
//     note 45 0.9 0.0 0.6 midi note, velocity, start seconds, hold seconds
//     wave saw            single-cycle shape in every oscillator table
//     skip 0.25           seconds rendered but not written (see the Rust half)
//     set osc_1_level 0.7 a control, by its Vital parameter name
//     modulate lfo_1 filter_1_cutoff 0.5
//                         a modulation connection and its amount
//
// The output is raw little-endian f32, interleaved stereo, which both
// sides read without needing a WAV parser.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "line_generator.h"
#include "modulation_connection_processor.h"
#include "synth_types.h"
#include "sound_engine.h"
#include "synth_parameters.h"
#include "wave_frame.h"
#include "wavetable.h"

namespace {

struct Note {
  int midi = 45;
  float velocity = 0.9f;
  float start_seconds = 0.0f;
  float hold_seconds = 0.5f;
};

struct Modulation {
  std::string source;
  std::string destination;
  float amount = 0.5f;
};

struct Case {
  int sample_rate = 44100;
  /// Which predefined single-cycle shape fills every oscillator's table.
  vital::PredefinedWaveFrames::Shape shape = vital::PredefinedWaveFrames::kSaw;
  float seconds = 1.0f;
  /// Seconds rendered but not written: the primer note, which puts the
  /// engine in the right state without being compared.
  float skip_seconds = 0.0f;
  std::vector<Note> notes;
  std::vector<std::pair<std::string, float>> controls;
  std::vector<Modulation> modulations;
};

bool readCase(const char* path, Case& result, std::string& error) {
  std::ifstream file(path);
  if (!file) {
    error = std::string("cannot open ") + path;
    return false;
  }

  std::string line;
  int line_number = 0;
  while (std::getline(file, line)) {
    ++line_number;
    // Blank lines and `#` comments keep the case files readable.
    size_t comment = line.find('#');
    if (comment != std::string::npos)
      line = line.substr(0, comment);
    std::istringstream stream(line);
    std::string directive;
    if (!(stream >> directive))
      continue;

    if (directive == "rate") {
      stream >> result.sample_rate;
    }
    else if (directive == "seconds") {
      stream >> result.seconds;
    }
    else if (directive == "note") {
      Note note;
      stream >> note.midi >> note.velocity >> note.start_seconds >> note.hold_seconds;
      result.notes.push_back(note);
    }
    else if (directive == "skip") {
      stream >> result.skip_seconds;
    }
    else if (directive == "wave") {
      std::string name;
      stream >> name;
      if (name == "sin") result.shape = vital::PredefinedWaveFrames::kSin;
      else if (name == "triangle") result.shape = vital::PredefinedWaveFrames::kTriangle;
      else if (name == "square") result.shape = vital::PredefinedWaveFrames::kSquare;
      else if (name == "pulse") result.shape = vital::PredefinedWaveFrames::kPulse;
      else if (name == "saw") result.shape = vital::PredefinedWaveFrames::kSaw;
      else {
        error = "unknown wave shape '" + name + "'";
        return false;
      }
    }
    else if (directive == "modulate") {
      Modulation modulation;
      stream >> modulation.source >> modulation.destination >> modulation.amount;
      result.modulations.push_back(modulation);
    }
    else if (directive == "set") {
      std::string name;
      float value = 0.0f;
      stream >> name >> value;
      result.controls.push_back({name, value});
    }
    else {
      error = "line " + std::to_string(line_number) + ": unknown directive '" + directive + "'";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: vital_golden <case-file> <out.raw>\n");
    return 2;
  }

  Case test_case;
  std::string error;
  if (!readCase(argv[1], test_case, error)) {
    std::fprintf(stderr, "vital_golden: %s\n", error.c_str());
    return 1;
  }

  vital::SoundEngine engine;
  engine.setSampleRate(test_case.sample_rate);
  engine.setBpm(120.0f);

  // Vital's own startup does this through SynthBase: without it the
  // oscillators have no waveform and the LFOs no shape, and the engine
  // renders silence. The wavetable creator lives in JUCE-heavy code, so
  // each oscillator gets one predefined shape repeated across its frames.
  // That is deliberate: the Rust side loads the same single shape, which
  // keeps the comparison about the DSP rather than about two different
  // wavetable builders.
  const vital::WaveFrame* frame =
      vital::PredefinedWaveFrames::getWaveFrame(test_case.shape);
  for (int i = 0; i < vital::kNumOscillators; ++i) {
    vital::Wavetable* wavetable = engine.getWavetable(i);
    if (wavetable == nullptr)
      continue;
    wavetable->setNumFrames(1);
    wavetable->loadWaveFrame(frame, 0);
    wavetable->postProcess(1.0f);
  }
  for (int i = 0; i < vital::kNumLfos; ++i) {
    LineGenerator* lfo = engine.getLfoSource(i);
    if (lfo)
      lfo->initTriangle();
  }

  // Start every control at Vital's own default, then apply the case. This
  // matters: an unset control must hold the value the reference ships, not
  // whatever the engine happens to construct with.
  vital::control_map controls = engine.getControls();
  for (auto& entry : controls) {
    entry.second->set(vital::Parameters::getDetails(entry.first).default_value);
  }

  for (const auto& control : test_case.controls) {
    auto found = controls.find(control.first);
    if (found == controls.end()) {
      std::fprintf(stderr, "vital_golden: unknown control '%s'\n", control.first.c_str());
      return 1;
    }
    found->second->set(control.second);
  }

  // Modulation connections are not controls: the reference wires them
  // through its own bank, exactly as SynthBase::createModulationChange
  // does. The amount rides on the `modulation_N_amount` control, so the
  // connection has to be made before the controls are applied... which is
  // why this runs here, after the control defaults and before the loop.
  vital::ModulationConnectionBank& bank = engine.getModulationBank();
  for (size_t i = 0; i < test_case.modulations.size(); ++i) {
    const Modulation& wanted = test_case.modulations[i];
    vital::ModulationConnection* connection =
        bank.createConnection(wanted.source, wanted.destination);
    if (connection == nullptr) {
      std::fprintf(stderr, "vital_golden: no free modulation slot for %s -> %s\n",
                   wanted.source.c_str(), wanted.destination.c_str());
      return 1;
    }

    vital::modulation_change change;
    change.source = engine.getModulationSource(connection->source_name);
    change.mono_destination = engine.getMonoModulationDestination(connection->destination_name);
    change.mono_modulation_switch = engine.getMonoModulationSwitch(connection->destination_name);
    if (change.source == nullptr || change.mono_destination == nullptr) {
      std::fprintf(stderr, "vital_golden: cannot connect %s -> %s\n",
                   wanted.source.c_str(), wanted.destination.c_str());
      return 1;
    }
    change.destination_scale =
        vital::Parameters::getParameterRange(connection->destination_name);
    change.poly_modulation_switch = engine.getPolyModulationSwitch(connection->destination_name);
    change.poly_destination = engine.getPolyModulationDestination(connection->destination_name);
    change.modulation_processor = connection->modulation_processor.get();
    change.disconnecting = false;
    change.num_audio_rate = 0;
    engine.connectModulation(change);

    // The bank hands out slots in order, so slot i is `modulation_(i+1)`.
    std::string amount_name = "modulation_" + std::to_string(i + 1) + "_amount";
    auto found = controls.find(amount_name);
    if (found != controls.end())
      found->second->set(wanted.amount);
  }

  const int block_size = vital::kMaxBufferSize;
  const int total_samples = static_cast<int>(test_case.seconds * test_case.sample_rate);
  std::vector<float> interleaved;
  interleaved.reserve(static_cast<size_t>(total_samples) * 2);

  int position = 0;
  while (position < total_samples) {
    int block = std::min(block_size, total_samples - position);

    // Note events land on their exact sample inside the block, the way a
    // host delivers them, so onsets line up with the Rust side.
    for (const auto& note : test_case.notes) {
      int on = static_cast<int>(note.start_seconds * test_case.sample_rate);
      int off = static_cast<int>((note.start_seconds + note.hold_seconds) * test_case.sample_rate);
      if (on >= position && on < position + block)
        engine.noteOn(note.midi, note.velocity, on - position, 0);
      if (off >= position && off < position + block)
        engine.noteOff(note.midi, 0.5f, off - position, 0);
    }

    engine.process(block);

    const vital::mono_float* output = (const vital::mono_float*)engine.output(0)->buffer;
    for (int i = 0; i < block; ++i) {
      interleaved.push_back(output[vital::poly_float::kSize * i]);
      interleaved.push_back(output[vital::poly_float::kSize * i + 1]);
    }
    position += block;
  }

  // The skipped window is rendered, because the engine's state depends on
  // it, but not written: nothing reads it, and the corpus is committed.
  size_t skipped = static_cast<size_t>(test_case.skip_seconds * test_case.sample_rate) * 2;
  skipped = std::min(skipped, interleaved.size());

  std::ofstream out(argv[2], std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "vital_golden: cannot write %s\n", argv[2]);
    return 1;
  }
  out.write(reinterpret_cast<const char*>(interleaved.data() + skipped),
            static_cast<std::streamsize>((interleaved.size() - skipped) * sizeof(float)));
  return out ? 0 : 1;
}
