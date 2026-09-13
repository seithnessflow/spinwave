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
//     vital_golden <case-file> <out.raw> [--probe <source> ...]
//
// `--probe` writes `<out.raw>.probe.csv`: the control-rate value of each
// named modulation source, one row per block, alongside the audio. The
// Rust half reads the same values out of its own engine, so a case that
// diverges can be asked WHERE rather than only how far — the shape of the
// difference names the mechanism. It is a diagnostic: no committed case
// uses it, and it does not touch the audio.
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
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "line_generator.h"
#include "linkwitz_riley_filter.h"
#include "compressor.h"
#include "value.h"
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
  /// A flat LFO shape at this value, instead of the triangle. Set by
  /// `lfo_shape flat <v>`: a horizontal line drawn in the LFO editor.
  /// It makes an LFO that does not move, which separates "the source
  /// varies" from "the source is an LFO" — two things every earlier case
  /// changed together.
  bool lfo_flat = false;
  float lfo_flat_value = 0.0f;
  /// `random_seed <n>`: accepted and NOT applied. Vital seeds each
  /// RandomGenerator from a process-global counter, so the voice's
  /// random_1 holds whatever seed its construction order gave it (18 in
  /// this build, recovered from a `--probe random_1` curve by
  /// tools/golden/random_seed.py). The directive tells the Spinwave side
  /// to use that same seed; here it only documents the case.
  int random_seed = -1;
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
    else if (directive == "random_seed") {
      stream >> result.random_seed;
    }
    else if (directive == "lfo_shape") {
      std::string kind;
      stream >> kind;
      if (kind != "flat") {
        error = "unknown lfo shape '" + kind + "' (only 'flat' so far)";
        return false;
      }
      result.lfo_flat = true;
      stream >> result.lfo_flat_value;
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

// `vital_golden --crossover <cutoff> <rate> <out.raw>`: the reference's
// LinkwitzRileyFilter alone on a deterministic input (an impulse, then a
// 55 Hz saw), low and high outputs interleaved per sample, lane 0. A
// unit-level golden for the compressor's band split, whose 120 Hz
// crossover was the whole of fx_compressor's 1.6e-5 while the filter
// read the same, line for line.
static int runCrossoverProbe(int argc, char* argv[]) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: vital_golden --crossover <cutoff> <rate> <out.raw>\n");
    return 2;
  }
  float cutoff = std::atof(argv[2]);
  int rate = std::atoi(argv[3]);
  vital::LinkwitzRileyFilter filter(cutoff);
  filter.setSampleRate(rate);
  filter.reset(vital::constants::kFullMask);
  const int kSamples = 4096;
  std::vector<vital::poly_float> input(kSamples);
  for (int i = 0; i < kSamples; ++i) {
    float saw = 2.0f * std::fmod(55.0f * i / rate, 1.0f) - 1.0f;
    input[i] = (i == 0) ? 1.0f : 0.5f * saw;
  }
  std::vector<float> out;
  for (int start = 0; start < kSamples; start += 128) {
    filter.processWithInput(input.data() + start, 128);
    const vital::poly_float* low = filter.output(vital::LinkwitzRileyFilter::kAudioLow)->buffer;
    const vital::poly_float* high = filter.output(vital::LinkwitzRileyFilter::kAudioHigh)->buffer;
    for (int i = 0; i < 128; ++i) {
      out.push_back(low[i][0]);
      out.push_back(high[i][0]);
    }
  }
  std::ofstream file(argv[4], std::ios::binary);
  file.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
  return 0;
}

// `vital_golden --compressor <bands> <rate> <out.raw>`: the reference's
// MultibandCompressor alone (ratios and gains at zero, table thresholds,
// attack and release 0.5, mix 1) on the crossover probe's input, lanes 0
// and 1 interleaved.
static int runCompressorProbe(int argc, char* argv[]) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: vital_golden --compressor <bands> <rate> <out.raw>\n");
    return 2;
  }
  int bands = std::atoi(argv[2]);
  int rate = std::atoi(argv[3]);
  vital::MultibandCompressor compressor;
  vital::cr::Value zero(0.0f), half(0.5f), one(1.0f), enabled_bands((float)bands);
  vital::cr::Value low_upper(-28.0f), band_upper(-25.0f), high_upper(-30.0f);
  vital::cr::Value low_lower(-35.0f), band_lower(-36.0f), high_lower(-35.0f);
  vital::Output audio_input(vital::kMaxBufferSize, 2);
  compressor.plug(&audio_input, vital::MultibandCompressor::kAudio);
  for (int i : { vital::MultibandCompressor::kLowUpperRatio, vital::MultibandCompressor::kBandUpperRatio,
                 vital::MultibandCompressor::kHighUpperRatio, vital::MultibandCompressor::kLowLowerRatio,
                 vital::MultibandCompressor::kBandLowerRatio, vital::MultibandCompressor::kHighLowerRatio,
                 vital::MultibandCompressor::kLowOutputGain, vital::MultibandCompressor::kBandOutputGain,
                 vital::MultibandCompressor::kHighOutputGain })
    compressor.plug(&zero, i);
  compressor.plug(&low_upper, vital::MultibandCompressor::kLowUpperThreshold);
  compressor.plug(&band_upper, vital::MultibandCompressor::kBandUpperThreshold);
  compressor.plug(&high_upper, vital::MultibandCompressor::kHighUpperThreshold);
  compressor.plug(&low_lower, vital::MultibandCompressor::kLowLowerThreshold);
  compressor.plug(&band_lower, vital::MultibandCompressor::kBandLowerThreshold);
  compressor.plug(&high_lower, vital::MultibandCompressor::kHighLowerThreshold);
  compressor.plug(&half, vital::MultibandCompressor::kAttack);
  compressor.plug(&half, vital::MultibandCompressor::kRelease);
  compressor.plug(&enabled_bands, vital::MultibandCompressor::kEnabledBands);
  compressor.plug(&one, vital::MultibandCompressor::kMix);
  compressor.setSampleRate(rate);
  compressor.reset(vital::constants::kFullMask);
  const int kSamples = 4096;
  std::vector<float> out;
  for (int start = 0; start < kSamples; start += 128) {
    for (int i = 0; i < 128; ++i) {
      int n = start + i;
      float saw = 2.0f * std::fmod(55.0f * n / rate, 1.0f) - 1.0f;
      float v = (n == 0) ? 1.0f : 0.5f * saw;
      audio_input.buffer[i] = vital::poly_float(v, v, v, v);
    }
    compressor.processWithInput(audio_input.buffer, 128);
    const vital::poly_float* dest = compressor.output(vital::MultibandCompressor::kAudioOut)->buffer;
    for (int i = 0; i < 128; ++i) {
      out.push_back(dest[i][0]);
      out.push_back(dest[i][1]);
    }
  }
  std::ofstream file(argv[4], std::ios::binary);
  file.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc >= 2 && std::strcmp(argv[1], "--crossover") == 0)
    return runCrossoverProbe(argc, argv);
  if (argc >= 2 && std::strcmp(argv[1], "--compressor") == 0)
    return runCompressorProbe(argc, argv);
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

  std::vector<std::string> probe_names;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--probe") == 0 && i + 1 < argc)
      probe_names.push_back(argv[++i]);
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
    if (lfo == nullptr)
      continue;
    lfo->initTriangle();
    if (test_case.lfo_flat) {
      // The y axis is inverted here (initTriangle starts at 1.0 for a
      // value of 0), so a flat line at value v sits at 1 - v.
      float y = 1.0f - test_case.lfo_flat_value;
      lfo->setNumPoints(2);
      lfo->setPoint(0, { 0.0f, y });
      lfo->setPoint(1, { 1.0f, y });
      lfo->render();
    }
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

  // AUDIT. Three times now the harness has been found skipping an
  // initialisation SynthBase does — the wavetable, initTriangle(), and
  // the defaults of controls the wiring creates — and each was found the
  // slow way, one wrong reference at a time. So this asserts the invariant
  // instead: after wiring, every control the case does not name must hold
  // its table default. Anything else is printed and the run aborts. (The
  // amounts are set by the wiring itself, so they are the one exception.)
  //
  // What the audit found the first time it ran is recorded in the notes:
  // it is what made the LFO and random references bipolar while every
  // other source was fine, which no reading of the code had explained.
  auto audit_controls = [&](const char* stage) {
    vital::control_map now = engine.getControls();
    std::vector<std::string> offenders;
    for (auto& entry : now) {
      const std::string& name = entry.first;
      bool case_sets_it = false;
      for (const auto& control : test_case.controls)
        if (control.first == name) case_sets_it = true;
      bool is_amount = name.rfind("modulation_", 0) == 0 &&
                       name.size() > 7 && name.compare(name.size() - 7, 7, "_amount") == 0;
      if (case_sets_it || is_amount)
        continue;
      float expected = vital::Parameters::getDetails(name).default_value;
      float actual = entry.second->value();
      if (actual != expected) {
        char line[256];
        std::snprintf(line, sizeof(line), "  %s = %g (table default %g)", name.c_str(), actual, expected);
        offenders.push_back(line);
      }
    }
    if (!offenders.empty()) {
      std::fprintf(stderr, "vital_golden: %s: %zu control(s) not at their table default:\n",
                   stage, offenders.size());
      for (const std::string& line : offenders)
        std::fprintf(stderr, "%s\n", line.c_str());
    }
    return offenders.empty();
  };

  // First pass: what did the wiring leave behind? Printed, not fatal —
  // this is the diagnostic the fix below answers.
  if (!test_case.modulations.empty())
    audit_controls("after wiring, before re-initialisation");

  // Wiring a connection brings its OWN controls into existence — bipolar,
  // stereo, bypass, created by ModulationConnectionProcessor::init() — and
  // the objects the connection actually reads are not the ones the map
  // held beforehand. So run the whole initialisation again now: every
  // control to its table default, then the case's, then the amounts.
  if (!test_case.modulations.empty()) {
    controls = engine.getControls();
    for (auto& entry : controls)
      entry.second->set(vital::Parameters::getDetails(entry.first).default_value);
    for (const auto& control : test_case.controls) {
      auto found = controls.find(control.first);
      if (found != controls.end())
        found->second->set(control.second);
    }
    for (size_t i = 0; i < test_case.modulations.size(); ++i) {
      std::string amount_name = "modulation_" + std::to_string(i + 1) + "_amount";
      auto found = controls.find(amount_name);
      if (found != controls.end())
        found->second->set(test_case.modulations[i].amount);
    }
  }

  // Second pass: the invariant. Fatal.
  if (!audit_controls("after initialisation")) {
    std::fprintf(stderr, "vital_golden: refusing to render on a mis-initialised engine\n");
    return 1;
  }

  // Modulation sources are read after each block, so a probe reading is
  // the value that block was rendered with.
  //
  // Read the STATUS output, not `getModulationSource(name)->buffer`. Poly
  // sources live in the voice graph, which the voice handler CLONES per
  // aggregate voice: the Output the source map hands out belongs to the
  // template processor, which no voice ever writes. Reading it gives a
  // plausible-looking curve that is not the one driving the audio —
  // envelopes read a flat zero through a sounding note, which is what
  // caught it. `SynthVoiceHandler::process` updates the status outputs
  // from the active voice mask right after the voices run; that is the
  // readout Vital's own interface draws, and it is the one that is true.
  std::vector<const vital::StatusOutput*> probes;
  for (const std::string& name : probe_names) {
    const vital::StatusOutput* source = engine.getStatusOutput(name);
    if (source == nullptr) {
      std::fprintf(stderr, "vital_golden: no status output for '%s'\n", name.c_str());
      return 1;
    }
    probes.push_back(source);
  }
  std::vector<std::vector<float>> probe_curves(probes.size());
  // The probe's own witness channel; see the self-test below.
  const vital::StatusOutput* witness_output = engine.getStatusOutput("env_1");
  std::vector<float> witness_curve;

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
    if (witness_output && !probes.empty()) {
      float witness = witness_output->value()[0];
      witness_curve.push_back(witness_output->isClearValue(witness) ? 0.0f : witness);
    }
    for (size_t p = 0; p < probes.size(); ++p) {
      // With no voice active the status outputs hold a sentinel rather
      // than a value; write it as `nan` so nothing averages it in.
      float value = probes[p]->value()[0];
      probe_curves[p].push_back(probes[p]->isClearValue(value)
                                    ? std::numeric_limits<float>::quiet_NaN()
                                    : value);
    }
    position += block;
  }

  // Self-test: an instrument that returns a plausible but wrong curve is
  // the worst outcome, and this one already did it once (reading the
  // template processor's output instead of the voice's). env_1 runs on
  // every voice and is non-zero through any sounding note, so if the
  // witness channel is flat the readout is broken and every number below
  // it is worthless. Fail loudly rather than print them.
  if (!probes.empty()) {
    const vital::StatusOutput* witness = engine.getStatusOutput("env_1");
    bool witness_moved = false;
    if (witness == nullptr) {
      std::fprintf(stderr, "vital_golden: no env_1 status output to check the probe against\n");
      return 1;
    }
    for (float value : witness_curve) {
      if (std::isfinite(value) && value > 1e-6f) {
        witness_moved = true;
        break;
      }
    }
    if (!witness_moved) {
      std::fprintf(stderr,
                   "vital_golden: probe self-test FAILED - env_1 never left zero through a "
                   "sounding note, so the readout is not the one driving the audio. Refusing "
                   "to write probe output.\n");
      return 1;
    }
  }

  if (!probes.empty()) {
    std::string probe_path = std::string(argv[2]) + ".probe.csv";
    std::ofstream probe_out(probe_path);
    if (!probe_out) {
      std::fprintf(stderr, "vital_golden: cannot write %s\n", probe_path.c_str());
      return 1;
    }
    probe_out << "block";
    for (const std::string& name : probe_names)
      probe_out << "," << name;
    probe_out << "\n";
    for (size_t row = 0; row < probe_curves[0].size(); ++row) {
      probe_out << row;
      for (const auto& curve : probe_curves)
        probe_out << "," << curve[row];
      probe_out << "\n";
    }
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
