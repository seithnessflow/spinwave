# Builds the golden reference: Vital's own DSP core, driven by a small
# harness, so Spinwave can be compared against it.
#
# This is NOT part of Spinwave. No crate depends on it. It runs once, on
# one machine, to generate the reference audio that gets committed; after
# that the comparison is pure Rust and nobody needs a C++ compiler again.
#
#   cmake -S tools/golden -B tools/golden/build -G Ninja -DVITAL_SRC=<path>
#   cmake --build tools/golden/build

cmake_minimum_required(VERSION 3.20)
project(vital_golden CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimised by default: the reference ships optimised, and the bench is
# comparing against how it actually sounds, not against a debug build.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED VITAL_SRC)
  set(VITAL_SRC "$ENV{USERPROFILE}/vital-reference/src")
endif()
if(NOT EXISTS "${VITAL_SRC}/synthesis/synth_engine/sound_engine.cpp")
  message(FATAL_ERROR
    "VITAL_SRC does not look like Vital's src/ (no sound_engine.cpp): ${VITAL_SRC}")
endif()
message(STATUS "Vital reference sources: ${VITAL_SRC}")

# The DSP core, plus the few files from common/ it needs: the parameter
# table (for defaults), the line generator behind every LFO shape, the
# synth_types value wrappers and the tuning table.
file(GLOB_RECURSE VITAL_DSP_SOURCES "${VITAL_SRC}/synthesis/*.cpp")
# `effects_engine/` is Vital's separate effects-only product. It ships its
# own SoundEngine, which collides with the synth's; the bench wants the
# synth.
list(FILTER VITAL_DSP_SOURCES EXCLUDE REGEX "/effects_engine/")
set(VITAL_COMMON_SOURCES
  "${VITAL_SRC}/common/synth_parameters.cpp"
  "${VITAL_SRC}/common/synth_types.cpp"
  "${VITAL_SRC}/common/line_generator.cpp"
  "${VITAL_SRC}/common/tuning.cpp"
)
# The wavetable creator and its components: what a .vital's embedded
# tables are built with. Editor code, but JUCE-free apart from base64
# (the stub has a real one).
file(GLOB VITAL_WAVETABLE_SOURCES "${VITAL_SRC}/common/wavetable/*.cpp")

add_executable(vital_golden main.cpp reference_load.cpp
  ${VITAL_DSP_SOURCES} ${VITAL_COMMON_SOURCES} ${VITAL_WAVETABLE_SOURCES})

# `stub/` comes first so its JuceHeader.h wins: see the comment there for
# why the real framework is not needed.
target_include_directories(vital_golden PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/stub"
  "${VITAL_SRC}/../third_party"
  "${VITAL_SRC}/common"
  "${VITAL_SRC}/common/wavetable"
  "${VITAL_SRC}/synthesis/effects"
  "${VITAL_SRC}/synthesis/filters"
  "${VITAL_SRC}/synthesis/framework"
  "${VITAL_SRC}/synthesis/lookups"
  "${VITAL_SRC}/synthesis/modulators"
  "${VITAL_SRC}/synthesis/modules"
  "${VITAL_SRC}/synthesis/producers"
  "${VITAL_SRC}/synthesis/synth_engine"
  "${VITAL_SRC}/synthesis/utilities"
  # synth_strings.h lives under interface/ but is used by the effect chain
  # for its effect names.
  "${VITAL_SRC}/interface/look_and_feel"
)

if(MSVC)
  # /fp:fast and SSE2 match how Vital ships: the point of the bench is to
  # compare against the reference as built, not against a stricter build
  # of it.
  # MSVC never defines __SSE2__, which is how poly_values.h picks its
  # intrinsics; x64 always has SSE2, and the shipped Windows build defines
  # it the same way.
  target_compile_definitions(vital_golden PRIVATE __SSE2__=1)
  target_compile_options(vital_golden PRIVATE /fp:fast /W0
    "/FI${CMAKE_CURRENT_SOURCE_DIR}/stub/vital_prelude.h")
else()
  target_compile_options(vital_golden PRIVATE -ffast-math -w -msse2)
endif()
