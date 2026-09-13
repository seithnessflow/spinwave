// A stand-in for JUCE, so Vital's DSP core can be compiled on its own.
//
// The golden bench needs `vital::SoundEngine` and nothing else from Vital.
// Across the 68 sources under `src/synthesis` only two files include
// JuceHeader at all, and between them they use exactly one JUCE type:
// `String`. Building the real framework to satisfy that would mean a
// Projucer run and a JUCE compile; this header replaces it in a few lines.
//
// If a future file needs more of JUCE, the compiler says so immediately
// rather than silently linking something different: that is the point of
// keeping the stub minimal instead of pulling in a compatibility layer.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

/// Vital's two DSP-side uses are string literals for wavetable and preset
/// names, so a thin wrapper over std::string covers them.
class String {
  public:
    String() {}
    String(const char* text) : text_(text ? text : "") {}
    String(const std::string& text) : text_(text) {}

    const char* toRawUTF8() const { return text_.c_str(); }
    std::string toStdString() const { return text_; }
    operator std::string() const { return text_; }

    bool isEmpty() const { return text_.empty(); }
    bool operator==(const String& other) const { return text_ == other.text_; }
    String operator+(const String& other) const { return String(text_ + other.text_); }

    // The Scala parser in tuning.cpp uses exactly these.
    bool contains(const String& part) const {
      return text_.find(part.text_) != std::string::npos;
    }

    float getFloatValue() const {
      try {
        return std::stof(text_);
      } catch (...) {
        return 0.0f;
      }
    }

    int getIntValue() const {
      try {
        return std::stoi(text_);
      } catch (...) {
        return 0;
      }
    }

    String trim() const {
      size_t first = text_.find_first_not_of(" \t\r\n");
      if (first == std::string::npos)
        return String();
      size_t last = text_.find_last_not_of(" \t\r\n");
      return String(text_.substr(first, last - first + 1));
    }

    String substring(int start) const {
      if (start < 0 || static_cast<size_t>(start) >= text_.size())
        return String();
      return String(text_.substr(start));
    }

    String substring(int start, int end) const {
      if (start < 0 || end <= start || static_cast<size_t>(start) >= text_.size())
        return String();
      size_t stop = std::min(static_cast<size_t>(end), text_.size());
      return String(text_.substr(start, stop - start));
    }

    String toLowerCase() const {
      std::string lowered = text_;
      std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      return String(lowered);
    }

    String removeCharacters(const String& unwanted) const {
      std::string kept;
      for (char c : text_) {
        if (unwanted.text_.find(c) == std::string::npos)
          kept.push_back(c);
      }
      return String(kept);
    }

    String upToFirstOccurrenceOf(const String& part, bool include, bool /*ignore_case*/) const {
      size_t at = text_.find(part.text_);
      if (at == std::string::npos)
        return *this;
      return String(text_.substr(0, include ? at + part.text_.size() : at));
    }

    String fromLastOccurrenceOf(const String& part, bool include, bool /*ignore_case*/) const {
      size_t at = text_.rfind(part.text_);
      if (at == std::string::npos)
        return *this;
      return String(text_.substr(include ? at : at + part.text_.size()));
    }

    int length() const { return static_cast<int>(text_.size()); }

    char operator[](int index) const {
      if (index < 0 || static_cast<size_t>(index) >= text_.size())
        return 0;
      return text_[index];
    }

    bool operator!=(const String& other) const { return text_ != other.text_; }

  private:
    friend class StringArray;
    std::string text_;
};

/// Vital reads Scala scale files as a list of lines.
class StringArray {
  public:
    StringArray() {}
    int size() const { return static_cast<int>(lines_.size()); }
    const String& operator[](int index) const { return lines_[index]; }
    void add(const String& line) { lines_.push_back(line); }

    /// Splits `text` on any character in `separators`. JUCE also honours
    /// `quotes`; no Scala file the bench reads uses them.
    int addTokens(const String& text, const String& separators, const String& /*quotes*/) {
      std::string current;
      for (char c : text.text_) {
        if (separators.text_.find(c) != std::string::npos) {
          lines_.push_back(String(current));
          current.clear();
        }
        else {
          current.push_back(c);
        }
      }
      lines_.push_back(String(current));
      return size();
    }

    /// JUCE's two-argument overload splits on whitespace.
    int addTokens(const String& text, bool /*preserve_quotes*/) {
      return addTokens(text, String(" \t\r\n"), String());
    }

    /// Range-for over the lines, as JUCE's StringArray supports.
    const String* begin() const { return lines_.data(); }
    const String* end() const { return lines_.data() + lines_.size(); }

  private:
    std::vector<String> lines_;
};

/// JUCE sets the CPU's flush-to-zero flag here. Leaving it a no-op does
/// change the arithmetic: denormals get computed instead of flushed. They
/// are around 1e-38, some thirty orders below the comparison tolerance, and
/// the Rust side does not set the flag either, so both halves treat them
/// the same way.
struct FloatVectorOperations {
    static void disableDenormalisedNumberSupport() {}
};

/// `Tuning` takes JUCE files to load Scala and .tun scales. The bench
/// never loads one (it renders in equal temperament, like the Rust side's
/// default), so this only has to satisfy the declarations.
class File {
  public:
    File() {}
    explicit File(const String& path) : path_(path) {}
    String getFullPathName() const { return path_; }
    String getFileNameWithoutExtension() const { return path_; }
    String getFileExtension() const {
      return path_.fromLastOccurrenceOf(String("."), true, false);
    }
    bool exists() const { return false; }
    /// No scale file is ever read: `exists()` is false and the bench
    /// renders in equal temperament, which is `Tuning`'s default.
    StringArray readLines(StringArray& into) const { return into; }

  private:
    String path_;
};

/// `Sample::jsonToState`, `WaveSource::jsonToState` and `FileSource::
/// jsonToState` decode base64 audio from preset files, and the migration
/// re-encodes it. Real implementations (JUCE's Base64 is the standard
/// alphabet with `=` padding; `MemoryOutputStream` grows as written).
class MemoryOutputStream {
  public:
    MemoryOutputStream() {}
    explicit MemoryOutputStream(size_t size) { bytes_.reserve(size); }
    const void* getData() const { return bytes_.data(); }
    size_t getDataSize() const { return bytes_.size(); }
    void write(const void* data, size_t size) {
      const char* bytes = static_cast<const char*>(data);
      bytes_.insert(bytes_.end(), bytes, bytes + size);
    }

  private:
    std::vector<char> bytes_;
};

struct Base64 {
    static String toBase64(const void* data, size_t size) {
      static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      const unsigned char* bytes = static_cast<const unsigned char*>(data);
      std::string out;
      out.reserve((size + 2) / 3 * 4);
      for (size_t i = 0; i < size; i += 3) {
        unsigned int chunk = bytes[i] << 16;
        if (i + 1 < size) chunk |= bytes[i + 1] << 8;
        if (i + 2 < size) chunk |= bytes[i + 2];
        out.push_back(alphabet[(chunk >> 18) & 63]);
        out.push_back(alphabet[(chunk >> 12) & 63]);
        out.push_back(i + 1 < size ? alphabet[(chunk >> 6) & 63] : '=');
        out.push_back(i + 2 < size ? alphabet[chunk & 63] : '=');
      }
      return String(out);
    }
    static bool convertFromBase64(MemoryOutputStream& stream, const std::string& text) {
      auto value = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
      };
      unsigned int chunk = 0;
      int bits = 0;
      for (char c : text) {
        int v = value(c);
        if (v < 0) {
          if (c == '=') break;
          continue;
        }
        chunk = (chunk << 6) | v;
        bits += 6;
        if (bits >= 8) {
          bits -= 8;
          unsigned char byte = (chunk >> bits) & 0xff;
          stream.write(&byte, 1);
        }
      }
      return true;
    }
};

class InputStream;
class MemoryInputStream;

// JUCE's debug helpers. In the real framework these install leak
// tracking and delete the copy constructor; neither affects the audio, so
// the bench compiles them away. The list is exactly what the DSP sources
// use, checked by grepping them.
#define JUCE_LEAK_DETECTOR(ClassName)
#define JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClassName)

// JUCE defines these; the DSP core only ever reads them.
#ifndef JUCE_MSVC
  #if defined(_MSC_VER)
    #define JUCE_MSVC 1
  #endif
#endif

// JUCE's generated project info; the wavetable creator stamps its
// version into the JSON it writes.
namespace ProjectInfo {
  const char* const versionString = "1.0.7";
}
