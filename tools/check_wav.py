#!/usr/bin/env python3
"""Quick WAV sanity checks for Spinwave renders (stdlib only).

Usage: python tools/check_wav.py render.wav [freq_hz ...]

Prints peak/RMS/DC and, for each requested frequency, the Goertzel power
relative to the strongest requested bin — enough to confirm that expected
note fundamentals are present without numpy.
"""

import math
import struct
import sys
import wave


def read_wav(path):
    # Try the stdlib reader first (PCM); fall back to raw float32 parsing.
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if width == 2:
            samples = [s / 32768.0 for s in struct.unpack(f"<{len(raw)//2}h", raw)]
        else:
            samples = list(struct.unpack(f"<{len(raw)//4}f", raw))
        return rate, channels, samples
    except wave.Error:
        with open(path, "rb") as f:
            data = f.read()
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a WAV"
        # Minimal parse: find fmt and data chunks.
        pos = 12
        rate, channels, fmt = 44100, 2, 3
        while pos + 8 <= len(data):
            chunk_id = data[pos : pos + 4]
            size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
            body = data[pos + 8 : pos + 8 + size]
            if chunk_id == b"fmt ":
                fmt, channels, rate = struct.unpack("<HHI", body[:8])
            elif chunk_id == b"data":
                assert fmt == 3, "expected float32 WAV"
                samples = list(struct.unpack(f"<{len(body)//4}f", body))
                return rate, channels, samples
            pos += 8 + size + (size & 1)
        raise ValueError("no data chunk")


def goertzel(samples, rate, freq):
    k = 2.0 * math.pi * freq / rate
    coeff = 2.0 * math.cos(k)
    s_prev = s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2, s_prev = s_prev, s
    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return power / len(samples)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    freqs = [float(f) for f in sys.argv[2:]]

    rate, channels, samples = read_wav(path)
    mono = [sum(samples[i : i + channels]) / channels for i in range(0, len(samples), channels)]

    peak = max(abs(s) for s in mono)
    rms = math.sqrt(sum(s * s for s in mono) / len(mono))
    dc = sum(mono) / len(mono)
    print(f"{path}: {len(mono)} frames @ {rate} Hz, {channels}ch")
    print(f"peak {peak:.4f}  rms {rms:.4f}  dc {dc:+.6f}")

    if freqs:
        powers = {f: goertzel(mono, rate, f) for f in freqs}
        reference = max(powers.values())
        for f, p in powers.items():
            rel_db = 10.0 * math.log10(max(p / reference, 1e-12))
            print(f"  {f:8.2f} Hz: {rel_db:+7.2f} dB rel")


if __name__ == "__main__":
    main()
