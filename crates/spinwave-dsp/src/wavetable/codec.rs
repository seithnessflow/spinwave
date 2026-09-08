//! Small encoding helpers for `.vital` wavetable payloads: base64, PCM
//! sample packing and the Mersenne Twister the reference uses for
//! vocoder phase randomization.

/// Decodes standard base64 (`+`, `/`, optional `=` padding). Whitespace is
/// skipped; any other invalid character aborts the decode.
pub(crate) fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn value(byte: u8) -> Option<u32> {
        match byte {
            b'A'..=b'Z' => Some((byte - b'A') as u32),
            b'a'..=b'z' => Some((byte - b'a') as u32 + 26),
            b'0'..=b'9' => Some((byte - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    let mut output = Vec::with_capacity(input.len() / 4 * 3);
    let mut accumulator: u32 = 0;
    let mut bits = 0u32;
    for &byte in input.as_bytes() {
        if byte.is_ascii_whitespace() || byte == b'=' {
            continue;
        }
        accumulator = (accumulator << 6) | value(byte)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((accumulator >> bits) as u8);
        }
    }
    Some(output)
}

/// Encodes bytes as standard base64 with padding (test helper for
/// building `.vital`-style payloads).
#[cfg(test)]
pub(crate) fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        output.push(ALPHABET[(triple >> 18) as usize & 63] as char);
        output.push(ALPHABET[(triple >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            output.push(ALPHABET[(triple >> 6) as usize & 63] as char);
        } else {
            output.push('=');
        }
        if chunk.len() > 2 {
            output.push(ALPHABET[triple as usize & 63] as char);
        } else {
            output.push('=');
        }
    }
    output
}

/// Reinterprets little-endian bytes as `f32` samples.
pub(crate) fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Reinterprets bytes as 16-bit PCM and scales to float like the
/// reference's `pcmToFloatData` (`1 / 32767`).
pub(crate) fn pcm_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    const SCALE: f32 = 1.0 / 32767.0;
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 * SCALE)
        .collect()
}

/// Packs `f32` samples as little-endian bytes (test helper).
#[cfg(test)]
pub(crate) fn f32_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// Minimal MT19937 (matches `std::mt19937`), used to reproduce the
/// reference's seeded vocode phase randomization.
pub(crate) struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub(crate) fn new(seed: u32) -> Mt19937 {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut next = y >> 1;
            if y & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = self.state[(i + 397) % 624] ^ next;
        }
        self.index = 0;
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Uniform float in `[min, max)`, mirroring
    /// `std::uniform_real_distribution<float>` closely enough for random
    /// phase generation.
    pub(crate) fn next_in_range(&mut self, min: f32, max: f32) -> f32 {
        let unit = self.next_u32() as f64 / (u32::MAX as f64 + 1.0);
        (unit * (max as f64 - min as f64) + min as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trip() {
        let data: Vec<u8> = (0..=255u8).collect();
        for len in [0, 1, 2, 3, 4, 255, 256] {
            let encoded = base64_encode(&data[..len]);
            assert_eq!(base64_decode(&encoded).unwrap(), &data[..len]);
        }
    }

    #[test]
    fn base64_known_vector() {
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_decode("TWFuTQ==").unwrap(), b"ManM");
        assert!(base64_decode("bad*data").is_none());
    }

    #[test]
    fn float_bytes_round_trip() {
        let samples = [0.0f32, 1.0, -1.0, 0.25, f32::MIN_POSITIVE];
        let bytes = f32_to_bytes(&samples);
        assert_eq!(bytes_to_f32(&bytes), samples);
    }

    #[test]
    fn pcm_decodes_full_scale() {
        let bytes = [0xff, 0x7f, 0x01, 0x80]; // 32767, -32767
        let floats = pcm_bytes_to_f32(&bytes);
        assert!((floats[0] - 1.0).abs() < 1e-6);
        assert!((floats[1] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn mt19937_matches_reference_first_output() {
        // First output of std::mt19937 seeded with 5489 is 3499211612.
        let mut rng = Mt19937::new(5489);
        assert_eq!(rng.next_u32(), 3_499_211_612);
    }
}
