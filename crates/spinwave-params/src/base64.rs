//! Minimal standard base64 (RFC 4648, `+` `/`, `=` padding) for the binary
//! payloads `.vital` presets embed: PCM16 samples (`settings.sample`) and
//! wavetable keyframes (`wave_data`). Decoding tolerates missing padding
//! and whitespace, like JUCE's `Base64::convertFromBase64`.

const ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encodes `bytes` as padded base64 text.
#[must_use]
pub fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(triple >> 18) as usize & 63] as char);
        out.push(ALPHABET[(triple >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[(triple >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[triple as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn decode_char(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((c - b'a') as u32 + 26),
        b'0'..=b'9' => Some((c - b'0') as u32 + 52),
        b'+' | b'-' => Some(62),
        b'/' | b'_' => Some(63),
        _ => None,
    }
}

/// Decodes base64 text; whitespace is skipped, padding is optional.
/// Returns `None` on any other invalid character.
#[must_use]
pub fn decode(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    for &c in text.as_bytes() {
        if c == b'=' {
            break;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        let value = decode_char(c)?;
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((accumulator >> bits) as u8);
            accumulator &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Decodes little-endian PCM16 base64 (Vital's `samples` /
/// `samples_stereo` fields) into `[-1, 1]` floats.
#[must_use]
pub fn decode_pcm16(text: &str) -> Option<Vec<f32>> {
    let bytes = decode(text)?;
    Some(
        bytes
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
            .collect(),
    )
}

/// Encodes floats as little-endian PCM16 base64 (`utils::floatToPcmData`
/// then `Base64::toBase64`).
#[must_use]
pub fn encode_pcm16(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &value in samples {
        let pcm = (value.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        bytes.extend_from_slice(&pcm.to_le_bytes());
    }
    encode(&bytes)
}

/// Decodes little-endian float32 base64 (pre-0.3.7 sample payloads and
/// wavetable `wave_data`).
#[must_use]
pub fn decode_f32(text: &str) -> Option<Vec<f32>> {
    let bytes = decode(text)?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
            .collect(),
    )
}

/// Encodes floats as little-endian float32 base64.
#[must_use]
pub fn encode_f32(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for &value in samples {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    encode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_bytes() {
        for len in 0..12usize {
            let bytes: Vec<u8> = (0..len as u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();
            let text = encode(&bytes);
            assert_eq!(text.len() % 4, 0);
            assert_eq!(decode(&text).unwrap(), bytes, "len {len}");
        }
        assert_eq!(encode(b"Man"), "TWFu");
        assert_eq!(encode(b"Ma"), "TWE=");
        assert_eq!(encode(b"M"), "TQ==");
        assert_eq!(decode("TWE").unwrap(), b"Ma");
        assert_eq!(decode("TW Fu\n").unwrap(), b"Man");
        assert!(decode("TW$u").is_none());
    }

    #[test]
    fn pcm16_and_f32_round_trip() {
        let samples = [0.0f32, 0.5, -0.5, 0.999, -1.0];
        let pcm = decode_pcm16(&encode_pcm16(&samples)).unwrap();
        for (a, b) in samples.iter().zip(&pcm) {
            assert!((a - b).abs() < 1e-4);
        }
        assert_eq!(decode_f32(&encode_f32(&samples)).unwrap(), samples);
    }
}
