//! Audio file decoding (WAV/FLAC/OGG/MP3/M4A via symphonia) into
//! interleaved stereo f32 for analysis.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decodes a file to (interleaved stereo, sample_rate). Multi-channel
/// sources fold to stereo; mono duplicates. `start`/`duration` in seconds
/// select a segment (applied after decode).
pub fn decode_file(
    path: &str,
    start: Option<f32>,
    duration: Option<f32>,
) -> Result<(Vec<f32>, u32), String> {
    let file = File::open(path).map_err(|e| format!("cannot open '{path}': {e}"))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(extension);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, stream, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("unsupported format: {e}"))?;
    let mut format = probed.format;

    let track = format
        .default_track()
        .ok_or("no audio track in file")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("no decoder: {e}"))?;

    let mut stereo: Vec<f32> = Vec::new();
    // End of stream (or a decode error) ends the loop — take what we have.
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }
        let Ok(decoded) = decoder.decode(&packet) else { continue };
        append_stereo(&decoded, &mut stereo);
    }

    if stereo.is_empty() {
        return Err("decoded zero samples".into());
    }

    // Segment selection.
    let frames = stereo.len() / 2;
    let start_frame = ((start.unwrap_or(0.0).max(0.0) * sample_rate as f32) as usize).min(frames);
    let end_frame = match duration {
        Some(seconds) => (start_frame + (seconds.max(0.0) * sample_rate as f32) as usize)
            .min(frames),
        None => frames,
    };
    if end_frame <= start_frame {
        return Err("selected segment is empty".into());
    }
    let segment = stereo[start_frame * 2..end_frame * 2].to_vec();
    Ok((segment, sample_rate))
}

fn append_stereo(decoded: &AudioBufferRef, out: &mut Vec<f32>) {
    macro_rules! fold {
        ($buffer:expr, $convert:expr) => {{
            let buffer = $buffer;
            let channels = buffer.spec().channels.count();
            let frames = buffer.frames();
            for frame in 0..frames {
                match channels {
                    1 => {
                        let value = $convert(buffer.chan(0)[frame]);
                        out.push(value);
                        out.push(value);
                    }
                    _ => {
                        // Fold >2 channels into stereo pairs L/R.
                        let mut left = 0.0f32;
                        let mut right = 0.0f32;
                        let mut left_count = 0.0f32;
                        let mut right_count = 0.0f32;
                        for channel in 0..channels {
                            let value = $convert(buffer.chan(channel)[frame]);
                            if channel % 2 == 0 {
                                left += value;
                                left_count += 1.0;
                            } else {
                                right += value;
                                right_count += 1.0;
                            }
                        }
                        out.push(left / left_count.max(1.0));
                        out.push(right / right_count.max(1.0));
                    }
                }
            }
        }};
    }

    match decoded {
        AudioBufferRef::F32(buffer) => fold!(buffer, |v: f32| v),
        AudioBufferRef::F64(buffer) => fold!(buffer, |v: f64| v as f32),
        AudioBufferRef::S16(buffer) => fold!(buffer, |v: i16| v as f32 / 32768.0),
        AudioBufferRef::S32(buffer) => fold!(buffer, |v: i32| v as f32 / 2147483648.0),
        AudioBufferRef::S24(buffer) => {
            fold!(buffer, |v: symphonia::core::sample::i24| v.inner() as f32 / 8388608.0)
        }
        AudioBufferRef::U8(buffer) => fold!(buffer, |v: u8| (v as f32 - 128.0) / 128.0),
        AudioBufferRef::S8(buffer) => fold!(buffer, |v: i8| v as f32 / 128.0),
        AudioBufferRef::U16(buffer) => fold!(buffer, |v: u16| (v as f32 - 32768.0) / 32768.0),
        AudioBufferRef::U24(buffer) => {
            fold!(buffer, |v: symphonia::core::sample::u24| (v.inner() as f32 - 8388608.0)
                / 8388608.0)
        }
        AudioBufferRef::U32(buffer) => {
            fold!(buffer, |v: u32| (v as f64 / 2147483648.0 - 1.0) as f32)
        }
    }
}
