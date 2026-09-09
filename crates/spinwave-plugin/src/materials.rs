//! Oscillator-slot material helpers shared by the plugin's live channel and
//! the MCP server: sample <-> `.vital` JSON codecs, wavetable JSON export,
//! SFZ instrument building, a small render cache for embedded wavetables,
//! and the float32 WAV writer.
//!
//! Materials live INSIDE the preset (`settings.sample`,
//! `settings.wavetables[slot]`, `settings.spinwave_materials.slots[slot]`),
//! so `get_patch` / DAW state / `save_preset` all read one source of truth.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use spinwave_dsp::oscillator::sample_source::BUFFER_SAMPLES;
use spinwave_dsp::oscillator::{Multisample, MultisampleSource, Sample};
use spinwave_dsp::wavetable::creator::wavetable_from_json;
use spinwave_dsp::wavetable::Wavetable;
use spinwave_params::base64;
use spinwave_params::preset::{SampleJson, SfzMaterial};
use spinwave_params::Preset;

/// Oscillator slots the engine has (`osc_1` .. `osc_4`).
pub const NUM_SLOTS: usize = spinwave_engine::kernel::mod_matrix::NUM_OSCILLATORS;

/// Slots whose wavetable Vital stores in `settings.wavetables` (one per
/// Vital oscillator); the remaining slots use `spinwave_materials`.
pub const NUM_VITAL_WAVETABLE_SLOTS: usize = spinwave_params::constants::NUM_OSCILLATORS;

/// Keyframe cap when exporting a rendered wavetable as a Wave Source (a
/// 257-frame table keeps every 4th frame; the creator interpolates).
const MAX_EXPORT_KEYFRAMES: usize = 65;

/// Number of rendered wavetables kept by [`cached_wavetable`].
const WAVETABLE_CACHE_SIZE: usize = 8;

// -- Channels ---------------------------------------------------------------

/// Splits interleaved stereo into `(left, right)`.
#[must_use]
pub fn deinterleave(stereo: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let left = stereo.iter().step_by(2).copied().collect();
    let right = stereo.iter().skip(1).step_by(2).copied().collect();
    (left, right)
}

/// The original-rate channels of a sample read back through the tier
/// accessors (tier 1 is the unfiltered original, guarded by
/// [`BUFFER_SAMPLES`] on each side): `(left, Some(right))` for stereo.
#[must_use]
pub fn sample_channels(sample: &Sample) -> (Vec<f32>, Option<Vec<f32>>) {
    let length = sample.original_length();
    if length == 0 {
        return (Vec::new(), None);
    }
    let left = sample.left_buffer(1)[BUFFER_SAMPLES..BUFFER_SAMPLES + length].to_vec();
    let right = sample
        .stereo()
        .then(|| sample.right_buffer(1)[BUFFER_SAMPLES..BUFFER_SAMPLES + length].to_vec());
    (left, right)
}

/// The original-rate audio of a sample folded to mono.
#[must_use]
pub fn sample_mono(sample: &Sample) -> Vec<f32> {
    let (left, right) = sample_channels(sample);
    match right {
        Some(right) => left.iter().zip(&right).map(|(l, r)| 0.5 * (l + r)).collect(),
        None => left,
    }
}

// -- Sample <-> .vital JSON -------------------------------------------------

/// Encodes a sample in Vital's `settings.sample` format.
#[must_use]
pub fn sample_to_json(sample: &Sample) -> SampleJson {
    let (left, right) = sample_channels(sample);
    SampleJson::from_channels(&sample.name, &left, right.as_deref(), sample.sample_rate())
}

/// Decodes a `settings.sample` payload into a sample (builds the
/// band-limited pyramid; call off the audio thread).
#[must_use]
pub fn sample_from_json(payload: &SampleJson) -> Option<Sample> {
    let (left, right) = payload.decode()?;
    let rate = if payload.sample_rate == 0 { 44100 } else { payload.sample_rate };
    Some(match right {
        Some(right) => Sample::from_stereo(&payload.name, &left, &right, rate),
        None => Sample::from_mono(&payload.name, &left, rate),
    })
}

// -- Wavetable -> creator JSON ----------------------------------------------

/// Serializes a rendered wavetable as a `.vital` wavetable-creator state:
/// one Wave Source whose keyframes hold the frames' time-domain data
/// (float32 base64, like `WaveSourceKeyframe::stateToJson`). Frames are
/// thinned to at most [`MAX_EXPORT_KEYFRAMES`] with linear interpolation
/// between them.
#[must_use]
pub fn wavetable_to_json(wavetable: &Wavetable, name: &str) -> Value {
    let num_frames = wavetable.num_frames().max(1);
    let stride = num_frames.div_ceil(MAX_EXPORT_KEYFRAMES).max(1);
    let mut keyframes = Vec::new();
    let mut frame = 0usize;
    while frame < num_frames {
        keyframes.push(serde_json::json!({
            "position": frame,
            "wave_data": base64::encode_f32(wavetable.data().wave_data(frame)),
        }));
        frame += stride;
    }
    if num_frames > 1 && !(num_frames - 1).is_multiple_of(stride) {
        keyframes.push(serde_json::json!({
            "position": num_frames - 1,
            "wave_data": base64::encode_f32(wavetable.data().wave_data(num_frames - 1)),
        }));
    }
    serde_json::json!({
        "name": name,
        "author": "spinwave",
        "version": spinwave_params::migrate::CURRENT_FORMAT_VERSION,
        "remove_all_dc": false,
        "full_normalize": false,
        "groups": [{
            "components": [{
                "type": "Wave Source",
                "interpolation_style": if keyframes.len() > 1 { 1 } else { 0 },
                "interpolation": 1,
                "keyframes": keyframes,
            }]
        }]
    })
}

fn hash_json(value: &Value) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Serialization is deterministic for a given Value (map keys sorted).
    serde_json::to_string(value).unwrap_or_default().hash(&mut hasher);
    hasher.finish()
}

static WAVETABLE_CACHE: Mutex<Vec<(u64, Arc<Wavetable>)>> = Mutex::new(Vec::new());

/// Renders a wavetable-creator JSON payload, memoized on the payload's
/// content so re-applying a preset (every live `set_params`) does not
/// re-render its embedded tables. Never called on the audio thread.
#[must_use]
pub fn cached_wavetable(value: &Value) -> Option<Arc<Wavetable>> {
    let key = hash_json(value);
    if let Ok(cache) = WAVETABLE_CACHE.lock() {
        if let Some((_, table)) = cache.iter().find(|(hash, _)| *hash == key) {
            return Some(table.clone());
        }
    }
    let table = Arc::new(wavetable_from_json(value)?);
    if let Ok(mut cache) = WAVETABLE_CACHE.lock() {
        cache.insert(0, (key, table.clone()));
        cache.truncate(WAVETABLE_CACHE_SIZE);
    }
    Some(table)
}

static SAMPLE_CACHE: Mutex<Vec<(u64, Arc<Sample>)>> = Mutex::new(Vec::new());

/// Decodes an embedded sample payload, memoized on its content (the
/// pyramid build is the expensive part). Never called on the audio thread.
#[must_use]
pub fn cached_sample(payload: &SampleJson) -> Option<Arc<Sample>> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    payload.samples.hash(&mut hasher);
    payload.samples_stereo.hash(&mut hasher);
    payload.sample_rate.hash(&mut hasher);
    payload.name.hash(&mut hasher);
    let key = hasher.finish();
    if let Ok(cache) = SAMPLE_CACHE.lock() {
        if let Some((_, sample)) = cache.iter().find(|(hash, _)| *hash == key) {
            return Some(sample.clone());
        }
    }
    let sample = Arc::new(sample_from_json(payload)?);
    if let Ok(mut cache) = SAMPLE_CACHE.lock() {
        cache.insert(0, (key, sample.clone()));
        cache.truncate(WAVETABLE_CACHE_SIZE);
    }
    Some(sample)
}

// -- SFZ --------------------------------------------------------------------

/// Decoded audio for one SFZ zone: `(left, right, sample_rate)`; mono
/// zones set `right` to `None`.
pub type ZoneFrames = (Vec<f32>, Option<Vec<f32>>, u32);

/// Parses SFZ text into one `Multisample`. `decode` resolves a zone's
/// `sample` opcode (already joined to `base_dir`) to audio; each file
/// decodes once. Zone material is shared, so the result clones cheaply.
pub fn multisample_from_sfz(
    text: &str,
    base_dir: &Path,
    mut decode: impl FnMut(&Path) -> Option<ZoneFrames>,
) -> Result<Multisample, String> {
    let mut cache: HashMap<String, Option<ZoneFrames>> = HashMap::new();
    Multisample::from_sfz(text, |sample_path| {
        let frames = cache
            .entry(sample_path.to_string())
            .or_insert_with(|| decode(&base_dir.join(sample_path.replace('\\', "/"))))
            .as_ref()?;
        Some(match &frames.1 {
            Some(right) => Sample::from_stereo(sample_path, &frames.0, right, frames.2),
            None => Sample::from_mono(sample_path, &frames.0, frames.2),
        })
    })
    .map_err(|e| format!("invalid SFZ: {e}"))
}

/// Builds `count` playback sources from SFZ text (one per voice kernel:
/// the per-zone playback state is private, the zone audio is shared).
/// Allocating here keeps it off the audio thread.
pub fn multisample_sources_from_sfz(
    text: &str,
    base_dir: &Path,
    count: usize,
    decode: impl FnMut(&Path) -> Option<ZoneFrames>,
) -> Result<Vec<MultisampleSource>, String> {
    let instrument = multisample_from_sfz(text, base_dir, decode)?;
    let mut sources = Vec::with_capacity(count);
    for _ in 1..count {
        sources.push(MultisampleSource::new(instrument.clone()));
    }
    sources.push(MultisampleSource::new(instrument));
    Ok(sources)
}

/// Zone decoder reading WAV files through the engine's own parser (PCM16
/// / float32 only) — what the plugin uses. Decoded files are memoized on
/// (path, modification time) so re-applying a preset does not re-read
/// them.
#[must_use]
pub fn decode_wav_zone(path: &Path) -> Option<ZoneFrames> {
    cached_zone(path, |path| {
        let bytes = std::fs::read(path).ok()?;
        let sample = Sample::from_wav_bytes(&bytes).ok()?;
        let (left, right) = sample_channels(&sample);
        Some((left, right, sample.sample_rate()))
    })
}

/// (path, modification time, decoded frames).
type ZoneCacheEntry = (std::path::PathBuf, Option<std::time::SystemTime>, Arc<ZoneFrames>);

static ZONE_CACHE: Mutex<Vec<ZoneCacheEntry>> = Mutex::new(Vec::new());

/// Memoizes a zone decoder on (path, mtime); at most [`ZONE_CACHE_SIZE`]
/// files. The frames are cloned out (the multisample builder needs owned
/// channels per instance anyway).
pub fn cached_zone(
    path: &Path,
    decode: impl FnOnce(&Path) -> Option<ZoneFrames>,
) -> Option<ZoneFrames> {
    let modified = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    if let Ok(cache) = ZONE_CACHE.lock() {
        if let Some((_, _, frames)) =
            cache.iter().find(|(p, m, _)| p == path && *m == modified)
        {
            return Some((**frames).clone());
        }
    }
    let frames = Arc::new(decode(path)?);
    if let Ok(mut cache) = ZONE_CACHE.lock() {
        cache.retain(|(p, _, _)| p != path);
        cache.insert(0, (path.to_path_buf(), modified, frames.clone()));
        cache.truncate(ZONE_CACHE_SIZE);
    }
    Some((*frames).clone())
}

/// Number of decoded SFZ zone files kept by [`cached_zone`].
pub const ZONE_CACHE_SIZE: usize = 64;

/// Reads an SFZ file into a preset material descriptor.
pub fn sfz_material_from_path(path: &str) -> Result<SfzMaterial, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
    Ok(SfzMaterial { path: path.to_string(), text })
}

/// The directory an SFZ material's sample opcodes resolve against.
#[must_use]
pub fn sfz_base_dir(material: &SfzMaterial) -> std::path::PathBuf {
    Path::new(&material.path).parent().map(Path::to_path_buf).unwrap_or_default()
}

// -- Preset material accessors --------------------------------------------

/// The wavetable JSON stored for a slot: `settings.wavetables[slot]` for
/// Vital's slots, `spinwave_materials.slots[slot].wavetable` beyond.
#[must_use]
pub fn slot_wavetable_json(preset: &Preset, slot: usize) -> Option<&Value> {
    if slot < NUM_VITAL_WAVETABLE_SLOTS {
        preset.settings.wavetables.as_ref()?.as_array()?.get(slot)
    } else {
        preset.settings.spinwave_materials.as_ref()?.slot(slot)?.wavetable.as_ref()
    }
}

/// Stores a slot's wavetable JSON where [`slot_wavetable_json`] reads it.
pub fn set_slot_wavetable_json(preset: &mut Preset, slot: usize, table: Value) {
    if slot < NUM_VITAL_WAVETABLE_SLOTS {
        let mut tables = match preset.settings.wavetables.take() {
            Some(Value::Array(tables)) => tables,
            _ => Vec::new(),
        };
        while tables.len() <= slot {
            tables.push(Value::Null);
        }
        tables[slot] = table;
        preset.settings.wavetables = Some(Value::Array(tables));
    } else {
        preset
            .settings
            .spinwave_materials
            .get_or_insert_with(Default::default)
            .slot_mut(slot)
            .wavetable = Some(table);
    }
}

/// Stores a slot's sample payload in `spinwave_materials`.
pub fn set_slot_sample_json(preset: &mut Preset, slot: usize, sample: SampleJson) {
    preset
        .settings
        .spinwave_materials
        .get_or_insert_with(Default::default)
        .slot_mut(slot)
        .sample = Some(sample);
}

/// Stores a slot's SFZ material in `spinwave_materials`.
pub fn set_slot_sfz(preset: &mut Preset, slot: usize, sfz: SfzMaterial) {
    preset
        .settings
        .spinwave_materials
        .get_or_insert_with(Default::default)
        .slot_mut(slot)
        .sfz = Some(sfz);
}

// -- WAV --------------------------------------------------------------------

/// Minimal 32-bit float stereo WAV writer.
pub fn write_wav(path: &str, interleaved: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    let data_bytes = (interleaved.len() * 4) as u32;
    let byte_rate = sample_rate * 2 * 4;

    let mut header = Vec::with_capacity(44);
    header.extend_from_slice(b"RIFF");
    header.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    header.extend_from_slice(b"WAVEfmt ");
    header.extend_from_slice(&16u32.to_le_bytes());
    header.extend_from_slice(&3u16.to_le_bytes());
    header.extend_from_slice(&2u16.to_le_bytes());
    header.extend_from_slice(&sample_rate.to_le_bytes());
    header.extend_from_slice(&byte_rate.to_le_bytes());
    header.extend_from_slice(&8u16.to_le_bytes());
    header.extend_from_slice(&32u16.to_le_bytes());
    header.extend_from_slice(b"data");
    header.extend_from_slice(&data_bytes.to_le_bytes());
    file.write_all(&header)?;
    let mut body = Vec::with_capacity(interleaved.len() * 4);
    for value in interleaved {
        body.extend_from_slice(&value.to_le_bytes());
    }
    file.write_all(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_json_round_trip_through_engine_sample() {
        let left: Vec<f32> = (0..300).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let right: Vec<f32> = left.iter().map(|v| -v).collect();
        let sample = Sample::from_stereo("pair", &left, &right, 48000);
        let payload = sample_to_json(&sample);
        assert_eq!(payload.length, 300);
        assert!(payload.samples_stereo.is_some());
        let back = sample_from_json(&payload).unwrap();
        assert_eq!(back.original_length(), 300);
        assert!(back.stereo());
        assert_eq!(back.sample_rate(), 48000);
        let (l, r) = sample_channels(&back);
        for (a, b) in left.iter().zip(&l) {
            assert!((a - b).abs() < 1e-3);
        }
        for (a, b) in right.iter().zip(r.as_ref().unwrap()) {
            assert!((a - b).abs() < 1e-3);
        }
        let mono = sample_mono(&back);
        assert!(mono.iter().all(|v| v.abs() < 1e-3));
    }

    #[test]
    fn wavetable_json_export_renders_back() {
        let table = spinwave_dsp::wavetable::factory::basic_shapes();
        let json = wavetable_to_json(&table, "shapes");
        let keyframes = json["groups"][0]["components"][0]["keyframes"].as_array().unwrap();
        assert!(!keyframes.is_empty() && keyframes.len() <= MAX_EXPORT_KEYFRAMES + 1);
        let rendered = cached_wavetable(&json).expect("renders");
        assert_eq!(rendered.num_frames(), table.num_frames());
        // The first frame comes back as exported.
        let original = table.data().wave_data(0);
        let back = rendered.data().wave_data(0);
        let error: f32 = original.iter().zip(back).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / original.len() as f32;
        assert!(error < 0.05, "mean error {error}");
        // Second lookup hits the cache (same Arc).
        let again = cached_wavetable(&json).unwrap();
        assert!(Arc::ptr_eq(&rendered, &again));
    }

    #[test]
    fn slot_material_placement() {
        let mut preset = Preset::default();
        set_slot_wavetable_json(&mut preset, 1, serde_json::json!({"name": "one"}));
        set_slot_wavetable_json(&mut preset, 3, serde_json::json!({"name": "four"}));
        assert_eq!(slot_wavetable_json(&preset, 1).unwrap()["name"], "one");
        assert!(slot_wavetable_json(&preset, 0).is_some_and(Value::is_null));
        assert_eq!(slot_wavetable_json(&preset, 3).unwrap()["name"], "four");
        // Slot 3 never lands in Vital's array (Vital would crash on a 4th).
        assert_eq!(preset.settings.wavetables.as_ref().unwrap().as_array().unwrap().len(), 2);
        set_slot_sample_json(&mut preset, 2, SampleJson::from_channels("s", &[0.1], None, 44100));
        assert!(preset.settings.spinwave_materials.as_ref().unwrap().slot(2).unwrap().sample.is_some());
    }

    #[test]
    fn sfz_builder_shares_decoded_zones() {
        let dir = std::env::temp_dir().join("spinwave-materials-sfz-test");
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("tone.wav");
        let frames: Vec<f32> = (0..400).flat_map(|i| {
            let v = (i as f32 * 0.2).sin() * 0.4;
            [v, v]
        }).collect();
        write_wav(wav.to_str().unwrap(), &frames, 44100).unwrap();
        let text = "<region> sample=tone.wav lokey=0 hikey=127 pitch_keycenter=60";
        let mut decodes = 0usize;
        let sources = multisample_sources_from_sfz(text, &dir, 3, |path| {
            decodes += 1;
            decode_wav_zone(path)
        })
        .unwrap();
        assert_eq!(sources.len(), 3);
        assert_eq!(decodes, 1, "one decode for all instances");
        assert_eq!(sources[0].zone_count(), 1);
        let _ = std::fs::remove_file(wav);
    }
}
