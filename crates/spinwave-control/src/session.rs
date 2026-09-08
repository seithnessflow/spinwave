//! The stateful synth session behind the MCP tools: one engine, one
//! current preset, the last render.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;
use spinwave_dsp::oscillator::{Multisample, Sample};
use spinwave_dsp::wavetable::{
    wavetable_from_audio, wavetable_from_png, AudioImportMode, AudioImportOptions,
    ImageImportOptions, Wavetable,
};
use spinwave_engine::engine::SoundEngine;
use spinwave_engine::kernel::mod_matrix::NUM_OSCILLATORS;
use spinwave_params::preset::ModulationConnection;
use spinwave_params::{parameters, Preset};
use spinwave_plugin::apply_preset;
use spinwave_plugin::patch::connections_from_preset;

use crate::analysis::{analyze, Analysis};
use crate::live_client::LiveLink;

pub const SAMPLE_RATE: u32 = 44100;
const MAX_RENDER_SECONDS: f32 = 60.0;

/// Loaded oscillator-slot material (samples, imported wavetables, SFZ
/// instruments), kept so it can be re-applied whenever the engine is
/// rebuilt (each `render` uses a fresh engine) or grows kernels.
#[derive(Default)]
struct Materials {
    samples: [Option<Arc<Sample>>; NUM_OSCILLATORS],
    wavetables: [Option<Arc<Wavetable>>; NUM_OSCILLATORS],
    /// SFZ text + base directory; `Multisample` is not `Clone`, so each
    /// re-application rebuilds one instance per kernel from this source.
    sfz: [Option<(String, PathBuf)>; NUM_OSCILLATORS],
}

fn check_slot(slot: usize) -> Result<(), String> {
    if slot >= NUM_OSCILLATORS {
        return Err(format!("slot must be 0..{}", NUM_OSCILLATORS - 1));
    }
    Ok(())
}

/// Splits interleaved stereo into (left, right).
fn deinterleave(stereo: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let left = stereo.iter().step_by(2).copied().collect();
    let right = stereo.iter().skip(1).step_by(2).copied().collect();
    (left, right)
}

/// Loads any audio file into a [`Sample`]: WAV bytes go straight through
/// the engine's own parser (preserving mono files as mono); everything else
/// (and WAV encodings the minimal parser rejects, e.g. 24-bit PCM) decodes
/// through symphonia to stereo f32.
fn sample_from_file(path: &str) -> Result<Sample, String> {
    let stem = Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "sample".to_string());
    let is_wav = Path::new(path)
        .extension()
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);
    if is_wav {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
        if let Ok(mut sample) = Sample::from_wav_bytes(&bytes) {
            sample.name = stem;
            return Ok(sample);
        }
    }
    let (stereo, sample_rate) = crate::decode::decode_file(path, None, None)?;
    let (left, right) = deinterleave(&stereo);
    Ok(Sample::from_stereo(&stem, &left, &right, sample_rate))
}

#[derive(Deserialize)]
pub struct NoteSpec {
    pub note: i32,
    /// Start time in seconds.
    pub start: f32,
    /// Duration in seconds.
    pub duration: f32,
    #[serde(default = "default_velocity")]
    pub velocity: f32,
    #[serde(default)]
    pub channel: usize,
}

fn default_velocity() -> f32 {
    0.8
}

pub struct Session {
    pub preset: Preset,
    engine: SoundEngine,
    materials: Materials,
    pub last_render: Option<Vec<f32>>,
    pub last_render_path: Option<String>,
    pub live: LiveLink,
}

impl Session {
    pub fn new() -> Session {
        let preset = Preset::from_json(
            r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#,
        )
        .expect("init preset");
        let mut engine = SoundEngine::new(SAMPLE_RATE);
        apply_preset(&preset, &mut engine);
        Session {
            preset,
            engine,
            materials: Materials::default(),
            last_render: None,
            last_render_path: None,
            live: LiveLink::default(),
        }
    }

    // -- Oscillator-slot material (samples, wavetables, SFZ) ----------------

    /// Loads an audio file into one oscillator slot's Sample/Granular
    /// engines of the offline engine (set `osc_N_engine` to 1 or 2 to hear
    /// it). Any format the analysis decoder reads works (WAV/MP3/FLAC/...).
    pub fn load_sample_offline(&mut self, path: &str, slot: usize) -> Result<String, String> {
        check_slot(slot)?;
        let sample = Arc::new(sample_from_file(path)?);
        let frames = sample.original_length();
        let rate = sample.sample_rate();
        for kernel in self.engine.allocator_mut().kernels_mut() {
            kernel.set_sample(slot, sample.clone());
        }
        self.materials.samples[slot] = Some(sample);
        Ok(format!(
            "loaded '{path}' into osc {} ({frames} frames @ {rate} Hz); set \
             osc_{}_engine to 1 (Sample) or 2 (Granular) to hear it",
            slot + 1,
            slot + 1,
        ))
    }

    /// Imports an audio file (mode `spectral`/`raw`) or a PNG spectrum
    /// (mode `png`) as the slot's wavetable in the offline engine.
    pub fn import_wavetable_offline(
        &mut self,
        path: &str,
        slot: usize,
        mode: &str,
    ) -> Result<String, String> {
        check_slot(slot)?;
        let table = match mode {
            "png" => {
                let bytes =
                    std::fs::read(path).map_err(|e| format!("cannot read '{path}': {e}"))?;
                wavetable_from_png(&bytes, &ImageImportOptions::default())?
            }
            "spectral" | "raw" => {
                let (stereo, sample_rate) = crate::decode::decode_file(path, None, None)?;
                let mono: Vec<f32> =
                    stereo.chunks_exact(2).map(|frame| 0.5 * (frame[0] + frame[1])).collect();
                let import_mode = if mode == "raw" {
                    AudioImportMode::RawSlice
                } else {
                    AudioImportMode::Spectral
                };
                wavetable_from_audio(
                    &mono,
                    sample_rate,
                    &AudioImportOptions { mode: import_mode, ..Default::default() },
                )
            }
            other => return Err(format!("unknown mode: {other} (spectral, raw or png)")),
        };
        let table = Arc::new(table);
        for kernel in self.engine.allocator_mut().kernels_mut() {
            kernel.set_wavetable(slot, table.clone());
        }
        self.materials.wavetables[slot] = Some(table);
        Ok(format!(
            "imported '{path}' ({mode}) as osc {}'s wavetable; the Wavetable \
             engine (osc_{}_engine 0) plays it",
            slot + 1,
            slot + 1,
        ))
    }

    /// Loads an SFZ instrument into one oscillator slot's Multisample
    /// engine of the offline engine (set `osc_N_engine` to 3 to hear it).
    /// Sample opcodes resolve relative to the SFZ file.
    pub fn load_sfz_offline(&mut self, path: &str, slot: usize) -> Result<String, String> {
        check_slot(slot)?;
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read '{path}': {e}"))?;
        let base_dir = Path::new(path).parent().map(|d| d.to_path_buf()).unwrap_or_default();
        self.apply_sfz_to_engine(slot, &text, &base_dir)?;
        self.materials.sfz[slot] = Some((text, base_dir));
        Ok(format!(
            "loaded SFZ '{path}' into osc {}; set osc_{}_engine to 3 \
             (Multisample) to hear it",
            slot + 1,
            slot + 1,
        ))
    }

    /// Builds one `Multisample` per kernel from SFZ source (it is not
    /// `Clone`); referenced audio decodes once into a frame cache, each
    /// kernel's instance rebuilding its zone pyramids from it.
    fn apply_sfz_to_engine(
        &mut self,
        slot: usize,
        text: &str,
        base_dir: &Path,
    ) -> Result<(), String> {
        type Frames = (Vec<f32>, Vec<f32>, u32);
        let mut cache: HashMap<String, Option<Frames>> = HashMap::new();
        let kernel_count = self.engine.allocator().kernels().len();
        let mut instruments = Vec::with_capacity(kernel_count);
        for _ in 0..kernel_count {
            let instrument = Multisample::from_sfz(text, |sample_path| {
                let frames = cache
                    .entry(sample_path.to_string())
                    .or_insert_with(|| {
                        let resolved = base_dir.join(sample_path.replace('\\', "/"));
                        crate::decode::decode_file(&resolved.to_string_lossy(), None, None)
                            .ok()
                            .map(|(stereo, rate)| {
                                let (left, right) = deinterleave(&stereo);
                                (left, right, rate)
                            })
                    })
                    .as_ref()?;
                Some(Sample::from_stereo(sample_path, &frames.0, &frames.1, frames.2))
            })
            .map_err(|e| format!("invalid SFZ: {e}"))?;
            instruments.push(instrument);
        }
        for kernel in self.engine.allocator_mut().kernels_mut() {
            let Some(instrument) = instruments.pop() else { break };
            kernel.set_multisample(slot, instrument);
        }
        Ok(())
    }

    /// Re-installs every loaded material on the current engine's kernels
    /// (after an engine rebuild or a kernel-pool growth).
    fn apply_materials(&mut self) {
        for slot in 0..NUM_OSCILLATORS {
            if let Some(sample) = self.materials.samples[slot].clone() {
                for kernel in self.engine.allocator_mut().kernels_mut() {
                    kernel.set_sample(slot, sample.clone());
                }
            }
            if let Some(table) = self.materials.wavetables[slot].clone() {
                for kernel in self.engine.allocator_mut().kernels_mut() {
                    kernel.set_wavetable(slot, table.clone());
                }
            }
            if let Some((text, base_dir)) = self.materials.sfz[slot].take() {
                // A once-valid SFZ only fails here if its files vanished.
                let _ = self.apply_sfz_to_engine(slot, &text, &base_dir);
                self.materials.sfz[slot] = Some((text, base_dir));
            }
        }
    }

    /// Pushes a sample file to the live instance's slot. Non-WAV audio is
    /// transcoded to a canonical float32 WAV in the temp directory first
    /// (the plugin's minimal parser reads PCM16/float32 WAV only).
    pub fn live_load_sample(&mut self, slot: usize, path: &str) -> Result<String, String> {
        check_slot(slot)?;
        let push_path = wav_for_live(path)?;
        self.live
            .send(&json!({"cmd": "load_sample", "slot": slot, "path": push_path}))?;
        Ok(format!("sample pushed to the live synth (osc {})", slot + 1))
    }

    /// Pushes a wavetable import to the live instance's slot.
    pub fn live_import_wavetable(
        &mut self,
        slot: usize,
        path: &str,
        mode: &str,
    ) -> Result<String, String> {
        check_slot(slot)?;
        let push_path = if mode == "png" { path.to_string() } else { wav_for_live(path)? };
        self.live.send(
            &json!({"cmd": "import_wavetable", "slot": slot, "path": push_path, "mode": mode}),
        )?;
        Ok(format!("wavetable pushed to the live synth (osc {})", slot + 1))
    }

    /// Pushes an SFZ load to the live instance's slot (the plugin resolves
    /// the sample paths itself; WAV files only on the live side).
    pub fn live_load_sfz(&mut self, slot: usize, path: &str) -> Result<String, String> {
        check_slot(slot)?;
        self.live.send(&json!({"cmd": "load_sfz", "slot": slot, "path": path}))?;
        Ok(format!("SFZ pushed to the live synth (osc {})", slot + 1))
    }

    /// Pushes the current preset to the running standalone.
    pub fn live_push_preset(&mut self) -> Result<String, String> {
        let preset_value =
            serde_json::to_value(&self.preset).map_err(|e| e.to_string())?;
        self.live
            .send(&serde_json::json!({"cmd": "preset", "preset": preset_value}))?;
        Ok("patch pushed to the live synth".into())
    }

    /// Pushes an arp/step-sequencer configuration to the live instance.
    /// The plugin's note processor validates and applies it at the next
    /// audio block.
    pub fn live_seq(&mut self, config: serde_json::Value) -> Result<String, String> {
        self.live.send(&serde_json::json!({"cmd": "seq", "config": config}))?;
        Ok("sequencer configured on the live synth".into())
    }

    /// Plays a note sequence on the live synth in (blocking) real time.
    pub fn live_sequence(&mut self, notes: &[NoteSpec]) -> Result<String, String> {
        const MAX_SECONDS: f32 = 20.0;
        if notes.is_empty() {
            return Err("no notes given".into());
        }

        #[derive(PartialEq)]
        enum Kind {
            On,
            Off,
        }
        let mut events: Vec<(f32, Kind, i32, f32, usize)> = Vec::new();
        for spec in notes {
            let start = spec.start.clamp(0.0, MAX_SECONDS);
            let end = (spec.start + spec.duration).clamp(start, MAX_SECONDS);
            events.push((start, Kind::On, spec.note, spec.velocity, spec.channel));
            events.push((end, Kind::Off, spec.note, 0.0, spec.channel));
        }
        events.sort_by(|a, b| a.0.total_cmp(&b.0));

        let started = std::time::Instant::now();
        for (time, kind, note, velocity, channel) in events {
            let target = std::time::Duration::from_secs_f32(time);
            let elapsed = started.elapsed();
            if target > elapsed {
                std::thread::sleep(target - elapsed);
            }
            match kind {
                Kind::On => self.live.note_on(note, velocity.clamp(0.0, 1.0), channel % 16)?,
                Kind::Off => self.live.note_off(note, channel % 16)?,
            };
        }
        Ok(format!("played {} note(s) live", notes.len()))
    }

    /// Re-applies the current preset to the engine (after any edit). Loaded
    /// slot material survives (it lives beside the params); it is only
    /// re-installed when the kernel pool grew (e.g. a polyphony change).
    pub fn sync_engine(&mut self) {
        let kernels_before = self.engine.allocator().kernels().len();
        apply_preset(&self.preset, &mut self.engine);
        if self.engine.allocator().kernels().len() != kernels_before {
            self.apply_materials();
        }
    }

    pub fn load_preset_json(&mut self, text: &str) -> Result<String, String> {
        // Tolerate a UTF-8 BOM (PowerShell's utf8 encoding writes one).
        let text = text.trim_start_matches('\u{feff}');
        let preset = Preset::from_json(text).map_err(|e| format!("invalid preset JSON: {e}"))?;
        self.preset = preset;
        self.sync_engine();
        Ok(format!(
            "loaded preset '{}' ({} set parameters, {} modulations mapped)",
            self.preset.preset_name,
            self.preset.settings.values.len(),
            connections_from_preset(&self.preset).len(),
        ))
    }

    pub fn set_params(&mut self, values: &serde_json::Map<String, serde_json::Value>)
        -> Result<String, String> {
        let table = parameters();
        let mut warnings = Vec::new();
        let mut applied = 0usize;
        for (name, value) in values {
            let Some(value) = value.as_f64() else {
                return Err(format!("'{name}': value must be a number"));
            };
            let value = value as f32;
            match table.lookup(name) {
                Some(details) => {
                    if value < details.min || value > details.max {
                        warnings.push(format!(
                            "'{name}' = {value} clamped to [{}, {}]",
                            details.min, details.max
                        ));
                    }
                    let clamped = value.clamp(details.min, details.max);
                    self.preset.settings.set_parameter(name, clamped);
                    applied += 1;
                }
                None => warnings.push(format!("'{name}' is not a known parameter — skipped")),
            }
        }
        self.sync_engine();
        let mut message = format!("{applied} parameter(s) applied");
        if !warnings.is_empty() {
            message.push_str(&format!("; warnings: {}", warnings.join("; ")));
        }
        Ok(message)
    }

    pub fn add_modulation(
        &mut self,
        source: &str,
        destination: &str,
        amount: f32,
        bipolar: bool,
        stereo: bool,
        power: f32,
    ) -> Result<String, String> {
        let known_sources = spinwave_params::constants::modulation_source_names();
        if !known_sources.iter().any(|s| s == source) {
            return Err(format!(
                "unknown modulation source '{source}'; valid sources: {}",
                known_sources.join(", ")
            ));
        }
        if !parameters().is_parameter(destination) {
            return Err(format!("unknown destination parameter '{destination}'"));
        }

        // First disconnected slot, or append.
        let slot = self
            .preset
            .settings
            .modulations
            .iter()
            .position(|m| !m.is_connected())
            .unwrap_or(self.preset.settings.modulations.len());
        if slot >= self.preset.settings.modulations.len() {
            self.preset.settings.modulations.push(ModulationConnection::default());
        }
        let connection = &mut self.preset.settings.modulations[slot];
        connection.source = source.to_string();
        connection.destination = destination.to_string();

        let n = slot + 1;
        let set = |session: &mut Session, key: String, value: f32| {
            session.preset.settings.set_parameter(&key, value);
        };
        set(self, format!("modulation_{n}_amount"), amount.clamp(-1.0, 1.0));
        set(self, format!("modulation_{n}_bipolar"), bipolar as i32 as f32);
        set(self, format!("modulation_{n}_stereo"), stereo as i32 as f32);
        set(self, format!("modulation_{n}_power"), power);
        set(self, format!("modulation_{n}_bypass"), 0.0);

        self.sync_engine();
        let mapped = connections_from_preset(&self.preset).len();
        Ok(format!(
            "modulation {n}: {source} -> {destination} (amount {amount}); \
             {mapped} connection(s) active in the engine"
        ))
    }

    pub fn clear_modulations(&mut self) -> String {
        let count = self
            .preset
            .settings
            .modulations
            .iter()
            .filter(|m| m.is_connected())
            .count();
        self.preset.settings.modulations.clear();
        self.sync_engine();
        format!("{count} modulation(s) removed")
    }

    pub fn render(
        &mut self,
        notes: &[NoteSpec],
        seconds: Option<f32>,
        bpm: f32,
        out_path: &str,
    ) -> Result<(String, Analysis), String> {
        if notes.is_empty() {
            return Err("no notes given".into());
        }
        let last_end = notes
            .iter()
            .map(|n| n.start + n.duration)
            .fold(0.0f32, f32::max);
        let total_seconds = seconds
            .unwrap_or(last_end + 1.5)
            .clamp(0.1, MAX_RENDER_SECONDS);

        // A fresh engine per render keeps results deterministic.
        self.engine = SoundEngine::new(SAMPLE_RATE);
        apply_preset(&self.preset, &mut self.engine);
        self.apply_materials();
        self.engine.set_bpm(bpm);

        let total_samples = (total_seconds * SAMPLE_RATE as f32) as usize;
        let block_size = 128usize;
        let mut stereo = Vec::with_capacity(total_samples * 2);
        let mut left = vec![0.0f32; block_size];
        let mut right = vec![0.0f32; block_size];

        let mut position = 0usize;
        while position < total_samples {
            let block = block_size.min(total_samples - position);
            for spec in notes {
                let start = (spec.start * SAMPLE_RATE as f32) as usize;
                let end = ((spec.start + spec.duration) * SAMPLE_RATE as f32) as usize;
                if start >= position && start < position + block {
                    self.engine.note_on(
                        spec.note,
                        spec.velocity.clamp(0.0, 1.0),
                        start - position,
                        spec.channel.min(15),
                    );
                }
                if end >= position && end < position + block {
                    self.engine
                        .note_off(spec.note, 0.5, end - position, spec.channel.min(15));
                }
            }
            self.engine
                .process(block, &mut left[..block], &mut right[..block]);
            for i in 0..block {
                stereo.push(left[i]);
                stereo.push(right[i]);
            }
            position += block;
        }

        let non_finite = stereo.iter().filter(|v| !v.is_finite()).count();
        if non_finite > 0 {
            return Err(format!("render produced {non_finite} non-finite samples"));
        }

        write_wav(out_path, &stereo, SAMPLE_RATE)
            .map_err(|e| format!("cannot write '{out_path}': {e}"))?;
        let analysis = analyze(&stereo, SAMPLE_RATE);
        self.last_render = Some(stereo);
        self.last_render_path = Some(out_path.to_string());
        let summary = format!(
            "rendered {total_seconds:.2}s ({} notes) to {out_path}",
            notes.len()
        );
        Ok((summary, analysis))
    }

    pub fn analyze_last(&self) -> Result<Analysis, String> {
        match &self.last_render {
            Some(buffer) => Ok(analyze(buffer, SAMPLE_RATE)),
            None => Err("nothing rendered yet — call play first".into()),
        }
    }

    /// Analyzes an external audio file (reference material).
    pub fn analyze_file(
        path: &str,
        start: Option<f32>,
        duration: Option<f32>,
    ) -> Result<Analysis, String> {
        let (stereo, sample_rate) = crate::decode::decode_file(path, start, duration)?;
        Ok(analyze(&stereo, sample_rate))
    }

    /// Compares a reference file against the last render and describes the
    /// gaps in sound-designer terms.
    pub fn compare(
        &self,
        reference_path: &str,
        start: Option<f32>,
        duration: Option<f32>,
    ) -> Result<serde_json::Value, String> {
        let reference = Self::analyze_file(reference_path, start, duration)?;
        let render = self.analyze_last()?;

        let mut notes: Vec<String> = Vec::new();
        let level = reference.rms_db - render.rms_db;
        if level.abs() > 1.5 {
            notes.push(format!(
                "reference is {:.1} dB {} than the render",
                level.abs(),
                if level > 0.0 { "louder" } else { "quieter" }
            ));
        }
        if reference.spectral_centroid_hz > 1.0 && render.spectral_centroid_hz > 1.0 {
            let semitones =
                12.0 * (reference.spectral_centroid_hz / render.spectral_centroid_hz).log2();
            if semitones.abs() > 2.0 {
                notes.push(format!(
                    "reference is ~{:.0} semitones {} (centroid {:.0} vs {:.0} Hz)",
                    semitones.abs(),
                    if semitones > 0.0 { "brighter" } else { "darker" },
                    reference.spectral_centroid_hz,
                    render.spectral_centroid_hz
                ));
            }
        }
        let band_pairs = [
            ("sub 0-60", reference.bands_db.sub_0_60, render.bands_db.sub_0_60),
            ("bass 60-250", reference.bands_db.bass_60_250, render.bands_db.bass_60_250),
            ("low-mid 250-1k", reference.bands_db.low_mid_250_1k, render.bands_db.low_mid_250_1k),
            ("mid 1k-4k", reference.bands_db.mid_1k_4k, render.bands_db.mid_1k_4k),
            ("high 4k-12k", reference.bands_db.high_4k_12k, render.bands_db.high_4k_12k),
            ("air 12k+", reference.bands_db.air_12k_up, render.bands_db.air_12k_up),
        ];
        for (name, ref_db, render_db) in band_pairs {
            let delta = ref_db - render_db;
            if delta.abs() > 4.0 {
                notes.push(format!(
                    "{name}: reference has {:.0} dB {} relative energy",
                    delta.abs(),
                    if delta > 0.0 { "more" } else { "less" }
                ));
            }
        }
        let format_rates = |rates: &[crate::analysis::ModRate]| -> String {
            if rates.is_empty() {
                "none".to_string()
            } else {
                rates
                    .iter()
                    .map(|r| format!("{:.1} Hz", r.hz))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        };
        notes.push(format!(
            "movement rates — reference: {} | render: {}",
            format_rates(&reference.movement.mod_rates_hz),
            format_rates(&render.movement.mod_rates_hz)
        ));
        let width = reference.stereo_width - render.stereo_width;
        if width.abs() > 0.15 {
            notes.push(format!(
                "reference is {} in stereo (width {:.2} vs {:.2})",
                if width > 0.0 { "wider" } else { "narrower" },
                reference.stereo_width,
                render.stereo_width
            ));
        }
        let flatness = reference.texture.spectral_flatness - render.texture.spectral_flatness;
        if flatness.abs() > 0.1 {
            notes.push(format!(
                "reference is {} (flatness {:.2} vs {:.2})",
                if flatness > 0.0 { "noisier/dirtier" } else { "more tonal/cleaner" },
                reference.texture.spectral_flatness,
                render.texture.spectral_flatness
            ));
        }

        Ok(serde_json::json!({
            "summary": notes,
            "reference": serde_json::to_value(&reference).unwrap_or_default(),
            "render": serde_json::to_value(&render).unwrap_or_default(),
        }))
    }

    /// Racks directory: `SPINWAVE_RACKS` env, else `<repo>/presets/racks`
    /// resolved relative to the executable, else `./presets/racks`.
    pub fn racks_dir() -> std::path::PathBuf {
        if let Ok(dir) = std::env::var("SPINWAVE_RACKS") {
            return dir.into();
        }
        if let Some(exe_dir) = std::env::current_exe().ok().and_then(|p| {
            p.parent()
                .and_then(|d| d.parent())
                .and_then(|d| d.parent())
                .map(|d| d.to_path_buf())
        }) {
            let candidate = exe_dir.join("presets").join("racks");
            if candidate.is_dir() {
                return candidate;
            }
        }
        std::path::PathBuf::from("presets/racks")
    }

    pub fn list_racks() -> Vec<(String, String, String)> {
        let mut racks = Vec::new();
        let Ok(entries) = std::fs::read_dir(Self::racks_dir()) else { return racks };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(
                text.trim_start_matches('\u{feff}'),
            ) else {
                continue;
            };
            racks.push((
                path.file_stem().unwrap_or_default().to_string_lossy().to_string(),
                value["description"].as_str().unwrap_or("").to_string(),
                path.to_string_lossy().to_string(),
            ));
        }
        racks
    }

    /// Applies an effect rack (a settings fragment) over the current patch,
    /// leaving the voice section untouched except keys the rack names.
    pub fn apply_rack(&mut self, rack: &str) -> Result<String, String> {
        let path = if std::path::Path::new(rack).exists() {
            std::path::PathBuf::from(rack)
        } else {
            Self::racks_dir().join(format!("{rack}.json"))
        };
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read rack '{}': {e}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_str(text.trim_start_matches('\u{feff}'))
                .map_err(|e| format!("invalid rack JSON: {e}"))?;
        let Some(settings) = value["settings"].as_object() else {
            return Err("rack has no settings object".into());
        };
        let applied = self.set_params(settings)?;
        Ok(format!(
            "rack '{}' applied: {applied}",
            value["name"].as_str().unwrap_or(rack)
        ))
    }

    pub fn describe_params(&self, search: Option<&str>, limit: usize) -> serde_json::Value {
        let table = parameters();
        match search {
            None => {
                // Group summary: prefix -> count.
                let mut groups: std::collections::BTreeMap<String, usize> =
                    std::collections::BTreeMap::new();
                for details in table.iter() {
                    let group = group_of(&details.name);
                    *groups.entry(group).or_insert(0) += 1;
                }
                json!({
                    "total_parameters": table.len(),
                    "groups": groups,
                    "hint": "call again with `search` (name substring or group prefix) for details",
                })
            }
            Some(search) => {
                let matches: Vec<_> = table
                    .iter()
                    .filter(|d| d.name.contains(search))
                    .take(limit)
                    .map(|d| {
                        json!({
                            "name": d.name,
                            "range": [d.min, d.max],
                            "default": d.default_value,
                            "display_name": d.display_name,
                            "units": d.display_units,
                        })
                    })
                    .collect();
                json!({ "matches": matches })
            }
        }
    }
}

fn group_of(name: &str) -> String {
    for prefix in [
        "osc_1", "osc_2", "osc_3", "sample", "filter_1", "filter_2", "filter_fx", "env_",
        "lfo_", "random_", "modulation_", "chorus", "compressor", "delay", "distortion",
        "eq_", "flanger", "phaser", "reverb", "macro",
    ] {
        if name.starts_with(prefix) {
            return prefix.trim_end_matches('_').to_string();
        }
    }
    "global".to_string()
}

/// Returns a path the live plugin can read as WAV: `.wav` files pass
/// through untouched; anything else is decoded and written as a canonical
/// float32 stereo WAV next to `%TEMP%`.
fn wav_for_live(path: &str) -> Result<String, String> {
    let is_wav = Path::new(path)
        .extension()
        .map(|e| e.eq_ignore_ascii_case("wav"))
        .unwrap_or(false);
    if is_wav {
        return Ok(path.to_string());
    }
    let (stereo, sample_rate) = crate::decode::decode_file(path, None, None)?;
    let stem = Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "material".to_string());
    let out = std::env::temp_dir().join(format!("spinwave-live-{stem}.wav"));
    let out = out.to_string_lossy().to_string();
    write_wav(&out, &stereo, sample_rate)
        .map_err(|e| format!("cannot write '{out}': {e}"))?;
    Ok(out)
}

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
    for value in interleaved {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_params_validates_and_clamps() {
        let mut session = Session::new();
        let mut values = serde_json::Map::new();
        values.insert("filter_1_cutoff".into(), json!(500.0)); // above max
        values.insert("bogus_param".into(), json!(1.0));
        let message = session.set_params(&values).unwrap();
        assert!(message.contains("1 parameter(s) applied"));
        assert!(message.contains("clamped"));
        assert!(message.contains("bogus_param"));
        assert_eq!(session.preset.settings.parameter("filter_1_cutoff"), Some(136.0));
    }

    #[test]
    fn add_modulation_and_render() {
        let mut session = Session::new();
        session
            .add_modulation("lfo_1", "filter_1_cutoff", 0.6, false, false, 0.0)
            .unwrap();
        assert!(session
            .add_modulation("nope", "filter_1_cutoff", 0.5, false, false, 0.0)
            .is_err());

        let scratch = std::env::temp_dir().join("spinwave-mcp-test.wav");
        let notes = vec![NoteSpec { note: 60, start: 0.0, duration: 0.5, velocity: 0.9, channel: 0 }];
        let (summary, analysis) = session
            .render(&notes, Some(1.0), 120.0, scratch.to_str().unwrap())
            .unwrap();
        assert!(summary.contains("rendered"));
        assert!(analysis.peak > 0.005, "peak {}", analysis.peak);
        assert!(analysis.pitch_hz.is_some());
        let _ = std::fs::remove_file(scratch);
    }

    #[test]
    fn load_sample_offline_changes_slot_material() {
        let mut session = Session::new();
        let default_len = session.engine.allocator().kernels()[0]
            .slot_sample(0)
            .original_length();

        // A small float32 stereo WAV as input material.
        let path = std::env::temp_dir().join("spinwave-load-sample-test.wav");
        let frames = 512usize;
        let interleaved: Vec<f32> = (0..frames * 2)
            .map(|i| ((i / 2) as f32 * 0.05).sin() * 0.5)
            .collect();
        write_wav(path.to_str().unwrap(), &interleaved, 44100).unwrap();

        let message = session.load_sample_offline(path.to_str().unwrap(), 0).unwrap();
        assert!(message.contains("512 frames"), "{message}");
        assert_ne!(default_len, frames);
        for kernel in session.engine.allocator().kernels() {
            assert_eq!(kernel.slot_sample(0).original_length(), frames);
        }

        // Out-of-range slots are rejected.
        assert!(session.load_sample_offline(path.to_str().unwrap(), 9).is_err());

        // The material survives the fresh engine a render builds.
        let out = std::env::temp_dir().join("spinwave-load-sample-render.wav");
        let notes =
            vec![NoteSpec { note: 60, start: 0.0, duration: 0.2, velocity: 0.9, channel: 0 }];
        session
            .render(&notes, Some(0.5), 120.0, out.to_str().unwrap())
            .unwrap();
        assert_eq!(
            session.engine.allocator().kernels()[0].slot_sample(0).original_length(),
            frames
        );
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn describe_params_groups_and_search() {
        let session = Session::new();
        let summary = session.describe_params(None, 10);
        assert!(summary["total_parameters"].as_u64().unwrap() > 700);
        let matches = session.describe_params(Some("filter_1_cut"), 10);
        assert_eq!(matches["matches"].as_array().unwrap().len(), 1);
    }
}
