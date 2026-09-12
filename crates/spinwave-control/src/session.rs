//! The stateful synth session behind the MCP tools: one engine, one
//! current preset, the last render.
//!
//! Oscillator-slot materials (samples, imported wavetables, SFZ
//! instruments) live INSIDE the preset (`settings.wavetables[slot]`,
//! `settings.spinwave_materials`): every `apply_preset` re-installs them,
//! and `save_preset` / `get_patch` / `live_apply` carry them along.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;
use serde_json::json;
use spinwave_dsp::oscillator::Sample;
use spinwave_dsp::wavetable::{
    wavetable_from_audio, wavetable_from_png, AudioImportMode, AudioImportOptions,
    ImageImportOptions,
};
use spinwave_engine::engine::SoundEngine;
use spinwave_engine::kernel::ModSource;
use spinwave_params::preset::{LoadReport, ModulationConnection};
use spinwave_params::{parameters, Preset};
use spinwave_plugin::materials::{self, ZoneFrames};
use spinwave_plugin::patch::{connections_from_preset, load_preset};
use spinwave_plugin::{apply_preset_with, materials::write_wav};

use crate::analysis::{analyze, Analysis};
use crate::live_client::LiveLink;

pub const SAMPLE_RATE: u32 = 44100;
const MAX_RENDER_SECONDS: f32 = 60.0;

/// Oscillator slots (`osc_1` .. `osc_4`).
const NUM_SLOTS: usize = materials::NUM_SLOTS;

fn check_slot(slot: usize) -> Result<(), String> {
    if slot >= NUM_SLOTS {
        return Err(format!("slot must be 0..{}", NUM_SLOTS - 1));
    }
    Ok(())
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
    let (left, right) = materials::deinterleave(&stereo);
    Ok(Sample::from_stereo(&stem, &left, &right, sample_rate))
}

/// SFZ zone decoder for the offline engine: any format symphonia reads
/// (memoized on the file's modification time).
fn decode_zone(path: &Path) -> Option<ZoneFrames> {
    materials::cached_zone(path, |path| {
        let (stereo, rate) = crate::decode::decode_file(&path.to_string_lossy(), None, None).ok()?;
        let (left, right) = materials::deinterleave(&stereo);
        Some((left, Some(right), rate))
    })
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
    pub last_render: Option<Vec<f32>>,
    pub last_render_path: Option<String>,
    pub live: LiveLink,
    /// Directory relative render paths resolve against.
    pub output_dir: PathBuf,
    /// Findings of the last preset load (`set_patch` / `load_preset`).
    pub last_report: LoadReport,
    /// Set by the golden bench: a single-cycle wavetable reinstalled after
    /// every engine rebuild, so the render matches the reference harness.
    forced_wavetable: Option<std::sync::Arc<spinwave_dsp::wavetable::Wavetable>>,
    /// Cleared by the golden bench (see `SoundEngine::set_master_dc_blocker`
    /// and `set_voice_dc_blockers`): both are Spinwave additions the
    /// reference lacks, so the bench compares without them.
    master_dc_blocker: bool,
    /// Set by the golden bench: the seed every voice's `random_1` starts
    /// from, so a case can draw the same values the reference did.
    random_seed: Option<u32>,
}

impl Session {
    /// A session whose relative render paths resolve against `output_dir`.
    pub fn with_output_dir(output_dir: PathBuf) -> Session {
        let preset = Preset::from_json(
            r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#,
        )
        .expect("init preset");
        let mut engine = SoundEngine::new(SAMPLE_RATE);
        apply_preset_with(&preset, &mut engine, &mut decode_zone);
        Session {
            preset,
            engine,
            last_render: None,
            last_render_path: None,
            live: LiveLink::default(),
            output_dir,
            last_report: LoadReport::default(),
            forced_wavetable: None,
            master_dc_blocker: true,
            random_seed: None,
        }
    }

    /// The offline engine (tests inspect installed material).
    #[cfg(test)]
    pub fn engine(&self) -> &SoundEngine {
        &self.engine
    }

    // -- Oscillator-slot material (samples, wavetables, SFZ) ----------------

    /// Loads an audio file into one oscillator slot's Sample/Granular
    /// engines (set `osc_N_engine` to 1 or 2 to hear it), embedding the
    /// audio in the preset. Any format the analysis decoder reads works.
    pub fn load_sample_offline(&mut self, path: &str, slot: usize) -> Result<String, String> {
        check_slot(slot)?;
        let sample = sample_from_file(path)?;
        let frames = sample.original_length();
        let rate = sample.sample_rate();
        materials::set_slot_sample_json(&mut self.preset, slot, materials::sample_to_json(&sample));
        self.sync_engine();
        Ok(format!(
            "loaded '{path}' into osc {} ({frames} frames @ {rate} Hz, embedded in the preset); \
             set osc_{}_engine to 1 (Sample) or 2 (Granular) to hear it",
            slot + 1,
            slot + 1,
        ))
    }

    /// Imports an audio file (mode `spectral`/`raw`) or a PNG spectrum
    /// (mode `png`) as the slot's wavetable, stored in the preset.
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
        let stem = Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "import".to_string());
        let json = materials::wavetable_to_json(&table, &stem);
        materials::set_slot_wavetable_json(&mut self.preset, slot, json);
        self.sync_engine();
        Ok(format!(
            "imported '{path}' ({mode}) as osc {}'s wavetable (stored in the preset); the \
             Wavetable engine (osc_{}_engine 0) plays it",
            slot + 1,
            slot + 1,
        ))
    }

    /// Loads an SFZ instrument into one oscillator slot's Multisample
    /// engine (set `osc_N_engine` to 3 to hear it). Sample opcodes resolve
    /// relative to the SFZ file; the path and text are kept in the preset.
    pub fn load_sfz_offline(&mut self, path: &str, slot: usize) -> Result<String, String> {
        check_slot(slot)?;
        let sfz = materials::sfz_material_from_path(path)?;
        // Validate before committing it to the preset.
        let base_dir = materials::sfz_base_dir(&sfz);
        let probe = materials::multisample_from_sfz(&sfz.text, &base_dir, decode_zone)?;
        let zones = probe.zones.len();
        let warnings: Vec<String> = probe.warnings.clone();
        materials::set_slot_sfz(&mut self.preset, slot, sfz);
        self.sync_engine();
        let mut message = format!(
            "loaded SFZ '{path}' into osc {} ({zones} zone(s)); set osc_{}_engine to 3 \
             (Multisample) to hear it",
            slot + 1,
            slot + 1,
        );
        if !warnings.is_empty() {
            message.push_str(&format!("; warnings: {}", warnings.join("; ")));
        }
        Ok(message)
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

    /// Pushes the current preset (materials included) to the live
    /// instance; relays its load report.
    pub fn live_push_preset(&mut self) -> Result<String, String> {
        let preset_value =
            serde_json::to_value(&self.preset).map_err(|e| e.to_string())?;
        let reply = self
            .live
            .send(&serde_json::json!({"cmd": "preset", "preset": preset_value}))?;
        let mut message = "patch pushed to the live synth".to_string();
        if let Some(body) = reply.strip_prefix("ok ") {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
                if let Ok(report) = serde_json::from_value::<LoadReport>(value["report"].clone()) {
                    if !report.is_clean() {
                        message.push_str(&format!(" ({})", report.summary()));
                    }
                }
            }
        }
        Ok(message)
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

    /// Re-applies the current preset to the engine (after any edit),
    /// materials included. Returns the load report.
    pub fn sync_engine(&mut self) -> LoadReport {
        apply_preset_with(&self.preset, &mut self.engine, &mut decode_zone)
    }

    /// Loads one single-cycle frame into every oscillator's wavetable, the
    /// way the golden bench's reference harness does: one frame, then the
    /// band-limited post-process. Both halves then play the same waveform,
    /// so a difference in the audio is a difference in the DSP rather than
    /// in two wavetable builders.
    ///
    /// Rendering rebuilds the engine, so this must be called after
    /// `load_preset_json` and before `render_samples`.
    pub fn load_single_frame_wavetables(&mut self, frame: &spinwave_dsp::wavetable::WaveFrame) {
        let mut table = spinwave_dsp::wavetable::Wavetable::new(1);
        table.set_num_frames(1);
        table.load_wave_frame_at(frame, 0);
        table.post_process(1.0);
        self.forced_wavetable = Some(std::sync::Arc::new(table));
        self.install_forced_wavetable();
    }

    /// Turns every DC blocker off, master and per-voice. The golden bench
    /// only: see `SoundEngine::set_master_dc_blocker`.
    pub fn set_dc_blockers(&mut self, enabled: bool) {
        self.master_dc_blocker = enabled;
    }

    /// Pins the random LFOs' seed for every render from here on (see
    /// `SoundEngine::reseed_random_lfos`).
    pub fn set_random_seed(&mut self, seed: Option<u32>) {
        self.random_seed = seed;
    }

    fn install_forced_wavetable(&mut self) {
        let Some(table) = &self.forced_wavetable else { return };
        for kernel in self.engine.allocator_mut().kernels_mut() {
            for slot in 0..NUM_SLOTS {
                kernel.set_wavetable(slot, table.clone());
            }
        }
    }

    /// Replaces the patch from `.vital` JSON: parses, migrates old
    /// versions, applies, and keeps the report in `last_report`.
    pub fn load_preset_json(&mut self, text: &str) -> Result<String, String> {
        let (preset, mut report) = load_preset(text)?;
        self.preset = preset;
        let applied = self.sync_engine();
        report.notes.extend(applied.notes);
        let mut message = format!(
            "loaded preset '{}' ({} set parameters, {} modulations mapped)",
            self.preset.preset_name,
            self.preset.settings.values.len(),
            connections_from_preset(&self.preset).len(),
        );
        if !report.is_clean() {
            message.push_str(&format!("; load report: {}", report.summary()));
        }
        self.last_report = report;
        Ok(message)
    }

    /// Sets parameter engine values by name. Every table parameter is
    /// accepted — Vital's and the Spinwave namespace (`osc_4_*`,
    /// `noise_*`, `bus_a_*`, `fx_split_*`, `*_gran_*`, ...).
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
        if !known_sources.contains(&source) {
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

        let report = self.sync_engine();
        let mapped = connections_from_preset(&self.preset).len();
        let mut message = format!(
            "modulation {n}: {source} -> {destination} (amount {amount}); \
             {mapped} connection(s) active in the engine"
        );
        if !report.ignored_connections.is_empty() {
            message.push_str(&format!(
                "; not routable by the engine: {}",
                report.ignored_connections.join(", ")
            ));
        }
        Ok(message)
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

    /// Resolves a render path: relative paths land in `output_dir`; an
    /// existing file is never overwritten silently — unless `overwrite`,
    /// a `-1`, `-2`... suffix is added.
    pub fn resolve_out_path(&self, out_path: &str, overwrite: bool) -> Result<PathBuf, String> {
        let requested = Path::new(out_path);
        let mut path = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.output_dir.join(requested)
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("cannot create '{}': {e}", parent.display()))?;
            }
        }
        if path.exists() && !overwrite {
            let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            let extension = path.extension().map(|e| e.to_string_lossy().to_string());
            let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
            for n in 1..10_000 {
                let name = match &extension {
                    Some(ext) => format!("{stem}-{n}.{ext}"),
                    None => format!("{stem}-{n}"),
                };
                let candidate = parent.join(name);
                if !candidate.exists() {
                    path = candidate;
                    break;
                }
            }
        }
        Ok(path)
    }

    /// Renders notes through a fresh engine to `out_path` (resolved with
    /// the overwrite policy of [`Session::resolve_out_path`]) and analyzes
    /// the result.
    pub fn render_to(
        &mut self,
        notes: &[NoteSpec],
        seconds: Option<f32>,
        bpm: f32,
        out_path: &str,
        overwrite: bool,
    ) -> Result<(String, Analysis), String> {
        if notes.is_empty() {
            return Err("no notes given".into());
        }
        let out_path = self.resolve_out_path(out_path, overwrite)?;
        let out_path = out_path.to_string_lossy().to_string();
        let stereo = self.render_samples(notes, seconds.unwrap_or(0.0), bpm);
        let total_seconds = stereo.len() as f32 / 2.0 / SAMPLE_RATE as f32;

        let non_finite = stereo.iter().filter(|v| !v.is_finite()).count();
        if non_finite > 0 {
            return Err(format!("render produced {non_finite} non-finite samples"));
        }

        write_wav(&out_path, &stereo, SAMPLE_RATE)
            .map_err(|e| format!("cannot write '{out_path}': {e}"))?;
        let analysis = analyze(&stereo, SAMPLE_RATE);
        self.last_render = Some(stereo);
        self.last_render_path = Some(out_path.clone());
        let summary = format!(
            "rendered {total_seconds:.2}s ({} notes) to {out_path}",
            notes.len()
        );
        Ok((summary, analysis))
    }

    /// The sample rate every render runs at.
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Renders notes and returns the interleaved stereo buffer without
    /// touching the filesystem. Unlike [`Session::render_to`] this keeps
    /// non-finite samples, because a caller checking for them needs to see
    /// them.
    pub fn render_samples(&mut self, notes: &[NoteSpec], seconds: f32, bpm: f32) -> Vec<f32> {
        self.render_samples_probed(notes, seconds, bpm, &[]).0
    }

    /// Renders, and alongside the audio samples the control-rate value of
    /// each requested modulation source, one reading per block.
    ///
    /// The golden bench uses this to compare the modulation curve itself
    /// rather than guessing at its shape from the audio it drives. The
    /// reference harness writes the same readings from Vital's own
    /// sources, so the two curves lie on top of each other or they do not.
    pub fn render_samples_probed(
        &mut self,
        notes: &[NoteSpec],
        seconds: f32,
        bpm: f32,
        probes: &[ModSource],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let last_end = notes
            .iter()
            .map(|n| n.start + n.duration)
            .fold(0.0f32, f32::max);
        let total_seconds = if seconds > 0.0 {
            seconds.clamp(0.1, MAX_RENDER_SECONDS)
        } else {
            (last_end + 1.5).clamp(0.1, MAX_RENDER_SECONDS)
        };

        // A fresh engine per render keeps results deterministic — with the
        // random seed counter rewound, since the generators are seeded
        // from a process-global counter (the reference's `next_seed_++`).
        spinwave_dsp::modulators::RandomGenerator::reset_seed_counter();
        self.engine = SoundEngine::new(SAMPLE_RATE);
        apply_preset_with(&self.preset, &mut self.engine, &mut decode_zone);
        if let Some(seed) = self.random_seed {
            self.engine.reseed_random_lfos(seed);
        }
        self.install_forced_wavetable();
        self.engine.set_master_dc_blocker(self.master_dc_blocker);
        self.engine.set_voice_dc_blockers(self.master_dc_blocker);
        self.engine.set_bpm(bpm);

        let total_samples = (total_seconds * SAMPLE_RATE as f32) as usize;
        let block_size = 128usize;
        let mut stereo = Vec::with_capacity(total_samples * 2);
        let mut left = vec![0.0f32; block_size];
        let mut right = vec![0.0f32; block_size];
        let mut probe_curves = vec![Vec::new(); probes.len()];
        // Four more curves, one per lane of the voice pair, carrying the
        // modulation offset that actually reached filter 1's cutoff.
        let mut destination_curves: Vec<Vec<f32>> =
            vec![Vec::new(); if probes.is_empty() { 0 } else { 4 }];

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
            // After the block, so the reading is the value that block was
            // rendered with — the same instant the reference reads.
            for (curve, &source) in probe_curves.iter_mut().zip(probes) {
                curve.push(self.engine.probe_source(source));
            }
            if !destination_curves.is_empty() {
                let cutoff = self.engine.probe_cutoff_offset();
                let level = self.engine.probe_osc_level_offset();
                let lanes = [cutoff[0], cutoff[1], level[0], level[1]];
                for (curve, value) in destination_curves.iter_mut().zip(lanes) {
                    curve.push(value);
                }
            }
            position += block;
        }
        probe_curves.extend(destination_curves);
        (stereo, probe_curves)
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
            "movement rates - reference: {} | render: {}",
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

    /// The repository root when the racks directory sits inside one.
    pub fn repo_dir() -> Option<PathBuf> {
        let racks = Self::racks_dir();
        let repo = racks.parent()?.parent()?.to_path_buf();
        repo.is_dir().then_some(repo)
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
                    "vital_parameters": table.len_vital(),
                    "spinwave_only_parameters": table.len() - table.len_vital(),
                    "groups": groups,
                    "hint": "call again with `search` (name substring or group prefix) for details; `spinwave_only` marks parameters Vital does not have",
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
                            "scale": format!("{:?}", d.scale),
                            "spinwave_only": d.spinwave_only,
                        })
                    })
                    .collect();
                json!({ "matches": matches })
            }
        }
    }
}

/// The group a parameter name belongs to (for `describe_params`).
pub fn group_of(name: &str) -> String {
    for prefix in [
        "bus_a_", "bus_b_", "fx_split_", "osc_1", "osc_2", "osc_3", "osc_4", "noise_", "sample",
        "filter_1", "filter_2", "filter_fx", "env_", "lfo_", "random_", "modulation_", "chorus",
        "compressor", "delay", "distortion", "eq_", "flanger", "phaser", "reverb", "macro",
        "portamento", "voice_", "stereo_",
    ] {
        if name.starts_with(prefix) {
            return prefix.trim_end_matches('_').to_string();
        }
    }
    "global".to_string()
}

/// Returns a path the live plugin can read as WAV: `.wav` files pass
/// through untouched; anything else is decoded and written as a canonical
/// float32 stereo WAV in `%TEMP%` under a unique name (pid + counter, so
/// two sources with the same stem never clobber each other).
fn wav_for_live(path: &str) -> Result<String, String> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
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
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!(
        "spinwave-live-{}-{unique}-{stem}.wav",
        std::process::id()
    ));
    let out = out.to_string_lossy().to_string();
    write_wav(&out, &stereo, sample_rate)
        .map_err(|e| format!("cannot write '{out}': {e}"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_session() -> Session {
        Session::with_output_dir(std::env::temp_dir())
    }

    #[test]
    fn set_params_validates_and_clamps() {
        let mut session = temp_session();
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
    fn set_params_accepts_the_spinwave_namespace() {
        let mut session = temp_session();
        let mut values = serde_json::Map::new();
        for (name, value) in [
            ("osc_4_on", 1.0),
            ("osc_4_level", 0.5),
            ("osc_1_engine", 2.0),
            ("osc_1_gran_density", 12.0),
            ("noise_on", 1.0),
            ("noise_level", 0.3),
            ("bus_a_on", 1.0),
            ("bus_a_reverb_on", 1.0),
            ("fx_split_delay", 3.0),
            ("lfo_10_frequency", 2.0),
            ("env_8_attack", 0.3),
            ("macro_control_7", 0.5),
        ] {
            values.insert(name.into(), json!(value));
        }
        let message = session.set_params(&values).unwrap();
        assert!(message.starts_with("12 parameter(s) applied"), "{message}");
        assert!(!message.contains("skipped"));
        let kernel = &session.engine().allocator().kernels()[0].params;
        assert!(kernel.oscillators[3].on);
        assert!(kernel.noise.on);
        assert_eq!(kernel.macros[6], 0.5);
        assert_eq!(group_of("bus_a_reverb_on"), "bus_a");
        assert_eq!(group_of("osc_4_level"), "osc_4");
        assert_eq!(group_of("noise_level"), "noise");
        assert_eq!(group_of("fx_split_delay"), "fx_split");
        let described = session.describe_params(Some("noise_"), 20);
        assert!(described["matches"].as_array().unwrap().iter().all(|m| m["spinwave_only"] == true));
    }

    #[test]
    fn add_modulation_and_render() {
        let mut session = temp_session();
        session
            .add_modulation("lfo_1", "filter_1_cutoff", 0.6, false, false, 0.0)
            .unwrap();
        assert!(session
            .add_modulation("nope", "filter_1_cutoff", 0.5, false, false, 0.0)
            .is_err());
        // Spinwave-only sources are accepted.
        session
            .add_modulation("lfo_11", "osc_4_level", 0.2, false, false, 0.0)
            .unwrap();

        let scratch = std::env::temp_dir().join("spinwave-mcp-test.wav");
        let notes = vec![NoteSpec { note: 60, start: 0.0, duration: 0.5, velocity: 0.9, channel: 0 }];
        let (summary, analysis) = session
            .render_to(&notes, Some(1.0), 120.0, scratch.to_str().unwrap(), true)
            .unwrap();
        assert!(summary.contains("rendered"));
        assert!(analysis.peak > 0.005, "peak {}", analysis.peak);
        assert!(analysis.pitch_hz.is_some());
        let _ = std::fs::remove_file(scratch);
    }

    #[test]
    fn load_sample_offline_embeds_and_survives_reapply() {
        let mut session = temp_session();
        let default_len = session.engine().allocator().kernels()[0]
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
        for kernel in session.engine().allocator().kernels() {
            assert_eq!(kernel.slot_sample(0).original_length(), frames);
        }
        // The material is in the preset (so get_patch / save_preset carry it).
        let block = session.preset.settings.spinwave_materials.as_ref().unwrap();
        assert_eq!(block.slot(0).unwrap().sample.as_ref().unwrap().length, frames as u64);

        // Out-of-range slots are rejected.
        assert!(session.load_sample_offline(path.to_str().unwrap(), 9).is_err());

        // A parameter edit re-applies the preset: the material survives.
        let mut values = serde_json::Map::new();
        values.insert("osc_1_engine".into(), json!(1.0));
        session.set_params(&values).unwrap();
        assert_eq!(
            session.engine().allocator().kernels()[0].slot_sample(0).original_length(),
            frames
        );

        // ... and the fresh engine a render builds.
        let out = std::env::temp_dir().join("spinwave-load-sample-render.wav");
        let notes =
            vec![NoteSpec { note: 60, start: 0.0, duration: 0.2, velocity: 0.9, channel: 0 }];
        session
            .render_to(&notes, Some(0.5), 120.0, out.to_str().unwrap(), true)
            .unwrap();
        assert_eq!(
            session.engine().allocator().kernels()[0].slot_sample(0).original_length(),
            frames
        );
        // Round trip through JSON keeps it.
        let json = session.preset.to_json().unwrap();
        let mut reloaded = temp_session();
        reloaded.load_preset_json(&json).unwrap();
        assert_eq!(
            reloaded.engine().allocator().kernels()[0].slot_sample(0).original_length(),
            frames
        );
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn load_preset_reports_and_output_paths_never_clobber() {
        let mut session = temp_session();
        let message = session
            .load_preset_json(
                r#"{"synth_version":"0.8.0","settings":{"filter_1_model":4.0,"filter_1_blend":1.0,
                    "modulations":[{"source":"lfo_1","destination":"nope"}]}}"#,
            )
            .unwrap();
        assert!(message.contains("load report"), "{message}");
        assert!(message.contains("migrated from 0.8.0"));
        assert!(message.contains("lfo_1 -> nope"));
        assert!(!session.last_report.is_clean());

        let dir = std::env::temp_dir().join("spinwave-out-path-test");
        std::fs::create_dir_all(&dir).unwrap();
        session.output_dir = dir.clone();
        let first = session.resolve_out_path("take.wav", false).unwrap();
        assert_eq!(first, dir.join("take.wav"));
        std::fs::write(&first, b"x").unwrap();
        let second = session.resolve_out_path("take.wav", false).unwrap();
        assert_eq!(second, dir.join("take-1.wav"));
        let forced = session.resolve_out_path("take.wav", true).unwrap();
        assert_eq!(forced, first);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn describe_params_groups_and_search() {
        let session = temp_session();
        let summary = session.describe_params(None, 10);
        assert_eq!(summary["vital_parameters"].as_u64().unwrap(), 794);
        assert!(summary["total_parameters"].as_u64().unwrap() > 794);
        let matches = session.describe_params(Some("filter_1_cut"), 10);
        assert_eq!(matches["matches"].as_array().unwrap().len(), 1);
    }
}
