//! The stateful synth session behind the MCP tools: one engine, one
//! current preset, the last render.

use std::io::Write;

use serde::Deserialize;
use serde_json::json;
use spinwave_engine::engine::SoundEngine;
use spinwave_params::preset::ModulationConnection;
use spinwave_params::{parameters, Preset};
use spinwave_plugin::apply_preset;
use spinwave_plugin::patch::connections_from_preset;

use crate::analysis::{analyze, Analysis};
use crate::live_client::LiveLink;

pub const SAMPLE_RATE: u32 = 44100;
const MAX_RENDER_SECONDS: f32 = 60.0;

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
            last_render: None,
            last_render_path: None,
            live: LiveLink::default(),
        }
    }

    /// Pushes the current preset to the running standalone.
    pub fn live_push_preset(&mut self) -> Result<String, String> {
        let preset_value =
            serde_json::to_value(&self.preset).map_err(|e| e.to_string())?;
        self.live
            .send(&serde_json::json!({"cmd": "preset", "preset": preset_value}))?;
        Ok("patch pushed to the live synth".into())
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

    /// Re-applies the current preset to the engine (after any edit).
    pub fn sync_engine(&mut self) {
        apply_preset(&self.preset, &mut self.engine);
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
    fn describe_params_groups_and_search() {
        let session = Session::new();
        let summary = session.describe_params(None, 10);
        assert!(summary["total_parameters"].as_u64().unwrap() > 700);
        let matches = session.describe_params(Some("filter_1_cut"), 10);
        assert_eq!(matches["matches"].as_array().unwrap().len(), 1);
    }
}
