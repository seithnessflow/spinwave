//! Live control channel: every Spinwave instance (standalone or hosted in
//! a DAW) opens a localhost TCP listener once the host activates it and
//! registers itself in a discovery file, so LLM agents can find and drive
//! running instances.
//!
//! Protocol: one JSON object per line, response `ok`/`ok {...}` or `err: ...`.
//! Commands:
//! `{"cmd":"preset","preset":{...}}`      — apply a full .vital preset; replies
//!                                          `ok {"report":{...}}` (see `LoadReport`)
//! `{"cmd":"get_patch"}`                  — returns the current preset JSON
//! `{"cmd":"note_on","note":48,"velocity":0.8,"channel":0}`
//! `{"cmd":"note_off","note":48,"channel":0}`
//! `{"cmd":"seq","config":{...}}`         — arp/step sequencer config
//!                                          (see `note_sequencer::SeqConfig::from_json`)
//! `{"cmd":"panic"}`                      — all sounds off (flushes the sequencer)
//! `{"cmd":"ping"}`                       — replies `ok blocks=<n>`
//!
//! Material loading (paths are local to this machine; files are read and
//! decoded on the network thread, the audio thread only swaps the result
//! in). Each command also records the material INSIDE the current preset
//! (`settings.wavetables[slot]` / `settings.spinwave_materials`), so
//! `get_patch` and the DAW's saved state carry it:
//! `{"cmd":"load_sample","slot":0..3,"path":"..."}` — WAV file into the
//!   slot's Sample/Granular engines.
//! `{"cmd":"import_wavetable","slot":0..3,"path":"...","mode":"spectral"}` —
//!   builds a wavetable from a WAV (`"spectral"` pitch-tracked resynthesis
//!   or `"raw"` single-period slices) or from a PNG spectrum (`"png"`).
//! `{"cmd":"load_sfz","slot":0..3,"path":"..."}` — SFZ instrument into the
//!   slot's Multisample engine (sample opcodes resolve relative to the SFZ
//!   file; WAV only).
//!
//! Set `SPINWAVE_LIVE=0` to disable, `SPINWAVE_LIVE_PORT` to force a port.

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use spinwave_dsp::oscillator::{MultisampleSource, Sample};
use spinwave_dsp::wavetable::{
    wavetable_from_audio, wavetable_from_png, AudioImportMode, AudioImportOptions,
    ImageImportOptions, Wavetable,
};
use spinwave_engine::kernel::mod_matrix::NUM_OSCILLATORS;
use spinwave_params::preset::LoadReport;
use spinwave_params::Preset;

use crate::materials;
use crate::patch::{self, BuiltPatch};

const PORT_RANGE_START: u16 = 41929;
const PORT_RANGE_END: u16 = 41979;

/// Probe timeout when checking whether a registry entry still answers.
const PROBE_TIMEOUT: Duration = Duration::from_millis(300);

/// Commands handed to the audio thread. Heavy structures are prebuilt on
/// the network thread so the audio thread only swaps them in.
pub enum LiveCommand {
    /// A complete prebuilt patch (see [`BuiltPatch`]).
    ApplyBuilt(Box<BuiltPatch>),
    /// Installs sample material in one oscillator slot of every kernel
    /// (Sample + Granular engines).
    SetSample { slot: usize, sample: Arc<Sample> },
    /// Installs a wavetable in one oscillator slot of every kernel.
    SetWavetable { slot: usize, table: Arc<Wavetable> },
    /// Installs an SFZ instrument in one oscillator slot. Each kernel owns
    /// its zone playback state, so the network thread prebuilds one source
    /// per kernel (see [`LiveShared::kernel_count`]); the zone audio itself
    /// is shared between them.
    SetMultisample { slot: usize, sources: Vec<MultisampleSource> },
    NoteOn { note: i32, velocity: f32, channel: usize },
    NoteOff { note: i32, channel: usize },
    /// Reconfigures the arp/step sequencer (parsed off the audio thread).
    Seq(Box<crate::note_sequencer::SeqConfig>),
    Panic,
}

/// The single source of truth for the current patch: the live channel's
/// `get_patch`/`set_patch`, the DAW's persisted state and the material
/// commands all read and write this preset.
pub struct PresetStore {
    preset: Mutex<Preset>,
    /// Bumped on every change.
    generation: AtomicU64,
    /// Generation last built and sent to the audio thread.
    applied: AtomicU64,
}

impl Default for PresetStore {
    fn default() -> Self {
        PresetStore::new(default_preset())
    }
}

impl PresetStore {
    #[must_use]
    pub fn new(preset: Preset) -> PresetStore {
        PresetStore {
            preset: Mutex::new(preset),
            generation: AtomicU64::new(1),
            applied: AtomicU64::new(0),
        }
    }

    /// A copy of the current preset.
    #[must_use]
    pub fn get(&self) -> Preset {
        self.preset.lock().map(|p| p.clone()).unwrap_or_default()
    }

    /// Replaces the preset; returns the new generation.
    pub fn set(&self, preset: Preset) -> u64 {
        if let Ok(mut slot) = self.preset.lock() {
            *slot = preset;
        }
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Edits the preset in place; returns the new generation.
    pub fn update(&self, edit: impl FnOnce(&mut Preset)) -> u64 {
        if let Ok(mut slot) = self.preset.lock() {
            edit(&mut slot);
        }
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// The current preset as JSON text.
    #[must_use]
    pub fn to_json(&self) -> String {
        self.preset
            .lock()
            .ok()
            .and_then(|p| p.to_json().ok())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Whether the audio thread has been sent the latest generation.
    #[must_use]
    pub fn is_applied(&self) -> bool {
        self.applied.load(Ordering::Acquire) == self.generation()
    }

    pub fn mark_applied(&self, generation: u64) {
        self.applied.fetch_max(generation, Ordering::AcqRel);
    }
}

/// nih-plug persistent field holding the preset as JSON text, backed by
/// the shared [`PresetStore`] (no second copy: DAW state save reads the
/// store, state restore writes it).
pub struct PersistedPreset(pub Arc<PresetStore>);

impl<'a> nih_plug::params::persist::PersistentField<'a, String> for PersistedPreset {
    fn set(&self, new_value: String) {
        match patch::load_preset(&new_value) {
            Ok((preset, report)) => {
                if !report.is_clean() {
                    eprintln!("spinwave: restored state: {}", report.summary());
                }
                self.0.set(preset);
            }
            Err(e) => eprintln!("spinwave: persisted preset ignored: {e}"),
        }
    }

    fn map<F, R>(&self, f: F) -> R
    where
        F: Fn(&String) -> R,
    {
        f(&self.0.to_json())
    }
}

/// State shared by the plugin, the network thread and the background task
/// executor.
pub struct LiveShared {
    pub store: Arc<PresetStore>,
    pub sender: Sender<LiveCommand>,
    /// Rendered-block counter, reported by the live ping as proof of life.
    pub blocks: AtomicU64,
    /// Kernel (voice-pair) count of the engine, published by the audio
    /// thread so prebuilt per-kernel structures match.
    pub kernel_count: AtomicUsize,
    /// Oversampled engine rate, published by the audio thread so prebuilt
    /// material (the convolution impulse) is rendered at the right rate.
    /// The host sample rate; a patch is built for the engine rate its own
    /// oversampling gives at this rate (`engine_rate_for`).
    pub sample_rate: AtomicU32,
}

impl LiveShared {
    /// Creates the shared state and the audio-thread end of the channel.
    #[must_use]
    pub fn new(store: Arc<PresetStore>) -> (Arc<LiveShared>, Receiver<LiveCommand>) {
        let (sender, receiver) = channel();
        let shared = Arc::new(LiveShared {
            store,
            sender,
            blocks: AtomicU64::new(0),
            kernel_count: AtomicUsize::new(0),
            sample_rate: AtomicU32::new(44_100),
        });
        (shared, receiver)
    }

    /// Builds the store's current preset for the engine and sends it
    /// (network / background thread only). Returns the load report.
    pub fn build_and_send(&self) -> Result<LoadReport, String> {
        let generation = self.store.generation();
        let preset = self.store.get();
        let mut report = LoadReport::default();
        patch::connections_report(&preset, &mut report);
        let master = patch::master_from_preset(&preset);
        let kernel_count =
            BuiltPatch::kernel_count(self.kernel_count.load(Ordering::Relaxed), master.polyphony);
        let engine_rate = crate::engine_rate_for(self.sample_rate(), master.oversampling);
        let built = BuiltPatch::build(&preset, kernel_count, engine_rate, &mut report);
        self.sender
            .send(LiveCommand::ApplyBuilt(Box::new(built)))
            .map_err(|_| "engine gone".to_string())?;
        self.store.mark_applied(generation);
        Ok(report)
    }

    fn send(&self, command: LiveCommand) -> String {
        if self.sender.send(command).is_err() {
            "err: engine gone".into()
        } else {
            "ok".into()
        }
    }

    /// Kernels a freshly built multisample set must cover.
    fn multisample_count(&self) -> usize {
        self.kernel_count.load(Ordering::Relaxed).clamp(1, patch::MAX_KERNELS)
    }

    /// The engine's oversampled rate, as last published by the audio
    /// thread (the constructor's guess until `initialize` runs).
    fn sample_rate(&self) -> u32 {
        self.sample_rate.load(Ordering::Relaxed).max(8000)
    }
}

/// A running listener: stops the thread and unregisters on drop.
pub struct LiveHandle {
    pub port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for LiveHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // Unblock `accept` with a throwaway connection.
        let address = SocketAddr::from(([127, 0, 0, 1], self.port));
        let _ = TcpStream::connect_timeout(&address, PROBE_TIMEOUT);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        unregister_instance();
    }
}

/// Whether the live channel is enabled (`SPINWAVE_LIVE` is not `0`).
#[must_use]
pub fn enabled() -> bool {
    std::env::var("SPINWAVE_LIVE").as_deref() != Ok("0")
}

/// Starts the listener unless disabled; picks the forced port or the first
/// free one in the range, and registers the instance for discovery. Call
/// from `initialize()` (never from `Default`, so scanning hosts open no
/// port).
pub fn start(shared: Arc<LiveShared>) -> Option<LiveHandle> {
    if !enabled() {
        return None;
    }

    let forced: Option<u16> =
        std::env::var("SPINWAVE_LIVE_PORT").ok().and_then(|p| p.parse().ok());
    let candidates: Vec<u16> = match forced {
        Some(port) => vec![port],
        None => (PORT_RANGE_START..=PORT_RANGE_END).collect(),
    };

    for port in candidates {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => {
                let stop = Arc::new(AtomicBool::new(false));
                let thread = spawn_listener(listener, shared, stop.clone());
                register_instance(port);
                eprintln!("spinwave: live control listening on 127.0.0.1:{port}");
                return Some(LiveHandle { port, stop, thread });
            }
            Err(_) => continue,
        }
    }
    eprintln!("spinwave: no free live control port in range");
    None
}

// -- Discovery registry -------------------------------------------------------

/// `%TEMP%/spinwave-instances.json`: an array of `{pid, port, exe, started}`
/// entries. Every writer purges the entries that no longer answer a ping
/// and rewrites the file atomically (temp file + rename).
#[must_use]
pub fn registry_path() -> std::path::PathBuf {
    std::env::temp_dir().join("spinwave-instances.json")
}

/// One registry entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistryEntry {
    pub pid: u64,
    pub port: u16,
    pub exe: String,
}

fn read_registry() -> Vec<serde_json::Value> {
    std::fs::read_to_string(registry_path())
        .ok()
        .and_then(|text| serde_json::from_str(text.trim_start_matches('\u{feff}')).ok())
        .unwrap_or_default()
}

fn write_registry(entries: &[serde_json::Value]) {
    let path = registry_path();
    let temp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let text = serde_json::to_string_pretty(entries).unwrap_or_default();
    if std::fs::write(&temp, text).is_ok() && std::fs::rename(&temp, &path).is_err() {
        let _ = std::fs::remove_file(&temp);
    }
}

/// Whether a live instance answers on `port` (connect + ping with a short
/// timeout).
#[must_use]
pub fn probe_port(port: u16) -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, PROBE_TIMEOUT) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(PROBE_TIMEOUT));
    let _ = stream.set_write_timeout(Some(PROBE_TIMEOUT));
    if writeln!(stream, "{{\"cmd\":\"ping\"}}").is_err() || stream.flush().is_err() {
        return false;
    }
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply).is_ok() && reply.starts_with("ok")
}

/// Drops the entries whose instance no longer answers (`keep_pid` is kept
/// without probing: the caller's own, possibly not listening yet).
fn purge_dead(entries: Vec<serde_json::Value>, keep_pid: Option<u64>) -> Vec<serde_json::Value> {
    entries
        .into_iter()
        .filter(|entry| {
            let pid = entry["pid"].as_u64();
            if pid.is_some() && pid == keep_pid {
                return true;
            }
            match entry["port"].as_u64() {
                Some(port) if port <= u16::MAX as u64 => probe_port(port as u16),
                _ => false,
            }
        })
        .collect()
}

fn register_instance(port: u16) {
    let pid = std::process::id() as u64;
    let mut entries = purge_dead(read_registry(), None);
    entries.retain(|e| e["pid"].as_u64() != Some(pid) || e["port"].as_u64() != Some(port as u64));
    entries.push(serde_json::json!({
        "pid": pid,
        "port": port,
        "exe": std::env::current_exe()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_default(),
        "started": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    }));
    write_registry(&entries);
}

/// Removes this process's entries (the listener is gone by now).
fn unregister_instance() {
    let pid = std::process::id() as u64;
    let mut entries = read_registry();
    entries.retain(|e| e["pid"].as_u64() != Some(pid));
    write_registry(&entries);
}

/// Reads the registry, probes every entry and returns the alive ones
/// (DAW-hosted plugins included), rewriting the file without the dead.
#[must_use]
pub fn alive_instances() -> Vec<RegistryEntry> {
    let entries = purge_dead(read_registry(), None);
    write_registry(&entries);
    entries
        .iter()
        .filter_map(|entry| {
            Some(RegistryEntry {
                pid: entry["pid"].as_u64()?,
                port: entry["port"].as_u64()? as u16,
                exe: entry["exe"].as_str().unwrap_or("?").to_string(),
            })
        })
        .collect()
}

// -- Listener ---------------------------------------------------------------

fn spawn_listener(
    listener: TcpListener,
    shared: Arc<LiveShared>,
    stop: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("spinwave-live".into())
        .spawn(move || {
            for stream in listener.incoming() {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                let Ok(stream) = stream else { continue };
                let shared = shared.clone();
                let _ = std::thread::Builder::new()
                    .name("spinwave-live-conn".into())
                    .spawn(move || handle_connection(stream, &shared));
            }
        })
        .ok()
}

/// The preset every instance starts from.
#[must_use]
pub fn default_preset() -> Preset {
    Preset::from_json(r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#)
        .expect("init preset")
}

fn handle_connection(stream: TcpStream, shared: &LiveShared) {
    let Ok(mut writer) = stream.try_clone() else { return };
    let reader = BufReader::new(stream);

    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = handle_line(&line, shared);
        if writeln!(writer, "{reply}").is_err() || writer.flush().is_err() {
            break;
        }
    }
}

/// Handles one protocol line (exposed for tests).
pub fn handle_line(line: &str, shared: &LiveShared) -> String {
    let value: serde_json::Value = match serde_json::from_str(line.trim_start_matches('\u{feff}'))
    {
        Ok(value) => value,
        Err(e) => return format!("err: invalid JSON: {e}"),
    };
    let Some(cmd) = value["cmd"].as_str() else { return "err: missing cmd".into() };

    match cmd {
        "ping" => format!("ok blocks={}", shared.blocks.load(Ordering::Relaxed)),
        "panic" => shared.send(LiveCommand::Panic),
        "get_patch" => format!("ok {}", shared.store.to_json()),
        "preset" => {
            let (preset, mut report) = match patch::load_preset_value(value["preset"].clone()) {
                Ok(loaded) => loaded,
                Err(e) => return format!("err: {e}"),
            };
            shared.store.set(preset);
            match shared.build_and_send() {
                Ok(built_report) => {
                    report.notes.extend(built_report.notes);
                    let report = serde_json::to_value(&report).unwrap_or_default();
                    format!("ok {}", serde_json::json!({ "report": report }))
                }
                Err(e) => format!("err: {e}"),
            }
        }
        "load_sample" => match slot_and_bytes(&value) {
            Ok((slot, bytes, stem)) => match Sample::from_wav_bytes(&bytes) {
                Ok(mut sample) => {
                    sample.name = stem;
                    let payload = materials::sample_to_json(&sample);
                    shared.store.update(|preset| materials::set_slot_sample_json(preset, slot, payload));
                    shared.send(LiveCommand::SetSample { slot, sample: Arc::new(sample) })
                }
                Err(e) => format!("err: {e}"),
            },
            Err(e) => e,
        },
        "import_wavetable" => match slot_and_bytes(&value) {
            Ok((slot, bytes, stem)) => {
                let table = match value["mode"].as_str().unwrap_or("spectral") {
                    "png" => wavetable_from_png(&bytes, &ImageImportOptions::default()),
                    mode @ ("spectral" | "raw") => Sample::from_wav_bytes(&bytes).map(|sample| {
                        let import_mode = if mode == "raw" {
                            AudioImportMode::RawSlice
                        } else {
                            AudioImportMode::Spectral
                        };
                        wavetable_from_audio(
                            &materials::sample_mono(&sample),
                            sample.sample_rate(),
                            &AudioImportOptions { mode: import_mode, ..Default::default() },
                        )
                    }),
                    other => Err(format!("unknown mode: {other} (spectral, raw or png)")),
                };
                match table {
                    Ok(table) => {
                        let json = materials::wavetable_to_json(&table, &stem);
                        shared.store.update(|preset| materials::set_slot_wavetable_json(preset, slot, json));
                        shared.send(LiveCommand::SetWavetable { slot, table: Arc::new(table) })
                    }
                    Err(e) => format!("err: {e}"),
                }
            }
            Err(e) => e,
        },
        "load_sfz" => {
            let Some(slot) = parse_slot(&value) else { return err_slot() };
            let Some(path) = value["path"].as_str() else { return "err: path required".into() };
            let sfz = match materials::sfz_material_from_path(path) {
                Ok(sfz) => sfz,
                Err(e) => return format!("err: {e}"),
            };
            // Each kernel owns its zone playback state: build one source
            // per kernel here on the network thread (the zones' audio is
            // decoded once and shared between them).
            let base_dir = materials::sfz_base_dir(&sfz);
            let sources = match materials::multisample_sources_from_sfz(
                &sfz.text,
                &base_dir,
                shared.multisample_count(),
                materials::decode_wav_zone,
            ) {
                Ok(sources) => sources,
                Err(e) => return format!("err: {e}"),
            };
            shared.store.update(|preset| materials::set_slot_sfz(preset, slot, sfz));
            shared.send(LiveCommand::SetMultisample { slot, sources })
        }
        "note_on" => {
            let Some(note) = value["note"].as_i64() else { return "err: note required".into() };
            shared.send(LiveCommand::NoteOn {
                note: note as i32,
                velocity: value["velocity"].as_f64().unwrap_or(0.8) as f32,
                channel: value["channel"].as_u64().unwrap_or(0) as usize % 16,
            })
        }
        "seq" => match crate::note_sequencer::SeqConfig::from_json(&value["config"]) {
            Ok(config) => shared.send(LiveCommand::Seq(Box::new(config))),
            Err(e) => format!("err: {e}"),
        },
        "note_off" => {
            let Some(note) = value["note"].as_i64() else { return "err: note required".into() };
            shared.send(LiveCommand::NoteOff {
                note: note as i32,
                channel: value["channel"].as_u64().unwrap_or(0) as usize % 16,
            })
        }
        other => format!("err: unknown cmd: {other}"),
    }
}

fn parse_slot(value: &serde_json::Value) -> Option<usize> {
    let slot = value["slot"].as_u64()? as usize;
    (slot < NUM_OSCILLATORS).then_some(slot)
}

fn err_slot() -> String {
    format!("err: slot must be 0..{}", NUM_OSCILLATORS - 1)
}

/// Shared front half of the material commands: validated slot, file bytes
/// and the file stem (used as the sample display name).
fn slot_and_bytes(value: &serde_json::Value) -> Result<(usize, Vec<u8>, String), String> {
    let Some(slot) = parse_slot(value) else { return Err(err_slot()) };
    let Some(path) = value["path"].as_str() else { return Err("err: path required".into()) };
    let bytes =
        std::fs::read(path).map_err(|e| format!("err: cannot read '{path}': {e}"))?;
    let stem = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "sample".to_string());
    Ok((slot, bytes, stem))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nih_plug::params::persist::PersistentField;

    fn shared() -> (Arc<LiveShared>, Receiver<LiveCommand>) {
        let store = Arc::new(PresetStore::default());
        LiveShared::new(store)
    }

    #[test]
    fn set_patch_updates_the_store_and_reports() {
        let (shared, receiver) = shared();
        shared.kernel_count.store(2, Ordering::Relaxed);
        let reply = handle_line(
            r#"{"cmd":"preset","preset":{"synth_version":"1.0.7","preset_name":"Live",
                "settings":{"filter_1_cutoff": 70.0,
                "modulations":[{"source":"lfo_1","destination":"bogus"}]}}}"#,
            &shared,
        );
        assert!(reply.starts_with("ok {"), "{reply}");
        let report: serde_json::Value =
            serde_json::from_str(reply.strip_prefix("ok ").unwrap()).unwrap();
        assert_eq!(report["report"]["ignored_connections"][0], "lfo_1 -> bogus");
        // The audio thread got a prebuilt patch sized for the kernel pool.
        match receiver.try_recv().unwrap() {
            LiveCommand::ApplyBuilt(patch) => {
                assert_eq!(patch.kernels.len(), 4); // polyphony 8 -> 4 pairs
                assert_eq!(patch.kernels[0].filters[0].params.state.midi_cutoff.lane(0), 70.0);
            }
            _ => panic!("expected ApplyBuilt"),
        }
        assert!(shared.store.is_applied());
        // get_patch reads the same store.
        let patch = handle_line(r#"{"cmd":"get_patch"}"#, &shared);
        assert!(patch.contains("\"preset_name\":\"Live\""));
    }

    #[test]
    fn persisted_preset_round_trips_through_the_store() {
        let store = Arc::new(PresetStore::default());
        let field = PersistedPreset(store.clone());
        let json = r#"{"synth_version":"1.0.7","preset_name":"Saved","settings":{"osc_1_level":0.4}}"#;
        let before = store.generation();
        field.set(json.to_string());
        assert!(store.generation() > before);
        assert!(!store.is_applied());
        let saved = field.map(|text| text.clone());
        let parsed = Preset::from_json(&saved).unwrap();
        assert_eq!(parsed.preset_name, "Saved");
        assert_eq!(parsed.settings.parameter("osc_1_level"), Some(0.4));
        // Garbage state is ignored, the store keeps its preset.
        field.set("nope".to_string());
        assert_eq!(store.get().preset_name, "Saved");
    }

    #[test]
    fn material_commands_record_into_the_preset() {
        let (shared, receiver) = shared();
        let dir = std::env::temp_dir();
        let wav = dir.join("spinwave-live-material-test.wav");
        let frames: Vec<f32> = (0..2000).flat_map(|i| {
            let v = (i as f32 * 0.05).sin() * 0.5;
            [v, v]
        }).collect();
        materials::write_wav(wav.to_str().unwrap(), &frames, 44100).unwrap();
        let reply = handle_line(
            &serde_json::json!({"cmd": "load_sample", "slot": 2, "path": wav.to_str().unwrap()}).to_string(),
            &shared,
        );
        assert_eq!(reply, "ok");
        assert!(matches!(receiver.try_recv().unwrap(), LiveCommand::SetSample { slot: 2, .. }));
        let preset = shared.store.get();
        let block = preset.settings.spinwave_materials.as_ref().unwrap();
        assert_eq!(block.slot(2).unwrap().sample.as_ref().unwrap().length, 2000);

        let reply = handle_line(
            &serde_json::json!({"cmd": "import_wavetable", "slot": 3, "path": wav.to_str().unwrap(), "mode": "raw"}).to_string(),
            &shared,
        );
        assert_eq!(reply, "ok");
        assert!(matches!(receiver.try_recv().unwrap(), LiveCommand::SetWavetable { slot: 3, .. }));
        let preset = shared.store.get();
        assert!(materials::slot_wavetable_json(&preset, 3).is_some());
        assert!(preset.settings.wavetables.is_none(), "slot 3 stays out of Vital's array");

        assert!(handle_line(r#"{"cmd":"load_sample","slot":9,"path":"x"}"#, &shared).starts_with("err"));
        let _ = std::fs::remove_file(wav);
    }

    #[test]
    fn registry_purge_keeps_only_answering_entries() {
        // Nothing listens on a fresh ephemeral port: the entry is dropped.
        let free = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let entries = vec![
            serde_json::json!({"pid": 1, "port": free, "exe": "dead"}),
            serde_json::json!({"pid": 2, "exe": "malformed"}),
        ];
        assert!(purge_dead(entries.clone(), None).is_empty());
        assert_eq!(purge_dead(entries, Some(1)).len(), 1);
    }

    #[test]
    fn listener_stops_and_unregisters_on_drop() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (shared, _receiver) = shared();
        let stop = Arc::new(AtomicBool::new(false));
        let thread = spawn_listener(listener, shared, stop.clone());
        let handle = LiveHandle { port, stop, thread };
        assert!(probe_port(port));
        drop(handle);
        assert!(!probe_port(port));
    }
}
