//! Live control channel: every Spinwave instance (standalone or hosted in
//! a DAW) opens a localhost TCP listener and registers itself in a
//! discovery file, so LLM agents can find and drive running instances.
//!
//! Protocol: one JSON object per line, response `ok`/`ok {...}` or `err: ...`.
//! Commands:
//! `{"cmd":"preset","preset":{...}}`      — apply a full .vital preset
//! `{"cmd":"get_patch"}`                  — returns the current preset JSON
//! `{"cmd":"note_on","note":48,"velocity":0.8,"channel":0}`
//! `{"cmd":"note_off","note":48,"channel":0}`
//! `{"cmd":"seq","config":{...}}`         — arp/step sequencer config
//!                                          (see `note_sequencer::SeqConfig::from_json`)
//! `{"cmd":"panic"}`                      — all sounds off (flushes the sequencer)
//! `{"cmd":"ping"}`                       — replies `ok blocks=<n>`
//!
//! Material loading (paths are local to this machine; files are read and
//! decoded on the network thread, the audio thread only swaps the result in):
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
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use spinwave_dsp::oscillator::sample_source::BUFFER_SAMPLES;
use spinwave_dsp::oscillator::{Multisample, Sample};
use spinwave_dsp::wavetable::{
    wavetable_from_audio, wavetable_from_png, AudioImportMode, AudioImportOptions,
    ImageImportOptions, Wavetable,
};
use spinwave_engine::allocator::MAX_POLYPHONY;
use spinwave_engine::kernel::mod_matrix::{Connection, NUM_OSCILLATORS};
use spinwave_engine::kernel::KernelParams;
use spinwave_params::Preset;

use crate::patch;

const PORT_RANGE_START: u16 = 41929;
const PORT_RANGE_END: u16 = 41979;

/// Upper bound on voice-pair kernels (two voices per kernel): enough
/// prebuilt `Multisample` instances for every kernel at max polyphony.
const MAX_KERNEL_PAIRS: usize = MAX_POLYPHONY.div_ceil(2);

/// Commands handed to the audio thread. Heavy structures are prebuilt on
/// the network thread so the audio thread only swaps them in.
pub enum LiveCommand {
    ApplyBuilt {
        kernel: Box<KernelParams>,
        connections: Vec<Connection>,
        effects_connections: Vec<spinwave_engine::engine::EffectsConnection>,
        effects: Box<spinwave_engine::engine::EffectsParams>,
        bus_a: Box<spinwave_engine::engine::EffectsParams>,
        bus_b: Box<spinwave_engine::engine::EffectsParams>,
        master: patch::MasterFromPreset,
        wavetables: Vec<(usize, std::sync::Arc<spinwave_dsp::wavetable::Wavetable>)>,
    },
    /// Installs sample material in one oscillator slot of every kernel
    /// (Sample + Granular engines).
    SetSample { slot: usize, sample: Arc<Sample> },
    /// Installs a wavetable in one oscillator slot of every kernel.
    SetWavetable { slot: usize, table: Arc<Wavetable> },
    /// Installs an SFZ instrument in one oscillator slot. `Multisample` is
    /// not `Clone`, so the network thread prebuilds one per possible kernel
    /// ([`MAX_KERNEL_PAIRS`]); the drain pops one per kernel.
    SetMultisample { slot: usize, instruments: Vec<Multisample> },
    NoteOn { note: i32, velocity: f32, channel: usize },
    NoteOff { note: i32, channel: usize },
    /// Reconfigures the arp/step sequencer (parsed off the audio thread).
    Seq(Box<crate::note_sequencer::SeqConfig>),
    Panic,
}

/// The original-rate audio of one channel folded to mono, read back through
/// the public tier accessors (tier 1 is the unfiltered original, guarded by
/// [`BUFFER_SAMPLES`] on each side).
fn sample_mono(sample: &Sample) -> Vec<f32> {
    let length = sample.original_length();
    if length == 0 {
        return Vec::new();
    }
    let left = &sample.left_buffer(1)[BUFFER_SAMPLES..BUFFER_SAMPLES + length];
    if !sample.stereo() {
        return left.to_vec();
    }
    let right = &sample.right_buffer(1)[BUFFER_SAMPLES..BUFFER_SAMPLES + length];
    left.iter().zip(right).map(|(l, r)| 0.5 * (l + r)).collect()
}

pub struct LiveState {
    pub receiver: Receiver<LiveCommand>,
    pub port: u16,
}

/// Starts the listener unless disabled; picks the forced port or the first
/// free one in the range, and registers the instance for discovery.
pub fn start(processed_blocks: Arc<AtomicU64>) -> Option<LiveState> {
    if std::env::var("SPINWAVE_LIVE").as_deref() == Ok("0") {
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
                let receiver = spawn_listener(listener, processed_blocks);
                register_instance(port);
                eprintln!("spinwave: live control listening on 127.0.0.1:{port}");
                return Some(LiveState { receiver, port });
            }
            Err(_) => continue,
        }
    }
    eprintln!("spinwave: no free live control port in range");
    None
}

/// Discovery registry: `%TEMP%/spinwave-instances.json`, an array of
/// `{pid, port, exe, started}` entries. Stale entries are pruned by
/// readers (a dead port simply refuses the connection).
fn registry_path() -> std::path::PathBuf {
    std::env::temp_dir().join("spinwave-instances.json")
}

fn register_instance(port: u16) {
    let path = registry_path();
    let mut entries: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str(text.trim_start_matches('\u{feff}')).ok())
        .unwrap_or_default();

    let pid = std::process::id();
    entries.retain(|e| e["pid"].as_u64() != Some(pid as u64));
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
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&entries).unwrap_or_default());
}

fn spawn_listener(
    listener: TcpListener,
    processed_blocks: Arc<AtomicU64>,
) -> Receiver<LiveCommand> {
    let (sender, receiver) = channel();
    // The network side owns the authoritative preset copy: get_patch reads
    // it, and the heavy preset→params mapping happens here, off the audio
    // thread.
    let current_preset: Arc<Mutex<Preset>> = Arc::new(Mutex::new(default_preset()));

    let _ = std::thread::Builder::new().name("spinwave-live".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let sender = sender.clone();
            let blocks = processed_blocks.clone();
            let preset = current_preset.clone();
            std::thread::spawn(move || handle_connection(stream, sender, blocks, preset));
        }
    });
    receiver
}

fn default_preset() -> Preset {
    Preset::from_json(r#"{"synth_version":"1.0.7","preset_name":"Init","settings":{}}"#)
        .expect("init preset")
}

fn handle_connection(
    stream: TcpStream,
    sender: Sender<LiveCommand>,
    processed_blocks: Arc<AtomicU64>,
    current_preset: Arc<Mutex<Preset>>,
) {
    let Ok(mut writer) = stream.try_clone() else { return };
    let reader = BufReader::new(stream);

    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = handle_line(&line, &sender, &processed_blocks, &current_preset);
        if writeln!(writer, "{reply}").is_err() || writer.flush().is_err() {
            break;
        }
    }
}

fn handle_line(
    line: &str,
    sender: &Sender<LiveCommand>,
    processed_blocks: &AtomicU64,
    current_preset: &Mutex<Preset>,
) -> String {
    let value: serde_json::Value = match serde_json::from_str(line.trim_start_matches('\u{feff}'))
    {
        Ok(value) => value,
        Err(e) => return format!("err: invalid JSON: {e}"),
    };
    let Some(cmd) = value["cmd"].as_str() else { return "err: missing cmd".into() };

    let send = |command: LiveCommand| -> String {
        if sender.send(command).is_err() {
            "err: engine gone".into()
        } else {
            "ok".into()
        }
    };

    match cmd {
        "ping" => format!("ok blocks={}", processed_blocks.load(Ordering::Relaxed)),
        "panic" => send(LiveCommand::Panic),
        "get_patch" => match current_preset.lock() {
            Ok(preset) => match preset.to_json() {
                Ok(json) => format!("ok {json}"),
                Err(e) => format!("err: {e}"),
            },
            Err(_) => "err: preset lock poisoned".into(),
        },
        "preset" => {
            let preset: Preset = match serde_json::from_value(value["preset"].clone()) {
                Ok(preset) => preset,
                Err(e) => return format!("err: invalid preset: {e}"),
            };
            // Build the heavy structures here, off the audio thread —
            // including rendering any embedded wavetables.
            let kernel = Box::new(patch::kernel_params_from_preset(&preset));
            let connections = patch::connections_from_preset(&preset);
            let effects_connections = patch::effects_connections_from_preset(&preset);
            let effects = Box::new(patch::effects_params_from_preset(&preset));
            let bus_a = Box::new(patch::effects_params_from_preset_prefixed(&preset, "bus_a_"));
            let bus_b = Box::new(patch::effects_params_from_preset_prefixed(&preset, "bus_b_"));
            let master = patch::master_from_preset(&preset);
            let wavetables = patch::wavetables_from_preset(&preset);
            if let Ok(mut slot) = current_preset.lock() {
                *slot = preset;
            }
            send(LiveCommand::ApplyBuilt {
                kernel,
                connections,
                effects_connections,
                effects,
                bus_a,
                bus_b,
                master,
                wavetables,
            })
        }
        "load_sample" => match slot_and_bytes(&value) {
            Ok((slot, bytes, stem)) => match Sample::from_wav_bytes(&bytes) {
                Ok(mut sample) => {
                    sample.name = stem;
                    send(LiveCommand::SetSample { slot, sample: Arc::new(sample) })
                }
                Err(e) => format!("err: {e}"),
            },
            Err(e) => e,
        },
        "import_wavetable" => match slot_and_bytes(&value) {
            Ok((slot, bytes, _)) => {
                let table = match value["mode"].as_str().unwrap_or("spectral") {
                    "png" => wavetable_from_png(&bytes, &ImageImportOptions::default()),
                    mode @ ("spectral" | "raw") => Sample::from_wav_bytes(&bytes).map(|sample| {
                        let import_mode = if mode == "raw" {
                            AudioImportMode::RawSlice
                        } else {
                            AudioImportMode::Spectral
                        };
                        wavetable_from_audio(
                            &sample_mono(&sample),
                            sample.sample_rate(),
                            &AudioImportOptions { mode: import_mode, ..Default::default() },
                        )
                    }),
                    other => Err(format!("unknown mode: {other} (spectral, raw or png)")),
                };
                match table {
                    Ok(table) => send(LiveCommand::SetWavetable { slot, table: Arc::new(table) }),
                    Err(e) => format!("err: {e}"),
                }
            }
            Err(e) => e,
        },
        "load_sfz" => {
            let Some(slot) = parse_slot(&value) else { return err_slot() };
            let Some(path) = value["path"].as_str() else { return "err: path required".into() };
            let text = match std::fs::read_to_string(path) {
                Ok(text) => text,
                Err(e) => return format!("err: cannot read '{path}': {e}"),
            };
            let base_dir = std::path::Path::new(path)
                .parent()
                .map(|d| d.to_path_buf())
                .unwrap_or_default();
            // Multisample is not Clone: build one instance per possible
            // kernel here on the network thread. The referenced WAV bytes
            // are read once and cached; each instance still rebuilds its
            // zones' band-limited pyramids, an accepted one-time load cost.
            let mut cache: std::collections::HashMap<String, Option<Vec<u8>>> =
                std::collections::HashMap::new();
            let mut instruments = Vec::with_capacity(MAX_KERNEL_PAIRS);
            for _ in 0..MAX_KERNEL_PAIRS {
                let built = Multisample::from_sfz(&text, |sample_path| {
                    let bytes = cache
                        .entry(sample_path.to_string())
                        .or_insert_with(|| {
                            let relative = sample_path.replace('\\', "/");
                            std::fs::read(base_dir.join(relative)).ok()
                        })
                        .as_ref()?;
                    Sample::from_wav_bytes(bytes).ok()
                });
                match built {
                    Ok(instrument) => instruments.push(instrument),
                    Err(e) => return format!("err: invalid SFZ: {e}"),
                }
            }
            send(LiveCommand::SetMultisample { slot, instruments })
        }
        "note_on" => {
            let Some(note) = value["note"].as_i64() else { return "err: note required".into() };
            send(LiveCommand::NoteOn {
                note: note as i32,
                velocity: value["velocity"].as_f64().unwrap_or(0.8) as f32,
                channel: value["channel"].as_u64().unwrap_or(0) as usize % 16,
            })
        }
        "seq" => match crate::note_sequencer::SeqConfig::from_json(&value["config"]) {
            Ok(config) => send(LiveCommand::Seq(Box::new(config))),
            Err(e) => format!("err: {e}"),
        },
        "note_off" => {
            let Some(note) = value["note"].as_i64() else { return "err: note required".into() };
            send(LiveCommand::NoteOff {
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
