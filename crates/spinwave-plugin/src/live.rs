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
//! `{"cmd":"panic"}`                      — all sounds off
//! `{"cmd":"ping"}`                       — replies `ok blocks=<n>`
//!
//! Set `SPINWAVE_LIVE=0` to disable, `SPINWAVE_LIVE_PORT` to force a port.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use spinwave_engine::kernel::mod_matrix::Connection;
use spinwave_engine::kernel::KernelParams;
use spinwave_params::Preset;

use crate::patch;

const PORT_RANGE_START: u16 = 41929;
const PORT_RANGE_END: u16 = 41979;

/// Commands handed to the audio thread. Heavy structures are prebuilt on
/// the network thread so the audio thread only swaps them in.
pub enum LiveCommand {
    ApplyBuilt {
        kernel: Box<KernelParams>,
        connections: Vec<Connection>,
        effects: Box<spinwave_engine::engine::EffectsParams>,
        master: patch::MasterFromPreset,
    },
    NoteOn { note: i32, velocity: f32, channel: usize },
    NoteOff { note: i32, channel: usize },
    Panic,
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
            // Build the heavy structures here, off the audio thread.
            let kernel = Box::new(patch::kernel_params_from_preset(&preset));
            let connections = patch::connections_from_preset(&preset);
            let effects = Box::new(patch::effects_params_from_preset(&preset));
            let master = patch::master_from_preset(&preset);
            if let Ok(mut slot) = current_preset.lock() {
                *slot = preset;
            }
            send(LiveCommand::ApplyBuilt { kernel, connections, effects, master })
        }
        "note_on" => {
            let Some(note) = value["note"].as_i64() else { return "err: note required".into() };
            send(LiveCommand::NoteOn {
                note: note as i32,
                velocity: value["velocity"].as_f64().unwrap_or(0.8) as f32,
                channel: value["channel"].as_u64().unwrap_or(0) as usize % 16,
            })
        }
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
