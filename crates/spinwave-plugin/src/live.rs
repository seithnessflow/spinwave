//! Live control channel: a localhost TCP listener that feeds commands to
//! the running engine (used by the standalone; enabled via the
//! `SPINWAVE_LIVE_PORT` environment variable).
//!
//! Protocol: one JSON object per line, response `ok` or `err: ...`.
//! Commands:
//! `{"cmd":"preset","preset":{...}}`      — apply a full .vital preset
//! `{"cmd":"note_on","note":48,"velocity":0.8,"channel":0}`
//! `{"cmd":"note_off","note":48,"channel":0}`
//! `{"cmd":"panic"}`                      — all sounds off
//! `{"cmd":"ping"}`

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

use spinwave_params::Preset;

pub enum LiveCommand {
    ApplyPreset(Box<Preset>),
    NoteOn { note: i32, velocity: f32, channel: usize },
    NoteOff { note: i32, channel: usize },
    Panic,
}

/// Starts the listener thread; returns the audio-side receiver.
/// The audio thread drains it at block boundaries and increments
/// `processed_blocks` — the ping reply carries it as a proof of life
/// that audio is actually being rendered.
pub fn start_listener(
    port: u16,
    processed_blocks: Arc<AtomicU64>,
) -> std::io::Result<Receiver<LiveCommand>> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let (sender, receiver) = channel();

    std::thread::Builder::new()
        .name("spinwave-live".into())
        .spawn(move || {
            for stream in listener.incoming().flatten() {
                let sender = sender.clone();
                let blocks = processed_blocks.clone();
                std::thread::spawn(move || handle_connection(stream, sender, blocks));
            }
        })?;

    eprintln!("spinwave: live control listening on 127.0.0.1:{port}");
    Ok(receiver)
}

fn handle_connection(
    stream: TcpStream,
    sender: Sender<LiveCommand>,
    processed_blocks: Arc<AtomicU64>,
) {
    let Ok(write_half) = stream.try_clone() else { return };
    let mut writer = write_half;
    let reader = BufReader::new(stream);

    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match parse_command(&line) {
            Ok(Some(command)) => {
                if sender.send(command).is_err() {
                    "err: engine gone".to_string()
                } else {
                    "ok".to_string()
                }
            }
            // Ping reports how many audio blocks the engine has rendered.
            Ok(None) => format!("ok blocks={}", processed_blocks.load(Ordering::Relaxed)),
            Err(message) => format!("err: {message}"),
        };
        if writeln!(writer, "{reply}").is_err() || writer.flush().is_err() {
            break;
        }
    }
}

fn parse_command(line: &str) -> Result<Option<LiveCommand>, String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("invalid JSON: {e}"))?;
    let cmd = value["cmd"].as_str().ok_or("missing cmd")?;
    match cmd {
        "ping" => Ok(None),
        "panic" => Ok(Some(LiveCommand::Panic)),
        "preset" => {
            let preset: Preset = serde_json::from_value(value["preset"].clone())
                .map_err(|e| format!("invalid preset: {e}"))?;
            Ok(Some(LiveCommand::ApplyPreset(Box::new(preset))))
        }
        "note_on" => Ok(Some(LiveCommand::NoteOn {
            note: value["note"].as_i64().ok_or("note required")? as i32,
            velocity: value["velocity"].as_f64().unwrap_or(0.8) as f32,
            channel: value["channel"].as_u64().unwrap_or(0) as usize % 16,
        })),
        "note_off" => Ok(Some(LiveCommand::NoteOff {
            note: value["note"].as_i64().ok_or("note required")? as i32,
            channel: value["channel"].as_u64().unwrap_or(0) as usize % 16,
        })),
        other => Err(format!("unknown cmd: {other}")),
    }
}

/// Port from `SPINWAVE_LIVE_PORT`, if configured.
pub fn configured_port() -> Option<u16> {
    std::env::var("SPINWAVE_LIVE_PORT").ok()?.parse().ok()
}
