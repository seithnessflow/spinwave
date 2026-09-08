//! Spinwave MCP server: exposes the synth engine to LLM agents over the
//! Model Context Protocol (stdio transport, newline-delimited JSON-RPC).
//!
//! Register with: `claude mcp add spinwave -- <path-to>/spinwave-mcp.exe`

mod analysis;
mod live_client;
mod session;

use std::io::{BufRead, Write};

use serde_json::{json, Value};
use session::{NoteSpec, Session};

const PROTOCOL_VERSION: &str = "2024-11-05";

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut session = Session::new();

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim_start_matches('\u{feff}');
        if line.trim().is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(line) else {
            eprintln!("spinwave-mcp: ignoring invalid JSON line");
            continue;
        };

        let method = message["method"].as_str().unwrap_or("");
        let id = message.get("id").cloned();

        // Notifications (no id) get no response.
        let Some(id) = id else {
            continue;
        };

        let result = match method {
            "initialize" => Ok(json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "spinwave",
                    "version": env!("CARGO_PKG_VERSION"),
                },
                "instructions": SERVER_INSTRUCTIONS,
            })),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tool_definitions() })),
            "tools/call" => handle_tool_call(&mut session, &message["params"]),
            _ => Err((-32601, format!("method not found: {method}"))),
        };

        let response = match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err((code, text)) => json!({
                "jsonrpc": "2.0", "id": id,
                "error": { "code": code, "message": text },
            }),
        };
        let mut out = stdout.lock();
        let _ = writeln!(out, "{response}");
        let _ = out.flush();
    }
}

const SERVER_INSTRUCTIONS: &str = "Spinwave is a wavetable synthesizer (Rust rework of \
Vital). Workflow: inspect parameters with describe_params, edit the patch with set_params \
/ add_modulation (or load a full .vital preset with set_patch), then hear the result with \
play — it renders notes to a WAV and returns an audio analysis (levels, spectrum, envelope, \
pitch). Iterate: tweak, play, read the analysis. Parameter values are ENGINE values \
(ranges from describe_params). Notable scales: envelope times are quartic (stored 2.0 = \
16 s), lfo/effect frequencies are log2 Hz (stored 3.0 = 8 Hz).";

fn tool_definitions() -> Value {
    json!([
        {
            "name": "describe_params",
            "description": "Lists synth parameters. Without arguments: group summary. With `search`: parameters whose name contains the string, with range/default/units.",
            "inputSchema": { "type": "object", "properties": {
                "search": { "type": "string", "description": "Substring or group prefix, e.g. 'osc_1', 'filter', 'reverb'" },
                "limit": { "type": "integer", "default": 40 }
            }}
        },
        {
            "name": "get_patch",
            "description": "Returns the current patch as .vital preset JSON.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "set_patch",
            "description": "Replaces the whole patch with a .vital preset JSON string.",
            "inputSchema": { "type": "object", "properties": {
                "preset_json": { "type": "string" }
            }, "required": ["preset_json"] }
        },
        {
            "name": "set_params",
            "description": "Sets parameter engine values by name, e.g. {\"filter_1_on\": 1, \"filter_1_cutoff\": 70}. Unknown names are skipped with a warning; out-of-range values are clamped.",
            "inputSchema": { "type": "object", "properties": {
                "params": { "type": "object", "additionalProperties": { "type": "number" } }
            }, "required": ["params"] }
        },
        {
            "name": "add_modulation",
            "description": "Connects a modulation source (e.g. lfo_1, env_2, velocity, macro_control_1) to a destination parameter (e.g. filter_1_cutoff).",
            "inputSchema": { "type": "object", "properties": {
                "source": { "type": "string" },
                "destination": { "type": "string" },
                "amount": { "type": "number", "description": "-1..1" },
                "bipolar": { "type": "boolean", "default": false },
                "stereo": { "type": "boolean", "default": false },
                "power": { "type": "number", "default": 0.0 }
            }, "required": ["source", "destination", "amount"] }
        },
        {
            "name": "clear_modulations",
            "description": "Removes every modulation connection.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "play",
            "description": "Renders notes through the full engine to a WAV file and returns an audio analysis (peak/RMS, spectral centroid and bands, envelope timing, pitch, stereo width). Times in seconds, notes as MIDI numbers.",
            "inputSchema": { "type": "object", "properties": {
                "notes": { "type": "array", "items": { "type": "object", "properties": {
                    "note": { "type": "integer" },
                    "start": { "type": "number" },
                    "duration": { "type": "number" },
                    "velocity": { "type": "number", "default": 0.8 },
                    "channel": { "type": "integer", "default": 0 }
                }, "required": ["note", "start", "duration"] }},
                "seconds": { "type": "number", "description": "Total render length; default = last note end + 1.5s tail" },
                "bpm": { "type": "number", "default": 120.0 },
                "out_path": { "type": "string", "description": "Output WAV path; default spinwave-render.wav in the working directory" }
            }, "required": ["notes"] }
        },
        {
            "name": "analyze",
            "description": "Re-analyzes the last render without re-rendering.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "save_preset",
            "description": "Saves the current patch to a .vital file.",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" },
                "name": { "type": "string", "description": "Preset display name" }
            }, "required": ["path"] }
        },
        {
            "name": "load_preset",
            "description": "Loads a .vital preset file as the current patch.",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" }
            }, "required": ["path"] }
        },
        {
            "name": "list_audio_devices",
            "description": "Lists the available audio output devices for live mode. Pick the user's actual monitors (e.g. their audio interface), not the Windows default.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "live_start",
            "description": "Launches the standalone synth with real audio output and connects the live control channel. ALWAYS pass output_device (from list_audio_devices) matching the user's monitors — the Windows default is often the wrong device. The response reports the device actually opened.",
            "inputSchema": { "type": "object", "properties": {
                "output_device": { "type": "string", "description": "Exact device name from list_audio_devices" },
                "sample_rate": { "type": "integer", "description": "Try 44100 if the device rejects the default 48000" },
                "period_size": { "type": "integer", "description": "Buffer size; default 512" }
            }}
        },
        {
            "name": "live_apply",
            "description": "Pushes the current patch to the running live synth (takes effect immediately, audible).",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "live_set_params",
            "description": "Sets parameters AND pushes them to the live synth in one step — the audible way to tweak while sound plays.",
            "inputSchema": { "type": "object", "properties": {
                "params": { "type": "object", "additionalProperties": { "type": "number" } }
            }, "required": ["params"] }
        },
        {
            "name": "live_sequence",
            "description": "Plays notes on the live synth in real time (blocking; max 20 s). Same note format as play.",
            "inputSchema": { "type": "object", "properties": {
                "notes": { "type": "array", "items": { "type": "object", "properties": {
                    "note": { "type": "integer" },
                    "start": { "type": "number" },
                    "duration": { "type": "number" },
                    "velocity": { "type": "number", "default": 0.8 },
                    "channel": { "type": "integer", "default": 0 }
                }, "required": ["note", "start", "duration"] }}
            }, "required": ["notes"] }
        },
        {
            "name": "live_note",
            "description": "Holds or releases a single note on the live synth: action 'on' or 'off'.",
            "inputSchema": { "type": "object", "properties": {
                "action": { "type": "string", "enum": ["on", "off"] },
                "note": { "type": "integer" },
                "velocity": { "type": "number", "default": 0.8 }
            }, "required": ["action", "note"] }
        },
        {
            "name": "live_instances",
            "description": "Discovers running Spinwave instances (standalone AND plugins hosted in a DAW) and pings them. Use live_attach to control one.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "live_attach",
            "description": "Points the live_* tools at a specific running instance's port (from live_instances) — e.g. a Spinwave loaded in the user's DAW.",
            "inputSchema": { "type": "object", "properties": {
                "port": { "type": "integer" }
            }, "required": ["port"] }
        },
        {
            "name": "live_get_patch",
            "description": "Reads the current patch (.vital JSON) back from the attached live instance.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "live_panic",
            "description": "Immediately silences every voice on the live synth.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "live_stop",
            "description": "Stops the live standalone synth.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

fn handle_tool_call(session: &mut Session, params: &Value) -> Result<Value, (i64, String)> {
    let name = params["name"].as_str().unwrap_or("");
    let args = &params["arguments"];

    match call_tool(session, name, args) {
        Ok(value) => {
            let text = match value {
                Value::String(text) => text,
                other => serde_json::to_string_pretty(&other).unwrap_or_default(),
            };
            Ok(json!({ "content": [{ "type": "text", "text": text }] }))
        }
        Err(text) => Ok(json!({
            "content": [{ "type": "text", "text": format!("Error: {text}") }],
            "isError": true,
        })),
    }
}

fn call_tool(session: &mut Session, name: &str, args: &Value) -> Result<Value, String> {
    match name {
        "describe_params" => {
            let search = args["search"].as_str();
            let limit = args["limit"].as_u64().unwrap_or(40) as usize;
            Ok(session.describe_params(search, limit.clamp(1, 200)))
        }
        "get_patch" => session
            .preset
            .to_json_pretty()
            .map(Value::String)
            .map_err(|e| e.to_string()),
        "set_patch" => {
            let json_text = args["preset_json"].as_str().ok_or("preset_json required")?;
            session.load_preset_json(json_text).map(Value::String)
        }
        "set_params" => {
            let values = args["params"].as_object().ok_or("params object required")?;
            session.set_params(values).map(Value::String)
        }
        "add_modulation" => {
            let source = args["source"].as_str().ok_or("source required")?;
            let destination = args["destination"].as_str().ok_or("destination required")?;
            let amount = args["amount"].as_f64().ok_or("amount required")? as f32;
            session
                .add_modulation(
                    source,
                    destination,
                    amount,
                    args["bipolar"].as_bool().unwrap_or(false),
                    args["stereo"].as_bool().unwrap_or(false),
                    args["power"].as_f64().unwrap_or(0.0) as f32,
                )
                .map(Value::String)
        }
        "clear_modulations" => Ok(Value::String(session.clear_modulations())),
        "play" => {
            let notes: Vec<NoteSpec> = serde_json::from_value(args["notes"].clone())
                .map_err(|e| format!("invalid notes: {e}"))?;
            let seconds = args["seconds"].as_f64().map(|s| s as f32);
            let bpm = args["bpm"].as_f64().unwrap_or(120.0) as f32;
            let out_path = args["out_path"].as_str().unwrap_or("spinwave-render.wav");
            session.render(&notes, seconds, bpm, out_path).map(|(summary, analysis)| {
                json!({ "summary": summary, "analysis": analysis })
            })
        }
        "analyze" => session.analyze_last().map(|a| serde_json::to_value(a).unwrap()),
        "save_preset" => {
            let path = args["path"].as_str().ok_or("path required")?;
            if let Some(name) = args["name"].as_str() {
                session.preset.preset_name = name.to_string();
            }
            let text = session.preset.to_json_pretty().map_err(|e| e.to_string())?;
            std::fs::write(path, text).map_err(|e| e.to_string())?;
            Ok(Value::String(format!("saved to {path}")))
        }
        "load_preset" => {
            let path = args["path"].as_str().ok_or("path required")?;
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            session.load_preset_json(&text).map(Value::String)
        }
        "list_audio_devices" => crate::live_client::LiveLink::list_output_devices()
            .map(|devices| Value::String(devices.join("\n"))),
        "live_start" => {
            let device = args["output_device"].as_str();
            let sample_rate = args["sample_rate"].as_u64().map(|r| r as u32);
            let period = args["period_size"].as_u64().map(|p| p as u32);
            let message = session.live.start(device, sample_rate, period)?;
            // Bring the running synth in line with the current patch.
            let patch = session.live_push_preset()?;
            Ok(Value::String(format!("{message}; {patch}")))
        }
        "live_apply" => session.live_push_preset().map(Value::String),
        "live_set_params" => {
            let values = args["params"].as_object().ok_or("params object required")?;
            let applied = session.set_params(values)?;
            let pushed = session.live_push_preset()?;
            Ok(Value::String(format!("{applied}; {pushed}")))
        }
        "live_sequence" => {
            let notes: Vec<NoteSpec> = serde_json::from_value(args["notes"].clone())
                .map_err(|e| format!("invalid notes: {e}"))?;
            session.live_sequence(&notes).map(Value::String)
        }
        "live_note" => {
            let note = args["note"].as_i64().ok_or("note required")? as i32;
            let velocity = args["velocity"].as_f64().unwrap_or(0.8) as f32;
            match args["action"].as_str() {
                Some("on") => session.live.note_on(note, velocity, 0).map(Value::String),
                Some("off") => session.live.note_off(note, 0).map(Value::String),
                _ => Err("action must be 'on' or 'off'".into()),
            }
        }
        "live_instances" => {
            let instances = crate::live_client::LiveLink::list_instances();
            if instances.is_empty() {
                Ok(Value::String(
                    "no live Spinwave instance found (standalone not running, no plugin loaded)"
                        .into(),
                ))
            } else {
                let lines: Vec<String> = instances
                    .iter()
                    .map(|(pid, port, exe)| format!("port {port}: {exe} (pid {pid})"))
                    .collect();
                Ok(Value::String(lines.join("\n")))
            }
        }
        "live_attach" => {
            let port = args["port"].as_u64().ok_or("port required")? as u16;
            session.live.attach(port).map(Value::String)
        }
        "live_get_patch" => {
            let json = session.live.get_patch()?;
            // Adopt the live patch as the session's current patch too.
            let note = session.load_preset_json(&json)?;
            Ok(Value::String(format!("{json}\n\n({note})")))
        }
        "live_panic" => session
            .live
            .send(&json!({"cmd": "panic"}))
            .map(|_| Value::String("all sounds off".into())),
        "live_stop" => Ok(Value::String(session.live.stop())),
        other => Err(format!("unknown tool: {other}")),
    }
}
