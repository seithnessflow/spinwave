//! Spinwave MCP server: exposes the synth engine to LLM agents over the
//! Model Context Protocol (stdio transport, newline-delimited JSON-RPC).
//!
//! Register with: `claude mcp add spinwave -- <path-to>/spinwave-mcp.exe`

mod analysis;
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
        other => Err(format!("unknown tool: {other}")),
    }
}
