//! Spinwave MCP server: exposes the synth engine to LLM agents over the
//! Model Context Protocol (stdio transport, newline-delimited JSON-RPC).
//!
//! Register with: `claude mcp add spinwave -- <path-to>/spinwave-mcp.exe`
//!
//! Renders with a relative `out_path` land in the output directory chosen
//! at startup: `--out-dir <dir>` (or `SPINWAVE_OUT_DIR`), else the
//! repository root when the server runs from a build tree, else the
//! current directory.

// The tool-definition `json!` literal nests deeper than the default limit.
#![recursion_limit = "256"]

use spinwave_control::ops;
use spinwave_control::session;

use std::io::{BufRead, Write};

use serde_json::{json, Value};
use session::{NoteSpec, Session};

const PROTOCOL_VERSION: &str = "2024-11-05";

/// The output directory for relative render paths (see the module docs).
fn output_dir() -> std::path::PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--out-dir" {
            if let Some(dir) = args.next() {
                return dir.into();
            }
        } else if let Some(dir) = arg.strip_prefix("--out-dir=") {
            return dir.into();
        }
    }
    if let Ok(dir) = std::env::var("SPINWAVE_OUT_DIR") {
        return dir.into();
    }
    Session::repo_dir()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default()
}

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut session = Session::with_output_dir(output_dir());

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
16 s), lfo/effect frequencies are log2 Hz (stored 3.0 = 8 Hz). Live flow: live_start (or \
live_attach), live_apply, then live_note / live_sequence — and the `sequencer` tool turns \
held notes into an arpeggio or step pattern on the live instance. Material: load_sample / \
import_wavetable / load_sfz put audio files, drawn-spectrum PNGs and SFZ instruments into \
an oscillator slot; pick the engine with osc_N_engine (0 wavetable, 1 sample, 2 granular, \
3 multisample).";

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
            "description": "Replaces the whole patch with a .vital preset JSON string. Old Vital versions are migrated; the response includes a load report (ignored modulations, unknown parameters, migrations applied).",
            "inputSchema": { "type": "object", "properties": {
                "preset_json": { "type": "string" }
            }, "required": ["preset_json"] }
        },
        {
            "name": "set_params",
            "description": "Sets parameter engine values by name, e.g. {\"filter_1_on\": 1, \"filter_1_cutoff\": 70}. Accepts Vital's 794 parameters and the Spinwave namespace (osc_4_*, env_7..8, lfo_9..12, macro_control_5..8, osc_N_engine, osc_N_smp_*/gran_*, noise_*, bus_a_*/bus_b_* mixer + effect chains, fx_split_*, lfo_N_generator). Unknown names are skipped with a warning; out-of-range values are clamped.",
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
                "out_path": { "type": "string", "description": "Output WAV path; relative paths land in the server's output directory (see the response). Default spinwave-render.wav" },
                "overwrite": { "type": "boolean", "default": false, "description": "Replace an existing file; otherwise a -1, -2... suffix keeps the previous take" }
            }, "required": ["notes"] }
        },
        {
            "name": "analyze",
            "description": "Re-analyzes the last render without re-rendering.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "analyze_file",
            "description": "Listens to ANY audio file (WAV/MP3/FLAC/OGG/M4A) — reference tracks, samples — and returns the full analysis: levels, spectrum, movement (wobble/LFO rates!), envelope, pitch, texture. Use start/duration to target a section (e.g. the drop).",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" },
                "start": { "type": "number", "description": "Segment start in seconds" },
                "duration": { "type": "number", "description": "Segment length in seconds (keep <= 30 for speed)" }
            }, "required": ["path"] }
        },
        {
            "name": "listen",
            "description": "LISTENS to a track over time with a clock (up to 3 min): walks the audio in ~0.5s observation frames and returns the musical STRUCTURE — BPM estimate, a narrated timeline (drops, breakdowns, buildups, bass entries, stereo moves) and a compact frame overview. The complete way to hear a song unfold; use analyze_file for a single static snapshot instead.",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" },
                "start": { "type": "number", "description": "Segment start in seconds" },
                "duration": { "type": "number", "description": "Up to 180 s; default 120" },
                "step": { "type": "number", "description": "Observation step in seconds (0.1-2, default 0.5)" },
                "full_frames": { "type": "boolean", "description": "Include every frame instead of the 2s overview", "default": false }
            }, "required": ["path"] }
        },
        {
            "name": "compare",
            "description": "Compares a reference audio file against the LAST RENDER and describes the gaps in sound-design terms (louder/brighter/more sub/movement rates/width/dirtiness). The tool for 'make it sound like this'.",
            "inputSchema": { "type": "object", "properties": {
                "reference_path": { "type": "string" },
                "start": { "type": "number" },
                "duration": { "type": "number" }
            }, "required": ["reference_path"] }
        },
        {
            "name": "save_preset",
            "description": "Saves the current patch to a .vital file (materials embedded). Spinwave-only parameters at their default are omitted so Vital can load the file too.",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" },
                "name": { "type": "string", "description": "Preset display name" }
            }, "required": ["path"] }
        },
        {
            "name": "load_preset",
            "description": "Loads a .vital preset file as the current patch (any Vital version: old ones are migrated). The response includes a load report.",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" }
            }, "required": ["path"] }
        },
        {
            "name": "load_sample",
            "description": "Loads an audio file (WAV/MP3/FLAC/OGG/M4A) into one oscillator slot's Sample AND Granular engines. Then set osc_N_engine to 1 (Sample) or 2 (Granular) and osc_N_on to 1 to hear it. With live=true it is also pushed to the attached live instance (non-WAV audio is transcoded to a temp WAV for it).",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" },
                "slot": { "type": "integer", "minimum": 0, "maximum": 3, "description": "Oscillator slot 0..3 (osc_1..osc_4)" },
                "live": { "type": "boolean", "default": false }
            }, "required": ["path", "slot"] }
        },
        {
            "name": "import_wavetable",
            "description": "Builds a wavetable from a file and installs it in one oscillator slot (the Wavetable engine, osc_N_engine 0). mode 'spectral' = pitch-tracked resynthesis of an audio file (best for pitched material), 'raw' = single-period slices, 'png' = image drawn as a spectrum (X=frame, Y=harmonic, brightness=amplitude). With live=true also pushed to the attached live instance.",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" },
                "slot": { "type": "integer", "minimum": 0, "maximum": 3 },
                "mode": { "type": "string", "enum": ["spectral", "raw", "png"], "default": "spectral" },
                "live": { "type": "boolean", "default": false }
            }, "required": ["path", "slot"] }
        },
        {
            "name": "load_sfz",
            "description": "Loads an SFZ multisample instrument into one oscillator slot's Multisample engine (osc_N_engine 3). Sample opcodes resolve relative to the SFZ file. With live=true also pushed to the attached live instance (live side reads WAV zone samples only).",
            "inputSchema": { "type": "object", "properties": {
                "path": { "type": "string" },
                "slot": { "type": "integer", "minimum": 0, "maximum": 3 },
                "live": { "type": "boolean", "default": false }
            }, "required": ["path", "slot"] }
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
            "name": "sequencer",
            "description": "Configures the note sequencer (arpeggiator / step sequencer) on the ATTACHED LIVE instance. LIVE-ONLY: it sits between incoming notes (live_note, live_sequence, MIDI keyboard) and the engine; the offline `play` render path is NOT affected. mode 'arp' arpeggiates held notes, mode 'step' plays the step pattern relative to the lowest held note, mode 'off' restores direct playthrough and releases everything. Hold notes with live_note action 'on' (use latch to keep them running hands-free).",
            "inputSchema": { "type": "object", "properties": {
                "mode": { "type": "string", "enum": ["off", "arp", "step"] },
                "pattern": { "type": "string", "enum": ["up", "down", "updown", "played", "random", "chord"], "description": "Arp note order; default up" },
                "rate": {
                    "description": "Tempo division string '1/1'..'1/32', optional 'd' (dotted) or 't' (triplet) suffix, e.g. '1/8d' — or a number for a free rate in Hz. Default 1/8.",
                    "oneOf": [{ "type": "string" }, { "type": "number" }]
                },
                "gate": { "type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.8, "description": "Note length as a fraction of a step" },
                "swing": { "type": "number", "minimum": 0.0, "maximum": 0.75, "default": 0.0, "description": "Delay of every 2nd step, as a fraction of a step" },
                "latch": { "type": "boolean", "default": false, "description": "Keep arpeggiating after keys are released; the next press starts a new held set" },
                "octaves": { "type": "integer", "minimum": 1, "maximum": 4, "default": 1, "description": "Arp octave range" },
                "steps": { "type": "array", "maxItems": 32, "description": "Step mode pattern; steps play relative to the lowest held note", "items": { "type": "object", "properties": {
                    "on": { "type": "boolean", "default": true, "description": "false = rest" },
                    "transpose": { "type": "integer", "default": 0, "description": "Semitones relative to the lowest held note" },
                    "velocity": { "type": "number", "default": 0.8 },
                    "tie": { "type": "boolean", "default": false, "description": "Hold into the next step (same note merges into one long note)" }
                }}},
                "length": { "type": "integer", "minimum": 1, "maximum": 32, "description": "Pattern length in steps; default = steps array length; missing steps are rests" },
                "seed": { "type": "integer", "description": "Deterministic seed for the random pattern" }
            }, "required": ["mode"] }
        },
        {
            "name": "patching_guide",
            "description": "Returns the Spinwave patching guide: signal flow, value scales (quartic envelope times, log2 frequencies...), modulation sources, sound design recipes, loudness workflow. READ THIS FIRST before designing a sound.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "list_racks",
            "description": "Lists the prebuilt effect racks (loudness chain, neuro crush, wide&wet space, dub delays, vintage warmth, club sub...). Apply one with apply_rack.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "apply_rack",
            "description": "Applies a prebuilt effect rack over the current patch (voice section untouched). Set push_live to also push to the attached live instance.",
            "inputSchema": { "type": "object", "properties": {
                "rack": { "type": "string", "description": "Rack name from list_racks, or a path to a rack JSON" },
                "push_live": { "type": "boolean", "default": false }
            }, "required": ["rack"] }
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
            "name": "measure_patch",
            "description": "Renders the CURRENT patch under a scenario and returns every descriptor with its unit (peak/RMS/LUFS, octave bands, centroid + trajectory, attack, decay to -20 dB, f0 by YIN, harmonicity, inharmonicity, width, mono compatibility, clipping, DC, aliasing proxy). Refuses a silent render while a source is on (code `silent`), a non-finite one, a patch that does not load.",
            "inputSchema": { "type": "object", "properties": {
                "lite": { "type": "boolean", "description": "Polyphony 1, no oversampling, one 0.6 s note: what searches spend (default false: 2.5 s, C3 held 1.5 s)" },
                "notes": { "type": "array", "items": { "type": "object", "properties": { "note": {"type":"integer"}, "start": {"type":"number"}, "duration": {"type":"number"}, "velocity": {"type":"number"} }, "required": ["note","start","duration"] } },
                "seconds": { "type": "number" },
                "bpm": { "type": "number" },
                "seed": { "type": "integer", "description": "Every random draw follows from it; echoed in the result" }
            } }
        },
        {
            "name": "aliasing_patch",
            "description": "The honest aliasing measure on the CURRENT patch: two renders a semitone apart, the power of the prominent peaks above 2 kHz that do not follow the key over all of them (FM sidebands and ring-mod products follow the key; fold-over does not). Validated: it falls to zero as the patch's oversampling rises. Also available as quality `aliasing` in explain_patch / suggest_moves.",
            "inputSchema": { "type": "object", "properties": {
                "lite": { "type": "boolean" },
                "seed": { "type": "integer" }
            } }
        },
        {
            "name": "compare_patches",
            "description": "The current patch against another patch file (.vital or .spinwave): the parameter diff spelled like the text format, and the perceptual distance (multi-resolution half-octave band spectrogram, dB) with per-band and per-time decompositions. `normalize_loudness` aligns levels first so the distance is timbre only.",
            "inputSchema": { "type": "object", "properties": {
                "other_path": { "type": "string" },
                "normalize_loudness": { "type": "boolean" },
                "lite": { "type": "boolean", "description": "Polyphony 1, no oversampling, one 0.6 s note: what searches spend (default false: 2.5 s, C3 held 1.5 s)" },
                "notes": { "type": "array", "items": { "type": "object", "properties": { "note": {"type":"integer"}, "start": {"type":"number"}, "duration": {"type":"number"}, "velocity": {"type":"number"} }, "required": ["note","start","duration"] } },
                "seconds": { "type": "number" },
                "bpm": { "type": "number" },
                "seed": { "type": "integer", "description": "Every random draw follows from it; echoed in the result" }
            }, "required": ["other_path"] }
        },
        {
            "name": "explain_patch",
            "description": "Which parameters of the CURRENT patch make it sound the way it does, for one quality (level, brightness, harshness, warmth, width, attack, sustain, noise, movement, band0..band7): each active parameter neutralised and re-rendered, ranked by measured effect. Defaults to the lite scenario.",
            "inputSchema": { "type": "object", "properties": {
                "quality": { "type": "string" },
                "max_renders": { "type": "integer" },
                "max_seconds": { "type": "number" },
                "lite": { "type": "boolean", "description": "Polyphony 1, no oversampling, one 0.6 s note: what searches spend (default false: 2.5 s, C3 held 1.5 s)" },
                "notes": { "type": "array", "items": { "type": "object", "properties": { "note": {"type":"integer"}, "start": {"type":"number"}, "duration": {"type":"number"}, "velocity": {"type":"number"} }, "required": ["note","start","duration"] } },
                "seconds": { "type": "number" },
                "bpm": { "type": "number" },
                "seed": { "type": "integer", "description": "Every random draw follows from it; echoed in the result" }
            }, "required": ["quality"] }
        },
        {
            "name": "suggest_moves",
            "description": "Which parameter moves push a quality the requested way, ranked by measured effect, with the collateral band distance of each. Continuous parameters only unless `switches` is true. Defaults to the lite scenario.",
            "inputSchema": { "type": "object", "properties": {
                "quality": { "type": "string" },
                "direction": { "type": "string", "enum": ["more", "less"] },
                "switches": { "type": "boolean" },
                "max_renders": { "type": "integer" },
                "max_seconds": { "type": "number" },
                "lite": { "type": "boolean", "description": "Polyphony 1, no oversampling, one 0.6 s note: what searches spend (default false: 2.5 s, C3 held 1.5 s)" },
                "notes": { "type": "array", "items": { "type": "object", "properties": { "note": {"type":"integer"}, "start": {"type":"number"}, "duration": {"type":"number"}, "velocity": {"type":"number"} }, "required": ["note","start","duration"] } },
                "seconds": { "type": "number" },
                "bpm": { "type": "number" },
                "seed": { "type": "integer", "description": "Every random draw follows from it; echoed in the result" }
            }, "required": ["quality", "direction"] }
        },
        {
            "name": "apply_diff",
            "description": "Applies a diff to the CURRENT patch under validation: `changes` ([{name, value}] with engine values or text spellings like \"800 Hz\") or `fragment` (.spinwave text with only the keys to change). Refuses what does not load; returns the changes made, the format's report, descriptors before and after, the distance, and whether the optional goal (quality + direction) was met. The patch is replaced only when `commit` is true (default).",
            "inputSchema": { "type": "object", "properties": {
                "changes": { "type": "array", "items": { "type": "object", "properties": { "name": {"type":"string"}, "value": {} }, "required": ["name","value"] } },
                "fragment": { "type": "string" },
                "goal_quality": { "type": "string" },
                "goal_direction": { "type": "string", "enum": ["more", "less"] },
                "commit": { "type": "boolean" },
                "lite": { "type": "boolean", "description": "Polyphony 1, no oversampling, one 0.6 s note: what searches spend (default false: 2.5 s, C3 held 1.5 s)" },
                "notes": { "type": "array", "items": { "type": "object", "properties": { "note": {"type":"integer"}, "start": {"type":"number"}, "duration": {"type":"number"}, "velocity": {"type":"number"} }, "required": ["note","start","duration"] } },
                "seconds": { "type": "number" },
                "bpm": { "type": "number" },
                "seed": { "type": "integer", "description": "Every random draw follows from it; echoed in the result" }
            } }
        },
        {
            "name": "explore_patch",
            "description": "Variations around the CURRENT patch: only active parameters move, each by an amplitude weighted by its measured sensitivity on this patch; indexed parameters and switches hold unless `switch_indexed` > 0. Each variant comes with its diff, its distance from the origin and its descriptors; with `out_dir` the variants are written as .spinwave files.",
            "inputSchema": { "type": "object", "properties": {
                "count": { "type": "integer" },
                "amplitude": { "type": "number", "description": "0..1, default 0.25" },
                "switch_indexed": { "type": "number", "description": "probability an indexed parameter switches, default 0" },
                "prior": { "type": "string", "enum": ["live", "measured", "measured_then_live"], "description": "where the weights come from: the knowledge store (`measured`), live renders (`live`), or the store first and live renders for what it lacks (default)" },
                "out_dir": { "type": "string" },
                "max_renders": { "type": "integer" },
                "max_seconds": { "type": "number" },
                "lite": { "type": "boolean", "description": "Polyphony 1, no oversampling, one 0.6 s note: what searches spend (default false: 2.5 s, C3 held 1.5 s)" },
                "notes": { "type": "array", "items": { "type": "object", "properties": { "note": {"type":"integer"}, "start": {"type":"number"}, "duration": {"type":"number"}, "velocity": {"type":"number"} }, "required": ["note","start","duration"] } },
                "seconds": { "type": "number" },
                "bpm": { "type": "number" },
                "seed": { "type": "integer", "description": "Every random draw follows from it; echoed in the result" }
            } }
        },
        {
            "name": "interpolate_patches",
            "description": "The patches on the line from the CURRENT patch (t = 0) to another patch file (t = 1): continuous values lerp, indexed values and switches take the far side from t >= 0.5, connections are the union with amounts lerped. Written as .spinwave files into `out_dir`.",
            "inputSchema": { "type": "object", "properties": {
                "other_path": { "type": "string" },
                "steps": { "type": "integer" },
                "out_dir": { "type": "string" }
            }, "required": ["other_path", "out_dir"] }
        },
        {
            "name": "live_stop",
            "description": "Stops the standalone synth this server spawned (an attached DAW instance is only detached, never killed).",
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

/// A refusal travels as its JSON (with the `code`), not as prose.
fn op_error(e: ops::OpError) -> String {
    serde_json::to_string(&e).unwrap_or_else(|_| e.to_string())
}

fn scenario_of(args: &Value, lite_default: bool) -> (ops::Scenario, u64) {
    let lite = args["lite"].as_bool().unwrap_or(lite_default);
    let mut scenario = if lite { ops::Scenario::lite() } else { ops::Scenario::faithful() };
    if let Ok(notes) = serde_json::from_value::<Vec<NoteSpec>>(args["notes"].clone()) {
        if !notes.is_empty() {
            scenario.notes = notes;
        }
    }
    if let Some(s) = args["seconds"].as_f64() {
        scenario.seconds = s as f32;
    }
    if let Some(b) = args["bpm"].as_f64() {
        scenario.bpm = b as f32;
    }
    (scenario, args["seed"].as_u64().unwrap_or(0))
}

fn budget_of(args: &Value) -> ops::Budget {
    let mut budget = ops::Budget::default();
    if let Some(n) = args["max_renders"].as_u64() {
        budget.max_renders = n as usize;
    }
    if let Some(s) = args["max_seconds"].as_f64() {
        budget.max_seconds = s as f32;
    }
    budget
}

fn quality_of(args: &Value) -> Result<ops::Quality, String> {
    let id = args["quality"].as_str().ok_or("quality required")?;
    ops::Quality::from_id(id).ok_or_else(|| format!("unknown quality `{id}`; one of: {}", ops::Quality::ALL.join(", ")))
}

fn call_tool(session: &mut Session, name: &str, args: &Value) -> Result<Value, String> {
    match name {
        "describe_params" => {
            let search = args["search"].as_str();
            let limit = args["limit"].as_u64().unwrap_or(40) as usize;
            let mut value = session.describe_params(search, limit.clamp(1, 200));
            // The engine this binary runs: a client compares it with
            // `spinwave-cli fingerprint` and warns when they differ (a
            // stale server cost a turn on 2026-09-13).
            if let Some(object) = value.as_object_mut() {
                object.insert("engine_fingerprint".into(), Value::String(spinwave_control::knowledge::ENGINE_FINGERPRINT.into()));
                object.insert("engine_git".into(), Value::String(spinwave_control::knowledge::GIT_COMMIT.into()));
            }
            Ok(value)
        }
        "get_patch" => session
            .preset
            .to_json_pretty()
            .map(Value::String)
            .map_err(|e| e.to_string()),
        "set_patch" => {
            let json_text = args["preset_json"].as_str().ok_or("preset_json required")?;
            let message = session.load_preset_json(json_text)?;
            Ok(json!({
                "summary": message,
                "report": serde_json::to_value(&session.last_report).unwrap_or_default(),
            }))
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
            let overwrite = args["overwrite"].as_bool().unwrap_or(false);
            session.render_to(&notes, seconds, bpm, out_path, overwrite).map(|(summary, analysis)| {
                json!({
                    "summary": summary,
                    "path": session.last_render_path,
                    "output_dir": session.output_dir.to_string_lossy(),
                    "analysis": analysis,
                })
            })
        }
        "analyze" => session.analyze_last().map(|a| serde_json::to_value(a).unwrap()),
        "analyze_file" => {
            let path = args["path"].as_str().ok_or("path required")?;
            let start = args["start"].as_f64().map(|v| v as f32);
            let duration = args["duration"].as_f64().map(|v| v as f32);
            Session::analyze_file(path, start, duration)
                .map(|a| serde_json::to_value(a).unwrap())
        }
        "listen" => {
            let path = args["path"].as_str().ok_or("path required")?;
            let start = args["start"].as_f64().map(|v| v as f32);
            let duration = Some(args["duration"].as_f64().unwrap_or(120.0).min(180.0) as f32);
            let step = args["step"].as_f64().unwrap_or(0.5) as f32;
            let (stereo, sample_rate) = spinwave_control::decode::decode_file(path, start, duration)?;
            let timeline = spinwave_control::listen::listen(&stereo, sample_rate, step);

            let mut text = String::new();
            if let Some(bpm) = timeline.bpm_estimate {
                if bpm < 100.0 {
                    // Halftime feel is ambiguous: report double-time too.
                    text.push_str(&format!("BPM estimate: {bpm:.0} (or {:.0} double-time)\n", bpm * 2.0));
                } else {
                    text.push_str(&format!("BPM estimate: {bpm:.0}\n"));
                }
            }
            text.push_str("\n== Structure ==\n");
            for line in &timeline.narrative {
                text.push_str(line);
                text.push('\n');
            }
            text.push_str("\n== Timeline (t | rms | sub/bass/mid/high dB | centroid | width | flat | onsets) ==\n");
            let stride = if args["full_frames"].as_bool().unwrap_or(false) {
                1
            } else {
                ((2.0 / timeline.step_seconds) as usize).max(1)
            };
            for frame in timeline.frames.iter().step_by(stride) {
                text.push_str(&format!(
                    "{:6.1}s | {:5.1} | {:4.0}/{:4.0}/{:4.0}/{:4.0} | {:5.0} | {:.2} | {:.2} | {}\n",
                    frame.t,
                    frame.rms_db,
                    frame.sub_db,
                    frame.bass_db,
                    frame.mid_db,
                    frame.high_db,
                    frame.centroid_hz,
                    frame.width,
                    frame.flatness,
                    frame.onsets
                ));
            }
            Ok(Value::String(text))
        }
        "compare" => {
            let path = args["reference_path"].as_str().ok_or("reference_path required")?;
            let start = args["start"].as_f64().map(|v| v as f32);
            let duration = args["duration"].as_f64().map(|v| v as f32);
            session.compare(path, start, duration)
        }
        "save_preset" => {
            let path = args["path"].as_str().ok_or("path required")?;
            if let Some(name) = args["name"].as_str() {
                session.preset.preset_name = name.to_string();
            }
            let text = session.preset.for_vital_file().to_json_pretty().map_err(|e| e.to_string())?;
            std::fs::write(path, text).map_err(|e| e.to_string())?;
            Ok(Value::String(format!("saved to {path}")))
        }
        "load_preset" => {
            let path = args["path"].as_str().ok_or("path required")?;
            let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            let message = session.load_preset_json(&text)?;
            Ok(json!({
                "summary": message,
                "report": serde_json::to_value(&session.last_report).unwrap_or_default(),
            }))
        }
        "load_sample" => {
            let path = args["path"].as_str().ok_or("path required")?;
            let slot = args["slot"].as_u64().ok_or("slot required")? as usize;
            let mut message = session.load_sample_offline(path, slot)?;
            if args["live"].as_bool().unwrap_or(false) {
                message.push_str("; ");
                message.push_str(&session.live_load_sample(slot, path)?);
            }
            Ok(Value::String(message))
        }
        "import_wavetable" => {
            let path = args["path"].as_str().ok_or("path required")?;
            let slot = args["slot"].as_u64().ok_or("slot required")? as usize;
            let mode = args["mode"].as_str().unwrap_or("spectral");
            let mut message = session.import_wavetable_offline(path, slot, mode)?;
            if args["live"].as_bool().unwrap_or(false) {
                message.push_str("; ");
                message.push_str(&session.live_import_wavetable(slot, path, mode)?);
            }
            Ok(Value::String(message))
        }
        "load_sfz" => {
            let path = args["path"].as_str().ok_or("path required")?;
            let slot = args["slot"].as_u64().ok_or("slot required")? as usize;
            let mut message = session.load_sfz_offline(path, slot)?;
            if args["live"].as_bool().unwrap_or(false) {
                message.push_str("; ");
                message.push_str(&session.live_load_sfz(slot, path)?);
            }
            Ok(Value::String(message))
        }
        "list_audio_devices" => spinwave_control::live_client::LiveLink::list_output_devices()
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
        "sequencer" => {
            if !args.is_object() {
                return Err("sequencer config object required".into());
            }
            session.live_seq(args.clone()).map(Value::String)
        }
        "patching_guide" => {
            let path = Session::racks_dir()
                .parent()
                .and_then(|p| p.parent())
                .map(|repo| repo.join("PATCHING.md"))
                .filter(|p| p.exists())
                .unwrap_or_else(|| "PATCHING.md".into());
            std::fs::read_to_string(&path)
                .map(|guide| {
                    Value::String(format!(
                        "<!-- engine fingerprint {} (git {}); compare with `spinwave-cli fingerprint` -->\n{guide}",
                        spinwave_control::knowledge::ENGINE_FINGERPRINT,
                        spinwave_control::knowledge::GIT_COMMIT
                    ))
                })
                .map_err(|e| format!("cannot read {}: {e}", path.display()))
        }
        "list_racks" => {
            let racks = Session::list_racks();
            if racks.is_empty() {
                Ok(Value::String(format!(
                    "no racks found in {}",
                    Session::racks_dir().display()
                )))
            } else {
                let lines: Vec<String> = racks
                    .iter()
                    .map(|(name, description, _)| format!("{name} — {description}"))
                    .collect();
                Ok(Value::String(lines.join("\n")))
            }
        }
        "apply_rack" => {
            let rack = args["rack"].as_str().ok_or("rack required")?;
            let mut message = session.apply_rack(rack)?;
            if args["push_live"].as_bool().unwrap_or(false) {
                message.push_str("; ");
                message.push_str(&session.live_push_preset()?);
            }
            Ok(Value::String(message))
        }
        "live_instances" => {
            let instances = spinwave_control::live_client::LiveLink::list_instances();
            let link = match session.live.target() {
                spinwave_control::live_client::Target::None => "this server is not attached".to_string(),
                target => format!("this server: {target:?} on port {}", session.live.port()),
            };
            if instances.is_empty() {
                Ok(Value::String(format!(
                    "no live Spinwave instance found (standalone not running, no plugin loaded); {link}"
                )))
            } else {
                let mut lines: Vec<String> = instances
                    .iter()
                    .map(|(pid, port, exe)| format!("port {port}: {exe} (pid {pid})"))
                    .collect();
                lines.push(link);
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
        // -- The operations: thin calls into `spinwave_control::ops` -----
        "measure_patch" => {
            let (scenario, seed) = scenario_of(args, false);
            let m = ops::measure(&session.preset, &scenario, seed).map_err(op_error)?;
            Ok(serde_json::to_value(m).unwrap_or_default())
        }
        "aliasing_patch" => {
            let (scenario, seed) = scenario_of(args, false);
            let r = ops::aliasing(&session.preset, &scenario, seed).map_err(op_error)?;
            Ok(serde_json::to_value(r).unwrap_or_default())
        }
        "compare_patches" => {
            let other = ops::load_patch(args["other_path"].as_str().ok_or("other_path required")?)?;
            let (scenario, seed) = scenario_of(args, false);
            let options = ops::DistanceOptions { normalize_loudness: args["normalize_loudness"].as_bool().unwrap_or(false) };
            let c = ops::compare(&session.preset, &other, &scenario, seed, options).map_err(op_error)?;
            Ok(serde_json::to_value(c).unwrap_or_default())
        }
        "explain_patch" => {
            let quality = quality_of(args)?;
            let (scenario, seed) = scenario_of(args, true);
            let e = ops::explain(&session.preset, &scenario, quality, seed, budget_of(args)).map_err(op_error)?;
            Ok(serde_json::to_value(e).unwrap_or_default())
        }
        "suggest_moves" => {
            let quality = quality_of(args)?;
            let direction = if args["direction"].as_str() == Some("less") { ops::Direction::Less } else { ops::Direction::More };
            let (scenario, seed) = scenario_of(args, true);
            let s = ops::suggest(&session.preset, &scenario, quality, direction, args["switches"].as_bool().unwrap_or(false), seed, budget_of(args)).map_err(op_error)?;
            Ok(serde_json::to_value(s).unwrap_or_default())
        }
        "apply_diff" => {
            let diff = if let Some(fragment) = args["fragment"].as_str() {
                ops::Diff::Fragment(fragment.to_string())
            } else {
                let changes: Vec<ops::Change> = serde_json::from_value(args["changes"].clone()).map_err(|e| format!("changes: {e}"))?;
                ops::Diff::Changes(changes)
            };
            let goal = match args["goal_quality"].as_str() {
                Some(q) => {
                    let quality = ops::Quality::from_id(q).ok_or_else(|| format!("unknown quality `{q}`"))?;
                    let direction = if args["goal_direction"].as_str() == Some("less") { ops::Direction::Less } else { ops::Direction::More };
                    Some((quality, direction))
                }
                None => None,
            };
            let (scenario, seed) = scenario_of(args, false);
            let applied = ops::apply(&session.preset, &diff, &scenario, goal, seed).map_err(op_error)?;
            if args["commit"].as_bool().unwrap_or(true) {
                let json = applied.preset.to_json().map_err(|e| e.to_string())?;
                session.load_preset_json(&json)?;
            }
            let mut value = serde_json::to_value(&applied).unwrap_or_default();
            value.as_object_mut().map(|o| o.remove("preset"));
            Ok(value)
        }
        "explore_patch" => {
            let (scenario, seed) = scenario_of(args, true);
            let spec = ops::ExploreSpec {
                count: args["count"].as_u64().unwrap_or(8) as usize,
                amplitude: args["amplitude"].as_f64().unwrap_or(0.25) as f32,
                seed,
                switch_indexed: args["switch_indexed"].as_f64().unwrap_or(0.0) as f32,
                budget: budget_of(args),
                prior: serde_json::from_value(args["prior"].clone()).unwrap_or_default(),
            };
            let e = ops::explore(&session.preset, &scenario, &spec).map_err(op_error)?;
            let out_dir = args["out_dir"].as_str();
            if let Some(dir) = out_dir {
                std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
            }
            let mut value = serde_json::to_value(&e).unwrap_or_default();
            for (i, v) in e.variants.iter().enumerate() {
                if let Some(dir) = out_dir {
                    let file = format!("{dir}/variant_{:03}.spinwave", v.index);
                    ops::save_patch(&v.preset, &file)?;
                    value["variants"][i]["path"] = Value::from(file);
                }
                value["variants"][i].as_object_mut().map(|o| o.remove("preset"));
            }
            Ok(value)
        }
        "interpolate_patches" => {
            let other = ops::load_patch(args["other_path"].as_str().ok_or("other_path required")?)?;
            let dir = args["out_dir"].as_str().ok_or("out_dir required")?;
            let steps = args["steps"].as_u64().unwrap_or(5).max(2) as usize;
            let ts: Vec<f32> = (0..steps).map(|i| i as f32 / (steps - 1) as f32).collect();
            let patches = ops::interpolate(&session.preset, &other, &ts).map_err(op_error)?;
            std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
            let mut files = Vec::new();
            for (t, p) in ts.iter().zip(&patches) {
                let file = format!("{dir}/t_{t:.3}.spinwave");
                ops::save_patch(p, &file)?;
                files.push(json!({ "t": t, "path": file }));
            }
            Ok(Value::Array(files))
        }
        "live_stop" => Ok(Value::String(session.live.stop())),
        "live_panic" => session
            .live
            .send(&json!({"cmd": "panic"}))
            .map(|_| Value::String("all sounds off".into())),
        other => Err(format!("unknown tool: {other}")),
    }
}
