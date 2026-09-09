//! Client for the live control channel: spawns the standalone synth or
//! attaches to a running instance (standalone or DAW-hosted), pushes
//! patches and plays notes while audio runs on the user's device.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::json;

const DEFAULT_PORT: u16 = 41929;
const PORT_RANGE_END: u16 = 41979;

/// What the link points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Nothing attached yet (commands go to the default port).
    None,
    /// A standalone this link spawned (`stop` kills it).
    Spawned,
    /// An instance found by discovery (`stop` only detaches).
    Attached,
}

pub struct LiveLink {
    child: Option<Child>,
    port: u16,
    target: Target,
}

impl Default for LiveLink {
    fn default() -> Self {
        LiveLink { child: None, port: DEFAULT_PORT, target: Target::None }
    }
}

impl LiveLink {
    /// Whether the standalone THIS link spawned is still alive.
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => child.try_wait().map(|status| status.is_none()).unwrap_or(false),
            None => false,
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn target(&self) -> Target {
        self.target
    }

    fn standalone_path() -> Result<std::path::PathBuf, String> {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
            .ok_or("cannot locate executable directory")?;
        let standalone = exe_dir.join("spinwave.exe");
        if !standalone.exists() {
            return Err(format!(
                "standalone not found at {} — build with `cargo build -p spinwave-plugin --release`",
                standalone.display()
            ));
        }
        Ok(standalone)
    }

    fn log_path() -> std::path::PathBuf {
        std::env::temp_dir().join("spinwave-live.log")
    }

    /// Lists the audio output devices the standalone can use.
    pub fn list_output_devices() -> Result<Vec<String>, String> {
        let standalone = Self::standalone_path()?;
        let output = Command::new(&standalone)
            .args(["-b", "wasapi", "--output-device", "?"])
            .env("SPINWAVE_LIVE", "0")
            .output()
            .map_err(|e| format!("cannot run standalone: {e}"))?;
        let text = String::from_utf8_lossy(&output.stderr);
        let devices: Vec<String> = text
            .lines()
            .skip_while(|line| !line.contains("Available devices are:"))
            .skip(1)
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect();
        if devices.is_empty() {
            Err(format!("could not list devices; standalone said: {text}"))
        } else {
            Ok(devices)
        }
    }

    /// A port nobody listens on, so the spawned standalone's ping can only
    /// be answered by the standalone itself (never by a DAW instance that
    /// happened to hold the default port).
    fn free_port() -> Result<u16, String> {
        for port in DEFAULT_PORT..=PORT_RANGE_END {
            if TcpListener::bind(("127.0.0.1", port)).is_ok() {
                return Ok(port);
            }
        }
        Err(format!("no free live port in {DEFAULT_PORT}..={PORT_RANGE_END}"))
    }

    /// Spawns the standalone (next to this executable) with the live
    /// listener enabled on a port verified free, waits until it answers a
    /// ping, and reports which backend/device it actually opened (from its
    /// captured log).
    pub fn start(
        &mut self,
        output_device: Option<&str>,
        sample_rate: Option<u32>,
        period_size: Option<u32>,
    ) -> Result<String, String> {
        if self.is_running() {
            return Ok(format!("already running on port {}", self.port));
        }

        let standalone = Self::standalone_path()?;
        let port = Self::free_port()?;
        let log_path = Self::log_path();
        let log_file = std::fs::File::create(&log_path)
            .map_err(|e| format!("cannot create live log: {e}"))?;

        let mut command = Command::new(&standalone);
        command
            .env("SPINWAVE_LIVE_PORT", port.to_string())
            .args(["-b", "wasapi"])
            .stdout(Stdio::null())
            .stderr(Stdio::from(log_file));
        if let Some(device) = output_device {
            command.args(["--output-device", device]);
        }
        if let Some(rate) = sample_rate {
            command.args(["-r", &rate.to_string()]);
        }
        if let Some(period) = period_size {
            command.args(["-p", &period.to_string()]);
        }
        let child = command
            .spawn()
            .map_err(|e| format!("cannot spawn standalone: {e}"))?;
        self.child = Some(child);
        self.port = port;
        self.target = Target::Spawned;

        // The audio backend takes a moment; retry the ping.
        for _ in 0..40 {
            std::thread::sleep(Duration::from_millis(250));
            if let Ok(first_ping) = self.send(&json!({"cmd": "ping"})) {
                let log = std::fs::read_to_string(&log_path).unwrap_or_default();
                if log.contains("dummy backend") {
                    self.stop();
                    return Err(format!(
                        "audio backend fell back to DUMMY (no sound!). Log: {log}"
                    ));
                }

                // Proof of life: the rendered-block counter must advance.
                let first_blocks = parse_blocks(&first_ping);
                std::thread::sleep(Duration::from_millis(600));
                let second_blocks =
                    self.send(&json!({"cmd": "ping"})).ok().and_then(|r| parse_blocks(&r));
                let audio_alive = match (first_blocks, second_blocks) {
                    (Some(a), Some(b)) => b > a,
                    _ => false,
                };
                if !audio_alive {
                    self.stop();
                    return Err(format!(
                        "the audio thread is NOT rendering (block counter stalled at {:?}) — \
                         wrong device or dead stream. Log tail: {}",
                        first_blocks,
                        log_tail(&log_path)
                    ));
                }

                let info_lines: Vec<&str> = log
                    .lines()
                    .filter(|l| l.contains("[INFO]") || l.contains("[ERROR]"))
                    .collect();
                return Ok(format!(
                    "standalone running, audio thread rendering (blocks {} -> {}), \
                     live control on 127.0.0.1:{}. Log: {}",
                    first_blocks.unwrap_or(0),
                    second_blocks.unwrap_or(0),
                    self.port,
                    info_lines.join(" | ")
                ));
            }
            if !self.is_running() {
                self.child = None;
                self.target = Target::None;
                return Err(format!(
                    "standalone exited during startup. Log tail: {}",
                    log_tail(&log_path)
                ));
            }
        }
        self.stop();
        Err("standalone did not answer the live ping within 10s".into())
    }

    /// Stops the standalone this link spawned; an attached instance (e.g.
    /// the user's DAW) is only detached, never silenced or killed.
    pub fn stop(&mut self) -> String {
        match self.child.take() {
            Some(mut child) => {
                // Only our own child gets the panic + kill.
                let _ = self.send(&json!({"cmd": "panic"}));
                let _ = child.kill();
                let _ = child.wait();
                self.target = Target::None;
                self.port = DEFAULT_PORT;
                "standalone stopped".into()
            }
            None => {
                if self.target == Target::Attached {
                    self.target = Target::None;
                    let port = self.port;
                    self.port = DEFAULT_PORT;
                    format!("detached from the instance on port {port} (left running)")
                } else {
                    "standalone was not running".into()
                }
            }
        }
    }

    pub fn send(&mut self, message: &serde_json::Value) -> Result<String, String> {
        let stream = TcpStream::connect(("127.0.0.1", self.port))
            .map_err(|e| format!("live connection failed on port {}: {e}", self.port))?;
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .map_err(|e| e.to_string())?;
        let mut writer = stream.try_clone().map_err(|e| e.to_string())?;
        writeln!(writer, "{message}").map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;

        let mut reply = String::new();
        BufReader::new(stream)
            .read_line(&mut reply)
            .map_err(|e| e.to_string())?;
        let reply = reply.trim().to_string();
        if reply.starts_with("err") {
            Err(reply)
        } else {
            Ok(reply)
        }
    }

    /// Reads the discovery registry and pings each instance; returns the
    /// alive ones as (pid, port, exe) — DAW-hosted plugins included.
    pub fn list_instances() -> Vec<(u64, u16, String)> {
        spinwave_plugin::live::alive_instances()
            .into_iter()
            .map(|entry| (entry.pid, entry.port, entry.exe))
            .collect()
    }

    /// Points this link at a specific instance's port (no process spawn).
    pub fn attach(&mut self, port: u16) -> Result<String, String> {
        let previous = self.port;
        let previous_target = self.target;
        self.port = port;
        match self.send(&json!({"cmd": "ping"})) {
            Ok(reply) => {
                if self.child.is_none() {
                    self.target = Target::Attached;
                }
                Ok(format!("attached to 127.0.0.1:{port} ({reply})"))
            }
            Err(e) => {
                self.port = previous;
                self.target = previous_target;
                Err(format!("no live instance on port {port}: {e}"))
            }
        }
    }

    /// Reads the current patch JSON back from the attached instance.
    pub fn get_patch(&mut self) -> Result<String, String> {
        let reply = self.send(&json!({"cmd": "get_patch"}))?;
        Ok(reply.strip_prefix("ok ").unwrap_or(&reply).to_string())
    }

    pub fn note_on(&mut self, note: i32, velocity: f32, channel: usize) -> Result<String, String> {
        self.send(&json!({"cmd": "note_on", "note": note, "velocity": velocity, "channel": channel}))
    }

    pub fn note_off(&mut self, note: i32, channel: usize) -> Result<String, String> {
        self.send(&json!({"cmd": "note_off", "note": note, "channel": channel}))
    }
}

fn parse_blocks(ping_reply: &str) -> Option<u64> {
    ping_reply.split("blocks=").nth(1)?.trim().parse().ok()
}

fn log_tail(log_path: &std::path::Path) -> String {
    let log = std::fs::read_to_string(log_path).unwrap_or_default();
    log.lines()
        .filter(|l| l.contains("[INFO]") || l.contains("[ERROR]"))
        .rev()
        .take(5)
        .collect::<Vec<_>>()
        .join(" | ")
}

impl Drop for LiveLink {
    fn drop(&mut self) {
        if self.child.is_some() {
            self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_never_kills_an_attached_instance() {
        // A fake instance answering pings on an ephemeral port.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let received = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let log = received.clone();
        let server = std::thread::spawn(move || {
            for stream in listener.incoming().take(2) {
                let Ok(stream) = stream else { continue };
                let mut writer = stream.try_clone().unwrap();
                let mut line = String::new();
                if BufReader::new(stream).read_line(&mut line).is_ok() {
                    log.lock().unwrap().push(line.trim().to_string());
                    let _ = writeln!(writer, "ok blocks=1");
                }
            }
        });

        let mut link = LiveLink::default();
        assert!(link.attach(port).unwrap().contains("attached"));
        assert_eq!(link.target(), Target::Attached);
        assert!(!link.is_running());
        let message = link.stop();
        assert!(message.contains("detached"), "{message}");
        assert_eq!(link.target(), Target::None);
        assert_eq!(link.port(), DEFAULT_PORT);
        // Second connection so the server loop can finish.
        let mut probe = LiveLink::default();
        let _ = probe.attach(port);
        server.join().unwrap();
        let lines = received.lock().unwrap();
        assert!(lines.iter().all(|l| l.contains("ping")), "no panic sent: {lines:?}");
    }

    #[test]
    fn attach_failure_keeps_the_previous_target() {
        let free = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let mut link = LiveLink::default();
        assert!(link.attach(free).is_err());
        assert_eq!(link.port(), DEFAULT_PORT);
        assert_eq!(link.target(), Target::None);
    }

    #[test]
    fn free_port_is_not_in_use() {
        let port = LiveLink::free_port().unwrap();
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
    }
}
