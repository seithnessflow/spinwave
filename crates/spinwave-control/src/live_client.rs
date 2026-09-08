//! Client for the standalone's live control channel: spawns the synth,
//! pushes patches and plays notes while audio runs on the user's device.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde_json::json;

const DEFAULT_PORT: u16 = 41929;

pub struct LiveLink {
    child: Option<Child>,
    port: u16,
}

impl Default for LiveLink {
    fn default() -> Self {
        LiveLink { child: None, port: DEFAULT_PORT }
    }
}

impl LiveLink {
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => child.try_wait().map(|status| status.is_none()).unwrap_or(false),
            None => false,
        }
    }

    /// Spawns the standalone (next to this executable) with the live
    /// listener enabled, and waits until it answers a ping.
    pub fn start(&mut self) -> Result<String, String> {
        if self.is_running() {
            return Ok(format!("already running on port {}", self.port));
        }

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

        let child = Command::new(&standalone)
            .env("SPINWAVE_LIVE_PORT", self.port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot spawn standalone: {e}"))?;
        self.child = Some(child);

        // The audio backend takes a moment; retry the ping.
        for _ in 0..40 {
            std::thread::sleep(Duration::from_millis(250));
            if self.send(&json!({"cmd": "ping"})).is_ok() {
                return Ok(format!(
                    "standalone running with audio output, live control on 127.0.0.1:{}",
                    self.port
                ));
            }
            if !self.is_running() {
                return Err("standalone exited during startup (no audio device?)".into());
            }
        }
        Err("standalone did not answer the live ping within 10s".into())
    }

    pub fn stop(&mut self) -> String {
        let _ = self.send(&json!({"cmd": "panic"}));
        match self.child.take() {
            Some(mut child) => {
                let _ = child.kill();
                let _ = child.wait();
                "standalone stopped".into()
            }
            None => "standalone was not running".into(),
        }
    }

    pub fn send(&mut self, message: &serde_json::Value) -> Result<String, String> {
        let stream = TcpStream::connect(("127.0.0.1", self.port))
            .map_err(|e| format!("live connection failed: {e}"))?;
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

    pub fn note_on(&mut self, note: i32, velocity: f32, channel: usize) -> Result<String, String> {
        self.send(&json!({"cmd": "note_on", "note": note, "velocity": velocity, "channel": channel}))
    }

    pub fn note_off(&mut self, note: i32, channel: usize) -> Result<String, String> {
        self.send(&json!({"cmd": "note_off", "note": note, "channel": channel}))
    }
}

impl Drop for LiveLink {
    fn drop(&mut self) {
        if self.child.is_some() {
            self.stop();
        }
    }
}
