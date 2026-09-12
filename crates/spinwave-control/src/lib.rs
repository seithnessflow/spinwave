//! Spinwave control: a synth session an agent (or a command line) can
//! drive, plus the listening tools that judge what came out.
//!
//! The MCP server (`spinwave-mcp`) and the command-line tool
//! (`spinwave-cli`) are both thin shells over [`session::Session`]: load or
//! edit a patch, render notes to a WAV, then read the render back through
//! [`analysis`] (levels, spectrum, envelope, pitch) and [`listen`]
//! (temporal structure). Sound design happens by iterating on that loop.

pub mod analysis;
pub mod bounds;
pub mod decode;
pub mod fuzz;
pub mod golden;
pub mod judge;
pub mod listen;
pub mod live_client;
pub mod sensitivity;
pub mod session;
pub mod text_preset;
pub mod ops;
