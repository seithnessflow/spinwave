//! The `.spinwave` text preset: a readable, editable, diffable view of a
//! `.vital` preset, in strict bijection with it.
//!
//! The design and its six decisions are in `notes/preset-text-format.md`.
//! In short: TOML; only what departs from the default; values in their
//! real unit, exact or written `raw:`; modules as tables with each
//! modulation under its destination; drawn shapes inline, bulky material
//! in a content-addressed sidecar; a derived-and-checked `requires` for
//! the Spinwave-only namespace; `format = 1` first.
//!
//! [`write`] and [`read`] are the two halves. [`Blobs`] is the sidecar.

mod layout;
mod read;
#[cfg(test)]
mod roundtrip;
mod units;
mod write;

use std::collections::BTreeMap;

use serde_json::Value;
use sha2::{Digest, Sha256};
use spinwave_params::preset::LineShape;
use spinwave_params::{LoadReport, Preset};

pub use read::read;
pub use write::write;

/// The format version this code writes and the highest it reads.
pub const FORMAT_VERSION: i64 = 1;

/// The sidecar: material too bulky for the text, keyed by the SHA-256 of
/// its verbatim JSON. Written to `name.spinwave.d/<hash>.json`, one file
/// per blob, so a changed wavetable is one new file and one changed line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Blobs {
    entries: BTreeMap<String, String>,
}

impl Blobs {
    /// Stores a JSON value verbatim and returns its reference for the text.
    pub fn put(&mut self, value: &Value) -> String {
        let json = serde_json::to_string(value).expect("a JSON value serialises");
        let hash = format!("{:x}", Sha256::digest(json.as_bytes()));
        self.entries.insert(hash.clone(), json);
        format!("blob:sha256:{hash}")
    }

    /// The JSON behind a reference, if this store has it.
    pub fn get(&self, reference: &str) -> Option<Value> {
        let hash = reference.strip_prefix("blob:sha256:")?;
        serde_json::from_str(self.entries.get(hash)?).ok()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `(hash, json)` pairs, for writing the sidecar files.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Reads a sidecar directory back. Files that are not `<hex>.json` are
    /// ignored; a file whose content does not hash to its name is refused,
    /// because the reference in the text would then point at something
    /// else than what was written.
    pub fn from_dir(dir: &std::path::Path) -> Result<Blobs, String> {
        let mut blobs = Blobs::default();
        let Ok(entries) = std::fs::read_dir(dir) else { return Ok(blobs) };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let json = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let hash = format!("{:x}", Sha256::digest(json.as_bytes()));
            if hash != stem {
                return Err(format!("{}: content hashes to {hash}, not to its name", path.display()));
            }
            blobs.entries.insert(hash, json);
        }
        Ok(blobs)
    }

    /// Writes the sidecar directory (created if needed).
    pub fn write_dir(&self, dir: &std::path::Path) -> Result<(), String> {
        if self.is_empty() {
            return Ok(());
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (hash, json) in self.iter() {
            let path = dir.join(format!("{hash}.json"));
            std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        Ok(())
    }
}

/// What [`write`] produces: the text, and the sidecar it references.
#[derive(Clone, Debug, PartialEq)]
pub struct Written {
    pub text: String,
    pub blobs: Blobs,
}

/// What [`read`] produces: the preset, and the report of everything the
/// parser normalised or refused. `preset` is `None` when any error was
/// recorded — a file with a refused value does not load half-way.
#[derive(Clone, Debug)]
pub struct ReadResult {
    pub preset: Option<Preset>,
    pub report: LoadReport,
}

// ------------------------------------------------------------ factory shapes

type ShapeMaker = fn() -> spinwave_dsp::modulators::LineGenerator;

/// The LFO shapes the engine can regenerate from a name. Written by name
/// when a drawn shape equals one exactly, points and powers alike.
const FACTORY_SHAPES: [(&str, ShapeMaker); 6] = [
    ("linear", spinwave_dsp::modulators::LineGenerator::linear),
    ("triangle", spinwave_dsp::modulators::LineGenerator::triangle),
    ("square", spinwave_dsp::modulators::LineGenerator::square),
    ("sin", spinwave_dsp::modulators::LineGenerator::sin),
    ("saw_up", spinwave_dsp::modulators::LineGenerator::saw_up),
    ("saw_down", spinwave_dsp::modulators::LineGenerator::saw_down),
];

fn factory_shape(name: &str) -> Option<LineShape> {
    let (_, make) = FACTORY_SHAPES.iter().find(|(n, _)| *n == name)?;
    let generator = make();
    let n = generator.num_points();
    let mut points = Vec::with_capacity(2 * n);
    let mut powers = Vec::with_capacity(n);
    for i in 0..n {
        let (x, y) = generator.point(i);
        points.push(x);
        points.push(y);
        powers.push(generator.power(i));
    }
    Some(LineShape {
        num_points: n as u32,
        points,
        powers,
        name: Some(factory_display_name(name).to_string()),
        smooth: generator.smooth(),
        ..Default::default()
    })
}

/// The name Vital writes into the file for a factory shape.
fn factory_display_name(name: &str) -> &'static str {
    match name {
        "linear" => "Linear",
        "triangle" => "Triangle",
        "square" => "Square",
        "sin" => "Sin",
        "saw_up" => "Saw Up",
        "saw_down" => "Saw Down",
        _ => "",
    }
}

/// The factory name of a shape, if it is one exactly — same points, same
/// powers, same smoothing, same name, nothing extra. Anything looser would
/// not round-trip.
fn factory_name_of(shape: &LineShape) -> Option<&'static str> {
    if !shape.extra.is_empty() {
        return None;
    }
    FACTORY_SHAPES.iter().map(|(n, _)| *n).find(|name| factory_shape(name).as_ref() == Some(shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_factory_shape_is_recognised_from_its_own_points() {
        for (name, _) in FACTORY_SHAPES {
            let shape = factory_shape(name).unwrap();
            assert_eq!(factory_name_of(&shape), Some(name));
        }
    }

    #[test]
    fn a_renamed_or_smoothed_triangle_is_not_the_factory_one() {
        let mut shape = factory_shape("triangle").unwrap();
        shape.name = Some("Custom".into());
        assert_eq!(factory_name_of(&shape), None);
        let mut shape = factory_shape("triangle").unwrap();
        shape.smooth = true;
        assert_eq!(factory_name_of(&shape), None);
    }

    #[test]
    fn blobs_are_content_addressed_and_verbatim() {
        let mut blobs = Blobs::default();
        let value = serde_json::json!({"name": "table", "data": [1, 2, 3]});
        let reference = blobs.put(&value);
        assert!(reference.starts_with("blob:sha256:"));
        assert_eq!(blobs.get(&reference), Some(value.clone()));
        let again = blobs.put(&value);
        assert_eq!(again, reference);
        assert_eq!(blobs.iter().count(), 1);
    }
}
