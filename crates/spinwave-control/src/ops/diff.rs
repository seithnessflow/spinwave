//! Parameter diffs: what changed between two patches, spelled the way the
//! `.spinwave` text spells it; and the [`Diff`] an operation applies —
//! a change list or a `.spinwave` fragment, read by the format's own
//! parser so there is one spelling of a parameter and one report.

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use spinwave_params::preset::ModulationConnection;
use spinwave_params::{parameters, ParamDetails, Preset};

use super::OpError;
use crate::text_preset::{self, place, read_value_text, spell_value, Blobs, TextValue};

/// One parameter that differs, in engine units and in the text's spelling.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ParamChange {
    pub name: String,
    /// The `[module]` and key the text writes it under.
    pub module: String,
    pub key: String,
    pub from: f32,
    pub to: f32,
    /// The two values as the text spells them (`"800 Hz"`, `"-6.0 dB"`,
    /// `"ladder"`), from the same code that writes the file.
    pub from_text: String,
    pub to_text: String,
    /// `to − from` in engine units; for an indexed parameter, the step.
    pub delta: f32,
    pub indexed: bool,
}

/// A connection present in one patch only, or with a different amount.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConnectionChange {
    pub source: String,
    pub destination: String,
    /// `None` = absent in that patch.
    pub from_amount: Option<f32>,
    pub to_amount: Option<f32>,
}

fn engine_value(preset: &Preset, details: &ParamDetails) -> f32 {
    preset.settings.values.get(&details.name).and_then(Json::as_f64).map(|v| v as f32).unwrap_or(details.default_value)
}

fn text_of(value: TextValue) -> String {
    match value {
        TextValue::Bool(b) => b.to_string(),
        TextValue::Integer(i) => i.to_string(),
        TextValue::Number(n) => n,
        TextValue::Str(s) => s,
    }
}

/// Spells an engine value the way the text file would.
pub fn spell(name: &str, engine: f32) -> Option<String> {
    let details = parameters().lookup(name)?;
    let p = place(name);
    Some(text_of(spell_value(details, &p.module, &p.key, engine).0))
}

/// Every table parameter that differs (absent counts as the default) and
/// every connection present in one patch only or with a different amount.
pub fn param_diff(a: &Preset, b: &Preset) -> (Vec<ParamChange>, Vec<ConnectionChange>) {
    let table = parameters();
    let mut params = Vec::new();
    for details in table.iter() {
        if details.name.starts_with("modulation_") {
            continue; // reported through the connections
        }
        let (x, y) = (engine_value(a, details), engine_value(b, details));
        if x == y {
            continue;
        }
        let p = place(&details.name);
        params.push(ParamChange {
            name: details.name.clone(),
            module: p.module.clone(),
            key: p.key.clone(),
            from: x,
            to: y,
            from_text: text_of(spell_value(details, &p.module, &p.key, x).0),
            to_text: text_of(spell_value(details, &p.module, &p.key, y).0),
            delta: y - x,
            indexed: details.scale == spinwave_params::ParamScale::Indexed,
        });
    }
    let mut connections = Vec::new();
    let (ca, cb) = (connections_of(a), connections_of(b));
    for (route, amount) in &ca {
        match cb.iter().find(|(r, _)| r == route) {
            Some((_, other)) if other == amount => {}
            Some((_, other)) => connections.push(ConnectionChange {
                source: route.0.clone(),
                destination: route.1.clone(),
                from_amount: Some(*amount),
                to_amount: Some(*other),
            }),
            None => connections.push(ConnectionChange {
                source: route.0.clone(),
                destination: route.1.clone(),
                from_amount: Some(*amount),
                to_amount: None,
            }),
        }
    }
    for (route, amount) in &cb {
        if !ca.iter().any(|(r, _)| r == route) {
            connections.push(ConnectionChange {
                source: route.0.clone(),
                destination: route.1.clone(),
                from_amount: None,
                to_amount: Some(*amount),
            });
        }
    }
    (params, connections)
}

/// `((source, destination), amount)` for every connected slot.
pub(crate) fn connections_of(preset: &Preset) -> Vec<((String, String), f32)> {
    preset
        .settings
        .modulations
        .iter()
        .enumerate()
        .filter(|(_, m)| m.is_connected())
        .map(|(i, m)| {
            let amount = preset
                .settings
                .values
                .get(&format!("modulation_{}_amount", i + 1))
                .and_then(Json::as_f64)
                .map(|v| v as f32)
                .unwrap_or(1.0);
            ((m.source.clone(), m.destination.clone()), amount)
        })
        .collect()
}

/// One requested change.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Change {
    pub name: String,
    /// An engine value, or the text spelling (`"800 Hz"`), read by the
    /// format's unit parser.
    pub value: ChangeValue,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ChangeValue {
    Engine(f32),
    Text(String),
}

/// What [`super::apply`] takes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Diff {
    Changes(Vec<Change>),
    /// `.spinwave` text with only the keys to change (`format = 1` may be
    /// omitted). Connections given as `[[module.mod]]` replace the
    /// connection with the same source and destination, or add one.
    Fragment(String),
}

/// A diff resolved against the table: engine values per name, and the
/// connections to merge (from a fragment).
pub struct Resolved {
    pub values: Vec<(String, f32)>,
    pub connections: Vec<(ModulationConnection, Vec<(String, Json)>)>,
    /// What the format's reader corrected or normalised, verbatim.
    pub report: spinwave_params::LoadReport,
}

/// Resolves a diff without applying it.
pub fn resolve(diff: &Diff) -> Result<Resolved, OpError> {
    let table = parameters();
    match diff {
        Diff::Changes(changes) => {
            let mut values = Vec::new();
            for change in changes {
                let Some(details) = table.lookup(&change.name) else {
                    return Err(OpError::UnknownParameter { name: change.name.clone() });
                };
                let engine = match &change.value {
                    ChangeValue::Engine(v) => {
                        if !(details.min..=details.max).contains(v) {
                            return Err(OpError::BadValue {
                                name: change.name.clone(),
                                message: format!("{v} outside {}..={}", details.min, details.max),
                            });
                        }
                        *v
                    }
                    ChangeValue::Text(text) => {
                        let p = place(&change.name);
                        read_value_text(details, &p.module, &p.key, text)
                            .map(|r| r.engine)
                            .map_err(|e| OpError::BadValue { name: change.name.clone(), message: format!("{e:?}") })?
                    }
                };
                values.push((change.name.clone(), engine));
            }
            Ok(Resolved { values, connections: Vec::new(), report: Default::default() })
        }
        Diff::Fragment(text) => {
            let text = if text.contains("format") { text.clone() } else { format!("format = {}\n{text}", text_preset::FORMAT_VERSION) };
            let result = text_preset::read(&text, &Blobs::default());
            let Some(preset) = result.preset else {
                let messages = result.report.errors.iter().map(|e| format!("line {} {}: {}", e.line, e.key, e.message)).collect();
                return Err(OpError::Rejected { messages });
            };
            let values: Vec<(String, f32)> = preset
                .settings
                .values
                .iter()
                .filter(|(name, _)| !name.starts_with("modulation_") && *name != "format")
                .filter_map(|(name, v)| v.as_f64().map(|v| (name.clone(), v as f32)))
                .collect();
            let connections = preset
                .settings
                .modulations
                .iter()
                .enumerate()
                .filter(|(_, m)| m.is_connected())
                .map(|(i, m)| {
                    let prefix = format!("modulation_{}_", i + 1);
                    let fields = preset
                        .settings
                        .values
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix))
                        .map(|(k, v)| (k[prefix.len()..].to_string(), v.clone()))
                        .collect();
                    (m.clone(), fields)
                })
                .collect();
            Ok(Resolved { values, connections, report: result.report })
        }
    }
}

/// Applies a resolved diff to a copy of `preset`.
pub fn apply_resolved(preset: &Preset, resolved: &Resolved) -> Preset {
    let mut out = preset.clone();
    for (name, value) in &resolved.values {
        out.settings.values.insert(name.clone(), Json::from(*value as f64));
    }
    for (connection, fields) in &resolved.connections {
        let slot = out
            .settings
            .modulations
            .iter()
            .position(|m| m.source == connection.source && m.destination == connection.destination)
            .or_else(|| out.settings.modulations.iter().position(|m| !m.is_connected()))
            .unwrap_or_else(|| {
                out.settings.modulations.push(ModulationConnection::default());
                out.settings.modulations.len() - 1
            });
        out.settings.modulations[slot] = connection.clone();
        for (field, value) in fields {
            out.settings.values.insert(format!("modulation_{}_{field}", slot + 1), value.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::saw_patch;

    #[test]
    fn a_diff_spells_values_like_the_file_and_finds_connections() {
        let a = saw_patch();
        let mut b = a.clone();
        b.settings.values.insert("filter_1_cutoff".into(), 60.0.into());
        b.settings.values.insert("filter_1_model".into(), 2.0.into());
        b.settings.modulations.push(ModulationConnection { source: "lfo_1".into(), destination: "filter_1_cutoff".into(), ..Default::default() });
        b.settings.values.insert("modulation_1_amount".into(), 0.5.into());
        let (params, connections) = param_diff(&a, &b);
        let cutoff = params.iter().find(|c| c.name == "filter_1_cutoff").expect("cutoff changed");
        assert_eq!((cutoff.from, cutoff.to), (80.0, 60.0));
        // The text spells a cutoff in its shortest exact form: st or Hz.
        assert!(cutoff.from_text.ends_with("st") || cutoff.from_text.ends_with("Hz"), "{cutoff:?}");
        let model = params.iter().find(|c| c.name == "filter_1_model").unwrap();
        assert!(model.indexed);
        assert_eq!(model.to_text, "ladder");
        assert_eq!(connections.len(), 1);
        assert_eq!(connections[0].to_amount, Some(0.5));
        assert_eq!(connections[0].from_amount, None);
    }

    #[test]
    fn a_fragment_is_read_by_the_format_and_merged_by_route() {
        let a = saw_patch();
        let fragment = "[filter_1]\ncutoff = \"200 Hz\"\n[[filter_1.mod]]\nfrom = \"env_2\"\nto = \"cutoff\"\namount = \"40%\"\n";
        let resolved = resolve(&Diff::Fragment(fragment.into())).expect("reads");
        assert_eq!(resolved.values.len(), 1);
        assert_eq!(resolved.values[0].0, "filter_1_cutoff");
        let applied = apply_resolved(&a, &resolved);
        let (params, connections) = param_diff(&a, &applied);
        assert_eq!(params.len(), 1);
        assert_eq!(connections.len(), 1, "{connections:?}");
        assert_eq!(connections[0].source, "env_2");
        // Applying the same fragment again changes nothing: merged by route.
        let again = apply_resolved(&applied, &resolved);
        assert!(param_diff(&applied, &again).1.is_empty());
    }

    #[test]
    fn a_change_list_refuses_unknown_names_and_out_of_range_values() {
        let bad = Diff::Changes(vec![Change { name: "filter_1_cutof".into(), value: ChangeValue::Engine(60.0) }]);
        assert!(matches!(resolve(&bad), Err(OpError::UnknownParameter { .. })));
        let out = Diff::Changes(vec![Change { name: "filter_1_cutoff".into(), value: ChangeValue::Engine(999.0) }]);
        assert!(matches!(resolve(&out), Err(OpError::BadValue { .. })));
        let text = Diff::Changes(vec![Change { name: "filter_1_cutoff".into(), value: ChangeValue::Text("440 Hz".into()) }]);
        let r = resolve(&text).unwrap();
        assert!((r.values[0].1 - 69.0).abs() < 1e-3, "440 Hz is MIDI 69: {}", r.values[0].1);
    }
}
