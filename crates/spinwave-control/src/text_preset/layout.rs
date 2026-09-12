//! Where a parameter lives in the text: its module (a TOML table) and its
//! key inside it. Derived from the name, never declared twice.
//!
//! The table registers parameters in groups (`osc_N_*`, `env_N_*`, ...) and
//! as flat names with a family prefix (`chorus_*`, `bus_a_*`, ...). The
//! text mirrors that: `osc_1_level` is `[osc_1] level`, `chorus_dry_wet` is
//! `[chorus] dry_wet`, and the handful of names with no family go under
//! `[global]`. The `modulation_N_*` slots never appear as keys: they are
//! spelled out as connections under their destination.

/// The module a parameter is written under, and its key there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    pub module: String,
    pub key: String,
}

/// Indexed groups, in the order their modules are written.
const INDEXED_FAMILIES: [&str; 6] = ["osc", "filter", "env", "lfo", "random", "macro_control"];

/// Flat families, in the order their modules are written. Effects follow
/// the reference's chain order; the Spinwave extensions come after.
pub const FLAT_FAMILIES: [&str; 22] = [
    "sample",
    "noise",
    "filter_fx",
    "portamento",
    "chorus",
    "compressor",
    "delay",
    "distortion",
    "eq",
    "flanger",
    "phaser",
    "reverb",
    "frequency_shifter",
    "convolution",
    "bus_a",
    "bus_b",
    "fx_split",
    "voice",
    "stereo",
    "pitch",
    "velocity",
    "view",
];

/// Splits `family_N_rest` for the indexed families.
fn split_indexed(name: &str) -> Option<(&str, usize, &str)> {
    for family in INDEXED_FAMILIES {
        let Some(tail) = name.strip_prefix(family) else { continue };
        let Some(tail) = tail.strip_prefix('_') else { continue };
        let digits: String = tail.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() {
            continue;
        }
        let index: usize = digits.parse().ok()?;
        let rest = &tail[digits.len()..];
        // `macro_control_1` has no rest; everything else does.
        let rest = rest.strip_prefix('_').unwrap_or(rest);
        return Some((family, index, rest));
    }
    None
}

/// The place of a parameter, by its table name.
pub fn place(name: &str) -> Place {
    if let Some((family, index, rest)) = split_indexed(name) {
        if family == "macro_control" {
            return Place { module: "macros".into(), key: format!("macro_{index}") };
        }
        return Place { module: format!("{family}_{index}"), key: rest.to_string() };
    }
    // `filter_fx_*` before the generic scan, since `filter` alone is an
    // indexed family and would not match `filter_fx`.
    for family in FLAT_FAMILIES {
        if let Some(rest) = name.strip_prefix(family).and_then(|r| r.strip_prefix('_')) {
            return Place { module: family.to_string(), key: rest.to_string() };
        }
    }
    Place { module: "global".into(), key: name.to_string() }
}

/// The table name for a module and a key: the inverse of [`place`].
pub fn table_name(module: &str, key: &str) -> String {
    if module == "macros" {
        if let Some(n) = key.strip_prefix("macro_") {
            return format!("macro_control_{n}");
        }
    }
    if module == "global" {
        return key.to_string();
    }
    format!("{module}_{key}")
}

/// Canonical order of modules in a file. Unknown modules sort last, by
/// name, so a future family still has a stable place.
pub fn module_rank(module: &str) -> (usize, usize, String) {
    if module == "global" {
        return (0, 0, String::new());
    }
    if let Some((family, index, _)) = split_indexed(&format!("{module}_x")) {
        let family_rank = INDEXED_FAMILIES.iter().position(|f| *f == family).unwrap_or(99);
        return (1 + family_rank, index, String::new());
    }
    if module == "macros" {
        return (1 + INDEXED_FAMILIES.len(), 0, String::new());
    }
    if let Some(rank) = FLAT_FAMILIES.iter().position(|f| *f == module) {
        return (20 + rank, 0, String::new());
    }
    (99, 0, module.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_indexed_flat_and_global_names() {
        assert_eq!(place("osc_1_level"), Place { module: "osc_1".into(), key: "level".into() });
        assert_eq!(place("filter_fx_cutoff"), Place { module: "filter_fx".into(), key: "cutoff".into() });
        assert_eq!(place("filter_2_cutoff"), Place { module: "filter_2".into(), key: "cutoff".into() });
        assert_eq!(place("macro_control_3"), Place { module: "macros".into(), key: "macro_3".into() });
        assert_eq!(place("bus_a_delay_on"), Place { module: "bus_a".into(), key: "delay_on".into() });
        assert_eq!(place("polyphony"), Place { module: "global".into(), key: "polyphony".into() });
        assert_eq!(place("lfo_12_frequency"), Place { module: "lfo_12".into(), key: "frequency".into() });
    }

    #[test]
    fn place_and_table_name_are_inverses() {
        for name in ["osc_4_wave_frame", "env_7_attack", "macro_control_8", "chorus_dry_wet", "volume", "bus_b_reverb_dry_wet"] {
            let p = place(name);
            assert_eq!(table_name(&p.module, &p.key), name);
        }
    }

    #[test]
    fn modules_sort_in_a_stable_order() {
        let mut modules = vec!["chorus", "osc_2", "global", "lfo_1", "osc_1", "filter_fx", "macros", "bus_a", "env_1"];
        modules.sort_by_key(|m| module_rank(m));
        assert_eq!(modules, ["global", "osc_1", "osc_2", "env_1", "lfo_1", "macros", "filter_fx", "chorus", "bus_a"]);
    }
}
