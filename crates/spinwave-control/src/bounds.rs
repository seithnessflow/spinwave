//! The bounds check of the golden bench.
//!
//! Rule (2026-09-12): a case whose value rests against a bound measures
//! nothing — every value of a case must be verified as interior to its
//! range for the whole measured duration. The rule was born of a
//! meta-modulation chain wired with amounts of 0.5 whose sum saturated at
//! 1 and hid the chain behind the clamp; it holds for every case, and
//! this module is the automatic form of it: every control a case sets,
//! and every value a case modulates (base + what the engine's matrices
//! summed into it, per block, per active voice, per sample where the
//! destination is audio rate), checked against the parameter's range.
//! Two static twins (`meta_bounds_twin_overflow`, `meta_power_bounds_...`)
//! saturate on purpose and say so in the corpus test's allowlist.

use spinwave_engine::engine::{EffectsModDest, SoundEngine};
use spinwave_engine::kernel::mod_matrix::ModDest;
use spinwave_params::preset::Preset;
use spinwave_plugin::patch::{parse_effects_mod_dest, parse_mod_dest};

/// A value that left its range.
#[derive(Clone, Debug, PartialEq)]
pub struct Excursion {
    /// Parameter name (`filter_1_cutoff`, `modulation_2_amount`...).
    pub name: String,
    /// The worst value seen (the farthest outside).
    pub value: f32,
    pub min: f32,
    pub max: f32,
    /// Block index the worst value was seen in; `None` for a static
    /// control set outside its range.
    pub block: Option<usize>,
}

impl Excursion {
    pub fn describe(&self) -> String {
        match self.block {
            Some(block) => format!(
                "{} reached {:.4} at block {} (range [{}, {}])",
                self.name, self.value, block, self.min, self.max
            ),
            None => format!(
                "{} is set to {:.4} (range [{}, {}])",
                self.name, self.value, self.min, self.max
            ),
        }
    }
}

enum Route {
    Voice(ModDest),
    Effects(EffectsModDest),
}

/// One modulated destination, with its base value and range.
struct Watched {
    name: String,
    route: Route,
    base: f32,
    min: f32,
    max: f32,
}

/// The check for one render: built from the preset before the blocks run,
/// fed every block, read at the end.
pub struct BoundsCheck {
    watched: Vec<Watched>,
    excursions: Vec<Excursion>,
}

impl BoundsCheck {
    /// Watches every destination the preset's connections modulate. Also
    /// checks, once, every control the preset sets against its range
    /// (a case cannot set a value outside the range either).
    pub fn for_preset(preset: &Preset) -> BoundsCheck {
        let table = spinwave_params::table::parameters();
        let mut check = BoundsCheck { watched: Vec::new(), excursions: Vec::new() };
        for (name, value) in &preset.settings.values {
            let (Some(details), Some(value)) = (table.lookup(name), value.as_f64()) else {
                continue;
            };
            let value = value as f32;
            if value < details.min || value > details.max {
                check.excursions.push(Excursion {
                    name: name.clone(),
                    value,
                    min: details.min,
                    max: details.max,
                    block: None,
                });
            }
        }
        for connection in &preset.settings.modulations {
            let name = connection.destination.clone();
            if check.watched.iter().any(|w| w.name == name) {
                continue;
            }
            let Some(details) = table.lookup(&name) else { continue };
            let route = match parse_mod_dest(&name) {
                Some(dest) => Route::Voice(dest),
                None => match parse_effects_mod_dest(&name) {
                    Some(dest) => Route::Effects(dest),
                    None => continue,
                },
            };
            let base = preset
                .settings
                .values
                .get(&name)
                .and_then(serde_json::Value::as_f64)
                .map_or(details.default_value, |v| v as f32);
            check.watched.push(Watched { name, route, base, min: details.min, max: details.max });
        }
        check
    }

    /// Reads the engine after a block.
    pub fn after_block(&mut self, engine: &SoundEngine, block: usize) {
        for watched in &self.watched {
            let (lo, hi) = match watched.route {
                Route::Voice(dest) => match engine.offset_extrema(dest) {
                    Some(extrema) => extrema,
                    None => continue,
                },
                Route::Effects(dest) => {
                    let offset = engine.effects_offset(dest);
                    (offset, offset)
                }
            };
            let (lo, hi) = (watched.base + lo, watched.base + hi);
            let worst = if lo < watched.min {
                lo
            } else if hi > watched.max {
                hi
            } else {
                continue;
            };
            let farther = |value: f32| {
                (value - watched.max).max(watched.min - value)
            };
            match self.excursions.iter_mut().find(|e| e.name == watched.name && e.block.is_some()) {
                Some(existing) => {
                    if farther(worst) > farther(existing.value) {
                        existing.value = worst;
                        existing.block = Some(block);
                    }
                }
                None => self.excursions.push(Excursion {
                    name: watched.name.clone(),
                    value: worst,
                    min: watched.min,
                    max: watched.max,
                    block: Some(block),
                }),
            }
        }
    }

    /// Every value that left its range, worst occurrence each.
    pub fn excursions(&self) -> &[Excursion] {
        &self.excursions
    }

    pub fn into_excursions(self) -> Vec<Excursion> {
        self.excursions
    }

    /// Number of destinations being watched (a check that watches nothing
    /// on a case with connections is a check that parsed nothing).
    pub fn watched(&self) -> usize {
        self.watched.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{NoteSpec, Session};

    fn render(json: &str) -> Vec<Excursion> {
        let mut session = Session::with_output_dir(std::env::temp_dir());
        session.set_check_bounds(true);
        session.load_preset_json(json).unwrap();
        let note = NoteSpec { note: 57, velocity: 0.9, start: 0.0, duration: 0.3, channel: 0 };
        session.render_samples(&[note], 0.5, 120.0);
        session.last_excursions.clone()
    }

    fn preset(level_amount: f32, flanger_dry_wet: f32) -> String {
        format!(
            r#"{{"synth_version":"1.0.7","preset_name":"b","settings":{{
                "osc_1_on":1,"osc_1_level":0.7,"flanger_dry_wet":{flanger_dry_wet},
                "modulation_1_amount":{level_amount},
                "modulations":[{{"source":"lfo_1","destination":"osc_1_level"}}]}}}}"#
        )
    }

    /// The check must see a value leave its range — and must see nothing
    /// when every value stays interior — or the corpus test guards nothing.
    #[test]
    fn flags_a_modulated_value_past_its_range_and_only_that() {
        // 0.7 + LFO x 0.7 passes 1 in [0, 1] within the note.
        let excursions = render(&preset(0.7, 0.4));
        assert_eq!(excursions.len(), 1, "{excursions:?}");
        assert_eq!(excursions[0].name, "osc_1_level");
        assert!(excursions[0].value > 1.0 && excursions[0].block.is_some(), "{:?}", excursions[0]);

        // 0.7 + LFO x 0.2 peaks at 0.9.
        assert!(render(&preset(0.2, 0.4)).is_empty());
    }

    #[test]
    fn flags_a_control_set_outside_its_range() {
        // flanger_dry_wet is [0, 0.5].
        let excursions = render(&preset(0.2, 0.8));
        assert_eq!(excursions.len(), 1, "{excursions:?}");
        assert_eq!(excursions[0].name, "flanger_dry_wet");
        assert_eq!(excursions[0].block, None);
    }

    #[test]
    fn watches_every_routed_connection() {
        let json = preset(0.2, 0.4);
        let preset = Preset::from_json(&json).unwrap();
        assert_eq!(BoundsCheck::for_preset(&preset).watched(), 1);
    }
}
