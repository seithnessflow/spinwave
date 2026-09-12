//! Operation 4: a diff in, a validated patch out, with the report and the
//! measured effect. What makes an edit visible and reversible by
//! construction: a parameter diff, never an opaque rewrite.

use serde::Serialize;
use spinwave_params::{LoadReport, Preset};

use super::diff::{apply_resolved, param_diff, resolve, ConnectionChange, Diff, ParamChange};
use super::distance::{distance, Distance, Options};
use super::explain::{Direction, Quality};
use super::{describe, render, render_seed, Descriptors, OpError, Scenario};
use crate::session::SAMPLE_RATE;

/// Whether the change went the way the caller asked.
#[derive(Clone, Debug, Serialize)]
pub struct GoalCheck {
    pub quality: Quality,
    pub direction: Direction,
    pub unit: &'static str,
    pub before: f32,
    pub after: f32,
    pub achieved: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Applied {
    pub seed: u64,
    /// The patch after the diff, loaded and rendered without error.
    pub preset: Preset,
    /// The changes actually made, spelled like the file.
    pub parameters: Vec<ParamChange>,
    pub connections: Vec<ConnectionChange>,
    /// The format's report on the diff: corrections, normalisations.
    pub report: LoadReport,
    pub before: Descriptors,
    pub after: Descriptors,
    pub distance: Distance,
    pub goal: Option<GoalCheck>,
}

/// Resolves the diff (refusing what does not parse or load), renders
/// before and after with the same seed, and measures what changed.
pub fn apply(preset: &Preset, diff: &Diff, scenario: &Scenario, goal: Option<(Quality, Direction)>, seed: u64) -> Result<Applied, OpError> {
    let resolved = resolve(diff)?;
    let after_preset = apply_resolved(preset, &resolved);
    let (parameters, connections) = param_diff(preset, &after_preset);
    if parameters.is_empty() && connections.is_empty() {
        return Err(OpError::Nothing { message: "the diff changes nothing".into() });
    }
    let mut session = super::session();
    let before = render(&mut session, preset, scenario, render_seed(seed, 0))?;
    let after = render(&mut session, &after_preset, scenario, render_seed(seed, 0))?;
    let (db, da) = (describe(&before), describe(&after));
    let goal = goal.map(|(quality, direction)| {
        let (b, a) = (quality.measure(&db), quality.measure(&da));
        let achieved = match direction {
            Direction::More => a > b,
            Direction::Less => a < b,
        };
        GoalCheck { quality, direction, unit: quality.unit(), before: b, after: a, achieved }
    });
    Ok(Applied {
        seed,
        preset: after_preset,
        parameters,
        connections,
        report: resolved.report,
        distance: distance(&before.samples, &after.samples, SAMPLE_RATE, Options::default()),
        before: db,
        after: da,
        goal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::diff::{Change, ChangeValue};
    use crate::ops::tests::saw_patch;

    #[test]
    fn a_cutoff_change_is_applied_measured_and_checked_against_its_goal() {
        let p = saw_patch();
        let diff = Diff::Changes(vec![Change { name: "filter_1_cutoff".into(), value: ChangeValue::Text("300 Hz".into()) }]);
        let a = apply(&p, &diff, &Scenario::lite(), Some((Quality::Brightness, Direction::Less)), 3).expect("applies");
        assert_eq!(a.parameters.len(), 1);
        assert!(a.parameters[0].to < a.parameters[0].from);
        let goal = a.goal.as_ref().unwrap();
        assert!(goal.achieved, "300 Hz is darker than the 80-semitone cutoff: {goal:?}");
        assert!(a.distance.total_db > 1.0);
        let wrong = apply(&p, &diff, &Scenario::lite(), Some((Quality::Brightness, Direction::More)), 3).unwrap();
        assert!(!wrong.goal.unwrap().achieved);
    }

    #[test]
    fn a_diff_that_changes_nothing_or_breaks_the_patch_is_refused() {
        let p = saw_patch();
        let same = Diff::Changes(vec![Change { name: "filter_1_cutoff".into(), value: ChangeValue::Engine(80.0) }]);
        assert!(matches!(apply(&p, &same, &Scenario::lite(), None, 1), Err(OpError::Nothing { .. })));
        let bad = Diff::Fragment("[filter_1]\ncutoff = \"eight hundred\"\n".into());
        assert!(matches!(apply(&p, &bad, &Scenario::lite(), None, 1), Err(OpError::Rejected { .. })));
    }
}
