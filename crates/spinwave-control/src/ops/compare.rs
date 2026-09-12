//! Operation 2: two patches, one scenario — what changed in the text and
//! how far apart the renders are, where.

use serde::Serialize;
use spinwave_params::Preset;

use super::diff::{param_diff, ConnectionChange, ParamChange};
use super::distance::{distance, Distance, Options};
use super::{describe, render, render_seed, Descriptors, OpError, Scenario};
use crate::session::SAMPLE_RATE;

#[derive(Clone, Debug, Serialize)]
pub struct Comparison {
    pub seed: u64,
    pub parameters: Vec<ParamChange>,
    pub connections: Vec<ConnectionChange>,
    pub distance: Distance,
    pub a: Descriptors,
    pub b: Descriptors,
}

/// Both patches rendered under the same scenario with the SAME render
/// seed, so every random draw matches and the parameters are the only
/// difference; then measured and compared.
pub fn compare(a: &Preset, b: &Preset, scenario: &Scenario, seed: u64, options: Options) -> Result<Comparison, OpError> {
    let mut session = super::session();
    let ra = render(&mut session, a, scenario, render_seed(seed, 0))?;
    let rb = render(&mut session, b, scenario, render_seed(seed, 0))?;
    let (parameters, connections) = param_diff(a, b);
    Ok(Comparison {
        seed,
        parameters,
        connections,
        distance: distance(&ra.samples, &rb.samples, SAMPLE_RATE, options),
        a: describe(&ra),
        b: describe(&rb),
    })
}
