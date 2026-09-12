//! Operation 1: render a patch under a scenario and measure it.

use serde::Serialize;
use spinwave_params::Preset;

use super::{describe, render, render_seed, Descriptors, OpError, Scenario, SelfTest};

#[derive(Clone, Debug, Serialize)]
pub struct Measurement {
    /// The operation's seed, echoed; the render used `render_seed(seed, 0)`.
    pub seed: u64,
    pub scenario: Scenario,
    pub descriptors: Descriptors,
    pub self_test: SelfTest,
    pub cost_ms: f32,
}

/// Renders and measures. The audio is returned too, for callers that
/// compare or write it.
pub fn measure_with_audio(preset: &Preset, scenario: &Scenario, seed: u64) -> Result<(Measurement, Vec<f32>), OpError> {
    let mut session = super::session();
    let rendered = render(&mut session, preset, scenario, render_seed(seed, 0))?;
    let started = std::time::Instant::now();
    let descriptors = describe(&rendered);
    let cost_ms = (rendered.cost + started.elapsed()).as_secs_f32() * 1000.0;
    Ok((Measurement { seed, scenario: scenario.clone(), descriptors, self_test: rendered.self_test, cost_ms }, rendered.samples))
}

pub fn measure(preset: &Preset, scenario: &Scenario, seed: u64) -> Result<Measurement, OpError> {
    measure_with_audio(preset, scenario, seed).map(|(m, _)| m)
}
