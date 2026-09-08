//! Voice filters: SVF, ladder, diode, dirty, Sallen-Key, comb, formant, and
//! the crossover/decimation utilities used for oversampling.

pub mod comb;
pub mod dc_filter;
pub mod decimator;
pub mod digital_svf;
pub mod diode;
pub mod dirty;
pub mod filter_state;
pub mod formant;
pub mod ladder;
pub mod linkwitz_riley;
pub mod one_pole;
pub mod sallen_key;
pub mod upsampler;

pub use comb::{CombFilter, CombFilterStyle, FeedbackStyle};
pub use dc_filter::DcFilter;
pub use decimator::{Decimator, FirHalfbandDecimator, IirHalfbandDecimator};
pub use digital_svf::{DigitalSvf, FilterValues};
pub use diode::DiodeFilter;
pub use dirty::DirtyFilter;
pub use filter_state::{CoefficientLookup, FilterState, FilterStyle};
pub use formant::{FormantFilter, VocalTract};
pub use ladder::LadderFilter;
pub use linkwitz_riley::LinkwitzRileyFilter;
pub use one_pole::{OnePoleFilter, Saturation};
pub use sallen_key::SallenKeyFilter;
pub use upsampler::Upsampler;
