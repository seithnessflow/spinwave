//! The multiband compressor alone, as `vital_golden --compressor` runs it:
//! ratios and gains at zero, table thresholds, attack and release 0.5,
//! mix 1, on the crossover probe's input, lanes 0 and 1 interleaved.
//!
//!     cargo run --release -p spinwave-dsp --example compressor_probe -- <bands> <rate> out.raw
use spinwave_dsp::effects::{BandOptions, MultibandCompressor, MultibandCompressorParams};
use spinwave_poly::PolyF32;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bands: usize = args[1].parse().unwrap();
    let rate: f32 = args[2].parse().unwrap();
    let mut compressor = MultibandCompressor::new(rate);
    compressor.set_sample_rate(rate);
    compressor.reset();
    let z = PolyF32::ZERO;
    let params = MultibandCompressorParams {
        enabled_bands: [BandOptions::Multiband, BandOptions::LowBand, BandOptions::HighBand, BandOptions::SingleBand][bands],
        attack: PolyF32::splat(0.5),
        release: PolyF32::splat(0.5),
        mix: PolyF32::ONE,
        low_upper_ratio: z, band_upper_ratio: z, high_upper_ratio: z,
        low_lower_ratio: z, band_lower_ratio: z, high_lower_ratio: z,
        low_upper_threshold_db: PolyF32::splat(-28.0),
        band_upper_threshold_db: PolyF32::splat(-25.0),
        high_upper_threshold_db: PolyF32::splat(-30.0),
        low_lower_threshold_db: PolyF32::splat(-35.0),
        band_lower_threshold_db: PolyF32::splat(-36.0),
        high_lower_threshold_db: PolyF32::splat(-35.0),
        low_output_gain_db: z, band_output_gain_db: z, high_output_gain_db: z,
    };
    let input: Vec<PolyF32> = (0..4096)
        .map(|i| {
            let saw = 2.0 * (55.0 * i as f32 / rate).rem_euclid(1.0) - 1.0;
            PolyF32::splat(if i == 0 { 1.0 } else { 0.5 * saw })
        })
        .collect();
    let mut output = vec![PolyF32::ZERO; 128];
    let mut out: Vec<f32> = Vec::new();
    for block in input.chunks(128) {
        compressor.process(&params, block, &mut output[..block.len()]);
        for s in &output[..block.len()] {
            out.push(s.lane(0));
            out.push(s.lane(1));
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[3], bytes).unwrap();
}
