//! The distortion alone, as `vital_golden --distortion` runs it: a 110 Hz
//! saw at 0.5 from sample 0 through one type at one drive (dB), lanes 0
//! and 1 interleaved, 4096 samples in blocks of 128.
//!
//!     cargo run --release -p spinwave-dsp --example distortion_probe -- <type> <drive_db> <rate> out.raw
use spinwave_dsp::effects::{Distortion, DistortionType};
use spinwave_poly::PolyF32;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dtype = match args[1].parse::<usize>().unwrap() {
        0 => DistortionType::SoftClip,
        1 => DistortionType::HardClip,
        2 => DistortionType::LinearFold,
        3 => DistortionType::SinFold,
        4 => DistortionType::BitCrush,
        _ => DistortionType::DownSample,
    };
    let drive_db: f32 = args[2].parse().unwrap();
    let rate: f32 = args[3].parse().unwrap();
    let mut distortion = Distortion::new(rate);
    let drive = vec![PolyF32::splat(drive_db); 128];
    let mut out: Vec<f32> = Vec::new();
    for start in (0..4096).step_by(128) {
        let mut block: Vec<PolyF32> = (start..start + 128)
            .map(|i| PolyF32::splat(0.5 * (2.0 * (110.0 * i as f32 / rate).rem_euclid(1.0) - 1.0)))
            .collect();
        distortion.process(dtype, &drive, &mut block);
        for s in &block {
            out.push(s.lane(0));
            out.push(s.lane(1));
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[4], bytes).unwrap();
}
