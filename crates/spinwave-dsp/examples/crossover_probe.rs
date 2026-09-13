//! The Linkwitz-Riley crossover alone, as `vital_golden --crossover` runs
//! it: an impulse then a 55 Hz saw (no input file), or a raw stereo f32
//! file (`--input <raw>`), low and high outputs of lane 0 interleaved —
//! or, with `--sum`, their sum per channel (the compressor's low-only
//! band path at ratio 0, an allpass). Compared bit for bit against the
//! reference's file to place a residual inside or outside the filter.
//!
//!     cargo run --release -p spinwave-dsp --example crossover_probe -- 120 88200 out.raw [--input in.raw] [--sum]
use spinwave_dsp::filters::LinkwitzRileyFilter;
use spinwave_poly::PolyF32;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cutoff: f32 = args[1].parse().unwrap();
    let rate: f32 = args[2].parse().unwrap();
    let input_path = args.iter().position(|a| a == "--input").map(|i| args[i + 1].clone());
    let sum = args.iter().any(|a| a == "--sum");
    let mut filter = LinkwitzRileyFilter::new(cutoff, rate);
    let input: Vec<PolyF32> = match input_path {
        Some(path) => {
            let bytes = std::fs::read(path).unwrap();
            let samples: Vec<f32> =
                bytes.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            samples.chunks(2).map(|f| PolyF32::stereo(f[0], f[1])).collect()
        }
        None => (0..4096)
            .map(|i| {
                let saw = 2.0 * (55.0 * i as f32 / rate).rem_euclid(1.0) - 1.0;
                PolyF32::splat(if i == 0 { 1.0 } else { 0.5 * saw })
            })
            .collect(),
    };
    let mut low = vec![PolyF32::ZERO; 128];
    let mut high = vec![PolyF32::ZERO; 128];
    let mut out: Vec<f32> = Vec::new();
    for block in input.chunks(128) {
        let n = block.len();
        filter.process(block, &mut low[..n], &mut high[..n]);
        for i in 0..n {
            if sum {
                let s = low[i] + high[i];
                out.push(s.lane(0));
                out.push(s.lane(1));
            } else {
                out.push(low[i].lane(0));
                out.push(high[i].lane(0));
            }
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[3], bytes).unwrap();
}
