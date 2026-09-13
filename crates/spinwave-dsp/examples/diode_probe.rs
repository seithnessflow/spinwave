//! The diode filter alone, as `vital_golden --diode` runs it: a fresh
//! filter reset at sample 0, a 110 Hz saw at 0.5 from sample 0, the
//! cutoff (MIDI), resonance and drive (dB) from the command line, style
//! 12 dB, blend 0, lanes 0 and 1 interleaved. 8192 samples.
//!
//!     cargo run --release -p spinwave-dsp --example diode_probe -- <cutoff> <resonance> <drive_db> <rate> out.raw [--input in.raw]
use spinwave_dsp::filters::diode::DiodeFilter;
use spinwave_dsp::filters::filter_state::FilterState;
use spinwave_poly::{PolyF32, PolyMask};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cutoff: f32 = args[1].parse().unwrap();
    let resonance: f32 = args[2].parse().unwrap();
    let drive_db: f32 = args[3].parse().unwrap();
    let rate: f32 = args[4].parse().unwrap();
    let mut state = FilterState { midi_cutoff: PolyF32::splat(cutoff), ..FilterState::default() };
    state.resonance_percent = PolyF32::splat(resonance);
    state.set_drive_db(PolyF32::splat(drive_db));
    state.set_pass_blend(PolyF32::ZERO);
    let mut diode = DiodeFilter::new();
    // `--input <raw>`: mono f32 samples to filter instead of the saw (a
    // voice's filter input dumped from an engine).
    let input: Vec<PolyF32> = match args.iter().position(|a| a == "--input") {
        Some(i) => std::fs::read(&args[i + 1])
            .unwrap()
            .chunks_exact(4)
            .map(|b| PolyF32::splat(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
        None => (0..8192)
            .map(|i| PolyF32::splat(0.5 * (2.0 * (110.0 * i as f32 / rate).rem_euclid(1.0) - 1.0)))
            .collect(),
    };
    let cutoff_buffer = vec![PolyF32::splat(cutoff); 128];
    let mut output = vec![PolyF32::ZERO; 128];
    let mut out: Vec<f32> = Vec::new();
    for (index, block) in input.chunks(128).enumerate() {
        // As the voice kernel drives it: setup every block, the reset on
        // the block the note starts in.
        diode.setup(&state, rate);
        if index == 0 {
            diode.reset(PolyMask::all_on());
        }
        diode.process_modulated(block, &cutoff_buffer[..block.len()], &mut output[..block.len()]);
        for s in &output[..block.len()] {
            out.push(s.lane(0));
            out.push(s.lane(1));
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[5], bytes).unwrap();
}
