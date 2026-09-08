//! Sample playback: band-limited buffers with looping, bouncing, root
//! pitch tracking and per-lane mip selection.
//!
//! Rework of Vital's `sample_source.{h,cpp}`. A loaded sample is expanded
//! into a pyramid of quality tiers: one 2x-upsampled buffer for playing
//! below root pitch, the original, and successively FIR-downsampled
//! octaves for playing above it, each with a loop-wrapped twin.

use spinwave_poly::utils::catmull_interpolation_matrix;
use spinwave_poly::{constants, math, utils, PolyF32, PolyMask, PolyU32};

use super::phase::u32_lt_signed;
use super::rng::Xorshift32;
use super::synth_oscillator::value_matrix_multi;

pub const DEFAULT_SAMPLE_LENGTH: usize = 44100;
pub const UPSAMPLE_TIMES: usize = 1;
pub const BUFFER_SAMPLES: usize = 4;
pub const MIN_SIZE: usize = 4;

pub const MAX_TRANSPOSE: f32 = 96.0;
pub const MIN_TRANSPOSE: f32 = -96.0;
pub const MIN_RATE: f32 = 0.25;
pub const MAX_RATE: f32 = 4.0;
pub const MAX_SAMPLE_AMPLITUDE: f32 = std::f32::consts::SQRT_2;

pub const NUM_DOWNSAMPLE_TAPS: usize = 55;
pub const NUM_UPSAMPLE_TAPS: usize = 52;

// FIR taps are kept digit-for-digit identical to the reference.
#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const UPSAMPLE_COEFFICIENTS: [f32; NUM_UPSAMPLE_TAPS] = [
    -0.000159813115702086552469274316479186382,
    0.000225405365781280835058009159865832771,
    -0.000378616814007205900686342525673921955,
    0.000594907533596884547516525643118256994,
    -0.000890530515941817101682742574553230952,
    0.001284040046393844676508866342601322685,
    -0.001796543223638378920792302295694753411,
    0.002451862103068884121692683208948437823,
    -0.003276873018553504678107568537370752892,
    0.004302012661141991003987961050825106213,
    -0.005561976429934398571952591794342879439,
    0.007097105459677621741576558633823879063,
    -0.008955232561651555595050311353588767815,
    0.011195057708851860467369476737076183781,
    -0.013890548104646217864033275191104621626,
    0.017139719620821350365424962092220084742,
    -0.021077036318492142763503238711564335972,
    0.025897497908177177783350941808748757467,
    -0.03189749744607761616776997470878995955,
    0.039555400754278852160084056777122896165,
    -0.049699764879031965714162311087420675904,
    0.063901297378209126476278356676630210131,
    -0.08553732517833501081128133591846562922,
    0.123410206086688845061871688812971115112,
    -0.209837893291539345774765479291090741754,
    0.63582677174146173815216798175242729485,
    0.63582677174146173815216798175242729485,
    -0.209837893291539345774765479291090741754,
    0.123410206086688845061871688812971115112,
    -0.08553732517833501081128133591846562922,
    0.063901297378209126476278356676630210131,
    -0.049699764879031965714162311087420675904,
    0.039555400754278852160084056777122896165,
    -0.03189749744607761616776997470878995955,
    0.025897497908177177783350941808748757467,
    -0.021077036318492142763503238711564335972,
    0.017139719620821350365424962092220084742,
    -0.013890548104646217864033275191104621626,
    0.011195057708851860467369476737076183781,
    -0.008955232561651555595050311353588767815,
    0.007097105459677621741576558633823879063,
    -0.005561976429934398571952591794342879439,
    0.004302012661141991003987961050825106213,
    -0.003276873018553504678107568537370752892,
    0.002451862103068884121692683208948437823,
    -0.001796543223638378920792302295694753411,
    0.001284040046393844676508866342601322685,
    -0.000890530515941817101682742574553230952,
    0.000594907533596884547516525643118256994,
    -0.000378616814007205900686342525673921955,
    0.000225405365781280835058009159865832771,
    -0.000159813115702086552469274316479186382,
];

#[rustfmt::skip]
#[allow(clippy::excessive_precision)]
const DOWNSAMPLE_COEFFICIENTS: [f32; NUM_DOWNSAMPLE_TAPS] = [
    -0.0013796309221920304,
    -0.0008322130675804714,
    0.0030100376204235577,
    0.00666031332700994,
    0.0040620073330527315,
    -0.003019073425031439,
    -0.004450269579432283,
    0.0030526281279541555,
    0.007614361286489334,
    -0.000546514301955849,
    -0.010099270019478761,
    -0.003465846383906444,
    0.011760981765402261,
    0.009402148654924303,
    -0.011429260748035207,
    -0.016935843679984037,
    0.008026778073943279,
    0.025557280950428782,
    -0.0002093220301655805,
    -0.03448379812688787,
    -0.013983156365753766,
    0.04279770831566429,
    0.03889228625534586,
    -0.049566024787935245,
    -0.09025827224454164,
    0.05398926693924448,
    0.31285587793730246,
    0.4444714418837066,
    0.31285587793730246,
    0.05398926693924448,
    -0.09025827224454164,
    -0.049566024787935245,
    0.03889228625534586,
    0.04279770831566429,
    -0.013983156365753766,
    -0.03448379812688787,
    -0.0002093220301655805,
    0.025557280950428782,
    0.008026778073943279,
    -0.016935843679984037,
    -0.011429260748035207,
    0.009402148654924303,
    0.011760981765402261,
    -0.003465846383906444,
    -0.010099270019478761,
    -0.000546514301955849,
    0.007614361286489334,
    0.0030526281279541555,
    -0.004450269579432283,
    -0.003019073425031439,
    0.0040620073330527315,
    0.00666031332700994,
    0.0030100376204235577,
    -0.0008322130675804714,
    -0.0013796309221920304,
];

fn filtered_sample(buffer: &[f32], index: usize) -> f32 {
    let radius = (NUM_DOWNSAMPLE_TAPS / 2) as i32;
    let index = index as i32;
    let start = (index - radius).max(0);
    let end = (buffer.len() as i32 - 1).min(index + radius);
    let mut total = 0.0;
    for i in start..=end {
        let coefficient = DOWNSAMPLE_COEFFICIENTS[(i - index + radius) as usize];
        total += coefficient * buffer[i as usize];
    }
    total
}

fn filtered_loop_sample(buffer: &[f32], index: usize) -> f32 {
    let radius = (NUM_DOWNSAMPLE_TAPS / 2) as i32;
    let size = buffer.len() as i32;
    let index = index as i32;
    let mut total = 0.0;
    for i in index - radius..=index + radius {
        let buffer_index = (i + size * radius).rem_euclid(size) as usize;
        let coefficient = DOWNSAMPLE_COEFFICIENTS[(i - index + radius) as usize];
        total += coefficient * buffer[buffer_index];
    }
    total
}

fn interpolated_sample(buffer: &[f32], index: usize) -> f32 {
    let radius = (NUM_UPSAMPLE_TAPS / 2) as i32;
    let index = index as i32;
    let start = (index - radius + 1).max(0);
    let end = (buffer.len() as i32 - 1).min(index + radius);
    let mut total = 0.0;
    for i in start..=end {
        let coefficient_index = (i - index + radius - 1) as usize;
        total += UPSAMPLE_COEFFICIENTS[coefficient_index] * buffer[i as usize];
    }
    total
}

fn upsample(original: &[f32], dest: &mut [f32]) {
    for i in 0..original.len() {
        dest[2 * i] = original[i];
        dest[2 * i + 1] = interpolated_sample(original, i);
    }
}

fn downsample(original: &[f32], dest: &mut [f32]) {
    for (i, value) in dest.iter_mut().enumerate() {
        *value = filtered_sample(original, 2 * i);
    }
}

fn downsample_loop(original: &[f32], dest: &mut [f32]) {
    for (i, value) in dest.iter_mut().enumerate() {
        *value = filtered_loop_sample(original, 2 * i);
    }
}

/// Builds the quality tiers for one channel: index 0 is 2x upsampled,
/// index 1 the original rate, then downsampled octaves. Every buffer is
/// padded with `BUFFER_SAMPLES` guard samples on both sides; the loop
/// variants wrap those guards around the cycle.
fn create_band_limited_buffers(buffer: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let size = buffer.len();

    let mut play = vec![0.0f32; size + 2 * BUFFER_SAMPLES];
    play[BUFFER_SAMPLES..BUFFER_SAMPLES + size].copy_from_slice(buffer);
    let mut looped = play.clone();
    for i in 0..BUFFER_SAMPLES {
        looped[i] = looped[size + i];
        looped[size + BUFFER_SAMPLES + i] = looped[BUFFER_SAMPLES + i];
    }

    let upsampled_size = size * 2;
    let mut up = vec![0.0f32; upsampled_size + 2 * BUFFER_SAMPLES];
    let (play_body, up_body) = (
        &buffer[..size],
        &mut up[BUFFER_SAMPLES..BUFFER_SAMPLES + upsampled_size],
    );
    upsample(play_body, up_body);
    let up_loop = up.clone();

    let mut destination = vec![up, play.clone()];
    let mut loop_destination = vec![up_loop, looped.clone()];

    let mut play_prev = play;
    let mut loop_prev = looped;
    let mut current_size = size;
    while current_size >= MIN_SIZE {
        let next_size = current_size.div_ceil(2);
        let mut next = vec![0.0f32; next_size + 2 * BUFFER_SAMPLES];
        let mut next_loop = vec![0.0f32; next_size + 2 * BUFFER_SAMPLES];

        downsample(
            &play_prev[BUFFER_SAMPLES..BUFFER_SAMPLES + current_size],
            &mut next[BUFFER_SAMPLES..BUFFER_SAMPLES + next_size],
        );
        downsample_loop(
            &loop_prev[BUFFER_SAMPLES..BUFFER_SAMPLES + current_size],
            &mut next_loop[BUFFER_SAMPLES..BUFFER_SAMPLES + next_size],
        );

        for i in 0..BUFFER_SAMPLES {
            next_loop[i] = loop_prev[next_size + i];
            next_loop[next_size + BUFFER_SAMPLES + i] = next_loop[BUFFER_SAMPLES + i];
        }

        destination.push(next.clone());
        loop_destination.push(next_loop.clone());
        play_prev = next;
        loop_prev = next_loop;
        current_size = next_size;
    }

    (destination, loop_destination)
}

/// A loaded sample with band-limited quality tiers per channel.
pub struct Sample {
    pub name: String,
    length: usize,
    sample_rate: u32,
    stereo: bool,
    left_buffers: Vec<Vec<f32>>,
    left_loop_buffers: Vec<Vec<f32>>,
    right_buffers: Vec<Vec<f32>>,
    right_loop_buffers: Vec<Vec<f32>>,
    /// Slice markers in original sample frames, sorted ascending.
    slices: Vec<usize>,
}

impl Default for Sample {
    /// One second of white noise, like the reference's default sample.
    fn default() -> Self {
        let mut rng = Xorshift32::new(0x517_c0de);
        let buffer: Vec<f32> =
            (0..DEFAULT_SAMPLE_LENGTH).map(|_| rng.next_in(-0.9, 0.9)).collect();
        let mut sample = Sample::blank("White Noise");
        sample.load_sample(&buffer, constants::DEFAULT_SAMPLE_RATE);
        sample
    }
}

impl Sample {
    const MAX_SIZE: usize = 1_764_000;

    pub fn new() -> Sample {
        Sample::default()
    }

    /// A sample with no audio loaded yet (no tier pyramid).
    fn blank(name: &str) -> Sample {
        Sample {
            name: name.to_string(),
            length: 0,
            sample_rate: constants::DEFAULT_SAMPLE_RATE,
            stereo: false,
            left_buffers: Vec::new(),
            left_loop_buffers: Vec::new(),
            right_buffers: Vec::new(),
            right_loop_buffers: Vec::new(),
            slices: Vec::new(),
        }
    }

    /// Builds a mono sample directly, without the default noise detour.
    pub fn from_mono(name: &str, buffer: &[f32], sample_rate: u32) -> Sample {
        let mut sample = Sample::blank(name);
        sample.load_sample(buffer, sample_rate);
        sample
    }

    /// Builds a stereo sample directly, without the default noise detour.
    pub fn from_stereo(name: &str, left: &[f32], right: &[f32], sample_rate: u32) -> Sample {
        let mut sample = Sample::blank(name);
        sample.load_stereo_sample(left, right, sample_rate);
        sample
    }

    pub fn load_sample(&mut self, buffer: &[f32], sample_rate: u32) {
        let size = buffer.len().min(Self::MAX_SIZE);
        let (buffers, loop_buffers) = create_band_limited_buffers(&buffer[..size]);
        self.length = size;
        self.sample_rate = sample_rate;
        self.stereo = false;
        self.left_buffers = buffers;
        self.left_loop_buffers = loop_buffers;
        self.right_buffers = Vec::new();
        self.right_loop_buffers = Vec::new();
        self.slices.clear();
    }

    pub fn load_stereo_sample(&mut self, left: &[f32], right: &[f32], sample_rate: u32) {
        let size = left.len().min(right.len()).min(Self::MAX_SIZE);
        let (left_buffers, left_loop) = create_band_limited_buffers(&left[..size]);
        let (right_buffers, right_loop) = create_band_limited_buffers(&right[..size]);
        self.length = size;
        self.sample_rate = sample_rate;
        self.stereo = true;
        self.left_buffers = left_buffers;
        self.left_loop_buffers = left_loop;
        self.right_buffers = right_buffers;
        self.right_loop_buffers = right_loop;
        self.slices.clear();
    }

    #[inline]
    pub fn original_length(&self) -> usize {
        self.length
    }

    /// Playback length in upsampled steps.
    #[inline]
    pub fn active_length(&self) -> usize {
        self.length * (1 << UPSAMPLE_TIMES)
    }

    #[inline]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    pub fn stereo(&self) -> bool {
        self.stereo
    }

    /// Quality tier for a playback step size (in upsampled steps).
    pub fn active_index(&self, delta: f32) -> usize {
        let value = (delta as i32).max(1) as u32;
        let octaves = (31 - value.leading_zeros()) as usize;
        octaves.min(self.left_buffers.len() - 1)
    }

    #[inline]
    pub fn left_buffer(&self, index: usize) -> &[f32] {
        &self.left_buffers[index]
    }

    #[inline]
    pub fn left_loop_buffer(&self, index: usize) -> &[f32] {
        &self.left_loop_buffers[index]
    }

    #[inline]
    pub fn right_buffer(&self, index: usize) -> &[f32] {
        if self.stereo {
            &self.right_buffers[index]
        } else {
            &self.left_buffers[index]
        }
    }

    #[inline]
    pub fn right_loop_buffer(&self, index: usize) -> &[f32] {
        if self.stereo {
            &self.right_loop_buffers[index]
        } else {
            &self.left_loop_buffers[index]
        }
    }

    /// Slice markers in original sample frames.
    #[inline]
    pub fn slices(&self) -> &[usize] {
        &self.slices
    }

    /// Replaces the slice markers (sorted, clamped to the sample length).
    pub fn set_slices(&mut self, mut slices: Vec<usize>) {
        slices.retain(|&position| position < self.length);
        slices.sort_unstable();
        slices.dedup();
        self.slices = slices;
    }

    /// The original-rate audio of one channel (tier 1, guard samples stripped).
    fn original_channel(&self, right: bool) -> &[f32] {
        let buffers = if right && self.stereo { &self.right_buffers } else { &self.left_buffers };
        &buffers[1][BUFFER_SAMPLES..BUFFER_SAMPLES + self.length]
    }

    /// Original audio folded to mono (`(L + R) / 2` for stereo samples).
    fn mono_frames(&self) -> Vec<f32> {
        let left = self.original_channel(false);
        if !self.stereo {
            return left.to_vec();
        }
        let right = self.original_channel(true);
        left.iter().zip(right).map(|(l, r)| 0.5 * (l + r)).collect()
    }

    /// Parses a minimal WAV file (PCM16 or float32, mono or stereo).
    ///
    /// Mono files build a mono pyramid, stereo files a stereo one; other
    /// channel counts, bit depths and codecs are rejected.
    pub fn from_wav_bytes(bytes: &[u8]) -> Result<Sample, String> {
        let (format, channels, sample_rate, bits, data) = parse_wav_chunks(bytes)?;

        let frames: Vec<Vec<f32>> = match (format, bits) {
            (1, 16) => decode_wav_samples(data, channels as usize, 2, |b| {
                i16::from_le_bytes([b[0], b[1]]) as f32 * (1.0 / 32768.0)
            }),
            (3, 32) => decode_wav_samples(data, channels as usize, 4, |b| {
                f32::from_le_bytes([b[0], b[1], b[2], b[3]])
            }),
            (format, bits) => {
                return Err(format!("unsupported WAV encoding: format {format}, {bits} bits"))
            }
        };
        if frames[0].is_empty() {
            return Err("WAV data chunk holds no complete frame".to_string());
        }

        match channels {
            1 => Ok(Sample::from_mono("wav", &frames[0], sample_rate)),
            2 => Ok(Sample::from_stereo("wav", &frames[0], &frames[1], sample_rate)),
            n => Err(format!("unsupported WAV channel count: {n}")),
        }
    }

    /// Searches the middle-to-late region for a sustain loop: the pair of
    /// positive-going zero crossings at least ~50 ms apart whose surrounding
    /// windows correlate best. Returns `(loop_start, loop_end)` in original
    /// sample frames, ready for the [`SampleSourceParams`] loop overrides.
    pub fn detect_loop(&self) -> Option<(usize, usize)> {
        let data = self.mono_frames();
        let length = data.len();
        let min_gap = ((self.sample_rate as usize) / 20).max(64);
        let window = 256.min(length / 8).max(16);
        if length < min_gap + 2 * window + 2 {
            return None;
        }

        let search_start = (length * 2 / 5).max(1);
        let search_end = length - window;
        let mut crossings = Vec::new();
        for i in search_start..search_end {
            if data[i - 1] <= 0.0 && data[i] > 0.0 {
                crossings.push(i);
            }
        }
        if crossings.len() < 2 {
            return None;
        }

        // Keep the pair search bounded on long, high-pitched material.
        const MAX_CANDIDATES: usize = 96;
        let stride = crossings.len().div_ceil(MAX_CANDIDATES);
        let candidates: Vec<usize> = crossings.iter().copied().step_by(stride).collect();

        let energy = |at: usize| -> f32 {
            data[at..at + window].iter().map(|v| v * v).sum::<f32>()
        };
        let correlate = |a: usize, b: usize| -> f32 {
            data[a..a + window].iter().zip(&data[b..b + window]).map(|(x, y)| x * y).sum()
        };

        let mut best: Option<(usize, usize)> = None;
        let mut best_score = 0.0f32;
        for (i, &start) in candidates.iter().enumerate() {
            let start_energy = energy(start);
            if start_energy <= f32::EPSILON {
                continue;
            }
            for &end in &candidates[i + 1..] {
                if end - start < min_gap {
                    continue;
                }
                let end_energy = energy(end);
                if end_energy <= f32::EPSILON {
                    continue;
                }
                let score = correlate(start, end) / (start_energy * end_energy).sqrt();
                if score > best_score {
                    best_score = score;
                    best = Some((start, end));
                }
            }
        }

        // A loop only counts when the seam windows genuinely match.
        if best_score > 0.5 { best } else { None }
    }

    /// Onset-based slice markers: positions (in original sample frames) where
    /// the short-time energy rises sharply. `sensitivity` in `0..=1` — higher
    /// values detect softer onsets. Feed the result to [`Sample::set_slices`]
    /// to make the markers playable through `SampleSourceParams::slice`.
    pub fn detect_slices(&self, sensitivity: f32) -> Vec<usize> {
        let data = self.mono_frames();
        let length = data.len();
        const HOP: usize = 64;
        const WINDOW: usize = 256;
        if length < WINDOW {
            return Vec::new();
        }

        let num_hops = (length - WINDOW) / HOP + 1;
        let energies: Vec<f32> = (0..num_hops)
            .map(|k| data[k * HOP..k * HOP + WINDOW].iter().map(|v| v * v).sum())
            .collect();
        let max_energy = energies.iter().copied().fold(0.0f32, f32::max);
        if max_energy <= 1e-9 {
            return Vec::new();
        }

        let sensitivity = sensitivity.clamp(0.0, 1.0);
        let floor = max_energy * (0.002 + 0.2 * (1.0 - sensitivity));
        let rise_ratio = 1.5 + 6.0 * (1.0 - sensitivity);
        let refractory = ((self.sample_rate as usize) / 20).div_ceil(HOP);

        let mut markers = Vec::new();
        let mut last_onset: Option<usize> = None;
        for (k, &e) in energies.iter().enumerate() {
            let previous = if k == 0 { 0.0 } else { energies[k - 1] };
            let in_refractory = last_onset.is_some_and(|last| k - last < refractory);
            if e > floor && e > previous * rise_ratio && !in_refractory {
                markers.push(k * HOP);
                last_onset = Some(k);
            }
        }
        markers
    }
}

/// Locates the fmt/data chunks: `(format, channels, sample_rate, bits, data)`.
fn parse_wav_chunks(bytes: &[u8]) -> Result<(u16, u16, u32, u16, &[u8]), String> {
    let u16le = |offset: usize| -> Option<u16> {
        Some(u16::from_le_bytes([*bytes.get(offset)?, *bytes.get(offset + 1)?]))
    };
    let u32le = |offset: usize| -> Option<u32> {
        Some(u32::from_le_bytes([
            *bytes.get(offset)?,
            *bytes.get(offset + 1)?,
            *bytes.get(offset + 2)?,
            *bytes.get(offset + 3)?,
        ]))
    };

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_string());
    }

    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32le(offset + 4).ok_or("truncated chunk header")? as usize;
        let body_start = offset + 8;
        let body_end = body_start + size;
        if body_end > bytes.len() {
            return Err("truncated WAV chunk".to_string());
        }
        match id {
            b"fmt " => {
                if size < 16 {
                    return Err("fmt chunk too small".to_string());
                }
                let mut format = u16le(body_start).unwrap();
                let channels = u16le(body_start + 2).unwrap();
                let sample_rate = u32le(body_start + 4).unwrap();
                let bits = u16le(body_start + 14).unwrap();
                // WAVE_FORMAT_EXTENSIBLE: the real codec sits in the subformat.
                if format == 0xFFFE && size >= 40 {
                    format = u16le(body_start + 24).unwrap();
                }
                fmt = Some((format, channels, sample_rate, bits));
            }
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {}
        }
        offset = body_end + (size & 1);
    }

    let (format, channels, sample_rate, bits) = fmt.ok_or("missing fmt chunk")?;
    let data = data.ok_or("missing data chunk")?;
    if channels == 0 {
        return Err("fmt chunk declares zero channels".to_string());
    }
    Ok((format, channels, sample_rate, bits, data))
}

/// Deinterleaves `data` into one `Vec<f32>` per channel.
fn decode_wav_samples(
    data: &[u8],
    channels: usize,
    bytes_per_sample: usize,
    decode: impl Fn(&[u8]) -> f32,
) -> Vec<Vec<f32>> {
    let frame_bytes = channels * bytes_per_sample;
    let num_frames = data.len() / frame_bytes;
    let mut out = vec![Vec::with_capacity(num_frames); channels];
    for frame in 0..num_frames {
        for (channel, samples) in out.iter_mut().enumerate() {
            let at = frame * frame_bytes + channel * bytes_per_sample;
            samples.push(decode(&data[at..at + bytes_per_sample]));
        }
    }
    out
}

/// Block-rate parameters for [`SampleSource`].
#[derive(Clone, Debug)]
pub struct SampleSourceParams {
    pub midi: PolyF32,
    pub keytrack: bool,
    pub level: PolyF32,
    pub random_phase: bool,
    pub transpose: PolyF32,
    pub transpose_quantize: u32,
    pub tune: PolyF32,
    pub loop_sample: bool,
    pub bounce: bool,
    pub pan: PolyF32,
    /// Loop region start override, in original sample frames. Both overrides
    /// must be set (with `loop_start < loop_end <= length`) and `loop_sample`
    /// on for the custom region to engage; playback then runs from the note
    /// start into the region and wraps `loop_end -> loop_start` (bounce keeps
    /// full-buffer semantics and ignores the overrides). Feed
    /// [`Sample::detect_loop`] results here.
    pub loop_start: Option<usize>,
    /// Loop region end override, in original sample frames (exclusive).
    pub loop_end: Option<usize>,
    /// Slice playback: note-on starts at slice marker `N` (wrapping modulo
    /// the marker count) from [`Sample::slices`], so a sliced break can be
    /// played chromatically with `keytrack` off. Overrides `random_phase`
    /// and `start_offset`.
    pub slice: Option<usize>,
    /// Note-on start position in original sample frames (SFZ `offset`).
    pub start_offset: usize,
    /// Tape-style playback rate in `0.25..=4.0`, multiplied on top of the
    /// transpose/tune pitch ratio: rate 2.0 plays one octave higher AND in
    /// half the time, exactly like doubling tape speed. With transpose the
    /// ratios multiply (rate 2.0 + transpose +12 = ×4 speed).
    pub rate: f32,
}

impl Default for SampleSourceParams {
    fn default() -> Self {
        SampleSourceParams {
            midi: PolyF32::splat(constants::MIDI_TRACK_CENTER as f32),
            keytrack: false,
            level: PolyF32::ONE,
            random_phase: false,
            transpose: PolyF32::ZERO,
            transpose_quantize: 0,
            tune: PolyF32::ZERO,
            loop_sample: false,
            bounce: false,
            pan: PolyF32::ZERO,
            loop_start: None,
            loop_end: None,
            slice: None,
            start_offset: 0,
            rate: 1.0,
        }
    }
}

/// The sample playback engine (raw + leveled outputs).
pub struct SampleSource {
    sample: Sample,
    pan_amplitude: PolyF32,
    transpose_quantize: u32,
    last_quantized_transpose: PolyF32,
    sample_index: PolyF32,
    sample_fraction: PolyF32,
    phase_inc: PolyF32,
    bounce_mask: PolyMask,
    playback_phase: PolyF32,
    sample_rate: f32,
    pending_reset: PolyMask,
    pending_reset_offset: PolyU32,
    rng: Xorshift32,
}

impl Default for SampleSource {
    fn default() -> Self {
        Self::new()
    }
}

impl SampleSource {
    pub fn new() -> SampleSource {
        Self::with_sample(Sample::default())
    }

    /// Builds a source around an already-loaded sample (skips the default
    /// white-noise pyramid).
    pub fn with_sample(sample: Sample) -> SampleSource {
        SampleSource {
            sample,
            pan_amplitude: PolyF32::ZERO,
            transpose_quantize: 0,
            last_quantized_transpose: PolyF32::ZERO,
            sample_index: PolyF32::ZERO,
            sample_fraction: PolyF32::ZERO,
            phase_inc: PolyF32::ZERO,
            bounce_mask: PolyMask::NONE,
            playback_phase: PolyF32::ZERO,
            sample_rate: constants::DEFAULT_SAMPLE_RATE as f32,
            pending_reset: PolyMask::NONE,
            pending_reset_offset: PolyU32::ZERO,
            rng: Xorshift32::new(0x5a17),
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn sample(&self) -> &Sample {
        &self.sample
    }

    pub fn sample_mut(&mut self) -> &mut Sample {
        &mut self.sample
    }

    /// Schedules a note-on reset at `sample_offset` inside the next block.
    pub fn note_on(&mut self, mask: PolyMask, sample_offset: PolyU32) {
        self.pending_reset |= mask;
        self.pending_reset_offset = mask.select_u32(sample_offset, self.pending_reset_offset);
    }

    /// Normalized playback position per lane, `0..=1`.
    pub fn playback_phase(&self) -> PolyF32 {
        self.playback_phase
    }

    fn snap_transpose(&mut self, input_midi: PolyF32, transpose: PolyF32, quantize: u32) -> PolyF32 {
        if quantize == 0 {
            return input_midi + transpose;
        }

        let global = quantize >> constants::NOTES_PER_OCTAVE != 0;
        let (pre_add, post_add) = if global {
            (input_midi, PolyF32::ZERO)
        } else {
            (PolyF32::ZERO, input_midi)
        };

        let snapped = utils::snap_transpose(pre_add + transpose, quantize);

        if self.transpose_quantize != 0 {
            self.phase_inc *= math::midi_offset_to_ratio(snapped - self.last_quantized_transpose);
        }

        self.last_quantized_transpose = snapped;
        self.transpose_quantize = quantize;
        post_add + snapped
    }

    pub fn process(
        &mut self,
        params: &SampleSourceParams,
        num_samples: usize,
        raw_out: &mut [PolyF32],
        leveled_out: &mut [PolyF32],
    ) {
        assert!(num_samples > 0);
        assert!(raw_out.len() >= num_samples && leveled_out.len() >= num_samples);

        let mut current_pan_amplitude = self.pan_amplitude;
        self.pan_amplitude = math::pan_amplitude(params.pan.clamp(-1.0, 1.0));

        let input_midi = if params.keytrack {
            params.midi - constants::MIDI_TRACK_CENTER as f32
        } else {
            PolyF32::ZERO
        };

        let transpose =
            self.snap_transpose(input_midi, params.transpose, params.transpose_quantize);
        let transpose = (transpose + params.tune).clamp(MIN_TRANSPOSE, MAX_TRANSPOSE);

        let sample_rate_ratio = self.sample.sample_rate() as f32 / self.sample_rate;
        let mut current_phase_inc = self.phase_inc;
        // Tape-style rate: scales speed AND pitch on top of transpose/tune.
        self.phase_inc = math::midi_offset_to_ratio(transpose)
            * (params.rate.clamp(MIN_RATE, MAX_RATE) * sample_rate_ratio)
            * (1 << UPSAMPLE_TIMES) as f32;

        let audio_length = self.sample.active_length();
        let reset_mask = self.pending_reset;
        let trigger_offset = self.pending_reset_offset;
        self.pending_reset = PolyMask::NONE;
        let mut reset_offset = trigger_offset.to_f32_signed();
        current_pan_amplitude = reset_mask.select(self.pan_amplitude, current_pan_amplitude);
        current_phase_inc = reset_mask.select(self.phase_inc, current_phase_inc);
        self.bounce_mask &= !reset_mask;
        reset_offset *= current_phase_inc;

        let mut reset_value = -reset_offset;
        if params.random_phase {
            let first_mask =
                PolyMask::from_u32(PolyU32::from_lanes([u32::MAX, u32::MAX, 0, 0]));
            let value1 = PolyF32::splat(self.rng.next_f32() * audio_length as f32);
            let value2 = PolyF32::splat(self.rng.next_f32() * audio_length as f32);
            reset_value = first_mask.select(value2, value1) - reset_offset;
        }

        // Slice / offset start: notes begin at a marker instead of frame 0.
        let start_frame = match params.slice {
            Some(slice) => {
                let slices = self.sample.slices();
                if slices.is_empty() { None } else { Some(slices[slice % slices.len()]) }
            }
            None if params.start_offset > 0 => Some(params.start_offset),
            None => None,
        };
        if let Some(frame) = start_frame {
            let start = (frame.min(self.sample.original_length()) << UPSAMPLE_TIMES) as f32;
            reset_value = PolyF32::splat(start) - reset_offset;
        }

        self.sample_index = reset_mask.select(reset_value.floor(), self.sample_index);
        self.sample_fraction =
            reset_mask.select(reset_value - reset_value.floor(), self.sample_fraction);

        let loop_enabled_mask = if params.loop_sample {
            PolyMask::all_on()
        } else {
            PolyMask::NONE
        };

        let bounce_enabled_mask = if params.bounce {
            PolyMask::all_on()
        } else {
            self.bounce_mask = PolyMask::NONE;
            PolyMask::NONE
        };

        let mut phase_mult = PolyF32::ONE;
        let mut buffer_indices = [0usize; 4];
        for (i, slot) in buffer_indices.iter_mut().enumerate() {
            let index = self.sample.active_index(self.phase_inc.lane(i));
            *slot = index;
            phase_mult.set_lane(i, 1.0 / (1 << index) as f32);
        }
        // Custom loop region (frames -> upsampled steps). Only meaningful for
        // plain looping: bounce keeps full-buffer semantics.
        let custom_loop = match (params.loop_start, params.loop_end) {
            (Some(start), Some(end))
                if params.loop_sample
                    && !params.bounce
                    && start < end
                    && end <= self.sample.original_length() =>
            {
                Some((start << UPSAMPLE_TIMES, end << UPSAMPLE_TIMES))
            }
            _ => None,
        };

        let pick = |i: usize| -> &[f32] {
            let index = buffer_indices[i];
            // A custom region never crosses the buffer end, so the plain
            // (non-wrapped) tiers interpolate correctly across the seam.
            if params.loop_sample && !params.bounce && custom_loop.is_none() {
                if i % 2 == 1 {
                    self.sample.right_loop_buffer(index)
                } else {
                    self.sample.left_loop_buffer(index)
                }
            } else if i % 2 == 1 {
                self.sample.right_buffer(index)
            } else {
                self.sample.left_buffer(index)
            }
        };
        let audio_buffers: [&[f32]; 4] = [pick(0), pick(1), pick(2), pick(3)];

        let sample_inc = 1.0 / num_samples as f32;
        let delta_pan_amplitude = (self.pan_amplitude - current_pan_amplitude) * sample_inc;
        let delta_phase_inc = (self.phase_inc - current_phase_inc) * sample_inc;

        let (wrap_end, wrap_amount) = match custom_loop {
            Some((start, end)) => (end as f32, (end - start) as f32),
            None => (audio_length as f32, audio_length as f32),
        };
        let length = PolyF32::splat(wrap_end);
        let wrap_amount = PolyF32::splat(wrap_amount);
        let mut current_fraction = self.sample_fraction;
        let mut current_index = self.sample_index.min(length);
        let mut current_bounce = self.bounce_mask;

        for out in raw_out.iter_mut().take(num_samples) {
            current_phase_inc += delta_phase_inc;

            let adjusted = current_bounce.select(length - current_index, current_index);
            let index_phase = adjusted.max(PolyF32::ZERO) * phase_mult;
            let fraction_phase = current_fraction * phase_mult;

            let start_indices = index_phase.to_i32_floor();
            let rounded_down_phase = start_indices.to_f32_signed();
            let mut t = index_phase - rounded_down_phase + fraction_phase;
            t = current_bounce.select(PolyF32::ONE - t, t);

            let interpolation_matrix = catmull_interpolation_matrix(t);
            let mut values = value_matrix_multi(&audio_buffers, start_indices);
            values.transpose();
            *out = interpolation_matrix.multiply_and_sum_rows(&values);

            current_fraction += current_phase_inc;
            let increment = current_fraction.floor();
            current_fraction -= increment;

            current_index += increment;
            let done_mask = current_index.ge(length);
            let bounced_mask = done_mask & !current_bounce & bounce_enabled_mask;
            let loop_over_mask =
                done_mask & (current_bounce | !bounce_enabled_mask) & loop_enabled_mask;
            current_bounce = (bounced_mask | current_bounce) & !loop_over_mask;

            current_index = (bounced_mask | loop_over_mask)
                .select(current_index - wrap_amount, current_index);
            current_index = current_index.min(length);
            current_fraction = current_fraction & !done_mask;
        }

        self.bounce_mask = current_bounce;
        if reset_mask.any() {
            for (i, out) in raw_out.iter_mut().take(num_samples).enumerate() {
                let zero_mask =
                    u32_lt_signed(PolyU32::splat(i as u32), trigger_offset) & reset_mask;
                *out = *out & !zero_mask;
            }
        }

        for (out, &raw) in leveled_out.iter_mut().zip(raw_out.iter()).take(num_samples) {
            current_pan_amplitude += delta_pan_amplitude;
            let level = params.level.clamp(0.0, MAX_SAMPLE_AMPLITUDE);
            *out = current_pan_amplitude * level * level * raw;
        }

        self.sample_index = current_index;
        self.sample_fraction = current_fraction;
        let phase = self
            .bounce_mask
            .select(length - self.sample_index, self.sample_index);
        self.playback_phase = phase * (1.0 / audio_length as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramp_plays_back_at_unity_pitch() {
        let length = 1000;
        let ramp: Vec<f32> = (0..length).map(|i| i as f32 / length as f32).collect();
        let mut source = SampleSource::new();
        source.set_sample_rate(44100.0);
        source.sample_mut().load_sample(&ramp, 44100);

        source.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let params = SampleSourceParams::default();

        const BLOCK: usize = 128;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];
        let mut output = Vec::new();
        for _ in 0..7 {
            source.process(&params, BLOCK, &mut raw, &mut leveled);
            for value in &raw {
                output.push(value.lane(0));
            }
        }

        // Playback runs at unity with a 3-sample interpolation latency.
        let checked = output.len().min(length - 10);
        for i in 10..checked {
            let expected = ramp[i - 3];
            assert!(
                (output[i] - expected).abs() < 1e-4,
                "sample {i}: got {} expected {expected}",
                output[i]
            );
        }
    }

    #[test]
    fn mip_selection_tracks_pitch() {
        let sample = Sample::default();
        // Unity playback (delta = 2 with the upsampled tiers) reads the
        // original-rate buffer; faster playback picks smaller tiers.
        assert_eq!(sample.active_index(2.0), 1);
        assert!(sample.active_index(4.0) > sample.active_index(2.0));
        assert_eq!(sample.active_index(1.0), 0);
    }

    #[test]
    fn tier_sizes_halve() {
        let length = 1000;
        let ramp: Vec<f32> = (0..length).map(|i| i as f32).collect();
        let mut sample = Sample::default();
        sample.load_sample(&ramp, 44100);
        assert_eq!(sample.left_buffer(0).len(), 2 * length + 2 * BUFFER_SAMPLES);
        assert_eq!(sample.left_buffer(1).len(), length + 2 * BUFFER_SAMPLES);
        assert_eq!(sample.left_buffer(2).len(), length / 2 + 2 * BUFFER_SAMPLES);
    }

    /// Builds a WAV byte blob around raw interleaved sample data.
    fn wav_bytes(format: u16, channels: u16, bits: u16, sample_rate: u32, data: &[u8]) -> Vec<u8> {
        let byte_rate = sample_rate * channels as u32 * (bits / 8) as u32;
        let block_align = channels * (bits / 8);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&format.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    fn original_frames(sample: &Sample, right: bool) -> &[f32] {
        sample.original_channel(right)
    }

    #[test]
    fn wav_pcm16_roundtrips_mono_and_stereo() {
        let left: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.31).sin() * 0.8).collect();
        let right: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.17).cos() * 0.6).collect();

        let encode = |v: f32| ((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes();
        let mono_data: Vec<u8> = left.iter().flat_map(|&v| encode(v)).collect();
        let mono = Sample::from_wav_bytes(&wav_bytes(1, 1, 16, 22050, &mono_data)).unwrap();
        assert!(!mono.stereo());
        assert_eq!(mono.sample_rate(), 22050);
        assert_eq!(mono.original_length(), 64);
        for (got, expected) in original_frames(&mono, false).iter().zip(&left) {
            assert!((got - expected).abs() < 1.0 / 32000.0);
        }

        let stereo_data: Vec<u8> = left
            .iter()
            .zip(&right)
            .flat_map(|(&l, &r)| {
                let mut frame = encode(l).to_vec();
                frame.extend(encode(r));
                frame
            })
            .collect();
        let stereo = Sample::from_wav_bytes(&wav_bytes(1, 2, 16, 44100, &stereo_data)).unwrap();
        assert!(stereo.stereo());
        assert_eq!(stereo.original_length(), 64);
        for (got, expected) in original_frames(&stereo, true).iter().zip(&right) {
            assert!((got - expected).abs() < 1.0 / 32000.0);
        }
    }

    #[test]
    fn wav_float32_roundtrips_mono_and_stereo() {
        let left: Vec<f32> = (0..48).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let right: Vec<f32> = (0..48).map(|i| 0.3 - (i as f32) * 0.005).collect();

        let mono_data: Vec<u8> = left.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mono = Sample::from_wav_bytes(&wav_bytes(3, 1, 32, 48000, &mono_data)).unwrap();
        assert_eq!(mono.sample_rate(), 48000);
        assert_eq!(original_frames(&mono, false), left.as_slice());

        let stereo_data: Vec<u8> = left
            .iter()
            .zip(&right)
            .flat_map(|(l, r)| {
                let mut frame = l.to_le_bytes().to_vec();
                frame.extend(r.to_le_bytes());
                frame
            })
            .collect();
        let stereo = Sample::from_wav_bytes(&wav_bytes(3, 2, 32, 48000, &stereo_data)).unwrap();
        assert!(stereo.stereo());
        assert_eq!(original_frames(&stereo, false), left.as_slice());
        assert_eq!(original_frames(&stereo, true), right.as_slice());
    }

    #[test]
    fn wav_rejects_garbage_and_unsupported() {
        assert!(Sample::from_wav_bytes(b"not a wav").is_err());
        let data = [0u8; 8];
        // 8-bit PCM is not supported by the minimal parser.
        assert!(Sample::from_wav_bytes(&wav_bytes(1, 1, 8, 44100, &data)).is_err());
    }

    #[test]
    fn loop_detection_finds_low_error_seam() {
        let sample_rate = 44100;
        let frequency = 220.0f32;
        let buffer: Vec<f32> = (0..sample_rate as usize)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Short attack so the sustain region is the natural loop home.
                let envelope = (i as f32 / 2000.0).min(1.0);
                envelope * (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.8
            })
            .collect();
        let sample = Sample::from_mono("tone", &buffer, sample_rate);

        let (start, end) = sample.detect_loop().expect("sustained tone should loop");
        assert!(end > start);
        assert!(end - start >= sample_rate as usize / 20, "loop shorter than 50 ms");
        assert!(end < buffer.len());

        // Crossfade error at the seam: the windows around start and end match.
        let window = 256.min(buffer.len() - end);
        let error: f32 = (0..window)
            .map(|i| (buffer[start + i] - buffer[end + i]).powi(2))
            .sum::<f32>()
            / window as f32;
        assert!(error < 1e-3, "seam error too high: {error}");
    }

    #[test]
    fn slice_detection_finds_four_hits() {
        let sample_rate = 44100u32;
        let hit_positions = [0usize, 11025, 22050, 33075];
        let mut buffer = vec![0.0f32; sample_rate as usize];
        for &position in &hit_positions {
            for i in 0..1500 {
                let decay = (-(i as f32) / 300.0).exp();
                buffer[position + i] = decay * ((i as f32) * 0.9).sin() * 0.8;
            }
        }
        let sample = Sample::from_mono("hits", &buffer, sample_rate);

        let markers = sample.detect_slices(0.5);
        assert_eq!(markers.len(), 4, "markers: {markers:?}");
        for (marker, position) in markers.iter().zip(&hit_positions) {
            let distance = marker.abs_diff(*position);
            assert!(distance <= 512, "marker {marker} far from onset {position}");
        }
    }

    #[test]
    fn slice_playback_starts_at_marker() {
        let length = 1000;
        let ramp: Vec<f32> = (0..length).map(|i| i as f32 / length as f32).collect();
        let mut source = SampleSource::with_sample(Sample::from_mono("ramp", &ramp, 44100));
        source.set_sample_rate(44100.0);
        source.sample_mut().set_slices(vec![0, 250, 500, 750]);

        source.note_on(PolyMask::all_on(), PolyU32::ZERO);
        let params = SampleSourceParams { slice: Some(2), ..Default::default() };

        const BLOCK: usize = 128;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];
        source.process(&params, BLOCK, &mut raw, &mut leveled);

        // Unity playback from frame 500 (3-sample interpolation latency).
        for i in 10..BLOCK {
            let expected = ramp[500 + i - 3];
            assert!(
                (raw[i].lane(0) - expected).abs() < 1e-4,
                "sample {i}: got {} expected {expected}",
                raw[i].lane(0)
            );
        }
    }

    #[test]
    fn rate_two_doubles_pitch_and_halves_duration() {
        let length = 1000;
        let ramp: Vec<f32> = (0..length).map(|i| i as f32 / length as f32).collect();

        // Returns the raw output and playback phase after each block.
        let render = |rate: f32| -> (Vec<f32>, Vec<f32>) {
            let mut source = SampleSource::with_sample(Sample::from_mono("ramp", &ramp, 44100));
            source.set_sample_rate(44100.0);
            source.note_on(PolyMask::all_on(), PolyU32::ZERO);
            let params = SampleSourceParams { rate, ..Default::default() };
            const BLOCK: usize = 128;
            let mut raw = [PolyF32::ZERO; BLOCK];
            let mut leveled = [PolyF32::ZERO; BLOCK];
            let mut output = Vec::new();
            let mut phases = Vec::new();
            for _ in 0..6 {
                source.process(&params, BLOCK, &mut raw, &mut leveled);
                output.extend(raw.iter().map(|v| v.lane(0)));
                phases.push(source.playback_phase().lane(0));
            }
            (output, phases)
        };

        let (fast, fast_phases) = render(2.0);
        let (_unity, unity_phases) = render(1.0);

        // Doubled pitch: the ramp slope doubles.
        let unity_slope = 1.0 / length as f32;
        for i in 50..400 {
            let slope = fast[i + 1] - fast[i];
            assert!(
                (slope - 2.0 * unity_slope).abs() < 5e-4,
                "sample {i}: slope {slope} expected {}",
                2.0 * unity_slope
            );
        }

        // Halved duration: the 1000-frame sample is exhausted after ~500
        // outputs at rate 2 while unity playback is only half way.
        assert!((fast_phases[2] - 0.768).abs() < 0.02, "rate 2 phase: {}", fast_phases[2]);
        assert!(fast_phases[3] > 0.999, "rate 2 not finished: {}", fast_phases[3]);
        assert!((unity_phases[3] - 0.512).abs() < 0.02, "unity phase: {}", unity_phases[3]);
        assert!(unity_phases[5] < 0.8, "unity finished early: {}", unity_phases[5]);
    }

    #[test]
    fn custom_loop_region_wraps_inside_bounds() {
        let length = 1000;
        let ramp: Vec<f32> = (0..length).map(|i| i as f32 / length as f32).collect();
        let mut source = SampleSource::with_sample(Sample::from_mono("ramp", &ramp, 44100));
        source.set_sample_rate(44100.0);
        source.note_on(PolyMask::all_on(), PolyU32::ZERO);

        let params = SampleSourceParams {
            loop_sample: true,
            loop_start: Some(200),
            loop_end: Some(400),
            ..Default::default()
        };

        const BLOCK: usize = 128;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];
        let mut output = Vec::new();
        for _ in 0..8 {
            source.process(&params, BLOCK, &mut raw, &mut leveled);
            output.extend(raw.iter().map(|v| v.lane(0)));
        }

        // After entering the region, playback stays within [0.2, 0.4].
        for (i, &value) in output.iter().enumerate().skip(410) {
            assert!(
                (0.19..=0.41).contains(&value),
                "sample {i} escaped the loop region: {value}"
            );
        }
        // And it keeps moving (looping, not frozen at the clamp).
        assert!((output[900] - output[901]).abs() > 1e-6 || output[900] != output[950]);
    }
}
