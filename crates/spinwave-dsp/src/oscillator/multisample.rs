//! SFZ multisample: zones of [`Sample`]s mapped by key/velocity range,
//! played through per-zone [`SampleSource`] voices.
//!
//! [`Multisample::from_sfz`] parses the SFZ subset that matters for zone
//! mapping (`<group>`/`<region>` headers; sample, lokey/hikey/key,
//! pitch_keycenter, lovel/hivel, loop_start/loop_end/loop_mode, tune,
//! volume, offset), resolving sample paths through a caller-supplied
//! callback so file IO stays out of the DSP crate. [`MultisampleSource`]
//! then plays notes with the same voice semantics as [`SampleSource`],
//! selecting the zone per note-on and pitching playback relative to the
//! zone's keycenter.

use std::sync::Arc;

use spinwave_poly::{constants, PolyF32, PolyMask, PolyU32};

use super::sample_source::{Sample, SampleSource, SampleSourceParams};

/// SFZ `loop_mode` values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SfzLoopMode {
    #[default]
    NoLoop,
    OneShot,
    LoopContinuous,
    LoopSustain,
}

impl SfzLoopMode {
    fn loops(self) -> bool {
        matches!(self, SfzLoopMode::LoopContinuous | SfzLoopMode::LoopSustain)
    }
}

/// One SFZ region resolved into a playable zone.
#[derive(Clone)]
pub struct MultisampleZone {
    /// The `sample` opcode value, as written in the SFZ text.
    pub sample_path: String,
    pub lokey: u8,
    pub hikey: u8,
    pub pitch_keycenter: u8,
    pub lovel: u8,
    pub hivel: u8,
    pub loop_start: Option<usize>,
    pub loop_end: Option<usize>,
    pub loop_mode: SfzLoopMode,
    /// Tuning in cents.
    pub tune: f32,
    /// Volume in dB.
    pub volume: f32,
    /// Playback start offset in sample frames.
    pub offset: usize,
    /// Shared material: cloning a zone (or a whole `Multisample`, one per
    /// voice kernel) only bumps refcounts.
    pub sample: Arc<Sample>,
}

impl MultisampleZone {
    #[inline]
    pub fn in_range(&self, note: u8, velocity: u8) -> bool {
        (self.lokey..=self.hikey).contains(&note) && (self.lovel..=self.hivel).contains(&velocity)
    }
}

/// A parsed SFZ instrument: zones plus non-fatal load warnings. `Clone` is
/// cheap (zone samples are shared), so one parse serves every kernel.
#[derive(Clone)]
pub struct Multisample {
    pub zones: Vec<MultisampleZone>,
    /// Zones skipped during parsing (missing samples, bad ranges) land here
    /// instead of failing the whole instrument.
    pub warnings: Vec<String>,
}

impl Multisample {
    /// Parses SFZ text; `load_sample` resolves each `sample` opcode value to
    /// audio (return `None` for a missing file — the zone is skipped with a
    /// warning, never failing the instrument).
    pub fn from_sfz(
        sfz_text: &str,
        mut load_sample: impl FnMut(&str) -> Option<Sample>,
    ) -> Result<Multisample, String> {
        let tokens = tokenize_sfz(sfz_text);
        if !tokens.iter().any(|t| matches!(t, SfzToken::Header(h) if h == "region")) {
            return Err("no <region> header found".to_string());
        }

        let mut group = ZoneSpec::default();
        let mut region: Option<ZoneSpec> = None;
        let mut target = Target::None;
        let mut zones = Vec::new();
        let mut warnings = Vec::new();

        let mut finalize = |region: &mut Option<ZoneSpec>,
                            zones: &mut Vec<MultisampleZone>,
                            warnings: &mut Vec<String>| {
            if let Some(spec) = region.take() {
                spec.finalize(&mut load_sample, zones, warnings);
            }
        };

        for token in tokens {
            match token {
                SfzToken::Header(header) => {
                    finalize(&mut region, &mut zones, &mut warnings);
                    match header.as_str() {
                        "group" => {
                            group = ZoneSpec::default();
                            target = Target::Group;
                        }
                        "region" => {
                            region = Some(group.clone());
                            target = Target::Region;
                        }
                        // <control>, <global>, <curve>... are out of scope:
                        // ignore their opcodes rather than misfiling them.
                        _ => target = Target::None,
                    }
                }
                SfzToken::Opcode(name, value) => {
                    let spec = match target {
                        Target::Group => Some(&mut group),
                        Target::Region => region.as_mut(),
                        Target::None => None,
                    };
                    if let Some(spec) = spec {
                        spec.apply(&name, value.trim(), &mut warnings);
                    }
                }
            }
        }
        finalize(&mut region, &mut zones, &mut warnings);

        Ok(Multisample { zones, warnings })
    }

    /// The zone for a note: key + velocity range match, nearest
    /// `pitch_keycenter` when ranges overlap.
    pub fn select_zone(&self, note: u8, velocity: u8) -> Option<usize> {
        select_zone_impl(self.zones.iter().map(|z| (z.lokey, z.hikey, z.lovel, z.hivel, z.pitch_keycenter)), note, velocity)
    }
}

fn select_zone_impl(
    zones: impl Iterator<Item = (u8, u8, u8, u8, u8)>,
    note: u8,
    velocity: u8,
) -> Option<usize> {
    let mut best: Option<(usize, u8)> = None;
    for (index, (lokey, hikey, lovel, hivel, keycenter)) in zones.enumerate() {
        if !(lokey..=hikey).contains(&note) || !(lovel..=hivel).contains(&velocity) {
            continue;
        }
        let distance = note.abs_diff(keycenter);
        if best.is_none_or(|(_, best_distance)| distance < best_distance) {
            best = Some((index, distance));
        }
    }
    best.map(|(index, _)| index)
}

enum Target {
    None,
    Group,
    Region,
}

enum SfzToken {
    Header(String),
    Opcode(String, String),
}

/// Splits SFZ text into headers and `name=value` opcodes. Values may contain
/// spaces (sample paths): words without `=` extend the previous value.
fn tokenize_sfz(text: &str) -> Vec<SfzToken> {
    let mut tokens = Vec::new();
    for line in text.lines() {
        let line = line.split("//").next().unwrap_or("");
        for word in line.split_whitespace() {
            let mut rest = word;
            while !rest.is_empty() {
                if let Some(stripped) = rest.strip_prefix('<') {
                    // Headers may be glued to what follows: `<region>sample=...`.
                    let Some(close) = stripped.find('>') else { break };
                    tokens.push(SfzToken::Header(stripped[..close].to_ascii_lowercase()));
                    rest = &stripped[close + 1..];
                } else if let Some(eq) = rest.find('=') {
                    tokens.push(SfzToken::Opcode(
                        rest[..eq].to_ascii_lowercase(),
                        rest[eq + 1..].to_string(),
                    ));
                    break;
                } else {
                    // Continuation of the previous value (path with spaces).
                    if let Some(SfzToken::Opcode(_, value)) = tokens.last_mut() {
                        value.push(' ');
                        value.push_str(rest);
                    }
                    break;
                }
            }
        }
    }
    tokens
}

/// Parses a key value: a MIDI number or a note name like `c4` / `c#4` / `db3`
/// (`c-1` = 0, so `c4` = 60).
fn parse_key(value: &str) -> Option<u8> {
    if let Ok(number) = value.parse::<i32>() {
        return u8::try_from(number).ok().filter(|&n| n <= 127);
    }
    let lower = value.to_ascii_lowercase();
    let mut chars = lower.chars();
    let letter = chars.next()?;
    let semitone = match letter {
        'c' => 0,
        'd' => 2,
        'e' => 4,
        'f' => 5,
        'g' => 7,
        'a' => 9,
        'b' => 11,
        _ => return None,
    };
    let rest = chars.as_str();
    let (accidental, octave_text) = match rest.chars().next()? {
        '#' => (1, &rest[1..]),
        'b' => (-1, &rest[1..]),
        _ => (0, rest),
    };
    let octave: i32 = octave_text.parse().ok()?;
    let midi = (octave + 1) * 12 + semitone + accidental;
    u8::try_from(midi).ok().filter(|&n| n <= 127)
}

/// Opcode accumulator: `None` fields inherit from the enclosing group.
#[derive(Clone, Default)]
struct ZoneSpec {
    sample: Option<String>,
    key: Option<u8>,
    lokey: Option<u8>,
    hikey: Option<u8>,
    pitch_keycenter: Option<u8>,
    lovel: Option<u8>,
    hivel: Option<u8>,
    loop_start: Option<usize>,
    loop_end: Option<usize>,
    loop_mode: Option<SfzLoopMode>,
    tune: Option<f32>,
    volume: Option<f32>,
    offset: Option<usize>,
}

impl ZoneSpec {
    fn apply(&mut self, name: &str, value: &str, warnings: &mut Vec<String>) {
        let mut warn = |what: &str| warnings.push(format!("bad {name} value '{value}': {what}"));
        match name {
            "sample" => self.sample = Some(value.to_string()),
            "key" => match parse_key(value) {
                Some(key) => self.key = Some(key),
                None => warn("not a note"),
            },
            "lokey" | "hikey" | "pitch_keycenter" => match parse_key(value) {
                Some(key) => {
                    *match name {
                        "lokey" => &mut self.lokey,
                        "hikey" => &mut self.hikey,
                        _ => &mut self.pitch_keycenter,
                    } = Some(key);
                }
                None => warn("not a note"),
            },
            "lovel" | "hivel" => match value.parse::<u8>().ok().filter(|&v| v <= 127) {
                Some(velocity) => {
                    *if name == "lovel" { &mut self.lovel } else { &mut self.hivel } =
                        Some(velocity);
                }
                None => warn("not a velocity"),
            },
            "loop_start" | "loopstart" => match value.parse::<usize>() {
                Ok(frame) => self.loop_start = Some(frame),
                Err(_) => warn("not a frame index"),
            },
            "loop_end" | "loopend" => match value.parse::<usize>() {
                Ok(frame) => self.loop_end = Some(frame),
                Err(_) => warn("not a frame index"),
            },
            "loop_mode" | "loopmode" => match value {
                "no_loop" => self.loop_mode = Some(SfzLoopMode::NoLoop),
                "one_shot" => self.loop_mode = Some(SfzLoopMode::OneShot),
                "loop_continuous" => self.loop_mode = Some(SfzLoopMode::LoopContinuous),
                "loop_sustain" => self.loop_mode = Some(SfzLoopMode::LoopSustain),
                _ => warn("unknown loop mode"),
            },
            "tune" => match value.parse::<f32>() {
                Ok(cents) => self.tune = Some(cents),
                Err(_) => warn("not a number"),
            },
            "volume" => match value.parse::<f32>() {
                Ok(db) => self.volume = Some(db),
                Err(_) => warn("not a number"),
            },
            "offset" => match value.parse::<usize>() {
                Ok(frames) => self.offset = Some(frames),
                Err(_) => warn("not a frame index"),
            },
            // Unknown opcodes are ignored (the SFZ spec is vast).
            _ => {}
        }
    }

    fn finalize(
        self,
        load_sample: &mut impl FnMut(&str) -> Option<Sample>,
        zones: &mut Vec<MultisampleZone>,
        warnings: &mut Vec<String>,
    ) {
        let Some(path) = self.sample else {
            warnings.push("region without sample opcode skipped".to_string());
            return;
        };
        let Some(sample) = load_sample(&path) else {
            warnings.push(format!("sample '{path}' not found; zone skipped"));
            return;
        };

        // `key` sets range and keycenter at once, explicit opcodes win.
        let lokey = self.lokey.or(self.key).unwrap_or(0);
        let hikey = self.hikey.or(self.key).unwrap_or(127);
        let pitch_keycenter = self.pitch_keycenter.or(self.key).unwrap_or(60);
        let lovel = self.lovel.unwrap_or(0);
        let hivel = self.hivel.unwrap_or(127);
        if lokey > hikey || lovel > hivel {
            warnings.push(format!("zone '{path}' has an empty key/velocity range; skipped"));
            return;
        }

        zones.push(MultisampleZone {
            sample_path: path,
            lokey,
            hikey,
            pitch_keycenter,
            lovel,
            hivel,
            loop_start: self.loop_start,
            loop_end: self.loop_end,
            loop_mode: self.loop_mode.unwrap_or_default(),
            tune: self.tune.unwrap_or(0.0),
            volume: self.volume.unwrap_or(0.0),
            offset: self.offset.unwrap_or(0),
            sample: Arc::new(sample),
        });
    }
}

struct ZoneVoice {
    lokey: u8,
    hikey: u8,
    pitch_keycenter: u8,
    lovel: u8,
    hivel: u8,
    loops: bool,
    loop_start: Option<usize>,
    loop_end: Option<usize>,
    /// Zone tune converted to semitones.
    tune_semitones: f32,
    /// Zone volume as a level multiplier (`level` is squared downstream, so
    /// this is the square root of the dB gain).
    level_multiplier: f32,
    offset: usize,
    source: SampleSource,
    /// Lanes currently assigned to this zone.
    active: PolyMask,
}

/// Plays a [`Multisample`] with [`SampleSource`] voice semantics: `note_on`
/// selects the zone per lane mask, `process` renders every sounding zone
/// pitched relative to its `pitch_keycenter` with tune/volume/loop applied.
///
/// Callers drive pitch through `params.midi` exactly like a keytracked
/// [`SampleSource`]; `keytrack`, `loop_sample` and the loop/slice/offset
/// overrides in the incoming params are replaced by each zone's settings.
pub struct MultisampleSource {
    zones: Vec<ZoneVoice>,
    /// Parse/load warnings carried over from the [`Multisample`].
    pub warnings: Vec<String>,
}

impl MultisampleSource {
    /// How many zones this instrument mapped.
    pub fn zone_count(&self) -> usize {
        self.zones.len()
    }

    pub fn new(multisample: Multisample) -> MultisampleSource {
        let zones = multisample
            .zones
            .into_iter()
            .map(|zone| ZoneVoice {
                lokey: zone.lokey,
                hikey: zone.hikey,
                pitch_keycenter: zone.pitch_keycenter,
                lovel: zone.lovel,
                hivel: zone.hivel,
                loops: zone.loop_mode.loops(),
                loop_start: zone.loop_start,
                // SFZ `loop_end` names the last looped frame (inclusive);
                // `SampleSourceParams::loop_end` is exclusive.
                loop_end: zone.loop_end.map(|end| end + 1),
                tune_semitones: zone.tune / 100.0,
                level_multiplier: 10.0f32.powf(zone.volume / 40.0),
                offset: zone.offset,
                source: SampleSource::with_sample(zone.sample.clone()),
                active: PolyMask::NONE,
            })
            .collect();
        MultisampleSource { zones, warnings: multisample.warnings }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        for zone in &mut self.zones {
            zone.source.set_sample_rate(sample_rate);
        }
    }

    #[inline]
    pub fn num_zones(&self) -> usize {
        self.zones.len()
    }

    /// Schedules a note-on: picks the zone for `midi_note`/`velocity`
    /// (key + velocity range, nearest keycenter on overlap) and hands the
    /// lanes in `mask` to it. Without a matching zone the note is silent.
    pub fn note_on(&mut self, mask: PolyMask, sample_offset: PolyU32, midi_note: u8, velocity: u8) {
        let selected = select_zone_impl(
            self.zones.iter().map(|z| (z.lokey, z.hikey, z.lovel, z.hivel, z.pitch_keycenter)),
            midi_note,
            velocity,
        );
        let Some(selected) = selected else {
            return;
        };
        for (index, zone) in self.zones.iter_mut().enumerate() {
            if index == selected {
                zone.active |= mask;
                zone.source.note_on(mask, sample_offset);
            } else {
                zone.active &= !mask;
            }
        }
    }

    /// Renders all sounding zones into `raw_out`/`leveled_out` (overwritten).
    /// Pitch follows `params.midi` relative to each zone's keycenter.
    pub fn process(
        &mut self,
        params: &SampleSourceParams,
        num_samples: usize,
        raw_out: &mut [PolyF32],
        leveled_out: &mut [PolyF32],
    ) {
        assert!(num_samples <= constants::MAX_BUFFER_SIZE);
        assert!(raw_out.len() >= num_samples && leveled_out.len() >= num_samples);
        raw_out[..num_samples].fill(PolyF32::ZERO);
        leveled_out[..num_samples].fill(PolyF32::ZERO);

        let mut zone_raw = [PolyF32::ZERO; constants::MAX_BUFFER_SIZE];
        let mut zone_leveled = [PolyF32::ZERO; constants::MAX_BUFFER_SIZE];
        for zone in &mut self.zones {
            if !zone.active.any() {
                continue;
            }

            let mut zone_params = params.clone();
            zone_params.keytrack = true;
            // midi - 60 (keytrack) + (60 - keycenter) + cents = midi - keycenter.
            zone_params.tune = params.tune
                + PolyF32::splat(
                    (constants::MIDI_TRACK_CENTER - zone.pitch_keycenter as i32) as f32
                        + zone.tune_semitones,
                );
            zone_params.level = params.level * zone.level_multiplier;
            zone_params.loop_sample = zone.loops;
            zone_params.loop_start = zone.loop_start;
            zone_params.loop_end = zone.loop_end;
            zone_params.slice = None;
            zone_params.start_offset = zone.offset;

            zone.source.process(&zone_params, num_samples, &mut zone_raw, &mut zone_leveled);
            for (out, &value) in raw_out[..num_samples].iter_mut().zip(&zone_raw) {
                *out += value & zone.active;
            }
            for (out, &value) in leveled_out[..num_samples].iter_mut().zip(&zone_leveled) {
                *out += value & zone.active;
            }

            // Release lanes whose one-shot playback ran off the end.
            if !zone.loops {
                let done = zone.source.playback_phase().ge(PolyF32::splat(1.0 - 1e-6));
                zone.active &= !done;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn short_sample(name: &str) -> Sample {
        let buffer: Vec<f32> = (0..64).map(|i| (i as f32 * 0.2).sin()).collect();
        Sample::from_mono(name, &buffer, 44100)
    }

    const TEST_SFZ: &str = "
// two-layer test kit
<group> lovel=0 hivel=63 volume=-6 tune=25
<region> sample=kick_soft.wav lokey=c1 hikey=b1 pitch_keycenter=c1
<region> sample=snare_soft.wav lokey=36 hikey=47 pitch_keycenter=42 \
 loop_mode=loop_continuous loop_start=100 loop_end=900
<group> lovel=64 hivel=127
<region> sample=kick_hard.wav key=c1 offset=50
<region> sample=snare hard.wav lokey=c#2 hikey=g2 pitch_keycenter=e2 tune=-10
";

    fn load_test_multisample() -> Multisample {
        Multisample::from_sfz(TEST_SFZ, |path| Some(short_sample(path))).unwrap()
    }

    #[test]
    fn sfz_parses_groups_regions_and_inheritance() {
        let multisample = load_test_multisample();
        assert!(multisample.warnings.is_empty(), "{:?}", multisample.warnings);
        assert_eq!(multisample.zones.len(), 4);

        // Group 1 opcodes inherit into its regions.
        let kick_soft = &multisample.zones[0];
        assert_eq!(kick_soft.sample_path, "kick_soft.wav");
        assert_eq!((kick_soft.lokey, kick_soft.hikey), (24, 35)); // c1..b1
        assert_eq!(kick_soft.pitch_keycenter, 24);
        assert_eq!((kick_soft.lovel, kick_soft.hivel), (0, 63));
        assert_eq!(kick_soft.volume, -6.0);
        assert_eq!(kick_soft.tune, 25.0);
        assert_eq!(kick_soft.loop_mode, SfzLoopMode::NoLoop);

        // Numeric keys, loop opcodes.
        let snare_soft = &multisample.zones[1];
        assert_eq!((snare_soft.lokey, snare_soft.hikey), (36, 47));
        assert_eq!(snare_soft.pitch_keycenter, 42);
        assert_eq!(snare_soft.loop_mode, SfzLoopMode::LoopContinuous);
        assert_eq!(snare_soft.loop_start, Some(100));
        assert_eq!(snare_soft.loop_end, Some(900));
        assert_eq!(snare_soft.volume, -6.0); // inherited

        // Second group resets inheritance; `key` sets range + keycenter.
        let kick_hard = &multisample.zones[2];
        assert_eq!((kick_hard.lokey, kick_hard.hikey), (24, 24));
        assert_eq!(kick_hard.pitch_keycenter, 24);
        assert_eq!((kick_hard.lovel, kick_hard.hivel), (64, 127));
        assert_eq!(kick_hard.volume, 0.0);
        assert_eq!(kick_hard.offset, 50);

        // Sharp note names, spaces in sample paths, region tune override.
        let snare_hard = &multisample.zones[3];
        assert_eq!(snare_hard.sample_path, "snare hard.wav");
        assert_eq!((snare_hard.lokey, snare_hard.hikey), (37, 43)); // c#2..g2
        assert_eq!(snare_hard.pitch_keycenter, 40); // e2
        assert_eq!(snare_hard.tune, -10.0);
    }

    #[test]
    fn zone_selection_maps_key_and_velocity() {
        let multisample = load_test_multisample();
        assert_eq!(multisample.select_zone(30, 40), Some(0)); // soft kick range
        assert_eq!(multisample.select_zone(24, 100), Some(2)); // hard layer
        assert_eq!(multisample.select_zone(40, 30), Some(1)); // soft snare
        assert_eq!(multisample.select_zone(40, 100), Some(3)); // hard snare
        assert_eq!(multisample.select_zone(90, 100), None); // out of range
    }

    #[test]
    fn overlapping_zones_pick_nearest_keycenter() {
        let sfz = "
<region> sample=low.wav lokey=0 hikey=127 pitch_keycenter=40
<region> sample=high.wav lokey=0 hikey=127 pitch_keycenter=80
";
        let multisample = Multisample::from_sfz(sfz, |path| Some(short_sample(path))).unwrap();
        assert_eq!(multisample.select_zone(50, 100), Some(0));
        assert_eq!(multisample.select_zone(70, 100), Some(1));
    }

    #[test]
    fn missing_sample_zone_is_skipped_with_warning() {
        let multisample = Multisample::from_sfz(TEST_SFZ, |path| {
            (path != "kick_hard.wav").then(|| short_sample(path))
        })
        .unwrap();
        assert_eq!(multisample.zones.len(), 3);
        assert_eq!(multisample.warnings.len(), 1);
        assert!(multisample.warnings[0].contains("kick_hard.wav"));
        // The remaining zones are intact.
        assert_eq!(multisample.select_zone(40, 100), Some(2));
    }

    #[test]
    fn from_sfz_rejects_text_without_regions() {
        assert!(Multisample::from_sfz("<group> lokey=0", |_| None).is_err());
        assert!(Multisample::from_sfz("just some text", |_| None).is_err());
    }

    #[test]
    fn zone_playback_pitch_matches_keycenter_math() {
        let length = 1000usize;
        let ramp: Vec<f32> = (0..length).map(|i| i as f32 / length as f32).collect();
        let sfz = "<region> sample=ramp.wav lokey=0 hikey=127 pitch_keycenter=48";
        let build = || {
            let multisample =
                Multisample::from_sfz(sfz, |_| Some(Sample::from_mono("ramp", &ramp, 44100)))
                    .unwrap();
            let mut source = MultisampleSource::new(multisample);
            source.set_sample_rate(44100.0);
            source
        };

        const BLOCK: usize = 128;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];

        // Playing the keycenter renders the sample at unity pitch.
        let mut source = build();
        source.note_on(PolyMask::all_on(), PolyU32::ZERO, 48, 100);
        let params = SampleSourceParams { midi: PolyF32::splat(48.0), ..Default::default() };
        let mut output = Vec::new();
        for _ in 0..4 {
            source.process(&params, BLOCK, &mut raw, &mut leveled);
            output.extend(raw.iter().map(|v| v.lane(0)));
        }
        for (i, &value) in output.iter().enumerate().skip(10) {
            let expected = ramp[i - 3];
            assert!((value - expected).abs() < 1e-4, "sample {i}: {value} vs {expected}");
        }

        // One octave above the keycenter doubles the playback slope.
        let mut source = build();
        source.note_on(PolyMask::all_on(), PolyU32::ZERO, 60, 100);
        let params = SampleSourceParams { midi: PolyF32::splat(60.0), ..Default::default() };
        let mut output = Vec::new();
        for _ in 0..4 {
            source.process(&params, BLOCK, &mut raw, &mut leveled);
            output.extend(raw.iter().map(|v| v.lane(0)));
        }
        let unity_slope = 1.0 / length as f32;
        for i in 50..400 {
            let slope = output[i + 1] - output[i];
            assert!(
                (slope - 2.0 * unity_slope).abs() < 5e-4,
                "sample {i}: slope {slope} expected {}",
                2.0 * unity_slope
            );
        }
    }

    #[test]
    fn sfz_inclusive_loop_end_becomes_exclusive() {
        let multisample = load_test_multisample();
        assert_eq!(multisample.zones[1].loop_end, Some(900));
        let source = MultisampleSource::new(multisample);
        // Frame 900 is part of the loop in SFZ terms, so the exclusive end
        // handed to the sample source is 901.
        assert_eq!(source.zones[1].loop_start, Some(100));
        assert_eq!(source.zones[1].loop_end, Some(901));
        assert_eq!(source.zones[0].loop_end, None);
    }

    #[test]
    fn velocity_layers_route_to_different_zones() {
        // Distinguish layers by constant sample values.
        let sfz = "
<region> sample=soft lokey=0 hikey=127 pitch_keycenter=60 lovel=0 hivel=63
<region> sample=loud lokey=0 hikey=127 pitch_keycenter=60 lovel=64 hivel=127
";
        let multisample = Multisample::from_sfz(sfz, |path| {
            let value = if path == "soft" { 0.25 } else { 0.75 };
            Some(Sample::from_mono(path, &vec![value; 512], 44100))
        })
        .unwrap();
        let mut source = MultisampleSource::new(multisample);
        source.set_sample_rate(44100.0);

        const BLOCK: usize = 64;
        let mut raw = [PolyF32::ZERO; BLOCK];
        let mut leveled = [PolyF32::ZERO; BLOCK];
        let params = SampleSourceParams { midi: PolyF32::splat(60.0), ..Default::default() };

        source.note_on(PolyMask::all_on(), PolyU32::ZERO, 60, 30);
        source.process(&params, BLOCK, &mut raw, &mut leveled);
        assert!((raw[BLOCK - 1].lane(0) - 0.25).abs() < 1e-3, "soft layer expected");

        source.note_on(PolyMask::all_on(), PolyU32::ZERO, 60, 120);
        source.process(&params, BLOCK, &mut raw, &mut leveled);
        assert!((raw[BLOCK - 1].lane(0) - 0.75).abs() < 1e-3, "loud layer expected");
    }
}
