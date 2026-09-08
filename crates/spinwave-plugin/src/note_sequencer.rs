//! Note processor between incoming MIDI and the `SoundEngine`: an
//! arpeggiator and a step sequencer with a bpm-synced clock.
//!
//! In `Off` mode notes pass straight through (`note_on`/`note_off` return
//! the event to forward). In `Arp`/`StepSeq` mode incoming notes are
//! collected into a held set and `process` emits engine note events with
//! sample offsets via the `emit` callback, once per audio block.

use serde_json::Value;

/// What the sequencer does with incoming notes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SeqMode {
    /// Pure passthrough (default).
    Off,
    /// Arpeggiate the held notes.
    Arp,
    /// Play the step pattern relative to the lowest held note.
    StepSeq,
}

/// Note ordering for the arpeggiator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArpPattern {
    Up,
    Down,
    UpDown,
    /// Notes in the order they were played.
    Played,
    /// Deterministic pseudo-random walk (seeded).
    Random,
    /// All held notes at once, retriggered every step.
    Chord,
}

/// Step clock rate: free-running or tempo-synced.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Rate {
    /// Steps per second, independent of tempo.
    Hz(f32),
    /// Step length in beats (quarter notes): 1/8 note = 0.5 beats.
    Beats(f64),
}

/// One step of the step sequencer.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Step {
    pub on: bool,
    /// Semitones relative to the lowest held note.
    pub transpose_semitones: i32,
    pub velocity: f32,
    /// Hold the note into the next step (merged when the next step plays
    /// the same note; a rest or a different note releases it).
    pub tie: bool,
}

impl Default for Step {
    fn default() -> Step {
        Step { on: true, transpose_semitones: 0, velocity: 0.8, tie: false }
    }
}

/// Full sequencer configuration, applied atomically at a block boundary.
#[derive(Clone, PartialEq, Debug)]
pub struct SeqConfig {
    pub mode: SeqMode,
    pub pattern: ArpPattern,
    pub rate: Rate,
    /// Fraction of a step the note sounds, 0..1.
    pub gate: f32,
    /// Delay applied to every 2nd step, as a fraction of a step, 0..0.75.
    pub swing: f32,
    /// Keep notes in the held set after release; the next fresh press
    /// starts a new held set.
    pub latch: bool,
    /// Arp octave range, 1..4.
    pub octaves: u32,
    /// Step pattern (step mode), up to 32 entries.
    pub steps: Vec<Step>,
    /// Pattern length in steps, 1..32 (missing steps are rests).
    pub length: usize,
    /// Seed for the deterministic `Random` pattern.
    pub seed: u64,
}

impl Default for SeqConfig {
    fn default() -> SeqConfig {
        SeqConfig {
            mode: SeqMode::Off,
            pattern: ArpPattern::Up,
            rate: Rate::Beats(0.5), // 1/8 note
            gate: 0.8,
            swing: 0.0,
            latch: false,
            octaves: 1,
            steps: Vec::new(),
            length: 1,
            seed: 0x5eed,
        }
    }
}

/// Beats per step for a division string: `"1/4"`, `"1/8d"` (dotted),
/// `"1/8t"` (triplet). Denominators 1..=32.
fn division_beats(text: &str) -> Option<f64> {
    let (core, multiplier) = if let Some(core) = text.strip_suffix('d') {
        (core, 1.5)
    } else if let Some(core) = text.strip_suffix('t') {
        (core, 2.0 / 3.0)
    } else {
        (text, 1.0)
    };
    let denominator: u32 = core.strip_prefix("1/")?.parse().ok()?;
    if !(1..=32).contains(&denominator) {
        return None;
    }
    Some(4.0 / denominator as f64 * multiplier)
}

impl SeqConfig {
    /// Builds a config from the live-channel JSON (`{"mode":"arp",...}`).
    /// Missing fields take their defaults; out-of-range values are clamped.
    pub fn from_json(value: &Value) -> Result<SeqConfig, String> {
        let mode = match value["mode"].as_str().unwrap_or("off") {
            "off" => SeqMode::Off,
            "arp" => SeqMode::Arp,
            "step" | "stepseq" | "step_seq" => SeqMode::StepSeq,
            other => return Err(format!("unknown mode '{other}' (off|arp|step)")),
        };
        let mut config = SeqConfig { mode, ..SeqConfig::default() };
        if let Some(pattern) = value["pattern"].as_str() {
            config.pattern = match pattern {
                "up" => ArpPattern::Up,
                "down" => ArpPattern::Down,
                "updown" | "up_down" => ArpPattern::UpDown,
                "played" => ArpPattern::Played,
                "random" => ArpPattern::Random,
                "chord" => ArpPattern::Chord,
                other => {
                    return Err(format!(
                        "unknown pattern '{other}' (up|down|updown|played|random|chord)"
                    ))
                }
            };
        }
        match &value["rate"] {
            Value::Null => {}
            Value::String(text) => {
                let beats = division_beats(text).ok_or_else(|| {
                    format!("bad rate '{text}' (use 1/1..1/32 with optional d/t, or a Hz number)")
                })?;
                config.rate = Rate::Beats(beats);
            }
            number if number.is_number() => {
                let hz = number.as_f64().unwrap_or(2.0) as f32;
                config.rate = Rate::Hz(hz.clamp(0.01, 200.0));
            }
            _ => return Err("rate must be a division string or a Hz number".into()),
        }
        if let Some(gate) = value["gate"].as_f64() {
            config.gate = (gate as f32).clamp(0.05, 1.0);
        }
        if let Some(swing) = value["swing"].as_f64() {
            config.swing = (swing as f32).clamp(0.0, 0.75);
        }
        if let Some(latch) = value["latch"].as_bool() {
            config.latch = latch;
        }
        if let Some(octaves) = value["octaves"].as_u64() {
            config.octaves = (octaves as u32).clamp(1, 4);
        }
        if let Some(steps) = value["steps"].as_array() {
            if steps.len() > 32 {
                return Err("at most 32 steps".into());
            }
            config.steps = steps
                .iter()
                .map(|step| Step {
                    on: step["on"].as_bool().unwrap_or(true),
                    transpose_semitones: step["transpose"].as_i64().unwrap_or(0).clamp(-64, 64)
                        as i32,
                    velocity: step["velocity"].as_f64().unwrap_or(0.8).clamp(0.0, 1.0) as f32,
                    tie: step["tie"].as_bool().unwrap_or(false),
                })
                .collect();
        }
        config.length = value["length"]
            .as_u64()
            .map(|length| length as usize)
            .unwrap_or(config.steps.len().max(1))
            .clamp(1, 32);
        if let Some(seed) = value["seed"].as_u64() {
            config.seed = seed;
        }
        Ok(config)
    }
}

/// A note event for the engine, with an in-block sample offset.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NoteEventOut {
    /// true = note on, false = note off.
    pub on: bool,
    pub note: i32,
    pub velocity: f32,
    pub channel: usize,
    pub offset: usize,
}

#[derive(Clone, Copy)]
struct HeldNote {
    note: i32,
    velocity: f32,
    channel: usize,
}

struct ActiveNote {
    note: i32,
    channel: usize,
    /// Samples until the note off; `None` = held indefinitely (tie).
    off_in: Option<usize>,
}

/// The note processor: passthrough, arpeggiator or step sequencer.
pub struct NoteSequencer {
    config: SeqConfig,
    bpm: f32,
    /// Held (or latched) notes in played order.
    held: Vec<HeldNote>,
    /// Notes whose key is physically down right now.
    physical: Vec<i32>,
    /// Notes this sequencer has turned on and not yet turned off.
    active: Vec<ActiveNote>,
    running: bool,
    /// Samples until the next step fires (fractional carry preserved).
    until_next_step: f64,
    step_index: usize,
    rng: u64,
    /// Note currently sustained by a tie (step mode).
    tied: Option<i32>,
}

impl Default for NoteSequencer {
    fn default() -> NoteSequencer {
        NoteSequencer {
            config: SeqConfig::default(),
            bpm: 120.0,
            held: Vec::new(),
            physical: Vec::new(),
            active: Vec::new(),
            running: false,
            until_next_step: 0.0,
            step_index: 0,
            rng: 0x5eed,
            tied: None,
        }
    }
}

impl NoteSequencer {
    pub fn set_bpm(&mut self, bpm: f32) {
        self.bpm = bpm;
    }

    pub fn mode(&self) -> SeqMode {
        self.config.mode
    }

    /// Applies a new configuration. A mode change flushes everything
    /// sounding (note offs are emitted through `emit` at offset 0).
    pub fn set_config(&mut self, config: SeqConfig, mut emit: impl FnMut(NoteEventOut)) {
        if config.mode != self.config.mode {
            self.flush(&mut emit);
        }
        if !config.latch && self.config.latch {
            let physical = &self.physical;
            self.held.retain(|held| physical.contains(&held.note));
        }
        self.config = config;
        self.rng = self.config.seed;
        if self.config.mode == SeqMode::Off || self.held.is_empty() {
            self.stop_clock();
        } else if !self.running {
            self.running = true;
            self.until_next_step = 0.0;
            self.step_index = 0;
        }
    }

    /// Registers a pressed key. In `Off` mode returns the event to forward
    /// to the engine immediately; otherwise the note joins the held set.
    pub fn note_on(&mut self, note: i32, velocity: f32, channel: usize) -> Option<NoteEventOut> {
        // A fresh press after everything was released replaces a latched set.
        if self.config.latch && self.physical.is_empty() && !self.held.is_empty() {
            self.held.clear();
        }
        if !self.physical.contains(&note) {
            self.physical.push(note);
        }
        self.held.retain(|held| held.note != note);
        self.held.push(HeldNote { note, velocity, channel });

        if self.config.mode == SeqMode::Off {
            return Some(NoteEventOut { on: true, note, velocity, channel, offset: 0 });
        }
        if !self.running {
            self.running = true;
            self.until_next_step = 0.0;
            self.step_index = 0;
            self.rng = self.config.seed;
        }
        None
    }

    /// Registers a released key. In `Off` mode returns the event to
    /// forward; with latch on the note stays in the held set.
    pub fn note_off(&mut self, note: i32, velocity: f32, channel: usize) -> Option<NoteEventOut> {
        self.physical.retain(|&physical| physical != note);
        if !self.config.latch {
            self.held.retain(|held| held.note != note);
        }
        if self.config.mode == SeqMode::Off {
            return Some(NoteEventOut { on: false, note, velocity, channel, offset: 0 });
        }
        if self.held.is_empty() {
            self.stop_clock();
        }
        None
    }

    /// Emits note offs (offset 0) for everything sounding because of this
    /// sequencer — in `Off` mode that is the forwarded held notes.
    pub fn flush(&mut self, mut emit: impl FnMut(NoteEventOut)) {
        for active in self.active.drain(..) {
            emit(NoteEventOut {
                on: false,
                note: active.note,
                velocity: 0.5,
                channel: active.channel,
                offset: 0,
            });
        }
        if self.config.mode == SeqMode::Off {
            for held in &self.held {
                emit(NoteEventOut {
                    on: false,
                    note: held.note,
                    velocity: 0.5,
                    channel: held.channel,
                    offset: 0,
                });
            }
        }
        self.tied = None;
        self.running = false;
        self.step_index = 0;
    }

    /// Clears all state without emitting (pair with `engine.all_sounds_off`).
    pub fn reset(&mut self) {
        self.held.clear();
        self.physical.clear();
        self.active.clear();
        self.tied = None;
        self.running = false;
        self.step_index = 0;
        self.until_next_step = 0.0;
    }

    /// Advances the clock over one audio block, emitting engine note
    /// events with sample offsets in `0..num_samples`.
    pub fn process(
        &mut self,
        num_samples: usize,
        sample_rate: f32,
        mut emit: impl FnMut(NoteEventOut),
    ) {
        let mut t = 0usize;
        while t < num_samples {
            let remaining = num_samples - t;
            let step_due = if self.running && self.config.mode != SeqMode::Off {
                self.until_next_step.max(0.0) as usize
            } else {
                usize::MAX
            };
            let off_due =
                self.active.iter().filter_map(|active| active.off_in).min().unwrap_or(usize::MAX);
            let advance = step_due.min(off_due).min(remaining);
            if advance > 0 {
                for active in &mut self.active {
                    if let Some(off_in) = &mut active.off_in {
                        *off_in -= advance;
                    }
                }
                if self.running {
                    self.until_next_step -= advance as f64;
                }
                t += advance;
                continue;
            }

            // Something is due exactly at offset t: gate/tie note offs
            // first, then the step itself.
            let mut index = 0;
            while index < self.active.len() {
                if self.active[index].off_in == Some(0) {
                    let active = self.active.remove(index);
                    if self.tied == Some(active.note) {
                        self.tied = None;
                    }
                    emit(NoteEventOut {
                        on: false,
                        note: active.note,
                        velocity: 0.5,
                        channel: active.channel,
                        offset: t,
                    });
                } else {
                    index += 1;
                }
            }
            if step_due == 0 {
                self.fire_step(t, sample_rate, &mut emit);
            }
        }
    }

    /// Samples per (un-swung) step at the current bpm.
    fn step_period(&self, sample_rate: f32) -> f64 {
        match self.config.rate {
            Rate::Hz(hz) => sample_rate as f64 / hz.max(0.01) as f64,
            Rate::Beats(beats) => beats * 60.0 / self.bpm.max(1.0) as f64 * sample_rate as f64,
        }
    }

    fn stop_clock(&mut self) {
        self.running = false;
        self.step_index = 0;
        self.tied = None;
        // Tied notes would never end without a clock: release at the next
        // process call.
        for active in &mut self.active {
            if active.off_in.is_none() {
                active.off_in = Some(0);
            }
        }
    }

    fn next_rand(&mut self) -> u64 {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng >> 33
    }

    /// The arp note cycle for the current held set: pattern order across
    /// the octave range.
    fn arp_sequence(&self) -> Vec<(i32, f32, usize)> {
        let mut ordered: Vec<(i32, f32, usize)> =
            self.held.iter().map(|held| (held.note, held.velocity, held.channel)).collect();
        if ordered.is_empty() {
            return ordered;
        }
        if self.config.pattern != ArpPattern::Played {
            ordered.sort_by_key(|&(note, _, _)| note);
        }
        let octaves = self.config.octaves.clamp(1, 4) as i32;
        let mut sequence = Vec::with_capacity(ordered.len() * octaves as usize);
        for octave in 0..octaves {
            for &(note, velocity, channel) in &ordered {
                sequence.push((note + 12 * octave, velocity, channel));
            }
        }
        match self.config.pattern {
            ArpPattern::Down => sequence.reverse(),
            ArpPattern::UpDown if sequence.len() > 2 => {
                let middle: Vec<_> =
                    sequence[1..sequence.len() - 1].iter().rev().copied().collect();
                sequence.extend(middle);
            }
            _ => {}
        }
        sequence
    }

    fn fire_step(
        &mut self,
        offset: usize,
        sample_rate: f32,
        emit: &mut impl FnMut(NoteEventOut),
    ) {
        let period = self.step_period(sample_rate);
        let cur = self.step_index;
        self.step_index = cur.wrapping_add(1);
        // Swing: the interval INTO every 2nd step grows, the one out of it
        // shrinks, so odd steps are delayed and the grid stays in place.
        let swing = self.config.swing as f64;
        let interval = period * if cur.is_multiple_of(2) { 1.0 + swing } else { 1.0 - swing };
        self.until_next_step = interval.max(1.0);
        let gate_len = ((self.config.gate as f64 * period) as usize).max(1);

        match self.config.mode {
            SeqMode::Off => {}
            SeqMode::Arp => {
                let sequence = self.arp_sequence();
                if sequence.is_empty() {
                    self.stop_clock();
                    return;
                }
                if self.config.pattern == ArpPattern::Chord {
                    for (note, velocity, channel) in sequence {
                        self.start_note(note, velocity, channel, Some(gate_len), offset, emit);
                    }
                } else {
                    let index = if self.config.pattern == ArpPattern::Random {
                        self.next_rand() as usize % sequence.len()
                    } else {
                        cur % sequence.len()
                    };
                    let (note, velocity, channel) = sequence[index];
                    self.start_note(note, velocity, channel, Some(gate_len), offset, emit);
                }
            }
            SeqMode::StepSeq => self.fire_seq_step(cur, gate_len, offset, emit),
        }
    }

    fn fire_seq_step(
        &mut self,
        cur: usize,
        gate_len: usize,
        offset: usize,
        emit: &mut impl FnMut(NoteEventOut),
    ) {
        let length = self.config.length.clamp(1, 32);
        let step = self
            .config
            .steps
            .get(cur % length)
            .copied()
            .unwrap_or(Step { on: false, ..Step::default() });
        let base = self
            .held
            .iter()
            .min_by_key(|held| held.note)
            .map(|held| (held.note, held.channel));

        let (base_note, channel) = match (step.on, base) {
            (true, Some(base)) => base,
            _ => {
                // Rest (or nothing held): a tied note ends here.
                if let Some(tied) = self.tied.take() {
                    self.end_note(tied, offset, emit);
                }
                return;
            }
        };

        let note = base_note + step.transpose_semitones;
        if let Some(tied) = self.tied {
            if tied == note {
                // Tie continuation: extend the sounding note, no retrigger.
                if let Some(active) = self.active.iter_mut().find(|active| active.note == note) {
                    active.off_in = if step.tie { None } else { Some(gate_len) };
                }
                if !step.tie {
                    self.tied = None;
                }
                return;
            }
            self.tied = None;
            self.end_note(tied, offset, emit);
        }
        let gate = if step.tie { None } else { Some(gate_len) };
        self.start_note(note, step.velocity, channel, gate, offset, emit);
        if step.tie {
            self.tied = Some(note);
        }
    }

    fn start_note(
        &mut self,
        note: i32,
        velocity: f32,
        channel: usize,
        off_in: Option<usize>,
        offset: usize,
        emit: &mut impl FnMut(NoteEventOut),
    ) {
        // Retrigger: release the previous instance first.
        if let Some(position) = self
            .active
            .iter()
            .position(|active| active.note == note && active.channel == channel)
        {
            let previous = self.active.remove(position);
            emit(NoteEventOut {
                on: false,
                note: previous.note,
                velocity: 0.5,
                channel: previous.channel,
                offset,
            });
        }
        emit(NoteEventOut { on: true, note, velocity, channel, offset });
        self.active.push(ActiveNote { note, channel, off_in });
    }

    fn end_note(&mut self, note: i32, offset: usize, emit: &mut impl FnMut(NoteEventOut)) {
        if let Some(position) = self.active.iter().position(|active| active.note == note) {
            let active = self.active.remove(position);
            emit(NoteEventOut {
                on: false,
                note: active.note,
                velocity: 0.5,
                channel: active.channel,
                offset,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SR: f32 = 1000.0; // 1 kHz keeps the math readable: 120 bpm, 1/4 => 500 samples.

    /// Runs `process` in 128-sample blocks, returning (on, note, absolute time).
    fn run(seq: &mut NoteSequencer, total: usize) -> Vec<(bool, i32, usize)> {
        let mut out = Vec::new();
        let mut t = 0usize;
        while t < total {
            let block = 128.min(total - t);
            seq.process(block, SR, |event| out.push((event.on, event.note, t + event.offset)));
            t += block;
        }
        out
    }

    fn ons(events: &[(bool, i32, usize)]) -> Vec<(i32, usize)> {
        events.iter().filter(|e| e.0).map(|e| (e.1, e.2)).collect()
    }

    fn offs(events: &[(bool, i32, usize)]) -> Vec<(i32, usize)> {
        events.iter().filter(|e| !e.0).map(|e| (e.1, e.2)).collect()
    }

    fn arp(pattern: ArpPattern) -> SeqConfig {
        SeqConfig {
            mode: SeqMode::Arp,
            pattern,
            rate: Rate::Beats(1.0), // 1/4 note = 500 samples at 120 bpm / 1 kHz
            gate: 0.5,
            ..SeqConfig::default()
        }
    }

    fn sequencer(config: SeqConfig) -> NoteSequencer {
        let mut seq = NoteSequencer::default();
        seq.set_bpm(120.0);
        seq.set_config(config, |_| {});
        seq
    }

    fn hold_chord(seq: &mut NoteSequencer) {
        assert!(seq.note_on(60, 0.8, 0).is_none(), "arp mode consumes notes");
        assert!(seq.note_on(64, 0.8, 0).is_none());
        assert!(seq.note_on(67, 0.8, 0).is_none());
    }

    #[test]
    fn arp_up_plays_held_notes_in_order_at_rate() {
        let mut seq = sequencer(arp(ArpPattern::Up));
        hold_chord(&mut seq);
        let events = run(&mut seq, 1500);
        assert_eq!(ons(&events), vec![(60, 0), (64, 500), (67, 1000)]);
    }

    #[test]
    fn arp_down_reverses_order() {
        let mut seq = sequencer(arp(ArpPattern::Down));
        hold_chord(&mut seq);
        let events = run(&mut seq, 1500);
        assert_eq!(ons(&events), vec![(67, 0), (64, 500), (60, 1000)]);
    }

    #[test]
    fn arp_updown_bounces_without_repeating_endpoints() {
        let mut seq = sequencer(arp(ArpPattern::UpDown));
        hold_chord(&mut seq);
        let events = run(&mut seq, 3000);
        assert_eq!(
            ons(&events),
            vec![(60, 0), (64, 500), (67, 1000), (64, 1500), (60, 2000), (64, 2500)]
        );
    }

    #[test]
    fn arp_played_keeps_press_order() {
        let mut seq = sequencer(arp(ArpPattern::Played));
        seq.note_on(64, 0.8, 0);
        seq.note_on(60, 0.8, 0);
        seq.note_on(67, 0.8, 0);
        let events = run(&mut seq, 1500);
        assert_eq!(ons(&events), vec![(64, 0), (60, 500), (67, 1000)]);
    }

    #[test]
    fn arp_chord_retriggers_all_notes_each_step() {
        let mut seq = sequencer(arp(ArpPattern::Chord));
        hold_chord(&mut seq);
        let events = run(&mut seq, 600);
        assert_eq!(ons(&events), vec![(60, 0), (64, 0), (67, 0), (60, 500), (64, 500), (67, 500)]);
    }

    #[test]
    fn arp_octave_range_extends_the_cycle() {
        let mut config = arp(ArpPattern::Up);
        config.octaves = 2;
        let mut seq = sequencer(config);
        seq.note_on(60, 0.8, 0);
        let events = run(&mut seq, 1500);
        assert_eq!(ons(&events), vec![(60, 0), (72, 500), (60, 1000)]);
    }

    #[test]
    fn arp_random_is_deterministic_for_a_seed() {
        let make = || {
            let mut config = arp(ArpPattern::Random);
            config.seed = 42;
            let mut seq = sequencer(config);
            hold_chord(&mut seq);
            ons(&run(&mut seq, 4000))
        };
        let first = make();
        let second = make();
        assert_eq!(first, second);
        assert_eq!(first.len(), 8);
        assert!(first.iter().all(|&(note, _)| [60, 64, 67].contains(&note)));
    }

    #[test]
    fn gate_releases_before_the_next_step() {
        let mut seq = sequencer(arp(ArpPattern::Up)); // gate 0.5 of 500
        hold_chord(&mut seq);
        let events = run(&mut seq, 1500);
        assert_eq!(offs(&events), vec![(60, 250), (64, 750), (67, 1250)]);
    }

    #[test]
    fn swing_delays_every_second_step() {
        let mut config = arp(ArpPattern::Up);
        config.swing = 0.5;
        config.gate = 0.2;
        let mut seq = sequencer(config);
        hold_chord(&mut seq);
        let events = run(&mut seq, 2000);
        // Steps: 0, 500*(1+0.5)=750, 1000, 1750.
        assert_eq!(
            ons(&events).iter().map(|&(_, t)| t).collect::<Vec<_>>(),
            vec![0, 750, 1000, 1750]
        );
    }

    #[test]
    fn latch_holds_after_release_and_new_press_starts_over() {
        let mut config = arp(ArpPattern::Up);
        config.latch = true;
        let mut seq = sequencer(config);
        seq.note_on(60, 0.8, 0);
        run(&mut seq, 400);
        assert!(seq.note_off(60, 0.5, 0).is_none());
        let held = run(&mut seq, 1000);
        assert!(!ons(&held).is_empty(), "latch keeps the arp running after release");
        assert!(ons(&held).iter().all(|&(note, _)| note == 60));

        // A fresh press replaces the latched set.
        seq.note_on(62, 0.8, 0);
        let replaced = run(&mut seq, 1000);
        assert!(!ons(&replaced).is_empty());
        assert!(ons(&replaced).iter().all(|&(note, _)| note == 62));
    }

    #[test]
    fn step_seq_transposes_and_merges_ties() {
        let step = |on: bool, transpose: i32, tie: bool| Step {
            on,
            transpose_semitones: transpose,
            velocity: 0.9,
            tie,
        };
        let config = SeqConfig {
            mode: SeqMode::StepSeq,
            rate: Rate::Beats(1.0),
            gate: 0.5,
            steps: vec![
                step(true, 0, false),
                step(true, 12, false),
                step(false, 0, false), // rest
                step(true, 7, true),   // tied into...
                step(true, 7, false),  // ...the same note: one long note
            ],
            length: 5,
            ..SeqConfig::default()
        };
        let mut seq = sequencer(config);
        seq.note_on(60, 0.8, 0);
        let events = run(&mut seq, 2400);
        assert_eq!(ons(&events), vec![(60, 0), (72, 500), (67, 1500)]);
        // 60 and 72 gate off after 250; the tied 67 spans steps 3+4 and
        // gates off at 2000 + 250.
        assert_eq!(offs(&events), vec![(60, 250), (72, 750), (67, 2250)]);
    }

    #[test]
    fn step_seq_tie_into_rest_releases_at_the_rest() {
        let config = SeqConfig {
            mode: SeqMode::StepSeq,
            rate: Rate::Beats(1.0),
            gate: 0.5,
            steps: vec![
                Step { tie: true, ..Step::default() },
                Step { on: false, ..Step::default() },
            ],
            length: 2,
            ..SeqConfig::default()
        };
        let mut seq = sequencer(config);
        seq.note_on(60, 0.8, 0);
        let events = run(&mut seq, 900);
        assert_eq!(ons(&events), vec![(60, 0)]);
        assert_eq!(offs(&events), vec![(60, 500)]);
    }

    #[test]
    fn off_mode_passes_events_through_and_emits_nothing() {
        let mut seq = NoteSequencer::default();
        let on = seq.note_on(60, 0.9, 2).expect("off mode forwards note on");
        assert_eq!((on.on, on.note, on.velocity, on.channel), (true, 60, 0.9, 2));
        assert!(run(&mut seq, 1000).is_empty());
        let off = seq.note_off(60, 0.4, 2).expect("off mode forwards note off");
        assert_eq!((off.on, off.note, off.channel), (false, 60, 2));
    }

    #[test]
    fn flush_releases_everything_sounding() {
        let mut config = arp(ArpPattern::Chord);
        config.gate = 1.0;
        let mut seq = sequencer(config);
        hold_chord(&mut seq);
        let events = run(&mut seq, 100); // fires the chord, gates still open
        assert_eq!(ons(&events).len(), 3);
        let mut released = Vec::new();
        seq.flush(|event| released.push(event));
        let mut notes: Vec<i32> =
            released.iter().filter(|e| !e.on).map(|e| e.note).collect();
        notes.sort();
        assert_eq!(notes, vec![60, 64, 67]);
    }

    #[test]
    fn mode_change_flushes_sounding_notes() {
        let mut seq = sequencer(arp(ArpPattern::Up));
        seq.note_on(60, 0.8, 0);
        run(&mut seq, 100);
        let mut released = Vec::new();
        seq.set_config(SeqConfig::default(), |event| released.push(event));
        assert_eq!(released.len(), 1);
        assert!(!released[0].on && released[0].note == 60);
    }

    #[test]
    fn config_json_round_trip() {
        let config = SeqConfig::from_json(&json!({
            "mode": "arp",
            "pattern": "updown",
            "rate": "1/8d",
            "gate": 0.6,
            "swing": 0.25,
            "latch": true,
            "octaves": 3,
            "seed": 7
        }))
        .unwrap();
        assert_eq!(config.mode, SeqMode::Arp);
        assert_eq!(config.pattern, ArpPattern::UpDown);
        assert_eq!(config.rate, Rate::Beats(0.75));
        assert_eq!(config.gate, 0.6);
        assert_eq!(config.swing, 0.25);
        assert!(config.latch);
        assert_eq!(config.octaves, 3);
        assert_eq!(config.seed, 7);

        let steps = SeqConfig::from_json(&json!({
            "mode": "step",
            "rate": 4.0,
            "steps": [{"transpose": 12, "velocity": 0.5}, {"on": false}]
        }))
        .unwrap();
        assert_eq!(steps.mode, SeqMode::StepSeq);
        assert_eq!(steps.rate, Rate::Hz(4.0));
        assert_eq!(steps.length, 2);
        assert_eq!(steps.steps[0].transpose_semitones, 12);
        assert!(!steps.steps[1].on);

        assert!(SeqConfig::from_json(&json!({"mode": "bogus"})).is_err());
        assert!(SeqConfig::from_json(&json!({"mode": "arp", "rate": "1/64"})).is_err());
    }

    #[test]
    fn division_table_covers_dotted_and_triplet() {
        assert_eq!(division_beats("1/1"), Some(4.0));
        assert_eq!(division_beats("1/4"), Some(1.0));
        assert_eq!(division_beats("1/8"), Some(0.5));
        assert_eq!(division_beats("1/8d"), Some(0.75));
        assert_eq!(division_beats("1/8t"), Some(0.5 * 2.0 / 3.0));
        assert_eq!(division_beats("1/32"), Some(0.125));
        assert_eq!(division_beats("1/64"), None);
        assert_eq!(division_beats("nope"), None);
    }
}
