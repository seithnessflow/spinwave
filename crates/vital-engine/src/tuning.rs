//! Microtuning support (rework of Vital's `common/tuning.cpp`).
//!
//! Parses Scala (`.scl`), keyboard mapping (`.kbm`) and AnaMark (`.tun`)
//! content from strings and maps MIDI notes to tuned (fractional) note
//! numbers. The default tuning is 12-TET identity.

use vital_poly::constants::{CENTS_PER_NOTE, MIDI_0_FREQUENCY, MIDI_SIZE, NOTES_PER_OCTAVE};

pub const TUNING_SIZE: usize = 2 * MIDI_SIZE;
pub const TUNING_CENTER: usize = MIDI_SIZE;
const DEFAULT_MIDI_REFERENCE: i32 = 60;
const SCALA_KBM_COMMENT: char = '!';
const TUN_COMMENT: char = ';';

#[inline]
fn ratio_to_midi_transpose(ratio: f32) -> f32 {
    ratio.log2() * NOTES_PER_OCTAVE as f32
}

#[inline]
fn frequency_to_midi_note(frequency: f32) -> f32 {
    (frequency / MIDI_0_FREQUENCY).log2() * NOTES_PER_OCTAVE as f32
}

fn first_token(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

fn read_cents_to_transpose(cents: &str) -> f32 {
    cents.parse::<f32>().unwrap_or(0.0) / CENTS_PER_NOTE as f32
}

fn read_ratio_to_transpose(ratio: &str) -> f32 {
    let mut parts = ratio.split('/');
    let mut value = parts
        .next()
        .and_then(|t| t.trim().parse::<i64>().ok())
        .unwrap_or(0) as f32;
    if let Some(den) = parts.next().and_then(|t| t.trim().parse::<i64>().ok()) {
        value /= den as f32;
    }
    ratio_to_midi_transpose(value)
}

#[derive(Clone, Debug)]
pub struct Tuning {
    scale_start_midi_note: i32,
    reference_midi_note: f32,
    scale: Vec<f32>,
    keyboard_mapping: Vec<usize>,
    tuning: Vec<f32>,
    tuning_name: String,
    mapping_name: String,
    is_default: bool,
}

impl Default for Tuning {
    fn default() -> Self {
        let mut tuning = Tuning {
            scale_start_midi_note: DEFAULT_MIDI_REFERENCE,
            reference_midi_note: 0.0,
            scale: Vec::new(),
            keyboard_mapping: Vec::new(),
            tuning: vec![0.0; TUNING_SIZE],
            tuning_name: String::new(),
            mapping_name: String::new(),
            is_default: true,
        };
        tuning.set_default_tuning();
        tuning
    }
}

impl Tuning {
    /// Reference note spelling → MIDI key (e.g. `"c4"` → 60). Follows the
    /// reference's convention (octave applies to the letter as written).
    pub fn note_to_midi_key(note_text: &str) -> Option<i32> {
        const SCALE: [i32; 7] = [-3, -1, 0, 2, 4, 5, 7];
        const OCTAVE_START: i32 = -1;

        let text: String = note_text.to_lowercase().replace(' ', "");
        if text.len() < 2 {
            return None;
        }
        let mut chars = text.chars();
        let letter = chars.next()?;
        let note_in_scale = (letter as i32) - ('a' as i32);
        if !(0..7).contains(&note_in_scale) {
            return None;
        }
        let mut offset = SCALE[note_in_scale as usize];

        let mut rest: &str = chars.as_str();
        if let Some(stripped) = rest.strip_prefix('#') {
            rest = stripped;
            offset += 1;
        } else if let Some(stripped) = rest.strip_prefix('b') {
            rest = stripped;
            offset -= 1;
        }
        if rest.is_empty() {
            return None;
        }

        let mut negative = false;
        if let Some(stripped) = rest.strip_prefix('-') {
            rest = stripped;
            negative = true;
            if rest.is_empty() {
                return None;
            }
        }
        let mut octave = rest.chars().next()?.to_digit(10)? as i32;
        if negative {
            octave = -octave;
        }
        octave -= OCTAVE_START;
        Some(NOTES_PER_OCTAVE * octave + offset)
    }

    pub fn name(&self) -> String {
        if self.mapping_name.is_empty() {
            self.tuning_name.clone()
        } else if self.tuning_name.is_empty() {
            self.mapping_name.clone()
        } else {
            format!("{} / {}", self.tuning_name, self.mapping_name)
        }
    }

    pub fn set_name(&mut self, name: &str) {
        self.mapping_name.clear();
        self.tuning_name = name.to_string();
    }

    pub fn is_default(&self) -> bool {
        self.is_default
    }

    /// Tuned (fractional) note number for a MIDI note.
    #[inline]
    pub fn convert_midi_note(&self, note: i32) -> f32 {
        let scale_offset = note - self.scale_start_midi_note;
        let index = (TUNING_CENTER as i32 + scale_offset)
            .clamp(0, TUNING_SIZE as i32 - 1) as usize;
        self.tuning[index] + self.scale_start_midi_note as f32 + self.reference_midi_note
    }

    pub fn set_start_midi_note(&mut self, note: i32) {
        self.scale_start_midi_note = note;
    }

    pub fn set_reference_note(&mut self, note: f32) {
        self.reference_midi_note = note;
    }

    pub fn set_reference_frequency(&mut self, frequency: f32) {
        self.set_reference_note_frequency(0, frequency);
    }

    pub fn set_reference_note_frequency(&mut self, midi_note: i32, frequency: f32) {
        self.reference_midi_note = frequency_to_midi_note(frequency) - midi_note as f32;
    }

    pub fn set_reference_ratio(&mut self, ratio: f32) {
        self.reference_midi_note = ratio_to_midi_transpose(ratio);
    }

    pub fn set_default_tuning(&mut self) {
        for (i, value) in self.tuning.iter_mut().enumerate() {
            *value = i as f32 - TUNING_CENTER as f32;
        }
        self.scale = (0..=NOTES_PER_OCTAVE).map(|i| i as f32).collect();
        self.keyboard_mapping.clear();
        self.is_default = true;
        self.tuning_name.clear();
        self.mapping_name.clear();
    }

    pub fn set_constant_tuning(&mut self, note: f32) {
        self.tuning.fill(note);
    }

    /// Rebuilds the note table from a scale (offsets in fractional notes,
    /// first entry 0, last entry = octave span) and the keyboard mapping.
    pub fn load_scale(&mut self, scale: Vec<f32>) {
        self.scale = scale.clone();
        if scale.len() <= 1 {
            self.set_constant_tuning(DEFAULT_MIDI_REFERENCE as f32);
            return;
        }

        let scale_size = scale.len() - 1;
        let mapping_size = if self.keyboard_mapping.is_empty() {
            scale_size
        } else {
            self.keyboard_mapping.len()
        };

        let octave_offset = scale[scale_size];
        let start_octave = -(TUNING_CENTER as i32) / mapping_size as i32 - 1;
        let mut mapping_position =
            (-(TUNING_CENTER as i32) - start_octave * mapping_size as i32) as usize;

        let mut current_offset = start_octave as f32 * octave_offset;
        for i in 0..TUNING_SIZE {
            if mapping_position >= mapping_size {
                current_offset += octave_offset;
                mapping_position = 0;
            }

            let note_in_scale = if self.keyboard_mapping.is_empty() {
                mapping_position
            } else {
                self.keyboard_mapping[mapping_position]
            };

            self.tuning[i] = current_offset + scale.get(note_in_scale).copied().unwrap_or(0.0);
            mapping_position += 1;
        }
    }

    /// Parses Scala `.scl` content.
    pub fn load_scala(&mut self, content: &str, name: &str) {
        enum State {
            Description,
            ScaleLength,
            ScaleRatios,
        }
        let mut state = State::Description;
        let mut scale_length = 1usize;
        let mut scale = vec![0.0f32];

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(SCALA_KBM_COMMENT) {
                continue;
            }
            if scale.len() >= scale_length + 1 {
                break;
            }
            match state {
                State::Description => state = State::ScaleLength,
                State::ScaleLength => {
                    scale_length = first_token(trimmed).parse().unwrap_or(1);
                    state = State::ScaleRatios;
                }
                State::ScaleRatios => {
                    let tuning = first_token(trimmed);
                    if tuning.contains('.') {
                        scale.push(read_cents_to_transpose(tuning));
                    } else {
                        scale.push(read_ratio_to_transpose(tuning));
                    }
                }
            }
        }

        self.keyboard_mapping = (0..scale.len().saturating_sub(1)).collect();
        self.scale_start_midi_note = DEFAULT_MIDI_REFERENCE;
        self.reference_midi_note = 0.0;
        self.load_scale(scale);
        self.is_default = false;
        self.tuning_name = name.to_string();
    }

    /// Parses Scala keyboard mapping `.kbm` content over the current scale.
    pub fn load_keyboard_map(&mut self, content: &str, name: &str) {
        const HEADER_SIZE: usize = 7;
        const MAP_SIZE_POSITION: usize = 0;
        const MIDI_MAP_MIDDLE_POSITION: usize = 3;
        const REFERENCE_NOTE_POSITION: usize = 4;
        const REFERENCE_FREQUENCY_POSITION: usize = 5;

        let mut header = [0.0f32; HEADER_SIZE];
        let mut header_position = 0usize;
        let mut map_size = 0usize;
        let mut last_scale_value = 0usize;
        self.keyboard_mapping.clear();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(SCALA_KBM_COMMENT) {
                continue;
            }
            if header_position >= HEADER_SIZE {
                let token = first_token(trimmed);
                if !token.to_lowercase().starts_with('x') {
                    last_scale_value = token.parse().unwrap_or(0);
                }
                self.keyboard_mapping.push(last_scale_value);
                if self.keyboard_mapping.len() >= map_size {
                    break;
                }
            } else {
                header[header_position] = first_token(trimmed).parse().unwrap_or(0.0);
                if header_position == MAP_SIZE_POSITION {
                    map_size = header[header_position] as usize;
                }
                header_position += 1;
            }
        }

        self.set_start_midi_note(header[MIDI_MAP_MIDDLE_POSITION] as i32);
        self.set_reference_note_frequency(
            header[REFERENCE_NOTE_POSITION] as i32,
            header[REFERENCE_FREQUENCY_POSITION],
        );
        self.load_scale(self.scale.clone());
        self.is_default = false;
        self.mapping_name = name.to_string();
    }

    /// Parses AnaMark `.tun` content ([Tuning] / [Exact Tuning] sections).
    pub fn load_tun(&mut self, content: &str, name: &str) {
        #[derive(PartialEq)]
        enum State {
            Scanning,
            Tuning,
        }
        self.keyboard_mapping.clear();
        let mut state = State::Scanning;
        let mut last_read_note = 0usize;
        let mut base_frequency = MIDI_0_FREQUENCY;
        let mut scale: Vec<f32> = (0..MIDI_SIZE).map(|i| i as f32).collect();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with(TUN_COMMENT) {
                continue;
            }
            if let Some(section) = trimmed.strip_prefix('[') {
                let section = section.trim_end_matches(']').to_lowercase();
                state = if section == "tuning" || section == "exact tuning" {
                    State::Tuning
                } else {
                    State::Scanning
                };
            } else if state == State::Tuning {
                let Some((variable, value)) = trimmed.split_once('=') else { continue };
                let value: f32 = value.trim().parse().unwrap_or(0.0);
                let variable = variable.trim().to_lowercase();
                if variable == "basefreq" {
                    base_frequency = value;
                } else {
                    let mut tokens = variable.split_whitespace();
                    if tokens.next() == Some("note") {
                        if let Some(index) =
                            tokens.next().and_then(|t| t.parse::<usize>().ok())
                        {
                            if index < MIDI_SIZE {
                                last_read_note = last_read_note.max(index);
                                scale[index] = value / CENTS_PER_NOTE as f32;
                            }
                        }
                    }
                }
            }
        }

        scale.truncate(last_read_note + 1);
        self.load_scale(scale);
        self.set_start_midi_note(0);
        self.set_reference_frequency(base_frequency);
        self.is_default = false;
        self.tuning_name = name.to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tuning_is_identity() {
        let tuning = Tuning::default();
        assert!(tuning.is_default());
        for note in [0, 21, 60, 69, 127] {
            assert_eq!(tuning.convert_midi_note(note), note as f32);
        }
    }

    #[test]
    fn note_names_parse() {
        assert_eq!(Tuning::note_to_midi_key("c4"), Some(60));
        assert_eq!(Tuning::note_to_midi_key("C#4"), Some(61));
        assert_eq!(Tuning::note_to_midi_key("db4"), Some(61));
        assert_eq!(Tuning::note_to_midi_key("h4"), None);
        assert_eq!(Tuning::note_to_midi_key(""), None);
    }

    #[test]
    fn scala_12_tet_is_identity() {
        let scl = "! comment\n12-TET\n12\n100.0\n200.0\n300.0\n400.0\n500.0\n\
                   600.0\n700.0\n800.0\n900.0\n1000.0\n1100.0\n1200.0\n";
        let mut tuning = Tuning::default();
        tuning.load_scala(scl, "12tet");
        assert!(!tuning.is_default());
        for note in [48, 60, 61, 72] {
            assert!((tuning.convert_midi_note(note) - note as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn scala_ratios_parse() {
        // Just intonation fifth: 3/2 above the root.
        let scl = "just fifth\n2\n3/2\n2/1\n";
        let mut tuning = Tuning::default();
        tuning.load_scala(scl, "just");
        let root = tuning.convert_midi_note(60);
        let fifth = tuning.convert_midi_note(61);
        assert!((fifth - root - 1.5f32.log2() * 12.0).abs() < 1e-4);
        // Octave wraps: two steps up = 12 notes.
        let octave = tuning.convert_midi_note(62);
        assert!((octave - root - 12.0).abs() < 1e-4);
    }

    #[test]
    fn quarter_tone_scale_halves_steps() {
        let mut lines = String::from("24-TET\n24\n");
        for i in 1..=24 {
            lines.push_str(&format!("{}.0\n", i * 50));
        }
        let mut tuning = Tuning::default();
        tuning.load_scala(&lines, "24tet");
        let step = tuning.convert_midi_note(61) - tuning.convert_midi_note(60);
        assert!((step - 0.5).abs() < 1e-4);
    }

    #[test]
    fn tun_file_sets_scale_and_reference() {
        let tun = "; AnaMark\n[Tuning]\nbasefreq = 8.1757989156\n\
                   note 0 = 0\nnote 1 = 50\nnote 2 = 100\n";
        let mut tuning = Tuning::default();
        tuning.load_tun(tun, "quarter");
        let step = tuning.convert_midi_note(1) - tuning.convert_midi_note(0);
        assert!((step - 0.5).abs() < 1e-3, "step {step}");
    }

    #[test]
    fn name_combines_tuning_and_mapping() {
        let mut tuning = Tuning::default();
        tuning.load_scala("x\n1\n1200.0\n", "octaves");
        assert_eq!(tuning.name(), "octaves");
        tuning.load_keyboard_map("! map\n1\n0\n127\n60\n69\n440.0\n1\n0\n", "concert");
        assert_eq!(tuning.name(), "octaves / concert");
    }
}
