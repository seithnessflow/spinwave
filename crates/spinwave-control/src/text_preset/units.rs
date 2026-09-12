//! Engine value ↔ text, in the unit a person thinks in.
//!
//! The parameter table carries the display law (scale, offset, multiplier,
//! units, option names); this module decides how that law is spelled and,
//! more importantly, guarantees the spelling is **exact**: a value is
//! written with the fewest digits from which the inverse recovers the same
//! f32 bit pattern, or, when no decimal spelling can (a fourth root in f32
//! occasionally cannot), as `raw:<engine value>`. Nothing is rounded on
//! the way back.
//!
//! Reading is generous about spelling — `8kHz`, `8 kHz`, `8000 Hz` are one
//! value — and strict about meaning: a number with no unit where one is
//! expected is refused naming the unit, an unknown unit is refused, an
//! out-of-range value is refused with the range in the same unit.

use spinwave_params::{ParamDetails, ParamScale};

/// How a parameter is spelled in the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spelling {
    /// `true` / `false`.
    Bool,
    /// An option name from the table's lookup: `"ladder"`.
    Named,
    /// A bare integer for an indexed parameter without names: `-12`.
    Integer,
    /// A filter cutoff: MIDI semitones in the engine, Hz in the text.
    CutoffHz,
    /// A time in seconds (any scale): `"90 ms"` below one second, `"1.6 s"` above.
    Time,
    /// A frequency in Hz from an exponential parameter.
    Hertz,
    /// A level with no unit under a quadratic scale: dB of the final gain.
    GainDb,
    /// A number followed by the table's own unit, or bare when the table
    /// has none: `"6.0 dB"`, `"50%"`, `"-12 st"`, `0.78`.
    Unit(UnitKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitKind {
    None,
    Percent,
    Decibel,
    Semitones,
    Voices,
    Hertz,
    Multiplier,
    GrainsPerSecond,
}

impl UnitKind {
    fn suffix(self) -> &'static str {
        match self {
            UnitKind::None => "",
            UnitKind::Percent => "%",
            UnitKind::Decibel => " dB",
            UnitKind::Semitones => " st",
            UnitKind::Voices => " voices",
            UnitKind::Hertz => " Hz",
            UnitKind::Multiplier => "x",
            UnitKind::GrainsPerSecond => " grains/s",
        }
    }

    /// Every spelling of this unit the reader accepts.
    fn aliases(self) -> &'static [&'static str] {
        match self {
            UnitKind::None => &[""],
            UnitKind::Percent => &["%", "pct", "percent"],
            UnitKind::Decibel => &["db"],
            UnitKind::Semitones => &["st", "semi", "semitone", "semitones", "semis"],
            UnitKind::Voices => &["voices", "voice", "v", ""],
            UnitKind::Hertz => &["hz"],
            UnitKind::Multiplier => &["x", ""],
            UnitKind::GrainsPerSecond => &["grains/s", "grains", "g/s", ""],
        }
    }
}

fn unit_kind(units: &str) -> UnitKind {
    match units.trim() {
        "" => UnitKind::None,
        "%" => UnitKind::Percent,
        "dB" => UnitKind::Decibel,
        "semitones" => UnitKind::Semitones,
        "voices" => UnitKind::Voices,
        "Hz" => UnitKind::Hertz,
        "x" => UnitKind::Multiplier,
        "grains/s" => UnitKind::GrainsPerSecond,
        // The table also uses free-text units ("filter", "osc", "lfo", ...)
        // for indexed parameters that name a target; those are `Named` or
        // `Integer` and never reach here. Anything unknown is written bare.
        _ => UnitKind::None,
    }
}

/// Decides the spelling of a parameter. `key` is the parameter's key inside
/// its module (`cutoff`, `frequency`), `module` the module name.
pub fn spelling(details: &ParamDetails, module: &str, key: &str) -> Spelling {
    if details.is_boolean() {
        return Spelling::Bool;
    }
    match details.scale {
        ParamScale::Indexed => {
            if details.string_lookup.is_some() {
                Spelling::Named
            } else {
                Spelling::Integer
            }
        }
        ParamScale::Linear => {
            let units = details.display_units.trim();
            if units == "semitones" && key.ends_with("cutoff") {
                Spelling::CutoffHz
            } else if units == "secs" || units == "ms" || units == "s" {
                Spelling::Time
            } else {
                Spelling::Unit(unit_kind(units))
            }
        }
        ParamScale::Quadratic => {
            if details.display_units.trim().is_empty() {
                Spelling::GainDb
            } else {
                Spelling::Unit(unit_kind(&details.display_units))
            }
        }
        ParamScale::Cubic | ParamScale::Quartic => Spelling::Time,
        ParamScale::Exponential => {
            // The table displays some exponential parameters as a period in
            // seconds (`display_invert`) or as a time (`chorus_delay_1` is
            // 1000 * 2^x ms). An LFO's rate is the one place everyone
            // thinks in Hz, so it is written in Hz regardless; both
            // spellings are accepted on input for every exponential value.
            let is_rate = key == "frequency" && (module.starts_with("lfo_") || module.starts_with("random_"));
            let units = details.display_units.trim();
            if is_rate {
                Spelling::Hertz
            } else if units == "secs" || units == "ms" || units == "s" {
                Spelling::Time
            } else {
                Spelling::Hertz
            }
        }
        ParamScale::SquareRoot => Spelling::Unit(unit_kind(&details.display_units)),
    }
}

/// Where a value came from when read back, for the report.
#[derive(Clone, Debug, PartialEq)]
pub struct Read {
    pub engine: f32,
    /// The canonical spelling, when the input was not already it.
    pub normalised: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UnitError {
    /// No unit where one is required. Carries the acceptable spellings.
    MissingUnit { expected: String },
    /// A unit that does not belong to this parameter.
    WrongUnit { found: String, expected: String },
    /// Parsed, but outside the table's range. Range given in the input's unit.
    OutOfRange { value: String, range: String },
    /// Not a number, not a known option, not parseable.
    Bad { message: String },
}

// ---------------------------------------------------------------- display

/// The table's own display value: `multiply * skew(engine) + offset`, in
/// f64 so that the inverse has the precision to land back on the f32.
fn table_display(details: &ParamDetails, x: f64) -> f64 {
    let skewed = match details.scale {
        ParamScale::Quadratic => x * x,
        ParamScale::Cubic => x * x * x,
        ParamScale::Quartic => x * x * x * x,
        ParamScale::SquareRoot => x.sqrt(),
        ParamScale::Exponential => {
            if details.display_invert {
                1.0 / x.exp2()
            } else {
                x.exp2()
            }
        }
        ParamScale::Indexed | ParamScale::Linear => x,
    };
    details.display_multiply as f64 * skewed + details.post_offset as f64
}

/// The exact inverse of [`table_display`]. Unlike the C++ `unskewValue`,
/// SquareRoot is inverted for real: the format needs the bijection that
/// Vital's own text entry does without.
fn table_inverse(details: &ParamDetails, display: f64) -> f64 {
    let skewed = (display - details.post_offset as f64) / details.display_multiply as f64;
    match details.scale {
        ParamScale::Quadratic => skewed.sqrt(),
        ParamScale::Cubic => skewed.cbrt(),
        ParamScale::Quartic => skewed.sqrt().sqrt(),
        ParamScale::SquareRoot => skewed * skewed,
        ParamScale::Exponential => {
            if details.display_invert {
                (1.0 / skewed).log2()
            } else {
                skewed.log2()
            }
        }
        ParamScale::Indexed | ParamScale::Linear => skewed,
    }
}

/// Whether the table's time unit is milliseconds (else seconds).
fn table_time_is_ms(details: &ParamDetails) -> bool {
    details.display_units.trim() == "ms"
}

/// The display value the text carries: seconds for a time, Hz for a rate,
/// dB for a gain, the table's display value otherwise.
fn to_display(details: &ParamDetails, spelling: Spelling, engine: f32) -> f64 {
    let x = engine as f64;
    match spelling {
        Spelling::CutoffHz => 440.0 * ((x - 69.0) / 12.0).exp2(),
        Spelling::GainDb => 20.0 * (x * x).log10(),
        Spelling::Hertz => {
            let d = table_display(details, x);
            if details.display_invert {
                1.0 / d
            } else {
                d
            }
        }
        Spelling::Time => {
            let d = table_display(details, x);
            if table_time_is_ms(details) {
                d / 1000.0
            } else {
                d
            }
        }
        Spelling::Unit(_) => table_display(details, x),
        Spelling::Bool | Spelling::Named | Spelling::Integer => x,
    }
}

/// The engine value for a display value: the exact inverse of
/// [`to_display`], in f64, rounded once to f32.
fn from_display(details: &ParamDetails, spelling: Spelling, display: f64) -> f32 {
    let x = match spelling {
        Spelling::CutoffHz => 69.0 + 12.0 * (display / 440.0).log2(),
        Spelling::GainDb => (10.0f64.powf(display / 20.0)).sqrt(),
        Spelling::Hertz => {
            let d = if details.display_invert { 1.0 / display } else { display };
            table_inverse(details, d)
        }
        Spelling::Time => {
            let d = if table_time_is_ms(details) { display * 1000.0 } else { display };
            table_inverse(details, d)
        }
        Spelling::Unit(_) => table_inverse(details, display),
        Spelling::Bool | Spelling::Named | Spelling::Integer => display,
    };
    x as f32
}

/// `x` with `digits` significant digits, trailing zeros trimmed, never in
/// exponent notation for the magnitudes a synth parameter reaches.
fn format_significant(x: f64, digits: usize) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let magnitude = x.abs().log10().floor() as i32;
    let decimals = (digits as i32 - 1 - magnitude).max(0) as usize;
    let s = format!("{x:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// A time in seconds as text: milliseconds below one second.
fn format_time(seconds: f64, digits: usize) -> String {
    if seconds.abs() < 1.0 {
        format!("{} ms", format_significant(seconds * 1000.0, digits))
    } else {
        format!("{} s", format_significant(seconds, digits))
    }
}

fn format_at(details: &ParamDetails, spelling: Spelling, display: f64, digits: usize) -> String {
    match spelling {
        Spelling::CutoffHz | Spelling::Hertz => {
            if display >= 1000.0 {
                format!("{} kHz", format_significant(display / 1000.0, digits))
            } else {
                format!("{} Hz", format_significant(display, digits))
            }
        }
        Spelling::GainDb => format!("{} dB", format_significant(display, digits)),
        Spelling::Time => format_time(display, digits),
        Spelling::Unit(kind) => {
            format!("{}{}", format_significant(display, digits), kind.suffix())
        }
        Spelling::Bool | Spelling::Named | Spelling::Integer => {
            unreachable!("{}: discrete spellings do not go through format_at", details.name)
        }
    }
}

/// The text for an engine value, plus a derived comment where one helps
/// (the semitone value behind a Hz, the knob value behind a dB).
///
/// Exact by construction: the returned text parses back to `engine` bit
/// for bit, or is the `raw:` form.
pub fn write(details: &ParamDetails, module: &str, key: &str, engine: f32) -> (Value, Option<String>) {
    let spelling = spelling(details, module, key);
    match spelling {
        Spelling::Bool => return (Value::Bool(engine >= 0.5), None),
        Spelling::Integer => {
            // An indexed parameter that is not on an integer (a fuzzed
            // patch, or a file edited by hand) keeps its exact value.
            if engine.fract() != 0.0 {
                return (Value::Str(format!("raw:{engine}")), None);
            }
            return (Value::Integer(engine as i64), None);
        }
        Spelling::Named => {
            if engine.fract() != 0.0 {
                return (Value::Str(format!("raw:{engine}")), None);
            }
            let index = engine as i64;
            let names = details.string_lookup.unwrap_or(&[]);
            let offset = details.min as i64;
            let position = usize::try_from(index - offset).ok().filter(|i| *i < names.len());
            return match position {
                Some(i) => (Value::Str(option_text(names, i)), None),
                // An index the table has no name for: honest fallback.
                None => (Value::Str(format!("raw:{engine}")), None),
            };
        }
        _ => {}
    }

    let display = to_display(details, spelling, engine);
    let exact = |text: &str| parse_engine(details, spelling, text).is_ok_and(|p| p.to_bits() == engine.to_bits());

    if spelling == Spelling::GainDb {
        // A level has two exact spellings: the knob value Vital shows and
        // stores, and the dB of the gain it produces. Pinning an f32
        // through dB needs seven digits (a half-ulp of 0.7 is 7e-7 dB),
        // while the knob value is usually short — so the knob wins where it
        // is shorter, and a level typed in dB stays in dB. Either way the
        // other form is the comment.
        if engine <= 0.0 {
            return (Value::Number("0".into()), Some("-inf dB".into()));
        }
        let knob: Option<String> = (1..=9).map(|d| format_significant(engine as f64, d)).find(|t| exact(t));
        let db: Option<String> = (3..=9).map(|d| format_at(details, spelling, display, d)).find(|t| exact(t));
        return match (knob, db) {
            (Some(k), Some(d)) if k.len() <= d.len() => (Value::Number(k), Some(format_at(details, spelling, display, 3))),
            (Some(k), None) => (Value::Number(k), Some(format_at(details, spelling, display, 3))),
            (_, Some(d)) => (Value::Str(d), Some(format!("knob {}", format_significant(engine as f64, 4)))),
            (None, None) => (Value::Str(format!("raw:{engine}")), None),
        };
    }

    if spelling == Spelling::CutoffHz {
        // Two exact spellings exist for a cutoff, Hz and the engine's own
        // semitones. Hz reads better but a cutoff authored in Vital sits
        // on a whole semitone whose Hz value needs seven digits; take the
        // shorter exact form and put the other in the comment.
        let hz: Option<String> = (3..=9).map(|d| format_at(details, spelling, display, d)).find(|t| exact(t));
        let st: Option<String> = (3..=9).map(|d| format!("{} st", format_significant(engine as f64, d))).find(|t| exact(t));
        let (text, other) = match (hz, st) {
            (Some(hz), Some(st)) if st.len() < hz.len() => (st, Some(format_at(details, spelling, display, 4))),
            (Some(hz), _) => (hz, Some(format!("{} st", format_significant(engine as f64, 4)))),
            (None, Some(st)) => (st, Some(format_at(details, spelling, display, 4))),
            (None, None) => (format!("raw:{engine}"), None),
        };
        return (Value::Str(text), other);
    }

    for digits in 3..=9 {
        let text = format_at(details, spelling, display, digits);
        if exact(&text) {
            let value = match spelling {
                Spelling::Unit(UnitKind::None) => Value::Number(text),
                _ => Value::Str(text),
            };
            return (value, comment_for(details, spelling, engine));
        }
    }
    (Value::Str(format!("raw:{engine}")), comment_for(details, spelling, engine))
}

fn comment_for(details: &ParamDetails, spelling: Spelling, engine: f32) -> Option<String> {
    match spelling {
        Spelling::CutoffHz => Some(format!("{} st", format_significant(engine as f64, 4))),
        Spelling::GainDb => Some(format!("knob {}", format_significant(engine as f64, 4))),
        Spelling::Hertz if details.display_invert => {
            Some(format!("{} period", format_time(1.0 / (engine as f64).exp2(), 3)))
        }
        _ => None,
    }
}

/// A written value: TOML scalar shapes the serializer emits.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Integer(i64),
    /// A bare number, for a parameter whose table unit is empty.
    Number(String),
    Str(String),
}

/// Option names are written lowercase with spaces kept; matching is
/// case-insensitive and ignores spaces and hyphens.
fn canonical_option(name: &str) -> String {
    name.trim().to_lowercase()
}

/// The text for option `i` of a lookup. Vital reuses a display name for
/// two options in a few lists ("FM <- Osc" is both oscillator A and B, the
/// interface tells them apart by position), so a name that is not unique
/// in its list carries its index: `"fm <- osc [8]"`.
fn option_text(names: &[&str], i: usize) -> String {
    let canonical = canonical_option(names[i]);
    let duplicates = names.iter().filter(|n| option_matches(n, names[i])).count();
    if duplicates > 1 {
        format!("{canonical} [{i}]")
    } else {
        canonical
    }
}

fn option_matches(candidate: &str, input: &str) -> bool {
    let fold = |s: &str| {
        s.chars()
            .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    fold(candidate) == fold(input)
}

// ---------------------------------------------------------------- reading

/// Splits `"8 kHz"` into `(8.0, "khz")`. A bare number gives an empty unit.
fn split_number_unit(text: &str) -> Option<(f64, String)> {
    let text = text.trim();
    let end = text
        .char_indices()
        .take_while(|(i, c)| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+' || (*c == 'e' && *i > 0))
        .map(|(i, c)| i + c.len_utf8())
        .last()?;
    let (number, unit) = text.split_at(end);
    let number: f64 = number.parse().ok()?;
    Some((number, unit.trim().to_lowercase()))
}

/// Reads a bare TOML scalar already typed (bool / integer / float).
pub fn read_scalar(details: &ParamDetails, module: &str, key: &str, value: Scalar) -> Result<Read, UnitError> {
    let spelling = spelling(details, module, key);
    match (spelling, value) {
        (Spelling::Bool, Scalar::Bool(b)) => Ok(Read { engine: if b { 1.0 } else { 0.0 }, normalised: None }),
        (Spelling::Bool, Scalar::Integer(i)) if i == 0 || i == 1 => {
            Ok(Read { engine: i as f32, normalised: Some(if i == 1 { "true" } else { "false" }.into()) })
        }
        (Spelling::Integer, Scalar::Integer(i)) => {
            check_range(details, spelling, i as f64, i as f32).map(|engine| Read { engine, normalised: None })
        }
        (Spelling::Integer, Scalar::Float(f)) if f.fract() == 0.0 => {
            check_range(details, spelling, f, f as f32).map(|engine| Read { engine, normalised: Some(format!("{}", f as i64)) })
        }
        (Spelling::Named, Scalar::Integer(i)) => {
            // An index instead of a name: accepted, normalised to the name.
            let engine = check_range(details, spelling, i as f64, i as f32)?;
            Ok(Read { engine, normalised: Some(canonical_for(details, module, key, engine)) })
        }
        (Spelling::Unit(UnitKind::None), Scalar::Integer(i)) => {
            let engine = from_display(details, spelling, i as f64);
            check_range(details, spelling, i as f64, engine).map(|engine| Read { engine, normalised: None })
        }
        (Spelling::Unit(UnitKind::None), Scalar::Float(f)) => {
            let engine = from_display(details, spelling, f);
            check_range(details, spelling, f, engine).map(|engine| Read { engine, normalised: None })
        }
        // A bare number on a level is the knob value, as Vital shows and
        // stores it. Not ambiguous any more: dB always carries its unit.
        (Spelling::GainDb, Scalar::Integer(i)) => {
            check_range(details, spelling, i as f64, i as f32).map(|engine| Read { engine, normalised: None })
        }
        (Spelling::GainDb, Scalar::Float(f)) => {
            check_range(details, spelling, f, f as f32).map(|engine| Read { engine, normalised: None })
        }
        (_, Scalar::Bool(_)) => Err(UnitError::Bad { message: "a boolean where a value is expected".into() }),
        (sp, Scalar::Integer(_) | Scalar::Float(_)) => Err(UnitError::MissingUnit { expected: expected_units(sp, details) }),
    }
}

/// A TOML scalar as the parser hands it over.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Integer(i64),
    Float(f64),
}

fn expected_units(spelling: Spelling, details: &ParamDetails) -> String {
    match spelling {
        Spelling::CutoffHz => "Hz, kHz or st (e.g. \"440 Hz\", \"69 st\")".into(),
        Spelling::Hertz => "Hz, kHz, or a period in s/ms (e.g. \"2 Hz\", \"500 ms\")".into(),
        Spelling::Time => "ms or s (e.g. \"90 ms\", \"1.5 s\")".into(),
        Spelling::GainDb => "the knob value (0 to 1, as Vital shows it) or dB of the final gain (e.g. \"-6.2 dB\")".into(),
        Spelling::Unit(kind) => match kind {
            UnitKind::None => "a bare number".into(),
            other => other.suffix().trim().to_string(),
        },
        Spelling::Bool => "true or false".into(),
        Spelling::Named => details
            .string_lookup
            .map(|names| (0..names.len()).map(|i| option_text(names, i)).collect::<Vec<_>>().join(", "))
            .unwrap_or_default(),
        Spelling::Integer => "an integer".into(),
    }
}

/// Reads a string value in any accepted spelling, and says how it would
/// be written canonically when the input was not already that.
pub fn read_str(details: &ParamDetails, module: &str, key: &str, text: &str) -> Result<Read, UnitError> {
    let spelling = spelling(details, module, key);
    let engine = parse_engine(details, spelling, text)?;
    let canonical = canonical_for(details, module, key, engine);
    let trimmed = text.trim();
    Ok(Read { engine, normalised: (canonical != trimmed).then_some(canonical) })
}

/// The engine value a string means, checked against the table's range.
/// This is the half [`write`] uses to prove its own output exact, so it
/// must not itself call [`write`].
fn parse_engine(details: &ParamDetails, spelling: Spelling, text: &str) -> Result<f32, UnitError> {
    let trimmed = text.trim();

    if let Some(raw) = trimmed.strip_prefix("raw:") {
        let engine: f32 = raw.trim().parse().map_err(|_| UnitError::Bad { message: format!("raw value '{raw}' is not a number") })?;
        return check_range(details, spelling, engine as f64, engine);
    }

    match spelling {
        Spelling::Bool => {
            return match trimmed.to_lowercase().as_str() {
                "true" | "on" | "yes" | "1" => Ok(1.0),
                "false" | "off" | "no" | "0" => Ok(0.0),
                _ => Err(UnitError::Bad { message: format!("'{trimmed}' is not true/false") }),
            };
        }
        Spelling::Named => {
            let names = details.string_lookup.unwrap_or(&[]);
            let offset = details.min as i64;
            // "name [i]": the index decides, the name is checked.
            if let Some((name, rest)) = trimmed.rsplit_once('[') {
                if let Some(index) = rest.strip_suffix(']').and_then(|d| d.trim().parse::<usize>().ok()) {
                    if names.get(index).is_some_and(|n| option_matches(n, name)) {
                        return Ok((index as i64 + offset) as f32);
                    }
                }
            }
            let matches: Vec<usize> = names.iter().enumerate().filter(|(_, n)| option_matches(n, trimmed)).map(|(i, _)| i).collect();
            return match matches.as_slice() {
                [i] => Ok((*i as i64 + offset) as f32),
                [] => Err(UnitError::Bad { message: format!("'{trimmed}' is not one of: {}", expected_units(spelling, details)) }),
                several => Err(UnitError::Bad {
                    message: format!(
                        "'{trimmed}' names {} options; write one of: {}",
                        several.len(),
                        several.iter().map(|i| format!("\"{}\"", option_text(names, *i))).collect::<Vec<_>>().join(", ")
                    ),
                }),
            };
        }
        Spelling::Integer => {
            let value: i64 = trimmed.parse().map_err(|_| UnitError::Bad { message: format!("'{trimmed}' is not an integer") })?;
            return check_range(details, spelling, value as f64, value as f32);
        }
        _ => {}
    }

    if spelling == Spelling::GainDb && trimmed.to_lowercase().replace(' ', "") == "-infdb" {
        return check_range(details, spelling, 0.0, 0.0);
    }


    let (number, unit) = split_number_unit(trimmed).ok_or_else(|| UnitError::Bad { message: format!("'{trimmed}' is not a number with a unit") })?;

    // Which unit was written, and what display value it means.
    let display = match spelling {
        Spelling::CutoffHz => match unit.as_str() {
            "hz" => number,
            "khz" => number * 1000.0,
            "st" | "semi" | "semitone" | "semitones" | "semis" => {
                return check_range(details, spelling, number, number as f32);
            }
            "" => return Err(UnitError::MissingUnit { expected: expected_units(spelling, details) }),
            other => return Err(UnitError::WrongUnit { found: other.into(), expected: expected_units(spelling, details) }),
        },
        Spelling::Hertz => match unit.as_str() {
            "hz" => number,
            "khz" => number * 1000.0,
            "s" | "sec" | "secs" => 1.0 / number,
            "ms" => 1000.0 / number,
            "" => return Err(UnitError::MissingUnit { expected: expected_units(spelling, details) }),
            other => return Err(UnitError::WrongUnit { found: other.into(), expected: expected_units(spelling, details) }),
        },
        Spelling::Time => match unit.as_str() {
            "ms" => number / 1000.0,
            "s" | "sec" | "secs" => number,
            "" => return Err(UnitError::MissingUnit { expected: expected_units(spelling, details) }),
            other => return Err(UnitError::WrongUnit { found: other.into(), expected: expected_units(spelling, details) }),
        },
        Spelling::GainDb => match unit.as_str() {
            "db" => number,
            // Bare: the knob value itself.
            "" => return check_range(details, spelling, number, number as f32),
            other => return Err(UnitError::WrongUnit { found: other.into(), expected: expected_units(spelling, details) }),
        },
        Spelling::Unit(kind) => {
            let accepted = kind.aliases();
            if !accepted.contains(&unit.as_str()) {
                if unit.is_empty() {
                    return Err(UnitError::MissingUnit { expected: expected_units(spelling, details) });
                }
                return Err(UnitError::WrongUnit { found: unit, expected: expected_units(spelling, details) });
            }
            number
        }
        Spelling::Bool | Spelling::Named | Spelling::Integer => unreachable!(),
    };

    let engine = from_display(details, spelling, display);
    check_range(details, spelling, display, engine)
}

/// The canonical text for an engine value, without the comment.
fn canonical_for(details: &ParamDetails, module: &str, key: &str, engine: f32) -> String {
    match write(details, module, key, engine).0 {
        Value::Str(s) | Value::Number(s) => s,
        Value::Integer(i) => i.to_string(),
        Value::Bool(b) => b.to_string(),
    }
}

fn check_range(details: &ParamDetails, spelling: Spelling, display: f64, engine: f32) -> Result<f32, UnitError> {
    if engine.is_nan() || engine < details.min || engine > details.max {
        let (lo, hi) = match spelling {
            Spelling::Bool | Spelling::Named | Spelling::Integer | Spelling::GainDb => (details.min as f64, details.max as f64),
            _ => (to_display(details, spelling, details.min), to_display(details, spelling, details.max)),
        };
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        let fmt = |d: f64| match spelling {
            Spelling::Bool | Spelling::Named | Spelling::Integer | Spelling::GainDb => format_significant(d, 4),
            _ => format_at(details, spelling, d, 4),
        };
        return Err(UnitError::OutOfRange {
            value: match spelling {
                Spelling::Bool | Spelling::Named | Spelling::Integer | Spelling::GainDb => format_significant(display, 4),
                _ => format_at(details, spelling, display, 4),
            },
            range: format!("{} to {}", fmt(lo), fmt(hi)),
        });
    }
    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spinwave_params::parameters;

    fn details(name: &str) -> ParamDetails {
        parameters().lookup(name).expect(name).clone()
    }

    fn roundtrip(name: &str, engine: f32) -> String {
        let d = details(name);
        let place = super::super::layout::place(name);
        let (value, _) = write(&d, &place.module, &place.key, engine);
        let text = match &value {
            Value::Str(s) | Value::Number(s) => s.clone(),
            Value::Integer(i) => i.to_string(),
            Value::Bool(b) => b.to_string(),
        };
        let back = match value {
            Value::Str(s) | Value::Number(s) => read_str(&d, &place.module, &place.key, &s).unwrap().engine,
            Value::Integer(i) => read_scalar(&d, &place.module, &place.key, Scalar::Integer(i)).unwrap().engine,
            Value::Bool(b) => read_scalar(&d, &place.module, &place.key, Scalar::Bool(b)).unwrap().engine,
        };
        assert_eq!(back.to_bits(), engine.to_bits(), "{name} = {engine} wrote {text}");
        text
    }

    #[test]
    fn spellings_follow_the_table() {
        assert_eq!(roundtrip("filter_1_cutoff", 69.0), "69 st");
        let c = details("filter_1_cutoff");
        let hz = read_str(&c, "filter_1", "cutoff", "1 kHz").unwrap().engine;
        assert_eq!(roundtrip("filter_1_cutoff", hz), "1 kHz");
        // Vital's default attack is 0.5476 in the engine, whose fourth power
        // has no short decimal spelling that inverts to the same f32: the
        // writer adds digits until it does. A value authored in the text as
        // "90 ms" stays "90 ms".
        assert!(roundtrip("env_1_attack", 0.5476).ends_with(" ms"));
        let d = details("env_1_attack");
        let authored = read_str(&d, "env_1", "attack", "90 ms").unwrap().engine;
        assert_eq!(roundtrip("env_1_attack", authored), "90 ms");
        // A level written by Vital is a short knob value; one typed in dB
        // stays in dB, because that is its shorter exact form.
        assert_eq!(roundtrip("osc_1_level", 0.7), "0.7");
        let l = details("osc_1_level");
        let authored = read_str(&l, "osc_1", "level", "-6.2 dB").unwrap().engine;
        assert_eq!(roundtrip("osc_1_level", authored), "-6.2 dB");
        assert_eq!(roundtrip("osc_1_level", 0.0), "0");
        assert_eq!(roundtrip("lfo_1_frequency", 1.0), "2 Hz");
        assert_eq!(roundtrip("filter_1_model", 2.0), "ladder");
        assert_eq!(roundtrip("filter_1_resonance", 0.5), "50%");
        assert_eq!(roundtrip("osc_1_transpose", -12.0), "-12");
        assert_eq!(roundtrip("env_1_sustain", 0.78), "0.78");
        assert_eq!(roundtrip("osc_1_unison_detune", 4.0), "16%");
    }

    #[test]
    fn every_scale_round_trips_exactly_across_its_range() {
        // Sweep each parameter across its range in f32 steps that are not
        // "nice"; the write must land back on the same bits every time,
        // or fall back to raw:, which round-trips by definition.
        let mut raw = 0usize;
        let mut total = 0usize;
        for d in parameters().iter() {
            if d.is_boolean() || d.scale == ParamScale::Indexed {
                continue;
            }
            let place = super::super::layout::place(&d.name);
            for i in 0..=40 {
                let t = i as f32 / 40.0;
                let engine = d.min + (d.max - d.min) * t * t;
                let (value, _) = write(d, &place.module, &place.key, engine);
                let text = match &value {
                    Value::Str(t) | Value::Number(t) => t,
                    _ => continue,
                };
                total += 1;
                if text.starts_with("raw:") {
                    raw += 1;
                }
                let back = read_str(d, &place.module, &place.key, text).unwrap_or_else(|e| panic!("{}: {text}: {e:?}", d.name));
                assert_eq!(back.engine.to_bits(), engine.to_bits(), "{} = {engine} wrote {text}", d.name);
            }
        }
        // The raw fallback exists, and it must stay rare. Measured, not guessed.
        assert!(raw * 100 < total, "{raw} raw fallbacks out of {total} values");
    }

    #[test]
    fn reads_every_alias_and_reports_the_normalisation() {
        let d = details("filter_1_cutoff");
        for text in ["8kHz", "8 kHz", "8000Hz", "8000 Hz"] {
            let r = read_str(&d, "filter_1", "cutoff", text).unwrap();
            assert!((r.engine - 119.21).abs() < 0.02, "{text}: {}", r.engine);
        }
        let r = read_str(&d, "filter_1", "cutoff", "8kHz").unwrap();
        assert_eq!(r.normalised.as_deref(), Some("8 kHz"));

        let a = details("env_1_attack");
        let ms = read_str(&a, "env_1", "attack", "90ms").unwrap();
        let s = read_str(&a, "env_1", "attack", "0.09 s").unwrap();
        assert_eq!(ms.engine.to_bits(), s.engine.to_bits());
    }

    #[test]
    fn refuses_a_missing_unit_naming_the_expected_one() {
        let d = details("filter_1_cutoff");
        match read_str(&d, "filter_1", "cutoff", "440") {
            Err(UnitError::MissingUnit { expected }) => assert!(expected.contains("Hz")),
            other => panic!("{other:?}"),
        }
        let l = details("osc_1_level");
        assert_eq!(read_str(&l, "osc_1", "level", "0.7").unwrap().engine, 0.7);
        match read_str(&l, "osc_1", "level", "70%") {
            Err(UnitError::WrongUnit { expected, .. }) => assert!(expected.contains("knob")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn refuses_out_of_range_with_the_range_in_the_input_unit() {
        let d = details("env_1_attack");
        match read_str(&d, "env_1", "attack", "90 s") {
            Err(UnitError::OutOfRange { range, .. }) => assert!(range.contains(" s"), "{range}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn option_names_match_loosely_and_write_canonically() {
        let d = details("filter_1_style");
        let r = read_str(&d, "filter_1", "style", "24DB").unwrap();
        assert_eq!(r.engine, 1.0);
        assert_eq!(r.normalised.as_deref(), Some("24db"));
    }
}
