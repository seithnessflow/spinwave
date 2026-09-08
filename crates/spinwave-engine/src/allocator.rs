//! Voice allocation and per-block dispatch (rework of Vital's
//! `voice_handler.{h,cpp}`).
//!
//! Owns the pool of voice pairs (two stereo voices per SIMD kernel),
//! note-on/off routing with priority and steal policies, sustain /
//! sostenuto / MPE state, and drives each active kernel once per block.

use spinwave_poly::constants::{VoiceEvent, NOTES_PER_OCTAVE, NUM_MIDI_CHANNELS};
use spinwave_poly::utils::silent_mask;
use spinwave_poly::{PolyF32, PolyMask, LANES};

use crate::tuning::Tuning;
use crate::voice::{KeyState, Voice, VoiceControls};

pub const PARALLEL_VOICES: usize = LANES / 2;
pub const MAX_POLYPHONY: usize = 33;
pub const MAX_ACTIVE_POLYPHONY: usize = 32;
pub const LOCAL_PITCH_BEND_RANGE: f32 = 48.0;

const CHANNEL_SHIFT: u32 = 8;
const NOTE_MASK: i32 = (1 << CHANNEL_SHIFT) - 1;

#[inline(always)]
fn combine_note_channel(note: i32, channel: usize) -> i32 {
    ((channel as i32) << CHANNEL_SHIFT) + note
}

#[inline(always)]
fn get_channel(value: i32) -> usize {
    (value >> CHANNEL_SHIFT) as usize
}

#[inline(always)]
fn get_note(value: i32) -> i32 {
    value & NOTE_MASK
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoicePriority {
    Newest,
    Oldest,
    Highest,
    Lowest,
    RoundRobin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceOverride {
    Kill,
    Steal,
}

/// One SIMD kernel processing a pair of voices. The engine implements this
/// for the full synth voice; tests use lightweight kernels.
pub trait VoiceKernel {
    fn set_sample_rate(&mut self, sample_rate: u32);
    /// Renders one block. Trigger offsets in `controls` are in (oversampled)
    /// samples within this block.
    fn process(&mut self, controls: &VoiceControls, num_samples: usize);
    /// Audio output of the last processed block (both voices in lanes).
    fn output(&self) -> &[PolyF32];
    /// Direct-out bus of the last processed block: producers routed past
    /// the bus effect chain (still gated by the voice amplitude). `None`
    /// when the kernel has no separate direct bus.
    fn direct_output(&self) -> Option<&[PolyF32]> {
        None
    }
    /// Hard-routed effect bus outputs (A, B) of the last processed block,
    /// gated by the voice amplitude like the other buses.
    fn bus_outputs(&self) -> (Option<&[PolyF32]>, Option<&[PolyF32]>) {
        (None, None)
    }
    /// Buffer watched for voice death (the amplitude envelope output);
    /// lanes silent across the whole block let their voice be retired.
    /// Return `None` to keep voices alive until explicitly released.
    fn voice_killer(&self) -> Option<&[PolyF32]> {
        None
    }
}

/// All output buses of one kernel for one block, handed to the
/// accumulation callback of [`VoiceAllocator::process`].
pub struct KernelOutputs<'a> {
    /// Feeds the main effect chain.
    pub main: &'a [PolyF32],
    /// Bypasses every effect chain (summed after them).
    pub direct: Option<&'a [PolyF32]>,
    /// Hard-routed into effect bus A.
    pub bus_a: Option<&'a [PolyF32]>,
    /// Hard-routed into effect bus B.
    pub bus_b: Option<&'a [PolyF32]>,
}

/// Voice pool + articulation state, generic over the kernel.
pub struct VoiceAllocator<K: VoiceKernel> {
    kernels: Vec<K>,
    /// Per-kernel controls (kept between blocks: control values persist).
    controls: Vec<VoiceControls>,
    voices: Vec<Voice>,
    free_voices: Vec<usize>,
    /// Ordered by current priority policy; last entry is the most recent.
    active_voices: Vec<usize>,
    pressed_notes: Vec<i32>,

    polyphony: usize,
    priority: VoicePriority,
    override_mode: VoiceOverride,
    legato: bool,
    oversample: usize,

    sustain: [bool; NUM_MIDI_CHANNELS],
    sostenuto: [bool; NUM_MIDI_CHANNELS],
    mod_wheel_values: [f32; NUM_MIDI_CHANNELS],
    pitch_wheel_values: [f32; NUM_MIDI_CHANNELS],
    zoned_pitch_wheel_values: [f32; NUM_MIDI_CHANNELS],
    pressure_values: [f32; NUM_MIDI_CHANNELS],
    slide_values: [f32; NUM_MIDI_CHANNELS],

    total_notes: i32,
    last_played_note: PolyF32,
    has_played_note: bool,
    tuning: Tuning,
}

impl<K: VoiceKernel> VoiceAllocator<K> {
    pub fn new(polyphony: usize, mut make_kernel: impl FnMut() -> K) -> Self {
        let mut allocator = VoiceAllocator {
            kernels: Vec::new(),
            controls: Vec::new(),
            voices: Vec::new(),
            free_voices: Vec::new(),
            active_voices: Vec::new(),
            pressed_notes: Vec::new(),
            polyphony: 0,
            priority: VoicePriority::RoundRobin,
            override_mode: VoiceOverride::Kill,
            legato: false,
            oversample: 1,
            sustain: [false; NUM_MIDI_CHANNELS],
            sostenuto: [false; NUM_MIDI_CHANNELS],
            mod_wheel_values: [0.0; NUM_MIDI_CHANNELS],
            pitch_wheel_values: [0.0; NUM_MIDI_CHANNELS],
            zoned_pitch_wheel_values: [0.0; NUM_MIDI_CHANNELS],
            pressure_values: [0.0; NUM_MIDI_CHANNELS],
            slide_values: [0.0; NUM_MIDI_CHANNELS],
            total_notes: 0,
            last_played_note: PolyF32::splat(-1.0),
            has_played_note: false,
            tuning: Tuning::default(),
        };
        allocator.set_polyphony_with(polyphony, &mut make_kernel);
        allocator
    }

    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        for kernel in &mut self.kernels {
            kernel.set_sample_rate(sample_rate);
        }
    }

    pub fn set_oversample(&mut self, oversample: usize) {
        self.oversample = oversample;
    }

    pub fn set_priority(&mut self, priority: VoicePriority) {
        self.priority = priority;
    }

    pub fn set_override(&mut self, override_mode: VoiceOverride) {
        self.override_mode = override_mode;
    }

    pub fn set_legato(&mut self, legato: bool) {
        self.legato = legato;
    }

    pub fn set_tuning(&mut self, tuning: Tuning) {
        self.tuning = tuning;
    }

    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    pub fn polyphony(&self) -> usize {
        self.polyphony
    }

    pub fn num_active_voices(&self) -> usize {
        self.active_voices.len()
    }

    pub fn num_pressed_notes(&self) -> usize {
        self.pressed_notes.len()
    }

    pub fn kernels(&self) -> &[K] {
        &self.kernels
    }

    pub fn kernels_mut(&mut self) -> &mut [K] {
        &mut self.kernels
    }

    /// Grows the pool if needed and kills excess voices if shrinking.
    pub fn set_polyphony_with(&mut self, polyphony: usize, make_kernel: &mut impl FnMut() -> K) {
        let polyphony = polyphony.clamp(1, MAX_ACTIVE_POLYPHONY);
        while self.voices.len() < polyphony {
            self.add_voice_pair(make_kernel());
        }

        let excess = self.active_voices.len().saturating_sub(polyphony);
        for _ in 0..excess {
            if let Some(sacrifice) = self.voice_to_kill(polyphony) {
                self.voices[sacrifice].kill(0);
            }
        }
        self.polyphony = polyphony;
    }

    fn add_voice_pair(&mut self, kernel: K) {
        let pair = self.kernels.len();
        self.kernels.push(kernel);
        self.controls.push(VoiceControls::default());
        for slot in 0..PARALLEL_VOICES {
            let index = self.voices.len();
            self.voices.push(Voice::new(pair, slot));
            self.free_voices.push(index);
        }
    }

    // -- Note events ---------------------------------------------------------

    pub fn note_on(&mut self, note: i32, velocity: f32, sample: usize, channel: usize) {
        debug_assert!(channel < NUM_MIDI_CHANNELS);
        let Some(voice_index) = self.grab_voice() else { return };

        let tuned_note = self.tuning.convert_midi_note(note);

        let last_note = if self.has_played_note {
            self.last_played_note
        } else {
            PolyF32::splat(tuned_note)
        };
        self.last_played_note = PolyF32::splat(tuned_note);
        self.has_played_note = true;

        let note_value = combine_note_channel(note, channel);
        self.pressed_notes.retain(|&v| v != note_value);
        self.pressed_notes.push(note_value);

        self.total_notes += 1;
        let voice = &mut self.voices[voice_index];
        voice.activate(
            note,
            tuned_note,
            velocity,
            last_note,
            self.pressed_notes.len() as i32,
            self.total_notes,
            sample,
            channel,
        );
        voice.state.local_pitch_bend = self.pitch_wheel_values[channel];
        voice.aftertouch = self.pressure_values[channel];
        voice.slide = self.slide_values[channel];
        self.active_voices.push(voice_index);

        self.sort_voice_priority();
    }

    pub fn note_off(&mut self, note: i32, lift: f32, sample: usize, channel: usize) {
        let note_value = combine_note_channel(note, channel);
        self.pressed_notes.retain(|&v| v != note_value);

        let matching: Vec<usize> = self
            .active_voices
            .iter()
            .copied()
            .filter(|&v| {
                self.voices[v].state.midi_note == note && self.voices[v].state.channel == channel
            })
            .collect();

        for voice_index in matching {
            if self.sustain[channel] {
                let voice = &mut self.voices[voice_index];
                voice.sustain();
                voice.state.lift = lift;
            } else if self.polyphony <= self.pressed_notes.len()
                && self.voices[voice_index].state.event != VoiceEvent::Kill
            {
                // More pressed notes than voices: reuse this voice (or a
                // fresh one when killing) for the eldest unplayed note.
                let new_voice = if self.override_mode == VoiceOverride::Kill {
                    self.voices[voice_index].kill(sample);
                    self.grab_voice()
                } else {
                    self.active_voices.retain(|&v| v != voice_index);
                    Some(voice_index)
                };
                let Some(new_voice) = new_voice else { continue };

                if self.priority == VoicePriority::Newest {
                    self.active_voices.insert(0, new_voice);
                } else {
                    self.active_voices.push(new_voice);
                }

                let old_note_value = self.grab_next_unplayed_pressed_note();
                let old_note = get_note(old_note_value);
                let old_channel = get_channel(old_note_value);
                let tuned_note = self.tuning.convert_midi_note(old_note);

                self.total_notes += 1;
                let velocity = self.voices[voice_index].state.velocity;
                let last_played = self.last_played_note;
                let pressed = self.pressed_notes.len() as i32 + 1;
                let total = self.total_notes;
                let stolen = &mut self.voices[new_voice];
                stolen.activate(
                    old_note, tuned_note, velocity, last_played, pressed, total, sample,
                    old_channel,
                );
                stolen.state.local_pitch_bend = self.pitch_wheel_values[channel];
                stolen.aftertouch = self.pressure_values[channel];
                stolen.slide = self.slide_values[channel];
            } else {
                let voice = &mut self.voices[voice_index];
                voice.deactivate(sample);
                voice.state.lift = lift;
            }
        }

        self.sort_voice_priority();
    }

    pub fn all_notes_off(&mut self, sample: usize) {
        self.pressed_notes.clear();
        for &v in &self.active_voices.clone() {
            self.voices[v].deactivate(sample);
        }
    }

    pub fn all_sounds_off(&mut self) {
        self.pressed_notes.clear();
        for &v in &self.active_voices.clone() {
            self.voices[v].kill(0);
            self.voices[v].mark_dead();
            self.free_voices.push(v);
        }
        self.active_voices.clear();
    }

    pub fn is_note_playing(&self, note: i32, channel: usize) -> bool {
        self.active_voices.iter().any(|&v| {
            let voice = &self.voices[v];
            voice.state.event != VoiceEvent::Kill
                && voice.state.midi_note == note
                && voice.state.channel == channel
        })
    }

    // -- Pedals and expression ----------------------------------------------

    pub fn sustain_on(&mut self, channel: usize) {
        self.sustain[channel] = true;
    }

    pub fn sustain_off(&mut self, sample: usize, channel: usize) {
        self.sustain[channel] = false;
        for &v in &self.active_voices.clone() {
            let voice = &mut self.voices[v];
            if voice.sustained() && !voice.state.sostenuto_pressed && voice.state.channel == channel
            {
                voice.deactivate(sample);
            }
        }
    }

    pub fn sostenuto_on(&mut self, channel: usize) {
        self.sostenuto[channel] = true;
        for &v in &self.active_voices.clone() {
            let voice = &mut self.voices[v];
            if voice.state.channel == channel {
                voice.state.sostenuto_pressed = true;
            }
        }
    }

    pub fn sostenuto_off(&mut self, sample: usize, channel: usize) {
        self.sostenuto[channel] = false;
        let sustain = self.sustain[channel];
        for &v in &self.active_voices.clone() {
            let voice = &mut self.voices[v];
            if voice.state.channel == channel {
                voice.state.sostenuto_pressed = false;
                if voice.sustained() && !sustain {
                    voice.deactivate(sample);
                }
            }
        }
    }

    pub fn set_mod_wheel(&mut self, value: f32, channel: usize) {
        self.mod_wheel_values[channel] = value;
    }

    pub fn set_mod_wheel_all_channels(&mut self, value: f32) {
        self.mod_wheel_values = [value; NUM_MIDI_CHANNELS];
    }

    pub fn set_pitch_wheel(&mut self, value: f32, channel: usize) {
        self.pitch_wheel_values[channel] = value;
        for &v in &self.active_voices.clone() {
            let voice = &mut self.voices[v];
            if voice.state.channel == channel && voice.held() {
                voice.state.local_pitch_bend = value;
            }
        }
    }

    pub fn set_zoned_pitch_wheel(&mut self, value: f32, from_channel: usize, to_channel: usize) {
        for channel in from_channel..=to_channel {
            self.zoned_pitch_wheel_values[channel] = value;
        }
    }

    pub fn set_aftertouch(&mut self, note: i32, value: f32, sample: usize, channel: usize) {
        for &v in &self.active_voices.clone() {
            let voice = &mut self.voices[v];
            if voice.state.midi_note == note && voice.state.channel == channel {
                voice.set_aftertouch(value, sample);
            }
        }
    }

    pub fn set_channel_aftertouch(&mut self, channel: usize, value: f32, sample: usize) {
        self.pressure_values[channel] = value;
        for &v in &self.active_voices.clone() {
            let voice = &mut self.voices[v];
            if voice.state.channel == channel && voice.held() {
                voice.set_aftertouch(value, sample);
            }
        }
    }

    pub fn set_channel_slide(&mut self, channel: usize, value: f32, sample: usize) {
        self.slide_values[channel] = value;
        for &v in &self.active_voices.clone() {
            let voice = &mut self.voices[v];
            if voice.state.channel == channel && voice.held() {
                voice.set_slide(value, sample);
            }
        }
    }

    /// Kernel-pair index of the most recently activated voice, if any voice
    /// is active. The engine reads this pair's modulation sources for the
    /// mono (bus-effect) modulation matrix.
    pub fn last_active_pair(&self) -> Option<usize> {
        self.active_voices.last().map(|&v| self.voices[v].pair)
    }

    pub fn last_active_note(&self) -> f32 {
        self.active_voices
            .last()
            .map(|&v| self.voices[v].state.tuned_note)
            .unwrap_or(0.0)
    }

    // -- Voice grabbing ------------------------------------------------------

    fn grab_voice(&mut self) -> Option<usize> {
        if self.active_voices.len() < self.polyphony
            || (self.override_mode == VoiceOverride::Kill && !self.legato)
        {
            if let Some(v) = self.grab_free_parallel_voice() {
                return Some(v);
            }
            if let Some(v) = self.grab_free_voice() {
                return Some(v);
            }
        }

        for state in [
            KeyState::Released,
            KeyState::Sustained,
            KeyState::Held,
            KeyState::Triggering,
        ] {
            if let Some(v) = self.grab_voice_of_type(state) {
                return Some(v);
            }
        }
        None
    }

    fn grab_free_voice(&mut self) -> Option<usize> {
        if self.free_voices.is_empty() {
            None
        } else {
            Some(self.free_voices.remove(0))
        }
    }

    /// Prefers a dead slot whose pair sibling is active: the kernel is
    /// already running, so adding the second voice costs nothing.
    fn grab_free_parallel_voice(&mut self) -> Option<usize> {
        for pair in 0..self.kernels.len() {
            let mut dead_voice = None;
            let mut has_active = false;
            for slot in 0..PARALLEL_VOICES {
                let index = pair * PARALLEL_VOICES + slot;
                if self.voices[index].dead() {
                    dead_voice = Some(index);
                } else {
                    has_active = true;
                }
            }
            if has_active {
                if let Some(dead) = dead_voice {
                    self.free_voices.retain(|&v| v != dead);
                    return Some(dead);
                }
            }
        }
        None
    }

    fn grab_voice_of_type(&mut self, key_state: KeyState) -> Option<usize> {
        let position = self
            .active_voices
            .iter()
            .position(|&v| self.voices[v].key_state() == key_state)?;
        Some(self.active_voices.remove(position))
    }

    fn voice_to_kill(&self, max_voices: usize) -> Option<usize> {
        let mut excess = self.active_voices.len() as i32 - max_voices as i32;
        let mut released = None;
        let mut sustained = None;
        let mut held = None;

        for &v in &self.active_voices {
            let voice = &self.voices[v];
            if voice.state.event == VoiceEvent::Kill {
                excess -= 1;
            } else if released.is_none() && voice.key_state() == KeyState::Released {
                released = Some(v);
            } else if sustained.is_none() && voice.key_state() == KeyState::Sustained {
                sustained = Some(v);
            } else if held.is_none() {
                held = Some(v);
            }
        }

        if excess <= 0 {
            return None;
        }
        released.or(sustained).or(held)
    }

    fn grab_next_unplayed_pressed_note(&mut self) -> i32 {
        let find_unplayed = |notes: &[i32], allocator: &Self| -> usize {
            if allocator.priority == VoicePriority::Newest {
                let mut index = notes.len();
                while index > 0 {
                    index -= 1;
                    if !allocator.is_note_playing(get_note(notes[index]), get_channel(notes[index]))
                    {
                        break;
                    }
                }
                index
            } else {
                notes
                    .iter()
                    .position(|&n| !allocator.is_note_playing(get_note(n), get_channel(n)))
                    .unwrap_or(0)
            }
        };

        let index = find_unplayed(&self.pressed_notes, self);
        let old_note_value = self.pressed_notes[index];
        if self.priority == VoicePriority::RoundRobin {
            self.pressed_notes.remove(index);
            self.pressed_notes.push(old_note_value);
        }
        old_note_value
    }

    fn sort_voice_priority(&mut self) {
        match self.priority {
            VoicePriority::Highest => {
                let voices = &self.voices;
                self.active_voices
                    .sort_by_key(|&v| -voices[v].state.midi_note);
                self.pressed_notes.sort_by_key(|&n| get_note(n));
            }
            VoicePriority::Lowest => {
                let voices = &self.voices;
                self.active_voices.sort_by_key(|&v| voices[v].state.midi_note);
                self.pressed_notes.sort_by_key(|&n| -get_note(n));
            }
            VoicePriority::Oldest => {
                let voices = &self.voices;
                self.active_voices
                    .sort_by_key(|&v| voices[v].state.note_count);
            }
            _ => {}
        }
    }

    // -- Block processing ----------------------------------------------------

    /// Renders one block: triggers, control values, kernel dispatch, voice
    /// retirement. Calls `accumulate(outputs)` for each active pair; the
    /// caller sums each bus into its mix buffers (remember lanes hold two
    /// voices — add `swap_voices()` of the sum to fold them together).
    pub fn process(
        &mut self,
        num_samples: usize,
        mut accumulate: impl FnMut(KernelOutputs),
    ) {
        if self.active_voices.is_empty() {
            return;
        }

        // Unique active pairs, keeping the most recent pair last so its
        // control readouts win for mono modulation sources.
        let mut active_pairs: Vec<usize> = Vec::with_capacity(self.kernels.len());
        let mut last_pair = None;
        for &v in &self.active_voices {
            let pair = self.voices[v].pair;
            if !active_pairs.contains(&pair) {
                active_pairs.push(pair);
            }
            last_pair = Some(pair);
        }
        if let Some(last) = last_pair {
            active_pairs.retain(|&p| p != last);
            active_pairs.push(last);
        }

        for pair in active_pairs {
            self.prepare_voice_triggers(pair, num_samples);
            self.prepare_voice_values(pair);

            let kernel = &mut self.kernels[pair];
            kernel.process(&self.controls[pair], num_samples);
            let kernel = &self.kernels[pair];
            let (bus_a, bus_b) = kernel.bus_outputs();
            accumulate(KernelOutputs {
                main: &kernel.output()[..num_samples],
                direct: kernel.direct_output().map(|direct| &direct[..num_samples]),
                bus_a: bus_a.map(|bus| &bus[..num_samples]),
                bus_b: bus_b.map(|bus| &bus[..num_samples]),
            });

            // Retire voices whose killer buffer stayed silent after release.
            let alive_mask = match kernel.voice_killer() {
                Some(buffer) => !silent_mask(&buffer[..num_samples]),
                None => PolyMask::all_on(),
            };
            for slot in 0..PARALLEL_VOICES {
                let index = pair * PARALLEL_VOICES + slot;
                let voice = &self.voices[index];
                let released = voice.state.event == VoiceEvent::Off
                    || voice.state.event == VoiceEvent::Kill;
                let alive = (voice.mask() & alive_mask).any();
                let active = self.active_voices.contains(&index);
                if released && !alive && active {
                    self.active_voices.retain(|&v| v != index);
                    self.free_voices.push(index);
                    self.voices[index].mark_dead();
                }
            }
        }
    }

    fn prepare_voice_triggers(&mut self, pair: usize, num_samples: usize) {
        let controls = &mut self.controls[pair];
        controls.voice_event.clear();
        controls.retrigger.clear();
        controls.reset.clear();
        controls.note.clear();
        controls.last_note.clear();
        controls.velocity.clear();
        controls.lift.clear();
        controls.aftertouch.clear();
        controls.slide.clear();
        controls.channel.clear();

        let oversample = self.oversample;
        let legato = self.legato;

        for slot in 0..PARALLEL_VOICES {
            let index = pair * PARALLEL_VOICES + slot;
            let voice = &mut self.voices[index];
            let mask = voice.mask();

            if let Some(event_sample) = voice.event_sample {
                let offset = event_sample * oversample;
                if num_samples <= offset {
                    voice.event_sample = Some(event_sample - num_samples / oversample);
                } else {
                    let event_value = PolyF32::splat(voice.state.event.as_f32());
                    controls.voice_event.fire(mask, event_value, offset);

                    if voice.state.event == VoiceEvent::On {
                        controls
                            .note
                            .fire(mask, PolyF32::splat(voice.state.tuned_note), offset);
                        controls.last_note.fire(mask, voice.state.last_note, offset);
                        controls
                            .velocity
                            .fire(mask, PolyF32::splat(voice.state.velocity), offset);
                        controls
                            .channel
                            .fire(mask, PolyF32::splat(voice.state.channel as f32), offset);

                        if voice.last_key_state() == KeyState::Dead {
                            controls
                                .reset
                                .fire(mask, PolyF32::splat(VoiceEvent::On.as_f32()), offset);
                        }
                    } else if voice.state.event == VoiceEvent::Off {
                        controls
                            .lift
                            .fire(mask, PolyF32::splat(voice.state.lift), offset);
                    }

                    let suppress_retrigger = legato
                        && voice.last_key_state() == KeyState::Held
                        && voice.state.event == VoiceEvent::On;
                    if !suppress_retrigger {
                        controls.retrigger.fire(mask, event_value, offset);
                    }

                    voice.complete_voice_event();
                }
            }

            if let Some(aftertouch_sample) = voice.aftertouch_sample {
                let offset = aftertouch_sample * oversample;
                if num_samples <= offset {
                    voice.aftertouch_sample = Some(aftertouch_sample - num_samples / oversample);
                } else {
                    controls
                        .aftertouch
                        .fire(mask, PolyF32::splat(voice.aftertouch), offset);
                    voice.aftertouch_sample = None;
                }
            }

            if let Some(slide_sample) = voice.slide_sample {
                let offset = slide_sample * oversample;
                if num_samples <= offset {
                    voice.slide_sample = Some(slide_sample - num_samples / oversample);
                } else {
                    controls.slide.fire(mask, PolyF32::splat(voice.slide), offset);
                    voice.slide_sample = None;
                }
            }
        }
    }

    fn prepare_voice_values(&mut self, pair: usize) {
        for slot in 0..PARALLEL_VOICES {
            let index = pair * PARALLEL_VOICES + slot;
            let voice = &self.voices[index];
            let controls = &mut self.controls[pair];
            let mask = voice.mask();
            let channel = voice.state.channel;

            controls
                .note
                .set_value(mask, PolyF32::splat(voice.state.tuned_note));
            let note = controls.note.value;
            controls.last_note.set_value(mask, voice.state.last_note);

            controls.note_pressed =
                mask.select(PolyF32::splat(voice.state.note_pressed as f32), controls.note_pressed);
            controls.note_count =
                mask.select(PolyF32::splat(voice.state.note_count as f32), controls.note_count);
            controls.note_in_octave = (note * (1.0 / NOTES_PER_OCTAVE as f32)).fract();
            controls
                .channel
                .set_value(mask, PolyF32::splat(channel as f32));
            controls
                .velocity
                .set_value(mask, PolyF32::splat(voice.state.velocity));

            let lift = if voice.released() { voice.state.lift } else { 0.0 };
            controls.lift.set_value(mask, PolyF32::splat(lift));

            controls
                .aftertouch
                .set_value(mask, PolyF32::splat(voice.aftertouch));
            controls.slide.set_value(mask, PolyF32::splat(voice.slide));

            let active_value = if voice.dead() { 0.0 } else { 1.0 };
            controls.active_mask = mask.select(PolyF32::splat(active_value), controls.active_mask);

            controls.mod_wheel =
                mask.select(PolyF32::splat(self.mod_wheel_values[channel]), controls.mod_wheel);

            let pitch_wheel = self.zoned_pitch_wheel_values[channel];
            controls.pitch_wheel =
                mask.select(PolyF32::splat(pitch_wheel), controls.pitch_wheel);
            controls.pitch_wheel_percent = mask.select(
                PolyF32::splat(pitch_wheel * 0.5 + 0.5),
                controls.pitch_wheel_percent,
            );

            let local_bend = voice.state.local_pitch_bend * LOCAL_PITCH_BEND_RANGE;
            controls.local_pitch_bend =
                mask.select(PolyF32::splat(local_bend), controls.local_pitch_bend);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spinwave_poly::constants::MAX_BUFFER_SIZE;

    /// Test kernel: plays 1.0 on lanes with a held voice, fading nothing.
    /// The "killer" mirrors the active gate so released voices die at the
    /// next block.
    struct GateKernel {
        output: Vec<PolyF32>,
        gate: PolyF32,
    }

    impl GateKernel {
        fn new() -> Self {
            GateKernel { output: vec![PolyF32::ZERO; MAX_BUFFER_SIZE], gate: PolyF32::ZERO }
        }
    }

    impl VoiceKernel for GateKernel {
        fn set_sample_rate(&mut self, _sample_rate: u32) {}

        fn process(&mut self, controls: &VoiceControls, num_samples: usize) {
            // Rise on voice-on trigger; fall on off/kill.
            let on = PolyF32::splat(VoiceEvent::On.as_f32());
            let is_on = controls.voice_event.value.eq(on);
            self.gate = (controls.voice_event.mask & is_on).select(PolyF32::ONE, self.gate);
            let is_off = !controls.voice_event.value.eq(on);
            self.gate = (controls.voice_event.mask & is_off).select(PolyF32::ZERO, self.gate);

            let value = self.gate * controls.active_mask;
            for sample in &mut self.output[..num_samples] {
                *sample = value;
            }
        }

        fn output(&self) -> &[PolyF32] {
            &self.output
        }

        fn voice_killer(&self) -> Option<&[PolyF32]> {
            Some(&self.output)
        }
    }

    fn make() -> VoiceAllocator<GateKernel> {
        let mut allocator = VoiceAllocator::new(8, GateKernel::new);
        allocator.set_sample_rate(44100);
        allocator
    }

    fn render(allocator: &mut VoiceAllocator<GateKernel>) -> Vec<PolyF32> {
        let mut out = vec![PolyF32::ZERO; 16];
        allocator.process(16, |outputs| {
            for (dest, src) in out.iter_mut().zip(outputs.main) {
                *dest += *src + src.swap_voices();
            }
        });
        out
    }

    #[test]
    fn single_note_activates_one_voice() {
        let mut allocator = make();
        allocator.note_on(60, 0.8, 0, 0);
        assert_eq!(allocator.num_active_voices(), 1);
        let out = render(&mut allocator);
        // One gated voice folded into both stereo lanes.
        assert_eq!(out[15].lane(0), 1.0);
        assert_eq!(out[15].lane(1), 1.0);
    }

    #[test]
    fn note_off_releases_and_voice_dies() {
        let mut allocator = make();
        allocator.note_on(60, 0.8, 0, 0);
        render(&mut allocator);
        allocator.note_off(60, 0.5, 0, 0);
        render(&mut allocator); // gate drops, killer sees silence
        assert_eq!(allocator.num_active_voices(), 0);
    }

    #[test]
    fn two_notes_share_one_kernel_pair() {
        let mut allocator = make();
        allocator.note_on(60, 0.8, 0, 0);
        allocator.note_on(64, 0.8, 0, 0);
        assert_eq!(allocator.num_active_voices(), 2);
        let out = render(&mut allocator);
        // Both voices sum: each stereo lane carries 2.0.
        assert_eq!(out[15].lane(0), 2.0);
    }

    #[test]
    fn sustain_pedal_holds_released_notes() {
        let mut allocator = make();
        allocator.note_on(60, 0.8, 0, 0);
        allocator.sustain_on(0);
        allocator.note_off(60, 0.5, 0, 0);
        render(&mut allocator);
        assert_eq!(allocator.num_active_voices(), 1);
        allocator.sustain_off(0, 0);
        render(&mut allocator);
        render(&mut allocator);
        assert_eq!(allocator.num_active_voices(), 0);
    }

    #[test]
    fn polyphony_limit_steals_voices() {
        let mut allocator = VoiceAllocator::new(2, GateKernel::new);
        allocator.note_on(60, 0.8, 0, 0);
        allocator.note_on(64, 0.8, 0, 0);
        allocator.note_on(67, 0.8, 0, 0);
        // Third note grabbed a voice; with Kill override the sacrificed
        // voice fades via kill event â€” active count includes the dying one.
        assert!(allocator.num_active_voices() >= 2);
        assert!(allocator.is_note_playing(67, 0));
    }

    #[test]
    fn note_stealing_returns_pressed_note_on_release() {
        let mut allocator = VoiceAllocator::new(1, GateKernel::new);
        allocator.note_on(60, 0.8, 0, 0);
        allocator.note_on(64, 0.8, 0, 0);
        assert!(allocator.is_note_playing(64, 0));
        // Releasing the sounding note revives the still-pressed one.
        allocator.note_off(64, 0.5, 0, 0);
        assert!(allocator.is_note_playing(60, 0));
    }

    #[test]
    fn all_sounds_off_clears_everything() {
        let mut allocator = make();
        allocator.note_on(60, 0.8, 0, 0);
        allocator.note_on(64, 0.8, 0, 0);
        allocator.all_sounds_off();
        assert_eq!(allocator.num_active_voices(), 0);
        assert_eq!(allocator.num_pressed_notes(), 0);
    }
}
