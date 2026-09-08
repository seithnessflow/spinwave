//! Spinwave standalone runner (CPAL audio + MIDI via nih-plug).

fn main() {
    nih_plug::nih_export_standalone::<vital_plugin::Spinwave>();
}
