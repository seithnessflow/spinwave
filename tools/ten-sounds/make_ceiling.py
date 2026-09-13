#!/usr/bin/env python3
"""Writes the hand-written ceiling patch of every ten-sounds target.

A ceiling is a patch a person wrote that meets the target's criterion:
it says the criterion CAN be met with this engine, and by how much a
model's attempt falls short of a plain human answer. The settings here
are the judge's own known positives (judge.rs tests), which is where a
person first wrote them; each is written as `.vital` JSON, then as
`.spinwave` through the CLI so both conditions have their ceiling in
their own format. `run.py --ceiling` judges them all; a target whose
ceiling fails is a broken target, not a hard one.

    python tools/ten-sounds/make_ceiling.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLI = ROOT / "target" / "release" / "spinwave-cli.exe"
if not CLI.exists():
    CLI = ROOT / "target" / "release" / "spinwave-cli"
OUT = HERE / "ceiling"

# settings (engine values, Vital's names), modulations (source, destination, amount)
CEILING = {
    # A saw through a low-pass an octave above C2's fundamental: the first
    # harmonics stay (rolloff 194 Hz), the top does not (centroid 129 Hz).
    "sub_bass": ({"osc_1_wave_frame": 128.0, "osc_1_level": 0.8, "filter_1_on": 1.0, "filter_1_cutoff": 48.0,
                  "filter_1_resonance": 0.3}, []),
    "pluck": ({"osc_1_wave_frame": 128.0, "env_1_attack": 0.0, "env_1_decay": 0.55, "env_1_sustain": 0.0, "env_1_release": 0.3}, []),
    "pad": ({"osc_1_wave_frame": 128.0, "osc_1_unison_voices": 6.0, "osc_1_unison_detune": 3.0, "osc_1_stereo_spread": 1.0,
             "env_1_attack": 1.05, "env_1_sustain": 1.0}, []),
    "fm_bell": ({"osc_1_wave_frame": 0.0, "osc_1_distortion_type": 8.0, "osc_1_distortion_amount": 0.6,
                 "osc_2_on": 1.0, "osc_2_wave_frame": 0.0, "osc_2_level": 0.0, "osc_2_transpose": 19.0, "osc_2_tune": 0.3,
                 "env_1_attack": 0.0, "env_1_decay": 1.4, "env_1_sustain": 0.0, "env_1_release": 1.5}, []),
    "lead": ({"osc_1_wave_frame": 128.0, "filter_1_on": 1.0, "filter_1_blend": 1.0, "filter_1_cutoff": 96.0,
              "filter_1_resonance": 0.6, "env_1_sustain": 1.0}, []),
    "filter_sweep_up": ({"osc_1_wave_frame": 128.0, "filter_1_on": 1.0, "filter_1_cutoff": 30.0, "env_1_sustain": 1.0,
                         "env_2_attack": 1.2, "env_2_sustain": 1.0}, [("env_2", "filter_1_cutoff", 0.8)]),
    "velocity_dark_soft": ({"osc_1_wave_frame": 128.0, "filter_1_on": 1.0, "filter_1_cutoff": 40.0, "env_1_sustain": 1.0},
                           [("velocity", "filter_1_cutoff", 0.7)]),
    "keytrack_bright_high": ({"osc_1_wave_frame": 128.0, "filter_1_on": 1.0, "filter_1_cutoff": 30.0, "filter_1_keytrack": 1.0,
                              "env_1_sustain": 1.0}, [("note", "filter_1_cutoff", 0.9)]),
    "noise_riser": ({"osc_1_level": 0.0, "noise_on": 1.0, "noise_level": 0.8, "noise_destination": 0.0,
                     "filter_1_on": 1.0, "filter_1_cutoff": 40.0, "env_1_sustain": 1.0, "env_2_attack": 1.3, "env_2_sustain": 1.0},
                    [("env_2", "filter_1_cutoff", 0.7)]),
    "tempo_wobble": ({"osc_1_wave_frame": 128.0, "filter_1_on": 1.0, "filter_1_cutoff": 50.0, "lfo_1_tempo": 5.0, "env_1_sustain": 1.0},
                     [("lfo_1", "filter_1_cutoff", 0.6)]),
}


def preset(settings, modulations):
    values = {"osc_1_on": 1.0}
    values.update(settings)
    for i, (_, _, amount) in enumerate(modulations, start=1):
        values[f"modulation_{i}_amount"] = amount
    values["modulations"] = [{"source": s, "destination": d} for s, d, _ in modulations]
    return {"synth_version": "1.0.7", "preset_name": "ceiling", "settings": values}


def main():
    OUT.mkdir(exist_ok=True)
    for target, (settings, modulations) in CEILING.items():
        vital = OUT / f"{target}.vital"
        vital.write_text(json.dumps(preset(settings, modulations), indent=1), encoding="utf-8")
        text = OUT / f"{target}.spinwave"
        proc = subprocess.run([str(CLI), "to-text", str(vital), str(text)], capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"{target}: {proc.stderr}")
    # The two controls: the reconstruct truth is its own ceiling (distance
    # 0); the edit ceiling darkens neuro-trinity where its brightness
    # actually comes from — the hard clip's drive, the EQ's high shelf, the
    # noise — plus the filter and the attack: six parameters. (Lowering
    # the cutoff alone, the first idea, did not darken it at all: 1.08 on
    # the centroid ratio. The notch-spread filter under two LFOs is not
    # where the top end is.)
    truth = json.loads((ROOT / "presets" / "packs" / "lush-pad.vital").read_text(encoding="utf-8-sig"))
    (OUT / "reconstruct.vital").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    origin = json.loads((ROOT / "presets" / "packs" / "neuro-trinity.vital").read_text(encoding="utf-8-sig"))
    edited = json.loads(json.dumps(origin))
    values = edited["settings"]
    values["distortion_drive"] = 6.0
    values["eq_high_gain"] = -12.0
    values["eq_high_cutoff"] = 96.0
    values["filter_1_cutoff"] = float(origin["settings"].get("filter_1_cutoff", 64.0)) - 12.0
    values["noise_level"] = 0.02
    values["env_1_attack"] = float(origin["settings"].get("env_1_attack", 0.0)) + 0.8
    (OUT / "edit.vital").write_text(json.dumps(edited, indent=1), encoding="utf-8")
    for target in ("reconstruct", "edit"):
        proc = subprocess.run([str(CLI), "to-text", str(OUT / f"{target}.vital"), str(OUT / f"{target}.spinwave")],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"{target}: {proc.stderr}")
    print(f"wrote {len(CEILING) + 2} ceiling patches to {OUT}")


if __name__ == "__main__":
    main()
