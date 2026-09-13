#!/usr/bin/env python3
"""Where a preset's distance between the engines comes from: the preset
is rendered by both engines as it is and with one thing removed at a
time (an effect off, a producer off, the filters off, the modulations
gone one source at a time, unison to 1, the LFO sync types to trigger),
and the band distance is printed per variant. The variant that brings
the distance to the floor names the module; a distance that survives
everything is the voice's core (oscillator, envelope, level).

    python tools/golden/bank_bisect.py <preset.vital> [--reference exe] [--work dir]

Both engines read the same modified JSON, so the comparison stays fair;
the Release reference build is 60x faster than the Debug one and agrees
with it at 4e-6 rms (Analog Pad), which is what a bisection needs.
"""
import argparse
import json
import re
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLI = ROOT / "target" / "release" / "spinwave-cli.exe"
EFFECTS = ["chorus", "compressor", "delay", "distortion", "eq", "filter_fx", "flanger", "phaser", "reverb"]


def run(*args):
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout, proc.stderr


def distance(reference, work, name, data):
    path = work / (name + ".vital")
    text = json.dumps(data)
    ref_raw, sw_raw = work / (name + "_ref.raw"), work / (name + "_sw.raw")
    # The reference render is the slow side (Debug harness: minutes per
    # variant); an unchanged variant in the work directory keeps it.
    cached = ref_raw.exists() and path.exists() and path.read_text(encoding="utf-8") == text
    path.write_text(text, encoding="utf-8")
    if not cached:
        code, _, err = run(reference, "--preset", path, ref_raw, "--fixed-phase", "--skip", "5")
        if code != 0:
            return None, "reference: " + err.strip()[:80]
    code, _, err = run(CLI, "preset-render", path, sw_raw, "--fixed-phase", "--skip", "5")
    if code != 0:
        return None, "spinwave: " + err.strip()[:80]
    code, out, err = run(CLI, "raw-distance", sw_raw, ref_raw)
    if code != 0:
        return None, "distance: " + err.strip()[:80]
    return json.loads(out), ""


def without(settings, keep):
    """The modulation list with only the entries whose index `keep` admits,
    and the `modulation_N_*` settings renumbered to follow: the slots are
    positional (slot i is `modulation_(i+1)`), so dropping an entry
    without shifting the settings would hand every later connection its
    neighbour's amount and polarity - the first version of this tool did,
    and its "no X" rows measured the shift, not X."""
    connections = settings.get("modulations", [])
    kept_indices = [i for i in range(len(connections)) if keep(i)]
    # A meta connection (destination `modulation_N_amount` / `_power`)
    # names its target SLOT: dropping the target drops it too, and a
    # surviving one is re-pointed at the target's new slot. The second
    # version of this tool left these names alone, and every "no X" row on
    # a preset with meta connections measured a meta link swung onto a
    # neighbour (VLT Future Gun, 2026-09-13: "no stereo" read as a fix).
    meta = re.compile(r"^modulation_([0-9]+)_(amount|power)$")
    renumber = {old: new for new, old in enumerate(kept_indices)}
    while True:
        dropped = False
        for i in list(kept_indices):
            match = meta.match(connections[i].get("destination", ""))
            if match and int(match.group(1)) - 1 not in renumber:
                kept_indices.remove(i)
                renumber = {old: new for new, old in enumerate(kept_indices)}
                dropped = True
        if not dropped:
            break
    kept = []
    for i in kept_indices:
        connection = dict(connections[i])
        match = meta.match(connection.get("destination", ""))
        if match:
            connection["destination"] = "modulation_%d_%s" % (renumber[int(match.group(1)) - 1] + 1, match.group(2))
        kept.append(connection)
    patch = {"modulations": kept}
    fields = ("amount", "power", "bipolar", "stereo", "bypass")
    for new, old in enumerate(kept_indices):
        for field in fields:
            key_old, key_new = "modulation_%d_%s" % (old + 1, field), "modulation_%d_%s" % (new + 1, field)
            if key_old in settings:
                patch[key_new] = settings[key_old]
    for slot in range(len(kept_indices) + 1, len(connections) + 1):
        for field in fields:
            patch["modulation_%d_%s" % (slot, field)] = 0.0
    return patch


def variants(settings):
    """(name, patch) pairs; a patch is applied on top of the settings."""
    out = [("as is", {})]
    for effect in EFFECTS:
        if settings.get(effect + "_on", 0) == 1:
            out.append((effect + " off", {effect + "_on": 0}))
    on_effects = {e + "_on": 0 for e in EFFECTS if settings.get(e + "_on", 0) == 1}
    if len(on_effects) > 1:
        out.append(("all effects off", on_effects))
    for producer in ["osc_1", "osc_2", "osc_3", "sample"]:
        if settings.get(producer + "_on", 0) == 1:
            out.append((producer + " off", {producer + "_on": 0}))
    for f in ["filter_1", "filter_2"]:
        if settings.get(f + "_on", 0) == 1:
            out.append((f + " off", {f + "_on": 0}))
    for i in (1, 2, 3):
        if settings.get("osc_%d_on" % i, 0) == 1 and settings.get("osc_%d_unison_voices" % i, 1) > 1:
            out.append(("osc_%d unison 1" % i, {"osc_%d_unison_voices" % i: 1}))
    lfo_sync = {"lfo_%d_sync_type" % i: 0 for i in range(1, 9) if settings.get("lfo_%d_sync_type" % i, 0) != 0}
    if lfo_sync:
        out.append(("lfo sync types 0", lfo_sync))
    if settings.get("portamento_time", -10) > -10:
        out.append(("portamento off", {"portamento_time": -10.0}))
    connections = settings.get("modulations", [])
    sources = sorted({m["source"] for m in connections if m.get("source")})
    if sources:
        out.append(("no modulations", without(settings, lambda i: False)))
    for source in sources:
        out.append(("no " + source, without(settings, lambda i, s=source: connections[i].get("source") != s)))
    live = [(i, m) for i, m in enumerate(connections) if m.get("source")]
    if len(live) > 1:
        for index, connection in live:
            out.append(("no %s->%s" % (connection["source"], connection["destination"]),
                        without(settings, lambda i, k=index: i != k)))
    used_lfos = [i for i in range(1, 9) if "lfo_%d" % i in sources]
    smooth = {"lfo_%d_smooth_mode" % i: 0 for i in used_lfos if settings.get("lfo_%d_smooth_mode" % i, 0) != 0}
    if smooth:
        out.append(("lfo smooth mode 0", smooth))
    sync = {"lfo_%d_sync" % i: 0 for i in used_lfos if settings.get("lfo_%d_sync" % i, 0) != 0}
    if sync:
        out.append(("lfo tempo sync off", sync))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("preset")
    parser.add_argument("--reference", default=str(HERE / "build" / "vital_golden.exe"))
    parser.add_argument("--work", default=os.path.join(os.environ.get("TEMP", "/tmp"), "bank_bisect"))
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    text = Path(args.preset).read_text(encoding="utf-8-sig")
    data = json.loads(text)
    settings = data["settings"]
    slug = "".join(c if c.isalnum() else "_" for c in Path(args.preset).stem)
    print(Path(args.preset).stem)
    for index, (name, patch) in enumerate(variants(settings)):
        variant = json.loads(json.dumps(data))
        variant["settings"].update(patch)
        result, error = distance(args.reference, work, "%s_%02d" % (slug, index), variant)
        if result is None:
            print("  %-24s %s" % (name, error))
            continue
        clamped = " (clamped: peak 2.1 both)" if result["reference_peak"] >= 2.0999 and result["rms"] == 0 else ""
        print("  %-24s band %6.2f dB  rms %.2e  ref peak %.3f%s" % (name, result["band_db"], result["rms"], result["reference_peak"], clamped))


if __name__ == "__main__":
    main()
