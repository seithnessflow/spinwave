#!/usr/bin/env python3
"""Every .vital of a bank through both engines: the wavetable construction
compared frame by frame, then the audio, so the two sources of a gap are
told apart before anything is listened to.

    python tools/golden/bank_compare.py ~/Documents/Vital [--out results.md]

The table of record lives in notes/bank-compare.md (its rows come from
this script; the default --out is a scratch file next to it).

Per preset: the reference (`vital_golden --preset`) and Spinwave
(`spinwave-cli preset-render`) render the same note the same way (a
primer hidden by a 5 s skip - long enough for its echo through any delay
or reverb to die - then C3 at 0.8 for 1.5 s, 2.5 s rendered),
random phases off on both sides (the engines' phase seeds are not
aligned, so a unison patch would compare draws, not DSP), and dump each
oscillator's built table and the SMP sample. Columns: the table
difference (max |Δ| over frames and samples, per oscillator, then the
sample), the audio RMS difference,
the peak difference, the reference's peak, the band distance in dB.
Sorted by band distance, descending.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REFERENCE = HERE / "build" / "vital_golden.exe"
CLI = ROOT / "target" / "release" / "spinwave-cli.exe"


def run(*args):
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout, proc.stderr


def table_diff(ref_dir, sw_dir):
    """Max |Δ| per oscillator table and for the SMP sample, or a word when
    they differ in shape (the sample's first float is its length)."""
    out = []
    for i in (1, 2, 3, "sample"):
        a_path, b_path = ref_dir / f"osc_{i}.raw", sw_dir / f"osc_{i}.raw"
        if i == "sample":
            a_path, b_path = ref_dir / "sample.raw", sw_dir / "sample.raw"
        if not a_path.exists() or not b_path.exists():
            out.append("missing" if a_path.exists() != b_path.exists() else "none")
            continue
        a = np.fromfile(a_path, dtype="<f4")
        b = np.fromfile(b_path, dtype="<f4")
        if len(a) == 0 or len(b) == 0 or int(a[0]) != int(b[0]) or len(a) != len(b):
            out.append(f"{'frames' if i != 'sample' else 'length'} {int(a[0]) if len(a) else 0}/{int(b[0]) if len(b) else 0}")
            continue
        out.append("%.1e" % float(np.abs(a[1:] - b[1:]).max()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bank")
    parser.add_argument("--out", default=str(ROOT / "notes" / "bank-compare.table.md"))
    parser.add_argument("--work", default=os.path.join(os.environ.get("TEMP", "/tmp"), "bank_compare"))
    parser.add_argument("--only", default="", help="substring of preset names to run")
    parser.add_argument("--reference", default=str(REFERENCE),
                        help="the reference harness exe (the Debug build by default; a Release build "
                             "agrees with it at 4e-6 rms and is 60x faster, for iteration)")
    parser.add_argument("--reuse-reference", action="store_true",
                        help="skip the reference render when its raw and table dump exist in --work "
                             "(the reference is the slow half; the Spinwave half is re-rendered)")
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    presets = sorted(Path(args.bank).rglob("*.vital"))
    rows = []
    for preset in presets:
        name = preset.stem
        if args.only and args.only.lower() not in name.lower():
            continue
        slug = "".join(c if c.isalnum() else "_" for c in name)
        ref_dir, sw_dir = work / f"{slug}_ref", work / f"{slug}_sw"
        ref_dir.mkdir(exist_ok=True)
        sw_dir.mkdir(exist_ok=True)
        ref_raw, sw_raw = work / f"{slug}_ref.raw", work / f"{slug}_sw.raw"
        reuse = args.reuse_reference and ref_raw.exists() and (ref_dir / "osc_1.raw").exists()
        if reuse:
            code, err = 0, ""
        else:
            code, _, err = run(args.reference, "--preset", preset, ref_raw, "--dump-tables", ref_dir, "--fixed-phase", "--skip", "5")
        ignored = [line for line in err.splitlines() if "ignored" in line]
        if code != 0:
            rows.append({"name": name, "error": f"reference: {err.strip()[:120]}"})
            print(f"{name:<32} reference failed: {err.strip()[:100]}")
            continue
        code, _, err = run(CLI, "preset-render", preset, sw_raw, "--dump-tables", sw_dir, "--fixed-phase", "--skip", "5")
        if code != 0:
            rows.append({"name": name, "error": f"spinwave: {err.strip()[:120]}"})
            print(f"{name:<32} spinwave failed: {err.strip()[:100]}")
            continue
        code, out, err = run(CLI, "raw-distance", sw_raw, ref_raw)
        if code != 0 or not out.strip().startswith("{"):
            rows.append({"name": name, "error": f"distance: {err.strip()[:120]}"})
            continue
        distance = json.loads(out)
        tables = table_diff(ref_dir, sw_dir)
        row = {"name": name, "tables": tables, "ignored": len(ignored), **distance}
        rows.append(row)
        print(f"{name:<32} band {distance['band_db']:6.2f} dB  rms {distance['rms']:.2e}  peak {distance['peak']:.2e} / ref {distance['reference_peak']:.3f}  tables {' '.join(tables)}")

    scored = [r for r in rows if "band_db" in r]
    scored.sort(key=lambda r: -r["band_db"])
    lines = ["| preset | band distance (dB) | RMS diff | peak diff | ref peak | tables max |Δ| (osc 1 / 2 / 3 / sample) | ref ignored |",
             "|---|---|---|---|---|---|---|"]
    for r in scored:
        lines.append("| %s | %.2f | %.1e | %.1e | %.3f | %s | %d |" % (
            r["name"], r["band_db"], r["rms"], r["peak"], r["reference_peak"], " / ".join(r["tables"]), r["ignored"]))
    for r in rows:
        if "error" in r:
            lines.append("| %s | — | — | — | — | %s | — |" % (r["name"], r["error"]))
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (work / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\n{len(scored)} presets compared, {len(rows) - len(scored)} failed; table in {args.out}")


if __name__ == "__main__":
    main()
