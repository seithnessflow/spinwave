#!/usr/bin/env python3
"""The ten-sounds test: does the .spinwave format and the measure loop help a
model make a sound from a sentence?

Three conditions on the same targets, so the comparison is what carries
the meaning:

  A  raw .vital JSON, one shot, no rendering       (the model's prior alone)
  B  .spinwave, one shot, no rendering              (B - A = what the format adds)
  C  .spinwave with the loop: render, analyse,       (C - B = what measuring adds)
     correct, up to five rounds

The judge's thresholds never reach the model. Condition C feeds back the
ANALYSIS of the render (spinwave-cli judge --analysis-only) and the load
report; pass/fail is computed on the side and logged. Protocol and the
reasons behind it: notes/ten-sounds-protocol.md.

    python tools/ten-sounds/run.py --samples 3 [--targets sub_bass,pluck] [--conditions A,B,C]

Needs `pip install anthropic` and credentials (ANTHROPIC_API_KEY, or a
profile from `ant auth login`). Writes tools/ten-sounds/results/<stamp>/.
"""

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLI = ROOT / "target" / "release" / "spinwave-cli.exe"
if not CLI.exists():
    CLI = ROOT / "target" / "release" / "spinwave-cli"

MODEL = "claude-opus-5"
MAX_ROUNDS = 5

# The two controls use presets the model is NOT shown as examples.
RECONSTRUCT_TRUTH = ROOT / "presets" / "packs" / "lush-pad.vital"
EDIT_ORIGIN = ROOT / "presets" / "packs" / "neuro-trinity.vital"
EXAMPLES = [ROOT / "presets" / "text" / f for f in ("glass-pluck.spinwave", "neuro-crache.spinwave", "extensions.spinwave")]

# A description of the reconstruct truth written from its settings, in the
# words a person would use, with no numbers.
RECONSTRUCT_DESCRIPTION = (
    "A lush pad: a wide unison saw with a second, thinner layer a fifth above "
    "it, a slow swell on the way in and a long release on the way out, a gently "
    "resonant low-pass that breathes slowly under an LFO, and chorus, delay and "
    "reverb behind it."
)


def cli(*args, check=False):
    """Runs spinwave-cli and returns (exit_code, stdout, stderr)."""
    proc = subprocess.run([str(CLI), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"spinwave-cli {' '.join(args)}: {proc.stderr}")
    return proc.returncode, proc.stdout, proc.stderr


def targets():
    _, out, _ = cli("targets", check=True)
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        ident, description = line.split(None, 1)
        rows.append((ident, description.strip()))
    return rows


def format_reference():
    """What condition B and C get: the format's design notes and the three
    committed examples. Nothing about the judge."""
    doc = (ROOT / "notes" / "preset-text-format.md").read_text(encoding="utf-8")
    examples = "\n\n".join(f"### {p.name}\n```toml\n{p.read_text(encoding='utf-8')}\n```" for p in EXAMPLES)
    return f"{doc}\n\n## Examples\n\n{examples}"


SYSTEM_A = (
    "You write presets for the Vital wavetable synthesizer as .vital JSON files. "
    "Reply with ONE fenced ```json block containing the complete preset and nothing else. "
    "Use Vital's own parameter names and engine value ranges."
)


def system_b():
    return (
        "You write presets for the Spinwave synthesizer (a Rust rework of Vital) in the "
        ".spinwave text format described below. Reply with ONE fenced ```toml block containing "
        "the complete patch and nothing else.\n\n" + format_reference()
    )


SYSTEM_C_LOOP = (
    "\n\nAfter each patch you send, you will receive either a load report (errors to fix, "
    "with lines and suggestions) or a measurement of your patch rendered through the synth: "
    "levels, spectrum, envelope, pitch, stereo width, movement. Compare the measurement with "
    "what was asked and send a corrected patch, or reply with the single word DONE if the "
    "patch already does what was asked."
)


def extract_block(text, lang):
    m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.S) or re.search(r"```\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def user_prompt(target_id, description, condition):
    if target_id == "reconstruct":
        return f"{RECONSTRUCT_DESCRIPTION}\n\nWrite the patch."
    if target_id == "edit":
        if condition == "A":
            origin = EDIT_ORIGIN.read_text(encoding="utf-8-sig")
            return f"Here is a patch:\n```json\n{origin}\n```\n\n{description}\n\nReply with the complete edited patch."
        _, _, _ = cli("to-text", str(EDIT_ORIGIN), str(HERE / "results" / "_edit_origin.spinwave"), check=True)
        origin = (HERE / "results" / "_edit_origin.spinwave").read_text(encoding="utf-8")
        return f"Here is a patch:\n```toml\n{origin}\n```\n\n{description}\n\nReply with the complete edited patch."
    return f"{description}\n\nWrite the patch."


def load_and_judge(text, condition, target_id, workdir, tag):
    """Writes the model's output, loads it, judges it. Returns a dict with
    everything the analysis needs, plus the feedback text for condition C."""
    ext = "vital" if condition == "A" else "spinwave"
    path = workdir / f"{tag}.{ext}"
    path.write_text(text, encoding="utf-8")
    record = {"file": str(path), "loaded": False, "errors": [], "pass": None, "checks": None}

    if condition != "A":
        code, out, err = cli("check", str(path))
        report = json.loads(out) if out.strip().startswith("{") else {"errors": [{"code": "syntax", "message": err.strip()}]}
        record["report"] = report
        record["errors"] = [e.get("code") for e in report.get("errors", [])]
        if code != 0:
            return record, "The patch did not load. Load report:\n```json\n" + json.dumps(report, indent=1) + "\n```"
    else:
        try:
            json.loads(text)
        except json.JSONDecodeError as e:
            record["errors"] = ["json_syntax"]
            return record, f"The patch is not valid JSON: {e}"

    record["loaded"] = True
    args = ["judge", str(path), "--target", target_id]
    if target_id == "reconstruct":
        args += ["--reference", str(RECONSTRUCT_TRUTH)]
    if target_id == "edit":
        args += ["--reference", str(EDIT_ORIGIN)]
    code, out, err = cli(*args)
    if not out.strip().startswith("{"):
        record["errors"].append("judge_refused")
        record["judge_error"] = err.strip()
        return record, f"The patch loaded but could not be measured: {err.strip()}"
    verdict = json.loads(out)
    record["pass"] = verdict["pass"]
    record["checks"] = verdict["checks"]
    feedback = "Measurement of your patch, rendered:\n```json\n" + json.dumps(verdict["analysis"], indent=1) + "\n```"
    return record, feedback


def run_cell(client, condition, target_id, description, sample, workdir, log):
    system = SYSTEM_A if condition == "A" else system_b()
    if condition == "C":
        system += SYSTEM_C_LOOP
    lang = "json" if condition == "A" else "toml"
    messages = [{"role": "user", "content": user_prompt(target_id, description, condition)}]
    rounds = []
    final = None
    for round_index in range(1, (MAX_ROUNDS if condition == "C" else 1) + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            messages=messages,
            thinking={"type": "adaptive"},
            cache_control={"type": "ephemeral"},
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        messages.append({"role": "assistant", "content": text})
        if condition == "C" and round_index > 1 and text.strip().upper().startswith("DONE"):
            rounds.append({"round": round_index, "done": True, "request_id": response._request_id})
            break
        patch = extract_block(text, lang)
        record, feedback = load_and_judge(patch, condition, target_id, workdir, f"{condition}-{target_id}-{sample}-r{round_index}")
        record.update({"round": round_index, "request_id": response._request_id, "usage": response.usage.to_dict()})
        rounds.append(record)
        final = record
        log.write(json.dumps({"condition": condition, "target": target_id, "sample": sample, **record}) + "\n")
        log.flush()
        if condition != "C":
            break
        messages.append({"role": "user", "content": feedback})
    return {"condition": condition, "target": target_id, "sample": sample, "rounds": rounds, "final": final}


def summarise(cells):
    """Pass rate, rounds, first-load rate and the error-code ranking, per
    condition. The error ranking is the most actionable number here: each
    frequent code names an alias to add, a unit to tolerate, or a sentence
    to fix in the format's documentation."""
    out = {}
    for condition in ("A", "B", "C"):
        rows = [c for c in cells if c["condition"] == condition]
        if not rows:
            continue
        finals = [c["final"] for c in rows if c["final"]]
        first_rounds = [c["rounds"][0] for c in rows if c["rounds"]]
        codes = {}
        for c in rows:
            for r in c["rounds"]:
                for code in r.get("errors", []):
                    codes[code] = codes.get(code, 0) + 1
        out[condition] = {
            "cells": len(rows),
            "pass_rate": sum(1 for f in finals if f.get("pass")) / max(len(finals), 1),
            "first_load_rate": sum(1 for r in first_rounds if r.get("loaded")) / max(len(first_rounds), 1),
            "mean_rounds": sum(len(c["rounds"]) for c in rows) / max(len(rows), 1),
            "error_codes": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
            "by_target": {
                t: sum(1 for c in rows if c["target"] == t and c["final"] and c["final"].get("pass")) / max(sum(1 for c in rows if c["target"] == t), 1)
                for t in sorted({c["target"] for c in rows})
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=3, help="patches per target per condition")
    parser.add_argument("--targets", default="", help="comma-separated target ids (default: all twelve)")
    parser.add_argument("--conditions", default="A,B,C")
    args = parser.parse_args()

    try:
        import anthropic
    except ImportError:
        sys.exit("pip install anthropic")
    if not CLI.exists():
        sys.exit(f"build the CLI first: cargo build --release -p spinwave-control --bin spinwave-cli ({CLI} missing)")

    client = anthropic.Anthropic()
    wanted = [t.strip() for t in args.targets.split(",") if t.strip()]
    conditions = [c.strip().upper() for c in args.conditions.split(",")]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    workdir = HERE / "results" / stamp
    workdir.mkdir(parents=True, exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)

    cells = []
    with open(workdir / "runs.jsonl", "w", encoding="utf-8") as log:
        for target_id, description in targets():
            if wanted and target_id not in wanted:
                continue
            for condition in conditions:
                for sample in range(1, args.samples + 1):
                    print(f"{condition} {target_id:<22} sample {sample} ...", end=" ", flush=True)
                    try:
                        cell = run_cell(client, condition, target_id, description, sample, workdir, log)
                    except anthropic.APIStatusError as e:
                        print(f"API error {e.status_code}: {e.message}")
                        continue
                    final = cell["final"] or {}
                    print(f"{'PASS' if final.get('pass') else 'fail'} after {len(cell['rounds'])} round(s)" + ("" if final.get("loaded") else f" (did not load: {final.get('errors')})"))
                    cells.append(cell)

    summary = summarise(cells)
    (workdir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (workdir / "cells.json").write_text(json.dumps(cells, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"\nresults in {workdir}")


if __name__ == "__main__":
    main()
