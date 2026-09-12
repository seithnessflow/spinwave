#!/usr/bin/env python3
"""Recovers the seed behind a Perlin-style random LFO from its probe curve.

Both engines seed every `RandomGenerator` from a process-global counter
(`next_seed_++`), so the seed a voice's `random_1` holds is a matter of
how many generators the process built before it — different on the two
sides of the bench, and on the Spinwave side different from one run to
the next until the counter was rewound per render. A case that modulates
from a random source compares noise unless both sides draw the same
values; `random_seed <n>` in a case file pins Spinwave's seed, and this
script says which one the reference happens to use.

    python tools/golden/random_seed.py <probe.csv> --hz 2 [--column random_1]

The probe is a `--probe random_1` CSV from either harness. In the Perlin
style the output between two draws is gradient noise,

    v(t) = 2 * lerp(from * t, to * (t - 1), t^2 (3 - 2t)),   t = phase in [0, 1)

which is linear in (from, to), so a least-squares fit over the first note
gives both draws — the residual says whether the model fits at all — and
a search over mt19937 seeds finds which generator makes those two values
among its first draws (`polyVoiceNext` draws two values per call, one per
voice slot, so the first note's `from`/`to` are draws 0/2 for slot 0 and
1/3 for slot 1).

The script self-tests before printing anything: a curve synthesised from
a known seed must come back as that seed, or it stops.
"""

import argparse
import csv
import math
import sys

import numpy as np

SAMPLE_RATE = 44100
BLOCK = 128
FIRST_DRAWS = 16


def draws(seed, count=FIRST_DRAWS):
    """The first `count` values a `RandomGenerator(-1, 1)` seeded with `seed`
    returns: mt19937 (numpy seeds it with `init_genrand`, like C++), one
    32-bit output scaled to [0, 1) in float32, then to [-1, 1)."""
    raw = np.random.RandomState(seed).randint(0, 2**32, size=count, dtype=np.uint64)
    canonical = raw.astype(np.float32) * np.float32(1.0 / 2**32)
    return canonical * np.float32(2.0) - np.float32(1.0)


def perlin(from_value, to_value, t):
    s = t * t * (3.0 - 2.0 * t)
    return 2.0 * ((from_value * t) * (1.0 - s) + (to_value * (t - 1.0)) * s)


def fit(curve, first_block, last_block, hz, phase_at_first):
    """Least-squares (from, to) over blocks [first_block, last_block), the
    phase at `first_block` being `phase_at_first` and advancing by one block
    of `hz` per block. Returns (from, to, residual)."""
    delta = hz * BLOCK / SAMPLE_RATE
    rows, targets = [], []
    for block in range(first_block, min(last_block, len(curve))):
        t = phase_at_first + (block - first_block) * delta
        if t >= 1.0:
            break
        s = t * t * (3.0 - 2.0 * t)
        rows.append([2.0 * t * (1.0 - s), 2.0 * (t - 1.0) * s])
        targets.append(2.0 * (curve[block] - 0.5))
    solution, residual, _, _ = np.linalg.lstsq(np.array(rows), np.array(targets), rcond=None)
    residual = float(residual[0]) if len(residual) else float("nan")
    return solution[0], solution[1], residual


def best_fit(curve, hz, note_blocks):
    """Tries the first few blocks as the reset block and a few phase
    offsets (a block's worth each), keeps the fit with the least residual."""
    delta = hz * BLOCK / SAMPLE_RATE
    best = None
    for reset in range(0, 4):
        for lead in (0.5, 1.0, 1.5, 2.0, 2.5):
            start = reset + 2
            a, b, residual = fit(curve, start, reset + note_blocks, hz, (lead + 2) * delta)
            if not math.isnan(residual) and (best is None or residual < best[0]):
                best = (residual, reset, lead, a, b)
    return best


def find_seed(from_value, to_value, max_seed, tolerance=2e-4):
    """Every seed whose first draws contain `from` then, later, `to`."""
    hits = []
    for seed in range(max_seed):
        values = draws(seed)
        for i in range(FIRST_DRAWS):
            if abs(values[i] - from_value) > tolerance:
                continue
            for j in range(i + 1, FIRST_DRAWS):
                if abs(values[j] - to_value) <= tolerance:
                    hits.append((seed, i, j))
    return hits


def self_test():
    """A curve made from a known seed must give that seed back."""
    seed, hz = 4242, 2.0
    values = draws(seed)
    delta = hz * BLOCK / SAMPLE_RATE
    curve = [0.5] * 2
    for block in range(2, 140):
        t = (block + 1) * delta
        curve.append(0.5 * perlin(values[0], values[2], t) + 0.5)
    best = best_fit(curve, hz, 130)
    if best is None or best[0] > 1e-8:
        sys.exit("random_seed.py: self-test FAILED — the fit does not recover a synthetic curve")
    hits = find_seed(best[3], best[4], seed + 1)
    if (seed, 0, 2) not in hits:
        sys.exit("random_seed.py: self-test FAILED — the search does not recover seed %d (%r)"
                 % (seed, hits))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("probe", help="probe CSV with a block column and the source column")
    parser.add_argument("--hz", type=float, required=True,
                        help="the random LFO's rate in Hz (random_N_frequency is log2 Hz)")
    parser.add_argument("--column", default="random_1")
    parser.add_argument("--note-blocks", type=int, default=100,
                        help="blocks of the first note to fit (it must stay within one draw period)")
    parser.add_argument("--max-seed", type=int, default=40000)
    args = parser.parse_args()

    self_test()

    with open(args.probe, newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    if args.column not in header:
        sys.exit("random_seed.py: no column %r in %s (has %s)" % (args.column, args.probe, header))
    index = header.index(args.column)
    curve = [float(row[index]) for row in rows[1:]]
    if any(math.isnan(v) for v in curve[: args.note_blocks + 4]):
        sys.exit("random_seed.py: the source reads NaN inside the first note; is the voice alive?")

    best = best_fit(curve, args.hz, args.note_blocks)
    if best is None:
        sys.exit("random_seed.py: no fit at all")
    residual, reset, lead, from_value, to_value = best
    print("fit: reset at block %d, phase lead %.1f blocks, from=%.5f to=%.5f, residual %.2e"
          % (reset, lead, from_value, to_value, residual))
    if residual > 1e-6:
        sys.exit("random_seed.py: the Perlin model does not fit this curve (residual %.2e); "
                 "wrong --hz, wrong style, or not a random LFO" % residual)

    hits = find_seed(from_value, to_value, args.max_seed)
    if not hits:
        sys.exit("random_seed.py: no seed below %d makes those draws" % args.max_seed)
    for seed, i, j in hits:
        print("seed %d: from = draw %d, to = draw %d (slot %d)" % (seed, i, j, i % 2))


if __name__ == "__main__":
    main()
