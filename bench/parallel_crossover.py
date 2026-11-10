"""Isolated kernel-call timing: serial vs 1/2/4 workers vs the shipped gate.

Times only the ctypes call, so run-to-run noise from the Python wrapper drops
out. Four temporary libraries are built from the shipped source: the parallel
path forced on with 1, 2 and 4 workers (1 worker is the serial baseline through
the exact same loop), plus the shipped build with its own threshold.

    taskset -c 0,2,4,6 pixi run python bench/parallel_crossover.py

Builds go to /tmp; dist/ and src/ are not touched. The numbers in README
"Parallel batch decoding" were produced by this script.
"""

from __future__ import annotations

import ctypes
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "bench"))

from bench import layered_lattice  # noqa: E402

I = ctypes.c_int64
SOURCE = os.path.join(ROOT, "src", "kernels.mojo")

# (positions, batch): 600 positions is 3,602 nodes, the batch benchmark's size.
CASES = [(32, 64), (600, 4), (600, 16), (600, 64), (2_000, 64), (6_000, 64),
         (12_000, 12), (12_000, 64), (24_000, 4), (24_000, 64)]
COLUMNS = ("1 worker (serial)", "2 workers", "4 workers", "shipped")


def build(workers: int | None) -> str:
    """Build to /tmp; ``workers=None`` keeps the shipped thresholds and cap."""
    text = open(SOURCE, encoding="utf-8").read()
    label = "shipped" if workers is None else f"w{workers}"
    if workers is not None:
        text = re.sub(r"comptime PARALLEL_NODE_THRESHOLD = \d+",
                      "comptime PARALLEL_NODE_THRESHOLD = 0", text)
        text = re.sub(r"comptime PARALLEL_EDGE_THRESHOLD = \d+",
                      "comptime PARALLEL_EDGE_THRESHOLD = 0", text)
        text = re.sub(r"comptime MAX_WORKERS = \d+",
                      f"comptime MAX_WORKERS = {workers}", text)
    out = os.path.join(tempfile.gettempdir(), f"mm-parallel-{label}.so")
    with tempfile.NamedTemporaryFile("w", suffix=".mojo", delete=False) as handle:
        handle.write(text)
        path = handle.name
    try:
        proc = subprocess.run(
            ["mojo", "build", "--emit", "shared-lib", path, "-o", out],
            capture_output=True, text=True,
        )
    finally:
        os.remove(path)
    if proc.returncode:
        raise SystemExit(proc.stderr[-2000:])
    return out


def load(path: str):
    library = ctypes.CDLL(path)
    library.mm_viterbi_decode_batch.argtypes = [I] * 8
    library.mm_viterbi_decode_batch.restype = I
    return library


def main() -> None:
    builds = [1, 2, 4, None]
    libraries = dict(zip(COLUMNS, (load(build(w)) for w in builds), strict=True))
    print("Isolated kernel-call time, median ms, one ctypes call per cell.")
    print()
    print("| nodes | edges | rows | " + " | ".join(COLUMNS) + " |")
    print("|---:|---:|---:|" + "---:|" * len(COLUMNS))
    for positions, batch in CASES:
        lattice = layered_lattice(positions=positions, width=6)
        rng = np.random.default_rng(7)
        weights = rng.integers(-100, 400, size=(batch, lattice.edge_count),
                               dtype=np.int64)
        totals = np.empty((batch, lattice.node_count), dtype=np.int64)
        back = np.empty_like(totals)
        args = [
            batch, lattice.node_count, lattice.edge_count,
            lattice.edge_offsets.ctypes.data, lattice.predecessors.ctypes.data,
            weights.ctypes.data, totals.ctypes.data, back.ctypes.data,
        ]
        rounds = 31 if lattice.node_count <= 40_000 else 11
        timings: dict[str, list[float]] = {name: [] for name in COLUMNS}
        for _ in range(rounds):
            for name, library in libraries.items():
                start = time.perf_counter()
                library.mm_viterbi_decode_batch(*args)
                timings[name].append(time.perf_counter() - start)
        cells = " | ".join(
            f"{statistics.median(timings[name]) * 1e3:.2f}" for name in COLUMNS
        )
        print(f"| {lattice.node_count:,} | {lattice.edge_count:,} | {batch} "
              f"| {cells} |")


if __name__ == "__main__":
    main()
