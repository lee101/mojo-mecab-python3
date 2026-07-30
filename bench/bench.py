"""Honest benchmarks for the Mojo kernels and compatibility surface."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import MeCab
import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_mecab as mm  # noqa: E402


def timeit(function, repeat=5):
    best = math.inf
    value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - start)
    return best, value


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def layered_lattice(positions=8_000, width=6):
    incoming = [[]]
    previous = [0]
    rng = np.random.default_rng(0)
    for _ in range(positions):
        current = []
        for _ in range(width):
            node = len(incoming)
            incoming.append(
                [
                    (predecessor, int(cost))
                    for predecessor, cost in zip(
                        previous,
                        rng.integers(-100, 400, size=len(previous)),
                        strict=True,
                    )
                ]
            )
            current.append(node)
        previous = current
    incoming.append([(predecessor, 0) for predecessor in previous])
    return mm.make_lattice(incoming)


def python_utf8_boundaries(text):
    data = text.encode("utf-8")
    offsets = [0]
    index = 0
    while index < len(data):
        lead = data[index]
        index += 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
        offsets.append(index)
    return offsets


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
    print()
    print("| benchmark | Mojo / compatibility | reference | ratio | result |")
    print("|---|---:|---:|---:|---|")

    lattice = layered_lattice()
    mojo_result = mm.decode(lattice)
    python_result = mm.reference_decode(lattice)
    assert mojo_result.path == python_result.path
    mojo_time, _ = timeit(lambda: mm.decode(lattice))
    reference_time, _ = timeit(lambda: mm.reference_decode(lattice), repeat=3)
    emit(
        f"Viterbi, {lattice.node_count:,} nodes / "
        f"{lattice.edge_count:,} edges",
        mojo_time,
        reference_time,
    )

    rng = np.random.default_rng(1)
    small = layered_lattice(positions=600, width=6)
    weights = rng.integers(
        -100, 400, size=(64, small.edge_count), dtype=np.int64
    )
    batch = mm.decode_batch(small, weights)

    def reference_batch():
        return [mm.reference_decode(small, row) for row in weights]

    assert batch.paths == tuple(result.path for result in reference_batch())
    mojo_time, _ = timeit(lambda: mm.decode_batch(small, weights))
    reference_time, _ = timeit(reference_batch, repeat=2)
    emit(
        f"Batched Viterbi, {weights.shape[0]} x "
        f"{small.node_count:,} nodes",
        mojo_time,
        reference_time,
    )

    real = mm.mecab_lattice("すももももももものうち" * 80)
    assert mm.decode(real).cost == mm.reference_decode(real).cost
    mojo_time, _ = timeit(lambda: mm.decode(real))
    reference_time, _ = timeit(lambda: mm.reference_decode(real), repeat=3)
    emit(
        f"UniDic lattice re-decode, {real.node_count:,} nodes",
        mojo_time,
        reference_time,
    )

    text = "日本語の形態素解析を高速化します。" * 30_000
    np.testing.assert_array_equal(
        mm.utf8_boundaries(text), python_utf8_boundaries(text)
    )
    mojo_time, _ = timeit(lambda: mm.utf8_boundaries(text))
    reference_time, _ = timeit(lambda: python_utf8_boundaries(text), repeat=3)
    emit(f"UTF-8 boundaries, {len(text):,} code points", mojo_time, reference_time)

    sample = "すももももももものうち。pythonが大好きです。" * 2_000
    ours = mm.Tagger("-Owakati")
    upstream = MeCab.Tagger("-Owakati")
    assert ours.parse(sample) == upstream.parse(sample)
    ours_time, _ = timeit(lambda: ours.parse(sample))
    upstream_time, _ = timeit(lambda: upstream.parse(sample))
    emit(
        f"Tagger.parse compatibility, {len(sample):,} chars",
        ours_time,
        upstream_time,
        mojo_label="compat",
        reference_label="mecab-python3",
        verdict="same C++ backend",
    )


def emit(
    name,
    mojo_time,
    reference_time,
    mojo_label="Mojo",
    reference_label="Python",
    verdict=None,
):
    ratio = reference_time / mojo_time
    result = verdict or ("faster" if ratio >= 1 else "slower")
    print(
        f"| {name} | {mojo_time * 1e3:.3f} ms ({mojo_label}) "
        f"| {reference_time * 1e3:.3f} ms ({reference_label}) "
        f"| {ratio:.2f}x | {result} |"
    )


if __name__ == "__main__":
    main()
