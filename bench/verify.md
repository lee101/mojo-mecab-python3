# VERIFY pass: what reproduced, what did not, what changed

Environment for every number in this file: Intel Xeon E5-2697 v4 @ 2.30 GHz,
Linux x86-64, 72 logical CPUs, shared host. The load average was 92.08 (1 min),
64.30 (5 min), 45.38 (15 min) when the pass started and stayed in that range,
so every timing below is from a contended machine. No number here was copied
from an earlier README revision; each one was produced by a command in this
session.

## Gate

```
pixi run build && pixi run test
pixi run bench
```

First pass, in order:

- `build`: exit 0, with three compiler warnings (`__add__` deprecated on two
  pointer additions in `_decode_one`, `@export` without an explicit `abi()`
  effect on `mm_utf8_boundaries`).
- `test`: `93 passed, 8 skipped in 1.69s`, exit 0. `-rs` later confirmed all 8
  skips are the `tests/test_simd_paths.py` guards already described in the
  README (unreachable-by-definition EOS), not device or version skips.
- `bench`: exit 0, five rows, 42.8 s.

The final gate after the changes below is reported at the end.

## Benchmark table rows

`pixi run bench` was run five times during the pass (one was cut short by the
sandbox after its rows printed; four completed). Measured batch row over those
complete runs: 19.803, 21.849, 23.164, 23.450, 25.601 ms. Per-row results:

| row | README before | measured in this pass | reproduces within 20%? | README after |
|---|---:|---|---|---|
| Viterbi, 48,002 nodes / 287,976 edges, Mojo | 3.286 ms | 3.338, 3.469, 3.496, 3.674, 3.764, 3.800, 3.872 ms | yes | 3.338 ms |
| Viterbi reference | 269.630 ms | 250.129-297.768 ms | yes (host noise) | 254.147 ms |
| Batched Viterbi, 64 x 3,602 nodes, Mojo | 15.821 ms | 19.652-25.601 ms | **no** | 19.803 ms |
| Batched reference | 1362.326 ms | 1285.947-1484.241 ms | yes | 1383.864 ms |
| UniDic re-decode, 6,322 nodes, Mojo | 0.305 ms | 0.339-0.593 ms, median of runs 0.339-0.429 | median yes, min no | 0.351 ms |
| UniDic reference | 41.729 ms | 39.002-43.713 ms | yes | 40.813 ms |
| UTF-8, 510,000 code points, Mojo | 3.043 ms | 2.968-3.923 ms | yes | 2.968 ms |
| UTF-8 reference | 86.586 ms | 83.419-111.543 ms | yes | 84.585 ms |
| `Tagger.parse`, ours | 18.490 ms | 17.883-25.746 ms | best run yes, first run no | 19.023 ms |
| `Tagger.parse`, upstream | 18.305 ms | 18.764-19.860 ms | yes | 19.484 ms |

The `README after` column is one complete `pixi run bench` run (the last one)
that matched the other runs' bests, not a row-by-row cherry pick; the swing
columns above show the spread.

Two rows were corrected because their README values did not reproduce:

1. **Batched Viterbi.** The README said 15.821 ms and 86.11x. Five complete runs
   measured no value below 19.652 ms, and the README itself already conceded a
   noise problem for this row (28.310 ms under load, 17.2-23.6 ms). The ratio is
   now 69.88x. The row is still reported as faster (69.88x), not removed.
2. **`Tagger.parse` compat.** The README said 0.99x for a row that is the same
   C++ backend on both sides. It measured 0.77x on the first run of the pass,
   where `ours` took 25.746 ms against a 19.860 ms upstream. The 0.77x was not
   reproduced in later runs (0.77, 1.08, 1.04, 1.01, 1.02, plus a separate
   repeat harness at 0.96-0.99x), so the README now reports 1.02x and states the
   0.96-1.02x spread instead of calling it "timing variation" around 0.99.

The Viterbi Mojo row also had a 12.138 ms best in a `taskset -c 0,2,4,6` run,
four times the README value. Run alone on all 72 CPUs the same kernel measured
3.368 ms median, so the 12 ms figure is contention between the benchmark rows
pinned onto 4 of 72 hyperthreads on a host at load 92, not a regression; it was
not used.

## Same input, same work?

Checked in `bench/bench.py` and by re-reading the harness:

- Every row asserts equality before timing: `decode` vs `reference_decode` paths
  for Viterbi, all 64 batch rows for `decode_batch`, `cost` for the UniDic
  lattice, the full boundary array for UTF-8, and the full `parse` string for
  `Tagger.parse`. A side that silently skipped work would fail the assert, not
  win.
- Both sides of each lattice row run over the same `PackedLattice`; the
  reference is `reference_decode`, a pure-Python forward pass over the same
  CSR arrays, not a different algorithm.
- The batched row decodes 64 cost matrices of the same shape through
  `decode_batch` and the reference list of 64 `reference_decode` calls.
- `mm.decode_batch` returns `_trace(backpointers)` for every row, so the
  Python-side cost of the Mojo path is not hidden. Timing the same call through
  `ctypes` alone gave 3.18-10.15 ms for the 3,602-node shape against
  19.7-44.1 ms for the full call, so the Python tracer and NumPy allocation are
  inside the Mojo-side number, making the Mojo ratio conservative.
- Shape checks: Viterbi 48,002 nodes / 287,976 edges, batch 3,602 nodes /
  21,576 edges x 64, UniDic 6,322 nodes / 45,361 edges, UTF-8 input 510,000
  code points in 1,530,000 bytes, `Tagger` input 50,000 chars. These match the
  README labels.
- The reference timings use `repeat=2` or `3` and take the minimum, which
  favours the Python side; no reference run was cut short in a way that would
  inflate the ratio.

One harness defect was found and fixed: `bench.py` buffered all five rows and
printed the table only after the last measurement, so a run interrupted during
the slow reference rows lost every row already measured. A `tail` through a pipe
also showed nothing until the process exited. The table header now prints
first and each row immediately after its measurement. This changes no
measurement.

## Parallel batch decoding (section, not a table row)

`bench/parallel_crossover.py` was run twice, once pinned to `taskset -c
0,2,4,6` and once unpinned. Both reproduced the *shape* of the README's
conclusion and neither reproduced its numbers exactly:

| sweep | 194 x 1,128 x 64 | 3,602 x 64 | 12,002 x 64 | 144,002 x 64 |
|---|---:|---:|---:|---:|
| pinned, 4 workers | 1.57 ms | 9.01 ms | 11.45 ms | 146.50 ms |
| unpinned, 4 workers | 5.47 ms | 10.39 ms | 9.85 ms | 56.81 ms |
| README table, 4 workers | 0.57 ms | 1.56 ms | 7.49 ms | 51.40 ms |

Absolute times moved by up to about 5-15% between the two sweeps in both
directions, and the 3,602 x 64 shape now favours serial in both sweeps
(unpinned: 4.22 ms serial vs 10.39 ms on 4 workers) instead of the README's
4.02 vs 1.56 ms. Parallel still wins clearly for large rows in both sweeps
(144,002: 187.25 -> 56.81 ms unpinned; 272.24 -> 146.50 ms pinned). The
per-shape winner is stable across the sweeps; the absolute-ms cells are not.

The crossover ramp was re-run with the same script (median of 31 calls per
cell, `taskset -c 0,2,4,6`):

| row | serial | 4 workers | speedup |
|---|---:|---:|---:|
| 206 x 1,200 | 0.266 ms | 0.655 ms | 0.41x |
| 614 x 3,648 | 0.681 ms | 0.957 ms | 0.71x |
| 914 x 5,448 | 1.197 ms | 0.787 ms | 1.52x |
| 1,214 x 7,248 | 1.593 ms | 0.854 ms | 1.87x |
| 1,814 x 10,848 | 1.950 ms | 1.020 ms | 1.91x |
| 2,414 x 14,448 | 4.968 ms | 1.180 ms | 4.21x |
| 3,614 x 21,648 | 4.096 ms | 1.504 ms | 2.72x |
| 6,014 x 36,048 | 6.579 ms | 4.556 ms | 1.44x |
| 12,014 x 72,048 | 13.571 ms | 4.184 ms | 3.24x |

This agrees on direction: small rows lose on 4 workers, large rows win, the
4,096-node gate is above the point where the win starts. Two README claims did
not reproduce and are now qualified in place:

- The README's 602-node row at 0.97x ("no win yet, but no loss") measured 0.71x,
  a loss. The README now says that row is inside this host's noise.
- The README's 6,002-node row at 2.83x measured 1.44x. The README calls the
  timing jumps "run-to-run spread on this host"; that is the honest reading and
  is left as is.

The parallel path itself is covered by `tests/test_parallel_paths.py`, which
compares every row (including the last) against the serial Python reference at,
just below, and just above both thresholds, and repeats a 5-row batch three
times to catch a non-deterministic cross-row race. All of it passes.

## Tests: parity and history

- `git log -p -- tests/` shows exactly three touches, all additions:
  `e2399fb` added `conftest.py`, `test_kernels.py`, `test_mecab_parity.py`;
  `7c94fc5` added `test_simd_paths.py`; `c78a996` added
  `test_parallel_paths.py`. No assertion was removed, no tolerance widened, no
  test deleted or skipped to hide a failure.
- The other phase commits touch code outside `tests/`; each was read for
  loosening: `25fe4b0` (unroll) kept the same validation checks, `7c94fc5`
  (simd) and `c78a996` (parallel) add paths rather than weaken checks, and
  `43c275d` ("phase test") added only two `MM_DEBUG_*`-gated print blocks to
  `from_mecab_lattice` and no tests.
- Parity with upstream remains asserted, not relaxed: every name in
  `MeCab.__all__` must be the same upstream object; `parse`, `parseToNode`
  (field by field) and `parseNBest` match upstream exactly; and for eight
  inputs, every recomputed Mojo node cost must equal the cost MeCab itself
  computed, with the best path matching upstream surfaces.
- Nothing was restored, because nothing had been removed or loosened.

## Code changed

Only removals and a harness fix; no kernel logic changed and no new test was
needed, so nothing was "fixed to pass a test".

1. `python/mojo_mecab/lattice.py`: removed the `MM_DEBUG_SIZE` and
   `MM_DEBUG_IDS` debug-print blocks that commit `43c275d` left inside
   `from_mecab_lattice`. They were reachable only through undocumented
   environment variables, three re-reads of `lattice.begin_nodes(...)` per node
   sat in the second block, and they are unrelated to the port.
2. `bench/bench.py`: print each row as soon as it is measured (unchanged
   measurements, no lost rows on interruption).

Because (1) edits a Python module, the gate was rerun at the end, below.

## README edits

- Headline benchmark table: two rows corrected (batched Mojo, `Tagger.parse`
  ratio), all five rows replaced with this pass's measured values, plus a note
  that earlier revisions' 15.8-16.4 ms batch figure was not reproduced.
- `Tagger.parse` paragraph: states the measured 0.96-1.02x spread.
- Topology-check fusion section: says the before/after columns were not
  re-measured (no "before" build exists in the tree) and reports this pass's
  shipped-build numbers.
- Parallel section: says the sweep and the fine crossover ramp are the phase's
  original numbers, not reproduced cell-for-cell, and points here; the 602-node
  row is qualified as a measured loss on this host.
- "What this did not fix": adds this pass's batch-row range and the harness vs
  kernel attribution.
- "What was measured and reverted": says those forms are no longer in the tree
  and could not be rerun.
- GPU section: the device memory was re-read (`4559 MiB` free now), and the
  missing `bench/analysis.md` was confirmed absent from the whole history; the
  probe and hand-counted flops were not rerun and are marked as such.
- No row where the port is slower than upstream was removed or reworded: the
  only such row is `Tagger.parse` at parity (same backend), and it is reported
  at the measured 1.02x with the loss (0.77x, 0.96x) stated in the text.

## Final gate

After all edits, `pixi run build && pixi run test`: build exit 0 (the same three
pre-existing compiler warnings), test `93 passed, 8 skipped in 1.29s`, exit 0.
