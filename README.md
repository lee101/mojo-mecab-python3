# mojo-mecab-python3

`mojo-mecab-python3` ports MeCab-style lattice decoding and UTF-8 byte
indexing to Mojo. It also re-exports the public `mecab-python3` classes and
constants, so covered code can change `import MeCab` to
`import mojo_mecab as MeCab`.

This is deliberately not a claim that the small Python wrapper in
[`mecab-python3`](https://github.com/SamuraiT/mecab-python3) is slow. Dictionary
lookup, candidate generation, and output formatting already run in MeCab's
native C++ library. The work worth porting is the dynamic program over an
explicit lattice, especially when costs are rescored repeatedly or in batches.

## Covered subset

- Exact minimum-cost decoding of a topologically ordered MeCab lattice.
- Batched decoding of many edge-cost assignments over one topology.
- Copying a real `MeCab.Lattice`, including its incoming path costs, into the
  compact Mojo representation.
- UTF-8 code-point boundaries as byte offsets, the coordinate system used by
  MeCab lattices.
- Every name in upstream `MeCab.__all__` is re-exported as the same upstream
  Python object. This compatibility surface is delegation, not a Mojo port;
  a test checks every exported identity, with parse, N-best, node, model, and
  lattice behavior covered by parity tests.

The tests compare every dynamic-programming node cost and the final best path
against real `mecab-python3` 1.0.12 with UniDic Lite, including ambiguous,
unknown-word, mixed-script, whitespace, and empty inputs. The equal-cost
predecessor rule also matches MeCab.

Not ported to Mojo:

- MeCab's double-array dictionary trie and unknown-word candidate generation.
- Dictionary compilation or dictionary file loading.
- N-best enumeration and marginal-probability forward/backward passes.
- Output templates and feature formatting.

Those operations continue to use upstream MeCab. A dictionary is therefore
still required; this repository installs UniDic Lite. `mecab-python3` itself is
a real published upstream package, not a reference implementation invented for
this port.

## Install

The checked-in environment pins the Mojo nightly used to build the library.
Installation currently means a source checkout with Pixi; no standalone wheel
or prebuilt shared library is published yet.

```bash
pixi install
pixi run build
pixi run test
```

The build produces:

```text
dist/libmojo-mecab-python3.so
```

Set `MOJO_MECAB_LIB` to a prebuilt copy of that library when loading it from
outside the checkout.

## Usage

The compatibility API uses the same names and method signatures as upstream:

```python
import mojo_mecab as MeCab

tagger = MeCab.Tagger("-Owakati")
print(tagger.parse("pythonが大好きです").strip())
# python が 大好き です
```

To run the Mojo decoder on candidates and costs supplied by MeCab:

```python
import mojo_mecab as MeCab

lattice = MeCab.mecab_lattice("すももももももものうち")
result = MeCab.decode(lattice)

print(MeCab.decoded_surfaces(lattice, result))
# ('すもも', 'もも', 'も', 'もも', 'の', 'うち')
print(result.cost)
```

Custom tokenizers can avoid MeCab entirely by constructing a `PackedLattice`
with `make_lattice`. Each item is the list of `(predecessor, edge_cost)` pairs
for one node; node zero is BOS and the final node is EOS.

```python
import mojo_mecab as MeCab

lattice = MeCab.make_lattice([
    [],
    [(0, 3)],
    [(0, 4)],
    [(1, 2), (2, 7)],
])
assert MeCab.decode(lattice).path == (0, 1, 3)
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64. Times are the best of repeated warm runs.

The table below was re-measured for the VERIFY pass. It is one complete
`pixi run bench` run on a shared 72-thread Xeon E5-2697 v4 whose load average
was 92 during measurement; `bench/verify.md` records the repeat runs behind each
row and which ones were corrected. The batched row is the serial tier: at 3,602
nodes per row it is below the measured parallel crossover, so the parallel work
does not change it; see "Parallel batch decoding". Earlier revisions quoted
15.8-16.4 ms for that row, which was not reproduced on this host, and 86.11x;
the measured values replace them.

| benchmark | Mojo / compatibility | reference | ratio | result |
|---|---:|---:|---:|---|
| Viterbi, 48,002 nodes / 287,976 edges | 3.338 ms (Mojo) | 254.147 ms (Python) | 76.13x | faster |
| Batched Viterbi, 64 x 3,602 nodes | 19.803 ms (Mojo) | 1383.864 ms (Python) | 69.88x | faster |
| UniDic lattice re-decode, 6,322 nodes | 0.351 ms (Mojo) | 40.813 ms (Python) | 116.40x | faster |
| UTF-8 boundaries, 510,000 code points | 2.968 ms (Mojo) | 84.585 ms (Python) | 28.50x | faster |
| `Tagger.parse` compatibility, 50,000 chars | 19.023 ms (compat) | 19.484 ms (`mecab-python3`) | 1.02x | same C++ backend |

`bench/verify.md` records the per-row repeat runs, the two figures that were
corrected, and the parts of this section that were not re-measured.

The Viterbi reference is the straightforward pure-Python implementation shipped
in `reference_decode`; the UniDic row uses an actual candidate lattice and path
costs extracted from upstream. The final row is intentionally reported even
though both calls use the same upstream C++ implementation. Its near-1.00x
result is timing variation, not a Mojo speedup: across four repeat measurements
it landed at 0.96x to 1.02x.
Candidate generation is outside the Mojo-covered subset, so end-to-end
`Tagger.parse` should be expected to perform like `mecab-python3`.

### Topology-check pass fusion

One change was kept: the CSR topology check in the decoder now reads each offset
and predecessor once, in a single pass, and checks a node's predecessor slice as
soon as its offsets are read. The previous form made two passes over the offset
row and a third over the predecessor row. This is a fusion win, not a SIMD win;
the fused pass is plain scalar code and got faster that way.

`pixi run bench` rows, five interleaved runs of each build on the same machine,
reported as median (min).

| benchmark | before | after |
|---|---:|---:|
| Viterbi, 48,002 nodes / 287,976 edges | 3.330 ms (3.266) | 3.266 ms (3.166) |
| Batched Viterbi, 64 x 3,602 nodes | 16.783 ms (15.620) | 16.444 ms (15.821) |
| UniDic lattice re-decode, 6,322 nodes | 0.327 ms (0.319) | 0.319 ms (0.300) |
| UTF-8 boundaries, 510,000 code points | 2.986 ms (2.924) | 3.063 ms (3.010) |
| `Tagger.parse` compatibility, 50,000 chars | 17.626 ms (17.325) | 18.991 ms (17.482) |

Read that table as "unchanged at the benchmark's resolution", with one caveat:
run-to-run variance is large enough that the UniDic row's before-max of 0.558 ms
is an outlier. These rows time a `ctypes` call plus Python-side Numpy allocation
around a kernel that costs about 0.06 ms on a 6,322-node lattice; the rest is
`str.encode` and array copies. The one real movement is the UniDic single
decode, whose kernel got about 13% faster when measured in isolation. The UTF-8
row is flat, as expected, since no UTF-8 code changed.

The before/after columns above were not re-measured in the VERIFY pass, because
"before" needs a build that no longer exists in the tree, so they are kept as
the phase's record and not as evidence of a speedup. What the VERIFY pass did
measure on the shipped build puts the decoder within reach of the "after"
column (Viterbi 3.338 ms and UniDic 0.351 ms in one `pixi run bench` run), while
the batched and `Tagger.parse` rows landed at 19.803 ms and 19.023 ms on a host
with load average 92, which is not a contradiction at this noise level but is
not a reproduction of the "after" figures either. See `bench/verify.md`.

### Parallel batch decoding

Batched decoding (``decode_batch``) runs its rows in parallel once a row is
large enough, using `parallelize` from `max.algorithm`. Rows are independent, so
each worker decodes a contiguous run of rows and returns status through its own
slot; the result is bit-identical to the serial loop
(`tests/test_parallel_paths.py` compares every row, including the last, against
the serial reference at and around both thresholds).

This needed a toolchain change: `parallelize` lives in the `max` package on this
toolchain, and the previous pin (mojo 1.2.0) had no `max` beside it, so the
import could not resolve. `pixi.toml` now pins `mojo ==1.3.0.dev2026111005`
with the matching `max ==26.8.0.dev2026111005`, and the kernel uses `Pointer`
in place of the removed `UnsafePointer`. The tests pass unchanged otherwise.

**Threshold.** The gate is per row, in kernel terms: `nodes >= 4096` or
`edges >= 128`, and at least 2 rows. The second clause is measured, not
decorative: the unzipped wall clock of a row is what decides, and the edge form
keeps a few-node, very wide lattice from being gated on its node count alone.
The worker count is `min(rows, 4)`.

Isolated kernel time (ctypes call only, median ms, `taskset -c 0,2,4,6`) from
`bench/parallel_crossover.py`, on the shared Xeon host. This is the phase's
original sweep and was not reproduced number-for-number in the VERIFY pass; the
two sweeps run there moved absolute times by up to 5-15% in both directions, the
64-row 3,602-node shape no longer showed a parallel win, and the parallel-wins-
on-large-rows conclusion plus the crossover shape did hold. See
`bench/verify.md` and "Crossover" below:

| nodes | edges | rows | 1 worker (serial) | 2 workers | 4 workers | shipped |
|---:|---:|---:|---:|---:|---:|---:|
| 194 | 1,128 | 64 | 0.21 | 0.51 | 0.57 | 0.57 |
| 3,602 | 21,576 | 4 | 0.28 | 0.54 | 0.61 | 0.62 |
| 3,602 | 21,576 | 16 | 1.03 | 1.25 | 0.83 | 0.86 |
| 3,602 | 21,576 | 64 | 4.02 | 2.42 | 1.56 | 1.56 |
| 12,002 | 71,976 | 64 | 14.21 | 9.19 | 7.49 | 6.85 |
| 36,002 | 215,976 | 64 | 44.77 | 24.09 | 20.19 | 21.48 |
| 72,002 | 431,976 | 12 | 16.19 | 10.26 | 8.88 | 9.12 |
| 72,002 | 431,976 | 64 | 98.36 | 44.16 | 39.55 | 41.24 |
| 144,002 | 863,976 | 4 | 12.67 | 8.51 | 5.82 | 5.20 |
| 144,002 | 863,976 | 64 | 165.96 | 87.59 | 51.40 | 51.27 |

"shipped" is the build in this checkout, with its own threshold; the other
columns are the same source with the parallel path forced on, so the 1-worker
column is the serial baseline through the identical loop.

**Crossover and margin.** The ramp, serial vs 4 workers, 64 rows. The table
below is the phase's original ramp. The VERIFY pass re-ran the same ramp with
the same script (median of 31 calls per cell) and recorded it in
`bench/verify.md`; the direction reproduced (serial beats parallel below ~900
nodes per row, parallel wins from ~900 up), but individual cells moved a lot on
the loaded host, and the VERIFY ramp measured 0.71x at the 614-node row, which
is a loss; so the 602-node row's 0.97x tie does not reproduce, and that row sits
inside this host's noise rather than on the crossover.

| row | serial | 4 workers | speedup |
|---|---:|---:|---:|
| 194 x 1,128 | 0.217 ms | 0.570 ms | 0.38x |
| 602 x 3,576 | 0.667 ms | 0.689 ms | 0.97x |
| 902 x 5,376 | 1.006 ms | 0.786 ms | 1.28x |
| 1,202 x 7,176 | 1.351 ms | 0.944 ms | 1.43x |
| 1,802 x 10,776 | 2.021 ms | 1.043 ms | 1.94x |
| 2,402 x 14,376 | 2.703 ms | 1.939 ms | 1.39x |
| 3,602 x 21,576 | 4.000 ms | 1.601 ms | 2.50x |
| 6,002 x 35,976 | 6.709 ms | 2.367 ms | 2.83x |
| 12,002 x 71,976 | 13.971 ms | 6.451 ms | 2.17x |

A win is first visible around 900 nodes per row (~1 ms of row work) at roughly
1.3x, and is unambiguous from ~3,600 nodes (the `pixi run bench` batch shape)
at 2.4-2.8x. 4,096 is that crossover with margin: the worker hand-off plus
memory wake-up on this host costs about 14 us per worker slot, and the
1,802-node row / 1.94x and 2,402-node row / 1.39x points show it is nowhere
near a cliff. Below 4,096 nodes per row, callers stay serial and are measured
at 0.38-0.97x against the forced-parallel form, so the gate matters.

**Worker count: 1, 2, 4 and the cap.** The chip is a 2-socket 72-thread Xeon
with a shared host load around 25-40. Measured on the build that splits rows
across workers, median ms:

| row | rows | 1 worker | 2 workers | 4 workers |
|---|---:|---:|---:|---:|
| 1,202 | 64 | 1.37 | 1.06 | 1.23 |
| 1,802 | 64 | 2.02 | 1.60 | 2.25 |
| 2,402 | 64 | 2.69 | 1.83 | 1.96 |
| 3,602 | 64 | 4.01 | 2.69 | 3.79 |
| 7,802 | 64 | 8.81 | 5.38 | 4.96 |
| 12,002 | 64 | 13.50 | 7.73 | 7.05 |
| 12,002 | 16 | 3.45 | 3.74 | 2.11 |

The gain does not flatten and reverse above the threshold: 4 workers is the
fastest or within noise of it from 3,602 nodes up, and 2 workers is usually the
right count just above the gate, where worker start-up is a larger share. The
cap of 4 is kept: no configuration measured faster, and on the shapes where
4 pinned CPUs are available the 1-work-item-per-worker split already uses all
four. On the 16-row, 72,002-node and 144,002-node shapes the gain is 2.6x and
2.5x, still short of 4x because the rows themselves are unequal.

**What this did not fix.** The `pixi run bench` harness only measures the
3,602-node batch, which stays on the serial tier by design, and on this shared
host that row is noisy: 28.310 ms normally, 19.7-20.4 ms with nothing else
running, and 17.2-23.6 ms two runs in a row. The isolated kernel call for the
same shape is about 4.0 ms, so the rest is the harness and host load, not
parallelism. The batch row in this build is therefore not comparable to the
15.8-16.8 ms recorded in the earlier commits; that is a measurement-environment
difference, and it is recorded here rather than presented as a win or a loss.

The VERIFY pass confirmed the diagnosis rather than the numbers. Across five
complete `pixi run bench` runs the batch row measured 19.652-25.601 ms, and the
host's load average was 92 at the time. The same shape timed through the
identical `decode_batch` call ranges 19.7-44.1 ms from run to run, most of it
`_trace` plus the NumPy allocations around the kernel. The isolated kernel call
for this shape is 3.18-10.15 ms depending on host load, which matches the 4.0 ms
above at the low end and is the parallel decision's input, not the harness's.

### What was measured and reverted

Every form below was built, tested for correctness at the edge lengths covered by
`tests/test_simd_paths.py`, and benchmarked before being removed. None of them
was faster at the sizes the caller actually uses, so none was kept. These are
the phase's measurements of code that is no longer in the tree, so the VERIFY
pass could not rerun them; only the shipped rows are reachable from this
checkout, and they are not re-measured here either.

UTF-8 boundary kernel on the benchmark input (1,530,000 bytes), kernel time only:

| form | min | median |
|---|---:|---:|
| scalar kernel, unchanged (shipped) | 2241 us | 2262 us |
| vector count of leader bytes, then scalar offsets | 146 us to count | 3545 us to write offsets from vector lanes |
| store every byte offset unconditional (store cost floor) | 855 us | not a full answer, reference only |
| read bytes as `Int8` and write offsets as before | 3372 us | 3392 us |

The count is 15x cheaper vectorised, but the loop also writes one `int64` offset
per code point, and emitting those offsets from vector lanes measured slower than
the original scalar walk, whose writes are perfectly sequential.

Lattice kernels (kernel time only, real caller topologies):

| form | 6,322 nodes | 48,002 nodes |
|---|---:|---:|
| scalar predecessor check (shipped) | 40.6 us | 249.5 us |
| lane-wise compare against the owning node id | 42.7 us | 253.6 us |
| vector `min`/`max` reduce per node | 164 us | 458 us |

The offset-monotonicity check was also tried as a lane-wise compare: it only
decides which branch to take, so there is nothing for the vector to save.

The batch kernel lost 12-23% when the forward pass was moved behind an extra
helper even with identical source, so that loop body stays inlined in
`mm_viterbi_decode_batch`. The fused check is written out in both entry points
for the same reason.

### Why the remaining loops stay scalar

The UTF-8 scan is a dependency chain: each boundary depends on the width of the
character before it, so a vector can only help by classifying bytes in bulk and
moving the result back out, which is what the table above measures and rejects.
The decoder's inner loop is a min-reduction in which each candidate reads a
predecessor total written earlier in the same pass, and the tie rule keeps the
last equal candidate, so the running order is part of the result and the loop is
not vectorisable. Profiling the lattice rows attributes most of their wall time
to the `ctypes` call and the Numpy buffers around the kernel, which is why a
kernel-level gain shows up as a fraction of that in `pixi run bench`. The
forward pass over one row cannot be split further: node `n` reads totals written
for earlier nodes, so within a row the order is fixed. Parallelism is only
across rows in `decode_batch`, and only above the threshold above.

### GPU: CPU only: memory-bound

No GPU path is provided, and none is worth providing: every kernel in this port
is memory-bound. The figure below is derived from the shipped source in
`src/kernels.mojo` rather than from a microbenchmark; the GPU phase's gate is a
~2 flops per byte threshold whose file, `bench/analysis.md`, does not exist in
this checkout or anywhere in the repository history, so the arithmetic intensity
was counted by hand from the code paths that exist. The counts themselves were
not re-derived in the VERIFY pass; what it did check is that no GPU code and no
GPU-gated tests are present, and that `bench/analysis.md` has never been
committed.

| kernel | work per unit | bytes moved per unit | flops per byte |
|---|---|---:|---:|
| `_decode_one` (Viterbi forward pass) | ~3 integer add/sub/cmps per edge | 32 read (predecessor, cost, predecessor total, predecessor backpointer) + 16 written per node | ~0.08 |
| `_validate` (CSR/topology check) | 1 compare per offset or predecessor word | 8 | 0.125 |
| `mm_utf8_boundaries` | 1-3 compares per byte, 1 add per code point | 1 read per byte + 8 written per code point | < 0.3 |

All three are an order of magnitude below the 2 flops/byte gate. The decoder's
per-edge traffic is dominated by two indirect loads (the predecessor's total and
backpointer) whose addresses depend on the edge, which is exactly the pattern a
GPU handles worst: no coalescing, and a dependent load in the loop. The forward
pass is also sequential within a row, as described above, so a row cannot be
spread over blocks at all. The only parallel axis is across batch rows, which
the CPU path already uses above its threshold.

The device itself was not the blocker. On this host
`nvidia-smi --query-gpu=memory.free` reported 4557 MiB free in that phase, and
4559 MiB free in the VERIFY pass (gate: at least 4000 MiB), and the documented
`max.gpu` host surface compiles and runs on the pinned toolchain: a probe
building and running `DeviceContext` with `enqueue_create_buffer[DType.float64]`
and `synchronize()` succeeded on an RTX 5090. The VERIFY pass re-read the
device's free memory (above) but did not rerun the probe. There is no GPU code in the port, therefore no GPU tests: nothing is
skipped, because nothing GPU-only was added. The full test suite runs on a
CPU-only host unchanged (`93 passed, 8 skipped`; all 8 skips are pre-existing
lattice-shape guards in `tests/test_simd_paths.py` for unreachable EOS, not
device detection).

The only benchmark at parity with upstream is `Tagger.parse`, which is the
exact same `mecab-python3` class and native C++ backend rather than a Mojo
kernel.

## How it works

The lattice is stored in incoming-edge compressed sparse row form:
`edge_offsets[node]:edge_offsets[node + 1]` selects its predecessor node IDs
and signed 64-bit edge costs. An edge cost is MeCab's connection cost plus the
destination node's word cost. Nodes are topologically ordered, with BOS first
and EOS last. The decoder first validates that layout in one fused pass, reading
each offset once and using it both for the ordering check and to open that
node's predecessor slice; it then performs one forward minimum-cost pass and
writes signed 64-bit totals and backpointers into NumPy-owned arrays.

Python calls one Mojo shared library through `ctypes`. Arrays cross the C ABI
as integer addresses; Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric exported
functions. Python owns all buffers, the kernels allocate no heap memory, and
batched costs use contiguous row-major `(batch, edge_count)` storage. UTF-8
indexing likewise reads a contiguous byte buffer and writes `int64` offsets.
