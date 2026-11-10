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

The table below is the current build. The rows that moved since the previous
commit, and the sizes of the moves, are in the next section.

| benchmark | Mojo / compatibility | reference | ratio | result |
|---|---:|---:|---:|---|
| Viterbi, 48,002 nodes / 287,976 edges | 3.286 ms (Mojo) | 269.630 ms (Python) | 82.06x | faster |
| Batched Viterbi, 64 x 3,602 nodes | 15.821 ms (Mojo) | 1362.326 ms (Python) | 86.11x | faster |
| UniDic lattice re-decode, 6,322 nodes | 0.305 ms (Mojo) | 41.729 ms (Python) | 136.91x | faster |
| UTF-8 boundaries, 510,000 code points | 3.043 ms (Mojo) | 86.586 ms (Python) | 28.46x | faster |
| `Tagger.parse` compatibility, 50,000 chars | 18.490 ms (compat) | 18.305 ms (`mecab-python3`) | 0.99x | same C++ backend |

The Viterbi reference is the straightforward pure-Python implementation shipped
in `reference_decode`; the UniDic row uses an actual candidate lattice and path
costs extracted from upstream. The final row is intentionally reported even
though both calls use the same upstream C++ implementation. Its near-1.00x
result is timing variation, not a Mojo speedup.
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

### What was measured and reverted

Every form below was built, tested for correctness at the edge lengths covered by
`tests/test_simd_paths.py`, and benchmarked before being removed. None of them
was faster at the sizes the caller actually uses, so none was kept.

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
kernel-level gain shows up as a fraction of that in `pixi run bench`.

No GPU path is provided. The lattice decoder performs roughly one integer add
per edge while loading an edge cost, predecessor index, and predecessor total;
UTF-8 indexing similarly does little arithmetic per byte read and offset
written. The only benchmark at parity with upstream is `Tagger.parse`, which is
the exact same
`mecab-python3` class and native C++ backend rather than a Mojo kernel.

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
