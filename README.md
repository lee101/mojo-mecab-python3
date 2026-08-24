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

| benchmark | Mojo / compatibility | reference | ratio | result |
|---|---:|---:|---:|---|
| Viterbi, 48,002 nodes / 287,976 edges | 3.543 ms (Mojo) | 267.985 ms (Python) | 75.63x | faster |
| Batched Viterbi, 64 x 3,602 nodes | 17.879 ms (Mojo) | 1333.131 ms (Python) | 74.56x | faster |
| UniDic lattice re-decode, 6,322 nodes | 0.367 ms (Mojo) | 40.635 ms (Python) | 110.81x | faster |
| UTF-8 boundaries, 510,000 code points | 3.075 ms (Mojo) | 86.197 ms (Python) | 28.03x | faster |
| `Tagger.parse` compatibility, 50,000 chars | 19.705 ms (compat) | 19.190 ms (`mecab-python3`) | 0.97x | same C++ backend |

The Viterbi reference is the straightforward pure-Python implementation shipped
in `reference_decode`; the UniDic row uses an actual candidate lattice and path
costs extracted from upstream. The final row is intentionally reported even
though both calls use the same upstream C++ implementation. Its near-1.00x
result is timing variation, not a Mojo speedup.
Candidate generation is outside the Mojo-covered subset, so end-to-end
`Tagger.parse` should be expected to perform like `mecab-python3`.

No SIMD or parallel path was added: every Mojo-owned kernel is already more
than 5x ahead of its reference, while profiling the parity row attributes all
time directly to the shared upstream `MeCab.Tagger.parse` method. There is no
Mojo loop on that path to optimize.

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
and EOS last. The decoder performs one forward minimum-cost pass and writes
signed 64-bit totals and backpointers into NumPy-owned arrays.

Python calls one Mojo shared library through `ctypes`. Arrays cross the C ABI
as integer addresses; Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside non-parametric exported
functions. Python owns all buffers, the kernels allocate no heap memory, and
batched costs use contiguous row-major `(batch, edge_count)` storage. UTF-8
indexing likewise reads a contiguous byte buffer and writes `int64` offsets.
