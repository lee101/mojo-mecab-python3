"""Edge-shape coverage for the fused lattice topology check.

``mm_viterbi_decode`` validates the CSR topology in a single pass: the offset
row must start at zero, end at ``edge_count`` and never decrease, and each
destination's predecessor slice may only name earlier nodes. These tests cover
the edge counts that a chunked or vectorised version would have to split into
body and tail (0, 1, W-1, W, W+1 and a large odd count), and prove the check
accepts valid lattices and refuses damaged ones at every one of them.

``mojo_mecab._lib`` is imported explicitly because ``mojo_mecab`` re-exports
upstream ``MeCab.__all__``, so ``mojo_mecab._lib`` is not the submodule.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_mecab as mm
from mojo_mecab import _lib

# Shapes are parametrised on W so every width a vectorised version might pick is
# covered. The shipped kernels do not vectorise these loops: every SIMD form was
# measured slower at the caller's lattice sizes and was reverted.
W = 4

EDGE_COUNTS = [0, 1, W - 1, W, W + 1, 4 * W + 3]


def fan_predecessors(edge_count: int):
    """Alternate BOS (0) and node 1, both legal predecessors of any later node."""
    return [(1 if index % 2 else 0, (index % 5) - 2) for index in range(edge_count)]


def fan_lattice(edge_count: int):
    """BOS -> node 1 -> EOS, where EOS receives ``edge_count`` edges.

    EOS is last, so its slice is the wide one; with no incoming BOS edges on
    node 1, EOS still receives its single edge from node 1 and stays reachable.
    """
    return mm.make_lattice(
        [[], [(0, (edge_count % 5) - 2)], fan_predecessors(edge_count)]
    )


def eos_slice(lattice) -> int:
    offsets = lattice.edge_offsets
    return int(offsets[-1] - offsets[-2])


def call_kernel(edge_offsets, predecessors, edge_costs) -> int:
    """Call the exported decoder directly, bypassing ``PackedLattice``."""
    node_count = edge_offsets.size - 1
    totals = np.zeros(node_count, dtype=np.int64)
    backpointers = np.zeros(node_count, dtype=np.int64)
    return _lib.lib().mm_viterbi_decode(
        node_count,
        predecessors.size,
        _lib.addr(edge_offsets),
        _lib.addr(predecessors),
        _lib.addr(edge_costs),
        _lib.addr(totals),
        _lib.addr(backpointers),
    )


def damaged(edge_count: int, lane: int, value: int) -> int:
    """Rewrite one EOS predecessor to an id EOS must reject.

    ``make_lattice`` and ``PackedLattice`` both validate in Python, so the
    damaged arrays go straight to the exported kernel to exercise its check.
    """
    lattice = fan_lattice(edge_count)
    predecessors = lattice.predecessors.copy()
    predecessors[lattice.edge_offsets[2] + lane] = value
    return call_kernel(lattice.edge_offsets, predecessors, lattice.edge_costs)


@pytest.mark.parametrize("edge_count", EDGE_COUNTS)
def test_wide_last_node_matches_reference(edge_count):
    if edge_count == 0:
        pytest.skip("EOS with no incoming edges is unreachable by definition")
    lattice = fan_lattice(edge_count)
    assert eos_slice(lattice) == edge_count
    mojo = mm.decode(lattice)
    reference = mm.reference_decode(lattice)
    assert mojo.path == reference.path
    assert mojo.cost == reference.cost
    np.testing.assert_array_equal(mojo.costs, reference.costs)


@pytest.mark.parametrize("edge_count", EDGE_COUNTS)
def test_wide_last_node_batch_matches_reference(edge_count):
    if edge_count == 0:
        pytest.skip("EOS with no incoming edges is unreachable by definition")
    lattice = fan_lattice(edge_count)
    rng = np.random.default_rng(edge_count)
    weights = rng.integers(-40, 90, size=(5, lattice.edge_count), dtype=np.int64)
    batch = mm.decode_batch(lattice, weights)
    for index, row in enumerate(weights):
        expected = mm.reference_decode(lattice, row)
        assert batch.paths[index] == expected.path
        np.testing.assert_array_equal(batch.costs[index], expected.costs)


@pytest.mark.parametrize("lane", ["first", "middle", "last"])
@pytest.mark.parametrize("bad_value", [2, -1])
@pytest.mark.parametrize("edge_count", EDGE_COUNTS)
def test_bad_predecessor_is_rejected(edge_count, bad_value, lane):
    """An id that is too large, or negative, must be refused wherever it sits.

    ``2`` is EOS's own id: legal for every earlier node, so only the owning
    node's check can reject it, which is exactly what is under test.
    """
    if edge_count == 0:
        pytest.skip("no EOS edges to damage")
    position = {"first": 0, "middle": edge_count // 2, "last": edge_count - 1}[lane]
    assert damaged(edge_count, position, bad_value) == 0


@pytest.mark.parametrize("edge_count", [W, W + 1, 4 * W + 3])
def test_decreasing_offsets_are_rejected(edge_count):
    """An interior offset that drops below its predecessor must be refused.

    Start and end offsets are kept valid, and node ids are chosen so every
    predecessor is in range, leaving the ordering check as the only reason to
    fail. A lattice shape with at least two wide nodes is built directly for
    this case rather than through ``make_lattice``, which would refuse it.
    """
    node_count = 5
    edges_per_node = [0, 0, edge_count, 1, 1]
    offsets = np.cumsum([0] + edges_per_node).astype(np.int64)
    predecessors = np.zeros(int(offsets[-1]), dtype=np.int64)
    costs = np.zeros(int(offsets[-1]), dtype=np.int64)
    start = int(offsets[2])
    predecessors[start:start + edge_count] = 1
    predecessors[start + edge_count] = 2
    predecessors[start + edge_count + 1] = 3
    # Bend the third offset below the second: monotonicity is now violated.
    offsets[3] = offsets[2] - 1
    status = call_kernel(offsets, predecessors, costs)
    assert status == 0
