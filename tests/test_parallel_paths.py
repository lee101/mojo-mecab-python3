"""Serial and parallel batch decoding must produce the same answer.

``mm_viterbi_decode_batch`` switches to ``max.algorithm.parallelize`` once a row
has ``PARALLEL_NODE_THRESHOLD`` nodes or more (see "Parallel batch decoding" in
README.md). The rows are independent, so the only thing that can go wrong is a
row boundary: a worker reading another row's costs, or writing another row's
totals and backpointers. These tests therefore cross the threshold and compare
every row, including the last, against the serial Python reference, which knows
nothing about threading.
"""

from __future__ import annotations

import numpy as np
import pytest

import mojo_mecab as mm

# The shipped thresholds, mirrored here because the constants are compile-time
# inside the Mojo source. If these drift, the tests stop covering the switch.
PARALLEL_NODE_THRESHOLD = 4096
PARALLEL_EDGE_THRESHOLD = 128
MAX_WORKERS = 4


def chain_lattice(node_count: int):
    """A narrow, highly ambiguous lattice of exactly ``node_count`` nodes.

    Every node after the first takes edges from the previous three nodes. That
    is the shape the threshold is measured on: cost ~ nodes, predecessor reads
    scattered across recent rows, and ties that decide the path.
    """
    incoming = [[]]
    for node in range(1, node_count - 1):
        first = max(0, node - 3)
        incoming.append([(pred, (node * 7 + pred) % 11 - 5) for pred in range(first, node)])
    incoming.append([(pred, 0) for pred in range(max(0, node_count - 4), node_count - 1)])
    lattice = mm.make_lattice(incoming)
    assert lattice.node_count == node_count
    return lattice


def check_batch(lattice, weights) -> None:
    batch = mm.decode_batch(lattice, weights)
    for index, row in enumerate(weights):
        expected = mm.reference_decode(lattice, row)
        assert batch.paths[index] == expected.path, index
        assert int(batch.costs[index, -1]) == expected.cost, index
        np.testing.assert_array_equal(batch.costs[index], expected.costs)
        np.testing.assert_array_equal(batch.backpointers[index], expected.backpointers)


@pytest.mark.parametrize(
    "node_count", [PARALLEL_NODE_THRESHOLD, PARALLEL_NODE_THRESHOLD + 1]
)
def test_batch_at_threshold_matches_reference(node_count):
    """The threshold itself and the first size above it switch to parallel."""
    lattice = chain_lattice(node_count)
    rng = np.random.default_rng(node_count)
    weights = rng.integers(-50, 200, size=(2 * MAX_WORKERS + 1, lattice.edge_count))
    check_batch(lattice, weights.astype(np.int64))


@pytest.mark.parametrize("node_count", [64, PARALLEL_NODE_THRESHOLD - 1])
def test_batch_below_threshold_matches_reference(node_count):
    """Sizes below the threshold stay serial and still agree."""
    lattice = chain_lattice(node_count)
    rng = np.random.default_rng(node_count)
    weights = rng.integers(-50, 200, size=(3, lattice.edge_count))
    check_batch(lattice, weights.astype(np.int64))


def wide_eos_lattice(edge_count: int):
    """Few nodes, but EOS gathers ``edge_count`` edges: the edge-only gate.

    Node count stays far below ``PARALLEL_NODE_THRESHOLD``, so only the edge
    threshold can switch this batch to the parallel path. Edges alternate
    between two predecessors and the row weights are random with ties, so a
    worker reading another row's cost slice would pick a different path.
    """
    # Each interior node adds exactly one edge, so `edge_count` is reached by a
    # short chain and EOS's slice is `edge_count - chain_edges`.
    incoming = [[]]
    chain_nodes = min(edge_count // 2, PARALLEL_NODE_THRESHOLD // 2)
    for node in range(1, chain_nodes):
        incoming.append([(node - 1, (node * 3) % 5 - 2)])
    chain_edges = sum(len(edges) for edges in incoming)
    node = len(incoming)
    incoming.append([(1 if edge % 2 else node - 1, (edge * 7) % 9 - 4)
                     for edge in range(edge_count - chain_edges)])
    lattice = mm.make_lattice(incoming)
    assert lattice.node_count < PARALLEL_NODE_THRESHOLD
    assert lattice.edge_count == edge_count
    return lattice


@pytest.mark.parametrize("edge_count", [PARALLEL_EDGE_THRESHOLD, PARALLEL_EDGE_THRESHOLD - 1])
def test_batch_edge_gate_matches_reference(edge_count):
    """The edge threshold alone decides, at and just under it."""
    lattice = wide_eos_lattice(edge_count)
    rng = np.random.default_rng(edge_count)
    weights = rng.integers(-50, 200, size=(2 * MAX_WORKERS + 1, lattice.edge_count))
    check_batch(lattice, weights.astype(np.int64))


def test_parallel_batch_identical_across_repeat_calls():
    """Two parallel calls on the same input must be bit-identical.

    A cross-row race need not fail the reference comparison if it is benign on
    one interleaving; repeating with an odd batch size (so the last of five rows
    can land on any worker) makes the comparison catch a non-deterministic one.
    """
    lattice = chain_lattice(PARALLEL_NODE_THRESHOLD)
    rng = np.random.default_rng(PARALLEL_NODE_THRESHOLD)
    weights = rng.integers(
        -50, 200, size=(MAX_WORKERS + 1, lattice.edge_count), dtype=np.int64
    )
    first = mm.decode_batch(lattice, weights)
    for _ in range(2):
        again = mm.decode_batch(lattice, weights)
        assert again.paths == first.paths
        np.testing.assert_array_equal(again.costs, first.costs)
        np.testing.assert_array_equal(again.backpointers, first.backpointers)
