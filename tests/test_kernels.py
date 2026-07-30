from __future__ import annotations

import numpy as np
import pytest

import mojo_mecab as mm


def test_known_shortest_path():
    lattice = mm.make_lattice(
        [[], [(0, 3)], [(0, 4)], [(1, 2), (2, 7)]],
        ["BOS", "a", "b", "EOS"],
    )
    result = mm.decode(lattice)
    assert result.path == (0, 1, 3)
    assert result.cost == 5
    np.testing.assert_array_equal(result.costs, [0, 3, 4, 5])


def test_negative_edge_costs_on_dag():
    lattice = mm.make_lattice(
        [[], [(0, 8)], [(0, 3)], [(1, -10), (2, 1)]]
    )
    assert mm.decode(lattice).path == (0, 1, 3)
    assert mm.decode(lattice).cost == -2


def test_equal_cost_uses_last_incoming_path_like_mecab():
    lattice = mm.make_lattice(
        [[], [(0, 2)], [(0, 3)], [(2, 4), (1, 5)]]
    )
    assert mm.decode(lattice).path == (0, 1, 3)


def test_mojo_matches_reference_on_dense_random_lattice():
    rng = np.random.default_rng(42)
    incoming = [[]]
    for node in range(1, 300):
        count = min(node, 9)
        predecessors = rng.choice(node, size=count, replace=False)
        incoming.append(
            [(int(pred), int(cost)) for pred, cost in zip(
                predecessors, rng.integers(-20, 80, size=count), strict=True
            )]
        )
    incoming[-1].append((298, 0))
    lattice = mm.make_lattice(incoming)
    mojo = mm.decode(lattice)
    reference = mm.reference_decode(lattice)
    assert mojo.path == reference.path
    assert mojo.cost == reference.cost
    np.testing.assert_array_equal(mojo.costs, reference.costs)
    np.testing.assert_array_equal(mojo.backpointers, reference.backpointers)


def test_batch_matches_individual_reference_decodes():
    lattice = mm.make_lattice(
        [[], [(0, 0)], [(0, 0)], [(1, 0), (2, 0)], [(3, 0)]]
    )
    rng = np.random.default_rng(7)
    weights = rng.integers(-50, 100, size=(32, lattice.edge_count))
    batch = mm.decode_batch(lattice, weights)
    for index, row in enumerate(weights):
        expected = mm.reference_decode(lattice, row)
        assert batch.paths[index] == expected.path
        np.testing.assert_array_equal(batch.costs[index], expected.costs)
        np.testing.assert_array_equal(
            batch.backpointers[index], expected.backpointers
        )


def test_zero_sized_batch():
    lattice = mm.make_lattice([[], [(0, 1)]])
    result = mm.decode_batch(lattice, np.empty((0, 1), dtype=np.int64))
    assert result.paths == ()
    assert result.costs.shape == (0, 2)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", [0]),
        ("ascii", [0, 1, 2, 3, 4, 5]),
        ("日本語", [0, 3, 6, 9]),
        ("A日 B", [0, 1, 4, 5, 6]),
        ("𠮷野家", [0, 4, 7, 10]),
        ("éガ", [0, 2, 5]),
    ],
)
def test_utf8_boundaries(text, expected):
    np.testing.assert_array_equal(mm.utf8_boundaries(text), expected)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: mm.PackedLattice([1, 1, 1], [], []),
        lambda: mm.PackedLattice([0, 0, 2], [0], [1]),
        lambda: mm.PackedLattice([0, 1, 1], [0], [1]),
        lambda: mm.PackedLattice([0, 0, 1], [2], [1]),
        lambda: mm.PackedLattice([0, 0, 1], [1], [1]),
        lambda: mm.PackedLattice([0, 0, 0], [], [], ["only one"]),
    ],
)
def test_invalid_lattices_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_unreachable_eos_is_rejected():
    lattice = mm.make_lattice([[], [(0, 1)], []])
    with pytest.raises(ValueError, match="unreachable"):
        mm.decode(lattice)


def test_bad_batch_shape_is_rejected():
    lattice = mm.make_lattice([[], [(0, 1)]])
    with pytest.raises(ValueError, match="shape"):
        mm.decode_batch(lattice, np.ones((3, 2), dtype=np.int64))


def test_unreachable_batch_is_rejected():
    lattice = mm.make_lattice([[], [(0, 1)], []])
    with pytest.raises(ValueError, match="unreachable"):
        mm.decode_batch(lattice, np.empty((1, 1), dtype=np.int64))


@pytest.mark.parametrize(
    "costs",
    [
        [1.5],
        np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        [np.iinfo(np.int64).max + 1],
    ],
)
def test_costs_are_not_silently_narrowed(costs):
    lattice = mm.make_lattice([[], [(0, 1)]])
    with pytest.raises((TypeError, OverflowError)):
        mm.decode(lattice, costs)


def test_path_cost_overflow_is_reported():
    maximum = np.iinfo(np.int64).max
    lattice = mm.make_lattice([[], [(0, maximum)], [(1, 1)]])
    with pytest.raises(OverflowError, match="path cost"):
        mm.decode(lattice)
