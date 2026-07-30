"""MeCab-compatible shortest-path decoding over an explicit word lattice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ._lib import addr, i64, lib

INF = np.int64(0x3FFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class DecodeResult:
    path: tuple[int, ...]
    cost: int
    costs: np.ndarray
    backpointers: np.ndarray


@dataclass(frozen=True)
class BatchDecodeResult:
    paths: tuple[tuple[int, ...], ...]
    costs: np.ndarray
    backpointers: np.ndarray


@dataclass(frozen=True)
class PackedLattice:
    """A topologically ordered lattice in incoming-edge CSR form.

    Node zero is BOS and the last node is EOS. Each edge cost must already
    contain the destination word cost plus its connection cost, matching
    MeCab's ``Path.cost`` convention.
    """

    edge_offsets: np.ndarray
    predecessors: np.ndarray
    edge_costs: np.ndarray
    labels: tuple[str, ...] = ()
    upstream_costs: np.ndarray | None = None

    def __post_init__(self) -> None:
        offsets = i64(self.edge_offsets)
        predecessors = i64(self.predecessors)
        costs = i64(self.edge_costs)
        object.__setattr__(self, "edge_offsets", offsets)
        object.__setattr__(self, "predecessors", predecessors)
        object.__setattr__(self, "edge_costs", costs)
        if offsets.ndim != 1 or offsets.size < 3:
            raise ValueError("a lattice needs BOS, EOS, and node_count + 1 offsets")
        if offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("edge_offsets must start at zero and be nondecreasing")
        if offsets[1] != 0:
            raise ValueError("BOS cannot have incoming edges")
        if int(offsets[-1]) != predecessors.size or costs.size != predecessors.size:
            raise ValueError("edge arrays do not match edge_offsets")
        node_count = offsets.size - 1
        if np.any(predecessors < 0) or np.any(predecessors >= node_count):
            raise ValueError("predecessor index out of range")
        for node in range(1, node_count):
            start, stop = int(offsets[node]), int(offsets[node + 1])
            if np.any(predecessors[start:stop] >= node):
                raise ValueError("nodes must be in topological order")
        if self.labels and len(self.labels) != node_count:
            raise ValueError("labels must have one entry per node")
        if self.upstream_costs is not None:
            upstream = i64(self.upstream_costs)
            if upstream.shape != (node_count,):
                raise ValueError("upstream_costs must have one entry per node")
            object.__setattr__(self, "upstream_costs", upstream)

    @property
    def node_count(self) -> int:
        return self.edge_offsets.size - 1

    @property
    def edge_count(self) -> int:
        return self.predecessors.size


def _trace(backpointers: np.ndarray) -> tuple[int, ...]:
    node = backpointers.size - 1
    reverse_path = [node]
    while node:
        node = int(backpointers[node])
        if node < 0:
            raise ValueError("EOS is unreachable from BOS")
        reverse_path.append(node)
        if len(reverse_path) > backpointers.size:
            raise ValueError("backpointer cycle")
    return tuple(reversed(reverse_path))


def decode(lattice: PackedLattice, edge_costs=None) -> DecodeResult:
    """Return the minimum-cost BOS-to-EOS path."""

    costs = lattice.edge_costs if edge_costs is None else i64(edge_costs)
    if costs.shape != (lattice.edge_count,):
        raise ValueError("edge_costs must have one value per edge")
    totals = np.empty(lattice.node_count, dtype=np.int64)
    backpointers = np.empty(lattice.node_count, dtype=np.int64)
    ok = lib().mm_viterbi_decode(
        lattice.node_count,
        lattice.edge_count,
        addr(lattice.edge_offsets),
        addr(lattice.predecessors),
        addr(costs),
        addr(totals),
        addr(backpointers),
    )
    if ok < 0:
        raise OverflowError("path cost does not fit in int64")
    if not ok:
        raise ValueError("EOS is unreachable from BOS")
    path = _trace(backpointers)
    return DecodeResult(path, int(totals[-1]), totals, backpointers)


def decode_batch(lattice: PackedLattice, edge_costs) -> BatchDecodeResult:
    """Decode many cost assignments over the same lattice topology."""

    weights = i64(edge_costs)
    if weights.ndim != 2 or weights.shape[1] != lattice.edge_count:
        raise ValueError("edge_costs must have shape (batch, edge_count)")
    totals = np.empty((weights.shape[0], lattice.node_count), dtype=np.int64)
    backpointers = np.empty_like(totals)
    if weights.shape[0]:
        ok = lib().mm_viterbi_decode_batch(
            weights.shape[0],
            lattice.node_count,
            lattice.edge_count,
            addr(lattice.edge_offsets),
            addr(lattice.predecessors),
            addr(weights),
            addr(totals),
            addr(backpointers),
        )
        if ok < 0:
            raise OverflowError("path cost does not fit in int64")
        if not ok:
            raise ValueError("EOS is unreachable from BOS")
    paths = tuple(_trace(row) for row in backpointers)
    return BatchDecodeResult(paths, totals, backpointers)


def reference_decode(
    lattice: PackedLattice, edge_costs: Sequence[int] | np.ndarray | None = None
) -> DecodeResult:
    """Straight Python implementation used as a readable parity reference."""

    weights = lattice.edge_costs if edge_costs is None else i64(edge_costs)
    if weights.shape != (lattice.edge_count,):
        raise ValueError("edge_costs must have one value per edge")
    totals = np.full(lattice.node_count, INF, dtype=np.int64)
    backpointers = np.full(lattice.node_count, -1, dtype=np.int64)
    totals[0] = 0
    for node in range(1, lattice.node_count):
        start = int(lattice.edge_offsets[node])
        stop = int(lattice.edge_offsets[node + 1])
        for edge in range(start, stop):
            predecessor = int(lattice.predecessors[edge])
            if totals[predecessor] == INF:
                continue
            candidate = int(totals[predecessor]) + int(weights[edge])
            if candidate <= int(totals[node]):
                totals[node] = candidate
                backpointers[node] = predecessor
    path = _trace(backpointers)
    return DecodeResult(path, int(totals[-1]), totals, backpointers)


def from_mecab_lattice(lattice) -> PackedLattice:
    """Copy a populated ``MeCab.Lattice`` into the compact Mojo layout."""

    nodes = {lattice.bos_node().id: lattice.bos_node()}
    for position in range(lattice.size() + 1):
        node = lattice.begin_nodes(position)
        while node is not None:
            nodes[node.id] = node
            node = node.bnext
    eos = lattice.eos_node()
    nodes[eos.id] = eos
    ordered_ids = sorted(nodes)
    if ordered_ids != list(range(len(ordered_ids))):
        raise ValueError("MeCab returned non-contiguous node IDs")

    offsets = [0]
    predecessors: list[int] = []
    edge_costs: list[int] = []
    labels: list[str] = []
    upstream_costs: list[int] = []
    for node_id in ordered_ids:
        node = nodes[node_id]
        labels.append(node.surface)
        upstream_costs.append(node.cost)
        path = node.lpath
        while path is not None:
            predecessors.append(path.lnode.id)
            edge_costs.append(path.cost)
            path = path.lnext
        offsets.append(len(predecessors))
    return PackedLattice(
        offsets, predecessors, edge_costs, tuple(labels), upstream_costs
    )


def mecab_lattice(text: str, model=None) -> PackedLattice:
    """Ask MeCab for candidates, then return a Mojo-decodable copied lattice."""

    import MeCab

    active_model = model if model is not None else MeCab.Model()
    tagger = active_model.createTagger()
    lattice = active_model.createLattice()
    lattice.set_request_type(MeCab.MECAB_ALL_MORPHS | MeCab.MECAB_MARGINAL_PROB)
    lattice.set_sentence(text)
    if not tagger.parse(lattice):
        raise RuntimeError(lattice.what())
    return from_mecab_lattice(lattice)


def decoded_surfaces(lattice: PackedLattice, result: DecodeResult) -> tuple[str, ...]:
    if not lattice.labels:
        raise ValueError("the lattice has no labels")
    return tuple(lattice.labels[node] for node in result.path[1:-1])


def make_lattice(
    incoming: Iterable[Iterable[tuple[int, int]]], labels: Sequence[str] = ()
) -> PackedLattice:
    """Build a lattice from ``(predecessor, edge_cost)`` pairs per node."""

    offsets = [0]
    predecessors: list[int] = []
    costs: list[int] = []
    for edges in incoming:
        for predecessor, cost in edges:
            predecessors.append(predecessor)
            costs.append(cost)
        offsets.append(len(predecessors))
    return PackedLattice(offsets, predecessors, costs, tuple(labels))
