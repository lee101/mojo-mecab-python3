"""MeCab-style lattice dynamic programming and UTF-8 indexing kernels."""

from std.sys import simd_width_of

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]

comptime I64_MAX = Int64(0x7FFFFFFFFFFFFFFF)
comptime I64_MIN = -I64_MAX - 1

comptime W64 = simd_width_of[DType.int64]()


@always_inline
def _decode_one(
    node_count: Int,
    edge_offsets: I64Ptr,
    predecessors: I64Ptr,
    edge_costs: I64Ptr,
    totals: I64Ptr,
    backpointers: I64Ptr,
    edge_base: Int,
    node_base: Int,
) -> Int:
    var totals_base = totals + node_base
    var backpointers_base = backpointers + node_base
    totals_base.unsafe_store(0, 0)
    backpointers_base.unsafe_store(0, -1)
    var best = Int64(0)
    var best_predecessor = Int64(-1)
    var edge = 0
    var edge_end = 0
    var node = 0
    while node + 1 < node_count:
        best = I64_MAX
        best_predecessor = Int64(-1)
        edge = edge_end
        edge_end = Int(edge_offsets.unsafe_load(node + 2))
        while edge < edge_end:
            var predecessor = Int(predecessors.unsafe_load(edge))
            # Every node except BOS is written below with a backpointer that is
            # -1 exactly when it is unreachable, so the bounds are implied.
            var previous = totals_base.unsafe_load(predecessor)
            var reachable = predecessor == 0
            if not reachable:
                reachable = backpointers_base.unsafe_load(predecessor) >= 0
            if reachable:
                var cost = edge_costs.unsafe_load(edge_base + edge)
                var overflow = previous > I64_MAX - cost
                if cost < 0:
                    overflow = previous < I64_MIN - cost
                if overflow:
                    return -1
                var candidate = previous + cost
                # MeCab replaces the predecessor on ties.
                if candidate <= best:
                    best = candidate
                    best_predecessor = Int64(predecessor)
            edge += 1
        totals_base.unsafe_store(node + 1, best)
        backpointers_base.unsafe_store(node + 1, best_predecessor)
        node += 1
    return 1 if backpointers_base.unsafe_load(node_count - 1) >= 0 else 0


@always_inline
def _validate_offsets(
    node_count: Int,
    edge_count: Int,
    edge_offsets: I64Ptr,
) -> Bool:
    """Check the CSR offset row with one read of each offset word.

    ``offsets[0..node_count+1]`` must start at zero, end at ``edge_count`` and
    never decrease. Each vector load of ``W64`` offsets is reused as the upper
    operand of the previous vector's comparison, so the seam pair is compared
    twice but no offset word is ever fetched twice. The old scalar loop loaded
    both neighbours for every node.
    """
    if edge_offsets.unsafe_load(0) != 0:
        return False
    if edge_offsets.unsafe_load(node_count) != Int64(edge_count):
        return False
    var bad = SIMD[DType.int64, W64](0)
    # Compare offsets[j] with offsets[j + 1] for every adjacent pair. Load the
    # upper operand first so the vector loaded on this iteration can serve as
    # the lower operand of the next one.
    var pairs = node_count
    var index = 0
    var lo = SIMD[DType.int64, W64](0)
    var hi = edge_offsets.unsafe_load[width=W64](0)
    while index + W64 <= pairs:
        lo = hi
        hi = edge_offsets.unsafe_load[width=W64](index + 1)
        bad += SIMD[DType.int64, W64](lo.gt(hi))
        index += W64
    if bad.reduce_add() != 0:
        return False
    while index < pairs:
        if edge_offsets.unsafe_load(index) > edge_offsets.unsafe_load(index + 1):
            return False
        index += 1
    return True


@always_inline
def _validate(
    node_count: Int,
    edge_count: Int,
    edge_offsets: I64Ptr,
    predecessors: I64Ptr,
) -> Bool:
    if not _validate_offsets(node_count, edge_count, edge_offsets):
        return False
    # Predicate each node's incoming predecessor slice with SIMD min/max: the
    # whole slice is invalid when its minimum is negative or its maximum is
    # >= the destination node, so no per-edge compare-and-branch is needed.
    var node = 1
    while node < node_count:
        var start = Int(edge_offsets.unsafe_load(node))
        var end = Int(edge_offsets.unsafe_load(node + 1))
        var lowest = SIMD[DType.int64, W64](I64_MAX)
        var highest = SIMD[DType.int64, W64](I64_MIN)
        var edge = start
        while edge + W64 <= end:
            var chunk = predecessors.unsafe_load[width=W64](edge)
            lowest = min(lowest, chunk)
            highest = max(highest, chunk)
            edge += W64
        var low = lowest.reduce_min()
        var high = highest.reduce_max()
        while edge < end:
            var predecessor = predecessors.unsafe_load(edge)
            low = min(low, predecessor)
            high = max(high, predecessor)
            edge += 1
        if low < 0 or high >= Int64(node):
            return False
        node += 1
    return True


@export("mm_viterbi_decode")
def mm_viterbi_decode(
    node_count: Int,
    edge_count: Int,
    edge_offsets_addr: Int,
    predecessors_addr: Int,
    edge_costs_addr: Int,
    totals_addr: Int,
    backpointers_addr: Int,
) abi("C") -> Int:
    if node_count < 2 or edge_count < 0:
        return 0
    if (
        edge_offsets_addr == 0
        or predecessors_addr == 0
        or edge_costs_addr == 0
        or totals_addr == 0
        or backpointers_addr == 0
    ):
        return 0
    var edge_offsets = I64Ptr(unsafe_from_address=edge_offsets_addr)
    var predecessors = I64Ptr(unsafe_from_address=predecessors_addr)
    var edge_costs = I64Ptr(unsafe_from_address=edge_costs_addr)
    var totals = I64Ptr(unsafe_from_address=totals_addr)
    var backpointers = I64Ptr(unsafe_from_address=backpointers_addr)
    if not _validate(node_count, edge_count, edge_offsets, predecessors):
        return 0
    return _decode_one(
        node_count,
        edge_offsets,
        predecessors,
        edge_costs,
        totals,
        backpointers,
        0,
        0,
    )


@export("mm_viterbi_decode_batch")
def mm_viterbi_decode_batch(
    batch_size: Int,
    node_count: Int,
    edge_count: Int,
    edge_offsets_addr: Int,
    predecessors_addr: Int,
    edge_costs_addr: Int,
    totals_addr: Int,
    backpointers_addr: Int,
) abi("C") -> Int:
    if batch_size <= 0 or node_count < 2 or edge_count < 0:
        return 0
    if (
        edge_offsets_addr == 0
        or predecessors_addr == 0
        or edge_costs_addr == 0
        or totals_addr == 0
        or backpointers_addr == 0
    ):
        return 0
    var edge_offsets = I64Ptr(unsafe_from_address=edge_offsets_addr)
    var predecessors = I64Ptr(unsafe_from_address=predecessors_addr)
    var edge_costs = I64Ptr(unsafe_from_address=edge_costs_addr)
    var totals = I64Ptr(unsafe_from_address=totals_addr)
    var backpointers = I64Ptr(unsafe_from_address=backpointers_addr)
    if not _validate(node_count, edge_count, edge_offsets, predecessors):
        return 0
    for batch in range(batch_size):
        var status = _decode_one(
            node_count,
            edge_offsets,
            predecessors,
            edge_costs,
            totals,
            backpointers,
            batch * edge_count,
            batch * node_count,
        )
        if status != 1:
            return status
    return 1


@export("mm_utf8_boundaries")
def mm_utf8_boundaries(
    bytes_addr: Int,
    byte_count: Int,
    boundaries_addr: Int,
    boundary_capacity: Int,
) -> Int:
    if (
        byte_count < 0
        or boundary_capacity < byte_count + 1
        or bytes_addr == 0
        or boundaries_addr == 0
    ):
        return 0
    var data = U8Ptr(unsafe_from_address=bytes_addr)
    var boundaries = I64Ptr(unsafe_from_address=boundaries_addr)
    var offset = 0
    var count = 1
    boundaries.unsafe_store(0, 0)
    while offset < byte_count:
        var lead = data.unsafe_load(offset)
        var width = 1
        if lead >= 0xF0:
            width = 4
        elif lead >= 0xE0:
            width = 3
        elif lead >= 0xC0:
            width = 2
        offset += width
        if offset > byte_count:
            return 0
        boundaries.unsafe_store(count, Int64(offset))
        count += 1
    return count
