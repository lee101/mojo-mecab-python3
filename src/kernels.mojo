"""MeCab-style lattice dynamic programming and UTF-8 indexing kernels."""

comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime U8Ptr = UnsafePointer[UInt8, AnyOrigin[mut=True]]

comptime I64_MAX = Int64(0x7FFFFFFFFFFFFFFF)
comptime I64_MIN = -I64_MAX - 1


def decode_one(
    node_count: Int,
    edge_offsets: I64Ptr,
    predecessors: I64Ptr,
    edge_costs: I64Ptr,
    totals: I64Ptr,
    backpointers: I64Ptr,
    edge_base: Int = 0,
    node_base: Int = 0,
) -> Int:
    if node_count <= 0:
        return 0
    totals[node_base] = 0
    backpointers[node_base] = -1
    for node in range(1, node_count):
        var best = I64_MAX
        var best_predecessor = Int64(-1)
        var edge = Int(edge_offsets[node])
        var edge_end = Int(edge_offsets[node + 1])
        while edge < edge_end:
            var predecessor = Int(predecessors[edge])
            var previous = totals[node_base + predecessor]
            if predecessor == 0 or backpointers[node_base + predecessor] >= 0:
                var cost = edge_costs[edge_base + edge]
                if (cost > 0 and previous > I64_MAX - cost) or (
                    cost < 0 and previous < I64_MIN - cost
                ):
                    return -1
                var candidate = previous + cost
                # MeCab replaces the predecessor on ties.
                if candidate <= best:
                    best = candidate
                    best_predecessor = Int64(predecessor)
            edge += 1
        totals[node_base + node] = best
        backpointers[node_base + node] = best_predecessor
    return 1 if backpointers[node_base + node_count - 1] >= 0 else 0


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
    if edge_offsets[0] != 0 or edge_offsets[node_count] != Int64(edge_count):
        return 0
    for node in range(node_count):
        if edge_offsets[node] > edge_offsets[node + 1]:
            return 0
    for node in range(1, node_count):
        for edge in range(Int(edge_offsets[node]), Int(edge_offsets[node + 1])):
            if predecessors[edge] < 0 or predecessors[edge] >= Int64(node):
                return 0
    return decode_one(
        node_count,
        edge_offsets,
        predecessors,
        edge_costs,
        totals,
        backpointers,
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
    if edge_offsets[0] != 0 or edge_offsets[node_count] != Int64(edge_count):
        return 0
    for node in range(node_count):
        if edge_offsets[node] > edge_offsets[node + 1]:
            return 0
    for node in range(1, node_count):
        for edge in range(Int(edge_offsets[node]), Int(edge_offsets[node + 1])):
            if predecessors[edge] < 0 or predecessors[edge] >= Int64(node):
                return 0
    for batch in range(batch_size):
        var status = decode_one(
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
) abi("C") -> Int:
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
    boundaries[0] = 0
    while offset < byte_count:
        var lead = data[offset]
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
        boundaries[count] = Int64(offset)
        count += 1
    return count
