"""Vector execution kernels for the Python C ABI."""

from std.math import sqrt
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime AGGREGATE_WORKERS = 8
comptime AGGREGATE_PARALLEL_THRESHOLD = 4000000


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bptr(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def binary_chunk(a: FPtr, b: FPtr, dst: FPtr, begin: Int, end: Int, op: Int):
    var i = begin
    if op == 0:
        while i + W <= end:
            dst.store(i, a.load[width=W](i) + b.load[width=W](i))
            i += W
        while i < end:
            dst[i] = a[i] + b[i]
            i += 1
    elif op == 1:
        while i + W <= end:
            dst.store(i, a.load[width=W](i) - b.load[width=W](i))
            i += W
        while i < end:
            dst[i] = a[i] - b[i]
            i += 1
    elif op == 2:
        while i + W <= end:
            dst.store(i, a.load[width=W](i) * b.load[width=W](i))
            i += W
        while i < end:
            dst[i] = a[i] * b[i]
            i += 1
    else:
        while i + W <= end:
            dst.store(i, a.load[width=W](i) / b.load[width=W](i))
            i += W
        while i < end:
            dst[i] = a[i] / b[i]
            i += 1


def binary(a: FPtr, b: FPtr, dst: FPtr, n: Int, op: Int):
    binary_chunk(a, b, dst, 0, n, op)


def multiply_add(a: FPtr, b: FPtr, c: FPtr, dst: FPtr, n: Int):
    var i = 0
    while i + W <= n:
        dst.store(
            i,
            a.load[width=W](i) * b.load[width=W](i) + c.load[width=W](i),
        )
        i += W
    while i < n:
        dst[i] = a[i] * b[i] + c[i]
        i += 1


def compare(a: FPtr, b: FPtr, dst: BPtr, n: Int, op: Int):
    for i in range(n):
        var value = False
        var a_nan = a[i] != a[i]
        var b_nan = b[i] != b[i]
        if op == 0:
            value = (a_nan and b_nan) or a[i] == b[i]
        elif op == 1:
            value = not ((a_nan and b_nan) or a[i] == b[i])
        elif op == 2:
            value = (not a_nan and b_nan) or a[i] < b[i]
        elif op == 3:
            value = (b_nan and not a_nan) or (a_nan == b_nan and a[i] <= b[i])
            if a_nan and b_nan:
                value = True
        elif op == 4:
            value = (a_nan and not b_nan) or a[i] > b[i]
        else:
            value = (a_nan and not b_nan) or (a_nan == b_nan and a[i] >= b[i])
            if a_nan and b_nan:
                value = True
        dst[i] = UInt8(1) if value else UInt8(0)


def aggregate(values: FPtr, valid: BPtr, n: Int, dst: FPtr):
    var count = 0
    var total = 0.0
    var mean = 0.0
    var m2 = 0.0
    var lo = 0.0
    var hi = 0.0
    var prod = 1.0
    for i in range(n):
        if valid[i] == 0:
            continue
        var x = values[i]
        count += 1
        total += x
        prod *= x
        if count == 1:
            mean = x
            lo = x
            hi = x
        else:
            var delta = x - mean
            mean += delta / Float64(count)
            m2 += delta * (x - mean)
            if x != x:
                hi = x
            else:
                if lo != lo or x < lo:
                    lo = x
                if hi == hi and x > hi:
                    hi = x
    dst[0] = Float64(count)
    dst[1] = total
    dst[2] = mean
    dst[3] = m2
    dst[4] = lo
    dst[5] = hi
    dst[6] = prod


def aggregate_dense_chunk(
    values: FPtr, begin: Int, end: Int, dst: FPtr
):
    var totals = SIMD[DType.float64, W](0.0)
    var products = SIMD[DType.float64, W](1.0)
    var lows = SIMD[DType.float64, W](values[begin])
    var highs = lows
    var i = begin
    while i + W <= end:
        var batch = values.load[width=W](i)
        totals += batch
        products *= batch
        lows = min(lows, batch)
        highs = max(highs, batch)
        i += W
    var total = totals.reduce_add()
    var prod = 1.0
    var lo = lows[0]
    var hi = highs[0]
    for lane in range(W):
        prod *= products[lane]
        var low = lows[lane]
        var high = highs[lane]
        if low == low and (lo != lo or low < lo):
            lo = low
        if high != high or (hi == hi and high > hi):
            hi = high
    while i < end:
        var x = values[i]
        total += x
        prod *= x
        if x != x:
            hi = x
        else:
            if lo != lo or x < lo:
                lo = x
            if hi == hi and x > hi:
                hi = x
        i += 1

    var count = end - begin
    var mean = total / Float64(count)
    var m2_batches = SIMD[DType.float64, W](0.0)
    i = begin
    while i + W <= end:
        var delta = values.load[width=W](i) - mean
        m2_batches += delta * delta
        i += W
    var m2 = m2_batches.reduce_add()
    while i < end:
        var delta = values[i] - mean
        m2 += delta * delta
        i += 1

    dst[0] = Float64(count)
    dst[1] = total
    dst[2] = mean
    dst[3] = m2
    dst[4] = lo
    dst[5] = hi
    dst[6] = prod


def aggregate_dense(values: FPtr, n: Int, dst: FPtr, scratch: FPtr):
    if n == 0:
        dst[0] = 0.0
        return
    if n < AGGREGATE_PARALLEL_THRESHOLD:
        aggregate_dense_chunk(values, 0, n, dst)
        return

    for partition in range(AGGREGATE_WORKERS):
        var begin = partition * n // AGGREGATE_WORKERS
        var end = (partition + 1) * n // AGGREGATE_WORKERS
        aggregate_dense_chunk(
            values, begin, end, scratch + partition * 7
        )

    for j in range(7):
        dst[j] = scratch[j]
    for partition in range(1, AGGREGATE_WORKERS):
        var part = scratch + partition * 7
        var left_count = dst[0]
        var right_count = part[0]
        var combined_count = left_count + right_count
        var delta = part[2] - dst[2]
        dst[1] += part[1]
        dst[2] += delta * right_count / combined_count
        dst[3] += (
            part[3]
            + delta * delta * left_count * right_count / combined_count
        )
        if part[4] == part[4] and (dst[4] != dst[4] or part[4] < dst[4]):
            dst[4] = part[4]
        if part[5] != part[5] or (dst[5] == dst[5] and part[5] > dst[5]):
            dst[5] = part[5]
        dst[6] *= part[6]
        dst[0] = combined_count


def bivariate(a: FPtr, b: FPtr, av: BPtr, bv: BPtr, n: Int, dst: FPtr):
    var count = 0
    var mean_a = 0.0
    var mean_b = 0.0
    var co = 0.0
    var m2a = 0.0
    var m2b = 0.0
    for i in range(n):
        if av[i] == 0 or bv[i] == 0:
            continue
        count += 1
        var x = a[i]
        var y = b[i]
        var dx = x - mean_a
        mean_a += dx / Float64(count)
        var dy = y - mean_b
        mean_b += dy / Float64(count)
        co += dx * (y - mean_b)
        m2a += dx * (x - mean_a)
        m2b += dy * (y - mean_b)
    dst[0] = Float64(count)
    dst[1] = co
    dst[2] = m2a
    dst[3] = m2b


def compact_range(
    values: FPtr,
    valid: BPtr,
    predicate: BPtr,
    dst: FPtr,
    dst_valid: BPtr,
    begin: Int,
    end: Int,
    destination: Int,
) -> Int:
    var written = destination
    var i = begin
    while i + W <= end:
        var batch = values.load[width=W](i)
        var batch_valid = valid.load[width=W](i)
        var selected = predicate.load[width=W](i).ne(0)
        comptime for lane in range(W):
            if selected[lane]:
                dst[written] = batch[lane]
                dst_valid[written] = batch_valid[lane]
                written += 1
        i += W
    while i < end:
        if predicate[i] != 0:
            dst[written] = values[i]
            dst_valid[written] = valid[i]
            written += 1
        i += 1
    return written


def compact_dense_range(
    values: FPtr,
    predicate: BPtr,
    dst: FPtr,
    begin: Int,
    end: Int,
    destination: Int,
) -> Int:
    var written = destination
    var i = begin
    while i + W <= end:
        var batch = values.load[width=W](i)
        var selected = predicate.load[width=W](i).ne(0)
        comptime for lane in range(W):
            if selected[lane]:
                dst[written] = batch[lane]
                written += 1
        i += W
    while i < end:
        if predicate[i] != 0:
            dst[written] = values[i]
            written += 1
        i += 1
    return written


def count_selected(predicate: BPtr, begin: Int, end: Int) -> Int:
    var count = 0
    var i = begin
    while i + W <= end:
        var selected = predicate.load[width=W](i).ne(0)
        comptime for lane in range(W):
            count += Int(selected[lane])
        i += W
    while i < end:
        count += Int(predicate[i] != 0)
        i += 1
    return count


def compact(
    values: FPtr,
    valid: BPtr,
    predicate: BPtr,
    dst: FPtr,
    dst_valid: BPtr,
    n: Int,
) -> Int:
    return compact_range(values, valid, predicate, dst, dst_valid, 0, n, 0)


def compact_dense(
    values: FPtr, predicate: BPtr, dst: FPtr, n: Int
) -> Int:
    return compact_dense_range(values, predicate, dst, 0, n, 0)


def list_metric(a: FPtr, b: FPtr, dst: FPtr, rows: Int, width: Int, op: Int):
    for r in range(rows):
        var dot = SIMD[DType.float64, W](0.0)
        var aa = SIMD[DType.float64, W](0.0)
        var bb = SIMD[DType.float64, W](0.0)
        var distance = SIMD[DType.float64, W](0.0)
        var i = 0
        var base = r * width
        while i + W <= width:
            var va = a.load[width=W](base + i)
            var vb = b.load[width=W](base + i)
            dot += va * vb
            aa += va * va
            bb += vb * vb
            var diff = va - vb
            distance += diff * diff
            i += W
        var sd = dot.reduce_add()
        var sa = aa.reduce_add()
        var sb = bb.reduce_add()
        var dist = distance.reduce_add()
        while i < width:
            var x = a[base + i]
            var y = b[base + i]
            sd += x * y
            sa += x * x
            sb += y * y
            var diff = x - y
            dist += diff * diff
            i += 1
        if op == 0:
            dst[r] = sd
        elif op == 1:
            dst[r] = sd / sqrt(sa * sb)
        else:
            dst[r] = sqrt(dist)


def hash_slot(key: Int64, capacity: Int) -> Int:
    return Int(key) & (capacity - 1)


def group_i64(
    keys: IPtr,
    key_valid: BPtr,
    values: FPtr,
    value_valid: BPtr,
    n: Int,
    table_keys: IPtr,
    occupied: BPtr,
    null_key: BPtr,
    sums: FPtr,
    counts: IPtr,
    mins: FPtr,
    maxs: FPtr,
    capacity: Int,
) -> Int:
    for i in range(capacity):
        occupied[i] = UInt8(0)
        null_key[i] = UInt8(0)
        sums[i] = 0.0
        counts[i] = 0
        mins[i] = 0.0
        maxs[i] = 0.0
    var groups = 0
    var null_slot = -1
    for i in range(n):
        var slot = 0
        if key_valid[i] == 0:
            if null_slot >= 0:
                slot = null_slot
            else:
                while slot < capacity and occupied[slot] != 0:
                    slot += 1
                if slot == capacity:
                    continue
                null_slot = slot
                occupied[slot] = UInt8(1)
                null_key[slot] = UInt8(1)
                groups += 1
        else:
            var key = keys[i]
            slot = hash_slot(key, capacity)
            while occupied[slot] != 0:
                if null_key[slot] == 0 and table_keys[slot] == key:
                    break
                slot += 1
                if slot == capacity:
                    slot = 0
            if occupied[slot] == 0:
                occupied[slot] = UInt8(1)
                table_keys[slot] = key
                groups += 1
        if value_valid[i] != 0:
            var value = values[i]
            if counts[slot] == 0:
                mins[slot] = value
                maxs[slot] = value
            else:
                if value != value:
                    maxs[slot] = value
                else:
                    if mins[slot] != mins[slot] or value < mins[slot]:
                        mins[slot] = value
                    if maxs[slot] == maxs[slot] and value > maxs[slot]:
                        maxs[slot] = value
            sums[slot] += value
            counts[slot] += 1
    return groups


def group_dense_i64(
    keys: IPtr,
    values: FPtr,
    n: Int,
    min_key: Int64,
    occupied: BPtr,
    sums: FPtr,
    counts: IPtr,
    mins: FPtr,
    maxs: FPtr,
    capacity: Int,
) -> Int:
    for i in range(capacity):
        occupied[i] = UInt8(0)
        sums[i] = 0.0
        counts[i] = 0
    var groups = 0
    for i in range(n):
        var slot = Int(keys[i] - min_key)
        var value = values[i]
        if occupied[slot] == 0:
            occupied[slot] = UInt8(1)
            sums[slot] = value
            counts[slot] = 1
            mins[slot] = value
            maxs[slot] = value
            groups += 1
        else:
            sums[slot] += value
            counts[slot] += 1
            if value != value:
                maxs[slot] = value
            else:
                if mins[slot] != mins[slot] or value < mins[slot]:
                    mins[slot] = value
                if maxs[slot] == maxs[slot] and value > maxs[slot]:
                    maxs[slot] = value
    return groups


def hash_join_i64(
    left: IPtr,
    left_valid: BPtr,
    nl: Int,
    right: IPtr,
    right_valid: BPtr,
    nr: Int,
    heads: IPtr,
    links: IPtr,
    capacity: Int,
    dst_left: IPtr,
    dst_right: IPtr,
    write_results: Bool,
    build_hash_table: Bool,
) -> Int:
    if build_hash_table:
        for i in range(capacity):
            heads[i] = -1
        for j in range(nr):
            if right_valid[j] == 0:
                links[j] = -1
                continue
            var slot = hash_slot(right[j], capacity)
            links[j] = heads[slot]
            heads[slot] = Int64(j)
    var matches = 0
    for i in range(nl):
        if left_valid[i] == 0:
            continue
        var slot = hash_slot(left[i], capacity)
        var j = Int(heads[slot])
        while j >= 0:
            if right[j] == left[i]:
                if write_results:
                    dst_left[matches] = Int64(i)
                    dst_right[matches] = Int64(j)
                matches += 1
            j = Int(links[j])
    return matches


def hash_join_dense_i64(
    left: IPtr,
    nl: Int,
    right: IPtr,
    nr: Int,
    heads: IPtr,
    links: IPtr,
    capacity: Int,
    dst_left: IPtr,
    dst_right: IPtr,
    write_results: Bool,
    build_hash_table: Bool,
) -> Int:
    if build_hash_table:
        for i in range(capacity):
            heads[i] = -1
        for j in range(nr):
            var slot = hash_slot(right[j], capacity)
            links[j] = heads[slot]
            heads[slot] = Int64(j)
    var matches = 0
    for i in range(nl):
        var slot = hash_slot(left[i], capacity)
        var j = Int(heads[slot])
        while j >= 0:
            if right[j] == left[i]:
                if write_results:
                    dst_left[matches] = Int64(i)
                    dst_right[matches] = Int64(j)
                matches += 1
            j = Int(links[j])
    return matches


def range_join_dense_i64(
    left: IPtr,
    nl: Int,
    right_first: Int64,
    nr: Int,
    dst_left: IPtr,
    dst_right: IPtr,
) -> Int:
    var written = 0
    var right_last = right_first + Int64(nr - 1)
    for i in range(nl):
        var key = left[i]
        if key >= right_first and key <= right_last:
            dst_left[written] = Int64(i)
            dst_right[written] = key - right_first
            written += 1
    return written


@export("mdb_binary")
def mdb_binary(a: Int, b: Int, dst: Int, n: Int, op: Int) abi("C"):
    binary(fptr(a), fptr(b), fptr(dst), n, op)


@export("mdb_multiply_add")
def mdb_multiply_add(
    a: Int, b: Int, c: Int, dst: Int, n: Int
) abi("C"):
    multiply_add(fptr(a), fptr(b), fptr(c), fptr(dst), n)


@export("mdb_compare")
def mdb_compare(a: Int, b: Int, dst: Int, n: Int, op: Int) abi("C"):
    compare(fptr(a), fptr(b), bptr(dst), n, op)


@export("mdb_aggregate")
def mdb_aggregate(values: Int, valid: Int, n: Int, dst: Int) abi("C"):
    aggregate(fptr(values), bptr(valid), n, fptr(dst))


@export("mdb_aggregate_dense")
def mdb_aggregate_dense(
    values: Int, n: Int, dst: Int, scratch: Int
) abi("C"):
    aggregate_dense(fptr(values), n, fptr(dst), fptr(scratch))


@export("mdb_bivariate")
def mdb_bivariate(
    a: Int, b: Int, av: Int, bv: Int, n: Int, dst: Int
) abi("C"):
    bivariate(fptr(a), fptr(b), bptr(av), bptr(bv), n, fptr(dst))


@export("mdb_compact")
def mdb_compact(
    values: Int,
    valid: Int,
    predicate: Int,
    dst: Int,
    dst_valid: Int,
    n: Int,
) abi("C") -> Int:
    return compact(
        fptr(values),
        bptr(valid),
        bptr(predicate),
        fptr(dst),
        bptr(dst_valid),
        n,
    )


@export("mdb_compact_dense")
def mdb_compact_dense(
    values: Int, predicate: Int, dst: Int, n: Int
) abi("C") -> Int:
    return compact_dense(fptr(values), bptr(predicate), fptr(dst), n)


@export("mdb_list_metric")
def mdb_list_metric(
    a: Int, b: Int, dst: Int, rows: Int, width: Int, op: Int
) abi("C"):
    list_metric(fptr(a), fptr(b), fptr(dst), rows, width, op)


@export("mdb_group_i64")
def mdb_group_i64(
    keys: Int,
    key_valid: Int,
    values: Int,
    value_valid: Int,
    n: Int,
    table_keys: Int,
    occupied: Int,
    null_key: Int,
    sums: Int,
    counts: Int,
    mins: Int,
    maxs: Int,
    capacity: Int,
) abi("C") -> Int:
    return group_i64(
        iptr(keys),
        bptr(key_valid),
        fptr(values),
        bptr(value_valid),
        n,
        iptr(table_keys),
        bptr(occupied),
        bptr(null_key),
        fptr(sums),
        iptr(counts),
        fptr(mins),
        fptr(maxs),
        capacity,
    )


@export("mdb_group_dense_i64")
def mdb_group_dense_i64(
    keys: Int,
    values: Int,
    n: Int,
    min_key: Int64,
    occupied: Int,
    sums: Int,
    counts: Int,
    mins: Int,
    maxs: Int,
    capacity: Int,
) abi("C") -> Int:
    return group_dense_i64(
        iptr(keys),
        fptr(values),
        n,
        min_key,
        bptr(occupied),
        fptr(sums),
        iptr(counts),
        fptr(mins),
        fptr(maxs),
        capacity,
    )


@export("mdb_hash_join_i64")
def mdb_hash_join_i64(
    left: Int,
    left_valid: Int,
    nl: Int,
    right: Int,
    right_valid: Int,
    nr: Int,
    heads: Int,
    links: Int,
    capacity: Int,
    dst_left: Int,
    dst_right: Int,
    write_results: Int,
) abi("C") -> Int:
    if write_results != 0:
        return hash_join_i64(
            iptr(left),
            bptr(left_valid),
            nl,
            iptr(right),
            bptr(right_valid),
            nr,
            iptr(heads),
            iptr(links),
            capacity,
            iptr(dst_left),
            iptr(dst_right),
            True,
            False,
        )
    # Pointers cannot be constructed from the null output addresses used in count mode.
    return hash_join_i64(
        iptr(left),
        bptr(left_valid),
        nl,
        iptr(right),
        bptr(right_valid),
        nr,
        iptr(heads),
        iptr(links),
        capacity,
        iptr(heads),
        iptr(links),
        False,
        True,
    )


@export("mdb_hash_join_dense_i64")
def mdb_hash_join_dense_i64(
    left: Int,
    nl: Int,
    right: Int,
    nr: Int,
    heads: Int,
    links: Int,
    capacity: Int,
    dst_left: Int,
    dst_right: Int,
    write_results: Int,
) abi("C") -> Int:
    if write_results != 0:
        return hash_join_dense_i64(
            iptr(left),
            nl,
            iptr(right),
            nr,
            iptr(heads),
            iptr(links),
            capacity,
            iptr(dst_left),
            iptr(dst_right),
            True,
            False,
        )
    return hash_join_dense_i64(
        iptr(left),
        nl,
        iptr(right),
        nr,
        iptr(heads),
        iptr(links),
        capacity,
        iptr(heads),
        iptr(links),
        False,
        True,
    )


@export("mdb_range_join_dense_i64")
def mdb_range_join_dense_i64(
    left: Int,
    nl: Int,
    right_first: Int64,
    nr: Int,
    dst_left: Int,
    dst_right: Int,
) abi("C") -> Int:
    return range_join_dense_i64(
        iptr(left),
        nl,
        right_first,
        nr,
        iptr(dst_left),
        iptr(dst_right),
    )
