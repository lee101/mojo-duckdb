"""DuckDB-compatible vector functions backed by Mojo."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from typing import Any, NamedTuple

import numpy as np

from ._lib import addr, lib


_PARALLEL_THRESHOLD = 1_000_000
_PARALLEL_WORKERS = min(8, os.cpu_count() or 1)
_WORK_POOL = ThreadPoolExecutor(
    max_workers=_PARALLEL_WORKERS, thread_name_prefix="mojo-duckdb"
)


def _ranges(size: int):
    workers = _PARALLEL_WORKERS if size >= _PARALLEL_WORKERS else size
    return tuple(
        (worker * size // workers, (worker + 1) * size // workers)
        for worker in range(workers)
    )


def _parallel_call(size: int, call) -> None:
    futures = [_WORK_POOL.submit(call, begin, end) for begin, end in _ranges(size)]
    for future in futures:
        future.result()


class AggregateResult(NamedTuple):
    count: int
    sum: float | None
    avg: float | None
    min: float | None
    max: float | None
    product: float | None
    var_pop: float | None
    var_samp: float | None
    stddev_pop: float | None
    stddev_samp: float | None


@dataclass(frozen=True)
class GroupResult:
    keys: np.ndarray | np.ma.MaskedArray
    count: np.ndarray
    sum: np.ndarray | np.ma.MaskedArray
    avg: np.ndarray | np.ma.MaskedArray
    min: np.ndarray | np.ma.MaskedArray
    max: np.ndarray | np.ma.MaskedArray


@dataclass(frozen=True)
class JoinResult:
    left: np.ndarray
    right: np.ndarray

    def __len__(self) -> int:
        return len(self.left)


def _converted(data: np.ndarray, dtype) -> np.ndarray:
    """Convert public inputs without lossy or surprising NumPy narrowing."""
    dtype = np.dtype(dtype)
    if data.size == 0:
        return np.ascontiguousarray(data, dtype=dtype)
    if np.issubdtype(data.dtype, np.complexfloating):
        raise TypeError("complex values are not supported")
    if data.dtype.kind == "O":
        flat = data.reshape(-1)
        if any(isinstance(x, (complex, np.complexfloating)) for x in flat):
            raise TypeError("complex values are not supported")
        if dtype == np.dtype(np.int64) and any(
            not isinstance(x, (int, np.integer, bool, np.bool_)) for x in flat
        ):
            raise TypeError("integer keys must contain only integers or NULL")
        if dtype == np.dtype(np.float64):
            for value in flat:
                if isinstance(value, (int, np.integer)) and not -(2**53) <= int(
                    value
                ) <= 2**53:
                    raise OverflowError(
                        "integer value cannot be represented exactly as float64"
                    )
    if dtype == np.dtype(np.int64):
        if data.dtype.kind not in "biu":
            raise TypeError("integer keys must contain only integers or NULL")
        if data.dtype.kind == "u" and data.size and data.max() > np.iinfo(np.int64).max:
            raise OverflowError("integer key is outside the int64 range")
    elif dtype == np.dtype(np.float64) and data.dtype.kind in "iu" and data.size:
        # Above 2**53, conversion to Float64 can silently change an integer.
        if data.dtype.kind == "u":
            too_large = data.max() > 2**53
        else:
            too_large = data.min() < -(2**53) or data.max() > 2**53
        if too_large:
            raise OverflowError("integer value cannot be represented exactly as float64")
    return np.ascontiguousarray(data, dtype=dtype)


def _column(values: Any, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    masked = np.ma.asarray(values)
    data = np.asarray(masked.data)
    mask = np.ma.getmaskarray(masked)
    if data.dtype.kind == "O":
        flat = data.reshape(-1)
        none_mask = np.fromiter((x is None for x in flat), bool, flat.size)
        present = [x for x in flat if x is not None]
        if np.dtype(dtype) == np.dtype(np.int64) and any(
            not isinstance(x, (int, np.integer, bool, np.bool_)) for x in present
        ):
            raise TypeError("integer keys must contain only integers or NULL")
        if any(isinstance(x, (complex, np.complexfloating)) for x in present):
            raise TypeError("complex values are not supported")
        clean = np.asarray(
            [0 if x is None else x for x in flat],
        ).reshape(data.shape)
        mask = np.logical_or(mask, none_mask.reshape(data.shape))
        data = clean
    data = _converted(np.asarray(data), dtype).reshape(-1)
    valid = np.ascontiguousarray(~np.asarray(mask, dtype=bool).reshape(-1), dtype=np.uint8)
    return data, valid


def _dense_column(values: Any, dtype=np.float64) -> np.ndarray | None:
    if np.ma.isMaskedArray(values):
        return None
    data = np.asarray(values)
    if data.dtype.kind == "O":
        return None
    return _converted(data, dtype).reshape(-1)


def _pair(a: Any, b: Any, dtype=np.float64):
    aa = np.ma.asarray(a)
    bb = np.ma.asarray(b)
    ba, bmask = np.broadcast_arrays(np.asarray(aa.data), np.asarray(bb.data))
    amask, bmask2 = np.broadcast_arrays(np.ma.getmaskarray(aa), np.ma.getmaskarray(bb))
    if ba.dtype.kind == "O":
        ba = np.where(ba == None, 0, ba)  # noqa: E711
        amask = np.logical_or(amask, np.asarray(np.broadcast_to(aa.data, ba.shape)) == None)  # noqa: E711
    if bmask.dtype.kind == "O":
        bmask = np.where(bmask == None, 0, bmask)  # noqa: E711
        bmask2 = np.logical_or(
            bmask2, np.asarray(np.broadcast_to(bb.data, bmask.shape)) == None  # noqa: E711
        )
    shape = ba.shape
    av = _converted(np.asarray(ba), dtype).reshape(-1)
    bv = _converted(np.asarray(bmask), dtype).reshape(-1)
    valid = np.ascontiguousarray(~(amask | bmask2), dtype=np.uint8).reshape(-1)
    return av, bv, valid, shape


def _nullable(data: np.ndarray, valid: np.ndarray, shape):
    data = data.reshape(shape)
    valid = valid.astype(bool).reshape(shape)
    if data.ndim == 0:
        return data.item() if valid.item() else None
    if valid.all():
        return data
    result = np.ma.MaskedArray(data, mask=~valid, copy=False)
    return result


_BINARY = {"add": 0, "subtract": 1, "multiply": 2, "divide": 3}
_COMPARE = {
    "equal": 0,
    "not_equal": 1,
    "less_than": 2,
    "less_than_or_equal": 3,
    "greater_than": 4,
    "greater_than_or_equal": 5,
}


def _binary(a: Any, b: Any, op: str):
    if not np.ma.isMaskedArray(a) and not np.ma.isMaskedArray(b):
        raw_a = np.asarray(a)
        raw_b = np.asarray(b)
        if raw_a.dtype.kind != "O" and raw_b.dtype.kind != "O":
            ba, bb = np.broadcast_arrays(raw_a, raw_b)
            shape = ba.shape
            av = _converted(ba, np.float64).reshape(-1)
            bv = _converted(bb, np.float64).reshape(-1)
            dst = np.empty(av.size, dtype=np.float64)
            if av.size:
                kernel = lib().mdb_binary
                opcode = _BINARY[op]
                if av.size >= _PARALLEL_THRESHOLD:
                    def process(begin, end):
                        kernel(
                            addr(av[begin:]),
                            addr(bv[begin:]),
                            addr(dst[begin:]),
                            end - begin,
                            opcode,
                        )

                    _parallel_call(av.size, process)
                else:
                    kernel(addr(av), addr(bv), addr(dst), av.size, opcode)
            result = dst.reshape(shape)
            return result.item() if result.ndim == 0 else result
    av, bv, valid, shape = _pair(a, b)
    dst = np.empty(av.size, dtype=np.float64)
    if av.size:
        lib().mdb_binary(addr(av), addr(bv), addr(dst), av.size, _BINARY[op])
    return _nullable(dst, valid, shape)


def add(a: Any, b: Any):
    return _binary(a, b, "add")


def subtract(a: Any, b: Any):
    return _binary(a, b, "subtract")


def multiply(a: Any, b: Any):
    return _binary(a, b, "multiply")


def divide(a: Any, b: Any):
    return _binary(a, b, "divide")


def multiply_add(a: Any, b: Any, c: Any):
    if not any(np.ma.isMaskedArray(value) for value in (a, b, c)):
        raw_a = np.asarray(a)
        raw_b = np.asarray(b)
        raw_c = np.asarray(c)
        if all(value.dtype.kind != "O" for value in (raw_a, raw_b, raw_c)):
            ba, bb, bc = np.broadcast_arrays(raw_a, raw_b, raw_c)
            shape = ba.shape
            av = _converted(ba, np.float64).reshape(-1)
            bv = _converted(bb, np.float64).reshape(-1)
            cv = _converted(bc, np.float64).reshape(-1)
            dst = np.empty(av.size, dtype=np.float64)
            if av.size:
                kernel = lib().mdb_multiply_add
                if av.size >= _PARALLEL_THRESHOLD:
                    def process(begin, end):
                        kernel(
                            addr(av[begin:]),
                            addr(bv[begin:]),
                            addr(cv[begin:]),
                            addr(dst[begin:]),
                            end - begin,
                        )

                    _parallel_call(av.size, process)
                else:
                    kernel(addr(av), addr(bv), addr(cv), addr(dst), av.size)
            result = dst.reshape(shape)
            return result.item() if result.ndim == 0 else result
    return add(multiply(a, b), c)


def _compare(a: Any, b: Any, op: str):
    av, bv, valid, shape = _pair(a, b)
    dst = np.empty(av.size, dtype=np.uint8)
    if av.size:
        lib().mdb_compare(addr(av), addr(bv), addr(dst), av.size, _COMPARE[op])
    return _nullable(dst.astype(bool), valid, shape)


def equal(a: Any, b: Any):
    return _compare(a, b, "equal")


def not_equal(a: Any, b: Any):
    return _compare(a, b, "not_equal")


def less_than(a: Any, b: Any):
    return _compare(a, b, "less_than")


def less_than_or_equal(a: Any, b: Any):
    return _compare(a, b, "less_than_or_equal")


def greater_than(a: Any, b: Any):
    return _compare(a, b, "greater_than")


def greater_than_or_equal(a: Any, b: Any):
    return _compare(a, b, "greater_than_or_equal")


def aggregate(values: Any) -> AggregateResult:
    raw = np.empty(7, dtype=np.float64)
    dense = _dense_column(values)
    if dense is not None and dense.size == 0:
        return AggregateResult(0, None, None, None, None, None, None, None, None, None)
    if dense is None:
        data, valid = _column(values)
        if data.size == 0:
            return AggregateResult(0, None, None, None, None, None, None, None, None, None)
        lib().mdb_aggregate(addr(data), addr(valid), data.size, addr(raw))
    else:
        data = dense
        scratch = np.empty(7 * 8, dtype=np.float64)
        lib().mdb_aggregate_dense(addr(data), data.size, addr(raw), addr(scratch))
    n = int(raw[0])
    if n == 0:
        return AggregateResult(0, None, None, None, None, None, None, None, None, None)
    vp = raw[3] / n
    vs = raw[3] / (n - 1) if n > 1 else None
    return AggregateResult(
        n,
        float(raw[1]),
        float(raw[2]),
        float(raw[4]),
        float(raw[5]),
        float(raw[6]),
        float(vp),
        None if vs is None else float(vs),
        float(math.sqrt(vp)),
        None if vs is None else float(math.sqrt(vs)),
    )


def count(values: Any) -> int:
    return aggregate(values).count


def count_if(values: Any) -> int:
    data, valid = _column(values)
    return int(np.count_nonzero(data.astype(bool) & valid.astype(bool)))


def sum(values: Any) -> float | None:
    return aggregate(values).sum


def avg(values: Any) -> float | None:
    return aggregate(values).avg


def min(values: Any) -> float | None:
    return aggregate(values).min


def max(values: Any) -> float | None:
    return aggregate(values).max


def product(values: Any) -> float | None:
    return aggregate(values).product


def var_pop(values: Any) -> float | None:
    return aggregate(values).var_pop


def var_samp(values: Any) -> float | None:
    return aggregate(values).var_samp


def variance(values: Any) -> float | None:
    return var_samp(values)


def stddev_pop(values: Any) -> float | None:
    return aggregate(values).stddev_pop


def stddev_samp(values: Any) -> float | None:
    return aggregate(values).stddev_samp


def stddev(values: Any) -> float | None:
    return stddev_samp(values)


def _bivariate(a: Any, b: Any):
    av, bv, valid, _ = _pair(a, b)
    if av.size == 0:
        return 0, 0.0, 0.0, 0.0
    raw = np.empty(4, dtype=np.float64)
    lib().mdb_bivariate(addr(av), addr(bv), addr(valid), addr(valid), av.size, addr(raw))
    return int(raw[0]), raw[1], raw[2], raw[3]


def covar_pop(a: Any, b: Any) -> float | None:
    n, co, _, _ = _bivariate(a, b)
    return None if n == 0 else float(co / n)


def covar_samp(a: Any, b: Any) -> float | None:
    n, co, _, _ = _bivariate(a, b)
    return None if n < 2 else float(co / (n - 1))


def corr(a: Any, b: Any) -> float | None:
    n, co, m2a, m2b = _bivariate(a, b)
    if n == 0:
        return None
    if m2a == 0.0 or m2b == 0.0:
        return math.nan
    return float(co / math.sqrt(m2a * m2b))


def filter(values: Any, predicate: Any):
    dense_data = _dense_column(values)
    dense_predicate = _dense_column(predicate, np.bool_)
    if dense_data is not None and dense_predicate is not None:
        if dense_data.size != dense_predicate.size:
            raise ValueError("values and predicate must have the same length")
        dst = np.empty(dense_data.size, dtype=np.float64)
        if dense_data.size == 0:
            return dst
        kernel = lib().mdb_compact_dense
        if dense_data.size >= _PARALLEL_THRESHOLD:
            ranges = _ranges(dense_data.size)
            counts = tuple(
                int(np.count_nonzero(dense_predicate[begin:end]))
                for begin, end in ranges
            )
            offsets = []
            n = 0
            for count in counts:
                offsets.append(n)
                n += count

            def process(partition):
                begin, end = ranges[partition]
                destination = offsets[partition]
                if counts[partition]:
                    kernel(
                        addr(dense_data[begin:]),
                        addr(dense_predicate[begin:]),
                        addr(dst[destination:]),
                        end - begin,
                    )

            futures = [
                _WORK_POOL.submit(process, partition)
                for partition in range(len(ranges))
            ]
            for future in futures:
                future.result()
        else:
            n = kernel(
                addr(dense_data),
                addr(dense_predicate),
                addr(dst),
                dense_data.size,
            )
        return dst[:n]
    data, valid = _column(values)
    pred, pred_valid = _column(predicate)
    if data.size != pred.size:
        raise ValueError("values and predicate must have the same length")
    selected = np.ascontiguousarray(
        pred.astype(bool) & pred_valid.astype(bool), dtype=np.uint8
    )
    dst = np.empty(data.size, dtype=np.float64)
    dst_valid = np.empty(data.size, dtype=np.uint8)
    if data.size == 0:
        return dst
    n = lib().mdb_compact(
        addr(data),
        addr(valid),
        addr(selected),
        addr(dst),
        addr(dst_valid),
        data.size,
    )
    return _nullable(dst[:n], dst_valid[:n], (n,))


def _list_metric(a: Any, b: Any, op: int):
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        raise TypeError("NULL list elements are not supported")
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise TypeError("complex values are not supported")
    if av.shape != bv.shape or av.ndim not in (1, 2):
        raise ValueError("arguments must have the same one- or two-dimensional shape")
    matrix_a = np.ascontiguousarray(av.reshape(1, -1) if av.ndim == 1 else av)
    matrix_b = np.ascontiguousarray(bv.reshape(1, -1) if bv.ndim == 1 else bv)
    dst = np.empty(matrix_a.shape[0], dtype=np.float64)
    if matrix_a.shape[0] == 0:
        return dst
    lib().mdb_list_metric(
        addr(matrix_a), addr(matrix_b), addr(dst), matrix_a.shape[0], matrix_a.shape[1], op
    )
    return float(dst[0]) if av.ndim == 1 else dst


def list_inner_product(a: Any, b: Any):
    return _list_metric(a, b, 0)


def list_cosine_similarity(a: Any, b: Any):
    return _list_metric(a, b, 1)


def list_distance(a: Any, b: Any):
    return _list_metric(a, b, 2)


array_inner_product = list_inner_product
array_cosine_similarity = list_cosine_similarity
array_distance = list_distance


def _capacity(n: int) -> int:
    capacity = 1
    while capacity < 2 * n + 1:
        capacity *= 2
    return capacity


def group_by(keys: Any, values: Any) -> GroupResult:
    dense_keys = _dense_column(keys, np.int64)
    dense_values = _dense_column(values)
    if dense_keys is not None and dense_values is not None:
        if dense_keys.size != dense_values.size:
            raise ValueError("keys and values must have the same length")
        if dense_keys.size:
            min_key = int(dense_keys.min())
            max_key = int(dense_keys.max())
            span = max_key - min_key + 1
            direct_limit = dense_keys.size // 2
            if direct_limit < 4096:
                direct_limit = 4096
            if direct_limit > 1_048_576:
                direct_limit = 1_048_576
            if span <= direct_limit:
                return _group_by_dense(dense_keys, dense_values, min_key, span)

    key_data, key_valid = _column(keys, np.int64)
    value_data, value_valid = _column(values)
    if key_data.size != value_data.size:
        raise ValueError("keys and values must have the same length")
    if key_data.size == 0:
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=np.float64)
        return GroupResult(
            empty_i64,
            empty_i64.copy(),
            empty_f64,
            empty_f64.copy(),
            empty_f64.copy(),
            empty_f64.copy(),
        )
    capacity = _capacity(key_data.size)
    table_keys = np.empty(capacity, dtype=np.int64)
    occupied = np.empty(capacity, dtype=np.uint8)
    null_key = np.empty(capacity, dtype=np.uint8)
    sums = np.empty(capacity, dtype=np.float64)
    counts = np.empty(capacity, dtype=np.int64)
    mins = np.empty(capacity, dtype=np.float64)
    maxs = np.empty(capacity, dtype=np.float64)
    lib().mdb_group_i64(
        addr(key_data),
        addr(key_valid),
        addr(value_data),
        addr(value_valid),
        key_data.size,
        addr(table_keys),
        addr(occupied),
        addr(null_key),
        addr(sums),
        addr(counts),
        addr(mins),
        addr(maxs),
        capacity,
    )
    take = occupied.astype(bool)
    missing_key = null_key[take].astype(bool)
    missing_value = counts[take] == 0
    group_keys = np.ma.MaskedArray(table_keys[take], mask=missing_key)
    group_sum = np.ma.MaskedArray(sums[take], mask=missing_value)
    group_min = np.ma.MaskedArray(mins[take], mask=missing_value)
    group_max = np.ma.MaskedArray(maxs[take], mask=missing_value)
    avg_data = np.zeros_like(sums[take])
    np.divide(sums[take], counts[take], out=avg_data, where=~missing_value)
    group_avg = np.ma.MaskedArray(avg_data, mask=missing_value)
    if not missing_key.any():
        group_keys = group_keys.data
    if not missing_value.any():
        group_sum = group_sum.data
        group_avg = group_avg.data
        group_min = group_min.data
        group_max = group_max.data
    return GroupResult(
        group_keys, counts[take], group_sum, group_avg, group_min, group_max
    )


def _group_by_dense(
    keys: np.ndarray, values: np.ndarray, min_key: int, capacity: int
) -> GroupResult:
    occupied = np.empty(capacity, dtype=np.uint8)
    sums = np.empty(capacity, dtype=np.float64)
    counts = np.empty(capacity, dtype=np.int64)
    mins = np.empty(capacity, dtype=np.float64)
    maxs = np.empty(capacity, dtype=np.float64)
    lib().mdb_group_dense_i64(
        addr(keys),
        addr(values),
        keys.size,
        min_key,
        addr(occupied),
        addr(sums),
        addr(counts),
        addr(mins),
        addr(maxs),
        capacity,
    )
    indices = np.flatnonzero(occupied)
    group_counts = counts[indices]
    group_sums = sums[indices]
    return GroupResult(
        indices.astype(np.int64) + min_key,
        group_counts,
        group_sums,
        group_sums / group_counts,
        mins[indices],
        maxs[indices],
    )


def hash_join(left_keys: Any, right_keys: Any) -> JoinResult:
    dense_left = _dense_column(left_keys, np.int64)
    dense_right = _dense_column(right_keys, np.int64)
    if dense_left is not None and dense_right is not None:
        return _hash_join_dense(dense_left, dense_right)

    left, left_valid = _column(left_keys, np.int64)
    right, right_valid = _column(right_keys, np.int64)
    if left.size == 0 or right.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return JoinResult(empty, empty.copy())
    capacity = _capacity(right.size)
    heads = np.empty(capacity, dtype=np.int64)
    links = np.empty(right.size if right.size else 1, dtype=np.int64)
    n = lib().mdb_hash_join_i64(
        addr(left),
        addr(left_valid),
        left.size,
        addr(right),
        addr(right_valid),
        right.size,
        addr(heads),
        addr(links),
        capacity,
        0,
        0,
        0,
    )
    dst_left = np.empty(n, dtype=np.int64)
    dst_right = np.empty(n, dtype=np.int64)
    if n:
        written = lib().mdb_hash_join_i64(
            addr(left),
            addr(left_valid),
            left.size,
            addr(right),
            addr(right_valid),
            right.size,
            addr(heads),
            addr(links),
            capacity,
            addr(dst_left),
            addr(dst_right),
            1,
        )
        if written != n:
            raise RuntimeError("hash join count changed between passes")
    return JoinResult(dst_left, dst_right)


def _hash_join_dense(left: np.ndarray, right: np.ndarray) -> JoinResult:
    if left.size == 0 or right.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return JoinResult(empty, empty.copy())
    # Check in Python integers to avoid overflow in right_first + nr inside Mojo.
    unit_range = int(right[-1]) - int(right[0]) == right.size - 1 and bool(
        np.all(np.diff(right) == 1)
    )
    range_end_safe = int(right[0]) <= np.iinfo(np.int64).max - (right.size - 1)
    if unit_range and range_end_safe:
        size = left.size if left.size else 1
        dst_left = np.empty(size, dtype=np.int64)
        dst_right = np.empty(size, dtype=np.int64)
        n = lib().mdb_range_join_dense_i64(
            addr(left),
            left.size,
            int(right[0]),
            right.size,
            addr(dst_left),
            addr(dst_right),
        )
        return JoinResult(dst_left[:n], dst_right[:n])

    capacity = _capacity(right.size)
    heads = np.empty(capacity, dtype=np.int64)
    links = np.empty(right.size if right.size else 1, dtype=np.int64)
    n = lib().mdb_hash_join_dense_i64(
        addr(left),
        left.size,
        addr(right),
        right.size,
        addr(heads),
        addr(links),
        capacity,
        0,
        0,
        0,
    )
    dst_left = np.empty(n, dtype=np.int64)
    dst_right = np.empty(n, dtype=np.int64)
    if n:
        written = lib().mdb_hash_join_dense_i64(
            addr(left),
            left.size,
            addr(right),
            right.size,
            addr(heads),
            addr(links),
            capacity,
            addr(dst_left),
            addr(dst_right),
            1,
        )
        if written != n:
            raise RuntimeError("hash join count changed between passes")
    return JoinResult(dst_left, dst_right)
