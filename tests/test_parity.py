from __future__ import annotations

import math

import duckdb
import numpy as np
import pytest

import mojo_duckdb as mdb


CON = duckdb.connect()
RNG = np.random.default_rng(20260729)


def sql_aggregate(values):
    return CON.execute(
        """
        select count(x), sum(x), avg(x), min(x), max(x), product(x),
               var_pop(x), var_samp(x), stddev_pop(x), stddev_samp(x)
        from (select unnest(?)::double as x)
        """,
        [values],
    ).fetchone()


def assert_optional_close(got, expected, *, rel=1e-11, abs=1e-12):
    if expected is None:
        assert got is None
    elif math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == pytest.approx(expected, rel=rel, abs=abs)


def normalized(values):
    result = []
    array = np.ma.asarray(values)
    mask = np.ma.getmaskarray(array).reshape(-1)
    for value, missing in zip(np.asarray(array.data).reshape(-1), mask):
        result.append(None if missing else value.item())
    return result


def test_all_aggregates_match_duckdb():
    values = RNG.normal(size=1003).tolist()
    values[3] = values[400] = values[998] = None
    got = mdb.aggregate(values)
    expected = sql_aggregate(values)
    assert got.count == expected[0]
    for actual, reference in zip(got[1:], expected[1:]):
        assert_optional_close(actual, reference)


def test_empty_and_all_null_aggregates_match_duckdb():
    for values in ([], [None, None, None]):
        got = mdb.aggregate(values)
        expected = sql_aggregate(values)
        assert got.count == expected[0] == 0
        assert tuple(got[1:]) == expected[1:]


def test_single_value_sample_statistics_are_null():
    assert mdb.var_pop([5.0]) == 0.0
    assert mdb.stddev_pop([5.0]) == 0.0
    assert mdb.var_samp([5.0]) is None
    assert mdb.stddev_samp([5.0]) is None


def test_named_aggregate_functions():
    values = [1.0, 2.0, None, 4.0]
    expected = sql_aggregate(values)
    functions = (
        mdb.count,
        mdb.sum,
        mdb.avg,
        mdb.min,
        mdb.max,
        mdb.product,
        mdb.var_pop,
        mdb.var_samp,
        mdb.stddev_pop,
        mdb.stddev_samp,
    )
    for fn, reference in zip(functions, expected):
        assert_optional_close(fn(values), reference)
    assert mdb.variance(values) == mdb.var_samp(values)
    assert mdb.stddev(values) == mdb.stddev_samp(values)


def test_nan_aggregate_ordering_matches_duckdb():
    values = [float("nan"), 1.0, -2.0]
    expected = CON.execute(
        """
        select min(x), max(x)
        from (values ('nan'::double), (1.0), (-2.0)) t(x)
        """
    ).fetchone()
    got = mdb.aggregate(values)
    assert got.min == expected[0]
    assert math.isnan(got.max) and math.isnan(expected[1])


def test_dense_aggregate_simd_tail_matches_duckdb():
    values = RNG.normal(size=13)
    got = mdb.aggregate(values)
    expected = sql_aggregate(values.tolist())
    for actual, reference in zip(got, expected):
        assert_optional_close(actual, reference)


@pytest.mark.parametrize("size", range(18))
def test_simd_boundaries_for_projection_and_list_metric(size):
    a = np.arange(size, dtype=np.float64) + 0.25
    b = np.arange(size, dtype=np.float64) + 1.5
    assert mdb.add(a, b) == pytest.approx(a + b)
    assert mdb.list_inner_product(a, b) == pytest.approx(float(np.dot(a, b)))


def test_dense_aggregate_parallel_threshold_matches_duckdb():
    values = 1.0 + RNG.normal(scale=1e-4, size=4_000_003)
    got = mdb.aggregate(values)
    CON.register("parallel_aggregate_data", {"x": values})
    expected = CON.execute(
        """
        select count(x), sum(x), avg(x), min(x), max(x), product(x),
               var_pop(x), var_samp(x), stddev_pop(x), stddev_samp(x)
        from parallel_aggregate_data
        """
    ).fetchone()
    CON.unregister("parallel_aggregate_data")
    for actual, reference in zip(got, expected):
        assert_optional_close(actual, reference)


def test_count_if_matches_duckdb():
    values = [True, False, None, True, False, True]
    expected = CON.execute(
        "select count_if(x) from (select unnest(?)::boolean x)", [values]
    ).fetchone()[0]
    assert mdb.count_if(values) == expected


def test_count_if_does_not_narrow_before_boolean_conversion():
    values = [256, 0, -256, None]
    expected = CON.execute(
        "select count_if(x) from (select unnest(?)::boolean x)", [values]
    ).fetchone()[0]
    assert mdb.count_if(values) == expected


def test_bivariate_statistics_match_duckdb():
    a = RNG.normal(size=701).astype(object)
    b = (0.3 * np.asarray(a, dtype=float) + RNG.normal(size=701)).astype(object)
    a[[3, 19, 200]] = None
    b[[4, 19, 500]] = None
    expected = CON.execute(
        """
        select covar_pop(a,b), covar_samp(a,b), corr(a,b)
        from (select unnest(?)::double a, unnest(?)::double b)
        """,
        [a.tolist(), b.tolist()],
    ).fetchone()
    for actual, reference in zip(
        (mdb.covar_pop(a, b), mdb.covar_samp(a, b), mdb.corr(a, b)), expected
    ):
        assert_optional_close(actual, reference)


def test_corr_constant_and_empty_match_duckdb():
    assert math.isnan(mdb.corr([1.0, 1.0], [2.0, 3.0]))
    assert mdb.corr([], []) is None


@pytest.mark.parametrize(
    ("name", "operator"),
    [
        ("add", "+"),
        ("subtract", "-"),
        ("multiply", "*"),
        ("divide", "/"),
    ],
)
def test_binary_projection_matches_duckdb(name, operator):
    a = [1.0, None, -3.0, 8.0]
    b = [2.0, 5.0, None, 4.0]
    expected = CON.execute(
        f"""
        select a {operator} b
        from (select unnest(?)::double a, unnest(?)::double b)
        """,
        [a, b],
    ).fetchnumpy()["(a " + operator + " b)"]
    got = getattr(mdb, name)(a, b)
    assert normalized(got) == normalized(expected)


def test_projection_supports_scalar_broadcast():
    assert np.array_equal(mdb.add(np.arange(6.0), 2.0), np.arange(6.0) + 2.0)
    assert mdb.multiply(3.0, 4.0) == 12.0
    assert mdb.add(None, 4.0) is None
    assert mdb.equal(None, 4.0) is None


@pytest.mark.parametrize(
    ("name", "operator"),
    [
        ("equal", "="),
        ("not_equal", "!="),
        ("less_than", "<"),
        ("less_than_or_equal", "<="),
        ("greater_than", ">"),
        ("greater_than_or_equal", ">="),
    ],
)
def test_comparisons_include_duckdb_nan_ordering(name, operator):
    a = [float("nan"), float("nan"), 1.0, None, 5.0]
    b = [float("nan"), 1.0, float("nan"), 2.0, 5.0]
    expected = CON.execute(
        f"""
        select a {operator} b
        from (values
            ('nan'::double, 'nan'::double),
            ('nan'::double, 1.0),
            (1.0, 'nan'::double),
            (NULL::double, 2.0),
            (5.0, 5.0)
        ) t(a, b)
        """
    ).fetchall()
    expected = [row[0] for row in expected]
    assert normalized(getattr(mdb, name)(a, b)) == expected


def test_filter_matches_where_null_semantics():
    values = [10.0, None, 30.0, 40.0, 50.0]
    predicate = [True, True, None, False, True]
    expected = CON.execute(
        """
        select x from (select unnest(?)::double x, unnest(?)::boolean p)
        where p
        """,
        [values, predicate],
    ).fetchall()
    assert normalized(mdb.filter(values, predicate)) == [row[0] for row in expected]


@pytest.mark.parametrize(
    "name",
    ["list_inner_product", "list_cosine_similarity", "list_distance"],
)
def test_list_functions_match_duckdb(name):
    a = RNG.normal(size=37)
    b = RNG.normal(size=37)
    expected = CON.execute(
        f"select {name}(?::double[], ?::double[])", [a.tolist(), b.tolist()]
    ).fetchone()[0]
    assert getattr(mdb, name)(a, b) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("list_inner_product", lambda a, b: np.sum(a * b, axis=1)),
        (
            "list_cosine_similarity",
            lambda a, b: np.sum(a * b, axis=1)
            / np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)),
        ),
        ("list_distance", lambda a, b: np.sqrt(np.sum((a - b) ** 2, axis=1))),
    ],
)
def test_list_functions_batch_rows(name, reference):
    a = RNG.normal(size=(23, 11))
    b = RNG.normal(size=(23, 11))
    got = getattr(mdb, name)(a, b)
    assert got == pytest.approx(reference(a, b))


def test_list_functions_reject_null_elements():
    values = np.ma.array([1.0, 2.0], mask=[False, True])
    with pytest.raises(TypeError, match="NULL list elements"):
        mdb.list_distance(values, values)


def test_array_aliases_are_duckdb_names():
    a, b = [1.0, 2.0], [3.0, 4.0]
    assert mdb.array_inner_product(a, b) == mdb.list_inner_product(a, b)
    assert mdb.array_cosine_similarity(a, b) == mdb.list_cosine_similarity(a, b)
    assert mdb.array_distance(a, b) == mdb.list_distance(a, b)


def test_group_by_matches_duckdb_including_nulls():
    keys = [2, 1, 2, None, 1, None, 3]
    values = [4.0, 3.0, None, 5.0, 7.0, None, None]
    got = mdb.group_by(keys, values)
    actual = {}
    for i, count in enumerate(got.count):
        key = None if np.ma.is_masked(got.keys[i]) else int(got.keys[i])
        actual[key] = (
            int(count),
            None if np.ma.is_masked(got.sum[i]) else float(got.sum[i]),
            None if np.ma.is_masked(got.avg[i]) else float(got.avg[i]),
            None if np.ma.is_masked(got.min[i]) else float(got.min[i]),
            None if np.ma.is_masked(got.max[i]) else float(got.max[i]),
        )
    expected_rows = CON.execute(
        """
        select k, count(v), sum(v), avg(v), min(v), max(v)
        from (select unnest(?)::bigint k, unnest(?)::double v)
        group by k
        """,
        [keys, values],
    ).fetchall()
    expected = {row[0]: row[1:] for row in expected_rows}
    assert actual == expected


def test_group_by_handles_hash_collisions_and_negative_keys():
    keys = np.array([-1, 7, 15, -1, 7, 23], dtype=np.int64)
    values = np.arange(1.0, 7.0)
    got = mdb.group_by(keys, values)
    actual = {int(k): s for k, s in zip(got.keys, got.sum)}
    assert actual == {-1: 5.0, 7: 7.0, 15: 3.0, 23: 6.0}


def test_group_by_dense_key_range_matches_duckdb():
    keys = RNG.integers(-32, 33, size=10_003, dtype=np.int64)
    values = RNG.normal(size=keys.size)
    got = mdb.group_by(keys, values)
    actual = {
        int(key): (int(count), float(total), float(low), float(high))
        for key, count, total, low, high in zip(
            got.keys, got.count, got.sum, got.min, got.max
        )
    }
    expected = CON.execute(
        """
        select k, count(v), sum(v), min(v), max(v)
        from (select unnest(?)::bigint k, unnest(?)::double v)
        group by k
        """,
        [keys.tolist(), values.tolist()],
    ).fetchall()
    assert actual.keys() == {row[0] for row in expected}
    for key, count, total, low, high in expected:
        got_count, got_total, got_low, got_high = actual[key]
        assert got_count == count
        assert got_total == pytest.approx(total)
        assert got_low == low
        assert got_high == high


def test_group_by_sparse_keys_uses_hash_fallback():
    keys = np.array([-(1 << 60), 1 << 60, -(1 << 60)], dtype=np.int64)
    got = mdb.group_by(keys, [1.0, 2.0, 3.0])
    assert {int(key): float(total) for key, total in zip(got.keys, got.sum)} == {
        -(1 << 60): 4.0,
        1 << 60: 2.0,
    }


@pytest.mark.parametrize("function", [mdb.group_by, mdb.hash_join])
def test_integer_keys_reject_fractional_values(function):
    with pytest.raises(TypeError, match="integer keys"):
        if function is mdb.group_by:
            function([1.5], [2.0])
        else:
            function([1.5], [1])


def test_float_kernels_reject_silent_integer_and_complex_narrowing():
    with pytest.raises(OverflowError, match="represented exactly"):
        mdb.sum([2**53 + 1])
    with pytest.raises(TypeError, match="complex"):
        mdb.add([1 + 2j], [3.0])


def test_empty_vector_paths():
    assert mdb.add([], []).size == 0
    assert mdb.equal([], []).size == 0
    assert mdb.filter([], []).size == 0
    assert len(mdb.group_by([], []).keys) == 0


def test_noncontiguous_inputs_are_safely_materialized():
    base = np.arange(40, dtype=np.float64)
    a = base[::2]
    b = base[1::2]
    assert not a.flags.c_contiguous
    assert mdb.multiply(a, b) == pytest.approx(a * b)
    assert mdb.list_distance(a, b) == pytest.approx(np.linalg.norm(a - b))


def test_hash_join_matches_duckdb_multiplicity_and_null_rules():
    left = [2, 1, 2, None, -3]
    right = [2, 2, 3, None, -3]
    got = mdb.hash_join(left, right)
    actual = sorted(zip(got.left.tolist(), got.right.tolist()))
    expected = CON.execute(
        """
        select li - 1, ri - 1
        from unnest(?) with ordinality l(k, li)
        join unnest(?) with ordinality r(k, ri) using (k)
        """,
        [left, right],
    ).fetchall()
    assert actual == sorted(expected)


def test_hash_join_empty_and_no_match():
    assert len(mdb.hash_join([], [])) == 0
    assert len(mdb.hash_join([1, 2], [3, 4])) == 0


def test_hash_join_dense_negative_keys_and_collisions():
    left = np.array([-1, 7, 15, 23], dtype=np.int64)
    right = np.array([15, -1, 7, 15], dtype=np.int64)
    got = mdb.hash_join(left, right)
    actual = sorted(zip(got.left.tolist(), got.right.tolist()))
    assert actual == [(0, 1), (1, 2), (2, 0), (2, 3)]


def test_hash_join_contiguous_right_range():
    left = np.array([-4, -3, -1, 0, 2, 3], dtype=np.int64)
    right = np.arange(-3, 3, dtype=np.int64)
    got = mdb.hash_join(left, right)
    assert list(zip(got.left.tolist(), got.right.tolist())) == [
        (1, 0),
        (2, 2),
        (3, 3),
        (4, 5),
    ]
