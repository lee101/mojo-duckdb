"""Interface-level benchmarks against DuckDB on the same NumPy columns."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import duckdb
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojo_duckdb as mdb  # noqa: E402


def timeit(fn, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    rng = np.random.default_rng(20260729)
    con = duckdb.connect()
    threads = con.execute("select current_setting('threads')").fetchone()[0]
    cases = []

    aggregate_x = np.ascontiguousarray(1.0 + rng.normal(scale=1e-4, size=2_000_000))
    con.register("aggregate_data", {"x": aggregate_x})
    cases.append(
        (
            "fused numeric aggregates",
            "2,000,000",
            lambda: mdb.aggregate(aggregate_x),
            lambda: con.execute(
                """
                select count(x), sum(x), avg(x), min(x), max(x), product(x),
                       var_pop(x), var_samp(x), stddev_pop(x), stddev_samp(x)
                from aggregate_data
                """
            ).fetchone(),
        )
    )

    projection_a = np.ascontiguousarray(rng.normal(size=5_000_000))
    projection_b = np.ascontiguousarray(rng.normal(size=5_000_000))
    con.register("projection_data", {"a": projection_a, "b": projection_b})
    cases.append(
        (
            "fused projection a * b + a",
            "5,000,000",
            lambda: mdb.multiply_add(projection_a, projection_b, projection_a),
            lambda: con.execute("select a * b + a as x from projection_data").fetchnumpy()[
                "x"
            ],
        )
    )

    filter_x = np.ascontiguousarray(rng.normal(size=5_000_000))
    filter_p = np.ascontiguousarray(filter_x > 0.5)
    con.register("filter_data", {"x": filter_x, "p": filter_p})
    cases.append(
        (
            "filter/compact",
            "5,000,000",
            lambda: mdb.filter(filter_x, filter_p),
            lambda: con.execute("select x from filter_data where p").fetchnumpy()["x"],
        )
    )

    group_keys = np.ascontiguousarray(rng.integers(0, 4096, size=1_000_000))
    group_values = np.ascontiguousarray(rng.normal(size=1_000_000))
    con.register("group_data", {"k": group_keys, "v": group_values})
    cases.append(
        (
            "integer group by (4,096 groups)",
            "1,000,000",
            lambda: mdb.group_by(group_keys, group_values),
            lambda: con.execute(
                "select k, count(v), sum(v), avg(v), min(v), max(v) "
                "from group_data group by k"
            ).fetchnumpy(),
        )
    )

    right_keys = np.arange(500_000, dtype=np.int64)
    left_keys = np.ascontiguousarray(rng.integers(0, 1_000_000, size=500_000))
    left_index = np.arange(left_keys.size, dtype=np.int64)
    right_index = np.arange(right_keys.size, dtype=np.int64)
    con.register("join_left", {"k": left_keys, "i": left_index})
    con.register("join_right", {"k": right_keys, "i": right_index})
    cases.append(
        (
            "integer inner join (contiguous right keys)",
            "500,000 x 500,000",
            lambda: mdb.hash_join(left_keys, right_keys),
            lambda: con.execute(
                "select l.i as li, r.i as ri from join_left l join join_right r using(k)"
            ).fetchnumpy(),
        )
    )

    print(
        f"Machine: {cpu_name()}; {os.cpu_count()} logical CPUs; "
        f"Linux {platform.release()}; DuckDB {duckdb.__version__}; threads={threads}"
    )
    print()
    print("| Kernel | Input | Mojo | DuckDB | DuckDB / Mojo | Result |")
    print("|---|---:|---:|---:|---:|---|")
    for name, size, mojo_fn, duck_fn in cases:
        mojo_fn()
        duck_fn()
        mojo_time = timeit(mojo_fn)
        duck_time = timeit(duck_fn)
        ratio = duck_time / mojo_time
        result = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| {name} | {size} | {mojo_time * 1e3:.2f} ms | "
            f"{duck_time * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
