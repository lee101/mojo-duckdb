# mojo-duckdb

`mojo-duckdb` is a standalone Mojo implementation of compute-bound vector
execution kernels used by analytical SQL engines. It is not a fork of DuckDB
and does not embed DuckDB. The Python API uses DuckDB's SQL function names for
the covered subset, and its results are tested against the real `duckdb`
package.

The useful boundary is a column batch: NumPy columns enter the wrapper, Mojo
processes their contiguous buffers, and NumPy scalars or columns come back.
This makes the project suitable for custom execution engines and data
pipelines that need DuckDB semantics without routing each batch through a SQL
parser.

## Covered subset

- Float64 projection: `add`, `subtract`, `multiply`, and `divide`.
- Float64 predicates: `equal`, `not_equal`, `less_than`,
  `less_than_or_equal`, `greater_than`, and `greater_than_or_equal`, including
  DuckDB's NaN ordering.
- SQL aggregates: `count`, `count_if`, `sum`, `avg`, `min`, `max`, `product`,
  `var_pop`, `var_samp`/`variance`, `stddev_pop`, `stddev_samp`/`stddev`,
  `covar_pop`, `covar_samp`, and `corr`. `aggregate` computes all univariate
  statistics in one pass.
- Selection-vector compaction through `filter`.
- `list_inner_product`, `list_cosine_similarity`, and `list_distance`, with
  their `array_*` aliases. Two-dimensional inputs execute a batch of list
  operations.
- Integer group aggregation through `group_by`, returning count, sum, average,
  minimum, and maximum.
- Integer equi-join through `hash_join`, including duplicate multiplicity and
  SQL's rule that NULL does not join to NULL.

Python `None` and NumPy masked entries are SQL NULL. IEEE NaN remains a valid
floating-point value. Nullable vector results are NumPy masked arrays.

This is not a SQL parser, optimizer, connection object, or storage engine. It
does not cover tables, files, transactions, window functions, sorting,
strings, decimals, timestamps, nested NULL list elements, outer joins, or
general expression compilation. Import `mojo_duckdb`, not `duckdb`; the latter
remains available for the complete database API.

## Install

The checked-in Pixi manifest pins the tested Mojo nightly and installs Python,
NumPy, pytest, and DuckDB:

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-duckdb.so`. For a copied Python installation,
set `MOJO_DUCKDB_LIB` to an already-built shared library.

## Usage

```python
import numpy as np
import mojo_duckdb as mdb

values = np.ma.array([1.0, 2.0, 99.0, 4.0], mask=[0, 0, 1, 0])

print(mdb.sum(values))                    # 7.0
print(mdb.var_pop(values))                # 1.5555555555555554
print(mdb.filter(values, [True, False, True, True]))
                                             # [1.0 -- 4.0]

groups = mdb.group_by([10, 20, 10, 20], [1.0, 2.0, 3.0, None])
print(sorted((int(k), float(v)) for k, v in zip(groups.keys, groups.sum)))
                                             # [(10, 4.0), (20, 2.0)]

pairs = mdb.hash_join([7, 8, 8], [8, 8, 9])
print(sorted((int(i), int(j)) for i, j in zip(pairs.left, pairs.right)))
                                             # [(1, 0), (1, 1), (2, 0), (2, 1)]
```

The same code is checked in and runs with
`pixi run python examples/basic.py`.

## How it works

All kernels live in one Mojo compilation unit to avoid repeated compiler
startup cost. Exported functions use a C ABI and receive NumPy buffer addresses
as 64-bit integers. Mojo reconstructs typed mutable pointers with
`AnyOrigin[mut=True]`; Python owns every allocation, so no cross-language
allocator or release protocol is needed.

Values and validity are separate contiguous buffers, like DuckDB's vector
representation. Numeric values are row-major float64, keys and result indices
are int64, and validity/selection vectors are uint8. Dense arithmetic and list
metrics use native-width SIMD. Dense aggregates use vector reductions with a
scalar remainder and parallelize above an internal threshold; nullable
aggregates retain stable online updates. Compact integer group ranges use
direct indexing, with open addressing as the general fallback. Joins reuse
their hash build across count and materialization, while a validated contiguous
right-key range uses direct lookup.

No GPU path is included.

## Benchmarks

Measured with `pixi run bench`; each entry is the best of five warmed runs and
includes the public Python interface, output allocation, and materialization.
DuckDB used its configured 72 threads. Lower time is better; "DuckDB / Mojo"
above 1 means Mojo was faster.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; 72 logical CPUs; Linux
6.8.0-136-generic; DuckDB 1.5.5; threads=72.

| Kernel | Input | Mojo | DuckDB | DuckDB / Mojo | Result |
|---|---:|---:|---:|---:|---|
| fused numeric aggregates | 2,000,000 | 5.68 ms | 15.35 ms | 2.70x | faster |
| projection a * b + a | 5,000,000 | 120.65 ms | 157.20 ms | 1.30x | faster |
| filter/compact | 5,000,000 | 95.35 ms | 81.81 ms | 0.86x | slower |
| integer group by (4,096 groups) | 1,000,000 | 7.88 ms | 19.05 ms | 2.42x | faster |
| integer inner join (contiguous right keys) | 500,000 x 500,000 | 3.99 ms | 50.02 ms | 12.53x | faster |

Mojo was faster in four of the five interface-level cases in this run. Dense
NumPy inputs remain zero-copy across the FFI boundary.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The parity suite executes equivalent expressions in DuckDB and asserts values,
NULL behavior, NaN behavior, group aggregates, and join output multiplicity.
