"""ctypes bindings for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_DUCKDB_LIB") or os.path.join(
    ROOT, "dist", "libmojo-duckdb.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mdb_binary": ([I, I, I, I, I], None),
    "mdb_compare": ([I, I, I, I, I], None),
    "mdb_aggregate": ([I, I, I, I], None),
    "mdb_aggregate_dense": ([I, I, I, I], None),
    "mdb_bivariate": ([I, I, I, I, I, I], None),
    "mdb_compact": ([I, I, I, I, I, I], I),
    "mdb_list_metric": ([I, I, I, I, I, I], None),
    "mdb_group_i64": ([I] * 13, I),
    "mdb_group_dense_i64": ([I] * 10, I),
    "mdb_hash_join_i64": ([I] * 12, I),
    "mdb_hash_join_dense_i64": ([I] * 10, I),
    "mdb_range_join_dense_i64": ([I] * 6, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_DUCKDB_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_DUCKDB_LIB does not exist: {LIB}")
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    stale = force or not os.path.exists(LIB)
    if not stale and sources:
        stale = os.path.getmtime(LIB) < max(os.path.getmtime(p) for p in sources)
    if stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def addr(array) -> int:
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null NumPy buffer to Mojo")
    return address
