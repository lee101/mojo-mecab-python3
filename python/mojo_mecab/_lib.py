"""Load the compiled Mojo kernels through a small ctypes boundary."""

from __future__ import annotations

import ctypes
import os
import subprocess
from numbers import Integral

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LIB_OVERRIDE = os.environ.get("MOJO_MECAB_LIB")
LIB = _LIB_OVERRIDE or os.path.join(ROOT, "dist", "libmojo-mecab-python3.so")

I = ctypes.c_int64

_SIGNATURES = {
    "mm_viterbi_decode": ([I, I, I, I, I, I, I], I),
    "mm_viterbi_decode_batch": ([I, I, I, I, I, I, I, I], I),
    "mm_utf8_boundaries": ([I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if _LIB_OVERRIDE:
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_MECAB_LIB does not exist: {LIB}")
    if not os.path.exists(source):
        if os.path.exists(LIB):
            return LIB
        raise BuildError("no Mojo source or compiled library was found")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(
        ["bash", script], capture_output=True, text=True, timeout=1800
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def i64(values) -> np.ndarray:
    source = np.asarray(values)
    if not source.size:
        return np.ascontiguousarray(source, dtype=np.int64)
    if source.dtype.kind == "O":
        lower, upper = np.iinfo(np.int64).min, np.iinfo(np.int64).max
        for value in source.flat:
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
            ):
                raise TypeError("integer arrays are required")
            if not lower <= int(value) <= upper:
                raise OverflowError("integer does not fit in int64")
        return np.ascontiguousarray(source, dtype=np.int64)
    if source.dtype.kind not in "iu":
        raise TypeError("integer arrays are required")
    lower, upper = np.iinfo(np.int64).min, np.iinfo(np.int64).max
    if int(source.min()) < lower or int(source.max()) > upper:
        raise OverflowError("integer does not fit in int64")
    return np.ascontiguousarray(source, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    return array.ctypes.data
