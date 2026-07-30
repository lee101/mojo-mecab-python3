"""UTF-8 byte indexing used by MeCab lattices."""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib


def utf8_boundaries(text: str) -> np.ndarray:
    """Return every code-point boundary as a UTF-8 byte offset."""

    encoded = text.encode("utf-8")
    byte_count = len(encoded)
    data = np.empty(max(1, byte_count), dtype=np.uint8)
    if encoded:
        data[:byte_count] = np.frombuffer(encoded, dtype=np.uint8)
    boundaries = np.empty(byte_count + 1, dtype=np.int64)
    count = lib().mm_utf8_boundaries(
        addr(data), byte_count, addr(boundaries), boundaries.size
    )
    if not count:
        raise ValueError("invalid UTF-8")
    return boundaries[:count]
