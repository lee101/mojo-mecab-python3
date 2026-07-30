"""Mojo lattice kernels with the mecab-python3 compatibility surface."""

from __future__ import annotations

import MeCab as _MeCab

from .lattice import (
    BatchDecodeResult,
    DecodeResult,
    PackedLattice,
    decode,
    decode_batch,
    decoded_surfaces,
    from_mecab_lattice,
    make_lattice,
    mecab_lattice,
    reference_decode,
)
from .utf8 import utf8_boundaries

for _name in _MeCab.__all__:
    if _name not in globals():
        globals()[_name] = getattr(_MeCab, _name)

VERSION = _MeCab.VERSION

__all__ = list(_MeCab.__all__) + [
    "BatchDecodeResult",
    "DecodeResult",
    "PackedLattice",
    "decode",
    "decode_batch",
    "decoded_surfaces",
    "from_mecab_lattice",
    "make_lattice",
    "mecab_lattice",
    "reference_decode",
    "utf8_boundaries",
]
