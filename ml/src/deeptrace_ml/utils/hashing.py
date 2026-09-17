"""Content hashing and order-independent deterministic randomness.

`stable_uniform_int` derives a "random" value from a key (e.g. a file's sha256) instead of a global
RNG stream. The result for a file never depends on how many files were processed before it, on
worker count, or on resuming a crashed job — which a sequential RNG cannot guarantee.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_int(key: str, seed: int) -> int:
    """64-bit integer derived from (seed, key); identical across machines and Python runs."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def stable_uniform_int(key: str, seed: int, low: int, high: int) -> int:
    """Deterministic integer in [low, high] (inclusive) for a given key and seed."""
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return low + stable_hash_int(key, seed) % (high - low + 1)
