"""Deterministic, stratified split of the official validation set into val_select / val_calib.

Why hash-based ordering instead of `train_test_split(random_state=...)`: the assignment of an image
depends only on its content hash and the seed — not on file listing order, which can differ between
operating systems and Kaggle mounts.
"""

from __future__ import annotations

import pandas as pd

from deeptrace_ml.utils.hashing import stable_hash_int


def split_validation(df: pd.DataFrame, seed: int, source_split: str, first: str, second: str) -> pd.DataFrame:
    """Return a copy where rows of `source_split` become `first` or `second` (50/50 per label)."""
    if first == second:
        raise ValueError("first and second subsplit names must differ")
    out = df.copy()
    mask = out["split"] == source_split
    if not mask.any():
        raise ValueError(f"No rows with split '{source_split}' to subdivide")

    for label, group in out[mask].groupby("label"):
        # Tie-break on rel_path so identical files (same sha256) still get a stable order.
        keys = group.apply(lambda r: (stable_hash_int(str(r["sha256"]), seed), str(r["rel_path"])), axis=1)
        ordered = keys.sort_values().index
        half = len(ordered) // 2
        out.loc[ordered[:half], "split"] = first
        out.loc[ordered[half:], "split"] = second
        if half == 0:
            raise ValueError(f"Label '{label}' has too few rows ({len(ordered)}) to split")
    return out
