"""Near-duplicate detection across splits and datasets with perceptual hashes.

Why: if (almost) the same face sits in train and test, test accuracy measures memorisation.

Scaling trick (pigeonhole principle): split each 64-bit hash into (d + 1) chunks. Two hashes within
Hamming distance d must match EXACTLY on at least one chunk, so we only compare items that share a
chunk bucket instead of all ~10^10 pairs for 140k images.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

HASH_BITS = 64
_POPCOUNT_8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(values: np.ndarray) -> np.ndarray:
    """Number of set bits for each uint64 (works on numpy 1.x and 2.x)."""
    arr = np.ascontiguousarray(values, dtype=np.uint64)
    as_bytes = arr.view(np.uint8).reshape(*arr.shape, 8)
    return _POPCOUNT_8[as_bytes].sum(axis=-1, dtype=np.int64)


def _hash_one(path: str) -> tuple[int, int] | None:
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        return int(str(imagehash.phash(rgb, hash_size=8)), 16), int(str(imagehash.dhash(rgb, hash_size=8)), 16)
    except (OSError, ValueError):
        return None


def compute_hashes(paths: Sequence[Path], workers: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return (hashes[n, 2] uint64 as [phash, dhash], ok_mask[n]). Failed rows are zeros + False."""
    jobs = [str(p) for p in paths]
    if workers <= 1:
        results = [_hash_one(j) for j in tqdm(jobs, desc="hash")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(_hash_one, jobs, chunksize=256), total=len(jobs), desc="hash"))
    hashes = np.zeros((len(jobs), 2), dtype=np.uint64)
    ok = np.zeros(len(jobs), dtype=bool)
    for i, res in enumerate(results):
        if res is not None:
            hashes[i] = res
            ok[i] = True
    return hashes, ok


def _chunk_layout(max_distance: int) -> list[tuple[int, int]]:
    """(shift, bit_width) for each of the (max_distance + 1) chunks covering 64 bits."""
    n_chunks = max_distance + 1
    base, extra = divmod(HASH_BITS, n_chunks)
    layout, shift = [], 0
    for i in range(n_chunks):
        width = base + (1 if i < extra else 0)
        layout.append((shift, width))
        shift += width
    return layout


def find_near_duplicates(hashes: np.ndarray, max_distance: int, max_bucket_size: int = 50_000) -> pd.DataFrame:
    """Pairs (i < j) whose pHash AND dHash distances are both <= max_distance."""
    if hashes.ndim != 2 or hashes.shape[1] != 2:
        raise ValueError("hashes must have shape (n, 2)")
    phash = hashes[:, 0].astype(np.uint64)
    dhash = hashes[:, 1].astype(np.uint64)
    found: dict[tuple[int, int], tuple[int, int]] = {}

    for shift, width in _chunk_layout(max_distance):
        mask = np.uint64((1 << width) - 1)
        keys = (phash >> np.uint64(shift)) & mask
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
        for group in np.split(order, boundaries):
            if len(group) < 2:
                continue
            if len(group) > max_bucket_size:
                raise RuntimeError(
                    f"Hash bucket of {len(group)} items (e.g. many blank/identical images). "
                    "Inspect those files or lower max_distance."
                )
            group = np.sort(group)
            for a in range(len(group) - 1):
                i, rest = group[a], group[a + 1 :]
                dp = popcount64(phash[i] ^ phash[rest])
                dd = popcount64(dhash[i] ^ dhash[rest])
                close = (dp <= max_distance) & (dd <= max_distance)
                for j, p_dist, d_dist in zip(rest[close], dp[close], dd[close], strict=False):
                    found[(int(i), int(j))] = (int(p_dist), int(d_dist))

    rows = [(i, j, p, d) for (i, j), (p, d) in sorted(found.items())]
    return pd.DataFrame(rows, columns=["i", "j", "phash_dist", "dhash_dist"])


def resolve_drops(
    manifest: pd.DataFrame, pairs: pd.DataFrame, split_priority: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decide which image of each cross-split pair to drop.

    `manifest` must be positionally aligned with the hashes used to build `pairs`.
    Returns (all_pairs_annotated, drops) where drops has one row per dropped image.
    Same-split pairs are reported but never dropped (they don't leak across splits).
    """
    rank = {name: i for i, name in enumerate(split_priority)}
    unknown = set(manifest["split"].unique()) - set(rank)
    if unknown:
        raise ValueError(f"Splits missing from split_priority: {sorted(unknown)}")

    m = manifest.reset_index(drop=True)
    annotated = pairs.copy()
    for side in ("i", "j"):
        for col in ("dataset", "split", "rel_path", "label"):
            annotated[f"{col}_{side}"] = m.loc[annotated[side], col].to_numpy()

    rank_i = annotated["split_i"].map(rank)
    rank_j = annotated["split_j"].map(rank)
    annotated["cross_split"] = rank_i != rank_j
    annotated["label_conflict"] = annotated["label_i"] != annotated["label_j"]
    annotated["drop"] = np.where(~annotated["cross_split"], "none", np.where(rank_i > rank_j, "i", "j"))

    drop_rows = []
    for _, row in annotated[annotated["cross_split"]].iterrows():
        dropped, kept = ("i", "j") if row["drop"] == "i" else ("j", "i")
        drop_rows.append(
            {
                "dataset": row[f"dataset_{dropped}"],
                "split": row[f"split_{dropped}"],
                "rel_path": row[f"rel_path_{dropped}"],
                "duplicate_of_dataset": row[f"dataset_{kept}"],
                "duplicate_of_split": row[f"split_{kept}"],
                "duplicate_of_rel_path": row[f"rel_path_{kept}"],
                "phash_dist": row["phash_dist"],
                "dhash_dist": row["dhash_dist"],
            }
        )
    drops = pd.DataFrame(
        drop_rows,
        columns=[
            "dataset",
            "split",
            "rel_path",
            "duplicate_of_dataset",
            "duplicate_of_split",
            "duplicate_of_rel_path",
            "phash_dist",
            "dhash_dist",
        ],
    ).drop_duplicates(subset=["dataset", "rel_path"])
    return annotated, drops
