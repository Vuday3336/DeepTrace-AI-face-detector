"""Dataset manifests: one row per image with label, split, hash and file-level metadata.

Paths are stored RELATIVE to a per-dataset root, so the same manifest works on Kaggle
(/kaggle/input/...) and on your laptop. Map dataset names to roots with `parse_roots`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from deeptrace_ml.data.image_meta import ImageReadError, read_image_meta
from deeptrace_ml.utils.hashing import sha256_file

LABELS = ("real", "fake")

MANIFEST_COLUMNS = [
    "dataset",
    "source",
    "label",
    "split",
    "rel_path",
    "sha256",
    "format",
    "width",
    "height",
    "mode",
    "file_size",
    "jpeg_quality_est",
    "has_exif",
    "has_icc",
    "has_xmp",
    "read_error",
]


@dataclass(frozen=True)
class RecordSpec:
    rel_path: str
    dataset: str
    source: str
    label: str
    split: str


def parse_roots(items: Iterable[str]) -> dict[str, Path]:
    """Parse CLI values like `140k=/kaggle/input/140k` into {dataset: root}."""
    roots: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"Expected DATASET=PATH, got '{item}'")
        roots[name] = Path(path)
    return roots


def resolve_path(roots: Mapping[str, Path], dataset: str, rel_path: str) -> Path:
    if dataset not in roots:
        raise KeyError(f"No root given for dataset '{dataset}'. Known: {sorted(roots)}")
    return roots[dataset] / rel_path


def _scan_one(args: tuple[str, str]) -> dict[str, Any]:
    root, rel_path = args
    path = Path(root) / rel_path
    row: dict[str, Any] = {"rel_path": rel_path, "read_error": None}
    try:
        row["sha256"] = sha256_file(path)
        row.update(read_image_meta(path).to_dict())
    except (ImageReadError, OSError) as exc:
        row["read_error"] = str(exc)
    return row


def build_manifest(root: Path, specs: Sequence[RecordSpec], workers: int = 4) -> pd.DataFrame:
    """Hash + read metadata for every spec. Unreadable files are KEPT with `read_error` set."""
    if not specs:
        raise ValueError("No records to scan — check the dataset path.")
    jobs = [(str(root), s.rel_path) for s in specs]
    if workers <= 1:
        scanned = [_scan_one(j) for j in tqdm(jobs, desc="scan")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scanned = list(tqdm(pool.map(_scan_one, jobs, chunksize=256), total=len(jobs), desc="scan"))

    base = pd.DataFrame([s.__dict__ for s in specs])
    meta = pd.DataFrame(scanned)
    df = base.merge(meta, on="rel_path", how="left", validate="one_to_one")
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[MANIFEST_COLUMNS]
    validate_manifest(df)
    return df


def validate_manifest(df: pd.DataFrame) -> None:
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    bad_labels = set(df["label"].unique()) - set(LABELS)
    if bad_labels:
        raise ValueError(f"Unknown labels in manifest: {bad_labels}")
    dupes = df.duplicated(subset=["dataset", "rel_path"])
    if dupes.any():
        raise ValueError(f"{int(dupes.sum())} duplicate (dataset, rel_path) rows in manifest")


def save_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    validate_manifest(df)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)  # .csv.gz suffix -> gzip-compressed automatically
    return out


def load_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    df = pd.read_csv(p, dtype={"sha256": str, "rel_path": str, "split": str, "dataset": str})
    validate_manifest(df)
    return df


def summarize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per dataset/split/source/label, plus unreadable files."""
    return (
        df.assign(unreadable=df["read_error"].notna())
        .groupby(["dataset", "split", "source", "label"], dropna=False)
        .agg(images=("rel_path", "size"), unreadable=("unreadable", "sum"))
        .reset_index()
    )
