"""Discover images for each dataset and turn them into RecordSpecs.

- 140k Real and Fake Faces: folder layout `real-vs-fake/{train,valid,test}/{real,fake}/*.jpg`.
- External sets (cross-generator, own photos): organised by `scripts/organize_external.py` into
  `<root>/<label>/<source>/<file>`; their split is the dataset name (they are test-only).
"""

from __future__ import annotations

import os
from pathlib import Path

from deeptrace_ml.data.image_meta import IMAGE_EXTENSIONS
from deeptrace_ml.data.manifest import LABELS, RecordSpec

SPLITS_140K = ("train", "valid", "test")
_140K_DIRNAME = "real-vs-fake"


def find_140k_root(search_root: Path, max_depth: int = 6) -> Path:
    """Locate the `real-vs-fake` folder anywhere below `search_root`.

    Kaggle has changed its /kaggle/input mount layout before; searching is more robust than
    hardcoding a path.
    """
    search_root = Path(search_root)
    if not search_root.is_dir():
        raise FileNotFoundError(f"Search root does not exist: {search_root}")
    base_depth = len(search_root.parts)
    for dirpath, dirnames, _ in os.walk(search_root):
        current = Path(dirpath)
        if len(current.parts) - base_depth >= max_depth:
            dirnames.clear()
            continue
        if current.name == _140K_DIRNAME and all((current / s).is_dir() for s in SPLITS_140K):
            return current
    raise FileNotFoundError(
        f"Could not find '{_140K_DIRNAME}/{{train,valid,test}}' under {search_root}. "
        "On Kaggle: Add Input -> '140k Real and Fake Faces'. Locally: download and unzip it first."
    )


def _list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def specs_140k(
    dataset_dir: Path, manifest_root: Path, dataset_name: str, real_source: str, fake_source: str
) -> list[RecordSpec]:
    """Specs for every image; `rel_path` is relative to `manifest_root`."""
    sources = {"real": real_source, "fake": fake_source}
    specs: list[RecordSpec] = []
    for split in SPLITS_140K:
        for label in LABELS:
            folder = dataset_dir / split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected folder missing: {folder}")
            images = _list_images(folder)
            if not images:
                raise FileNotFoundError(f"No images found in {folder}")
            specs.extend(
                RecordSpec(
                    rel_path=p.relative_to(manifest_root).as_posix(),
                    dataset=dataset_name,
                    source=sources[label],
                    label=label,
                    split=split,
                )
                for p in images
            )
    return specs


def specs_external(root: Path, dataset_name: str) -> list[RecordSpec]:
    """Specs for an organised external dataset: `<root>/<label>/<source>/<file>`."""
    root = Path(root)
    specs: list[RecordSpec] = []
    for label in LABELS:
        label_dir = root / label
        if not label_dir.is_dir():
            continue
        for source_dir in sorted(d for d in label_dir.iterdir() if d.is_dir()):
            specs.extend(
                RecordSpec(
                    rel_path=p.relative_to(root).as_posix(),
                    dataset=dataset_name,
                    source=source_dir.name,
                    label=label,
                    split=dataset_name,
                )
                for p in _list_images(source_dir)
            )
    if not specs:
        raise FileNotFoundError(f"No images under {root}/{{real,fake}}/<source>/")
    return specs
