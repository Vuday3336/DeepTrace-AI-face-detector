"""Index the Kaggle "140k Real and Fake Faces" dataset into a manifest.

Example (Kaggle):
    python scripts/build_manifest_140k.py --search-root /kaggle/input --workers 4
Example (local, after `kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw --unzip`):
    python scripts/build_manifest_140k.py --search-root data/raw --workers 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deeptrace_ml.config import load_data_config
from deeptrace_ml.data.datasets import find_140k_root, specs_140k
from deeptrace_ml.data.manifest import build_manifest, save_manifest, summarize_manifest
from deeptrace_ml.data.splits import split_validation
from deeptrace_ml.utils.logging import get_logger

log = get_logger("build_manifest_140k")
ML_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search-root", type=Path, required=True, help="Folder containing the unzipped dataset")
    p.add_argument("--data-config", type=Path, default=ML_DIR / "configs/data.yaml")
    p.add_argument("--out", type=Path, default=ML_DIR / "data/manifests/140k_raw.csv.gz")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_data_config(args.data_config)
    dataset_dir = find_140k_root(args.search_root)
    log.info("Found dataset at %s", dataset_dir)

    specs = specs_140k(
        dataset_dir=dataset_dir,
        manifest_root=dataset_dir,
        dataset_name=cfg.dataset_140k_name,
        real_source=cfg.real_source_140k,
        fake_source=cfg.fake_source_140k,
    )
    log.info("Scanning %d images (hash + metadata + full decode)...", len(specs))
    df = build_manifest(dataset_dir, specs, workers=args.workers)
    df = split_validation(df, cfg.seed, cfg.valid_source_split, cfg.val_first, cfg.val_second)

    unreadable = int(df["read_error"].notna().sum())
    if unreadable:
        log.warning("%d unreadable files (kept in manifest with read_error; excluded later)", unreadable)
    save_manifest(df, args.out)
    log.info("Saved %s\n%s", args.out, summarize_manifest(df).to_string(index=False))
    log.info("Use this root in later steps:  --root %s=%s", cfg.dataset_140k_name, dataset_dir)


if __name__ == "__main__":
    main()
