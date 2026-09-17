"""Organise test-only image collections (cross-generator set, your own photos) + build a manifest.

Each --source is NAME:LABEL:PATH (PATH may itself contain ':' e.g. D:\\photos). Images are found
recursively, verified by magic bytes, sampled deterministically, and COPIED to
    <out-root>/<label>/<name>/<sha256[:16]>.<ext>
Originals are never modified.

Cross-generator set (Kaggle):
    python scripts/organize_external.py --dataset crossgen --out-root data/raw/crossgen \
        --source celebahq:real:/kaggle/input/celebahq-folder \
        --source sdxl:fake:/kaggle/input/deeptrace-generated/sdxl \
        --source flux_schnell:fake:/kaggle/input/deeptrace-generated/flux-schnell \
        --max-per-source 1000

Your own photos (local):
    python scripts/organize_external.py --dataset own --out-root data/raw/own \
        --source phone:real:"D:/Pictures/deeptrace_phone" --source mygen:fake:"D:/Pictures/deeptrace_ai"
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from deeptrace_ml.data.datasets import specs_external
from deeptrace_ml.data.image_meta import IMAGE_EXTENSIONS, SUPPORTED_FORMATS, detect_format
from deeptrace_ml.data.manifest import LABELS, build_manifest, save_manifest, summarize_manifest
from deeptrace_ml.utils.hashing import sha256_file, stable_hash_int
from deeptrace_ml.utils.logging import get_logger

log = get_logger("organize_external")
ML_DIR = Path(__file__).resolve().parents[1]
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_EXT = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    label: str
    path: Path


def parse_source(value: str) -> SourceSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected NAME:LABEL:PATH, got '{value}'")
    name, label, path = parts
    if not _NAME_RE.match(name):
        raise argparse.ArgumentTypeError(f"Source name '{name}' must match {_NAME_RE.pattern}")
    if label not in LABELS:
        raise argparse.ArgumentTypeError(f"Label must be one of {LABELS}, got '{label}'")
    if not Path(path).is_dir():
        raise argparse.ArgumentTypeError(f"Source folder does not exist: {path}")
    return SourceSpec(name, label, Path(path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Dataset name, also used as its split (e.g. crossgen, own)")
    p.add_argument("--source", type=parse_source, action="append", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None, help="Default: data/manifests/<dataset>_raw.csv.gz")
    p.add_argument("--max-per-source", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def find_images(folder: Path) -> list[Path]:
    """Recursively list files whose MAGIC BYTES are JPEG/PNG/WebP (extensions are only a pre-filter)."""
    found: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with path.open("rb") as fh:
            if detect_format(fh.read(16)) in SUPPORTED_FORMATS:
                found.append(path)
    return found


def sample_paths(paths: list[Path], root: Path, limit: int | None, seed: int) -> list[Path]:
    """Deterministic sample: order by a hash of the relative path, take the first `limit`."""
    if limit is None or len(paths) <= limit:
        return paths
    ranked = sorted(paths, key=lambda p: stable_hash_int(p.relative_to(root).as_posix(), seed))
    return sorted(ranked[:limit])


def copy_source(src: SourceSpec, out_root: Path, limit: int | None, seed: int) -> Counter[str]:
    stats: Counter[str] = Counter()
    candidates = find_images(src.path)
    if not candidates:
        raise FileNotFoundError(f"No JPEG/PNG/WebP images found in {src.path}")
    chosen = sample_paths(candidates, src.path, limit, seed)
    dest_dir = out_root / src.label / src.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for path in chosen:
        digest = sha256_file(path)
        if digest in seen:
            stats["identical_skipped"] += 1
            continue
        seen.add(digest)
        with path.open("rb") as fh:
            fmt = detect_format(fh.read(16))
        dest = dest_dir / f"{digest[:16]}{_EXT[fmt]}"
        if dest.exists():
            stats["already_present"] += 1
            continue
        shutil.copy2(path, dest)
        stats["copied"] += 1
    stats["found"] = len(candidates)
    stats["selected"] = len(chosen)
    return stats


def main() -> None:
    args = parse_args()
    if not _NAME_RE.match(args.dataset):
        raise SystemExit(f"--dataset must match {_NAME_RE.pattern}")
    manifest_path = args.manifest or ML_DIR / f"data/manifests/{args.dataset}_raw.csv.gz"

    for src in args.source:
        stats = copy_source(src, args.out_root, args.max_per_source, args.seed)
        log.info("%s (%s): %s", src.name, src.label, dict(stats))

    specs = specs_external(args.out_root, args.dataset)
    df = build_manifest(args.out_root, specs, workers=args.workers)
    save_manifest(df, manifest_path)
    log.info("Saved %s\n%s", manifest_path, summarize_manifest(df).to_string(index=False))

    counts = df[df["read_error"].isna()]["label"].value_counts()
    real, fake = int(counts.get("real", 0)), int(counts.get("fake", 0))
    if min(real, fake) == 0:
        log.warning("Only one class present (real=%d, fake=%d): metrics like AUC will be undefined.", real, fake)
    elif max(real, fake) / min(real, fake) > 1.1:
        log.warning("Classes are imbalanced (real=%d, fake=%d). Report balanced metrics or re-sample.", real, fake)
    if int(df["has_exif"].astype(str).str.lower().eq("true").sum()):
        log.warning(
            "Some copies still contain EXIF (possibly GPS location). %s is git-ignored — never commit "
            "or publish raw personal photos. Processed crops have all metadata stripped.",
            args.out_root,
        )
    log.info("Use this root in later steps:  --root %s=%s", args.dataset, args.out_root)


if __name__ == "__main__":
    main()
