"""Apply canonical preprocessing to a manifest: decode -> strip metadata -> MTCNN -> square crop ->
224x224 -> JPEG (quality from sha256). Produces the images every model trains/evaluates on.

    python scripts/prepare_dataset.py \
        --manifest data/manifests/140k_raw.csv.gz --root 140k=/kaggle/input/.../real-vs-fake \
        --duplicates data/manifests/duplicates_drop.csv \
        --out-root /kaggle/working/processed --device auto --batch-size 64

Resumable: progress is appended to <out-root>/progress_<manifest>.jsonl, so re-running after a
Kaggle timeout continues where it stopped. Every skipped image gets an explicit status
(read_error / no_face / duplicate_dropped) — nothing disappears silently.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from deeptrace_ml.config import PreprocessConfig, export_preprocess_json, load_preprocess_config
from deeptrace_ml.data.manifest import load_manifest, parse_roots, resolve_path
from deeptrace_ml.preprocessing.face import FaceDetector, MTCNNFaceDetector
from deeptrace_ml.preprocessing.io import ImageValidationError, load_image_path
from deeptrace_ml.preprocessing.pipeline import crop_faces, encode_offline
from deeptrace_ml.utils.env import write_environment
from deeptrace_ml.utils.hashing import stable_hash_int
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import reports_dir

log = get_logger("prepare_dataset")
ML_DIR = Path(__file__).resolve().parents[1]

PROCESSED_COLUMNS = [
    "dataset",
    "source",
    "label",
    "split",
    "rel_path",
    "sha256",
    "status",
    "error",
    "num_faces",
    "face_prob",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "crop_left",
    "crop_top",
    "crop_side",
    "crop_clamped",
    "jpeg_quality",
    "processed_rel_path",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--root", action="append", required=True, help="DATASET=PATH")
    p.add_argument("--config", type=Path, default=ML_DIR / "configs/preprocess.yaml")
    p.add_argument("--duplicates", type=Path, default=None, help="duplicates_drop.csv from dedupe.py")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument(
        "--out-manifest", type=Path, default=None, help="Default: data/manifests/<manifest name>_processed.csv.gz"
    )
    p.add_argument("--report", type=Path, default=None, help="Default: reports/prepare_<manifest name>.json")
    p.add_argument("--splits", nargs="*", default=None, help="Only process these splits")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (smoke test)")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--io-workers", type=int, default=8)
    return p.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for MTCNN. See ml/README.md for install commands.") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def manifest_stem(path: Path) -> str:
    name = path.name
    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_drop_keys(path: Path | None) -> set[tuple[str, str]]:
    if path is None:
        return set()
    drops = pd.read_csv(path, dtype=str)
    return set(zip(drops["dataset"], drops["rel_path"], strict=False))


def read_progress(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    done: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave a half-written LAST line; anything earlier is real corruption.
                log.warning("Ignoring unparsable progress line %d in %s", line_no, path)
                continue
            done[(rec["dataset"], rec["rel_path"])] = rec
    return done


def base_record(row: pd.Series) -> dict[str, Any]:
    rec: dict[str, Any] = dict.fromkeys(PROCESSED_COLUMNS)
    for col in ("dataset", "source", "label", "split", "rel_path", "sha256"):
        rec[col] = row[col]
    return rec


def output_rel_path(row: pd.Series) -> str:
    # sha prefix + rel_path hash: unique even when the same file appears twice in one split
    suffix = stable_hash_int(str(row["rel_path"]), 0) % 10**8
    return f"{row['dataset']}/{row['split']}/{row['label']}/{str(row['sha256'])[:16]}_{suffix:08d}.jpg"


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def process_batch(
    rows: list[pd.Series],
    roots: dict[str, Path],
    detector: FaceDetector,
    cfg: PreprocessConfig,
    out_root: Path,
    pool: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    def load(row: pd.Series) -> Image.Image | ImageValidationError:
        try:
            return load_image_path(resolve_path(roots, row["dataset"], row["rel_path"]), cfg.io)
        except ImageValidationError as exc:
            return exc
        except OSError as exc:
            return ImageValidationError("INVALID_IMAGE", str(exc))

    loaded = list(pool.map(load, rows))
    records = [base_record(r) for r in rows]
    ok_idx = [i for i, img in enumerate(loaded) if isinstance(img, Image.Image)]
    for i, img in enumerate(loaded):
        if isinstance(img, ImageValidationError):
            records[i].update(status="read_error", error=img.code + ": " + img.message)

    detections = detector.detect([loaded[i] for i in ok_idx]) if ok_idx else []
    for i, faces in zip(ok_idx, detections, strict=False):
        rec, row, image = records[i], rows[i], loaded[i]
        assert isinstance(image, Image.Image)
        rec["num_faces"] = len(faces)
        if not faces:
            rec.update(status="no_face")
            continue
        # Training images are single-subject: keep the top-ranked face (prob x area).
        cropped = crop_faces(image, faces[:1], cfg)[0]
        data, quality = encode_offline(cropped.image, str(row["sha256"]), cfg)
        rel = output_rel_path(row)
        write_atomic(out_root / rel, data)
        box, crop = cropped.face.box, cropped.crop
        rec.update(
            status="ok",
            face_prob=cropped.face.prob,
            box_x1=box.x1,
            box_y1=box.y1,
            box_x2=box.x2,
            box_y2=box.y2,
            crop_left=crop.left,
            crop_top=crop.top,
            crop_side=crop.side,
            crop_clamped=crop.clamped,
            jpeg_quality=quality,
            processed_rel_path=rel,
        )
    return records


def main() -> None:
    args = parse_args()
    cfg = load_preprocess_config(args.config)
    roots = parse_roots(args.root)
    stem = manifest_stem(args.manifest)
    out_manifest = args.out_manifest or ML_DIR / f"data/manifests/{stem.replace('_raw', '')}_processed.csv.gz"
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.manifest)
    if args.splits:
        manifest = manifest[manifest["split"].isin(args.splits)]
    if args.limit:
        manifest = manifest.head(args.limit)
    manifest = manifest.reset_index(drop=True)
    if manifest.empty:
        raise SystemExit("Nothing to process after filtering.")

    drop_keys = load_drop_keys(args.duplicates)
    progress_path = args.out_root / f"progress_{stem}.jsonl"
    done = read_progress(progress_path)

    todo: list[pd.Series] = []
    with progress_path.open("a", encoding="utf-8") as progress:

        def log_record(rec: dict[str, Any]) -> None:
            progress.write(json.dumps(rec) + "\n")
            done[(rec["dataset"], rec["rel_path"])] = rec

        for _, row in manifest.iterrows():
            key = (row["dataset"], row["rel_path"])
            prev = done.get(key)
            if prev is not None and (prev["status"] != "ok" or (args.out_root / prev["processed_rel_path"]).exists()):
                continue
            if pd.notna(row["read_error"]):
                log_record({**base_record(row), "status": "read_error", "error": str(row["read_error"])})
            elif key in drop_keys:
                log_record({**base_record(row), "status": "duplicate_dropped"})
            else:
                todo.append(row)
        progress.flush()

        log.info("%d rows total, %d already done, %d to process", len(manifest), len(manifest) - len(todo), len(todo))
        if todo:
            device = resolve_device(args.device)
            log.info("Loading MTCNN on %s", device)
            detector = MTCNNFaceDetector(cfg.face, device=device)
            with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
                batches = list(batched(todo, args.batch_size))
                for batch in tqdm(batches, desc="prepare", unit="batch"):
                    for rec in process_batch(list(batch), roots, detector, cfg, args.out_root, pool):
                        log_record(rec)
                    progress.flush()

    keys = set(zip(manifest["dataset"], manifest["rel_path"], strict=False))
    processed = pd.DataFrame([rec for key, rec in done.items() if key in keys], columns=PROCESSED_COLUMNS)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_manifest, index=False)

    ok = processed[processed["status"] == "ok"]
    summary = {
        "preprocess_version": cfg.version,
        "manifest": args.manifest.as_posix(),
        "rows": int(len(processed)),
        "status_counts": processed["status"].value_counts().to_dict(),
        "status_by_group": (
            processed.groupby(["dataset", "split", "source", "label", "status"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict("records")
        ),
        "crop_clamped_rate_by_source": ok.groupby("source")["crop_clamped"].mean().to_dict() if len(ok) else {},
        "multi_face_rate_by_source": (
            ok.assign(multi=ok["num_faces"].astype(int) > 1).groupby("source")["multi"].mean().to_dict()
            if len(ok)
            else {}
        ),
    }
    write_json(summary, args.report or reports_dir() / f"prepare_{stem}.json")
    export_preprocess_json(cfg, args.out_root / "preprocess.json")
    write_environment(args.out_root / f"env_prepare_{stem}.json", repo_dir=ML_DIR)
    log.info("Saved %s. Status counts: %s", out_manifest, summary["status_counts"])


if __name__ == "__main__":
    main()
