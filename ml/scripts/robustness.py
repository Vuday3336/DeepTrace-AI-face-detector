"""Accuracy vs JPEG quality / downscaling / blur / screenshot chains, through the SERVING pipeline.

Each sampled image: original file -> perturbation -> (known face box, rescaled) -> square crop ->
224px -> JPEG at inference quality -> model -> calibrated probability. Reusing the face box isolates
the classifier's robustness from detector failures (detector misses are a separate question).

    python scripts/robustness.py --checkpoint runs/effnet_b0_seed42/best.pt --model-name effnet_b0 \
        --processed-manifest data/manifests/140k_processed.csv.gz \
        --processed-manifest data/manifests/crossgen_processed.csv.gz \
        --root 140k=<140k real-vs-fake folder> --root crossgen=/kaggle/working/raw/crossgen \
        --calibration reports/calibration_effnet_b0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from deeptrace_ml.config import load_preprocess_config, read_yaml
from deeptrace_ml.data.manifest import parse_roots, resolve_path
from deeptrace_ml.evaluation.calibration import apply_calibration
from deeptrace_ml.evaluation.inference import logits_for_images
from deeptrace_ml.evaluation.metrics import ranking_metrics, threshold_metrics
from deeptrace_ml.evaluation.perturbations import build_grid
from deeptrace_ml.evaluation.plots import plot_robustness
from deeptrace_ml.evaluation.test_sets import iter_test_sets
from deeptrace_ml.models.classifier import load_checkpoint
from deeptrace_ml.preprocessing.crop import Box, crop_and_resize, square_crop_box
from deeptrace_ml.preprocessing.io import ImageValidationError, load_image_path
from deeptrace_ml.preprocessing.pipeline import to_inference_image
from deeptrace_ml.training.engine import resolve_device
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import plots_dir, reports_dir

log = get_logger("robustness")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--processed-manifest", type=Path, action="append", required=True)
    p.add_argument("--root", action="append", required=True, help="DATASET=PATH of the RAW images")
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--sets", nargs="+", default=["id_test", "crossgen_full"])
    p.add_argument("--preprocess-config", type=Path, default=ML_DIR / "configs/preprocess.yaml")
    p.add_argument("--eval-config", type=Path, default=ML_DIR / "configs/eval.yaml")
    p.add_argument("--n-per-class", type=int, default=None, help="Override eval.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def face_crop(image: Image.Image, row: pd.Series, scale: float, cfg: Any) -> Image.Image:
    box = Box(row["box_x1"] * scale, row["box_y1"] * scale, row["box_x2"] * scale, row["box_y2"] * scale)
    crop = square_crop_box(box, image.width, image.height, cfg.crop.margin_ratio)
    return to_inference_image(crop_and_resize(image, crop, cfg.crop.output_size, cfg.crop.resample), cfg)


def main() -> None:
    args = parse_args()
    pre_cfg = load_preprocess_config(args.preprocess_config)
    rob_cfg = read_yaml(args.eval_config)["robustness"]
    n_per_class = args.n_per_class or int(rob_cfg["n_per_class"])
    seed = int(rob_cfg["seed"])
    roots = parse_roots(args.root)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    model, _ = load_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    grid = build_grid({k: list(v) for k, v in rob_cfg["grid"].items()}, list(rob_cfg["screenshot_seeds"]))

    processed = pd.concat(
        [pd.read_csv(m, dtype={"split": str, "rel_path": str}) for m in args.processed_manifest], ignore_index=True
    )
    processed = processed[processed["status"] == "ok"]

    results: dict[str, list[dict[str, Any]]] = {}
    for name, _desc, rows in iter_test_sets(processed):
        if name not in args.sets:
            continue
        sample = rows.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(n_per_class, len(g)), random_state=seed)
        )
        originals: list[tuple[pd.Series, Image.Image]] = []
        for _, row in sample.iterrows():
            try:
                originals.append(
                    (row, load_image_path(resolve_path(roots, row["dataset"], row["rel_path"]), pre_cfg.io))
                )
            except (ImageValidationError, OSError, KeyError) as exc:
                log.warning("Skipping %s: %s", row["rel_path"], exc)
        if not originals:
            raise SystemExit(f"No raw images loadable for {name}; check --root values")
        labels = np.array([int(r["label"] == "fake") for r, _ in originals])
        log.info("%s: %d images x %d perturbations", name, len(originals), len(grid))

        rows_out = []
        for spec in tqdm(grid, desc=name):
            crops = []
            for row, image in originals:
                perturbed, scale = spec.fn(image)
                crops.append(face_crop(perturbed, row, scale, pre_cfg))
            probs = apply_calibration(
                logits_for_images(model, crops, device, args.batch_size), calibration["temperature"]
            )
            at_t = threshold_metrics(labels, probs, calibration["threshold"])
            rows_out.append(
                {
                    "family": spec.family,
                    "level": spec.level,
                    "n": int(len(labels)),
                    "roc_auc": ranking_metrics(labels, probs)["roc_auc"],
                    "balanced_accuracy": at_t["balanced_accuracy"],
                    "accuracy": at_t["accuracy"],
                    "fpr": at_t["fpr"],
                    "tpr": at_t["recall_tpr"],
                }
            )
        results[name] = rows_out

    if not results:
        raise SystemExit(f"None of the requested sets {args.sets} exist in the processed manifests")
    plots = plots_dir() / f"robustness/{args.model_name}"
    for family in ("jpeg_quality", "downscale", "blur_sigma"):
        for metric in ("balanced_accuracy", "roc_auc"):
            plot_robustness(
                results, family, metric, plots / f"{family}_{metric}.png", f"{args.model_name}: {metric} vs {family}"
            )
    write_json(
        {
            "model": args.model_name,
            "n_per_class": n_per_class,
            "seed": seed,
            "note": "Face box reused from clean image (scaled); isolates classifier robustness.",
            "results": results,
        },
        args.out or reports_dir() / f"robustness_{args.model_name}.json",
    )
    log.info("Saved robustness report and plots in %s", plots)


if __name__ == "__main__":
    main()
