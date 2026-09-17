"""Explainability galleries, error analysis and the model-randomisation sanity check.

Outputs (docs/plots/explain/<model>/ and reports/explain_<model>.json):
  * outcomes_<set>.png   seeded random TP / TN / FP / FN examples with every heatmap method
  * worst_errors_<set>.png  the most CONFIDENT false positives and false negatives
  * sanity check: rank correlation between heatmaps of the trained and a randomly re-initialised
    model. High correlation = heatmaps mostly show image structure, not learned evidence.

    python scripts/explain.py --checkpoint runs/effnet_b0_seed42/best.pt --model-name effnet_b0 \
        --predictions reports/predictions_effnet_b0.csv --calibration reports/calibration_effnet_b0.json \
        --data-root /kaggle/input/deeptrace-processed/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

from deeptrace_ml.evaluation.calibration import apply_calibration
from deeptrace_ml.evaluation.inference import load_predictions
from deeptrace_ml.evaluation.plots import plot_gallery
from deeptrace_ml.evaluation.test_sets import iter_test_sets
from deeptrace_ml.explain.closed_form import heatmap_rgba
from deeptrace_ml.explain.torch_cam import gradcam_maps, randomize_weights, rollout_maps
from deeptrace_ml.models.classifier import BinaryClassifier, load_checkpoint
from deeptrace_ml.preprocessing.pipeline import normalize_image
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import plots_dir, reports_dir

log = get_logger("explain")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--predictions", type=Path, action="append", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--sets", nargs="+", default=["id_test", "crossgen_full"])
    p.add_argument("--n-per-outcome", type=int, default=4)
    p.add_argument("--n-worst", type=int, default=8)
    p.add_argument("--n-sanity", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="Grad-CAM needs gradients; CPU is fine for galleries")
    return p.parse_args()


def methods_for(model: BinaryClassifier) -> list[str]:
    return ["gradcam", "gradcam++"] if model.spec.family == "cnn" else ["gradcam", "rollout"]


def compute_maps(model: BinaryClassifier, images: list[Image.Image], method: str, device: str) -> np.ndarray:
    batch = torch.from_numpy(np.concatenate([normalize_image(im, model.mean, model.std) for im in images])).to(device)
    if method == "rollout":
        return rollout_maps(model, batch)
    return gradcam_maps(model, batch, method=method, target="fake")


def overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    rgba = heatmap_rgba(cam, image.width).astype(np.float32)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    a = alpha * rgba[..., 3:4] / 255.0
    return (base * (1 - a) + rgba[..., :3] * a).astype(np.uint8)


def load_images(rows: pd.DataFrame, root: Path) -> list[Image.Image]:
    images = []
    for rel in rows["processed_rel_path"]:
        with Image.open(root / rel) as im:
            images.append(im.convert("RGB"))
    return images


def gallery(
    model: BinaryClassifier, rows: pd.DataFrame, root: Path, device: str, title: str, path: Path
) -> list[dict[str, Any]]:
    images = load_images(rows, root)
    methods = methods_for(model)
    maps = {m: compute_maps(model, images, m, device) for m in methods}
    entries, listing = [], []
    for i, (_, row) in enumerate(rows.iterrows()):
        entry: dict[str, Any] = {
            "image": np.asarray(images[i]),
            "caption": f"{row['outcome']} {row['source']} p={row['prob']:.2f}",
        }
        for m in methods:
            entry[m] = overlay(images[i], maps[m][i])
        entries.append(entry)
        listing.append(
            {
                "rel_path": row["rel_path"],
                "source": row["source"],
                "label": row["label"],
                "outcome": row["outcome"],
                "prob": float(row["prob"]),
            }
        )
    plot_gallery(entries, path, title, columns=["image", *methods])
    return listing


def sanity_check(model: BinaryClassifier, rows: pd.DataFrame, root: Path, device: str, seed: int) -> dict[str, Any]:
    images = load_images(rows, root)
    random_model = randomize_weights(model, seed)
    out: dict[str, Any] = {"n_images": len(images)}
    for method in methods_for(model):
        trained, randomized = (
            compute_maps(model, images, method, device),
            compute_maps(random_model, images, method, device),
        )
        corrs = [spearmanr(a.ravel(), b.ravel()).statistic for a, b in zip(trained, randomized, strict=True)]
        corrs = [c for c in corrs if np.isfinite(c)]
        out[method] = {
            "mean_spearman_trained_vs_random": float(np.mean(corrs)) if corrs else None,
            "interpretation": "near 0 is good: heatmaps depend on learned weights",
        }
    return out


def main() -> None:
    args = parse_args()
    model, _ = load_checkpoint(args.checkpoint)
    model.to(args.device)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    preds = load_predictions(args.predictions)
    preds["prob"] = apply_calibration(preds["logit"].to_numpy(), calibration["temperature"])
    threshold = calibration["threshold"]
    is_fake = preds["label"] == "fake"
    flagged = preds["prob"] >= threshold
    preds["outcome"] = np.select([is_fake & flagged, ~is_fake & ~flagged, ~is_fake & flagged], ["TP", "TN", "FP"], "FN")
    plots = plots_dir() / f"explain/{args.model_name}"

    report: dict[str, Any] = {"model": args.model_name, "methods": methods_for(model), "sets": {}}
    for name, _desc, rows in iter_test_sets(preds):
        if name not in args.sets:
            continue
        picked = pd.concat(
            [g.sample(min(args.n_per_outcome, len(g)), random_state=args.seed) for _, g in rows.groupby("outcome")]
        )
        errors_fp = rows[rows["outcome"] == "FP"].nlargest(args.n_worst // 2, "prob")
        errors_fn = rows[rows["outcome"] == "FN"].nsmallest(args.n_worst // 2, "prob")
        worst = pd.concat([errors_fp, errors_fn])
        report["sets"][name] = {
            "outcome_counts": rows["outcome"].value_counts().to_dict(),
            "outcome_gallery": gallery(
                model,
                picked,
                args.data_root,
                args.device,
                f"{args.model_name} — {name}: sampled outcomes",
                plots / f"outcomes_{name}.png",
            ),
        }
        if len(worst):
            report["sets"][name]["worst_errors"] = gallery(
                model,
                worst,
                args.data_root,
                args.device,
                f"{args.model_name} — {name}: most confident errors",
                plots / f"worst_errors_{name}.png",
            )
        log.info("%s: galleries saved", name)

    id_rows = preds[(preds["dataset"] == "140k") & (preds["split"] == "test")]
    if len(id_rows):
        sample = id_rows.sample(min(args.n_sanity, len(id_rows)), random_state=args.seed)
        report["sanity_check"] = sanity_check(model, sample, args.data_root, args.device, args.seed)
        log.info("Sanity check: %s", report["sanity_check"])
    write_json(report, reports_dir() / f"explain_{args.model_name}.json")


if __name__ == "__main__":
    main()
