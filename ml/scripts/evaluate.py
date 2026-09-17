"""Metrics on every test set (ID, cross-generator variants, own set) with bootstrap CIs.

    python scripts/evaluate.py --predictions reports/predictions_effnet_b0.csv \
        --calibration reports/calibration_effnet_b0.json --model-name effnet_b0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deeptrace_ml.config import read_yaml
from deeptrace_ml.evaluation.calibration import apply_calibration
from deeptrace_ml.evaluation.inference import binary_labels, load_predictions
from deeptrace_ml.evaluation.metrics import full_report, sigmoid
from deeptrace_ml.evaluation.plots import plot_confusion, plot_reliability, plot_roc_curves
from deeptrace_ml.evaluation.test_sets import iter_test_sets, per_source_rates
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import plots_dir, reports_dir

log = get_logger("evaluate")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, action="append", required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--eval-config", type=Path, default=ML_DIR / "configs/eval.yaml")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    boot = read_yaml(args.eval_config)["bootstrap"]
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    temperature, threshold = float(calibration["temperature"]), float(calibration["threshold"])
    preds = load_predictions(args.predictions)
    preds["prob"] = apply_calibration(preds["logit"].to_numpy(), temperature)
    plots = plots_dir() / f"eval/{args.model_name}"

    report: dict[str, Any] = {"model": args.model_name, "calibration": calibration, "test_sets": {}}
    curves = {}
    for name, description, rows in iter_test_sets(preds):
        labels, probs = binary_labels(rows), rows["prob"].to_numpy()
        metrics = full_report(labels, probs, threshold, int(boot["n_resamples"]), int(boot["seed"]))
        metrics["description"] = description
        metrics["per_source"] = per_source_rates(rows, "prob", threshold)
        report["test_sets"][name] = metrics
        curves[name] = (labels, probs)
        plot_confusion(
            metrics["at_tuned_threshold"]["confusion_matrix"],
            plots / f"confusion_{name}.png",
            f"{args.model_name} — {name}",
        )
        plot_reliability(
            labels,
            sigmoid(rows["logit"].to_numpy()),
            probs,
            plots / f"reliability_{name}.png",
            f"{args.model_name} — {name}",
        )
        log.info(
            "%-28s n=%5d AUC=%s bal_acc=%.4f FPR=%.3f TPR=%.3f ECE=%.4f",
            name,
            metrics["n"],
            f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "n/a",
            metrics["at_tuned_threshold"]["balanced_accuracy"],
            metrics["at_tuned_threshold"]["fpr"],
            metrics["at_tuned_threshold"]["recall_tpr"],
            metrics["ece_15"],
        )
    if not report["test_sets"]:
        raise SystemExit("No test set had both classes. Did predict.py include the test/crossgen/own splits?")
    plot_roc_curves(curves, plots / "roc_all_sets.png", f"{args.model_name} — ROC per test set")
    write_json(report, args.out or reports_dir() / f"eval_{args.model_name}.json")


if __name__ == "__main__":
    main()
