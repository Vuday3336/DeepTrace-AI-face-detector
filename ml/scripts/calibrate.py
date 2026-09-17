"""Fit temperature, decision threshold and uncertainty band on val_calib ONLY.

python scripts/calibrate.py --predictions reports/predictions_effnet_b0.csv --model-name effnet_b0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deeptrace_ml.config import read_yaml
from deeptrace_ml.evaluation.calibration import apply_calibration, calibrate
from deeptrace_ml.evaluation.inference import binary_labels, load_predictions
from deeptrace_ml.evaluation.metrics import sigmoid
from deeptrace_ml.evaluation.plots import plot_reliability
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.paths import plots_dir, reports_dir

log = get_logger("calibrate")
ML_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ML_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, action="append", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--eval-config", type=Path, default=ML_DIR / "configs/eval.yaml")
    p.add_argument("--split", default="val_calib")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cal_cfg = read_yaml(args.eval_config)["calibration"]
    preds = load_predictions(args.predictions)
    rows = preds[(preds["dataset"] == "140k") & (preds["split"] == args.split)]
    if rows.empty:
        raise SystemExit(f"No 140k/{args.split} rows in predictions — run predict.py with --splits {args.split}")
    logits, labels = rows["logit"].to_numpy(), binary_labels(rows)
    result = calibrate(
        logits,
        labels,
        float(cal_cfg["target_fpr"]),
        float(cal_cfg["max_abstention"]),
        float(cal_cfg["target_error_capture"]),
    )
    out = args.out or reports_dir() / f"calibration_{args.model_name}.json"
    write_json({"model": args.model_name, "fitted_on": f"140k/{args.split}", **result.to_dict()}, out)
    plot_reliability(
        labels,
        sigmoid(logits),
        apply_calibration(logits, result.temperature),
        plots_dir() / f"eval/{args.model_name}/reliability_val_calib.png",
        f"{args.model_name} — reliability on val_calib",
    )
    log.info(
        "T=%.3f threshold=%.4f delta=%.2f | ECE %.4f -> %.4f | FPR on calib %.3f",
        result.temperature,
        result.threshold,
        result.uncertainty_delta,
        result.ece_before,
        result.ece_after,
        result.achieved_fpr_on_calib,
    )


if __name__ == "__main__":
    main()
