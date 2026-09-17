"""Logits for processed crops -> predictions CSV (input to calibrate / evaluate / explain).

    python scripts/predict.py --checkpoint runs/effnet_b0_seed42/best.pt \
        --manifest data/manifests/140k_processed.csv.gz --manifest data/manifests/crossgen_processed.csv.gz \
        --data-root /kaggle/input/deeptrace-processed/processed \
        --splits val_calib test crossgen own --out reports/predictions_effnet_b0.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deeptrace_ml.evaluation.inference import predict_frame
from deeptrace_ml.models.classifier import load_checkpoint
from deeptrace_ml.training.data import load_processed_split
from deeptrace_ml.training.engine import resolve_device
from deeptrace_ml.utils.logging import get_logger

log = get_logger("predict")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path, action="append", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["val_calib", "test", "crossgen", "own"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, extra = load_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    frames = []
    for manifest in args.manifest:
        try:
            frames.append(load_processed_split(manifest, splits=args.splits))
        except ValueError as exc:
            log.warning("Skipping %s: %s", manifest, exc)
    if not frames:
        raise SystemExit("No rows to predict for the requested splits.")
    frame = pd.concat(frames, ignore_index=True)
    log.info("Predicting %d crops with %s (checkpoint epoch %s)", len(frame), model.spec.name, extra.get("epoch"))
    preds = predict_frame(model, frame, args.data_root, device, args.batch_size, args.num_workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.out, index=False)
    log.info("Saved %s\n%s", args.out, preds.groupby(["dataset", "split", "label"]).size().to_string())


if __name__ == "__main__":
    main()
