"""Frozen CLIP ViT-B/16 + logistic-regression probe (C chosen on val_select).

    python scripts/train_clip_probe.py --config configs/train/clip_probe.yaml \
        --manifest data/manifests/140k_processed.csv.gz --data-root /kaggle/input/deeptrace-processed/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from deeptrace_ml.models.classifier import build_model, save_checkpoint
from deeptrace_ml.training.config import load_train_config
from deeptrace_ml.training.data import ProcessedFaceDataset, eval_transform, load_processed_split
from deeptrace_ml.training.engine import make_loader, resolve_device
from deeptrace_ml.training.probe import extract_features, fit_probe, install_head
from deeptrace_ml.utils.env import write_environment
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.seed import seed_everything

log = get_logger("train_clip_probe")
ML_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=ML_DIR / "configs/train/clip_probe.yaml")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true", help="Random init (tests only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(
        args.config, {"seed": args.seed, "train_fraction": args.train_fraction, "num_workers": args.num_workers}
    )
    seed_everything(cfg.seed)
    run_dir = args.run_dir or ML_DIR / f"runs/{cfg.model}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    write_environment(run_dir / "env.json", repo_dir=ML_DIR)

    device = resolve_device(args.device)
    model = build_model(cfg.model, pretrained=not args.no_pretrained).to(device)
    if not model.spec.frozen_backbone:
        raise SystemExit(f"{cfg.model} is not a frozen-backbone model; use scripts/train.py")

    features = {}
    for split in ("train", "val_select"):
        frame = load_processed_split(
            args.manifest,
            splits=[split],
            datasets=["140k"],
            fraction=cfg.train_fraction if split == "train" else 1.0,
            seed=cfg.seed,
        )
        loader = make_loader(
            ProcessedFaceDataset(frame, args.data_root, eval_transform(model.mean, model.std)), cfg, shuffle=False
        )
        log.info("Extracting %s features (%d images)", split, len(frame))
        features[split] = extract_features(model, loader, device, cfg.amp)
        np.savez_compressed(run_dir / f"features_{split}.npz", x=features[split][0], y=features[split][1])

    weights, bias, result = fit_probe(*features["train"], *features["val_select"], cfg.probe_c_grid, cfg.seed)
    install_head(model, weights, bias)
    best_auc = result.val_auc_by_c[result.best_c]
    save_checkpoint(
        model.cpu(), run_dir / "best.pt", extra={"probe_c": result.best_c, "val_auc": best_auc, "config": cfg.to_dict()}
    )
    write_json(
        {
            "model": cfg.model,
            "best_c": result.best_c,
            "val_auc_by_c": result.val_auc_by_c,
            "best_val_auc": best_auc,
            "checkpoint": run_dir / "best.pt",
        },
        run_dir / "final_metrics.json",
    )
    log.info("Probe C=%g val_select AUC=%.4f -> %s", result.best_c, best_auc, run_dir / "best.pt")


if __name__ == "__main__":
    main()
