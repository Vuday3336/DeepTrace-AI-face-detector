"""Fine-tune a model on the processed 140k data (train split), early-stopping on val_select AUC.

    python scripts/train.py --config configs/train/effnet_b0.yaml \
        --manifest data/manifests/140k_processed.csv.gz --data-root /kaggle/input/deeptrace-processed/processed \
        --run-dir runs/effnet_b0_seed42

Seeds for the final comparison (3 runs):  --seed 42, --seed 43, --seed 44
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deeptrace_ml.training.config import load_train_config
from deeptrace_ml.training.data import load_processed_split
from deeptrace_ml.training.engine import train_model
from deeptrace_ml.utils.logging import get_logger

log = get_logger("train")
ML_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True, help="Processed manifest (140k_processed.csv.gz)")
    p.add_argument("--data-root", type=Path, required=True, help="Folder containing processed crops")
    p.add_argument("--run-dir", type=Path, default=None, help="Default: runs/<model>_seed<seed>")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--max-train-batches", type=int, default=None, help="Smoke test / budget cap per epoch")
    p.add_argument("--max-eval-batches", type=int, default=None)
    p.add_argument("--no-pretrained", action="store_true", help="Random init (tests only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(
        args.config,
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "train_fraction": args.train_fraction,
            "max_train_batches_per_epoch": args.max_train_batches,
            "max_eval_batches": args.max_eval_batches,
        },
    )
    run_dir = args.run_dir or ML_DIR / f"runs/{cfg.model}_seed{cfg.seed}"
    train_df = load_processed_split(
        args.manifest, splits=["train"], datasets=["140k"], fraction=cfg.train_fraction, seed=cfg.seed
    )
    val_df = load_processed_split(args.manifest, splits=["val_select"], datasets=["140k"])
    result = train_model(
        cfg, train_df, val_df, args.data_root, run_dir, device=args.device, pretrained=not args.no_pretrained
    )
    log.info("Best epoch %d, val_select AUC %.4f -> %s", result.best_epoch, result.best_val_auc, result.checkpoint)


if __name__ == "__main__":
    main()
