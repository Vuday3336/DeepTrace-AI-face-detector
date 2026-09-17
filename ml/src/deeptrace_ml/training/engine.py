"""Fine-tuning loop: AdamW + warmup/cosine + AMP + early stopping on validation ROC-AUC.

Everything a run produces lands in its run directory:
  config.yaml, env.json, metrics_epoch.jsonl, best.pt, final_metrics.json, val_select_predictions.csv
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from deeptrace_ml.models.classifier import BinaryClassifier, build_model, save_checkpoint
from deeptrace_ml.training.config import TrainConfig
from deeptrace_ml.training.data import ProcessedFaceDataset, eval_transform, train_transform
from deeptrace_ml.utils.env import write_environment
from deeptrace_ml.utils.jsonio import write_json
from deeptrace_ml.utils.logging import get_logger
from deeptrace_ml.utils.seed import seed_everything, worker_init_fn

log = get_logger("train")


@dataclass
class TrainResult:
    best_epoch: int
    best_val_auc: float
    checkpoint: Path
    epochs_run: int
    stopped_early: bool


def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear 0->1 over warmup, then cosine 1->0."""

    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def predict_logits(
    model: BinaryClassifier, loader: DataLoader, device: torch.device, amp: bool, max_batches: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    use_amp = amp and device.type == "cuda"
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu().numpy())
        labels.append(y.numpy())
    if not logits:
        raise RuntimeError("Evaluation loader produced no batches")
    return np.concatenate(logits), np.concatenate(labels)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        raise ValueError("Validation set needs both classes to compute ROC-AUC")
    return float(roc_auc_score(labels, scores))


def make_loader(dataset: ProcessedFaceDataset, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) > cfg.batch_size,
        worker_init_fn=worker_init_fn if cfg.num_workers > 0 else None,
        generator=generator,
        persistent_workers=cfg.num_workers > 0,
    )


def train_model(
    cfg: TrainConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    data_root: Path,
    run_dir: Path,
    device: str = "auto",
    pretrained: bool = True,
) -> TrainResult:
    seed_everything(cfg.seed, cfg.deterministic)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    write_environment(run_dir / "env.json")
    dev = resolve_device(device)

    model = build_model(cfg.model, pretrained=pretrained).to(dev)
    if model.spec.frozen_backbone:
        raise ValueError(f"{cfg.model} is a linear probe — use scripts/train_clip_probe.py")
    log.info("Model %s on %s | train=%d val=%d", cfg.model, dev, len(train_df), len(val_df))

    train_loader = make_loader(
        ProcessedFaceDataset(train_df, data_root, train_transform(cfg.augment, model.mean, model.std)), cfg, True
    )
    val_loader = make_loader(ProcessedFaceDataset(val_df, data_root, eval_transform(model.mean, model.std)), cfg, False)

    steps_per_epoch = len(train_loader)
    if cfg.max_train_batches_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_train_batches_per_epoch)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = torch.optim.AdamW(model.param_groups(cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, int(cfg.warmup_epochs * steps_per_epoch))
    )
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_epoch, bad_epochs = -1.0, -1, 0
    checkpoint = run_dir / "best.pt"
    history_path = run_dir / "metrics_epoch.jsonl"
    epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        start, running, seen = time.perf_counter(), 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} step {step}: lower the LR")
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        logits, labels = predict_logits(model, val_loader, dev, cfg.amp, cfg.max_eval_batches)
        val_loss = float(
            nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits), torch.from_numpy(labels)).item()
        )
        val_auc = safe_auc(labels, logits)
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "seconds": round(time.perf_counter() - start, 1),
        }
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info(
            "epoch %d | train_loss %.4f | val_loss %.4f | val_auc %.4f | %.0fs",
            epoch,
            record["train_loss"],
            val_loss,
            val_auc,
            record["seconds"],
        )

        if val_auc > best_auc:
            best_auc, best_epoch, bad_epochs = val_auc, epoch, 0
            save_checkpoint(model, checkpoint, extra={"epoch": epoch, "val_auc": val_auc, "config": cfg.to_dict()})
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                log.info("Early stopping: no val AUC improvement for %d epochs", bad_epochs)
                break

    result = TrainResult(best_epoch, best_auc, checkpoint, epoch, epoch < cfg.epochs)
    write_json({**result.__dict__, "model": cfg.model}, run_dir / "final_metrics.json")
    return result
