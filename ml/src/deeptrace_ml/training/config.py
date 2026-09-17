"""Training configuration (one YAML per model in configs/train/)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from deeptrace_ml.config import ConfigError
from deeptrace_ml.models.registry import get_spec


@dataclass(frozen=True)
class AugmentConfig:
    jpeg_quality_min: int = 30
    jpeg_quality_max: int = 100
    jpeg_p: float = 0.5
    blur_sigma_max: float = 2.0
    blur_p: float = 0.2
    downscale_min: float = 0.25
    downscale_p: float = 0.3
    color_jitter: float = 0.1
    color_jitter_p: float = 0.3
    hflip_p: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    model: str
    seed: int = 42
    epochs: int = 15
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    early_stopping_patience: int = 3
    amp: bool = True
    num_workers: int = 4
    grad_clip_norm: float | None = 1.0
    max_train_batches_per_epoch: int | None = None  # budget / smoke-test cap
    max_eval_batches: int | None = None
    train_fraction: float = 1.0  # subsample train split (seeded) if GPU time is short
    deterministic: bool = False
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    # CLIP linear probe only
    probe_c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate(cfg: TrainConfig) -> None:
    get_spec(cfg.model)  # raises on unknown model
    checks = [
        (cfg.epochs >= 1, "epochs must be >= 1"),
        (cfg.batch_size >= 1, "batch_size must be >= 1"),
        (cfg.lr_backbone >= 0 and cfg.lr_head > 0, "learning rates must be positive"),
        (0 <= cfg.warmup_epochs < cfg.epochs, "warmup_epochs must be in [0, epochs)"),
        (cfg.early_stopping_patience >= 1, "early_stopping_patience must be >= 1"),
        (0 < cfg.train_fraction <= 1, "train_fraction must be in (0, 1]"),
        (1 <= cfg.augment.jpeg_quality_min <= cfg.augment.jpeg_quality_max <= 100, "bad JPEG quality range"),
        (0 < cfg.augment.downscale_min < 1, "downscale_min must be in (0, 1)"),
        (len(cfg.probe_c_grid) > 0 and all(c > 0 for c in cfg.probe_c_grid), "probe_c_grid must be positive"),
    ]
    for ok, message in checks:
        if not ok:
            raise ConfigError(message)


def load_train_config(path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Training config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown training config keys: {sorted(unknown)}")
    augment_raw = raw.pop("augment", {}) or {}
    aug_known = {f.name for f in fields(AugmentConfig)}
    if set(augment_raw) - aug_known:
        raise ConfigError(f"Unknown augment keys: {sorted(set(augment_raw) - aug_known)}")
    if "probe_c_grid" in raw:
        raw["probe_c_grid"] = tuple(float(c) for c in raw["probe_c_grid"])
    cfg = TrainConfig(**raw, augment=AugmentConfig(**augment_raw))
    _validate(cfg)
    return cfg
