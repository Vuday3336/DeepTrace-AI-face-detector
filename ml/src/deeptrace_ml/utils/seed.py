"""Reproducibility: one call seeds every RNG a training run touches."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch. `deterministic=True` also forces deterministic cuDNN kernels
    (slower; bit-exact reruns). Default favours speed: reruns match up to GPU kernel noise."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def worker_init_fn(worker_id: int) -> None:
    """DataLoader workers get distinct but reproducible seeds (derived from torch's base seed)."""
    import torch

    worker_seed = torch.utils.data.get_worker_info().seed % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
