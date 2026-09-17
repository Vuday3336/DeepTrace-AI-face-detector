"""CLIP linear probe: frozen features + logistic regression, folded into an nn.Linear head.

Why a logistic regression instead of SGD on a linear layer: it is convex, so it finds THE optimum for
each regularisation strength C in seconds, and C is chosen on val_select. The fitted
StandardScaler + LogisticRegression is folded into one linear layer on raw features, so the saved
model is an ordinary BinaryClassifier (same eval/export path as every other model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from deeptrace_ml.models.classifier import BinaryClassifier
from deeptrace_ml.utils.logging import get_logger

log = get_logger("probe")


@dataclass
class ProbeResult:
    best_c: float
    val_auc_by_c: dict[float, float]


@torch.no_grad()
def extract_features(
    model: BinaryClassifier, loader: DataLoader, device: torch.device, amp: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    use_amp = amp and device.type == "cuda"
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pooled = model.pooled(model.features(x.to(device, non_blocking=True)))
        feats.append(pooled.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels).astype(int)


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    c_grid: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, float, ProbeResult]:
    """Return (weights[D], bias) on RAW features for the C with the best validation AUC."""
    scaler = StandardScaler().fit(train_x)
    tx, vx = scaler.transform(train_x), scaler.transform(val_x)
    scores: dict[float, float] = {}
    best: tuple[float, LogisticRegression] | None = None
    for c in c_grid:
        clf = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(tx, train_y)
        auc = float(roc_auc_score(val_y, clf.decision_function(vx)))
        scores[c] = auc
        log.info("C=%g val_auc=%.4f", c, auc)
        if best is None or auc > scores[best[0]]:
            best = (c, clf)
    assert best is not None
    c, clf = best
    # logit = w . (x - mu) / sigma + b  ==  (w / sigma) . x + (b - sum(w * mu / sigma))
    sigma = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    weights = clf.coef_[0] / sigma
    bias = float(clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / sigma))
    return weights.astype(np.float32), bias, ProbeResult(best_c=c, val_auc_by_c=scores)


def install_head(model: BinaryClassifier, weights: np.ndarray, bias: float) -> None:
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(weights).reshape(1, -1))
        model.head.bias.fill_(bias)
