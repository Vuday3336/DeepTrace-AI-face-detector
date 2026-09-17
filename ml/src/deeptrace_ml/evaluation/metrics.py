"""Classification metrics with bootstrap confidence intervals.

Convention: label 1 = AI-generated (positive), 0 = real. A false positive is a REAL photo called fake.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """ECE for the positive-class probability: weighted mean |accuracy - confidence| per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def reliability_bins(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    return [
        {
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "count": int((idx == b).sum()),
            "mean_prob": float(probs[idx == b].mean()) if (idx == b).any() else None,
            "frac_positive": float(labels[idx == b].mean()) if (idx == b).any() else None,
        }
        for b in range(n_bins)
    ]


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(labels)),
        "balanced_accuracy": float(0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0))),
        "precision": float(precision),
        "recall_tpr": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ranking_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float | None]:
    """Threshold-free metrics; None when a class is missing (AUC undefined)."""
    if len(np.unique(labels)) < 2:
        return {"roc_auc": None, "pr_auc": None, "tpr_at_fpr_5pct": None}
    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "tpr_at_fpr_5pct": tpr_at_fpr(labels, probs, 0.05),
    }


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float | None],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float] | None:
    """Percentile bootstrap, resampling within each class to keep the class balance fixed."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        return None
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        value = metric(labels[idx], probs[idx])
        if value is not None:
            values.append(value)
    if not values:
        return None
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"low": float(lo), "high": float(hi)}


def full_report(
    labels: np.ndarray, probs: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 42
) -> dict[str, Any]:
    labels = labels.astype(int)
    report: dict[str, Any] = {
        "n": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        **ranking_metrics(labels, probs),
        "at_tuned_threshold": threshold_metrics(labels, probs, threshold),
        "at_0_5": threshold_metrics(labels, probs, 0.5),
        "ece_15": expected_calibration_error(labels, probs),
        "brier": float(np.mean((probs - labels) ** 2)),
    }
    both = report["n_real"] > 0 and report["n_fake"] > 0
    report["ci95"] = {
        "roc_auc": bootstrap_ci(labels, probs, lambda y, p: ranking_metrics(y, p)["roc_auc"], n_boot, seed)
        if both
        else None,
        "balanced_accuracy": bootstrap_ci(
            labels, probs, lambda y, p: threshold_metrics(y, p, threshold)["balanced_accuracy"], n_boot, seed
        )
        if both
        else None,
    }
    return report
