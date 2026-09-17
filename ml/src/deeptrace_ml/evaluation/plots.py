"""Evaluation figures: ROC, confusion matrix, reliability diagram, robustness curves, image galleries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from deeptrace_ml.evaluation.metrics import reliability_bins  # noqa: E402


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> Path:
    """curves: name -> (labels, probs)"""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (labels, probs) in curves.items():
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate (real called fake)")
    ax.set_ylabel("True positive rate (fake caught)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, path)


def plot_confusion(cm: Mapping[str, int], path: Path, title: str) -> Path:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["pred REAL", "pred AI"])
    ax.set_yticks([0, 1], ["true REAL", "true AI"])
    ax.set_title(title, fontsize=9)
    return _save(fig, path)


def plot_reliability(
    labels: np.ndarray, probs_before: np.ndarray, probs_after: np.ndarray, path: Path, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, probs in (("before temperature", probs_before), ("after temperature", probs_after)):
        bins = [b for b in reliability_bins(labels, probs) if b["count"] > 0]
        ax.plot([b["mean_prob"] for b in bins], [b["frac_positive"] for b in bins], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax.set_xlabel("predicted P(AI-generated)")
    ax.set_ylabel("observed fraction AI-generated")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_robustness(
    results: Mapping[str, Sequence[Mapping[str, Any]]], family: str, metric: str, path: Path, title: str
) -> Path:
    """results: model/test-set name -> rows with keys family, level, <metric>."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in results.items():
        pts = sorted((r["level"], r[metric]) for r in rows if r["family"] == family and r.get(metric) is not None)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=name)
    if family in ("jpeg_quality", "downscale"):
        ax.invert_xaxis()  # left = clean, right = more degraded
    ax.set_xlabel(family)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=7)
    return _save(fig, path)


def plot_gallery(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    title: str,
    columns: Sequence[str] = ("image", "overlay"),
) -> Path:
    """rows: dicts with numpy/PIL images under `columns` and a `caption`."""
    n = max(1, len(rows))
    fig, axes = plt.subplots(n, len(columns), figsize=(2.3 * len(columns), 2.4 * n), squeeze=False)
    for r, row in enumerate(rows):
        for c, col in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(row[col])
            ax.axis("off")
            if r == 0:
                ax.set_title(col, fontsize=8)
        axes[r, 0].axis("on")  # keep the label visible; hide ticks/frame only
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
        axes[r, 0].set_ylabel(row.get("caption", ""), fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)
