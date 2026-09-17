"""Audit plots. Each function saves a PNG and returns the figure (so notebooks can display it)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks still display returned figures
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402


def _save(fig: Figure, path: Path) -> Figure:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fig


def _grid(n: int, ncols: int = 3, cell: tuple[float, float] = (4.2, 3.0)) -> tuple[Figure, np.ndarray]:
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def plot_numeric_distributions(
    df: pd.DataFrame, columns: Sequence[str], group_col: str, path: Path, title: str, bins: int = 40
) -> Figure:
    fig, axes = _grid(len(columns))
    groups = sorted(df[group_col].dropna().unique())
    for ax, col in zip(axes, columns, strict=False):
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            ax.set_title(f"{col} (no data)")
            continue
        lo, hi = finite.quantile(0.005), finite.quantile(0.995)
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
        for group in groups:
            subset = values[(df[group_col] == group) & np.isfinite(values)]
            ax.hist(subset.clip(lo, hi), bins=edges, density=True, alpha=0.45, label=str(group))
        ax.set_title(col)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_rates(df: pd.DataFrame, columns: Sequence[str], group_col: str, path: Path, title: str) -> Figure:
    """Share of images per group where a 0/1 column is 1 (e.g. has_exif, is_png)."""
    rates = df.groupby(group_col)[list(columns)].mean().T
    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 1.2), 3.5))
    rates.plot.bar(ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of images")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def plot_radial_spectra(profiles: Mapping[str, np.ndarray], path: Path, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, profile in profiles.items():
        freqs = np.linspace(0, 1, len(profile))
        ax.plot(freqs, profile, label=name)
    ax.set_xlabel("normalised spatial frequency (1 = Nyquist)")
    ax.set_ylabel("mean log power")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_spectra_2d(spectra: Mapping[str, np.ndarray], path: Path, title: str) -> Figure:
    fig, axes = _grid(len(spectra), ncols=min(4, max(1, len(spectra))), cell=(3.2, 3.2))
    for ax, (name, spec) in zip(axes, spectra.items(), strict=False):
        ax.imshow(spec, cmap="magma")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_image_pairs(
    pairs: Sequence[tuple[Path, Path, str]], path: Path, title: str, loader: Callable[[Path], Image.Image] | None = None
) -> Figure:
    """Side-by-side pairs (e.g. near-duplicates) for eyeballing false positives."""

    def _default_load(p: Path) -> Image.Image:
        with Image.open(p) as im:
            return im.convert("RGB")

    load = loader or _default_load
    n = max(1, len(pairs))
    fig, axes = plt.subplots(n, 2, figsize=(4.4, 2.3 * n), squeeze=False)
    for row, (a, b, caption) in enumerate(pairs):
        for col, img_path in enumerate((a, b)):
            ax = axes[row, col]
            ax.imshow(load(img_path))
            ax.axis("off")
        axes[row, 0].set_title(caption, fontsize=7, loc="left")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)
