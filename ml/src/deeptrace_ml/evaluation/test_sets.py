"""Test-set definitions from docs/ARCHITECTURE.md §6.2, built from one combined predictions table.

Splitting the cross-generator evaluation into "swap only fakes" and "swap only reals" separates the
two changes in the full cross-generator set (new generator AND new real-photo source), so a drop can
be attributed to the right cause.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

ID_DATASET = "140k"
ID_REAL_SOURCE = "ffhq"
ID_FAKE_SOURCE = "stylegan"


def iter_test_sets(pred: pd.DataFrame) -> Iterator[tuple[str, str, pd.DataFrame]]:
    """Yield (name, description, rows). Sets without both classes present are skipped."""
    id_test = pred[(pred["dataset"] == ID_DATASET) & (pred["split"] == "test")]
    id_real = id_test[id_test["label"] == "real"]
    id_fake = id_test[id_test["label"] == "fake"]
    cross = pred[pred["dataset"] == "crossgen"]
    own = pred[pred["dataset"] == "own"]

    candidates: list[tuple[str, str, pd.DataFrame]] = [
        ("id_test", "In-distribution: FFHQ real vs StyleGAN fake (140k test split)", id_test),
        ("crossgen_full", "Cross-generator: all crossgen reals vs all crossgen fakes", cross),
    ]
    for source in sorted(cross.loc[cross["label"] == "fake", "source"].unique()):
        candidates.append(
            (
                f"fake_swap_{source}",
                f"Generator shift only: FFHQ real (140k test) vs {source} fake",
                pd.concat([id_real, cross[(cross["label"] == "fake") & (cross["source"] == source)]]),
            )
        )
    for source in sorted(cross.loc[cross["label"] == "real", "source"].unique()):
        candidates.append(
            (
                f"real_swap_{source}",
                f"Real-source shift only: {source} real vs StyleGAN fake (140k test)",
                pd.concat([cross[(cross["label"] == "real") & (cross["source"] == source)], id_fake]),
            )
        )
    candidates.append(("own", "Own sanity set: phone photos vs self-generated fakes", own))

    for name, description, rows in candidates:
        if rows["label"].nunique() == 2:
            yield name, description, rows.reset_index(drop=True)


def per_source_rates(rows: pd.DataFrame, prob_col: str, threshold: float) -> dict[str, dict[str, float]]:
    """For each source: share flagged as fake (= FPR for real sources, TPR for fake sources)."""
    out: dict[str, dict[str, float]] = {}
    for (source, label), g in rows.groupby(["source", "label"]):
        flagged = float((g[prob_col] >= threshold).mean())
        key = "fpr" if label == "real" else "tpr"
        out[str(source)] = {"label": str(label), "n": int(len(g)), key: flagged}
    return out
