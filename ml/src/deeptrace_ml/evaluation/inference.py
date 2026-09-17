"""Batch inference helpers shared by predict/robustness/explain scripts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deeptrace_ml.models.classifier import BinaryClassifier
from deeptrace_ml.preprocessing.pipeline import normalize_image
from deeptrace_ml.training.data import ProcessedFaceDataset, eval_transform

PREDICTION_COLUMNS = ["dataset", "split", "source", "label", "rel_path", "processed_rel_path", "logit"]


@torch.no_grad()
def predict_frame(
    model: BinaryClassifier,
    frame: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    amp: bool = True,
) -> pd.DataFrame:
    """Logits for every processed crop in `frame` (same order)."""
    model.eval().to(device)
    dataset = ProcessedFaceDataset(frame, data_root, eval_transform(model.mean, model.std))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    use_amp = amp and device.type == "cuda"
    logits: list[np.ndarray] = []
    for x, _ in tqdm(loader, desc="predict"):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    out = frame[[c for c in PREDICTION_COLUMNS if c != "logit"]].copy()
    out["logit"] = np.concatenate(logits)
    return out.reset_index(drop=True)


@torch.no_grad()
def logits_for_images(
    model: BinaryClassifier,
    images: Sequence[Image.Image],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Logits for already-cropped 224px PIL images (numpy normalisation = serving path)."""
    model.eval().to(device)
    out: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch = np.concatenate(
            [normalize_image(im, model.mean, model.std) for im in images[start : start + batch_size]]
        )
        out.append(model(torch.from_numpy(batch).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def load_predictions(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype={"split": str, "dataset": str, "rel_path": str})
        missing = set(PREDICTION_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{p} is not a predictions file (missing {sorted(missing)})")
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["dataset", "rel_path"])


def binary_labels(frame: pd.DataFrame) -> np.ndarray:
    return (frame["label"] == "fake").to_numpy(dtype=int)
