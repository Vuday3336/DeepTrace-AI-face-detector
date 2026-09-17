"""Datasets and augmentations over the PROCESSED manifest (224x224 face crops)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no network call on import

import albumentations as A  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from albumentations.pytorch import ToTensorV2  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from deeptrace_ml.training.config import AugmentConfig  # noqa: E402

LABEL_TO_INT = {"real": 0, "fake": 1}


def load_processed_split(
    manifest_path: str | Path,
    splits: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    fraction: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Rows with status == ok, optionally filtered. `fraction` subsamples per label (seeded)."""
    df = pd.read_csv(manifest_path, dtype={"rel_path": str, "sha256": str, "split": str, "dataset": str})
    required = {"status", "label", "split", "dataset", "processed_rel_path"}
    if required - set(df.columns):
        raise ValueError(f"{manifest_path} is not a processed manifest (missing {required - set(df.columns)})")
    df = df[df["status"] == "ok"]
    if splits:
        df = df[df["split"].isin(splits)]
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    if fraction < 1.0:
        df = df.groupby("label", group_keys=False).sample(frac=fraction, random_state=seed)
    if df.empty:
        raise ValueError(f"No processed rows for splits={splits} datasets={datasets} in {manifest_path}")
    return df.reset_index(drop=True)


def train_transform(aug: AugmentConfig, mean: Sequence[float], std: Sequence[float]) -> A.Compose:
    """Degradations that real uploads suffer, so the model can't rely on pristine artifacts."""
    return A.Compose(
        [
            A.ImageCompression(quality_range=(aug.jpeg_quality_min, aug.jpeg_quality_max), p=aug.jpeg_p),
            A.GaussianBlur(sigma_limit=(0.1, aug.blur_sigma_max), blur_limit=0, p=aug.blur_p),
            A.Downscale(scale_range=(aug.downscale_min, 0.99), p=aug.downscale_p),
            A.ColorJitter(
                brightness=aug.color_jitter,
                contrast=aug.color_jitter,
                saturation=aug.color_jitter,
                hue=min(0.05, aug.color_jitter / 5),
                p=aug.color_jitter_p,
            ),
            A.HorizontalFlip(p=aug.hflip_p),
            A.Normalize(mean=tuple(mean), std=tuple(std)),
            ToTensorV2(),
        ]
    )


def eval_transform(mean: Sequence[float], std: Sequence[float]) -> A.Compose:
    return A.Compose([A.Normalize(mean=tuple(mean), std=tuple(std)), ToTensorV2()])


class ProcessedFaceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: str | Path, transform: A.Compose) -> None:
        self.paths = [Path(root) / p for p in frame["processed_rel_path"]]
        self.labels = frame["label"].map(LABEL_TO_INT).to_numpy(dtype=np.float32)
        if np.isnan(self.labels).any():
            raise ValueError("Unknown labels in frame")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[idx]) as im:
            image = np.asarray(im.convert("RGB"))
        tensor = self.transform(image=image)["image"]
        return tensor, torch.tensor(self.labels[idx])
