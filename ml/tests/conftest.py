from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deeptrace_ml.config import load_preprocess_config

ML_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def preprocess_cfg():
    return load_preprocess_config(ML_DIR / "configs/preprocess.yaml")


def make_image(width: int = 128, height: int = 96, seed: int = 0, mode: str = "RGB") -> Image.Image:
    """Textured random image (flat images compress/hash degenerately)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height // 8 + 1, width // 8 + 1, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((width, height), Image.Resampling.BICUBIC)
    noise = rng.integers(-20, 21, size=(height, width, 3))
    arr = np.clip(np.asarray(img, dtype=np.int32) + noise, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    return out.convert(mode) if mode != "RGB" else out
