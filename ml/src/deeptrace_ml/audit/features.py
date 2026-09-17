"""Hand-crafted features for the shortcut audit.

Two groups, interpreted differently:
  * METADATA features (size, format, JPEG quality, EXIF...) say nothing about whether a face is
    synthetic. If they predict the label, that is a pure SHORTCUT and must be removed.
  * PIXEL statistics (colour, sharpness, high-frequency energy) can be partly legitimate signal
    (generators do leave texture traces), but a high score still warns that a model could win with
    "sharper = fake" style rules that break on real-world images.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

METADATA_FEATURES = [
    "width",
    "height",
    "aspect",
    "megapixels",
    "file_size_kb",
    "bytes_per_pixel",
    "jpeg_quality_est",
    "jpeg_quality_missing",
    "is_jpeg",
    "is_png",
    "is_webp",
    "has_exif",
    "has_icc",
    "has_xmp",
]
PIXEL_FEATURES = [
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_saturation",
    "mean_brightness",
    "laplacian_var",
    "hf_energy_ratio",
]


def metadata_feature_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest
    out = pd.DataFrame(index=df.index)
    width = df["width"].astype(float)
    height = df["height"].astype(float)
    out["width"] = width
    out["height"] = height
    out["aspect"] = width / height
    out["megapixels"] = width * height / 1e6
    out["file_size_kb"] = df["file_size"].astype(float) / 1024
    out["bytes_per_pixel"] = df["file_size"].astype(float) / (width * height)
    quality = pd.to_numeric(df["jpeg_quality_est"], errors="coerce")
    out["jpeg_quality_missing"] = quality.isna().astype(int)
    out["jpeg_quality_est"] = quality.fillna(-1)
    for fmt in ("jpeg", "png", "webp"):
        out[f"is_{fmt}"] = (df["format"] == fmt).astype(int)
    for flag in ("has_exif", "has_icc", "has_xmp"):
        out[flag] = df[flag].astype(str).str.lower().isin(["true", "1"]).astype(int)
    return out[METADATA_FEATURES]


def pixel_features(img: Image.Image) -> dict[str, float]:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(img.convert("L"), dtype=np.float64)

    laplacian = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4.0 * gray[1:-1, 1:-1]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt(((yy - h // 2) / (h / 2)) ** 2 + ((xx - w // 2) / (w / 2)) ** 2)
    total = spectrum.sum()

    return {
        "mean_r": float(rgb[..., 0].mean()),
        "mean_g": float(rgb[..., 1].mean()),
        "mean_b": float(rgb[..., 2].mean()),
        "std_r": float(rgb[..., 0].std()),
        "std_g": float(rgb[..., 1].std()),
        "std_b": float(rgb[..., 2].std()),
        "mean_saturation": float(hsv[..., 1].mean()),
        "mean_brightness": float(hsv[..., 2].mean()),
        "laplacian_var": float(laplacian.var()),
        "hf_energy_ratio": float(spectrum[radius > 0.5].sum() / total) if total > 0 else 0.0,
    }


def _pixel_features_path(path: str) -> dict[str, float] | None:
    try:
        with Image.open(path) as im:
            return pixel_features(im)
    except (OSError, ValueError):
        return None


def pixel_feature_frame(paths: Sequence[Path], workers: int = 4) -> pd.DataFrame:
    """One row per path; rows for unreadable files are NaN (counted by the caller)."""
    jobs = [str(p) for p in paths]
    if workers <= 1:
        rows = [_pixel_features_path(j) for j in tqdm(jobs, desc="pixel features")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(
                tqdm(pool.map(_pixel_features_path, jobs, chunksize=64), total=len(jobs), desc="pixel features")
            )
    empty = dict.fromkeys(PIXEL_FEATURES, np.nan)
    return pd.DataFrame([r if r is not None else empty for r in rows], columns=PIXEL_FEATURES)
