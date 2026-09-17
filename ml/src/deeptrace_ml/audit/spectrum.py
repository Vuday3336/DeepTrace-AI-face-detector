"""Average Fourier spectra per image source.

GAN generators upsample feature maps (transposed convs / nearest upsampling), which tends to leave
periodic high-frequency patterns: bright dots or grid lines in the 2D spectrum and bumps at the tail
of the 1D radial profile. Comparing sources before/after preprocessing shows how much of that
signal (and of JPEG's 8x8 block grid) survives.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


@dataclass(frozen=True)
class SpectrumResult:
    mean_log_power_2d: np.ndarray  # (size, size), centred
    radial_profile: np.ndarray  # (size // 2,)
    n_used: int
    n_skipped: int  # images smaller than `size` or unreadable


def centered_log_power(img: Image.Image, size: int) -> np.ndarray | None:
    """Log power spectrum of the central size x size grayscale patch (no resizing: resizing
    would itself reshape the spectrum). Returns None if the image is smaller than `size`."""
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    h, w = gray.shape
    if h < size or w < size:
        return None
    top, left = (h - size) // 2, (w - size) // 2
    patch = gray[top : top + size, left : left + size]
    patch = patch - patch.mean()
    window = np.outer(np.hanning(size), np.hanning(size))  # suppresses edge-discontinuity leakage
    power = np.abs(np.fft.fftshift(np.fft.fft2(patch * window))) ** 2
    return np.log(power + 1e-8)


def radial_profile(spectrum_2d: np.ndarray) -> np.ndarray:
    h, w = spectrum_2d.shape
    yy, xx = np.indices((h, w))
    radius = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(np.int64)
    sums = np.bincount(radius.ravel(), weights=spectrum_2d.ravel())
    counts = np.bincount(radius.ravel())
    profile = sums / np.maximum(counts, 1)
    return profile[: min(h, w) // 2]


def _spectrum_for_path(path: str, size: int) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            return centered_log_power(im, size)
    except (OSError, ValueError):
        return None


def mean_spectrum(paths: Sequence[Path], size: int = 224, workers: int = 4) -> SpectrumResult:
    if not paths:
        raise ValueError("mean_spectrum needs at least one path")
    fn = partial(_spectrum_for_path, size=size)
    jobs = [str(p) for p in paths]
    total = np.zeros((size, size), dtype=np.float64)
    used = 0
    if workers <= 1:
        iterator = map(fn, jobs)
        results = tqdm(iterator, total=len(jobs), desc="spectra")
        for spec in results:
            if spec is not None:
                total += spec
                used += 1
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for spec in tqdm(pool.map(fn, jobs, chunksize=32), total=len(jobs), desc="spectra"):
                if spec is not None:
                    total += spec
                    used += 1
    if used == 0:
        raise ValueError(f"No image was >= {size}px and readable; cannot compute a spectrum")
    mean = total / used
    return SpectrumResult(mean, radial_profile(mean), used, len(jobs) - used)
