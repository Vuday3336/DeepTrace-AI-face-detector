"""Real-world degradations for robustness curves (docs/ARCHITECTURE.md §6.3).

Perturbations are applied to the ORIGINAL uploaded-size image before face cropping, because that is
where real degradation happens (a platform recompresses the whole photo, not your 224px crop).
Each function returns (image, scale) where `scale` maps original coordinates to the new image, so the
known face box can be reused.
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

Perturbation = Callable[[Image.Image], tuple[Image.Image, float]]


def jpeg(quality: int) -> Perturbation:
    def apply(img: Image.Image) -> tuple[Image.Image, float]:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as out:
            return out.convert("RGB"), 1.0

    return apply


def downscale(factor: float) -> Perturbation:
    """Shrink then return at the SMALLER size: detail is lost, as with a thumbnail re-upload."""

    def apply(img: Image.Image) -> tuple[Image.Image, float]:
        if factor >= 1.0:
            return img, 1.0
        size = (max(16, round(img.width * factor)), max(16, round(img.height * factor)))
        return img.resize(size, Image.Resampling.BILINEAR), size[0] / img.width

    return apply


def blur(sigma: float) -> Perturbation:
    def apply(img: Image.Image) -> tuple[Image.Image, float]:
        if sigma <= 0:
            return img, 1.0
        return img.filter(ImageFilter.GaussianBlur(radius=sigma)), 1.0

    return apply


def screenshot_chain(seed: int) -> Perturbation:
    """Screenshot / re-upload simulation: display rescale -> PNG -> gamma shift -> resize -> JPEG."""

    def apply(img: Image.Image) -> tuple[Image.Image, float]:
        rng = random.Random(seed)
        scale = rng.uniform(0.5, 1.5)
        size = (max(16, round(img.width * scale)), max(16, round(img.height * scale)))
        out = img.resize(size, Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        with Image.open(io.BytesIO(buf.getvalue())) as png:
            out = png.convert("RGB")
        gamma = rng.uniform(0.85, 1.15)
        arr = np.clip(255.0 * (np.asarray(out, dtype=np.float32) / 255.0) ** gamma, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        rescale = rng.uniform(0.7, 1.0)
        final = (max(16, round(out.width * rescale)), max(16, round(out.height * rescale)))
        out = out.resize(final, Image.Resampling.BILINEAR)
        out, _ = jpeg(rng.randint(70, 90))(out)
        return out, final[0] / img.width

    return apply


@dataclass(frozen=True)
class PerturbationSpec:
    family: str
    level: float | int
    fn: Perturbation


def build_grid(grid: dict[str, list[float]], screenshot_seeds: list[int]) -> list[PerturbationSpec]:
    specs: list[PerturbationSpec] = [PerturbationSpec("none", 0, lambda im: (im, 1.0))]
    builders: dict[str, Callable[[float], Perturbation]] = {
        "jpeg_quality": lambda v: jpeg(int(v)),
        "downscale": downscale,
        "blur_sigma": blur,
    }
    for family, levels in grid.items():
        if family not in builders:
            raise ValueError(f"Unknown perturbation family '{family}'")
        specs.extend(PerturbationSpec(family, lvl, builders[family](lvl)) for lvl in levels)
    specs.extend(PerturbationSpec("screenshot", s, screenshot_chain(s)) for s in screenshot_seeds)
    return specs
