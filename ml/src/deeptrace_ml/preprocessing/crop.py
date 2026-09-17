"""Square face crops that never leave the image (no padding)."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class SquareCrop:
    left: int
    top: int
    side: int
    clamped: bool  # True if the ideal square had to be shifted or shrunk to fit

    @property
    def pil_box(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.side, self.top + self.side)


def square_crop_box(box: Box, image_width: int, image_height: int, margin_ratio: float) -> SquareCrop:
    if box.width <= 0 or box.height <= 0:
        raise ValueError(f"Degenerate face box: {box}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    ideal_side = max(box.width, box.height) * (1.0 + margin_ratio)
    side = max(1, round(ideal_side))
    clamped = False
    max_side = min(image_width, image_height)
    if side > max_side:
        side, clamped = max_side, True

    cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
    ideal_left, ideal_top = round(cx - side / 2.0), round(cy - side / 2.0)
    left = min(max(ideal_left, 0), image_width - side)
    top = min(max(ideal_top, 0), image_height - side)
    clamped = clamped or left != ideal_left or top != ideal_top
    return SquareCrop(left=left, top=top, side=side, clamped=clamped)


def crop_and_resize(img: Image.Image, crop: SquareCrop, output_size: int, resample: str) -> Image.Image:
    if resample not in _RESAMPLE:
        raise ValueError(f"Unknown resample '{resample}'")
    return img.crop(crop.pil_box).resize((output_size, output_size), _RESAMPLE[resample])
