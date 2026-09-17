"""Safe image decoding and JPEG re-encoding shared by data prep and the backend.

Every image leaves `load_image_bytes` as a plain RGB array-backed PIL image with:
  * EXIF orientation applied (phone photos are often stored sideways + a rotate flag),
  * transparency flattened onto a fixed background,
  * ALL metadata gone (EXIF, ICC, XMP) — the model can never see it.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from deeptrace_ml.config import IOConfig
from deeptrace_ml.data.image_meta import SUPPORTED_FORMATS, detect_format

_SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}


class ImageValidationError(Exception):
    """Input rejected. `code` matches the API error codes in docs/ARCHITECTURE.md §9.2."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _flatten_to_rgb(img: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*background, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    return img.convert("RGB")


def strip_metadata(img: Image.Image) -> Image.Image:
    """Rebuild the image from raw pixels so no `info`/EXIF/ICC survives."""
    return Image.fromarray(np.array(img.convert("RGB"), dtype=np.uint8))


def load_image_bytes(data: bytes, cfg: IOConfig) -> Image.Image:
    if len(data) > cfg.max_file_bytes:
        raise ImageValidationError("FILE_TOO_LARGE", f"{len(data)} bytes > {cfg.max_file_bytes}")
    fmt = detect_format(data[:16])
    if fmt not in SUPPORTED_FORMATS:
        raise ImageValidationError("UNSUPPORTED_MEDIA_TYPE", f"detected format '{fmt}'")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > cfg.max_pixels:
                raise ImageValidationError("INVALID_IMAGE", f"{im.width}x{im.height} exceeds {cfg.max_pixels} pixels")
            im.load()
            oriented = ImageOps.exif_transpose(im)
            rgb = _flatten_to_rgb(oriented, cfg.alpha_background)
    except ImageValidationError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("INVALID_IMAGE", f"{type(exc).__name__}: {exc}") from exc
    return strip_metadata(rgb)


def load_image_path(path: str | Path, cfg: IOConfig) -> Image.Image:
    return load_image_bytes(Path(path).read_bytes(), cfg)


def encode_jpeg(img: Image.Image, quality: int, subsampling: str) -> bytes:
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")
    if subsampling not in _SUBSAMPLING:
        raise ValueError(f"Unknown subsampling '{subsampling}'")
    buf = io.BytesIO()
    img.convert("RGB").save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling=_SUBSAMPLING[subsampling],
        optimize=False,
        progressive=False,
    )
    return buf.getvalue()


def jpeg_roundtrip(img: Image.Image, quality: int, subsampling: str) -> Image.Image:
    with Image.open(io.BytesIO(encode_jpeg(img, quality, subsampling))) as decoded:
        return decoded.convert("RGB")
