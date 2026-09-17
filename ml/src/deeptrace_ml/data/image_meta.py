"""File-level image metadata: the cheap signals a model could cheat with.

Format is detected from magic bytes, never from the extension (extensions lie, e.g. PNGs renamed
to .jpg). JPEG quality is *estimated* from the luminance quantization table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = ("jpeg", "png", "webp")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

# Standard IJG (libjpeg) luminance quantization table at quality 50.
_IJG_LUMA = np.array(
    [
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ],
    dtype=np.int64,
)


def _ijg_table(quality: int) -> np.ndarray:
    """Replicates libjpeg's jpeg_quality_scaling with integer arithmetic."""
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((_IJG_LUMA * scale + 50) // 100, 1, 255)


# Mean table value per quality 1..100. The mean is independent of zigzag vs natural ordering,
# which differs between Pillow versions, so we never depend on coefficient order.
_QUALITY_MEANS = np.array([_ijg_table(q).mean() for q in range(1, 101)])


class ImageReadError(RuntimeError):
    """The file could not be opened or fully decoded."""


@dataclass(frozen=True)
class ImageMeta:
    format: str
    width: int
    height: int
    mode: str
    file_size: int
    jpeg_quality_est: int | None
    has_exif: bool
    has_icc: bool
    has_xmp: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_format(header: bytes) -> str:
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if header[:2] == b"BM":
        return "bmp"
    return "unknown"


def estimate_jpeg_quality(quant_tables: Mapping[int, Sequence[int]] | None) -> int | None:
    """Closest IJG quality (1-100) for the luminance table, or None if there is no table.

    Exact for libjpeg/Pillow encoders; approximate for cameras and editors that use custom tables.
    """
    if not quant_tables or 0 not in quant_tables:
        return None
    luma_mean = float(np.mean(np.asarray(quant_tables[0], dtype=np.float64)))
    return int(np.argmin(np.abs(_QUALITY_MEANS - luma_mean))) + 1


def _has_xmp(info: Mapping[str, Any]) -> bool:
    return any("xmp" in str(key).lower() for key in info)


def read_image_meta(path: str | Path) -> ImageMeta:
    """Read metadata AND fully decode the image, so truncated files are caught here, not in training."""
    p = Path(path)
    try:
        with p.open("rb") as fh:
            header = fh.read(16)
        file_size = p.stat().st_size
        with Image.open(p) as im:
            fmt = detect_format(header)
            quality = estimate_jpeg_quality(getattr(im, "quantization", None)) if fmt == "jpeg" else None
            meta = ImageMeta(
                format=fmt,
                width=im.width,
                height=im.height,
                mode=im.mode,
                file_size=file_size,
                jpeg_quality_est=quality,
                has_exif=len(im.getexif()) > 0,
                has_icc=bool(im.info.get("icc_profile")),
                has_xmp=_has_xmp(im.info),
            )
            im.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, SyntaxError, ValueError) as exc:
        raise ImageReadError(f"{p}: {type(exc).__name__}: {exc}") from exc
    return meta
