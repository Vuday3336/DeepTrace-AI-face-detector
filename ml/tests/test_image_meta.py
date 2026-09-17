from __future__ import annotations

import io

import pytest
from conftest import make_image
from PIL import Image

from deeptrace_ml.data.image_meta import ImageReadError, detect_format, estimate_jpeg_quality, read_image_meta


def _encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


@pytest.mark.parametrize("fmt,expected", [("JPEG", "jpeg"), ("PNG", "png"), ("WEBP", "webp"), ("BMP", "bmp")])
def test_detect_format_from_magic_bytes(fmt, expected):
    assert detect_format(_encode(make_image(), fmt)[:16]) == expected


def test_detect_format_unknown():
    assert detect_format(b"not an image at all") == "unknown"


@pytest.mark.parametrize("quality", [30, 50, 65, 75, 85, 90, 95])
def test_jpeg_quality_estimate_matches_pillow_encoder(quality):
    data = _encode(make_image(), "JPEG", quality=quality)
    with Image.open(io.BytesIO(data)) as im:
        estimate = estimate_jpeg_quality(im.quantization)
    assert estimate is not None
    assert abs(estimate - quality) <= 1


def test_quality_none_without_tables():
    assert estimate_jpeg_quality(None) is None
    assert estimate_jpeg_quality({}) is None


def test_read_image_meta_detects_exif(tmp_path):
    img = make_image(64, 48)
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"  # Make
    path = tmp_path / "with_exif.jpg"
    img.save(path, format="JPEG", quality=88, exif=exif.tobytes())
    meta = read_image_meta(path)
    assert (meta.format, meta.width, meta.height) == ("jpeg", 64, 48)
    assert meta.has_exif is True
    assert abs(meta.jpeg_quality_est - 88) <= 1


def test_read_image_meta_png_extension_lies(tmp_path):
    path = tmp_path / "actually_png.jpg"
    make_image().save(path, format="PNG")
    meta = read_image_meta(path)
    assert meta.format == "png"
    assert meta.jpeg_quality_est is None
    assert meta.has_exif is False


def test_truncated_file_raises(tmp_path):
    data = _encode(make_image(256, 256), "PNG")
    path = tmp_path / "truncated.png"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(ImageReadError):
        read_image_meta(path)
