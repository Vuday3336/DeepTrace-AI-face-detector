from __future__ import annotations

import io

import numpy as np
import pytest
from conftest import make_image
from PIL import Image

from deeptrace_ml.config import FaceConfig
from deeptrace_ml.preprocessing.crop import Box, crop_and_resize, square_crop_box
from deeptrace_ml.preprocessing.face import DetectedFace, downscale_for_detection, filter_and_rank
from deeptrace_ml.preprocessing.io import ImageValidationError, encode_jpeg, load_image_bytes
from deeptrace_ml.preprocessing.pipeline import crop_faces, encode_offline, offline_jpeg_quality


def _bytes(img: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


# ---------- io ----------


def test_exif_orientation_applied_and_metadata_stripped(preprocess_cfg):
    img = make_image(120, 80)
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW when displayed
    exif[0x010F] = "PhoneMaker"
    data = _bytes(img, "JPEG", quality=95, exif=exif.tobytes())
    loaded = load_image_bytes(data, preprocess_cfg.io)
    assert loaded.size == (80, 120)  # width/height swapped
    assert loaded.mode == "RGB"
    assert loaded.info == {}
    assert len(loaded.getexif()) == 0


def test_transparent_png_composited_on_background(preprocess_cfg):
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    loaded = load_image_bytes(_bytes(rgba, "PNG"), preprocess_cfg.io)
    assert np.asarray(loaded)[0, 0].tolist() == list(preprocess_cfg.io.alpha_background)


def test_unsupported_format_rejected(preprocess_cfg):
    with pytest.raises(ImageValidationError) as err:
        load_image_bytes(_bytes(make_image(), "BMP"), preprocess_cfg.io)
    assert err.value.code == "UNSUPPORTED_MEDIA_TYPE"


def test_too_many_pixels_rejected(preprocess_cfg):
    from dataclasses import replace

    tiny_limit = replace(preprocess_cfg.io, max_pixels=100)
    with pytest.raises(ImageValidationError) as err:
        load_image_bytes(_bytes(make_image(20, 20), "PNG"), tiny_limit)
    assert err.value.code == "INVALID_IMAGE"


def test_file_too_large_rejected(preprocess_cfg):
    from dataclasses import replace

    small = replace(preprocess_cfg.io, max_file_bytes=10)
    with pytest.raises(ImageValidationError) as err:
        load_image_bytes(_bytes(make_image(), "PNG"), small)
    assert err.value.code == "FILE_TOO_LARGE"


def test_corrupt_image_rejected(preprocess_cfg):
    data = _bytes(make_image(200, 200), "PNG")
    with pytest.raises(ImageValidationError) as err:
        load_image_bytes(data[:200], preprocess_cfg.io)
    assert err.value.code == "INVALID_IMAGE"


def test_encode_jpeg_validates_quality():
    with pytest.raises(ValueError):
        encode_jpeg(make_image(), 0, "4:2:0")


# ---------- crop ----------


def test_square_crop_centered_without_clamping():
    crop = square_crop_box(Box(40, 40, 60, 70), 200, 200, margin_ratio=0.3)
    assert crop.side == 39  # max(20, 30) * 1.3
    assert crop.clamped is False
    assert crop.left + crop.side / 2 == pytest.approx(50, abs=1)


def test_square_crop_shifted_at_edge():
    crop = square_crop_box(Box(0, 0, 50, 50), 200, 200, margin_ratio=0.3)
    assert (crop.left, crop.top) == (0, 0)
    assert crop.clamped is True


def test_square_crop_shrunk_when_larger_than_image():
    crop = square_crop_box(Box(10, 5, 90, 95), 100, 100, margin_ratio=0.3)
    assert crop.side == 100
    assert crop.clamped is True
    assert (crop.left, crop.top) == (0, 0)


def test_square_crop_rejects_degenerate_box():
    with pytest.raises(ValueError):
        square_crop_box(Box(10, 10, 10, 20), 100, 100, 0.3)


def test_crop_and_resize_output_size():
    crop = square_crop_box(Box(30, 30, 90, 90), 128, 128, 0.3)
    out = crop_and_resize(make_image(128, 128), crop, 224, "bicubic")
    assert out.size == (224, 224)


# ---------- face filtering (no MTCNN needed) ----------


def _face_cfg(**overrides) -> FaceConfig:
    base = dict(
        mtcnn_min_face_size=20,
        mtcnn_thresholds=(0.6, 0.7, 0.7),
        min_face_size=40,
        prob_threshold=0.9,
        detect_max_side=1024,
        max_faces=10,
    )
    base.update(overrides)
    return FaceConfig(**base)


def test_filter_and_rank_rescales_filters_and_sorts():
    boxes = np.array([[0, 0, 30, 30], [10, 10, 15, 15], [50, 50, 70, 70], [0, 0, 40, 40]], dtype=float)
    probs = np.array([0.99, 0.99, 0.95, 0.5])
    faces = filter_and_rank(boxes, probs, scale=0.5, cfg=_face_cfg())
    # box 1 too small after rescale (10px), box 3 low prob -> 2 faces, larger one first
    assert len(faces) == 2
    assert faces[0].box.width == pytest.approx(60)
    assert faces[1].box.x1 == pytest.approx(100)


def test_filter_and_rank_handles_no_detections():
    assert filter_and_rank(None, None, 1.0, _face_cfg()) == []


def test_downscale_for_detection():
    small, scale = downscale_for_detection(make_image(2048, 1024), 1024)
    assert small.size == (1024, 512)
    assert scale == pytest.approx(0.5)
    same, scale1 = downscale_for_detection(make_image(300, 200), 1024)
    assert same.size == (300, 200) and scale1 == 1.0


# ---------- pipeline ----------


def test_offline_quality_deterministic_and_in_range(preprocess_cfg):
    qualities = {offline_jpeg_quality(f"{i:064x}", preprocess_cfg) for i in range(500)}
    lo, hi = preprocess_cfg.encode.offline_quality_min, preprocess_cfg.encode.offline_quality_max
    assert min(qualities) >= lo and max(qualities) <= hi
    assert len(qualities) > (hi - lo) // 2  # actually spread over the range
    assert offline_jpeg_quality("abc", preprocess_cfg) == offline_jpeg_quality("abc", preprocess_cfg)


def test_crop_faces_and_encode_offline(preprocess_cfg):
    img = make_image(300, 300)
    faces = [DetectedFace(Box(100, 100, 200, 220), 0.99)]
    [cropped] = crop_faces(img, faces, preprocess_cfg)
    assert cropped.image.size == (224, 224)
    data, quality = encode_offline(cropped.image, "f" * 64, preprocess_cfg)
    with Image.open(io.BytesIO(data)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.size == (224, 224)
        assert len(decoded.getexif()) == 0
    assert preprocess_cfg.encode.offline_quality_min <= quality <= preprocess_cfg.encode.offline_quality_max
