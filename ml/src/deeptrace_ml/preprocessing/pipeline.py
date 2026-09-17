"""Canonical crop -> resize -> JPEG steps (docs/ARCHITECTURE.md §4.1).

Offline (dataset prep) and online (API) differ ONLY in the JPEG quality rule:
  offline: quality ~ U[min, max], derived from the source file's sha256 (deterministic)
  online:  fixed `inference_quality`
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from deeptrace_ml.config import PreprocessConfig
from deeptrace_ml.preprocessing.crop import SquareCrop, crop_and_resize, square_crop_box
from deeptrace_ml.preprocessing.face import DetectedFace
from deeptrace_ml.preprocessing.io import encode_jpeg, jpeg_roundtrip
from deeptrace_ml.utils.hashing import stable_uniform_int


@dataclass(frozen=True)
class CroppedFace:
    face: DetectedFace
    crop: SquareCrop
    image: Image.Image  # output_size x output_size RGB, before JPEG


def crop_faces(image: Image.Image, faces: Sequence[DetectedFace], cfg: PreprocessConfig) -> list[CroppedFace]:
    out: list[CroppedFace] = []
    for face in faces:
        crop = square_crop_box(face.box, image.width, image.height, cfg.crop.margin_ratio)
        resized = crop_and_resize(image, crop, cfg.crop.output_size, cfg.crop.resample)
        out.append(CroppedFace(face=face, crop=crop, image=resized))
    return out


def offline_jpeg_quality(source_sha256: str, cfg: PreprocessConfig) -> int:
    return stable_uniform_int(source_sha256, cfg.seed, cfg.encode.offline_quality_min, cfg.encode.offline_quality_max)


def encode_offline(face_image: Image.Image, source_sha256: str, cfg: PreprocessConfig) -> tuple[bytes, int]:
    quality = offline_jpeg_quality(source_sha256, cfg)
    return encode_jpeg(face_image, quality, cfg.encode.subsampling), quality


def to_inference_image(face_image: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    return jpeg_roundtrip(face_image, cfg.encode.inference_quality, cfg.encode.subsampling)


def normalize_image(img: Image.Image, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """RGB image -> float32 array (1, 3, H, W), normalised like training's albumentations Normalize."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)
