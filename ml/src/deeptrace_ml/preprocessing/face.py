"""Face detection with MTCNN (facenet-pytorch).

Why MTCNN: it is light enough for CPU serving, returns confidence scores, and the SAME detector is
used for training data and uploads, so crops are framed identically in both places.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from PIL import Image

from deeptrace_ml.config import FaceConfig
from deeptrace_ml.preprocessing.crop import Box


@dataclass(frozen=True)
class DetectedFace:
    box: Box  # in ORIGINAL image pixels
    prob: float


class FaceDetector(Protocol):
    def detect(self, images: Sequence[Image.Image]) -> list[list[DetectedFace]]: ...


def downscale_for_detection(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    scale = min(1.0, max_side / max(img.width, img.height))
    if scale >= 1.0:
        return img, 1.0
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.Resampling.BILINEAR), scale


def filter_and_rank(
    boxes: np.ndarray | None, probs: np.ndarray | None, scale: float, cfg: FaceConfig
) -> list[DetectedFace]:
    """Rescale to original pixels, drop small/uncertain faces, rank by prob * area (desc)."""
    if boxes is None or probs is None:
        return []
    faces: list[DetectedFace] = []
    for raw_box, prob in zip(boxes, probs, strict=False):
        if prob is None or float(prob) < cfg.prob_threshold:
            continue
        x1, y1, x2, y2 = (float(v) / scale for v in raw_box)
        box = Box(x1, y1, x2, y2)
        if box.width < cfg.min_face_size or box.height < cfg.min_face_size:
            continue
        faces.append(DetectedFace(box=box, prob=float(prob)))
    faces.sort(key=lambda f: f.prob * f.box.width * f.box.height, reverse=True)
    return faces[: cfg.max_faces]


class MTCNNFaceDetector:
    def __init__(self, cfg: FaceConfig, device: str = "cpu") -> None:
        try:
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise ImportError(
                "facenet-pytorch is missing. Install with: " "pip install -r requirements/facenet.txt --no-deps"
            ) from exc
        self._cfg = cfg
        self._mtcnn: Any = MTCNN(
            keep_all=True,
            min_face_size=cfg.mtcnn_min_face_size,
            thresholds=list(cfg.mtcnn_thresholds),
            post_process=False,
            device=device,
        )

    def detect(self, images: Sequence[Image.Image]) -> list[list[DetectedFace]]:
        """Detect faces; same-size images are batched together (fast for the 256x256 140k set)."""
        prepared = [downscale_for_detection(im.convert("RGB"), self._cfg.detect_max_side) for im in images]
        results: list[list[DetectedFace]] = [[] for _ in images]
        by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (small, _) in enumerate(prepared):
            by_size[small.size].append(idx)

        for indices in by_size.values():
            batch = [prepared[i][0] for i in indices]
            boxes, probs = self._mtcnn.detect(batch)
            for k, i in enumerate(indices):
                results[i] = filter_and_rank(boxes[k], probs[k], prepared[i][1], self._cfg)
        return results
