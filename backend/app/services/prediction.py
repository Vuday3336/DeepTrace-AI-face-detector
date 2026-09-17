"""Upload -> validated image -> per-face analysis -> API response (CPU work runs in a threadpool)."""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.core.errors import AppError
from app.schemas import FaceOut, ImageInfo, ModelSummary
from deeptrace_ml.preprocessing.io import ImageValidationError, load_image_bytes
from deeptrace_ml.serving.runtime import AnalysisResult, Analyzer

_STATUS_BY_CODE = {"FILE_TOO_LARGE": 413, "UNSUPPORTED_MEDIA_TYPE": 415, "INVALID_IMAGE": 422}


@dataclass
class PredictionOutcome:
    image: Image.Image  # metadata-stripped RGB (what may be stored if the user opts in)
    faces: list[FaceOut]
    image_info: ImageInfo
    timings_ms: dict[str, float]


def png_base64(image: Image.Image | np.ndarray) -> str:
    pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def model_summary(analyzer: Analyzer) -> ModelSummary:
    bundle = analyzer.bundle
    return ModelSummary(
        name=bundle.name,
        version=bundle.version,
        threshold=bundle.threshold,
        uncertainty_delta=bundle.uncertainty_delta,
        explanation_method=bundle.explanation["method"],
    )


def _decode(data: bytes, analyzer: Analyzer, max_bytes: int) -> Image.Image:
    io_cfg = analyzer.bundle.preprocess.io
    if len(data) > min(max_bytes, io_cfg.max_file_bytes):
        raise AppError(413, "FILE_TOO_LARGE", f"Upload exceeds {min(max_bytes, io_cfg.max_file_bytes)} bytes")
    try:
        return load_image_bytes(data, io_cfg)
    except ImageValidationError as exc:
        raise AppError(_STATUS_BY_CODE.get(exc.code, 422), exc.code, exc.message) from exc


def _to_faces(result: AnalysisResult) -> list[FaceOut]:
    return [
        FaceOut(
            face_index=f.index,
            bbox=[round(v) for v in f.box],
            detection_score=round(f.detection_score, 4),
            label=f.label,  # type: ignore[arg-type]
            prob_ai_generated=round(f.prob_ai_generated, 4),
            heatmap_png_base64=png_base64(f.heatmap_rgba) if f.heatmap_rgba is not None else None,
            face_crop_png_base64=png_base64(f.face_crop),
        )
        for f in result.faces
    ]


def _run(data: bytes, analyzer: Analyzer, max_bytes: int, explain: bool) -> PredictionOutcome:
    start = time.perf_counter()
    image = _decode(data, analyzer, max_bytes)
    decode_ms = (time.perf_counter() - start) * 1000
    result = analyzer.analyze(image, explain=explain)
    if not result.faces:
        raise AppError(
            422, "NO_FACE_DETECTED", "No face was found. Use a clear, reasonably large, front-facing face photo."
        )
    start = time.perf_counter()
    faces = _to_faces(result)
    timings = {"decode": decode_ms, **result.timings_ms, "encode": (time.perf_counter() - start) * 1000}
    return PredictionOutcome(
        image=image,
        faces=faces,
        image_info=ImageInfo(width=result.width, height=result.height),
        timings_ms={k: round(v, 1) for k, v in timings.items()},
    )


async def run_prediction(data: bytes, analyzer: Analyzer, max_bytes: int, explain: bool) -> PredictionOutcome:
    return await run_in_threadpool(_run, data, analyzer, max_bytes, explain)
