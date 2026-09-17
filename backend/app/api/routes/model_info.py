from __future__ import annotations

import copy
from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import require_analyzer
from deeptrace_ml.serving.runtime import Analyzer

router = APIRouter(tags=["model"])


def public_card(card: dict[str, Any], preprocess: dict[str, Any]) -> dict[str, Any]:
    """The model card without internal heatmap parameters (thousands of head weights)."""
    out = copy.deepcopy(card)
    explanation = out.get("explanation", {})
    explanation.pop("head", None)
    out["preprocessing"] = {
        "face_detector": "MTCNN",
        "crop_margin_ratio": preprocess["crop"]["margin_ratio"],
        "input_size": preprocess["crop"]["output_size"],
        "inference_jpeg_quality": preprocess["encode"]["inference_quality"],
        "max_upload_bytes": preprocess["io"]["max_file_bytes"],
        "max_faces": preprocess["face"]["max_faces"],
    }
    return out


@router.get("/model-info", summary="Training data, metrics, decision settings and known limitations")
async def model_info(analyzer: Analyzer = Depends(require_analyzer)) -> dict[str, Any]:
    return public_card(analyzer.bundle.card, analyzer.bundle.preprocess.to_dict())
