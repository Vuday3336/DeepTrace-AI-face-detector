"""Request/response models (they also generate the OpenAPI docs at /docs)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field

Label = Literal["REAL", "AI_GENERATED", "UNCERTAIN"]
DISCLAIMER = (
    "DeepTrace is a detection aid, not proof. Results can be wrong, especially for new generators and "
    "heavily compressed images. Never use a result as the sole basis for an accusation."
)


class ModelSummary(BaseModel):
    name: str
    version: str
    threshold: float
    uncertainty_delta: float
    explanation_method: str


class ImageInfo(BaseModel):
    width: int
    height: int


class FaceOut(BaseModel):
    face_index: int
    bbox: list[int] = Field(description="[x1, y1, x2, y2] in original image pixels")
    detection_score: float
    label: Label
    prob_ai_generated: float = Field(ge=0, le=1, description="Calibrated probability the face is AI-generated")
    heatmap_png_base64: str | None = Field(description="RGBA PNG, same size as the face crop; alpha = importance")
    face_crop_png_base64: str


class PredictResponse(BaseModel):
    request_id: str
    model: ModelSummary
    image: ImageInfo
    faces: list[FaceOut]
    stored: bool
    history_id: str | None = None
    timings_ms: dict[str, float]
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    model_error: str | None = None
    history: Literal["enabled", "disabled"]
    db: Literal["ok", "error", "disabled"]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=64)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in_minutes: int


class UserOut(BaseModel):
    id: str
    email: EmailStr
    created_at: datetime


class HistoryFace(BaseModel):
    face_index: int
    bbox: list[int]
    label: Label
    prob_ai_generated: float


class HistoryItem(BaseModel):
    id: str
    created_at: datetime
    model_name: str
    model_version: str
    num_faces: int
    faces: list[HistoryFace]
    has_image: bool


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


ModelInfo = dict[str, Any]
