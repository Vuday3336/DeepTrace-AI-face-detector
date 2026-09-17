from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from app.api.deps import AppState, enforce_rate_limit, get_state, optional_user, require_analyzer
from app.core.errors import AppError
from app.db.models import User
from app.schemas import ErrorResponse, PredictResponse
from app.services.history import save_prediction
from app.services.prediction import model_summary, run_prediction
from deeptrace_ml.serving.runtime import Analyzer

log = logging.getLogger("deeptrace.predict")
router = APIRouter(tags=["detection"])

_ERRORS = {code: {"model": ErrorResponse} for code in (401, 413, 415, 422, 429, 503)}


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses=_ERRORS,
    summary="Detect faces and estimate whether each is AI-generated",
    dependencies=[Depends(enforce_rate_limit)],
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="JPEG, PNG or WebP, max 10 MB"),
    store: bool = Query(False, description="Save to your history (requires login). Default: nothing is stored."),
    explain: bool = Query(True, description="Include heatmaps"),
    analyzer: Analyzer = Depends(require_analyzer),
    user: User | None = Depends(optional_user),
    state: AppState = Depends(get_state),
) -> PredictResponse:
    if store and user is None:
        raise AppError(401, "UNAUTHORIZED", "Log in to save results to history (or send store=false)")

    limit = state.settings.max_upload_bytes
    data = await file.read(limit + 1)  # read at most limit+1 bytes: never buffer a huge upload
    await file.close()
    if not data:
        raise AppError(422, "INVALID_IMAGE", "Empty upload")

    outcome = await run_prediction(data, analyzer, limit, explain)
    del data  # uploaded bytes are not kept anywhere past this point

    history_id = None
    if store and user is not None and state.db is not None:
        async with state.db.sessionmaker() as session:
            row = await save_prediction(
                session,
                user,
                analyzer.bundle.name,
                analyzer.bundle.version,
                outcome.faces,
                outcome.image,
                state.settings.upload_dir,
                sum(outcome.timings_ms.values()),
            )
            history_id = row.id

    log.info(
        "predict faces=%d labels=%s stored=%s ms=%s",
        len(outcome.faces),
        [f.label for f in outcome.faces],
        history_id is not None,
        outcome.timings_ms,
    )
    return PredictResponse(
        request_id=request.state.request_id,
        model=model_summary(analyzer),
        image=outcome.image_info,
        faces=outcome.faces,
        stored=history_id is not None,
        history_id=history_id,
        timings_ms=outcome.timings_ms,
    )
