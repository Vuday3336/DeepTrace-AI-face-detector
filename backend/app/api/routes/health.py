from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import AppState, get_state
from app.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and model/database status")
async def health(state: AppState = Depends(get_state)) -> HealthResponse:
    db_status = "disabled"
    if state.db is not None:
        db_status = "ok" if await state.db.ping() else "error"
    analyzer = state.analyzer
    return HealthResponse(
        status="ok" if analyzer is not None and db_status != "error" else "degraded",
        model_loaded=analyzer is not None,
        model_name=analyzer.bundle.name if analyzer else None,
        model_version=analyzer.bundle.version if analyzer else None,
        model_error=state.model_error,
        history="enabled" if state.settings.history_enabled else "disabled",
        db=db_status,  # type: ignore[arg-type]
    )
