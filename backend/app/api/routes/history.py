from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AppState, current_user, get_session, get_state
from app.core.errors import AppError
from app.db.models import User
from app.schemas import HistoryItem, HistoryPage
from app.services.history import delete_prediction, get_owned, list_predictions, to_item

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryPage, summary="Your saved analyses (newest first)")
async def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> HistoryPage:
    rows, total = await list_predictions(session, user, limit, offset)
    return HistoryPage(items=[to_item(r) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/{prediction_id}", response_model=HistoryItem)
async def get_history_item(
    prediction_id: str, user: User = Depends(current_user), session: AsyncSession = Depends(get_session)
) -> HistoryItem:
    row = await get_owned(session, user, prediction_id)
    if row is None:
        raise AppError(404, "NOT_FOUND", "No such history item")
    return to_item(row)


@router.get("/{prediction_id}/image", response_class=FileResponse, summary="The stored (metadata-free) image")
async def get_history_image(
    prediction_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
    state: AppState = Depends(get_state),
) -> FileResponse:
    row = await get_owned(session, user, prediction_id)
    if row is None or row.image_path is None:
        raise AppError(404, "NOT_FOUND", "No stored image for this item")
    path = state.settings.upload_dir / row.image_path
    if not path.is_file():
        raise AppError(404, "NOT_FOUND", "Stored image has expired or was removed")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@router.delete("/{prediction_id}", status_code=204, summary="Delete an item and its stored image")
async def delete_history_item(
    prediction_id: str,
    user: User = Depends(current_user),
    session: AsyncSession = Depends(get_session),
    state: AppState = Depends(get_state),
) -> Response:
    row = await get_owned(session, user, prediction_id)
    if row is None:
        raise AppError(404, "NOT_FOUND", "No such history item")
    await delete_prediction(session, row, state.settings.upload_dir)
    return Response(status_code=204)
