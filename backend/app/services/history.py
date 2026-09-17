"""Opt-in prediction history: rows + stored images (re-encoded, metadata-free), with retention."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Prediction, User
from app.schemas import FaceOut, HistoryFace, HistoryItem

log = logging.getLogger("deeptrace.history")


def _store_image(image: Image.Image, upload_dir: Path, prediction_id: str) -> str:
    """Store the METADATA-STRIPPED decoded image (never the raw upload, which may carry GPS EXIF)."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{prediction_id}.jpg"
    image.save(path, format="JPEG", quality=95)
    return path.name


async def save_prediction(
    session: AsyncSession,
    user: User,
    model_name: str,
    model_version: str,
    faces: list[FaceOut],
    image: Image.Image,
    upload_dir: Path,
    latency_ms: float,
) -> Prediction:
    row = Prediction(
        user_id=user.id,
        model_name=model_name,
        model_version=model_version,
        num_faces=len(faces),
        faces=[
            HistoryFace(
                face_index=f.face_index, bbox=f.bbox, label=f.label, prob_ai_generated=f.prob_ai_generated
            ).model_dump()
            for f in faces
        ],
        latency_ms=latency_ms,
    )
    session.add(row)
    await session.flush()  # assigns the id
    try:
        row.image_path = _store_image(image, upload_dir, row.id)
        await session.commit()
    except Exception:
        await session.rollback()
        (upload_dir / f"{row.id}.jpg").unlink(missing_ok=True)
        raise
    return row


def to_item(row: Prediction) -> HistoryItem:
    return HistoryItem(
        id=row.id,
        created_at=row.created_at,
        model_name=row.model_name,
        model_version=row.model_version,
        num_faces=row.num_faces,
        faces=[HistoryFace(**f) for f in row.faces],
        has_image=row.image_path is not None,
    )


async def list_predictions(session: AsyncSession, user: User, limit: int, offset: int) -> tuple[list[Prediction], int]:
    total = await session.scalar(select(func.count()).select_from(Prediction).where(Prediction.user_id == user.id))
    rows = await session.scalars(
        select(Prediction)
        .where(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(rows), int(total or 0)


async def get_owned(session: AsyncSession, user: User, prediction_id: str) -> Prediction | None:
    row = await session.get(Prediction, prediction_id)
    return row if row is not None and row.user_id == user.id else None


async def delete_prediction(session: AsyncSession, row: Prediction, upload_dir: Path) -> None:
    if row.image_path:
        (upload_dir / row.image_path).unlink(missing_ok=True)
    await session.delete(row)
    await session.commit()


async def purge_expired_images(session: AsyncSession, upload_dir: Path, retention_days: int) -> int:
    """Delete stored images older than the retention period (rows stay, marked has_image=false)."""
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    rows = await session.scalars(
        select(Prediction).where(Prediction.created_at < cutoff, Prediction.image_path.is_not(None))
    )
    count = 0
    for row in rows:
        (upload_dir / str(row.image_path)).unlink(missing_ok=True)
        row.image_path = None
        count += 1
    await session.commit()
    if count:
        log.info("Purged %d stored images older than %d days", count, retention_days)
    return count
