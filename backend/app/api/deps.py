"""Shared FastAPI dependencies: app state, model availability, DB session, users, rate limiting."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AppError
from app.core.rate_limit import SlidingWindowRateLimiter
from app.core.security import decode_access_token
from app.core.settings import Settings
from app.db.models import User
from app.db.session import Database
from deeptrace_ml.serving.runtime import Analyzer


@dataclass
class AppState:
    settings: Settings
    limiter: SlidingWindowRateLimiter
    analyzer: Analyzer | None = None
    model_error: str | None = None
    db: Database | None = None


def get_state(request: Request) -> AppState:
    return request.app.state.deeptrace


def require_analyzer(state: AppState = Depends(get_state)) -> Analyzer:
    if state.analyzer is None:
        raise AppError(503, "MODEL_NOT_LOADED", "The detection model is not available. Check /api/v1/health.")
    return state.analyzer


async def enforce_rate_limit(request: Request, state: AppState = Depends(get_state)) -> None:
    await state.limiter(request)


def require_history(state: AppState = Depends(get_state)) -> Database:
    if state.db is None or not state.settings.history_enabled:
        raise AppError(503, "HISTORY_DISABLED", "Accounts and history are disabled on this deployment.")
    return state.db


async def get_session(db: Database = Depends(require_history)) -> AsyncIterator[AsyncSession]:
    async with db.sessionmaker() as session:
        yield session


def _bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not header:
        return None
    if scheme.lower() != "bearer" or not token:
        raise AppError(401, "UNAUTHORIZED", "Use 'Authorization: Bearer <token>'")
    return token


async def optional_user(request: Request, state: AppState = Depends(get_state)) -> User | None:
    token = _bearer_token(request)
    if token is None:
        return None
    if state.db is None or not state.settings.jwt_secret:
        raise AppError(503, "HISTORY_DISABLED", "Accounts are disabled on this deployment.")
    user_id = decode_access_token(token, state.settings.jwt_secret)
    if user_id is None:
        raise AppError(401, "UNAUTHORIZED", "Invalid or expired token")
    async with state.db.sessionmaker() as session:
        user = await session.get(User, user_id)
    if user is None:
        raise AppError(401, "UNAUTHORIZED", "User no longer exists")
    return user


async def current_user(user: User | None = Depends(optional_user)) -> User:
    if user is None:
        raise AppError(401, "UNAUTHORIZED", "Login required")
    return user
