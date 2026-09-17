from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AppState, current_user, enforce_rate_limit, get_session, get_state
from app.core.errors import AppError
from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User
from app.schemas import Credentials, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"], dependencies=[Depends(enforce_rate_limit)])


@router.post("/register", response_model=UserOut, status_code=201, summary="Create an account")
async def register(body: Credentials, session: AsyncSession = Depends(get_session)) -> UserOut:
    user = User(email=body.email.lower(), password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise AppError(409, "EMAIL_TAKEN", "An account with this email already exists") from exc
    return UserOut(id=user.id, email=user.email, created_at=user.created_at)


@router.post("/login", response_model=TokenResponse, summary="Get an access token")
async def login(
    body: Credentials, session: AsyncSession = Depends(get_session), state: AppState = Depends(get_state)
) -> TokenResponse:
    user = await session.scalar(select(User).where(User.email == body.email.lower()))
    # Same error for unknown email and wrong password: don't reveal which accounts exist.
    if user is None or not verify_password(body.password, user.password_hash):
        raise AppError(401, "INVALID_CREDENTIALS", "Incorrect email or password")
    assert state.settings.jwt_secret is not None
    token = create_access_token(user.id, state.settings.jwt_secret, state.settings.jwt_expire_minutes)
    return TokenResponse(access_token=token, expires_in_minutes=state.settings.jwt_expire_minutes)


@router.get("/me", response_model=UserOut, summary="Current user")
async def me(user: User = Depends(current_user)) -> UserOut:
    return UserOut(id=user.id, email=user.email, created_at=user.created_at)
