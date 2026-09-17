"""FastAPI application factory.

uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from app.api.deps import AppState
from app.api.routes import auth, health, history, model_info, predict
from app.core.errors import register_error_handlers
from app.core.rate_limit import SlidingWindowRateLimiter
from app.core.settings import Settings, get_settings
from app.db.session import Database
from app.inference.loader import load_bundle, load_detector
from app.services.history import purge_expired_images
from deeptrace_ml.serving.runtime import Analyzer

API_PREFIX = "/api/v1"
log = logging.getLogger("deeptrace")


class SPAStaticFiles(StaticFiles):
    """Serve the built React app; unknown paths fall back to index.html for client-side routing."""

    async def get_response(self, path: str, scope: dict) -> Response:  # type: ignore[override]
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:  # newer Starlette raises instead of returning a 404
            if exc.status_code != 404:
                raise
            response = Response(status_code=404)
        if response.status_code == 404 and not path.startswith("api/"):
            return FileResponse(Path(str(self.directory)) / "index.html")
        return response


def create_app(settings: Settings | None = None, analyzer: Analyzer | None = None) -> FastAPI:
    """`analyzer` can be injected (tests); otherwise the bundle and MTCNN are loaded at startup."""
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    state = AppState(
        settings=settings,
        limiter=SlidingWindowRateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds, settings.trust_forwarded_for
        ),
        analyzer=analyzer,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if state.analyzer is None:
            try:
                bundle = load_bundle(settings)
                state.analyzer = Analyzer(bundle, load_detector(bundle, settings))
            except Exception as exc:  # noqa: BLE001 — keep serving /health so the failure is visible
                state.model_error = f"{type(exc).__name__}: {exc}"
                log.error("Model failed to load: %s", state.model_error)
        if settings.history_enabled and settings.database_url:
            state.db = Database(settings.database_url)
            await state.db.create_all()
            async with state.db.sessionmaker() as session:
                await purge_expired_images(session, settings.upload_dir, settings.retention_days)
        yield
        if state.db is not None:
            await state.db.dispose()

    app = FastAPI(
        title="DeepTrace API",
        version="1.0.0",
        description="Explainable AI-generated face detection. A detection aid, not proof.",
        lifespan=lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.state.deeptrace = state
    register_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        log.info(
            "%s %s %d %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    for module in (health, model_info, predict, auth, history):
        app.include_router(module.router, prefix=API_PREFIX)

    if settings.frontend_dir is not None:
        if not (settings.frontend_dir / "index.html").is_file():
            raise RuntimeError(f"DEEPTRACE_FRONTEND_DIR={settings.frontend_dir} has no index.html")
        app.mount("/", SPAStaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app
