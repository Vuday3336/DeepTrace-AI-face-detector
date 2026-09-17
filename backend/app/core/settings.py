"""Runtime configuration from environment variables (see backend/.env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DEEPTRACE_", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    # --- model bundle ---
    model_dir: Path = Path("models/current")
    model_repo_id: str | None = None  # e.g. "<user>/deeptrace-model" on the Hugging Face Hub
    model_revision: str = "main"  # pin a commit hash in production
    verify_model_checksums: bool = True
    onnx_threads: int | None = None
    detector_device: str = "cpu"

    # --- request limits ---
    max_upload_bytes: int = 10 * 1024 * 1024
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60
    trust_forwarded_for: bool = False  # enable only behind a proxy you control (HF Spaces, nginx)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- optional history (needs a database + JWT secret) ---
    database_url: str | None = None  # postgresql+asyncpg://user:pass@db:5432/deeptrace
    jwt_secret: str | None = None
    jwt_expire_minutes: int = 60
    upload_dir: Path = Path("data/uploads")
    retention_days: int = 30

    # --- single-container deployment: serve the built React app ---
    frontend_dir: Path | None = None

    @property
    def history_enabled(self) -> bool:
        return bool(self.database_url and self.jwt_secret)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.database_url and not self.jwt_secret:
            raise ValueError("DEEPTRACE_JWT_SECRET is required when DEEPTRACE_DATABASE_URL is set")
        if self.jwt_secret and len(self.jwt_secret) < 32:
            raise ValueError("DEEPTRACE_JWT_SECRET must be at least 32 characters")
        if self.rate_limit_requests < 1 or self.rate_limit_window_seconds < 1:
            raise ValueError("Rate limit values must be positive")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
