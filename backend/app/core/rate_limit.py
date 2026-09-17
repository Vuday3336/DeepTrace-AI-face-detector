"""Sliding-window rate limiter (in memory, per client IP).

Why not a library: the API runs as a single process on free hosting, so an in-process limiter is
sufficient and trivially testable. With several replicas this must move to Redis (documented limit).
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import Request

from app.core.errors import AppError


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int, trust_forwarded_for: bool = False) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self.trust_forwarded_for = trust_forwarded_for
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def client_key(self, request: Request) -> str:
        if self.trust_forwarded_for:
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = max(1, int(self.window - (now - hits[0])) + 1)
                raise AppError(
                    429,
                    "RATE_LIMITED",
                    f"Too many requests; retry in {retry_after}s",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)
            if len(self._hits) > 10_000:  # bound memory: drop idle clients
                for stale in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[stale]

    async def __call__(self, request: Request) -> None:
        self.check(self.client_key(request))
