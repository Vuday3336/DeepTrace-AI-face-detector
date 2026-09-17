from __future__ import annotations

import json
import shutil

from fastapi.testclient import TestClient

from app.core.errors import AppError
from app.core.rate_limit import SlidingWindowRateLimiter
from app.main import create_app
from tests.conftest import make_settings


def test_health_ok(client: TestClient):
    body = client.get("/api/v1/health").json()
    assert body == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "tiny_test",
        "model_version": "0.0.0-test",
        "model_error": None,
        "history": "disabled",
        "db": "disabled",
    }


def test_model_info_hides_internal_head_weights(client: TestClient):
    body = client.get("/api/v1/model-info").json()
    assert body["model"]["name"] == "tiny_test"
    assert "head" not in body["explanation"]
    assert body["preprocessing"]["input_size"] == 224
    assert body["known_limitations"]


def test_missing_model_degrades_gracefully(tmp_path):
    app = create_app(make_settings(tmp_path, model_dir=tmp_path / "nothing-here"))
    with TestClient(app) as c:
        health = c.get("/api/v1/health").json()
        predict = c.post("/api/v1/predict", files={"file": ("a.jpg", b"\xff\xd8\xff", "image/jpeg")})
    assert health["status"] == "degraded" and health["model_loaded"] is False
    assert "No model bundle" in health["model_error"]
    assert predict.status_code == 503 and predict.json()["error"]["code"] == "MODEL_NOT_LOADED"


def test_corrupted_bundle_detected_by_checksum(tmp_path, bundle_dir):
    broken = tmp_path / "broken"
    shutil.copytree(bundle_dir, broken)
    card = json.loads((broken / "model_card.json").read_text())
    card["decision"]["threshold"] = 0.01  # tampering
    (broken / "model_card.json").write_text(json.dumps(card))
    app = create_app(make_settings(tmp_path, model_dir=broken))
    with TestClient(app) as c:
        health = c.get("/api/v1/health").json()
    assert health["model_loaded"] is False and "Checksum mismatch" in health["model_error"]


def test_openapi_documents_endpoints(client: TestClient):
    paths = client.get("/api/v1/openapi.json").json()["paths"]
    assert {"/api/v1/predict", "/api/v1/model-info", "/api/v1/health"} <= set(paths)


def test_unknown_route_uses_error_envelope(client: TestClient):
    res = client.get("/api/v1/nope")
    assert res.status_code == 404 and res.json()["error"]["code"] == "NOT_FOUND"


def test_history_disabled_without_database(client: TestClient):
    res = client.post("/api/v1/auth/register", json={"email": "a@example.com", "password": "password123"})
    assert res.status_code == 503 and res.json()["error"]["code"] == "HISTORY_DISABLED"


def test_rate_limiter_window_expires():
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=10)
    limiter.check("ip", now=0.0)
    limiter.check("ip", now=1.0)
    try:
        limiter.check("ip", now=2.0)
        raise AssertionError("expected rate limit")
    except AppError as exc:
        assert exc.status_code == 429
    limiter.check("ip", now=10.5)  # first hit expired
    limiter.check("other", now=2.0)  # separate client


def test_spa_fallback_serves_index(tmp_path, bundle):
    from deeptrace_ml.serving.runtime import Analyzer
    from tests.conftest import StubDetector

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>DeepTrace</html>")
    app = create_app(make_settings(tmp_path, frontend_dir=dist), analyzer=Analyzer(bundle, StubDetector()))
    with TestClient(app) as c:
        assert "DeepTrace" in c.get("/history").text  # client-side route
        assert c.get("/api/v1/health").status_code == 200  # API still wins
