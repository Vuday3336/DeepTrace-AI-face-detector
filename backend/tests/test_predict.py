from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from deeptrace_ml.serving.runtime import Analyzer
from tests.conftest import StubDetector, image_bytes, make_settings


def _post(client: TestClient, data: bytes, name: str = "face.jpg", mime: str = "image/jpeg", **params: object):
    return client.post("/api/v1/predict", files={"file": (name, data, mime)}, params=params)


def test_predict_single_face_full_response(client: TestClient):
    res = _post(client, image_bytes())
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["model"]["name"] == "tiny_test"
    assert body["image"] == {"width": 320, "height": 240}
    assert body["stored"] is False and body["history_id"] is None
    assert "not proof" in body["disclaimer"]
    [face] = body["faces"]
    assert face["label"] in {"REAL", "AI_GENERATED", "UNCERTAIN"}
    assert 0.0 <= face["prob_ai_generated"] <= 1.0
    assert len(face["bbox"]) == 4
    heatmap = Image.open(io.BytesIO(base64.b64decode(face["heatmap_png_base64"])))
    crop = Image.open(io.BytesIO(base64.b64decode(face["face_crop_png_base64"])))
    assert heatmap.mode == "RGBA" and heatmap.size == (224, 224)
    assert crop.size == (224, 224)
    assert set(body["timings_ms"]) >= {"decode", "detect", "inference", "explain", "encode"}
    assert res.headers["X-Request-ID"] == body["request_id"]


def test_label_follows_threshold_and_uncertainty_band(client: TestClient):
    bright = _post(client, image_bytes(value=250)).json()["faces"][0]
    dark = _post(client, image_bytes(value=5)).json()["faces"][0]
    assert bright["prob_ai_generated"] > dark["prob_ai_generated"]
    for face in (bright, dark):
        p = face["prob_ai_generated"]
        expected = "UNCERTAIN" if abs(p - 0.5) < 0.05 else ("AI_GENERATED" if p >= 0.5 else "REAL")
        assert face["label"] == expected


def test_explain_false_omits_heatmap(client: TestClient):
    face = _post(client, image_bytes(), explain="false").json()["faces"][0]
    assert face["heatmap_png_base64"] is None


def test_multiple_faces_each_analysed(tmp_path, bundle):
    app = create_app(make_settings(tmp_path), analyzer=Analyzer(bundle, StubDetector(faces_per_image=3)))
    with TestClient(app) as c:
        body = _post(c, image_bytes(width=900, height=300)).json()
    assert [f["face_index"] for f in body["faces"]] == [0, 1, 2]


def test_png_and_webp_accepted(client: TestClient):
    assert _post(client, image_bytes(fmt="PNG"), "a.png", "image/png").status_code == 200
    assert _post(client, image_bytes(fmt="WEBP"), "a.webp", "image/webp").status_code == 200


def test_no_face_returns_422(client: TestClient):
    res = _post(client, image_bytes(width=80, height=80))
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "NO_FACE_DETECTED"


def test_unsupported_type_by_magic_bytes_even_with_jpg_name(client: TestClient):
    res = _post(client, image_bytes(fmt="BMP"), "sneaky.jpg", "image/jpeg")
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_corrupt_image_returns_422(client: TestClient):
    res = _post(client, image_bytes(fmt="PNG")[:100], "broken.png", "image/png")
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_IMAGE"


def test_too_large_returns_413(tmp_path, bundle):
    app = create_app(make_settings(tmp_path, max_upload_bytes=1000), analyzer=Analyzer(bundle, StubDetector()))
    with TestClient(app) as c:
        res = _post(c, image_bytes(width=800, height=800, fmt="PNG"), "big.png", "image/png")
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_missing_file_is_validation_error(client: TestClient):
    res = client.post("/api/v1/predict")
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"


def test_store_without_login_rejected(client: TestClient):
    res = _post(client, image_bytes(), store="true")
    assert res.status_code == 401


def test_rate_limit(tmp_path, bundle):
    app = create_app(make_settings(tmp_path, rate_limit_requests=2), analyzer=Analyzer(bundle, StubDetector()))
    with TestClient(app) as c:
        codes = [_post(c, image_bytes()).status_code for _ in range(3)]
        limited = _post(c, image_bytes())
    assert codes == [200, 200, 429]
    assert limited.json()["error"]["code"] == "RATE_LIMITED"
    assert int(limited.headers["Retry-After"]) >= 1


def test_exif_gps_never_echoed(client: TestClient):
    exif = Image.Exif()
    exif[0x010F] = "SecretPhoneMaker"
    res = _post(client, image_bytes(exif=exif.tobytes()))
    assert res.status_code == 200
    assert "SecretPhoneMaker" not in res.text
