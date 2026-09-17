from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import image_bytes

CREDS = {"email": "Alice@Example.com", "password": "correct-horse-1"}


def _token(c: TestClient) -> dict[str, str]:
    assert c.post("/api/v1/auth/register", json=CREDS).status_code == 201
    res = c.post("/api/v1/auth/login", json=CREDS)
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _predict(c: TestClient, headers: dict[str, str], store: bool):
    return c.post(
        "/api/v1/predict",
        files={"file": ("f.jpg", image_bytes(), "image/jpeg")},
        params={"store": str(store).lower()},
        headers=headers,
    )


def test_register_login_me(history_client: TestClient):
    headers = _token(history_client)
    me = history_client.get("/api/v1/auth/me", headers=headers).json()
    assert me["email"] == "alice@example.com"  # normalised
    dup = history_client.post("/api/v1/auth/register", json=CREDS)
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "EMAIL_TAKEN"


def test_login_errors_do_not_reveal_accounts(history_client: TestClient):
    _token(history_client)
    wrong_pw = history_client.post("/api/v1/auth/login", json={**CREDS, "password": "wrong-password"})
    unknown = history_client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "whatever1"})
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["error"]["message"] == unknown.json()["error"]["message"]


def test_bad_tokens_rejected(history_client: TestClient):
    assert history_client.get("/api/v1/history").status_code == 401
    assert history_client.get("/api/v1/history", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    assert history_client.get("/api/v1/history", headers={"Authorization": "Basic abc"}).status_code == 401


def test_privacy_default_stores_nothing(history_client: TestClient, tmp_path):
    headers = _token(history_client)
    res = _predict(history_client, headers, store=False)
    assert res.json()["stored"] is False
    assert history_client.get("/api/v1/history", headers=headers).json()["total"] == 0
    uploads = tmp_path / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())


def test_store_list_image_delete(history_client: TestClient, tmp_path):
    headers = _token(history_client)
    body = _predict(history_client, headers, store=True).json()
    assert body["stored"] is True
    item_id = body["history_id"]

    page = history_client.get("/api/v1/history", headers=headers).json()
    assert page["total"] == 1 and page["items"][0]["id"] == item_id and page["items"][0]["has_image"]
    assert page["items"][0]["faces"][0]["label"] == body["faces"][0]["label"]

    image = history_client.get(f"/api/v1/history/{item_id}/image", headers=headers)
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"
    stored = tmp_path / "uploads" / f"{item_id}.jpg"
    assert stored.is_file()

    assert history_client.delete(f"/api/v1/history/{item_id}", headers=headers).status_code == 204
    assert not stored.exists()
    assert history_client.get(f"/api/v1/history/{item_id}", headers=headers).status_code == 404


def test_users_cannot_see_each_other(history_client: TestClient):
    alice = _token(history_client)
    item_id = _predict(history_client, alice, store=True).json()["history_id"]
    bob_creds = {"email": "bob@example.com", "password": "another-pass-2"}
    history_client.post("/api/v1/auth/register", json=bob_creds)
    bob = {
        "Authorization": "Bearer " + history_client.post("/api/v1/auth/login", json=bob_creds).json()["access_token"]
    }
    assert history_client.get(f"/api/v1/history/{item_id}", headers=bob).status_code == 404
    assert history_client.get(f"/api/v1/history/{item_id}/image", headers=bob).status_code == 404
    assert history_client.delete(f"/api/v1/history/{item_id}", headers=bob).status_code == 404
    assert history_client.get("/api/v1/history", headers=bob).json()["total"] == 0


def test_short_password_rejected(history_client: TestClient):
    res = history_client.post("/api/v1/auth/register", json={"email": "c@example.com", "password": "short"})
    assert res.status_code == 422
