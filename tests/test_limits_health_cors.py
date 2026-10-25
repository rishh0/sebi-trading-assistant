from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app as app_module
from scripts.rate_limit import RateLimiter, client_ip

client = TestClient(app_module.app)

ALLOWED_ORIGIN = "https://frontend-omega-six-90.vercel.app"


@pytest.fixture
def fake_answer(mocker):
    mocker.patch.object(app_module, "generate_answer", return_value=("ok", []))


# --- RateLimiter ---

def test_limiter_allows_up_to_limit_then_blocks():
    limiter = RateLimiter(limit=3, window_seconds=60)
    assert [limiter.check("ip") for _ in range(3)] == [None, None, None]
    retry_after = limiter.check("ip")
    assert retry_after is not None and 1 <= retry_after <= 61


def test_limiter_tracks_keys_independently():
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("a") is not None


def test_limiter_frees_up_after_window(mocker):
    clock = mocker.patch("scripts.rate_limit.time.monotonic", return_value=1000.0)
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.check("ip") is None
    assert limiter.check("ip") is not None
    clock.return_value = 1061.0
    assert limiter.check("ip") is None


def test_client_ip_prefers_first_forwarded_for_entry():
    request = MagicMock()
    request.headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
    assert client_ip(request) == "203.0.113.7"


def test_client_ip_falls_back_to_socket_address():
    request = MagicMock()
    request.headers = {}
    request.client.host = "198.51.100.2"
    assert client_ip(request) == "198.51.100.2"


# --- Endpoint rate limits ---

def test_query_returns_429_with_retry_after_once_limit_hit(fake_answer, monkeypatch):
    monkeypatch.setattr(app_module, "query_limiter_per_ip", RateLimiter(limit=2, window_seconds=60))
    for _ in range(2):
        assert client.post("/query", json={"question": "hi"}).status_code == 200
    res = client.post("/query", json={"question": "hi"})
    assert res.status_code == 429
    assert int(res.headers["Retry-After"]) >= 1
    assert "too quickly" in res.json()["detail"]


def test_query_limit_is_per_ip(fake_answer, monkeypatch):
    monkeypatch.setattr(app_module, "query_limiter_per_ip", RateLimiter(limit=1, window_seconds=60))
    ip_a = {"X-Forwarded-For": "203.0.113.1"}
    ip_b = {"X-Forwarded-For": "203.0.113.2"}
    assert client.post("/query", json={"question": "hi"}, headers=ip_a).status_code == 200
    assert client.post("/query", json={"question": "hi"}, headers=ip_a).status_code == 429
    assert client.post("/query", json={"question": "hi"}, headers=ip_b).status_code == 200


def test_global_cap_holds_even_when_ip_is_spoofed(fake_answer, monkeypatch):
    monkeypatch.setattr(app_module, "query_limiter_global", RateLimiter(limit=2, window_seconds=60))
    for i in range(2):
        headers = {"X-Forwarded-For": f"203.0.113.{i}"}
        assert client.post("/query", json={"question": "hi"}, headers=headers).status_code == 200
    res = client.post("/query", json={"question": "hi"}, headers={"X-Forwarded-For": "203.0.113.99"})
    assert res.status_code == 429
    assert "busy" in res.json()["detail"]


def test_stream_endpoint_is_rate_limited_too(monkeypatch):
    monkeypatch.setattr(app_module, "query_limiter_per_ip", RateLimiter(limit=0, window_seconds=60))
    assert client.post("/query-stream", json={"question": "hi"}).status_code == 429


def test_upload_is_rate_limited(monkeypatch):
    monkeypatch.setattr(app_module, "upload_limiter_per_ip", RateLimiter(limit=0, window_seconds=60))
    res = client.post("/upload", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert res.status_code == 429


# --- Question length ---

def test_overlong_question_is_rejected(fake_answer):
    res = client.post("/query", json={"question": "x" * (app_module.MAX_QUESTION_CHARS + 1)})
    assert res.status_code == 422


def test_empty_question_is_rejected(fake_answer):
    assert client.post("/query", json={"question": ""}).status_code == 422


# --- Health ---

def test_health_is_ok_and_cheap(mocker):
    llm = mocker.patch.object(app_module, "get_llm")
    store = mocker.patch.object(app_module, "get_vectorstore")
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
    llm.assert_not_called()
    store.assert_not_called()


# --- CORS ---

def test_cors_preflight_allows_frontend_post():
    res = client.options("/query", headers={
        "Origin": ALLOWED_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_cors_preflight_rejects_unused_method():
    res = client.options("/query", headers={
        "Origin": ALLOWED_ORIGIN,
        "Access-Control-Request-Method": "DELETE",
    })
    assert res.status_code == 400


def test_cors_preflight_rejects_unknown_origin():
    res = client.options("/query", headers={
        "Origin": "https://evil.example.com",
        "Access-Control-Request-Method": "POST",
    })
    assert res.status_code == 400
