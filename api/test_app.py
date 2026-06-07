import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app, query_prometheus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin_token(client):
    resp = client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    return resp.json()["access_token"]


@pytest.fixture
def viewer_token(client):
    resp = client.post(
        "/auth/token",
        data={"username": "viewer", "password": "viewer123"},
    )
    return resp.json()["access_token"]


@pytest.fixture
def auth_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


# ---------- Public endpoint ----------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200  # nosec B101
    assert resp.json()["status"] == "ok"  # nosec B101


# ---------- Auth ----------
def test_login_success(client):
    resp = client.post(
        "/auth/token",
        data={"username": "admin", "password": "admin123"},
    )
    assert resp.status_code == 200  # nosec B101
    data = resp.json()
    assert "access_token" in data  # nosec B101
    assert data["token_type"] == "bearer"  # nosec B101
    assert data["role"] == "admin"  # nosec B101




def test_login_wrong_password(client):
    resp = client.post(
        "/auth/token",
        data={"username": "admin", "password": "wrong"},
    )
    assert resp.status_code == 401  # nosec B101


def test_login_unknown_user(client):
    resp = client.post(
        "/auth/token",
        data={"username": "nobody", "password": "pw"},
    )
    assert resp.status_code == 401  # nosec B101


def test_protected_endpoint_no_token(client):
    resp = client.get("/metrics/summary")
    assert resp.status_code == 401  # nosec B101


def test_protected_endpoint_bad_token(client):
    resp = client.get(
        "/metrics/summary",
        headers={"Authorization": "Bearer invalid.token.here"},
    )
    assert resp.status_code == 401  # nosec B101



def test_viewer_can_access_metrics(client, viewer_token):
    with patch("app.query_prometheus", return_value=50.0):
        resp = client.get(
            "/metrics/summary",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 200  # nosec B101


def test_viewer_cannot_access_gpu(client, viewer_token):
    with patch("app.query_prometheus", return_value=1200.0):
        resp = client.get(
            "/gpu",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403  # nosec B101


def test_viewer_cannot_access_vllm(client, viewer_token):
    with patch("app.query_prometheus", return_value=5.0):
        resp = client.get(
            "/vllm",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        assert resp.status_code == 403  # nosec B101


def test_admin_can_access_gpu(client, auth_headers):
    with patch("app.query_prometheus", return_value=1200.0):
        resp = client.get("/gpu", headers=auth_headers)
        assert resp.status_code == 200  # nosec B101


# ---------- Metrics summary ----------
def test_metrics_summary_returns_expected_keys(client, auth_headers):
    with patch("app.query_prometheus", return_value=123.4):
        resp = client.get("/metrics/summary", headers=auth_headers)
        assert resp.status_code == 200  # nosec B101
        data = resp.json()
        assert "ttft_ms" in data  # nosec B101
        assert "latency_ms" in data  # nosec B101
        assert "throughput_tokens_per_sec" in data  # nosec B101


def test_metrics_summary_has_steady_and_burst(client, auth_headers):
    with patch("app.query_prometheus", return_value=50.0):
        resp = client.get("/metrics/summary", headers=auth_headers)
        data = resp.json()
        assert "steady" in data["ttft_ms"]  # nosec B101
        assert "burst" in data["ttft_ms"]  # nosec B101


def test_metrics_summary_prometheus_down(client, auth_headers):
    with patch("app.query_prometheus", return_value=None):
        resp = client.get("/metrics/summary", headers=auth_headers)
        assert resp.status_code == 200  # nosec B101
        data = resp.json()
        assert data["ttft_ms"]["steady"] is None  # nosec B101


# ---------- GPU ----------
def test_gpu_returns_expected_keys(client, auth_headers):
    with patch("app.query_prometheus", return_value=1200.0):
        resp = client.get("/gpu", headers=auth_headers)
        assert resp.status_code == 200  # nosec B101
        data = resp.json()
        assert "sm_clock_mhz" in data  # nosec B101
        assert "memory_used_mib" in data  # nosec B101
        assert "temperature_c" in data  # nosec B101
        assert "power_w" in data  # nosec B101


# ---------- vLLM ----------
def test_vllm_returns_expected_keys(client, auth_headers):
    with patch("app.query_prometheus", return_value=5.0):
        resp = client.get("/vllm", headers=auth_headers)
        assert resp.status_code == 200  # nosec B101
        data = resp.json()
        assert "requests_waiting" in data  # nosec B101
        assert "requests_running" in data  # nosec B101
        assert "kv_cache_usage_pct" in data  # nosec B101



# ---------- Helper ----------
def test_query_prometheus_handles_exception():
    with patch("requests.get", side_effect=Exception("connection refused")):
        result = query_prometheus("ttft_ms")
        assert result is None  # nosec B101