import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
import app as app_module
from app import app, query_prometheus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_thresholds():
    app_module._thresholds.clear()
    yield
    app_module._thresholds.clear()


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


def test_query_prometheus_uses_configured_url():
    class FakeResp:
        def json(self):
            return {"status": "success", "data": {"result": [{"value": [0, "42.5"]}]}}

    with patch.object(app_module, "PROMETHEUS_URL", "http://prometheus:9090"), \
            patch("requests.get", return_value=FakeResp()) as mock_get:
        assert query_prometheus("ttft_ms") == 42.5  # nosec B101
        assert mock_get.call_args.args[0] == "http://prometheus:9090/api/v1/query"  # nosec B101


# ---------- Thresholds CRUD ----------
THRESHOLD = {"name": "High TTFT", "metric": "ttft_ms", "operator": ">", "value": 500.0}


@pytest.fixture
def viewer_headers(viewer_token):
    return {"Authorization": f"Bearer {viewer_token}"}


def test_thresholds_require_auth(client):
    assert client.get("/thresholds").status_code == 401  # nosec B101
    assert client.post("/thresholds", json=THRESHOLD).status_code == 401  # nosec B101
    assert client.put("/thresholds/1", json=THRESHOLD).status_code == 401  # nosec B101
    assert client.delete("/thresholds/1").status_code == 401  # nosec B101


def test_create_threshold(client, viewer_headers):
    resp = client.post("/thresholds", json=THRESHOLD, headers=viewer_headers)
    assert resp.status_code == 201  # nosec B101
    data = resp.json()
    assert data["id"] == 1  # nosec B101
    assert data["created_by"] == "viewer"  # nosec B101
    assert data["enabled"] is True  # nosec B101
    assert data["metric"] == "ttft_ms"  # nosec B101


def test_create_threshold_invalid_metric(client, auth_headers):
    bad = {**THRESHOLD, "metric": "not_a_metric"}
    assert client.post("/thresholds", json=bad, headers=auth_headers).status_code == 422  # nosec B101


def test_create_threshold_missing_field(client, auth_headers):
    resp = client.post("/thresholds", json={"name": "x"}, headers=auth_headers)
    assert resp.status_code == 422  # nosec B101


def test_list_and_get_threshold(client, auth_headers, viewer_headers):
    created = client.post("/thresholds", json=THRESHOLD, headers=auth_headers).json()
    listing = client.get("/thresholds", headers=viewer_headers)
    assert listing.status_code == 200  # nosec B101
    assert [t["id"] for t in listing.json()] == [created["id"]]  # nosec B101
    one = client.get(f"/thresholds/{created['id']}", headers=viewer_headers)
    assert one.status_code == 200  # nosec B101
    assert one.json() == created  # nosec B101


def test_get_threshold_not_found(client, auth_headers):
    assert client.get("/thresholds/999", headers=auth_headers).status_code == 404  # nosec B101


def test_admin_can_update_threshold(client, auth_headers):
    created = client.post("/thresholds", json=THRESHOLD, headers=auth_headers).json()
    new = {**THRESHOLD, "value": 800.0, "enabled": False}
    resp = client.put(f"/thresholds/{created['id']}", json=new, headers=auth_headers)
    assert resp.status_code == 200  # nosec B101
    data = resp.json()
    assert data["value"] == 800.0  # nosec B101
    assert data["enabled"] is False  # nosec B101
    assert data["id"] == created["id"]  # nosec B101
    assert data["created_by"] == "admin"  # nosec B101
    fetched = client.get(f"/thresholds/{created['id']}", headers=auth_headers).json()
    assert fetched["value"] == 800.0  # nosec B101


def test_viewer_cannot_update_threshold(client, auth_headers, viewer_headers):
    created = client.post("/thresholds", json=THRESHOLD, headers=auth_headers).json()
    resp = client.put(f"/thresholds/{created['id']}", json=THRESHOLD, headers=viewer_headers)
    assert resp.status_code == 403  # nosec B101


def test_update_threshold_not_found(client, auth_headers):
    resp = client.put("/thresholds/999", json=THRESHOLD, headers=auth_headers)
    assert resp.status_code == 404  # nosec B101


def test_admin_can_delete_threshold(client, auth_headers):
    created = client.post("/thresholds", json=THRESHOLD, headers=auth_headers).json()
    resp = client.delete(f"/thresholds/{created['id']}", headers=auth_headers)
    assert resp.status_code == 204  # nosec B101
    assert client.get(f"/thresholds/{created['id']}", headers=auth_headers).status_code == 404  # nosec B101


def test_viewer_cannot_delete_threshold(client, auth_headers, viewer_headers):
    created = client.post("/thresholds", json=THRESHOLD, headers=auth_headers).json()
    resp = client.delete(f"/thresholds/{created['id']}", headers=viewer_headers)
    assert resp.status_code == 403  # nosec B101
    assert client.get(f"/thresholds/{created['id']}", headers=auth_headers).status_code == 200  # nosec B101


def test_delete_threshold_not_found(client, auth_headers):
    assert client.delete("/thresholds/999", headers=auth_headers).status_code == 404  # nosec B101


def test_openapi_lists_all_four_methods(client):
    paths = client.get("/openapi.json").json()["paths"]
    methods = set(paths["/thresholds"]) | set(paths["/thresholds/{threshold_id}"])
    assert {"get", "post", "put", "delete"} <= methods  # nosec B101
