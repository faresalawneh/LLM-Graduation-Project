import pytest
from unittest.mock import patch
from app import app, query_prometheus


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200  # nosec B101
    assert resp.get_json()["status"] == "ok"  # nosec B101


def test_metrics_summary_returns_expected_keys(client):
    with patch("app.query_prometheus", return_value=123.4):
        resp = client.get("/metrics/summary")
        assert resp.status_code == 200  # nosec B101
        data = resp.get_json()
        assert "ttft_ms" in data  # nosec B101
        assert "latency_ms" in data  # nosec B101
        assert "throughput_tokens_per_sec" in data  # nosec B101


def test_metrics_summary_has_steady_and_burst(client):
    with patch("app.query_prometheus", return_value=50.0):
        resp = client.get("/metrics/summary")
        data = resp.get_json()
        assert "steady" in data["ttft_ms"]  # nosec B101
        assert "burst" in data["ttft_ms"]  # nosec B101


def test_metrics_summary_prometheus_down(client):
    with patch("app.query_prometheus", return_value=None):
        resp = client.get("/metrics/summary")
        assert resp.status_code == 200  # nosec B101
        data = resp.get_json()
        assert data["ttft_ms"]["steady"] is None  # nosec B101


def test_gpu_returns_expected_keys(client):
    with patch("app.query_prometheus", return_value=1200.0):
        resp = client.get("/gpu")
        assert resp.status_code == 200  # nosec B101
        data = resp.get_json()
        assert "sm_clock_mhz" in data  # nosec B101
        assert "memory_used_mib" in data  # nosec B101
        assert "temperature_c" in data  # nosec B101
        assert "power_w" in data  # nosec B101


def test_vllm_returns_expected_keys(client):
    with patch("app.query_prometheus", return_value=5.0):
        resp = client.get("/vllm")
        assert resp.status_code == 200  # nosec B101
        data = resp.get_json()
        assert "requests_waiting" in data  # nosec B101
        assert "requests_running" in data  # nosec B101
        assert "kv_cache_usage_pct" in data  # nosec B101


def test_query_prometheus_handles_exception():
    with patch("requests.get", side_effect=Exception("connection refused")):
        result = query_prometheus("ttft_ms")
        assert result is None  # nosec B101
