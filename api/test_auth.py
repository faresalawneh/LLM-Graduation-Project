"""REQ-04 (#26): identity management tests.

Covers the token itself (expired, tampered, wrong key, alg none, bad claims),
the full role x endpoint access matrix, that the matrix matches the README API
reference, and that credentials come from configuration, not source code.
"""
import base64
import json
import re
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from jose import jwt

import app as app_module
import auth
from app import app

README = Path(__file__).resolve().parent.parent / "README.md"
THRESHOLD = {"name": "High TTFT", "metric": "ttft_ms", "operator": ">", "value": 500.0}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_thresholds():
    app_module._thresholds.clear()
    yield
    app_module._thresholds.clear()


def _login(client, role):
    resp = client.post("/auth/token", data={"username": role, "password": f"{role}123"})
    assert resp.status_code == 200  # nosec B101
    return resp.json()["access_token"]


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _b64(obj):
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()


def _assert_rejected(client, token):
    resp = client.get("/metrics/summary", headers=_bearer(token))
    assert resp.status_code == 401  # nosec B101
    assert resp.headers["www-authenticate"] == "Bearer"  # nosec B101


# ---------- Token validation: every case must be 401 ----------
def test_expired_token_rejected(client):
    token = auth.create_access_token({"sub": "admin", "role": "admin"}, expires_delta=timedelta(minutes=-1))
    _assert_rejected(client, token)


def test_token_signed_with_other_key_rejected(client):
    token = jwt.encode({"sub": "admin", "role": "admin"}, "attacker-key", algorithm="HS256")
    _assert_rejected(client, token)


def test_payload_tampered_viewer_to_admin_rejected(client):
    header, payload, signature = _login(client, "viewer").split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["role"] = "admin"
    forged = f"{header}.{_b64(claims)}.{signature}"
    resp = client.get("/gpu", headers=_bearer(forged))
    assert resp.status_code == 401  # nosec B101


def test_signature_tampered_rejected(client):
    token = _login(client, "admin")
    last = "A" if token[-1] != "A" else "B"
    _assert_rejected(client, token[:-1] + last)


def test_alg_none_rejected(client):
    token = f"{_b64({'alg': 'none', 'typ': 'JWT'})}.{_b64({'sub': 'admin', 'role': 'admin'})}."
    _assert_rejected(client, token)


@pytest.mark.parametrize(
    "claims",
    [
        {"sub": "admin"},                               # no role
        {"role": "admin"},                              # no subject
        {"sub": "root", "role": "superadmin"},          # unknown role
    ],
    ids=["missing-role", "missing-sub", "unknown-role"],
)
def test_bad_claims_rejected(client, claims):
    _assert_rejected(client, auth.create_access_token(claims))


@pytest.mark.parametrize(
    "headers",
    [{"Authorization": "Token abc"}, {"Authorization": "Bearer"}, {"Authorization": ""}],
    ids=["wrong-scheme", "empty-bearer", "empty-header"],
)
def test_malformed_authorization_header_rejected(client, headers):
    assert client.get("/metrics/summary", headers=headers).status_code == 401  # nosec B101


def test_token_carries_expiry_and_role(client):
    claims = jwt.get_unverified_claims(_login(client, "viewer"))
    assert claims["sub"] == "viewer" and claims["role"] == "viewer"  # nosec B101
    assert "exp" in claims  # nosec B101


# ---------- Role x endpoint matrix ----------
# Access levels exactly as written in the README API reference table.
ACCESS = {
    ("GET", "/health"): "public",
    ("GET", "/metrics/summary"): "any authenticated user",
    ("GET", "/gpu"): "admin",
    ("GET", "/vllm"): "admin",
    ("POST", "/thresholds"): "any authenticated user",
    ("GET", "/thresholds"): "any authenticated user",
    ("GET", "/thresholds/{id}"): "any authenticated user",
    ("PUT", "/thresholds/{id}"): "admin",
    ("DELETE", "/thresholds/{id}"): "admin",
}
SUCCESS = {"POST": 201, "DELETE": 204}


def _expected(access, caller):
    if access == "public":
        return None  # checked separately, status is endpoint-specific
    if caller == "anonymous":
        return 401
    if access == "admin" and caller == "viewer":
        return 403
    return "ok"


@pytest.mark.parametrize("caller", ["anonymous", "viewer", "admin"])
@pytest.mark.parametrize("method,path", list(ACCESS), ids=[f"{m} {p}" for m, p in ACCESS])
def test_access_matrix(client, method, path, caller):
    expected = _expected(ACCESS[(method, path)], caller)
    if expected is None:
        assert client.request(method, path).status_code == 200  # nosec B101
        return

    # Seed one threshold so /thresholds/{id} exists.
    seeded = client.post("/thresholds", json=THRESHOLD, headers=_bearer(_login(client, "admin"))).json()
    url = path.replace("{id}", str(seeded["id"]))
    headers = {} if caller == "anonymous" else _bearer(_login(client, caller))
    body = THRESHOLD if method in ("POST", "PUT") else None

    with patch("app.query_prometheus", return_value=1.0):
        resp = client.request(method, url, headers=headers, json=body)

    if expected == "ok":
        assert resp.status_code == SUCCESS.get(method, 200), resp.text  # nosec B101
    else:
        assert resp.status_code == expected, resp.text  # nosec B101


def test_matrix_covers_every_route():
    routes = {
        (method, route.path.replace("{threshold_id}", "{id}"))
        for route in app.routes
        if getattr(route, "include_in_schema", False)
        for method in route.methods
        if method != "HEAD"
    }
    routes.discard(("POST", "/auth/token"))
    assert routes == set(ACCESS)  # nosec B101


def test_readme_role_table_matches_code():
    """Every row of the README API reference must agree with ACCESS."""
    section = README.read_text(encoding="utf-8").split("## API reference", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| (GET|POST|PUT|DELETE) \| `([^`]+)` \| ([^|]+?) \|", section, re.M)
    documented = {(m, p): a for m, p, a in rows if p not in ("/", "/auth/token")}
    assert documented == ACCESS  # nosec B101


# ---------- Configuration: no credentials in source ----------
def test_no_credentials_in_auth_source():
    source = Path(auth.__file__).read_text(encoding="utf-8")
    for leaked in ("admin123", "viewer123", "dev-secret-change-in-production"):
        assert leaked not in source  # nosec B101


def test_users_loaded_from_environment():
    users = auth.load_users({"API_ADMIN_PASSWORD": "s3cret"})
    assert set(users) == {"admin"}  # nosec B101  viewer disabled when unset
    assert auth.verify_password("s3cret", users["admin"]["hashed_password"])  # nosec B101
    assert auth.load_users({}) == {}  # nosec B101


def test_disabled_account_cannot_log_in(client):
    with patch.object(auth, "USERS_DB", auth.load_users({"API_ADMIN_PASSWORD": "x"})):
        resp = client.post("/auth/token", data={"username": "viewer", "password": "viewer123"})
    assert resp.status_code == 401  # nosec B101


def test_secret_from_environment_or_random():
    assert auth.load_secret({"JWT_SECRET": "k" * 40}) == "k" * 40  # nosec B101
    first, second = auth.load_secret({}), auth.load_secret({})
    assert first != second and len(first) >= auth.MIN_SECRET_LENGTH  # nosec B101


def test_expiry_from_environment():
    assert auth.load_expire_minutes({}) == 60  # nosec B101
    assert auth.load_expire_minutes({"JWT_EXPIRE_MINUTES": "15"}) == 15  # nosec B101
