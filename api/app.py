"""LLM Observability API - FastAPI with JWT auth."""
import os
from itertools import count
from typing import Annotated, Dict, List, Literal, Optional

import requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from auth import (
    Token,
    User,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_role,
)



# Inside Docker, "localhost" is the API container itself. Set PROMETHEUS_URL to
# the Prometheus service name (e.g. http://prometheus:9090) when the API shares
# the monitoring compose network.
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")

app = FastAPI(
    title="LLM Observability API",
    description="Real-time observability for vLLM inference serving on Tesla P40",
    version="1.0.0",
)


# ---------- Response models (for auto Swagger) ----------
class HealthResponse(BaseModel):
    status: str


class ScenarioMetric(BaseModel):
    steady: Optional[float] = None
    burst: Optional[float] = None


class MetricsSummary(BaseModel):
    ttft_ms: ScenarioMetric
    latency_ms: ScenarioMetric
    throughput_tokens_per_sec: ScenarioMetric


class GPUStats(BaseModel):
    sm_clock_mhz: Optional[float] = None
    memory_used_mib: Optional[float] = None
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None


class VLLMStats(BaseModel):
    requests_waiting: Optional[float] = None
    requests_running: Optional[float] = None
    kv_cache_usage_pct: Optional[float] = None


# Metrics an alert threshold can be attached to.
ThresholdMetric = Literal[
    "ttft_ms",
    "request_latency_ms",
    "throughput_tokens_per_sec",
    "gpu_temperature_c",
    "gpu_power_w",
    "gpu_memory_used_mib",
    "vllm_requests_waiting",
    "vllm_kv_cache_usage_pct",
]


class ThresholdBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    metric: ThresholdMetric
    operator: Literal[">", "<"] = ">"
    value: float
    enabled: bool = True


class ThresholdCreate(ThresholdBase):
    pass


class ThresholdUpdate(ThresholdBase):
    pass


class Threshold(ThresholdBase):
    id: int
    created_by: str


# ---------- In-memory threshold store (lost on restart) ----------
_thresholds: Dict[int, Threshold] = {}
_threshold_ids = count(1)


# ---------- Helper ----------
def query_prometheus(promql: str) -> Optional[float]:
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=5,
        )
        data = resp.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
        return None
    except Exception:
        return None


# ---------- Auth endpoint ----------
@app.post("/auth/token", response_model=Token, tags=["auth"])
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return Token(access_token=token, token_type="bearer", role=user["role"]) # nosec


# ---------- Public endpoint ----------
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["public"])
async def health():
    return HealthResponse(status="ok")


# ---------- Protected endpoints (any authenticated user) ----------
@app.get("/metrics/summary", response_model=MetricsSummary, tags=["metrics"])
async def metrics_summary(user: Annotated[User, Depends(get_current_user)]):
    return MetricsSummary(
        ttft_ms=ScenarioMetric(
            steady=query_prometheus('ttft_ms{scenario="steady"}'),
            burst=query_prometheus('ttft_ms{scenario="burst"}'),
        ),
        latency_ms=ScenarioMetric(
            steady=query_prometheus('request_latency_ms{scenario="steady"}'),
            burst=query_prometheus('request_latency_ms{scenario="burst"}'),
        ),
        throughput_tokens_per_sec=ScenarioMetric(
            steady=query_prometheus('throughput_tokens_per_sec{scenario="steady"}'),
            burst=query_prometheus('throughput_tokens_per_sec{scenario="burst"}'),
        ),
    )


@app.get("/gpu", response_model=GPUStats, tags=["metrics"])
async def gpu(user: Annotated[User, Depends(require_role("admin"))]):
    return GPUStats(
        sm_clock_mhz=query_prometheus("DCGM_FI_DEV_SM_CLOCK"),
        memory_used_mib=query_prometheus("DCGM_FI_DEV_FB_USED"),
        temperature_c=query_prometheus("DCGM_FI_DEV_GPU_TEMP"),
        power_w=query_prometheus("DCGM_FI_DEV_POWER_USAGE"),
    )


@app.get("/vllm", response_model=VLLMStats, tags=["metrics"])
async def vllm(user: Annotated[User, Depends(require_role("admin"))]):
    return VLLMStats(
        requests_waiting=query_prometheus("vllm:num_requests_waiting"),
        requests_running=query_prometheus("vllm:num_requests_running"),
        kv_cache_usage_pct=query_prometheus("vllm:gpu_cache_usage_perc"),
    )


# ---------- Alert thresholds: full CRUD ----------
# GET/POST: any authenticated user. PUT/DELETE: admin only.
@app.post(
    "/thresholds",
    response_model=Threshold,
    status_code=status.HTTP_201_CREATED,
    tags=["thresholds"],
)
async def create_threshold(
    body: ThresholdCreate, user: Annotated[User, Depends(get_current_user)]
):
    threshold = Threshold(id=next(_threshold_ids), created_by=user.username, **body.model_dump())
    _thresholds[threshold.id] = threshold
    return threshold


@app.get("/thresholds", response_model=List[Threshold], tags=["thresholds"])
async def list_thresholds(user: Annotated[User, Depends(get_current_user)]):
    return list(_thresholds.values())


@app.get("/thresholds/{threshold_id}", response_model=Threshold, tags=["thresholds"])
async def get_threshold(
    threshold_id: int, user: Annotated[User, Depends(get_current_user)]
):
    threshold = _thresholds.get(threshold_id)
    if threshold is None:
        raise HTTPException(status_code=404, detail="Threshold not found")
    return threshold


@app.put("/thresholds/{threshold_id}", response_model=Threshold, tags=["thresholds"])
async def update_threshold(
    threshold_id: int,
    body: ThresholdUpdate,
    user: Annotated[User, Depends(require_role("admin"))],
):
    existing = _thresholds.get(threshold_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Threshold not found")
    updated = Threshold(id=threshold_id, created_by=existing.created_by, **body.model_dump())
    _thresholds[threshold_id] = updated
    return updated


@app.delete("/thresholds/{threshold_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["thresholds"])
async def delete_threshold(
    threshold_id: int, user: Annotated[User, Depends(require_role("admin"))]
):
    if _thresholds.pop(threshold_id, None) is None:
        raise HTTPException(status_code=404, detail="Threshold not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) # nosec