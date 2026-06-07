"""LLM Observability API - FastAPI with JWT auth."""
from typing import Annotated, Optional

import requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from auth import (
    Token,
    User,
    authenticate_user,
    create_access_token,
    get_current_user,
    require_role,
)



PROMETHEUS_URL = "http://localhost:9090"

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) # nosec