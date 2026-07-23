#!/usr/bin/env bash
set -euo pipefail

# Local laptop replay runner - WSL2 + RTX 2050 (4GB)
# Location: monitoring/scripts/run_local_replay.sh
# Usage: ./run_local_replay.sh [steady|burst]

SCENARIO="${1:-steady}"

# This script lives in monitoring/scripts/ -> repo root is two levels up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export INFERENCE_URL="http://localhost:8000"
export ENDPOINT_TYPE="completions"
export OTEL_URL="http://localhost:4318/v1/metrics"
export PROMETHEUS_URL="http://localhost:9090"
export ARTIFACT_ROOT="${REPO_ROOT}/artifacts"
export ENABLE_GPU_SCRAPE=1
export SLICE_DURATION=5
export MAX_CONCURRENCY=20          # 4GB card - do not raise without watching VRAM
export SCENARIO

case "$SCENARIO" in
  steady) export TRACE_FILE="${SCRIPT_DIR}/steady_window.csv" ;;
  burst)  export TRACE_FILE="${SCRIPT_DIR}/burst_window.csv"  ;;
  *) echo "usage: $0 [steady|burst]" >&2; exit 1 ;;
esac

# --- preflight -------------------------------------------------------------
[ -f "$TRACE_FILE" ] || { echo "missing trace: $TRACE_FILE" >&2; exit 1; }

curl -sf "${INFERENCE_URL}/v1/models" > /dev/null \
  || { echo "vLLM not reachable at ${INFERENCE_URL}" >&2; exit 1; }

curl -sf "localhost:9400/metrics" > /dev/null \
  || echo "WARN: nvml-exporter down - GPU panels will be empty" >&2

curl -sf "localhost:9090/-/healthy" > /dev/null \
  || echo "WARN: prometheus down" >&2

echo "scenario=${SCENARIO} model=${MODEL} concurrency=${MAX_CONCURRENCY}"
echo "trace=${TRACE_FILE}"
echo "artifacts=${ARTIFACT_ROOT}"
echo

python "${SCRIPT_DIR}/aiperf_trace_replay.py"