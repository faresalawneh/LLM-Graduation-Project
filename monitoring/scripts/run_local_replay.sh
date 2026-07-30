#!/usr/bin/env bash
set -euo pipefail

# Local laptop replay runner - WSL2 + RTX 5060 Laptop (8GB, Blackwell sm_120)
# Location: monitoring/scripts/run_local_replay.sh
# Usage: ./run_local_replay.sh [steady|burst]
#
# Assumes vLLM and the monitoring stack are already up (see stack_up.sh).
# Waits for vLLM readiness rather than failing immediately, so this can be
# launched right after the container starts.

SCENARIO="${1:-steady}"

# This script lives in monitoring/scripts/ -> repo root is two levels up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export INFERENCE_URL="http://localhost:8000"
export ENDPOINT_TYPE="completions"
export OTEL_URL="http://localhost:4318/v1/metrics"
export PROMETHEUS_URL="http://localhost:9090"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts}"
export ENABLE_GPU_SCRAPE=1
export SLICE_DURATION=5
export MAX_CONCURRENCY=20          # frozen for comparability with prior runs
export SCENARIO

case "$SCENARIO" in
  steady) export TRACE_FILE="${TRACE_FILE:-${SCRIPT_DIR}/steady_window.csv}" ;;
  burst)  export TRACE_FILE="${TRACE_FILE:-${SCRIPT_DIR}/burst_window.csv}"  ;;
  *) echo "usage: $0 [steady|burst]" >&2; exit 1 ;;
esac

# --- preflight -------------------------------------------------------------

[ -f "$TRACE_FILE" ] || { echo "missing trace: $TRACE_FILE" >&2; exit 1; }

command -v aiperf >/dev/null \
  || { echo "aiperf not on PATH - activate venv: source ~/venvs/aiperf-env/bin/activate" >&2; exit 1; }

# LFS checkout re-materializes these with CRLF; normalize idempotently.
if grep -q $'\r' "$TRACE_FILE"; then
  echo "normalizing CRLF in $(basename "$TRACE_FILE")"
  sed -i 's/\r$//' "$TRACE_FILE"
fi

# vLLM takes ~60-120s to load the 1.5B model; wait rather than fail.
VLLM_WAIT="${VLLM_WAIT:-300}"
elapsed=0
until curl -sf "${INFERENCE_URL}/v1/models" >/dev/null 2>&1; do
  if [ "$elapsed" -ge "$VLLM_WAIT" ]; then
    echo "vLLM not ready at ${INFERENCE_URL} after ${VLLM_WAIT}s" >&2
    exit 1
  fi
  printf '\rwaiting for vLLM... %ss' "$elapsed"
  sleep 5
  elapsed=$((elapsed + 5))
done
[ "$elapsed" -gt 0 ] && echo

curl -sf "localhost:9400/metrics" > /dev/null \
  || echo "WARN: nvml-exporter down - GPU panels will be empty" >&2
curl -sf "localhost:9090/-/healthy" > /dev/null \
  || echo "WARN: prometheus down" >&2

echo "scenario=${SCENARIO} model=${MODEL} concurrency=${MAX_CONCURRENCY}"
echo "trace=${TRACE_FILE}"
echo "artifacts=${ARTIFACT_ROOT}"
echo

python "${SCRIPT_DIR}/aiperf_trace_replay.py"
