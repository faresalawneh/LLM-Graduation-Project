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
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-20}"   # default frozen at 20 for comparability; override per-run
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

# LFS checkout materializes these with CRLF. The traces are tracked, so never
# edit them in place: replay an LF copy and record where it came from, for
# mlflow_log_run.py. The copy keeps the file name and stays with the artifacts.
TRACE_ORIGINAL="$(realpath "$TRACE_FILE")"
mkdir -p "$ARTIFACT_ROOT"
TRACE_DIR="$(mktemp -d "$(realpath "$ARTIFACT_ROOT")/.trace.XXXXXX")"
TRACE_FILE="${TRACE_DIR}/$(basename "$TRACE_ORIGINAL")"
sed 's/\r$//' "$TRACE_ORIGINAL" > "$TRACE_FILE"

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

# Facts that can only be captured at run time.
TRACE_SHA256="$(sha256sum "$TRACE_ORIGINAL" | cut -d' ' -f1)"
if GIT_STATUS="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)"; then
  if [ -n "$GIT_STATUS" ]; then GIT_DIRTY=true; else GIT_DIRTY=false; fi
else
  echo "WARN: git status failed - git_dirty unavailable" >&2
  GIT_DIRTY=unavailable
fi
python3 - "${TRACE_DIR}/trace_provenance.json" "$TRACE_FILE" "$TRACE_ORIGINAL" \
  "$TRACE_SHA256" "$MAX_CONCURRENCY" "$GIT_DIRTY" <<'PYEOF'
import json, sys
keys = ("trace_used", "trace_original_path", "trace_original_sha256",
        "max_concurrency", "git_dirty")
with open(sys.argv[1], "w") as fh:
    json.dump(dict(zip(keys, sys.argv[2:])), fh, indent=2)
PYEOF

echo "scenario=${SCENARIO} model=${MODEL} concurrency=${MAX_CONCURRENCY}"
echo "trace=${TRACE_FILE} (LF copy of ${TRACE_ORIGINAL})"
echo "artifacts=${ARTIFACT_ROOT}"
echo

python "${SCRIPT_DIR}/aiperf_trace_replay.py"
