#!/usr/bin/env bash
set -euo pipefail

# One-command INFERSCOPE experiment runner.
#   ./run_experiment.sh            interactive menu
#   ./run_experiment.sh smoke      300 requests   ~30s   artifacts -> /tmp
#   ./run_experiment.sh steady      72 requests   ~15m
#   ./run_experiment.sh burst    9,483 requests   ~15m
# Options: --teardown   --no-mlflow

SCENARIO="${1:-}"
[ $# -gt 0 ] && shift

TEARDOWN=0
DO_MLFLOW=1
for arg in "$@"; do
  case "$arg" in
    --teardown)   TEARDOWN=1 ;;
    --no-mlflow)  DO_MLFLOW=0 ;;
    --keep-stack) TEARDOWN=0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

if [ -z "${SCENARIO}" ]; then
  echo
  echo "  select workload:"
  echo
  echo "    1) smoke     300 requests   ~30s    artifacts -> /tmp"
  echo "    2) steady     72 requests   ~15m    arrival-rate-bound"
  echo "    3) burst   9,483 requests   ~15m    bursty arrival"
  echo "    q) quit"
  echo
  read -rp "  > " choice
  case "${choice}" in
    1|smoke)  SCENARIO="smoke"  ;;
    2|steady) SCENARIO="steady" ;;
    3|burst)  SCENARIO="burst"  ;;
    q|Q)      echo "cancelled"; exit 0 ;;
    *) echo "invalid choice: ${choice}" >&2; exit 1 ;;
  esac
  echo
fi

case "${SCENARIO}" in
  steady|burst|smoke) ;;
  *) echo "usage: $0 [steady|burst|smoke] [--teardown] [--no-mlflow]" >&2; exit 1 ;;
esac

MON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MON_DIR}/.." && pwd)"
SCRIPTS="${MON_DIR}/scripts"
VENV="${AIPERF_VENV:-$HOME/venvs/aiperf-env}"

hr()   { printf '%s\n' "------------------------------------------------------------"; }
step() { hr; printf '  %s\n' "$1"; hr; }

if ! command -v aiperf >/dev/null 2>&1; then
  if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
  else
    echo "aiperf not found and no venv at ${VENV}" >&2
    exit 1
  fi
fi
echo "aiperf $(aiperf --version 2>/dev/null || echo '?')  |  venv: ${VIRTUAL_ENV:-system}"

if curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1; then
  step "stack already up - skipping startup"
else
  step "starting stack"
  ( cd "${MON_DIR}" && ./stack_up.sh )
fi

BASELINE="$(docker inspect vllm --format '{{.RestartCount}}' 2>/dev/null || echo 0)"
# Refresh the baseline the MLflow logger subtracts from. stack_up.sh writes this
# only when the stack is (re)started, so across several runs on a live stack it
# goes stale and vllm_restarts_during_run reports the cumulative count instead of
# the per-run delta.
STATE_FILE="${MON_DIR}/logs/stack_state.json"
if [ -f "${STATE_FILE}" ]; then
  python3 - "${STATE_FILE}" "${BASELINE}" <<'PYEOF'
import json, sys
path, baseline = sys.argv[1], int(sys.argv[2])
with open(path) as fh:
    state = json.load(fh)
state["restart_count_at_start"] = baseline
with open(path, "w") as fh:
    json.dump(state, fh, indent=2)
PYEOF
  echo "restart baseline for this run: ${BASELINE}"
fi

if [ "${SCENARIO}" = "smoke" ]; then
  SMOKE_TRACE="/tmp/burst_smoke.csv"
  head -301 "${SCRIPTS}/burst_window.csv" > "${SMOKE_TRACE}"
  export TRACE_FILE="${SMOKE_TRACE}"
  export ARTIFACT_ROOT="/tmp/smoke_artifacts"
  REPLAY_ARG="burst"
  PREFIX="burst"
else
  export ARTIFACT_ROOT="${REPO_ROOT}/artifacts"
  REPLAY_ARG="${SCENARIO}"
  PREFIX="${SCENARIO}"
fi

step "running: ${SCENARIO}"
START_EPOCH="$(date +%s)"
( cd "${SCRIPTS}" && ./run_local_replay.sh "${REPLAY_ARG}" )
ELAPSED=$(( $(date +%s) - START_EPOCH ))

RUN_DIR="$(ls -dt "${ARTIFACT_ROOT}"/${PREFIX}_*/ 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "no artifact directory found under ${ARTIFACT_ROOT}" >&2
  exit 1
fi
echo "artifacts: ${RUN_DIR}"

if [ "${DO_MLFLOW}" -eq 1 ]; then
  step "logging to MLflow"
  ( cd "${REPO_ROOT}" && python "${SCRIPTS}/mlflow_log_run.py" "${RUN_DIR}" )
fi

step "crash check"
NOW="$(docker inspect vllm --format '{{.RestartCount}}' 2>/dev/null || echo 0)"
OOM="$(docker inspect vllm --format '{{.State.OOMKilled}}' 2>/dev/null || echo unknown)"
CRASHES=$(( NOW - BASELINE ))
if [ "${CRASHES}" -gt 0 ]; then
  echo "*** vLLM restarted ${CRASHES} time(s) DURING this run - results suspect ***"
  echo "*** see ${MON_DIR}/logs/vllm_events.jsonl ***"
else
  echo "no restarts during run (count=${NOW}, oom=${OOM})"
fi

step "summary"
printf '  scenario : %s\n' "${SCENARIO}"
printf '  wall     : %dm %ds\n' $((ELAPSED / 60)) $((ELAPSED % 60))
printf '  artifacts: %s\n' "${RUN_DIR}"

CSV="${RUN_DIR}profile_export_aiperf.csv"
if [ -f "${CSV}" ]; then
  awk -F',' 'NR>1 && ($1 ~ /Time to First Token|Inter Token Latency|Request Latency|Request Count|Error/) {
    printf "  %-28s %s\n", $1, $2
  }' "${CSV}" 2>/dev/null || true
fi

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  vram     : /' || true

if [ "${TEARDOWN}" -eq 1 ]; then
  step "teardown"
  ( cd "${MON_DIR}" && docker compose --profile local down )
  pkill -f "docker events --filter container=vllm" 2>/dev/null || true
else
  echo
  echo "stack left running - next run skips the ~95s model load."
  echo "to stop:  cd ${MON_DIR} && docker compose --profile local down"
fi

echo
echo "view results:  ${MON_DIR}/mlflow_ui.sh"
