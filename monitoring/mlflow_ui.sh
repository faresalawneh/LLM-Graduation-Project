#!/usr/bin/env bash
set -euo pipefail

# Launch the MLflow UI against the project's SQLite store.
#   ./mlflow_ui.sh          -> http://localhost:5000
#   ./mlflow_ui.sh 5001     -> alternate port
# In the browser: click "Model training" (top-left), then inferscope-gp2.

PORT="${1:-5000}"
MON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MON_DIR}/.." && pwd)"
VENV="${AIPERF_VENV:-$HOME/venvs/aiperf-env}"

if ! command -v mlflow >/dev/null 2>&1; then
  [ -f "${VENV}/bin/activate" ] && source "${VENV}/bin/activate"
fi
command -v mlflow >/dev/null 2>&1 || {
  echo "mlflow not installed:" >&2
  echo "  source ${VENV}/bin/activate" >&2
  echo "  pip install -r ${REPO_ROOT}/requirements-experiments.txt" >&2
  exit 1
}

DB="${REPO_ROOT}/mlruns/mlflow.db"
[ -f "${DB}" ] || { echo "no MLflow store at ${DB} - run an experiment first" >&2; exit 1; }

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "port ${PORT} in use - UI may already be running: http://localhost:${PORT}"
  echo "or: $0 5001"
  exit 1
fi

echo "MLflow UI -> http://localhost:${PORT}"
echo "  toggle 'Model training' (not GenAI), experiment: inferscope-gp2"
echo "  Ctrl+C to stop"
echo
exec mlflow ui --backend-store-uri "sqlite:///${DB}" --port "${PORT}"
