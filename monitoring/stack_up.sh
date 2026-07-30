#!/usr/bin/env bash
set -euo pipefail

# Bring up the local INFERSCOPE stack (vLLM + monitoring) and instrument it.
# Usage: ./stack_up.sh
# Then:  monitoring/scripts/run_local_replay.sh [steady|burst]

MON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${MON_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${MON_DIR}"

command -v docker >/dev/null \
  || { echo "docker unreachable - enable WSL integration in Docker Desktop" >&2; exit 1; }

# Guard: compose defaults to :latest, which dies on Blackwell (UVA is not available)
IMG="$(grep -E '^VLLM_IMAGE=' .env 2>/dev/null | cut -d= -f2- || true)"
case "${IMG}" in
  ""|*":latest") echo "VLLM_IMAGE must be pinned (not :latest) in ${MON_DIR}/.env" >&2; exit 1 ;;
esac
echo "vllm image: ${IMG}"

echo "starting stack (profile: local)..."
docker compose --profile local up -d

# Crash instrumentation: vLLM has restart:unless-stopped, so a die/restart is
# otherwise invisible. Capture exit codes live.
EVENT_LOG="${LOG_DIR}/vllm_events.jsonl"
pkill -f "docker events --filter container=vllm" 2>/dev/null || true
nohup docker events --filter container=vllm \
      --filter event=die --filter event=oom --filter event=restart \
      --format '{{json .}}' >> "${EVENT_LOG}" 2>/dev/null &
echo "event watcher -> ${EVENT_LOG}"

# Health wait (compose healthcheck has start_period 300s)
WAIT="${VLLM_WAIT:-420}"; elapsed=0
until curl -sf "http://localhost:8000/v1/models" >/dev/null 2>&1; do
  if [ "${elapsed}" -ge "${WAIT}" ]; then
    echo "vLLM not ready after ${WAIT}s" >&2
    docker compose --profile local logs --tail 40 vllm >&2
    exit 1
  fi
  printf '\rloading model... %ss' "${elapsed}"
  sleep 5; elapsed=$((elapsed + 5))
done
[ "${elapsed}" -gt 0 ] && echo

# Provenance for run_manifest / MLflow
DIGEST="$(docker inspect "${IMG}" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "unknown")"
BASELINE="$(docker inspect vllm --format '{{.RestartCount}}' 2>/dev/null || echo 0)"
cat > "${LOG_DIR}/stack_state.json" <<EOF
{
  "vllm_image": "${IMG}",
  "vllm_digest": "${DIGEST}",
  "restart_count_at_start": ${BASELINE},
  "stack_up_time": "$(date -Iseconds)"
}
EOF

echo
docker compose --profile local ps --format 'table {{.Name}}\t{{.Status}}'
echo
echo "digest: ${DIGEST}"
echo "restart baseline: ${BASELINE}   (compare after run to detect crashes)"
echo "ready -> monitoring/scripts/run_local_replay.sh [steady|burst]"
