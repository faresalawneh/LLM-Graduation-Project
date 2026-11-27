#!/usr/bin/env bash
set -euo pipefail

# Bring up the local INFERSCOPE stack (vLLM + monitoring) and instrument it.
# Usage: ./stack_up.sh                  (dev: builds nvml-exporter locally)
#        STACK_ENV=demo ./stack_up.sh   (demo: published images, see docker-compose.demo.yml)
# INFERSCOPE_COMPOSE_OVERRIDE=<file> (dev only; set by scripts/start_run.py)
# layers the local version lock on top of docker-compose.yml.
# Then:  monitoring/scripts/run_local_replay.sh [steady|burst]

MON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${MON_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${MON_DIR}"

STACK_ENV="${STACK_ENV:-dev}"
COMPOSE=(docker compose)
case "${STACK_ENV}" in
  dev) ;;
  demo) COMPOSE+=(-f docker-compose.yml -f docker-compose.demo.yml) ;;
  *) echo "STACK_ENV must be dev or demo (got: ${STACK_ENV})" >&2; exit 1 ;;
esac

OVERRIDE="${INFERSCOPE_COMPOSE_OVERRIDE:-}"
if [ -n "${OVERRIDE}" ]; then
  [ "${STACK_ENV}" = "dev" ] \
    || { echo "the local override is only supported with STACK_ENV=dev" >&2; exit 1; }
  [ -f "${OVERRIDE}" ] || { echo "override not found: ${OVERRIDE}" >&2; exit 1; }
  COMPOSE+=(-f docker-compose.yml -f "${OVERRIDE}")
  echo "version lock: ${OVERRIDE}"
else
  echo "WARNING: no local override - the vLLM and model version is NOT locked." >&2
  echo "WARNING: use monitoring/scripts/start_run.py to start a locked stack." >&2
fi

command -v docker >/dev/null \
  || { echo "docker unreachable - enable WSL integration in Docker Desktop" >&2; exit 1; }

# Guard: compose defaults to :latest, which dies on Blackwell (UVA is not available)
IMG="$(grep -E '^VLLM_IMAGE=' .env 2>/dev/null | cut -d= -f2- || true)"
case "${IMG}" in
  ""|*":latest") echo "VLLM_IMAGE must be pinned (not :latest) in ${MON_DIR}/.env" >&2; exit 1 ;;
esac
echo "vllm image: ${IMG}"

# Guard: dev and demo are separate compose projects (own volumes and network)
# but share fixed container names. If the other environment's containers still
# exist, stop here instead of failing halfway through "up" on a name conflict.
PROJECT="$("${COMPOSE[@]}" config 2>/dev/null | sed -n 's/^name: //p' | head -n1 || true)"
if [ -n "${PROJECT}" ]; then
  for c in prometheus otel-collector nvml-exporter grafana vllm; do
    OWNER="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.project"}}' "${c}" 2>/dev/null || true)"
    if [ -n "${OWNER}" ] && [ "${OWNER}" != "${PROJECT}" ]; then
      echo "container '${c}' belongs to compose project '${OWNER}', not '${PROJECT}' (env: ${STACK_ENV})." >&2
      echo "Bring the other environment down first (volumes are kept):" >&2
      echo "  docker compose -p ${OWNER} down" >&2
      exit 1
    fi
  done
fi

echo "starting stack (env: ${STACK_ENV}, project: ${PROJECT:-unknown}, profile: local)..."
"${COMPOSE[@]}" --profile local up -d

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
    "${COMPOSE[@]}" --profile local logs --tail 40 vllm >&2
    exit 1
  fi
  printf '\rloading model... %ss' "${elapsed}"
  sleep 5; elapsed=$((elapsed + 5))
done
[ "${elapsed}" -gt 0 ] && echo

# Provenance for run_manifest / MLflow
DIGEST="$(docker inspect "${IMG}" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "unknown")"
BASELINE="$(docker inspect vllm --format '{{.RestartCount}}' 2>/dev/null || echo 0)"
# What the running vllm container was actually started with.
CMD_LINES="$(docker inspect vllm --format '{{range .Config.Cmd}}{{println .}}{{end}}' 2>/dev/null || true)"
ENV_LINES="$(docker inspect vllm --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null || true)"
revision_arg() {
  local value
  value="$(printf '%s\n' "${CMD_LINES}" | awk -v flag="$1" 'prev == flag { v = $0 } { prev = $0 } END { print v }')"
  if [[ "${value}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "${value}"
  else
    echo "WARN: $1 unavailable (not set to a commit id on the vllm container)" >&2
    echo "unavailable"
  fi
}
MODEL_REVISION="$(revision_arg --revision)"
TOKENIZER_REVISION="$(revision_arg --tokenizer-revision)"
HF_OFFLINE="$(printf '%s\n' "${ENV_LINES}" | sed -n 's/^HF_HUB_OFFLINE=//p')"
if [ -z "${HF_OFFLINE}" ]; then
  echo "WARN: HF_HUB_OFFLINE unavailable on the vllm container" >&2
  HF_OFFLINE="unavailable"
fi
if [ -n "${OVERRIDE}" ]; then
  OVERRIDE_ACTIVE=true
  OVERRIDE_PATH="$(realpath "${OVERRIDE}")"
  OVERRIDE_SHA256="$(sha256sum "${OVERRIDE}" | cut -d' ' -f1)"
else
  OVERRIDE_ACTIVE=false
  OVERRIDE_PATH="unavailable"
  OVERRIDE_SHA256="unavailable"
fi
cat > "${LOG_DIR}/stack_state.json" <<EOF
{
  "stack_env": "${STACK_ENV}",
  "vllm_image": "${IMG}",
  "vllm_digest": "${DIGEST}",
  "override_active": "${OVERRIDE_ACTIVE}",
  "override_path": "${OVERRIDE_PATH}",
  "override_sha256": "${OVERRIDE_SHA256}",
  "model_revision": "${MODEL_REVISION}",
  "tokenizer_revision": "${TOKENIZER_REVISION}",
  "hf_hub_offline": "${HF_OFFLINE}",
  "restart_count_at_start": ${BASELINE},
  "stack_up_time": "$(date -Iseconds)"
}
EOF

echo
"${COMPOSE[@]}" --profile local ps --format 'table {{.Name}}\t{{.Status}}'
echo
echo "digest: ${DIGEST}"
echo "restart baseline: ${BASELINE}   (compare after run to detect crashes)"
echo "ready -> monitoring/scripts/run_local_replay.sh [steady|burst]"
