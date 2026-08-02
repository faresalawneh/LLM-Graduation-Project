#!/usr/bin/env bash
# INFERSCOPE cohort-B protocol runner.
#
# Executes the 12-run counterbalanced sequence for the paper. Wraps
# monitoring/run_experiment.sh without modifying it, so every run stays
# comparable with burst_2026-07-30_0253.
#
#   ./run_protocol.sh warmup        discarded warm-up replay, never logged
#   ./run_protocol.sh plan          print the sequence and exit
#   ./run_protocol.sh 1             run step 1 only
#   ./run_protocol.sh 1 12          run steps 1..12 with cooldown gates
#
# Never use --teardown. The container must survive all 12 runs or the
# RestartCount evidence resets.
set -euo pipefail

MON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/monitoring" && pwd)"
REPO_ROOT="$(cd "${MON_DIR}/.." && pwd)"
ART_ROOT="${REPO_ROOT}/artifacts"
PROTO_DIR="${REPO_ROOT}/protocol_logs"
COHORT="B"
COOLDOWN_MIN="${COOLDOWN_MIN:-20}"
TEMP_TOLERANCE_C="${TEMP_TOLERANCE_C:-3}"
BASELINE_FILE="${PROTO_DIR}/baseline_temp_c"

# step | scenario | kill_time_s (0 = no fault)
#
# Kill times are 400s and 600s only. Earlier injections (200s into a ~900s
# replay) leave more in-flight records partially written in AIPerf's
# memory-mapped dataset; the client then cannot deserialise them, never
# reaches its expected record count, and waits indefinitely. Injections at
# 200s hung every time we tried them. Even at 400s only one attempt in
# several completed.
PLAN=(
  "1|steady|0"
  "2|burst|400"
  "3|burst|0"
  "4|burst|400"
  "5|steady|0"
  "6|burst|600"
  "7|burst|0"
  "8|burst|400"
  "9|steady|0"
  "10|burst|400"
  "11|burst|0"
  "12|burst|600"
)

mkdir -p "${PROTO_DIR}"
hr()   { printf '%s\n' "============================================================"; }
say()  { printf '[protocol] %s\n' "$*"; }
gputemp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }

# AIPerf exits 0 but leaves its ~20 worker processes resident. They must be
# reaped or the next run contends with them and blocks.
reap_aiperf() {
  local n
  n="$(pgrep -c aiperf 2>/dev/null || echo 0)"
  [ "${n}" -eq 0 ] && return 0
  say "reaping ${n} orphaned aiperf process(es)"
  pkill -TERM aiperf 2>/dev/null || true
  sleep 5
  pkill -9 aiperf 2>/dev/null || true
  sleep 3
  say "aiperf processes remaining: $(pgrep -c aiperf 2>/dev/null || echo 0)"
}

print_plan() {
  printf '%-5s %-8s %-12s\n' step scenario fault
  for row in "${PLAN[@]}"; do
    IFS='|' read -r n s k <<<"${row}"
    [ "${k}" = "0" ] && f="none" || f="kill @ ${k}s"
    printf '%-5s %-8s %-12s\n' "${n}" "${s}" "${f}"
  done
}

# Wait for the GPU to return to within TEMP_TOLERANCE_C of the session baseline,
# with a hard minimum wait of COOLDOWN_MIN. Gates on temperature, not the clock.
cooldown() {
  local base target waited t
  base="$(cat "${BASELINE_FILE}" 2>/dev/null || echo "")"
  [ -z "${base}" ] && { say "no baseline recorded, skipping gate"; return 0; }
  target=$(( base + TEMP_TOLERANCE_C ))
  say "cooldown: sleeping ${COOLDOWN_MIN}m, then gating until GPU <= ${target}C"
  sleep $(( COOLDOWN_MIN * 60 ))
  waited=0
  while :; do
    t="$(gputemp)"
    [ "${t}" -le "${target}" ] && { say "GPU at ${t}C (target ${target}C) - proceeding"; return 0; }
    [ "${waited}" -ge 1800 ] && { say "WARNING: GPU still ${t}C after 30m extra wait - proceeding, FLAG THIS RUN"; return 0; }
    say "GPU at ${t}C, waiting for ${target}C ..."
    sleep 60
    waited=$(( waited + 60 ))
  done
}

# Schedule the fault injection N seconds after the replay actually starts.
# Start is detected by the appearance of a new artifact directory, so the
# offset is measured from the replay, not from stack startup.
schedule_kill() {
  local scenario="$1" kill_at="$2" step="$3" before dir
  before="$(ls -d "${ART_ROOT}/${scenario}"_*/ 2>/dev/null | wc -l)"
  (
    for _ in $(seq 1 1800); do
      if [ "$(ls -d "${ART_ROOT}/${scenario}"_*/ 2>/dev/null | wc -l)" -gt "${before}" ]; then
        dir="$(ls -dt "${ART_ROOT}/${scenario}"_*/ | head -1)"
        say "replay started (${dir}); killing vllm in ${kill_at}s"
        sleep "${kill_at}"
        kill_epoch="$(date +%s)"
        docker exec vllm pkill -9 -f multiprocessing-fork >/dev/null 2>&1 || true
        printf '{"step":%s,"kill_time_s":%s,"kill_epoch":%s,"run_dir":"%s"}\n' \
          "${step}" "${kill_at}" "${kill_epoch}" "${dir}" \
          | tee -a "${PROTO_DIR}/fault_injections.jsonl" > "${dir}/fault_injection.json"
        say "vllm killed at epoch ${kill_epoch}"
        exit 0
      fi
      sleep 1
    done
    say "ERROR: replay never started, kill not injected"
  ) >/dev/null 2>&1 &
  echo $!
}

run_step() {
  local step="$1" scenario="$2" kill_at="$3"
  local tag killer smi_pid log_pid ev_pid rundir
  [ "${kill_at}" = "0" ] && tag="clean" || tag="kill${kill_at}"
  hr; say "STEP ${step}: ${scenario} / ${tag}  (cohort ${COHORT})"; hr

  local pre="${PROTO_DIR}/step${step}_${scenario}_${tag}"
  mkdir -p "${pre}"

  reap_aiperf
  free -m | awk '/Mem:/{printf "[protocol] free before run: %d MiB avail\n", $7}'
  if ! timeout 20 curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
    say "vLLM not responding - restarting"
    timeout 120 bash -c 'cd "$1" && docker compose --profile local up -d vllm' _ "${MON_DIR}" || true
    for _ in $(seq 1 60); do
      timeout 10 curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && break
      sleep 5
    done
  fi
  gputemp > "${pre}/temp_before_c"
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw \
             --format=csv > "${pre}/smi_before.csv"

  # per-run instrumentation the July runs lacked
  ( while :; do
      printf '%s,' "$(date +%s)"
      nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,memory.used \
                 --format=csv,noheader,nounits
      sleep 5
    done ) > "${pre}/smi_series.csv" 2>/dev/null & smi_pid=$!
  # No restart loop: "docker logs -f --since 0s" re-streams the entire container
  # log on each iteration, which after a fault restart becomes an unbounded write
  # loop that starves AIPerf's record processing.
  log_pid=""  # live log following corrupts AIPerf's mmap dataset; captured after the run instead
  docker events --filter container=vllm --filter event=die --filter event=restart \
                --filter event=start --format '{{json .}}' > "${pre}/docker_events.jsonl" 2>/dev/null & ev_pid=$!

  killer=""
  [ "${kill_at}" != "0" ] && killer="$(schedule_kill "${scenario}" "${kill_at}" "${step}")"

  set +e
  timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT_S:-2400}" \
    bash -c 'cd "$1" && ./run_experiment.sh "$2"' _ "${MON_DIR}" "${scenario}" \
    2>&1 | tee "${pre}/runner.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    say "STEP ${step} TIMED OUT after ${RUN_TIMEOUT_S:-2400}s - killing stragglers"
    pkill -f aiperf 2>/dev/null || true
    sleep 10
  fi
  echo "${rc}" > "${pre}/exit_code"

  [ -n "${killer}" ] && { kill "${killer}" 2>/dev/null || true; }
  kill "${smi_pid}" "${log_pid}" "${ev_pid}" 2>/dev/null || true
  wait "${smi_pid}" "${log_pid}" "${ev_pid}" 2>/dev/null || true

  kill "${smi_pid}" "${log_pid}" "${ev_pid}" 2>/dev/null || true
  docker logs --tail 400 vllm > "${pre}/vllm_startup_after.log" 2>&1 || true
  reap_aiperf
  gputemp > "${pre}/temp_after_c"
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw \
             --format=csv > "${pre}/smi_after.csv"

  rundir="$(ls -dt "${ART_ROOT}/${scenario}"_*/ | head -1)"
  cp -r "${pre}" "${rundir}protocol/" 2>/dev/null || true

  python3 "${REPO_ROOT}/tag_protocol_run.py" \
    --run-dir "${rundir}" --step "${step}" --cohort "${COHORT}" \
    --fault "${tag}" --kill-time "${kill_at}" || say "WARNING: MLflow tagging failed"

  say "STEP ${step} done (rc=${rc}) -> ${rundir}"
  return 0
}

case "${1:-}" in
  plan) print_plan; exit 0 ;;
  warmup)
    say "WARM-UP - results discarded, not logged to MLflow"
    ( cd "${MON_DIR}" && ./run_experiment.sh burst --no-mlflow )
    gputemp > "${BASELINE_FILE}"
    say "session baseline GPU temp recorded: $(cat "${BASELINE_FILE}")C"
    say "wait ${COOLDOWN_MIN}m, then: ./run_protocol.sh 1"
    exit 0 ;;
esac

FROM="${1:?usage: $0 [plan|warmup|<from> [to]]}"
TO="${2:-$FROM}"
for row in "${PLAN[@]}"; do
  IFS='|' read -r n s k <<<"${row}"
  [ "${n}" -lt "${FROM}" ] && continue
  [ "${n}" -gt "${TO}" ] && continue
  [ "${n}" -gt "${FROM}" ] && cooldown
  run_step "${n}" "${s}" "${k}"
done
hr; say "steps ${FROM}..${TO} complete"
