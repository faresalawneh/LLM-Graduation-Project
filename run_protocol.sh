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

# step | scenario | fault_spec
#   0          = clean control
#   <seconds>  = F1, kill the engine at that offset into the replay
#   f2@<secs>  = F2a, pause nvml-exporter for F2_PAUSE_S at that offset
#
# The 200s restriction from fe65b5c is lifted. That workaround existed because
# AIPerf's finalisation hangs after a mid-replay kill; ef8e1e7 recovers
# aggregates from profile_export.jsonl (validated 36/36 against native output)
# and 73ec141 bounds and reaps the hang. Kill-time spread is now data: corrupt
# in-flight record counts were 0 at 400s, 6 and 54 at other offsets.
#
# F2a pauses the out-of-band collector, not AIPerf's stream: serving stays
# healthy and the run completes, so §V-E's preconditions are evaluable.
# See docs/F2_INJECTION_DECISION.md.
PLAN=(
  "1|burst|400"
  "2|burst|0"
  "3|burst|200"
  "4|burst|0"
  "5|burst|600"
  "6|burst|f2@450"
  "7|burst|0"
  "8|burst|400"
  "9|burst|f2@450"
)

mkdir -p "${PROTO_DIR}"
# never leave the collector paused if the runner dies mid-injection
trap 'docker unpause nvml-exporter >/dev/null 2>&1 || true' EXIT
hr()   { printf '%s\n' "============================================================"; }
say()  { printf '[protocol] %s\n' "$*"; }
gputemp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1; }
gpufb()   { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

# Clean burst replays of an identical trace separate into two mean-ITL clusters
# (29.6 vs 36.0 ms) tracking a ~360 MiB difference in device memory occupancy
# that is present before the run starts. Controls and fault runs landing in
# different clusters would confound every latency comparison, so FB_USED is
# recorded per run and a mid-sequence cluster change is flagged loudly.
FB_CLUSTER_BOUNDARY_MIB="${FB_CLUSTER_BOUNDARY_MIB:-6600}"
CLUSTER_FILE="${PROTO_DIR}/session_cluster"

fb_cluster() {
  local fb="$1"
  [ "${fb}" -ge "${FB_CLUSTER_BOUNDARY_MIB}" ] && echo "slow" || echo "fast"
}

check_cluster() {
  local fb cluster prev
  fb="$(gpufb)"; cluster="$(fb_cluster "${fb}")"
  prev="$(cat "${CLUSTER_FILE}" 2>/dev/null || echo "")"
  if [ -z "${prev}" ]; then
    echo "${cluster}" > "${CLUSTER_FILE}"
    say "session cluster: ${cluster} (FB_USED ${fb} MiB, boundary ${FB_CLUSTER_BOUNDARY_MIB})"
  elif [ "${cluster}" != "${prev}" ]; then
    say "*** CLUSTER CHANGE: ${prev} -> ${cluster} (FB_USED ${fb} MiB) ***"
    say "*** latency is NOT comparable across the boundary - FLAG THIS RUN ***"
  fi
  echo "${fb}"
}

# AIPerf exits 0 but leaves its ~20 worker processes resident. They must be
# reaped or the next run contends with them and blocks.
reap_aiperf() {
  local n
  n="$(pgrep -c aiperf 2>/dev/null || echo 0)"
  [ "${n}" -eq 0 ] && return 0
  say "reaping ${n} orphaned aiperf process(es)"
  pkill -TERM -f "aiperf " 2>/dev/null || true
  sleep 5
  pkill -9 -f "aiperf " 2>/dev/null || true
  sleep 3
  say "aiperf processes remaining: $(pgrep -c aiperf 2>/dev/null || echo 0)"
}

print_plan() {
  printf '%-5s %-8s %-12s\n' step scenario fault
  for row in "${PLAN[@]}"; do
    IFS='|' read -r n s k <<<"${row}"
    case "${k}" in
      0)    f="none" ;;
      f2@*) f="f2a pause @ ${k#f2@}s" ;;
      *)    f="kill @ ${k}s" ;;
    esac
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

# Pause the out-of-band collector for F2_PAUSE_S, N seconds into the replay.
# Serving and AIPerf are untouched: the run completes and the GPU field series
# shows a gap, so the §V-E preconditions stay evaluable.
F2_PAUSE_S="${F2_PAUSE_S:-60}"
schedule_f2() {
  local scenario="$1" at="$2" step="$3" before dir
  before="$(ls -d "${ART_ROOT}/${scenario}"_*/ 2>/dev/null | wc -l)"
  (
    for _ in $(seq 1 1800); do
      if [ "$(ls -d "${ART_ROOT}/${scenario}"_*/ 2>/dev/null | wc -l)" -gt "${before}" ]; then
        dir="$(ls -dt "${ART_ROOT}/${scenario}"_*/ | head -1)"
        sleep "${at}"
        pause_epoch="$(date +%s)"
        docker pause nvml-exporter >/dev/null 2>&1 || true
        sleep "${F2_PAUSE_S}"
        docker unpause nvml-exporter >/dev/null 2>&1 || true
        printf '{"step":%s,"fault":"f2a","pause_at_s":%s,"pause_s":%s,"pause_epoch":%s,"run_dir":"%s"}\n' \
          "${step}" "${at}" "${F2_PAUSE_S}" "${pause_epoch}" "${dir}" \
          | tee -a "${PROTO_DIR}/fault_injections.jsonl" > "${dir}/fault_injection.json"
        exit 0
      fi
      sleep 1
    done
  ) >/dev/null 2>&1 &
  echo $!
}

run_step() {
  local step="$1" scenario="$2" spec="$3"
  local tag killer smi_pid log_pid ev_pid rundir kill_at f2_at fb
  kill_at="0"; f2_at=""
  case "${spec}" in
    0)      tag="clean" ;;
    f2@*)   f2_at="${spec#f2@}"; tag="f2a${f2_at}" ;;
    *)      kill_at="${spec}"; tag="kill${kill_at}" ;;
  esac
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
  fb="$(check_cluster)"
  echo "${fb}" > "${pre}/fb_used_before_mib"
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
  [ -n "${f2_at}" ]       && killer="$(schedule_f2   "${scenario}" "${f2_at}"  "${step}")"

  set +e
  timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT_S:-2400}" \
    bash -c 'cd "$1" && ./run_experiment.sh "$2"' _ "${MON_DIR}" "${scenario}" \
    2>&1 | tee "${pre}/runner.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    say "STEP ${step} TIMED OUT after ${RUN_TIMEOUT_S:-2400}s - killing stragglers"
    pkill -9 -f "aiperf " 2>/dev/null || true
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

  # A mid-replay kill can leave profile_export.jsonl complete but AIPerf's
  # finalisation never finished. Recompute from the per-request records
  # (validated 36/36 against native aggregates on clean runs).
  if [ ! -f "${rundir}profile_export_aiperf.json" ] && [ -f "${rundir}profile_export.jsonl" ]; then
    say "no native aggregates - recovering from per-request records"
    python3 "${REPO_ROOT}/monitoring/scripts/recover_aggregates.py" "${rundir}" \
      || say "WARNING: aggregate recovery failed"
  fi
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
