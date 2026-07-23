# Laptop Runbook — MSI, RTX 5060 Laptop (Blackwell, 8GB), WSL2

Verified working 2026-07-23. Differs from SETUP_TEAM.md, which targets a
4GB RTX 2050 (Turing) and pins a vLLM version that CANNOT run on Blackwell.

## 0. Windows prerequisites

- Recent NVIDIA driver (Blackwell requires it)
- WSL2 + Ubuntu-24.04, Docker Desktop with WSL integration enabled
- Create `C:\Users\<you>\.wslconfig`:

```
[wsl2]
memory=12GB
swap=2GB
```

Then `wsl --shutdown` in PowerShell. **This is mandatory.** Default WSL has no
memory ceiling and will crash the whole VM mid-run. An earlier `swap=4GB`
setting left only ~5.7GB usable and caused vLLM to hang forever while loading
1.5B weights.

Verify: `free -h` must show ~11Gi total, 2.0Gi swap.

## 1. Clone (WSL home, NOT /mnt/c)

```bash
cd ~ && git clone https://github.com/faresalawneh/LLM-Graduation-Project.git
cd LLM-Graduation-Project && git checkout waqfi_experiments
```

Working from `/mnt/c/...` is 10-20x slower and causes bind-mount problems.

## 2. Verify GPU passthrough

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Must list "NVIDIA GeForce RTX 5060 Laptop GPU". Nothing else works until it does.

## 3. vLLM image — version is critical

```bash
echo "VLLM_IMAGE=vllm/vllm-openai:v0.9.2" > monitoring/.env
docker pull vllm/vllm-openai:v0.9.2
```

| Tag | Result on RTX 5060 + WSL2 |
|---|---|
| `:latest` | FAILS — `RuntimeError: UVA is not available` |
| `:v0.6.3` (SETUP_TEAM.md) | FAILS — no sm_120 Blackwell kernels |
| **`:v0.9.2`** | **WORKS** |

`:latest` uses the V1 engine, which allocates a UVA (Unified Virtual Addressing)
buffer at worker init. WSL2's paravirtualized GPU cannot provide pinned memory,
so it raises. `docker-compose.yml` also sets `VLLM_USE_V1=0` as belt-and-braces.

## 4. Line endings (if cloned via GitHub Desktop on Windows)

```bash
sed -i 's/\r$//' monitoring/scripts/*.sh monitoring/scripts/*.py
chmod +x monitoring/scripts/*.sh
```

Otherwise: `/usr/bin/env: 'bash\r': No such file or directory`.
`.gitattributes` now enforces LF, so fresh clones should be clean.

## 5. Start the stack

```bash
cd monitoring
docker compose --profile local up -d --build
docker logs -f vllm     # wait for "Application startup complete", Ctrl+C
```

Starts vllm, prometheus, otel-collector, nvml-exporter, grafana.
`--build` only builds the small NVML exporter. `dcgm-exporter` is absent by
design — DCGM needs a privileged daemon that cannot run under WSL2. The NVML
exporter emits DCGM-compatible metric names (`DCGM_FI_DEV_SM_CLOCK` etc.) with
`source="nvml"`, so the GP1 Grafana dashboards work unchanged.

## 6. Health check — all four must print

```bash
curl -sf localhost:8000/v1/models > /dev/null && echo vllm OK
curl -sf localhost:9400/metrics   > /dev/null && echo nvml OK
curl -sf localhost:9090/-/healthy > /dev/null && echo prometheus OK
curl -s 'localhost:9090/api/v1/query?query=DCGM_FI_DEV_SM_CLOCK' | grep -q '"value"' && echo chain OK
```

`chain OK` is the important one — it proves NVML -> exporter -> Prometheus works.

## 7. Python env

```bash
sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv ~/venvs/aiperf-env
source ~/venvs/aiperf-env/bin/activate
pip install "aiperf[otel]" requests
```

3.12 specifically — 3.14 has no prebuilt pyzmq wheels.

## 8. Run

```bash
source ~/venvs/aiperf-env/bin/activate
cd ~/LLM-Graduation-Project/monitoring/scripts
./run_local_replay.sh steady     # 72 requests, ~14 min
./run_local_replay.sh burst      # 9,483 requests
```

Do NOT use standalone `docker run` for vLLM (old instructions) — it collides
with the compose service on port 8000.

Steady replays 72 requests over ~14 min at original BurstGPT inter-arrival
timestamps. **Long silent gaps are normal**, not a hang. Artifacts are written
at the END of the run to `artifacts/<scenario>_<timestamp>/`.

During the run: close Chrome. Watch Task Manager memory; sustained 93%+ is the
danger zone.

## 9. Results

- Live: Grafana `localhost:3000` (admin/admin)
- Saved: `artifacts/<scenario>_<timestamp>/` —
  `profile_export_aiperf.csv` (summary), `profile_export.jsonl` (per-request),
  `gpu_telemetry_export.jsonl`, `server_metrics_export.csv`, `run_manifest.json`

## 10. Measured results (2026-07-23, 72-request steady trace)

| Metric | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---|---|
| Benchmark duration | 826.22 s | 828.29 s |
| TTFT avg | 154.28 ms | 246.07 ms |
| Request latency avg | 2,051.95 ms | 4,332.96 ms |
| Output TPS/user | 134.94 | 61.04 |
| Total token throughput | 106.13 tok/s | 103.94 tok/s |
| Total GPU power | 14.86 W | 25.02 W |
| OSL mismatch | 4/72 | 7/72 |

**Key finding:** tripling model size left wall-clock duration unchanged
(826 vs 828 s). The steady workload is arrival-rate-bound, not compute-bound —
the GPU idles between timestamped requests. Per-token speed halved and power
rose 68%, but total duration did not move. Burst is required to produce queue
depth and contention.

`gpu_utilization` reads 0.0 via NVML on this card, exactly as DCGM did on the
P40s in GP1 — **SM clock is the correct utilization proxy on both**.

## 11. Known issues

| Symptom | Cause / fix |
|---|---|
| `UVA is not available` | on `:latest` — use v0.9.2 |
| `bash\r: No such file` | CRLF — `sed -i 's/\r$//'` |
| nvml-exporter build fails, `/requirements.txt not found` | `monitoring/exporters/requirements.txt` missing (fixed in repo) |
| vLLM hangs at "Using model weights format" | WSL RAM starvation — fix `.wslconfig` |
| WSL VM unreachable, `Wsl/Service/0x8007274c` | memory exhaustion — `wsl --shutdown`, set the cap |
| `GPU scrape missing: DCGM_FI_DEV_SM_CLOCK{gpu="1"}` | cosmetic — P40 2-GPU leftover; this laptop is `gpu="0"`. Console GPU panel blank, exported telemetry unaffected |
| `Record amount must be non-negative on ... osl_mismatch_diff_pct` | model stops before trace target length. Harmless |
| docker CLI "not supported in this WSL2 distro" | you are in `docker-desktop`, not `Ubuntu-24.04` |

## 12. Config notes

`docker-compose.yml` currently: Qwen2.5-1.5B-Instruct, float16,
max-model-len 4096, gpu-memory-utilization 0.85. Needs ~7GB VRAM and the
12GB `.wslconfig`. For 4GB cards use 0.5B at 0.75.

`MAX_CONCURRENCY=20` in `run_local_replay.sh`.
