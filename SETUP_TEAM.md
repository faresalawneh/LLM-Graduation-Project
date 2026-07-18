# Team Setup - Fresh Clone to Grafana in ~30 min
# For Mahmoud / Abdelrahman - Windows laptop with NVIDIA GPU

## Prerequisites (one-time, Windows side)

1. **NVIDIA driver** - normal Game Ready / Studio driver, recent version. Check: `nvidia-smi` works in PowerShell.
2. **WSL2 + Ubuntu** - PowerShell (admin): `wsl --install`, reboot, create Linux user.
3. **Docker Desktop** - install, then Settings > Resources > WSL Integration > enable for your Ubuntu distro. Confirm GPU works:
   `docker run --rm --gpus all nvcr.io/nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
   (if this fails, nothing else will work - fix this first)

## Clone and set up (WSL terminal)

```bash
cd ~
git clone https://github.com/faresalawneh/LLM-Graduation-Project.git
cd LLM-Graduation-Project

# Python 3.12 venv - NOT the system python3 (Ubuntu 26.04 ships 3.14 which breaks things)
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv aiperf-env
source aiperf-env/bin/activate
pip install -r requirements-local.txt

# runner script permissions + line endings (git on Windows may add CRLF)
chmod +x monitoring/scripts/run_local_replay.sh
sed -i 's/\r$//' monitoring/scripts/run_local_replay.sh
```

## Pin the vLLM image (important)

`latest` vLLM does NOT work under WSL2 (fails with "UVA is not available").

```bash
echo "VLLM_IMAGE=vllm/vllm-openai:v0.6.3" >> monitoring/.env
```

## Start the stack

```bash
cd monitoring
docker compose --profile local up -d
```

First start downloads the vLLM image (~5 GB) and Qwen2.5-0.5B weights - be patient.
Watch: `docker logs -f vllm` until the server startup completes. Ctrl-C detaches safely.

## Health check - all four must print OK

```bash
curl -sf localhost:8000/v1/models > /dev/null && echo vllm OK
curl -sf localhost:9400/metrics   > /dev/null && echo nvml OK
curl -sf localhost:9090/-/healthy > /dev/null && echo prometheus OK
curl -s 'localhost:9090/api/v1/query?query=DCGM_FI_DEV_SM_CLOCK' | grep -q '"value"' && echo chain OK
```

## Run a benchmark

```bash
cd ~/LLM-Graduation-Project
source aiperf-env/bin/activate
./monitoring/scripts/run_local_replay.sh steady
```

Open http://localhost:3000 (admin/admin) - both dashboards. Steady = 72 requests
over 15 min at real BurstGPT inter-arrival times; long quiet gaps are normal.
Artifacts land in `artifacts/<scenario>_<date>_<time>/` - `profile_export.jsonl`
is written at the END of the run, not live.

Burst (`./monitoring/scripts/run_local_replay.sh burst`) only after steady works.
9,483 requests - keep MAX_CONCURRENCY=20 on cards with 4-8 GB VRAM.

## Stop everything

```bash
cd monitoring && docker compose --profile local down
```

## GPU differences per machine

The runner defaults fit a 4 GB card (RTX 2050: Qwen2.5-0.5B, gpu-memory-utilization 0.75, concurrency 20).
On an 8 GB card (RTX 5060) you can raise to Qwen2.5-1.5B-Instruct and utilization 0.85 -
edit the vllm service command in docker-compose.yml and MODEL in run_local_replay.sh together.

## If something breaks

| Symptom | Fix |
|---|---|
| `bash\r: No such file or directory` | `sed -i 's/\r$//' <script>` - Windows line endings |
| `command not found` running the script | prefix with `./` |
| vLLM crash-loops, "UVA is not available" | you're on `latest` - pin v0.6.3 (see above) |
| vLLM OOM / engine dies | another app holds VRAM - close browsers, check ollama isn't running: `docker stop ollama` |
| Panels empty but services up | check http://localhost:9090/targets - every target UP except dcgm-exporter (server-only, DOWN is normal) |
| `ModuleNotFoundError` | wrong venv - `source ~/LLM-Graduation-Project/aiperf-env/bin/activate` |
| Port already in use | old containers from manual runs: `docker ps -a`, remove strays |
