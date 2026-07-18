1. Windows: install WSL2 (Ubuntu), Docker Desktop with WSL integration,
   latest NVIDIA driver (must be recent for Blackwell anyway)

2. WSL:
   git clone github.com/faresalawneh/LLM-Graduation-Project
   git checkout claude/aiperf-trace-replay-7xibz1   # or main after PR merge

   # NVIDIA container toolkit (RUN_LOCAL_FULL.md bottom section)
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi  # verify

   # python env (3.12, NOT 3.14 - pyzmq wheels)
   sudo apt install python3.12 python3.12-venv -y
   python3.12 -m venv ~/venvs/aiperf-env
   source ~/venvs/aiperf-env/bin/activate
   pip install "aiperf[otel]" requests

3. vLLM:
   docker run --gpus all -d --name vllm-local \
     -v ~/hf_cache:/root/.cache/huggingface \
     -p 8000:8000 \
     vllm/vllm-openai:latest \
     --model Qwen/Qwen2.5-1.5B-Instruct \
     --dtype float16 --max-model-len 4096 \
     --gpu-memory-utilization 0.90
   # add --enforce-eager only if it errors without it
   docker logs -f vllm-local   # wait for startup complete
   # IMPORTANT: curl :8000/metrics | grep vllm: — newer vLLM may rename
   # metrics (e.g. gpu_cache_usage_perc → kv_cache_usage_perc); if renamed,
   # Grafana vLLM panels need query updates

4. Monitoring stack:
   cd monitoring && docker compose up -d
   # prometheus, grafana, otel-collector; dcgm stays DOWN (no DCGM on laptops)

5. Data: copy steady_window.csv + burst_window.csv from this laptop
   (or the full BurstGPT csv + run slice_burstgpt_window.py)

6. Run:
   cd scripts
   TRACE_FILE=./steady_window.csv SCENARIO=steady ARTIFACT_ROOT=./aiperf_local \
   MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 aiperf_trace_replay.py && \
   TRACE_FILE=./burst_window.csv SCENARIO=burst ARTIFACT_ROOT=./aiperf_local \
   MODEL=Qwen/Qwen2.5-1.5B-Instruct python3 aiperf_trace_replay.py

7. Watch: Grafana localhost:3000, live via OTel. Mid-run check:
   curl -s localhost:8889/metrics | grep ttft_ms