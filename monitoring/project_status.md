# Real-Time Observability Pipeline for LLM Serving — Project Status

## What We're Building

A full-stack monitoring and benchmarking system for LLM inference that correlates LLM performance metrics with GPU hardware metrics, visualized in Grafana.

**Stack:** vLLM → AIPerf → Prometheus Pushgateway → Prometheus → Grafana

---

## Architecture Overview

```
┌──────────────────────────────────┐       ┌──────────────────────────────┐
│     JUST University GPU Server   │       │      Windows Machine         │
│                                  │       │                              │
│  BurstGPT CSV                    │       │  Docker Compose:             │
│       ↓                          │       │  ┌────────────────────────┐  │
│  benchmark.py                    │       │  │ Pushgateway (:9091)    │  │
│   - converts CSV → JSONL        │  push  │  │     ↓                  │  │
│   - starts vLLM server          │ ──────→│  │ Prometheus  (:9090)    │  │
│   - runs AIPerf benchmark       │       │  │     ↓                  │  │
│   - pushes metrics via HTTP     │       │  │ Grafana     (:3000)    │  │
│                                  │       │  └────────────────────────┘  │
│  DCGM Exporter (future)         │       │                              │
└──────────────────────────────────┘       └──────────────────────────────┘
```

---

## Current File Structure

```
project/
├── docker-compose.yml          # Prometheus + Pushgateway + Grafana
├── prometheus/
│   └── prometheus.yml          # Scrape config (targets: pushgateway:9091)
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   ├── dashboard.yml   # Dashboard provisioning
│       │   └── dashboard.json  # Panels: TTFT, Throughput, GPU Util, GPU Mem
│       └── datasources/
│           └── prometheus.yml  # Points Grafana → Prometheus
├── benchmark.py                # Main pipeline script (runs on GPU server)
└── BurstGPT_without_fails_3.csv  # 5M rows, NOT on GitHub (too large)
```

---

## What's Done

- [x] Docker Compose stack (Prometheus, Pushgateway, Grafana) running on Windows machine
- [x] All three services verified: Grafana (:3000), Prometheus (:9090), Pushgateway (:9091)
- [x] Prometheus scrape config targeting Pushgateway with `honor_labels: true`
- [x] Grafana dashboard provisioned with TTFT, Throughput, GPU Util, GPU Memory panels
- [x] benchmark.py script written — converts BurstGPT CSV → JSONL, starts vLLM, runs AIPerf, pushes to Pushgateway
- [x] BurstGPT dataset ready (5M rows, `BurstGPT_without_fails_3.csv`)
- [x] Previous Colab-based testing phase complete (local Ollama phase also complete)

## What's Next

- [ ] **Upload BurstGPT dataset to university server** (too big for GitHub — use SCP, USB, or Google Drive)
- [ ] **Configure benchmark.py** on the server:
  - Set `BURSTGPT_CSV_PATH` to actual path on server
  - Set `PUSHGATEWAY_URL` to Windows machine IP (e.g., `http://192.168.x.x:9091`)
- [ ] **Install deps on server:** `pip install vllm aiperf pandas requests`
- [ ] **Test connectivity:** server must reach Windows machine on port 9091
- [ ] **Run first benchmark** and verify metrics appear in Grafana
- [ ] **Fix flat metrics issue:** push metrics incrementally during benchmark, not just once at the end
- [ ] **Filter BurstGPT rows** exceeding model's 2048-token context limit
- [ ] **Integrate DCGM Exporter** on university server for GPU hardware metrics
- [ ] **Higher concurrency tests** / consider larger model for GPU saturation
- [ ] **Research writing phase**

---

## Key Info

| Item | Value |
|------|-------|
| Model | `facebook/opt-125m` |
| vLLM port | 8000 |
| Pushgateway | `:9091` on Windows machine |
| Prometheus | `:9090` on Windows machine |
| Grafana | `:3000` on Windows machine (admin/admin) |
| AIPerf concurrency | 10 |
| N_REQUESTS | 500 |
| Dataset | BurstGPT — 5M rows, no actual prompts (dummy prompts sized to match token counts) |

## Known Issues

- BurstGPT has no actual prompt text — benchmark uses `"hello " * n` as dummy prompts
- 5 requests failed in earlier runs due to exceeding 2048-token context limit — need to filter those rows
- Grafana dashboards may show flat lines if metrics are pushed as a single summary instead of incrementally
- DCGM Exporter won't work on consumer GPUs (Colab T4 / laptop) — only on university server
- ngrok free tier gives random URLs each restart — Cloudflare Tunnel is a free alternative
