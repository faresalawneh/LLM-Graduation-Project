**1. Install Docker + NVIDIA Container Toolkit**
```bash
# Docker
curl -fsSL https://get.docker.com | sh

# NVIDIA Container Toolkit (for GPU access in Docker)
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```

**2. Copy your project folder to the server** (USB, git clone, or drag-and-drop via Remote Desktop)

**3. Install vLLM + AIPerf**
```bash
pip install vllm aiperf
```

**4. Start the monitoring stack**
```bash
docker compose up -d
docker-compose ps
```

**5. Serve the model**
```bash
vllm serve facebook/opt-125m --port 8000
```

**6. Run AIPerf benchmark** — point it at `localhost:8000`, push metrics to `localhost:9091`

**7. Open Grafana** at `localhost:3000` in the server's browser

**8. Later — add DCGM Exporter** to compose file, update prometheus target, `docker compose up -d` again

That's it. No ngrok, no tunnels, everything talks over localhost.

Dateset: https://drive.google.com/file/d/1DOkwV9YwKWCePLIEwAreIoT4SJxilozG/view?usp=sharing
