# 1. Go to monitoring folder
cd C:\Users\alawn\Documents\GitHub\LLM-Graduation-Project\monitoring

# 2. Set image overrides to use your Docker Hub builds
$env:GRAFANA_IMAGE="faresalawneh/llm-observability-grafana:latest"
$env:PROMETHEUS_IMAGE="faresalawneh/llm-observability-prometheus:latest"
$env:PUSHGATEWAY_IMAGE="faresalawneh/llm-observability-pushgateway:latest"

# 3. Bring up the whole stack (Prometheus, Pushgateway, Grafana, Ollama)
docker compose --profile local up -d

# 4. Wait ~10 seconds, then verify all 4 are running
docker ps

# 5. Pull the model into Ollama (only needed once, persists in volume)
docker exec ollama ollama pull llama3.2:1b

# 6. Confirm model is loaded
docker exec ollama ollama list

# 7. Wipe any old Pushgateway data
curl.exe -X PUT http://localhost:9091/api/v1/admin/wipe

# 8. Open Grafana in browser while waiting: http://localhost:3000 (admin/admin)

# 9. Go to the scripts folder
cd C:\Users\alawn\Documents\GitHub\LLM-Graduation-Project\monitoring\scripts

# 10. Set bench script env vars
$env:INFERENCE_URL="http://localhost:11434/v1/completions"
$env:MODEL="llama3.2:1b"
$env:DATASET_PATH="C:\Users\alawn\Documents\GitHub\LLM-Graduation-Project\BurstGPT_without_fails_3.csv"
$env:ARTIFACT_ROOT="C:\Users\alawn\Documents\GitHub\LLM-Graduation-Project\replay_results"
$env:ENABLE_GPU_SCRAPE="0"

# 11. Run the replay
python burstgpt_real_replay.py