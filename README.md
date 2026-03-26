🚀 Quick Start Guide
1. Run Local Stack
Navigate to the project root where docker-compose.yml is located and run:

Bash
docker-compose up -d
Grafana: http://localhost:3000 (Default: admin/admin)

Pushgateway: http://localhost:9091

2. Establish Data Tunnel (ngrok)
Open your terminal and start a tunnel to the Pushgateway port:

Bash
ngrok http 9091
⚠️ Copy the "Forwarding" URL (e.g., https://xxxx.ngrok-free.dev).

3. Execute Benchmark (Google Colab)
Open the provided Notebook in Google Colab.

Upload BurstGPT_without_fails_3.csv to the /content/ directory.

Replace the NGROK_URL variable in the code with your copied ngrok URL.

Run the vLLM Server cell first, then run the Benchmark/Stress Test cell.

4. Setup Grafana Dashboard
Add Data Source: Select Prometheus and set the URL to http://prometheus:9090.

Import Dashboard: Go to Dashboards -> Import and upload the dashboard.json file included in this repo.

📂 Required Repository Files:
Ensure the following files are in the root directory for this guide to work:

docker-compose.yml

prometheus.yml

demo.ipynb
