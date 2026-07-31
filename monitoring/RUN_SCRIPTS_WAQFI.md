# RUN SCRIPTS — WAQFI

Commands for running INFERSCOPE GP2 experiments.

## 1. Run

```bash
cd ~/LLM-Graduation-Project/monitoring
./run_experiment.sh
```

Menu:~[200~smoke 300 requests ~30s artifacts -> /tmp
steady 72 requests ~15m
burst 9,483 requests ~15m
q) quit~Or name the workload directly:

```bash
./run_experiment.sh smoke
./run_experiment.sh steady
./run_experiment.sh burst
```

Options:

```bash
./run_experiment.sh burst --teardown      # stop the stack afterwards
./run_experiment.sh smoke --no-mlflow     # run without logging to MLflow
```

## 2. View results

```bash
cd ~/LLM-Graduation-Project/monitoring
./mlflow_ui.sh
```

Open http://localhost:5000 -> click "Model training" (top-left toggle) ->
inferscope-gp2. Ctrl+C to stop. Alternate port: `./mlflow_ui.sh 5001`

## 3. Stop the stack

Only when finished - leaving it up saves the ~95s model load next time.

```bash
cd ~/LLM-Graduation-Project/monitoring
docker compose --profile local down
pkill -f "docker events --filter container=vllm"
```

## One-time setup (per machine)

```bash
sudo apt-get install -y git-lfs && git lfs install --skip-repo
cd ~/LLM-Graduation-Project && git lfs checkout
source ~/venvs/aiperf-env/bin/activate
pip install -r requirements-experiments.txt
```

Docker Desktop -> Settings -> Resources -> WSL Integration -> enable Ubuntu-24.04.

No venv activation needed afterwards - run_experiment.sh handles it.

## Notes

- Docker Desktop must be running before any command above.
- Output goes to `artifacts/<scenario>_<timestamp>/` (gitignored, local only).
- Smoke runs go to `/tmp/smoke_artifacts/` and are cleared on reboot.
- MLflow's Duration column is logging time, not benchmark time - use the
  `wall` figure from the run summary.
