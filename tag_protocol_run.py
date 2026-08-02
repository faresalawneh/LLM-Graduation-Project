#!/usr/bin/env python3
"""Attach cohort and fault-injection tags to the MLflow run just logged.

``run_experiment.sh`` logs each run through ``mlflow_log_run.py``. This adds the
protocol metadata afterwards, matching on the ``artifact_dir`` param so the
existing logging path is left untouched.

Example:
    $ python tag_protocol_run.py --run-dir artifacts/burst_2026-08-02_1130/ \
          --step 4 --cohort B --fault kill200 --kill-time 200
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.INFO, format="[tag] %(message)s")
log: Final = logging.getLogger(__name__)


def main() -> int:
    """Tag the MLflow run whose ``artifact_dir`` matches ``--run-dir``.

    Returns:
        0 on success, 1 if no matching run was found.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--step", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--fault", required=True, help="clean | kill200 | kill400 | kill600")
    ap.add_argument("--kill-time", required=True)
    ap.add_argument("--tracking-uri", default="sqlite:///mlruns/mlflow.db")
    ns = ap.parse_args()

    import mlflow

    mlflow.set_tracking_uri(ns.tracking_uri)
    target = str(ns.run_dir).rstrip("/")

    runs = mlflow.search_runs(
        experiment_names=["inferscope-gp2"],
        order_by=["start_time DESC"],
        max_results=25,
        output_format="pandas",
    )
    col = "params.artifact_dir"
    if col not in runs.columns:
        log.error("no %s column in MLflow runs", col)
        return 1

    match = runs[runs[col].astype(str).str.rstrip("/") == target]
    if match.empty:
        log.error("no MLflow run with artifact_dir=%s", target)
        return 1

    run_id = match.iloc[0]["run_id"]
    tags = {
        "cohort": ns.cohort,
        "protocol_step": str(ns.step),
        "fault_injected": "false" if ns.fault == "clean" else "true",
        "fault_label": ns.fault,
        "kill_time_s": str(ns.kill_time),
        "run_type": "full",
    }

    fault_file = ns.run_dir / "fault_injection.json"
    if fault_file.is_file():
        try:
            tags["kill_epoch"] = str(json.loads(fault_file.read_text())["kill_epoch"])
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("could not read kill_epoch: %s", exc)

    client = mlflow.tracking.MlflowClient()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)
    log.info("tagged run %s: %s", run_id[:8], tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
