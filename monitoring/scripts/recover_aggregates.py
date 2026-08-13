#!/usr/bin/env python3
"""Recompute AIPerf run aggregates from per-request records.

AIPerf writes profile_export.jsonl incrementally, one record per attempted
request, with per-request metrics already computed in milliseconds. Its
finalisation step (which produces profile_export_aiperf.json) can stall
indefinitely after a mid-replay engine kill, leaving a complete record file
but no aggregates.

This tool reproduces that aggregation step. It does not recompute any
measurement: it consumes AIPerf's own per-request values and only replaces the
percentile pass. Validated against clean runs to four decimal places on avg,
p50 and p99 for request_latency, time_to_first_token and inter_token_latency.

Output goes to profile_export_recovered.json, deliberately not named
profile_export_aiperf.json so recovered and native aggregates are never
confused downstream.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
VALIDATE_KEYS = ("request_latency", "time_to_first_token", "inter_token_latency")


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile matching AIPerf's convention."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p / 100.0
    lower = int(rank)
    if lower + 1 >= len(ordered):
        return ordered[lower]
    return ordered[lower] + (rank - lower) * (ordered[lower + 1] - ordered[lower])


def summarize(values: list[float], unit: str) -> dict[str, Any]:
    """Build an AIPerf-shaped statistics block for one metric."""
    total = math.fsum(values)
    mean = total / len(values)
    if len(values) > 1:
        variance = math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    block: dict[str, Any] = {"unit": unit, "avg": mean}
    for p in PERCENTILES:
        block["p{}".format(p)] = percentile(values, p)
    block["min"] = min(values)
    block["max"] = max(values)
    block["std"] = std
    block["count"] = len(values)
    block["sum"] = total
    return block


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL record file, dropping unparseable trailing lines."""
    records: list[dict[str, Any]] = []
    dropped = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1
    if dropped:
        print("  warning: dropped {} unparseable line(s)".format(dropped), file=sys.stderr)
    return records


def recover(artifact_dir: Path) -> dict[str, Any]:
    """Recompute aggregates for one run directory."""
    source = artifact_dir / "profile_export.jsonl"
    if not source.exists():
        raise FileNotFoundError("no profile_export.jsonl in {}".format(artifact_dir))

    records = load_records(source)
    if not records:
        raise ValueError("no records in {}".format(source))

    buckets: dict[str, list[float]] = defaultdict(list)
    units: dict[str, str] = {}
    cancelled = 0
    starts: list[int] = []
    ends: list[int] = []

    for record in records:
        meta = record.get("metadata", {})
        if meta.get("was_cancelled"):
            cancelled += 1
        start = meta.get("request_start_ns")
        if start is not None:
            starts.append(start)
        end = meta.get("request_end_ns")
        if end is not None:
            ends.append(end)
        for key, payload in record.get("metrics", {}).items():
            value = payload.get("value")
            if value is None:
                continue
            # Some metrics are per-chunk arrays (one element per streamed
            # chunk); AIPerf aggregates these by flattening across requests.
            if isinstance(value, list):
                buckets[key].extend(
                    float(item) for item in value if isinstance(item, (int, float))
                )
            elif isinstance(value, (int, float)):
                buckets[key].append(float(value))
            else:
                continue
            units.setdefault(key, payload.get("unit", ""))

    result: dict[str, Any] = {
        "recovered": True,
        "recovered_from": str(source),
        "record_count": len(records),
        "cancelled_count": cancelled,
    }

    if starts and ends:
        duration_s = (max(ends) - min(starts)) / 1e9
        result["benchmark_duration"] = {"unit": "s", "avg": duration_s}
        if duration_s > 0:
            result["request_throughput"] = {
                "unit": "requests/sec",
                "avg": len(records) / duration_s,
            }

    for key, values in sorted(buckets.items()):
        result[key] = summarize(values, units.get(key, ""))

    result["request_count"] = {
        "unit": "requests",
        "avg": float(len(buckets.get("request_latency", records))),
    }
    return result


CSV_LABELS = {
    "request_latency": "Request Latency (ms)",
    "time_to_first_token": "Time to First Token (ms)",
    "time_to_second_token": "Time to Second Token (ms)",
    "inter_token_latency": "Inter Token Latency (ms)",
    "inter_chunk_latency": "Inter Chunk Latency (ms)",
    "output_sequence_length": "Output Sequence Length (tokens)",
    "input_sequence_length": "Input Sequence Length (tokens)",
    "output_token_count": "Output Token Count (tokens)",
    "time_to_first_output_token": "Time to First Output Token (ms)",
    "http_req_duration": "HTTP Duration (excl. conn) (ms)",
    "http_req_waiting": "HTTP Waiting (TTFB) (ms)",
    "http_req_receiving": "HTTP Receiving (ms)",
    "http_req_sending": "HTTP Sending (ms)",
}
CSV_COLUMNS = ("avg", "min", "max", "sum", "p1", "p5", "p10",
               "p25", "p50", "p75", "p90", "p95", "p99", "std")


def write_csv(result: dict[str, Any], destination: Path) -> None:
    """Emit an AIPerf-shaped summary CSV from recovered aggregates.

    mlflow_log_run.py reads profile_export_aiperf.csv; recovered runs have no
    native CSV, so without this they cannot be logged to MLflow.
    """
    import csv as _csv

    with destination.open("w", newline="") as handle:
        writer = _csv.writer(handle)
        writer.writerow(("Metric",) + CSV_COLUMNS)
        for key, label in CSV_LABELS.items():
            block = result.get(key)
            if not isinstance(block, dict):
                continue
            row = [label]
            for column in CSV_COLUMNS:
                value = block.get(column)
                row.append("" if value is None else "{:.2f}".format(float(value)))
            writer.writerow(row)
        count = result.get("request_latency", {}).get("count")
        if count is not None:
            writer.writerow(["Request Count (count)"] + ["{:.2f}".format(float(count))] * len(CSV_COLUMNS))


def validate(artifact_dir: Path) -> bool:
    """Compare recovered aggregates against AIPerf's own, when both exist."""
    native_path = artifact_dir / "profile_export_aiperf.json"
    if not native_path.exists():
        print("  no native aggregates in {}; cannot validate".format(artifact_dir))
        return False

    native = json.loads(native_path.read_text())
    recovered = recover(artifact_dir)
    ok = True

    for key in VALIDATE_KEYS:
        if key not in native or key not in recovered:
            print("  {:24s} absent in one source".format(key))
            ok = False
            continue
        for stat in ("avg", "p50", "p99", "count"):
            mine = recovered[key].get(stat)
            theirs = native[key].get(stat)
            if mine is None or theirs is None:
                continue
            match = math.isclose(float(mine), float(theirs), rel_tol=1e-9, abs_tol=1e-6)
            ok = ok and match
            flag = "ok " if match else "MISMATCH"
            print("  {} {}.{}: recovered={!r} native={!r}".format(flag, key, stat, mine, theirs))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dirs", type=Path, nargs="+")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="compare against native aggregates instead of writing output",
    )
    args = parser.parse_args()

    failures = 0
    for artifact_dir in args.artifact_dirs:
        print("--- {}".format(artifact_dir.name))
        try:
            if args.validate:
                if not validate(artifact_dir):
                    failures += 1
                continue
            result = recover(artifact_dir)
            destination = artifact_dir / "profile_export_recovered.json"
            destination.write_text(json.dumps(result, indent=1))
            write_csv(result, artifact_dir / "profile_export_recovered.csv")
            latency = result.get("request_latency", {})
            print("  records={} latency.avg={:.4f} -> {}".format(
                result["record_count"],
                latency.get("avg", float("nan")),
                destination.name,
            ))
        except (OSError, ValueError) as exc:
            print("  FAILED: {}".format(exc), file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
