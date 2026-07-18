#!/usr/bin/env python3
"""
Slice a 900-second window out of the full BurstGPT CSV for AIPerf trace replay.

Buckets follow burstgpt_real_replay.py: bucket N covers
[N * 900, (N + 1) * 900) seconds, e.g. 21661 (steady) and 23832 (burst).

Rebase note (verified empirically with aiperf 0.11.0): the burst_gpt_trace
loader normalizes timestamps to the first request in the file, so a window
starting at e.g. 19,494,900s replays immediately without stalling. --rebase
therefore defaults to OFF; use it if you want the window CSV to be
self-describing with timestamps starting at 0.

Usage:
    python3 slice_burstgpt_window.py --bucket 21661 \
        --input BurstGPT_without_fails_3.csv --output steady_window.csv [--rebase]
"""

import argparse
import csv
import logging
from pathlib import Path

BUCKET_DURATION = 900  # seconds, same 15-minute buckets as burstgpt_real_replay.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("slice_burstgpt_window")


def slice_window(input_path: Path, output_path: Path, bucket_id: int, rebase: bool) -> int:
    start_ts = bucket_id * BUCKET_DURATION
    end_ts   = start_ts + BUCKET_DURATION

    rows = []
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or "Timestamp" not in fieldnames:
            raise SystemExit("Input CSV has no 'Timestamp' column: {}".format(input_path))
        for row in reader:
            try:
                ts = float(row["Timestamp"])
            except (TypeError, ValueError):
                continue
            if start_ts <= ts < end_ts:
                rows.append(row)

    rows.sort(key=lambda r: float(r["Timestamp"]))
    log.info("Found %d rows for bucket %d (%.0fs to %.0fs)",
             len(rows), bucket_id, start_ts, end_ts)

    if rows and rebase:
        base_ts = float(rows[0]["Timestamp"])
        for row in rows:
            row["Timestamp"] = "{:.3f}".format(float(row["Timestamp"]) - base_ts)
        log.info("Rebased timestamps to start at 0 (base was %.3f)", base_ts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %d rows to %s", len(rows), output_path)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Slice a 900s bucket window from a BurstGPT CSV")
    parser.add_argument("--bucket", type=int, required=True,
                        help="Bucket ID (900s buckets, e.g. 21661 steady / 23832 burst)")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the full BurstGPT CSV")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path for the sliced window CSV")
    parser.add_argument("--rebase", action="store_true",
                        help="Rewrite Timestamp to start at 0 (optional; aiperf "
                             "auto-normalizes to the first timestamp anyway)")
    args = parser.parse_args()

    count = slice_window(args.input, args.output, args.bucket, args.rebase)
    if count == 0:
        raise SystemExit("No rows found for bucket {}".format(args.bucket))


if __name__ == "__main__":
    main()
