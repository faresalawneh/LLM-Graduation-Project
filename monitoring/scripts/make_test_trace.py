#!/usr/bin/env python3
"""
Writes a small BurstGPT-format test trace CSV for smoke-testing AIPerf trace replay.

Header matches the real BurstGPT dataset:
    Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type

Timestamps are seconds from the start of the trace (~10 rows spanning ~10s).
"""

import csv
import os
from pathlib import Path

OUTPUT_PATH = Path(os.getenv("TRACE_FILE", "test_trace.csv"))

# (timestamp_sec, request_tokens, response_tokens)
ROWS = [
    (0.0,  32,  48),
    (0.5,  64,  96),
    (1.2, 128,  64),
    (2.0,  16,  32),
    (3.5,  96, 128),
    (4.1,  48,  24),
    (5.8, 200, 100),
    (7.0,  24,  80),
    (8.4, 150,  60),
    (9.9,  40,  40),
]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Model", "Request tokens",
                         "Response tokens", "Total tokens", "Log Type"])
        for ts, req_tok, resp_tok in ROWS:
            writer.writerow([ts, "ChatGPT", req_tok, resp_tok,
                             req_tok + resp_tok, "Conversation log"])
    print("Wrote {} rows to {}".format(len(ROWS), OUTPUT_PATH))


if __name__ == "__main__":
    main()
