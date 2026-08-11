# F2 injection target — decision

## Problem

§V-E requires an F2 run to satisfy three preconditions: RestartCount = 0,
client failures at clean-run levels, latency within clean variation. The F2
fault as observed (§III-B, the NVML Hostname bug) hung AIPerf indefinitely,
so no aggregates exist to evaluate those preconditions against. As specified,
the condition cannot produce data.

## Two distinct faults were conflated

| | F2a: stall the collector's stream | F2b: stall AIPerf's stream |
|---|---|---|
| Broken | Prometheus scrape of nvml-exporter | AIPerf's OTLP telemetry consumer |
| Serving plane | Healthy | Healthy |
| Run outcome | Completes normally | Hangs indefinitely |
| §V-E preconditions | Evaluable | Unsatisfiable |
| Detected by | I_telemetry (gap in collector series) | Watchdog timeout only |
| Matches §III-B | No | Yes |

Verified independent paths in monitoring/docker-compose.yml:
- nvml-exporter:9400 -> Prometheus -> gpu_scrape_thread (out-of-band collector)
- AIPerf -> otel-collector:4318 -> :8889 -> Prometheus (client's own export)

## Decision

Run **F2a** as the controlled condition. Cite the §III-B incident as the
observed instance of **F2b**, motivating §IV-D's out-of-band watchdog rather
than exercising the coverage matrix.

Rationale: F2a is the only variant that yields a measurable run, and it is the
fault the checker is designed to catch — a healthy server whose telemetry
silently stops, producing a plausible but incomplete result set. F2b is a
client liveness failure, not a measurement-plane fault in the paper's sense.

## Injection method

`docker pause nvml-exporter`, hold 60 s, `docker unpause`. Serving and AIPerf
untouched. Prometheus continues scraping and records failed scrapes, so the
gap appears as absent samples with no error surfaced — the realistic form of
the fault.

Since commit 4de647b the scrape isolates fields individually, so a paused
exporter yields snapshots containing only the vllm: gauges. The gap is
field-specific: I_telemetry should fire on the GPU fields while I_counter and
I_request stay green. That disjointness is what the two-plane coverage
argument requires.

## Preconditions (replacing §V-E as written)

1. RestartCount = 0 for the vllm container across the run
2. Client error count within clean-run range for the matched cluster
3. Mean ITL within the run's own FB_USED cluster range
4. GPU telemetry gap present, 55-70 s, at the injected offset

Failing 1-3 invalidates the run as an F2 condition (serving was disturbed).
Failing 4 means the injection did not take.
