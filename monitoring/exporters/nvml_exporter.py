#!/usr/bin/env python3
"""
NVML-based GPU exporter for the local laptop environment (WSL2 + RTX 5060).

DCGM cannot run under WSL2 (paravirtualized GPU, nv-hostengine will not start),
so this exporter reads the same fields straight from NVML and emits them under
the EXACT DCGM Exporter metric names and labels (gpu, UUID, modelName), on the
same port 9400 — the Grafana dashboards and Prometheus job work unchanged.

Only DCGM_FI_DEV_* gauges are covered. DCGM_FI_PROF_* profiling counters are
DCP-family only and do not exist here. Under WSL2, temperature and power are
commonly unsupported; each field is probed individually and skipped if NVML
rejects it, so one unsupported field never kills the whole scrape.
"""

import atexit
import logging
import socket
import os
import time

import pynvml
from prometheus_client import start_http_server
from prometheus_client.core import GaugeMetricFamily, REGISTRY
from prometheus_client import PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR

NVML_EXPORTER_PORT = int(os.getenv("NVML_EXPORTER_PORT", "9400"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("nvml_exporter")

LABEL_NAMES = ["gpu", "UUID", "modelName", "source", "Hostname"]
# DCGM emits Hostname on every metric; AIPerf 0.11+ requires it and will
# hang on "Waiting for telemetry results..." if it is absent.
HOSTNAME = os.getenv("NVML_EXPORTER_HOSTNAME") or socket.gethostname()


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value


class NVMLCollector:
    """Custom collector emitting DCGM-named gauges from NVML reads."""

    def __init__(self):
        self._warned_unsupported = False

    # (metric_name, help_text, reader) — reader takes an NVML device handle.
    def _field_readers(self):
        return [
            ("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (%).",
             lambda h: float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)),
            ("DCGM_FI_DEV_MEM_COPY_UTIL", "Memory copy utilization (%).",
             lambda h: float(pynvml.nvmlDeviceGetUtilizationRates(h).memory)),
            ("DCGM_FI_DEV_SM_CLOCK", "SM clock frequency (MHz).",
             lambda h: float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))),
            ("DCGM_FI_DEV_MEM_CLOCK", "Memory clock frequency (MHz).",
             lambda h: float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))),
            ("DCGM_FI_DEV_FB_USED", "Framebuffer memory used (MiB).",
             lambda h: float(pynvml.nvmlDeviceGetMemoryInfo(h).used // 1024**2)),
            ("DCGM_FI_DEV_FB_FREE", "Framebuffer memory free (MiB).",
             lambda h: float(pynvml.nvmlDeviceGetMemoryInfo(h).free // 1024**2)),
            ("DCGM_FI_DEV_FB_TOTAL", "Framebuffer memory total (MiB).",
             lambda h: float(pynvml.nvmlDeviceGetMemoryInfo(h).total // 1024**2)),
            ("DCGM_FI_DEV_POWER_USAGE", "Power usage (W).",
             lambda h: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0),
            ("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (C).",
             lambda h: float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))),
        ]

    def collect(self):
        unsupported = []
        families = {
            name: GaugeMetricFamily(name, help_text, labels=LABEL_NAMES)
            for name, help_text, _ in self._field_readers()
        }

        device_count = pynvml.nvmlDeviceGetCount()
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            labels = [
                str(index),
                _decode(pynvml.nvmlDeviceGetUUID(handle)),
                _decode(pynvml.nvmlDeviceGetName(handle)),
                "nvml",
                HOSTNAME,
            ]
            for name, _, reader in self._field_readers():
                try:
                    value = reader(handle)
                except pynvml.NVMLError:
                    unsupported.append("{}[gpu={}]".format(name, index))
                    continue
                families[name].add_metric(labels, value)

        if unsupported and not self._warned_unsupported:
            log.warning("Unsupported NVML fields (skipped): %s", ", ".join(unsupported))
            self._warned_unsupported = True

        for family in families.values():
            yield family


def main():
    pynvml.nvmlInit()
    atexit.register(pynvml.nvmlShutdown)
    log.info("NVML initialized: driver=%s devices=%d",
             _decode(pynvml.nvmlSystemGetDriverVersion()),
             pynvml.nvmlDeviceGetCount())

    # keep /metrics clean: DCGM-named gauges only
    for collector in (PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR):
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass
    REGISTRY.register(NVMLCollector())

    start_http_server(NVML_EXPORTER_PORT)
    log.info("Serving /metrics on port %d", NVML_EXPORTER_PORT)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
