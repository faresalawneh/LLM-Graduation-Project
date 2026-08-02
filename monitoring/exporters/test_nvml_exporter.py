"""Tests for nvml_exporter. pynvml is mocked entirely — no GPU needed in CI."""

import sys
import types
from unittest import mock

import pytest


EXPECTED_METRICS = [
    "DCGM_FI_DEV_GPU_UTIL",
    "DCGM_FI_DEV_MEM_COPY_UTIL",
    "DCGM_FI_DEV_SM_CLOCK",
    "DCGM_FI_DEV_MEM_CLOCK",
    "DCGM_FI_DEV_FB_USED",
    "DCGM_FI_DEV_FB_FREE",
    "DCGM_FI_DEV_FB_TOTAL",
    "DCGM_FI_DEV_POWER_USAGE",
    "DCGM_FI_DEV_GPU_TEMP",
]


class NVMLError(Exception):
    pass


class NVMLError_NotSupported(NVMLError):
    pass


def _make_fake_pynvml():
    fake = types.ModuleType("pynvml")
    fake.NVMLError = NVMLError
    fake.NVMLError_NotSupported = NVMLError_NotSupported
    fake.NVML_CLOCK_SM = 1
    fake.NVML_CLOCK_MEM = 2
    fake.NVML_TEMPERATURE_GPU = 0

    util = types.SimpleNamespace(gpu=42, memory=17)
    mem = types.SimpleNamespace(used=3 * 1024**2, free=5 * 1024**2, total=8 * 1024**2)

    fake.nvmlInit = mock.Mock()
    fake.nvmlShutdown = mock.Mock()
    fake.nvmlSystemGetDriverVersion = mock.Mock(return_value=b"580.00")
    fake.nvmlDeviceGetCount = mock.Mock(return_value=1)
    fake.nvmlDeviceGetHandleByIndex = mock.Mock(return_value="handle0")
    fake.nvmlDeviceGetUUID = mock.Mock(return_value=b"GPU-fake-uuid")
    fake.nvmlDeviceGetName = mock.Mock(return_value=b"NVIDIA GeForce RTX 5060")
    fake.nvmlDeviceGetUtilizationRates = mock.Mock(return_value=util)
    fake.nvmlDeviceGetClockInfo = mock.Mock(return_value=1500)
    fake.nvmlDeviceGetMemoryInfo = mock.Mock(return_value=mem)
    fake.nvmlDeviceGetPowerUsage = mock.Mock(return_value=45000)
    fake.nvmlDeviceGetTemperature = mock.Mock(return_value=61)
    return fake


@pytest.fixture
def exporter():
    fake = _make_fake_pynvml()
    with mock.patch.dict(sys.modules, {"pynvml": fake}):
        sys.modules.pop("nvml_exporter", None)
        import nvml_exporter
        yield nvml_exporter, fake
        sys.modules.pop("nvml_exporter", None)


def _collect(module):
    return list(module.NVMLCollector().collect())


def test_all_metrics_emitted(exporter):
    module, _ = exporter
    families = _collect(module)
    names = [f.name for f in families]
    assert sorted(names) == sorted(EXPECTED_METRICS)
    for family in families:
        assert len(family.samples) == 1, "expected one sample for {}".format(family.name)


def test_unsupported_field_is_skipped(exporter):
    module, fake = exporter
    fake.nvmlDeviceGetTemperature.side_effect = NVMLError_NotSupported("not supported")

    families = _collect(module)  # must not raise

    by_name = {f.name: f for f in families}
    assert by_name["DCGM_FI_DEV_GPU_TEMP"].samples == []
    for name in EXPECTED_METRICS:
        if name != "DCGM_FI_DEV_GPU_TEMP":
            assert len(by_name[name].samples) == 1, "{} missing".format(name)


def test_labels_match_dcgm_contract(exporter):
    module, _ = exporter
    for family in _collect(module):
        for sample in family.samples:
            assert set(sample.labels) == {"gpu", "UUID", "modelName", "source", "Hostname"}
            assert sample.labels["gpu"] == "0"
            assert sample.labels["UUID"] == "GPU-fake-uuid"
            assert sample.labels["modelName"] == "NVIDIA GeForce RTX 5060"
            assert sample.labels["source"] == "nvml"
