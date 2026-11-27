"""Tests for the provenance params in mlflow_log_run.py.

mlflow is not installed in CI, so a stand-in module is used.
"""
import importlib
import json
import logging
import sys
from unittest import mock

import pytest

NEW_PARAMS = ("qwen_revision", "max_concurrency", "git_dirty",
              "trace_original_path", "trace_original_sha256")
REVISION = "a" * 40


@pytest.fixture
def logger_module(monkeypatch):
    fake = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.delitem(sys.modules, "mlflow_log_run", raising=False)
    module = importlib.import_module("mlflow_log_run")
    yield module, fake
    sys.modules.pop("mlflow_log_run", None)


def write_record(directory, trace_used=None, **changes):
    trace = directory / "burst_window.csv"
    record = {
        "trace_used": str(trace_used or trace),
        "trace_original_path": "/repo/monitoring/scripts/burst_window.csv",
        "trace_original_sha256": "d" * 64,
        "max_concurrency": "7",
        "git_dirty": "false",
    }
    record.update(changes)
    (directory / "trace_provenance.json").write_text(json.dumps(record))
    return {"trace_file": str(trace)}, record


def test_all_values_are_reported(logger_module, tmp_path):
    module, _ = logger_module
    manifest, record = write_record(tmp_path)
    params = module.run_provenance(manifest, {"qwen_revision": REVISION})
    assert params == {
        "qwen_revision": REVISION,
        **{k: record[k] for k in NEW_PARAMS[1:]},
    }


@pytest.mark.parametrize("manifest", [{}, {"trace_file": "/nowhere/x.csv"}])
def test_missing_values_are_unavailable_with_a_warning(
        logger_module, manifest, caplog):
    module, _ = logger_module
    params = module.run_provenance(manifest, {})
    assert params == {key: "unavailable" for key in NEW_PARAMS}
    for key in NEW_PARAMS:
        assert f"{key} unavailable" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_a_record_for_another_trace_is_ignored(logger_module, tmp_path, caplog):
    module, _ = logger_module
    manifest, _ = write_record(tmp_path, trace_used="/other/trace.csv")
    params = module.run_provenance(manifest, {})
    assert params["trace_original_sha256"] == "unavailable"
    assert params["max_concurrency"] == "unavailable"
    assert "different trace" in caplog.text


def test_broken_record_and_unavailable_markers(logger_module, tmp_path):
    module, _ = logger_module
    manifest, _ = write_record(tmp_path, git_dirty="unavailable", max_concurrency="")
    params = module.run_provenance(manifest, {"qwen_revision": "unavailable"})
    assert params["qwen_revision"] == "unavailable"
    assert params["git_dirty"] == "unavailable"
    assert params["max_concurrency"] == "unavailable"
    (tmp_path / "trace_provenance.json").write_text("{broken")
    assert module.run_provenance(manifest, {})["git_dirty"] == "unavailable"


def test_main_logs_the_new_params(logger_module, tmp_path, monkeypatch):
    module, fake = logger_module
    run_dir = tmp_path / "burst_run"
    run_dir.mkdir()
    manifest, _ = write_record(tmp_path)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
    state = tmp_path / "stack_state.json"
    state.write_text(json.dumps({"qwen_revision": REVISION, "stack_env": "dev"}))
    # Keep the logger's mlruns/ inside the temp directory and away from Docker.
    monkeypatch.setattr(module, "__file__", str(tmp_path / "m" / "s" / "x.py"))
    monkeypatch.setattr(module, "docker_state", lambda: {})
    monkeypatch.setattr(module, "git_sha", lambda root: "abc1234")
    monkeypatch.setattr(
        sys, "argv", ["x", str(run_dir), "--stack-state", str(state)])
    assert module.main() == 0
    logged = fake.log_params.call_args.args[0]
    assert logged["qwen_revision"] == REVISION
    assert logged["max_concurrency"] == "7"
    assert logged["git_dirty"] == "false"
    assert logged["trace_original_sha256"] == "d" * 64
    assert logged["trace_original_path"].endswith("burst_window.csv")
