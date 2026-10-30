"""Proteção do teste e tentativas de OOM simuladas; nenhum acesso à GPU."""

from pathlib import Path

import pytest

from reviewsense.train_transformer import (
    diagnose_cuda,
    execute_with_oom_retry,
    protect_test_csv,
    run_smoke,
)
from reviewsense.transformer_data import SmokeConfig


def test_guard_blocks_even_attempting_to_open_test(tmp_path):
    with protect_test_csv(tmp_path) as state:
        with pytest.raises(PermissionError, match="test.csv proibida"):
            (tmp_path / "test.csv").open()
        assert state["blocked_open_attempts"] == 1
        other = tmp_path / "allowed.txt"
        other.write_text("synthetic")
        assert other.read_text() == "synthetic"
    assert not state["active"]


def test_existing_output_is_rejected_before_any_model_or_cuda_access(tmp_path):
    with pytest.raises(FileExistsError):
        run_smoke(Path("not-read"), Path("not-read"), SmokeConfig(), tmp_path, tmp_path / "cache")


def test_no_silent_cpu_training_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="sem fallback CPU"):
        diagnose_cuda()


def test_oom_retries_once_with_conservative_config(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls, errors = [], []

    def execute(config, index):
        calls.append(config)
        if index == 1:
            raise torch.cuda.OutOfMemoryError("synthetic oom")
        return "completed"

    def on_oom(index, config, error, attempts):
        errors.append(index)

    result, attempts = execute_with_oom_retry(SmokeConfig(), execute, on_oom)
    assert result == "completed" and errors == [1]
    assert len(calls) == len(attempts) == 2
    assert calls[1].train_batch_size == 1 and calls[1].gradient_checkpointing
    assert calls[1].effective_batch_size == calls[0].effective_batch_size
    assert [attempt["status"] for attempt in attempts] == ["cuda_oom", "completed"]


def test_oom_cannot_cause_unlimited_attempts(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def execute(config, index):
        calls.append(index)
        raise torch.cuda.OutOfMemoryError("synthetic oom")

    with pytest.raises(RuntimeError, match="duas tentativas"):
        execute_with_oom_retry(SmokeConfig(), execute, lambda *args: None)
    assert calls == [1, 2]


def test_unrelated_errors_do_not_trigger_retries():
    pytest.importorskip("torch")
    calls = []

    def execute(config, index):
        calls.append(index)
        raise ValueError("synthetic non-oom error")

    with pytest.raises(ValueError, match="non-oom"):
        execute_with_oom_retry(SmokeConfig(), execute, lambda *args: None)
    assert calls == [1]
