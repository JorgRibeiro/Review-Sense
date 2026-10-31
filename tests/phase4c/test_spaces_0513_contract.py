"""SDK identity/cache and scheduler wrapper tests, not a ZeroGPU hardware simulation."""

from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from upstream_spaces import upstream_sdk

from app.app import create_app
from app.config import AppSettings
from reviewsense.predict import ClassScore, Prediction
from reviewsense.zerogpu import gpu_callback_transform, import_zero_runtime
from scripts.verify_phase4b_package import no_network


@pytest.mark.parametrize("value", ["1", "t", "true", "T", "TRUE", "TrUe"])
def test_runtime_accepts_exact_sdk_truthy_values(monkeypatch, value):
    sdk = upstream_sdk(monkeypatch)
    sdk.config.Config.zero_gpu = sdk.config.boolean(value)
    monkeypatch.setenv("SPACES_ZERO_GPU", value)
    with no_network():
        assert import_zero_runtime() is sdk.package


@pytest.mark.parametrize(
    "value", [None, "", "0", "false", "False", "yes", "on", "2", " true", "true ", " t "]
)
def test_runtime_rejects_absent_or_invalid_values_matching_sdk(monkeypatch, value):
    sdk = upstream_sdk(monkeypatch)
    sdk.config.Config.zero_gpu = sdk.config.boolean(value)
    if value is None:
        monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    else:
        monkeypatch.setenv("SPACES_ZERO_GPU", value)
    assert sdk.config.Config.zero_gpu is False
    with no_network(), pytest.raises(RuntimeError, match="runtime ZeroGPU"):
        import_zero_runtime()


def test_truthy_env_cannot_override_inactive_loaded_sdk(monkeypatch):
    sdk = upstream_sdk(monkeypatch)
    monkeypatch.setenv("SPACES_ZERO_GPU", "1")
    sdk.config.Config.zero_gpu = False
    with pytest.raises(RuntimeError, match="emulação"):
        import_zero_runtime()


def test_wraps_alone_does_not_register_callback_in_real_sdk(monkeypatch):
    sdk = upstream_sdk(monkeypatch)
    original = Mock()
    allocated = sdk.package.GPU(duration=10)(original)

    @wraps(allocated)
    def previous_outer(*args):
        return allocated(*args)

    assert hasattr(previous_outer, "zerogpu")
    assert previous_outer not in sdk.decorator.decorated_cache
    assert sdk.package.GPU(duration=10)(previous_outer) is not previous_outer
    sdk.client.schedule.assert_not_called()


def test_final_callback_identity_registered_and_allocated_by_actual_sdk_code(monkeypatch):
    sdk = upstream_sdk(monkeypatch)

    def callback(text, model):
        assert sdk.state["allocated"]
        return "synthetic result"

    with no_network() as network:
        final = gpu_callback_transform(sdk.package)(callback)
        registry = sdk.decorator.decorated_cache
        assert registry[final] is final
        assert sdk.package.GPU(duration=10)(final) is final
        assert registry[callback] is not final
        assert final("synthetic", "bertimbau") == "synthetic result"
    assert sdk.client.schedule.call_args.kwargs["task_id"] == id(callback)
    assert sdk.state["order"] == [
        ("schedule", id(callback)),
        ("worker", id(callback)),
        ("release", None),
    ]
    assert network["blocked_connections"] == 0 and not sdk.state["allocated"]


def test_sdk_scheduler_denial_cannot_call_inference_and_is_sanitized(monkeypatch):
    sdk = upstream_sdk(monkeypatch)
    callback = Mock(side_effect=AssertionError("inference forbidden"))
    sdk.client.schedule.side_effect = RuntimeError("submitted secret /home/private")
    final = gpu_callback_transform(sdk.package)(callback)
    with no_network():
        assert "Escreva" in final("", "bertimbau")
        assert "8.000" in final("x" * 8001, "bertimbau")
        assert "Selecione" in final("text", "svm")
        sdk.client.schedule.assert_not_called()
        html = final("submitted secret", "bertimbau")
    assert "GPU temporariamente indisponível" in html
    assert "submitted secret" not in html and "/home/" not in html
    callback.assert_not_called()
    assert sdk.client.schedule.call_args.kwargs["task_id"] == id(callback)


def test_gradio_registers_the_exact_final_cached_callback(monkeypatch):
    sdk = upstream_sdk(monkeypatch)

    def predict(*args, **kwargs):
        assert sdk.state["allocated"]
        return Prediction(
            "positivo",
            2,
            "bertimbau",
            "cuda:0",
            "fp16",
            (
                ClassScore(2, "positivo", 0.8),
                ClassScore(1, "neutro", 0.15),
                ClassScore(0, "negativo", 0.05),
            ),
            "softmax_uncalibrated",
            0.001,
            0,
            1,
        )

    demo = create_app(
        SimpleNamespace(predict=predict),
        settings=AppSettings(enable_svm=False),
        callback_transform=gpu_callback_transform(sdk.package),
    )
    try:
        events = [f for f in demo.fns.values() if f.api_name in ("analyze", "analyze_enter")]
        assert len(events) == 2 and events[0].fn is events[1].fn
        final = events[0].fn
        assert sdk.decorator.decorated_cache[final] is final
        assert sdk.package.GPU(final) is final
        assert all(f.concurrency_id == "inference" and f.concurrency_limit == 1 for f in events)
        assert demo._queue.max_size == 8
        with no_network():
            rendered = final("synthetic", "bertimbau")
        assert "Scores normalizados" in rendered
        assert sdk.client.schedule.call_count == 1
    finally:
        demo.close()


def test_unknown_registry_contract_fails_before_binding_ui(monkeypatch):
    sdk = upstream_sdk(monkeypatch)
    sdk.package.GPU = lambda **kwargs: lambda callback: callback
    with pytest.raises(RuntimeError, match="Registro GPU incompatível"):
        gpu_callback_transform(sdk.package)(lambda *a: None)
