"""Opções locais, mensagens de hospedagem e falhas públicas seguras."""

from types import SimpleNamespace

import pytest

from app.app import create_app, launch_options, make_analyzer
from app.config import AppSettings, service_from_settings
from reviewsense.predict import ArtifactError, DeviceError, InferenceError, InputError


def test_local_default_lazy_and_never_public():
    service = service_from_settings(AppSettings())
    assert service.loaded_models == ()
    assert service.model_source is not None
    options = launch_options()
    assert options["server_name"] == "127.0.0.1" and options["share"] is False


@pytest.mark.parametrize(
    "values", [dict(origin="hub-cache"), dict(environment="wrong"), dict(origin="other")]
)
def test_incomplete_or_invalid_settings(values):
    with pytest.raises(ValueError):
        AppSettings(**values)


def test_hosted_blocked_with_current_descriptor():
    with pytest.raises(PermissionError, match="bloqueada"):
        service_from_settings(AppSettings(environment="hosted"))
    with pytest.raises(PermissionError, match="bloqueada"):
        create_app(SimpleNamespace(), settings=AppSettings(environment="hosted"))


def test_hosted_preview_copy_limits_and_shared_queue(monkeypatch):
    # Somente constrói UI com serviço falso; não executa launch nem hospeda pesos.
    monkeypatch.setattr("app.app.check_hosting_authorization", lambda settings: None)
    demo = create_app(
        SimpleNamespace(predict=lambda *a, **k: None),
        settings=AppSettings(environment="hosted", enable_svm=False),
    )
    try:
        config = repr(demo.config)
        assert "Demonstração hospedada" in config and "Demonstração local" not in config
        assert "plataforma de hospedagem" in config and "delivery" in config
        radio = next(c for c in demo.config["components"] if c["type"] == "radio")
        assert len(radio["props"]["choices"]) == 1
        assert demo._queue.max_size == 8
        # Dois eventos compartilham explicitamente a mesma fila de execução.
        events = [fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "analyze"]
        assert len(events) == 2
        assert all(fn.concurrency_id == "inference" and fn.concurrency_limit == 1 for fn in events)
    finally:
        demo.close()


@pytest.mark.parametrize(
    "kind", [ArtifactError, DeviceError, InferenceError, InputError, RuntimeError]
)
def test_typed_and_unknown_errors_cannot_leak_internal_paths_or_input(kind, caplog):
    def fail(*a, **k):
        raise kind("sensitive-user-text /home/private hf_" + "X" * 30)

    rendered = make_analyzer(SimpleNamespace(predict=fail))("sensitive-user-text", "bertimbau")
    assert "sensitive-user-text" not in rendered + caplog.text
    assert "/home/" not in rendered and "hf_" not in rendered


def test_artifact_and_approval_directories_blocked_from_file_serving():
    from pathlib import Path

    names = {Path(p).name for p in launch_options()["blocked_paths"]}
    assert {"outputs", "models", "data", "reports", "publishing", "cache"} <= names


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model-origin", "hub-cache"],
        ["--environment", "hosted"],
    ],
)
def test_cli_invalid_or_unauthorized_configuration_never_launches(arguments, monkeypatch, capsys):
    from unittest.mock import Mock

    from app.app import main

    launch = Mock(side_effect=AssertionError("must not build app"))
    monkeypatch.setattr("app.app.create_app", launch)
    monkeypatch.setattr("sys.argv", ["reviewsense-app", *arguments])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "não autorizado" in error and "Traceback" not in error and "/home/" not in error
    launch.assert_not_called()
