"""Interface construída e callbacks executados com serviço sintético em memória."""

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from reviewsense.predict import ClassScore, InputError, Prediction

pytest.importorskip("gradio")
app_module = importlib.import_module("app.app")
create_app = app_module.create_app
launch_options = app_module.launch_options
make_analyzer = app_module.make_analyzer
result_html = app_module.result_html


@pytest.fixture
def prediction():
    return Prediction(
        "positivo",
        2,
        "bertimbau",
        "cpu",
        "fp32",
        (
            ClassScore(2, "positivo", 0.8),
            ClassScore(1, "neutro", 0.15),
            ClassScore(0, "negativo", 0.05),
        ),
        "softmax_uncalibrated",
        0.012,
        0,
        1,
    )


def test_app_builds_without_loading_and_defaults_to_bertimbau(prediction):
    calls = []
    service = SimpleNamespace(predict=lambda *a, **k: calls.append(1))
    demo = create_app(service)
    assert not calls and demo.analytics_enabled is False
    assert demo.title == "ReviewSense PT-BR"
    components = demo.config["components"]
    radio = next(c for c in components if c["type"] == "radio")
    assert radio["props"]["value"] == "bertimbau"
    text = next(c for c in components if c["type"] == "textbox")
    assert text["props"]["max_length"] == 8000
    assert "sensitive_text" not in repr(demo.config)
    demo.close()


def test_launch_options_are_private_by_default():
    options = launch_options()
    assert options["server_name"] == "127.0.0.1" and options["share"] is False
    assert options["run_history"] is False and options["enable_monitoring"] is False
    assert options["ssr_mode"] is False and options["mcp_server"] is False
    assert options["pwa"] is False and options["footer_links"] == []
    assert options["allowed_paths"] == [] and options["show_error"] is False
    assert launch_options(public_network=True)["server_name"] == "0.0.0.0"
    assert any(Path(p).name == "data" for p in options["blocked_paths"])


def test_bert_scores_and_svm_no_probability_display(prediction):
    html = result_html(prediction)
    assert "Scores normalizados" in html and "não calibrados" in html and "12.0 ms" in html
    assert html.index("Positivo</span>") < html.index("Neutro</span>")
    svm = replace(prediction, model="svm", scores=(), score_type=None)
    html = result_html(svm)
    assert "score-track" not in html and "80.0%" not in html
    assert "sem probabilidades calibradas" in html


def test_callback_never_echoes_inputs_and_sanitizes_unexpected_error(prediction, caplog):
    service = SimpleNamespace(predict=lambda *a, **k: prediction)
    assert "secret_text" not in make_analyzer(service)("secret_text", "bertimbau")

    def fail(*args, **kwargs):
        raise RuntimeError("secret_text /home/private/token")

    html = make_analyzer(SimpleNamespace(predict=fail))("secret_text", "bertimbau")
    assert "Não foi possível concluir" in html
    assert "secret_text" not in html + caplog.text and "/home/" not in html


def test_input_error_is_helpful_and_does_not_preserve_old_result():
    def fail(*a, **k):
        raise InputError("Escreva uma avaliação antes de analisar.")

    html = make_analyzer(SimpleNamespace(predict=fail))("", "bertimbau")
    assert 'role="alert"' in html and "Escreva uma avaliação" in html
    assert "sentiment-pill" not in html
