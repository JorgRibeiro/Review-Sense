"""Serviço offline: somente artefatos sintéticos; nenhuma leitura de dataset real."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from reviewsense.data import file_sha256
from reviewsense.labels import LABEL_NAMES
from reviewsense.predict import (
    MAX_INPUT_CHARACTERS,
    MODEL_PREFIX,
    ArtifactError,
    ArtifactPaths,
    DeviceError,
    InferenceError,
    InferenceService,
    InputError,
    _BertBackend,
    normalized_scores,
    validate_text,
)


@pytest.fixture
def tiny_paths(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    bert_dir, svm_dir = tmp_path / "bert", tmp_path / "svm"
    bert_dir.mkdir()
    svm_dir.mkdir()
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Não", "é", "ótimo", "ruim", "regular"]
        )
    }
    tokenizer = transformers.BertTokenizerFast(vocab=vocab, do_lower_case=False)
    model = transformers.BertForSequenceClassification(
        transformers.BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            id2label=LABEL_NAMES,
            label2id={name: label for label, name in LABEL_NAMES.items()},
        )
    )
    model.save_pretrained(bert_dir)
    tokenizer.save_pretrained(bert_dir)
    # Ajuste permitido exclusivamente da fixture sintética; nunca dos modelos reais.
    svm = make_pipeline(TfidfVectorizer(), LinearSVC()).fit(
        ["produto ruim", "produto regular", "produto ótimo"], [0, 1, 2]
    )
    joblib.dump(svm, svm_dir / "selected_model.joblib")
    svm_sha = file_sha256(svm_dir / "selected_model.joblib")
    svm_record = {"model_artifact": {"sha256": svm_sha}}
    (svm_dir / "results.json").write_text(json.dumps(svm_record))
    (tmp_path / "svm_record.json").write_text(json.dumps(svm_record))
    (tmp_path / "bert_record.json").write_text(
        json.dumps(
            {
                "saved_model_sha256": file_sha256(bert_dir / "model.safetensors"),
                "best_checkpoint": "synthetic/checkpoint-5543",
            }
        )
    )
    inventory = {f"{MODEL_PREFIX}/{p.name}": file_sha256(p) for p in bert_dir.iterdir()}
    inventory["models/baseline/phase2-v1/selected_model.joblib"] = svm_sha
    (tmp_path / "final_record.json").write_text(json.dumps({"inventory": inventory}))
    return ArtifactPaths(
        bert_dir,
        svm_dir,
        tmp_path / "bert_record.json",
        tmp_path / "svm_record.json",
        tmp_path / "final_record.json",
    )


@pytest.mark.parametrize("text", [None, 5, [], {}, b"texto", "", " ", "\n\t", "x" * 8001])
def test_invalid_input_does_not_load_models(text, monkeypatch):
    service = InferenceService()
    monkeypatch.setattr(service, "_load_bert", lambda: pytest.fail("Carregamento indevido"))
    with pytest.raises(InputError):
        service.predict(text)
    assert service.loaded_models == ()


def test_preserves_accents_case_negation_and_original_whitespace():
    text = "  Não é ÓTIMO! 😕\n"
    assert validate_text(text) == text
    assert len(validate_text("á" * MAX_INPUT_CHARACTERS)) == MAX_INPUT_CHARACTERS


@pytest.mark.parametrize("model", ["other", None, [], {}])
def test_invalid_model(model):
    with pytest.raises(InputError):
        InferenceService().predict("Avaliação válida", model=model)


@pytest.mark.parametrize("values", [[], "string", ["texto"] * 17])
def test_batch_limits(values):
    with pytest.raises(InputError):
        InferenceService().predict_batch(values)


def test_score_order_sum_mapping_and_numerical_stability():
    rows = normalized_scores([[10000, 10001, 9999], [2, 2, 2]], 2)
    assert [score.class_id for score in rows[0]] == [1, 0, 2]
    assert [score.label for score in rows[0]] == ["neutro", "negativo", "positivo"]
    assert sum(score.value for score in rows[0]) == pytest.approx(1)
    assert [score.class_id for score in rows[1]] == [0, 1, 2]


@pytest.mark.parametrize("logits", [[[0, np.inf, 1]], [[0, np.nan, 1]], [[0, 1]], []])
def test_invalid_logits(logits):
    with pytest.raises(InferenceError):
        normalized_scores(logits, 1)


def test_cpu_inference_lazy_cache_response_and_unchanged_weights(tiny_paths, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    service = InferenceService(tiny_paths, device="auto")
    assert service.loaded_models == ()
    create = service._create_bert
    calls = []

    def load(*args, **kwargs):
        calls.append(args[0])
        return create(*args, **kwargs)

    monkeypatch.setattr(service, "_create_bert", load)
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("Backward indevido"))
    first = service.predict("Não é ótimo")
    before = {key: tensor.clone() for key, tensor in service._bert.model.state_dict().items()}
    result = service.predict("Não é ótimo")
    assert first.device == result.device == "cpu"
    assert first.cold_start_seconds > 0 and result.cold_start_seconds == 0
    assert result.precision == "fp32" and result.score_type == "softmax_uncalibrated"
    assert result.class_id in LABEL_NAMES and result.label == LABEL_NAMES[result.class_id]
    assert result.scores[0].class_id == result.class_id
    assert result.inference_seconds > 0 and result.batch_size == 1
    assert first.scores == result.scores
    assert calls == ["cpu"] and service.loaded_models == ("bertimbau",)
    assert not service._bert.model.training
    assert all(
        torch.equal(before[key], tensor) for key, tensor in service._bert.model.state_dict().items()
    )
    assert "Não é ótimo" not in json.dumps(result.to_dict(), ensure_ascii=False)
    assert isinstance(result.to_dict()["scores"], list)
    assert (
        service._bert.tokenizer("Não")["input_ids"] != service._bert.tokenizer("não")["input_ids"]
    )


def test_svm_no_probabilities_or_refit_and_only_loads_once(tiny_paths, monkeypatch):
    service = InferenceService(tiny_paths)
    first = service.predict("produto ótimo", model="svm")
    monkeypatch.setattr(service, "_load_svm", lambda: pytest.fail("Recarga indevida"))
    monkeypatch.setattr(service._svm, "fit", lambda *a, **k: pytest.fail("Fit indevido"))
    monkeypatch.setattr(
        service._svm.steps[0][1], "fit_transform", lambda *a, **k: pytest.fail("Refit indevido")
    )
    monkeypatch.setattr(
        service._svm, "decision_function", lambda *a, **k: pytest.fail("Margens não solicitadas")
    )
    results = service.predict_batch(
        ["produto ruim", "produto regular", "produto ótimo"], model="svm"
    )
    assert [r.class_id for r in results] == [0, 1, 2]
    assert all(not r.scores and r.score_type is None and r.device == "cpu" for r in results)
    assert service.loaded_models == ("svm",) and first.cold_start_seconds > 0


@pytest.mark.parametrize(
    "name", ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]
)
def test_changed_bert_artifact_rejected_before_loading(tiny_paths, name, monkeypatch):
    path = tiny_paths.bert_dir / name
    path.write_bytes(path.read_bytes() + b"changed")
    service = InferenceService(tiny_paths, device="cpu")
    monkeypatch.setattr(
        service, "_create_bert", lambda *a, **k: pytest.fail("Carregamento de artefato adulterado")
    )
    with pytest.raises(ArtifactError, match="integridade"):
        service.predict("Não é ótimo")


def test_missing_artifacts_and_sanitized_paths(tiny_paths):
    paths = replace(tiny_paths, bert_dir=tiny_paths.bert_dir / "missing")
    with pytest.raises(ArtifactError) as caught:
        InferenceService(paths).predict("Não é ótimo")
    assert "README" in str(caught.value) and str(tiny_paths.bert_dir) not in str(caught.value)


def test_tampered_svm_and_incorrect_mapping(tiny_paths):
    path = tiny_paths.svm_dir / "selected_model.joblib"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ArtifactError, match="integridade"):
        InferenceService(tiny_paths).predict("produto bom", model="svm")
    config = json.loads((tiny_paths.bert_dir / "config.json").read_text())
    config["id2label"] = {"0": "positivo", "1": "neutro", "2": "negativo"}
    (tiny_paths.bert_dir / "config.json").write_text(json.dumps(config))
    with pytest.raises(ArtifactError, match="mapeamento"):
        InferenceService(tiny_paths)._create_bert("cpu")


def test_explicit_cuda_unavailable(tiny_paths, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(DeviceError, match="CUDA indisponível"):
        InferenceService(tiny_paths, device="cuda").predict("Não é ótimo")


def test_cuda_initialization_failure_recovers_once_without_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    service = InferenceService()
    monkeypatch.setattr(service, "_verify_bert", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def create(device, *, notes=()):
        calls.append(device)
        if device.startswith("cuda"):
            raise torch.cuda.OutOfMemoryError("CUDA /private/path secret")
        return _BertBackend(None, None, "cpu", False, notes)

    monkeypatch.setattr(service, "_create_bert", create)
    backend = service._load_bert()
    assert calls == ["cuda:0", "cpu"] and backend.device == "cpu"
    assert "/private" not in repr(backend.notes)


@pytest.mark.parametrize("preference", ["auto", "cuda"])
def test_cuda_prediction_failure_is_bounded_and_safe(monkeypatch, preference):
    service = InferenceService(device=preference)
    service._bert = _BertBackend(None, None, "cuda:0", True)
    recovered = []

    def logits(backend, texts):
        if backend.device.startswith("cuda"):
            raise RuntimeError("CUDA /private/path user text")
        return np.tile([0, 1, 2], (len(texts), 1))

    def recover():
        recovered.append(1)
        return _BertBackend(None, None, "cpu", False)

    monkeypatch.setattr(service, "_bert_logits", logits)
    monkeypatch.setattr(service, "_recover_cpu", recover)
    if preference == "cuda":
        with pytest.raises(DeviceError) as caught:
            service.predict("segredo")
        assert "/private" not in str(caught.value) and not recovered
    else:
        assert service.predict("segredo").device == "cpu"
        assert service.predict("segredo").device == "cpu"
        assert recovered == [1]


def test_invalid_logits_do_not_trigger_device_recovery(monkeypatch):
    service = InferenceService()
    service._bert = _BertBackend(None, None, "cuda:0", True)
    monkeypatch.setattr(service, "_bert_logits", lambda *a: np.array([[np.inf, 1, 2]]))
    monkeypatch.setattr(service, "_recover_cpu", lambda: pytest.fail("Recuperação indevida"))
    with pytest.raises(InferenceError):
        service.predict("texto")


def test_no_dataset_access_during_service_prediction(tiny_paths):
    from reviewsense.final_evaluation import test_access_policy

    path = Path(__file__).resolve().parents[2] / "data/processed/phase1-final/test.csv"
    with test_access_policy(path) as access:
        service = InferenceService(tiny_paths, device="cpu")
        service.predict("produto ótimo", model="svm")
        service.predict("Não é ótimo")
        assert access["read_opens"] == access["blocked_opens"] == 0


def test_backend_exception_never_echoes_text_or_paths(monkeypatch, capsys, caplog):
    service = InferenceService()
    service._svm = SimpleNamespace(
        predict=lambda _: (_ for _ in ()).throw(RuntimeError("sensitive_text /home/private/token"))
    )
    with pytest.raises(InferenceError) as caught:
        service.predict("sensitive_text", model="svm")
    assert "sensitive_text" not in str(caught.value) and "/home/" not in str(caught.value)
    captured = capsys.readouterr()
    assert "sensitive_text" not in captured.out + captured.err + caplog.text


def test_long_batch_uses_max_256_tokens_and_controlled_microbatches(tiny_paths):
    service = InferenceService(tiny_paths, device="cpu")
    service.predict("Não é ótimo")
    shapes = []
    handle = service._bert.model.register_forward_pre_hook(
        lambda model, args, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    try:
        results = service.predict_batch([("Não é ótimo " * 700)[:8000]] * 3)
    finally:
        handle.remove()
    assert shapes == [(2, 256), (1, 256)]
    assert len(results) == 3 and all(result.batch_size == 3 for result in results)
    assert all(sum(s.value for s in result.scores) == pytest.approx(1) for result in results)


def test_concurrent_requests_share_one_lazy_model(monkeypatch):
    service = InferenceService()
    calls = []

    def load():
        calls.append(1)
        return SimpleNamespace(predict=lambda texts: np.zeros(len(texts), dtype=int))

    monkeypatch.setattr(service, "_load_svm", load)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: service.predict("Produto sintético", model="svm"), range(4))
        )
    assert calls == [1] and all(result.class_id == 0 for result in results)
