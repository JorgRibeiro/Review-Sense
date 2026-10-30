"""Avaliação final testada apenas com dados/artefatos sintéticos locais."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from reviewsense import final_evaluation as final
from reviewsense.baseline import load_experiment
from reviewsense.data import build_manifest, dataset_fingerprint, file_sha256
from reviewsense.final_reporting import render_report
from reviewsense.labels import LABEL_NAMES


@pytest.fixture
def frozen(tmp_path):
    splits = {}
    for split in ("train", "val", "test"):
        texts = [f"Avaliação {word} do split {split}." for word in ("ruim", "regular", "ótima")]
        splits[split] = pd.DataFrame(
            {
                "sample_id": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
                "review_text": texts,
                "label": [0, 1, 2],
            }
        )
    combined = pd.concat(splits.values(), ignore_index=True)
    manifest = build_manifest(combined, splits, source={"kind": "synthetic"}, config={})
    reference = json.loads(json.dumps(manifest))
    for info in reference["splits"].values():
        del info["sample_ids"]
    for name, frame in splits.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "reference.json").write_text(json.dumps(reference))
    config = {"data_dir": str(tmp_path), "reference": str(tmp_path / "reference.json")}
    return config, splits, manifest, reference


def test_preflight_metadata_never_opens_test(frozen):
    config, _, _, _ = frozen
    with final.test_access_policy(Path(config["data_dir"]) / "test.csv") as access:
        train, val, data = final.validate_manifest_metadata(config)
        assert len(train) == len(val) == 3
        assert data["test_csv_opened"] is False
        assert access["read_opens"] == access["blocked_opens"] == 0


@pytest.mark.parametrize("mode", ["r", "rb", "w", "a"])
def test_preflight_blocks_all_test_access(tmp_path, mode):
    path = tmp_path / "test.csv"
    path.write_text("synthetic")
    with final.test_access_policy(path) as access:
        with pytest.raises(PermissionError), path.open(mode):
            pass
    assert access["blocked_opens"] == 1 and access["read_opens"] == 0
    assert path.read_text() == "synthetic"


def test_final_guard_allows_only_one_read_and_no_write(tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("synthetic")
    with final.test_access_policy(path, final=True) as access:
        with pytest.raises(PermissionError), path.open("w"):
            pass
        assert path.read_bytes() == b"synthetic"
        with pytest.raises(PermissionError):
            path.read_bytes()
    assert access["read_opens"] == 1 and access["blocked_opens"] == 2


def test_final_integrity_and_manifest_order(frozen):
    _, frames, manifest, reference = frozen
    summary = final.validate_final_frame(
        frames["test"], frames["train"], frames["val"], manifest, reference
    )
    assert summary["fingerprint"] == dataset_fingerprint(frames["test"])
    with pytest.raises(ValueError, match="Ordem"):
        final.validate_final_frame(
            frames["test"].iloc[::-1], frames["train"], frames["val"], manifest, reference
        )


@pytest.mark.parametrize("mutation", ["label", "missing", "text", "id", "duplicate", "schema"])
def test_rejects_invalid_final_frame(frozen, mutation):
    _, frames, manifest, reference = frozen
    frame = frames["test"].copy()
    if mutation == "label":
        frame.loc[0, "label"] = 5
    elif mutation == "missing":
        frame.loc[0, "review_text"] = None
    elif mutation == "text":
        frame.loc[0, "review_text"] += "  "
    elif mutation == "id":
        frame.loc[0, "sample_id"] = "invalid"
    elif mutation == "duplicate":
        frame.iloc[0] = frame.iloc[1]
    else:
        frame = frame[["review_text", "label", "sample_id"]]
    with pytest.raises(ValueError):
        final.validate_final_frame(frame, frames["train"], frames["val"], manifest, reference)


def test_rejects_fingerprint_distribution_and_overlap(frozen):
    _, frames, manifest, reference = frozen
    reference["splits"]["test"]["fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        final.validate_final_frame(
            frames["test"], frames["train"], frames["val"], manifest, reference
        )
    reference["splits"]["test"]["fingerprint"] = dataset_fingerprint(frames["test"])
    with pytest.raises(ValueError, match="Vazamento"):
        final.validate_final_frame(
            frames["test"], frames["test"], frames["val"], manifest, reference
        )


def test_predictions_identity_order_and_invalid_classes():
    frame = pd.DataFrame({"sample_id": ["a", "b", "c"]})
    p = final.Predictions(tuple(frame.sample_id), np.array([0, 1, 2]))
    assert np.array_equal(p.aligned(frame), [0, 1, 2])
    with pytest.raises(ValueError, match="ordem"):
        replace(p, ids=("b", "a", "c")).aligned(frame)
    for labels in ([0, 1], [0, 1, 3], [0.0, 1.0, 2.0], [0, 1, np.nan]):
        with pytest.raises(ValueError):
            final.validate_labels(labels, 3)


@pytest.mark.parametrize(
    "logits", [np.zeros((2, 2)), np.full((2, 3), np.inf), np.full((2, 3), np.nan)]
)
def test_rejects_nonfinite_or_wrong_logits(logits):
    with pytest.raises(ValueError):
        final.validate_logits(logits, 2)


def test_synthetic_svm_loading_checksum_and_prediction_without_fit(tmp_path, monkeypatch):
    pipeline = make_pipeline(TfidfVectorizer(), LinearSVC())
    # Ajuste apenas da fixture sintética; proibido nos modelos reais da Fase 3C.
    pipeline.fit(["produto ruim", "produto regular", "produto ótimo"], [0, 1, 2])
    path = tmp_path / "selected_model.joblib"
    joblib.dump(pipeline, path)
    (tmp_path / "results.json").write_text(
        json.dumps({"model_artifact": {"sha256": file_sha256(path)}})
    )
    loaded, _ = load_experiment(tmp_path)
    monkeypatch.setattr(loaded, "fit", lambda *a, **k: pytest.fail("Fit indevido"))
    monkeypatch.setattr(
        loaded.steps[0][1], "fit_transform", lambda *a, **k: pytest.fail("Refit indevido")
    )
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "review_text": ["produto ruim", "produto regular", "produto ótimo"],
        }
    )
    predictions = final.predict_svm(loaded, frame, {"svm_threads": 1, "svm_batch_size": 2})
    assert np.array_equal(predictions.aligned(frame), [0, 1, 2])
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Checksum"):
        load_experiment(tmp_path)


def test_local_tiny_bert_inference_preserves_weights_and_casing(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Não", "é", "ótimo", "ruim"]
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
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    loaded, tokenizer = final.load_local_bert(tmp_path, "cpu")
    original = {k: v.clone() for k, v in loaded.state_dict().items()}
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("Backward indevido"))
    frame = pd.DataFrame(
        {"sample_id": ["a", "b", "c"], "review_text": ["Não é ótimo", "ruim", "ótimo"]}
    )
    config = {"bert_batch_size": 2, "max_length": 256}
    a = final.predict_bert(loaded, tokenizer, frame, config).aligned(frame)
    b = final.predict_bert(loaded, tokenizer, frame, config).aligned(frame)
    assert np.array_equal(a, b) and not loaded.training
    assert all(torch.equal(v, original[k]) for k, v in loaded.state_dict().items())
    assert tokenizer("Não")["input_ids"] != tokenizer("não")["input_ids"]
    loaded.config.id2label = {0: "positivo", 1: "neutro", 2: "negativo"}
    with pytest.raises(ValueError, match="Mapeamentos"):
        final.validate_mappings(loaded, tokenizer)


def test_mapping_and_tokenizer_rejection():
    config = SimpleNamespace(
        id2label=LABEL_NAMES, label2id={v: k for k, v in LABEL_NAMES.items()}, num_labels=3
    )
    with pytest.raises(ValueError, match="cased"):
        final.validate_mappings(SimpleNamespace(config=config), SimpleNamespace(do_lower_case=True))


def test_aggregated_metrics_agreement_and_rendering(tmp_path):
    # Quatro combinações pareadas, exatamente uma ocorrência de cada.
    result = final.summarize(
        [0, 1, 2, 0], np.array([0, 0, 2, 1]), np.array([0, 1, 0, 1]), {"svm": {}, "bertimbau": {}}
    )
    assert result["paired_agreement"] == {
        "both_correct": 1,
        "both_wrong": 1,
        "only_svm_correct": 1,
        "only_bertimbau_correct": 1,
        "samples": 4,
    }
    final.validate_summary(result)
    assert result["models"]["svm"]["metrics"]["confusion_matrix"] == [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]
    render_report(result, tmp_path, detailed=False)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 4
    result["models"]["svm"]["metrics"]["per_class"]["neutro"]["support"] = 100
    with pytest.raises(ValueError, match="inconsistentes"):
        final.validate_summary(result)


def test_atomic_writes_no_overwrite_and_no_partial_report(tmp_path):
    path = tmp_path / "result.json"
    with pytest.raises(ValueError):
        final.atomic_json(path, {"bad": float("nan")})
    assert not path.exists() and not list(tmp_path.iterdir())
    final.atomic_json(path, {"valid": True})
    with pytest.raises(FileExistsError):
        final.atomic_json(path, {"valid": False})
    assert final.read_json(path) == {"valid": True}


def test_explicit_authorization_and_single_start(tmp_path):
    with pytest.raises(PermissionError):
        final.begin_final(tmp_path, authorized=False, metadata={})
    assert not (tmp_path / "started.json").exists()
    final.begin_final(tmp_path, authorized=True, metadata={})
    with pytest.raises(FileExistsError):
        final.begin_final(tmp_path, authorized=True, metadata={})
    with pytest.raises(PermissionError):
        final.execute_final(tmp_path / "absent.json")


def test_failure_after_synthetic_test_read_is_recorded_and_not_retried(frozen, monkeypatch):
    config, frames, _, _ = frozen
    config["run_dir"] = str(Path(config["data_dir"]) / "run")
    state = {"commit": "synthetic", "branch": "synthetic", "clean": True}
    approved = {
        "status": "approved",
        "git": state,
        "inventory": {},
        "software": {},
        "test_access": {"read_opens": 0, "blocked_opens": 0},
    }
    final.atomic_json(Path(config["run_dir"]) / "preflight.json", approved)
    monkeypatch.setattr(final, "read_config", lambda path: config)
    monkeypatch.setattr(final, "git_state", lambda: state)
    monkeypatch.setattr(final, "versions", lambda: {})
    monkeypatch.setattr(final, "artifact_inventory", lambda *args: {})
    monkeypatch.setattr(final, "load_models", lambda config: (None, None, None, {}))
    monkeypatch.setattr(
        final, "validate_manifest_metadata", lambda config: (frames["train"], frames["val"], {})
    )
    # Integridade passa, mas tamanho sintético 3 != 19.004: falha depois da única leitura.
    with pytest.raises(ValueError, match="Número"):
        final.execute_final(Path("synthetic.json"), authorized=True)
    failure = final.read_json(Path(config["run_dir"]) / "failure.json")
    assert failure["test_read_opens"] == 1
    assert failure["status"] == "failed_no_automatic_retry"
    with pytest.raises(FileExistsError):
        final.execute_final(Path("synthetic.json"), authorized=True)
    assert not (Path(config["run_dir"]) / "results.json").exists()


def test_cli_requires_separate_explicit_operation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["final_evaluation"])
    with pytest.raises(SystemExit) as error:
        final.main()
    assert error.value.code == 2


def test_preflight_guard_covers_model_loading(frozen, monkeypatch):
    config, frames, _, _ = frozen
    config.update({"run_dir": str(Path(config["data_dir"]) / "run"), "expected_versions": {}})
    monkeypatch.setattr(final, "read_config", lambda _: config)
    monkeypatch.setattr(final, "versions", lambda: {})
    monkeypatch.setattr(final, "artifact_inventory", lambda *args: {})
    monkeypatch.setattr(
        final, "validate_manifest_metadata", lambda _: (frames["train"], frames["val"], {})
    )
    monkeypatch.setattr(
        final, "load_models", lambda _: (Path(config["data_dir"]) / "test.csv").read_text()
    )
    with pytest.raises(PermissionError):
        final.preflight(Path("synthetic.json"))
    assert not (Path(config["run_dir"]) / "preflight.json").exists()


def test_inventory_rejects_changed_model_and_tokenizer_checksums(monkeypatch):
    config = {
        "data_dir": "synthetic",
        "reference": "synthetic/reference.json",
        "protocol": "synthetic/protocol.md",
        "svm_dir": "synthetic/svm",
        "bert_dir": "synthetic/bert",
        "bert_checkpoint": "synthetic/ckpt",
        "svm_reference": "synthetic/svm.json",
        "bert_reference": "synthetic/bert.json",
        "svm_sha256": "expected",
        "bert_sha256": "expected",
        "bert_file_sha256": {"tokenizer.json": "expected"},
    }
    monkeypatch.setattr(final, "file_sha256", lambda _: "expected")
    assert final.artifact_inventory(config, Path("synthetic/config.json"))
    monkeypatch.setattr(
        final, "file_sha256", lambda p: "bad" if p.name == "tokenizer.json" else "expected"
    )
    with pytest.raises(ValueError, match="Checksum"):
        final.artifact_inventory(config, Path("synthetic/config.json"))


def test_detailed_report_and_all_five_figures_from_synthetic_aggregates(tmp_path):
    gpu_memory = {"peak_allocated_bytes": 100, "peak_reserved_bytes": 200}
    driver = {"peak_used_bytes_sampled": 300, "min_free_bytes": 500}
    timing = {
        "seconds": 1.0,
        "texts_per_second": 3.0,
        "milliseconds_per_text_amortized": 1000 / 3,
        "process_rss_peak_sampled_bytes": 400,
        "pytorch": gpu_memory,
        "driver": driver,
    }
    result = final.summarize(
        [0, 1, 2], np.array([0, 1, 2]), np.array([0, 1, 2]), {"svm": timing, "bertimbau": timing}
    )
    benchmark = {
        name: {
            "batch_size": 2,
            "median_seconds": 1,
            "texts_per_second": 3,
            "milliseconds_per_text_amortized": 1000 / 3,
        }
        for name in ("svm", "bertimbau")
    }
    result.update(
        {
            "started_at": "synthetic",
            "finished_at": "synthetic",
            "inventory": {},
            "git": {"commit": "synthetic", "branch": "synthetic"},
            "test": {
                "rows": 3,
                "preparation_read_opens": 0,
                "read_opens": 1,
                "blocked_opens": 0,
                "csv_sha256": "synthetic",
                "fingerprint": "synthetic",
            },
            "historical_training": {
                "cpu_model": "synthetic",
                "logical_cpus": 1,
                "svm_fit_seconds": 1,
                "svm_search_validation_seconds": 2,
                "bert_three_epochs_validation_checkpoints_seconds": 3,
                "artifact_bytes": {"svm_joblib": 1, "bert_weights": 2, "bert_tokenizer": 3},
            },
            "preflight": {"benchmark": benchmark},
            "gpu": {"name": "synthetic", "cuda_build": "synthetic"},
            "software": {
                key: "synthetic" for key in ("python", "torch", "transformers", "scikit-learn")
            },
        }
    )
    text = render_report(result, tmp_path)
    assert "0.000000" in text
    assert len(list((tmp_path / "figures").glob("*.png"))) == 5
    assert "both_correct" not in text  # Identificadores internos são apresentados em português.
    final.atomic_json(tmp_path / "aggregate.json", result)
    final.validate_summary(final.read_json(tmp_path / "aggregate.json"))


def test_successful_single_evaluation_with_19004_synthetic_rows(frozen, monkeypatch):
    config, frames, manifest, reference = frozen
    directory = Path(config["data_dir"])
    texts = [f"Avaliação sintética número {i}, não contém dados reais." for i in range(19004)]
    frame = pd.DataFrame(
        {
            "sample_id": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
            "review_text": texts,
            "label": np.arange(19004) % 3,
        }
    )
    frame.to_csv(directory / "test.csv", index=False)
    combined = pd.concat([frames["train"], frames["val"], frame], ignore_index=True)
    counts = frame.label.value_counts()
    summary = {
        "rows": len(frame),
        "fingerprint": dataset_fingerprint(frame),
        "class_distribution": {
            name: {"count": int(counts[label]), "fraction": float(counts[label] / len(frame))}
            for label, name in LABEL_NAMES.items()
        },
    }
    manifest["splits"]["test"] = {**summary, "sample_ids": frame.sample_id.tolist()}
    reference["splits"]["test"] = summary
    manifest["prepared_fingerprint"] = reference["prepared_fingerprint"] = dataset_fingerprint(
        combined
    )
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "reference.json").write_text(json.dumps(reference))
    config["run_dir"] = str(directory / "run")
    state = {"commit": "synthetic", "branch": "synthetic", "clean": True}
    final.atomic_json(
        Path(config["run_dir"]) / "preflight.json",
        {
            "status": "approved",
            "git": state,
            "inventory": {},
            "software": {},
            "test_access": {"read_opens": 0, "blocked_opens": 0},
        },
    )
    monkeypatch.setattr(final, "read_config", lambda _: config)
    monkeypatch.setattr(final, "git_state", lambda: state)
    monkeypatch.setattr(final, "versions", lambda: {})
    monkeypatch.setattr(final, "artifact_inventory", lambda *args: {})
    monkeypatch.setattr(final, "load_models", lambda _: (None, None, None, {}))
    monkeypatch.setattr(final, "historical_training", lambda _: {})
    calls = []

    def predict_without_real_models(_, incoming, *, gpu):
        calls.append(gpu)
        return final.Predictions(tuple(incoming.sample_id), incoming.label.to_numpy()), {}

    monkeypatch.setattr(final, "timed_prediction", predict_without_real_models)
    result = final.execute_final(Path("synthetic.json"), authorized=True)
    assert result["status"] == "completed" and result["test"]["read_opens"] == 1
    assert result["test"]["preparation_read_opens"] == 0
    assert result["test"]["csv_sha256"] == file_sha256(directory / "test.csv")
    assert calls == [False, True]
    assert result["paired_agreement"]["both_correct"] == 19004
    final.validate_summary(final.read_json(Path(config["run_dir"]) / "results.json"))
    with pytest.raises(FileExistsError):
        final.execute_final(Path("synthetic.json"), authorized=True)
    with pytest.raises(FileExistsError):
        final.preflight(Path("synthetic.json"))
