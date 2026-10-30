"""Pacotes sintéticos e BERT minúsculo; sem baixar pesos ou abrir teste real."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from reviewsense.labels import LABEL_NAMES
from reviewsense.model_packaging import (
    ARTIFACT_FILES,
    PACKAGE_FILES,
    build_package,
    compare_outputs,
    file_inventory,
    local_logits,
    read_json,
    safe_file,
    validate_configuration,
    validate_manifest,
    validate_model_card,
    validate_package,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def inputs(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    source, documents = tmp_path / "original", tmp_path / "documents"
    source.mkdir()
    shutil.copytree(ROOT / "publishing/model_card", documents)
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "O", "produto", "é", "Não", "bom"]
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
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    (source / "training_args.bin").write_bytes(b"synthetic, never export")
    expected = {
        name: entry["sha256"] for name, entry in file_inventory(source, ARTIFACT_FILES).items()
    }
    # Agregados versionados, não CSVs ou pesos reais.
    metrics = read_json(ROOT / "reports/phase3c_final_metrics.json")["models"]["bertimbau"][
        "metrics"
    ]
    return {
        "source": source,
        "documents": documents,
        "expected": expected,
        "metrics": metrics,
        "provenance": {"packaging_code_commit": "synthetic"},
    }


def build(inputs, destination):
    return build_package(destination=destination, **inputs)


def test_exact_allowlist_bytes_and_deterministic_manifest(inputs, tmp_path):
    original = file_inventory(inputs["source"], ARTIFACT_FILES)
    a, b = tmp_path / "package-a", tmp_path / "package-b"
    first, second = build(inputs, a), build(inputs, b)
    assert first == second
    assert set(p.name for p in a.iterdir()) == set(PACKAGE_FILES)
    assert not (a / "training_args.bin").exists()
    assert file_inventory(inputs["source"], ARTIFACT_FILES) == original
    assert file_inventory(a, ARTIFACT_FILES) == original
    assert first["license_status"] == "REQUIRES_REVIEW" and first["upload_authorized"] is False
    assert first["huggingface_repo_id"] is None and str(tmp_path) not in json.dumps(first)
    validate_package(a, first)


@pytest.mark.parametrize("name", ARTIFACT_FILES)
def test_missing_artifacts(inputs, tmp_path, name):
    (inputs["source"] / name).unlink()
    with pytest.raises(ValueError):
        build(inputs, tmp_path / "package")
    assert not (tmp_path / "package").exists()


@pytest.mark.parametrize("name", ["test.csv", "optimizer.pt", ".env", "surprise.json"])
def test_unexpected_source_files(inputs, tmp_path, name):
    (inputs["source"] / name).write_text("synthetic")
    with pytest.raises(ValueError, match="inesperados"):
        build(inputs, tmp_path / "package")


def test_wrong_checkpoint(inputs, tmp_path):
    inputs["expected"]["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint"):
        build(inputs, tmp_path / "package")


@pytest.mark.parametrize("mode", ["file", "source", "document", "destination_parent", "dangling"])
def test_symlinks_rejected(inputs, tmp_path, mode):
    destination = tmp_path / "package"
    if mode in {"file", "document"}:
        path = (
            inputs["source"] / "config.json"
            if mode == "file"
            else inputs["documents"] / "README.md"
        )
        external = tmp_path / "external"
        path.rename(external)
        path.symlink_to(external)
    elif mode == "source":
        alias = tmp_path / "alias"
        alias.symlink_to(inputs["source"], target_is_directory=True)
        inputs["source"] = alias
    elif mode == "destination_parent":
        alias = tmp_path / "parent"
        alias.symlink_to(tmp_path, target_is_directory=True)
        destination = alias / "package"
    else:
        destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        build(inputs, destination)


def test_existing_destination_overlap_and_traversal(inputs, tmp_path):
    destination = tmp_path / "package"
    manifest = build(inputs, destination)
    with pytest.raises(FileExistsError):
        build(inputs, destination)
    validate_package(destination, manifest)
    with pytest.raises(ValueError):
        safe_file(inputs["source"], "../outside.json")
    with pytest.raises(ValueError):
        build(inputs, inputs["source"] / "subdir")


@pytest.mark.parametrize("mutation", ["missing", "extra", "tampered", "manifest_hash"])
def test_bad_package_rejected(inputs, tmp_path, mutation):
    destination = tmp_path / "package"
    manifest = build(inputs, destination)
    if mutation == "missing":
        (destination / "tokenizer.json").unlink()
    elif mutation == "extra":
        (destination / "test.csv").write_text("synthetic")
    elif mutation == "tampered":
        (destination / "model.safetensors").write_bytes(b"tampered synthetic weights")
    else:
        manifest["files"]["README.md"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_package(destination, manifest)


@pytest.mark.parametrize("mutation", ["mapping", "lowercase", "remote_code"])
def test_configuration_policy(inputs, mutation):
    path = inputs["source"] / (
        "tokenizer_config.json" if mutation == "lowercase" else "config.json"
    )
    config = read_json(path)
    if mutation == "mapping":
        config["id2label"] = {"0": "positivo", "1": "neutro", "2": "negativo"}
    elif mutation == "lowercase":
        config["do_lower_case"] = True
    else:
        config["auto_map"] = {"AutoModelForSequenceClassification": "untrusted.CustomModel"}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        validate_configuration(inputs["source"])


@pytest.mark.parametrize("mutation", ["license", "metric", "section", "credential", "path"])
def test_card_structure_metrics_and_metadata_hygiene(inputs, mutation):
    card = (inputs["documents"] / "README.md").read_text()
    if mutation == "license":
        card = card.replace("license: unknown", "license: mit")
    elif mutation == "metric":
        card = card.replace(str(inputs["metrics"]["f1_macro"]), "0.9")
    elif mutation == "section":
        card = card.replace("## Limitações", "## Removed")
    elif mutation == "credential":
        card += "\nhf_" + "a" * 30
    else:
        card += "\n/home/private-person/model"
    with pytest.raises(ValueError):
        validate_model_card(card, inputs["metrics"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("upload_authorized", True),
        ("license_status", "READY"),
        ("huggingface_repo_id", "invented/target"),
        ("max_length", 128),
    ],
)
def test_manifest_publication_block(inputs, tmp_path, field, value):
    manifest = copy.deepcopy(build(inputs, tmp_path / "package"))
    manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_numeric_tolerances_without_gpu():
    a = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    b = a + 0.0002
    result = compare_outputs(a, b, device="cuda", precision="fp16")
    assert result["same_predicted_classes"] and result["logits_allclose"]
    assert result["max_absolute_logit_difference"] > 0
    with pytest.raises(ValueError):
        compare_outputs(a, b, device="cpu", precision="fp32")
    for bad in (np.full_like(a, np.nan), a[:, ::-1], np.zeros((2, 2))):
        with pytest.raises(ValueError):
            compare_outputs(a, bad, device="cpu", precision="fp32")


def test_tiny_model_equivalence_no_training_or_upload(inputs, tmp_path, monkeypatch):
    import torch
    from huggingface_hub import HfApi, ModelCard

    def forbidden(*args, **kwargs):
        pytest.fail("Treino/upload/validação remota não permitidos")

    for method in ("create_repo", "upload_folder", "upload_file"):
        monkeypatch.setattr(HfApi, method, forbidden)
    monkeypatch.setattr(ModelCard, "validate", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    destination = tmp_path / "package"
    manifest = build(inputs, destination)
    texts = ["O produto é bom.", "Não é bom.", "O produto " * 300]
    a, config_a = local_logits(inputs["source"], texts, device="cpu", precision="fp32")
    b, config_b = local_logits(destination, texts, device="cpu", precision="fp32")
    assert config_a == config_b and config_b["maximum_padded_tokens"] == 256
    assert compare_outputs(a, b, device="cpu", precision="fp32")["same_predicted_classes"]
    validate_package(destination, manifest)


def test_independent_card_example_empty_cache(inputs, tmp_path):
    from scripts.verify_phase4b_package import verify_independent_example

    destination = tmp_path / "package"
    build(inputs, destination)
    proof = verify_independent_example(destination, ROOT)
    assert proof["literal_model_card_example_executed"]
    assert proof["original_models_reports_data_accesses"] == proof["network_connections"] == 0


def test_no_real_test_access(inputs, tmp_path):
    from reviewsense.final_evaluation import test_access_policy

    with test_access_policy(ROOT / "data/processed/phase1-final/test.csv") as guard:
        manifest = build(inputs, tmp_path / "package")
        validate_package(tmp_path / "package", manifest)
        assert guard["read_opens"] == guard["blocked_opens"] == 0
