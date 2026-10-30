"""Contrato de distribuição, gates e inferência offline com BERT sintético."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from reviewsense.final_evaluation import test_access_policy
from reviewsense.model_packaging import PACKAGE_FILES, file_inventory
from reviewsense.model_resolution import (
    HubSnapshot,
    LocalPackage,
    canonical_digest,
    release_gates,
    scan_metadata,
    validate_descriptor,
    validate_runtime_package,
)
from reviewsense.predict import ArtifactError, InferenceService
from reviewsense.release_readiness import (
    check_known_links,
    inspect_weight_header,
    preflight,
    require_remote_authorization,
)


def local(candidate, root=None):
    return LocalPackage(
        candidate["package"], candidate["descriptor_path"], root or candidate["package"].parent
    )


def test_local_independent_service_cache_and_unchanged_weights(candidate, monkeypatch):
    import huggingface_hub
    import torch

    from scripts.verify_phase4b_package import no_network

    forbidden = Mock(side_effect=AssertionError("network/train forbidden"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    before = file_inventory(candidate["package"], PACKAGE_FILES)
    service = InferenceService(device="cpu", model_source=local(candidate))
    monkeypatch.setattr("reviewsense.predict._record", forbidden)
    assert service.loaded_models == ()
    with no_network() as network:
        service.prepare()
        prediction = service.predict("Não gostei do produto.")
        cached = service.predict("O produto é bom.")
    assert network["blocked_connections"] == 0
    assert prediction.class_id in {0, 1, 2} and prediction.precision == "fp32"
    assert cached.cold_start_seconds == prediction.cold_start_seconds == 0
    assert len(prediction.scores) == 3 and sum(s.value for s in prediction.scores) == pytest.approx(
        1
    )
    assert file_inventory(candidate["package"], PACKAGE_FILES) == before
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_length", 512),
        ("labels", {"0": "positivo"}),
        ("architecture", "AutoCode"),
        ("model_card", "../README.md"),
        ("extra", "not allowed"),
        ("files", {}),
        ("reference", None),
        ("schema_version", 3),
        ("upload_authorized", True),
        ("license", "mit"),
        ("package_version", "main"),
        ("license_status", "READY"),
    ],
)
def test_descriptor_invalid(candidate, key, value):
    descriptor = copy.deepcopy(candidate["descriptor"])
    descriptor[key] = value
    with pytest.raises(ValueError):
        validate_descriptor(descriptor)


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        "test.csv",
        "optimizer.pt",
        "training_args.bin",
        "pytorch_model.bin",
        "copy.safetensors",
    ],
)
def test_reject_extra_files(candidate, name):
    (candidate["package"] / name).write_text("synthetic")
    with pytest.raises(ValueError, match="inesperados"):
        local(candidate).resolve()


@pytest.mark.parametrize(
    "mode", ["missing", "hash", "symlink", "outside_root", "ancestor", "traversal"]
)
def test_reject_invalid_local(candidate, tmp_path, mode):
    package = candidate["package"]
    if mode == "missing":
        (package / "config.json").unlink()
    elif mode == "hash":
        (package / "config.json").write_text("{}")
    elif mode == "symlink":
        original = package / "config.json"
        original.unlink()
        original.symlink_to(candidate["source"] / "config.json")
    elif mode == "outside_root":
        with pytest.raises(ValueError, match="raiz"):
            local(candidate, tmp_path / "different").resolve()
        return
    elif mode == "ancestor":
        alias = tmp_path / "alias"
        alias.symlink_to(package, target_is_directory=True)
        candidate["package"] = alias
    else:
        candidate["package"] = package / ".." / "package"
    with pytest.raises(ValueError):
        local(candidate).resolve()


@pytest.mark.parametrize("revision", ["main", "latest", "v1", "abc123", "", "A" * 40])
def test_hub_requires_immutable_revision(candidate, tmp_path, revision):
    source = HubSnapshot(
        "example/model", revision, candidate["descriptor_path"], tmp_path, tmp_path
    )
    with pytest.raises(ValueError, match="imutável"):
        source.resolve()


def test_review_blocks_remote_before_any_hub_call(candidate, tmp_path, monkeypatch):
    import huggingface_hub

    spy = Mock(side_effect=AssertionError("unexpected Hub call"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", spy)
    with pytest.raises(PermissionError):
        HubSnapshot(
            "example/model", "a" * 40, candidate["descriptor_path"], tmp_path, tmp_path
        ).resolve()
    with pytest.raises(PermissionError):
        require_remote_authorization(candidate["descriptor"])
    assert all(release_gates(candidate["descriptor"]).values())
    spy.assert_not_called()


def approve_synthetic(candidate, tmp_path):
    # Apenas artefato sintético de teste: não representa autorização do ReviewSense real.
    from hashlib import sha256

    package = candidate["package"]
    card = package / "README.md"
    card.write_text(card.read_text().replace("license: unknown", "license: mit"))
    status = package / "DISTRIBUTION_STATUS.md"
    status.write_text("licensing_status: APPROVED\nupload_authorized: true\n")
    d = candidate["descriptor"]
    d.update(license_status="APPROVED", license="mit", upload_authorized=True)
    d["files"] = file_inventory(package, PACKAGE_FILES)
    d["reference"]["package_content_sha256"] = canonical_digest(d["files"])
    candidate["descriptor_path"].write_text(json.dumps(d))
    root = tmp_path / "approvals"
    root.mkdir()
    (root / "evidence.txt").write_text("Synthetic review only")
    common = {"descriptor_sha256": canonical_digest(d)}
    receipts = {
        "rights": {
            **common,
            "decision": "APPROVED",
            "scope": ["public_weights", "public_inference"],
            "evidence_file": "evidence.txt",
            "evidence_sha256": sha256((root / "evidence.txt").read_bytes()).hexdigest(),
        },
        "destination": {
            **common,
            "repo_id": "example/model",
            "revision": "a" * 40,
            "confirmed": True,
        },
        "human": {
            **common,
            "repo_id": "example/model",
            "revision": "a" * 40,
            "authorized": True,
            "actions": ["download", "public_weights", "public_inference"],
        },
    }
    for name, value in receipts.items():
        (root / f"{name}.json").write_text(json.dumps(value))
    return root


def test_hub_mock_pins_revision_allowlist_no_auth_no_download(candidate, tmp_path, monkeypatch):
    import huggingface_hub

    approvals = approve_synthetic(candidate, tmp_path)
    spy = Mock(return_value=str(candidate["package"]))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", spy)
    source = HubSnapshot(
        "example/model", "a" * 40, candidate["descriptor_path"], approvals, tmp_path
    )
    assert source.resolve() == candidate["package"]
    assert not any(release_gates(candidate["descriptor"], approvals).values())
    kwargs = spy.call_args.kwargs
    assert kwargs["local_files_only"] is True and kwargs["token"] is False
    assert kwargs["revision"] == "a" * 40 and set(kwargs["allow_patterns"]) == set(PACKAGE_FILES)


@pytest.mark.parametrize("receipt", ["rights", "destination", "human"])
def test_each_gate_independent_and_evidence_bound(candidate, tmp_path, receipt):
    approvals = approve_synthetic(candidate, tmp_path)
    (approvals / f"{receipt}.json").unlink()
    gates = release_gates(candidate["descriptor"], approvals)
    assert gates[
        {
            "rights": "LEGAL_REVIEW_REQUIRED",
            "destination": "DESTINATION_UNCONFIRMED",
            "human": "PUBLISH_AUTHORIZATION_REQUIRED",
        }[receipt]
    ]
    with pytest.raises(PermissionError):
        require_remote_authorization(candidate["descriptor"], approvals)


def test_no_simple_authorization_flag_bypasses_evidence(candidate, tmp_path):
    approvals = approve_synthetic(candidate, tmp_path)
    (approvals / "evidence.txt").write_text("tampered")
    assert release_gates(candidate["descriptor"], approvals)["LEGAL_REVIEW_REQUIRED"]
    candidate["descriptor"]["package_version"] = "1.0.0-rc.2"
    assert all(release_gates(candidate["descriptor"], approvals).values())


@pytest.mark.parametrize(
    "secret",
    [
        "hf_" + "X" * 30,
        "ghp_" + "A" * 30,
        "AKIA" + "B" * 16,
        "-----BEGIN PRIVATE KEY-----",
        "password=synthetic-secret-value",
        "/home/synthetic/private",
    ],
)
def test_secret_patterns(secret):
    with pytest.raises(ValueError, match="segredo"):
        scan_metadata(secret)


def test_offline_preflight_links_and_test_guard(candidate):
    import re

    from scripts.verify_phase4b_package import no_network

    urls = set()
    for p in candidate["package"].glob("*.md"):
        urls.update(u.rstrip(".,);") for u in re.findall(r'https?://[^\s<>"\x27`]+', p.read_text()))
    receipt = {"checked_at_utc": "synthetic", "links": [{"url": u, "status": 200} for u in urls]}
    real_test = Path(__file__).resolve().parents[2] / "data/processed/phase1-final/test.csv"
    with test_access_policy(real_test) as guard, no_network() as network:
        result = preflight(
            candidate["package"], candidate["manifest"], candidate["descriptor"], receipt
        )
    assert result["integrity_and_metadata_valid"] and not result["TECHNICALLY_READY"]
    assert result["LEGAL_REVIEW_REQUIRED"] and not result["remote_execution_allowed"]
    assert network["blocked_connections"] == guard["read_opens"] == guard["blocked_opens"] == 0
    with pytest.raises(ValueError, match="evidência"):
        check_known_links(candidate["package"], {"checked_at_utc": "synthetic", "links": []})


def test_weight_header_and_duplicate_metadata(candidate):
    inspect_weight_header(candidate["package"])
    (candidate["package"] / "model.safetensors").write_bytes(b"not weights")
    with pytest.raises(ValueError, match="Header"):
        inspect_weight_header(candidate["package"])


def test_missing_source_sanitized(candidate):
    candidate["package"] = candidate["package"] / "missing"
    service = InferenceService(model_source=local(candidate), device="cpu")
    with pytest.raises(ArtifactError) as caught:
        service.predict("Texto sintético")
    assert str(candidate["package"]) not in str(caught.value)


def test_metadata_secret_detected_with_recomputed_hashes(candidate):
    d = candidate["descriptor"]
    card = candidate["package"] / "ATTRIBUTIONS.md"
    card.write_text(card.read_text() + "\nhf_" + "X" * 30)
    d["files"] = file_inventory(candidate["package"], PACKAGE_FILES)
    d["reference"]["package_content_sha256"] = canonical_digest(d["files"])
    with pytest.raises(ValueError, match="segredo"):
        validate_runtime_package(candidate["package"], d)


def test_duplicate_content_rejected(candidate):
    package, descriptor = candidate["package"], candidate["descriptor"]
    (package / "PROJECT_CODE_LICENSE.txt").write_bytes(
        (package / "BASE_MODEL_LICENSE.txt").read_bytes()
    )
    descriptor["files"] = file_inventory(package, PACKAGE_FILES)
    descriptor["reference"]["package_content_sha256"] = canonical_digest(descriptor["files"])
    with pytest.raises(ValueError, match="duplicados"):
        validate_runtime_package(package, descriptor)


@pytest.mark.parametrize("failure", ["outside", "missing_cache", "wrong_destination"])
def test_approved_mock_snapshot_still_fails_safely(candidate, tmp_path, monkeypatch, failure):
    import huggingface_hub

    approvals = approve_synthetic(candidate, tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    spy = Mock(return_value=str(candidate["package"]))
    if failure == "missing_cache":
        spy.side_effect = FileNotFoundError("synthetic cache miss")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", spy)
    repo = "other/model" if failure == "wrong_destination" else "example/model"
    source = HubSnapshot(repo, "a" * 40, candidate["descriptor_path"], approvals, cache)
    with pytest.raises((ValueError, FileNotFoundError, PermissionError)):
        source.resolve()
    if failure == "wrong_destination":
        spy.assert_not_called()


def test_malformed_receipts_never_satisfy_gates(candidate, tmp_path):
    approvals = approve_synthetic(candidate, tmp_path)
    for name in ("rights", "human", "destination"):
        (approvals / f"{name}.json").write_text("[]")
    assert all(release_gates(candidate["descriptor"], approvals).values())


def test_service_synthetic_scores_match_original_tiny_model(candidate):
    import numpy as np

    from reviewsense.model_packaging import local_logits

    texts = ["O produto bom", "Não bom"]
    logits, _ = local_logits(candidate["source"], texts, device="cpu", precision="fp32")
    reference = np.exp(logits - logits.max(axis=1, keepdims=True))
    reference /= reference.sum(axis=1, keepdims=True)
    predictions = InferenceService(device="cpu", model_source=local(candidate)).predict_batch(texts)
    for row, prediction in zip(reference, predictions, strict=True):
        by_id = {score.class_id: score.value for score in prediction.scores}
        np.testing.assert_allclose([by_id[k] for k in range(3)], row, atol=1e-6, rtol=1e-5)
        assert prediction.class_id == int(row.argmax())
