"""Distribuição v2: todas as aprovações desta suíte pertencem a pesos minúsculos sintéticos."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from reviewsense.artifacts import ARTIFACT_FILES, file_inventory
from reviewsense.hf_launch import (
    PUBLIC_DOCUMENTS,
    build_candidate,
    json_bytes,
    publication_plan,
    validate_candidate,
    validate_operational_confirmation,
    validate_public_card,
)
from reviewsense.model_resolution import (
    PUBLIC_PACKAGE_FILES,
    SCOPED_TERMS,
    HubSnapshot,
    LocalPackage,
    canonical_digest,
    provision_operational_approvals,
    release_gates,
    validate_descriptor,
)
from reviewsense.space_bundle import bundle_inventory, prepare_bundle, validate_bundle
from reviewsense.zerogpu import ZeroGPUSettings
from scripts.verify_phase4b_package import no_network

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def public_candidate(candidate, tmp_path):
    expected = {
        k: v["sha256"] for k, v in file_inventory(candidate["source"], ARTIFACT_FILES).items()
    }
    package = tmp_path / "public"
    manifest, descriptor = build_candidate(
        candidate["source"],
        ROOT / "publishing/hf_release/model_card",
        package,
        expected=expected,
        metrics=candidate["manifest"]["frozen_test_metrics"],
        provenance={"kind": "SYNTHETIC TEST ONLY"},
    )
    path = tmp_path / "public_descriptor.json"
    path.write_bytes(json_bytes(descriptor))
    return {
        **candidate,
        "package": package,
        "manifest": manifest,
        "descriptor": descriptor,
        "descriptor_path": path,
    }


def private_payload(descriptor):
    # Vinculado ao modelo BERT minúsculo do fixture, nunca ao checkpoint real.
    evidence = "SYNTHETIC TEST ONLY — operational confirmation, not legal clearance."
    digest = canonical_digest(descriptor)
    destination = {
        "descriptor_sha256": digest,
        "confirmed": True,
        "repo_id": SCOPED_TERMS["model_repo_id"],
        "revision": "a" * 40,
        "visibility": "public",
    }
    return {
        "evidence": evidence,
        "risk": {
            "descriptor_sha256": digest,
            "decision": "ACKNOWLEDGED_UNRESOLVED_RIGHTS",
            "purpose": "educational_noncommercial",
            "third_party_permissions": "NOT_ASSERTED",
            "scope": ["public_weights", "public_inference"],
            "evidence_file": "owner_decision.md",
            "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(),
        },
        "destination": destination,
        "human": {
            "descriptor_sha256": digest,
            "actor": "JorgeRibeiru",
            "authorized": True,
            "repo_id": destination["repo_id"],
            "revision": destination["revision"],
            "visibility": "public",
            "space_id": SCOPED_TERMS["space_repo_id"],
            "actions": ["download", "public_weights", "public_inference", "host_zerogpu"],
        },
    }


def test_preparation_no_remote_and_preserves_four_artifacts(candidate, tmp_path, monkeypatch):
    import huggingface_hub

    denied = Mock(side_effect=AssertionError("no remote operation"))
    for name in ["create_repo", "upload_folder", "create_commit", "whoami"]:
        monkeypatch.setattr(huggingface_hub.HfApi, name, denied)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", denied)
    before = file_inventory(candidate["source"], ARTIFACT_FILES)
    with no_network() as network:
        manifest, descriptor = build_candidate(
            candidate["source"],
            ROOT / "publishing/hf_release/model_card",
            tmp_path / "new",
            expected={n: v["sha256"] for n, v in before.items()},
            metrics=candidate["manifest"]["frozen_test_metrics"],
            provenance={},
        )
        validate_candidate(tmp_path / "new", manifest, descriptor)
    assert network["blocked_connections"] == 0
    assert file_inventory(candidate["source"], ARTIFACT_FILES) == before
    assert {n: descriptor["files"][n] for n in ARTIFACT_FILES} == before
    assert set(descriptor["files"]) == set(PUBLIC_PACKAGE_FILES)
    assert len(PUBLIC_PACKAGE_FILES) == 10 and len(PUBLIC_DOCUMENTS) == 6
    assert descriptor["license_status"] == "REQUIRES_REVIEW"
    assert "upload_authorized" not in descriptor
    assert all(release_gates(descriptor).values())
    denied.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("license", "mit"),
        ("license", "cc-by-4.0"),
        ("license_status", "APPROVED"),
        ("upload_authorized", True),
        ("files", {}),
        ("schema_version", True),
        ("distribution_terms", {**SCOPED_TERMS, "legal_clearance": "APPROVED"}),
        ("distribution_terms", {**SCOPED_TERMS, "operational_confirmation_required": 1}),
    ],
)
def test_v2_rejects_false_clearance_and_invalid_contract(public_candidate, key, value):
    descriptor = copy.deepcopy(public_candidate["descriptor"])
    descriptor[key] = value
    with pytest.raises(ValueError):
        validate_descriptor(descriptor)


def test_operational_consent_never_changes_legal_status(public_candidate, tmp_path):
    descriptor = public_candidate["descriptor"]
    root = tmp_path / "private"
    provision_operational_approvals(json.dumps(private_payload(descriptor)), root, descriptor)
    assert not any(release_gates(descriptor, root).values())
    assert descriptor["license_status"] == "REQUIRES_REVIEW"
    assert descriptor["distribution_terms"]["legal_clearance"] == "NOT_ESTABLISHED"
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in root.iterdir())
    with pytest.raises(FileExistsError):
        provision_operational_approvals(json.dumps(private_payload(descriptor)), root, descriptor)
    settings = ZeroGPUSettings(
        public_candidate["descriptor_path"],
        ROOT / "deploy/zerogpu/runtime.json",
        root,
        tmp_path / "cache",
        SCOPED_TERMS["model_repo_id"],
        "a" * 40,
        SCOPED_TERMS["space_repo_id"],
    )
    settings.authorize()  # Contrato somente; não importa spaces nem consulta CUDA.


@pytest.mark.parametrize(
    "record,key,value",
    [
        ("risk", "decision", "APPROVED"),
        ("risk", "purpose", "commercial"),
        ("risk", "third_party_permissions", "GRANTED"),
        ("risk", "evidence_file", "../outside"),
        ("risk", "descriptor_sha256", "b" * 64),
        ("risk", "evidence_sha256", "0" * 64),
        ("destination", "revision", "main"),
        ("destination", "repo_id", "other/model"),
        ("destination", "visibility", "private"),
        ("human", "authorized", False),
        ("human", "descriptor_sha256", "0" * 64),
        ("human", "revision", "b" * 40),
        ("human", "actor", "other"),
        ("human", "actions", ["download"]),
    ],
)
def test_fail_closed_for_missing_or_wrong_consent(public_candidate, tmp_path, record, key, value):
    payload = private_payload(public_candidate["descriptor"])
    payload[record][key] = value
    with pytest.raises((ValueError, PermissionError)):
        provision_operational_approvals(
            json.dumps(payload), tmp_path / "invalid", public_candidate["descriptor"]
        )


def test_secret_schema_and_content_are_not_executable(public_candidate, tmp_path):
    for payload in ["[]", '{"arbitrary": "__import__(os)"}', "x" * 128001]:
        with pytest.raises(ValueError):
            provision_operational_approvals(
                payload, tmp_path / "invalid", public_candidate["descriptor"]
            )


@pytest.mark.parametrize("failure", ["extra", "license_missing", "changed_weight", "symlink"])
def test_v2_local_package_security(public_candidate, tmp_path, failure):
    package = public_candidate["package"]
    if failure == "extra":
        (package / "consumer.csv").write_text("synthetic forbidden extra")
    elif failure == "license_missing":
        (package / "LICENSE").unlink()
    elif failure == "changed_weight":
        with (package / "model.safetensors").open("ab") as f:
            f.write(b"x")
    else:
        (package / "LICENSE").unlink()
        (package / "LICENSE").symlink_to(ROOT / "publishing/hf_release/model_card/LICENSE")
    with pytest.raises(ValueError):
        LocalPackage(package, public_candidate["descriptor_path"], tmp_path).resolve()


def test_v2_realistic_symlink_cache_materializes_ten_files(public_candidate, tmp_path):
    descriptor = public_candidate["descriptor"]
    approvals = tmp_path / "records"
    provision_operational_approvals(json.dumps(private_payload(descriptor)), approvals, descriptor)
    cache = tmp_path / "cache"
    repo = cache / ("models--" + SCOPED_TERMS["model_repo_id"].replace("/", "--"))
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    blobs = repo / "blobs"
    blobs.mkdir()
    for name in PUBLIC_PACKAGE_FILES:
        data = (public_candidate["package"] / name).read_bytes()
        etag = hashlib.sha256(data).hexdigest()
        (blobs / etag).write_bytes(data)
        (snapshot / name).symlink_to(f"../../blobs/{etag}")
    (snapshot / "unapproved.csv").write_text("synthetic extra never copied")
    with no_network() as access:
        path = HubSnapshot(
            SCOPED_TERMS["model_repo_id"],
            "a" * 40,
            public_candidate["descriptor_path"],
            approvals,
            cache,
        ).resolve()
    assert access["blocked_connections"] == 0
    assert {p.name for p in path.iterdir()} == set(PUBLIC_PACKAGE_FILES)
    assert not any(p.is_symlink() for p in path.iterdir())
    assert file_inventory(path, PUBLIC_PACKAGE_FILES) == descriptor["files"]


def test_bundle_17_files_and_plan_binding(public_candidate, tmp_path):
    space = tmp_path / "space"
    inventory = prepare_bundle(ROOT, space, descriptor=public_candidate["descriptor_path"])
    validate_bundle(space, inventory)
    plan = publication_plan(public_candidate["descriptor"], bundle_inventory(space))
    assert len(plan["model"]["files"]) == 10 and len(plan["space"]["files"]) == 17
    assert plan["model_hub_revision"] is None
    assert plan["space"]["paid_resources_allowed"] is False
    assert not any(
        "private" in name or name.endswith(".safetensors") for name in plan["space"]["files"]
    )
    with pytest.raises(PermissionError):
        validate_operational_confirmation(plan, {})
    confirmation = {
        "authorized": True,
        "plan_sha256": canonical_digest(plan),
        "actor": "JorgeRibeiru",
        "model_repo_id": plan["model"]["repo_id"],
        "space_repo_id": plan["space"]["repo_id"],
        "visibility": "public",
        "paid_resources_allowed": False,
        "operations": plan["allowed_operations_after_confirmation"],
        "legal_clearance": "NOT_ESTABLISHED",
    }
    validate_operational_confirmation(
        plan, confirmation
    )  # Confirmação SOMENTE do fixture sintético.
    for key, value in [
        ("plan_sha256", "0" * 64),
        ("visibility", "private"),
        ("actor", "other"),
        ("paid_resources_allowed", True),
        ("legal_clearance", "APPROVED"),
    ]:
        with pytest.raises(PermissionError):
            validate_operational_confirmation(plan, {**confirmation, key: value})


def test_public_card_rejects_changed_frozen_metric(public_candidate):
    card = (
        (public_candidate["package"] / "README.md")
        .read_text()
        .replace("0.7544611794363947", "0.9999")
    )
    with pytest.raises(ValueError):
        validate_public_card(card, public_candidate["manifest"]["frozen_test_metrics"])


def test_space_secret_bootstrap_is_private_and_sanitized(
    public_candidate, tmp_path, monkeypatch, capsys
):
    import reviewsense.zerogpu as zero

    spec = importlib.util.spec_from_file_location(
        "space_launch_test", ROOT / "deploy/zerogpu/app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "space-root"
    root.mkdir()
    (root / "model_descriptor.json").write_bytes(json_bytes(public_candidate["descriptor"]))
    monkeypatch.setenv(
        "REVIEWSENSE_OPERATIONAL_APPROVAL_JSON",
        json.dumps(private_payload(public_candidate["descriptor"])),
    )
    monkeypatch.delenv("REVIEWSENSE_APPROVAL_ROOT", raising=False)
    monkeypatch.setenv("REVIEWSENSE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("REVIEWSENSE_MODEL_REPO_ID", SCOPED_TERMS["model_repo_id"])
    monkeypatch.setenv("REVIEWSENSE_MODEL_REVISION", "a" * 40)
    monkeypatch.setenv("SPACE_ID", SCOPED_TERMS["space_repo_id"])
    locations = []

    def create(settings):
        settings.authorize()
        locations.append(settings.approval_root)
        assert settings.approval_root.exists()
        assert "REVIEWSENSE_OPERATIONAL_APPROVAL_JSON" not in os.environ
        return Mock()  # Não serve HTTP nem importa/inicializa CUDA.

    monkeypatch.setattr(zero, "create_zerogpu_app", create)
    from contextlib import ExitStack

    with ExitStack() as stack:
        module._launch(root, stack)
    assert locations and not locations[0].exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("change", ["paid", "revision", "extra", "other_destination"])
def test_plan_itself_cannot_expand_operations(public_candidate, tmp_path, change):
    from reviewsense.hf_launch import validate_publication_plan

    space = tmp_path / "restricted-space"
    prepare_bundle(ROOT, space, descriptor=public_candidate["descriptor_path"])
    plan = publication_plan(public_candidate["descriptor"], bundle_inventory(space))
    if change == "paid":
        plan["space"]["paid_resources_allowed"] = True
    elif change == "revision":
        plan["model_hub_revision"] = "main"
    elif change == "extra":
        plan["space"]["files"]["private.json"] = {"bytes": 1, "sha256": "a" * 64}
    else:
        plan["space"]["repo_id"] = "another/space"
    with pytest.raises(ValueError):
        validate_publication_plan(plan)


def test_independent_example_accepts_exact_ten_files(public_candidate):
    from scripts.verify_phase4b_package import verify_independent_example

    result = verify_independent_example(
        public_candidate["package"], ROOT, names=PUBLIC_PACKAGE_FILES
    )
    assert result["original_models_reports_data_accesses"] == 0
    assert result["network_connections"] == 0
    assert result["finite_three_class_scores"] is True


def test_secret_failure_does_not_serve_or_echo(public_candidate, tmp_path, monkeypatch):
    from contextlib import ExitStack

    import reviewsense.zerogpu as zero

    spec = importlib.util.spec_from_file_location(
        "negative_space_test", ROOT / "deploy/zerogpu/app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "model_descriptor.json").write_bytes(json_bytes(public_candidate["descriptor"]))
    monkeypatch.setenv("REVIEWSENSE_OPERATIONAL_APPROVAL_JSON", '{"secret":"SENSITIVE_SENTINEL"}')
    monkeypatch.delenv("REVIEWSENSE_APPROVAL_ROOT", raising=False)
    called = Mock(side_effect=AssertionError("UI must not be constructed"))
    monkeypatch.setattr(zero, "create_zerogpu_app", called)
    with ExitStack() as stack, pytest.raises(SystemExit) as error:
        module._launch(tmp_path, stack)
    assert "SENSITIVE_SENTINEL" not in str(error.value)
    assert str(tmp_path) not in str(error.value)
    called.assert_not_called()
