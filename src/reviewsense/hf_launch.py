"""Preparação local do candidato v2 e plano exato; nenhuma API remota ou autenticação."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from reviewsense.artifacts import (
    ARTIFACT_FILES,
    BASE_MODEL,
    BASE_REVISION,
    _no_symlink_ancestors,
    file_inventory,
    file_sha256,
    inspect_directory,
    safe_file,
)
from reviewsense.labels import LABEL_NAMES
from reviewsense.model_packaging import METRIC_NAMES, card_example, release_inputs
from reviewsense.model_resolution import (
    PUBLIC_PACKAGE_FILES,
    SCOPED_TERMS,
    canonical_digest,
    scan_metadata,
    validate_descriptor,
    validate_runtime_package,
)
from reviewsense.space_bundle import BUNDLE_FILES, bundle_inventory, prepare_bundle

PUBLIC_DOCUMENTS = tuple(n for n in PUBLIC_PACKAGE_FILES if n not in ARTIFACT_FILES)
PACKAGE_VERSION = "1.0.0-rc.2"
OPERATIONS = [
    "create_public_model",
    "upload_model_allowlist",
    "verify_pinned_model",
    "configure_private_operational_secret",
    "create_public_zerogpu_space",
    "upload_space_allowlist",
    "verify_real_synthetic_inference",
]


def validate_public_card(content: str, metrics: dict) -> None:
    """Confere licença de escopo limitado e números congelados, sem validador remoto."""
    from huggingface_hub import ModelCard

    metadata = ModelCard(content).data.to_dict()
    expected = {
        "language": "pt",
        "library_name": "transformers",
        "pipeline_tag": "text-classification",
        "base_model": BASE_MODEL,
        "base_model_relation": "finetune",
        "license": "other",
        "license_name": SCOPED_TERMS["license_name"],
        "license_link": "LICENSE",
        "datasets": ["ruanchaves/b2w-reviews01"],
    }
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Metadata pública ou escopo de licença divergente.")
    if "license-under-review" not in metadata.get("tags", []):
        raise ValueError("Incerteza de direitos ausente.")
    result = metadata["model-index"][0]["results"][0]
    measured = {m["name"]: m["value"] for m in result["metrics"]}
    if (
        result["task"]["type"] != "text-classification"
        or result["dataset"]["split"] != "test"
        or result["dataset"]["type"] != "ruanchaves/b2w-reviews01"
        or measured != {name: metrics[key] for name, key in METRIC_NAMES.items()}
    ):
        raise ValueError("Resultados públicos divergem da referência congelada.")
    for values in metrics["per_class"].values():
        if any(str(values[key]) not in content for key in ("precision", "recall", "f1-score")):
            raise ValueError("Métricas por classe ausentes ou divergentes.")
    if any(str(metrics[key]) not in content for key in METRIC_NAMES.values()):
        raise ValueError("Resultados exatos ausentes do corpo do Card.")
    card_example(content)
    scan_metadata(content)


def build_candidate(
    source: Path,
    documents: Path,
    destination: Path,
    *,
    expected: dict[str, str],
    metrics: dict,
    provenance: dict,
) -> tuple[dict, dict]:
    """Copia só dez nomes fixos para destino novo; não altera o candidato 4B original."""
    inspect_directory(source, ARTIFACT_FILES, ignored={"training_args.bin"})
    inspect_directory(documents, PUBLIC_DOCUMENTS)
    if set(expected) != set(ARTIFACT_FILES) or any(
        file_sha256(safe_file(source, n)) != expected[n] for n in ARTIFACT_FILES
    ):
        raise ValueError("Artefatos divergem do checkpoint selecionado.")
    validate_public_card(safe_file(documents, "README.md").read_text(), metrics)
    _no_symlink_ancestors(destination)
    if os.path.lexists(destination):
        raise FileExistsError("Candidato existente; não sobrescrever.")
    for origin in (source, documents):
        if destination.absolute().is_relative_to(
            origin.absolute()
        ) or origin.absolute().is_relative_to(destination.absolute()):
            raise ValueError("Destino sobrepõe origem.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-candidate-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        for name in PUBLIC_PACKAGE_FILES:
            origin = source if name in ARTIFACT_FILES else documents
            path = safe_file(origin, name)
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
                with (staging / name).open("xb") as target:
                    shutil.copyfileobj(stream, target)
        inventory = file_inventory(staging, PUBLIC_PACKAGE_FILES)
        if any(inventory[n]["sha256"] != expected[n] for n in ARTIFACT_FILES):
            raise ValueError("Artefato mudou durante a cópia.")
        descriptor = {
            "schema_version": 2,
            "artifact_identity": "reviewsense-ptbr-bertimbau-v1",
            "package_version": PACKAGE_VERSION,
            "architecture": "BertForSequenceClassification",
            "max_length": 256,
            "labels": {str(k): v for k, v in LABEL_NAMES.items()},
            "files": inventory,
            "license_status": "REQUIRES_REVIEW",
            "license": "other",
            "distribution_terms": copy.deepcopy(SCOPED_TERMS),
            "model_card": "README.md",
            "reference": {
                "base_revision": BASE_REVISION,
                "package_content_sha256": canonical_digest(inventory),
            },
        }
        manifest = {
            "schema_version": 2,
            "descriptor_sha256": canonical_digest(descriptor),
            "files": inventory,
            "original_artifact_sha256": expected,
            "selected_checkpoint": {"epoch": 1, "global_step": 5543},
            "frozen_test_metrics": metrics,
            "provenance": provenance,
            "operational_confirmation_required": True,
            "legal_clearance": "NOT_ESTABLISHED",
        }
        validate_candidate(staging, manifest, descriptor)
        if os.path.lexists(destination):
            raise FileExistsError("Destino criado durante preparação; não sobrescrever.")
        staging.rename(destination)
    return manifest, descriptor


def validate_candidate(package: Path, manifest: dict, descriptor: dict) -> None:
    validate_runtime_package(package, descriptor)
    if (
        descriptor["schema_version"] != 2
        or manifest["schema_version"] != 2
        or manifest["descriptor_sha256"] != canonical_digest(descriptor)
        or manifest["files"] != descriptor["files"]
        or manifest["selected_checkpoint"] != {"epoch": 1, "global_step": 5543}
        or manifest["operational_confirmation_required"] is not True
        or manifest["legal_clearance"] != "NOT_ESTABLISHED"
        or set(manifest["original_artifact_sha256"]) != set(ARTIFACT_FILES)
        or any(
            descriptor["files"][n]["sha256"] != manifest["original_artifact_sha256"][n]
            for n in ARTIFACT_FILES
        )
    ):
        raise ValueError("Manifest do candidato público divergente.")
    validate_public_card(
        safe_file(package, "README.md").read_text(), manifest["frozen_test_metrics"]
    )
    scan_metadata(json.dumps(manifest, ensure_ascii=False))


def publication_plan(descriptor: dict, space_inventory: dict) -> dict:
    """Vincula a confirmação futura aos bytes e destinos, sem inventar revisão do Hub."""
    validate_descriptor(descriptor)
    if descriptor["schema_version"] != 2 or set(space_inventory) != set(BUNDLE_FILES):
        raise ValueError("Plano exige descritor v2 e bundle de 17 arquivos.")
    if space_inventory["model_descriptor.json"]["sha256"] != canonical_file_digest(descriptor):
        raise ValueError("Bundle não contém o descritor público selecionado.")
    plan = {
        "schema_version": 2,
        "account": "JorgeRibeiru",
        "operational_confirmation_required": True,
        "legal_clearance": "NOT_ESTABLISHED",
        "descriptor_sha256": canonical_digest(descriptor),
        "model": {
            "repo_id": SCOPED_TERMS["model_repo_id"],
            "visibility": "public",
            "files": descriptor["files"],
        },
        "space": {
            "repo_id": SCOPED_TERMS["space_repo_id"],
            "visibility": "public",
            "files": space_inventory,
            "hardware": "zero-a10g",
            "paid_resources_allowed": False,
        },
        "model_hub_revision": None,
        "revision_policy": "PIN_VERIFIED_COMMIT_AFTER_MODEL_UPLOAD_BEFORE_SPACE",
        "allowed_operations_after_confirmation": list(OPERATIONS),
    }
    validate_publication_plan(plan)
    return plan


def validate_publication_plan(plan: dict) -> None:
    """Recusa destino, hardware pago, extras e revisões inventadas mesmo num plano alterado."""
    if (
        set(plan)
        != {
            "schema_version",
            "account",
            "operational_confirmation_required",
            "legal_clearance",
            "descriptor_sha256",
            "model",
            "space",
            "model_hub_revision",
            "revision_policy",
            "allowed_operations_after_confirmation",
        }
        or set(plan["model"]) != {"repo_id", "visibility", "files"}
        or set(plan["space"])
        != {
            "repo_id",
            "visibility",
            "files",
            "hardware",
            "paid_resources_allowed",
        }
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 2
        or plan["account"] != "JorgeRibeiru"
        or plan["operational_confirmation_required"] is not True
        or plan["legal_clearance"] != "NOT_ESTABLISHED"
        or plan["model"]["repo_id"] != SCOPED_TERMS["model_repo_id"]
        or plan["space"]["repo_id"] != SCOPED_TERMS["space_repo_id"]
        or plan["model"]["visibility"] != "public"
        or plan["space"]["visibility"] != "public"
        or plan["space"]["paid_resources_allowed"] is not False
        or plan["space"]["hardware"] != "zero-a10g"
        or plan["model_hub_revision"] is not None
        or plan["revision_policy"] != "PIN_VERIFIED_COMMIT_AFTER_MODEL_UPLOAD_BEFORE_SPACE"
        or plan["allowed_operations_after_confirmation"] != OPERATIONS
        or not re.fullmatch(r"[a-f0-9]{64}", plan["descriptor_sha256"])
        or set(plan["model"]["files"]) != set(PUBLIC_PACKAGE_FILES)
        or set(plan["space"]["files"]) != set(BUNDLE_FILES)
    ):
        raise ValueError("Plano público incompatível com o escopo solicitado.")
    for inventory in (plan["model"]["files"], plan["space"]["files"]):
        for record in inventory.values():
            if (
                set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int
                or record["bytes"] < 1
                or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])
            ):
                raise ValueError("Inventário público inválido.")


def canonical_file_digest(value: dict) -> str:
    """Digest dos bytes JSON usados nos arquivos, distinto do digest canônico do contrato."""
    import hashlib

    return hashlib.sha256(json_bytes(value)).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def validate_operational_confirmation(plan: dict, confirmation: dict) -> None:
    """Só chamar depois de receber consentimento real; não produz recibo de revisão legal."""
    validate_publication_plan(plan)
    if (
        confirmation.get("authorized") is not True
        or confirmation.get("plan_sha256") != canonical_digest(plan)
        or confirmation.get("actor") != plan["account"]
        or confirmation.get("model_repo_id") != plan["model"]["repo_id"]
        or confirmation.get("space_repo_id") != plan["space"]["repo_id"]
        or confirmation.get("visibility") != "public"
        or confirmation.get("paid_resources_allowed") is not False
        or confirmation.get("operations") != plan["allowed_operations_after_confirmation"]
        or confirmation.get("legal_clearance") != "NOT_ESTABLISHED"
    ):
        raise PermissionError("Confirmação operacional ausente ou divergente do plano revisado.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    root, output = Path.cwd(), args.destination
    _no_symlink_ancestors(output)
    if os.path.lexists(output):
        raise FileExistsError("Saída existente; fornecer destino novo.")
    for source in ("models", "publishing", "src", "app", "deploy", "data"):
        if output.absolute().is_relative_to((root / source).absolute()):
            raise ValueError("Saída sobrepõe origens protegidas.")
    expected, metrics, provenance = release_inputs(root)
    provenance["preparation_code_sha256"] = {
        name: file_sha256(root / name)
        for name in (
            "src/reviewsense/hf_launch.py",
            "src/reviewsense/model_resolution.py",
            "src/reviewsense/space_bundle.py",
            "deploy/zerogpu/app.py",
        )
    }
    output.mkdir(parents=True)
    manifest, descriptor = build_candidate(
        root / "models/bertimbau/phase3b-full-v3/best_model",
        root / "publishing/hf_release/model_card",
        output / "model",
        expected=expected,
        metrics=metrics,
        provenance=provenance,
    )
    descriptor_path = output / "inference_descriptor.json"
    descriptor_path.write_bytes(json_bytes(descriptor))
    (output / "package_manifest.json").write_bytes(json_bytes(manifest))
    prepare_bundle(root, output / "space", descriptor=descriptor_path)
    inventory = bundle_inventory(output / "space")
    plan = publication_plan(descriptor, inventory)
    (output / "launch_plan.json").write_bytes(json_bytes(plan))
    print(
        json.dumps(
            {
                "model_files": len(descriptor["files"]),
                "space_files": len(inventory),
                "plan_sha256": canonical_digest(plan),
                "operational_confirmation_required": True,
            }
        )
    )


if __name__ == "__main__":
    main()
