"""Descritor de inferência independente e resolução offline de pacotes verificados."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from reviewsense.artifacts import (
    BASE_REVISION,
    DOCUMENT_FILES,
    PACKAGE_FILES,
    _no_symlink_ancestors,
    file_inventory,
    inspect_directory,
    read_json,
    safe_file,
    validate_configuration,
)
from reviewsense.labels import LABEL_NAMES

DESCRIPTOR_KEYS = {
    "schema_version",
    "artifact_identity",
    "package_version",
    "architecture",
    "max_length",
    "labels",
    "files",
    "reference",
    "license_status",
    "license",
    "upload_authorized",
    "model_card",
}
SCOPED_TERMS = {
    "license_name": "reviewsense-controlled-rights-cc-by-nc-sa-4.0",
    "license_link": "LICENSE",
    "controlled_rights_license": "cc-by-nc-sa-4.0",
    "legal_clearance": "NOT_ESTABLISHED",
    "operational_confirmation_required": True,
    "model_repo_id": "JorgeRibeiru/reviewsense-ptbr-bertimbau",
    "space_repo_id": "JorgeRibeiru/reviewsense-ptbr",
}
PUBLIC_PACKAGE_FILES = tuple(sorted((*PACKAGE_FILES, "LICENSE")))
SHA256 = r"[a-f0-9]{64}"
REVISION = r"[a-f0-9]{40}"
REPO_ID = r"[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def scan_metadata(text: str) -> None:
    """Rejeita caminhos pessoais e padrões conhecidos; não é um detector universal de segredos."""
    if re.search(
        r"/home/|/Users/|[A-Za-z]:[\\/]Users[\\/]|\bhf_[A-Za-z0-9]{20,}"
        r"|\bgh[pousr]_[A-Za-z0-9]{20,}|\bAKIA[A-Z0-9]{16}"
        r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        r"|(?i:(?:api_key|password|access_token)\s*[:=]\s*[\"']?\S{8,})",
        text,
    ):
        raise ValueError("Possível segredo ou caminho pessoal nos metadados.")


def descriptor_from_manifest(manifest: dict) -> dict:
    """Extrai somente identidade, contrato e hashes; não exporta histórico ou métricas."""
    result = {
        key: manifest[key]
        for key in (
            "schema_version",
            "artifact_identity",
            "package_version",
            "architecture",
            "max_length",
            "labels",
            "files",
            "license_status",
            "upload_authorized",
            "model_card",
        )
    }
    result["license"] = "unknown"
    result["reference"] = {
        "base_revision": manifest["base_revision"],
        "package_content_sha256": manifest["package_content_sha256"],
    }
    validate_descriptor(result)
    return result


def validate_descriptor(value: dict) -> None:
    try:
        _validate_descriptor(value)
    except (KeyError, TypeError, AttributeError):
        raise ValueError("Estrutura do descritor inválida.") from None


def _validate_descriptor(value: dict) -> None:
    version = value["schema_version"]
    expected_keys = DESCRIPTOR_KEYS
    if version == 2:
        expected_keys = (DESCRIPTOR_KEYS - {"upload_authorized"}) | {"distribution_terms"}
    if (
        set(value) != expected_keys
        or type(version) is not int
        or version not in {1, 2}
        or value["artifact_identity"] != "reviewsense-ptbr-bertimbau-v1"
        or not re.fullmatch(r"1\.0\.0(?:-rc\.[1-9][0-9]*)?", value["package_version"])
        or value["architecture"] != "BertForSequenceClassification"
        or value["max_length"] != 256
        or value["labels"] != {str(k): v for k, v in LABEL_NAMES.items()}
        or value["model_card"] != "README.md"
        or set(value["files"]) != set(package_files(value))
        or value["license_status"] not in {"REQUIRES_REVIEW", "APPROVED", "BLOCKED"}
        or not isinstance(value["license"], str)
        or not re.fullmatch(r"[a-z0-9.-]+", value["license"])
        or set(value["reference"]) != {"base_revision", "package_content_sha256"}
        or value["reference"]["base_revision"] != BASE_REVISION
    ):
        raise ValueError("Descritor de inferência incompatível.")
    if version == 1:
        if type(value["upload_authorized"]) is not bool:
            raise ValueError("Flag de autorização inválida.")
        if value["license_status"] != "APPROVED" and (
            value["upload_authorized"] or value["license"] != "unknown"
        ):
            raise ValueError("Licença pendente não permite autorização de distribuição.")
        if value["license_status"] == "APPROVED" and value["license"] == "unknown":
            raise ValueError("Licença aprovada precisa de identificação explícita.")
    elif (
        value["license"] != "other"
        or value["license_status"] != "REQUIRES_REVIEW"
        or value["distribution_terms"] != SCOPED_TERMS
        or value["distribution_terms"].get("operational_confirmation_required") is not True
    ):
        raise ValueError("Política v2 exige licença de escopo limitado e incerteza explícita.")
    for record in value["files"].values():
        if (
            set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] < 1
            or not re.fullmatch(SHA256, record["sha256"])
        ):
            raise ValueError("Inventário público inválido.")
    if canonical_digest(value["files"]) != value["reference"]["package_content_sha256"]:
        raise ValueError("Fingerprint do descritor divergente.")
    scan_metadata(json.dumps(value, ensure_ascii=False))


def package_files(descriptor: dict) -> tuple[str, ...]:
    """Allowlists fixas por versão; nomes arbitrários do descritor nunca são aceitos."""
    return PUBLIC_PACKAGE_FILES if descriptor.get("schema_version") == 2 else PACKAGE_FILES


def validate_runtime_package(directory: Path, descriptor: dict) -> None:
    """Valida integridade/contrato sem inferir autorização ou aprovação jurídica."""
    from huggingface_hub import ModelCard

    validate_descriptor(descriptor)
    names = package_files(descriptor)
    inspect_directory(directory, names)
    if file_inventory(directory, names) != descriptor["files"]:
        raise ValueError("Integridade do pacote incompatível com o descritor.")
    validate_configuration(directory)
    for name in (*DOCUMENT_FILES, "tokenizer.json"):
        scan_metadata(safe_file(directory, name).read_text())
    metadata = ModelCard(safe_file(directory, "README.md").read_text()).data.to_dict()
    if (
        metadata.get("license") != descriptor["license"]
        or metadata.get("pipeline_tag") != "text-classification"
        or metadata.get("language") != "pt"
    ):
        raise ValueError("Model Card incompatível com o descritor.")
    status = safe_file(directory, "DISTRIBUTION_STATUS.md").read_text()
    if f"licensing_status: {descriptor['license_status']}" not in status:
        raise ValueError("Aviso de distribuição incompatível com o descritor.")
    if descriptor["schema_version"] == 1:
        if f"upload_authorized: {str(descriptor['upload_authorized']).lower()}" not in status:
            raise ValueError("Aviso de distribuição incompatível com o descritor.")
    else:
        terms = descriptor["distribution_terms"]
        license_text = safe_file(directory, "LICENSE").read_text()
        scan_metadata(license_text)
        if (
            metadata.get("license_name") != terms["license_name"]
            or metadata.get("license_link") != terms["license_link"]
            or "legal_clearance: NOT_ESTABLISHED" not in status
            or "operational_confirmation_required: true" not in status
            or "https://creativecommons.org/licenses/by-nc-sa/4.0/" not in license_text
            or "somente os direitos que o responsável controla" not in license_text
        ):
            raise ValueError("Avisos de escopo e autorização operacional v2 ausentes.")
    hashes = [descriptor["files"][n]["sha256"] for n in names]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Arquivos duplicados no candidato.")


class ModelSource(Protocol):
    def resolve(self) -> Path: ...


@dataclass(frozen=True)
class LocalPackage:
    directory: Path
    descriptor_path: Path
    allowed_root: Path

    def resolve(self) -> Path:
        # Sem resolve() antes da inspeção: não esconder symlinks do validador.
        directory = self.directory.absolute()
        if not directory.is_relative_to(self.allowed_root.absolute()):
            raise ValueError("Pacote fora da raiz permitida.")
        descriptor = read_json(safe_file(self.descriptor_path.parent, self.descriptor_path.name))
        validate_runtime_package(directory, descriptor)
        return directory


def release_gates(descriptor: dict, approval_root: Path | None = None) -> dict[str, bool]:
    """Confere recibos separados, vinculados ao candidato; revisão humana continua indispensável.

    v1 conserva a política anterior; v2 confere decisão operacional sob incerteza,
    mantendo REQUIRES_REVIEW no descritor. Registros do operador não são assinaturas
    digitais nem prova automática da suficiência jurídica das permissões.
    """
    validate_descriptor(descriptor)
    if descriptor["schema_version"] == 2:
        return _operational_gates(descriptor, approval_root)
    digest = canonical_digest(descriptor)
    receipts = {}
    for name in ("rights", "destination", "human"):
        try:
            receipt = read_json(safe_file(approval_root, f"{name}.json")) if approval_root else {}
            if not isinstance(receipt, dict) or receipt.get("descriptor_sha256") != digest:
                receipt = {}
            receipts[name] = receipt
        except (ValueError, OSError, TypeError):
            receipts[name] = {}
    rights, destination, human = (receipts[k] for k in ("rights", "destination", "human"))
    # Evidência documental fica fora do pacote público; o recibo confere seus bytes.
    evidence_ok = False
    try:
        evidence = safe_file(approval_root, rights.get("evidence_file", ""))
        evidence_ok = hashlib.sha256(evidence.read_bytes()).hexdigest() == rights.get(
            "evidence_sha256"
        )
    except (ValueError, OSError, TypeError, AttributeError):
        pass
    legal = (
        descriptor["license_status"] == "APPROVED"
        and rights.get("decision") == "APPROVED"
        and isinstance(rights.get("scope"), list)
        and "public_weights" in rights["scope"]
        and "public_inference" in rights["scope"]
        and evidence_ok
    )
    target = bool(
        destination.get("confirmed") is True
        and isinstance(destination.get("repo_id"), str)
        and isinstance(destination.get("revision"), str)
        and re.fullmatch(REPO_ID, destination.get("repo_id", ""))
        and re.fullmatch(REVISION, destination.get("revision", ""))
    )
    authorized = (
        descriptor["upload_authorized"] is True
        and human.get("authorized") is True
        and human.get("repo_id") == destination.get("repo_id")
        and human.get("revision") == destination.get("revision")
        and isinstance(human.get("actions"), list)
        and all(
            action in human["actions"]
            for action in ("download", "public_weights", "public_inference")
        )
    )
    return {
        "LEGAL_REVIEW_REQUIRED": not legal,
        "DESTINATION_UNCONFIRMED": not target,
        "PUBLISH_AUTHORIZATION_REQUIRED": not authorized,
    }


def _operational_gates(descriptor: dict, root: Path | None) -> dict[str, bool]:
    """v2: autorização de operação sob incerteza, nunca aprovação jurídica implícita."""
    digest = canonical_digest(descriptor)
    records = {}
    for name in ("risk", "destination", "human"):
        try:
            record = read_json(safe_file(root, f"{name}.json")) if root else {}
            records[name] = record if record.get("descriptor_sha256") == digest else {}
        except (ValueError, OSError, TypeError, AttributeError):
            records[name] = {}
    risk, destination, human = (records[n] for n in ("risk", "destination", "human"))
    evidence_ok = False
    try:
        evidence = safe_file(root, risk.get("evidence_file", ""))
        evidence_ok = hashlib.sha256(evidence.read_bytes()).hexdigest() == risk.get(
            "evidence_sha256"
        )
    except (ValueError, OSError, TypeError, AttributeError):
        pass
    acknowledged = (
        evidence_ok
        and risk.get("decision") == "ACKNOWLEDGED_UNRESOLVED_RIGHTS"
        and risk.get("purpose") == "educational_noncommercial"
        and risk.get("third_party_permissions") == "NOT_ASSERTED"
        and isinstance(risk.get("scope"), list)
        and all(s in risk["scope"] for s in ("public_weights", "public_inference"))
    )
    target = (
        destination.get("confirmed") is True
        and destination.get("repo_id") == SCOPED_TERMS["model_repo_id"]
        and destination.get("visibility") == "public"
        and isinstance(destination.get("revision"), str)
        and re.fullmatch(REVISION, destination["revision"]) is not None
    )
    authorized = (
        human.get("authorized") is True
        and human.get("actor") == "JorgeRibeiru"
        and human.get("repo_id") == destination.get("repo_id")
        and human.get("revision") == destination.get("revision")
        and human.get("visibility") == "public"
        and isinstance(human.get("actions"), list)
        and all(a in human["actions"] for a in ("download", "public_weights", "public_inference"))
    )
    # Situação jurídica continua no descritor (REQUIRES_REVIEW/NOT_ESTABLISHED).
    # Estes três valores são somente bloqueios operacionais da política v2 explícita.
    return {
        "RIGHTS_POLICY_UNACKNOWLEDGED": not acknowledged,
        "DESTINATION_UNCONFIRMED": not target,
        "PUBLISH_AUTHORIZATION_REQUIRED": not authorized,
    }


def provision_operational_approvals(payload: str, root: Path, descriptor: dict) -> None:
    """Materializa um Secret privado já autorizado; não inventa nem aprova seu conteúdo.

    Usado no startup do Space. O operador é a fronteira de confiança; não é assinatura
    digital. Nunca registrar payload/evidência em logs ou incluir no bundle público.
    """
    validate_descriptor(descriptor)
    if descriptor["schema_version"] != 2 or len(payload.encode()) > 128_000:
        raise ValueError("Pacote de autorizações operacionais incompatível.")
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or set(value) != {"risk", "destination", "human", "evidence"}
        or not isinstance(value["evidence"], str)
        or not all(isinstance(value[n], dict) for n in ("risk", "destination", "human"))
        or value["risk"].get("evidence_file") != "owner_decision.md"
    ):
        raise ValueError("Estrutura de autorização privada inválida.")
    _no_symlink_ancestors(root)
    if root.exists():
        raise FileExistsError("Não sobrescrever autorizações existentes.")
    root.mkdir(mode=0o700)
    for name in ("risk", "destination", "human"):
        path = root / f"{name}.json"
        path.write_text(json.dumps(value[name], ensure_ascii=False), encoding="utf-8")
        path.chmod(0o600)
    evidence = root / "owner_decision.md"
    evidence.write_text(value["evidence"], encoding="utf-8")
    evidence.chmod(0o600)
    if any(release_gates(descriptor, root).values()):
        raise PermissionError("Autorizações privadas não correspondem ao candidato/revisão.")


@dataclass(frozen=True)
class HubSnapshot:
    """Adaptador opcional de cache: nunca baixa, autentica ou publica nesta versão."""

    repo_id: str
    revision: str
    descriptor_path: Path
    approval_root: Path
    cache_root: Path

    def resolve(self) -> Path:
        if not re.fullmatch(REPO_ID, self.repo_id) or not re.fullmatch(REVISION, self.revision):
            raise ValueError("Destino explícito e revisão SHA imutável são obrigatórios.")
        descriptor = read_json(safe_file(self.descriptor_path.parent, self.descriptor_path.name))
        if any(release_gates(descriptor, self.approval_root).values()):
            raise PermissionError("Distribuição bloqueada pelos gates de lançamento.")
        destination = read_json(safe_file(self.approval_root, "destination.json"))
        if (self.repo_id, self.revision) != (destination["repo_id"], destination["revision"]):
            raise PermissionError("Snapshot não corresponde ao destino autorizado.")
        from reviewsense.hub_cache import materialize_hub_snapshot

        # Lê diretamente o cache fixado: nenhuma API, autenticação ou consulta à rede.
        directory = materialize_hub_snapshot(
            self.cache_root, self.repo_id, self.revision, descriptor
        )
        return LocalPackage(directory, self.descriptor_path, self.cache_root).resolve()
