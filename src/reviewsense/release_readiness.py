"""Preflight local do candidato; não contém publicação, autenticação ou bypass."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from reviewsense.model_packaging import (
    read_json,
    safe_file,
    validate_model_card,
    validate_package,
)
from reviewsense.model_resolution import (
    descriptor_from_manifest,
    release_gates,
    scan_metadata,
    validate_descriptor,
    validate_runtime_package,
)


def check_known_links(package: Path, receipt: dict) -> dict:
    """Offline: referências locais e recibo HTTP separado, sem abrir conexões."""
    urls = set()
    local = 0
    for path in package.glob("*.md"):
        text = path.read_text()
        urls.update(u.rstrip(".,);") for u in re.findall(r'https?://[^\s<>"\x27`]+', text))
        for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            target = target.split("#")[0]
            safe_file(package, target)
            local += 1
    known = {item["url"]: item for item in receipt.get("links", [])}
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.username or parts.password or parts.query:
            raise ValueError("Link inseguro ou com parâmetros inesperados no pacote.")
        if url not in known or known[url].get("status") != 200:
            raise ValueError(
                "Link conhecido sem evidência HTTP 200; atualizar auditoria separadamente."
            )
    if not receipt.get("checked_at_utc"):
        raise ValueError("Recibo de links sem data de consulta.")
    return {
        "external_links": len(urls),
        "local_references": local,
        "network_performed_by_preflight": False,
        "http_receipt_checked_at_utc": receipt["checked_at_utc"],
        "limitations": "Âncoras não verificadas; HTTP prévio não garante disponibilidade futura.",
    }


def inspect_weight_header(package: Path) -> None:
    """Inspeciona apenas o header JSON do formato seguro, nunca desserializa pickle."""
    with safe_file(package, "model.safetensors").open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 2 <= header_size <= 16 * 1024**2:
            raise ValueError("Header safetensors inválido ou excessivo.")
        header = stream.read(header_size).decode()
        json.loads(header)
        scan_metadata(header)


def preflight(
    package: Path,
    manifest: dict,
    descriptor: dict,
    links: dict,
    *,
    approval_root: Path | None = None,
) -> dict:
    """Integridade completa do candidato 4B, descritor e gates separados.

    TECHNICALLY_READY só é promovido pelo runner após a inferência e equivalência.
    A aprovação legal não é inferida de sucesso técnico.
    """
    validate_package(package, manifest)
    validate_descriptor(descriptor)
    if descriptor != descriptor_from_manifest(manifest):
        raise ValueError("Descritor não corresponde ao candidato congelado.")
    validate_runtime_package(package, descriptor)
    validate_model_card(
        safe_file(package, "README.md").read_text(), manifest["frozen_test_metrics"]
    )
    inspect_weight_header(package)
    checked_links = check_known_links(package, links)
    return {
        "phase": "4C-readiness",
        "TECHNICALLY_READY": False,
        **release_gates(descriptor, approval_root),
        "integrity_and_metadata_valid": True,
        "links": checked_links,
        "license_status": descriptor["license_status"],
        "upload_authorized": descriptor["upload_authorized"],
        "remote_execution_allowed": False,
        "package_content_sha256": manifest["package_content_sha256"],
        "test_metrics_recomputed": False,
    }


def require_remote_authorization(descriptor: dict, approval_root: Path | None = None) -> None:
    """Gate reutilizável para etapa futura; não há uploader nesta entrega."""
    if any(release_gates(descriptor, approval_root).values()):
        raise PermissionError("Operação remota recusada: lançamento ainda não autorizado.")


def load_preflight_inputs(manifest: Path, descriptor: Path, links: Path) -> tuple[dict, dict, dict]:
    return tuple(read_json(safe_file(p.parent, p.name)) for p in (manifest, descriptor, links))
