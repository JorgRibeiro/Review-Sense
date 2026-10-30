"""Configuração explícita: nenhuma variável de plataforma ativa rede ou downloads."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reviewsense.model_resolution import HubSnapshot, LocalPackage, read_json, release_gates
from reviewsense.predict import InferenceService

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppSettings:
    environment: Literal["local", "hosted"] = "local"
    origin: Literal["local", "hub-cache"] = "local"
    model_dir: Path = ROOT / "outputs/phase4b/model-repo-staging"
    descriptor: Path = ROOT / "publishing/inference_descriptor.json"
    allowed_root: Path = ROOT / "outputs/phase4b"
    repo_id: str | None = None
    revision: str | None = None
    approval_root: Path | None = None
    cache_root: Path = ROOT / "cache/approved-hub"
    enable_svm: bool = True

    def __post_init__(self):
        if self.environment not in {"local", "hosted"} or self.origin not in {"local", "hub-cache"}:
            raise ValueError("Ambiente ou origem inválidos.")
        if self.origin == "hub-cache" and not (
            self.repo_id and self.revision and self.approval_root
        ):
            raise ValueError("Hub exige destino, revisão e recibos explícitos.")


def check_hosting_authorization(settings: AppSettings) -> None:
    """Chamado tanto pela CLI quanto pela construção direta da UI hospedada."""
    if settings.environment == "hosted":
        descriptor = read_json(settings.descriptor)
        if any(release_gates(descriptor, settings.approval_root).values()):
            raise PermissionError("Hospedagem pública bloqueada pelos gates de lançamento.")


def service_from_settings(settings: AppSettings, *, device: str = "auto") -> InferenceService:
    # Construção local continua lazy. Hospedagem não prossegue com gates pendentes.
    check_hosting_authorization(settings)
    if settings.origin == "hub-cache":
        source = HubSnapshot(
            settings.repo_id,
            settings.revision,
            settings.descriptor,
            settings.approval_root,
            settings.cache_root,
        )
    else:
        source = LocalPackage(settings.model_dir, settings.descriptor, settings.allowed_root)
    return InferenceService(device=device, model_source=source)
