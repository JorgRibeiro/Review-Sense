"""Adaptador ZeroGPU isolado. Importar não importa spaces/torch nem faz operações remotas."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from reviewsense.artifacts import PACKAGE_FILES, read_json, safe_file
from reviewsense.model_resolution import REPO_ID, REVISION, HubSnapshot, release_gates
from reviewsense.predict import (
    ArtifactError,
    InferenceError,
    InferenceService,
    InputError,
    _BertBackend,
    load_bert_cpu,
    validate_text,
)

GPU_DURATION_SECONDS = 10


@dataclass(frozen=True)
class ZeroGPUSettings:
    descriptor: Path
    runtime: Path
    approval_root: Path
    cache_root: Path
    repo_id: str
    revision: str
    space_id: str
    provision_public_snapshot: bool = False

    def __post_init__(self):
        if (
            not re.fullmatch(REPO_ID, self.repo_id)
            or not re.fullmatch(REPO_ID, self.space_id)
            or not re.fullmatch(REVISION, self.revision)
        ):
            raise ValueError("Destino explícito e revisão SHA imutável obrigatórios.")

    def authorize(self) -> None:
        descriptor = read_json(safe_file(self.descriptor.parent, self.descriptor.name))
        if any(release_gates(descriptor, self.approval_root).values()):
            raise PermissionError("Direitos/destino/autorização pendentes para ZeroGPU.")
        human = read_json(safe_file(self.approval_root, "human.json"))
        destination = read_json(safe_file(self.approval_root, "destination.json"))
        if (
            human.get("space_id") != self.space_id
            or "host_zerogpu" not in human.get("actions", [])
            or (destination["repo_id"], destination["revision"]) != (self.repo_id, self.revision)
        ):
            raise PermissionError("Destino ZeroGPU não corresponde à autorização humana.")

    def source(self) -> HubSnapshot:
        self.authorize()
        if self.provision_public_snapshot:
            from huggingface_hub import snapshot_download

            # Somente futuro repo público autorizado; sem credenciais, main ou código remoto.
            snapshot_download(
                repo_id=self.repo_id,
                revision=self.revision,
                repo_type="model",
                allow_patterns=list(PACKAGE_FILES),
                cache_dir=self.cache_root,
                local_files_only=False,
                token=False,
            )
        return HubSnapshot(
            self.repo_id, self.revision, self.descriptor, self.approval_root, self.cache_root
        )


class ZeroGPUInferenceService(InferenceService):
    """Carga eager fora da função GPU; sem descoberta CUDA, lazy load ou fallback CPU."""

    def __init__(self, model_source):
        super().__init__(device="cuda", model_source=model_source)

    def prepare(self) -> None:
        with self._lock:
            if self._bert is None:
                self._verify_bert()
                import torch

                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                torch.set_num_threads(2)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                model, tokenizer = load_bert_cpu(self._bert_directory)
                # spaces deve ter instalado a emulação antes deste ponto.
                # Não consultar is_available/capability nem executar kernels CUDA no startup.
                model = model.to("cuda").eval()
                self._bert = _BertBackend(model, tokenizer, "cuda:0", True)

    def predict_batch(self, texts: list[str], *, model="bertimbau"):
        if model != "bertimbau":
            raise InputError("Selecione BERTimbau ou SVM.")
        if self._bert is None:
            raise InferenceError("Modelo não foi preparado na inicialização.")
        return super().predict_batch(texts, model=model)


def verify_runtime_versions(path: Path) -> dict:
    expected = read_json(safe_file(path.parent, path.name))
    actual = {name: importlib.metadata.version(name) for name in expected["packages"]}
    if (
        actual["torch"].endswith("+cpu")
        or platform.python_version() != expected["python_version"]
        or any(
            actual[name].split("+")[0] != version for name, version in expected["packages"].items()
        )
    ):
        raise RuntimeError("Runtime diferente da combinação preparada; requer nova validação.")
    return actual


def import_zero_runtime():
    if os.environ.get("SPACES_ZERO_GPU", "").lower() != "true":
        raise RuntimeError("Este backend requer o runtime ZeroGPU, não a GPU local.")
    if "torch" in sys.modules and "spaces" not in sys.modules:
        raise RuntimeError("spaces precisa ser importado antes de torch.")
    spaces_module = importlib.import_module("spaces")
    if not importlib.import_module("spaces.config").Config.zero_gpu:
        raise RuntimeError("A emulação CUDA do runtime ZeroGPU não foi habilitada.")
    return spaces_module


def gpu_callback_transform(spaces_module):
    """Preserva metadata zerogpu no callback Gradio e sanitiza falhas do scheduler."""

    def transform(callback):
        allocated = spaces_module.GPU(duration=GPU_DURATION_SECONDS)(callback)

        @wraps(allocated)
        def guarded(text, model):
            from app.app import public_error_html

            try:
                validate_text(text)
                if model != "bertimbau":
                    return public_error_html("Selecione BERTimbau.")
            except InputError as error:
                return public_error_html(str(error))
            try:
                return allocated(text, model)
            except Exception:
                return public_error_html(
                    "GPU temporariamente indisponível. Tente novamente mais tarde."
                )

        return guarded

    return transform


def create_zerogpu_app(settings: ZeroGPUSettings):
    # Gates e versões antes de criar UI ou carregar pesos. Nenhuma aprovação fictícia aqui.
    settings.authorize()
    verify_runtime_versions(settings.runtime)
    spaces_module = import_zero_runtime()
    service = ZeroGPUInferenceService(settings.source())
    try:
        service.prepare()
    except Exception:
        raise ArtifactError(
            "Modelo ZeroGPU indisponível ou inválido; inicialização interrompida."
        ) from None
    from app.app import create_app
    from app.config import AppSettings

    app_settings = AppSettings(
        environment="hosted",
        descriptor=settings.descriptor,
        approval_root=settings.approval_root,
    )
    return create_app(
        service, settings=app_settings, callback_transform=gpu_callback_transform(spaces_module)
    )
