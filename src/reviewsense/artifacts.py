"""Contratos e integridade de inferência, sem dependências de dados ou treinamento."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path

from reviewsense.labels import LABEL_NAMES


def file_sha256(path: str | Path) -> str:
    """Calcula checksum em streaming, sem carregar pesos inteiros em memória."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


ARTIFACT_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
DOCUMENT_FILES = (
    "README.md",
    "BASE_MODEL_LICENSE.txt",
    "PROJECT_CODE_LICENSE.txt",
    "ATTRIBUTIONS.md",
    "DISTRIBUTION_STATUS.md",
)
PACKAGE_FILES = tuple(sorted(ARTIFACT_FILES + DOCUMENT_FILES))
BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
BASE_REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _no_symlink_ancestors(path: Path) -> None:
    if ".." in path.parts:
        raise ValueError("Caminho com travessia de diretórios rejeitado.")
    for item in (path.absolute(), *path.absolute().parents):
        if item.is_symlink():
            raise ValueError("Links simbólicos não são permitidos.")


def safe_file(directory: Path, name: str) -> Path:
    _no_symlink_ancestors(directory)
    if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
        raise ValueError("Nome de arquivo inseguro.")
    path = directory / name
    if path.is_symlink():
        raise ValueError("Link simbólico rejeitado.")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Arquivo necessário ausente ou não regular: {name}.")
    return path


def inspect_directory(
    directory: Path, required: tuple[str, ...], *, ignored: set[str] | None = None
):
    _no_symlink_ancestors(directory)
    actual = {item.name for item in directory.iterdir()}
    if set(required) - actual or actual - set(required) - (ignored or set()):
        raise ValueError("Diretório incompleto ou com arquivos inesperados.")
    # Mesmo arquivos explicitamente excluídos não podem ser links ou diretórios.
    for name in actual:
        safe_file(directory, name)


def _metadata_hygiene(text: str) -> None:
    if re.search(r"/home/|/Users/|[A-Za-z]:[\\/]Users[\\/]|\bhf_[A-Za-z0-9]{20,}", text):
        raise ValueError("Caminho pessoal ou possível credencial nos metadados.")


def validate_configuration(directory: Path) -> None:
    config = read_json(safe_file(directory, "config.json"))
    tokenizer = read_json(safe_file(directory, "tokenizer_config.json"))
    if (
        config.get("model_type") != "bert"
        or config.get("architectures") != ["BertForSequenceClassification"]
        or config.get("id2label") != {str(k): v for k, v in LABEL_NAMES.items()}
        or config.get("label2id") != {v: k for k, v in LABEL_NAMES.items()}
        or tokenizer.get("do_lower_case") is not False
        or tokenizer.get("strip_accents") is True
        or config.get("auto_map")
        or tokenizer.get("auto_map")
    ):
        raise ValueError("Arquitetura, tokenizer ou mapeamento incompatível com o modelo aprovado.")
    for name in ("config.json", "tokenizer_config.json"):
        _metadata_hygiene((directory / name).read_text())


def file_inventory(directory: Path, names: tuple[str, ...]) -> dict:
    return {
        name: {
            "bytes": safe_file(directory, name).stat().st_size,
            "sha256": file_sha256(safe_file(directory, name)),
        }
        for name in sorted(names)
    }
