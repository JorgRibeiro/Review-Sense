"""Prepara bundle de código ZeroGPU local por allowlist; não publica nem copia pesos."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from reviewsense.artifacts import _no_symlink_ancestors, file_sha256, safe_file
from reviewsense.model_resolution import scan_metadata, validate_descriptor

RUNTIME_MODULES = (
    "__init__.py",
    "labels.py",
    "artifacts.py",
    "model_resolution.py",
    "hub_cache.py",
    "predict.py",
    "zerogpu.py",
)
BUNDLE_FILES = {
    **{
        name: f"deploy/zerogpu/{name}"
        for name in (
            "app.py",
            "README.md",
            "requirements.txt",
            "runtime.json",
        )
    },
    **{f"reviewsense/{name}": f"src/reviewsense/{name}" for name in RUNTIME_MODULES},
    **{
        f"app/{name}": f"app/{name}"
        for name in (
            "__init__.py",
            "app.py",
            "config.py",
            "style.css",
        )
    },
    "model_descriptor.json": "publishing/inference_descriptor.json",
    "LICENSE": "LICENSE",
}


def validate_bundle(directory: Path, inventory: dict) -> None:
    _no_symlink_ancestors(directory)
    if set(inventory) != set(BUNDLE_FILES):
        raise ValueError("Allowlist do bundle de código divergente.")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink no bundle de código.")
        if path.is_dir():
            if path.relative_to(directory).as_posix() not in {"app", "reviewsense"}:
                raise ValueError("Diretório não permitido no bundle.")
            continue
        actual.add(path.relative_to(directory).as_posix())
    if actual != set(BUNDLE_FILES):
        raise ValueError("Bundle incompleto ou com arquivos extras.")
    for name in sorted(actual):
        path = directory / name
        safe_file(path.parent, path.name)
        if file_sha256(path) != inventory[name]:
            raise ValueError("Hash de código divergente no bundle.")
        text = path.read_text()
        if path.suffix == ".py":
            # Os próprios validadores contêm regex /home/|/Users/ sem dados pessoais.
            # Código exige um nome de usuário; credenciais continuam proibidas.
            if re.search(
                r"/(?:home|Users)/[A-Za-z0-9_.-]+|\bhf_[A-Za-z0-9]{20,}"
                r"|\bgh[pousr]_[A-Za-z0-9]{20,}|\bAKIA[A-Z0-9]{16}"
                r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                text,
            ):
                raise ValueError("Possível segredo ou caminho pessoal no código do bundle.")
        else:
            scan_metadata(text)
    validate_descriptor(json.loads((directory / "model_descriptor.json").read_text()))


def prepare_bundle(root: Path, destination: Path) -> dict:
    _no_symlink_ancestors(destination)
    if os.path.lexists(destination):
        raise FileExistsError("Candidato de código existente; não sobrescrever.")
    # Não permitir saída dentro das origens de código.
    for source in ("src", "app", "deploy", "publishing"):
        if destination.absolute().is_relative_to((root / source).absolute()):
            raise ValueError("Destino sobrepõe uma origem de código.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".space-code-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        inventory = {}
        for name, origin in sorted(BUNDLE_FILES.items()):
            path = root / origin
            safe_file(path.parent, path.name)
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target, follow_symlinks=False)
            inventory[name] = file_sha256(path)
        validate_bundle(staging, inventory)
        if os.path.lexists(destination):
            raise FileExistsError("Destino criado durante preparação; não sobrescrever.")
        staging.rename(destination)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=Path("outputs/phase4c-zerogpu/space-candidate")
    )
    args = parser.parse_args()
    inventory = prepare_bundle(Path.cwd(), args.destination)
    print(json.dumps({"files": inventory, "upload_authorized": False}, indent=2))


if __name__ == "__main__":
    main()
