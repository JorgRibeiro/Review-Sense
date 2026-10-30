"""Materializa somente bytes aprovados de um cache Hub local, sem seguir links arbitrários."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

from reviewsense.model_packaging import PACKAGE_FILES, _no_symlink_ancestors
from reviewsense.model_resolution import (
    REPO_ID,
    REVISION,
    canonical_digest,
    validate_descriptor,
    validate_runtime_package,
)


@contextmanager
def _directory_fd(path: Path):
    """Abre cada ancestral sem seguir symlinks e mantém os diretórios fixos por fd."""
    if ".." in path.parts:
        raise ValueError("Travessia de diretórios rejeitada.")
    if not hasattr(os, "O_NOFOLLOW") or os.open not in os.supports_dir_fd:
        raise NotImplementedError("Materialização segura requer suporte POSIX a dir_fd/O_NOFOLLOW.")
    with ExitStack() as stack:
        absolute = path.absolute()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(absolute.anchor, flags)
        stack.callback(os.close, fd)
        for part in absolute.parts[1:]:
            fd = os.open(part, flags, dir_fd=fd)
            stack.callback(os.close, fd)
        yield fd


def _copy_approved_file(snapshot_fd: int, blobs_fd: int, name: str, target: Path, record: dict):
    entry = os.stat(name, dir_fd=snapshot_fd, follow_symlinks=False)
    parent_fd, source_name = snapshot_fd, name
    if stat.S_ISLNK(entry.st_mode):
        link = os.readlink(name, dir_fd=snapshot_fd)
        # Somente o layout Hub de arquivos na raiz: snapshots/<sha>/x -> ../../blobs/<etag>.
        match = re.fullmatch(r"\.\./\.\./blobs/([a-f0-9]{40}|[a-f0-9]{64})", link)
        if not match:
            raise ValueError("Link de snapshot fora do blob permitido.")
        parent_fd, source_name = blobs_fd, match[1]
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(source_name, flags, dir_fd=parent_fd), "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("Arquivo de cache não regular.")
        digest, size = hashlib.sha256(), 0
        with target.open("xb") as output:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > record["bytes"]:
                    raise ValueError("Tamanho do cache diverge do descritor aprovado.")
                digest.update(chunk)
                output.write(chunk)
        if size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError("Hash/tamanho do cache diverge do descritor aprovado.")


def materialize_hub_snapshot(
    cache_root: Path, repo_id: str, revision: str, descriptor: dict
) -> Path:
    """Copia allowlist para diretório novo de arquivos regulares, validado antes da promoção.

    A raiz do cache é explícita e confiada pelo operador. Links são aceitos somente
    nas folhas do snapshot e para blobs regulares do mesmo repo; não se chama o Hub.
    Entradas não aprovadas do cache não são abertas ou copiadas. Destino existente
    só é reutilizado após validação integral; nunca é reparado ou sobrescrito.
    """
    validate_descriptor(descriptor)
    if not re.fullmatch(REPO_ID, repo_id) or not re.fullmatch(REVISION, revision):
        raise ValueError("Repo_id e revisão SHA imutável obrigatórios.")
    repo = cache_root / ("models--" + repo_id.replace("/", "--"))
    parent = cache_root / "reviewsense-materialized" / repo.name / revision
    destination = parent / canonical_digest(descriptor)
    _no_symlink_ancestors(parent)
    if os.path.lexists(destination):
        validate_runtime_package(destination, descriptor)
        return destination
    # Abertura segura antes de criar qualquer saída; cache ausente falha sem download.
    with (
        _directory_fd(repo / "snapshots" / revision) as snapshot_fd,
        _directory_fd(repo / "blobs") as blobs_fd,
    ):
        parent.mkdir(parents=True, exist_ok=True)
        _no_symlink_ancestors(parent)
        with tempfile.TemporaryDirectory(prefix=".preparing-", dir=parent) as temporary:
            staging = Path(temporary) / "model"
            staging.mkdir()
            for name in PACKAGE_FILES:
                _copy_approved_file(
                    snapshot_fd, blobs_fd, name, staging / name, descriptor["files"][name]
                )
            validate_runtime_package(staging, descriptor)
            if os.path.lexists(destination):
                raise FileExistsError("Destino criado durante a preparação; não sobrescrever.")
            staging.rename(destination)
    return destination
