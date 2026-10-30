"""Materialização offline; aceita somente links folha para blobs regulares do mesmo repo."""

import os
from unittest.mock import Mock

import pytest

from reviewsense.hub_cache import materialize_hub_snapshot
from reviewsense.model_packaging import PACKAGE_FILES, file_inventory
from reviewsense.model_resolution import LocalPackage, canonical_digest
from scripts.verify_phase4b_package import no_network


def materialize(cache, candidate):
    return materialize_hub_snapshot(
        cache["root"], "example/model", "a" * 40, candidate["descriptor"]
    )


def test_realistic_links_materialized_offline_and_source_preserved(
    candidate, hub_cache, monkeypatch
):
    import huggingface_hub

    cache = hub_cache(candidate["package"])
    # Entradas extras de um cache pré-existente nunca entram na allowlist ou são abertas.
    extra = cache["snapshot"] / "test.csv"
    extra.symlink_to("/untrusted/must-not-open")
    forbidden = Mock(side_effect=AssertionError("Hub/API/token forbidden"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "get_token", forbidden)
    before = file_inventory(candidate["package"], PACKAGE_FILES)
    links = {name: os.readlink(cache["snapshot"] / name) for name in PACKAGE_FILES}
    with pytest.raises(ValueError):
        LocalPackage(cache["snapshot"], candidate["descriptor_path"], cache["root"]).resolve()
    with no_network() as network:
        directory = materialize(cache, candidate)
        assert (
            LocalPackage(directory, candidate["descriptor_path"], cache["root"]).resolve()
            == directory
        )
    assert network["blocked_connections"] == 0
    assert {p.name for p in directory.iterdir()} == set(PACKAGE_FILES)
    assert all(p.is_file() and not p.is_symlink() for p in directory.iterdir())
    assert file_inventory(directory, PACKAGE_FILES) == before
    assert file_inventory(candidate["package"], PACKAGE_FILES) == before
    assert all(os.readlink(cache["snapshot"] / name) == links[name] for name in PACKAGE_FILES)
    inode = (directory / "model.safetensors").stat().st_ino
    assert materialize(cache, candidate) == directory
    assert (directory / "model.safetensors").stat().st_ino == inode
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "mode",
    [
        "absolute",
        "escape",
        "other_repo",
        "chained",
        "dangling",
        "blob_directory",
        "leaf_directory",
        "snapshot_ancestor",
        "blobs_ancestor",
        "cache_ancestor",
        "wrong_hash",
        "oversize",
        "fifo",
    ],
)
def test_unsafe_cache_rejected_without_promoting_package(candidate, hub_cache, tmp_path, mode):
    cache = hub_cache(candidate["package"])
    leaf = cache["snapshot"] / "README.md"
    blob = cache["files"]["README.md"]
    if mode in {"absolute", "escape", "other_repo", "chained", "dangling"}:
        leaf.unlink()
        target = {
            "absolute": str(blob),
            "escape": "../../../../private.txt",
            "other_repo": "../../../../models--other--repo/blobs/" + blob.name,
            "chained": "README-alias.md",
            "dangling": "../../blobs/" + "f" * 40,
        }[mode]
        leaf.symlink_to(target)
    elif mode in {"blob_directory", "fifo"}:
        blob.unlink()
        if mode == "fifo":
            os.mkfifo(blob)
        else:
            blob.mkdir()
    elif mode == "leaf_directory":
        leaf.unlink()
        leaf.mkdir()
    elif mode in {"snapshot_ancestor", "blobs_ancestor"}:
        folder = cache["snapshot"].parent if mode == "snapshot_ancestor" else cache["blobs"]
        outside = tmp_path / "relocated"
        folder.rename(outside)
        folder.symlink_to(outside, target_is_directory=True)
    elif mode == "cache_ancestor":
        alias = tmp_path / "cache-alias"
        alias.symlink_to(cache["root"], target_is_directory=True)
        cache["root"] = alias
    elif mode == "wrong_hash":
        blob.write_bytes(b"X" * blob.stat().st_size)
    else:
        blob.write_bytes(blob.read_bytes() + b"extra")
    with pytest.raises((ValueError, OSError)):
        materialize(cache, candidate)
    outputs = cache["root"] / "reviewsense-materialized"
    assert not list(outputs.glob("**/" + canonical_digest(candidate["descriptor"])))
    assert not list(outputs.glob("**/.preparing-*"))


def test_blob_symlink_chain_rejected_even_for_identical_approved_bytes(
    candidate, hub_cache, tmp_path
):
    cache = hub_cache(candidate["package"])
    blob = cache["files"]["README.md"]
    outside = tmp_path / "approved-but-outside"
    blob.rename(outside)
    blob.symlink_to(outside)
    with pytest.raises(OSError):
        materialize(cache, candidate)


def test_materialized_corruption_never_repaired_or_overwritten(candidate, hub_cache):
    cache = hub_cache(candidate["package"])
    directory = materialize(cache, candidate)
    file = directory / "README.md"
    file.write_text("synthetic corruption")
    with pytest.raises(ValueError, match="Integridade"):
        materialize(cache, candidate)
    assert file.read_text() == "synthetic corruption"


def test_destination_parent_symlink_rejected(candidate, hub_cache, tmp_path):
    cache = hub_cache(candidate["package"])
    external = tmp_path / "external"
    external.mkdir()
    (cache["root"] / "reviewsense-materialized").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        materialize(cache, candidate)
    assert not list(external.iterdir())


def test_regular_files_in_snapshot_supported(candidate, hub_cache):
    cache = hub_cache(candidate["package"])
    for name in PACKAGE_FILES:
        leaf = cache["snapshot"] / name
        data = cache["files"][name].read_bytes()
        leaf.unlink()
        leaf.write_bytes(data)
    directory = materialize(cache, candidate)
    assert file_inventory(directory, PACKAGE_FILES) == candidate["descriptor"]["files"]


def test_unavailable_safe_open_support_fails_closed(candidate, hub_cache, monkeypatch):
    cache = hub_cache(candidate["package"])
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(NotImplementedError, match="POSIX"):
        materialize(cache, candidate)
