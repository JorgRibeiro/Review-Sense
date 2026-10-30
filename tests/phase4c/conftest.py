"""Artefatos pequenos, exclusivamente sintéticos; nenhuma dependência do checkpoint real."""

import json
import shutil
from pathlib import Path

import pytest

from reviewsense.labels import LABEL_NAMES
from reviewsense.model_packaging import ARTIFACT_FILES, build_package, file_inventory
from reviewsense.model_resolution import descriptor_from_manifest


@pytest.fixture
def candidate(tmp_path):
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    documents = tmp_path / "documents"
    shutil.copytree(root / "publishing/model_card", documents)
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "O", "produto", "Não", "bom"]
        )
    }
    BertTokenizerFast(vocab=vocab, do_lower_case=False).save_pretrained(source)
    BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            id2label=LABEL_NAMES,
            label2id={v: k for k, v in LABEL_NAMES.items()},
        )
    ).save_pretrained(source)
    manifest = build_package(
        source,
        documents,
        tmp_path / "package",
        expected={k: v["sha256"] for k, v in file_inventory(source, ARTIFACT_FILES).items()},
        metrics=json.loads((root / "reports/phase3c_final_metrics.json").read_text())["models"][
            "bertimbau"
        ]["metrics"],
        provenance={"packaging_code_commit": "synthetic"},
    )
    descriptor = descriptor_from_manifest(manifest)
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor))
    return {
        "package": tmp_path / "package",
        "descriptor": descriptor,
        "descriptor_path": descriptor_path,
        "manifest": manifest,
        "source": source,
    }


@pytest.fixture
def hub_cache(tmp_path):
    """Layout realista de cache Hub: snapshots/<sha>/arquivo -> ../../blobs/<etag>."""
    from hashlib import sha1, sha256

    from reviewsense.model_packaging import PACKAGE_FILES

    def build(package):
        cache = tmp_path / "hf-cache"
        repo = cache / "models--example--model"
        blobs = repo / "blobs"
        snapshot = repo / "snapshots" / ("a" * 40)
        blobs.mkdir(parents=True)
        snapshot.mkdir(parents=True)
        paths = {}
        for name in PACKAGE_FILES:
            data = (package / name).read_bytes()
            # Cache pode usar etags Git de 40 caracteres ou SHA-256 de 64.
            etag = sha256(data).hexdigest()
            if name != "model.safetensors":
                etag = sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
                ).hexdigest()
            paths[name] = blobs / etag
            paths[name].write_bytes(data)
            (snapshot / name).symlink_to(f"../../blobs/{etag}")
        (snapshot / ".gitattributes").write_text("synthetic cached metadata")
        return {"root": cache, "snapshot": snapshot, "blobs": blobs, "files": paths}

    return build
