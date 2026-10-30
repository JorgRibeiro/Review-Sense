"""Leitura exclusiva de treino/validação, sem abrir o CSV de teste da Fase 1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from reviewsense.data import (
    PREPARED_COLUMNS,
    PREPROCESSING_VERSION,
    dataset_fingerprint,
    normalize_text,
)
from reviewsense.labels import LABEL_NAMES


def validate_training_frame(frame: pd.DataFrame) -> None:
    """Valida esquema, classes, normalização e IDs sem alterar os dados."""
    if list(frame.columns) != PREPARED_COLUMNS or frame.empty:
        raise ValueError("Esperado split preparado com sample_id, review_text e label.")
    if frame.isna().any().any() or not frame["label"].isin(LABEL_NAMES).all():
        raise ValueError("Split contém valores ausentes ou classes inválidas.")
    if set(frame["label"].unique()) != set(LABEL_NAMES):
        raise ValueError("As três classes devem estar presentes no split.")
    texts = frame["review_text"].map(normalize_text)
    if texts.eq("").any() or not texts.equals(frame["review_text"]):
        raise ValueError("Textos vazios ou não normalizados no split.")
    expected_ids = texts.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    if not expected_ids.equals(frame["sample_id"]) or expected_ids.duplicated().any():
        raise ValueError("IDs inválidos ou textos duplicados no split.")


def validate_training_pair(train: pd.DataFrame, val: pd.DataFrame) -> None:
    """Exige splits válidos e disjuntos; só o texto será usado como feature."""
    validate_training_frame(train)
    validate_training_frame(val)
    if set(train["sample_id"]) & set(val["sample_id"]):
        raise ValueError("Vazamento entre treino e validação.")


def load_training_data(
    artifacts_dir: str | Path,
    reference_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Confere treino/validação contra manifest local e resumo versionado da Fase 1.

    Não chama load_artifacts, pois ele abre também test.csv. Não lê nem verifica
    o conteúdo do CSV de teste. A referência versionada ancora os fingerprints,
    evitando aceitar mudanças simultâneas de CSVs e manifest local.
    """
    directory = Path(artifacts_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Versão do manifest não suportada.")
    if not (
        manifest.get("preprocessing") == reference.get("preprocessing") == PREPROCESSING_VERSION
    ):
        raise ValueError("Política de preparação diverge da Fase 1.")
    if manifest.get("source") != reference.get("source"):
        raise ValueError("Fonte do manifest diverge da referência da Fase 1.")
    if manifest.get("prepared_fingerprint") != reference.get("prepared_fingerprint"):
        raise ValueError("Fingerprint global do manifest diverge da referência.")
    frames = {}
    provenance: dict[str, Any] = {
        "preprocessing": manifest["preprocessing"],
        "source": manifest["source"],
        "prepared_fingerprint": manifest["prepared_fingerprint"],
        "splits": {},
        "test_csv_opened": False,
    }
    for name in ("train", "val"):
        declared = manifest["splits"][name]
        expected = reference["splits"][name]
        frame = pd.read_csv(
            directory / f"{name}.csv",
            keep_default_na=False,
            dtype={"sample_id": str, "review_text": str, "label": "int64"},
        )
        validate_training_frame(frame)
        fingerprint = dataset_fingerprint(frame)
        counts = frame["label"].value_counts()
        distribution = {
            label_name: {"count": int(counts[label]), "fraction": float(counts[label] / len(frame))}
            for label, label_name in LABEL_NAMES.items()
        }
        if (
            fingerprint != declared["fingerprint"]
            or fingerprint != expected["fingerprint"]
            or len(frame) != declared["rows"]
            or len(frame) != expected["rows"]
            or frame["sample_id"].tolist() != declared["sample_ids"]
            or distribution != declared["class_distribution"]
            or distribution != expected["class_distribution"]
        ):
            raise ValueError(f"Split {name} diverge dos fingerprints congelados da Fase 1.")
        frames[name] = frame
        provenance["splits"][name] = {
            "rows": len(frame),
            "fingerprint": fingerprint,
            "class_distribution": distribution,
        }
    validate_training_pair(frames["train"], frames["val"])
    return frames["train"], frames["val"], provenance
