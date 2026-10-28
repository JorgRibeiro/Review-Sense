"""Carga, auditoria, limpeza conservadora e splits sem textos idênticos.

Nenhum vocabulário, tokenizer ou modelo é ajustado neste módulo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd
from sklearn.model_selection import train_test_split

from reviewsense.labels import LABEL_NAMES, rating_to_label

DATASET_REVISION = "4639429ec698d7821fc99a0bc665fa213d9fcd5a"
DATASET_URL = (
    "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/"
    f"{DATASET_REVISION}/B2W-Reviews01.csv"
)
# Checksum da cópia oficial fixada, validado na integração real.
DATASET_SHA256 = "821fb0bf9f7230b0fba4e4f9fadd75a66d1a9ff0b1657810791d33007eb2ab38"
PREPROCESSING_VERSION = "nfc-whitespace-deduplicate-drop-label-conflicts-v1"
REQUIRED_COLUMNS = {"review_text", "overall_rating"}
PREPARED_COLUMNS = ["sample_id", "review_text", "label"]


def file_sha256(path: str | Path) -> str:
    """Calcula checksum sem carregar o arquivo inteiro em memória."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_b2w(cache_dir: str | Path = "data/raw") -> Path:
    """Baixa a fonte original fixada, verifica integridade e reutiliza o cache.

    Download temporário e substituição apenas após verificar o checksum.
    Cache divergente causa erro, em vez de uma troca silenciosa de dados.
    """
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "B2W-Reviews01.csv"
    if destination.exists():
        if file_sha256(destination) != DATASET_SHA256:
            raise ValueError("Checksum do cache divergente; remova-o e baixe novamente.")
        return destination
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(DATASET_URL, timeout=120) as response:
                shutil.copyfileobj(response, stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if file_sha256(temporary) != DATASET_SHA256:
            raise ValueError("Checksum inesperado no download do B2W-Reviews01.")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_b2w(path: str | Path | None = None, *, cache_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Carrega CSV UTF-8 com vírgula ou ponto e vírgula; preserva textos como 'NA'.

    Um caminho explícito permite uso offline e testes. Seu checksum é registrado,
    mas não precisa corresponder ao corpus oficial. Apenas campos vazios do CSV
    são interpretados como ausentes; valores inválidos do alvo causam erro.
    """
    source = Path(path) if path is not None else download_b2w(cache_dir)
    with source.open(encoding="utf-8-sig") as stream:
        header = stream.readline()
    delimiter = None
    for candidate in (",", ";"):
        columns = next(csv.reader([header], delimiter=candidate))
        if REQUIRED_COLUMNS.issubset(columns):
            if len(columns) != len(set(columns)):
                raise ValueError("O CSV contém nomes de colunas duplicados.")
            delimiter = candidate
            break
    if delimiter is None:
        raise ValueError("Cabeçalho inválido: esperados review_text e overall_rating.")
    frame = pd.read_csv(
        source,
        sep=delimiter,
        encoding="utf-8-sig",
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
        dtype={"review_text": "string"},
    )
    validate_dataset(frame)
    frame.attrs["source"] = {
        "kind": "local_csv" if path is not None else "official_csv",
        "url": DATASET_URL if path is None else None,
        "revision": DATASET_REVISION if path is None else None,
        "sha256": file_sha256(source),
    }
    return frame


def normalize_text(value: object) -> str:
    """Normaliza NFC, controles ASCII e espaços; preserva caixa e conteúdo.

    Ausentes viram string vazia, para posterior descarte documentado. Não remove
    acentos, stopwords, negações, emojis ou pontuação. Não converte números em texto.
    """
    if not isinstance(value, str):
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return ""
        raise ValueError("review_text deve conter somente strings ou valores ausentes.")
    text = unicodedata.normalize("NFC", value)
    return " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", text).split())


def validate_dataset(frame: pd.DataFrame) -> None:
    """Valida contrato mínimo; textos ausentes são auditados antes da limpeza."""
    if not frame.columns.is_unique:
        raise ValueError("O dataset contém nomes de colunas duplicados.")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    if frame.empty:
        raise ValueError("O dataset está vazio.")
    try:
        frame["overall_rating"].map(rating_to_label)
        frame["review_text"].map(normalize_text)
    except ValueError as error:
        raise ValueError(f"Dataset inválido: {error}") from error


def _class_distribution(labels: pd.Series) -> dict[str, dict[str, int | float]]:
    counts = labels.value_counts()
    return {
        name: {
            "count": int(counts.get(label, 0)),
            "fraction": float(counts.get(label, 0) / len(labels)) if len(labels) else 0.0,
        }
        for label, name in LABEL_NAMES.items()
    }


def audit_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Retorna somente estatísticas agregadas, sem exemplos ou identificadores pessoais.

    Contagens de duplicatas excedentes indicam quantas linhas além da primeira
    existem; grupos conflitantes têm mais de uma classe derivada das estrelas.
    Estatísticas de comprimento excluem textos vazios/ausentes.
    """
    validate_dataset(frame)
    texts = frame["review_text"].map(normalize_text)
    labels = frame["overall_rating"].map(rating_to_label)
    valid = texts.ne("")
    groups = pd.DataFrame(
        {"text": texts[valid], "label": labels[valid], "rating": frame.loc[valid, "overall_rating"]}
    )
    grouped = groups.groupby("text", sort=False)
    sizes = grouped.size()
    conflicts = grouped["label"].nunique().gt(1)
    characters = texts[valid].str.len()
    words = texts[valid].str.split().str.len()
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "empty_texts_including_missing": int((~valid).sum()),
        "duplicate_rows_excess": int(frame.duplicated().sum()),
        "duplicate_text_groups": int(sizes.gt(1).sum()),
        "duplicate_text_rows_excess": int((sizes - 1).sum()),
        "conflicting_label_groups": int(conflicts.sum()),
        "conflicting_label_rows": int(sizes[conflicts].sum()),
        "conflicting_rating_groups": int(grouped["rating"].nunique().gt(1).sum()),
        "rating_distribution": {
            str(rating): int((frame["overall_rating"] == rating).sum()) for rating in range(1, 6)
        },
        "class_distribution": _class_distribution(labels),
        "text_characters": characters.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .dropna()
        .to_dict(),
        "text_words_whitespace": words.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .dropna()
        .to_dict(),
    }


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Remove vazios, todos os grupos de classes conflitantes e repetições.

    Textos com notas diferentes mas mesma classe não são conflitos de classe.
    Retorna apenas texto, alvo e hash do texto (sample_id), sem metadados de origem.
    Ordenar pelos hashes torna os splits independentes da ordem das linhas brutas.
    """
    validate_dataset(frame)
    prepared = pd.DataFrame(
        {
            "review_text": frame["review_text"].map(normalize_text),
            "label": frame["overall_rating"].map(rating_to_label),
        }
    )
    empty = prepared["review_text"].eq("")
    prepared = prepared.loc[~empty].copy()
    conflicting = prepared.groupby("review_text")["label"].transform("nunique").gt(1)
    conflicts_removed = int(conflicting.sum())
    prepared = prepared.loc[~conflicting].copy()
    before_deduplication = len(prepared)
    prepared = prepared.drop_duplicates("review_text").copy()
    prepared["sample_id"] = prepared["review_text"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    prepared = prepared[PREPARED_COLUMNS].sort_values("sample_id").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("Nenhuma avaliação utilizável após a limpeza.")
    report: dict[str, int | str] = {
        "policy": PREPROCESSING_VERSION,
        "input_rows": len(frame),
        "empty_rows_removed": int(empty.sum()),
        "conflicting_rows_removed": conflicts_removed,
        "duplicate_rows_removed": before_deduplication - len(prepared),
        "output_rows": len(prepared),
    }
    return prepared, report


def _validate_prepared(frame: pd.DataFrame) -> None:
    if set(frame.columns) != set(PREPARED_COLUMNS) or frame.empty:
        raise ValueError("Esperado dataset preparado não vazio com sample_id, review_text e label.")
    if frame.isna().any().any() or not frame["label"].isin(LABEL_NAMES).all():
        raise ValueError("Dataset preparado contém valores ausentes ou classes inválidas.")
    texts = frame["review_text"].map(normalize_text)
    if texts.eq("").any() or not texts.equals(frame["review_text"]):
        raise ValueError("Dataset preparado contém textos vazios ou não normalizados.")
    expected = texts.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    if not expected.equals(frame["sample_id"]) or frame["sample_id"].duplicated().any():
        raise ValueError("IDs inválidos ou textos duplicados; execute prepare_dataset primeiro.")


def split_dataset(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Divide dados deduplicados em ~70/15/15, com estratificação por classe.

    Falha explicitamente caso o conjunto seja pequeno demais para estratificar.
    A garantia cobre textos normalizados idênticos; não similares, autores ou produtos.
    """
    fractions = (train_size, val_size, test_size)
    if any(not math.isfinite(size) or not 0 < size < 1 for size in fractions):
        raise ValueError("As frações devem estar entre 0 e 1.")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("As frações de treino, validação e teste devem somar 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed deve ser um inteiro entre 0 e 2**32 - 1.")
    _validate_prepared(frame)
    if set(frame["label"].unique()) != set(LABEL_NAMES):
        raise ValueError("As três classes devem estar presentes para estratificação.")
    ordered = frame.sort_values("sample_id").reset_index(drop=True)
    try:
        train, held_out = train_test_split(
            ordered, train_size=train_size, random_state=seed, stratify=ordered["label"]
        )
        val, test = train_test_split(
            held_out,
            test_size=test_size / (val_size + test_size),
            random_state=seed,
            stratify=held_out["label"],
        )
    except ValueError as error:
        raise ValueError(
            "Amostras insuficientes para os splits estratificados solicitados."
        ) from error
    splits = {
        name: part.sort_values("sample_id").reset_index(drop=True)
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    validate_splits(splits, expected=ordered)
    return splits


def validate_splits(
    splits: Mapping[str, pd.DataFrame], *, expected: pd.DataFrame | None = None
) -> None:
    """Verifica integridade, disjunção por texto/ID, classes e cobertura opcional."""
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Esperados os splits train, val e test.")
    seen: set[str] = set()
    for name, part in splits.items():
        _validate_prepared(part)
        ids = set(part["sample_id"])
        if seen & ids:
            raise ValueError("Vazamento detectado: texto presente em mais de um split.")
        if set(part["label"].unique()) != set(LABEL_NAMES):
            raise ValueError(f"Split {name} não contém todas as classes.")
        seen.update(ids)
    if expected is not None:
        _validate_prepared(expected)
        combined = pd.concat(splits.values(), ignore_index=True)
        if dataset_fingerprint(combined) != dataset_fingerprint(expected):
            raise ValueError("Os splits não cobrem exatamente os dados preparados.")


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 do conteúdo preparado em ordem canônica (texto, alvo e ID)."""
    digest = hashlib.sha256()
    for row in frame[PREPARED_COLUMNS].sort_values("sample_id").itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(
    prepared: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
    *,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Registra configuração, fonte, ambiente, fingerprints e IDs, sem textos."""
    validate_splits(splits, expected=prepared)
    return {
        "schema_version": 1,
        "preprocessing": PREPROCESSING_VERSION,
        "source": dict(source),
        "config": dict(config),
        "python": platform.python_version(),
        "dataset_license": "CC-BY-NC-SA-4.0 (fonte original)",
        "label_names": {str(label): name for label, name in LABEL_NAMES.items()},
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("reviewsense-ptbr", "pandas", "numpy", "scikit-learn")
        },
        "prepared_fingerprint": dataset_fingerprint(prepared),
        "splits": {
            name: {
                "rows": len(part),
                "fingerprint": dataset_fingerprint(part),
                "class_distribution": _class_distribution(part["label"]),
                "sample_ids": part["sample_id"].tolist(),
            }
            for name, part in splits.items()
        },
    }


def _validate_manifest(splits: Mapping[str, pd.DataFrame], manifest: Mapping[str, Any]) -> None:
    """Compara os arquivos preparados com os IDs e fingerprints congelados."""
    validate_splits(splits)
    if manifest.get("schema_version") != 1:
        raise ValueError("Versão de manifest não suportada.")
    if manifest.get("preprocessing") != PREPROCESSING_VERSION:
        raise ValueError("Política do manifest difere da implementação atual.")
    combined = pd.concat(splits.values(), ignore_index=True)
    if manifest.get("prepared_fingerprint") != dataset_fingerprint(combined):
        raise ValueError("Fingerprint do conjunto preparado diverge do manifest.")
    declared = manifest.get("splits", {})
    if set(declared) != set(splits):
        raise ValueError("Splits declarados no manifest são inválidos.")
    for name, part in splits.items():
        info = declared[name]
        if (
            info.get("fingerprint") != dataset_fingerprint(part)
            or info.get("sample_ids") != part["sample_id"].tolist()
            or info.get("rows") != len(part)
            or info.get("class_distribution") != _class_distribution(part["label"])
        ):
            raise ValueError(f"Split {name} diverge do manifest.")


def load_artifacts(
    output_dir: str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Reabre splits congelados, preservando textos literais e verificando integridade.

    Deve ser preferido à leitura genérica de CSV, que pode interpretar 'NA' ou
    'null' como ausentes. Rejeita modificações no texto, alvo, IDs ou atribuição.
    """
    directory = Path(output_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    splits = {
        name: pd.read_csv(
            directory / f"{name}.csv",
            keep_default_na=False,
            dtype={"sample_id": str, "review_text": str, "label": "int64"},
        )
        for name in ("train", "val", "test")
    }
    _validate_manifest(splits, manifest)
    return splits, manifest, audit


def save_artifacts(
    output_dir: str | Path,
    *,
    splits: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> Path:
    """Salva CSVs locais e JSONs; recusa substituir uma execução já congelada.

    O diretório de saída deve estar ausente. Todos os arquivos são preparados em
    um diretório temporário antes da publicação local por rename.
    """
    _validate_manifest(splits, manifest)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Artefatos já existem em {destination}; use outro diretório.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        for name, part in splits.items():
            part.to_csv(staging / f"{name}.csv", index=False)
        for name, value in (("manifest", manifest), ("audit", audit)):
            (staging / f"{name}.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        staging.rename(destination)
    return destination


def main() -> None:
    """CLI da Fase 1; produz artefatos locais, sem treinar modelos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, help="CSV local opcional, delimitado por vírgula ou ponto e vírgula."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.json"))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Diretório de saída já existe; use outro para preservar os splits.")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frame = load_b2w(args.csv, cache_dir=args.cache_dir)
    audit = audit_dataset(frame)
    prepared, cleaning = prepare_dataset(frame)
    splits = split_dataset(prepared, **config)
    manifest = build_manifest(prepared, splits, source=frame.attrs["source"], config=config)
    audit["cleaning"] = cleaning
    save_artifacts(args.output_dir, splits=splits, manifest=manifest, audit=audit)
    print(
        json.dumps(
            {
                "cleaning": cleaning,
                "splits": {name: len(part) for name, part in splits.items()},
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
