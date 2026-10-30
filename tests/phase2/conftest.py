"""Fixtures sintéticas da Fase 2; nunca usam os CSVs reais."""

import json

import pytest

from reviewsense.data import build_manifest, prepare_dataset, save_artifacts, split_dataset


@pytest.fixture
def frozen_training(reviews, tmp_path):
    words = {
        1: "péssimo horrível defeito",
        3: "regular razoável mediano",
        5: "excelente ótimo perfeito",
    }
    reviews["review_text"] = [
        f"{words[row.overall_rating]} avaliação sintética exemplo{i}"
        for i, row in enumerate(reviews.itertuples())
    ]
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    manifest = build_manifest(prepared, splits, source={"kind": "synthetic"}, config={"seed": 42})
    directory = tmp_path / "frozen"
    save_artifacts(directory, splits=splits, manifest=manifest, audit={})
    reference = dict(manifest)
    reference["splits"] = {
        name: {key: value for key, value in info.items() if key != "sample_ids"}
        for name, info in manifest["splits"].items()
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    return directory, reference_path, splits
