"""Reprodutibilidade, estratificação, integridade e persistência dos splits."""

import json

import pandas as pd
import pytest

from reviewsense.data import (
    build_manifest,
    prepare_dataset,
    save_artifacts,
    split_dataset,
    validate_splits,
)


def test_splits_are_reproducible_stratified_and_cover_all_rows(reviews):
    # Duplicatas com variações de espaços não podem atravessar conjuntos.
    duplicates = reviews.iloc[:30].copy()
    duplicates["review_text"] = "  " + duplicates["review_text"] + "\n"
    prepared, _ = prepare_dataset(pd.concat([reviews, duplicates], ignore_index=True))
    splits = split_dataset(prepared)
    repeated = split_dataset(prepared.sample(frac=1, random_state=99))
    assert {name: len(part) for name, part in splits.items()} == {
        "train": 210,
        "val": 45,
        "test": 45,
    }
    assert sum(len(part) for part in splits.values()) == len(prepared) == len(reviews)
    validate_splits(splits, expected=prepared)
    for name, part in splits.items():
        pd.testing.assert_frame_equal(part, repeated[name])
        assert part["label"].value_counts().nunique() == 1
    different = split_dataset(prepared, seed=7)
    assert set(splits["train"].sample_id) != set(different["train"].sample_id)
    assert set(splits["train"].review_text).isdisjoint(splits["test"].review_text)


@pytest.mark.parametrize(
    "config",
    [
        {"train_size": 0.8},
        {"val_size": 0},
        {"test_size": -0.1},
        {"train_size": float("nan")},
        {"val_size": float("inf")},
        {"seed": True},
        {"seed": -1},
    ],
)
def test_invalid_split_config_is_rejected(reviews, config):
    prepared, _ = prepare_dataset(reviews)
    with pytest.raises(ValueError):
        split_dataset(prepared, **config)


def test_custom_fractions(reviews):
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared, train_size=0.6, val_size=0.2, test_size=0.2)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [180, 60, 60]


def test_duplicate_prepared_text_is_rejected(reviews):
    prepared, _ = prepare_dataset(reviews)
    with pytest.raises(ValueError, match="duplicados"):
        split_dataset(pd.concat([prepared, prepared.iloc[:1]], ignore_index=True))


def test_small_dataset_cannot_silently_skip_stratification(reviews):
    small = reviews.groupby("overall_rating").head(1)
    prepared, _ = prepare_dataset(small)
    with pytest.raises(ValueError, match="insuficientes"):
        split_dataset(prepared)


def test_missing_class_is_rejected(reviews):
    prepared, _ = prepare_dataset(reviews[reviews.overall_rating != 3])
    with pytest.raises(ValueError, match="três classes"):
        split_dataset(prepared)


def test_leakage_is_detected(reviews):
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    splits["val"] = pd.concat([splits["val"], splits["train"].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Vazamento"):
        validate_splits(splits)


def test_dropped_rows_and_changed_labels_are_detected(reviews):
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    splits["train"] = splits["train"].iloc[1:]
    with pytest.raises(ValueError, match="exatamente"):
        validate_splits(splits, expected=prepared)
    splits = split_dataset(prepared)
    splits["train"].loc[0, "label"] = (splits["train"].loc[0, "label"] + 1) % 3
    with pytest.raises(ValueError, match="exatamente"):
        validate_splits(splits, expected=prepared)


def test_manifest_and_artifacts_are_frozen(reviews, tmp_path):
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    manifest = build_manifest(prepared, splits, source={"kind": "synthetic"}, config={"seed": 42})
    assert "review_text" not in json.dumps(manifest)
    for name, part in splits.items():
        assert manifest["splits"][name]["sample_ids"] == part.sample_id.tolist()
    output = tmp_path / "v1"
    save_artifacts(output, splits=splits, manifest=manifest, audit={"rows": 300})
    assert json.loads((output / "manifest.json").read_text()) == manifest
    pd.testing.assert_frame_equal(pd.read_csv(output / "train.csv"), splits["train"])
    with pytest.raises(FileExistsError):
        save_artifacts(output, splits=splits, manifest=manifest, audit={})
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_loading_frozen_artifacts_preserves_literal_text(reviews, tmp_path):
    from reviewsense.data import load_artifacts

    reviews.loc[0, "review_text"] = "NA"
    reviews.loc[1, "review_text"] = "null"
    reviews.loc[2, "review_text"] = "123"
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    manifest = build_manifest(prepared, splits, source={}, config={"seed": 42})
    output = tmp_path / "frozen"
    save_artifacts(output, splits=splits, manifest=manifest, audit={"rows": 300})
    loaded, loaded_manifest, audit = load_artifacts(output)
    for name in splits:
        pd.testing.assert_frame_equal(loaded[name], splits[name])
    assert loaded_manifest == manifest
    assert audit == {"rows": 300}


@pytest.mark.parametrize("tamper", ["text", "label", "ids", "assignment"])
def test_loading_detects_changed_files_or_manifest(reviews, tmp_path, tamper):
    from reviewsense.data import load_artifacts

    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    manifest = build_manifest(prepared, splits, source={}, config={"seed": 42})
    output = tmp_path / "frozen"
    save_artifacts(output, splits=splits, manifest=manifest, audit={})
    if tamper == "text":
        changed = splits["train"].copy()
        changed.loc[0, "review_text"] = "texto adulterado"
        changed.to_csv(output / "train.csv", index=False)
    elif tamper == "label":
        changed = splits["train"].copy()
        changed.loc[0, "label"] = (changed.loc[0, "label"] + 1) % 3
        changed.to_csv(output / "train.csv", index=False)
    elif tamper == "assignment":
        # Trocar splits mantém o fingerprint global, mas altera a atribuição.
        splits["val"].to_csv(output / "test.csv", index=False)
        splits["test"].to_csv(output / "val.csv", index=False)
    else:
        manifest["splits"]["train"]["sample_ids"].reverse()
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_artifacts(output)
