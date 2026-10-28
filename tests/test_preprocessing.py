"""Normalização, validação, auditoria e políticas explícitas de limpeza."""

import pandas as pd
import pytest

from reviewsense.data import (
    audit_dataset,
    dataset_fingerprint,
    load_b2w,
    normalize_text,
    prepare_dataset,
    validate_dataset,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("  NÃO\t gostei!\n Péssimo 😞  ", "NÃO gostei! Péssimo 😞"),
        ("na\u0303o gostei", "não gostei"),
        ("bom\x00demais", "bom demais"),
        ("NA", "NA"),
        ("null", "null"),
        ("", ""),
        (None, ""),
        (pd.NA, ""),
        (float("nan"), ""),
    ],
)
def test_normalization_preserves_information(value, expected):
    assert normalize_text(value) == expected
    assert normalize_text(expected) == expected


@pytest.mark.parametrize("value", [123, True, [], {}])
def test_normalization_rejects_nontext(value):
    with pytest.raises(ValueError, match="strings"):
        normalize_text(value)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"review_text": ["bom"]}),
        pd.DataFrame({"review_text": [], "overall_rating": []}),
        pd.DataFrame({"review_text": ["bom"], "overall_rating": [None]}),
        pd.DataFrame({"review_text": [123], "overall_rating": [5]}),
        pd.DataFrame({"review_text": ["bom"], "overall_rating": [6]}),
        pd.DataFrame([["bom", "bom", 5]], columns=["review_text", "review_text", "overall_rating"]),
    ],
)
def test_schema_validation_rejects_bad_inputs(frame):
    with pytest.raises(ValueError):
        validate_dataset(frame)


def dirty_reviews():
    return pd.DataFrame(
        {
            "review_text": ["  ótimo! ", "ótimo!", "ruim", "ruim", "regular", None, " \t"],
            "overall_rating": [4, 5, 1, 5, 3, 2, 3],
        }
    )


def test_audit_reports_missing_duplicates_and_conflicts():
    frame = dirty_reviews()
    original = frame.copy(deep=True)
    report = audit_dataset(frame)
    assert report["rows"] == 7
    assert report["missing_values"]["review_text"] == 1
    assert report["empty_texts_including_missing"] == 2
    assert report["duplicate_text_groups"] == 2
    assert report["duplicate_text_rows_excess"] == 2
    assert report["conflicting_rating_groups"] == 2
    assert report["conflicting_label_groups"] == 1
    assert report["conflicting_label_rows"] == 2
    assert report["text_characters"]["count"] == 5
    assert report["class_distribution"]["positivo"]["count"] == 3
    pd.testing.assert_frame_equal(frame, original)


def test_cleaning_removes_whole_conflicting_groups_and_metadata():
    prepared, report = prepare_dataset(dirty_reviews())
    assert set(prepared.columns) == {"sample_id", "review_text", "label"}
    assert dict(zip(prepared.review_text, prepared.label, strict=True)) == {
        "ótimo!": 2,
        "regular": 1,
    }
    assert report["empty_rows_removed"] == 2
    assert report["conflicting_rows_removed"] == 2
    assert report["duplicate_rows_removed"] == 1
    assert report["input_rows"] == report["output_rows"] + sum(
        report[key]
        for key in ("empty_rows_removed", "conflicting_rows_removed", "duplicate_rows_removed")
    )


def test_preparation_is_order_independent_and_does_not_mutate(reviews):
    original = reviews.copy(deep=True)
    prepared, _ = prepare_dataset(reviews)
    shuffled, _ = prepare_dataset(reviews.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(prepared, shuffled)
    pd.testing.assert_frame_equal(reviews, original)
    assert "reviewer_id" not in prepared
    assert dataset_fingerprint(prepared) == dataset_fingerprint(shuffled)
    modified = prepared.copy()
    modified.loc[0, "label"] = (modified.loc[0, "label"] + 1) % 3
    assert dataset_fingerprint(modified) != dataset_fingerprint(prepared)


def test_unicode_equivalence_deduplicates():
    frame = pd.DataFrame(
        {"review_text": ["na\u0303o gostei", "não gostei"], "overall_rating": [1, 2]}
    )
    prepared, report = prepare_dataset(frame)
    assert len(prepared) == 1
    assert report["duplicate_rows_removed"] == 1


def test_no_usable_rows_is_an_explicit_error():
    with pytest.raises(ValueError, match="Nenhuma avaliação"):
        prepare_dataset(pd.DataFrame({"review_text": [None], "overall_rating": [5]}))


@pytest.mark.parametrize("delimiter", [",", ";"])
def test_csv_loading_handles_quotes_multiline_and_literal_na(tmp_path, delimiter):
    path = tmp_path / "reviews.csv"
    pd.DataFrame(
        {
            "review_text": ['não gostei; "ruim", péssimo\nmesmo', "NA", "null", None],
            "overall_rating": [1, 3, 4, 5],
        }
    ).to_csv(path, sep=delimiter, index=False)
    loaded = load_b2w(path)
    assert loaded.loc[0, "review_text"] == 'não gostei; "ruim", péssimo\nmesmo'
    assert loaded.loc[1, "review_text"] == "NA"
    assert loaded.loc[2, "review_text"] == "null"
    assert pd.isna(loaded.loc[3, "review_text"])
    assert len(loaded.attrs["source"]["sha256"]) == 64
    assert loaded.attrs["source"]["kind"] == "local_csv"


@pytest.mark.parametrize(
    "header", ["text,rating\nbom,5", "review_text,review_text,overall_rating\na,b,5"]
)
def test_csv_rejects_bad_headers(tmp_path, header):
    path = tmp_path / "bad.csv"
    path.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError):
        load_b2w(path)


def test_csv_preserves_numeric_looking_text(tmp_path):
    path = tmp_path / "reviews.csv"
    path.write_text("review_text,overall_rating\n123,1\n456,5\n", encoding="utf-8")
    loaded = load_b2w(path)
    assert loaded["review_text"].tolist() == ["123", "456"]
