"""Seleção somente na validação, limites de busca e persistência do pipeline."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from reviewsense.baseline import (
    build_svm_pipeline,
    candidate_parameters,
    load_experiment,
    run_experiment,
)
from reviewsense.frozen_data import load_training_data


@pytest.fixture
def baseline_config():
    return json.loads(Path("configs/baseline.json").read_text())


def test_grid_is_small_and_covers_requested_dimensions(baseline_config):
    candidates = candidate_parameters(baseline_config)
    assert len(candidates) == 12
    assert {params["C"] for params in candidates} == {0.1, 1.0, 10.0}
    assert {tuple(params["ngram_range"]) for params in candidates} == {(1, 1), (1, 2)}
    assert {params["class_weight"] for params in candidates} == {None, "balanced"}


@pytest.mark.parametrize("change", ["large", "duplicate", "C", "ngrams", "weight", "inference"])
def test_grid_rejects_invalid_or_excessive_search(baseline_config, change):
    config = copy.deepcopy(baseline_config)
    if change == "large":
        config["search"]["C"] = [0.1, 0.5, 1, 2]
    elif change == "duplicate":
        config["search"]["C"] = [1, 1]
    elif change == "C":
        config["search"]["C"] = [0]
    elif change == "ngrams":
        config["search"]["ngram_range"] = [[1, 3]]
    elif change == "weight":
        config["search"]["class_weight"] = ["invalid"]
    else:
        config["inference"]["batch_size"] = 0
    with pytest.raises(ValueError):
        candidate_parameters(config)


def test_pipeline_preserves_accents_negations_and_fits_only_training(baseline_config):
    params = candidate_parameters(baseline_config)[0]
    pipeline = build_svm_pipeline(baseline_config, params)
    texts = [
        "não gostei péssimo",
        "não gostei ruim",
        "regular razoável",
        "regular mediano",
        "ótimo excelente",
        "ótimo perfeito",
    ]
    pipeline.fit(texts, [0, 0, 1, 1, 2, 2])
    vocabulary = pipeline.named_steps["tfidf"].vocabulary_.copy()
    idf = pipeline.named_steps["tfidf"].idf_.copy()
    pipeline.predict(["exclusivavalidacao não ótimo"])
    assert "exclusivavalidacao" not in vocabulary
    assert "não" in vocabulary and "ótimo" in vocabulary
    assert pipeline.named_steps["tfidf"].vocabulary_ == vocabulary
    np.testing.assert_array_equal(pipeline.named_steps["tfidf"].idf_, idf)
    assert not hasattr(pipeline, "predict_proba")


def test_experiment_uses_training_only_selects_and_roundtrips(
    frozen_training, baseline_config, tmp_path, monkeypatch
):
    directory, reference, _ = frozen_training
    config = copy.deepcopy(baseline_config)
    config["search"]["C"] = [0.1, 1]
    config["search"]["ngram_range"] = [[1, 1]]
    config["search"]["class_weight"] = [None]
    config["inference"]["repeats"] = 1
    train, val, _ = load_training_data(directory, reference)
    fit_transform = TfidfVectorizer.fit_transform
    fitted_text_sets = []

    def track_fit(self, documents, *args, **kwargs):
        fitted_text_sets.append(set(documents))
        assert set(documents) == set(train.review_text)
        assert not set(documents) & set(val.review_text)
        return fit_transform(self, documents, *args, **kwargs)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", track_fit)
    output = tmp_path / "baseline"
    result = run_experiment(directory, reference, config, output, progress=False)
    assert len(fitted_text_sets) == 2
    assert len(result["candidates"]) == 2
    assert result["selected"]["metrics"]["f1_macro"] == max(
        result["dummy"]["metrics"]["f1_macro"],
        *(candidate["metrics"]["f1_macro"] for candidate in result["candidates"]),
    )
    # Ordem fixa resolve empates; neste fixture ambos os SVMs são perfeitos.
    assert result["best_svm_candidate_id"] == 1
    assert result["selected"]["refit_on_train_plus_val"] is False
    assert result["model_artifact"]["validation_roundtrip_verified"] is True
    assert result["data"]["test_csv_opened"] is False
    model, saved = load_experiment(output)
    assert saved == result
    assert model.predict(val.review_text.tolist()).shape == (len(val),)
    assert "sample_ids" not in json.dumps(result)
    assert "péssimo" not in json.dumps(result, ensure_ascii=False)
    with pytest.raises(FileExistsError):
        run_experiment(directory, reference, config, output, progress=False)
    (output / "selected_model.joblib").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        load_experiment(output)


def test_nonconverged_candidates_cannot_be_selected(
    frozen_training, baseline_config, tmp_path, monkeypatch
):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC

    directory, reference, _ = frozen_training
    baseline_config["search"] = {"C": [1.0], "ngram_range": [[1, 1]], "class_weight": [None]}
    baseline_config["inference"]["repeats"] = 1
    original_fit = LinearSVC.fit

    def nonconverged_fit(self, x, y, *args, **kwargs):
        fitted = original_fit(self, x, y, *args, **kwargs)
        warnings.warn("synthetic convergence warning", ConvergenceWarning, stacklevel=2)
        return fitted

    monkeypatch.setattr(LinearSVC, "fit", nonconverged_fit)
    output = tmp_path / "nonconverged"
    with pytest.raises(RuntimeError, match="Nenhum SVM convergiu"):
        run_experiment(directory, reference, baseline_config, output, progress=False)
    assert not output.exists()
