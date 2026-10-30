"""Métricas com classes ausentes nas predições e medição sem resultados inventados."""

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from reviewsense.evaluate import benchmark_inference, classification_metrics, plot_confusion_matrix


def test_metrics_include_all_classes_and_zero_divisions():
    metrics = classification_metrics([0, 0, 1, 2], [2, 2, 2, 2])
    assert metrics["confusion_matrix"] == [[0, 0, 2], [0, 0, 1], [0, 0, 1]]
    assert metrics["accuracy"] == 0.25
    assert metrics["f1_macro"] == pytest.approx(0.4 / 3)
    assert metrics["precision_macro"] == pytest.approx(0.25 / 3)
    assert metrics["recall_macro"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["neutro"]["f1-score"] == 0
    assert metrics["per_class"]["negativo"]["support"] == 2


@pytest.mark.parametrize("true, pred", [([], []), ([0], [0, 1]), ([3], [0]), ([0], [9])])
def test_metrics_reject_invalid_labels(true, pred):
    with pytest.raises(ValueError):
        classification_metrics(true, pred)


def test_benchmark_reports_real_repetitions_and_batches():
    texts = ["um", "dois", "três", "quatro", "cinco"]
    model = DummyClassifier(strategy="most_frequent").fit(texts, [0, 2, 2, 1, 2])
    pred, timing = benchmark_inference(model, texts, batch_size=2, repeats=2, warmup_batches=1)
    np.testing.assert_array_equal(pred, [2, 2, 2, 2, 2])
    assert len(timing["seconds_per_repeat"]) == 2
    assert timing["samples_per_repeat"] == 5
    assert timing["median_seconds"] > 0
    assert timing["milliseconds_per_text_amortized"] == timing["median_seconds"] * 1000 / 5


def test_confusion_plot_uses_aggregate_labels():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plot_confusion_matrix(classification_metrics([0, 1, 2], [0, 1, 2]), title="Sintético")
    assert figure.axes[0].get_title() == "Sintético"
    plt.close(figure)
