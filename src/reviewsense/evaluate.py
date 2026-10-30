"""Métricas agregadas e medição de inferência; nenhuma função carrega dados de teste."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from reviewsense.labels import LABEL_NAMES


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Calcula métricas macro/por classe e matriz com ordem explícita 0, 1, 2.

    Divisões indefinidas (classe nunca prevista, por exemplo) recebem zero.
    Linhas da matriz são classes reais; colunas, classes previstas.
    """
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if true.ndim != 1 or predicted.shape != true.shape or not len(true):
        raise ValueError("Rótulos reais e previstos devem ser vetores não vazios do mesmo tamanho.")
    if (
        not np.isin(true, list(LABEL_NAMES)).all()
        or not np.isin(predicted, list(LABEL_NAMES)).all()
    ):
        raise ValueError("Classes esperadas: 0, 1 e 2.")
    names = list(LABEL_NAMES.values())
    report = classification_report(
        true,
        predicted,
        labels=list(LABEL_NAMES),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    macro = report["macro avg"]
    return {
        "f1_macro": float(macro["f1-score"]),
        "precision_macro": float(macro["precision"]),
        "recall_macro": float(macro["recall"]),
        "accuracy": float(np.mean(true == predicted)),
        "per_class": {name: report[name] for name in names},
        "label_order": list(LABEL_NAMES),
        "label_names": names,
        "confusion_matrix": confusion_matrix(true, predicted, labels=list(LABEL_NAMES)).tolist(),
        "samples": len(true),
    }


def benchmark_inference(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 256,
    repeats: int = 3,
    warmup_batches: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cronometra predict, com pré-processamento e TF-IDF quando houver pipeline.

    Um warm-up explícito antecede as repetições sobre toda a validação. Reporta
    mediana e duração de cada passagem; ms/texto é throughput amortizado em lote,
    não latência de uma requisição individual. Não inclui I/O, métricas ou treino.
    """
    if batch_size < 1 or repeats < 1 or warmup_batches < 0 or not len(texts):
        raise ValueError("Amostras, batch_size e repeats positivos; warmup_batches não negativo.")
    values = list(texts)
    for _ in range(warmup_batches):
        model.predict(values[:batch_size])
    durations = []
    reference = None
    for _ in range(repeats):
        start = perf_counter()
        predictions = np.concatenate(
            [
                model.predict(values[offset : offset + batch_size])
                for offset in range(0, len(values), batch_size)
            ]
        )
        durations.append(perf_counter() - start)
        if reference is not None and not np.array_equal(reference, predictions):
            raise ValueError("Predições variaram entre repetições de inferência.")
        reference = predictions
    median = float(np.median(durations))
    return reference, {
        "batch_size": batch_size,
        "repeats": repeats,
        "warmup_batches": warmup_batches,
        "samples_per_repeat": len(values),
        "seconds_per_repeat": durations,
        "median_seconds": median,
        "milliseconds_per_text_amortized": median * 1000 / len(values),
        "texts_per_second": len(values) / median,
        "includes_text_preprocessing": True,
        "includes_disk_io": False,
    }


def plot_confusion_matrix(metrics: dict[str, Any], *, title: str):
    """Desenha somente contagens agregadas, sem avaliações ou IDs individuais."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axes = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        np.asarray(metrics["confusion_matrix"]), display_labels=metrics["label_names"]
    ).plot(ax=axes, colorbar=False, cmap="Blues", values_format="d")
    axes.set(title=title, xlabel="Classe prevista", ylabel="Classe real")
    figure.tight_layout()
    return figure
