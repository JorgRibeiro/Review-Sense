"""Contrato de rótulos, incluindo entradas inválidas comuns em CSVs."""

import numpy as np
import pandas as pd
import pytest

from reviewsense.labels import LABEL_NAMES, rating_to_label


@pytest.mark.parametrize(
    "rating, expected", [(1, 0), (2, 0), (3, 1), (4, 2), (5, 2), (3.0, 1), (np.int64(5), 2)]
)
def test_rating_mapping(rating, expected):
    assert rating_to_label(rating) == expected


@pytest.mark.parametrize(
    "rating",
    [0, 6, -1, 2.5, None, float("nan"), float("inf"), "5", True, False, pd.NA, np.bool_(True)],
)
def test_invalid_rating_is_rejected(rating):
    with pytest.raises(ValueError, match="inteiro entre 1 e 5"):
        rating_to_label(rating)


def test_class_names_are_stable():
    assert LABEL_NAMES == {0: "negativo", 1: "neutro", 2: "positivo"}
