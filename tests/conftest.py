"""Dados sintéticos, independentes da rede e sem avaliações reais."""

import pandas as pd
import pytest


@pytest.fixture
def reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_text": [
                f"Avaliação sintética {rating}-{i}: não é igual!"
                for rating in (1, 3, 5)
                for i in range(100)
            ],
            "overall_rating": [rating for rating in (1, 3, 5) for _ in range(100)],
            "reviewer_id": [f"autor-{i}" for i in range(300)],
        }
    )
