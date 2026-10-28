"""Classes derivadas de estrelas: uma proxy, não anotação humana de sentimento."""

from numbers import Real

LABEL_NAMES: dict[int, str] = {0: "negativo", 1: "neutro", 2: "positivo"}
RATING_TO_LABEL: dict[int, int] = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}


def rating_to_label(rating: Real) -> int:
    """Converte uma nota numérica inteira de 1 a 5; rejeita ausentes e booleanos.

    Floats integrais são aceitos porque CSVs com nulos podem gerar esse dtype.
    Strings não são convertidas silenciosamente.
    """
    if isinstance(rating, bool) or not isinstance(rating, Real):
        raise ValueError("A nota deve ser um número inteiro entre 1 e 5.")
    if rating not in RATING_TO_LABEL:
        raise ValueError("A nota deve ser um número inteiro entre 1 e 5.")
    return RATING_TO_LABEL[int(rating)]
