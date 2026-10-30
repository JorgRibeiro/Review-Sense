"""Configuração e amostragem do smoke: sem importação de Torch ou downloads."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from reviewsense.data import dataset_fingerprint
from reviewsense.frozen_data import validate_training_frame, validate_training_pair
from reviewsense.labels import LABEL_NAMES


@dataclass(frozen=True)
class SmokeConfig:
    """Limites explícitos impedem um treino completo acidental nesta fase."""

    checkpoint: str = "neuralmind/bert-base-portuguese-cased"
    revision: str = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
    seed: int = 42
    train_samples: int = 256
    val_samples: int = 96
    max_length: int = 128
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    epochs: int = 1
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = False
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    cpu_threads: int = 2

    def validate(self) -> None:
        """Recusa limites incompatíveis com smoke e revisões não fixadas."""
        if self.checkpoint != "neuralmind/bert-base-portuguese-cased":
            raise ValueError("Esta fase usa apenas o checkpoint BERTimbau Base cased.")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("A revisão deve ser um SHA de commit fixo do checkpoint.")
        limits = {
            "seed": (0, 2**32 - 1),
            "train_samples": (3, 256),
            "val_samples": (3, 96),
            "max_length": (8, 128),
            "train_batch_size": (1, 2),
            "eval_batch_size": (1, 2),
            "gradient_accumulation_steps": (1, 32),
            "epochs": (1, 1),
            "cpu_threads": (1, 16),
        }
        for name, (minimum, maximum) in limits.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"Configuração fora dos limites do smoke: {name}.")
        if self.effective_batch_size > 32:
            raise ValueError("Batch efetivo excede o limite diagnóstico de 32.")
        if self.max_length % 8:
            raise ValueError("max_length deve ser múltiplo de 8 para o padding do smoke.")
        if self.mixed_precision not in ("fp16", "no"):
            raise ValueError("Precisão esperada: fp16 ou no (FP32 explícito).")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing deve ser booleano.")
        for name in ("learning_rate", "max_grad_norm", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Parâmetro numérico inválido: {name}.")
            if value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Parâmetro numérico inválido: {name}.")

    @property
    def effective_batch_size(self) -> int:
        return self.train_batch_size * self.gradient_accumulation_steps


def label_mappings() -> tuple[dict[int, str], dict[str, int]]:
    """Mantém os mesmos IDs das Fases 1 e 2 na cabeça de classificação."""
    id2label = dict(LABEL_NAMES)
    return id2label, {name: label for label, name in id2label.items()}


def oom_fallback(config: SmokeConfig) -> SmokeConfig:
    """Uma única alternativa, preservando batch efetivo e comprimento máximo."""
    fallback = replace(
        config,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=config.effective_batch_size,
        gradient_checkpointing=True,
    )
    fallback.validate()
    return fallback


def stratified_sample(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Amostra sem reposição; ordenação por hash torna a escolha estável."""
    validate_training_frame(frame)
    if isinstance(size, bool) or not isinstance(size, int) or not 3 <= size <= len(frame):
        raise ValueError("Tamanho de amostra deve estar entre 3 e o tamanho do split.")
    ordered = frame.sort_values("sample_id").reset_index(drop=True)
    if size == len(ordered):
        return ordered.copy()
    sampled, _ = train_test_split(
        ordered, train_size=size, random_state=seed, stratify=ordered["label"]
    )
    sampled = sampled.sort_values("sample_id").reset_index(drop=True)
    validate_training_frame(sampled)
    return sampled


def make_smoke_samples(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: SmokeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Cada amostra nasce exclusivamente de seu split; não cria novos splits."""
    config.validate()
    validate_training_pair(train, val)
    samples = {
        "train": stratified_sample(train, config.train_samples, config.seed),
        "val": stratified_sample(val, config.val_samples, config.seed),
    }
    validate_training_pair(samples["train"], samples["val"])
    summary = {
        name: {
            "rows": len(part),
            "fingerprint": dataset_fingerprint(part),
            "class_counts": {
                label_name: int((part["label"] == label).sum())
                for label, label_name in LABEL_NAMES.items()
            },
        }
        for name, part in samples.items()
    }
    return samples["train"], samples["val"], summary


def tokenize_reviews(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    """Tokeniza textos sem lowercasing/limpeza adicional e sem padding antecipado.

    A escolha cased é validada no carregamento do tokenizer. O collator fará o
    padding de cada lote, enquanto truncamento é aplicado aqui. IDs não entram no modelo.
    """
    encoded = tokenizer(
        frame["review_text"].tolist(),
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return [
        {**{key: values[index] for key, values in encoded.items()}, "labels": int(label)}
        for index, label in enumerate(frame["label"])
    ]
