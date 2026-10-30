"""Configuração, classes e amostras sem rede, CUDA ou pesos externos."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from reviewsense.data import prepare_dataset, split_dataset
from reviewsense.transformer_data import (
    SmokeConfig,
    label_mappings,
    make_smoke_samples,
    oom_fallback,
    stratified_sample,
    tokenize_reviews,
)


def test_default_config_and_label_maps():
    config = SmokeConfig(**json.loads(Path("configs/bertimbau_smoke.json").read_text()))
    config.validate()
    assert config.train_samples == 256 and config.val_samples == 96
    assert config.max_length == 128 and config.effective_batch_size == 16
    assert config.learning_rate == 2e-5 and config.epochs == 1
    assert config.mixed_precision == "fp16"
    assert label_mappings() == (
        {0: "negativo", 1: "neutro", 2: "positivo"},
        {"negativo": 0, "neutro": 1, "positivo": 2},
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"train_samples": 88684},
        {"val_samples": 19004},
        {"train_samples": 2},
        {"epochs": 2},
        {"max_length": 256},
        {"max_length": 127},
        {"train_batch_size": 4},
        {"eval_batch_size": 8},
        {"seed": True},
        {"revision": "main"},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"mixed_precision": "bf16"},
        {"gradient_checkpointing": "yes"},
        {"checkpoint": "other/model"},
    ],
)
def test_smoke_rejects_full_training_and_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(SmokeConfig(), **changes).validate()


def test_oom_fallback_preserves_effective_batch():
    default = SmokeConfig()
    alternative = oom_fallback(default)
    assert alternative.train_batch_size == alternative.eval_batch_size == 1
    assert alternative.gradient_accumulation_steps == 16
    assert alternative.effective_batch_size == default.effective_batch_size
    assert alternative.gradient_checkpointing
    assert alternative.max_length == 128 and alternative.learning_rate == 2e-5
    assert default.train_batch_size == 2


def test_samples_are_deterministic_disjoint_and_from_the_correct_source(reviews):
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    original_train, original_val = splits["train"].copy(), splits["val"].copy()
    config = replace(SmokeConfig(), train_samples=60, val_samples=30)
    train, val, summary = make_smoke_samples(splits["train"], splits["val"], config)
    again_train, again_val, again_summary = make_smoke_samples(
        splits["train"].sample(frac=1, random_state=6),
        splits["val"].sample(frac=1, random_state=7),
        config,
    )
    pd.testing.assert_frame_equal(train, again_train)
    pd.testing.assert_frame_equal(val, again_val)
    pd.testing.assert_frame_equal(splits["train"], original_train)
    pd.testing.assert_frame_equal(splits["val"], original_val)
    assert summary == again_summary
    assert set(train.sample_id) <= set(splits["train"].sample_id)
    assert set(val.sample_id) <= set(splits["val"].sample_id)
    assert set(train.sample_id).isdisjoint(val.sample_id)
    assert train.label.value_counts().tolist() == [20, 20, 20]
    assert val.label.value_counts().tolist() == [10, 10, 10]
    assert "sample_ids" not in json.dumps(summary)
    different = stratified_sample(splits["train"], 60, seed=11)
    assert set(train.sample_id) != set(different.sample_id)


def test_sampling_rejects_excess_size_and_overlap(reviews):
    prepared, _ = prepare_dataset(reviews)
    splits = split_dataset(prepared)
    with pytest.raises(ValueError):
        stratified_sample(splits["val"], 96, seed=42)
    config = replace(SmokeConfig(), train_samples=30, val_samples=15)
    with pytest.raises(ValueError, match="Vazamento"):
        make_smoke_samples(splits["train"], splits["train"], config)


def test_tokenization_forwards_original_cased_accented_texts():
    frame = pd.DataFrame(
        {"review_text": ["NÃO gostei! Péssimo 😞", "Ótimo Produto!"], "label": [0, 2]}
    )
    observed = {}

    def fake_tokenizer(texts, **kwargs):
        observed["texts"] = texts
        observed["kwargs"] = kwargs
        return {"input_ids": [[1, 2, 3], [1, 4]], "attention_mask": [[1, 1, 1], [1, 1]]}

    records = tokenize_reviews(frame, fake_tokenizer, 128)
    assert observed["texts"] == frame.review_text.tolist()
    assert observed["kwargs"] == {
        "truncation": True,
        "max_length": 128,
        "padding": False,
        "return_attention_mask": True,
    }
    assert [row["labels"] for row in records] == [0, 2]
    assert [len(row["input_ids"]) for row in records] == [3, 2]
    assert not any("sample_id" in row or "review_text" in row for row in records)
