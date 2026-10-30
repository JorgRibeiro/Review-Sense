"""Comprimentos, cache local e critérios computacionais fixos da Fase 3B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from reviewsense.data import file_sha256
from reviewsense.frozen_data import load_training_data
from reviewsense.train_transformer import load_tokenizer, protect_test_csv, write_json
from reviewsense.transformer_data import SmokeConfig, tokenize_reviews


@dataclass(frozen=True)
class FullConfig:
    checkpoint: str = SmokeConfig().checkpoint
    revision: str = SmokeConfig().revision
    seed: int = 42
    initial_loss_scale: float = 16384.0
    loss_scale_growth_interval: int = 1000000
    max_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 256
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 2
    gradient_checkpointing: bool = False
    cpu_threads: int = 2
    checkpoint_every_steps: int = 500
    logging_steps: int = 100
    min_driver_free_bytes: int = 2**30
    monitor_interval_seconds: float = 0.05
    pilot_representative_samples: int = 256
    pilot_long_samples: int = 64
    pilot_repeats: int = 2
    pilot_warmup_samples: int = 32

    def validate(self):
        if self.checkpoint != SmokeConfig().checkpoint or self.revision != SmokeConfig().revision:
            raise ValueError("Manter checkpoint e revisão validados na Fase 3A.")
        if not 1 <= self.max_epochs <= 3 or self.learning_rate != 2e-5:
            raise ValueError("Até três épocas, learning rate fixo 2e-5; sem tuning nesta fase.")
        if self.initial_loss_scale != 16384.0 or self.loss_scale_growth_interval != 1000000:
            raise ValueError("Usar a configuração numérica FP16 validada no piloto.")
        if self.max_length not in (128, 256) or self.train_batch_size not in (1, 2, 4):
            raise ValueError("Configuração fora das alternativas computacionais previstas.")
        if self.train_batch_size * self.gradient_accumulation_steps != 16:
            raise ValueError("O batch efetivo deve permanecer 16.")
        if self.eval_batch_size not in (1, 2) or self.min_driver_free_bytes < 2**30:
            raise ValueError("Avaliação conservadora e margem mínima de 1 GiB.")
        for key in ("cpu_threads", "checkpoint_every_steps", "logging_steps", "pilot_repeats"):
            if isinstance(getattr(self, key), bool) or getattr(self, key) < 1:
                raise ValueError(f"Parâmetro inválido: {key}.")
        if self.pilot_repeats > 3 or not 0.01 <= self.monitor_interval_seconds <= 1:
            raise ValueError("Orçamento ou intervalo de monitoramento inválido.")
        for key in ("pilot_representative_samples", "pilot_long_samples", "pilot_warmup_samples"):
            if not 16 <= getattr(self, key) <= 512 or getattr(self, key) % 16:
                raise ValueError("Amostras do piloto devem ser múltiplos de 16, entre 16 e 512.")


def length_statistics(lengths: list[int] | np.ndarray) -> dict[str, Any]:
    """Mede antes do truncamento, incluindo CLS/SEP, com percentil linear."""
    values = np.asarray(lengths)
    if values.ndim != 1 or not len(values) or (values < 2).any():
        raise ValueError("Esperado vetor de comprimentos incluindo tokens especiais.")
    return {
        "samples": len(values),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "percentile_method": "linear",
        "percentiles": {f"P{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)},
        "exceeds": {
            str(limit): {
                "count": int((values > limit).sum()),
                "percent": float((values > limit).mean() * 100),
            }
            for limit in (128, 256)
        },
        "includes_special_tokens": True,
        "truncation": False,
    }


def representative_indices(lengths: np.ndarray, count: int, seed: int) -> list[int]:
    """Amostra estratos de comprimento; garante presença de todas as faixas existentes."""
    bins = np.digitize(lengths, [32, 64, 128, 256], right=True)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(bins == index) for index in np.unique(bins)]
    if count > len(lengths) or count < len(groups):
        raise ValueError("Tamanho incompatível com os estratos de comprimento.")
    selected = [int(rng.choice(group)) for group in groups]
    remainder = np.setdiff1d(np.arange(len(lengths)), selected)
    # Amostragem aleatória no restante mantém aproximadamente as proporções da origem.
    extra = rng.choice(remainder, count - len(selected), replace=False)
    selected.extend(int(index) for index in extra)
    rng.shuffle(selected)
    return selected


def prepare_phase3b(
    artifacts_dir: Path,
    reference: Path,
    config: FullConfig,
    output: Path,
    cache: Path,
) -> dict[str, Any]:
    """Analisa todos os textos permitidos e salva features individuais só no cache local."""
    config.validate()
    if output.exists():
        raise FileExistsError("Cache/análise já existe; use outra saída.")
    output.mkdir(parents=True)
    with protect_test_csv(artifacts_dir) as guard:
        checks = {
            name: file_sha256(artifacts_dir / name)
            for name in ("train.csv", "val.csv", "manifest.json")
        }
        train, val, provenance = load_training_data(artifacts_dir, reference)
        tokenizer = load_tokenizer(SmokeConfig(), cache, True)
        lengths = {}
        records = {}
        for name, frame in (("train", train), ("val", val)):
            sizes = []
            for start in range(0, len(frame), 512):
                encoded = tokenizer(
                    frame.iloc[start : start + 512]["review_text"].tolist(),
                    truncation=False,
                    padding=False,
                    verbose=False,
                )
                sizes.extend(len(ids) for ids in encoded["input_ids"])
            lengths[name] = np.asarray(sizes)
            records[name] = tokenize_reviews(frame, tokenizer, 256)
        natural = representative_indices(
            lengths["train"], config.pilot_representative_samples, config.seed
        )
        longest = np.argsort(-lengths["train"], kind="stable")[: config.pilot_long_samples].tolist()
        features = {
            "train": records["train"],
            "val": records["val"],
            "pilot_representative_indices": natural,
            "pilot_long_indices": longest,
            "lengths": lengths,
            "provenance": provenance,
        }
        joblib.dump(features, output / "features.joblib", compress=3)
        tokenizer.save_pretrained(output / "tokenizer")
        summary = {
            "config": asdict(config),
            "data": provenance,
            "lengths": {name: length_statistics(values) for name, values in lengths.items()},
            "pilot_samples": {
                "representative": length_statistics(lengths["train"][natural]),
                "long_stress": length_statistics(lengths["train"][longest]),
                "source": "train_only",
                "same_examples_all_variants": True,
            },
            "features_sha256": file_sha256(output / "features.joblib"),
            "execution_checks": {
                "test_csv_open_attempts": guard["blocked_open_attempts"],
                "source_artifacts_unchanged": all(
                    file_sha256(artifacts_dir / name) == value for name, value in checks.items()
                ),
            },
        }
        write_json(output / "analysis.json", summary)
    return summary


def load_features(directory: Path) -> tuple[dict, dict]:
    """Confere cache local confiável antes de carregar seus tokens/labels."""
    summary = json.loads((directory / "analysis.json").read_text())
    if file_sha256(directory / "features.joblib") != summary["features_sha256"]:
        raise ValueError("Cache de features divergente.")
    return joblib.load(directory / "features.joblib"), summary


def truncate_records(records: list[dict], length: int, sep_token_id: int) -> list[dict]:
    """Recorta cache de 256 para 128 preservando SEP e campos alinhados."""
    result = []
    for row in records:
        shortened = {
            key: (value[:length] if isinstance(value, list) else value)
            for key, value in row.items()
        }
        if len(row["input_ids"]) > length:
            shortened["input_ids"][-1] = sep_token_id
        result.append(shortened)
    return result


def variant_config(config: FullConfig, length: int, batch: int, checkpointing=False) -> FullConfig:
    alternative = replace(
        config,
        max_length=length,
        train_batch_size=batch,
        gradient_accumulation_steps=16 // batch,
        gradient_checkpointing=checkpointing,
        eval_batch_size=min(config.eval_batch_size, batch),
    )
    alternative.validate()
    return alternative


def choose_gpu_configuration(results: list[dict], config: FullConfig) -> dict:
    """Prioridade fixa, apenas estabilidade e memória; nenhuma métrica de qualidade."""
    for batch, checkpointing in ((4, False), (2, False), (1, True)):
        for result in results:
            variant = result["config"]
            if (
                variant["max_length"] == 256
                and variant["train_batch_size"] == batch
                and variant["gradient_checkpointing"] == checkpointing
                and result["status"] == "completed"
                and result["numerically_stable"]
                and result["driver"]["min_free_bytes"] >= config.min_driver_free_bytes
                and result["driver"].get("samples", 0) > 0
                and not result["driver"].get("errors")
            ):
                return {
                    "config": variant,
                    "reason": "stable_and_at_least_1GiB_driver_free",
                    "pilot_result_id": result["variant_id"],
                }
    raise RuntimeError("Nenhuma configuração de 256 tokens passou estabilidade e margem de VRAM.")
