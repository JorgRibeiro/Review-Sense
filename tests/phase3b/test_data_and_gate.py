"""Dados, truncamento e seleção computacional, sem GPU ou rede."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from reviewsense.phase3b_data import (
    FullConfig,
    choose_gpu_configuration,
    length_statistics,
    representative_indices,
    truncate_records,
    variant_config,
)
from reviewsense.train_bertimbau import checkpoint_files_valid, run_training


def test_untruncated_length_statistics_and_strict_thresholds():
    result = length_statistics([10, 128, 129, 256, 257])
    assert result["percentiles"]["P50"] == 129
    assert result["percentiles"]["P90"] == pytest.approx(256.6)
    assert result["exceeds"]["128"] == {"count": 3, "percent": 60.0}
    assert result["exceeds"]["256"] == {"count": 1, "percent": 20.0}
    assert result["truncation"] is False


def test_representative_sampling_covers_lengths_and_is_deterministic():
    lengths = np.array([10] * 400 + [40] * 200 + [100] * 100 + [200] * 30 + [500] * 20)
    indices = representative_indices(lengths, 256, 42)
    assert indices == representative_indices(lengths, 256, 42)
    assert len(set(indices)) == 256
    assert set(lengths[indices]) == {10, 40, 100, 200, 500}


def test_cache_truncation_preserves_sep_and_alignment_without_mutating():
    records = [
        {
            "input_ids": [2, 5, 6, 7, 8, 3],
            "attention_mask": [1] * 6,
            "token_type_ids": [0] * 6,
            "labels": 2,
        }
    ]
    result = truncate_records(records, 4, 3)
    assert result[0]["input_ids"] == [2, 5, 6, 3]
    assert len(result[0]["attention_mask"]) == len(result[0]["token_type_ids"]) == 4
    assert result[0]["labels"] == 2
    assert len(records[0]["input_ids"]) == 6


def result_for(batch, free=2**30, stable=True, checkpointing=False):
    return {
        "config": asdict(variant_config(FullConfig(), 256, batch, checkpointing)),
        "variant_id": f"b{batch}",
        "status": "completed",
        "numerically_stable": stable,
        "driver": {"min_free_bytes": free, "samples": 20, "errors": []},
    }


def test_compute_gate_prefers_requested_config_only_with_margin_and_stability():
    results = [result_for(2), result_for(4)]
    assert choose_gpu_configuration(results, FullConfig())["config"]["train_batch_size"] == 4
    results[1]["driver"]["min_free_bytes"] -= 1
    assert choose_gpu_configuration(results, FullConfig())["config"]["train_batch_size"] == 2
    results[0]["numerically_stable"] = False
    with pytest.raises(RuntimeError):
        choose_gpu_configuration(results, FullConfig())
    results.append(result_for(1, checkpointing=True))
    selected = choose_gpu_configuration(results, FullConfig())["config"]
    assert selected["gradient_checkpointing"] and selected["gradient_accumulation_steps"] == 16
    assert selected["eval_batch_size"] == 1


def test_gate_rejects_driver_monitor_failure():
    result = result_for(4)
    result["driver"]["errors"] = ["NVMLError"]
    with pytest.raises(RuntimeError):
        choose_gpu_configuration([result], FullConfig())


@pytest.mark.parametrize(
    "change",
    [
        {"learning_rate": 1e-4},
        {"max_epochs": 4},
        {"gradient_accumulation_steps": 8},
        {"max_length": 512},
        {"min_driver_free_bytes": 100},
        {"pilot_repeats": 10},
    ],
)
def test_no_quality_tuning_or_excessive_budget(change):
    with pytest.raises(ValueError):
        replace(FullConfig(), **change).validate()


def test_full_training_requires_pilot_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match="gate"):
        run_training(tmp_path / "nonexistent", FullConfig(), tmp_path / "run")


def test_incomplete_resume_state_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Checkpoint incompleto"):
        checkpoint_files_valid(tmp_path)
