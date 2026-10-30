"""Busca pequena TF-IDF + LinearSVC em CPU, com seleção pela validação."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import tempfile
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_info, threadpool_limits

from reviewsense.data import file_sha256
from reviewsense.evaluate import benchmark_inference, classification_metrics
from reviewsense.frozen_data import load_training_data

MAX_CANDIDATES = 12


def candidate_parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expande até 12 combinações, na ordem usada para desempate determinístico."""
    search = config["search"]
    combinations = list(product(search["ngram_range"], search["C"], search["class_weight"]))
    if not 1 <= len(combinations) <= MAX_CANDIDATES:
        raise ValueError("A busca deve conter entre 1 e 12 configurações.")
    parameters = []
    for ngrams, c_value, weight in combinations:
        if list(ngrams) not in ([1, 1], [1, 2]):
            raise ValueError("Usar unigramas [1,1] ou unigramas/bigramas [1,2].")
        if isinstance(c_value, bool) or not math.isfinite(c_value) or c_value <= 0:
            raise ValueError("C deve ser finito e positivo.")
        if weight not in (None, "balanced"):
            raise ValueError("class_weight deve ser null ou balanced.")
        parameters.append(
            {"C": float(c_value), "ngram_range": list(ngrams), "class_weight": weight}
        )
    if len({json.dumps(params, sort_keys=True) for params in parameters}) != len(parameters):
        raise ValueError("A busca contém configurações repetidas.")
    for key in ("seed", "thread_limit"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} deve ser inteiro.")
    if not 0 <= config["seed"] < 2**32 or config["thread_limit"] < 1:
        raise ValueError("seed inválida ou thread_limit não positivo.")
    inference = config["inference"]
    for key, minimum in (("batch_size", 1), ("repeats", 1), ("warmup_batches", 0)):
        value = inference[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Parâmetro de inferência inválido: {key}.")
    return parameters


def build_svm_pipeline(config: dict[str, Any], params: dict[str, Any]) -> Pipeline:
    """Cria pipeline não ajustado; fit recebe exclusivamente textos do treino."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**config["tfidf"], ngram_range=tuple(params["ngram_range"]))),
            (
                "svm",
                LinearSVC(
                    **config["svm"],
                    C=params["C"],
                    class_weight=params["class_weight"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def runtime_environment() -> dict[str, Any]:
    """Registra hardware, software e limites ativos, sem hostname ou dados pessoais."""
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu or "unknown",
        "logical_cpus": os.cpu_count(),
        "available_cpu_affinity": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "device": "cpu",
        "parallel_candidates": 1,
        "threadpools": [
            {
                key: info.get(key)
                for key in ("user_api", "internal_api", "num_threads", "version", "architecture")
            }
            for info in threadpool_info()
        ],
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "reviewsense-ptbr",
                "scikit-learn",
                "numpy",
                "scipy",
                "pandas",
                "joblib",
                "threadpoolctl",
            )
        },
    }


def code_provenance() -> dict[str, Any]:
    """Identifica os módulos usados, inclusive quando a árvore tiver mudanças locais."""
    directory = Path(__file__).parent
    hashes = {
        name: file_sha256(directory / name)
        for name in ("baseline.py", "frozen_data.py", "evaluate.py", "data.py", "labels.py")
    }
    try:
        root = directory.parent.parent
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"git_commit": commit, "working_tree_dirty": dirty, "module_sha256": hashes}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_experiment(
    artifacts_dir: str | Path,
    reference_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Verifica dados, treina Dummy e até 12 SVMs, seleciona e salva o vencedor.

    Cada pipeline é ajustado somente no treino. A validação serve à comparação;
    o modelo selecionado NÃO é reajustado em treino+validação. Não carrega teste.
    Em empate de F1 macro, vence a primeira configuração; Dummy vence empate global.
    Candidatos com aviso de não convergência são registrados, mas não selecionados.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("A execução já existe; escolha outro diretório de saída.")
    params_grid = candidate_parameters(config)
    started_at = datetime.now(ZoneInfo("America/Recife")).isoformat()
    experiment_start = perf_counter()
    train, val, provenance = load_training_data(artifacts_dir, reference_path)
    train_texts, val_texts = train["review_text"].tolist(), val["review_text"].tolist()
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    candidates: list[dict[str, Any]] = []
    best_svm = None
    best_pipeline = None
    search_start = perf_counter()
    with threadpool_limits(limits=config["thread_limit"]):
        environment = runtime_environment()
        dummy = DummyClassifier(strategy="most_frequent", random_state=config["seed"])
        start = perf_counter()
        dummy.fit(train_texts, y_train)
        dummy_fit_seconds = perf_counter() - start
        dummy_predictions, dummy_inference = benchmark_inference(
            dummy, val_texts, **config["inference"]
        )
        dummy_record = {
            "name": "dummy",
            "strategy": "most_frequent",
            "fit_seconds": dummy_fit_seconds,
            "metrics": classification_metrics(y_val, dummy_predictions),
            "inference": dummy_inference,
        }
        for index, params in enumerate(params_grid, start=1):
            pipeline = build_svm_pipeline(config, params)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                start = perf_counter()
                pipeline.fit(train_texts, y_train)
                fit_seconds = perf_counter() - start
            convergence_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in captured)
            start = perf_counter()
            predictions = pipeline.predict(val_texts)
            predict_seconds = perf_counter() - start
            record = {
                "candidate_id": index,
                "params": params,
                "fit_seconds": fit_seconds,
                "validation_predict_seconds_single_batch": predict_seconds,
                "metrics": classification_metrics(y_val, predictions),
                "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
                "n_iter": int(np.max(pipeline.named_steps["svm"].n_iter_)),
                "converged": convergence_warnings == 0,
                "warning_categories": sorted({w.category.__name__ for w in captured}),
            }
            candidates.append(record)
            if record["converged"] and (
                best_svm is None or record["metrics"]["f1_macro"] > best_svm["metrics"]["f1_macro"]
            ):
                best_svm, best_pipeline = record, pipeline
            if progress:
                print(
                    json.dumps(
                        {
                            "candidate": index,
                            "total": len(params_grid),
                            "params": params,
                            "fit_seconds": round(fit_seconds, 3),
                            "f1_macro_val": record["metrics"]["f1_macro"],
                            "converged": record["converged"],
                        }
                    ),
                    flush=True,
                )
        if best_svm is None:
            raise RuntimeError(
                "Nenhum SVM convergiu; resultados não publicados como modelo válido."
            )
        best_predictions, best_inference = benchmark_inference(
            best_pipeline, val_texts, **config["inference"]
        )
        best_svm["inference"] = best_inference
        if classification_metrics(y_val, best_predictions) != best_svm["metrics"]:
            raise RuntimeError("Métricas divergentes na nova passagem do modelo selecionado.")
        svm_wins = best_svm["metrics"]["f1_macro"] > dummy_record["metrics"]["f1_macro"]
        selected_model = best_pipeline if svm_wins else dummy
        expected_predictions = best_predictions if svm_wins else dummy_predictions
        selected = {
            "name": "svm" if svm_wins else "dummy",
            "candidate_id": best_svm["candidate_id"] if svm_wins else None,
            "metrics": best_svm["metrics"] if svm_wins else dummy_record["metrics"],
            "selection_metric": "validation_f1_macro",
            "tie_break": "first_grid_candidate; dummy_on_global_tie",
            "refit_on_train_plus_val": False,
        }
        search_seconds = perf_counter() - search_start
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            model_file = staging / "selected_model.joblib"
            joblib.dump(selected_model, model_file, compress=3)
            reloaded = joblib.load(model_file)
            if not np.array_equal(reloaded.predict(val_texts), expected_predictions):
                raise RuntimeError("As predições mudaram após salvar/carregar o modelo.")
            config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result = {
                "schema_version": 1,
                "phase": 2,
                "started_at": started_at,
                "finished_at": datetime.now(ZoneInfo("America/Recife")).isoformat(),
                "dataset_license": "CC-BY-NC-SA-4.0 (fonte original)",
                "data": provenance,
                "code": code_provenance(),
                "environment": environment,
                "config": config,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "dummy": dummy_record,
                "candidates": candidates,
                "best_svm_candidate_id": best_svm["candidate_id"],
                "selected": selected,
                "timing": {
                    "search_and_validation_seconds": search_seconds,
                    "experiment_seconds_before_report_write": perf_counter() - experiment_start,
                    "fit_includes": "TF-IDF fit_transform + LinearSVC fit; Dummy fit ignores text",
                    "inference_conditions": (
                        "sequential CPU; no GPU; warmed in-memory batches; wall clock"
                    ),
                    "thread_limit": config["thread_limit"],
                    "process_peak_rss_bytes": peak_rss
                    if platform.system() == "Darwin"
                    else peak_rss * 1024,
                    "peak_rss_scope": (
                        "whole process, including loading/search/serialization; not per model"
                    ),
                },
                "model_artifact": {
                    "file": model_file.name,
                    "sha256": file_sha256(model_file),
                    "bytes": model_file.stat().st_size,
                    "validation_roundtrip_verified": True,
                },
                "limitations": [
                    "Validation used for selection; not an unbiased final performance estimate.",
                    "Star-derived labels; commerce domain from 2018.",
                    "Near-duplicates/authors/products may overlap; exact duplicate texts removed.",
                    "Word TF-IDF ignores most punctuation/emojis and one-character tokens.",
                    "LinearSVC margins are not calibrated probabilities.",
                    "Times depend on host load/hardware; throughput is not request latency.",
                ],
            }
            _write_json(staging / "results.json", result)
            _write_json(staging / "config.json", config)
            staging.rename(destination)
    return result


def load_experiment(output_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Carrega apenas artefatos locais confiáveis e verifica checksum do modelo.

    Joblib usa pickle: não carregar modelos de origem desconhecida. O checksum
    detecta corrupção, mas não autentica a origem do arquivo.
    """
    directory = Path(output_dir)
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    path = directory / "selected_model.joblib"
    if file_sha256(path) != result["model_artifact"]["sha256"]:
        raise ValueError("Checksum do modelo salvo divergente.")
    return joblib.load(path), result


def main() -> None:
    """Executa a Fase 2 localmente, sem alterar os artefatos da Fase 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("data/processed/phase1-final"))
    parser.add_argument("--reference", type=Path, default=Path("reports/data_audit_summary.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/baseline/phase2-v1"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_experiment(args.artifacts_dir, args.reference, config, args.output_dir)
    print(
        json.dumps(
            {"selected": result["selected"], "output_dir": str(args.output_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
