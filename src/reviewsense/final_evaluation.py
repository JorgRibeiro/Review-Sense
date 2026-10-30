"""Avaliação definitiva: preflight bloqueia teste; execução explícita e única."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from reviewsense.baseline import load_experiment
from reviewsense.data import dataset_fingerprint, file_sha256
from reviewsense.evaluate import classification_metrics
from reviewsense.frozen_data import load_training_data, validate_training_frame
from reviewsense.labels import LABEL_NAMES


def now() -> str:
    return datetime.now(ZoneInfo("America/Recife")).isoformat()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    """Publica arquivo completo; hard link exclusivo impede sobrescrita concorrente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".phase3c-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def test_access_policy(path: Path, *, final: bool = False):
    """Audit hook: bloqueia escrita e permite no máximo uma abertura de leitura."""
    target = path.resolve()
    state = {"active": True, "read_opens": 0, "blocked_opens": 0}

    def guard(event, args):
        if not state["active"] or event != "open":
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        if Path(os.fsdecode(args[0])).resolve() != target:
            return
        flags = args[2]
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if not final or state["read_opens"] or flags & write_flags:
            state["blocked_opens"] += 1
            raise PermissionError("Acesso ao teste bloqueado pelo protocolo da Fase 3C.")
        state["read_opens"] += 1

    sys.addaudithook(guard)
    try:
        yield state
    finally:
        state["active"] = False


# Evita que pytest descubra a função importada como um teste.
test_access_policy.__test__ = False


def git_state() -> dict:
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "clean": not subprocess.check_output(["git", "status", "--porcelain"], text=True),
    }


def versions() -> dict:
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "scikit-learn", "numpy", "pandas", "accelerate")
        },
    }


def read_config(path: Path) -> dict:
    config = read_json(path)
    required = {
        "schema_version": 1,
        "max_length": 256,
        "bert_batch_size": 2,
        "svm_batch_size": 256,
        "selected_bert_epoch": 1,
        "selected_bert_step": 5543,
        "bootstrap": None,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Configuração diverge do protocolo congelado.")
    for key in ("torch_threads", "svm_threads", "benchmark_samples", "benchmark_repeats"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError("Configuração de medição inválida.")
    return config


def artifact_inventory(config: dict, config_path: Path) -> dict[str, str]:
    """Hash somente da lista permitida; nunca inclui conteúdo do CSV de teste."""
    bert = Path(config["bert_dir"])
    checkpoint = Path(config["bert_checkpoint"])
    svm = Path(config["svm_dir"])
    paths = [
        config_path,
        Path(config["protocol"]),
        Path(config["reference"]),
        Path(config["svm_reference"]),
        Path(config["bert_reference"]),
        Path(config["data_dir"]) / "manifest.json",
        *(Path(config["data_dir"]) / f"{name}.csv" for name in ("train", "val")),
        *(svm / name for name in ("selected_model.joblib", "results.json", "config.json")),
        bert / "model.safetensors",
        checkpoint / "model.safetensors",
        *(bert / name for name in config["bert_file_sha256"]),
        *(checkpoint / name for name in config["bert_file_sha256"]),
        *Path("src/reviewsense").glob("*.py"),
    ]
    inventory = {path.as_posix(): file_sha256(path) for path in paths}
    for path, expected in {
        svm / "selected_model.joblib": config["svm_sha256"],
        bert / "model.safetensors": config["bert_sha256"],
        checkpoint / "model.safetensors": config["bert_sha256"],
        **{bert / name: sha for name, sha in config["bert_file_sha256"].items()},
        **{checkpoint / name: sha for name, sha in config["bert_file_sha256"].items()},
    }.items():
        if inventory[path.as_posix()] != expected:
            raise ValueError("Checksum de artefato selecionado divergente.")
    return inventory


def validate_manifest_metadata(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Lê apenas treino/validação; verifica os metadados do teste sem abri-lo."""
    directory = Path(config["data_dir"])
    if not (directory / "test.csv").is_file():
        raise FileNotFoundError("Split de teste ausente.")
    train, val, provenance = load_training_data(directory, config["reference"])
    manifest = read_json(directory / "manifest.json")
    reference = read_json(config["reference"])
    if manifest.get("label_names") != {str(k): v for k, v in LABEL_NAMES.items()}:
        raise ValueError("Mapeamento do manifest divergente.")
    if set(manifest["splits"]) != {"train", "val", "test"}:
        raise ValueError("Splits inválidos no manifest.")
    declared = manifest["splits"]["test"]
    expected = reference["splits"]["test"]
    if {k: declared[k] for k in expected} != expected:
        raise ValueError("Metadados do teste divergentes da referência congelada.")
    ids = declared["sample_ids"]
    if len(ids) != expected["rows"] or len(set(ids)) != len(ids):
        raise ValueError("IDs do manifest inválidos.")
    if set(ids) & (set(train.sample_id) | set(val.sample_id)):
        raise ValueError("Sobreposição de IDs declarados entre splits.")
    return train, val, provenance


def validate_final_frame(
    frame: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, manifest: dict, reference: dict
) -> dict:
    """Validação final não corrige dados: qualquer divergência é fatal."""
    validate_training_frame(frame)
    declared, expected = manifest["splits"]["test"], reference["splits"]["test"]
    counts = frame.label.value_counts()
    summary = {
        "rows": len(frame),
        "fingerprint": dataset_fingerprint(frame),
        "class_distribution": {
            name: {"count": int(counts[label]), "fraction": float(counts[label] / len(frame))}
            for label, name in LABEL_NAMES.items()
        },
    }
    if any(summary != {key: other[key] for key in summary} for other in (declared, expected)):
        raise ValueError("Split final diverge do fingerprint/distribuição congelados.")
    if frame.sample_id.tolist() != declared["sample_ids"]:
        raise ValueError("Ordem/identidade do split final divergente.")
    if set(frame.sample_id) & (set(train.sample_id) | set(val.sample_id)):
        raise ValueError("Vazamento de IDs no split final.")
    combined = dataset_fingerprint(pd.concat([train, val, frame], ignore_index=True))
    if (
        combined != manifest["prepared_fingerprint"]
        or combined != reference["prepared_fingerprint"]
    ):
        raise ValueError("Fingerprint global divergente.")
    return summary


def validate_labels(labels, count: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.shape != (count,) or values.dtype.kind not in "iu":
        raise ValueError("Predições com tamanho/tipo inválido.")
    if not np.isin(values, list(LABEL_NAMES)).all():
        raise ValueError("Predição fora de [0,1,2].")
    return values


def validate_logits(logits, count: int) -> np.ndarray:
    values = np.asarray(logits)
    if values.shape != (count, 3) or not np.isfinite(values).all():
        raise ValueError("Logits não finitos ou shape inválido.")
    return values


@dataclass(frozen=True)
class Predictions:
    """Identidade e ordem ficam em memória/local, nunca no relatório público."""

    ids: tuple[str, ...]
    labels: np.ndarray

    def aligned(self, frame: pd.DataFrame) -> np.ndarray:
        if self.ids != tuple(frame.sample_id):
            raise ValueError("Identidade/ordem de predições divergente.")
        return validate_labels(self.labels, len(frame))


def paired_agreement(truth, svm, bert) -> dict:
    true = validate_labels(truth, len(truth))
    a, b = validate_labels(svm, len(true)) == true, validate_labels(bert, len(true)) == true
    return {
        "both_correct": int((a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "only_svm_correct": int((a & ~b).sum()),
        "only_bertimbau_correct": int((~a & b).sum()),
        "samples": len(true),
    }


def validate_mappings(model, tokenizer) -> None:
    if model.config.id2label != LABEL_NAMES or model.config.label2id != {
        value: key for key, value in LABEL_NAMES.items()
    }:
        raise ValueError("Mapeamentos de classe do Transformer inválidos.")
    if model.config.num_labels != 3 or tokenizer.do_lower_case:
        raise ValueError("Esperado checkpoint cased de três classes.")


def load_local_bert(directory: Path, device: str):
    """Carregamento exclusivamente local, reutilizado no teste com BERT minúsculo."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, attn_implementation="eager"
    )
    validate_mappings(model, tokenizer)
    return model.to(device).eval(), tokenizer


def load_models(config: dict):
    import torch

    from reviewsense.train_transformer import diagnose_cuda

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["torch_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    diagnostic = diagnose_cuda()
    if not diagnostic["fp16_operation_supported"]:
        raise RuntimeError("FP16 indisponível.")
    svm, previous = load_experiment(config["svm_dir"])
    public = read_json(config["svm_reference"])
    bert_reference = read_json(config["bert_reference"])
    if previous != public or previous["selected"]["refit_on_train_plus_val"]:
        raise ValueError("Registro do SVM diverge da seleção da Fase 2.")
    params = svm.get_params()
    # Os nomes dos passos são conferidos no próprio pipeline salvo.
    tfidf, classifier = svm.steps[0][1], svm.steps[-1][1]
    if (
        classifier.C != 0.1
        or classifier.class_weight != "balanced"
        or tfidf.ngram_range != (1, 2)
        or tfidf.max_features != 80000
        or not np.array_equal(classifier.classes_, [0, 1, 2])
        or not params
        or previous["model_artifact"]["sha256"] != config["svm_sha256"]
        or bert_reference["saved_model_sha256"] != config["bert_sha256"]
        or not bert_reference["best_checkpoint"].endswith("checkpoint-5543")
    ):
        raise ValueError("Artefatos não correspondem aos modelos pré-selecionados.")
    model, tokenizer = load_local_bert(Path(config["bert_dir"]), "cuda:0")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Pesos não finitos no modelo salvo.")
    if torch.cuda.mem_get_info()[0] < config["min_free_gpu_bytes"]:
        raise RuntimeError("Memória GPU livre insuficiente; não alterar batch automaticamente.")
    return svm, model, tokenizer, diagnostic


def predict_svm(model, frame: pd.DataFrame, config: dict) -> Predictions:
    with threadpool_limits(limits=config["svm_threads"]):
        predictions = np.concatenate(
            [
                model.predict(frame.review_text.iloc[start : start + config["svm_batch_size"]])
                for start in range(0, len(frame), config["svm_batch_size"])
            ]
        )
    return Predictions(tuple(frame.sample_id), validate_labels(predictions, len(frame)))


def predict_bert(model, tokenizer, frame: pd.DataFrame, config: dict) -> Predictions:
    import torch

    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"),
    ):
        for start in range(0, len(frame), config["bert_batch_size"]):
            texts = frame.review_text.iloc[start : start + config["bert_batch_size"]].tolist()
            tokens = tokenizer(
                texts,
                truncation=True,
                max_length=config["max_length"],
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            ).to(device)
            logits = model(**tokens).logits.float().cpu().numpy()
            predictions.extend(validate_logits(logits, len(texts)).argmax(axis=1).tolist())
    return Predictions(tuple(frame.sample_id), validate_labels(predictions, len(frame)))


def rss_bytes() -> int:
    """RSS atual Linux, sem nova dependência; não é memória exclusiva por modelo."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def timed_prediction(predict, frame: pd.DataFrame, *, gpu: bool) -> tuple[Predictions, dict]:
    import torch

    from reviewsense.gpu_benchmark import DriverMonitor
    from reviewsense.train_transformer import gpu_memory

    stop = threading.Event()
    rss = [rss_bytes()]

    def sample():
        while not stop.wait(0.01):
            rss.append(rss_bytes())

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    monitor = DriverMonitor() if gpu else None
    try:
        if monitor:
            monitor.__enter__()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        predictions = predict(frame)
        if gpu:
            torch.cuda.synchronize()
        seconds = perf_counter() - start
    finally:
        stop.set()
        thread.join()
        if monitor:
            monitor.__exit__(None, None, None)
    if monitor and monitor.errors:
        raise RuntimeError("Falha no monitoramento de memória GPU.")
    return predictions, {
        "samples": len(frame),
        "seconds": seconds,
        "texts_per_second": len(frame) / seconds,
        "milliseconds_per_text_amortized": seconds * 1000 / len(frame),
        "process_rss_peak_sampled_bytes": max(*rss, rss_bytes()),
        "rss_sampling_interval_seconds": 0.01,
        "rss_scope": "whole process including both loaded models; not incremental model memory",
        "pytorch": gpu_memory() if gpu else None,
        "driver": monitor.summary() if monitor else None,
        "includes_preprocessing_transfers_prediction": True,
        "includes_model_loading_disk_io_metrics_warmup": False,
        "request_latency_measured": False,
    }


def summarize(truth, svm, bert, timings: dict) -> dict:
    output = {}
    for name, prediction in (("svm", svm), ("bertimbau", bert)):
        metrics = classification_metrics(truth, prediction)
        errors = int(np.sum(np.asarray(truth) != prediction))
        output[name] = {
            "metrics": metrics,
            "errors": errors,
            "error_fraction": errors / len(truth),
            "inference": timings[name],
        }
    a, b = output["svm"]["metrics"], output["bertimbau"]["metrics"]
    difference = b["f1_macro"] - a["f1_macro"]
    return {
        "models": output,
        "paired_agreement": paired_agreement(truth, svm, bert),
        "comparison": {
            "f1_macro_difference_absolute": difference,
            "f1_macro_difference_relative": difference / a["f1_macro"] if a["f1_macro"] else None,
        },
    }


def validate_summary(result: dict) -> None:
    """Confere agregados, inclusive no dry-run com relatórios sintéticos."""
    total = result["paired_agreement"]["samples"]
    if total < 1:
        raise ValueError("Total agregado inválido.")
    if sum(v for k, v in result["paired_agreement"].items() if k != "samples") != total:
        raise ValueError("Concordância pareada inconsistente.")
    for model in result["models"].values():
        metrics = model["metrics"]
        cm = np.asarray(metrics["confusion_matrix"])
        supports = [metrics["per_class"][name]["support"] for name in LABEL_NAMES.values()]
        if (
            metrics["label_order"] != [0, 1, 2]
            or metrics["samples"] != total
            or cm.shape != (3, 3)
            or cm.sum() != total
            or (cm < 0).any()
            or not np.array_equal(cm.sum(axis=1), supports)
            or model["errors"] != total - int(cm.trace())
            or not np.isclose(metrics["accuracy"], cm.trace() / total)
            or not np.isclose(model["error_fraction"], model["errors"] / total)
        ):
            raise ValueError("Métricas/suportes/matriz inconsistentes.")
        precision = np.divide(
            cm.diagonal(), cm.sum(axis=0), out=np.zeros(3), where=cm.sum(axis=0) != 0
        )
        recall = np.divide(
            cm.diagonal(), cm.sum(axis=1), out=np.zeros(3), where=cm.sum(axis=1) != 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(3),
            where=precision + recall != 0,
        )
        for key, values in (("precision", precision), ("recall", recall), ("f1-score", f1)):
            actual = [metrics["per_class"][name][key] for name in LABEL_NAMES.values()]
            macro_key = "f1_macro" if key == "f1-score" else f"{key}_macro"
            if not np.allclose(actual, values) or not np.isclose(metrics[macro_key], values.mean()):
                raise ValueError("Métricas por classe/macro inconsistentes com a matriz.")


def preflight(config_path: Path) -> dict:
    """Não abre teste: valida modelos/sintéticos e mede inferência na validação."""
    from sklearn.model_selection import StratifiedShuffleSplit

    config = read_config(config_path)
    if (Path(config["run_dir"]) / "started.json").exists():
        raise FileExistsError("Avaliação final já iniciada; permitido somente --render-saved.")
    with test_access_policy(Path(config["data_dir"]) / "test.csv") as access:
        inventory = artifact_inventory(config, config_path)
        actual_versions = versions()
        if any(actual_versions[k] != v for k, v in config["expected_versions"].items()):
            raise ValueError("Versões incompatíveis com os artefatos congelados.")
        _, val, provenance = validate_manifest_metadata(config)
        svm, model, tokenizer, diagnostic = load_models(config)
        diagnostic_frame = pd.DataFrame(
            {
                "sample_id": ["synthetic-0", "synthetic-1", "synthetic-2"],
                "review_text": [
                    "Não gostei da entrega.",
                    "É um produto regular.",
                    "Ótimo produto!",
                ],
                "label": [0, 1, 2],
            }
        )
        a = predict_svm(svm, diagnostic_frame, config).aligned(diagnostic_frame)
        b = predict_bert(model, tokenizer, diagnostic_frame, config).aligned(diagnostic_frame)
        synthetic = summarize(diagnostic_frame.label.to_numpy(), a, b, {"svm": {}, "bertimbau": {}})
        validate_summary(synthetic)
        from reviewsense.final_reporting import render_report

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "synthetic.json"
            atomic_json(target, synthetic)
            validate_summary(read_json(target))
            render_report(synthetic, Path(temporary), detailed=False)
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=config["benchmark_samples"], random_state=config["seed"]
        )
        indices, _ = next(splitter.split(val, val.label))
        benchmark_frame = val.iloc[np.sort(indices)]
        benchmark = {}
        for name, predict in {
            "svm": lambda frame: predict_svm(svm, frame, config),
            "bertimbau": lambda frame: predict_bert(model, tokenizer, frame, config),
        }.items():
            predict(benchmark_frame.iloc[: config["warmup_samples"]])
            runs, reference = [], None
            for _ in range(config["benchmark_repeats"]):
                predictions, timing = timed_prediction(predict, benchmark_frame, gpu=name != "svm")
                labels = predictions.aligned(benchmark_frame)
                if reference is not None and not np.array_equal(labels, reference):
                    raise ValueError("Predições variaram no benchmark pré-teste.")
                reference = labels
                runs.append(timing)
            median = float(np.median([run["seconds"] for run in runs]))
            benchmark[name] = {
                "runs": runs,
                "median_seconds": median,
                "texts_per_second": len(benchmark_frame) / median,
                "milliseconds_per_text_amortized": median * 1000 / len(benchmark_frame),
                "samples_per_repeat": len(benchmark_frame),
                "batch_size": config[f"{'bert' if name != 'svm' else 'svm'}_batch_size"],
            }
        if artifact_inventory(config, config_path) != inventory:
            raise ValueError("Artefatos alterados durante preflight.")
        result = {
            "status": "approved",
            "created_at": now(),
            "git": git_state(),
            "inventory": inventory,
            "software": actual_versions,
            "gpu": diagnostic,
            "data": provenance,
            "benchmark": benchmark,
            "test_access": {k: v for k, v in access.items() if k != "active"},
            "synthetic_inference_and_report_validated": True,
        }
        atomic_json(Path(config["run_dir"]) / "preflight.json", result, replace=True)
        return result


def begin_final(run_dir: Path, *, authorized: bool, metadata: dict) -> Path:
    """Marcador exclusivo sobrevive a falhas; exige autorização da CLI."""
    if not authorized:
        raise PermissionError("Exigido --execute-final para abrir o teste.")
    marker = run_dir / "started.json"
    atomic_json(marker, {"status": "started", "started_at": now(), **metadata})
    return marker


def execute_final(config_path: Path, *, authorized: bool = False) -> dict:
    """Uma leitura, uma passagem/modelo; nunca retenta após falha ou conclusão."""
    if not authorized:
        raise PermissionError("Exigido --execute-final.")
    config = read_config(config_path)
    run_dir, test_path = Path(config["run_dir"]), Path(config["data_dir"]) / "test.csv"
    if (run_dir / "started.json").exists() or (run_dir / "results.json").exists():
        raise FileExistsError("Execução definitiva já iniciada; use somente --render-saved.")
    with test_access_policy(test_path) as preparation_access:
        approved = read_json(run_dir / "preflight.json")
        state = git_state()
        if (
            approved["status"] != "approved"
            or not state["clean"]
            or approved["git"] != state
            or approved["inventory"] != artifact_inventory(config, config_path)
            or approved["software"] != versions()
            or approved["test_access"] != {"read_opens": 0, "blocked_opens": 0}
        ):
            raise ValueError(
                "Preflight/commit/árvore/configuração inválidos; teste continua fechado."
            )
        train, val, _ = validate_manifest_metadata(config)
        svm, model, tokenizer, diagnostic = load_models(config)
        manifest = read_json(Path(config["data_dir"]) / "manifest.json")
        reference = read_json(config["reference"])
        original_stat = test_path.stat()
        marker = begin_final(
            run_dir,
            authorized=authorized,
            metadata={"git": state, "preflight_sha256": file_sha256(run_dir / "preflight.json")},
        )
    try:
        with test_access_policy(test_path, final=True) as access:
            raw = test_path.read_bytes()
            test_sha = hashlib.sha256(raw).hexdigest()
            frame = pd.read_csv(
                io.BytesIO(raw),
                keep_default_na=False,
                dtype={"sample_id": str, "review_text": str, "label": "int64"},
            )
            summary = validate_final_frame(frame, train, val, manifest, reference)
            if len(frame) != 19004:
                raise ValueError("Número de exemplos diferente do protocolo final.")
            timings, predictions = {}, {}
            for name, predict in {
                "svm": lambda frame: predict_svm(svm, frame, config),
                "bertimbau": lambda frame: predict_bert(model, tokenizer, frame, config),
            }.items():
                batch, timings[name] = timed_prediction(predict, frame, gpu=name != "svm")
                predictions[name] = batch.aligned(frame)
            result = summarize(
                frame.label.to_numpy(), predictions["svm"], predictions["bertimbau"], timings
            )
            validate_summary(result)
            final_stat = test_path.stat()
            if artifact_inventory(config, config_path) != approved["inventory"] or (
                original_stat.st_ino,
                original_stat.st_size,
                original_stat.st_mtime_ns,
            ) != (final_stat.st_ino, final_stat.st_size, final_stat.st_mtime_ns):
                raise ValueError("Artefatos originais alterados durante execução.")
            result.update(
                {
                    "schema_version": 1,
                    "phase": "3C",
                    "status": "completed",
                    "started_at": read_json(marker)["started_at"],
                    "finished_at": now(),
                    "git": state,
                    "software": versions(),
                    "gpu": diagnostic,
                    "inventory": approved["inventory"],
                    "preflight": approved,
                    "test": {
                        **summary,
                        "csv_sha256": test_sha,
                        "read_opens": access["read_opens"],
                        "blocked_opens": access["blocked_opens"],
                        "preparation_read_opens": preparation_access["read_opens"],
                        "original_stat_unchanged": True,
                        "overlap_ids": 0,
                        "same_order_and_labels_for_both_models": True,
                        "single_inference_passes": {"svm": 1, "bertimbau": 1},
                        "bert_logits_finite_three_classes": True,
                    },
                    "config": config,
                    "historical_training": historical_training(config),
                    "principal_candidate_preselected": "bertimbau",
                }
            )
            atomic_json(run_dir / "results.json", result)
            atomic_json(
                marker,
                {**read_json(marker), "status": "completed", "finished_at": now()},
                replace=True,
            )
            return result
    except BaseException as error:
        # Nunca serializar mensagem/traceback que possa conter textos ou caminhos pessoais.
        atomic_json(
            run_dir / "failure.json",
            {
                "status": "failed_no_automatic_retry",
                "error_type": type(error).__name__,
                "at": now(),
                "test_read_opens": access["read_opens"] if "access" in locals() else 0,
            },
        )
        raise


def historical_training(config: dict) -> dict:
    svm, bert = read_json(config["svm_reference"]), read_json(config["bert_reference"])
    candidate = next(
        c for c in svm["candidates"] if c["candidate_id"] == svm["selected"]["candidate_id"]
    )
    return {
        "svm_fit_seconds": candidate["fit_seconds"],
        "svm_search_validation_seconds": svm["timing"]["search_and_validation_seconds"],
        "bert_three_epochs_validation_checkpoints_seconds": bert["train_seconds"],
        "cpu_model": svm["environment"]["cpu_model"],
        "logical_cpus": os.cpu_count(),
        "artifact_bytes": {
            "svm_joblib": (Path(config["svm_dir"]) / "selected_model.joblib").stat().st_size,
            "bert_weights": (Path(config["bert_dir"]) / "model.safetensors").stat().st_size,
            "bert_tokenizer": sum(
                (Path(config["bert_dir"]) / name).stat().st_size
                for name in ("tokenizer.json", "tokenizer_config.json")
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3c_final.json"))
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--preflight", action="store_true")
    operation.add_argument("--execute-final", action="store_true")
    operation.add_argument("--render-saved", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.config)
        print(json.dumps({"status": result["status"], "test_access": result["test_access"]}))
    elif args.execute_final:
        result = execute_final(args.config, authorized=args.execute_final)
        print(json.dumps({"status": result["status"], "comparison": result["comparison"]}))
    else:
        from reviewsense.final_reporting import publish_saved

        publish_saved(read_config(args.config))


if __name__ == "__main__":
    main()
