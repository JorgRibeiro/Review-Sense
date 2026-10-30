"""Fine-tuning diagnóstico do BERTimbau em uma GPU local, limitado ao smoke."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from reviewsense.data import file_sha256
from reviewsense.evaluate import classification_metrics
from reviewsense.frozen_data import load_training_data
from reviewsense.transformer_data import (
    SmokeConfig,
    label_mappings,
    make_smoke_samples,
    oom_fallback,
    tokenize_reviews,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


@contextmanager
def protect_test_csv(artifacts_dir: str | Path):
    """Impede abertura do teste; não lê nem calcula checksum desse arquivo.

    O audit hook fica inativo ao sair do contexto. Não interfere nos arquivos
    sintéticos dos testes unitários nem em outros diretórios.
    """
    protected = os.path.realpath(Path(artifacts_dir) / "test.csv")
    state = {"active": True, "blocked_open_attempts": 0}

    def guard(event, args):
        if state["active"] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) == protected:
                state["blocked_open_attempts"] += 1
                raise PermissionError("Abertura do test.csv proibida na Fase 3A.")

    sys.addaudithook(guard)
    try:
        yield state
    finally:
        state["active"] = False


def query_nvidia() -> list[dict[str, str]]:
    """Snapshot agregado do dispositivo; não registra processos, nomes ou UUIDs."""
    fields = ["name", "driver_version", "memory.total", "memory.used", "utilization.gpu"]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [
            dict(zip(fields, (part.strip() for part in row), strict=True))
            for row in csv.reader(result.stdout.splitlines())
        ]
    except (OSError, subprocess.SubprocessError):
        return []


def diagnose_cuda() -> dict[str, Any]:
    """Exige CUDA e executa uma operação FP16 real; nunca migra o treino para CPU."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível. Este smoke deve executar na GPU, sem fallback CPU.")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        tensor = torch.ones((16, 16), device="cuda")
        product = tensor @ tensor
        fp16_supported = product.dtype == torch.float16 and bool(torch.isfinite(product).all())
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    diagnostic = {
        "device": "cuda:0",
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes_torch": properties.total_memory,
        "free_memory_bytes_at_diagnosis": free,
        "visible_memory_bytes": total,
        "fp16_operation_supported": fp16_supported,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": query_nvidia(),
    }
    del tensor, product
    torch.cuda.empty_cache()
    return diagnostic


def gpu_memory() -> dict[str, int]:
    """Medição do allocator PyTorch, separada da ocupação total da placa."""
    import torch

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_visible_total_bytes": total,
    }


def load_tokenizer(config: SmokeConfig, cache_dir: Path, local_files_only: bool):
    """Carrega AutoTokenizer cased na revisão fixa, sem código remoto."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.checkpoint,
        revision=config.revision,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        trust_remote_code=False,
        use_fast=True,
        do_lower_case=False,
        strip_accents=False,
    )
    if getattr(tokenizer, "do_lower_case", False):
        raise ValueError("Tokenizer incompatível: lowercasing ativo no checkpoint cased.")
    return tokenizer


def evaluate_on_gpu(model, loader, accelerator) -> tuple[np.ndarray, np.ndarray, float]:
    """Avalia somente o loader recebido; sincroniza a GPU ao medir tempo."""
    import torch

    model.eval()
    logits, labels = [], []
    torch.cuda.synchronize()
    start = perf_counter()
    with torch.inference_mode():
        for batch in loader:
            with accelerator.autocast():
                output = model(**batch)
            logits.append(output.logits.float().cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(logits), np.concatenate(labels), perf_counter() - start


def train_attempt(
    train_records: list[dict],
    val_records: list[dict],
    tokenizer,
    config: SmokeConfig,
    checkpoint_dir: Path,
    stage: dict[str, Any],
    validation_frame,
) -> dict[str, Any]:
    """Uma tentativa limitada: treino, avaliação, save_pretrained e recarga local.

    Accelerate gerencia FP16/GradScaler, transferência para CUDA e acumulação.
    AdamW não usa foreach para evitar temporários maiores. O loop é de uma época.
    """
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )

    set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None,
    )
    if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
        raise RuntimeError("O smoke exige exatamente um processo em uma GPU CUDA.")
    id2label, label2id = label_mappings()
    stage["name"] = "load_model"
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        revision=config.revision,
        cache_dir=stage["cache_dir"],
        local_files_only=stage["local_files_only"],
        trust_remote_code=False,
        use_safetensors=False,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    initial_head_hash = hashlib.sha256(
        model.classifier.weight.detach().numpy().tobytes()
    ).hexdigest()
    collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8)
    train_loader = DataLoader(
        train_records,
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_records,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8,
        foreach=False,
    )
    expected_updates = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, expected_updates)
    stage["name"] = "prepare_cuda"
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    observations = {"parameter_device": str(next(model.parameters()).device)}
    unwrapped = accelerator.unwrap_model(model)
    first_linear = next(
        module for module in unwrapped.modules() if isinstance(module, torch.nn.Linear)
    )

    def observe_linear(module, inputs, output):
        if "linear_output_dtype" not in observations:
            observations["linear_output_dtype"] = str(output.dtype)
            observations["linear_output_device"] = str(output.device)

    hook = first_linear.register_forward_hook(observe_linear)
    stage["name"] = "train"
    torch.cuda.reset_peak_memory_stats()
    memory_before_train = gpu_memory()
    torch.cuda.synchronize()
    start = perf_counter()
    loss_sum, seen, updates, skipped = 0.0, 0, 0, 0
    model.train()
    for index, batch in enumerate(train_loader, start=1):
        observations.setdefault("input_device", str(batch["input_ids"].device))
        with accelerator.accumulate(model):
            with accelerator.autocast():
                output = model(**batch)
                loss = output.loss
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError("Loss não finita no smoke; interrompendo sem tuning.")
            batch_rows = len(batch["labels"])
            loss_sum += loss_value * batch_rows
            seen += batch_rows
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                if accelerator.optimizer_step_was_skipped:
                    skipped += 1
                else:
                    updates += 1
                if "nvidia_smi_during_training" not in observations:
                    observations["nvidia_smi_during_training"] = query_nvidia()
        if index % 32 == 0 or index == len(train_loader):
            print(
                json.dumps(
                    {
                        "stage": "train",
                        "micro_batches": index,
                        "total_micro_batches": len(train_loader),
                        "optimizer_updates": updates,
                        "mean_loss": loss_sum / seen,
                    }
                ),
                flush=True,
            )
    torch.cuda.synchronize()
    train_seconds = perf_counter() - start
    train_memory = gpu_memory()
    hook.remove()
    del output, loss, batch
    unwrapped = accelerator.unwrap_model(model)
    final_head_hash = hashlib.sha256(
        unwrapped.classifier.weight.detach().float().cpu().numpy().tobytes()
    ).hexdigest()
    if seen != len(train_records) or updates == 0 or final_head_hash == initial_head_hash:
        raise RuntimeError("Não foi comprovada atualização dos pesos em toda a amostra de treino.")
    if (
        config.mixed_precision == "fp16"
        and observations.get("linear_output_dtype") != "torch.float16"
    ):
        raise RuntimeError("FP16 não observado na operação linear em CUDA.")
    stage["name"] = "evaluate"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    memory_before_eval = gpu_memory()
    logits, labels, eval_seconds = evaluate_on_gpu(model, val_loader, accelerator)
    eval_memory = gpu_memory()
    metrics = classification_metrics(labels, logits.argmax(axis=1))
    stage["name"] = "save"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    unwrapped.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    parameter_count = sum(parameter.numel() for parameter in unwrapped.parameters())
    del first_linear, unwrapped, model, optimizer, scheduler, train_loader, val_loader
    accelerator.free_memory()
    gc.collect()
    torch.cuda.empty_cache()
    stage["name"] = "reload"
    reload_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, local_files_only=True, do_lower_case=False, strip_accents=False
    )
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir, local_files_only=True, use_safetensors=True, attn_implementation="eager"
    )
    if reloaded.config.id2label != id2label or reloaded.config.label2id != label2id:
        raise RuntimeError("Mapeamentos de rótulos mudaram na recarga.")
    if getattr(reload_tokenizer, "do_lower_case", False):
        raise RuntimeError("Tokenizer recarregado perdeu a configuração cased.")
    reloaded = accelerator.prepare(reloaded)
    reload_records = tokenize_reviews(validation_frame, reload_tokenizer, config.max_length)
    if reload_records != val_records:
        raise RuntimeError("Tokenização da validação mudou após recarga.")
    reload_loader = accelerator.prepare(
        DataLoader(
            reload_records,
            batch_size=config.eval_batch_size,
            collate_fn=DataCollatorWithPadding(reload_tokenizer, pad_to_multiple_of=8),
            num_workers=0,
            pin_memory=True,
        )
    )
    torch.cuda.reset_peak_memory_stats()
    reloaded_logits, reloaded_labels, reload_seconds = evaluate_on_gpu(
        reloaded, reload_loader, accelerator
    )
    reload_memory = gpu_memory()
    same_predictions = np.array_equal(logits.argmax(axis=1), reloaded_logits.argmax(axis=1))
    if not same_predictions or not np.array_equal(labels, reloaded_labels):
        raise RuntimeError("Predições ou ordem da validação mudaram após recarga.")
    if not np.allclose(logits, reloaded_logits, rtol=1e-4, atol=1e-4):
        raise RuntimeError("Logits divergentes após recarga do modelo.")
    del reloaded, reload_loader
    accelerator.free_memory()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "parameter_count": parameter_count,
        "training": {
            "seconds": train_seconds,
            "mean_loss": loss_sum / seen,
            "samples_seen": seen,
            "micro_batches": math.ceil(seen / config.train_batch_size),
            "expected_optimizer_updates": expected_updates,
            "successful_optimizer_updates": updates,
            "skipped_optimizer_updates": skipped,
            "initial_classifier_sha256": initial_head_hash,
            "final_classifier_sha256": final_head_hash,
            "memory_before": memory_before_train,
            "memory": train_memory,
        },
        "evaluation": {
            "seconds": eval_seconds,
            "memory_before": memory_before_eval,
            "memory": eval_memory,
        },
        "reload": {
            "seconds_evaluation": reload_seconds,
            "same_predictions": same_predictions,
            "logits_allclose_rtol_atol_1e4": True,
            "max_absolute_logit_difference": float(np.abs(logits - reloaded_logits).max()),
            "label_mappings_verified": True,
            "tokenizer_outputs_verified": True,
            "memory": reload_memory,
        },
        "gpu_observations": observations,
    }


def execute_with_oom_retry(
    config: SmokeConfig, execute, on_oom
) -> tuple[Any, list[dict[str, Any]]]:
    """No máximo duas tentativas; só CUDA OOM autoriza a alternativa conservadora."""
    import torch

    attempts = []
    current = config
    for index in (1, 2):
        try:
            value = execute(current, index)
            attempts.append({"attempt": index, "status": "completed", "config": asdict(current)})
            return value, attempts
        except torch.cuda.OutOfMemoryError as error:
            attempts.append({"attempt": index, "status": "cuda_oom", "config": asdict(current)})
            on_oom(index, current, error, attempts)
            error.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            if index == 2:
                raise RuntimeError(
                    "CUDA OOM nas duas tentativas permitidas; smoke não concluído."
                ) from error
            current = oom_fallback(current)
    raise AssertionError("Fluxo de tentativas inválido.")


def run_smoke(
    artifacts_dir: str | Path,
    reference: str | Path,
    config: SmokeConfig,
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Executa somente o smoke, com dados protegidos e artefatos exclusivamente locais."""
    config.validate()
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("A saída já existe; escolha uma nova execução.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(ZoneInfo("America/Recife")).isoformat()
    start = perf_counter()
    diagnostic = diagnose_cuda()
    effective = config
    adjustments = []
    if config.mixed_precision == "fp16" and not diagnostic["fp16_operation_supported"]:
        effective = replace(config, mixed_precision="no")
        adjustments.append("FP16 indisponível no diagnóstico; precisão efetiva FP32.")
    with protect_test_csv(artifacts_dir) as guard:
        paths = {
            name: Path(artifacts_dir) / name for name in ("train.csv", "val.csv", "manifest.json")
        }
        before = {name: file_sha256(path) for name, path in paths.items()}
        train, val, provenance = load_training_data(artifacts_dir, reference)
        train_sample, val_sample, sample_summary = make_smoke_samples(train, val, effective)
        tokenizer = load_tokenizer(effective, cache, local_files_only)
        train_records = tokenize_reviews(train_sample, tokenizer, effective.max_length)
        val_records = tokenize_reviews(val_sample, tokenizer, effective.max_length)
        token_stats = {}
        for name, records in (("train", train_records), ("val", val_records)):
            lengths = np.array([len(row["input_ids"]) for row in records])
            token_stats[name] = {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": float(lengths.mean()),
                "at_max_length": int((lengths == effective.max_length).sum()),
                "at_max_length_fraction": float((lengths == effective.max_length).mean()),
            }
        del train, val
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            stage = {
                "name": "initial",
                "cache_dir": str(cache),
                "local_files_only": local_files_only,
            }

            def execute(attempt_config, index):
                print(json.dumps({"attempt": index, "config": asdict(attempt_config)}), flush=True)
                return train_attempt(
                    train_records,
                    val_records,
                    tokenizer,
                    attempt_config,
                    staging / f"attempt-{index}" / "checkpoint",
                    stage,
                    val_sample,
                )

            def on_oom(index, attempt_config, error, attempts):
                # Não persiste a mensagem completa, que pode conter caminhos locais.
                attempts[-1]["stage"] = stage["name"]
                print(
                    json.dumps({"attempt": index, "status": "cuda_oom", "stage": stage["name"]}),
                    flush=True,
                )
                failure = {"attempts": attempts, "diagnostic": diagnostic, "status": "cuda_oom"}
                write_json(destination.parent / f"{destination.name}-oom-diagnostic.json", failure)

            outcome, attempts = execute_with_oom_retry(effective, execute, on_oom)
            after = {name: file_sha256(path) for name, path in paths.items()}
            if before != after or guard["blocked_open_attempts"]:
                raise RuntimeError("Invariantes dos artefatos congelados violadas.")
            successful = attempts[-1]
            saved_config = successful["config"]
            checkpoint = staging / f"attempt-{successful['attempt']}" / "checkpoint"
            local_samples = {
                "train_sample_ids": train_sample["sample_id"].tolist(),
                "val_sample_ids": val_sample["sample_id"].tolist(),
            }
            write_json(staging / "samples_manifest.json", local_samples)
            write_json(staging / "config.json", saved_config)
            module_names = [
                "train_transformer.py",
                "transformer_data.py",
                "frozen_data.py",
                "evaluate.py",
                "data.py",
                "labels.py",
            ]
            modules = {name: file_sha256(Path(__file__).parent / name) for name in module_names}
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            result = {
                "schema_version": 1,
                "phase": "3A",
                "purpose": "technical_smoke_not_final_performance",
                "started_at": started_at,
                "finished_at": datetime.now(ZoneInfo("America/Recife")).isoformat(),
                "code": {
                    "git_commit": commit,
                    "module_sha256": modules,
                    "working_tree_dirty": bool(
                        subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
                    ),
                },
                "checkpoint_source": {
                    "repo_id": config.checkpoint,
                    "revision": config.revision,
                    "original_format": "pytorch_model.bin",
                    "trust_remote_code": False,
                },
                "requested_config": asdict(config),
                "effective_config": saved_config,
                "adjustments": adjustments,
                "attempts": attempts,
                "data": provenance,
                "smoke_samples": sample_summary,
                "token_lengths": token_stats,
                "software": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "accelerate",
                        "tokenizers",
                        "safetensors",
                        "huggingface-hub",
                        "numpy",
                        "scikit-learn",
                        "pandas",
                    )
                },
                "python": platform.python_version(),
                "platform": platform.platform(),
                "gpu_diagnostic": diagnostic,
                **outcome,
                "execution_checks": {
                    "test_csv_guard_active": True,
                    "test_csv_open_attempts": guard["blocked_open_attempts"],
                    "source_artifacts_unchanged": before == after,
                    "sha256_before_and_after": before,
                },
                "checkpoint_artifacts": {
                    "directory": str(checkpoint.relative_to(staging)),
                    "sha256": {
                        file.name: file_sha256(file)
                        for file in sorted(checkpoint.iterdir())
                        if file.is_file()
                    },
                },
                "total_seconds_before_report_write": perf_counter() - start,
                "measurement_conditions": {
                    "training": "CUDA synchronized; compute/optimizer/collator/transfers",
                    "evaluation": "CUDA synchronized; collator/transfers/forward/CPU logits",
                    "memory": "PyTorch allocator peaks reset per stage; not total board usage",
                    "evaluation_optimizer_retained": True,
                    "cuda_cache_cleared_before_evaluation": True,
                    "attention": "eager",
                    "tf32": False,
                    "deterministic_algorithms": True,
                    "optimizer": "AdamW foreach=False",
                    "scheduler": "linear decay; zero warmup",
                    "padding": "dynamic per batch; multiple of 8",
                    "dataloader_workers": 0,
                },
                "limitations": [
                    "Diagnostic smoke only; no comparison with the full-data baseline.",
                    "One epoch and 256 training examples; not a quality or throughput forecast.",
                    "CUDA allocator statistics exclude other processes and driver/context memory.",
                    "At-length-limit fraction is not an exact truncation rate.",
                ],
            }
            write_json(staging / "results.json", result)
            staging.rename(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("data/processed/phase1-final"))
    parser.add_argument("--reference", type=Path, default=Path("reports/data_audit_summary.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/bertimbau_smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/bertimbau/phase3a-v1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/huggingface/hub"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    if args.diagnose_only:
        print(json.dumps(diagnose_cuda(), ensure_ascii=False, indent=2))
        return
    config = SmokeConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    result = run_smoke(
        args.artifacts_dir,
        args.reference,
        config,
        args.output_dir,
        args.cache_dir,
        local_files_only=args.local_files_only,
    )
    print(
        json.dumps(
            {
                "metrics": result["metrics"],
                "training": result["training"],
                "evaluation": result["evaluation"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
