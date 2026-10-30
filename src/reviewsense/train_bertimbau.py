"""Treino 3B com Trainer: seleção por validação e checkpoints completos retomáveis."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from reviewsense.data import file_sha256
from reviewsense.evaluate import classification_metrics
from reviewsense.gpu_benchmark import DriverMonitor
from reviewsense.phase3b_data import FullConfig, load_features
from reviewsense.train_transformer import diagnose_cuda, gpu_memory, protect_test_csv, write_json
from reviewsense.transformer_data import label_mappings


def checkpoint_files_valid(checkpoint: Path, *, fp16=True):
    required = {
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    }
    if fp16:
        required.add("scaler.pt")
    missing = [name for name in sorted(required) if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"Checkpoint incompleto para retomada: {missing}")
    return {name: (checkpoint / name).stat().st_size for name in sorted(required)}


def training_arguments(config: FullConfig, output: Path, *, full: bool, cpu=False):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.max_epochs,
        max_steps=-1 if full else 4,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type="linear",
        warmup_steps=0,
        fp16=not cpu,
        bf16=False,
        tf32=False,
        optim="adamw_torch",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config.seed,
        data_seed=config.seed,
        full_determinism=True,
        eval_strategy="epoch" if full else "no",
        save_strategy="epoch" if full else "steps",
        save_steps=2,
        save_total_limit=2,
        save_only_model=False,
        load_best_model_at_end=full,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        eval_accumulation_steps=32,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to=[],
        push_to_hub=False,
        disable_tqdm=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=not cpu,
        dataloader_drop_last=False,
        ignore_data_skip=False,
        use_cpu=cpu,
        remove_unused_columns=False,
        auto_find_batch_size=False,
        enable_jit_checkpoint=False,
    )


def normalized_classification_loss(outputs, labels, num_items_in_batch=None):
    """Normaliza pela quantidade real de exemplos do grupo acumulado (inclusive o último).

    BERT desta versão retorna CE média por microbatch e ignora num_items_in_batch.
    Fornecer a loss explicitamente evita multiplicar gradientes pela acumulação.
    """
    import torch.nn.functional as functional

    if num_items_in_batch is None:
        return functional.cross_entropy(outputs.logits.float(), labels, reduction="mean")
    return (
        functional.cross_entropy(outputs.logits.float(), labels, reduction="sum")
        / num_items_in_batch
    )


def create_stable_trainer(*args, **kwargs):
    """Verifica loss e norma em todos os passos, usando o clipping já feito pelo Trainer."""
    import torch
    from transformers import Trainer

    class StableTrainer(Trainer):
        nonfinite_losses = 0
        nonfinite_norms = 0
        pending_amp_overflow = False

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not bool(torch.isfinite(loss).all()):
                self.nonfinite_losses += 1
                raise FloatingPointError("Loss não finita; treino interrompido.")
            return loss

        def _clip_grad_norm(self, model):
            norm = super()._clip_grad_norm(model)
            if not math.isfinite(float(norm)):
                self.nonfinite_norms += 1
                if self.accelerator.scaler is None:
                    raise FloatingPointError(
                        "Norma não finita sem GradScaler para proteger o update."
                    )
                self.pending_amp_overflow = True
                # O scaler já detectou o overflow durante unscale_ e deve pular o update.
                # None evita registrar uma norma falsa ou serializar Infinity nos logs.
                return None
            return norm

    kwargs.setdefault("compute_loss_func", normalized_classification_loss)
    return StableTrainer(*args, **kwargs)


def verify_amp_recovery(trainer, optimizer, model, numeric: dict, step: int):
    """Confirma que overflows de FP16 pulam o update sem contaminar pesos."""
    import torch

    skipped = bool(getattr(optimizer, "step_was_skipped", False))
    pending = trainer.pending_amp_overflow
    if pending and not skipped:
        raise FloatingPointError("Overflow detectado sem proteção do GradScaler.")
    if skipped:
        numeric["skipped_updates"] += 1
        numeric["consecutive_skips"] += 1
        finite = torch.stack([torch.isfinite(p.detach()).all() for p in model.parameters()]).all()
        if not bool(finite):
            numeric["nonfinite_weights"] += 1
            raise FloatingPointError("Pesos não finitos após overflow; interrompendo.")
        numeric["recovered_amp_overflows"] += 1
    else:
        numeric["consecutive_skips"] = 0
    trainer.pending_amp_overflow = False
    if numeric["consecutive_skips"] >= 3 or (
        step >= 1000 and numeric["skipped_updates"] / step > 0.01
    ):
        raise FloatingPointError("Overflows persistentes ou excessivos; interrompendo.")


def run_training(
    features_dir: Path,
    config: FullConfig,
    output: Path,
    *,
    mode="full",
    resume: Path | None = None,
    gate: Path | None = None,
):
    """O treino completo exige aprovação do piloto e retomada reproduzida antes de iniciar."""
    config.validate()
    if output.exists() and resume is None:
        raise FileExistsError("Saída já existe; use outra ou informe --resume.")
    if mode == "full":
        if gate is None:
            raise ValueError("Treino completo exige o gate do piloto.")
        approval = json.loads(gate.read_text())
        if not approval.get("approved") or not approval.get("resume_verified"):
            raise ValueError("Piloto/retomada ainda não aprovados; não iniciar treino completo.")
        if approval.get("runtime_source_sha256") != file_sha256(Path(__file__)):
            raise ValueError("Código de treino difere do validado na retomada.")
        if approval["selected_config"] != asdict(config):
            raise ValueError("Configuração difere do piloto aprovado.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from accelerate.utils import set_seed
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        TrainerCallback,
    )

    diagnostic = diagnose_cuda()
    if (
        mode == "full"
        and diagnostic["free_memory_bytes_at_diagnosis"] < approval["required_free_at_start_bytes"]
    ):
        raise RuntimeError("Memória livre atual insuficiente para a configuração aprovada.")
    torch.set_num_threads(config.cpu_threads)
    set_seed(config.seed)
    features, analysis = load_features(features_dir)
    if mode == "full" and approval["features_sha256"] != analysis["features_sha256"]:
        raise ValueError("Cache difere do piloto aprovado.")
    if resume:
        checkpoint_files_valid(resume)
        saved = json.loads((resume / "phase3b_state.json").read_text())
        if (
            saved["config"] != asdict(config)
            or saved["features_sha256"] != analysis["features_sha256"]
        ):
            raise ValueError("Retomada exige exatamente configuração e dados da execução original.")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(features_dir / "tokenizer", local_files_only=True)
    id2label, label2id = label_mappings()
    model = AutoModelForSequenceClassification.from_pretrained(
        config.checkpoint,
        revision=config.revision,
        cache_dir="cache/huggingface/hub",
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=False,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    train_records = features["train"]
    if mode != "full":
        indices = (
            features["pilot_representative_indices"][:48] + features["pilot_long_indices"][:16]
        )
        train_records = [features["train"][index] for index in indices]
    pending = {}
    history_path = output / "validation_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    if resume:
        resumed_step = json.loads((resume / "trainer_state.json").read_text())["global_step"]
        history = [entry for entry in history if entry["global_step"] <= resumed_step]
    numeric = {
        "nonfinite_losses": 0,
        "nonfinite_gradient_norms": 0,
        "skipped_updates": 0,
        "consecutive_skips": 0,
        "nonfinite_weights": 0,
        "recovered_amp_overflows": 0,
    }
    started = perf_counter()

    def compute_metrics(prediction):
        logits = prediction.predictions
        if not np.isfinite(logits).all():
            raise FloatingPointError("Logits não finitos na validação.")
        details = classification_metrics(prediction.label_ids, logits.argmax(axis=1))
        pending["details"] = details
        return {
            key: details[key] for key in ("f1_macro", "precision_macro", "recall_macro", "accuracy")
        }

    class RecordState(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if mode == "full" and state.global_step % config.checkpoint_every_steps == 0:
                control.should_save = True
            if mode == "interrupted" and state.global_step == 2:
                control.should_save = True
                control.should_training_stop = True
            optimizer = kwargs.get("optimizer")
            verify_amp_recovery(trainer, optimizer, kwargs["model"], numeric, state.global_step)
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            files = checkpoint_files_valid(checkpoint)
            write_json(
                checkpoint / "phase3b_state.json",
                {
                    "phase": "3B",
                    "config": asdict(config),
                    "features_sha256": analysis["features_sha256"],
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "data_seed": config.seed,
                    "ignore_data_skip": False,
                    "files": files,
                },
            )

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if "details" in pending:
                history.append(
                    {
                        "epoch": state.epoch,
                        "global_step": state.global_step,
                        "metrics": pending["details"],
                        "eval_runtime": metrics.get("eval_runtime"),
                    }
                )
                write_json(history_path, history)

        def on_log(self, args, state, control, logs, **kwargs):
            write_json(
                output / "progress.json",
                {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "max_steps": state.max_steps,
                    "elapsed_seconds": perf_counter() - started,
                    "latest_log": logs,
                    "numerical": {**numeric, "nonfinite_gradient_norms": trainer.nonfinite_norms},
                    "loss_scale": trainer.accelerator.scaler.get_scale()
                    if trainer.accelerator.scaler is not None
                    else None,
                },
            )

    arguments = training_arguments(config, output, full=mode == "full")
    trainer = create_stable_trainer(
        model=model,
        args=arguments,
        train_dataset=train_records,
        eval_dataset=features["val"] if mode == "full" else None,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=compute_metrics if mode == "full" else None,
        callbacks=[RecordState()],
        optimizer_cls_and_kwargs=(
            torch.optim.AdamW,
            {
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "foreach": False,
            },
        ),
    )
    if trainer.accelerator.scaler is not None:
        scale_state = trainer.accelerator.scaler.state_dict()
        scale_state["scale"] = config.initial_loss_scale
        scale_state["growth_interval"] = config.loss_scale_growth_interval
        trainer.accelerator.scaler.load_state_dict(scale_state)
    with protect_test_csv(Path("data/processed/phase1-final")) as guard:
        with DriverMonitor(config.monitor_interval_seconds) as monitor:
            torch.cuda.reset_peak_memory_stats()
            try:
                train_result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
            except torch.cuda.OutOfMemoryError:
                write_json(
                    output / "failure.json",
                    {
                        "status": "cuda_oom",
                        "config": asdict(config),
                        "pytorch": gpu_memory(),
                        "driver": monitor.summary(),
                        "latest_checkpoints": sorted(
                            path.name for path in output.glob("checkpoint-*")
                        ),
                    },
                )
                raise
            train_seconds = perf_counter() - started
            trainer.save_model(str(output / ("best_model" if mode == "full" else "final_model")))
            model_path = output / ("best_model" if mode == "full" else "final_model")
            best_metrics = None
            if mode == "full":
                prediction = trainer.predict(
                    features["val"], metric_key_prefix="selected_validation"
                )
                best_metrics = classification_metrics(
                    prediction.label_ids, prediction.predictions.argmax(axis=1)
                )
            numeric["nonfinite_losses"] = trainer.nonfinite_losses
            numeric["nonfinite_gradient_norms"] = trainer.nonfinite_norms
            result = {
                "phase": "3B",
                "mode": mode,
                "config": asdict(config),
                "features_sha256": analysis["features_sha256"],
                "data": analysis["data"],
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "source_sha256": file_sha256(Path(__file__)),
                "software": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "accelerate", "nvidia-ml-py")
                },
                "diagnostic": diagnostic,
                "train_seconds": train_seconds,
                "train_result": train_result.metrics,
                "global_step": trainer.state.global_step,
                "epoch": trainer.state.epoch,
                "best_metric": trainer.state.best_metric,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "selected_validation": best_metrics,
                "validation_history": history,
                "numerical": numeric,
                "pytorch": gpu_memory(),
                "driver": monitor.summary(),
                "saved_model_sha256": file_sha256(model_path / "model.safetensors"),
                "resume_from": str(resume) if resume else None,
                "test_csv_open_attempts": guard["blocked_open_attempts"],
                "checkpoint_files": {
                    path.name: checkpoint_files_valid(path)
                    for path in sorted(output.glob("checkpoint-*"))
                },
            }
            write_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("cache/phase3b/v1"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["full", "control", "interrupted", "resumed"], default="full"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    config = FullConfig(**json.loads(args.config.read_text()))
    result = run_training(
        args.features_dir,
        config,
        args.output_dir,
        mode=args.mode,
        resume=args.resume,
        gate=args.gate,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "mode",
                    "global_step",
                    "epoch",
                    "best_metric",
                    "selected_validation",
                    "numerical",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
