"""Benchmark computacional controlado e monitoramento NVML, sem seleção por qualidade."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from reviewsense.phase3b_data import FullConfig, load_features, truncate_records, variant_config
from reviewsense.train_transformer import diagnose_cuda, gpu_memory, write_json
from reviewsense.transformer_data import label_mappings


class DriverMonitor:
    """Memória total do driver amostrada por NVML; o máximo não é um pico contínuo exato."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = 0
        self.max_used = 0
        self.min_free = 2**64
        self.total = 0
        self.max_utilization = 0
        self.errors = []

    def _sample(self):
        try:
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.samples += 1
            self.total = memory.total
            self.max_used = max(self.max_used, memory.used)
            self.min_free = min(self.min_free, memory.free)
            self.max_utilization = max(self.max_utilization, utilization.gpu)
        except self.nvml.NVMLError as error:
            self.errors.append(type(error).__name__)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def summary(self):
        return {
            "source": "NVML",
            "sampling_interval_seconds": self.interval,
            "samples": self.samples,
            "total_bytes": self.total,
            "peak_used_bytes_sampled": self.max_used,
            "min_free_bytes": self.min_free,
            "peak_utilization_percent_sampled": self.max_utilization,
            "errors": list(self.errors),
            "includes_other_processes_and_driver": True,
        }

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self._sample()
        self.nvml.nvmlShutdown()


def benchmark_variant(features_dir: Path, config: FullConfig) -> dict:
    """Treina lotes representativos e longos com FP16 e batch efetivo 16, sem F1."""
    import torch
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs, set_seed
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    diagnostic = diagnose_cuda()
    features, analysis = load_features(features_dir)
    tokenizer = AutoTokenizer.from_pretrained(features_dir / "tokenizer", local_files_only=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[
            GradScalerKwargs(
                init_scale=config.initial_loss_scale,
                growth_interval=config.loss_scale_growth_interval,
            )
        ],
    )
    if accelerator.device.type != "cuda" or accelerator.num_processes != 1:
        raise RuntimeError("Benchmark exige uma GPU em um único processo.")
    result = {
        "variant_id": f"{config.max_length}-b{config.train_batch_size}"
        f"-gc{int(config.gradient_checkpointing)}",
        "config": asdict(config),
        "diagnostic": diagnostic,
        "features_sha256": analysis["features_sha256"],
        "initial_scaler_state": accelerator.scaler.state_dict(),
        "stage": "load_model",
        "status": "initializing",
        "numerically_stable": False,
    }
    with DriverMonitor(config.monitor_interval_seconds) as monitor:
        try:
            torch.cuda.reset_peak_memory_stats()
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
            if config.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                foreach=False,
            )
            model, optimizer = accelerator.prepare(model, optimizer)
            collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
            numeric = {
                "nonfinite_losses": 0,
                "nonfinite_gradient_norms": 0,
                "skipped_optimizer_updates": 0,
                "optimizer_updates": 0,
                "max_unclipped_gradient_norm": 0.0,
            }
            observations = {"parameter_device": str(next(model.parameters()).device)}
            first_linear = next(
                layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)
            )

            def observe(module, inputs, output):
                observations.setdefault("linear_dtype", str(output.dtype))
                observations.setdefault("linear_device", str(output.device))

            handle = first_linear.register_forward_hook(observe)

            def run_records(records):
                loader = accelerator.prepare(
                    DataLoader(
                        records,
                        batch_size=config.train_batch_size,
                        shuffle=False,
                        collate_fn=collator,
                        num_workers=0,
                        pin_memory=True,
                    )
                )
                torch.cuda.synchronize()
                start = perf_counter()
                loss_sum, rows, tokens = 0.0, 0, 0
                model.train()
                for batch in loader:
                    with accelerator.accumulate(model):
                        with accelerator.autocast():
                            output = model(**batch)
                        loss = output.loss
                        value = float(loss.detach())
                        numeric["nonfinite_losses"] += int(not math.isfinite(value))
                        if not math.isfinite(value):
                            raise FloatingPointError("Loss não finita no piloto.")
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            norm = float(
                                accelerator.clip_grad_norm_(
                                    model.parameters(), config.max_grad_norm
                                )
                            )
                            numeric["nonfinite_gradient_norms"] += int(not math.isfinite(norm))
                            if math.isfinite(norm):
                                numeric["max_unclipped_gradient_norm"] = max(
                                    numeric["max_unclipped_gradient_norm"], norm
                                )
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        if accelerator.sync_gradients:
                            if accelerator.optimizer_step_was_skipped:
                                numeric["skipped_optimizer_updates"] += 1
                            else:
                                numeric["optimizer_updates"] += 1
                    rows += len(batch["labels"])
                    tokens += int(batch["attention_mask"].sum())
                    loss_sum += value * len(batch["labels"])
                torch.cuda.synchronize()
                seconds = perf_counter() - start
                return {
                    "seconds": seconds,
                    "samples": rows,
                    "examples_per_second": rows / seconds,
                    "nonpadding_tokens_per_second": tokens / seconds,
                    "mean_loss_diagnostic_only": loss_sum / rows,
                }

            natural = [features["train"][i] for i in features["pilot_representative_indices"]]
            long = [features["train"][i] for i in features["pilot_long_indices"]]
            natural = truncate_records(natural, config.max_length, tokenizer.sep_token_id)
            long = truncate_records(long, config.max_length, tokenizer.sep_token_id)
            result["stage"] = "warmup"
            run_records(natural[: config.pilot_warmup_samples])
            result["regimes"] = {}
            for name, records in (("representative", natural), ("long_stress", long)):
                result["stage"] = name
                repeats = [run_records(records) for _ in range(config.pilot_repeats)]
                durations = [row["seconds"] for row in repeats]
                result["regimes"][name] = {
                    "repeats": repeats,
                    "median_seconds": float(np.median(durations)),
                    "median_examples_per_second": len(records) / float(np.median(durations)),
                    "pytorch": gpu_memory(),
                }
            result["pytorch"] = gpu_memory()
            result["numerical"] = numeric
            result["gpu_observations"] = observations
            result["status"] = "completed"
            result["numerically_stable"] = (
                numeric["nonfinite_losses"]
                == numeric["nonfinite_gradient_norms"]
                == numeric["skipped_optimizer_updates"]
                == 0
            )
            result["stage"] = "completed"
            handle.remove()
            model = optimizer = None
            del first_linear
            accelerator.free_memory()
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as error:
            result["status"] = "cuda_oom"
            result["pytorch"] = gpu_memory()
            error.__traceback__ = None
        except FloatingPointError:
            result["status"] = "nonfinite"
        result["driver"] = monitor.summary()
    result["software"] = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "accelerate", "nvidia-ml-py")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("cache/phase3b/v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/bertimbau_phase3b.json"))
    parser.add_argument("--length", type=int, required=True, choices=[128, 256])
    parser.add_argument("--batch", type=int, required=True, choices=[1, 2, 4])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Resultado já existe; use outra saída.")
    config = FullConfig(**json.loads(args.config.read_text()))
    config = variant_config(config, args.length, args.batch, args.gradient_checkpointing)
    result = benchmark_variant(args.features_dir, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "variant_id",
                    "status",
                    "numerically_stable",
                    "driver",
                    "pytorch",
                    "numerical",
                    "regimes",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
