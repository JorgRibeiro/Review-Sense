"""Etapas explícitas: preparação, piloto, prova de retomada e treino completo aprovado."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from reviewsense.data import file_sha256
from reviewsense.phase3b_data import FullConfig, choose_gpu_configuration, prepare_phase3b
from reviewsense.train_transformer import write_json


def worker(module: str, arguments: list[str], log_path: Path):
    """Cada worker usa novo processo, isolando contexto CUDA e estado do Accelerate."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [sys.executable, "-u", "-m", module, *arguments],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_pilot(config_path: Path, features: Path, work: Path):
    config = FullConfig(**json.loads(config_path.read_text()))
    config.validate()
    if work.exists():
        raise FileExistsError("Piloto já existe; use outra pasta.")
    work.mkdir(parents=True)
    results = []
    for length, batch in ((128, 2), (256, 2), (256, 4)):
        output = work / f"{length}-b{batch}.json"
        worker(
            "reviewsense.gpu_benchmark",
            [
                "--features-dir",
                str(features),
                "--config",
                str(config_path),
                "--length",
                str(length),
                "--batch",
                str(batch),
                "--output",
                str(output),
            ],
            work / f"{length}-b{batch}.log",
        )
        result = json.loads(output.read_text())
        results.append(result)
        print(result["variant_id"], result["status"], result["numerically_stable"], flush=True)
    try:
        selection = choose_gpu_configuration(results, config)
    except RuntimeError:
        output = work / "256-b1-gc.json"
        worker(
            "reviewsense.gpu_benchmark",
            [
                "--features-dir",
                str(features),
                "--config",
                str(config_path),
                "--length",
                "256",
                "--batch",
                "1",
                "--gradient-checkpointing",
                "--output",
                str(output),
            ],
            work / "256-b1-gc.log",
        )
        results.append(json.loads(output.read_text()))
        selection = choose_gpu_configuration(results, config)
    write_json(work / "selected_config.json", selection["config"])
    write_json(work / "selection.json", selection)
    return selection


def equal_states(left, right):
    """Compara estados de optimizer/scheduler/scaler sem executar o modelo."""
    import torch

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal_states(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            equal_states(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def verify_resume(resume_dir: Path) -> dict:
    """Comprova retomada por pesos e estados finais idênticos ao controle contínuo."""
    import torch

    control = json.loads((resume_dir / "control/results.json").read_text())
    resumed = json.loads((resume_dir / "interrupted/results.json").read_text())
    comparisons = {}
    for name in ("optimizer.pt", "scheduler.pt", "scaler.pt"):
        a = torch.load(
            resume_dir / "control/checkpoint-4" / name, map_location="cpu", weights_only=True
        )
        b = torch.load(
            resume_dir / "interrupted/checkpoint-4" / name, map_location="cpu", weights_only=True
        )
        comparisons[name] = equal_states(a, b)
        del a, b
    result = {
        "control_steps": control["global_step"],
        "interrupted_at_step": 2,
        "resumed_steps": resumed["global_step"],
        "final_model_identical": control["saved_model_sha256"] == resumed["saved_model_sha256"],
        "final_state_identical": comparisons,
        "same_runtime_source": control["source_sha256"] == resumed["source_sha256"],
        "runtime_source_sha256": resumed["source_sha256"],
        "config": resumed["config"],
        "features_sha256": resumed["features_sha256"],
        "numerical": {"control": control["numerical"], "resumed": resumed["numerical"]},
        "test_csv_open_attempts": control["test_csv_open_attempts"]
        + resumed["test_csv_open_attempts"],
    }
    result["verified"] = (
        result["final_model_identical"]
        and all(comparisons.values())
        and result["same_runtime_source"]
        and result["control_steps"] == 4
        and result["resumed_steps"] == 4
        and result["test_csv_open_attempts"] == 0
        and not any(resumed["numerical"].values())
        and not any(control["numerical"].values())
    )
    if not result["verified"]:
        raise RuntimeError(
            "Retomada divergiu do controle; treino completo não autorizado pelo gate."
        )
    write_json(resume_dir / "verification.json", result)
    return result


def run_resume_proof(work: Path, resume_dir: Path, features: Path):
    if resume_dir.exists():
        raise FileExistsError("Prova de retomada já existe; use outra pasta.")
    config = work / "selected_config.json"
    for mode, folder in (
        ("control", "control"),
        ("interrupted", "interrupted"),
        ("resumed", "interrupted"),
    ):
        arguments = [
            "--features-dir",
            str(features),
            "--config",
            str(config),
            "--output-dir",
            str(resume_dir / folder),
            "--mode",
            mode,
        ]
        if mode == "resumed":
            arguments.extend(["--resume", str(resume_dir / folder / "checkpoint-2")])
        worker("reviewsense.train_bertimbau", arguments, resume_dir / f"{mode}.log")
        print("Prova de retomada:", mode, "concluído", flush=True)
    return verify_resume(resume_dir)


def build_gate(work: Path, resume_dir: Path) -> dict:
    config = FullConfig(**json.loads((work / "selected_config.json").read_text()))
    results = [
        json.loads(path.read_text())
        for path in work.glob("*.json")
        if path.name not in ("selected_config.json", "selection.json", "acceptance.json")
    ]
    results = [result for result in results if "variant_id" in result]
    selected = choose_gpu_configuration(results, config)
    proof = verify_resume(resume_dir)
    runtime = Path(__file__).parent / "train_bertimbau.py"
    if (
        proof["runtime_source_sha256"] != file_sha256(runtime)
        or proof["config"] != selected["config"]
    ):
        raise ValueError("Código ou configuração mudou desde a prova de retomada.")
    pilot = next(
        result for result in results if result["variant_id"] == selected["pilot_result_id"]
    )
    gate = {
        "approved": True,
        "resume_verified": True,
        "selected_config": selected["config"],
        "selection": selected,
        "features_sha256": proof["features_sha256"],
        "runtime_source_sha256": proof["runtime_source_sha256"],
        "required_free_at_start_bytes": pilot["pytorch"]["peak_reserved_bytes"]
        + config.min_driver_free_bytes,
        "resume_proof": proof,
        "selection_uses_quality_metrics": False,
    }
    write_json(work / "acceptance.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "resume-proof", "accept", "train"])
    parser.add_argument("--features-dir", type=Path, default=Path("cache/phase3b/v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/bertimbau_phase3b.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("models/bertimbau/phase3b-pilot-v3"))
    parser.add_argument(
        "--resume-dir", type=Path, default=Path("models/bertimbau/phase3b-resume-v3")
    )
    parser.add_argument("--run-dir", type=Path, default=Path("models/bertimbau/phase3b-full-v2"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare_phase3b(
            Path("data/processed/phase1-final"),
            Path("reports/data_audit_summary.json"),
            FullConfig(**json.loads(args.config.read_text())),
            args.features_dir,
            Path("cache/huggingface/hub"),
        )
        print(json.dumps(result["lengths"], indent=2))
    elif args.stage == "pilot":
        print(json.dumps(run_pilot(args.config, args.features_dir, args.work_dir), indent=2))
    elif args.stage == "resume-proof":
        print(
            json.dumps(
                run_resume_proof(args.work_dir, args.resume_dir, args.features_dir), indent=2
            )
        )
    elif args.stage == "accept":
        print(json.dumps(build_gate(args.work_dir, args.resume_dir), indent=2))
    else:
        arguments = [
            "--features-dir",
            str(args.features_dir),
            "--config",
            str(args.work_dir / "selected_config.json"),
            "--output-dir",
            str(args.run_dir),
            "--gate",
            str(args.work_dir / "acceptance.json"),
        ]
        if args.resume:
            arguments.extend(["--resume", str(args.resume)])
        worker(
            "reviewsense.train_bertimbau",
            arguments,
            args.run_dir.parent / f"{args.run_dir.name}-{'resume' if args.resume else 'train'}.log",
        )
        print("Treino completo concluído:", args.run_dir)


if __name__ == "__main__":
    main()
