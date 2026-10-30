"""Smoke opcional com modelos originais e exemplos sintéticos; nunca lê datasets."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from app.app import EXAMPLES, create_app, launch_options
from reviewsense.data import file_sha256
from reviewsense.final_evaluation import test_access_policy
from reviewsense.predict import ArtifactPaths, InferenceService


def run_verification() -> dict:
    from gradio_client import Client

    root = Path.cwd()
    paths = ArtifactPaths.from_repository(root)
    files = [
        paths.bert_dir / name
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
    ]
    files.append(paths.svm_dir / "selected_model.joblib")
    before = {str(path.relative_to(root)): file_sha256(path) for path in files}
    cases = ["positive", "negative", "neutral", "ambiguous"]
    measurements = {}
    with test_access_policy(root / "data/processed/phase1-final/test.csv") as access:
        auto, cpu = InferenceService(device="auto"), InferenceService(device="cpu")
        for name, service, model in (
            ("bert_auto", auto, "bertimbau"),
            ("svm_cpu", auto, "svm"),
            ("bert_cpu", cpu, "bertimbau"),
        ):
            first = service.predict(EXAMPLES[0], model=model)
            latencies, predicted = [], {}
            for repeat in range(3):
                for case, text in zip(cases, EXAMPLES, strict=True):
                    result = service.predict(text, model=model)
                    assert result.cold_start_seconds == 0
                    assert result.class_id in {0, 1, 2}
                    if model == "bertimbau":
                        assert abs(sum(score.value for score in result.scores) - 1) < 1e-8
                    else:
                        assert not result.scores and result.score_type is None
                    latencies.append(result.inference_seconds)
                    if repeat == 0:
                        predicted[case] = {"class_id": result.class_id, "label": result.label}
            measurements[name] = {
                "device": first.device,
                "precision": first.precision,
                "cold_start_seconds": first.cold_start_seconds,
                "first_inference_seconds": first.inference_seconds,
                "warm_median_seconds": median(latencies),
                "warm_min_seconds": min(latencies),
                "warm_max_seconds": max(latencies),
                "requests_measured": len(latencies),
                "batch_size": 1,
                "synthetic_case_predictions": predicted,
            }
            long_text = ("O produto é bom, mas a entrega não foi adequada. " * 180)[:8000]
            bounded = service.predict(long_text, model=model)
            assert bounded.class_id in {0, 1, 2}
            measurements[name]["bounded_input_check"] = {
                "characters": len(long_text),
                "result_valid": True,
                "inference_seconds": bounded.inference_seconds,
            }
        demo = create_app(cpu)
        try:
            _, url, _ = demo.launch(prevent_thread_lock=True, **launch_options(port=7861))
            assert demo.run_history is False and demo.analytics_enabled is False
            client = Client(url, verbose=False, analytics_enabled=False)
            cpu_html = client.predict(EXAMPLES[0], "bertimbau", api_name="/analyze")
            assert "CPU" in cpu_html and "Scores normalizados" in cpu_html
            svm_html = client.predict(EXAMPLES[1], "svm", api_name="/analyze")
            assert "sem probabilidades calibradas" in svm_html and "score-track" not in svm_html
        finally:
            demo.close()
        after = {str(path.relative_to(root)): file_sha256(path) for path in files}
        assert before == after
        assert access["read_opens"] == access["blocked_opens"] == 0
        report = {
            "phase": "4A",
            "purpose": "synthetic_inference_smoke_not_dataset_evaluation",
            "executed_at": datetime.now(ZoneInfo("America/Recife")).isoformat(),
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_clean": not subprocess.check_output(["git", "status", "--porcelain"], text=True),
            "source_sha256": {
                name: file_sha256(root / name)
                for name in (
                    "src/reviewsense/predict.py",
                    "app/app.py",
                    "app/style.css",
                    "scripts/verify_phase4a_local.py",
                )
            },
            "software": {
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "scikit-learn",
                        "numpy",
                        "gradio",
                        "gradio-client",
                        "pillow",
                        "fastapi",
                        "pydantic",
                        "starlette",
                    )
                },
            },
            "conditions": {
                "requests": "one warmup then 3 passes over 4 fixed synthetic examples",
                "includes": "tokenization/TF-IDF, transfers, forward, normalized scores",
                "excludes": "cold loading, queue wait, HTTP roundtrip, rendering",
                "model_loading_measured_separately": True,
                "loading_scope": "lazy backend initialization; CPU follows GPU and reuses imports",
                "measurement_order": ["bert_auto", "svm_cpu", "bert_cpu"],
                "torch_intraop_threads": 2,
                "svm_threadpool_limit": 1,
                "isolated_host_or_thermal_control": False,
                "max_length": 256,
                "max_input_characters": 8000,
            },
            "measurements": measurements,
            "checks": {
                "weights_and_tokenizers_unchanged": before == after,
                "artifact_sha256": before,
                "cpu_gradio_http_bert": True,
                "cpu_gradio_http_svm": True,
                "test_read_opens": access["read_opens"],
                "blocked_test_opens": access["blocked_opens"],
            },
        }
    return report


def main() -> None:
    report = run_verification()
    destination = Path("outputs/phase4a/local_smoke.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "verified", "measurements": report["measurements"]}, indent=2))


if __name__ == "__main__":
    main()
