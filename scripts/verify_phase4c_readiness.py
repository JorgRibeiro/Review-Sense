"""Preflight offline com pesos reais e textos sintéticos; nenhuma operação remota."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np

from reviewsense.final_evaluation import atomic_json, test_access_policy
from reviewsense.model_packaging import DEFAULT_MANIFEST, DEFAULT_PACKAGE, SOURCE_PREFIX
from reviewsense.model_resolution import LocalPackage
from reviewsense.predict import InferenceService
from reviewsense.release_readiness import load_preflight_inputs, preflight
from scripts.verify_phase4b_package import TEXTS, no_network, verify


def run(package: Path, manifest: Path, descriptor: Path, links: Path, *, device: str) -> dict:
    root = Path.cwd()
    with test_access_policy(root / "data/processed/phase1-final/test.csv") as guard:
        with no_network() as network:
            m, d, link_receipt = load_preflight_inputs(manifest, descriptor, links)
            result = preflight(package, m, d, link_receipt)
            result["equivalence"] = verify(
                package,
                m,
                root / SOURCE_PREFIX,
                device=device,
                precision="fp16" if device == "cuda" else "fp32",
                repository=root,
            )
            service = InferenceService(
                device=device,
                model_source=LocalPackage(package, descriptor, package.parent),
            )
            service.prepare()
            predictions = service.predict_batch(TEXTS)
            cached = service.predict(TEXTS[0])
            assert len(predictions) == 5 and cached.cold_start_seconds == 0
            assert all(p.class_id in {0, 1, 2} and len(p.scores) == 3 for p in predictions)
            assert all(np.isfinite([s.value for s in p.scores]).all() for p in predictions)
            result["inference_service"] = {
                "synthetic_samples": 5,
                "finite_three_class_scores": True,
                "preload_and_cache_verified": True,
                "source": "local_package_and_public_descriptor",
                "internal_training_reports_required_by_service": False,
                "device": predictions[0].device,
                "precision": predictions[0].precision,
            }
            assert network["blocked_connections"] == 0
        assert guard["read_opens"] == guard["blocked_opens"] == 0
        result.update(
            {
                "TECHNICALLY_READY": True,
                "network_connections": 0,
                "test_csv_read_opens": guard["read_opens"],
                "blocked_test_csv_opens": guard["blocked_opens"],
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "software": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "gradio",
                        "huggingface-hub",
                        "tokenizers",
                    )
                },
            }
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Obrigatório; inteiramente offline, com inferência sintética local.",
    )
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--descriptor", type=Path, default=Path("publishing/inference_descriptor.json")
    )
    parser.add_argument("--links", type=Path, default=Path("publishing/readiness_links.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    result = run(args.package, args.manifest, args.descriptor, args.links, device=args.device)
    # Saída fixa ignorada: nunca permite sobrescrever os modelos ou resultados históricos.
    atomic_json(Path(f"outputs/phase4c/readiness_{args.device}.json"), result, replace=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
