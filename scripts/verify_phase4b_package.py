"""Verifica pacote local em exemplos sintéticos, sem upload ou avaliação de dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from reviewsense.model_packaging import (
    ARTIFACT_FILES,
    DEFAULT_MANIFEST,
    DEFAULT_PACKAGE,
    PACKAGE_FILES,
    SOURCE_PREFIX,
    card_example,
    compare_outputs,
    file_inventory,
    local_logits,
    read_json,
    safe_file,
    validate_package,
)

TEXTS = [
    "O produto chegou rápido e funciona bem.",
    "Não funcionou e a embalagem veio quebrada.",
    "O produto corresponde à descrição. Ainda estou testando.",
    "Gostei do material, mas a entrega demorou bastante.",
    ("O produto é bom, mas a entrega não foi adequada. " * 180)[:8000],
]


@contextmanager
def no_network():
    state = {"active": True, "blocked_connections": 0}

    def guard(event, args):
        if state["active"] and event == "socket.connect":
            state["blocked_connections"] += 1
            raise PermissionError("Conexões proibidas durante verificação local.")

    sys.addaudithook(guard)
    try:
        yield state
    finally:
        state["active"] = False


def verify_independent_example(
    package: Path, repository: Path, *, names: tuple[str, ...] = PACKAGE_FILES
) -> dict:
    """Executa o snippet literal fora do repo, com cache vazio e acesso à origem bloqueado."""
    if names not in (PACKAGE_FILES, tuple(sorted((*PACKAGE_FILES, "LICENSE")))):
        raise ValueError("Allowlist independente incompatível.")
    snippet = card_example((package / "README.md").read_text())
    worker = """import json, os, runpy, sys
from pathlib import Path
roots = [Path(value).resolve() for value in json.loads(os.environ["BLOCKED_REPOSITORY_ROOTS"])]
# -I não remove caminhos acrescentados por arquivos .pth de instalações editable.
sys.path[:] = [value for value in sys.path
               if not any(Path(value).resolve().is_relative_to(root) for root in roots)]
state = {"repository_opens": 0, "network_connections": 0}
def guard(event, args):
    if event == "socket.connect":
        state["network_connections"] += 1
        raise PermissionError("Network disabled")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(root) for root in roots):
            state["repository_opens"] += 1
            raise PermissionError("Original repository access blocked")
sys.addaudithook(guard)
variables = runpy.run_path(sys.argv[1])
assert "reviewsense" not in sys.modules
assert variables["device"].type == "cpu"
assert len(variables["scores"]) == 3
assert abs(sum(variables["scores"]) - 1) < 1e-6
print("INDEPENDENT_CHECK:" + json.dumps(state))
"""
    with tempfile.TemporaryDirectory(prefix="reviewsense-independent-") as temporary:
        outside = Path(temporary)
        copied = outside / "model"
        copied.mkdir()
        for name in names:
            shutil.copyfile(safe_file(package, name), copied / name, follow_symlinks=False)
        # Confirma que a cópia independente também tem os mesmos bytes.
        if file_inventory(copied, names) != file_inventory(package, names):
            raise ValueError("Cópia independente divergente.")
        example = outside / "example.py"
        example.write_text(snippet)
        harness = outside / "worker.py"
        harness.write_text(worker)
        environment = {
            **os.environ,
            "REVIEWSENSE_MODEL_DIR": str(copied),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "HF_HOME": str(outside / "empty-cache"),
            "HF_HUB_CACHE": str(outside / "empty-cache/hub"),
            "BLOCKED_REPOSITORY_ROOTS": json.dumps(
                [
                    str(repository / name)
                    for name in ("models", "reports", "cache", "src", "publishing", "docs", "data")
                ]
            ),
        }
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-I", str(harness), str(example)],
            cwd=outside,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(
                "O exemplo isolado do Model Card falhou; não considerar o pacote validado."
            )
        records = [
            line.removeprefix("INDEPENDENT_CHECK:")
            for line in result.stdout.splitlines()
            if line.startswith("INDEPENDENT_CHECK:")
        ]
        if len(records) != 1 or json.loads(records[0]) != {
            "repository_opens": 0,
            "network_connections": 0,
        }:
            raise ValueError("Verificação de independência não comprovada.")
    return {
        "literal_model_card_example_executed": True,
        "outside_repository": True,
        "empty_hf_cache": True,
        "isolated_python": True,
        "device": "cpu",
        "precision": "fp32",
        "original_models_reports_data_accesses": 0,
        "network_connections": 0,
        "reviewsense_module_not_imported": True,
        "finite_three_class_scores": True,
    }


def verify(
    package: Path, manifest: dict, source: Path, *, device: str, precision: str, repository: Path
) -> dict:
    validate_package(package, manifest)
    original_before = file_inventory(source, ARTIFACT_FILES)
    for name, record in original_before.items():
        if record != manifest["files"][name]:
            raise ValueError("Fonte original diverge do pacote/manifest congelado.")
    with no_network() as network:
        reference, original_config = local_logits(source, TEXTS, device=device, precision=precision)
        candidate, package_config = local_logits(package, TEXTS, device=device, precision=precision)
        comparison = compare_outputs(reference, candidate, device=device, precision=precision)
        if original_config != package_config or package_config["maximum_padded_tokens"] != 256:
            raise ValueError("Configuração/truncamento efetivo divergente.")
        independent = verify_independent_example(package, repository)
        if file_inventory(source, ARTIFACT_FILES) != original_before:
            raise ValueError("Artefatos originais alterados durante a verificação.")
        validate_package(package, manifest)
        assert network["blocked_connections"] == 0
    return {
        "status": "technically_verified",
        "upload_authorized": False,
        "license_status": manifest["license_status"],
        "artifact_bytes_identical": True,
        "original_files_unchanged": True,
        "comparison": comparison,
        "configuration_and_tokenization": package_config,
        "independent_loading": independent,
        "network_connections": 0,
        "test_metrics_recomputed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", type=Path, default=Path(SOURCE_PREFIX))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()
    from reviewsense.final_evaluation import atomic_json, test_access_policy

    with test_access_policy(Path("data/processed/phase1-final/test.csv")) as guard:
        result = verify(
            args.package,
            read_json(args.manifest),
            args.source,
            device=args.device,
            precision=args.precision,
            repository=Path.cwd(),
        )
        assert guard["read_opens"] == guard["blocked_opens"] == 0
        result["test_csv_read_opens"] = guard["read_opens"]
        result["blocked_test_csv_opens"] = guard["blocked_opens"]
        output = Path(f"outputs/phase4b/verification_{args.device}_{args.precision}.json")
        atomic_json(output, result, replace=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
