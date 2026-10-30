"""Candidato de distribuição local: cópia verificada, sem treino, datasets ou upload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from reviewsense.data import file_sha256
from reviewsense.labels import LABEL_NAMES

ARTIFACT_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")
DOCUMENT_FILES = (
    "README.md",
    "BASE_MODEL_LICENSE.txt",
    "PROJECT_CODE_LICENSE.txt",
    "ATTRIBUTIONS.md",
    "DISTRIBUTION_STATUS.md",
)
PACKAGE_FILES = tuple(sorted(ARTIFACT_FILES + DOCUMENT_FILES))
SOURCE_ONLY_FILES = {"training_args.bin"}
BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
BASE_REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
SOURCE_PREFIX = "models/bertimbau/phase3b-full-v3/best_model"
PACKAGE_VERSION = "1.0.0-rc.1"
LICENSE_STATUS = "REQUIRES_REVIEW"
DEFAULT_PACKAGE = Path("outputs/phase4b/model-repo-staging")
DEFAULT_MANIFEST = Path("reports/phase4b_package_manifest.json")
METRIC_NAMES = {
    "F1 Macro": "f1_macro",
    "Precision Macro": "precision_macro",
    "Recall Macro": "recall_macro",
    "Accuracy": "accuracy",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _no_symlink_ancestors(path: Path) -> None:
    if ".." in path.parts:
        raise ValueError("Caminho com travessia de diretórios rejeitado.")
    for item in (path.absolute(), *path.absolute().parents):
        if item.is_symlink():
            raise ValueError("Links simbólicos não são permitidos.")


def safe_file(directory: Path, name: str) -> Path:
    _no_symlink_ancestors(directory)
    if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
        raise ValueError("Nome de arquivo inseguro.")
    path = directory / name
    if path.is_symlink():
        raise ValueError("Link simbólico rejeitado.")
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Arquivo necessário ausente ou não regular: {name}.")
    return path


def inspect_directory(
    directory: Path, required: tuple[str, ...], *, ignored: set[str] | None = None
):
    _no_symlink_ancestors(directory)
    actual = {item.name for item in directory.iterdir()}
    if set(required) - actual or actual - set(required) - (ignored or set()):
        raise ValueError("Diretório incompleto ou com arquivos inesperados.")
    # Mesmo arquivos explicitamente excluídos não podem ser links ou diretórios.
    for name in actual:
        safe_file(directory, name)


def _metadata_hygiene(text: str) -> None:
    if re.search(r"/home/|/Users/|[A-Za-z]:[\\/]Users[\\/]|\bhf_[A-Za-z0-9]{20,}", text):
        raise ValueError("Caminho pessoal ou possível credencial nos metadados.")


def validate_configuration(directory: Path) -> None:
    config = read_json(safe_file(directory, "config.json"))
    tokenizer = read_json(safe_file(directory, "tokenizer_config.json"))
    if (
        config.get("model_type") != "bert"
        or config.get("architectures") != ["BertForSequenceClassification"]
        or config.get("id2label") != {str(k): v for k, v in LABEL_NAMES.items()}
        or config.get("label2id") != {v: k for k, v in LABEL_NAMES.items()}
        or tokenizer.get("do_lower_case") is not False
        or tokenizer.get("strip_accents") is True
        or config.get("auto_map")
        or tokenizer.get("auto_map")
    ):
        raise ValueError("Arquitetura, tokenizer ou mapeamento incompatível com o modelo aprovado.")
    for name in ("config.json", "tokenizer_config.json"):
        _metadata_hygiene((directory / name).read_text())


def card_example(content: str) -> str:
    snippets = re.findall(r"^```python\n(.*?)^```", content, flags=re.MULTILINE | re.DOTALL)
    if len(snippets) != 1:
        raise ValueError("Model Card deve conter um único exemplo Python verificável.")
    compile(snippets[0], "model-card-example", "exec")
    return snippets[0]


def validate_model_card(content: str, expected: dict | None = None) -> dict:
    """Validação offline do subconjunto documentado de metadata; não chama validate remoto."""
    from huggingface_hub import ModelCard

    _metadata_hygiene(content)
    card = ModelCard(content)
    metadata = card.data.to_dict()
    required = {
        "language": "pt",
        "library_name": "transformers",
        "pipeline_tag": "text-classification",
        "base_model": BASE_MODEL,
        "base_model_relation": "finetune",
        "license": "unknown",
        "datasets": ["ruanchaves/b2w-reviews01"],
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("Metadados do Model Card incompatíveis ou licença conclusiva indevida.")
    if "license-under-review" not in metadata.get("tags", []):
        raise ValueError("Status de revisão de licença ausente no Model Card.")
    for section in (
        "English summary",
        "Visão geral",
        "Dados",
        "Rótulos",
        "Treinamento",
        "Avaliação independente",
        "Desempenho computacional",
        "Limitações",
        "Uso responsável",
        "Inferência local",
        "Referências",
    ):
        if f"## {section}\n" not in content:
            raise ValueError(f"Seção necessária ausente: {section}.")
    result = metadata["model-index"][0]["results"][0]
    if (
        result["task"]["type"] != "text-classification"
        or result["dataset"]["split"] != "test"
        or result["dataset"]["type"] != "ruanchaves/b2w-reviews01"
    ):
        raise ValueError("Tarefa/split inconsistentes no Model Card.")
    measured = {item["name"]: item["value"] for item in result["metrics"]}
    if set(measured) != set(METRIC_NAMES) or any(
        not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1
        for value in measured.values()
    ):
        raise ValueError("Métricas estruturadas inválidas.")
    if expected is not None:
        for name, key in METRIC_NAMES.items():
            if measured[name] != expected[key] or str(expected[key]) not in content:
                raise ValueError("Model Card diverge das métricas congeladas da Fase 3C.")
        for values in expected["per_class"].values():
            if any(str(values[key]) not in content for key in ("precision", "recall", "f1-score")):
                raise ValueError("Métricas por classe divergentes do registro congelado.")
    card_example(content)
    return metadata


def file_inventory(directory: Path, names: tuple[str, ...]) -> dict:
    return {
        name: {
            "bytes": safe_file(directory, name).stat().st_size,
            "sha256": file_sha256(safe_file(directory, name)),
        }
        for name in sorted(names)
    }


def validate_manifest(manifest: dict) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("package_version") != PACKAGE_VERSION
        or manifest.get("architecture") != "BertForSequenceClassification"
        or manifest.get("artifact_identity") != "reviewsense-ptbr-bertimbau-v1"
        or manifest.get("model_card") != "README.md"
        or manifest.get("max_length") != 256
        or manifest.get("labels") != {str(k): v for k, v in LABEL_NAMES.items()}
        or manifest.get("license_status") != LICENSE_STATUS
        or manifest.get("upload_authorized") is not False
        or manifest.get("huggingface_repo_id") is not None
        or manifest.get("base_model") != BASE_MODEL
        or manifest.get("base_revision") != BASE_REVISION
        or manifest.get("selected_checkpoint") != {"epoch": 1, "global_step": 5543}
        or set(manifest.get("files", {})) != set(PACKAGE_FILES)
    ):
        raise ValueError(
            "Manifest inválido ou tentativa de alterar a política de publicação da Fase 4B."
        )
    for record in manifest["files"].values():
        if (
            not isinstance(record["bytes"], int)
            or record["bytes"] < 1
            or not re.fullmatch(r"[a-f0-9]{64}", record["sha256"])
        ):
            raise ValueError("Tamanho/hash inválido no manifest.")
    for name in ARTIFACT_FILES:
        if manifest["files"][name]["sha256"] != manifest["original_artifact_sha256"][name]:
            raise ValueError("Manifest não corresponde ao checkpoint original selecionado.")
    digest = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest.get("package_content_sha256"):
        raise ValueError("Fingerprint do conteúdo do pacote divergente.")
    _metadata_hygiene(json.dumps(manifest, ensure_ascii=False))


def validate_package(directory: Path, manifest: dict) -> None:
    validate_manifest(manifest)
    inspect_directory(directory, PACKAGE_FILES)
    if file_inventory(directory, PACKAGE_FILES) != manifest["files"]:
        raise ValueError("Hashes/tamanhos do pacote divergem do manifest.")
    validate_configuration(directory)
    validate_model_card((directory / "README.md").read_text(), manifest["frozen_test_metrics"])
    for name in DOCUMENT_FILES:
        _metadata_hygiene((directory / name).read_text())
    status = (directory / "DISTRIBUTION_STATUS.md").read_text()
    if "REQUIRES_REVIEW" not in status or "upload_authorized: false" not in status:
        raise ValueError("Aviso de distribuição não mantém o bloqueio de publicação.")


def build_package(
    source: Path,
    documents: Path,
    destination: Path,
    *,
    expected: dict[str, str],
    metrics: dict,
    provenance: dict,
) -> dict:
    """Cria somente destino novo; preserva bytes e nunca exporta training_args.bin."""
    inspect_directory(source, ARTIFACT_FILES, ignored=SOURCE_ONLY_FILES)
    inspect_directory(documents, DOCUMENT_FILES)
    validate_configuration(source)
    if set(expected) != set(ARTIFACT_FILES):
        raise ValueError("Referência de hashes incompleta.")
    for name in ARTIFACT_FILES:
        if file_sha256(safe_file(source, name)) != expected[name]:
            raise ValueError("Checkpoint incorreto ou corrompido frente à referência congelada.")
    validate_model_card((documents / "README.md").read_text(), metrics)
    for name in DOCUMENT_FILES:
        _metadata_hygiene((documents / name).read_text())
    _no_symlink_ancestors(destination)
    if os.path.lexists(destination):
        raise FileExistsError("O destino já existe; não substituir um pacote preparado.")
    for origin in (source, documents):
        if destination.absolute().is_relative_to(
            origin.absolute()
        ) or origin.absolute().is_relative_to(destination.absolute()):
            raise ValueError("Destino não pode sobrepor diretórios de origem.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Sem timestamps, caminhos absolutos ou repo_id inventado no manifest técnico.
    manifest = {
        "schema_version": 1,
        "package_version": PACKAGE_VERSION,
        "artifact_identity": "reviewsense-ptbr-bertimbau-v1",
        "architecture": "BertForSequenceClassification",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "selected_checkpoint": {"epoch": 1, "global_step": 5543},
        "max_length": 256,
        "labels": {str(k): v for k, v in LABEL_NAMES.items()},
        "license_status": LICENSE_STATUS,
        "upload_authorized": False,
        "huggingface_repo_id": None,
        "original_artifact_sha256": expected,
        "frozen_test_metrics": metrics,
        "provenance": provenance,
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")
        },
        "model_card": "README.md",
        "files": {},
    }
    with tempfile.TemporaryDirectory(prefix=".phase4b-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        for name in PACKAGE_FILES:
            origin = source if name in ARTIFACT_FILES else documents
            path = safe_file(origin, name)
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(path, flags), "rb") as input_stream:
                with (staging / name).open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
            if file_sha256(staging / name) != file_sha256(safe_file(origin, name)):
                raise ValueError("Bytes copiados não correspondem à origem.")
        manifest["files"] = file_inventory(staging, PACKAGE_FILES)
        manifest["package_content_sha256"] = hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        validate_package(staging, manifest)
        if os.path.lexists(destination):
            raise FileExistsError("O destino foi criado durante a preparação; não sobrescrever.")
        staging.rename(destination)
    return manifest


def release_inputs(root: Path) -> tuple[dict, dict, dict]:
    phase3c_path = root / "reports/phase3c_final_metrics.json"
    phase3b_path = root / "reports/bertimbau_phase3b_metrics.json"
    final = read_json(phase3c_path)
    checkpoint = read_json(phase3b_path)
    verified = read_json(root / "reports/phase3c_verification.json")
    if (
        file_sha256(phase3c_path) != verified["metrics_json_sha256"]
        or file_sha256(phase3b_path) != final["inventory"]["reports/bertimbau_phase3b_metrics.json"]
        or not checkpoint["best_checkpoint"].endswith("checkpoint-5543")
    ):
        raise ValueError("Registros imutáveis ou checkpoint selecionado divergentes.")
    expected = {name: final["inventory"][f"{SOURCE_PREFIX}/{name}"] for name in ARTIFACT_FILES}
    if expected["model.safetensors"] != checkpoint["saved_model_sha256"]:
        raise ValueError("Peso não corresponde à seleção da Fase 3B.")
    provenance = {
        "packaging_code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "training_code_commit": checkpoint["launch"]["code_commit"],
        "evaluation_code_commit": final["git"]["commit"],
        "metrics_reference": "reports/phase3c_final_metrics.json",
        "metrics_sha256": file_sha256(phase3c_path),
        "license_review_reference": "docs/FASE4B_LICENSE_REVIEW.md",
        "license_review_sha256": file_sha256(root / "docs/FASE4B_LICENSE_REVIEW.md"),
    }
    return expected, final["models"]["bertimbau"]["metrics"], provenance


def compare_outputs(original, packaged, *, device: str, precision: str) -> dict:
    a, b = np.asarray(original), np.asarray(packaged)
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 3 or not len(a):
        raise ValueError("Shape dos logits inválido.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Logits não finitos.")
    if device == "cpu" and precision != "fp32":
        raise ValueError("Comparação CPU deve utilizar FP32.")
    if device not in {"cpu", "cuda"} or precision not in {"fp32", "fp16"}:
        raise ValueError("Dispositivo/precisão de comparação inválidos.")
    atol, rtol = (1e-3, 1e-3) if precision == "fp16" else (1e-6, 1e-5)
    equal_classes = bool(np.array_equal(a.argmax(axis=1), b.argmax(axis=1)))
    close_logits = bool(np.allclose(a, b, atol=atol, rtol=rtol))

    def softmax(values):
        exponentials = np.exp(values - values.max(axis=1, keepdims=True))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    scores_a, scores_b = softmax(a.astype(np.float64)), softmax(b.astype(np.float64))
    close_scores = bool(np.allclose(scores_a, scores_b, atol=atol, rtol=rtol))
    if not equal_classes or not close_logits or not close_scores:
        raise ValueError("Pacote e original não são equivalentes na comparação controlada.")
    return {
        "same_predicted_classes": equal_classes,
        "logits_allclose": close_logits,
        "normalized_scores_allclose": close_scores,
        "atol": atol,
        "rtol": rtol,
        "max_absolute_logit_difference": float(np.max(np.abs(a.astype(float) - b))),
        "max_absolute_score_difference": float(np.max(np.abs(scores_a - scores_b))),
        "samples": len(a),
        "device": device,
        "precision": precision,
    }


def local_logits(directory: Path, texts: list[str], *, device: str, precision: str):
    """Transformers padrão, sem qualquer resolução de paths do projeto."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU requer FP32 nesta verificação.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível para a comparação solicitada.")
    validate_configuration(directory)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, token=False
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            directory,
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    if model.config.id2label != LABEL_NAMES or tokenizer.do_lower_case:
        raise ValueError("Mapeamento ou tokenizer carregado incompatível.")
    output, maximum = [], 0
    with (
        torch.inference_mode(),
        torch.autocast(device, dtype=torch.float16, enabled=precision == "fp16"),
    ):
        for start in range(0, len(texts), 2):
            tokens = tokenizer(
                texts[start : start + 2],
                max_length=256,
                truncation=True,
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            ).to(device)
            maximum = max(maximum, tokens["input_ids"].shape[1])
            output.append(model(**tokens).logits.float().cpu().numpy())
    logits = np.concatenate(output)
    if maximum > 256 or not np.isfinite(logits).all():
        raise ValueError("Truncamento/logits inválidos.")
    return logits, {
        "maximum_padded_tokens": maximum,
        "id2label_verified": True,
        "tokenizer_cased_verified": True,
        "eval_and_inference_mode": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    if os.path.lexists(args.manifest):
        raise FileExistsError("Manifest já existe; não substituir a preparação registrada.")
    for directory in (args.destination, Path(SOURCE_PREFIX), Path("publishing/model_card")):
        if args.manifest.absolute().is_relative_to(directory.absolute()):
            raise ValueError("Manifest técnico deve ficar fora do pacote e das origens.")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True):
        raise ValueError("Faça commit do código e documentação antes de preparar o candidato real.")
    from reviewsense.final_evaluation import atomic_json, test_access_policy

    with test_access_policy(Path("data/processed/phase1-final/test.csv")) as access:
        expected, metrics, provenance = release_inputs(Path.cwd())
        manifest = build_package(
            Path(SOURCE_PREFIX),
            Path("publishing/model_card"),
            args.destination,
            expected=expected,
            metrics=metrics,
            provenance=provenance,
        )
        assert access["read_opens"] == access["blocked_opens"] == 0
        atomic_json(args.manifest, manifest)
    print(
        json.dumps(
            {
                "status": "prepared_locally",
                "license_status": LICENSE_STATUS,
                "upload_authorized": False,
                "files": list(manifest["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
