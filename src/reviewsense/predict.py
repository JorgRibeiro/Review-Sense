"""Inferência local reutilizável, sem dados, treino, telemetria ou download de pesos."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from threadpoolctl import threadpool_limits

from reviewsense.baseline import load_experiment
from reviewsense.data import file_sha256
from reviewsense.labels import LABEL_NAMES

ModelName = Literal["bertimbau", "svm"]
DevicePreference = Literal["auto", "cpu", "cuda"]
MAX_INPUT_CHARACTERS = 8000
MAX_BATCH_ITEMS = 16
MAX_LENGTH = 256
MODEL_PREFIX = "models/bertimbau/phase3b-full-v3/best_model"


class InputError(ValueError):
    """Entrada rejeitada antes de carregar qualquer modelo."""


class ArtifactError(RuntimeError):
    """Artefatos locais ausentes, incompatíveis ou adulterados."""


class DeviceError(RuntimeError):
    """Dispositivo solicitado não pôde executar a inferência."""


class InferenceError(RuntimeError):
    """Saída inválida ou erro de execução, sem expor entrada/traceback ao usuário."""


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolução local separada da API; futuramente pode apontar ao cache do Hub."""

    bert_dir: Path
    svm_dir: Path
    bert_record: Path
    svm_record: Path
    final_record: Path

    @classmethod
    def from_repository(cls, root: Path) -> ArtifactPaths:
        return cls(
            root / MODEL_PREFIX,
            root / "models/baseline/phase2-v1",
            root / "reports/bertimbau_phase3b_metrics.json",
            root / "reports/baseline_validation.json",
            root / "reports/phase3c_final_metrics.json",
        )


@dataclass(frozen=True)
class ClassScore:
    class_id: int
    label: str
    value: float


@dataclass(frozen=True)
class Prediction:
    label: str
    class_id: int
    model: ModelName
    device: str
    precision: str
    scores: tuple[ClassScore, ...]
    score_type: str | None
    inference_seconds: float
    cold_start_seconds: float
    batch_size: int
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["scores"] = list(value["scores"])
        value["notes"] = list(value["notes"])
        return value


@dataclass
class _BertBackend:
    model: Any
    tokenizer: Any
    device: str
    fp16: bool
    notes: tuple[str, ...] = ()


def validate_text(text: object) -> str:
    """Preserva o texto original; strip é usado somente para detectar vazio."""
    if not isinstance(text, str):
        raise InputError("Insira uma avaliação em formato de texto.")
    if not text.strip():
        raise InputError("Escreva uma avaliação antes de analisar.")
    if len(text) > MAX_INPUT_CHARACTERS:
        raise InputError("A avaliação deve ter no máximo 8.000 caracteres.")
    return text


def normalized_scores(logits: Any, count: int) -> list[tuple[ClassScore, ...]]:
    """Softmax estável; scores não calibrados, em ordem decrescente e desempate por ID."""
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (count, 3) or not np.isfinite(values).all():
        raise InferenceError("O modelo retornou scores inválidos. A análise foi interrompida.")
    exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return [
        tuple(
            ClassScore(class_id, LABEL_NAMES[class_id], float(row[class_id]))
            for class_id in sorted(LABEL_NAMES, key=lambda label: (-row[label], label))
        )
        for row in probabilities
    ]


def _record(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ArtifactError(
            "Registro de integridade ausente ou inválido. Disponibilize os relatórios "
            "e artefatos das Fases 2, 3B e 3C conforme o guia de execução."
        ) from None


def _check_hash(path: Path, expected: str) -> None:
    try:
        if file_sha256(path) != expected:
            raise ArtifactError(
                "A integridade dos artefatos não confere com os registros anteriores."
            )
    except OSError:
        raise ArtifactError(
            "Modelo local indisponível. Disponibilize os artefatos treinados das Fases 2 e 3B "
            "conforme o README. Nenhum checkpoint será baixado automaticamente."
        ) from None


def _is_cuda_error(error: Exception, torch: Any) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or any(
        word in str(error).lower() for word in ("cuda", "cudnn", "cublas")
    )


class InferenceService:
    """Cache por processo e execução serializada; inicializar não lê artefatos ou dados.

    Cada batch tem até 16 entradas; BERT executa microbatches de 2. Latência inclui
    tokenização/transformação, transferência e pós-processamento, exclui carga fria
    e espera pelo lock. A carga fria é informada separadamente. Nada persiste entradas.
    """

    def __init__(self, paths: ArtifactPaths | None = None, *, device: DevicePreference = "auto"):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Dispositivo esperado: auto, cpu ou cuda.")
        self.paths = paths or ArtifactPaths.from_repository(Path(__file__).resolve().parents[2])
        self.device_preference = device
        self._lock = threading.RLock()
        self._svm = None
        self._bert: _BertBackend | None = None

    @property
    def loaded_models(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, loaded in (("bertimbau", self._bert), ("svm", self._svm))
            if loaded is not None
        )

    def _load_svm(self):
        record = _record(self.paths.svm_record)
        final = _record(self.paths.final_record)
        try:
            expected = record["model_artifact"]["sha256"]
            if final["inventory"]["models/baseline/phase2-v1/selected_model.joblib"] != expected:
                raise ArtifactError("O baseline diverge dos registros de integridade.")
            _check_hash(self.paths.svm_dir / "selected_model.joblib", expected)
            model, local_record = load_experiment(self.paths.svm_dir)
            if local_record["model_artifact"]["sha256"] != expected:
                raise ArtifactError("O baseline diverge do registro da Fase 2.")
            if not np.array_equal(model.classes_, [0, 1, 2]):
                raise ArtifactError("As classes do baseline devem ser negativo, neutro e positivo.")
            return model
        except ArtifactError:
            raise
        except Exception:
            raise ArtifactError("Não foi possível carregar o baseline local verificado.") from None

    def _verify_bert(self) -> None:
        record = _record(self.paths.bert_record)
        final = _record(self.paths.final_record)
        try:
            expected = final["inventory"]
            if expected[f"{MODEL_PREFIX}/model.safetensors"] != record["saved_model_sha256"]:
                raise ArtifactError("O BERTimbau diverge do registro da Fase 3B.")
            if not record["best_checkpoint"].endswith("checkpoint-5543"):
                raise ArtifactError("Esperado BERTimbau selecionado na época 1, update 5543.")
            for name in (
                "model.safetensors",
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                _check_hash(self.paths.bert_dir / name, expected[f"{MODEL_PREFIX}/{name}"])
        except ArtifactError:
            raise
        except (KeyError, TypeError):
            raise ArtifactError("Registro de integridade do BERTimbau incompatível.") from None

    def _create_bert(self, device: str, *, notes: tuple[str, ...] = ()) -> _BertBackend:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.paths.bert_dir, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.paths.bert_dir,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        if (
            model.config.id2label != LABEL_NAMES
            or model.config.label2id != {name: label for label, name in LABEL_NAMES.items()}
            or model.config.num_labels != 3
            or tokenizer.do_lower_case
        ):
            raise ArtifactError("Esperado tokenizer cased e mapeamento negativo/neutro/positivo.")
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ArtifactError("O modelo salvo contém pesos não finitos.")
        model = model.to(device).eval()
        fp16 = device.startswith("cuda") and torch.cuda.get_device_capability(0)[0] >= 7
        return _BertBackend(model, tokenizer, device, fp16, notes)

    def _load_bert(self) -> _BertBackend:
        self._verify_bert()
        try:
            import torch
        except ImportError:
            raise ArtifactError(
                "Instale as dependências Transformer conforme o guia de execução."
            ) from None
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.device_preference == "cuda" and not torch.cuda.is_available():
            raise DeviceError("CUDA indisponível. Inicie com --device cpu ou --device auto.")
        device = (
            "cuda:0" if self.device_preference != "cpu" and torch.cuda.is_available() else "cpu"
        )
        try:
            return self._create_bert(device)
        except ArtifactError:
            raise
        except Exception as error:
            if device.startswith("cuda") and _is_cuda_error(error, torch):
                if self.device_preference == "auto":
                    return self._recover_cpu()
                raise DeviceError(
                    "Falha na GPU. Inicie com --device cpu para usar o mesmo modelo."
                ) from None
            raise ArtifactError("Não foi possível carregar o BERTimbau local verificado.") from None

    def _recover_cpu(self) -> _BertBackend:
        """Uma recuperação por falha CUDA; fixa CPU no cache para as próximas chamadas."""
        try:
            self._bert = None
            self._verify_bert()
            return self._create_bert(
                "cpu", notes=("A GPU ficou indisponível; análise executada na CPU.",)
            )
        except Exception:
            raise DeviceError("Não foi possível executar na CPU após a falha da GPU.") from None

    @staticmethod
    def _bert_logits(backend: _BertBackend, texts: list[str]) -> np.ndarray:
        import torch

        output = []
        backend.model.eval()
        if backend.device.startswith("cuda"):
            torch.cuda.synchronize()
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda" if backend.device.startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=backend.fp16,
            ),
        ):
            for start in range(0, len(texts), 2):
                tokens = backend.tokenizer(
                    texts[start : start + 2],
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                ).to(backend.device)
                output.append(backend.model(**tokens).logits.float().cpu().numpy())
        if backend.device.startswith("cuda"):
            torch.cuda.synchronize()
        return np.concatenate(output)

    def predict(self, text: str, *, model: ModelName = "bertimbau") -> Prediction:
        return self.predict_batch([text], model=model)[0]

    def predict_batch(
        self, texts: list[str], *, model: ModelName = "bertimbau"
    ) -> list[Prediction]:
        if not isinstance(model, str) or model not in {"bertimbau", "svm"}:
            raise InputError("Selecione BERTimbau ou SVM.")
        if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= MAX_BATCH_ITEMS:
            raise InputError("Envie de 1 a 16 avaliações por chamada.")
        values = [validate_text(text) for text in texts]
        with self._lock:
            cold_start = 0.0
            if (self._svm if model == "svm" else self._bert) is None:
                loading = perf_counter()
                if model == "svm":
                    self._svm = self._load_svm()
                else:
                    self._bert = self._load_bert()
                cold_start = perf_counter() - loading
            start = perf_counter()
            if model == "svm":
                try:
                    with threadpool_limits(limits=1):
                        ids = np.asarray(self._svm.predict(values))
                    if (
                        ids.shape != (len(values),)
                        or ids.dtype.kind not in "iu"
                        or not np.isin(ids, [0, 1, 2]).all()
                    ):
                        raise InferenceError("O baseline retornou classes inválidas.")
                except InferenceError:
                    raise
                except Exception:
                    raise InferenceError("Não foi possível analisar com o SVM.") from None
                scores = [() for _ in values]
                device, precision, notes = "cpu", "sklearn_native", ()
            else:
                import torch

                try:
                    logits = self._bert_logits(self._bert, values)
                except Exception as error:
                    if self._bert.device.startswith("cuda") and _is_cuda_error(error, torch):
                        if self.device_preference == "auto":
                            self._bert = self._recover_cpu()
                            try:
                                logits = self._bert_logits(self._bert, values)
                            except Exception:
                                raise DeviceError(
                                    "A análise não pôde ser concluída na CPU."
                                ) from None
                        else:
                            raise DeviceError(
                                "Falha na GPU. Inicie a aplicação com --device cpu."
                            ) from None
                    else:
                        raise InferenceError("Não foi possível analisar com o BERTimbau.") from None
                scores = normalized_scores(logits, len(values))
                ids = np.array([row[0].class_id for row in scores])
                device = self._bert.device
                precision = "fp16" if self._bert.fp16 else "fp32"
                notes = self._bert.notes
            seconds = perf_counter() - start
            return [
                Prediction(
                    LABEL_NAMES[int(label)],
                    int(label),
                    model,
                    device,
                    precision,
                    scores[index],
                    "softmax_uncalibrated" if model == "bertimbau" else None,
                    seconds,
                    cold_start,
                    len(values),
                    notes,
                )
                for index, label in enumerate(ids)
            ]
