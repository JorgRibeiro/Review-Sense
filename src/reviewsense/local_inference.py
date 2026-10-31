"""Compatibilidade local dos artefatos históricos; excluída do bundle BERT-only."""

import numpy as np

MODEL_PREFIX = "models/bertimbau/phase3b-full-v3/best_model"


def repository_paths(root):
    return dict(
        bert_dir=root / MODEL_PREFIX,
        svm_dir=root / "models/baseline/phase2-v1",
        bert_record=root / "reports/bertimbau_phase3b_metrics.json",
        svm_record=root / "reports/baseline_validation.json",
        final_record=root / "reports/phase3c_final_metrics.json",
    )


def load_svm(paths):
    from reviewsense.baseline import load_experiment
    from reviewsense.predict import ArtifactError, _check_hash, _record

    record = _record(paths.svm_record)
    final = _record(paths.final_record)
    try:
        expected = record["model_artifact"]["sha256"]
        if final["inventory"]["models/baseline/phase2-v1/selected_model.joblib"] != expected:
            raise ArtifactError("O baseline diverge dos registros de integridade.")
        _check_hash(paths.svm_dir / "selected_model.joblib", expected)
        model, local_record = load_experiment(paths.svm_dir)
        if local_record["model_artifact"]["sha256"] != expected:
            raise ArtifactError("O baseline diverge do registro da Fase 2.")
        if not np.array_equal(model.classes_, [0, 1, 2]):
            raise ArtifactError("As classes do baseline devem ser negativo, neutro e positivo.")
        return model
    except ArtifactError:
        raise
    except Exception:
        raise ArtifactError("Não foi possível carregar o baseline local verificado.") from None


def verify_bert(paths):
    from reviewsense.predict import ArtifactError, _check_hash, _record

    record = _record(paths.bert_record)
    final = _record(paths.final_record)
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
            _check_hash(paths.bert_dir / name, expected[f"{MODEL_PREFIX}/{name}"])
    except ArtifactError:
        raise
    except (KeyError, TypeError):
        raise ArtifactError("Registro de integridade do BERTimbau incompatível.") from None
