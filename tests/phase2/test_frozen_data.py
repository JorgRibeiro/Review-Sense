"""Guardas da Fase 2: somente treino/validação, ancorados na Fase 1."""

import json

import pandas as pd
import pytest

from reviewsense.frozen_data import load_training_data, validate_training_pair


def test_load_training_data_never_opens_test(frozen_training, monkeypatch):
    directory, reference, expected = frozen_training
    read_csv = pd.read_csv
    reads = []

    def guarded_read(path, *args, **kwargs):
        assert path.name != "test.csv", "O teste final não pode ser aberto."
        reads.append(path.name)
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read)
    # O carregador deve funcionar até sem o arquivo de teste estar disponível.
    (directory / "test.csv").unlink()
    train, val, provenance = load_training_data(directory, reference)
    pd.testing.assert_frame_equal(train, expected["train"])
    pd.testing.assert_frame_equal(val, expected["val"])
    assert reads == ["train.csv", "val.csv"]
    assert provenance["test_csv_opened"] is False
    assert set(provenance["splits"]) == {"train", "val"}
    assert "sample_ids" not in json.dumps(provenance)


@pytest.mark.parametrize("tamper", ["text", "label", "manifest", "reference"])
def test_frozen_data_rejects_changes(frozen_training, tamper):
    directory, reference, splits = frozen_training
    if tamper in ("text", "label"):
        changed = splits["train"].copy()
        if tamper == "text":
            changed.loc[0, "review_text"] = "modificado"
        else:
            changed.loc[0, "label"] = (changed.loc[0, "label"] + 1) % 3
        changed.to_csv(directory / "train.csv", index=False)
    else:
        path = reference if tamper == "reference" else directory / "manifest.json"
        content = json.loads(path.read_text())
        content["splits"]["train"]["fingerprint"] = "wrong"
        path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_data(directory, reference)


def test_pair_rejects_overlap(frozen_training):
    _, _, splits = frozen_training
    overlapping = pd.concat([splits["val"], splits["train"].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="Vazamento"):
        validate_training_pair(splits["train"], overlapping)
