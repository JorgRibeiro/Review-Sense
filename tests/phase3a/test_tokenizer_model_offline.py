"""Integração mínima CPU com vocabulário e BERT minúsculos criados localmente."""

import pandas as pd
import pytest

from reviewsense.transformer_data import label_mappings, tokenize_reviews


def test_real_tokenizer_dynamic_padding_and_tiny_model_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = {
        token: index
        for index, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "Não",
                "não",
                "Ótimo",
                "ótimo",
                "gostei",
                "produto",
                "!",
            ]
        )
    }
    tokenizer = transformers.BertTokenizerFast(
        vocab=vocab, do_lower_case=False, strip_accents=False
    )
    assert tokenizer("Não")["input_ids"] != tokenizer("não")["input_ids"]
    assert tokenizer("Ótimo")["input_ids"] != tokenizer("Otimo")["input_ids"]
    frame = pd.DataFrame({"review_text": ["Não gostei!", "Ótimo"], "label": [0, 2]})
    records = tokenize_reviews(frame, tokenizer, 128)
    batch = transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)(records)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"].shape[1] < 128
    assert batch["labels"].tolist() == [0, 2]
    id2label, label2id = label_mappings()
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    model = transformers.AutoModelForSequenceClassification.from_config(config)
    output = model(**batch)
    assert output.logits.shape == (2, 3)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.device.type == "cpu"
    model.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer.save_pretrained(tmp_path)
    reloaded = transformers.AutoModelForSequenceClassification.from_pretrained(
        tmp_path, local_files_only=True
    )
    reloaded_tokenizer = transformers.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert reloaded.config.id2label == id2label
    assert reloaded.config.label2id == label2id
    assert tokenize_reviews(frame, reloaded_tokenizer, 128) == records
