"""Prova de retomada em CPU com BERT minúsculo e vocabulário local."""

from pathlib import Path

import pytest


def test_cpu_interruption_resume_matches_uninterrupted_weights(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from accelerate.utils import set_seed

    from reviewsense.phase3b_data import FullConfig
    from reviewsense.train_bertimbau import (
        checkpoint_files_valid,
        create_stable_trainer,
        training_arguments,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(2)
    vocab = {
        token: i
        for i, token in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "bom",
                "ruim",
                "regular",
            ]
        )
    }
    tokenizer = transformers.BertTokenizerFast(vocab=vocab, do_lower_case=False)
    records = [{**tokenizer(["bom", "ruim", "regular"][i % 3]), "labels": i % 3} for i in range(64)]
    config = FullConfig()

    def make(output, stop=False):
        set_seed(42)
        model = transformers.BertForSequenceClassification(
            transformers.BertConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_labels=3,
            )
        )

        class Interrupt(transformers.TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if stop and state.global_step == 2:
                    control.should_save = control.should_training_stop = True
                return control

        return create_stable_trainer(
            model=model,
            args=training_arguments(config, output, full=False, cpu=True),
            train_dataset=records,
            processing_class=tokenizer,
            data_collator=transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
            callbacks=[Interrupt()],
            optimizer_cls_and_kwargs=(torch.optim.AdamW, {"lr": 2e-5, "foreach": False}),
        )

    continuous = make(tmp_path / "control")
    continuous.train()
    expected = {
        name: value.detach().clone() for name, value in continuous.model.state_dict().items()
    }
    interrupted = make(tmp_path / "interrupted", stop=True)
    interrupted.train()
    checkpoint = Path(interrupted.args.output_dir) / "checkpoint-2"
    checkpoint_files_valid(checkpoint, fp16=False)
    resumed = make(tmp_path / "interrupted")
    resumed.train(resume_from_checkpoint=str(checkpoint))
    assert resumed.state.global_step == continuous.state.global_step == 4
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, expected[name]), f"Estado divergente após retomada: {name}"


def test_loss_is_normalized_over_the_actual_effective_batch():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from reviewsense.train_bertimbau import normalized_classification_loss

    torch.manual_seed(42)
    logits = torch.randn(12, 3, requires_grad=True)
    labels = torch.arange(12) % 3
    whole = torch.nn.functional.cross_entropy(logits, labels)
    expected_grad = torch.autograd.grad(whole, logits, retain_graph=True)[0]
    accumulated = sum(
        normalized_classification_loss(
            SimpleNamespace(logits=logits[start : start + 2]), labels[start : start + 2], 12
        )
        for start in range(0, 12, 2)
    )
    actual_grad = torch.autograd.grad(accumulated, logits)[0]
    assert torch.allclose(whole, accumulated)
    assert torch.allclose(expected_grad, actual_grad)


def test_amp_overflow_requires_skipped_update_and_finite_weights():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from reviewsense.train_bertimbau import verify_amp_recovery

    model = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(pending_amp_overflow=True)
    numeric = {
        "skipped_updates": 0,
        "consecutive_skips": 0,
        "nonfinite_weights": 0,
        "recovered_amp_overflows": 0,
    }
    with pytest.raises(FloatingPointError, match="sem proteção"):
        verify_amp_recovery(trainer, SimpleNamespace(step_was_skipped=False), model, numeric, 1)
    verify_amp_recovery(trainer, SimpleNamespace(step_was_skipped=True), model, numeric, 1)
    assert numeric["recovered_amp_overflows"] == 1
    assert not trainer.pending_amp_overflow
    verify_amp_recovery(trainer, SimpleNamespace(step_was_skipped=False), model, numeric, 2)
    assert numeric["consecutive_skips"] == 0


def test_amp_persistent_skips_and_corrupted_weights_are_rejected():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from reviewsense.train_bertimbau import verify_amp_recovery

    model = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(pending_amp_overflow=True)
    numeric = {
        "skipped_updates": 2,
        "consecutive_skips": 2,
        "nonfinite_weights": 0,
        "recovered_amp_overflows": 2,
    }
    with pytest.raises(FloatingPointError, match="persistentes"):
        verify_amp_recovery(trainer, SimpleNamespace(step_was_skipped=True), model, numeric, 3)
    model.weight.data.fill_(float("nan"))
    trainer.pending_amp_overflow = True
    numeric["consecutive_skips"] = 0
    with pytest.raises(FloatingPointError, match="Pesos não finitos"):
        verify_amp_recovery(trainer, SimpleNamespace(step_was_skipped=True), model, numeric, 4)


def test_trainer_does_not_unscale_twice_after_overflow(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from reviewsense.train_bertimbau import create_stable_trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(1, 1)
    args = transformers.TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[])
    trainer = create_stable_trainer(model=model, args=args)
    trainer.pending_amp_overflow = True

    def fail_if_called(*args, **kwargs):
        raise AssertionError("A norma não pode descalar novamente após detectar overflow.")

    monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", fail_if_called)
    assert trainer._get_grad_norm(model, grad_norm=None) is None
