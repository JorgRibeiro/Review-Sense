"""ZeroGPU: controle de ciclo de vida simulado; nunca valida GPU física compartilhada."""

import copy
import json
import subprocess
import sys
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_release_readiness import approve_synthetic

from reviewsense.artifacts import PACKAGE_FILES, file_inventory
from reviewsense.model_resolution import LocalPackage
from reviewsense.predict import ArtifactError, InferenceError, InferenceService, InputError
from reviewsense.space_bundle import BUNDLE_FILES, prepare_bundle, validate_bundle
from reviewsense.zerogpu import (
    GPU_DURATION_SECONDS,
    ZeroGPUInferenceService,
    ZeroGPUSettings,
    create_zerogpu_app,
    gpu_callback_transform,
    import_zero_runtime,
    verify_runtime_versions,
)
from scripts.verify_phase4b_package import no_network

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def simulated_runtime(monkeypatch):
    import reviewsense.zerogpu as zero

    events = []
    registry = {}
    original_loader = zero.load_bert_cpu

    def loader(directory):
        events.append("load_cpu")
        model, tokenizer = original_loader(directory)
        original_to = model.to

        def simulated_to(device):
            assert device == "cuda"
            events.append("to_emulated_cuda")
            return original_to("cpu")

        model.to = simulated_to
        return model, tokenizer

    def gpu(**kwargs):
        events.append(("decorate", kwargs))

        def decorate(callback):
            if callback in registry:
                return registry[callback]

            @wraps(callback)
            def allocated(*args):
                events.append("allocate_simulated")
                return callback(*args)

            allocated.zerogpu = True
            registry[callback] = allocated
            registry[allocated] = allocated
            return allocated

        return decorate

    monkeypatch.setattr(zero, "load_bert_cpu", loader)
    # Simula só controle de fluxo: toda matemática continua em CPU no modelo minúsculo.
    original_logits = InferenceService._bert_logits
    monkeypatch.setattr(
        ZeroGPUInferenceService,
        "_bert_logits",
        staticmethod(
            lambda backend, texts: original_logits(
                replace(backend, device="cpu", fp16=False), texts
            )
        ),
    )
    monkeypatch.setattr(zero, "verify_runtime_versions", lambda path: {})
    monkeypatch.setattr(
        ZeroGPUInferenceService, "_load_bert", Mock(side_effect=AssertionError("no lazy load"))
    )
    monkeypatch.setattr(
        ZeroGPUInferenceService, "_recover_cpu", Mock(side_effect=AssertionError("no fallback"))
    )
    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace(GPU=gpu))
    monkeypatch.setitem(
        sys.modules, "spaces.zero.decorator", SimpleNamespace(decorated_cache=registry)
    )
    monkeypatch.setitem(
        sys.modules, "spaces.config", SimpleNamespace(Config=SimpleNamespace(zero_gpu=True))
    )
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    return events


def make_settings(candidate, approvals, cache):
    return ZeroGPUSettings(
        candidate["descriptor_path"],
        ROOT / "deploy/zerogpu/runtime.json",
        approvals,
        cache,
        "example/model",
        "a" * 40,
        "example/space",
    )


def approve_zero(candidate, tmp_path):
    approvals = approve_synthetic(candidate, tmp_path)
    p = approvals / "human.json"
    human = json.loads(p.read_text())
    human["space_id"] = "example/space"
    human["actions"].append("host_zerogpu")
    p.write_text(json.dumps(human))
    return approvals


def test_local_imports_do_not_require_spaces_or_training_modules():
    script = """import sys, builtins
sys.path.insert(0, sys.argv[1])
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'spaces','pandas','sklearn','threadpoolctl'} or name in {
        'reviewsense.data','reviewsense.baseline','reviewsense.model_packaging',
        'reviewsense.train_transformer','reviewsense.final_evaluation'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from reviewsense.predict import InferenceService
from reviewsense.zerogpu import ZeroGPUSettings
assert InferenceService(device='cpu').loaded_models == ()
assert 'spaces' not in sys.modules and 'torch' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(ROOT / "src")], check=True, capture_output=True
    )


@pytest.mark.parametrize("revision", ["main", "v1", "abc", "A" * 40])
def test_zero_settings_reject_mobile_or_invalid_revision(candidate, tmp_path, revision):
    with pytest.raises(ValueError, match="imutável"):
        replace(make_settings(candidate, tmp_path, tmp_path), revision=revision)


def test_pending_real_descriptor_blocks_before_spaces_download_or_load(tmp_path, monkeypatch):
    import reviewsense.zerogpu as zero

    spy = Mock(side_effect=AssertionError("must stop before runtime"))
    monkeypatch.setattr(zero, "import_zero_runtime", spy)
    monkeypatch.setattr(zero, "load_bert_cpu", spy)
    settings = ZeroGPUSettings(
        ROOT / "publishing/inference_descriptor.json",
        ROOT / "deploy/zerogpu/runtime.json",
        tmp_path,
        tmp_path,
        "JorgeRibeiru/reviewsense-ptbr-bertimbau",
        "a" * 40,
        "JorgeRibeiru/reviewsense-ptbr",
        True,
    )
    with no_network() as network, pytest.raises(PermissionError):
        create_zerogpu_app(settings)
    assert network["blocked_connections"] == 0
    spy.assert_not_called()


def test_cpu_only_or_normal_cuda_is_not_zero_runtime(monkeypatch):
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    with pytest.raises(RuntimeError, match="runtime ZeroGPU"):
        import_zero_runtime()


def test_spaces_must_precede_torch(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    monkeypatch.delitem(sys.modules, "spaces", raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(RuntimeError, match="antes"):
        import_zero_runtime()


def test_preload_once_no_repeated_transfer_and_same_synthetic_scores(candidate, simulated_runtime):
    source = LocalPackage(
        candidate["package"], candidate["descriptor_path"], candidate["package"].parent
    )
    before = file_inventory(candidate["package"], PACKAGE_FILES)
    zero = ZeroGPUInferenceService(source)
    with pytest.raises(InferenceError, match="inicialização"):
        zero.predict("O produto bom")
    zero.prepare()
    zero.prepare()
    values = ["O produto bom", "Não bom"]
    observed = zero.predict_batch(values)
    expected = InferenceService(device="cpu", model_source=source).predict_batch(values)
    for a, b in zip(observed, expected, strict=True):
        assert a.class_id == b.class_id
        np.testing.assert_allclose(
            [s.value for s in a.scores], [s.value for s in b.scores], atol=1e-6
        )
    zero.predict(values[0])
    assert simulated_runtime.count("load_cpu") == simulated_runtime.count("to_emulated_cuda") == 1
    assert file_inventory(candidate["package"], PACKAGE_FILES) == before
    with pytest.raises(InputError):
        zero.predict("O produto bom", model="svm")


@pytest.mark.parametrize("mode", ["missing", "hash"])
def test_invalid_weights_never_load_or_transfer(candidate, simulated_runtime, mode):
    p = candidate["package"] / "model.safetensors"
    if mode == "missing":
        p.unlink()
    else:
        p.write_bytes(b"synthetic corruption")
    service = ZeroGPUInferenceService(
        LocalPackage(
            candidate["package"], candidate["descriptor_path"], candidate["package"].parent
        )
    )
    with pytest.raises(ArtifactError):
        service.prepare()
    assert "load_cpu" not in simulated_runtime and "to_emulated_cuda" not in simulated_runtime


def test_callback_zero_decorator_limits_and_sanitized_scheduler_error(monkeypatch):
    calls = []
    registry = {}
    monkeypatch.setitem(
        sys.modules, "spaces.zero.decorator", SimpleNamespace(decorated_cache=registry)
    )

    def gpu(**kwargs):
        assert kwargs == {"duration": GPU_DURATION_SECONDS}

        def decorate(fn):
            if fn in registry:
                return registry[fn]

            @wraps(fn)
            def allocate(*args):
                calls.append(args)
                raise RuntimeError("sensitive-text /home/private")

            allocate.zerogpu = True
            registry[fn] = allocate
            registry[allocate] = allocate
            return allocate

        return decorate

    callback = gpu_callback_transform(SimpleNamespace(GPU=gpu))(lambda *a: None)
    assert callback.zerogpu
    assert "Escreva" in callback("", "bertimbau")
    assert "8.000" in callback("a" * 8001, "bertimbau")
    assert "Selecione" in callback("produto", "svm")
    assert not calls
    result = callback("sensitive-text", "bertimbau")
    assert len(calls) == 1 and "GPU temporariamente indisponível" in result
    assert "sensitive-text" not in result and "/home/" not in result


def test_gradio_binds_shared_decorated_callback_after_preload(
    candidate,
    tmp_path,
    hub_cache,
    simulated_runtime,
    monkeypatch,
):
    import huggingface_hub

    approvals = approve_zero(candidate, tmp_path)
    cache = hub_cache(candidate["package"])
    spy = Mock(side_effect=AssertionError("no network/API"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", spy)
    with no_network() as network:
        demo = create_zerogpu_app(make_settings(candidate, approvals, cache["root"]))
        try:
            radio = next(c for c in demo.config["components"] if c["type"] == "radio")
            assert len(radio["props"]["choices"]) == 1
            text = next(c for c in demo.config["components"] if c["type"] == "textbox")
            assert text["props"]["max_length"] == 8000 and demo._queue.max_size == 8
            fns = [fn for fn in demo.fns.values() if fn.fn and getattr(fn.fn, "zerogpu", False)]
            assert len(fns) == 2 and fns[0].fn is fns[1].fn
            assert all(fn.concurrency_id == "inference" and fn.concurrency_limit == 1 for fn in fns)
            rendered = fns[0].fn("O produto bom", "bertimbau")
            assert "Scores normalizados" in rendered and "não calibrados" in rendered
            assert "allocate_simulated" in simulated_runtime
            assert simulated_runtime.index("load_cpu") < next(
                i for i, e in enumerate(simulated_runtime) if isinstance(e, tuple)
            )
        finally:
            demo.close()
    assert network["blocked_connections"] == 0
    spy.assert_not_called()


def test_space_specific_authorization_required(candidate, tmp_path):
    approvals = approve_synthetic(candidate, tmp_path)
    with pytest.raises(PermissionError, match="ZeroGPU"):
        make_settings(candidate, approvals, tmp_path).authorize()


def test_future_provision_pins_sha_allowlist_and_no_token(
    candidate, tmp_path, hub_cache, monkeypatch
):
    import huggingface_hub

    approvals = approve_zero(candidate, tmp_path)
    cache = hub_cache(candidate["package"])
    spy = Mock()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", spy)
    settings = replace(
        make_settings(candidate, approvals, cache["root"]), provision_public_snapshot=True
    )
    source = settings.source()
    assert source.resolve().is_relative_to(cache["root"])
    assert spy.call_count == 1
    kwargs = spy.call_args.kwargs
    assert kwargs["token"] is False and kwargs["revision"] == "a" * 40
    assert set(kwargs["allow_patterns"]) == set(PACKAGE_FILES)
    assert kwargs["repo_id"] == "example/model"


@pytest.mark.parametrize("mismatch", ["python", "torch", "cpu_build"])
def test_runtime_matrix_rejects_unverified_versions(monkeypatch, mismatch):
    path = ROOT / "deploy/zerogpu/runtime.json"
    expected = json.loads(path.read_text())
    actual = copy.deepcopy(expected["packages"])
    python = expected["python_version"]
    monkeypatch.setattr("reviewsense.zerogpu.platform.python_version", lambda: python)
    monkeypatch.setattr("reviewsense.zerogpu.importlib.metadata.version", lambda name: actual[name])
    assert verify_runtime_versions(path) == actual
    if mismatch == "python":
        python = "3.11.14"
    elif mismatch == "torch":
        actual["torch"] = "2.14.1+cu132"
    else:
        actual["torch"] = "2.13.0+cpu"
    with pytest.raises(RuntimeError, match="Runtime"):
        verify_runtime_versions(path)


def test_bundle_has_only_runtime_code_and_frozen_descriptor(tmp_path):
    directory = tmp_path / "space"
    inventory = prepare_bundle(ROOT, directory)
    validate_bundle(directory, inventory)
    assert set(inventory) == set(BUNDLE_FILES)
    assert not any(
        n.endswith((".safetensors", ".csv", ".joblib", ".pt", ".bin")) for n in inventory
    )
    assert not any(n.split("/")[0] in {"data", "models", "reports", "outputs"} for n in inventory)
    assert all(
        "train_" not in n and "baseline" not in n and "model_packaging" not in n for n in inventory
    )
    descriptor = json.loads((directory / "model_descriptor.json").read_text())
    assert (
        descriptor["upload_authorized"] is False
        and descriptor["license_status"] == "REQUIRES_REVIEW"
    )
    assert "preload_from_hub" not in (directory / "README.md").read_text().split("---")[1]
    with pytest.raises(FileExistsError):
        prepare_bundle(ROOT, directory)


@pytest.mark.parametrize("failure", ["extra", "symlink", "secret", "hash"])
def test_bundle_rejects_unsafe_or_modified_content(tmp_path, failure):
    directory = tmp_path / "space"
    inventory = prepare_bundle(ROOT, directory)
    p = directory / "app.py"
    if failure == "extra":
        (directory / "test.csv").write_text("synthetic")
    elif failure == "symlink":
        p.unlink()
        p.symlink_to(ROOT / "deploy/zerogpu/app.py")
    elif failure == "secret":
        p.write_text(p.read_text() + "\n# hf_" + "X" * 30)
        from reviewsense.artifacts import file_sha256

        inventory["app.py"] = file_sha256(p)
    else:
        p.write_text("altered")
    with pytest.raises(ValueError):
        validate_bundle(directory, inventory)


def test_built_entry_independent_offline_and_fails_safely_with_pending_rights(tmp_path):
    directory = tmp_path / "space"
    prepare_bundle(ROOT, directory)
    code = """import sys,runpy,os
from pathlib import Path
bundle=Path(sys.argv[1]); original=Path(sys.argv[2])
sys.path[:]=[p for p in sys.path if p not in {str(original),str(original/'src')}]
blocked=[original/p for p in ('src','app','data','models','reports','publishing','deploy')]
state={'network':0,'original_opens':0}
def guard(event,args):
    if event=='socket.connect':
        state['network']+=1
        raise AssertionError('network forbidden')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(root) for root in blocked):
            state['original_opens']+=1
            raise AssertionError('original repository forbidden')
sys.addaudithook(guard)
os.environ.update(REVIEWSENSE_MODEL_REPO_ID='JorgeRibeiru/reviewsense-ptbr-bertimbau',
 REVIEWSENSE_MODEL_REVISION='a'*40,SPACE_ID='JorgeRibeiru/reviewsense-ptbr',
 REVIEWSENSE_APPROVAL_ROOT=str(bundle/'absent-approvals'),REVIEWSENSE_CACHE_ROOT=str(bundle/'cache'))
try:runpy.run_path(str(bundle/'app.py'),run_name='__main__')
except SystemExit as error:
    assert 'Inicialização bloqueada' in str(error) and '/home/' not in str(error)
else:raise AssertionError('launch should fail')
for name in ['spaces','torch','reviewsense.data','reviewsense.baseline']:
    assert name not in sys.modules
assert state=={'network':0,'original_opens':0}
"""
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(directory), str(ROOT)],
        check=True,
        capture_output=True,
    )


def test_zero_marker_alone_cannot_enable_inactive_spaces_emulation(monkeypatch):
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "spaces.config", SimpleNamespace(Config=SimpleNamespace(zero_gpu=False))
    )
    with pytest.raises(RuntimeError, match="emulação"):
        import_zero_runtime()


@pytest.mark.parametrize("failure", ["missing", "hash", "cuda_init"])
def test_startup_failure_never_constructs_interface(
    candidate,
    tmp_path,
    hub_cache,
    simulated_runtime,
    monkeypatch,
    failure,
):
    import app.app as app_module
    import reviewsense.zerogpu as zero

    approvals = approve_zero(candidate, tmp_path)
    cache = hub_cache(candidate["package"])
    if failure == "missing":
        cache["files"]["model.safetensors"].unlink()
    elif failure == "hash":
        cache["files"]["model.safetensors"].write_bytes(b"corrupted synthetic weights")
    else:

        def fail_loader(*args):
            raise RuntimeError("/home/private cuda unavailable")

        monkeypatch.setattr(zero, "load_bert_cpu", fail_loader)
    spy = Mock(side_effect=AssertionError("UI must not be built"))
    monkeypatch.setattr(app_module, "create_app", spy)
    with no_network(), pytest.raises(ArtifactError) as caught:
        create_zerogpu_app(make_settings(candidate, approvals, cache["root"]))
    assert "/home/" not in str(caught.value)
    spy.assert_not_called()
