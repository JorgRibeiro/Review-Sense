"""Execute pinned SDK functions with fake transport/worker boundaries, never CUDA init."""

import hashlib
import inspect
import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from datetime import timedelta
from functools import partial, wraps
from pathlib import Path
from queue import Queue
from threading import Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/spaces_0_51_3"


def load_functions(module, filename):
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    source = (FIXTURES / filename).read_text()
    assert (
        hashlib.sha256(source.encode()).hexdigest()
        == provenance["excerpts"][filename]["fixture_sha256"]
    )
    # Only annotation evaluation changes; upstream function bodies remain exact.
    exec(
        compile("from __future__ import annotations\n" + source, filename, "exec"), module.__dict__
    )


def upstream_sdk(monkeypatch):
    config = ModuleType("spaces.config")
    load_functions(config, "spaces__config.py.txt")
    config.Config = SimpleNamespace(zero_gpu=True)
    client = ModuleType("spaces.zero.client")
    state = {"allocated": False, "order": []}

    def schedule(**kwargs):
        state["order"].append(("schedule", kwargs["task_id"]))
        state["allocated"] = True
        return SimpleNamespace(
            allowToken="synthetic", nvidiaIndex=0, nvidiaUUID="synthetic", idle=True
        )

    def release(*args, **kwargs):
        state["allocated"] = False
        state["order"].append(("release", None))

    client.schedule = Mock(side_effect=schedule)
    client.release = Mock(side_effect=release)
    client.get_duration_seconds = lambda *a: 10
    client.duration_warning = Mock()
    wrappers = ModuleType("spaces.zero.wrappers")

    class OkResult:
        def __init__(self, value):
            self.value = value

    class ExceptionResult:
        pass

    class GradioQueueEvent:
        pass

    class Worker:
        def __init__(self, target, *args):
            # Replace process execution only; the scheduler wrapper itself is upstream code.
            freevars = dict(
                zip(
                    target.__code__.co_freevars,
                    (cell.cell_contents for cell in target.__closure__),
                    strict=True,
                )
            )
            task = freevars["task"]
            self.res_queue = Queue()
            self.process = SimpleNamespace(is_alive=lambda: True)

            def put(payload):
                assert state["allocated"], "inference without allocation"
                (args, kwargs), _ = payload
                state["order"].append(("worker", id(task)))
                self.res_queue.put(OkResult(task(*args, **kwargs)))

            self.arg_queue = SimpleNamespace(put=put, empty=lambda: True)

    wrappers.__dict__.update(
        wraps=wraps,
        partial=partial,
        timedelta=timedelta,
        forked=False,
        client=client,
        gradio_request_var=lambda: ContextVar("synthetic-request", default=None),
        Worker=Worker,
        OkResult=OkResult,
        ExceptionResult=ExceptionResult,
        GradioQueueEvent=GradioQueueEvent,
        ThreadEvent=Event,
        ThreadPoolExecutor=ThreadPoolExecutor,
        copy_context=copy_context,
        GradioPartialContext=SimpleNamespace(get=lambda: None),
    )
    load_functions(wrappers, "spaces__zero__wrappers.py.txt")
    wrappers.generator_function_wrapper = Mock(side_effect=AssertionError("not a generator"))
    decorator = ModuleType("spaces.zero.decorator")
    decorator.__dict__.update(
        Config=config.Config,
        inspect=inspect,
        warnings=warnings,
        partial=partial,
        decorated_cache={},
    )
    load_functions(decorator, "spaces__zero__decorator.py.txt")
    package = ModuleType("spaces")
    package.__path__ = []
    package.GPU = decorator.GPU
    zero = ModuleType("spaces.zero")
    zero.__path__ = []
    zero.client = client
    zero.wrappers = wrappers
    for name, module in {
        "spaces": package,
        "spaces.config": config,
        "spaces.zero": zero,
        "spaces.zero.client": client,
        "spaces.zero.wrappers": wrappers,
        "spaces.zero.decorator": decorator,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(
        package=package, config=config, decorator=decorator, client=client, state=state
    )
