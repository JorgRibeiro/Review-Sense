"""Entrada do candidato Space; não lança servidor nem importa spaces ao ser importada."""

import os
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    with ExitStack() as stack:
        _launch(root, stack)


def _launch(root: Path, stack: ExitStack) -> None:
    try:
        from reviewsense.artifacts import read_json
        from reviewsense.model_resolution import provision_operational_approvals
        from reviewsense.zerogpu import ZeroGPUSettings, create_zerogpu_app

        payload = os.environ.pop("REVIEWSENSE_OPERATIONAL_APPROVAL_JSON", None)
        if payload is not None:
            if "REVIEWSENSE_APPROVAL_ROOT" in os.environ:
                raise ValueError("Origens de autorização ambíguas.")
            private = Path(stack.enter_context(TemporaryDirectory(prefix="reviewsense-private-")))
            approval_root = private / "records"
            provision_operational_approvals(
                payload, approval_root, read_json(root / "model_descriptor.json")
            )
        else:
            approval_root = Path(os.environ["REVIEWSENSE_APPROVAL_ROOT"])
        settings = ZeroGPUSettings(
            descriptor=root / "model_descriptor.json",
            runtime=root / "runtime.json",
            approval_root=approval_root,
            cache_root=Path(os.environ["REVIEWSENSE_CACHE_ROOT"]),
            repo_id=os.environ["REVIEWSENSE_MODEL_REPO_ID"],
            revision=os.environ["REVIEWSENSE_MODEL_REVISION"],
            space_id=os.environ["SPACE_ID"],
            provision_public_snapshot=os.environ.get("REVIEWSENSE_PROVISION_PUBLIC_SNAPSHOT")
            == "1",
        )
        demo = create_zerogpu_app(settings)
        from app.app import launch_options

        options = launch_options(public_network=True)
        options["blocked_paths"].extend(
            str(p.absolute())
            for p in (
                settings.cache_root,
                settings.approval_root,
                settings.descriptor,
            )
        )
    except Exception:
        raise SystemExit(
            "Inicialização bloqueada: modelo/runtime indisponível ou lançamento não autorizado."
        ) from None
    demo.launch(**options)


if __name__ == "__main__":
    main()
