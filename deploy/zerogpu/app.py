"""Entrada do candidato Space; não lança servidor nem importa spaces ao ser importada."""

import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    try:
        from reviewsense.zerogpu import ZeroGPUSettings, create_zerogpu_app

        settings = ZeroGPUSettings(
            descriptor=root / "model_descriptor.json",
            runtime=root / "runtime.json",
            approval_root=Path(os.environ["REVIEWSENSE_APPROVAL_ROOT"]),
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
