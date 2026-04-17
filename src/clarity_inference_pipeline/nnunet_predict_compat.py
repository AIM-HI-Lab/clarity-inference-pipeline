"""nnU-Net v1 ``predict`` entrypoint compatible with PyTorch 2.6+ legacy checkpoints."""

from __future__ import annotations

import inspect
import multiprocessing
import sys
from collections.abc import Callable
from typing import Any


def _prefer_fork_multiprocessing_on_macos() -> None:
    """
    On macOS, Python 3.8+ defaults to ``spawn``. nnU-Net v1 preprocess workers receive
    ``trainer.preprocess_patient``, which pulls in unpicklable lambdas (e.g. from
    ``nnunet.utilities.nd_softmax``). ``fork`` avoids pickling those arguments.
    """

    if sys.platform != "darwin":
        return
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass


def _patch_torch_load_for_legacy_checkpoints() -> None:
    """Default ``weights_only=False`` so nnU-Net v1 ``.model`` pickles load like PyTorch < 2.6."""

    import torch

    orig: Callable[..., Any] = torch.load

    def patched(*args: Any, **kwargs: Any) -> Any:
        try:
            sig = inspect.signature(orig)
        except (TypeError, ValueError):
            return orig(*args, **kwargs)
        if "weights_only" in sig.parameters:
            kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    torch.load = patched  # type: ignore[method-assign]


def main() -> None:
    _prefer_fork_multiprocessing_on_macos()
    _patch_torch_load_for_legacy_checkpoints()
    try:
        from nnunet.inference.predict_simple import main as nnunet_main
    except ImportError as err:
        print(
            "nnunet is not installed. Install nnU-Net v1 (nnunet package) for tumor segmentation.",
            file=sys.stderr,
        )
        raise SystemExit(1) from err

    raise SystemExit(nnunet_main())


if __name__ == "__main__":
    main()
