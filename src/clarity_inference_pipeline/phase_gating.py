"""Optional respiratory/cardiac phase gating hook (internal model package not required locally)."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any, Callable

from .config import PhaseGatingConfig


def _load_callable(spec: str) -> Callable[..., Any]:
    """Import ``module_path:callable_name``."""

    if ":" not in spec:
        raise ValueError(f"Expected 'module:callable', got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def run_phase_gating(
    nifti_dir: Path,
    output_dir: Path,
    config: PhaseGatingConfig,
) -> Path:
    """
    If ``config.enabled`` is false, returns ``nifti_dir`` unchanged.

    If ``entrypoint`` contains ``:``, it is treated as ``module:callable`` and invoked with
    keyword arguments from ``config.kwargs`` plus paths.

    Otherwise ``entrypoint`` is run as a subprocess with ``config.extra_args``.
    """

    if not config.enabled:
        return nifti_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    if not config.entrypoint:
        raise ValueError("phase_gating.enabled is true but entrypoint is empty")

    if ":" in config.entrypoint:
        fn = _load_callable(config.entrypoint)
        result = fn(
            nifti_dir=nifti_dir,
            output_dir=output_dir,
            extra_args=list(config.extra_args),
            **config.kwargs,
        )
        if isinstance(result, Path):
            return result
        return output_dir

    cmd = [config.entrypoint, str(nifti_dir), str(output_dir)]
    cmd.extend(list(config.extra_args))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Phase gating command failed ({proc.returncode}): {proc.stderr or proc.stdout}"
        )
    return output_dir
