"""TotalSegmentator CLI wrapper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import TotalSegmentatorConfig
from .subprocess_util import run_subprocess_logged


def resolve_totalsegmentator_binary(config: TotalSegmentatorConfig) -> str:
    """Return executable path or name; prefers explicit ``binary`` if found on PATH."""

    if shutil.which(config.binary):
        return config.binary
    return config.binary


def run_totalsegmentator(
    input_image: Path,
    output_dir: Path,
    config: TotalSegmentatorConfig,
) -> subprocess.CompletedProcess[str]:
    """
    Invoke TotalSegmentator on ``input_image``, writing segmentations under ``output_dir``.

    Default CLI shape matches common TotalSegmentator usage: ``-i``, ``-o``, optional ``--task``.
    Adjust flags to match your installed version.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    binary = resolve_totalsegmentator_binary(config)
    cmd: list[str] = [binary, "-i", str(input_image), "-o", str(output_dir)]
    if config.device:
        # TotalSegmentator: --device gpu|cpu|mps (we map cuda → gpu).
        ts_dev = "gpu" if config.device == "cuda" else config.device
        if ts_dev in ("gpu", "cpu", "mps"):
            cmd.extend(["--device", ts_dev])
    if config.task:
        cmd.extend(["--task", config.task])
    cmd.extend(list(config.extra_args))
    proc = run_subprocess_logged(cmd, label="TotalSegmentator")
    if proc.returncode == 0:
        _write_kidney_binary_mask(output_dir)
    return proc


def _write_kidney_binary_mask(output_dir: Path) -> None:
    """Create `kidney_binary_mask.nii.gz` from TotalSegmentator kidney outputs."""

    import nibabel as nib
    import numpy as np

    kidney_sources = [
        output_dir / "kidney_left.nii.gz",
        output_dir / "kidney_right.nii.gz",
    ]
    existing = [p for p in kidney_sources if p.exists()]
    if not existing:
        return

    ref = nib.load(str(existing[0]))
    binary = np.zeros(ref.shape, dtype=np.uint8)
    for src in existing:
        data = np.asarray(nib.load(str(src)).get_fdata())
        binary[data > 0] = 1

    out = output_dir / "kidney_binary_mask.nii.gz"
    nib.save(nib.Nifti1Image(binary, ref.affine, ref.header), str(out))
