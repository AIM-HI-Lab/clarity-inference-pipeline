"""Tumor segmentation via external CLI (configurable binary and args)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import TumorSegmentationConfig
from .subprocess_util import run_subprocess_logged


def resolve_tumor_binary(config: TumorSegmentationConfig) -> str:
    if shutil.which(config.binary):
        return config.binary
    return config.binary


def _executable_for_mode(config: TumorSegmentationConfig) -> str:
    """``clarity-nnunet-predict`` is the v1 PyTorch 2.6+ shim; v2 still uses ``nnUNetv2_predict``."""

    b = resolve_tumor_binary(config)
    if config.mode == "nnunetv2" and b == "clarity-nnunet-predict":
        return "nnUNetv2_predict"
    return b


def run_tumor_segmentation(
    input_image: Path,
    output_path: Path,
    config: TumorSegmentationConfig,
    *,
    totalseg_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run the configured tumor segmentation command.

    ``totalseg_dir`` is included for pipelines that pass organ context; it is appended
    to the environment as ``CLARITY_TOTALSEG_DIR`` when set.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = _executable_for_mode(config)
    env = os.environ.copy()
    env.update(dict(config.env))
    if totalseg_dir is not None:
        env.setdefault("CLARITY_TOTALSEG_DIR", str(totalseg_dir))

    if config.mode == "nnunetv2":
        case_id = output_path.stem.replace(".nii", "")
        with tempfile.TemporaryDirectory(prefix="clarity_tumor_input_") as tmp_input, tempfile.TemporaryDirectory(
            prefix="clarity_tumor_output_"
        ) as tmp_output:
            tmp_input_path = Path(tmp_input)
            tmp_output_path = Path(tmp_output)
            staged_input = tmp_input_path / f"{case_id}_0000.nii.gz"
            shutil.copy2(input_image, staged_input)

            cmd: list[str] = [
                binary,
                "-i",
                str(tmp_input_path),
                "-o",
                str(tmp_output_path),
                "-d",
                config.dataset_id,
                "-c",
                config.configuration,
            ]
            for fold in config.folds:
                cmd.extend(["-f", str(fold)])
            cmd.extend(list(config.extra_args))
            proc = run_subprocess_logged(cmd, env=env, label="nnUNetv2_predict")
            produced = tmp_output_path / f"{case_id}.nii.gz"
            if proc.returncode == 0 and produced.exists():
                shutil.copy2(produced, output_path)
            return proc

    if config.mode == "nnunetv1":
        case_id = output_path.stem.replace(".nii", "")
        with tempfile.TemporaryDirectory(prefix="clarity_tumor_input_") as tmp_input, tempfile.TemporaryDirectory(
            prefix="clarity_tumor_output_"
        ) as tmp_output:
            tmp_input_path = Path(tmp_input)
            tmp_output_path = Path(tmp_output)
            staged_input = tmp_input_path / f"{case_id}_0000.nii.gz"
            shutil.copy2(input_image, staged_input)

            cmd = [
                binary,
                "-i",
                str(tmp_input_path),
                "-o",
                str(tmp_output_path),
                "-t",
                str(config.task_id),
                "-m",
                str(config.model),
            ]
            cmd.extend(list(config.extra_args))
            proc = run_subprocess_logged(cmd, env=env, label="nnUNet_predict")
            produced = tmp_output_path / f"{case_id}.nii.gz"
            if proc.returncode == 0 and produced.exists():
                shutil.copy2(produced, output_path)
            return proc

    cmd = [binary, "-i", str(input_image), "-o", str(output_path)]
    cmd.extend(list(config.extra_args))
    return run_subprocess_logged(cmd, env=env, label="tumor_seg")
