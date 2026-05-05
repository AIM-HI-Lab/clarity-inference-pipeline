"""TCGA-style CT phase gate (corticomedullary / nephrographic), matching ccf-radiomics-pipelines."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_ALLOWED_PHASE_PREDICTIONS = frozenset({1, 2})


@contextmanager
def _temporary_swp_device(device: str | None) -> Iterator[None]:
    if not device:
        yield
        return
    key = "SWP_DEVICE"
    previous = os.environ.get(key)
    os.environ[key] = device
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def assert_tcga_phase_allowed(
    *,
    image_nii: Path,
    kidney_mask_nii: Path,
    model_parent_dir: Path,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Run the same ``tcga_phase`` SWP checkpoint layout used in ccf-radiomics-pipelines
    (``model_parent_dir / "tcga_phase"`` with ``run_inference_on_batch``).

    Class indices **1** and **2** are treated as allowable (corticomedullary / nephrographic),
    consistent with ``kidney_tumor.run_kidney_segmentator*`` phase checks.

    Returns a small metadata dict for logging (prediction value, paths used).
    """

    try:
        from swp.inference_v3 import run_inference_on_batch  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "TCGA phase gating requires the `swp` package (``swp.inference_v3.run_inference_on_batch``) "
            "installed in the same environment as the worker. This matches ccf-radiomics-pipelines."
        ) from e

    model_dir = model_parent_dir / "tcga_phase"
    if not model_dir.is_dir():
        raise RuntimeError(
            f"TCGA phase model directory not found: {model_dir} "
            f"(expected a ``tcga_phase`` folder under model parent {model_parent_dir})."
        )

    image_nii = image_nii.resolve()
    kidney_mask_nii = kidney_mask_nii.resolve()
    if not image_nii.is_file() or not kidney_mask_nii.is_file():
        raise RuntimeError("TCGA phase gating: image or kidney mask NIfTI is missing.")

    with tempfile.TemporaryDirectory(prefix="clarity-tcga-phase-") as tmp:
        case_root = Path(tmp) / "case"
        case_root.mkdir(parents=True, exist_ok=True)
        tseg = case_root / "total_seg"
        tseg.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_nii, case_root / "image.nii.gz")
        shutil.copy2(kidney_mask_nii, tseg / "kidney_binary_mask.nii.gz")

        out_batch = Path(tmp) / "batch_out"
        out_batch.mkdir(parents=True, exist_ok=True)

        map_cuda = {"cuda": "cuda", "gpu": "cuda", "cpu": "cpu"}
        swp_dev = map_cuda.get((device or "").strip().lower(), device)

        with _temporary_swp_device(swp_dev):
            run_inference_on_batch(
                img_pths=[case_root / "image.nii.gz"],
                mask_pths=[tseg / "kidney_binary_mask.nii.gz"],
                model_dir=model_dir,
                n_classes=4,
                sampling_mode="weighted",
                use_seg=True,
                output_pth=out_batch,
                model_name="tcga_phase",
            )

        pred_path = case_root / "tcga_phase_prediction.json"
        if not pred_path.is_file():
            raise RuntimeError(
                "TCGA phase model did not write tcga_phase_prediction.json; check swp install and model weights."
            )
        pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
        pred_val = pred_payload.get("prediction")
        if pred_val == "Error" or pred_val is None:
            raise RuntimeError(
                "The CT series found appears to be unenhanced or an incorrect phase for renal tumor "
                "scoring. CLARITY requires a corticomedullary (30–60 s post-contrast) or nephrographic "
                f"(80–120 s post-contrast) phase. Phase model error: {pred_payload!r}"
            )
        try:
            pred_int = int(pred_val)
        except (TypeError, ValueError):
            pred_int = -1
        if pred_int not in _ALLOWED_PHASE_PREDICTIONS:
            raise RuntimeError(
                "The CT series found appears to be unenhanced or an incorrect phase for renal tumor "
                "scoring. CLARITY requires a corticomedullary (30–60 s post-contrast) or nephrographic "
                f"(80–120 s post-contrast) phase (tcga_phase prediction was {pred_val!r}, expected 1 or 2)."
            )

        return {
            "tcga_phase_prediction": pred_val,
            "model_dir": str(model_dir),
            "image": str(image_nii),
            "kidney_mask": str(kidney_mask_nii),
            "swp_device": device,
        }
