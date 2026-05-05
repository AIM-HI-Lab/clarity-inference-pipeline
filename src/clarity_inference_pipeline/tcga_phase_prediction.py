"""TCGA-style contrast phase prediction (SWP v3 ResNet), separate from PNvsRN (SWP v5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import TcgaPhasePredictionConfig
from .tcga_phase_swp.inference_batch import PHASE_MAP, run_inference_on_batch

# Matches ccf-radiomics-pipelines kidney_tumor gating: class indices 1 and 2 only.
ALLOWED_TCGA_PHASE_CLASS_INDICES: frozenset[int] = frozenset({1, 2})


def _phase_label(class_index: int) -> str:
    if 0 <= class_index < len(PHASE_MAP):
        return PHASE_MAP[class_index]
    return f"class_{class_index}"


@dataclass(frozen=True)
class TcgaPhasePredictionResult:
    """Outcome of SWP v3 phase inference for one case."""

    prediction: int
    probabilities: list[float]
    metadata_block: dict[str, Any]


def run_tcga_phase_prediction(
    *,
    case_id: str,
    ct_nifti: Path,
    kidney_mask_nifti: Path,
    cases_root: Path,
    config: TcgaPhasePredictionConfig,
) -> TcgaPhasePredictionResult:
    """
    Run ensemble TCGA phase models (``*.pth`` under ``config.model_dir``).

    Uses the same preprocessing as ``swp.inference_v3`` / ccf-radiomics-pipelines ``tcga_phase``:
    weighted sampling, kidney mask, 4-class softmax.
    """

    model_dir = config.model_dir
    if model_dir is None:
        raise ValueError("tcga_phase_prediction.enabled requires model_dir")

    resolved = model_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"TCGA phase model directory not found: {resolved}")

    device = config.device
    predictions = run_inference_on_batch(
        img_pths=[ct_nifti.resolve()],
        mask_pths=[kidney_mask_nifti.resolve()],
        model_dir=resolved,
        n_classes=4,
        sampling_mode="weighted",
        use_seg=True,
        model_name="tcga_phase",
        output_pth=cases_root.resolve(),
        case_ids=[case_id],
        cache_pth=config.cache_root.expanduser().resolve() if config.cache_root else None,
        device=device,
    )
    if not predictions:
        raise RuntimeError(
            f"TCGA phase prediction produced no output for case {case_id!r}. "
            f"Check that {resolved} contains one or more ResNet50 .pth checkpoints."
        )
    pred = predictions.get(case_id)
    if pred is None:
        raise RuntimeError(f"TCGA phase prediction missing entry for case {case_id!r}.")

    pred_val = pred.get("prediction")
    if pred_val == "Error":
        err = pred.get("error", "unknown error")
        raise RuntimeError(
            f"TCGA phase model failed for case {case_id!r}: {err}. "
            "The series may be unsuitable for automated phase scoring."
        )

    if not isinstance(pred_val, int):
        raise RuntimeError(f"Unexpected TCGA phase prediction type for {case_id!r}: {type(pred_val)}")

    probs = pred.get("probabilities")
    if not isinstance(probs, list):
        probs = []

    meta = {
        "result": "success",
        "prediction": pred,
        "model_name": "tcga_phase",
    }
    return TcgaPhasePredictionResult(prediction=pred_val, probabilities=probs, metadata_block=meta)


def enforce_tcga_phase_allowed(
    result: TcgaPhasePredictionResult,
    *,
    case_id: str,
) -> None:
    """Raise if predicted class is not nephrographic (1) or arterial / corticomedullary (2)."""

    if result.prediction not in ALLOWED_TCGA_PHASE_CLASS_INDICES:
        label = _phase_label(result.prediction)
        allowed = ", ".join(_phase_label(i) for i in sorted(ALLOWED_TCGA_PHASE_CLASS_INDICES))
        raise RuntimeError(
            f"Predicted CT phase for case {case_id!r} is {label} (model class {result.prediction}). "
            "CLARITY requires nephrographic or corticomedullary (arterial) phase only "
            f"({allowed} in this model). Please upload a corticomedullary (30–60 s) or "
            "nephrographic (80–120 s) contrast series."
        )
