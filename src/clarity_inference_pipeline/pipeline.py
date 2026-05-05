"""End-to-end orchestration: DICOM -> NIfTI -> segmentation -> CLARITY (SWP) prediction."""

from __future__ import annotations

import subprocess
import sys
import warnings
from shutil import copy2
from pathlib import Path
from typing import Any

from . import __version__
from .config import DicomPaths, PipelineConfig
from .clarity import run_clarity_inference
from .dicom import discover_series_roots, run_dcm2niix, select_best_series, stage_series_for_conversion
from .dicom_sitk import convert_staged_dicom_to_nifti
from .mask_adaptation import adapt_masks, swp_segmentation_has_tumor_voxels
from .nifti_ct import select_primary_ct_nifti, write_nnunet_compatible_nifti
from .phase_gating import run_phase_gating
from .tcga_phase_prediction import (
    enforce_tcga_phase_allowed,
    run_tcga_phase_prediction,
)
from .totalsegmentator import run_totalsegmentator
from .tumor_segmentation import run_tumor_segmentation
from .workspace import (
    default_workspace_layout,
    ensure_case_workspace,
    ensure_workspace_dirs,
    write_case_metadata,
    write_manifest,
    write_swp_manifest,
)


def _pipeline_log(message: str) -> None:
    print(f"[clarity-pipeline] {message}", file=sys.stderr, flush=True)


def find_primary_nifti(nifti_dir: Path) -> Path:
    """Pick the primary CT NIfTI under ``nifti_dir`` (see :func:`select_primary_ct_nifti`)."""

    return select_primary_ct_nifti(nifti_dir)


def _stage_ct_nifti_for_downstream(raw_nifti: Path, case_paths: dict[str, Path]) -> None:
    """Write 3D float CT to ``imaging`` / ``image_compat`` for TotalSegmentator and nnU-Net."""

    write_nnunet_compatible_nifti(raw_nifti, case_paths["imaging"])
    copy2(case_paths["imaging"], case_paths["image_compat"])


def _cached_step_ok() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cached"], 0, stdout="", stderr="")


def _warn_if_cuda_requested_but_unavailable(device: str | None) -> None:
    """nnU-Net and TotalSegmentator use the same PyTorch; if CUDA is missing, both fall back to CPU."""

    if device != "cuda":
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        return
    _pipeline_log(
        "WARNING: --device cuda but torch.cuda.is_available() is False. "
        "TotalSegmentator and nnU-Net will run on CPU (slow). "
        "Fix: request a GPU in your job (Slurm: #SBATCH --gres=gpu:1; check echo $CUDA_VISIBLE_DEVICES); "
        "install a PyTorch wheel that matches the node driver (CLARITY_PYTORCH_CUDA in dev/setup_local_models.sh, e.g. cu118); "
        "in the job run: nvidia-smi && python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"."
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    series_instance_uid: str | None = None,
) -> Path:
    """
    Execute the full pipeline and write ``config.manifest_name`` under the workspace root.

    Raises ``RuntimeError`` if DICOM→NIfTI (``dcm2niix`` or SimpleITK), TotalSegmentator, or tumor steps fail
    when invoked (optional steps may be skipped via config).
    """

    layout = default_workspace_layout(config.workspace_root)
    ensure_workspace_dirs(layout)
    _warn_if_cuda_requested_but_unavailable(config.inference.device)

    steps_completed: list[str] = []
    artifacts: dict[str, Any] = {"cases": []}

    series_list = list(discover_series_roots(config.dicom.input_dir))
    if not series_list:
        raise RuntimeError(
            "No DICOM files were found in this submission. Check that you uploaded the correct study folder "
            "and that files have .dcm extensions. If your DICOM CD uses extensionless files "
            "(e.g., IM-0001-0001), rename them with a .dcm extension before uploading."
        )

    _pipeline_log(f"Found {len(series_list)} DICOM series under {config.dicom.input_dir}")

    if series_instance_uid is not None:
        selected_series = [s for s in series_list if s.series_instance_uid == series_instance_uid]
        if not selected_series:
            raise ValueError(f"No series with SeriesInstanceUID={series_instance_uid!r}")
    else:
        if config.auto_select_series:
            best, reasons = select_best_series(series_list)
            selected_series = [best]
            for reason in reasons:
                _pipeline_log(f"Series selection: {reason}")
        else:
            selected_series = series_list
            if len(selected_series) > 1:
                warnings.warn(
                    f"Processing {len(selected_series)} DICOM series under {config.dicom.input_dir}. "
                    "Use --series-uid to limit to one series.",
                    stacklevel=2,
                )

    processed_case_ids: list[str] = []
    clarity_case_ids: list[str] = []
    n_selected = len(selected_series)

    for case_idx, series in enumerate(selected_series, start=1):
        case_id = series.series_instance_uid
        _pipeline_log(f"Case {case_idx}/{n_selected}: SeriesInstanceUID {case_id}")
        case_steps: list[str] = []
        case_artifacts: dict[str, Any] = {}
        case_paths = ensure_case_workspace(layout, case_id)

        nifti_sub = layout["nifti"] / series.series_instance_uid
        ts_out = case_paths["total_seg"]
        kidney_mask = ts_out / config.mask_adaptation.kidney_mask_name
        tumor_out = case_paths["tumor_segmentation"]

        reuse = config.reuse_cached_artifacts
        imaging_ready = case_paths["imaging"].exists() and case_paths["image_compat"].exists()
        skip_dicom = reuse and imaging_ready
        skip_ts = reuse and kidney_mask.exists()
        skip_tumor_run = config.skip_tumor or (reuse and tumor_out.exists() and tumor_out.stat().st_size > 0)

        if skip_dicom:
            _pipeline_log("  Reusing cached imaging + NIfTI workspace (skip DICOM staging & dcm2niix)…")
            if config.phase_gating.enabled:
                warnings.warn(
                    "reuse_cached_artifacts: ignoring --enable-phase-gating because imaging was reused from workspace.",
                    stacklevel=2,
                )
            try:
                case_artifacts["primary_nifti"] = str(find_primary_nifti(nifti_sub))
            except FileNotFoundError:
                case_artifacts["primary_nifti"] = str(case_paths["imaging"])
            case_steps.append("dicom_to_nifti_cached")
        else:
            _pipeline_log("  Staging DICOM for dcm2niix…")
            staging = stage_series_for_conversion(
                list(series.files),
                layout["dicom_staging"],
                use_symlinks=True,
            )
            case_steps.append("dicom_stage")
            case_artifacts["dicom_staging"] = str(staging)

            nifti_sub.mkdir(parents=True, exist_ok=True)
            if config.dicom_backend == "sitk":
                _pipeline_log("  SimpleITK (GDCM): DICOM → NIfTI…")
                out_nii = nifti_sub / "series_sitk.nii.gz"
                try:
                    convert_staged_dicom_to_nifti(staging, out_nii)
                except Exception as e:
                    raise RuntimeError(f"SimpleITK DICOM→NIfTI failed for {case_id}: {e}") from e
                proc_dcm = subprocess.CompletedProcess(["SimpleITK"], 0, stdout="", stderr="")
                case_artifacts["dcm2niix"] = {
                    "returncode": 0,
                    "stdout": "",
                    "backend": "sitk",
                    "output_nifti": str(out_nii),
                }
            else:
                _pipeline_log("  dcm2niix: DICOM → NIfTI…")
                proc_dcm = run_dcm2niix(
                    staging,
                    nifti_sub,
                    binary=config.dcm2niix_binary,
                )
                case_artifacts["dcm2niix"] = {
                    "returncode": proc_dcm.returncode,
                    "stdout": proc_dcm.stdout[-4000:] if proc_dcm.stdout else "",
                    "backend": "dcm2niix",
                }
                if proc_dcm.returncode != 0:
                    raise RuntimeError(
                        f"dcm2niix failed for {case_id} ({proc_dcm.returncode}): "
                        f"{proc_dcm.stderr or proc_dcm.stdout}"
                    )
            case_steps.append("dicom_to_nifti")

            primary_nifti = find_primary_nifti(nifti_sub)
            _stage_ct_nifti_for_downstream(primary_nifti, case_paths)
            case_artifacts["primary_nifti"] = str(primary_nifti)

            working_nifti_dir = nifti_sub
            if config.phase_gating.enabled:
                _pipeline_log("  Phase gating…")
                pg_out = layout["phase_gating"] / series.series_instance_uid
                try:
                    working_nifti_dir = run_phase_gating(nifti_sub, pg_out, config.phase_gating)
                except Exception as e:
                    raise RuntimeError(
                        "The CT series found appears to be unenhanced or an incorrect phase for renal tumor "
                        "scoring. CLARITY requires a corticomedullary (30–60 s post-contrast) or nephrographic "
                        "(80–120 s post-contrast) phase. Please re-upload the correct contrast phase."
                    ) from e
                case_steps.append("phase_gating")
                case_artifacts["phase_gating_output"] = str(working_nifti_dir)
                try:
                    primary_nifti = find_primary_nifti(working_nifti_dir)
                except FileNotFoundError:
                    raise RuntimeError(
                        "The CT series found appears to be unenhanced or an incorrect phase for renal tumor "
                        "scoring. CLARITY requires a corticomedullary (30–60 s post-contrast) or nephrographic "
                        "(80–120 s post-contrast) phase. Please re-upload the correct contrast phase."
                    ) from None
                case_artifacts["primary_nifti_after_gating"] = str(primary_nifti)
                _stage_ct_nifti_for_downstream(primary_nifti, case_paths)

        if skip_ts:
            _pipeline_log("  Reusing cached TotalSegmentator output…")
            proc_ts = _cached_step_ok()
        else:
            _pipeline_log("  TotalSegmentator (organs / kidneys)…")
            proc_ts = run_totalsegmentator(case_paths["image_compat"], ts_out, config.totalsegmentator)
        case_artifacts["totalsegmentator"] = {
            "returncode": proc_ts.returncode,
            "stdout": proc_ts.stdout[-4000:] if proc_ts.stdout else "",
        }
        if proc_ts.returncode != 0:
            raise RuntimeError(
                f"TotalSegmentator failed for {case_id} ({proc_ts.returncode}): "
                f"{proc_ts.stderr or proc_ts.stdout}"
            )
        case_steps.append("totalsegmentator" if not skip_ts else "totalsegmentator_cached")

        if config.tcga_phase_prediction.enabled:
            if not kidney_mask.exists():
                raise RuntimeError(
                    "TCGA phase prediction requires a kidney mask from TotalSegmentator, but "
                    f"{kidney_mask} is missing."
                )
            _pipeline_log("  TCGA phase prediction (SWP v3; PNvsRN uses v5 separately)…")
            try:
                phase_result = run_tcga_phase_prediction(
                    case_id=case_id,
                    ct_nifti=case_paths["image_compat"],
                    kidney_mask_nifti=kidney_mask,
                    cases_root=layout["cases"],
                    config=config.tcga_phase_prediction,
                )
                enforce_tcga_phase_allowed(phase_result, case_id=case_id)
            except (RuntimeError, ValueError):
                raise
            except Exception as e:
                raise RuntimeError(
                    "Automated contrast-phase scoring failed for this series. CLARITY is validated on "
                    "corticomedullary and nephrographic phases only; ensure the upload is a suitable "
                    "contrast-enhanced abdominal CT."
                ) from e
            case_artifacts["tcga_phase_prediction"] = phase_result.metadata_block
            case_steps.append("tcga_phase_prediction")

        if not config.skip_tumor:
            if skip_tumor_run:
                _pipeline_log("  Reusing cached tumor segmentation…")
                proc_tu = _cached_step_ok()
            else:
                _pipeline_log("  Tumor segmentation (nnU-Net)…")
                proc_tu = run_tumor_segmentation(
                    case_paths["image_compat"],
                    tumor_out,
                    config.tumor,
                    totalseg_dir=ts_out,
                )
            case_artifacts["tumor_segmentation"] = {
                "returncode": proc_tu.returncode,
                "stdout": proc_tu.stdout[-4000:] if proc_tu.stdout else "",
                "tumor_seg_v2": {
                    "model_name": "nnUNet_KiTS23",
                    "mode": config.tumor.mode,
                    "dataset_id": config.tumor.dataset_id,
                    "configuration": config.tumor.configuration,
                },
            }
            if proc_tu.returncode != 0:
                raise RuntimeError(
                    f"Tumor segmentation failed for {case_id} ({proc_tu.returncode}): "
                    f"{proc_tu.stderr or proc_tu.stdout}"
                )
            case_steps.append(
                "tumor_segmentation_cached" if skip_tumor_run else "tumor_segmentation"
            )

        if not kidney_mask.exists():
            raise RuntimeError(
                "No kidney structures were detected in the CT. The scan may not cover the kidneys, may be a "
                "non-abdominal CT, or the image quality may be insufficient for automated segmentation. Confirm "
                "the scan is a full abdominal CT covering both kidneys."
            )

        _pipeline_log("  Mask adaptation (kidney + tumor → segmentation)…")
        tumor_path = case_paths["tumor_segmentation"] if case_paths["tumor_segmentation"].exists() else None
        adapt_cfg = config.mask_adaptation
        if adapt_cfg.reference_image is None:
            from dataclasses import replace

            adapt_cfg = replace(adapt_cfg, reference_image=case_paths["imaging"])

        adapted = adapt_masks(
            kidney_mask_path=kidney_mask,
            tumor_mask_path=tumor_path,
            output_path=case_paths["segmentation"],
            config=adapt_cfg,
        )
        case_steps.append("mask_adaptation")
        case_artifacts["segmentation"] = str(adapted)

        has_tumor = swp_segmentation_has_tumor_voxels(adapted)
        clarity_skip: str | None = None
        if not config.skip_tumor:
            if not has_tumor:
                if config.continue_on_empty_tumor:
                    clarity_skip = "no_tumor_voxels_in_segmentation"
                    case_artifacts["clarity_skip"] = clarity_skip
                    warnings.warn(
                        f"Case {case_id}: no SWP label 2 (tumor) after nnU-Net + mask fusion — "
                        f"omitting from CLARITY manifest. Inspect {case_paths['tumor_segmentation']} "
                        f"and {adapted}. Default is to continue; use --fail-on-empty-tumor to abort on this.",
                        stacklevel=2,
                    )
                else:
                    raise RuntimeError(
                        "No renal tumor was detected in the segmentation. Possible causes: (1) the tumor is too "
                        "small for automated detection at current resolution; (2) the CT phase is unenhanced or "
                        "incorrect — use corticomedullary or nephrographic phase; (3) post-operative scan — use "
                        "a pre-operative CT. If you believe a tumor is present, contact us with the case label "
                        "for manual review."
                    )
            else:
                clarity_case_ids.append(case_id)

        case_metadata: dict[str, Any] = {
            "case_id": case_id,
            "pipeline_version": __version__,
            "source_dicom_dir": str(series.series_dir),
            "study_instance_uid": series.study_instance_uid,
            "series_instance_uid": series.series_instance_uid,
            "modality": series.modality,
            "steps_completed": case_steps,
            "artifacts": {
                "imaging": str(case_paths["imaging"]),
                "segmentation": str(case_paths["segmentation"]),
                "total_seg": str(case_paths["total_seg"]),
                "tumor_segmentation": str(case_paths["tumor_segmentation"]),
            },
        }
        if "tcga_phase_prediction" in case_artifacts:
            case_metadata["tcga_phase_prediction"] = case_artifacts["tcga_phase_prediction"]
        if clarity_skip is not None:
            case_metadata["clarity_skip"] = clarity_skip

        write_case_metadata(case_paths["metadata"], case_metadata)
        artifacts["cases"].append(
            {
                "case_id": case_id,
                "case_root": str(case_paths["case_root"]),
                "imaging": str(case_paths["imaging"]),
                "segmentation": str(case_paths["segmentation"]),
                "study_instance_uid": series.study_instance_uid,
                "series_instance_uid": series.series_instance_uid,
                "steps_completed": case_steps,
            }
        )
        processed_case_ids.append(case_id)

    if not processed_case_ids:
        raise RuntimeError("No DICOM series were processed successfully.")

    _pipeline_log(
        f"Writing SWP manifest ({len(clarity_case_ids)} case(s) with tumor labels for CLARITY; "
        f"{len(processed_case_ids)} case(s) processed overall)…"
    )

    steps_completed.extend(
        [
            "dicom_stage",
            "dicom_to_nifti",
            "phase_gating" if config.phase_gating.enabled else "phase_gating_skipped",
            "totalsegmentator",
            (
                "tcga_phase_prediction"
                if config.tcga_phase_prediction.enabled
                else "tcga_phase_prediction_skipped"
            ),
            "tumor_segmentation" if not config.skip_tumor else "tumor_segmentation_skipped",
            "mask_adaptation",
        ]
    )

    swp_manifest_path = layout["root"] / "swp_manifest.json"
    write_swp_manifest(
        swp_manifest_path,
        case_ids=clarity_case_ids,
        data_root=layout["cases"],
    )
    steps_completed.append("swp_manifest")
    artifacts["swp_manifest"] = str(swp_manifest_path)

    if not config.skip_inference:
        if len(clarity_case_ids) == 0:
            _pipeline_log(
                "Skipping CLARITY inference: no case has SWP tumor label 2 "
                "(nnU-Net produced no tumor voxels, or all such cases were skipped)."
            )
        else:
            _pipeline_log("CLARITY inference (PN vs RN)…")
            prediction_path = layout["predictions"] / config.inference.output_name
            run_clarity_inference(
                manifest_path=swp_manifest_path,
                data_root=layout["cases"],
                output_json=prediction_path,
                config=config.inference,
            )
            steps_completed.append("clarity_inference")
            artifacts["predictions_json"] = str(prediction_path)

    artifacts["clarity_case_ids"] = clarity_case_ids

    manifest_path = layout["root"] / config.manifest_name
    write_manifest(
        manifest_path,
        pipeline_version=__version__,
        steps_completed=steps_completed,
        artifacts=artifacts,
        config_snapshot={
            "workspace_root": str(config.workspace_root),
            "dicom_input": str(config.dicom.input_dir),
            "case_ids": processed_case_ids,
            "clarity_case_ids": clarity_case_ids,
            "series_instance_uid": series_instance_uid,
            "phase_gating_enabled": config.phase_gating.enabled,
            "tcga_phase_prediction_enabled": config.tcga_phase_prediction.enabled,
            "skip_tumor": config.skip_tumor,
            "skip_inference": config.skip_inference,
            "reuse_cached_artifacts": config.reuse_cached_artifacts,
            "continue_on_empty_tumor": config.continue_on_empty_tumor,
            "dicom_backend": config.dicom_backend,
            "auto_select_series": config.auto_select_series,
        },
    )
    steps_completed.append("manifest")
    return manifest_path


def build_pipeline_config(
    *,
    workspace_root: Path,
    dicom_input: Path,
    series_output_dir: Path | None = None,
    **pipeline_kwargs: Any,
) -> PipelineConfig:
    """Helper to construct :class:`PipelineConfig` with a :class:`DicomPaths` block."""

    out = series_output_dir or workspace_root / "dicom_series_out"
    dicom_paths = DicomPaths(input_dir=dicom_input, series_output_dir=out)
    return PipelineConfig(workspace_root=workspace_root, dicom=dicom_paths, **pipeline_kwargs)
