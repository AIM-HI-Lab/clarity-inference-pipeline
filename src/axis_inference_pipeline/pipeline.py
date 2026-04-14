"""End-to-end orchestration: DICOM -> NIfTI -> segmentation -> axis-pn prediction."""

from __future__ import annotations

import subprocess
import sys
import warnings
from shutil import copy2
from pathlib import Path
from typing import Any

from . import __version__
from .config import DicomPaths, PipelineConfig
from .axis_pn import run_axis_pn_inference
from .dicom import discover_series_roots, run_dcm2niix, stage_series_for_conversion
from .dicom_sitk import convert_staged_dicom_to_nifti
from .mask_adaptation import adapt_masks
from .nifti_ct import select_primary_ct_nifti, write_nnunet_compatible_nifti
from .phase_gating import run_phase_gating
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
    print(f"[axis-pn] {message}", file=sys.stderr, flush=True)


def find_primary_nifti(nifti_dir: Path) -> Path:
    """Pick the primary CT NIfTI under ``nifti_dir`` (see :func:`select_primary_ct_nifti`)."""

    return select_primary_ct_nifti(nifti_dir)


def _stage_ct_nifti_for_downstream(raw_nifti: Path, case_paths: dict[str, Path]) -> None:
    """Write 3D float CT to ``imaging`` / ``image_compat`` for TotalSegmentator and nnU-Net."""

    write_nnunet_compatible_nifti(raw_nifti, case_paths["imaging"])
    copy2(case_paths["imaging"], case_paths["image_compat"])


def _cached_step_ok() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["cached"], 0, stdout="", stderr="")


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

    steps_completed: list[str] = []
    artifacts: dict[str, Any] = {"cases": []}

    series_list = list(discover_series_roots(config.dicom.input_dir))
    if not series_list:
        raise FileNotFoundError(f"No DICOM series found under {config.dicom.input_dir}")

    _pipeline_log(f"Found {len(series_list)} DICOM series under {config.dicom.input_dir}")

    if series_instance_uid is not None:
        selected_series = [s for s in series_list if s.series_instance_uid == series_instance_uid]
        if not selected_series:
            raise ValueError(f"No series with SeriesInstanceUID={series_instance_uid!r}")
    else:
        selected_series = series_list
        if len(selected_series) > 1:
            warnings.warn(
                f"Processing {len(selected_series)} DICOM series under {config.dicom.input_dir}. "
                "Use --series-uid to limit to one series.",
                stacklevel=2,
            )

    processed_case_ids: list[str] = []
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
                working_nifti_dir = run_phase_gating(nifti_sub, pg_out, config.phase_gating)
                case_steps.append("phase_gating")
                case_artifacts["phase_gating_output"] = str(working_nifti_dir)
                try:
                    primary_nifti = find_primary_nifti(working_nifti_dir)
                except FileNotFoundError:
                    primary_nifti = find_primary_nifti(nifti_sub)
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
            raise FileNotFoundError(
                f"Expected kidney mask at {kidney_mask}. Configure TotalSegmentator to produce it."
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

        write_case_metadata(
            case_paths["metadata"],
            {
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
            },
        )
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

    _pipeline_log(f"Writing SWP manifest ({len(processed_case_ids)} case(s))…")

    steps_completed.extend(
        [
            "dicom_stage",
            "dicom_to_nifti",
            "phase_gating" if config.phase_gating.enabled else "phase_gating_skipped",
            "totalsegmentator",
            "tumor_segmentation" if not config.skip_tumor else "tumor_segmentation_skipped",
            "mask_adaptation",
        ]
    )

    swp_manifest_path = layout["root"] / "swp_manifest.json"
    write_swp_manifest(
        swp_manifest_path,
        case_ids=processed_case_ids,
        data_root=layout["cases"],
    )
    steps_completed.append("swp_manifest")
    artifacts["swp_manifest"] = str(swp_manifest_path)

    if not config.skip_inference:
        _pipeline_log("axis-pn inference (PN vs RN)…")
        prediction_path = layout["predictions"] / config.inference.output_name
        run_axis_pn_inference(
            manifest_path=swp_manifest_path,
            data_root=layout["cases"],
            output_json=prediction_path,
            config=config.inference,
        )
        steps_completed.append("axis_pn_inference")
        artifacts["predictions_json"] = str(prediction_path)

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
            "series_instance_uid": series_instance_uid,
            "phase_gating_enabled": config.phase_gating.enabled,
            "skip_tumor": config.skip_tumor,
            "skip_inference": config.skip_inference,
            "reuse_cached_artifacts": config.reuse_cached_artifacts,
            "dicom_backend": config.dicom_backend,
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
