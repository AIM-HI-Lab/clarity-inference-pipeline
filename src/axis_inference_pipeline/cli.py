"""CLI entrypoint: ``axis-pn predict`` and related commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import (
    InferenceConfig,
    MaskAdaptationConfig,
    PhaseGatingConfig,
    TotalSegmentatorConfig,
    TumorSegmentationConfig,
)
from .pipeline import build_pipeline_config, run_pipeline
from .pipeline_profile import resolve_totalsegmentator_extra_args, resolve_tumor_extra_args

app = typer.Typer(
    name="axis-pn",
    help="DICOM-to-axis-pn inference pipeline (orchestration and external model wrappers).",
    no_args_is_help=True,
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "pnvrn_folds"


@app.callback()
def _root() -> None:
    """Axis PN pipeline CLI."""


@app.command("predict")
def predict(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory tree containing DICOM files for one or more series.",
        ),
    ],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "--work-dir",
            "--output",
            "-w",
            file_okay=False,
            help="Workspace root for case folders, logs, manifest, and predictions (created if missing).",
        ),
    ],
    series_uid: Annotated[
        Optional[str],
        typer.Option(
            "--series-uid",
            help="SeriesInstanceUID to process when multiple series are present.",
        ),
    ] = None,
    skip_tumor: Annotated[
        bool,
        typer.Option("--skip-tumor", help="Skip tumor segmentation step."),
    ] = False,
    skip_inference: Annotated[
        bool,
        typer.Option("--skip-inference", help="Stop after building SWP-ready NIfTI inputs."),
    ] = False,
    dcm2niix: Annotated[str, typer.Option("--dcm2niix", help="dcm2niix executable name or path.")] = "dcm2niix",
    totalseg_binary: Annotated[
        str,
        typer.Option("--totalseg-binary", help="TotalSegmentator executable name or path."),
    ] = "TotalSegmentator",
    totalseg_task: Annotated[
        Optional[str],
        typer.Option("--totalseg-task", help="Optional TotalSegmentator --task value."),
    ] = None,
    totalseg_extra: Annotated[
        Optional[str],
        typer.Option(
            "--totalseg-extra",
            help='Extra TotalSegmentator arguments (shell-style); else AXIS_TOTALSEG_EXTRA.',
        ),
    ] = None,
    tumor_binary: Annotated[
        str,
        typer.Option(
            "--tumor-binary",
            help="Tumor CLI (default: axis-nnunet-predict for nnunetv1 — PyTorch 2.6+ safe nnU-Net v1).",
        ),
    ] = "axis-nnunet-predict",
    tumor_mode: Annotated[
        str,
        typer.Option("--tumor-mode", help="Tumor segmentation wrapper mode: nnunetv1, nnunetv2, or simple."),
    ] = "nnunetv1",
    tumor_task_id: Annotated[
        str,
        typer.Option("--tumor-task-id", help="nnU-Net v1 task id when --tumor-mode=nnunetv1."),
    ] = "135",
    tumor_model: Annotated[
        str,
        typer.Option("--tumor-model", help="nnU-Net model name when --tumor-mode=nnunetv1."),
    ] = "3d_cascade_fullres",
    tumor_dataset_id: Annotated[
        str,
        typer.Option("--tumor-dataset-id", help="nnUNet dataset id when --tumor-mode=nnunetv2."),
    ] = "Dataset123_Kits23",
    tumor_configuration: Annotated[
        str,
        typer.Option("--tumor-configuration", help="nnUNet configuration when --tumor-mode=nnunetv2."),
    ] = "3d_fullres",
    tumor_extra: Annotated[
        Optional[str],
        typer.Option(
            "--tumor-extra",
            help='Extra tumor-segmentation CLI args (shell-style); else AXIS_TUMOR_EXTRA.',
        ),
    ] = None,
    checkpoint_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--checkpoint-dir",
            "--weights-dir",
            exists=True,
            file_okay=False,
            help="Directory containing axis-pn .pth checkpoints.",
        ),
    ] = None,
    checkpoint_dir_recursive: Annotated[
        bool,
        typer.Option(
            "--checkpoint-dir-recursive/--no-checkpoint-dir-recursive",
            help="Search for .pth in subfolders (default: on; needed for pnvrn_folds/m*/fold_*.pth).",
        ),
    ] = True,
    checkpoint_path: Annotated[
        Optional[list[Path]],
        typer.Option("--checkpoint-path", exists=True, dir_okay=False, help="Explicit checkpoint path; repeatable."),
    ] = None,
    model_root: Annotated[
        Optional[Path],
        typer.Option("--model-root", exists=True, file_okay=False, help="Model root with fold_*/checkpoints/best_model.pth."),
    ] = None,
    device: Annotated[
        Optional[str],
        typer.Option("--device", help="SWP inference device override, e.g. cpu or cuda."),
    ] = None,
    swp_cache_root: Annotated[
        Optional[Path],
        typer.Option("--swp-cache-root", file_okay=False, help="Cache root for SWP patch extraction."),
    ] = None,
    enable_phase_gating: Annotated[
        bool,
        typer.Option(
            "--enable-phase-gating",
            help="Run optional phase-gating step (requires --phase-entrypoint).",
        ),
    ] = False,
    phase_entrypoint: Annotated[
        Optional[str],
        typer.Option(
            "--phase-entrypoint",
            help="``module:callable`` or shell command; required when phase gating is enabled.",
        ),
    ] = None,
    mask_reference: Annotated[
        Optional[Path],
        typer.Option(
            "--mask-reference",
            exists=True,
            dir_okay=False,
            help="Reference NIfTI for mask resampling; default is primary CT volume.",
        ),
    ] = None,
    manifest_name: Annotated[
        str,
        typer.Option("--manifest-name", help="Manifest filename written under the workspace root."),
    ] = "run_manifest.json",
    reuse_cached_artifacts: Annotated[
        bool,
        typer.Option(
            "--reuse-cached-artifacts",
            help="Skip DICOM→NIfTI, TotalSegmentator, and tumor when those outputs already exist under --workspace.",
        ),
    ] = False,
) -> None:
    """Run the full DICOM → axis-pn pipeline (DICOM ingest, segmentation, optional gating)."""

    if enable_phase_gating and not phase_entrypoint:
        raise typer.BadParameter("--phase-entrypoint is required when --enable-phase-gating is set.")

    if checkpoint_dir is None and DEFAULT_MODEL_DIR.exists():
        checkpoint_dir = DEFAULT_MODEL_DIR
        checkpoint_dir_recursive = True

    phase_cfg = PhaseGatingConfig(
        enabled=enable_phase_gating,
        entrypoint=phase_entrypoint,
    )
    totalseg_extra_args = resolve_totalsegmentator_extra_args(cli_extra=totalseg_extra)
    tumor_extra_args = resolve_tumor_extra_args(cli_extra=tumor_extra)
    ts_cfg = TotalSegmentatorConfig(binary=totalseg_binary, task=totalseg_task, extra_args=totalseg_extra_args)
    tumor_cfg = TumorSegmentationConfig(
        binary=tumor_binary,
        mode=tumor_mode,
        task_id=tumor_task_id,
        model=tumor_model,
        dataset_id=tumor_dataset_id,
        configuration=tumor_configuration,
        extra_args=tumor_extra_args,
    )
    mask_cfg = MaskAdaptationConfig(reference_image=mask_reference)
    inference_cfg = InferenceConfig(
        checkpoint_paths=tuple(checkpoint_path or []),
        checkpoint_dir=checkpoint_dir,
        checkpoint_dir_recursive=checkpoint_dir_recursive,
        model_root=model_root,
        device=device,
        cache_root=swp_cache_root,
    )

    if not skip_inference and not (
        inference_cfg.checkpoint_paths or inference_cfg.checkpoint_dir or inference_cfg.model_root
    ):
        raise typer.BadParameter(
            "Provide --checkpoint-path, --checkpoint-dir, or --model-root unless --skip-inference is set."
        )

    cfg = build_pipeline_config(
        workspace_root=workspace,
        dicom_input=input_dir,
        totalsegmentator=ts_cfg,
        tumor=tumor_cfg,
        phase_gating=phase_cfg,
        mask_adaptation=mask_cfg,
        inference=inference_cfg,
        skip_tumor=skip_tumor,
        skip_inference=skip_inference,
        reuse_cached_artifacts=reuse_cached_artifacts,
        dcm2niix_binary=dcm2niix,
        manifest_name=manifest_name,
    )

    manifest = run_pipeline(cfg, series_instance_uid=series_uid)
    typer.echo(f"Wrote manifest: {manifest}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
