"""Pipeline configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DicomPaths:
    """Resolved inputs for DICOM discovery and conversion."""

    input_dir: Path
    """Directory tree to scan for DICOM files."""

    series_output_dir: Path
    """Per-series working output (NIfTI or intermediate)."""


@dataclass(frozen=True)
class TotalSegmentatorConfig:
    """Settings for TotalSegmentator CLI invocation."""

    binary: str = "TotalSegmentator"
    extra_args: Sequence[str] = ()
    task: str | None = None
    """If set, passed as ``--task`` (library-specific; adjust for your install)."""

    device: str | None = None
    """If set, passed as ``--device`` (``gpu`` / ``cpu``); aligns with pipeline ``--device`` cuda/cpu."""


@dataclass(frozen=True)
class TumorSegmentationConfig:
    """Settings for tumor segmentation (external CLI or script)."""

    binary: str = "axis-nnunet-predict"
    mode: str = "nnunetv1"
    task_id: str = "135"
    model: str = "3d_cascade_fullres"
    dataset_id: str = "Dataset123_Kits23"
    configuration: str = "3d_fullres"
    folds: Sequence[str] = ("all",)
    extra_args: Sequence[str] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseGatingConfig:
    """Optional phase gating; disabled unless ``enabled`` is true."""

    enabled: bool = False
    """When false, phase gating is skipped entirely."""

    entrypoint: str | None = None
    """Python module path (``pkg.module:callable``) or shell command name."""

    extra_args: Sequence[str] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    """Passed to a callable entrypoint when using dynamic import."""


@dataclass(frozen=True)
class MaskAdaptationConfig:
    """Resampling / label fusion for masks before downstream axis-pn."""

    reference_image: Path | None = None
    """If set, masks are adapted to this image grid."""

    interpolation: str = "nearest"
    kidney_mask_name: str = "kidney_binary_mask.nii.gz"
    tumor_output_name: str = "tumor_segmentation_v2.nii.gz"
    output_name: str = "segmentation.nii.gz"
    tumor_labels: Sequence[int] = (2,)
    extra_args: Sequence[str] = ()


@dataclass(frozen=True)
class InferenceConfig:
    """Settings for vendored SWP PNvsRN inference."""

    project_name: str = "pnvrn_nifti"
    expected_checkpoint_count: int | None = 25
    checkpoint_paths: Sequence[Path] = ()
    checkpoint_dir: Path | None = None
    checkpoint_dir_recursive: bool = True
    model_root: Path | None = None
    output_name: str = "predictions.json"
    device: str | None = None
    cache_root: Path | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level options for :func:`run_pipeline`."""

    workspace_root: Path
    dicom: DicomPaths
    totalsegmentator: TotalSegmentatorConfig = field(default_factory=TotalSegmentatorConfig)
    tumor: TumorSegmentationConfig = field(default_factory=TumorSegmentationConfig)
    phase_gating: PhaseGatingConfig = field(default_factory=PhaseGatingConfig)
    mask_adaptation: MaskAdaptationConfig = field(default_factory=MaskAdaptationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    skip_tumor: bool = False
    skip_inference: bool = False
    # When True, skip DICOM→NIfTI, TotalSegmentator, and tumor if those outputs already exist under the workspace.
    reuse_cached_artifacts: bool = False
    dicom_backend: str = "dcm2niix"
    """``dcm2niix`` (external CLI) or ``sitk`` (SimpleITK / GDCM in-process)."""
    dcm2niix_binary: str = "dcm2niix"
    manifest_name: str = "run_manifest.json"
