"""Adapt segmentation masks into the SWP PNvsRN label convention."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from .config import MaskAdaptationConfig


def _nnunet_labels_to_uint8(tumor_data):
    """Match SWP cache (data_loader_v5): discrete labels from float nnU-Net NIfTI voxels."""

    import numpy as np

    return np.round(np.asarray(tumor_data)).astype(np.uint8)


def _resample_to_reference(mask_path: Path, reference_image: Path, interpolation: str):
    import nibabel as nib
    from nibabel.processing import resample_from_to

    ref = nib.load(str(reference_image))
    mov = nib.load(str(mask_path))
    order = 0 if interpolation == "nearest" else 1
    return resample_from_to(mov, ref, order=order)


def _resolve_mask(mask_path: Path, config: MaskAdaptationConfig):
    if config.reference_image is None:
        import nibabel as nib

        return nib.load(str(mask_path))
    return _resample_to_reference(mask_path, config.reference_image, config.interpolation)


def create_swp_segmentation(
    *,
    kidney_mask_path: Path,
    tumor_mask_path: Path | None,
    output_path: Path,
    config: MaskAdaptationConfig,
) -> Path:
    """
    Create `segmentation.nii.gz` using the SWP PNvsRN convention.

    Label meanings:
    - `0`: background
    - `1`: support organ / kidney context
    - `2`: tumor
    """

    import nibabel as nib
    import numpy as np

    kidney_img = _resolve_mask(kidney_mask_path, config)
    kidney_data = np.asarray(kidney_img.get_fdata())
    output = np.zeros(kidney_data.shape, dtype=np.uint8)
    output[kidney_data > 0] = 1

    if tumor_mask_path is not None and tumor_mask_path.exists():
        tumor_img = _resolve_mask(tumor_mask_path, config)
        tumor_u8 = _nnunet_labels_to_uint8(tumor_img.get_fdata())
        tumor_mask = np.zeros(tumor_u8.shape, dtype=bool)
        for label in config.tumor_labels:
            tumor_mask |= tumor_u8 == int(label)
        output[tumor_mask] = 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(output, kidney_img.affine, kidney_img.header), str(output_path))
    return output_path


def adapt_masks(
    *,
    kidney_mask_path: Path,
    tumor_mask_path: Path | None,
    output_path: Path,
    config: MaskAdaptationConfig,
) -> Path:
    """Build the SWP-ready segmentation artifact for a single case."""

    return create_swp_segmentation(
        kidney_mask_path=kidney_mask_path,
        tumor_mask_path=tumor_mask_path,
        output_path=output_path,
        config=config,
    )


def swp_segmentation_has_tumor_voxels(
    segmentation_path: Path,
    *,
    tumor_label_ids: Sequence[int] | None = None,
) -> bool:
    """
    Return True if ``segmentation_path`` contains at least one voxel in ``tumor_label_ids``.

    axis-pn / SWP V5 require a non-empty primary (default label ``2``) to build patches.
    """

    import nibabel as nib
    import numpy as np

    ids = tumor_label_ids if tumor_label_ids is not None else (2,)
    data = np.round(np.asarray(nib.load(str(segmentation_path)).get_fdata())).astype(np.uint8)
    for lid in ids:
        if np.any(data == int(lid)):
            return True
    return False


def run_mask_tool_subprocess(
    command: str,
    mask_paths: list[Path],
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Escape hatch: run an external resampling tool with a fixed argument shape."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [command, "--out", str(output_dir)]
    if extra_args:
        cmd.extend(extra_args)
    for m in mask_paths:
        cmd.extend(["--mask", str(m)])
    return subprocess.run(cmd, check=False, capture_output=True, text=True)
