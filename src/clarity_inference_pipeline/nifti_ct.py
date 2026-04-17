"""CT NIfTI selection and 3D canonicalization for segmentation backends."""

from __future__ import annotations

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np


def select_primary_ct_nifti(nifti_dir: Path) -> Path:
    """
    Choose the main CT volume under ``nifti_dir``.

    dcm2niix may emit both diagnostic CT and derived SEG objects; the largest file is
    not always the CT. We deprioritize filenames that look like DICOM SEG exports.
    """

    candidates = sorted(nifti_dir.glob("*.nii.gz")) + sorted(nifti_dir.glob("*.nii"))
    if not candidates:
        raise FileNotFoundError(f"No NIfTI files found in {nifti_dir}")

    def sort_key(p: Path) -> tuple[int, int]:
        stem_l = p.stem.lower()
        looks_like_seg = "segmentation" in stem_l or stem_l.endswith("_seg")
        return (1 if looks_like_seg else 0, -p.stat().st_size)

    ranked = sorted(candidates, key=sort_key)
    best = ranked[0]
    if "segmentation" in best.stem.lower():
        warnings.warn(
            f"No non-segmentation NIfTI under {nifti_dir}; using {best.name}. "
            "Point --input at a directory that contains only the CT series if results look wrong.",
            stacklevel=2,
        )
    return best


def _array_to_cxyz(data: np.ndarray) -> np.ndarray:
    """Return a 3D voxel array (X, Y, Z) for nnU-Net / TotalSegmentator-style volumes."""

    d = np.asarray(data)
    if d.ndim == 3:
        return d.astype(np.float32, copy=False)

    if d.ndim == 4:
        # (X, Y, Z, T) / (X, Y, Z, echo) — common after dcm2niix multi-frame
        if d.shape[-1] <= 16:
            return d[..., 0].astype(np.float32, copy=False)
        # (C, X, Y, Z) with a small channel dim
        if d.shape[0] <= 16 and d.shape[0] < min(d.shape[1], d.shape[2], d.shape[3]):
            return d[0, ...].astype(np.float32, copy=False)
        return d[..., 0].astype(np.float32, copy=False)

    if d.ndim == 5:
        return d[..., 0, 0].astype(np.float32, copy=False)

    raise ValueError(f"Unsupported image rank {d.ndim} with shape {d.shape}")


def write_nnunet_compatible_nifti(src: Path, dst: Path) -> None:
    """
    Load ``src``, collapse extra dimensions to a single 3D volume, save float32 NIfTI at ``dst``.

    nnU-Net v1 expects spatially 3D CT; 4D NIfTI from multi-frame DICOM must be reduced.
    """

    img = nib.load(str(src))
    data3 = _array_to_cxyz(img.get_fdata())
    out = nib.Nifti1Image(data3.astype(np.float32, copy=False), img.affine)
    out.header.set_data_dtype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
