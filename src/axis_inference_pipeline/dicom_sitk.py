"""DICOM → NIfTI using SimpleITK (GDCM), for environments without ``dcm2niix`` on PATH."""

from __future__ import annotations

from pathlib import Path


def convert_staged_dicom_to_nifti(staging_dir: Path, output_nifti: Path) -> None:
    """
    Read one DICOM series from ``staging_dir`` and write a single compressed NIfTI.

    Uses :class:`SimpleITK.ImageSeriesReader` (same idea as ``dicom2niiser`` / GDCM).
    """

    import SimpleITK as sitk

    staging_dir = staging_dir.resolve()
    output_nifti = output_nifti.resolve()
    output_nifti.parent.mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(staging_dir))
    if not dicom_names:
        raise RuntimeError(
            f"SimpleITK found no DICOM series under {staging_dir}. "
            "Check that slices are readable and belong to one series."
        )
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    size = image.GetSize()
    if len(size) >= 3 and min(size[:3]) < 5:
        raise RuntimeError(f"Refusing tiny volume from SimpleITK (size={size!r}).")

    sitk.WriteImage(image, str(output_nifti))
