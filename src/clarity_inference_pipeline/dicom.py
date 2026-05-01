"""DICOM series discovery and conversion to NIfTI (via external tools)."""

from __future__ import annotations

import os
import shutil
import subprocess
from math import sqrt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pydicom

from .subprocess_util import run_subprocess_logged


@dataclass(frozen=True)
class DicomSeries:
    """A discovered series: files sharing Study/Series UIDs."""

    files: tuple[Path, ...]
    study_instance_uid: str
    series_instance_uid: str
    modality: str | None
    series_dir: Path
    """Parent directory of the first slice (informational)."""

    @property
    def file_count(self) -> int:
        return len(self.files)


_NON_IMAGE_SOP_CLASS_UIDS = {
    # Secondary Capture
    "1.2.840.10008.5.1.4.1.1.7",
    # Grayscale Softcopy Presentation State Storage
    "1.2.840.10008.5.1.4.1.1.11.1",
    # Color Softcopy Presentation State Storage
    "1.2.840.10008.5.1.4.1.1.11.2",
    # Pseudo-Color Softcopy Presentation State Storage
    "1.2.840.10008.5.1.4.1.1.11.3",
    # Blending Softcopy Presentation State Storage
    "1.2.840.10008.5.1.4.1.1.11.4",
    # XA/XRF Grayscale Softcopy Presentation State Storage
    "1.2.840.10008.5.1.4.1.1.11.5",
    # Encapsulated PDF Storage
    "1.2.840.10008.5.1.4.1.1.104.1",
    # Raw Data Storage
    "1.2.840.10008.5.1.4.1.1.66",
}


def _read_uids(path: Path) -> tuple[str, str] | None:
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None
    suid = getattr(ds, "StudyInstanceUID", None)
    seuid = getattr(ds, "SeriesInstanceUID", None)
    if not suid or not seuid:
        return None
    return str(suid), str(seuid)


def discover_series_roots(
    input_dir: Path,
    *,
    pattern: str = "**/*",
) -> Iterator[DicomSeries]:
    """
    Walk ``input_dir`` and group DICOM files by (StudyInstanceUID, SeriesInstanceUID).

    Files are grouped into per-series folders under a staging layout by copying or
    symlinking in :func:`stage_series_for_conversion` before calling ``dcm2niix``.
    """

    input_dir = input_dir.resolve()
    groups: dict[tuple[str, str], list[Path]] = {}
    for fp in sorted(input_dir.glob(pattern)):
        if not fp.is_file():
            continue
        if fp.name.startswith("."):
            continue
        uids = _read_uids(fp)
        if uids is None:
            continue
        groups.setdefault(uids, []).append(fp)

    for (study_uid, series_uid), file_list in sorted(groups.items()):
        file_tuple = tuple(sorted(file_list, key=lambda p: str(p)))
        modality: str | None = None
        if file_tuple:
            try:
                ds = pydicom.dcmread(file_tuple[0], stop_before_pixels=True, force=True)
                modality = str(getattr(ds, "Modality", "") or "") or None
            except Exception:
                modality = None
        series_dir = file_tuple[0].parent
        yield DicomSeries(
            files=file_tuple,
            study_instance_uid=study_uid,
            series_instance_uid=series_uid,
            modality=modality,
            series_dir=series_dir,
        )


def _read_first_dataset(series: DicomSeries) -> pydicom.dataset.FileDataset | None:
    try:
        return pydicom.dcmread(series.files[0], stop_before_pixels=True, force=True)
    except Exception:
        return None


def _image_type_primary_score(image_type: object) -> int:
    if not image_type:
        return 0
    if isinstance(image_type, str):
        vals = [part.strip().upper() for part in image_type.split("\\") if part.strip()]
    else:
        vals = [str(v).strip().upper() for v in image_type]
    has_primary = "PRIMARY" in vals
    has_derived_secondary = "DERIVED" in vals and "SECONDARY" in vals
    if has_primary and not has_derived_secondary:
        return 1
    if has_primary:
        return 1
    return 0


def _axial_orientation_score(iop: object) -> int:
    if not iop:
        return 0
    try:
        vals = [float(v) for v in iop]
    except Exception:
        return 0
    if len(vals) != 6:
        return 0
    rx, ry, rz, cx, cy, cz = vals
    nx = (ry * cz) - (rz * cy)
    ny = (rz * cx) - (rx * cz)
    nz = (rx * cy) - (ry * cx)
    mag = sqrt(nx * nx + ny * ny + nz * nz)
    if mag == 0:
        return 0
    nx /= mag
    ny /= mag
    nz /= mag
    # Axial series has slice normal closest to +/- Z.
    if abs(nz) >= abs(nx) and abs(nz) >= abs(ny):
        return 1
    return 0


def select_best_series(series_list: list[DicomSeries]) -> tuple[DicomSeries, list[str]]:
    """
    Select one best primary contrast-enhanced axial CT series candidate.

    Returns:
        (selected_series, reason_strings_for_logging)
    """

    reasons: list[str] = []
    if not series_list:
        raise RuntimeError(
            "No CT series with sufficient slices were found. The uploaded folder may be a patient-level "
            "folder containing only non-CT modalities (MRI, PET, dose reports), or all CT series had fewer "
            "than 30 slices (scouts/localizers only). Upload one contrast-enhanced abdominal CT study folder."
        )

    candidates: list[tuple[DicomSeries, pydicom.dataset.FileDataset | None]] = []
    for series in series_list:
        ds = _read_first_dataset(series)
        modality = (series.modality or "").strip().upper()
        if modality != "CT":
            reasons.append(
                f"Rejected {series.series_instance_uid}: modality={series.modality or 'unknown'} (requires CT)."
            )
            continue
        if series.file_count < 30:
            reasons.append(
                f"Rejected {series.series_instance_uid}: only {series.file_count} slices (<30)."
            )
            continue
        sop_class_uid = str(getattr(ds, "SOPClassUID", "") or "")
        if sop_class_uid in _NON_IMAGE_SOP_CLASS_UIDS:
            reasons.append(
                f"Rejected {series.series_instance_uid}: SOPClassUID={sop_class_uid} is non-image."
            )
            continue
        candidates.append((series, ds))

    if not candidates:
        raise RuntimeError(
            "No CT series with sufficient slices were found. The uploaded folder may be a patient-level "
            "folder containing only non-CT modalities (MRI, PET, dose reports), or all CT series had fewer "
            "than 30 slices (scouts/localizers only). Upload one contrast-enhanced abdominal CT study folder."
        )

    if len(candidates) == 1:
        chosen = candidates[0][0]
        reasons.append(
            f"Selected {chosen.series_instance_uid}: only one CT candidate remained after hard filters."
        )
        return chosen, reasons

    scored: list[tuple[int, int, int, DicomSeries]] = []
    for series, ds in candidates:
        primary_score = _image_type_primary_score(getattr(ds, "ImageType", None))
        axial_score = _axial_orientation_score(getattr(ds, "ImageOrientationPatient", None))
        scored.append((primary_score, axial_score, series.file_count, series))
        reasons.append(
            f"Candidate {series.series_instance_uid}: primary_score={primary_score}, "
            f"axial_score={axial_score}, slices={series.file_count}."
        )

    best_primary = max(row[0] for row in scored)
    scored = [row for row in scored if row[0] == best_primary]
    reasons.append(f"Step 2 kept {len(scored)} series with highest primary_score={best_primary}.")

    best_axial = max(row[1] for row in scored)
    scored = [row for row in scored if row[1] == best_axial]
    reasons.append(f"Step 3 kept {len(scored)} series with highest axial_score={best_axial}.")

    best_slices = max(row[2] for row in scored)
    scored = [row for row in scored if row[2] == best_slices]
    reasons.append(f"Step 4 kept {len(scored)} series with max slices={best_slices}.")

    selected = sorted(scored, key=lambda row: row[3].series_instance_uid)[0][3]
    reasons.append(f"Selected {selected.series_instance_uid} as best CT series.")
    return selected, reasons


def stage_series_for_conversion(
    files: list[Path],
    staging_dir: Path,
    *,
    use_symlinks: bool = True,
) -> Path:
    """Copy or symlink ``files`` into ``staging_dir/<series_uid>/`` and return that path."""

    if not files:
        raise ValueError("files must be non-empty")
    uids = _read_uids(files[0])
    if uids is None:
        raise ValueError(f"Not a DICOM file: {files[0]}")
    _, series_uid = uids
    target = staging_dir / series_uid
    target.mkdir(parents=True, exist_ok=True)
    for src in files:
        dest = target / src.name
        if dest.exists():
            continue
        if use_symlinks:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)
    return target


def run_dcm2niix(
    input_dir: Path,
    output_dir: Path,
    *,
    binary: str = "dcm2niix",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run ``dcm2niix`` to convert DICOM in ``input_dir`` to NIfTI in ``output_dir``.

    Requires ``dcm2niix`` on PATH in deployment environments.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [binary, "-o", str(output_dir), "-z", "y", str(input_dir)]
    if extra_args:
        cmd.extend(extra_args)
    return run_subprocess_logged(cmd, label="dcm2niix")


def collect_files_for_series(
    input_dir: Path,
    study_uid: str,
    series_uid: str,
    *,
    pattern: str = "**/*",
) -> list[Path]:
    """Return all files under ``input_dir`` belonging to the given series UIDs."""

    out: list[Path] = []
    for fp in sorted(input_dir.glob(pattern)):
        if not fp.is_file():
            continue
        uids = _read_uids(fp)
        if uids is None:
            continue
        su, se = uids
        if su == study_uid and se == series_uid:
            out.append(fp)
    return out


def executable_available(binary: str) -> bool:
    """True if ``binary`` is an executable path or a name found on ``PATH``."""

    p = Path(binary).expanduser()
    if p.is_file():
        return os.access(p, os.X_OK)
    return shutil.which(binary) is not None


def resolve_dicom_backend(mode: str, dcm2niix_binary: str) -> str:
    """
    Resolve ``auto`` → ``dcm2niix`` when the binary exists, else ``sitk`` if SimpleITK imports.

    ``mode`` is one of ``auto``, ``dcm2niix``, ``sitk`` (case-insensitive).
    """

    m = (mode or "auto").strip().lower()
    if m == "dcm2niix":
        return "dcm2niix"
    if m == "sitk":
        return "sitk"
    if m == "auto":
        if executable_available(dcm2niix_binary):
            return "dcm2niix"
        try:
            import SimpleITK  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "dicom backend 'auto': dcm2niix is not on PATH and SimpleITK is not installed. "
                "Install dcm2niix, or `pip install SimpleITK` and use --dicom-backend sitk."
            ) from e
        return "sitk"
    raise ValueError(f"Unknown --dicom-backend {mode!r} (use auto, dcm2niix, or sitk).")
