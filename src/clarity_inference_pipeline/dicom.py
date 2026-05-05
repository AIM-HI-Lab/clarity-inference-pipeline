"""DICOM series discovery and conversion to NIfTI (via external tools)."""

from __future__ import annotations

import os
import shutil
import subprocess
from math import sqrt
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pydicom

from .subprocess_util import run_subprocess_logged


MIN_CT_SLICES = 30


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


def _study_datetime_from_dataset(ds: pydicom.dataset.FileDataset | None) -> datetime:
    """Parse StudyDate (+ StudyTime) for sorting; missing values sort oldest."""

    if ds is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    raw_d = getattr(ds, "StudyDate", None)
    raw_t = getattr(ds, "StudyTime", None)
    if raw_d is None or str(raw_d).strip() in {"", "None"}:
        return datetime.min.replace(tzinfo=timezone.utc)
    d_compact = "".join(ch for ch in str(raw_d) if ch.isdigit())[:8]
    if len(d_compact) != 8:
        return datetime.min.replace(tzinfo=timezone.utc)
    year = int(d_compact[0:4])
    month = int(d_compact[4:6])
    day = int(d_compact[6:8])

    t_compact = "".join(ch for ch in str(raw_t or "") if ch.isdigit())
    hour = minute = second = 0
    micro = 0
    if len(t_compact) >= 6:
        hour = int(t_compact[0:2])
        minute = int(t_compact[2:4])
        second = int(t_compact[4:6])
    if len(t_compact) > 6:
        frac = (t_compact[6:] + "000000")[:6]
        micro = int(frac)

    try:
        return datetime(year, month, day, hour, minute, second, micro, tzinfo=timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


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


def iter_eligible_ct_series_with_scores(
    series_list: list[DicomSeries],
) -> tuple[list[tuple[DicomSeries, pydicom.dataset.FileDataset | None, int, int, int]], list[str]]:
    """
    Filter to CT series that meet pipeline thresholds and attach heuristic scores.

    Returns rows ``(series, first_dataset, primary_score, axial_score, slice_count)`` plus log reasons.
    """

    reasons: list[str] = []
    eligible: list[tuple[DicomSeries, pydicom.dataset.FileDataset | None, int, int, int]] = []

    if not series_list:
        return [], reasons

    for series in series_list:
        ds = _read_first_dataset(series)
        modality = (series.modality or "").strip().upper()
        if modality != "CT":
            reasons.append(
                f"Rejected {series.series_instance_uid}: modality={series.modality or 'unknown'} (requires CT)."
            )
            continue
        if series.file_count < MIN_CT_SLICES:
            reasons.append(
                f"Rejected {series.series_instance_uid}: only {series.file_count} slices (<{MIN_CT_SLICES})."
            )
            continue
        sop_class_uid = str(getattr(ds, "SOPClassUID", "") or "")
        if sop_class_uid in _NON_IMAGE_SOP_CLASS_UIDS:
            reasons.append(
                f"Rejected {series.series_instance_uid}: SOPClassUID={sop_class_uid} is non-image."
            )
            continue

        primary_score = _image_type_primary_score(getattr(ds, "ImageType", None))
        axial_score = _axial_orientation_score(getattr(ds, "ImageOrientationPatient", None))
        eligible.append((series, ds, primary_score, axial_score, series.file_count))
        reasons.append(
            f"Candidate {series.series_instance_uid}: primary_score={primary_score}, "
            f"axial_score={axial_score}, slices={series.file_count}."
        )

    return eligible, reasons


def rank_ct_series_candidates_for_fallback(
    series_list: list[DicomSeries],
) -> tuple[list[DicomSeries], list[str]]:
    """
    Order CT candidates for bounded retries: newest StudyDate first, then imaging heuristics.

    Uses the same primary / axial / slice lexicographic ordering as :func:`select_best_series`
    within ties on study timestamp (missing StudyDate sorts last).
    """

    eligible, reasons = iter_eligible_ct_series_with_scores(series_list)
    if not eligible:
        return [], reasons

    ranked_rows = sorted(
        eligible,
        key=lambda row: (
            -_study_datetime_from_dataset(row[1]).timestamp(),
            -row[2],
            -row[3],
            -row[4],
            row[0].series_instance_uid,
        ),
    )
    reasons.append(
        "Fallback ranking: newest StudyDate (StudyTime when present) first, "
        "then primary_score, axial_score, slice count, then SeriesInstanceUID."
    )
    return [row[0] for row in ranked_rows], reasons


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
            f"than {MIN_CT_SLICES} slices (scouts/localizers only). Upload one contrast-enhanced abdominal CT study folder."
        )

    eligible, filter_reasons = iter_eligible_ct_series_with_scores(series_list)
    reasons.extend(filter_reasons)

    if not eligible:
        raise RuntimeError(
            "No CT series with sufficient slices were found. The uploaded folder may be a patient-level "
            "folder containing only non-CT modalities (MRI, PET, dose reports), or all CT series had fewer "
            f"than {MIN_CT_SLICES} slices (scouts/localizers only). Upload one contrast-enhanced abdominal CT study folder."
        )

    if len(eligible) == 1:
        chosen = eligible[0][0]
        reasons.append(
            f"Selected {chosen.series_instance_uid}: only one CT candidate remained after hard filters."
        )
        return chosen, reasons

    sorted_eligible = sorted(
        eligible,
        key=lambda row: (-row[2], -row[3], -row[4], row[0].series_instance_uid),
    )
    selected = sorted_eligible[0][0]
    reasons.append(
        f"Selected {selected.series_instance_uid} as best CT series "
        f"(primary_score={sorted_eligible[0][2]}, axial_score={sorted_eligible[0][3]}, "
        f"slices={sorted_eligible[0][4]})."
    )
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
