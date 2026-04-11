"""DICOM series discovery and conversion to NIfTI (via external tools)."""

from __future__ import annotations

import shutil
import subprocess
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
