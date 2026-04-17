"""Run workspace layout and manifest writing."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def default_workspace_layout(root: Path) -> dict[str, Path]:
    """Standard subdirectories under ``root``."""

    root = root.resolve()
    return {
        "root": root,
        "cases": root / "cases",
        "dicom_staging": root / "dicom_staging",
        "nifti": root / "nifti",
        "totalseg": root / "totalseg",
        "tumor": root / "tumor",
        "phase_gating": root / "phase_gating",
        "masks_adapted": root / "masks_adapted",
        "predictions": root / "predictions",
        "logs": root / "logs",
    }


def ensure_workspace_dirs(paths: Mapping[str, Path]) -> None:
    """Create workspace directories (idempotent)."""

    for key, p in paths.items():
        if key == "root":
            continue
        p.mkdir(parents=True, exist_ok=True)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def write_manifest(
    path: Path,
    *,
    pipeline_version: str,
    steps_completed: list[str],
    artifacts: Mapping[str, Any],
    config_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    """Write a JSON manifest describing the run."""

    payload: dict[str, Any] = {
        "schema": "clarity_inference_pipeline.manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": pipeline_version,
        "steps_completed": list(steps_completed),
        "artifacts": _json_safe(dict(artifacts)),
    }
    if config_snapshot is not None:
        payload["config"] = _json_safe(dict(config_snapshot))

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def ensure_case_workspace(layout: Mapping[str, Path], case_id: str) -> dict[str, Path]:
    """Return and create the canonical per-case workspace used by SWP inference."""

    case_root = layout["cases"] / case_id
    paths = {
        "case_root": case_root,
        "metadata": case_root / "metadata.json",
        "imaging": case_root / "imaging.nii.gz",
        "image_compat": case_root / "image.nii.gz",
        "segmentation": case_root / "segmentation.nii.gz",
        "total_seg": case_root / "total_seg",
        "tumor_segmentation": case_root / "tumor_segmentation_v2.nii.gz",
    }
    case_root.mkdir(parents=True, exist_ok=True)
    paths["total_seg"].mkdir(parents=True, exist_ok=True)
    return paths


def write_case_metadata(path: Path, metadata: Mapping[str, Any]) -> Path:
    """Write per-case metadata JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(metadata)), indent=2), encoding="utf-8")
    return path


def write_swp_manifest(path: Path, *, case_ids: list[str], data_root: Path) -> Path:
    """Write the manifest consumed by vendored `segmentation_weighted_planes` inference."""

    payload = {
        "schema_version": 1,
        "data_root": str(data_root.resolve()),
        "image_filename": "imaging.nii.gz",
        "segmentation_filename": "segmentation.nii.gz",
        "mode": "list",
        "cases": [
            {
                "case_id": case_id,
                "subdir": case_id,
                "label": 0,
            }
            for case_id in case_ids
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
