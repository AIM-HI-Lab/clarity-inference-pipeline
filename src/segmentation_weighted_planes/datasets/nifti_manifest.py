"""
Resolve NIfTI paths + labels from a JSON manifest (dataset-agnostic).

Environment (optional defaults):
  SWP_MANIFEST_JSON  Path to manifest JSON
  SWP_DATA_ROOT      Root directory for case folders

CLI overrides via training_inputs_json (see inference --training-inputs-json):
  manifest          Path to manifest (overrides env)
  data_root         Root for case folders (overrides env)
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from segmentation_weighted_planes.projects import TrainingProject


def _resolve_paths(
    project_class: TrainingProject,
    training_inputs: Dict[str, Any],
) -> Tuple[Path, Path]:
    manifest_s = (
        training_inputs.get("manifest")
        or project_class.data_paths.get("manifest")
        or os.environ.get("SWP_MANIFEST_JSON")
        or ""
    )
    if not manifest_s:
        raise ValueError(
            "Set manifest path: export SWP_MANIFEST_JSON=/path/to/manifest.json "
            "or pass {\"manifest\": \"...\"} in --training-inputs-json."
        )
    manifest = Path(manifest_s).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest is not a file: {manifest}")

    data_root_s = (
        training_inputs.get("data_root")
        or project_class.data_paths.get("data_root")
        or os.environ.get("SWP_DATA_ROOT")
        or ""
    )
    if data_root_s:
        data_root = Path(data_root_s).expanduser().resolve()
    else:
        data_root = manifest.parent

    return manifest, data_root


def _load_manifest(manifest: Path) -> Dict[str, Any]:
    with manifest.open() as f:
        return json.load(f)


def _scan_cases(
    data_root: Path,
    image_key: str,
    seg_key: str,
) -> List[Tuple[str, Path, Path]]:
    out = []
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root is not a directory: {data_root}")
    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        img_p = sub / image_key
        seg_p = sub / seg_key
        if img_p.is_file() and seg_p.is_file():
            out.append((sub.name, img_p, seg_p))
    return out


def get_nifti_manifest_dataset_labels(
    project_class: TrainingProject,
    training_inputs_json: dict,
    image_paths: Optional[List[str]] = None,
):
    """
    Returns the same structure as other dataset helpers:
      img_nii_pths: list of [Path, Path]  (image, mask)
      labels: list[int]
      case_ids: list[str]
    """
    manifest_pth, data_root = _resolve_paths(project_class, training_inputs_json)
    doc = _load_manifest(manifest_pth)
    if not isinstance(doc, dict):
        raise TypeError(f"Manifest must be a JSON object, got {type(doc).__name__}")

    if doc.get("data_root") and not (
        training_inputs_json.get("data_root")
        or (project_class.data_paths.get("data_root") if project_class.data_paths else "")
        or os.environ.get("SWP_DATA_ROOT")
    ):
        dr = Path(doc["data_root"])
        data_root = dr.resolve() if dr.is_absolute() else (manifest_pth.parent / dr).resolve()

    image_key = doc.get("image_filename", (image_paths or project_class.image_path_filenames)[0])
    seg_key = doc.get("segmentation_filename", (image_paths or project_class.image_path_filenames)[1])
    mode = doc.get("mode", "list")

    img_rows: List[List[Path]] = []
    labels: List[int] = []
    case_ids: List[str] = []
    missing: List[str] = []
    missing_label: List[str] = []

    if mode == "scan":
        scanned = _scan_cases(data_root, image_key, seg_key)
        for case_id, img_p, seg_p in tqdm(scanned, desc="Scanning NIfTI case folders"):
            img_rows.append([img_p, seg_p])
            case_ids.append(case_id)
            labels.append(0)
    else:
        cases = doc.get("cases")
        if not cases:
            raise ValueError('Manifest must contain "cases" array, or set "mode": "scan".')
        declared_ids = []
        for entry in cases:
            if not isinstance(entry, dict):
                raise TypeError(f"Each case entry must be an object, got {type(entry).__name__}")
            if "case_id" not in entry:
                raise KeyError('Each case entry must include "case_id"')
            declared_ids.append(str(entry["case_id"]))
        if len(set(declared_ids)) != len(declared_ids):
            dup = [k for k, v in Counter(declared_ids).items() if v > 1]
            raise ValueError(f"Duplicate case_id values in manifest: {dup[:20]}")

        for entry in tqdm(cases, desc="NIfTI manifest cases"):
            case_id = str(entry["case_id"])
            sub = str(entry.get("subdir", case_id))
            img_p = data_root / sub / image_key
            seg_p = data_root / sub / seg_key
            if not img_p.is_file() or not seg_p.is_file():
                missing.append(case_id)
                continue
            lab = entry.get("label", None)
            if lab is None:
                missing_label.append(case_id)
                lab = 0
            try:
                lab_i = int(lab)
            except (TypeError, ValueError) as e:
                raise ValueError(f'Invalid "label" for case_id={case_id!r}: {lab!r}') from e
            img_rows.append([img_p, seg_p])
            case_ids.append(case_id)
            labels.append(lab_i)

    if missing:
        print(f"Missing imaging/segmentation for {len(missing)} case(s): {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if missing_label:
        print(
            f"No ground-truth label for {len(missing_label)} case(s); "
            f"using placeholder label 0 for caching/metrics: {missing_label[:20]}{'...' if len(missing_label) > 20 else ''}"
        )

    if not img_rows:
        raise ValueError(
            "No cases to run: manifest produced zero valid (image, segmentation) pairs. "
            "Check data_root, filenames, and paths (list mode: cases[].subdir; scan mode: subfolders under data_root)."
        )

    return {
        "img_nii_pths": img_rows,
        "labels": labels,
        "case_ids": case_ids,
    }
