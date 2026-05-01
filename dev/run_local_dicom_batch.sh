#!/usr/bin/env bash
# Batch runner: one patient folder or ALL cases under a data root (any DICOM layout you use).
# TCIA KiTS19-style folders (e.g. KiTS-00000) are a convenient test case; the same pipeline
# applies to your own datasets — one subdirectory per case, each containing a nested DICOM tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${CLARITY_VENV_DIR:-${REPO_ROOT}/.venv}"
VENV_PYTHON="${VENV_DIR}/bin/python"
CLARITY_BIN="${VENV_DIR}/bin/clarity-pipeline"
ENV_FILE="${REPO_ROOT}/dev/clarity_local_env.sh"

# Default: example tree from C4KC / KiTS; override with CLARITY_DICOM_CASES_ROOT for your data.
DICOM_CASES_ROOT_DEFAULT="${HOME}/Desktop/kits_data/C4KC-KiTS-NBIA-manifest (1)/c4kc_kits"
WEIGHTS_DIR_DEFAULT="${REPO_ROOT}/pnvrn_folds"
WORK_ROOT_DEFAULT="${REPO_ROOT}/local-runs"
DEVICE_DEFAULT="cpu"

if [[ ! -x "${CLARITY_BIN}" ]]; then
  echo "Missing ${CLARITY_BIN}."
  echo "Create the env first (default interpreter is python3.10; set CLARITY_PYTHON if needed):"
  echo "  ./dev/setup_local_models.sh"
  exit 1
fi

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing ${VENV_PYTHON}."
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

if [[ -n "${CLARITY_NNUNET_V1_RAW:-}" ]]; then
  export nnUNet_raw_data_base="${CLARITY_NNUNET_V1_RAW}"
fi
if [[ -n "${CLARITY_NNUNET_V1_PREPROCESSED:-}" ]]; then
  export nnUNet_preprocessed="${CLARITY_NNUNET_V1_PREPROCESSED}"
fi
if [[ -n "${CLARITY_NNUNET_V1_RESULTS:-}" ]]; then
  export RESULTS_FOLDER="${CLARITY_NNUNET_V1_RESULTS}"
fi
if [[ -n "${CLARITY_NNUNET_V2_RAW:-}" ]]; then
  export nnUNet_raw="${CLARITY_NNUNET_V2_RAW}"
fi
if [[ -n "${CLARITY_NNUNET_V2_PREPROCESSED:-}" ]]; then
  export nnUNet_preprocessed="${CLARITY_NNUNET_V2_PREPROCESSED}"
fi
if [[ -n "${CLARITY_NNUNET_V2_RESULTS:-}" ]]; then
  export nnUNet_results="${CLARITY_NNUNET_V2_RESULTS}"
fi

# nnU-Net spills large predictions to $TMPDIR; on Slurm, node-local temp avoids slow BeeGFS.
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="${TMPDIR:-${SLURM_TMPDIR}}"
fi

# Parent directory: each immediate subdirectory is one case/patient (any name).
# CLARITY_KITS_ROOT is a deprecated alias for CLARITY_DICOM_CASES_ROOT.
REQUEST="${1:-ALL}"
DEVICE="${CLARITY_DEVICE:-$DEVICE_DEFAULT}"
DICOM_CASES_ROOT="${CLARITY_DICOM_CASES_ROOT:-${CLARITY_KITS_ROOT:-$DICOM_CASES_ROOT_DEFAULT}}"
WEIGHTS_DIR="${CLARITY_WEIGHTS_DIR:-$WEIGHTS_DIR_DEFAULT}"
WORK_ROOT="${CLARITY_WORK_ROOT:-$WORK_ROOT_DEFAULT}"

if [[ ! -d "${DICOM_CASES_ROOT}" ]]; then
  echo "DICOM cases root not found: ${DICOM_CASES_ROOT}"
  echo "Set CLARITY_DICOM_CASES_ROOT to the parent of your per-patient folders."
  exit 1
fi

if [[ ! -d "${WEIGHTS_DIR}" ]]; then
  echo "Weights directory not found: ${WEIGHTS_DIR}"
  exit 1
fi

if [[ -z "${nnUNet_results:-}" || -z "${nnUNet_preprocessed:-}" || -z "${nnUNet_raw:-}" ]]; then
  echo "nnU-Net v2 setup is not initialized."
  echo "Run this once first:"
  echo "  ./dev/setup_local_models.sh"
  exit 1
fi

run_one_case() {
  local CASE_NAME="$1"
  local RUN_NAME="${CLARITY_RUN_NAME:-${CASE_NAME}}"
  local WORK_DIR="${WORK_ROOT}/${RUN_NAME}"

  if [[ ! -d "${DICOM_CASES_ROOT}/${CASE_NAME}" ]]; then
    echo "Case directory not found: ${DICOM_CASES_ROOT}/${CASE_NAME}"
    return 1
  fi

  local SERIES_DIR
  SERIES_DIR="$("${VENV_PYTHON}" - <<'PY' "${DICOM_CASES_ROOT}" "${CASE_NAME}"
from pathlib import Path
import sys

import pydicom

cases_root = Path(sys.argv[1])
case_name = sys.argv[2]
case_root = cases_root / case_name


def modality_for_series_dir(series_dir: Path) -> str | None:
    """Modality from the first .dcm file in the directory (header read only)."""

    for child in sorted(series_dir.iterdir()):
        if child.is_file() and child.suffix.lower() == ".dcm":
            try:
                ds = pydicom.dcmread(child, stop_before_pixels=True, force=True)
                m = getattr(ds, "Modality", None)
                return str(m).strip() if m else None
            except Exception:
                continue
    return None


candidates: list[tuple[Path, str | None]] = []
for directory in sorted(case_root.rglob("*")):
    if not directory.is_dir():
        continue
    try:
        if not any(
            child.is_file() and child.suffix.lower() == ".dcm" for child in directory.iterdir()
        ):
            continue
    except OSError:
        continue
    candidates.append((directory, modality_for_series_dir(directory)))

if not candidates:
    raise SystemExit(f"No DICOM series directory found under {case_root}")

# Prefer diagnostic CT. Some collections ship a DICOM SEG series; using it as CT yields
# label maps instead of HU — TotalSegmentator / nnU-Net then produce unusable masks.
non_seg = [(d, m) for d, m in candidates if m != "SEG"]
if not non_seg:
    raise SystemExit(
        f"Only DICOM SEG (or unreadable) series found under {case_root}. "
        "Use a directory whose Modality is CT (the diagnostic CT series), not SEG."
    )

ct_dirs = [(d, m) for d, m in non_seg if m == "CT"]
chosen = sorted(ct_dirs, key=lambda t: str(t[0]))[0][0] if ct_dirs else sorted(non_seg, key=lambda t: str(t[0]))[0][0]
print(chosen)
PY
)"

  mkdir -p "${WORK_ROOT}"

  echo "Running clarity-pipeline"
  echo "  case: ${CASE_NAME}"
  echo "  series: ${SERIES_DIR}"
  echo "  work dir: ${WORK_DIR}"
  echo "  weights: ${WEIGHTS_DIR}"
  echo "  device: ${DEVICE}"
  echo "  tumor backend: nnUNet v2 Dataset123_Kits23 3d_fullres (KiTS23-trained checkpoint)"

  # Set CLARITY_REUSE_CACHED=1 to skip DICOM/TotalSegmentator/tumor when outputs already exist under WORK_DIR (faster iteration on SWP inference).
  #
  # If macOS shows "Python quit unexpectedly" during TotalSegmentator, stderr lines like
  # MallocStackLogging are usually harmless. The crash is often memory pressure on large CTs.
  # Optional: add --force_split for huge volumes only (can break small/cropped scans):
  #   export CLARITY_TOTALSEG_EXTRA="-fs -nr 2 -ns 2"
  #
  # DICOM→NIfTI: default --dicom-backend auto (dcm2niix if on PATH, else SimpleITK). Set CLARITY_DICOM_BACKEND=sitk to force in-process conversion without dcm2niix.
  #
  # Build one argv array so we never expand an empty array under `set -u`.
  local _db="${CLARITY_DICOM_BACKEND:-auto}"
  local -a CLARITY_PREDICT_CMD=(
    "${CLARITY_BIN}" predict
    --input "${SERIES_DIR}"
    --work-dir "${WORK_DIR}"
    --weights-dir "${WEIGHTS_DIR}"
    --tumor-mode nnunetv2
    --tumor-dataset-id Dataset123_Kits23
    --tumor-configuration 3d_fullres
    --device "${DEVICE}"
    --dicom-backend "${_db}"
  )
  if [[ -n "${CLARITY_DCM2NIIX:-}" ]]; then
    CLARITY_PREDICT_CMD+=(--dcm2niix "${CLARITY_DCM2NIIX}")
  elif [[ "${_db}" == "dcm2niix" ]] && ! command -v dcm2niix >/dev/null 2>&1; then
    echo "dcm2niix not on PATH and CLARITY_DCM2NIIX not set; use --dicom-backend auto (default) or sitk." >&2
    return 1
  fi
  if [[ -n "${CLARITY_TOTALSEG_EXTRA:-}" ]]; then
    CLARITY_PREDICT_CMD+=(--totalseg-extra "${CLARITY_TOTALSEG_EXTRA}")
  fi
  if [[ -n "${CLARITY_TUMOR_EXTRA:-}" ]]; then
    CLARITY_PREDICT_CMD+=(--tumor-extra "${CLARITY_TUMOR_EXTRA}")
  fi
  if [[ "${CLARITY_REUSE_CACHED:-0}" == "1" ]]; then
    CLARITY_PREDICT_CMD+=(--reuse-cached-artifacts)
  fi
  if [[ "${CLARITY_FAIL_ON_EMPTY_TUMOR:-0}" == "1" ]]; then
    CLARITY_PREDICT_CMD+=(--fail-on-empty-tumor)
  fi

  PATH="${VENV_DIR}/bin:${PATH}" "${CLARITY_PREDICT_CMD[@]}"

  echo
  echo "Done: ${CASE_NAME}"
  echo "Prediction JSON: ${WORK_DIR}/predictions/predictions.json"
}

run_all_cases_under_root() {
  shopt -s nullglob
  local -a dirs
  dirs=("${DICOM_CASES_ROOT}"/*/)
  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "No case subdirectories under: ${DICOM_CASES_ROOT}"
    exit 1
  fi
  local -a names=()
  local d
  for d in "${dirs[@]}"; do
    [[ -d "$d" ]] || continue
    names+=("$(basename "${d%/}")")
  done
  if [[ ${#names[@]} -eq 0 ]]; then
    echo "No case subdirectories under: ${DICOM_CASES_ROOT}"
    exit 1
  fi
  local _sorted
  local -a names_sorted=()
  _sorted="$(printf '%s\n' "${names[@]}" | sort -V)"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" ]] && continue
    names_sorted+=("${line}")
  done <<< "${_sorted}"
  names=("${names_sorted[@]}")
  echo "Running ${#names[@]} case folder(s) under ${DICOM_CASES_ROOT}: ${names[*]}"
  local c
  for c in "${names[@]}"; do
    echo ""
    echo "========== ${c} =========="
    unset CLARITY_RUN_NAME || true
    run_one_case "${c}"
  done
  echo ""
  echo "All cases finished (${#names[@]} total)."
}

if [[ "${REQUEST}" == "ALL" || "${REQUEST}" == "--all" || "${REQUEST}" == "-a" ]]; then
  run_all_cases_under_root
else
  run_one_case "${REQUEST}"
fi
