#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${AXIS_VENV_DIR:-${REPO_ROOT}/.venv}"
VENV_PYTHON="${VENV_DIR}/bin/python"
AXIS_BIN="${VENV_DIR}/bin/axis-pn"
ENV_FILE="${REPO_ROOT}/dev/axis_local_env.sh"

KITS_ROOT_DEFAULT="${HOME}/Desktop/kits_data/C4KC-KiTS-NBIA-manifest (1)/c4kc_kits"
WEIGHTS_DIR_DEFAULT="${REPO_ROOT}/pnvrn_folds"
WORK_ROOT_DEFAULT="${REPO_ROOT}/local-runs"
DEVICE_DEFAULT="cpu"

if [[ ! -x "${AXIS_BIN}" ]]; then
  echo "Missing ${AXIS_BIN}."
  echo "Create the env first (default interpreter is python3.10; set AXIS_PYTHON if needed):"
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

if [[ -n "${AXIS_NNUNET_V1_RAW:-}" ]]; then
  export nnUNet_raw_data_base="${AXIS_NNUNET_V1_RAW}"
fi
if [[ -n "${AXIS_NNUNET_V1_PREPROCESSED:-}" ]]; then
  export nnUNet_preprocessed="${AXIS_NNUNET_V1_PREPROCESSED}"
fi
if [[ -n "${AXIS_NNUNET_V1_RESULTS:-}" ]]; then
  export RESULTS_FOLDER="${AXIS_NNUNET_V1_RESULTS}"
fi
if [[ -n "${AXIS_NNUNET_V2_RAW:-}" ]]; then
  export nnUNet_raw="${AXIS_NNUNET_V2_RAW}"
fi
if [[ -n "${AXIS_NNUNET_V2_RESULTS:-}" ]]; then
  export nnUNet_results="${AXIS_NNUNET_V2_RESULTS}"
fi

REQUEST="${1:-KiTS-00000}"
DEVICE="${AXIS_DEVICE:-$DEVICE_DEFAULT}"
KITS_ROOT="${AXIS_KITS_ROOT:-$KITS_ROOT_DEFAULT}"
WEIGHTS_DIR="${AXIS_WEIGHTS_DIR:-$WEIGHTS_DIR_DEFAULT}"
WORK_ROOT="${AXIS_WORK_ROOT:-$WORK_ROOT_DEFAULT}"

if [[ ! -d "${KITS_ROOT}" ]]; then
  echo "KiTS root not found: ${KITS_ROOT}"
  exit 1
fi

if [[ ! -d "${WEIGHTS_DIR}" ]]; then
  echo "Weights directory not found: ${WEIGHTS_DIR}"
  exit 1
fi

if [[ -z "${RESULTS_FOLDER:-}" || -z "${nnUNet_preprocessed:-}" || -z "${nnUNet_raw_data_base:-}" ]]; then
  echo "nnU-Net setup is not initialized."
  echo "Run this once first:"
  echo "  ./dev/setup_local_models.sh"
  exit 1
fi

run_one_case() {
  local CASE_NAME="$1"
  local RUN_NAME="${AXIS_RUN_NAME:-${CASE_NAME}}"
  local WORK_DIR="${WORK_ROOT}/${RUN_NAME}"

  if [[ ! -d "${KITS_ROOT}/${CASE_NAME}" ]]; then
    echo "Case directory not found: ${KITS_ROOT}/${CASE_NAME}"
    return 1
  fi

  local SERIES_DIR
  SERIES_DIR="$("${VENV_PYTHON}" - <<'PY' "${KITS_ROOT}" "${CASE_NAME}"
from pathlib import Path
import sys

import pydicom

kits_root = Path(sys.argv[1])
case_name = sys.argv[2]
case_root = kits_root / case_name


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

# Prefer diagnostic CT volumes. KiTS cases often include a DICOM SEG series; using it
# yields a label/probability map (0–1), not HU — TotalSegmentator and nnU-Net then
# produce empty masks and axis-pn fails with "Primary object not found in mask".
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

  echo "Running axis-pn"
  echo "  case: ${CASE_NAME}"
  echo "  series: ${SERIES_DIR}"
  echo "  work dir: ${WORK_DIR}"
  echo "  weights: ${WEIGHTS_DIR}"
  echo "  device: ${DEVICE}"
  echo "  tumor backend: nnUNet v1 Task135_KiTS2021"

  # Set AXIS_REUSE_CACHED=1 to skip DICOM/TotalSegmentator/tumor when outputs already exist under WORK_DIR (faster iteration on SWP inference).
  #
  # If macOS shows "Python quit unexpectedly" during TotalSegmentator, stderr lines like
  # MallocStackLogging are usually harmless. The crash is often memory pressure on large CTs.
  # Optional: add --force_split for huge volumes only (can break small/cropped scans):
  #   export AXIS_TOTALSEG_EXTRA="-fs -nr 2 -ns 2"
  #
  # Build one argv array so we never expand an empty array under `set -u`.
  local -a AXIS_PREDICT_CMD=(
    "${AXIS_BIN}" predict
    --input "${SERIES_DIR}"
    --work-dir "${WORK_DIR}"
    --weights-dir "${WEIGHTS_DIR}"
    --tumor-mode nnunetv1
    --tumor-task-id 135
    --tumor-model 3d_cascade_fullres
    --device "${DEVICE}"
  )
  if [[ -n "${AXIS_TOTALSEG_EXTRA:-}" ]]; then
    AXIS_PREDICT_CMD+=(--totalseg-extra "${AXIS_TOTALSEG_EXTRA}")
  fi
  if [[ -n "${AXIS_TUMOR_EXTRA:-}" ]]; then
    AXIS_PREDICT_CMD+=(--tumor-extra "${AXIS_TUMOR_EXTRA}")
  fi
  if [[ "${AXIS_REUSE_CACHED:-0}" == "1" ]]; then
    AXIS_PREDICT_CMD+=(--reuse-cached-artifacts)
  fi

  PATH="${VENV_DIR}/bin:${PATH}" "${AXIS_PREDICT_CMD[@]}"

  echo
  echo "Done: ${CASE_NAME}"
  echo "Prediction JSON: ${WORK_DIR}/predictions/predictions.json"
}

run_all_kits_cases() {
  shopt -s nullglob
  local -a dirs
  dirs=("${KITS_ROOT}"/KiTS-*/)
  if [[ ${#dirs[@]} -eq 0 ]]; then
    echo "No KiTS-* case directories under: ${KITS_ROOT}"
    exit 1
  fi
  local -a names=()
  local d
  for d in "${dirs[@]}"; do
    [[ -d "$d" ]] || continue
    names+=("$(basename "${d%/}")")
  done
  if [[ ${#names[@]} -eq 0 ]]; then
    echo "No KiTS-* case directories under: ${KITS_ROOT}"
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
  echo "Running ${#names[@]} cases under ${KITS_ROOT}: ${names[*]}"
  local c
  for c in "${names[@]}"; do
    echo ""
    echo "========== ${c} =========="
    unset AXIS_RUN_NAME || true
    run_one_case "${c}"
  done
  echo ""
  echo "All cases finished (${#names[@]} total)."
}

if [[ "${REQUEST}" == "ALL" || "${REQUEST}" == "--all" || "${REQUEST}" == "-a" ]]; then
  run_all_kits_cases
else
  run_one_case "${REQUEST}"
fi
