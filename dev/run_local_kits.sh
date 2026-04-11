#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv312/bin/python"
AXIS_BIN="${REPO_ROOT}/.venv312/bin/axis-pn"
ENV_FILE="${REPO_ROOT}/dev/axis_local_env.sh"

KITS_ROOT_DEFAULT="${HOME}/Desktop/kits_data/C4KC-KiTS-NBIA-manifest (1)/c4kc_kits"
WEIGHTS_DIR_DEFAULT="${REPO_ROOT}/pnvrn_folds"
WORK_ROOT_DEFAULT="${REPO_ROOT}/local-runs"
DEVICE_DEFAULT="cpu"

if [[ ! -x "${AXIS_BIN}" ]]; then
  echo "Missing ${AXIS_BIN}."
  echo "Create the env first:"
  echo "  python3.12 -m venv .venv312"
  echo "  .venv312/bin/pip install --upgrade pip setuptools wheel"
  echo "  .venv312/bin/pip install -e ."
  echo "  .venv312/bin/pip install TotalSegmentator nnunetv2"
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

CASE_NAME="${1:-KiTS-00000}"
DEVICE="${AXIS_DEVICE:-$DEVICE_DEFAULT}"
KITS_ROOT="${AXIS_KITS_ROOT:-$KITS_ROOT_DEFAULT}"
WEIGHTS_DIR="${AXIS_WEIGHTS_DIR:-$WEIGHTS_DIR_DEFAULT}"
WORK_ROOT="${AXIS_WORK_ROOT:-$WORK_ROOT_DEFAULT}"
RUN_NAME="${AXIS_RUN_NAME:-${CASE_NAME}}"
WORK_DIR="${WORK_ROOT}/${RUN_NAME}"

if [[ ! -d "${KITS_ROOT}/${CASE_NAME}" ]]; then
  echo "Case directory not found: ${KITS_ROOT}/${CASE_NAME}"
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

SERIES_DIR="$("${VENV_PYTHON}" - <<'PY' "${KITS_ROOT}" "${CASE_NAME}"
from pathlib import Path
import sys

kits_root = Path(sys.argv[1])
case_name = sys.argv[2]
case_root = kits_root / case_name

for directory in sorted(case_root.rglob("*")):
    if directory.is_dir():
        try:
            if any(child.is_file() and child.suffix.lower() == ".dcm" for child in directory.iterdir()):
                print(directory)
                break
        except Exception:
            pass
else:
    raise SystemExit(f"No DICOM series directory found under {case_root}")
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
# Build one argv array so we never expand an empty array under `set -u`.
AXIS_PREDICT_CMD=(
  "${AXIS_BIN}" predict
  --input "${SERIES_DIR}"
  --work-dir "${WORK_DIR}"
  --weights-dir "${WEIGHTS_DIR}"
  --tumor-mode nnunetv1
  --tumor-task-id 135
  --tumor-model 3d_cascade_fullres
  --device "${DEVICE}"
)
if [[ "${AXIS_REUSE_CACHED:-0}" == "1" ]]; then
  AXIS_PREDICT_CMD+=(--reuse-cached-artifacts)
fi

PATH="${REPO_ROOT}/.venv312/bin:${PATH}" "${AXIS_PREDICT_CMD[@]}"

echo
echo "Done."
echo "Prediction JSON: ${WORK_DIR}/predictions/predictions.json"
