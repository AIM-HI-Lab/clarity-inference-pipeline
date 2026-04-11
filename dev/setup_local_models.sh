#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv312"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

NNUNET_V1_ROOT="${REPO_ROOT}/.nnunetv1"
NNUNET_V2_ROOT="${REPO_ROOT}/.nnunetv2"
TOTALSEG_ROOT="${REPO_ROOT}/.totalsegmentator"

mkdir -p "${NNUNET_V1_ROOT}/raw" "${NNUNET_V1_ROOT}/preprocessed" "${NNUNET_V1_ROOT}/results"
mkdir -p "${NNUNET_V2_ROOT}/raw" "${NNUNET_V2_ROOT}/preprocessed" "${NNUNET_V2_ROOT}/results"
mkdir -p "${TOTALSEG_ROOT}"

cat > "${REPO_ROOT}/dev/axis_local_env.sh" <<EOF
#!/usr/bin/env bash
export AXIS_NNUNET_V1_RAW="${NNUNET_V1_ROOT}/raw"
export AXIS_NNUNET_V1_PREPROCESSED="${NNUNET_V1_ROOT}/preprocessed"
export AXIS_NNUNET_V1_RESULTS="${NNUNET_V1_ROOT}/results"
export AXIS_NNUNET_V2_RAW="${NNUNET_V2_ROOT}/raw"
export AXIS_NNUNET_V2_PREPROCESSED="${NNUNET_V2_ROOT}/preprocessed"
export AXIS_NNUNET_V2_RESULTS="${NNUNET_V2_ROOT}/results"
export TOTALSEG_HOME_DIR="${TOTALSEG_ROOT}"
EOF
chmod +x "${REPO_ROOT}/dev/axis_local_env.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  python3.12 -m venv "${VENV_DIR}"
fi

"${PIP_BIN}" install --upgrade pip "setuptools<82" wheel
"${PIP_BIN}" install -e .
"${PIP_BIN}" install TotalSegmentator nnunetv2 nnunet

source "${REPO_ROOT}/dev/axis_local_env.sh"
export PATH="${VENV_DIR}/bin:${PATH}"
export nnUNet_raw_data_base="${AXIS_NNUNET_V1_RAW}"
export nnUNet_preprocessed="${AXIS_NNUNET_V1_PREPROCESSED}"
export RESULTS_FOLDER="${AXIS_NNUNET_V1_RESULTS}"
export nnUNet_raw="${AXIS_NNUNET_V2_RAW}"
export nnUNet_results="${AXIS_NNUNET_V2_RESULTS}"

totalseg_download_weights -t total || true

MODEL_ZIP="${REPO_ROOT}/.cache/Task135_KiTS2021.zip"
mkdir -p "${REPO_ROOT}/.cache"
if [[ ! -f "${MODEL_ZIP}" ]]; then
  "${PYTHON_BIN}" - <<'PY' "${MODEL_ZIP}"
from pathlib import Path
import sys
import urllib.request

dest = Path(sys.argv[1])
url = "https://zenodo.org/records/5126443/files/Task135_KiTS2021.zip?download=1"
urllib.request.urlretrieve(url, dest)
print(dest)
PY
fi

nnUNet_install_pretrained_model_from_zip "${MODEL_ZIP}"

echo
echo "Setup complete."
echo "Environment file: ${REPO_ROOT}/dev/axis_local_env.sh"
echo "Legacy KiTS model root: ${AXIS_NNUNET_V1_RESULTS}"
echo "TotalSegmentator cache: ${TOTALSEG_HOME_DIR}"
