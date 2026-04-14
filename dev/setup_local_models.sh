#!/usr/bin/env bash
# Fresh clone / HPC workflow (recommended):
#   module load python/gpu/3.10.6    # example — your site's GPU Python + CUDA paths
#   cd axis-inference-pipeline
#   export AXIS_PYTHON="$(command -v python3.10 2>/dev/null || command -v python3)"
#   ./dev/setup_local_models.sh
#
# PyTorch is installed from PyTorch's CUDA wheel index *before* `pip install -e .` so the
# dependency resolver does not leave you on a `+cpu` build, then installed again *after* to
# overwrite any replacement from PyPI.
#
# AXIS_PYTORCH_CUDA=auto|cpu|cu118|cu121|cu124|skip
#   auto — nvidia-smi "CUDA Version"; Linux + no nvidia-smi → cu118
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AXIS_PYTHON="${AXIS_PYTHON:-python3.10}"
if ! command -v "${AXIS_PYTHON}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${AXIS_PYTHON}" >&2
  echo "Load your module first, e.g. module load python/gpu/3.10.6" >&2
  echo "Then: export AXIS_PYTHON=\"\$(command -v python3.10)\"" >&2
  exit 1
fi
VENV_DIR="${REPO_ROOT}/.venv"
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
  "${AXIS_PYTHON}" -m venv "${VENV_DIR}"
fi

"${PIP_BIN}" install --upgrade pip "setuptools<82" wheel

AXIS_PYTORCH_CUDA="${AXIS_PYTORCH_CUDA:-auto}"
TORCH_VARIANT=""

_install_torch_wheels() {
  local variant="$1"
  case "${variant}" in
    cpu)
      "${PIP_BIN}" install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
      ;;
    cu118)
      "${PIP_BIN}" install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/cu118"
      ;;
    cu121)
      "${PIP_BIN}" install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/cu121"
      ;;
    cu124)
      "${PIP_BIN}" install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/cu124"
      ;;
    *)
      echo "Unknown PyTorch variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

if [[ "${AXIS_PYTORCH_CUDA}" != "skip" ]]; then
  if [[ "${AXIS_PYTORCH_CUDA}" == "auto" ]]; then
    TORCH_VARIANT="$("${PYTHON_BIN}" - <<'PY'
import re
import subprocess
import sys

def main() -> None:
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL, timeout=60)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if sys.platform == "linux":
            print("cu118")
        else:
            print("cpu")
        return
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        print("cu118")
        return
    major, minor = int(m.group(1)), int(m.group(2))
    if (major, minor) >= (12, 4):
        print("cu124")
    elif (major, minor) >= (12, 1):
        print("cu121")
    else:
        print("cu118")

if __name__ == "__main__":
    main()
PY
)"
  else
    TORCH_VARIANT="${AXIS_PYTORCH_CUDA}"
  fi
  echo "AXIS_PYTORCH_CUDA (resolved)=${TORCH_VARIANT}"
  echo "Installing torch/torchvision (${TORCH_VARIANT}) before editable install…"
  _install_torch_wheels "${TORCH_VARIANT}"
fi

echo "Installing axis-inference-pipeline (editable)…"
"${PIP_BIN}" install -e .

if [[ "${AXIS_PYTORCH_CUDA}" != "skip" ]]; then
  echo "Re-installing torch/torchvision (${TORCH_VARIANT}) after editable install (prevents +cpu from PyPI)…"
  _install_torch_wheels "${TORCH_VARIANT}"
fi

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
echo "Python: ${AXIS_PYTHON} → ${VENV_DIR}"
echo "Environment file: ${REPO_ROOT}/dev/axis_local_env.sh"
echo "Legacy KiTS model root: ${AXIS_NNUNET_V1_RESULTS}"
echo "TotalSegmentator cache: ${TOTALSEG_HOME_DIR}"
echo
export TORCH_VARIANT="${TORCH_VARIANT:-}"
"${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

print(
    "PyTorch:",
    torch.__version__,
    "| cuda build:",
    torch.version.cuda,
    "| cuda available now:",
    torch.cuda.is_available(),
)
tv = os.environ.get("TORCH_VARIANT", "")
if tv in ("cu118", "cu121", "cu124") and sys.platform == "linux":
    if torch.version.cuda is None:
        print(
            "ERROR: torch is still CPU-only (+cpu) after setup. "
            "Try: .venv/bin/pip install -U torch torchvision --index-url https://download.pytorch.org/whl/"
            + tv,
            file=sys.stderr,
        )
        sys.exit(1)
PY
