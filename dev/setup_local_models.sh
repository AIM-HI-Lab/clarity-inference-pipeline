#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default matches typical HPC clusters; override e.g. AXIS_PYTHON=python3.12 on a dev machine.
AXIS_PYTHON="${AXIS_PYTHON:-python3.10}"
if ! command -v "${AXIS_PYTHON}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${AXIS_PYTHON}" >&2
  echo "Install Python 3.10+ or set AXIS_PYTHON to a working executable." >&2
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
"${PIP_BIN}" install -e .

# PyTorch: default `pip` often installs a very new CUDA build (e.g. cu124) that needs a newer
# NVIDIA driver than many shared clusters provide. Reinstall torch/torchvision from the PyTorch
# wheel index that matches the *driver* (see nvidia-smi "CUDA Version: X.Y").
#   AXIS_PYTORCH_CUDA=auto|cpu|cu118|cu121|cu124|skip
#   auto — pick from nvidia-smi; on Linux if nvidia-smi is missing (common on login nodes), use cu118
#   cpu — CPU-only wheels
#   skip — do not touch torch after `pip install -e .`
AXIS_PYTORCH_CUDA="${AXIS_PYTORCH_CUDA:-auto}"
if [[ "${AXIS_PYTORCH_CUDA}" != "skip" ]]; then
  if [[ "${AXIS_PYTORCH_CUDA}" == "auto" ]]; then
    AXIS_PYTORCH_CUDA="$("${PYTHON_BIN}" - <<'PY'
import re
import subprocess

def main() -> None:
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL, timeout=60)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Login nodes often have no GPU and no nvidia-smi; CPU-only torch breaks GPU jobs later.
        # On Linux, default to cu118 so compute nodes can use CUDA (still runs on CPU if no GPU).
        import sys

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
    # Match PyTorch wheel to *maximum* CUDA version the driver supports (nvidia-smi header).
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
    echo "AXIS_PYTORCH_CUDA (resolved)=${AXIS_PYTORCH_CUDA}"
  fi
  case "${AXIS_PYTORCH_CUDA}" in
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
      echo "Unknown AXIS_PYTORCH_CUDA=${AXIS_PYTORCH_CUDA} (use auto, cpu, cu118, cu121, cu124, or skip)" >&2
      exit 1
      ;;
  esac
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
"${PYTHON_BIN}" - <<'PY'
import torch

print(
    "PyTorch:",
    torch.__version__,
    "| cuda built:",
    torch.version.cuda,
    "| cuda available now:",
    torch.cuda.is_available(),
)
if not torch.cuda.is_available():
    print(
        "  (If you use GPU jobs: run setup on a GPU node or set AXIS_PYTORCH_CUDA=cu118 before setup;"
        " or run: dev/check_gpu_env.sh on a compute node.)"
    )
PY
