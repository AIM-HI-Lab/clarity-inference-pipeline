#!/usr/bin/env bash

#!/usr/bin/env bash
#SBATCH --job-name=axis-kits-gpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90000
#SBATCH --gres=gpu:1

# Quick check: GPU driver visibility + whether this venv's PyTorch sees CUDA.
# Run on a GPU compute node (or interactive GPU session), same as your jobs.
#
#   cd /path/to/axis-inference-pipeline
#   ./dev/check_gpu_env.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY} — run ./dev/setup_local_models.sh first." >&2
  exit 1
fi

echo "== CUDA_VISIBLE_DEVICES =="
echo "${CUDA_VISIBLE_DEVICES-<unset>}"

echo
echo "== nvidia-smi (if present) =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "(nvidia-smi not on PATH — not a GPU session?)"
fi

echo
echo "== PyTorch in .venv =="
"${PY}" - <<'PY'
import torch

print("torch.__file__:", torch.__file__)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda (build):", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
else:
    print()
    print("CUDA not available. Common causes:")
    print("  • This shell is not on a GPU node / Slurm job has no GPU (--gres=gpu:1).")
    print("  • PyTorch is CPU-only: reinstall CUDA wheel, e.g.")
    print('    AXIS_PYTORCH_CUDA=cu118 .venv/bin/pip install -U torch torchvision \\')
    print('      --index-url https://download.pytorch.org/whl/cu118')
    print("  • Driver too new/too old for this torch build: try cu118, or match pytorch.org.")
PY
