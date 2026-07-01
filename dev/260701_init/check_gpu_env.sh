#!/usr/bin/env bash
# GPU driver + PyTorch CUDA visibility (run on a GPU node or inside a Slurm GPU job).
#
#   ./dev/260701_init/check_gpu_env.sh
#
set -euo pipefail

DEV_INIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${DEV_INIT_DIR}/../.." && pwd)"

if [[ -f "${DEV_INIT_DIR}/clarity_local_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${DEV_INIT_DIR}/clarity_local_env.sh"
fi
PY="${CLARITY_VENV_DIR:-${HOME}/beegfs/env/clarity-inference-pipeline}/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY} — run ./dev/260701_init/setup.sh first." >&2
  exit 1
fi

echo "== Slurm / GPU assignment =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-<unset>}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE-<unset>}"

echo
echo "== nvidia-smi (if present) =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "(nvidia-smi not on PATH — not a GPU session?)"
fi

echo
echo "== PyTorch in ${CLARITY_VENV_DIR:-venv} =="
"${PY}" - <<'PY'
import torch

print("torch.__file__:", torch.__file__)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda (build):", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.version.cuda is None:
    print()
    print(">>> CPU-only PyTorch wheel. Re-run setup with CLARITY_PYTORCH_CUDA=cu126 (or cu124).")
    raise SystemExit(0)

if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    try:
        t = torch.tensor([1.0], device="cuda")
        print("cuda tensor smoke test: OK", t)
    except Exception as e:
        print("cuda tensor smoke test FAILED:", e)
else:
    print()
    print("CUDA build present but torch.cuda.is_available() is False.")
    print("Check partition #SBATCH --gres and module load cuda12.6/toolkit/12.6.2.")
PY
