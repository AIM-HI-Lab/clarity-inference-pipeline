#!/usr/bin/env bash
# GPU driver + PyTorch CUDA visibility (run on a GPU node or inside a Slurm GPU job).
#
#   ./dev/check_gpu_env.sh
#
# For Slurm-only access, this script runs at the start of dev/slurm_gpu_kits.job (or run it on an interactive GPU node).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "Missing ${PY} — run ./dev/setup_local_models.sh first." >&2
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
echo "== PyTorch in .venv =="
"${PY}" - <<'PY'
import os
import torch

print("torch.__file__:", torch.__file__)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda (build):", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.version.cuda is None:
    print()
    print(">>> This PyTorch is CPU-only (e.g. version shows '+cpu'). GPU and Slurm assignment are OK;")
    print(">>> only the wheel is wrong — often from `pip install -e .` without the CUDA reinstall step.")
    print(">>> Fix from repo root (pick one index that matches your driver; CUDA 12.x drivers: try cu124):")
    print(">>>   .venv/bin/pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    print(">>> Fallback (older drivers):")
    print(">>>   .venv/bin/pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print(">>> Then re-run dev/slurm_gpu_kits.job or clarity-pipeline on a GPU node.")
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
    print("CUDA build present but torch.cuda.is_available() is False. Common causes:")
    print("  • Job has no GPU: CUDA_VISIBLE_DEVICES empty → fix #SBATCH (--gres=gpu:1 vs --gpus-per-node=1).")
    print("  • Driver/library mismatch: try cu118 wheel; or module load cuda on your cluster.")
    print("  • Wrong node type: partition may not attach GPUs.")
PY

TORCH_LIB="$("${PY}" -c "
import pathlib
import torch
p = pathlib.Path(torch.__file__).resolve().parent / 'lib' / 'libtorch_cuda.so'
print(p if p.is_file() else '')
" 2>/dev/null || true)"
if [[ -n "${TORCH_LIB}" && -f "${TORCH_LIB}" ]]; then
  echo
  echo "== Missing shared libs for libtorch_cuda.so (if any) =="
  if command -v ldd >/dev/null 2>&1; then
    ldd "${TORCH_LIB}" 2>/dev/null | grep -i "not found" || echo "(none reported by ldd)"
  fi
fi
