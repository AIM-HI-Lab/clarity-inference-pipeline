#!/usr/bin/env bash
# Run clarity-pipeline inside Docker (build the image first: docker build -t clarity-inference-pipeline:local .)
#
# Usage:
#   ./dev/docker-predict.sh /path/to/dicom [output_dir [weights_dir]] [-- extra clarity-pipeline args...]
#
# Env:
#   CLARITY_DOCKER_IMAGE   (default clarity-inference-pipeline:local; use clarity-inference-pipeline:gpu for GPU image)
#   CLARITY_DOCKER_GPU=1   add --gpus all
#   CLARITY_DEVICE=cuda|cpu
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${CLARITY_DOCKER_IMAGE:-clarity-inference-pipeline:local}"
DEVICE="${CLARITY_DEVICE:-cpu}"

ALL=("$@")
SEP=-1
for i in "${!ALL[@]}"; do
  if [[ "${ALL[$i]}" == "--" ]]; then
    SEP=$i
    break
  fi
done
if [[ $SEP -ge 0 ]]; then
  BEFORE=("${ALL[@]:0:$SEP}")
  EXTRA=("${ALL[@]:$((SEP + 1))}")
else
  BEFORE=("${ALL[@]}")
  EXTRA=()
fi
set -- "${BEFORE[@]}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/dicom [output_dir [weights_dir]] [-- extra clarity-pipeline args...]" >&2
  exit 1
fi

DICOM="$1"
OUT="${2:-${REPO_ROOT}/data/out}"
WEIGHTS="${3:-${REPO_ROOT}/pnvrn_folds}"

if [[ ! -d "$DICOM" ]]; then
  echo "DICOM path must be a directory (folder of .dcm files), not a single file: $DICOM" >&2
  exit 1
fi
if [[ ! -e "$OUT" ]]; then
  echo "Path does not exist: $OUT" >&2
  exit 1
fi

if [[ ! -d "$WEIGHTS" ]]; then
  echo "Weights directory not found: $WEIGHTS" >&2
  echo "Provide PNvsRN checkpoints (same layout as pnvrn_folds/) or pass a third path." >&2
  exit 1
fi

DICOM="$(cd "$DICOM" && pwd)"
OUT="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"
WEIGHTS="$(cd "$WEIGHTS" && pwd)"

DOCKER_RUN=(docker run --rm)
if [[ "${CLARITY_DOCKER_GPU:-0}" == "1" ]]; then
  DOCKER_RUN+=(--gpus all)
fi

exec "${DOCKER_RUN[@]}" \
  -v "${DICOM}:/data/dicom:ro" \
  -v "${OUT}:/data/out" \
  -v "${WEIGHTS}:/models:ro" \
  "$IMAGE" \
  predict \
  --input /data/dicom \
  --work-dir /data/out \
  --weights-dir /models \
  --device "$DEVICE" \
  "${EXTRA[@]}"
