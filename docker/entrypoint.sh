#!/usr/bin/env bash
set -euo pipefail
# Consistent nnU-Net v1 + TotalSegmentator paths inside the container.
export nnUNet_raw_data_base="${nnUNet_raw_data_base:-/opt/nnunet/v1/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/opt/nnunet/v1/preprocessed}"
export RESULTS_FOLDER="${RESULTS_FOLDER:-/opt/nnunet/v1/results}"
export TOTALSEG_HOME_DIR="${TOTALSEG_HOME_DIR:-/opt/totalsegmentator}"
exec axis-pn "$@"
