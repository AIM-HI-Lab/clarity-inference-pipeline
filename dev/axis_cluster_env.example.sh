#!/usr/bin/env bash
# Copy to axis_cluster_env.sh (gitignored) and set paths for Slurm jobs on this cluster.
#   cp dev/axis_cluster_env.example.sh dev/axis_cluster_env.sh
#
export AXIS_KITS_ROOT="/path/to/kits-dicoms/c4kc_kits"
# Optional:
# export CASE_NAME="KiTS-00042"
# export AXIS_WEIGHTS_DIR="/path/to/pnvrn_folds"
# export AXIS_WORK_ROOT="/path/to/scratch/axis-runs"
