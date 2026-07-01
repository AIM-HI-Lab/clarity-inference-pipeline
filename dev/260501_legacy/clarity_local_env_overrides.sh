#!/usr/bin/env bash
# Site-specific overrides sourced AFTER dev/clarity_local_env.sh in run_local_dicom_batch.sh
# (and therefore also in dev/slurm_gpu_kits.job which delegates to that script).
# Values here override the repo-local defaults written by setup_local_models.sh.
# This file is gitignored — safe to put cluster paths here.

# ── Shared nnU-Net v2 model store (seshadr2's pre-trained KiTS23 weights) ──
export CLARITY_NNUNET_V2_RESULTS=/home/jonnalr/AIM-HI-Lab/models/radiomics_pipeline_models/nnUNet_v2/nnUNet_results_v2
export CLARITY_NNUNET_V2_RAW=/home/jonnalr/AIM-HI-Lab/models/radiomics_pipeline_models/nnUNet_v2/nnUNet_raw_data_v2
export CLARITY_NNUNET_V2_PREPROCESSED=/home/jonnalr/AIM-HI-Lab/models/radiomics_pipeline_models/nnUNet_v2/nnUNet_preprocessed_v2
