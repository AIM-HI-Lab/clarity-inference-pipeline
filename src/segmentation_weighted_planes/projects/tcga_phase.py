"""SWP ``TrainingProject`` for 4-class CT phase (tcga_phase / v3 ResNet path)."""

from __future__ import annotations

import os
from typing import Optional

from segmentation_weighted_planes.projects import TrainingProject


class TcgaPhaseNiftiGate(TrainingProject):
    """
    Descriptor for the tcga_phase model layout (corticomedullary / nephrographic / …).

    Patch sampling matches vendored ``tcga_phase_swp`` / CCF: primary label 1 on the kidney mask,
    secondary {1, 3}. Runtime phase checks use ``clarity_inference_pipeline.tcga_phase_gating``;
    this class exists so ``project_registry`` can resolve ``tcga_phase`` if needed.
    """

    project_name = "tcga_phase"
    dataset = "nifti_manifest"
    replicate = 0
    n_classes = 4
    class_of_interest = 1
    output_type = "integer"

    sampling_mode = "weighted"
    use_seg = True

    seg_class_definitions = {
        "primary": [1],
        "secondary": [1, 3],
    }

    image_path_filenames = ["image.nii.gz", "total_seg/kidney_binary_mask.nii.gz"]

    bag_k = 48
    bags_per_case_val = 6

    bag_mix = {
        "primary": 0.70,
        "boundary": 0.15,
        "secondary": 0.10,
        "background": 0.05,
    }

    def __init__(self) -> None:
        super().__init__()

    @property
    def data_paths(self) -> dict:
        return {
            "manifest": os.environ.get("SWP_MANIFEST_JSON", ""),
            "data_root": os.environ.get("SWP_DATA_ROOT", ""),
        }

    @staticmethod
    def prediction_target_classifier(labels: dict, training_inputs: dict) -> Optional[int]:
        return None
