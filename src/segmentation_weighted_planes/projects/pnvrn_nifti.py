import os
from typing import Optional

from segmentation_weighted_planes.projects import TrainingProject


class PNvsRN_NiftiInference(TrainingProject):
    """
    PN vs RN inference layout aligned with KiTS external-validation PNvRN and the
    kidney_radiomics PN vs RN training recipe (binary, PN=0, RN=1).

    Case data is resolved via datasets/nifti_manifest.py using a JSON manifest
    (see SWP_MANIFEST_JSON or --training-inputs-json).
    """

    project_name = "pnvrn_nifti"
    dataset = "nifti_manifest"
    pred_target = "pn-vs-rn"

    replicate = 0
    bag_k = 48
    bags_per_case_val = 16

    image_path_filenames = ["imaging.nii.gz", "segmentation.nii.gz"]

    seg_class_definitions = {
        "primary": [2],
        "secondary": [1],
    }

    bag_mix = {
        "primary": 0.40,
        "boundary": 0.30,
        "secondary": 0.15,
        "background": 0.05,
    }

    def __init__(self):
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


class PNvsRN_KiTS_ExternalVal(PNvsRN_NiftiInference):
    """
    Alias for --project-name kits__pn_vs_rn_external_val (Slurm / legacy scripts).
    Same NIfTI manifest behavior as PNvsRN_NiftiInference.
    """

    project_name = "kits__pn_vs_rn_external_val"
