from abc import abstractmethod
from typing import Any, Dict, Optional

project_registry = {}


class TrainingProject(object):
    @property
    @abstractmethod
    def project_name(self) -> str:
        pass

    @property
    @abstractmethod
    def dataset(self) -> str:
        pass

    @property
    @abstractmethod
    def data_paths(self) -> dict:
        pass

    n_classes: int = 2
    class_of_interest: int = 1
    output_type: str = "integer"

    replicate: int = 0
    sub_n: Optional[int] = None

    sampling_mode: str = "weighted"
    use_seg: bool = True

    seg_class_definitions: Dict[str, list] = {
        "primary": [2],
        "secondary": [1, 3],
    }

    image_path_filenames = ["image.nii.gz", "tumor_segmentation_v2.nii.gz"]

    bag_k: int = 48
    bags_per_case_val: int = 6
    pooling: str = "attn"
    topk: int = 8

    slab_depth: int = 1

    bag_mix: Dict[str, float] = {
        "primary": 0.70,
        "boundary": 0.15,
        "secondary": 0.10,
        "background": 0.05,
    }

    def __init__(self):
        pass

    @staticmethod
    def prediction_target_classifier(labels: dict, training_inputs: dict) -> Optional[int]:
        raise NotImplementedError

    @staticmethod
    def filter_in(labels: dict, training_inputs: dict) -> bool:
        return True

    @staticmethod
    def modify_instance_labels(instance_labels: list) -> list:
        return instance_labels

    @staticmethod
    def reclassify_prediction(pred):
        return pred

    def project_metadata(self) -> dict:
        return {
            "project_name": self.project_name,
            "dataset": self.dataset,
            "replicate": self.replicate,
            "sub_n": self.sub_n,
            "n_classes": self.n_classes,
            "class_of_interest": self.class_of_interest,
            "output_type": self.output_type,
            "sampling_mode": self.sampling_mode,
            "use_seg": self.use_seg,
            "bag_k": self.bag_k,
            "bags_per_case_val": self.bags_per_case_val,
            "pooling": self.pooling,
            "topk": self.topk,
            "slab_depth": self.slab_depth,
            "bag_mix": dict(self.bag_mix) if self.bag_mix else {},
        }

    def v5_sampling_config(self) -> Dict[str, Any]:
        return {
            "bag_k": int(self.bag_k),
            "bags_per_case_val": int(self.bags_per_case_val),
            "bag_mix": dict(self.bag_mix) if self.bag_mix else {},
            "slab_depth": int(self.slab_depth),
        }


def _register_project(instance: TrainingProject):
    project_registry[instance.project_name] = instance


from segmentation_weighted_planes.projects.pnvrn_nifti import (  # noqa: E402
    PNvsRN_KiTS_ExternalVal,
    PNvsRN_NiftiInference,
)
from segmentation_weighted_planes.projects.tcga_phase import TcgaPhaseNiftiGate  # noqa: E402

_register_project(PNvsRN_NiftiInference())
_register_project(PNvsRN_KiTS_ExternalVal())
_register_project(TcgaPhaseNiftiGate())
