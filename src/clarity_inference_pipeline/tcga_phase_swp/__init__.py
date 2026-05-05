"""Vendored SWP v3 batch inference (ResNet50 multi-patch) for ``tcga_phase`` CT phase classification."""

from .inference_batch import run_inference_on_batch

__all__ = ["run_inference_on_batch"]
