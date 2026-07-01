"""Tests for TCGA phase torch device resolution on CPU-only hosts."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from clarity_inference_pipeline.tcga_phase_swp.inference_batch import _resolve_torch_device


class TestResolveTorchDevice(unittest.TestCase):
    def test_cuda_requested_but_unavailable_falls_back_to_cpu(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            dev = _resolve_torch_device("cuda")
        self.assertEqual(dev.type, "cpu")

    def test_cpu_explicit(self) -> None:
        dev = _resolve_torch_device("cpu")
        self.assertEqual(dev.type, "cpu")

    def test_none_uses_cpu_when_cuda_missing(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            dev = _resolve_torch_device(None)
        self.assertEqual(dev.type, "cpu")


if __name__ == "__main__":
    unittest.main()
