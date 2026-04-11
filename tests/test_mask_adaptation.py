from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment-dependent
    nib = None
    np = None

from axis_inference_pipeline.config import MaskAdaptationConfig
from axis_inference_pipeline.mask_adaptation import adapt_masks


@unittest.skipIf(nib is None or np is None, "nibabel is not installed")
class MaskAdaptationTests(unittest.TestCase):
    def test_adapt_masks_creates_swp_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.eye(4)

            kidney = np.zeros((4, 4, 4), dtype=np.uint8)
            kidney[1:3, 1:3, 1:3] = 1
            tumor = np.zeros((4, 4, 4), dtype=np.uint8)
            tumor[2, 2, 2] = 2

            kidney_path = root / "kidney_binary_mask.nii.gz"
            tumor_path = root / "tumor_segmentation_v2.nii.gz"
            ref_path = root / "imaging.nii.gz"
            out_path = root / "segmentation.nii.gz"

            nib.save(nib.Nifti1Image(kidney, affine), str(kidney_path))
            nib.save(nib.Nifti1Image(tumor, affine), str(tumor_path))
            nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), affine), str(ref_path))

            result = adapt_masks(
                kidney_mask_path=kidney_path,
                tumor_mask_path=tumor_path,
                output_path=out_path,
                config=MaskAdaptationConfig(reference_image=ref_path),
            )

            data = np.asarray(nib.load(str(result)).get_fdata())
            self.assertEqual(int(data[1, 1, 1]), 1)
            self.assertEqual(int(data[2, 2, 2]), 2)


if __name__ == "__main__":
    unittest.main()
