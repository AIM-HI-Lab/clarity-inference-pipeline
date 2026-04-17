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

from clarity_inference_pipeline.totalsegmentator import _write_kidney_binary_mask


@unittest.skipIf(nib is None or np is None, "nibabel is not installed")
class TotalSegmentatorTests(unittest.TestCase):
    def test_write_kidney_binary_mask_combines_left_and_right(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            affine = np.eye(4)

            left = np.zeros((3, 3, 3), dtype=np.uint8)
            right = np.zeros((3, 3, 3), dtype=np.uint8)
            left[0, 0, 0] = 1
            right[2, 2, 2] = 1

            nib.save(nib.Nifti1Image(left, affine), str(root / "kidney_left.nii.gz"))
            nib.save(nib.Nifti1Image(right, affine), str(root / "kidney_right.nii.gz"))

            _write_kidney_binary_mask(root)

            combined = np.asarray(nib.load(str(root / "kidney_binary_mask.nii.gz")).get_fdata())
            self.assertEqual(int(combined[0, 0, 0]), 1)
            self.assertEqual(int(combined[2, 2, 2]), 1)


if __name__ == "__main__":
    unittest.main()
