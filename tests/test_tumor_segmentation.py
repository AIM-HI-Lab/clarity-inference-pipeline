from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from axis_inference_pipeline.config import TumorSegmentationConfig
from axis_inference_pipeline.tumor_segmentation import run_tumor_segmentation


class TumorSegmentationTests(unittest.TestCase):
    def test_nnunetv2_wrapper_builds_expected_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "input.nii.gz"
            output = root / "tumor_segmentation_v2.nii.gz"
            image.write_bytes(b"nifti")

            captured: dict[str, object] = {}

            def fake_logged(cmd, **kwargs):  # type: ignore[no-untyped-def]
                captured["cmd"] = cmd
                out_dir = Path(cmd[cmd.index("-o") + 1])
                (out_dir / "tumor_segmentation_v2.nii.gz").write_bytes(b"mask")

                class Result:
                    returncode = 0
                    stdout = ""
                    stderr = ""

                return Result()

            with patch(
                "axis_inference_pipeline.tumor_segmentation.run_subprocess_logged",
                side_effect=fake_logged,
            ):
                run_tumor_segmentation(
                    image,
                    output,
                    TumorSegmentationConfig(mode="nnunetv2"),
                )

            self.assertTrue(output.exists())
            cmd = captured["cmd"]
            self.assertIn("nnUNetv2_predict", cmd)
            self.assertIn("Dataset123_Kits23", cmd)
            self.assertIn("3d_fullres", cmd)


if __name__ == "__main__":
    unittest.main()
