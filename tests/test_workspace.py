from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from axis_inference_pipeline.workspace import ensure_case_workspace, write_swp_manifest


class WorkspaceTests(unittest.TestCase):
    def test_ensure_case_workspace_creates_expected_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = {"cases": root / "cases"}
            case = ensure_case_workspace(layout, "case_001")

            self.assertTrue(case["case_root"].exists())
            self.assertTrue(case["total_seg"].exists())
            self.assertEqual(case["imaging"].name, "imaging.nii.gz")
            self.assertEqual(case["segmentation"].name, "segmentation.nii.gz")

    def test_write_swp_manifest_uses_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_swp_manifest(root / "swp_manifest.json", case_ids=["a", "b"], data_root=root / "cases")
            payload = json.loads(path.read_text())

            self.assertEqual(payload["image_filename"], "imaging.nii.gz")
            self.assertEqual(payload["segmentation_filename"], "segmentation.nii.gz")
            self.assertEqual([row["case_id"] for row in payload["cases"]], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
