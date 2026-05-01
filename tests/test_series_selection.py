from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from clarity_inference_pipeline.dicom import DicomSeries, select_best_series


def _mk_series(uid: str, *, modality: str = "CT", slices: int = 50) -> DicomSeries:
    files = tuple(Path(f"/tmp/{uid}_{idx:04d}.dcm") for idx in range(slices))
    return DicomSeries(
        files=files,
        study_instance_uid="1.2.840.1",
        series_instance_uid=uid,
        modality=modality,
        series_dir=Path("/tmp"),
    )


class SeriesSelectionTests(unittest.TestCase):
    def test_drops_non_ct_modalities(self) -> None:
        mr = _mk_series("mr-1", modality="MR", slices=80)
        ct = _mk_series("ct-1", modality="CT", slices=60)
        mapping = {
            ct.files[0]: type("DS", (), {"SOPClassUID": "1.2.840.10008.5.1.4.1.1.2"})(),
        }
        with patch(
            "clarity_inference_pipeline.dicom.pydicom.dcmread",
            side_effect=lambda path, **kwargs: mapping[path],
        ):
            selected, reasons = select_best_series([mr, ct])
        self.assertEqual(selected.series_instance_uid, "ct-1")
        self.assertTrue(any("modality=MR" in reason for reason in reasons))

    def test_drops_series_with_too_few_slices(self) -> None:
        short_ct = _mk_series("ct-short", slices=20)
        good_ct = _mk_series("ct-good", slices=55)
        mapping = {
            short_ct.files[0]: type("DS", (), {"SOPClassUID": "1.2.840.10008.5.1.4.1.1.2"})(),
            good_ct.files[0]: type("DS", (), {"SOPClassUID": "1.2.840.10008.5.1.4.1.1.2"})(),
        }
        with patch(
            "clarity_inference_pipeline.dicom.pydicom.dcmread",
            side_effect=lambda path, **kwargs: mapping[path],
        ):
            selected, _ = select_best_series([short_ct, good_ct])
        self.assertEqual(selected.series_instance_uid, "ct-good")

    def test_prefers_primary_over_derived(self) -> None:
        derived = _mk_series("ct-derived", slices=80)
        primary = _mk_series("ct-primary", slices=60)
        mapping = {
            derived.files[0]: type(
                "DS",
                (),
                {
                    "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
                    "ImageType": ["DERIVED", "SECONDARY"],
                },
            )(),
            primary.files[0]: type(
                "DS",
                (),
                {"SOPClassUID": "1.2.840.10008.5.1.4.1.1.2", "ImageType": ["ORIGINAL", "PRIMARY"]},
            )(),
        }
        with patch(
            "clarity_inference_pipeline.dicom.pydicom.dcmread",
            side_effect=lambda path, **kwargs: mapping[path],
        ):
            selected, _ = select_best_series([derived, primary])
        self.assertEqual(selected.series_instance_uid, "ct-primary")

    def test_prefers_axial_over_coronal_or_sagittal(self) -> None:
        coronal = _mk_series("ct-coronal", slices=70)
        axial = _mk_series("ct-axial", slices=60)
        mapping = {
            coronal.files[0]: type(
                "DS",
                (),
                {
                    "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
                    "ImageType": ["ORIGINAL", "PRIMARY"],
                    "ImageOrientationPatient": [1, 0, 0, 0, 0, 1],
                },
            )(),
            axial.files[0]: type(
                "DS",
                (),
                {
                    "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
                    "ImageType": ["ORIGINAL", "PRIMARY"],
                    "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                },
            )(),
        }
        with patch(
            "clarity_inference_pipeline.dicom.pydicom.dcmread",
            side_effect=lambda path, **kwargs: mapping[path],
        ):
            selected, _ = select_best_series([coronal, axial])
        self.assertEqual(selected.series_instance_uid, "ct-axial")

    def test_prefers_more_slices_as_final_tiebreaker(self) -> None:
        ct_small = _mk_series("ct-small", slices=45)
        ct_large = _mk_series("ct-large", slices=80)
        mapping = {
            ct_small.files[0]: type(
                "DS",
                (),
                {
                    "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
                    "ImageType": ["ORIGINAL", "PRIMARY"],
                    "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                },
            )(),
            ct_large.files[0]: type(
                "DS",
                (),
                {
                    "SOPClassUID": "1.2.840.10008.5.1.4.1.1.2",
                    "ImageType": ["ORIGINAL", "PRIMARY"],
                    "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
                },
            )(),
        }
        with patch(
            "clarity_inference_pipeline.dicom.pydicom.dcmread",
            side_effect=lambda path, **kwargs: mapping[path],
        ):
            selected, _ = select_best_series([ct_small, ct_large])
        self.assertEqual(selected.series_instance_uid, "ct-large")

    def test_raises_descriptive_error_when_none_survive(self) -> None:
        mr = _mk_series("mr", modality="MR", slices=80)
        short_ct = _mk_series("ct-short", modality="CT", slices=10)
        with self.assertRaises(RuntimeError) as ctx:
            select_best_series([mr, short_ct])
        self.assertIn("No CT series with sufficient slices were found.", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
