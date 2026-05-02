from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pydicom
from botocore.exceptions import ClientError
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, MRImageStorage, generate_uid

from clarity_inference_pipeline.s3_worker import (
    MIN_CT_SLICES,
    _run_iteration,
    _submission_has_dicom,
    delete_input_objects,
    list_pending_submissions,
    write_result_json,
)


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        return [page for page in self._pages if page.get("_prefix") == prefix]


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.pages: dict[str, list[dict]] = {}
        self.deleted_batches: list[list[str]] = []
        self.download_count = 0

    def get_paginator(self, name: str):
        if name != "list_objects_v2":
            raise ValueError(name)
        pages = []
        for prefix, keys in self.pages.items():
            contents = []
            for row in keys:
                if isinstance(row, dict):
                    contents.append(row)
                else:
                    contents.append({"Key": row, "Size": 0})
            pages.append({"_prefix": prefix, "Contents": contents})
        return _FakePaginator(pages)

    def head_object(self, *, Bucket: str, Key: str):
        if Key in self.objects:
            return {"ETag": "etag"}
        raise ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "HeadObject",
        )

    def get_object(self, *, Bucket: str, Key: str):
        if Key in self.objects:
            return {"Body": BytesIO(self.objects[Key])}
        raise ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "GetObject",
        )

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str):
        self.objects[Key] = Body
        return {"ETag": "etag"}

    def delete_objects(self, *, Bucket: str, Delete: dict):
        keys = [row["Key"] for row in Delete["Objects"]]
        self.deleted_batches.append(keys)
        return {"Deleted": [{"Key": key} for key in keys]}

    def download_file(self, bucket: str, key: str, filename: str):
        if key not in self.objects:
            raise FileNotFoundError(key)
        self.download_count += 1
        with open(filename, "wb") as fh:
            fh.write(self.objects[key])


def _dicom_bytes(*, modality: str, study_uid: str, series_uid: str) -> bytes:
    sop_class_uid = CTImageStorage if modality == "CT" else MRImageStorage
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PatientName = "Worker^Test"
    ds.PatientID = "worker-test"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = modality

    buf = BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    return buf.getvalue()


def _add_submission(
    s3: _FakeS3Client,
    submission_id: str,
    objects: dict[str, bytes],
    *,
    sizes: dict[str, int] | None = None,
) -> None:
    root = "clarity/submissions/"
    input_prefix = f"{root}{submission_id}/input/"
    s3.objects.update(objects)
    s3.pages[root] = sorted({*s3.pages.get(root, []), *objects.keys()})
    s3.pages[input_prefix] = [
        {"Key": key, "Size": (sizes or {}).get(key, len(body))}
        for key, body in sorted(objects.items())
        if key.startswith(input_prefix)
    ]


def _add_dicom_series(
    s3: _FakeS3Client,
    submission_id: str,
    *,
    modality: str = "CT",
    count: int = MIN_CT_SLICES,
) -> None:
    study_uid = generate_uid()
    series_uid = generate_uid()
    payload = _dicom_bytes(modality=modality, study_uid=study_uid, series_uid=series_uid)
    objects = {
        f"clarity/submissions/{submission_id}/input/{idx:04d}.dcm": payload
        for idx in range(count)
    }
    _add_submission(s3, submission_id, objects)


def _run_once(s3: _FakeS3Client, *, work_root: Path, **kwargs) -> int:
    weights_dir = work_root / "weights"
    weights_dir.mkdir(exist_ok=True)
    return _run_iteration(
        s3,
        bucket="bucket",
        prefix_root="clarity/submissions",
        work_root=work_root,
        weights_dir=weights_dir,
        device="cpu",
        dicom_backend="sitk",
        dcm2niix_binary="dcm2niix",
        fail_on_empty_tumor=False,
        pipeline_version="test-version",
        max_cases=None,
        delete_input_after_success=False,
        delete_input_after_failure=False,
        auto_select_series=True,
        **kwargs,
    )


def _result_payload(s3: _FakeS3Client, submission_id: str) -> dict:
    key = f"clarity/submissions/{submission_id}/result.json"
    return json.loads(s3.objects[key].decode("utf-8"))


class S3WorkerTests(unittest.TestCase):
    def test_pending_submission_detection(self) -> None:
        s3 = _FakeS3Client()
        s3.pages = {
            "clarity/submissions/": [
                "clarity/submissions/sub-1/manifest.json",
                "clarity/submissions/sub-1/input/a/1.dcm",
                "clarity/submissions/sub-2/input/2.dcm",
                "clarity/submissions/sub-2/result.json",
                "clarity/submissions/sub-3/input/readme.txt",
            ],
            "clarity/submissions/sub-1/input/": ["clarity/submissions/sub-1/input/a/1.dcm"],
            "clarity/submissions/sub-2/input/": ["clarity/submissions/sub-2/input/2.dcm"],
            "clarity/submissions/sub-3/input/": ["clarity/submissions/sub-3/input/readme.txt"],
        }
        s3.objects = {"clarity/submissions/sub-2/result.json": b"{}"}

        pending = list_pending_submissions(s3, bucket="bucket", prefix_root="clarity/submissions")
        self.assertEqual([row.submission_id for row in pending], ["sub-1", "sub-3"])

    def test_result_json_writer_schema(self) -> None:
        s3 = _FakeS3Client()
        write_result_json(
            s3,
            bucket="bucket",
            result_key="clarity/submissions/sub-1/result.json",
            submission_id="sub-1",
            status="completed",
            pipeline_version="v1.0.0",
            message="Processing complete",
            clarity_score=0.73,
        )
        payload = json.loads(s3.objects["clarity/submissions/sub-1/result.json"].decode("utf-8"))

        self.assertEqual(
            set(payload.keys()),
            {
                "submission_id",
                "status",
                "clarity_score",
                "message",
                "error_code",
                "error_message",
                "processed_at",
                "pipeline_version",
            },
        )
        self.assertEqual(payload["submission_id"], "sub-1")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["clarity_score"], 0.73)
        self.assertEqual(payload["message"], "Processing complete")
        self.assertIsNone(payload["error_code"])
        self.assertIsNone(payload["error_message"])
        self.assertEqual(payload["pipeline_version"], "v1.0.0")
        datetime.fromisoformat(payload["processed_at"].replace("Z", "+00:00"))

    def test_cleanup_only_deletes_input_prefix(self) -> None:
        s3 = _FakeS3Client()
        s3.pages = {
            "clarity/submissions/sub-1/input/": [
                "clarity/submissions/sub-1/input/a/1.dcm",
                "clarity/submissions/sub-1/input/b/2.dcm",
                "clarity/submissions/sub-1/manifest.json",
                "clarity/submissions/sub-1/result.json",
            ]
        }

        deleted = delete_input_objects(
            s3, bucket="bucket", input_prefix="clarity/submissions/sub-1/input/"
        )
        self.assertEqual(deleted, 2)
        self.assertEqual(
            s3.deleted_batches,
            [[
                "clarity/submissions/sub-1/input/a/1.dcm",
                "clarity/submissions/sub-1/input/b/2.dcm",
            ]],
        )

    def test_submission_has_dicom_true_when_dcm_extension_present(self) -> None:
        s3 = _FakeS3Client()
        s3.pages = {
            "clarity/submissions/sub-1/input/": [
                {"Key": "clarity/submissions/sub-1/input/IM-0001-0001.dcm", "Size": 1024},
            ]
        }
        self.assertTrue(
            _submission_has_dicom(
                s3,
                bucket="bucket",
                input_prefix="clarity/submissions/sub-1/input/",
            )
        )

    def test_submission_has_dicom_fallback_sampling_success(self) -> None:
        s3 = _FakeS3Client()
        s3.pages = {
            "clarity/submissions/sub-1/input/": [
                {"Key": "clarity/submissions/sub-1/input/IM-0001-0001", "Size": 100},
            ]
        }
        s3.objects = {"clarity/submissions/sub-1/input/IM-0001-0001": b"extensionless-dicom"}

        fake_ds = type("DS", (), {"StudyInstanceUID": "1.2.3"})()
        with patch("clarity_inference_pipeline.s3_worker.pydicom.dcmread", return_value=fake_ds):
            self.assertTrue(
                _submission_has_dicom(
                    s3,
                    bucket="bucket",
                    input_prefix="clarity/submissions/sub-1/input/",
                )
            )

    def test_submission_has_dicom_fallback_sampling_failure(self) -> None:
        s3 = _FakeS3Client()
        s3.pages = {
            "clarity/submissions/sub-1/input/": [
                {"Key": "clarity/submissions/sub-1/input/00000001", "Size": 100},
                {"Key": "clarity/submissions/sub-1/input/readme.txt", "Size": 120},
            ]
        }
        s3.objects = {
            "clarity/submissions/sub-1/input/00000001": b"not-dicom-1",
            "clarity/submissions/sub-1/input/readme.txt": b"not-dicom-2",
        }

        with patch(
            "clarity_inference_pipeline.s3_worker.pydicom.dcmread",
            side_effect=Exception("invalid"),
        ):
            self.assertFalse(
                _submission_has_dicom(
                    s3,
                    bucket="bucket",
                    input_prefix="clarity/submissions/sub-1/input/",
                )
            )

    def test_random_dcm_fails_before_pipeline(self) -> None:
        s3 = _FakeS3Client()
        _add_submission(
            s3,
            "sub-random",
            {"clarity/submissions/sub-random/input/not-really.dcm": b"definitely not a dicom"},
        )

        with TemporaryDirectory() as tmp, patch(
            "clarity_inference_pipeline.s3_worker.run_pipeline"
        ) as run_pipeline_mock:
            _run_once(s3, work_root=Path(tmp))

        result = _result_payload(s3, "sub-random")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "INVALID_DICOM")
        self.assertEqual(result["message"], "One or more DICOM files could not be read.")
        run_pipeline_mock.assert_not_called()

    def test_mr_dicom_fails_wrong_modality_before_pipeline(self) -> None:
        s3 = _FakeS3Client()
        _add_dicom_series(s3, "sub-mr", modality="MR", count=MIN_CT_SLICES)

        with TemporaryDirectory() as tmp, patch(
            "clarity_inference_pipeline.s3_worker.run_pipeline"
        ) as run_pipeline_mock:
            _run_once(s3, work_root=Path(tmp))

        result = _result_payload(s3, "sub-mr")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "WRONG_MODALITY")
        self.assertEqual(result["message"], "Only CT studies are supported.")
        run_pipeline_mock.assert_not_called()

    def test_chest_ct_no_kidneys_maps_to_no_kidneys_detected(self) -> None:
        s3 = _FakeS3Client()
        _add_dicom_series(s3, "sub-no-kidneys")

        with TemporaryDirectory() as tmp, patch(
            "clarity_inference_pipeline.s3_worker.run_pipeline",
            side_effect=RuntimeError(
                "No kidney structures were detected in the CT. The scan may not cover the kidneys."
            ),
        ):
            _run_once(s3, work_root=Path(tmp))

        result = _result_payload(s3, "sub-no-kidneys")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "NO_KIDNEYS_DETECTED")
        self.assertEqual(result["message"], "No kidneys were detected in the CT.")

    def test_empty_tumor_maps_to_no_tumor_detected(self) -> None:
        s3 = _FakeS3Client()
        _add_dicom_series(s3, "sub-no-tumor")

        def _write_empty_tumor_manifest(cfg) -> None:
            manifest = {"artifacts": {"clarity_case_ids": []}}
            (cfg.workspace_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with TemporaryDirectory() as tmp, patch(
            "clarity_inference_pipeline.s3_worker.run_pipeline",
            side_effect=_write_empty_tumor_manifest,
        ):
            _run_once(s3, work_root=Path(tmp))

        result = _result_payload(s3, "sub-no-tumor")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "NO_TUMOR_DETECTED")
        self.assertEqual(result["message"], "No tumor was detected.")

    def test_oversized_submission_fails_before_download_or_pipeline(self) -> None:
        s3 = _FakeS3Client()
        key = "clarity/submissions/sub-big/input/huge.dcm"
        _add_submission(s3, "sub-big", {key: b"not downloaded"}, sizes={key: 10_001})

        with TemporaryDirectory() as tmp, patch(
            "clarity_inference_pipeline.s3_worker.run_pipeline"
        ) as run_pipeline_mock:
            _run_once(s3, work_root=Path(tmp), max_single_object_bytes=10_000)

        result = _result_payload(s3, "sub-big")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "UPLOAD_TOO_LARGE")
        self.assertEqual(s3.download_count, 0)
        run_pipeline_mock.assert_not_called()

    def test_stale_processing_result_becomes_pending_again(self) -> None:
        s3 = _FakeS3Client()
        _add_submission(
            s3,
            "sub-stale",
            {"clarity/submissions/sub-stale/input/1.dcm": b"placeholder"},
        )
        old_processed_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace(
            "+00:00", "Z"
        )
        s3.objects["clarity/submissions/sub-stale/result.json"] = json.dumps(
            {
                "submission_id": "sub-stale",
                "status": "processing",
                "clarity_score": None,
                "message": "Processing started",
                "processed_at": old_processed_at,
                "pipeline_version": "test",
            }
        ).encode("utf-8")
        s3.pages["clarity/submissions/"].append("clarity/submissions/sub-stale/result.json")

        pending = list_pending_submissions(
            s3,
            bucket="bucket",
            prefix_root="clarity/submissions",
            processing_ttl_seconds=60,
        )
        self.assertEqual([row.submission_id for row in pending], ["sub-stale"])

    def test_malformed_predictions_fail_result_instead_of_crashing_worker(self) -> None:
        s3 = _FakeS3Client()
        _add_dicom_series(s3, "sub-bad-preds")

        def _write_malformed_predictions(cfg) -> None:
            manifest = {"artifacts": {"clarity_case_ids": ["case-1"]}}
            (cfg.workspace_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            predictions_dir = cfg.workspace_root / "predictions"
            predictions_dir.mkdir()
            (predictions_dir / "predictions.json").write_text(json.dumps({"cases": [{}]}), encoding="utf-8")

        with TemporaryDirectory() as tmp, patch(
            "clarity_inference_pipeline.s3_worker.run_pipeline",
            side_effect=_write_malformed_predictions,
        ):
            processed = _run_once(s3, work_root=Path(tmp))

        result = _result_payload(s3, "sub-bad-preds")
        self.assertEqual(processed, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "PIPELINE_ERROR")
        self.assertEqual(result["message"], "Processing failed.")


if __name__ == "__main__":
    unittest.main()
