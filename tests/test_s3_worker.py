from __future__ import annotations

import json
import unittest
from datetime import datetime

from botocore.exceptions import ClientError

from clarity_inference_pipeline.s3_worker import delete_input_objects, list_pending_submissions, write_result_json


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

    def get_paginator(self, name: str):
        if name != "list_objects_v2":
            raise ValueError(name)
        pages = []
        for prefix, keys in self.pages.items():
            contents = [{"Key": key} for key in keys]
            pages.append({"_prefix": prefix, "Contents": contents})
        return _FakePaginator(pages)

    def head_object(self, *, Bucket: str, Key: str):
        if Key in self.objects:
            return {"ETag": "etag"}
        raise ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}, "ResponseMetadata": {"HTTPStatusCode": 404}},
            "HeadObject",
        )

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str):
        self.objects[Key] = Body
        return {"ETag": "etag"}

    def delete_objects(self, *, Bucket: str, Delete: dict):
        keys = [row["Key"] for row in Delete["Objects"]]
        self.deleted_batches.append(keys)
        return {"Deleted": [{"Key": key} for key in keys]}


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
        self.assertEqual([row.submission_id for row in pending], ["sub-1"])

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
            {"submission_id", "status", "clarity_score", "message", "processed_at", "pipeline_version"},
        )
        self.assertEqual(payload["submission_id"], "sub-1")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["clarity_score"], 0.73)
        self.assertEqual(payload["message"], "Processing complete")
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


if __name__ == "__main__":
    unittest.main()
