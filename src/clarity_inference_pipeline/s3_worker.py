"""S3 worker for CLARITY Upload Portal submissions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import boto3
import typer
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from . import __version__
from .config import (
    InferenceConfig,
    MaskAdaptationConfig,
    PhaseGatingConfig,
    TotalSegmentatorConfig,
    TumorSegmentationConfig,
)
from .pipeline import build_pipeline_config, run_pipeline

app = typer.Typer(name="clarity-s3-worker", help="Run CLARITY inference over S3 submissions.")


TRANSIENT_ERROR_CODES = {
    "RequestTimeout",
    "RequestTimeoutException",
    "Throttling",
    "ThrottlingException",
    "SlowDown",
    "ServiceUnavailable",
    "InternalError",
    "InternalServerError",
}


@dataclass(frozen=True)
class SubmissionPaths:
    submission_id: str
    base_prefix: str
    input_prefix: str
    manifest_key: str
    result_key: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log(level: str, event: str, **fields: Any) -> None:
    payload = {"ts": _utc_now_iso(), "level": level, "event": event, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)


def _normalize_prefix(prefix_root: str) -> str:
    return prefix_root.strip("/").rstrip("/")


def _submission_paths(prefix_root: str, submission_id: str) -> SubmissionPaths:
    root = _normalize_prefix(prefix_root)
    base = f"{root}/{submission_id}"
    return SubmissionPaths(
        submission_id=submission_id,
        base_prefix=base,
        input_prefix=f"{base}/input/",
        manifest_key=f"{base}/manifest.json",
        result_key=f"{base}/result.json",
    )


def _is_transient_s3_error(exc: Exception) -> bool:
    if isinstance(exc, ClientError):
        code = (exc.response.get("Error") or {}).get("Code", "")
        status = (exc.response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        return code in TRANSIENT_ERROR_CODES or status in {429, 500, 502, 503, 504}
    return isinstance(exc, BotoCoreError)


def _with_retries(func, *, attempts: int = 5, base_sleep_seconds: float = 1.0):
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts or not _is_transient_s3_error(exc):
                raise
            sleep_for = base_sleep_seconds * (2 ** (attempt - 1))
            _log(
                "warning",
                "s3_retry",
                attempt=attempt,
                max_attempts=attempts,
                sleep_seconds=sleep_for,
                error=str(exc),
            )
            time.sleep(sleep_for)
    raise RuntimeError("unreachable")


def _s3_key_exists(s3_client: Any, *, bucket: str, key: str) -> bool:
    def _call() -> bool:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            code = (exc.response.get("Error") or {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    return _with_retries(_call)


def _iter_submission_ids(s3_client: Any, *, bucket: str, prefix_root: str) -> list[str]:
    prefix = f"{_normalize_prefix(prefix_root)}/"
    paginator = s3_client.get_paginator("list_objects_v2")

    submission_ids: set[str] = set()
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix) :]
            if not remainder:
                continue
            submission_id = remainder.split("/", 1)[0]
            if submission_id:
                submission_ids.add(submission_id)
    return sorted(submission_ids)


def _submission_has_dicom(s3_client: Any, *, bucket: str, input_prefix: str) -> bool:
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=input_prefix)
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.lower().endswith(".dcm"):
                return True
    return False


def list_pending_submissions(s3_client: Any, *, bucket: str, prefix_root: str) -> list[SubmissionPaths]:
    pending: list[SubmissionPaths] = []
    for submission_id in _iter_submission_ids(s3_client, bucket=bucket, prefix_root=prefix_root):
        paths = _submission_paths(prefix_root, submission_id)
        has_result = _s3_key_exists(s3_client, bucket=bucket, key=paths.result_key)
        if has_result:
            continue
        has_dcm = _submission_has_dicom(s3_client, bucket=bucket, input_prefix=paths.input_prefix)
        if has_dcm:
            pending.append(paths)
    return pending


def _result_payload(
    *,
    submission_id: str,
    status: str,
    pipeline_version: str,
    message: str,
    clarity_score: float | None,
) -> dict[str, Any]:
    return {
        "submission_id": submission_id,
        "status": status,
        "clarity_score": clarity_score,
        "message": message,
        "processed_at": _utc_now_iso(),
        "pipeline_version": pipeline_version,
    }


def write_result_json(
    s3_client: Any,
    *,
    bucket: str,
    result_key: str,
    submission_id: str,
    status: str,
    pipeline_version: str,
    message: str,
    clarity_score: float | None,
) -> None:
    payload = _result_payload(
        submission_id=submission_id,
        status=status,
        pipeline_version=pipeline_version,
        message=message,
        clarity_score=clarity_score,
    )
    body = json.dumps(payload, indent=2).encode("utf-8")
    _with_retries(
        lambda: s3_client.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=body,
            ContentType="application/json",
        )
    )


def _download_submission_input(
    s3_client: Any,
    *,
    bucket: str,
    input_prefix: str,
    local_input_root: Path,
) -> int:
    paginator = s3_client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=input_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(input_prefix) :]
            if not rel:
                continue
            dst = local_input_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            _with_retries(lambda: s3_client.download_file(bucket, key, str(dst)))
            count += 1
    return count


def _extract_clarity_score(predictions_path: Path) -> float:
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    rows = payload.get("val_rows") or []
    scores: list[float] = []
    for row in rows:
        probs = row.get("pred_probs")
        if isinstance(probs, list) and len(probs) >= 2:
            scores.append(float(probs[1]))
    if not scores:
        raise ValueError("No prediction probabilities found in predictions output.")
    return float(sum(scores) / len(scores))


def delete_input_objects(s3_client: Any, *, bucket: str, input_prefix: str) -> int:
    paginator = s3_client.get_paginator("list_objects_v2")
    to_delete: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=input_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.startswith(input_prefix):
                continue
            to_delete.append({"Key": key})

    deleted = 0
    for idx in range(0, len(to_delete), 1000):
        batch = to_delete[idx : idx + 1000]
        if not batch:
            continue
        resp = _with_retries(lambda: s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch}))
        deleted += len(resp.get("Deleted", []))
    return deleted


def _safe_error_message(exc: Exception) -> str:
    msg = str(exc).strip().replace("\n", " ")
    if not msg:
        msg = exc.__class__.__name__
    return msg[:300]


def _process_submission(
    s3_client: Any,
    *,
    bucket: str,
    submission: SubmissionPaths,
    work_root: Path,
    weights_dir: Path,
    device: str,
    dicom_backend: str,
    dcm2niix_binary: str,
    pipeline_version: str,
    delete_input_after_success: bool,
) -> None:
    write_result_json(
        s3_client,
        bucket=bucket,
        result_key=submission.result_key,
        submission_id=submission.submission_id,
        status="processing",
        pipeline_version=pipeline_version,
        message="Processing started",
        clarity_score=None,
    )

    with TemporaryDirectory(prefix=f"clarity-submission-{submission.submission_id}-", dir=work_root) as tmp:
        submission_root = Path(tmp)
        input_root = submission_root / "input"
        pipeline_workspace = submission_root / "workspace"
        input_root.mkdir(parents=True, exist_ok=True)
        pipeline_workspace.mkdir(parents=True, exist_ok=True)

        _log("info", "download_start", submission_id=submission.submission_id)
        n_downloaded = _download_submission_input(
            s3_client,
            bucket=bucket,
            input_prefix=submission.input_prefix,
            local_input_root=input_root,
        )
        if n_downloaded == 0:
            raise RuntimeError("No input files downloaded from submission input prefix.")
        _log("info", "download_complete", submission_id=submission.submission_id, files=n_downloaded)

        cfg = build_pipeline_config(
            workspace_root=pipeline_workspace,
            dicom_input=input_root,
            totalsegmentator=TotalSegmentatorConfig(device=device),
            tumor=TumorSegmentationConfig(),
            phase_gating=PhaseGatingConfig(),
            mask_adaptation=MaskAdaptationConfig(),
            inference=InferenceConfig(
                checkpoint_dir=weights_dir,
                checkpoint_dir_recursive=True,
                device=device,
            ),
            skip_tumor=False,
            skip_inference=False,
            reuse_cached_artifacts=False,
            continue_on_empty_tumor=False,
            dicom_backend=dicom_backend,
            dcm2niix_binary=dcm2niix_binary,
        )
        _log("info", "pipeline_start", submission_id=submission.submission_id)
        run_pipeline(cfg)
        _log("info", "pipeline_complete", submission_id=submission.submission_id)

        predictions_path = pipeline_workspace / "predictions" / "predictions.json"
        clarity_score = _extract_clarity_score(predictions_path)
        write_result_json(
            s3_client,
            bucket=bucket,
            result_key=submission.result_key,
            submission_id=submission.submission_id,
            status="completed",
            pipeline_version=pipeline_version,
            message="Processing complete",
            clarity_score=clarity_score,
        )
        _log(
            "info",
            "submission_completed",
            submission_id=submission.submission_id,
            clarity_score=clarity_score,
        )

        if delete_input_after_success:
            deleted = delete_input_objects(s3_client, bucket=bucket, input_prefix=submission.input_prefix)
            _log("info", "input_deleted_after_success", submission_id=submission.submission_id, deleted=deleted)


def _run_iteration(
    s3_client: Any,
    *,
    bucket: str,
    prefix_root: str,
    work_root: Path,
    weights_dir: Path,
    device: str,
    dicom_backend: str,
    dcm2niix_binary: str,
    pipeline_version: str,
    max_cases: int | None,
    delete_input_after_success: bool,
    delete_input_after_failure: bool,
) -> int:
    pending = list_pending_submissions(s3_client, bucket=bucket, prefix_root=prefix_root)
    if max_cases is not None:
        pending = pending[:max_cases]
    _log("info", "pending_scan_complete", pending_count=len(pending))

    processed = 0
    for submission in pending:
        processed += 1
        _log("info", "submission_start", submission_id=submission.submission_id)
        try:
            _process_submission(
                s3_client,
                bucket=bucket,
                submission=submission,
                work_root=work_root,
                weights_dir=weights_dir,
                device=device,
                dicom_backend=dicom_backend,
                dcm2niix_binary=dcm2niix_binary,
                pipeline_version=pipeline_version,
                delete_input_after_success=delete_input_after_success,
            )
        except Exception as exc:  # noqa: BLE001
            err = _safe_error_message(exc)
            _log("error", "submission_failed", submission_id=submission.submission_id, error=err)
            write_result_json(
                s3_client,
                bucket=bucket,
                result_key=submission.result_key,
                submission_id=submission.submission_id,
                status="failed",
                pipeline_version=pipeline_version,
                message=err,
                clarity_score=None,
            )
            if delete_input_after_failure:
                deleted = delete_input_objects(s3_client, bucket=bucket, input_prefix=submission.input_prefix)
                _log(
                    "info",
                    "input_deleted_after_failure",
                    submission_id=submission.submission_id,
                    deleted=deleted,
                )
    return processed


@app.command("run")
def run(
    bucket: str = typer.Option(..., "--bucket", envvar="CLARITY_S3_BUCKET"),
    region: str = typer.Option("us-east-1", "--region", envvar="AWS_DEFAULT_REGION"),
    prefix_root: str = typer.Option(
        "clarity/submissions",
        "--prefix-root",
        envvar="CLARITY_S3_PREFIX_ROOT",
    ),
    work_root: Path = typer.Option(
        Path("/tmp/clarity-s3-worker"),
        "--work-root",
        envvar="CLARITY_WORK_ROOT",
        file_okay=False,
    ),
    weights_dir: Path = typer.Option(
        ...,
        "--weights-dir",
        envvar="CLARITY_WEIGHTS_DIR",
        exists=True,
        file_okay=False,
    ),
    device: str = typer.Option("cpu", "--device", envvar="CLARITY_DEVICE"),
    dicom_backend: str = typer.Option(
        "sitk",
        "--dicom-backend",
        envvar="CLARITY_DICOM_BACKEND",
        help="DICOM conversion backend: sitk or dcm2niix.",
    ),
    dcm2niix_binary: str = typer.Option(
        "dcm2niix",
        "--dcm2niix",
        envvar="CLARITY_DCM2NIIX",
        help="dcm2niix executable path/name when backend=dcm2niix.",
    ),
    poll_seconds: int = typer.Option(30, "--poll-seconds", envvar="CLARITY_S3_POLL_SECONDS"),
    once: bool = typer.Option(False, "--once", help="Run one scan/process cycle and exit."),
    max_cases: int | None = typer.Option(
        None,
        "--max-cases",
        envvar="CLARITY_S3_MAX_CASES",
        min=1,
    ),
    delete_input_after_success: bool = typer.Option(
        False,
        "--delete-input-after-success/--no-delete-input-after-success",
        envvar="CLARITY_DELETE_INPUT_AFTER_SUCCESS",
    ),
    delete_input_after_failure: bool = typer.Option(
        False,
        "--delete-input-after-failure/--no-delete-input-after-failure",
        envvar="CLARITY_DELETE_INPUT_AFTER_FAILURE",
    ),
    pipeline_version: str = typer.Option(
        __version__,
        "--pipeline-version",
        envvar="CLARITY_PIPELINE_VERSION",
    ),
) -> None:
    """Run S3 worker once or continuously."""

    work_root.mkdir(parents=True, exist_ok=True)

    boto_config = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "standard"},
    )
    s3_client = boto3.client("s3", region_name=region, config=boto_config)

    _log(
        "info",
        "worker_start",
        bucket=bucket,
        region=region,
        prefix_root=prefix_root,
        once=once,
        poll_seconds=poll_seconds,
        max_cases=max_cases,
        dicom_backend=dicom_backend,
        dcm2niix_binary=dcm2niix_binary,
    )

    while True:
        _run_iteration(
            s3_client,
            bucket=bucket,
            prefix_root=prefix_root,
            work_root=work_root,
            weights_dir=weights_dir,
            device=device,
            dicom_backend=dicom_backend,
            dcm2niix_binary=dcm2niix_binary,
            pipeline_version=pipeline_version,
            max_cases=max_cases,
            delete_input_after_success=delete_input_after_success,
            delete_input_after_failure=delete_input_after_failure,
        )
        if once:
            _log("info", "worker_exit_once")
            return
        time.sleep(poll_seconds)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
