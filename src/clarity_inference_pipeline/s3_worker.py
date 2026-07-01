"""S3 worker for CLARITY Upload Portal submissions."""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from tempfile import NamedTemporaryFile
from typing import Any

import boto3
import pydicom
import typer
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from . import __version__
from .config import (
    InferenceConfig,
    MaskAdaptationConfig,
    PhaseGatingConfig,
    TcgaPhasePredictionConfig,
    TotalSegmentatorConfig,
    TumorSegmentationConfig,
)
from .dicom import MIN_CT_SLICES, discover_series_roots, rank_ct_series_candidates_for_fallback
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


DEFAULT_MAX_TOTAL_BYTES = 3 * 1024 * 1024 * 1024
DEFAULT_MAX_OBJECT_COUNT = 5000
DEFAULT_MAX_SINGLE_OBJECT_BYTES = 512 * 1024 * 1024
DEFAULT_PROCESSING_TTL_SECONDS = 6 * 60 * 60
DEFAULT_MAX_SERIES_FALLBACK_ATTEMPTS = 20


@dataclass(frozen=True)
class SubmissionPaths:
    submission_id: str
    base_prefix: str
    input_prefix: str
    manifest_key: str
    result_key: str


@dataclass(frozen=True)
class S3InputObject:
    key: str
    size: int


@dataclass(frozen=True)
class SubmissionInventory:
    objects: tuple[S3InputObject, ...]

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def total_bytes(self) -> int:
        return sum(obj.size for obj in self.objects)


@dataclass(frozen=True)
class DicomProbe:
    key: str
    study_instance_uid: str
    series_instance_uid: str
    modality: str


class WorkerFailure(RuntimeError):
    def __init__(self, error_code: str, message: str, error_message: str) -> None:
        super().__init__(error_message)
        self.error_code = error_code
        self.message = message
        self.error_message = error_message


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


def _read_s3_json(s3_client: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    def _call() -> dict[str, Any] | None:
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            code = (exc.response.get("Error") or {}).get("Code", "")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        body = resp["Body"].read()
        return json.loads(body.decode("utf-8"))

    try:
        return _with_retries(_call)
    except Exception as exc:  # noqa: BLE001
        _log("warning", "result_json_unreadable", key=key, error=_safe_error_message(exc))
        return None


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_processing_result_stale(payload: dict[str, Any] | None, *, ttl_seconds: int) -> bool:
    if not payload or payload.get("status") != "processing":
        return False
    processed_at = _parse_iso_datetime(payload.get("processed_at"))
    if processed_at is None:
        return True
    age_seconds = (datetime.now(timezone.utc) - processed_at).total_seconds()
    return age_seconds >= ttl_seconds


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


def _collect_submission_inventory(
    s3_client: Any,
    *,
    bucket: str,
    input_prefix: str,
) -> SubmissionInventory:
    paginator = s3_client.get_paginator("list_objects_v2")
    objects: list[S3InputObject] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=input_prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/") or not key.startswith(input_prefix):
                continue
            rel = key[len(input_prefix) :]
            if not rel:
                continue
            objects.append(S3InputObject(key=key, size=int(obj.get("Size") or 0)))
    return SubmissionInventory(objects=tuple(objects))


def _submission_has_dicom(s3_client: Any, *, bucket: str, input_prefix: str) -> bool:
    paginator = s3_client.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=input_prefix)
    sampled: list[dict[str, Any]] = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key.lower().endswith(".dcm"):
                return True
            if key.endswith("/"):
                continue
            sampled.append(obj)

    # Fallback for extensionless PACS/CD exports: sample small files and probe with pydicom.
    sampled = sorted(sampled, key=lambda row: int(row.get("Size") or 0))[:20]
    for obj in sampled:
        key = obj.get("Key", "")
        try:
            with NamedTemporaryFile(suffix=".dcm") as tmp:
                _with_retries(lambda: s3_client.download_file(bucket, key, tmp.name))
                ds = pydicom.dcmread(tmp.name, stop_before_pixels=True, force=True)
                if getattr(ds, "StudyInstanceUID", None):
                    return True
        except Exception:  # noqa: BLE001
            continue
    return False


def list_pending_submissions(
    s3_client: Any,
    *,
    bucket: str,
    prefix_root: str,
    processing_ttl_seconds: int = DEFAULT_PROCESSING_TTL_SECONDS,
) -> list[SubmissionPaths]:
    pending: list[SubmissionPaths] = []
    for submission_id in _iter_submission_ids(s3_client, bucket=bucket, prefix_root=prefix_root):
        paths = _submission_paths(prefix_root, submission_id)
        has_result = _s3_key_exists(s3_client, bucket=bucket, key=paths.result_key)
        if has_result:
            result_payload = _read_s3_json(s3_client, bucket=bucket, key=paths.result_key)
            if not _is_processing_result_stale(
                result_payload,
                ttl_seconds=processing_ttl_seconds,
            ):
                continue
            _log("warning", "stale_processing_retry", submission_id=submission_id)
        pending.append(paths)
    return pending


def _result_payload(
    *,
    submission_id: str,
    status: str,
    pipeline_version: str,
    message: str,
    clarity_score: float | None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "submission_id": submission_id,
        "status": status,
        "clarity_score": clarity_score,
        "message": message,
        "error_code": error_code,
        "error_message": error_message,
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
    error_code: str | None = None,
    error_message: str | None = None,
) -> None:
    payload = _result_payload(
        submission_id=submission_id,
        status=status,
        pipeline_version=pipeline_version,
        message=message,
        clarity_score=clarity_score,
        error_code=error_code,
        error_message=error_message,
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


def _enforce_upload_caps(
    inventory: SubmissionInventory,
    *,
    max_total_bytes: int,
    max_object_count: int,
    max_single_object_bytes: int,
) -> None:
    if inventory.total_bytes > max_total_bytes:
        raise WorkerFailure(
            "UPLOAD_TOO_LARGE",
            "Upload is too large.",
            f"Submission total size {inventory.total_bytes} bytes exceeds limit {max_total_bytes} bytes.",
        )
    if inventory.object_count > max_object_count:
        raise WorkerFailure(
            "UPLOAD_TOO_LARGE",
            "Upload has too many files.",
            f"Submission object count {inventory.object_count} exceeds limit {max_object_count}.",
        )
    oversized = [obj for obj in inventory.objects if obj.size > max_single_object_bytes]
    if oversized:
        largest = max(oversized, key=lambda obj: obj.size)
        raise WorkerFailure(
            "UPLOAD_TOO_LARGE",
            "Upload contains a file that is too large.",
            f"Object {largest.key} is {largest.size} bytes; limit is {max_single_object_bytes} bytes.",
        )


def _probe_dicom_object(s3_client: Any, *, bucket: str, obj: S3InputObject) -> DicomProbe | None:
    try:
        with NamedTemporaryFile(suffix=".dcm") as tmp:
            _with_retries(lambda: s3_client.download_file(bucket, obj.key, tmp.name))
            ds = pydicom.dcmread(tmp.name, stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001
        return None

    study_uid = getattr(ds, "StudyInstanceUID", None)
    series_uid = getattr(ds, "SeriesInstanceUID", None)
    modality = str(getattr(ds, "Modality", "") or "").strip().upper()
    if not study_uid or not series_uid or not modality:
        return None
    return DicomProbe(
        key=obj.key,
        study_instance_uid=str(study_uid),
        series_instance_uid=str(series_uid),
        modality=modality,
    )


def _validate_dicom_headers(
    s3_client: Any,
    *,
    bucket: str,
    inventory: SubmissionInventory,
) -> None:
    if inventory.object_count == 0:
        raise WorkerFailure(
            "NO_DICOM_FILES",
            "No DICOM files were found.",
            "No objects were present under the submission input prefix.",
        )

    probes: list[DicomProbe] = []
    invalid_dcm_count = 0
    for obj in inventory.objects:
        probe = _probe_dicom_object(s3_client, bucket=bucket, obj=obj)
        if probe is None:
            if obj.key.lower().endswith(".dcm"):
                invalid_dcm_count += 1
            continue
        probes.append(probe)

    if not probes:
        if invalid_dcm_count:
            raise WorkerFailure(
                "INVALID_DICOM",
                "One or more DICOM files could not be read.",
                f"{invalid_dcm_count} .dcm object(s) had unreadable or incomplete DICOM headers.",
            )
        raise WorkerFailure(
            "NO_DICOM_FILES",
            "No DICOM files were found.",
            "No usable DICOM headers were found under the submission input prefix.",
        )

    ct_series_counts: dict[tuple[str, str], int] = {}
    modalities = sorted({probe.modality for probe in probes})
    for probe in probes:
        if probe.modality == "CT":
            key = (probe.study_instance_uid, probe.series_instance_uid)
            ct_series_counts[key] = ct_series_counts.get(key, 0) + 1

    if not ct_series_counts:
        raise WorkerFailure(
            "WRONG_MODALITY",
            "Only CT studies are supported.",
            f"No CT DICOM series found. Modalities present: {', '.join(modalities)}.",
        )
    if max(ct_series_counts.values()) < MIN_CT_SLICES:
        raise WorkerFailure(
            "INSUFFICIENT_CT_SERIES",
            "The CT series does not have enough slices.",
            f"Largest CT series has {max(ct_series_counts.values())} slice(s); "
            f"minimum is {MIN_CT_SLICES}.",
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
    scores: list[float] = []

    # Current SWP prediction-only payload shape:
    # { "cases": [ {"ensemble_pred_probs": [...], ...}, ... ], ... }
    for row in payload.get("cases") or []:
        probs = row.get("ensemble_pred_probs")
        if isinstance(probs, list) and len(probs) >= 2:
            scores.append(float(probs[1]))

    # Backward-compatibility with older payload shape:
    # { "val_rows": [ {"pred_probs": [...], ...}, ... ] }
    if not scores:
        for row in payload.get("val_rows") or []:
            probs = row.get("pred_probs")
            if isinstance(probs, list) and len(probs) >= 2:
                scores.append(float(probs[1]))

    if not scores:
        raise ValueError("No prediction probabilities found in predictions output.")
    return float(sum(scores) / len(scores))


def _read_clarity_case_count(workspace_root: Path) -> int:
    manifest_path = workspace_root / "run_manifest.json"
    if not manifest_path.exists():
        return 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts") or {}
    case_ids = artifacts.get("clarity_case_ids") or []
    return len(case_ids)


def _classify_runtime_error(exc: RuntimeError) -> WorkerFailure:
    detail = _safe_error_message(exc)
    lowered = detail.lower()
    if "no kidney structures were detected" in lowered:
        return WorkerFailure(
            "NO_KIDNEYS_DETECTED",
            "No kidneys were detected in the CT.",
            detail,
        )
    if "no renal tumor was detected" in lowered or "no scoreable tumor mask" in lowered:
        return WorkerFailure(
            "NO_TUMOR_DETECTED",
            "No tumor was detected.",
            detail,
        )
    if "modality=" in lowered and "requires ct" in lowered:
        return WorkerFailure(
            "WRONG_MODALITY",
            "Only CT studies are supported.",
            detail,
        )
    if "no ct series with sufficient slices" in lowered:
        return WorkerFailure(
            "INSUFFICIENT_CT_SERIES",
            "The CT series does not have enough slices.",
            detail,
        )
    if "no dicom" in lowered:
        return WorkerFailure(
            "NO_DICOM_FILES",
            "No DICOM files were found.",
            detail,
        )
    return WorkerFailure("PIPELINE_ERROR", "Processing failed.", detail)


def _retryable_failed_series_attempt(exc: BaseException) -> bool:
    """Whether trying another ranked CT series might recover from this failure."""

    if isinstance(exc, WorkerFailure):
        return exc.error_code == "NO_TUMOR_DETECTED"
    if isinstance(exc, RuntimeError):
        wf = _classify_runtime_error(exc)
        if wf.error_code in {"NO_KIDNEYS_DETECTED", "NO_TUMOR_DETECTED"}:
            return True
        if wf.error_code != "PIPELINE_ERROR":
            return False
        detail = wf.error_message.lower()
        needles = (
            "corticomedullary",
            "nephrographic",
            "unenhanced",
            "incorrect phase",
            "renal tumor scoring",
            "predicted ct phase",
            "tcga phase",
            "totalsegmentator failed",
            "tumor segmentation failed",
            "dcm2niix failed",
            "simpleitk",
            "dicom→nifti",
            "dicom to nifti",
        )
        return any(n in detail for n in needles)
    return False


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
    # Keep messages bounded for S3/result payloads, but preserve the tail where
    # the most specific/root exception text usually appears.
    max_len = 1200
    if len(msg) <= max_len:
        return msg
    head = msg[:700]
    tail = msg[-450:]
    return f"{head} ... {tail}"


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
    fail_on_empty_tumor: bool,
    pipeline_version: str,
    delete_input_after_success: bool,
    auto_select_series: bool,
    max_total_bytes: int,
    max_object_count: int,
    max_single_object_bytes: int,
    max_series_fallback_attempts: int,
    tcga_phase_model_dir: Path | None = None,
    tcga_phase_cache_root: Path | None = None,
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

    inventory = _collect_submission_inventory(
        s3_client,
        bucket=bucket,
        input_prefix=submission.input_prefix,
    )
    _enforce_upload_caps(
        inventory,
        max_total_bytes=max_total_bytes,
        max_object_count=max_object_count,
        max_single_object_bytes=max_single_object_bytes,
    )
    _validate_dicom_headers(s3_client, bucket=bucket, inventory=inventory)

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
            raise RuntimeError(
                "No DICOM files were found in this submission. Check that you uploaded the correct study "
                "folder and that files have .dcm extensions. If your DICOM CD uses extensionless files "
                "(e.g., IM-0001-0001), rename them with a .dcm extension before uploading."
            )
        _log("info", "download_complete", submission_id=submission.submission_id, files=n_downloaded)

        active_workspace = pipeline_workspace

        tcga_phase_cfg = TcgaPhasePredictionConfig(
            enabled=tcga_phase_model_dir is not None,
            model_dir=tcga_phase_model_dir,
            cache_root=tcga_phase_cache_root,
            device=device,
        )
        base_cfg_kwargs = dict(
            dicom_input=input_root,
            totalsegmentator=TotalSegmentatorConfig(device=device),
            tumor=TumorSegmentationConfig(device=device),
            phase_gating=PhaseGatingConfig(),
            tcga_phase_prediction=tcga_phase_cfg,
            mask_adaptation=MaskAdaptationConfig(),
            inference=InferenceConfig(
                checkpoint_dir=weights_dir,
                checkpoint_dir_recursive=True,
                device=device,
            ),
            skip_tumor=False,
            skip_inference=False,
            reuse_cached_artifacts=False,
            continue_on_empty_tumor=not fail_on_empty_tumor,
            auto_select_series=auto_select_series,
            dicom_backend=dicom_backend,
            dcm2niix_binary=dcm2niix_binary,
        )

        if max_series_fallback_attempts <= 1:
            cfg = build_pipeline_config(
                workspace_root=pipeline_workspace,
                **base_cfg_kwargs,
            )
            _log("info", "pipeline_start", submission_id=submission.submission_id)
            run_pipeline(cfg)
        else:
            series_list = list(discover_series_roots(input_root))
            ranked, rank_reasons = rank_ct_series_candidates_for_fallback(series_list)
            for detail in rank_reasons:
                _log(
                    "info",
                    "series_fallback_rank",
                    submission_id=submission.submission_id,
                    detail=detail,
                )
            if not ranked:
                raise RuntimeError(
                    "No CT series with sufficient slices were found. The uploaded folder may be a patient-level "
                    "folder containing only non-CT modalities (MRI, PET, dose reports), or all CT series had fewer "
                    f"than {MIN_CT_SLICES} slices (scouts/localizers only). Upload one contrast-enhanced abdominal CT study folder."
                )
            attempts_budget = min(max_series_fallback_attempts, len(ranked))
            last_error: BaseException | None = None
            for attempt_idx in range(attempts_budget):
                series = ranked[attempt_idx]
                ws_try = submission_root / f"workspace_try_{attempt_idx}"
                ws_try.mkdir(parents=True, exist_ok=True)
                cfg = build_pipeline_config(workspace_root=ws_try, **base_cfg_kwargs)
                _log(
                    "info",
                    "pipeline_start",
                    submission_id=submission.submission_id,
                    fallback_attempt=attempt_idx + 1,
                    fallback_attempts_max=attempts_budget,
                    series_instance_uid=series.series_instance_uid,
                )
                try:
                    run_pipeline(cfg, series_instance_uid=series.series_instance_uid)
                except RuntimeError as exc:
                    last_error = exc
                    if attempt_idx >= attempts_budget - 1 or not _retryable_failed_series_attempt(exc):
                        raise
                    _log(
                        "warning",
                        "series_fallback_retry",
                        submission_id=submission.submission_id,
                        series_instance_uid=series.series_instance_uid,
                        error=_safe_error_message(exc),
                    )
                    continue

                clarity_try = _read_clarity_case_count(ws_try)
                if clarity_try == 0:
                    tumor_miss = WorkerFailure(
                        "NO_TUMOR_DETECTED",
                        "No tumor was detected.",
                        "Processing completed, but no case produced a scoreable tumor mask.",
                    )
                    last_error = tumor_miss
                    if attempt_idx >= attempts_budget - 1 or not _retryable_failed_series_attempt(tumor_miss):
                        raise tumor_miss
                    _log(
                        "warning",
                        "series_fallback_retry",
                        submission_id=submission.submission_id,
                        series_instance_uid=series.series_instance_uid,
                        reason="no_scoreable_tumor_mask",
                    )
                    continue

                active_workspace = ws_try
                break
            else:
                if last_error is not None:
                    raise last_error
                raise RuntimeError("Series fallback exhausted without completing inference.")

        _log("info", "pipeline_complete", submission_id=submission.submission_id)

        clarity_case_count = _read_clarity_case_count(active_workspace)
        if clarity_case_count == 0:
            raise WorkerFailure(
                "NO_TUMOR_DETECTED",
                "No tumor was detected.",
                "Processing completed, but no case produced a scoreable tumor mask.",
            )

        predictions_path = active_workspace / "predictions" / "predictions.json"
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


def _submission_in_worker_shard(*, submission_id: str, worker_index: int, worker_count: int) -> bool:
    """Deterministically shard submission IDs across workers."""

    digest = hashlib.sha1(submission_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big")
    return (bucket % worker_count) == worker_index


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
    fail_on_empty_tumor: bool,
    pipeline_version: str,
    max_cases: int | None,
    delete_input_after_success: bool,
    delete_input_after_failure: bool,
    auto_select_series: bool,
    processing_ttl_seconds: int = DEFAULT_PROCESSING_TTL_SECONDS,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_object_count: int = DEFAULT_MAX_OBJECT_COUNT,
    max_single_object_bytes: int = DEFAULT_MAX_SINGLE_OBJECT_BYTES,
    max_series_fallback_attempts: int = DEFAULT_MAX_SERIES_FALLBACK_ATTEMPTS,
    tcga_phase_model_dir: Path | None = None,
    tcga_phase_cache_root: Path | None = None,
    worker_index: int = 0,
    worker_count: int = 1,
) -> int:
    pending = list_pending_submissions(
        s3_client,
        bucket=bucket,
        prefix_root=prefix_root,
        processing_ttl_seconds=processing_ttl_seconds,
    )
    if worker_count > 1:
        pending = [
            submission
            for submission in pending
            if _submission_in_worker_shard(
                submission_id=submission.submission_id,
                worker_index=worker_index,
                worker_count=worker_count,
            )
        ]
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
                fail_on_empty_tumor=fail_on_empty_tumor,
                pipeline_version=pipeline_version,
                delete_input_after_success=delete_input_after_success,
                auto_select_series=auto_select_series,
                max_total_bytes=max_total_bytes,
                max_object_count=max_object_count,
                max_single_object_bytes=max_single_object_bytes,
                max_series_fallback_attempts=max_series_fallback_attempts,
                tcga_phase_model_dir=tcga_phase_model_dir,
                tcga_phase_cache_root=tcga_phase_cache_root,
            )
        except WorkerFailure as exc:
            _log(
                "error",
                "submission_failed",
                submission_id=submission.submission_id,
                error_code=exc.error_code,
                error=exc.error_message,
            )
            write_result_json(
                s3_client,
                bucket=bucket,
                result_key=submission.result_key,
                submission_id=submission.submission_id,
                status="failed",
                pipeline_version=pipeline_version,
                message=exc.message,
                clarity_score=None,
                error_code=exc.error_code,
                error_message=exc.error_message,
            )
            if delete_input_after_failure:
                deleted = delete_input_objects(s3_client, bucket=bucket, input_prefix=submission.input_prefix)
                _log(
                    "info",
                    "input_deleted_after_failure",
                    submission_id=submission.submission_id,
                    deleted=deleted,
                )
        except RuntimeError as exc:
            failure = _classify_runtime_error(exc)
            _log(
                "error",
                "submission_failed",
                submission_id=submission.submission_id,
                error_code=failure.error_code,
                error=failure.error_message,
            )
            write_result_json(
                s3_client,
                bucket=bucket,
                result_key=submission.result_key,
                submission_id=submission.submission_id,
                status="failed",
                pipeline_version=pipeline_version,
                message=failure.message,
                clarity_score=None,
                error_code=failure.error_code,
                error_message=failure.error_message,
            )
            if delete_input_after_failure:
                deleted = delete_input_objects(s3_client, bucket=bucket, input_prefix=submission.input_prefix)
                _log(
                    "info",
                    "input_deleted_after_failure",
                    submission_id=submission.submission_id,
                    deleted=deleted,
                )
        except Exception as exc:  # noqa: BLE001
            err = _safe_error_message(exc)
            _log(
                "error",
                "submission_failed",
                submission_id=submission.submission_id,
                error_code="PIPELINE_ERROR",
                error=err,
            )
            write_result_json(
                s3_client,
                bucket=bucket,
                result_key=submission.result_key,
                submission_id=submission.submission_id,
                status="failed",
                pipeline_version=pipeline_version,
                message="Processing failed.",
                clarity_score=None,
                error_code="PIPELINE_ERROR",
                error_message=err,
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
    tcga_phase_model_dir: Path | None = typer.Option(
        None,
        "--tcga-phase-model-dir",
        envvar="CLARITY_TCGA_PHASE_MODEL_DIR",
        exists=True,
        file_okay=False,
        help="SWP v3 TCGA phase checkpoints (*.pth). When set, gating runs after TotalSegmentator.",
    ),
    tcga_phase_cache_root: Path | None = typer.Option(
        None,
        "--tcga-phase-cache-root",
        envvar="CLARITY_TCGA_PHASE_CACHE_ROOT",
        file_okay=False,
        help="Optional cache directory for TCGA phase (v3) patch extraction.",
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
    fail_on_empty_tumor: bool = typer.Option(
        False,
        "--fail-on-empty-tumor/--allow-empty-tumor",
        envvar="CLARITY_FAIL_ON_EMPTY_TUMOR",
        help="Match clarity-pipeline behavior. Default allows empty tumor series to be skipped.",
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
    auto_select_series: bool = typer.Option(
        True,
        "--auto-select-series/--no-auto-select-series",
        envvar="CLARITY_AUTO_SELECT_SERIES",
        help="Auto-select one best CT series per submission (recommended).",
    ),
    pipeline_version: str = typer.Option(
        __version__,
        "--pipeline-version",
        envvar="CLARITY_PIPELINE_VERSION",
    ),
    processing_ttl_seconds: int = typer.Option(
        DEFAULT_PROCESSING_TTL_SECONDS,
        "--processing-ttl-seconds",
        envvar="CLARITY_S3_PROCESSING_TTL_SECONDS",
        min=1,
        help="Retry submissions stuck in processing after this many seconds.",
    ),
    max_total_bytes: int = typer.Option(
        DEFAULT_MAX_TOTAL_BYTES,
        "--max-total-bytes",
        envvar="CLARITY_S3_MAX_TOTAL_BYTES",
        min=1,
    ),
    max_object_count: int = typer.Option(
        DEFAULT_MAX_OBJECT_COUNT,
        "--max-object-count",
        envvar="CLARITY_S3_MAX_OBJECT_COUNT",
        min=1,
    ),
    max_single_object_bytes: int = typer.Option(
        DEFAULT_MAX_SINGLE_OBJECT_BYTES,
        "--max-single-object-bytes",
        envvar="CLARITY_S3_MAX_SINGLE_OBJECT_BYTES",
        min=1,
    ),
    max_series_fallback_attempts: int = typer.Option(
        DEFAULT_MAX_SERIES_FALLBACK_ATTEMPTS,
        "--max-series-fallback-attempts",
        envvar="CLARITY_S3_MAX_SERIES_FALLBACK_ATTEMPTS",
        min=1,
        help=(
            "Try up to N ranked CT series when the previous candidate fails with a retryable clinical/tool error "
            "(e.g. no kidneys / no tumor / phase mismatch). Use 1 to disable multi-series retries."
        ),
    ),
    worker_index: int = typer.Option(
        0,
        "--worker-index",
        envvar="CLARITY_S3_WORKER_INDEX",
        min=0,
        help="0-based worker shard index when running multiple worker processes.",
    ),
    worker_count: int = typer.Option(
        1,
        "--worker-count",
        envvar="CLARITY_S3_WORKER_COUNT",
        min=1,
        help="Total number of worker shards. Use with --worker-index for safe parallel workers.",
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
        fail_on_empty_tumor=fail_on_empty_tumor,
        auto_select_series=auto_select_series,
        processing_ttl_seconds=processing_ttl_seconds,
        max_total_bytes=max_total_bytes,
        max_object_count=max_object_count,
        max_single_object_bytes=max_single_object_bytes,
        max_series_fallback_attempts=max_series_fallback_attempts,
        tcga_phase_model_dir=str(tcga_phase_model_dir) if tcga_phase_model_dir else None,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    if worker_index >= worker_count:
        raise typer.BadParameter("--worker-index must be < --worker-count.")

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
            fail_on_empty_tumor=fail_on_empty_tumor,
            pipeline_version=pipeline_version,
            max_cases=max_cases,
            delete_input_after_success=delete_input_after_success,
            delete_input_after_failure=delete_input_after_failure,
            auto_select_series=auto_select_series,
            processing_ttl_seconds=processing_ttl_seconds,
            max_total_bytes=max_total_bytes,
            max_object_count=max_object_count,
            max_single_object_bytes=max_single_object_bytes,
            max_series_fallback_attempts=max_series_fallback_attempts,
            tcga_phase_model_dir=tcga_phase_model_dir,
            tcga_phase_cache_root=tcga_phase_cache_root,
            worker_index=worker_index,
            worker_count=worker_count,
        )
        if once:
            _log("info", "worker_exit_once")
            return
        time.sleep(poll_seconds)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
