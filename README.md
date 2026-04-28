# Clarity Inference Pipeline

DICOM → NIfTI → TotalSegmentator → KiTS21-style kidney tumor segmentation (nnU-Net v1) → SWP-compatible volumes → **PNvsRN** ensemble inference producing the **CLARITY** score (`clarity-pipeline predict`).

## Input layout 

There are **two** layouts, depending on how you invoke the tool.

### A. `clarity-pipeline predict` (single `--input` directory)

- **`--input` must be a directory**, never a single `.dcm` file.
- The pipeline discovers DICOMs **recursively** under that directory.
- **One CT series:** point `--input` at the folder that contains **only** that series (all slices for one volume). Nested subfolders are fine if they still describe one series.
- **Several series under one tree:** you may see a “multiple series” warning. Either pass **`--series-uid …`** to process one series, or let it run each discovered series.

### B. Batch helpers (`dev/run_local_dicom_batch.sh`, `dev/slurm_gpu_kits.job`)

Use this when you have **many patients** and want one run per patient folder.

Set **`CLARITY_DICOM_CASES_ROOT`** to a directory whose **immediate subdirectories** are cases (any names — `KiTS-00042`, `patient_07`, etc.):

```text
CLARITY_DICOM_CASES_ROOT/
  case-a/          ← one patient
    … nested folders …
    … DICOM .dcm files …
  case-b/
  …
```

The batch script picks **one diagnostic CT series** per case (prefers `Modality == CT`, skips `SEG`). Structure inside each case folder can be anything that still contains a CT series directory with `.dcm` files.

**Example dataset, not a requirement:** the default paths in `dev/run_local_dicom_batch.sh` point at a **TCIA / KiTS-style** tree (C4KC manifest layout). That is only a **convenience default** for developers. **To use your own data:** set `CLARITY_DICOM_CASES_ROOT` to the parent of your per-case folders (and optionally `CLARITY_WORK_ROOT`, `CLARITY_WEIGHTS_DIR`, `CLARITY_DEVICE`). No KiTS-specific filenames or metadata are required.

## Run it

### Option 1 — Dev setup (venv + scripts)

**Once per clone:**

```bash
git clone https://github.com/AIM-HI-Lab/clarity-inference-pipeline.git
cd clarity-inference-pipeline
./dev/setup_local_models.sh
```

This creates `.venv`, installs the package + TotalSegmentator + nnU-Net, downloads the public KiTS21 tumor weights into repo-local nnU-Net paths, and writes `dev/clarity_local_env.sh`. Default interpreter is **`python3.10`**; use **`CLARITY_PYTHON=…`** if needed.

**Every new shell:**

```bash
cd clarity-inference-pipeline
source dev/clarity_local_env.sh
export PATH="$(pwd)/.venv/bin:$PATH"
```

**Single series / arbitrary folder** (same rules as **layout A** in [Input layout](#input-layout-read-this-first)):

```bash
clarity-pipeline predict \
  --input /path/to/dicom/folder \
  --work-dir /path/to/output/run1 \
  --weights-dir /path/to/pnvrn_folds \
  --device cpu
```

Omit `--weights-dir` if `pnvrn_folds/` exists in the repo root (25× `.pth` in a fold tree).

**Many cases on disk** (same rules as **layout B** in [Input layout](#input-layout-read-this-first)):

```bash
export CLARITY_DICOM_CASES_ROOT=/path/to/parent/of/case/folders
bash dev/run_local_dicom_batch.sh ALL          # all immediate subfolders
# or
bash dev/run_local_dicom_batch.sh case-a       # one subfolder name
```

Useful env vars: `CLARITY_DEVICE=cuda`, `CLARITY_WORK_ROOT`, `CLARITY_WEIGHTS_DIR`. See `dev/run_local_dicom_batch.sh` header for optional `CLARITY_TUMOR_EXTRA`, `CLARITY_FAIL_ON_EMPTY_TUMOR`, etc.

**GPU batch on Slurm** (after the same `./dev/setup_local_models.sh` on a node where the venv lives):

```bash
cd /path/to/clarity-inference-pipeline
export CLARITY_DICOM_CASES_ROOT=/path/to/parent/of/case/folders   # required in practice
sbatch dev/slurm_gpu_kits.job
```

Defaults: processes **all** case subfolders (`CASE_NAME=ALL`), `CLARITY_DEVICE=cuda`. One case: `sbatch --export=ALL,CASE_NAME=case-a dev/slurm_gpu_kits.job`. The job runs `dev/run_local_dicom_batch.sh`; edit `#SBATCH` lines in `dev/slurm_gpu_kits.job` for your scheduler (partition, GPUs, memory, time). Submit **from the repo** (or set `CLARITY_REPO_ROOT`) so Slurm can find the clone — see [Cluster notes](#cluster-slurm-and-hpc).

### Option 2 — Docker

Build once from the repo root:

```bash
docker build -t clarity-inference-pipeline:local .
# GPU host: docker build -f Dockerfile.gpu -t clarity-inference-pipeline:gpu .
```

Run (folder of DICOMs, output dir, PNvsRN tree):

```bash
chmod +x dev/docker-predict.sh
./dev/docker-predict.sh /path/to/dicom/folder /path/to/output /path/to/pnvrn_folds
```

Optional args after `--` go to `clarity-pipeline predict`. GPU: set `CLARITY_DOCKER_IMAGE`, `CLARITY_DOCKER_GPU=1`, `CLARITY_DEVICE=cuda` as in `dev/docker-predict.sh`.

## Prerequisites (local / venv)

- **Python 3.10+** (see `CLARITY_PYTHON`).
- **`dcm2niix`** on `PATH` *or* use **`--dicom-backend sitk`** / `CLARITY_DICOM_BACKEND=sitk` (SimpleITK / GDCM).
- **PNvsRN weights:** a `pnvrn_folds/`-style tree with **25** `.pth` checkpoints.

Phase gating is optional and off by default (internal classifier not in this repo).

Advanced: `python3 -m pip install -e .` alone installs the package; you must still provide TotalSegmentator, nnU-Net, and model weights on your own. Prefer `./dev/setup_local_models.sh` for the full stack.

## CLI (short)

```bash
clarity-pipeline predict --input DIR --work-dir DIR [--weights-dir DIR] [--device cpu|cuda]
```

Notable flags: `--series-uid`, `--dicom-backend auto|dcm2niix|sitk`, `--totalseg-extra` / `CLARITY_TOTALSEG_EXTRA`, `--tumor-extra` / `CLARITY_TUMOR_EXTRA`, `--skip-inference`, `--skip-tumor`, `--fail-on-empty-tumor`, `--enable-phase-gating` (with `--phase-entrypoint`).

Predictions are **inference-only** (`evaluation_mode: "prediction_only"`); no ground-truth labels are required.

## S3 Worker (Upload Portal Contract)

`clarity-s3-worker` polls `s3://<bucket>/clarity/submissions/{submission_id}` and writes
`result.json` in the Upload Portal schema.

### One-shot mode

```bash
clarity-s3-worker run \
  --bucket my-clarity-bucket \
  --region us-east-1 \
  --weights-dir /models/pnvrn_folds \
  --device cpu \
  --once
```

### Continuous polling mode

```bash
clarity-s3-worker run \
  --bucket my-clarity-bucket \
  --region us-east-1 \
  --prefix-root clarity/submissions \
  --weights-dir /models/pnvrn_folds \
  --work-root /var/lib/clarity-worker \
  --device cuda \
  --poll-seconds 30 \
  --max-cases 4 \
  --delete-input-after-success
```

Worker behavior:

- Pending submission: `input/` has at least one `.dcm` and `result.json` does not exist.
- Writes `status="processing"` before inference starts.
- Writes `status="completed"` with a single `clarity_score` (mean positive-class probability across produced case rows).
- On exceptions, writes `status="failed"`, `clarity_score: null`, and a concise error message.
- Optional cleanup only deletes `clarity/submissions/{submission_id}/input/*` (never `manifest.json` or `result.json`).

### Environment variables

| Variable | CLI flag | Default | Notes |
| --- | --- | --- | --- |
| `CLARITY_S3_BUCKET` | `--bucket` | _(required)_ | Target S3 bucket. |
| `AWS_DEFAULT_REGION` | `--region` | `us-east-1` | AWS region for S3 client. |
| `CLARITY_S3_PREFIX_ROOT` | `--prefix-root` | `clarity/submissions` | Submission root prefix. |
| `CLARITY_WORK_ROOT` | `--work-root` | `/tmp/clarity-s3-worker` | Local temp workspace parent. |
| `CLARITY_WEIGHTS_DIR` | `--weights-dir` | _(required)_ | PNvsRN fold checkpoints directory. |
| `CLARITY_DEVICE` | `--device` | `cpu` | `cpu` or `cuda`. |
| `CLARITY_S3_POLL_SECONDS` | `--poll-seconds` | `30` | Poll interval in loop mode. |
| `CLARITY_S3_MAX_CASES` | `--max-cases` | unset | Limit processed submissions per cycle. |
| `CLARITY_DELETE_INPUT_AFTER_SUCCESS` | `--delete-input-after-success` | `false` | Remove `input/*` after successful upload. |
| `CLARITY_DELETE_INPUT_AFTER_FAILURE` | `--delete-input-after-failure` | `false` | Remove `input/*` after failed run. |
| `CLARITY_PIPELINE_VERSION` | `--pipeline-version` | package version | Version string written into `result.json`. |

### IAM permissions (minimum)

- `s3:ListBucket` on bucket with prefix conditions for `clarity/submissions/*`
- `s3:GetObject` on `arn:aws:s3:::<bucket>/clarity/submissions/*`
- `s3:PutObject` on `arn:aws:s3:::<bucket>/clarity/submissions/*`
- `s3:DeleteObject` on `arn:aws:s3:::<bucket>/clarity/submissions/*` (only needed when delete flags are enabled)

### systemd example

```ini
[Unit]
Description=CLARITY S3 Worker
After=network-online.target

[Service]
Type=simple
User=clarity
WorkingDirectory=/opt/clarity-inference-pipeline
Environment=CLARITY_S3_BUCKET=my-clarity-bucket
Environment=AWS_DEFAULT_REGION=us-east-1
Environment=CLARITY_WEIGHTS_DIR=/opt/models/pnvrn_folds
Environment=CLARITY_DEVICE=cuda
ExecStart=/opt/clarity-inference-pipeline/.venv/bin/clarity-s3-worker run --poll-seconds 30
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### cron example (one-shot every 5 minutes)

```cron
*/5 * * * * cd /opt/clarity-inference-pipeline && /opt/clarity-inference-pipeline/.venv/bin/clarity-s3-worker run --bucket my-clarity-bucket --weights-dir /opt/models/pnvrn_folds --once >> /var/log/clarity-s3-worker.log 2>&1
```

## Output

Under `--work-dir`: `cases/<series_uid>/` (NIfTI + segmentations), `swp_manifest.json`, `predictions/predictions.json`, `run_manifest.json`. Main result file: **`predictions/predictions.json`** (ensemble over checkpoints).

## Cluster (Slurm and HPC)

- **Same pipeline** as `dev/run_local_dicom_batch.sh` / `dev/docker-predict.sh`: TotalSegmentator **total** task, nnU-Net v1 KiTS21 tumor model, then SWP ensemble.
- **One-time:** load your site’s Python/CUDA modules if needed, then `./dev/setup_local_models.sh`. PyTorch CUDA variant: **`CLARITY_PYTORCH_CUDA`** (`auto`, `cu118`, `cu121`, `cu124`, `cpu`, …) — the script installs CUDA wheels **before and after** `pip install -e .` so you do not stay on a `+cpu` build. Copy or link **`pnvrn_folds`** somewhere shared; set **`CLARITY_WEIGHTS_DIR`** if not `<repo>/pnvrn_folds`.
- **Submit from the clone** so `SLURM_SUBMIT_DIR` resolves: `cd …/clarity-inference-pipeline && sbatch dev/slurm_gpu_kits.job`, or `sbatch --export=ALL,CLARITY_REPO_ROOT=/path/to/clone dev/slurm_gpu_kits.job`. If the venv is not `<repo>/.venv`, set **`CLARITY_VENV_DIR`**.
- **`dcm2niix` missing in batch `PATH`:** use `CLARITY_DICOM_BACKEND=sitk` or install / `module load` dcm2niix; see also `CLARITY_DCM2NIIX`.
- **`clarity-pipeline` not found / REPO_ROOT looks like a spool path:** Slurm copies the batch script; always pass **`CLARITY_REPO_ROOT`** or submit from the repo directory.
- **No GPU / wrong PyTorch:** check `nvidia-smi`, partition and `#SBATCH --gres` (or your site’s GPU syntax), and reinstall CUDA-matched `torch` if `torch.version.cuda` is `None` or the build is `+cpu`. Optional: run **`./dev/check_gpu_env.sh`** on an interactive GPU node.

For CPU-only queues, copy the job file, set `CLARITY_DEVICE=cpu`, drop or change GPU `#SBATCH` lines, and point at a CPU partition.

## Docker details

Image installs dependencies, TotalSegmentator weights, and Task135 under fixed paths; you still mount **PNvsRN** weights. `docker compose` and raw `docker run` examples work the same way as `dev/docker-predict.sh`; see comments in `Dockerfile` / `docker-compose.yml`.

## Parity: Docker vs venv vs Slurm

| Piece | Docker | Venv + `run_local_dicom_batch.sh` / Slurm |
| --- | --- | --- |
| Entry | `clarity-pipeline predict` via `dev/docker-predict.sh` | Same CLI, invoked by batch script |
| nnU-Net / TotalSegmentator dirs | Set in image / entrypoint | `dev/setup_local_models.sh` → `dev/clarity_local_env.sh` |
| Tumor model | Baked or downloaded in image | Downloaded by `setup_local_models.sh` |
