# Axis Inference Pipeline

DICOM → NIfTI → TotalSegmentator → KiTS21-style kidney tumor segmentation (nnU-Net v1) → SWP-compatible volumes → **PNvsRN** ensemble inference (`axis-pn predict`).

## Input layout 

There are **two** layouts, depending on how you invoke the tool.

### A. `axis-pn predict` (single `--input` directory)

- **`--input` must be a directory**, never a single `.dcm` file.
- The pipeline discovers DICOMs **recursively** under that directory.
- **One CT series:** point `--input` at the folder that contains **only** that series (all slices for one volume). Nested subfolders are fine if they still describe one series.
- **Several series under one tree:** you may see a “multiple series” warning. Either pass **`--series-uid …`** to process one series, or let it run each discovered series.

### B. Batch helpers (`dev/run_local_dicom_batch.sh`, `dev/slurm_gpu_kits.job`)

Use this when you have **many patients** and want one run per patient folder.

Set **`AXIS_DICOM_CASES_ROOT`** to a directory whose **immediate subdirectories** are cases (any names — `KiTS-00042`, `patient_07`, etc.):

```text
AXIS_DICOM_CASES_ROOT/
  case-a/          ← one patient
    … nested folders …
    … DICOM .dcm files …
  case-b/
  …
```

The batch script picks **one diagnostic CT series** per case (prefers `Modality == CT`, skips `SEG`). Structure inside each case folder can be anything that still contains a CT series directory with `.dcm` files.

**Example dataset, not a requirement:** the default paths in `dev/run_local_dicom_batch.sh` point at a **TCIA / KiTS-style** tree (C4KC manifest layout). That is only a **convenience default** for developers. **To use your own data:** set `AXIS_DICOM_CASES_ROOT` to the parent of your per-case folders (and optionally `AXIS_WORK_ROOT`, `AXIS_WEIGHTS_DIR`, `AXIS_DEVICE`). No KiTS-specific filenames or metadata are required.

## Run it

### Option 1 — Dev setup (venv + scripts)

**Once per clone:**

```bash
git clone https://github.com/AIM-HI-Lab/axis-inference-pipeline.git
cd axis-inference-pipeline
./dev/setup_local_models.sh
```

This creates `.venv`, installs the package + TotalSegmentator + nnU-Net, downloads the public KiTS21 tumor weights into repo-local nnU-Net paths, and writes `dev/axis_local_env.sh`. Default interpreter is **`python3.10`**; use **`AXIS_PYTHON=…`** if needed.

**Every new shell:**

```bash
cd axis-inference-pipeline
source dev/axis_local_env.sh
export PATH="$(pwd)/.venv/bin:$PATH"
```

**Single series / arbitrary folder** (same rules as **layout A** in [Input layout](#input-layout-read-this-first)):

```bash
axis-pn predict \
  --input /path/to/dicom/folder \
  --work-dir /path/to/output/run1 \
  --weights-dir /path/to/pnvrn_folds \
  --device cpu
```

Omit `--weights-dir` if `pnvrn_folds/` exists in the repo root (25× `.pth` in a fold tree).

**Many cases on disk** (same rules as **layout B** in [Input layout](#input-layout-read-this-first)):

```bash
export AXIS_DICOM_CASES_ROOT=/path/to/parent/of/case/folders
bash dev/run_local_dicom_batch.sh ALL          # all immediate subfolders
# or
bash dev/run_local_dicom_batch.sh case-a       # one subfolder name
```

Useful env vars: `AXIS_DEVICE=cuda`, `AXIS_WORK_ROOT`, `AXIS_WEIGHTS_DIR`. See `dev/run_local_dicom_batch.sh` header for optional `AXIS_TUMOR_EXTRA`, `AXIS_FAIL_ON_EMPTY_TUMOR`, etc.

**GPU batch on Slurm** (after the same `./dev/setup_local_models.sh` on a node where the venv lives):

```bash
cd /path/to/axis-inference-pipeline
export AXIS_DICOM_CASES_ROOT=/path/to/parent/of/case/folders   # required in practice
sbatch dev/slurm_gpu_kits.job
```

Defaults: processes **all** case subfolders (`CASE_NAME=ALL`), `AXIS_DEVICE=cuda`. One case: `sbatch --export=ALL,CASE_NAME=case-a dev/slurm_gpu_kits.job`. The job runs `dev/run_local_dicom_batch.sh`; edit `#SBATCH` lines in `dev/slurm_gpu_kits.job` for your scheduler (partition, GPUs, memory, time). Submit **from the repo** (or set `AXIS_REPO_ROOT`) so Slurm can find the clone — see [Cluster notes](#cluster-slurm-and-hpc).

### Option 2 — Docker

Build once from the repo root:

```bash
docker build -t axis-inference-pipeline:local .
# GPU host: docker build -f Dockerfile.gpu -t axis-inference-pipeline:gpu .
```

Run (folder of DICOMs, output dir, PNvsRN tree):

```bash
chmod +x dev/docker-predict.sh
./dev/docker-predict.sh /path/to/dicom/folder /path/to/output /path/to/pnvrn_folds
```

Optional args after `--` go to `axis-pn predict`. GPU: set `AXIS_DOCKER_IMAGE`, `AXIS_DOCKER_GPU=1`, `AXIS_DEVICE=cuda` as in `dev/docker-predict.sh`.

## Prerequisites (local / venv)

- **Python 3.10+** (see `AXIS_PYTHON`).
- **`dcm2niix`** on `PATH` *or* use **`--dicom-backend sitk`** / `AXIS_DICOM_BACKEND=sitk` (SimpleITK / GDCM).
- **PNvsRN weights:** a `pnvrn_folds/`-style tree with **25** `.pth` checkpoints.

Phase gating is optional and off by default (internal classifier not in this repo).

Advanced: `python3 -m pip install -e .` alone installs the package; you must still provide TotalSegmentator, nnU-Net, and model weights on your own. Prefer `./dev/setup_local_models.sh` for the full stack.

## CLI (short)

```bash
axis-pn predict --input DIR --work-dir DIR [--weights-dir DIR] [--device cpu|cuda]
```

Notable flags: `--series-uid`, `--dicom-backend auto|dcm2niix|sitk`, `--totalseg-extra` / `AXIS_TOTALSEG_EXTRA`, `--tumor-extra` / `AXIS_TUMOR_EXTRA`, `--skip-inference`, `--skip-tumor`, `--fail-on-empty-tumor`, `--enable-phase-gating` (with `--phase-entrypoint`).

Predictions are **inference-only** (`evaluation_mode: "prediction_only"`); no ground-truth labels are required.

## Output

Under `--work-dir`: `cases/<series_uid>/` (NIfTI + segmentations), `swp_manifest.json`, `predictions/predictions.json`, `run_manifest.json`. Main result file: **`predictions/predictions.json`** (ensemble over checkpoints).

## Cluster (Slurm and HPC)

- **Same pipeline** as `dev/run_local_dicom_batch.sh` / `dev/docker-predict.sh`: TotalSegmentator **total** task, nnU-Net v1 KiTS21 tumor model, then SWP ensemble.
- **One-time:** load your site’s Python/CUDA modules if needed, then `./dev/setup_local_models.sh`. PyTorch CUDA variant: **`AXIS_PYTORCH_CUDA`** (`auto`, `cu118`, `cu121`, `cu124`, `cpu`, …) — the script installs CUDA wheels **before and after** `pip install -e .` so you do not stay on a `+cpu` build. Copy or link **`pnvrn_folds`** somewhere shared; set **`AXIS_WEIGHTS_DIR`** if not `<repo>/pnvrn_folds`.
- **Submit from the clone** so `SLURM_SUBMIT_DIR` resolves: `cd …/axis-inference-pipeline && sbatch dev/slurm_gpu_kits.job`, or `sbatch --export=ALL,AXIS_REPO_ROOT=/path/to/clone dev/slurm_gpu_kits.job`. If the venv is not `<repo>/.venv`, set **`AXIS_VENV_DIR`**.
- **`dcm2niix` missing in batch `PATH`:** use `AXIS_DICOM_BACKEND=sitk` or install / `module load` dcm2niix; see also `AXIS_DCM2NIIX`.
- **`axis-pn` not found / REPO_ROOT looks like a spool path:** Slurm copies the batch script; always pass **`AXIS_REPO_ROOT`** or submit from the repo directory.
- **No GPU / wrong PyTorch:** check `nvidia-smi`, partition and `#SBATCH --gres` (or your site’s GPU syntax), and reinstall CUDA-matched `torch` if `torch.version.cuda` is `None` or the build is `+cpu`. Optional: run **`./dev/check_gpu_env.sh`** on an interactive GPU node.

For CPU-only queues, copy the job file, set `AXIS_DEVICE=cpu`, drop or change GPU `#SBATCH` lines, and point at a CPU partition.

## Docker details

Image installs dependencies, TotalSegmentator weights, and Task135 under fixed paths; you still mount **PNvsRN** weights. `docker compose` and raw `docker run` examples work the same way as `dev/docker-predict.sh`; see comments in `Dockerfile` / `docker-compose.yml`.

## Parity: Docker vs venv vs Slurm

| Piece | Docker | Venv + `run_local_dicom_batch.sh` / Slurm |
| --- | --- | --- |
| Entry | `axis-pn predict` via `dev/docker-predict.sh` | Same CLI, invoked by batch script |
| nnU-Net / TotalSegmentator dirs | Set in image / entrypoint | `dev/setup_local_models.sh` → `dev/axis_local_env.sh` |
| Tumor model | Baked or downloaded in image | Downloaded by `setup_local_models.sh` |
