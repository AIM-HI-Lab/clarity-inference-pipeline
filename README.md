# Axis Inference Pipeline

`axis-inference-pipeline` packages the DICOM-to-`axis-pn` flow around the vendored PNvsRN inference stack.

## What It Does

Given a directory of DICOM files, the pipeline:

1. discovers DICOM series,
2. converts each series to NIfTI with `dcm2niix`,
3. runs `TotalSegmentator`,
4. runs kidney tumor segmentation with a public KiTS21 `nnU-Net v1` pretrained model by default,
5. builds SWP-compatible per-case files:
   - `imaging.nii.gz`
   - `segmentation.nii.gz`
6. runs the vendored PNvsRN ensemble and averages predictions across all 25 checkpoints.

By default, if `pnvrn_folds/` exists in the repo root, it is used as the model directory with recursive checkpoint discovery. That folder should resolve to all 25 `.pth` files.

## Install

```bash
python3 -m pip install -e .
```

You will also need external runtime tools on `PATH`. The dev setup script installs:

- `TotalSegmentator`
- `nnunetv2`
- `nnunet` (legacy v1, used for the public KiTS21 tumor model)

Phase gating is optional and disabled by default because the internal phase-classifier package is not present in this repo.

### Prerequisites (local runs)

- **Python 3.10+** (`dev/setup_local_models.sh` uses **`python3.10`** by default; set `AXIS_PYTHON` to use another interpreter, e.g. `AXIS_PYTHON=python3.12`)
- **`dcm2niix`** on your `PATH` (e.g. macOS: `brew install dcm2niix`)
- **PNvsRN weights**: a directory tree of **25** `.pth` checkpoints in the same layout as `pnvrn_folds/` (fold subfolders). Download or copy that tree somewhere, e.g. `~/models/pnvrn_folds`.

### Input path: always a folder

`axis-pn predict --input` must be a **directory**, not a path to a single `.dcm` file. Unzip or copy your DICOMs into a folder first. Typical CT scans are one folder with many slice files (one series); put that folder path as `--input`.

## Quick start (copy-paste)

### 1. Clone and install

```bash
git clone https://github.com/AIM-HI-Lab/axis-inference-pipeline.git
cd axis-inference-pipeline
./dev/setup_local_models.sh
```

This creates `.venv`, installs the package + TotalSegmentator + nnU-Net, downloads KiTS21 tumor weights into the repo-local nnU-Net paths, and writes `dev/axis_local_env.sh`.

### 2. Activate the environment (every new terminal)

```bash
cd axis-inference-pipeline
source dev/axis_local_env.sh
export PATH="$(pwd)/.venv/bin:$PATH"
```

### 3. Run on one CT series (one folder of DICOM slices)

Replace the three paths, then run:

```bash
axis-pn predict \
  --input /path/to/dicom/folder/one_series \
  --work-dir /path/to/output/run1 \
  --weights-dir /path/to/pnvrn_folds \
  --device cpu
```

- **`--input`**: folder that contains **only** that series (all `.dcm` slices for one volume).
- **`--work-dir`**: output folder (created if missing).
- **`--weights-dir`**: your PNvsRN checkpoint tree (25× `.pth`). If you keep `pnvrn_folds/` inside the repo clone, you can omit `--weights-dir` and it will auto-detect it when present.

Results: **`/path/to/output/run1/predictions/predictions.json`**, plus `cases/<SeriesInstanceUID>/` under the work dir.

### 4. Run on a directory tree (multiple series)

Use the same command when `--input` is a parent folder that contains **several** series (nested folders are fine). The pipeline discovers all DICOM files under that tree recursively.

If you see a warning that multiple series were found, either:

- **Process only one series** — pick its `SeriesInstanceUID` and re-run:

```bash
axis-pn predict \
  --input /path/to/dicom/folder \
  --work-dir /path/to/output/run1 \
  --weights-dir /path/to/pnvrn_folds \
  --device cpu \
  --series-uid "1.2.840.113619.2.55.3.XXXX.XXXX.XXXX.XXXXX"
```

- **Or** let it run all discovered series (it will process each one).

To **list** series UIDs and slice counts without running the full pipeline:

```bash
cd axis-inference-pipeline
source dev/axis_local_env.sh
export PATH="$(pwd)/.venv/bin:$PATH"
python -c "
from pathlib import Path
from axis_inference_pipeline.dicom import discover_series_roots
root = Path('/path/to/dicom/folder')
for s in discover_series_roots(root):
    print(s.series_instance_uid, s.modality, s.file_count, 'files')
"
```

Replace `/path/to/dicom/folder` with your `--input` directory.

### 5. Same thing with Docker

Build the image once (see [Docker](#docker)), then:

```bash
chmod +x dev/docker-predict.sh
./dev/docker-predict.sh /path/to/dicom/folder /path/to/output/run1 /path/to/pnvrn_folds
```

The first argument must be a **folder** of DICOMs (same rule as `--input` above). Optional: add `-- --series-uid ...` after the three paths.

## CLI

```bash
axis-pn predict \
  --input /path/to/dicoms \
  --work-dir /path/to/output
```

Useful flags:

- `--series-uid`: process only one discovered series
- `--weights-dir`: checkpoint root, defaults to `<repo>/pnvrn_folds` when present
- `--device cpu|cuda`: SWP inference device override
- `--totalseg-extra "..."` or env `AXIS_TOTALSEG_EXTRA`: optional extra TotalSegmentator CLI arguments (default: full `total` task).
- `--tumor-extra "..."` or env `AXIS_TUMOR_EXTRA`: optional extra nnU-Net tumor-segmentation CLI arguments.
- `--skip-inference`: stop after building SWP-ready NIfTI inputs
- `--skip-tumor`: create kidney-only SWP masks
- `--enable-phase-gating --phase-entrypoint ...`: enable optional phase selection

## Output Layout

The work directory contains:

- `cases/<series_uid>/imaging.nii.gz`
- `cases/<series_uid>/segmentation.nii.gz`
- `cases/<series_uid>/total_seg/`
- `cases/<series_uid>/metadata.json`
- `swp_manifest.json`
- `predictions/predictions.json`
- `run_manifest.json`

`predictions/predictions.json` is the averaged ensemble output across all checkpoints.

## Local Dev Runner

There is a convenience script at `dev/run_local_kits.sh`.

First-time setup:

```bash
./dev/setup_local_models.sh
```

That script:

- creates `.venv` (via `python3.10` unless `AXIS_PYTHON` is set)
- installs this package plus `TotalSegmentator`, `nnunetv2`, and `nnunet`
- creates repo-local nnU-Net env directories
- writes `dev/axis_local_env.sh`
- downloads the public KiTS21 pretrained tumor model
- preps the TotalSegmentator cache location

Then run a KiTS case with one command:

```bash
bash dev/run_local_kits.sh KiTS-00000
```

By default that script uses:

- KiTS root: `~/Desktop/kits_data/C4KC-KiTS-NBIA-manifest (1)/c4kc_kits`
- weights: `<repo>/pnvrn_folds`
- output root: `<repo>/local-runs`
- device: `cpu`
- tumor backend: `nnUNet v1` public KiTS21 baseline (`Task135_KiTS2021`, `3d_cascade_fullres`)

You can override them with:

```bash
AXIS_DEVICE=cuda AXIS_KITS_ROOT=/path/to/c4kc_kits bash dev/run_local_kits.sh KiTS-00000
```

## Cluster dry run (Slurm, CPU, no Docker)

This is the path for **validating the pipeline on a shared HPC node** when you cannot use Docker (typical on clusters). It runs the **same** `axis-pn predict` path as `dev/run_local_kits.sh`, which matches the default Docker invocation: TotalSegmentator **total** task, **nnU-Net v1** KiTS21 tumor model (`Task135`, `3d_cascade_fullres`), then SWP ensemble inference with **`--device cpu`**.

### Data layout (KiTS)

Point `AXIS_KITS_ROOT` at the directory that **directly contains** one folder per case:

```text
/path/to/kits-dicoms/c4kc_kits/
  KiTS-00000/
    ... nested DICOM series folders ...
  KiTS-00001/
  ...
```

Example used for AIM-HI Lab storage: `/home/jonnalr/AIM-HI-Lab/kits-dicoms/c4kc_kits`. Each `KiTS-XXXXX` directory should contain the usual nested DICOM tree (the helper picks a diagnostic **CT** series and skips **SEG**).

### One-time setup on the cluster

From an interactive session on a **login or build node** (adjust paths):

```bash
git clone https://github.com/AIM-HI-Lab/axis-inference-pipeline.git
cd axis-inference-pipeline
# Python 3.10 on PATH (or `AXIS_PYTHON=…`) + pip; ensure `dcm2niix` is on PATH (e.g. module load or conda).
./dev/setup_local_models.sh
# If you still have an old `.venv312` from earlier docs, remove or ignore it; the venv directory is now `.venv`.
```

That creates `.venv`, installs dependencies, downloads TotalSegmentator **total** weights and **Task135_KiTS2021**, and writes `dev/axis_local_env.sh` with **machine-local** nnU-Net directories under the clone.

Copy or link **PNvsRN weights** (`pnvrn_folds/`-style tree, 25× `.pth`) somewhere readable on the cluster and set `AXIS_WEIGHTS_DIR` if it is not `<repo>/pnvrn_folds`.

**Slurm says it cannot find `axis-pn` / `.venv`, or `REPO_ROOT` looks like `.../slurm/.../spool/...`:** Slurm **copies** the batch script to a **spool** directory and runs that copy, so **`$BASH_SOURCE` is not inside your git clone**. The scripts use **`SLURM_SUBMIT_DIR`** (the directory you were in when you ran `sbatch`) when it contains `dev/run_local_kits.sh`. **Submit from inside the repo:** `cd /path/to/axis-inference-pipeline && sbatch dev/slurm_….job`, or set **`AXIS_REPO_ROOT`**: `sbatch --export=ALL,AXIS_REPO_ROOT=/path/to/axis-inference-pipeline dev/slurm_….job`. If the venv is not `<repo>/.venv`, set **`AXIS_VENV_DIR`**. Run **`./dev/setup_local_models.sh` once** in that clone on the cluster so `.venv/bin/axis-pn` exists on the shared filesystem.

### Submit a single-patient CPU job (`xtreme`)

The batch file requests **1 node**, **96 CPUs**, **2.0 TB RAM**, partition **`xtreme`**, and **no** wall-clock limit (your site may still inject a default cap—add `#SBATCH --time=…` to the job file if required).

```bash
cd /path/to/axis-inference-pipeline
chmod +x dev/slurm_xtreme_kits_cpu.job
# Optional: export AXIS_KITS_ROOT=/your/path/c4kc_kits
# Optional: export AXIS_WEIGHTS_DIR=/your/path/pnvrn_folds
# Optional: export AXIS_WORK_ROOT=/your/scratch/axis-runs
sbatch dev/slurm_xtreme_kits_cpu.job
```

Run a specific case (default in the script is `KiTS-00000`):

```bash
sbatch --export=ALL,CASE_NAME=KiTS-00042 dev/slurm_xtreme_kits_cpu.job
```

Logs: `axis-kits-cpu-<jobid>.out` / `.err` in the submission directory.

**Runtime (rough, one patient, CPU):** dominated by TotalSegmentator (full **total** task) and nnU-Net tumor inference, then 25-fold SWP inference. Expect **on the order of several hours** per typical KiTS CT on a large CPU node—often roughly **~4–12+ hours** depending on voxel size, slice count, filesystem speed, and cluster load. Treat this as a **dry-run / validation** window, not a tight SLA.

### GPU job (`gpu` partition)

`dev/slurm_gpu_kits.job` requests **partition `gpu`**, **1 task**, **12 CPUs/task**, **`--mem=90000`** (megabytes on typical Slurm), **`--gres=gpu:1`**, and runs with **`AXIS_DEVICE=cuda`** (override with `AXIS_DEVICE` if needed). Edit the `#SBATCH` lines if your site uses different GPU or memory syntax.

```bash
chmod +x dev/slurm_gpu_kits.job
sbatch dev/slurm_gpu_kits.job
```

**Do you need a separate venv?** **No** — use the **same** `.venv` from `./dev/setup_local_models.sh` as for CPU. You still need **`axis-pn`** and deps installed there. For **`--device cuda`**, the PyTorch inside that venv must be **CUDA-enabled** (many default `pip install torch` wheels on clusters are CPU-only). After the normal setup, install a CUDA build that matches your node’s driver/CUDA stack, e.g. follow [PyTorch’s install selector](https://pytorch.org/get-started/locally/), or use a cluster module for PyTorch and point `AXIS_PYTHON` at that environment if your admins recommend it.

### Parity with Docker (same software path)

| Piece | Docker (CPU image) | Cluster (this repo) |
| --- | --- | --- |
| Entry | `axis-pn predict … --device cpu` (see `dev/docker-predict.sh`) | `dev/run_local_kits.sh` → same CLI flags + CT series selection |
| nnU-Net v1/v2 + TotalSegmentator dirs | Set in `Dockerfile` / `docker/entrypoint.sh` | Set by `dev/setup_local_models.sh` → `dev/axis_local_env.sh` |
| Tumor model | Task135 zip baked into image | Downloaded by `setup_local_models.sh` |
| Python | 3.11 in `Dockerfile` | 3.10 default in `setup_local_models.sh` (`AXIS_PYTHON` to override; `>=3.10` per `pyproject.toml`) |

For a laptop or server **with** Docker, use [step 5 under Quick start](#5-same-thing-with-docker) (or the [Docker](#docker) section) so external testers can reproduce the same flow without a cluster.

## Docker

The image installs `axis-inference-pipeline`, `TotalSegmentator`, nnU-Net v2, **legacy nnU-Net v1** (KiTS21 tumor), downloads TotalSegmentator `total` task weights, and installs **Task135_KiTS2021** under `/opt/nnunet/v1/results`. You still need **PNvsRN `.pth` weights** (same tree as `pnvrn_folds/`): mount them and pass `--weights-dir`.

By default, TotalSegmentator runs the full `total` task unless you pass `--totalseg-extra` or set `AXIS_TOTALSEG_EXTRA`.

### One-time: build the image

From the repo root:

```bash
git clone git@github.com:AIM-HI-Lab/axis-inference-pipeline.git
cd axis-inference-pipeline
docker build -t axis-inference-pipeline:local .
```

(GPU host: `docker build -f Dockerfile.gpu -t axis-inference-pipeline:gpu .`)

### Run: point at your DICOMs

1. Put **DICOMs** anywhere on disk (nested folders are fine). If you only have a zip/tar, extract it first so you have a directory of `.dcm` files.
2. Put **PNvsRN checkpoints** in a directory with the same layout as `pnvrn_folds/` (25× `.pth` under subfolders). If that folder is not in the clone, copy or symlink it next to the repo.
3. Choose an **empty output directory** for results.

**Easiest (helper script)** — builds paths for you:

```bash
chmod +x dev/docker-predict.sh
./dev/docker-predict.sh /path/to/dicom/folder /path/to/output /path/to/pnvrn_folds
```

Optional flags after `--` go to `axis-pn predict`, e.g. `-- --series-uid 1.2.840...`

**Same thing with `docker run`:**

```bash
docker run --rm \
  -v /path/to/dicoms:/data/dicom:ro \
  -v /path/to/output:/data/out \
  -v /path/to/pnvrn_folds:/models:ro \
  axis-inference-pipeline:local \
  predict \
  --input /data/dicom \
  --work-dir /data/out \
  --weights-dir /models \
  --device cpu
```

Recursive checkpoint discovery is **on by default**; you do not need `--checkpoint-dir-recursive` unless you turned it off.

**Compose (optional):**

```bash
mkdir -p data/dicom data/out
# copy or symlink DICOMs into data/dicom; ensure pnvrn_folds exists beside compose file
docker compose run --rm axis-pn predict \
  --input /data/dicom \
  --work-dir /data/out \
  --weights-dir /models \
  --device cpu
```

Override bind paths with env: `DICOM_DIR`, `OUT_DIR`, `WEIGHTS_DIR`.

### GPU (NVIDIA)

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.
2. Build `Dockerfile.gpu` (see above).
3. Run with GPU, for example:

```bash
AXIS_DOCKER_IMAGE=axis-inference-pipeline:gpu AXIS_DOCKER_GPU=1 AXIS_DEVICE=cuda \
  ./dev/docker-predict.sh /path/to/dicom /path/to/out /path/to/pnvrn_folds
```

Or: `docker compose --profile gpu run --rm axis-pn-gpu predict ... --device cuda` (requires a GPU-capable Compose setup).

### What you get

Under the output/work dir: `cases/`, `swp_manifest.json`, `predictions/predictions.json`, `run_manifest.json` (see [Output Layout](#output-layout)).
