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

**Labels:** You do not need ground-truth diagnosis labels. Point `--input` at a folder of DICOMs in the usual layout (see [Input path](#input-path-always-a-folder)); the pipeline only needs imaging plus internally built segmentations. **`predictions/predictions.json` contains model predictions and probabilities only** (`evaluation_mode: "prediction_only"` in the JSON metadata). Optional `label` fields in `swp_manifest.json` are ignored for reporting.

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
- `--dicom-backend auto|dcm2niix|sitk`: DICOM→NIfTI via external **dcm2niix** (default **auto** picks it when on `PATH`) or in-process **SimpleITK** (GDCM). Env `AXIS_DICOM_BACKEND` overrides the flag.
- `--totalseg-extra "..."` or env `AXIS_TOTALSEG_EXTRA`: optional extra TotalSegmentator CLI arguments (default: full `total` task).
- `--tumor-extra "..."` or env `AXIS_TUMOR_EXTRA`: optional extra nnU-Net tumor-segmentation CLI arguments.
- `--skip-inference`: stop after building SWP-ready NIfTI inputs
- `--skip-tumor`: create kidney-only SWP masks
- `--fail-on-empty-tumor`: abort if tumor segmentation is empty (default: **continue** and skip axis-pn for that case only)
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

`predictions/predictions.json` is the averaged ensemble output across all checkpoints (per-case `ensemble_pred`, `ensemble_pred_probs`, and per-checkpoint rows). It does **not** include accuracy metrics or patient labels—suitable for external DICOM folders where labels are unknown.

The vendored SWP CLI (`python -m segmentation_weighted_planes.inference`) follows the same rule: **prediction-only** combined JSON, no AUC/accuracy in the output.

## Local Dev Runner

There is a batch helper at **`dev/run_local_dicom_batch.sh`**: one folder per patient under a data root, each containing a nested DICOM tree (same rules as `axis-pn predict --input`). **TCIA KiTS19** is a convenient public test; **any** layout with one case per subdirectory works.

First-time setup:

```bash
./dev/setup_local_models.sh
```

That script:

- creates `.venv` (via `python3.10` unless `AXIS_PYTHON` is set)
- installs **CUDA `torch` / `torchvision` from PyTorch’s wheel index** (before and after `pip install -e .`) so you do not stay on a **`+cpu`** build from PyPI
- installs this package plus `TotalSegmentator`, `nnunetv2`, and `nnunet`
- creates repo-local nnU-Net env directories
- writes `dev/axis_local_env.sh`
- downloads the public KiTS21 pretrained tumor model
- preps the TotalSegmentator cache location

Then run one case folder (subdirectory name under the data root):

```bash
bash dev/run_local_dicom_batch.sh my_case_001
```

Run **every** immediate subdirectory under the data root (sorted): pass `ALL` (or `--all` / `-a`):

```bash
AXIS_DICOM_CASES_ROOT=/path/to/parent bash dev/run_local_dicom_batch.sh ALL
```

(`AXIS_KITS_ROOT` is still accepted as an alias for `AXIS_DICOM_CASES_ROOT`.)

By default that script uses:

- Data root: `~/Desktop/kits_data/C4KC-KiTS-NBIA-manifest (1)/c4kc_kits` (override with `AXIS_DICOM_CASES_ROOT`)
- weights: `<repo>/pnvrn_folds`
- output root: `<repo>/local-runs`
- device: `cpu`
- tumor backend: `nnUNet v1` public KiTS21 baseline (`Task135_KiTS2021`, `3d_cascade_fullres`)

You can override them with:

```bash
AXIS_DEVICE=cuda AXIS_DICOM_CASES_ROOT=/path/to/parent bash dev/run_local_dicom_batch.sh my_case_001
```

If nnU-Net finds **no tumor voxels** for a case, the pipeline **continues** by default and **skips axis-pn** for that case only. Set **`AXIS_FAIL_ON_EMPTY_TUMOR=1`** or pass **`--fail-on-empty-tumor`** to `axis-pn predict` to abort instead.

## Cluster (Slurm)

This is for **running on a shared HPC node** without Docker. It uses the **same** `axis-pn predict` path as `dev/run_local_dicom_batch.sh` and **`dev/docker-predict.sh`**: TotalSegmentator **total** task, **nnU-Net v1** KiTS21 tumor model (`Task135`, `3d_cascade_fullres`), then SWP ensemble inference. The checked-in Slurm script uses **`--device cuda`**; use CPU only if you edit it (see below).

### Data layout (one folder per case)

Point **`AXIS_DICOM_CASES_ROOT`** at the directory that **directly contains** one folder per case (any naming). Example with KiTS-style names:

```text
/path/to/dicom-cases/
  case-00000/
    ... nested DICOM series folders ...
  case-00001/
  ...
```

Example used for AIM-HI Lab storage: `/home/jonnalr/AIM-HI-Lab/kits-dicoms`. Each case directory should contain the usual nested DICOM tree (the helper picks a diagnostic **CT** series and skips **SEG**). If your tree has an extra level (e.g. `c4kc_kits/KiTS-XXXXX`), set **`AXIS_DICOM_CASES_ROOT`** to that parent. **`AXIS_KITS_ROOT`** is a deprecated alias for the same variable.

### One-time setup on the cluster

From a **login or build node** (adjust module names to your site):

```bash
git clone https://github.com/AIM-HI-Lab/axis-inference-pipeline.git
cd axis-inference-pipeline
module load python/gpu/3.10.6    # example: GPU-capable Python + CUDA libs on PATH
export AXIS_PYTHON="$(command -v python3.10 2>/dev/null || command -v python3)"
./dev/setup_local_models.sh
```

`setup_local_models.sh` installs **`torch` / `torchvision` from the [PyTorch CUDA index](https://pytorch.org/get-started/locally/) twice** — **before** and **after** `pip install -e .` — so `pip` does not leave you on a **`+cpu`** wheel. `AXIS_PYTORCH_CUDA` (**`auto`** by default) reads **`nvidia-smi`** when available; on **Linux** without **`nvidia-smi`** it defaults to **`cu118`**. Override: **`AXIS_PYTORCH_CUDA=cu124`**, **`cu121`**, **`cu118`**, **`cpu`**, or **`skip`**.

DICOM→NIfTI: **`pip install -e .`** brings **SimpleITK**; use **`--dicom-backend auto`** (default) or **`sitk`** without **dcm2niix**. Optional: **`module load …`** for **dcm2niix**, or **`export AXIS_DCM2NIIX=/path/to/dcm2niix`**.

That creates `.venv`, installs dependencies, downloads TotalSegmentator **total** weights and **Task135_KiTS2021**, and writes **`dev/axis_local_env.sh`**. On Linux, if you selected a CUDA variant but **`torch.version.cuda`** is still **`None`**, setup **exits with an error** so you do not proceed with a broken env.

Copy or link **PNvsRN weights** (`pnvrn_folds/`-style tree, 25× `.pth`) somewhere readable on the cluster and set `AXIS_WEIGHTS_DIR` if it is not `<repo>/pnvrn_folds`.

**`No such file or directory: 'dcm2niix'`:** With **`--dicom-backend auto`** (the default), the pipeline uses **SimpleITK** (GDCM) when `dcm2niix` is not on `PATH`, as long as **`pip install`** pulled **SimpleITK** (declared in this package). To force that path: **`export AXIS_DICOM_BACKEND=sitk`**. Alternatively install **`dcm2niix`** and/or set **`AXIS_DCM2NIIX`** to its full path. Batch jobs often have a minimal `PATH`; **`module load`** may be needed if you insist on **dcm2niix**.

**Slurm says it cannot find `axis-pn` / `.venv`, or `REPO_ROOT` looks like `.../slurm/.../spool/...`:** Slurm **copies** the batch script to a **spool** directory and runs that copy, so **`$BASH_SOURCE` is not inside your git clone**. The job script resolves the repo when **`SLURM_SUBMIT_DIR`** contains `dev/run_local_dicom_batch.sh`. **Submit from inside the repo:** `cd /path/to/axis-inference-pipeline && sbatch dev/slurm_gpu_kits.job`, or set **`AXIS_REPO_ROOT`**: `sbatch --export=ALL,AXIS_REPO_ROOT=/path/to/axis-inference-pipeline dev/slurm_gpu_kits.job`. If the venv is not `<repo>/.venv`, set **`AXIS_VENV_DIR`**. Run **`./dev/setup_local_models.sh` once** in that clone on the cluster so `.venv/bin/axis-pn` exists on the shared filesystem.

### GPU batch job (`dev/slurm_gpu_kits.job`)

The checked-in script requests **partition `gpu`**, **1 task**, **12 CPUs/task**, **`--mem=32200`** (megabytes; edit as needed), **`--gres=gpu:1`**, and **`AXIS_DEVICE=cuda`**. It **`exec`s `dev/run_local_dicom_batch.sh`**; **`CASE_NAME` defaults to `ALL`** (every immediate subdirectory under **`AXIS_DICOM_CASES_ROOT`**). Before the pipeline it runs **`dev/check_gpu_env.sh`** unless **`AXIS_DEBUG_CUDA=0`**. Edit **`#SBATCH`** if your site uses different GPU flags or partitions.

```bash
chmod +x dev/slurm_gpu_kits.job
sbatch dev/slurm_gpu_kits.job
```

Log files are typically **`axis-kits-gpu-<jobid>.out`** / **`.err`** in the submission directory (see the **`====`** GPU/torch block in `.out`).

**CPU-only cluster:** Copy **`dev/slurm_gpu_kits.job`** in your clone, set **`export AXIS_DEVICE=cpu`**, remove or relax **`--gres`**, point **`#SBATCH --partition`** at a CPU partition, and increase **`--mem`** / **`--cpus-per-task`** as needed — the same **`run_local_dicom_batch.sh`** entrypoint still applies.

**GPU sanity check without the full pipeline:** On an interactive GPU session, run **`./dev/check_gpu_env.sh`** (same script the Slurm job invokes first).

**Do you need a separate venv?** **No** — use the same `.venv` from `./dev/setup_local_models.sh`. GPU jobs need a **CUDA** PyTorch build that fits the **driver** on the compute nodes; `setup_local_models.sh` handles that via **`AXIS_PYTORCH_CUDA`** (see above). If you see **`The NVIDIA driver on your system is too old`** from TotalSegmentator or nnU-Net, reinstall with **`AXIS_PYTORCH_CUDA=cu118`** (or run the `pip install …/whl/cu118` one-liner above).

**Runtime (rough):** dominated by TotalSegmentator, nnU-Net, then 25-fold SWP — often **hours per case** depending on volume size and filesystem; treat as a validation window, not a tight SLA.

**Still no GPU after that?**

1. **Empty `CUDA_VISIBLE_DEVICES`** — Slurm did not assign a GPU. Fix partition / **`#SBATCH`** (try **`--gpus-per-node=1`**, **`--gpus-per-task=1`**, or your center’s GPU flags). Ask admins which partition and flags attach GPUs.
2. **`torch.version.cuda` is `None`** or **`torch.__version__` ends with `+cpu`** — CPU-only PyTorch (the GPU is fine). Reinstall CUDA wheels from the repo root, e.g. **`cu124`** if your driver reports CUDA 12.x (`nvidia-smi`):
   ```bash
   .venv/bin/pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
   Or **`cu118`** on older drivers. Then **`AXIS_PYTORCH_CUDA=cu124 ./dev/setup_local_models.sh`** on the next fresh venv so `pip install -e .` does not leave you on `+cpu` again without the second install step.
3. **CUDA build but `is_available()` False** — Often **driver vs wheel** (try **`cu118`**) or missing system libraries. Many clusters need **`module load cuda`** (and sometimes **`cudnn`**) *before* Python runs; add those lines **inside** your Slurm script (after **`cd`**, before **`axis-pn`**), using your site’s module names.
4. **Login-node setup** — If **`nvidia-smi`** was missing during setup, **`auto`** now defaults **Linux → `cu118`** so you are less likely to get CPU-only torch by accident.

**nnU-Net / TotalSegmentator “CUDA is not available”:** Same PyTorch as the rest of the venv; fix the checks above first.

### Parity with Docker (same software path)

| Piece | Docker (CPU image) | Cluster (this repo) |
| --- | --- | --- |
| Entry | `axis-pn predict … --device cpu` (see `dev/docker-predict.sh`) | `dev/run_local_dicom_batch.sh` → same CLI flags + CT series selection |
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
