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

- creates `.venv312`
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

## Docker

The image installs `axis-inference-pipeline`, `TotalSegmentator`, nnU-Net v2, **legacy nnU-Net v1** (KiTS21 tumor), downloads TotalSegmentator `total` task weights, and installs **Task135_KiTS2021** under `/opt/nnunet/v1/results`. You still need **PNvsRN `.pth` weights** (same tree as `pnvrn_folds/`): mount them and pass `--weights-dir`.

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
