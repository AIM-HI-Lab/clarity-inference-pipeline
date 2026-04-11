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

The image installs `axis-inference-pipeline`, `TotalSegmentator`, nnU-Net v2, **legacy nnU-Net v1** (for the KiTS21 tumor model), downloads TotalSegmentator `total` task weights, and installs the public **Task135_KiTS2021** nnU-Net v1 checkpoint into `/opt/nnunet/v1/results`. You still need the **PNvsRN `.pth` weights** (the `pnvrn_folds/` tree from this repo, or any equivalent directory on disk): mount it read-only and pass `--weights-dir`.

Build:

```bash
docker build -t axis-pn .
```

### Run on another machine (DICOM folder + weights)

1. Put **PNvsRN checkpoints** on disk (same layout as `pnvrn_folds/`, with `.pth` files under subfolders).
2. Put **DICOMs** in a directory tree (one or more series). If someone only gave you a **download link** (HTTP/S) to a zip or tarball, download and extract it first, then point the container at the folder that contains the `.dcm` files:

```bash
mkdir -p ~/axis-data/dicoms && cd ~/axis-data/dicoms
curl -fL "https://example.com/your-dicom-archive.zip" -o dicoms.zip
unzip dicoms.zip   # or tar xf …
# Use the directory that actually contains the series (nested subdirs are fine)
```

3. Pick an output directory and run:

```bash
docker run --rm \
  -v /path/to/dicoms:/input:ro \
  -v /path/to/output:/output \
  -v /path/to/pnvrn_folds:/models:ro \
  axis-pn predict \
  --input /input \
  --work-dir /output \
  --weights-dir /models \
  --checkpoint-dir-recursive
```

Replace `/path/to/dicoms` with the folder from step 2. The pipeline discovers series under `--input`; you do not need to flatten DICOMs into a single directory if they already live in a per-series folder.

For **GPU**, use your environment’s NVIDIA Container Toolkit settings and add `--device cuda` to the `predict` command.

**CPU-only** example (same mounts):

```bash
docker run --rm \
  -v /path/to/dicoms:/input:ro \
  -v /path/to/output:/output \
  -v /path/to/pnvrn_folds:/models:ro \
  axis-pn predict \
  --input /input \
  --work-dir /output \
  --weights-dir /models \
  --checkpoint-dir-recursive \
  --device cpu
```
