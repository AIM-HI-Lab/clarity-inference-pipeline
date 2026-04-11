"""
SWPDataset_V5
=============
Key upgrades vs V4
- 3D component-aware sampling (not "2D connected components per slice")
- Patch taxonomy w/ quotas: core / boundary / support / hard_neg / background
- Optional 2.5D+ (multi-slice stacks) per view
- Patch-level metadata cached (fractions, center coords, type, etc.) to debug without PHI
- Still organ-agnostic via seg_class_definitions + patch_quotas (works for kidney/prostate/bladder)
- Same external API shape: get_instance / get_instances / assemble_batch_np / get_batch_mixup_np

Notes
- This keeps the "cache-to-disk then fast training" workflow.
- It does NOT assume tumor is inside kidney label; KiTS-style mutually-exclusive labels are fine.
"""

from __future__ import annotations
import os
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

from segmentation_weighted_planes.data_loader_v3_core import (
    create_cum_wts,
    hash_str,
    no_augment_fn,
    reorient_to_identity,
)
from segmentation_weighted_planes.training.training_parameters import TrainingParameters

try:
    from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label
except Exception as e:
    raise ImportError(
        "SWPDataset_V5 requires scipy (ndimage). Please `pip install scipy`."
    ) from e


Numeric = Union[int, float]
VIEWS = ["axial", "coronal", "sagittal"]

RSMP_PIXEL_SIZE = 1
INNER_PATCH_SIZE_PX = 267

# legacy V4 diagonal crop then resize -> 267
PATCH_SIZE_PX_LEGACY = int(np.sqrt((INNER_PATCH_SIZE_PX**2) * 2)) + 1


# --------------------------
# Config / typing
# --------------------------
@dataclass(frozen=True)
class V5CacheConfig:
    # sampling
    max_patches_per_view: int = 240
    patch_quotas: Dict[str, float] = None  # fractions that sum ~1.0
    seed: int = 69_420 # 1337

    # 2.5D+
    stack_slices: int = 1  # 1=2D, 3/5/7 = multi-slice stack

    # geometry / crop
    crop_mode: str = "direct"  # "direct" (no resize) or "legacy" (diagonal crop + resize)
    inner_patch_size_px: int = INNER_PATCH_SIZE_PX
    legacy_patch_size_px: int = PATCH_SIZE_PX_LEGACY

    # distance-based sampling
    boundary_radius_vox: int = 1        # boundary thickness around tumor
    context_band_min_vox: int = 2       # support context must be at least this far from tumor
    context_band_max_vox: int = 18      # ...and within this band (voxels)
    hard_neg_min_vox: int = 25          # hard neg must be >= this far from tumor, but still in support
    background_margin_vox: int = 10     # avoid borders for background sampling

    # weighting
    weight_by_component_volume: bool = True
    weight_by_patch_type: Dict[str, float] = None  # optional multipliers

    def to_json(self) -> str:
        d = dict(self.__dict__)
        # normalize defaults
        if d["patch_quotas"] is None:
            d["patch_quotas"] = TrainingParameters.DEFAULT_PATCH_QUOTAS
        if d["weight_by_patch_type"] is None:
            d["weight_by_patch_type"] = TrainingParameters.DEFAULT_WEIGHT_BY_PATCH_TYPE
        # force int stack_slices odd
        if int(d["stack_slices"]) % 2 == 0:
            raise ValueError("stack_slices must be odd (1,3,5,...)")
        return json.dumps(d, sort_keys=True)


PATCH_TYPE_TO_ID = {
    "core": 0,
    "boundary": 1,
    "support": 2,
    "hard_neg": 3,
    "background": 4,
}
ID_TO_PATCH_TYPE = {v: k for k, v in PATCH_TYPE_TO_ID.items()}


# --------------------------
# Utilities
# --------------------------
def _safe_choice(rng: np.random.Generator, arr: np.ndarray) -> Optional[np.ndarray]:
    """Pick a random row from arr (N x 3 coords). Return None if empty."""
    if arr is None or arr.size == 0:
        return None
    idx = rng.integers(0, arr.shape[0])
    return arr[idx]


def _coords_of(mask: np.ndarray) -> np.ndarray:
    """Return coordinates (N,3) of True voxels (z,y,x)."""
    return np.argwhere(mask)


def _fraction_in_patch(seg2d: np.ndarray, class_ids: List[int]) -> float:
    if seg2d.size == 0:
        return 0.0
    if not class_ids:
        return 0.0
    m = np.zeros_like(seg2d, dtype=bool)
    for cid in class_ids:
        m |= (seg2d == cid)
    return float(m.mean())


def _extract_plane_stack(
    img3d: np.ndarray,
    seg3d: np.ndarray,
    center_zyx: Tuple[int, int, int],
    axis_ind: int,
    patch_size_px: int,
    stack_slices: int,
    pad_val_img: int = 0,
    pad_val_seg: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a (stack_slices, patch_size_px, patch_size_px) image stack AND a 2D seg slice (center slice),
    from a 3D volume for a given axis_ind plane.
      axis_ind=0 -> axial (z slices, yx plane)
      axis_ind=1 -> coronal (y slices, zx plane) but returned as (H,W) in that plane
      axis_ind=2 -> sagittal (x slices, zy plane)
    """
    zc, yc, xc = center_zyx
    half = patch_size_px // 2
    k = stack_slices
    kh = k // 2

    # Select slice indices along the plane axis
    if axis_ind == 0:
        # axial: slice index is z
        slice_inds = np.arange(zc - kh, zc + kh + 1)
        plane_img = img3d
        plane_seg = seg3d
        # in-plane dims are (y,x)
        in0, in1 = yc, xc
        plane_shape = (img3d.shape[1], img3d.shape[2])
    elif axis_ind == 1:
        # coronal: slice index is y, in-plane dims (z,x)
        slice_inds = np.arange(yc - kh, yc + kh + 1)
        # We'll use img3d[:, y, :] which is (z,x)
        plane_shape = (img3d.shape[0], img3d.shape[2])
        in0, in1 = zc, xc
    elif axis_ind == 2:
        # sagittal: slice index is x, in-plane dims (z,y)
        slice_inds = np.arange(xc - kh, xc + kh + 1)
        plane_shape = (img3d.shape[0], img3d.shape[1])
        in0, in1 = zc, yc
    else:
        raise ValueError("axis_ind must be 0,1,2")

    # pad in-plane so crops are safe
    pad = half
    H, W = plane_shape
    # padded arrays per-slice will be extracted on demand (cheap enough)

    stack_imgs = []
    center_seg2d = None

    for si_i, si in enumerate(slice_inds):
        if axis_ind == 0:
            if si < 0 or si >= img3d.shape[0]:
                img2d = np.full(plane_shape, pad_val_img, dtype=img3d.dtype)
                seg2d = np.full(plane_shape, pad_val_seg, dtype=seg3d.dtype)
            else:
                img2d = plane_img[si, :, :]
                seg2d = plane_seg[si, :, :]
        elif axis_ind == 1:
            if si < 0 or si >= img3d.shape[1]:
                img2d = np.full(plane_shape, pad_val_img, dtype=img3d.dtype)
                seg2d = np.full(plane_shape, pad_val_seg, dtype=seg3d.dtype)
            else:
                img2d = img3d[:, si, :]  # (z,x)
                seg2d = seg3d[:, si, :]
        else:  # axis_ind == 2
            if si < 0 or si >= img3d.shape[2]:
                img2d = np.full(plane_shape, pad_val_img, dtype=img3d.dtype)
                seg2d = np.full(plane_shape, pad_val_seg, dtype=seg3d.dtype)
            else:
                img2d = img3d[:, :, si]  # (z,y)
                seg2d = seg3d[:, :, si]

        # pad in-plane
        pimg = np.pad(img2d, pad, constant_values=pad_val_img)
        pseg = np.pad(seg2d, pad, constant_values=pad_val_seg)

        # crop
        y0 = pad + in0 - half
        y1 = y0 + patch_size_px
        x0 = pad + in1 - half
        x1 = x0 + patch_size_px

        crp_img = pimg[y0:y1, x0:x1]
        crp_seg = pseg[y0:y1, x0:x1]

        # If crop goes weird (shouldn't due to pad), hard-fix
        if crp_img.shape != (patch_size_px, patch_size_px):
            fixed = np.full((patch_size_px, patch_size_px), pad_val_img, dtype=pimg.dtype)
            hh = min(patch_size_px, crp_img.shape[0])
            ww = min(patch_size_px, crp_img.shape[1])
            fixed[:hh, :ww] = crp_img[:hh, :ww]
            crp_img = fixed

        if crp_seg.shape != (patch_size_px, patch_size_px):
            fixeds = np.full((patch_size_px, patch_size_px), pad_val_seg, dtype=pseg.dtype)
            hh = min(patch_size_px, crp_seg.shape[0])
            ww = min(patch_size_px, crp_seg.shape[1])
            fixeds[:hh, :ww] = crp_seg[:hh, :ww]
            crp_seg = fixeds

        stack_imgs.append(crp_img)
        if si_i == kh:
            center_seg2d = crp_seg

    stack_imgs = np.stack(stack_imgs, axis=0)  # (k, H, W)
    assert center_seg2d is not None
    return stack_imgs, center_seg2d


def _resize_stack_and_seg(
    img_stack: np.ndarray,
    seg2d: np.ndarray,
    out_hw: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resize image stack (k,H,W) -> (k,out_hw,out_hw) and seg2d -> (out_hw,out_hw)
    """
    k = img_stack.shape[0]
    out_imgs = np.zeros((k, out_hw, out_hw), dtype=np.uint8)
    for i in range(k):
        out_imgs[i] = cv2.resize(img_stack[i], (out_hw, out_hw), interpolation=cv2.INTER_LINEAR).astype(
            np.uint8
        )
    out_seg = cv2.resize(seg2d, (out_hw, out_hw), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    return out_imgs, out_seg


def _build_sampling_masks(
    mask_np: np.ndarray,
    class_defs: Dict[str, List[int]],
    cfg: V5CacheConfig,
) -> Dict[str, np.ndarray]:
    """
    Returns boolean 3D masks for:
      - primary (tumor)
      - secondary (support organ)
      - boundary (primary boundary band)
      - support_band (secondary near primary, within [min,max] vox)
      - hard_neg (secondary far from primary, >= hard_neg_min)
      - background (not secondary and not primary)
    """
    primary_ids = class_defs.get("primary", []) or []
    secondary_ids = class_defs.get("secondary", []) or []

    primary = np.zeros_like(mask_np, dtype=bool)
    for cid in primary_ids:
        primary |= (mask_np == cid)

    secondary = np.zeros_like(mask_np, dtype=bool)
    for cid in secondary_ids:
        secondary |= (mask_np == cid)

    # Tumor boundary: dilate - erode (band)
    if primary.any():
        dil = binary_dilation(primary, iterations=max(1, cfg.boundary_radius_vox))
        ero = binary_erosion(primary, iterations=max(1, cfg.boundary_radius_vox))
        boundary = np.logical_and(dil, np.logical_not(ero))
    else:
        boundary = np.zeros_like(primary, dtype=bool)

    # Distance to primary (for context/hard neg)
    # distance_transform_edt expects False as features, True as background -> invert
    if primary.any():
        dist = distance_transform_edt(~primary)  # 0 at tumor voxels, grows away
    else:
        dist = np.full_like(mask_np, fill_value=1e9, dtype=np.float32)

    support_band = secondary & (dist >= cfg.context_band_min_vox) & (dist <= cfg.context_band_max_vox)
    hard_neg = secondary & (dist >= cfg.hard_neg_min_vox)

    background = (~primary) & (~secondary)

    return {
        "primary": primary,
        "secondary": secondary,
        "boundary": boundary,
        "support_band": support_band,
        "hard_neg": hard_neg,
        "background": background,
        "dist_to_primary": dist,
    }


def _label_3d_components(primary_mask: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Label 3D connected components of primary (tumor).
    Returns (lab_vol, n_comp, comp_sizes) where comp_sizes is (n_comp,) voxel counts.
    """
    lab_vol, n = label(primary_mask.astype(np.uint8))
    if n == 0:
        return lab_vol, 0, np.zeros((0,), dtype=np.int64)
    sizes = np.bincount(lab_vol.ravel())[1:]  # skip 0 background
    return lab_vol, n, sizes.astype(np.int64)


def _sample_center_from_component(
    rng: np.random.Generator,
    lab_vol: np.ndarray,
    comp_sizes: np.ndarray,
    cfg: V5CacheConfig,
) -> Optional[Tuple[int, int, int]]:
    if comp_sizes.size == 0:
        return None
    if cfg.weight_by_component_volume:
        w = comp_sizes.astype(np.float64)
        w = w / w.sum()
        comp_id = int(rng.choice(np.arange(1, comp_sizes.size + 1), p=w))
    else:
        comp_id = int(rng.integers(1, comp_sizes.size + 1))
    coords = np.argwhere(lab_vol == comp_id)
    c = _safe_choice(rng, coords)
    if c is None:
        return None
    return int(c[0]), int(c[1]), int(c[2])

def _call_augment(fn, img, mask):
    out = fn(img, mask)
    if isinstance(out, tuple):
        if len(out) == 2:
            return out
        if len(out) >= 3:
            return out[0], out[1]
    raise ValueError(f"Unexpected return from {fn.__name__}: {type(out)}")

def no_augment_fn_v5(img: np.ndarray, mask: np.ndarray):
    """
    V5-safe replacement for V3 no_augment_fn.

    img: (C,H,W) or (H,W)
    mask: (2,H,W) expected in V5
    Returns float32 arrays unchanged (no padding/cropping).
    """
    if img.ndim == 2:
        img = img[None, :, :]
    img = img.astype(np.float32, copy=False)
    mask = mask.astype(np.float32, copy=False)
    return img, mask

# --------------------------
# Caching per case (V5)
# --------------------------
def cache_case_v5(
    img_pth: Path,
    mask_pth: Path,
    case_cache_pth: Path,
    class_defs: Dict[str, List[int]],
    cfg: V5CacheConfig,
):
    """
    Writes per-view NPZs with:
      imgs: (N, K, H, W) uint8      (K=stack_slices)
      masks: (N, H, W) uint8       (center-slice labels)
      mar_wts: (N,) float32
      cum_wts: (N,) float32
      meta arrays:
        patch_type: (N,) uint8
        center_zyx: (N,3) int16
        tumor_frac: (N,) float16
        support_frac: (N,) float16
        background_frac: (N,) float16

    Patch generation is component-aware (3D) and quota-based.
    """
    case_cache_pth.mkdir(parents=True, exist_ok=True)

    seed = (cfg.seed + (hash(str(mask_pth)) % 1_000_000_000)) % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)

    # Load + standardize
    img_nib = reorient_to_identity(nib.load(str(img_pth)))
    seg_nib = reorient_to_identity(nib.load(str(mask_pth)))

    img_np = np.asanyarray(img_nib.dataobj).astype(np.float32)
    seg_np = np.round(np.asanyarray(seg_nib.dataobj)).astype(np.uint8)

    # Normalize image -> uint8 (keep your V3 behavior)
    img_np = np.clip(255 * (img_np + 128) / (128 + 256), 0, 255).astype(np.uint8)

    sm = _build_sampling_masks(seg_np, class_defs, cfg)
    primary = sm["primary"]
    if not primary.any():
        raise ValueError(f"Primary object not found in mask for case: {mask_pth}")

    lab_vol, n_comp, comp_sizes = _label_3d_components(primary)
    if n_comp == 0:
        raise ValueError(f"No connected components found in primary for case: {mask_pth}")

    # Quotas
    pq = cfg.patch_quotas or TrainingParameters.DEFAULT_PATCH_QUOTAS
    # normalize quotas defensively
    s = float(sum(max(0.0, v) for v in pq.values()))
    if s <= 0:
        raise ValueError("patch_quotas must sum to > 0")
    pq = {k: max(0.0, float(v)) / s for k, v in pq.items()}

    type_w = cfg.weight_by_patch_type or TrainingParameters.DEFAULT_WEIGHT_BY_PATCH_TYPE

    # Decide crop mode
    if cfg.crop_mode not in ("direct", "legacy"):
        raise ValueError("crop_mode must be 'direct' or 'legacy'")
    if cfg.crop_mode == "legacy":
        crop_px = int(cfg.legacy_patch_size_px)
        out_px = int(cfg.inner_patch_size_px)
    else:
        crop_px = int(cfg.inner_patch_size_px)
        out_px = int(cfg.inner_patch_size_px)

    # Precompute coords for masks (for fast sampling)
    coords_boundary = _coords_of(sm["boundary"])
    coords_support = _coords_of(sm["support_band"])
    coords_hard_neg = _coords_of(sm["hard_neg"])
    coords_background = _coords_of(sm["background"])

    views_meta: Dict[str, Any] = {}

    for axis_ind, view_name in zip([0, 1, 2], VIEWS):
        n_total = int(cfg.max_patches_per_view)

        # assign counts per patch type
        counts = {k: int(round(pq.get(k, 0.0) * n_total)) for k in pq}
        # fix rounding to match total
        while sum(counts.values()) < n_total:
            # add to most important types first
            for k in ["core", "boundary", "support", "hard_neg", "background"]:
                if sum(counts.values()) >= n_total:
                    break
                counts[k] = counts.get(k, 0) + 1
        while sum(counts.values()) > n_total:
            # remove from least important types first
            for k in ["background", "hard_neg", "support", "boundary", "core"]:
                if sum(counts.values()) <= n_total:
                    break
                if counts.get(k, 0) > 0:
                    counts[k] -= 1

        imgs_list: List[np.ndarray] = []
        masks_list: List[np.ndarray] = []
        arb_wts: List[float] = []

        patch_type_list: List[int] = []
        center_list: List[Tuple[int, int, int]] = []
        tumor_frac_list: List[float] = []
        support_frac_list: List[float] = []
        background_frac_list: List[float] = []

        # helper to append patch
        def _append_patch(center_zyx: Tuple[int, int, int], ptype: str, comp_wt: float):
            img_stack, seg2d = _extract_plane_stack(
                img_np, seg_np, center_zyx, axis_ind=axis_ind, patch_size_px=crop_px, stack_slices=cfg.stack_slices
            )
            if cfg.crop_mode == "legacy":
                img_stack, seg2d = _resize_stack_and_seg(img_stack, seg2d, out_hw=out_px)

            imgs_list.append(img_stack)
            masks_list.append(seg2d)

            # weight = component weight * patch-type multiplier
            arb_wts.append(float(comp_wt) * float(type_w.get(ptype, 1.0)))

            patch_type_list.append(int(PATCH_TYPE_TO_ID[ptype]))
            center_list.append(center_zyx)

            tumor_frac_list.append(_fraction_in_patch(seg2d, class_defs.get("primary", []) or []))
            support_frac_list.append(_fraction_in_patch(seg2d, class_defs.get("secondary", []) or []))

            # background is "neither primary nor secondary"
            bg_ids = []
            # we can't list all IDs, so define background fraction as 1 - (tumor or support or other-labeled)
            # but "other" may exist; we include it as non-background
            other_ids = class_defs.get("other", []) or []
            frac_other = _fraction_in_patch(seg2d, other_ids)
            frac_bg = 1.0 - min(1.0, tumor_frac_list[-1] + support_frac_list[-1] + frac_other)
            background_frac_list.append(float(max(0.0, frac_bg)))

        # generate patches by type
        # core: random voxel inside 3D tumor component
        for _ in range(counts.get("core", 0)):
            c = _sample_center_from_component(rng, lab_vol, comp_sizes, cfg)
            if c is None:
                continue
            # component weight = its voxel count (if enabled)
            comp_id = int(lab_vol[c])
            comp_wt = float(comp_sizes[comp_id - 1]) if (cfg.weight_by_component_volume and comp_id > 0) else 1.0
            _append_patch(c, "core", comp_wt)

        # boundary: sample from boundary band, but bias to a component by choosing tumor point then snapping
        for _ in range(counts.get("boundary", 0)):
            # try: pick component center, then search boundary near it using dist
            c = _sample_center_from_component(rng, lab_vol, comp_sizes, cfg)
            if c is None:
                continue
            comp_id = int(lab_vol[c])
            comp_wt = float(comp_sizes[comp_id - 1]) if (cfg.weight_by_component_volume and comp_id > 0) else 1.0

            # pick a boundary voxel globally if available
            bc = _safe_choice(rng, coords_boundary)
            if bc is None:
                # fall back to core
                _append_patch(c, "core", comp_wt)
            else:
                _append_patch((int(bc[0]), int(bc[1]), int(bc[2])), "boundary", comp_wt)

        # support: kidney/prostate/bladder context band near tumor (but not too near)
        for _ in range(counts.get("support", 0)):
            sc = _safe_choice(rng, coords_support)
            if sc is None:
                # fallback: boundary or core
                c = _sample_center_from_component(rng, lab_vol, comp_sizes, cfg)
                if c is None:
                    continue
                comp_id = int(lab_vol[c])
                comp_wt = float(comp_sizes[comp_id - 1]) if (cfg.weight_by_component_volume and comp_id > 0) else 1.0
                _append_patch(c, "core", comp_wt)
            else:
                # weight by nearest component? approximate by sampling component again
                c = _sample_center_from_component(rng, lab_vol, comp_sizes, cfg)
                comp_id = int(lab_vol[c]) if c is not None else 1
                comp_wt = float(comp_sizes[comp_id - 1]) if (cfg.weight_by_component_volume and comp_id > 0) else 1.0
                _append_patch((int(sc[0]), int(sc[1]), int(sc[2])), "support", comp_wt)

        # hard negatives: in support organ but far from tumor (teaches specificity)
        for _ in range(counts.get("hard_neg", 0)):
            hc = _safe_choice(rng, coords_hard_neg)
            if hc is None:
                # fallback to support
                sc = _safe_choice(rng, coords_support)
                if sc is None:
                    continue
                hc = sc
            _append_patch((int(hc[0]), int(hc[1]), int(hc[2])), "hard_neg", comp_wt=1.0)

        # background: random non-organ voxels (small quota)
        for _ in range(counts.get("background", 0)):
            bc = _safe_choice(rng, coords_background)
            if bc is None:
                continue
            _append_patch((int(bc[0]), int(bc[1]), int(bc[2])), "background", comp_wt=1.0)

        # pack arrays
        if len(imgs_list) == 0:
            imgs = np.zeros((0, cfg.stack_slices, out_px, out_px), dtype=np.uint8)
            masks = np.zeros((0, out_px, out_px), dtype=np.uint8)
            mar_wts = np.zeros((0,), dtype=np.float32)
            cum_wts = np.zeros((0,), dtype=np.float32)

            patch_type = np.zeros((0,), dtype=np.uint8)
            center_zyx = np.zeros((0, 3), dtype=np.int16)
            tumor_frac = np.zeros((0,), dtype=np.float16)
            support_frac = np.zeros((0,), dtype=np.float16)
            background_frac = np.zeros((0,), dtype=np.float16)
        else:
            imgs = np.stack(imgs_list, axis=0)  # (N,K,H,W)
            masks = np.stack(masks_list, axis=0)  # (N,H,W)

            arb = np.asarray(arb_wts, dtype=np.float32)
            arb = np.maximum(arb, 1e-8)
            mar_wts = arb / float(arb.sum())
            cum_wts = np.asarray(create_cum_wts(list(arb)), dtype=np.float32)

            patch_type = np.asarray(patch_type_list, dtype=np.uint8)
            center_zyx = np.asarray(center_list, dtype=np.int16)
            tumor_frac = np.asarray(tumor_frac_list, dtype=np.float16)
            support_frac = np.asarray(support_frac_list, dtype=np.float16)
            background_frac = np.asarray(background_frac_list, dtype=np.float16)

        npz_path = case_cache_pth / f"{view_name}.npz"
        np.savez(
            npz_path,
            imgs=imgs,
            masks=masks,
            mar_wts=mar_wts,
            cum_wts=cum_wts,
            patch_type=patch_type,
            center_zyx=center_zyx,
            tumor_frac=tumor_frac,
            support_frac=support_frac,
            background_frac=background_frac,
        )

        views_meta[view_name] = {
            "n_patches": int(imgs.shape[0]),
            "stack_slices": int(cfg.stack_slices),
            "crop_mode": cfg.crop_mode,
        }

    meta = {
        "img_pth": str(img_pth.resolve()),
        "mask_pth": str(mask_pth.resolve()),
        "views": views_meta,
        "cfg": json.loads(cfg.to_json()),
        "class_defs": class_defs,
    }
    with (case_cache_pth / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def get_case_data_v5(case_pth: Path, label: Numeric, item_lsf: dict, case_id: str) -> Dict[str, Any]:
    meta = json.load(open(case_pth / "meta.json", "r"))
    views = []
    for view_name in VIEWS:
        npz_path = case_pth / f"{view_name}.npz"
        vmeta = meta["views"].get(view_name, {"n_patches": 0})
        views.append(
            {
                "name": view_name,
                "path": npz_path,
                "n_patches": int(vmeta.get("n_patches", 0)),
                "loaded": False,
                "imgs": None,
                "masks": None,
                "mar_wts": None,
                "cum_wts": None,
                # metadata arrays
                "patch_type": None,
                "center_zyx": None,
                "tumor_frac": None,
                "support_frac": None,
                "background_frac": None,
            }
        )
    return {
        "name": case_pth.name,  # keep for debugging if you want
        "case_id": str(case_id),  # real identifier
        "views": views,
        "label": label,
        "lsf": item_lsf,
    }


def prep_case_v5(
    img_pth: Path,
    mask_pth: Path,
    label: Numeric,
    case_id: str,
    item_lsf: dict,
    case_cache_pth: Path,
    class_defs: dict,
    cfg: V5CacheConfig,
):
    meta_path = case_cache_pth / "meta.json"
    recache = False

    if not case_cache_pth.exists() or not meta_path.exists():
        recache = True
    else:
        meta = json.load(open(meta_path, "r"))
        if meta.get("img_pth") != str(img_pth.resolve()):
            recache = True
        elif meta.get("mask_pth") != str(mask_pth.resolve()):
            recache = True
        else:
            # cfg change invalidates cache
            old_cfg = meta.get("cfg")
            if old_cfg is None or json.dumps(old_cfg, sort_keys=True) != cfg.to_json():
                recache = True
            # class_def change invalidates cache
            old_cd = meta.get("class_defs")
            if old_cd is None or json.dumps(old_cd, sort_keys=True) != json.dumps(class_defs, sort_keys=True):
                recache = True

    if recache:
        cache_case_v5(
            img_pth=img_pth,
            mask_pth=mask_pth,
            case_cache_pth=case_cache_pth,
            class_defs=class_defs,
            cfg=cfg,
        )

    return get_case_data_v5(case_cache_pth, label, item_lsf, case_id)


def _prep_case_v5_worker(args: Tuple):
    (
        img_pth,
        mask_pth,
        label,
        case_id,
        item_lsf,
        case_cache_pth,
        class_defs,
        cfg_dict,
    ) = args
    cfg = V5CacheConfig(**cfg_dict)
    return prep_case_v5(
        img_pth=img_pth,
        mask_pth=mask_pth,
        label=label,
        case_id=case_id,
        item_lsf=item_lsf,
        case_cache_pth=case_cache_pth,
        class_defs=class_defs,
        cfg=cfg,
    )


# --------------------------
# Augment (same feel as your V4 "fast")
# --------------------------
def augment_fn_fast(img: np.ndarray, mask: np.ndarray):
    """
    img: (K,H,W) or (H,W) float32
    mask: (C,H,W) float32
    """
    if img.ndim == 2:
        img = img[None, :, :]  # (1,H,W)

    # flips
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=-2)  # vertical
        mask = np.flip(mask, axis=-2)
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=-1)  # horizontal
        mask = np.flip(mask, axis=-1)

    # rot90
    k = np.random.randint(0, 4)
    if k:
        img = np.rot90(img, k, axes=(-2, -1))
        mask = np.rot90(mask, k, axes=(-2, -1))

    # V5: keep shapes, just ensure float32
    img, mask = no_augment_fn_v5(img, mask)
    return img, mask


# --------------------------
# Dataset V5
# --------------------------
class SWPDataset_V5:
    """
    V5 is a strict drop-in for the training loop side:
      - get_instance / get_instances
      - assemble_batch_np
      - get_batch_mixup_np

    Major difference:
      - imgs are stored as (K,H,W) stacks (K=stack_slices), not plain (H,W).
      - assemble_batch_np maps stacks into channels as desired.
    """

    def __init__(
            self,
            img_nii_pths: List[Path],
            mask_nii_pths: List[Path],
            labels: List[Numeric],
            case_ids: List[str],
            lsf_values: dict,
            cache_pth: Path,
            seg_class_definitions: Dict[str, List[int]] = None,
            num_workers: int = 1,
            max_loaded_views: int = 96,
            cfg: Optional[V5CacheConfig] = None,
            project_class: Optional[Any] = None,
    ):
        self.seg_class_definitions = seg_class_definitions or {
            "primary": [2],
            "secondary": [1],
            "other": [3],
        }
        self.cfg = cfg or V5CacheConfig()

        # LRU for loaded views
        self.max_loaded_views = int(max_loaded_views)
        self._loaded_views: List[dict] = []

        self.cache_pth = Path(cache_pth)
        self.cache_pth.mkdir(parents=True, exist_ok=True)

        # Keep project_class (used only to derive defaults)
        self.project_class = project_class

        # ---- defaults that the trainer should NOT pass anymore ----
        self.bag_k = int(getattr(project_class, "bag_k", 32)) if project_class else 32

        # Expect project_class.bag_mix like:
        # { "primary":0.5, "boundary":0.2, "secondary":0.2, "background":0.1 } (hard_neg gets leftover)
        self.bag_mix = dict(getattr(project_class, "bag_mix", {})) if project_class else {}

        self.use_seg_default = bool(getattr(project_class, "use_seg", True)) if project_class else True

        sd = int(getattr(project_class, "slab_depth", 1)) if project_class else 1
        # NOTE: 'multi' increases img channels when stack_slices>=3; keep trainer in 'center' until backbone supports 5ch.
        self.stack_mode = "multi" if sd >= 3 else "center"

        # Deterministic RNG for loader-level sampling (can still be stochastic per epoch)
        self._rng = np.random.default_rng(int(getattr(self.cfg, "seed", 1337)))

        # ---- Build tasks ----
        tasks = []
        cfg_dict = json.loads(self.cfg.to_json())

        for i, (img_pth, mask_pth, label, case_id) in enumerate(
                zip(img_nii_pths, mask_nii_pths, labels, case_ids)
        ):
            item_lsf = {k: lsf_values[k][i] for k in lsf_values}

            img_hash = hash_str(str(Path(img_pth).resolve()))
            settings_obj = {
                "img_hash": img_hash,
                "cfg": cfg_dict,
                "class_defs": self.seg_class_definitions,
                "rsmp_pixel_size": RSMP_PIXEL_SIZE,
            }
            settings_hash = hash_str(json.dumps(settings_obj, sort_keys=True))
            case_cache_pth = self.cache_pth / f"{case_id}_{settings_hash}"

            tasks.append(
                (
                    Path(img_pth),
                    Path(mask_pth),
                    label,
                    str(case_id),
                    item_lsf,
                    case_cache_pth,
                    self.seg_class_definitions,
                    cfg_dict,
                )
            )

        self.data: List[dict] = []
        if int(num_workers) == 1:
            for t in tqdm(tasks, desc="Caching cases V5 (serial)"):
                self.data.append(_prep_case_v5_worker(t))
        else:
            with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
                for case_data in tqdm(
                        ex.map(_prep_case_v5_worker, tasks),
                        total=len(tasks),
                        desc=f"Caching cases V5 (workers={num_workers})",
                ):
                    self.data.append(case_data)

    def __len__(self) -> int:
        return len(self.data)

    # ---------- view loading helpers ----------
    def _ensure_view_loaded(self, view: dict):
        if view["loaded"]:
            return

        if len(self._loaded_views) >= self.max_loaded_views:
            evict = self._loaded_views.pop(0)
            for k in [
                "imgs",
                "masks",
                "mar_wts",
                "cum_wts",
                "patch_type",
                "center_zyx",
                "tumor_frac",
                "support_frac",
                "background_frac",
            ]:
                evict[k] = None
            evict["loaded"] = False

        data = np.load(view["path"], mmap_mode="r", allow_pickle=False)
        view["imgs"] = data["imgs"]  # (N,K,H,W) uint8
        view["masks"] = data["masks"]  # (N,H,W) uint8
        view["mar_wts"] = data["mar_wts"]
        view["cum_wts"] = data["cum_wts"]

        # metadata
        view["patch_type"] = data["patch_type"]
        view["center_zyx"] = data["center_zyx"]
        view["tumor_frac"] = data["tumor_frac"]
        view["support_frac"] = data["support_frac"]
        view["background_frac"] = data["background_frac"]

        view["loaded"] = True
        self._loaded_views.append(view)

    # ---------- stats ----------
    def get_class_fractions(self):
        labels = [x["label"] for x in self.data]
        counts: Dict[Any, int] = {}
        for lab in labels:
            key = lab[0] if isinstance(lab, tuple) else lab
            counts[key] = counts.get(key, 0) + 1
        return {k: v / len(labels) for k, v in counts.items()}

    # ---------- case sampling ----------
    def _sample_case(self, balanced: bool) -> int:
        if not balanced:
            return int(self._rng.integers(0, len(self.data)))

        cases_by_label: Dict[Any, List[int]] = {}
        for idx, case in enumerate(self.data):
            lab = case["label"]
            lab = lab[0] if isinstance(lab, tuple) else lab
            cases_by_label.setdefault(lab, []).append(idx)

        labels = list(cases_by_label.keys())
        lab = labels[int(self._rng.integers(0, len(labels)))]
        return int(self._rng.choice(cases_by_label[lab]))

    def _sample_patch_indices(
            self,
            view: dict,
            n: int,
            patch_type_allow: Optional[List[str]] = None,
            prefer_tumorish: bool = False,
            used: Optional[set] = None,
    ):
        """
        Returns a list of img indices from this view.
        - tries to avoid reusing indices already in `used` (within the current bag)
        - samples without replacement when possible
        """
        N = int(view["imgs"].shape[0])
        if N <= 0:
            return []

        # build candidate indices
        cand = np.arange(N, dtype=np.int64)

        if patch_type_allow is not None:
            allowed_ids = np.array([PATCH_TYPE_TO_ID[x] for x in patch_type_allow], dtype=np.uint8)
            ptypes = np.asarray(view["patch_type"], dtype=np.uint8)
            ok = np.isin(ptypes, allowed_ids)
            cand = cand[ok]

        if used:
            # remove already-used indices
            if len(used) > 0:
                mask = np.array([i not in used for i in cand], dtype=bool)
                cand = cand[mask]

        if cand.size == 0:
            return []

        # if we can do true without-replacement
        take = min(int(n), int(cand.size))

        if prefer_tumorish:
            tf = np.asarray(view["tumor_frac"], dtype=np.float32)[cand] + 1e-6
            p = tf / tf.sum()
            # numpy choice with replace=False works with probabilities
            idxs = self._rng.choice(cand, size=take, replace=False, p=p)
        else:
            # uniform without replacement
            idxs = self._rng.choice(cand, size=take, replace=False)

        return [int(i) for i in np.atleast_1d(idxs)]

    def get_instance(
            self,
            balanced: bool = False,
            patch_type_allow: Optional[List[str]] = None,
            prefer_tumorish: bool = False,
    ) -> dict:
        case_idx = self._sample_case(balanced)
        case = self.data[case_idx]

        valid_view_indices = [i for i, v in enumerate(case["views"]) if v["n_patches"] > 0]
        if not valid_view_indices:
            raise RuntimeError(f"Case {case['name']} has no patches in any view.")
        view_ind = int(self._rng.choice(valid_view_indices))
        view_name = VIEWS[view_ind]
        view = case["views"][view_ind]

        self._ensure_view_loaded(view)
        N = int(view["imgs"].shape[0])
        if N <= 0:
            raise RuntimeError(f"View {view_name} in case {case['name']} has 0 patches after loading.")

        # optionally restrict by patch_type
        if patch_type_allow is not None:
            allowed_ids = np.array([PATCH_TYPE_TO_ID[x] for x in patch_type_allow], dtype=np.uint8)
            ptypes = view["patch_type"]
            ok = np.isin(ptypes, allowed_ids)
            inds = np.where(ok)[0]
            if inds.size > 0:
                if prefer_tumorish:
                    tf = np.asarray(view["tumor_frac"], dtype=np.float32)[inds] + 1e-6
                    p = tf / tf.sum()
                    img_ind = int(self._rng.choice(inds, p=p))
                else:
                    img_ind = int(self._rng.choice(inds))
            else:
                img_ind = int(self._rng.integers(0, N))
        else:
            rnd = float(self._rng.random())
            img_ind = int(np.searchsorted(view["cum_wts"], rnd, side="right"))
            img_ind = min(max(img_ind, 0), N - 1)

        img_stack = view["imgs"][img_ind]  # (K,H,W) uint8
        mask2d = view["masks"][img_ind]  # (H,W) uint8
        mar_wt = float(view["mar_wts"][img_ind]) if view["mar_wts"].size > 0 else 1.0

        return {
            "img": img_stack,
            "mask": mask2d,
            "label": case["label"],
            "view": view_name,
            "case_name": case["name"],
            "mar_wt": mar_wt,
            "img_ind": img_ind,
            "lsf": case["lsf"],
            "patch_type": int(view["patch_type"][img_ind]),
            "center_zyx": tuple(int(x) for x in view["center_zyx"][img_ind]),
            "tumor_frac": float(view["tumor_frac"][img_ind]),
            "support_frac": float(view["support_frac"][img_ind]),
            "background_frac": float(view["background_frac"][img_ind]),
        }

    def get_instances(
        self,
        n: int,
        balanced: bool = False,
        patch_type_allow: Optional[List[str]] = None,
        prefer_tumorish: bool = False,
    ) -> List[dict]:
        return [
            self.get_instance(
                balanced=balanced,
                patch_type_allow=patch_type_allow,
                prefer_tumorish=prefer_tumorish,
            )
            for _ in range(n)
        ]

    # ---------- batch assembly ----------
    def assemble_batch_np(
        self,
        instances: List[dict],
        augment: bool = False,
        use_seg: bool = True,
        # how to map 2.5D stacks into channels:
        # - "center": use center slice only (keeps channel count small)
        # - "stack": average stack into one channel (cheap)
        # - "multi": use 3 channels from stack (K must be >=3): [center-1, center, center+1]
        stack_mode: str = "center",
    ) -> np.ndarray:
        """
        Returns (B, C, H, W) float32.
        Default is compatible with your V4 trainer expecting 3 channels:
          C=3 => [img, primary_mask, secondary_mask]
        """
        if stack_mode not in ("center", "stack", "multi"):
            raise ValueError("stack_mode must be 'center', 'stack', or 'multi'")

        B = len(instances)
        H = INNER_PATCH_SIZE_PX
        W = INNER_PATCH_SIZE_PX

        # image channel strategy
        if stack_mode == "multi":
            C_img = 3
        else:
            C_img = 1

        C = C_img + (2 if use_seg else 2)  # keep same signature; if not use_seg, duplicate image like your V4

        batch_arr = np.zeros((B, C, H, W), dtype=np.float32)

        for i, inst in enumerate(instances):
            img_stack_u8 = inst["img"]  # (K,H,W) uint8
            mask_u8 = inst["mask"]      # (H,W) uint8

            # Convert image stack -> float32 normalized like V3
            img_stack = img_stack_u8.astype(np.float32)
            img_stack = (img_stack - 128.0) / 128.0

            K = img_stack.shape[0]
            if stack_mode == "center":
                img = img_stack[K // 2]
                img_ch = img[None, :, :]
            elif stack_mode == "stack":
                img = img_stack.mean(axis=0)
                img_ch = img[None, :, :]
            else:  # "multi"
                if K < 3:
                    # degrade gracefully
                    img = img_stack[K // 2]
                    img_ch = np.stack([img, img, img], axis=0)
                else:
                    c = K // 2
                    img_ch = np.stack([img_stack[c - 1], img_stack[c], img_stack[c + 1]], axis=0)

            # Build seg channels (2,H,W) float32
            converted_mask = np.zeros((2,) + mask_u8.shape, np.float32)
            for cid in self.seg_class_definitions.get("primary", []) or []:
                converted_mask[0][mask_u8 == cid] = 1.0
            for cid in self.seg_class_definitions.get("secondary", []) or []:
                converted_mask[1][mask_u8 == cid] = 1.0


            if augment:
                img_ch, converted_mask = augment_fn_fast(img_ch, converted_mask)
            else:
                img_ch, converted_mask = no_augment_fn_v5(img_ch, converted_mask)

            # Fill output
            batch_arr[i, :C_img] = img_ch
            if use_seg:
                batch_arr[i, C_img:] = converted_mask
            else:
                # mimic your V4 "use_seg=False": duplicate image into seg channels
                batch_arr[i, C_img:] = img_ch[:2] if C_img >= 2 else np.repeat(img_ch, 2, axis=0)

        return batch_arr

    # ---------- batch + mixup (same feel as V4) ----------
    def get_batch_mixup_np(
        self,
        batch_size: int,
        balanced: bool = False,
        augment: bool = False,
        use_seg: bool = True,
        n_classes: int = 2,
        stack_mode: str = "center",
        # optional ablation knobs:
        patch_type_allow: Optional[List[str]] = None,
        prefer_tumorish: bool = False,
    ):
        instances = self.get_instances(
            batch_size,
            balanced=balanced,
            patch_type_allow=patch_type_allow,
            prefer_tumorish=prefer_tumorish,
        )
        labels = np.array([x["label"] for x in instances])
        batch_arr = self.assemble_batch_np(instances, augment=augment, use_seg=use_seg, stack_mode=stack_mode)

        # LSF matrix
        lsf_data = []
        for i in range(batch_size):
            row = []
            for k in sorted(instances[i]["lsf"]):
                row.append(instances[i]["lsf"][k])
            lsf_data.append(row)
        lsf_data = np.asarray(lsf_data)

        if not augment:
            if labels.ndim == 2:  # time-to-event
                return batch_arr, (labels[:, 1], labels[:, 0]), lsf_data
            else:
                if n_classes == 1:
                    return batch_arr, np.stack([labels], axis=-1), lsf_data
                elif n_classes == 2:
                    ret_labels = np.stack([1.0 - labels, labels], axis=-1)
                    return batch_arr, ret_labels, lsf_data
                else:
                    ret_labels = np.stack([np.where(labels == i, 1.0, 0.0) for i in range(n_classes)], axis=-1)
                    return batch_arr, ret_labels, lsf_data

        # mixup branch
        instances_mix = self.get_instances(
            batch_size,
            balanced=balanced,
            patch_type_allow=patch_type_allow,
            prefer_tumorish=prefer_tumorish,
        )
        labels_mix = np.array([x["label"] for x in instances_mix])
        batch_mix = self.assemble_batch_np(instances_mix, augment=augment, use_seg=use_seg, stack_mode=stack_mode)

        lsf_mix = []
        for i in range(batch_size):
            row = []
            for k in sorted(instances_mix[i]["lsf"]):
                row.append(instances_mix[i]["lsf"][k])
            lsf_mix.append(row)
        lsf_mix = np.asarray(lsf_mix)

        mix_w = np.random.beta(0.9, 0.9, batch_size).astype(np.float32)
        mw_arr = mix_w.reshape(batch_size, 1, 1, 1)

        ret_arr = batch_arr * mw_arr + batch_mix * (1.0 - mw_arr)
        ret_lsf = lsf_data * mix_w.reshape(-1, 1) + lsf_mix * (1.0 - mix_w.reshape(-1, 1))

        if labels.ndim == 2:
            dur = labels[:, 1]
            dur_m = labels_mix[:, 1]
            ret_dur = dur * mix_w + dur_m * (1.0 - mix_w)

            ev = labels[:, 0]
            ev_m = labels_mix[:, 0]
            ret_ev = np.round(ev * mix_w + ev_m * (1.0 - mix_w)).astype(np.int64)
            return ret_arr, (ret_dur, ret_ev), ret_lsf
        else:
            ret_lab = labels * mix_w + labels_mix * (1.0 - mix_w)
            if n_classes == 1:
                return ret_arr, np.stack([ret_lab], axis=-1), ret_lsf
            elif n_classes == 2:
                ret_lab = np.stack([1.0 - ret_lab, ret_lab], axis=-1)
                return ret_arr, ret_lab, ret_lsf
            else:
                ret_lab = np.stack([np.where(labels == i, 1.0, 0.0) for i in range(n_classes)], axis=-1)
                return ret_arr, ret_lab, ret_lsf

    def sample_case_index(self, balanced: bool = False) -> int:
        if not balanced:
            return int(np.random.randint(len(self.data)))

        cases_by_label = {}
        for idx, case in enumerate(self.data):
            lab = case["label"]
            lab = lab[0] if isinstance(lab, tuple) else lab
            cases_by_label.setdefault(lab, []).append(idx)

        labels = list(cases_by_label.keys())
        lab = labels[np.random.randint(len(labels))]
        return int(np.random.choice(cases_by_label[lab]))

    def get_instance_from_case(
            self,
            case_idx: int,
            patch_type_allow: Optional[List[str]] = None,
            prefer_tumorish: bool = False,
    ) -> dict:
        case = self.data[case_idx]

        valid_view_indices = [i for i, v in enumerate(case["views"]) if v["n_patches"] > 0]
        if not valid_view_indices:
            raise RuntimeError(f"Case {case['name']} has no patches in any view.")

        view_ind = int(self._rng.choice(valid_view_indices))
        view_name = VIEWS[view_ind]
        view = case["views"][view_ind]

        self._ensure_view_loaded(view)
        N = int(view["imgs"].shape[0])
        if N <= 0:
            raise RuntimeError(f"View {view_name} in case {case['name']} has 0 patches after loading.")

        if patch_type_allow is not None:
            allowed_ids = np.array([PATCH_TYPE_TO_ID[x] for x in patch_type_allow], dtype=np.uint8)
            ptypes = view["patch_type"]
            ok = np.isin(ptypes, allowed_ids)
            inds = np.where(ok)[0]
            if inds.size > 0:
                if prefer_tumorish:
                    tf = np.asarray(view["tumor_frac"], dtype=np.float32)[inds] + 1e-6
                    p = tf / tf.sum()
                    img_ind = int(self._rng.choice(inds, p=p))
                else:
                    img_ind = int(self._rng.choice(inds))
            else:
                img_ind = int(self._rng.integers(0, N))
        else:
            rnd = float(self._rng.random())
            img_ind = int(np.searchsorted(view["cum_wts"], rnd, side="right"))
            img_ind = min(max(img_ind, 0), N - 1)

        img_stack = view["imgs"][img_ind]
        mask2d = view["masks"][img_ind]
        mar_wt = float(view["mar_wts"][img_ind]) if view["mar_wts"].size > 0 else 1.0

        return {
            "img": img_stack,
            "mask": mask2d,
            "label": case["label"],
            "view": view_name,
            "case_name": case["name"],
            "mar_wt": mar_wt,
            "img_ind": img_ind,
            "lsf": case["lsf"],
            "patch_type": int(view["patch_type"][img_ind]),
            "center_zyx": tuple(int(x) for x in view["center_zyx"][img_ind]),
            "tumor_frac": float(view["tumor_frac"][img_ind]),
            "support_frac": float(view["support_frac"][img_ind]),
            "background_frac": float(view["background_frac"][img_ind]),
        }

    def get_case_bag(self, case_idx: int, augment: bool):
        """
        Clean V5 trainer API:
            bag_x, y = dataset.get_case_bag(case_idx=..., augment=...)

        Returns:
          bag_x: (K, C, H, W) float32
          y: case label (scalar or tuple)
        """
        k = int(self.bag_k)
        use_seg = bool(self.use_seg_default)
        stack_mode = str(self.stack_mode)

        case = self.data[case_idx]
        y = case["label"]

        # Normalize bag_mix to patch taxonomy. Expected keys:
        #  - primary, boundary, secondary, background
        bm = self.bag_mix or {}
        mix = {
            "core": float(bm.get("primary", bm.get("core", 0.0))),
            "boundary": float(bm.get("boundary", 0.0)),
            "support": float(bm.get("secondary", bm.get("support", 0.0))),
            "background": float(bm.get("background", 0.0)),
        }

        # leftover -> hard_neg (acts as "everything else")
        total = sum(max(0.0, v) for v in mix.values())
        if total > 1.0:
            # renormalize if someone set weird fractions
            mix = {t: max(0.0, v) / total for t, v in mix.items()}
            total = 1.0
        mix["hard_neg"] = max(0.0, 1.0 - total)

        # Convert fractions -> integer counts (guarantee sum == k)
        order = ["core", "boundary", "support", "hard_neg", "background"]
        counts = {t: int(math.floor(mix[t] * k)) for t in order}
        # distribute the remainder by descending fractional parts
        rem = k - sum(counts.values())
        if rem > 0:
            fracs = [(t, (mix[t] * k) - counts[t]) for t in order]
            fracs.sort(key=lambda x: x[1], reverse=True)
            for i in range(rem):
                counts[fracs[i % len(fracs)][0]] += 1
        elif rem < 0:
            # trim extras from the least important types
            for t in ["background", "hard_neg", "support", "boundary", "core"]:
                if rem == 0:
                    break
                take = min(counts[t], -rem)
                counts[t] -= take
                rem += take

        # Build instances (WITHOUT replacement inside bag)
        instances = []

        order = ["core", "boundary", "support", "hard_neg", "background"]

        # Track used indices per (view_ind) so we don't repeat within a bag
        used_by_view = {}  # view_ind -> set(img_ind)

        case = self.data[case_idx]
        valid_view_indices = [i for i, v in enumerate(case["views"]) if v["n_patches"] > 0]
        if not valid_view_indices:
            raise RuntimeError(f"Case {case['name']} has no patches in any view.")

        for t in order:
            n = counts.get(t, 0)
            if n <= 0:
                continue

            # choose a view for this type (you can also loop views; this is the minimal change)
            view_ind = int(self._rng.choice(valid_view_indices))
            view_name = VIEWS[view_ind]
            view = case["views"][view_ind]
            self._ensure_view_loaded(view)

            if view_ind not in used_by_view:
                used_by_view[view_ind] = set()

            idxs = self._sample_patch_indices(
                view=view,
                n=n,
                patch_type_allow=[t],
                prefer_tumorish=(t in ["core", "boundary"]),
                used=used_by_view[view_ind],
            )

            # if we couldn't get enough from that view/type, fall back to unrestricted (still avoid repeats)
            if len(idxs) < n:
                more = self._sample_patch_indices(
                    view=view,
                    n=(n - len(idxs)),
                    patch_type_allow=None,
                    prefer_tumorish=False,
                    used=used_by_view[view_ind],
                )
                idxs.extend(more)

            for img_ind in idxs:
                used_by_view[view_ind].add(img_ind)

                img_stack = view["imgs"][img_ind]
                mask2d = view["masks"][img_ind]
                mar_wt = float(view["mar_wts"][img_ind]) if view["mar_wts"].size > 0 else 1.0

                instances.append({
                    "img": img_stack,
                    "mask": mask2d,
                    "label": case["label"],
                    "view": view_name,
                    "case_name": case["name"],
                    "mar_wt": mar_wt,
                    "img_ind": img_ind,
                    "lsf": case["lsf"],
                    "patch_type": int(view["patch_type"][img_ind]),
                    "center_zyx": tuple(int(x) for x in view["center_zyx"][img_ind]),
                    "tumor_frac": float(view["tumor_frac"][img_ind]),
                    "support_frac": float(view["support_frac"][img_ind]),
                    "background_frac": float(view["background_frac"][img_ind]),
                })

        # Safety: if still short, do unrestricted sampling but avoid repeats if possible
        while len(instances) < k:
            # pick a random view, avoid repeating within that view if possible
            view_ind = int(self._rng.choice(valid_view_indices))
            view_name = VIEWS[view_ind]
            view = case["views"][view_ind]
            self._ensure_view_loaded(view)
            if view_ind not in used_by_view:
                used_by_view[view_ind] = set()

            idxs = self._sample_patch_indices(view=view, n=1, used=used_by_view[view_ind])
            if len(idxs) == 0:
                # absolute fallback: allow repeats
                img_ind = int(self._rng.integers(0, int(view["imgs"].shape[0])))
            else:
                img_ind = idxs[0]
                used_by_view[view_ind].add(img_ind)

            img_stack = view["imgs"][img_ind]
            mask2d = view["masks"][img_ind]
            mar_wt = float(view["mar_wts"][img_ind]) if view["mar_wts"].size > 0 else 1.0

            instances.append({
                "img": img_stack,
                "mask": mask2d,
                "label": case["label"],
                "view": view_name,
                "case_name": case["name"],
                "mar_wt": mar_wt,
                "img_ind": img_ind,
                "lsf": case["lsf"],
                "patch_type": int(view["patch_type"][img_ind]),
                "center_zyx": tuple(int(x) for x in view["center_zyx"][img_ind]),
                "tumor_frac": float(view["tumor_frac"][img_ind]),
                "support_frac": float(view["support_frac"][img_ind]),
                "background_frac": float(view["background_frac"][img_ind]),
            })

        # Trim if over
        if len(instances) > k:
            instances = instances[:k]

        bag_x = self.assemble_batch_np(
            instances,
            augment=augment,
            use_seg=use_seg,
            stack_mode=stack_mode,
        )  # (k,C,H,W)

        return bag_x, y