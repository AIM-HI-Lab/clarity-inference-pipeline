"""
Minimal subset of data_loader_v3 used by data_loader_v5 (inference-only vendor build).
"""

from typing import Union

import numpy as np
import nibabel as nib
from nibabel.funcs import as_closest_canonical
from scipy.ndimage import affine_transform

Numeric = Union[int, float]

INNER_PATCH_SIZE_PX = 267


def hash_str(s: str) -> str:
    s = str(s)
    hash_val = 0
    for c in s:
        hash_val = (hash_val * 31 + ord(c)) % 2**32
    return hex(hash_val)[2:].zfill(8)


def reorient_to_identity(img):
    reoriented = as_closest_canonical(img)

    new_aff = reoriented.affine
    new_aff[0:3, 3] = 0
    shifted = nib.Nifti1Image(reoriented.get_fdata(), new_aff)

    new_size_1 = int(np.round(reoriented.shape[0] * new_aff[0, 0]))
    new_size_2 = int(np.round(reoriented.shape[1] * new_aff[1, 1]))
    new_size_3 = int(np.round(reoriented.shape[2] * new_aff[2, 2]))

    resampled = nib.Nifti1Image(
        affine_transform(
            shifted.get_fdata(),
            np.linalg.inv(shifted.affine),
            output_shape=(new_size_1, new_size_2, new_size_3),
            order=1,
        ),
        np.eye(4),
    )

    return resampled


def create_cum_wts(rel_arb_lst):
    ret = []
    arb_sum = sum(rel_arb_lst)
    running_tot = rel_arb_lst[0]
    ret.append(running_tot / arb_sum)
    for i in range(1, len(rel_arb_lst)):
        running_tot += rel_arb_lst[i]
        ret.append(running_tot / arb_sum)

    return ret


def pad_first_two_dims(img, inner_patch_size_px):
    img_shape = np.array(img.shape[:2])
    target_shape = np.maximum(img_shape, inner_patch_size_px)
    padding = target_shape - img_shape
    padding_before = padding // 2
    padding_after = padding - padding_before

    pad_width = tuple((padding_before[i], padding_after[i]) for i in range(2))
    remaining_dims_pad = tuple((0, 0) for _ in range(img.ndim - 2))
    pad_width = pad_width + remaining_dims_pad

    return np.pad(img, pad_width, mode="constant", constant_values=0)


def pad_last_two_dims(mask, inner_patch_size_px):
    img_shape = np.array(mask.shape[-2:])
    target_shape = np.maximum(img_shape, inner_patch_size_px)
    padding = target_shape - img_shape
    padding_before = padding // 2
    padding_after = padding - padding_before

    pad_width_last_two = tuple((padding_before[i], padding_after[i]) for i in range(2))
    preceding_dims_pad = tuple((0, 0) for _ in range(mask.ndim - 2))
    pad_width = preceding_dims_pad + pad_width_last_two

    return np.pad(mask, pad_width, mode="constant", constant_values=0)


def no_augment_fn(img, mask):
    padded_img = pad_first_two_dims(img, INNER_PATCH_SIZE_PX)
    padded_mask = pad_last_two_dims(mask, INNER_PATCH_SIZE_PX)

    h, w = padded_img.shape
    crop_h = (h - INNER_PATCH_SIZE_PX) // 2
    crop_w = (w - INNER_PATCH_SIZE_PX) // 2
    img = padded_img[
        crop_h : crop_h + INNER_PATCH_SIZE_PX, crop_w : crop_w + INNER_PATCH_SIZE_PX
    ]
    mask = padded_mask[
        :, crop_h : crop_h + INNER_PATCH_SIZE_PX, crop_w : crop_w + INNER_PATCH_SIZE_PX
    ]
    return img, mask
