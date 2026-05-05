import json
from time import sleep
from pathlib import Path
from typing import List, Union

import cv2
import scipy.ndimage
import numpy as np
import nibabel as nib
import imageio as iio
from tqdm import tqdm
from scipy.ndimage import affine_transform, center_of_mass
from nibabel.funcs import as_closest_canonical
import albumentations as A


RSMP_PIXEL_SIZE = 1
INNER_PATCH_SIZE_PX = 267
PATCH_SIZE_PX = int(np.sqrt((INNER_PATCH_SIZE_PX**2)*2)) + 1
VIEWS = ["axial", "coronal", "sagittal"]


Numeric = Union[int, float]


# Reproducible hash function to avoid name collisions
# Must return same value for same input always, even across invocations
def hash_str(s: str) -> str:
    # Ensure input is string
    s = str(s)
    # Hash function
    hash_val = 0
    for c in s:
        hash_val = (hash_val * 31 + ord(c)) % 2**32
    # Return as hex string
    return hex(hash_val)[2:].zfill(8)


def variable_axis_slice(block, axis_ind, slice_i):
    if axis_ind == 0:
        return block[slice_i]
    if axis_ind == 1:
        return block[:, slice_i]
    if axis_ind == 2:
        return block[:, :, slice_i]


def get_centroid(bin_seg):
    if np.sum(bin_seg) == 0:
        return 0, 0
    
    center_y_float, center_x_float = center_of_mass(bin_seg)

    cnt_y = int(np.round(center_y_float))
    cnt_x = int(np.round(center_x_float))

    return cnt_y, cnt_x

    # y_arange = np.arange(bin_seg.shape[0]).reshape((-1,1))
    # x_arange = np.arange(bin_seg.shape[1]).reshape((-1,1))

    # y_sums = np.sum(bin_seg, axis=1)
    # x_sums = np.sum(bin_seg, axis=0)

    # cnt_y = int(np.round(np.sum(np.multiply(y_arange, y_sums))/np.sum(y_sums)))
    # cnt_x = int(np.round(np.sum(np.multiply(x_arange, x_sums))/np.sum(x_sums)))

    # return cnt_y, cnt_x


def get_centroid_3d(bin_seg):
    x_arange = np.arange(bin_seg.shape[0]).reshape((-1,1))
    y_arange = np.arange(bin_seg.shape[1]).reshape((-1,1))
    z_arange = np.arange(bin_seg.shape[2]).reshape((-1,1))

    x_sums = np.sum(bin_seg, axis=(1,2)).reshape((-1,1))
    y_sums = np.sum(bin_seg, axis=(0,2)).reshape((-1,1))
    z_sums = np.sum(bin_seg, axis=(0,1)).reshape((-1,1))

    cnt_x = int(np.round(np.sum(np.multiply(x_arange, x_sums))/np.sum(x_sums)))
    cnt_y = int(np.round(np.sum(np.multiply(y_arange, y_sums))/np.sum(y_sums)))
    cnt_z = int(np.round(np.sum(np.multiply(z_arange, z_sums))/np.sum(z_sums)))

    return cnt_x, cnt_y, cnt_z


def crop_rois(
    rsmp_img, rsmp_seg, dst_prnt, slice_i, class_defs, sampling_mode,
    centroid_info, fov_slices, cache
):
    # print(f"DEBUG(crop_rois): rsmp_img.shape upon entry: {rsmp_img.shape}")
    # print(f"DEBUG(crop_rois): PATCH_SIZE_PX: {PATCH_SIZE_PX}")

    rois = []

    # Get binary image of tumor only
    tum_bin = np.zeros(rsmp_img.shape, np.uint8)
    for class_id in class_defs["primary"]:
        tum_bin[np.equal(rsmp_seg, class_id)] = 1

    # Find connected components
    label_arr, num_feats = scipy.ndimage.label(tum_bin)

    # TEMP
    # Scale seg to 0-255
    rsmp_seg = (rsmp_seg*200).astype(np.uint8)

    # If sampling mode is weighted...
    if sampling_mode == "weighted":
        patch_size_px = PATCH_SIZE_PX

        padded_img = np.pad(rsmp_img, patch_size_px//2, constant_values=0)
        padded_seg = np.pad(rsmp_seg, patch_size_px//2, constant_values=0)

        # print(f"DEBUG(crop_rois): padded_img.shape: {padded_img.shape}")

        # Iterate over features
        for feat_i in range(1, num_feats + 1):
            # Get that feature's centroid
            feat_bin = np.equal(label_arr, feat_i).astype(np.uint8)
            cnt_y, cnt_x = get_centroid(feat_bin)

            # print(f"DEBUG(crop_rois): Centroid (y, x): ({cnt_y}, {cnt_x})")

            # Get that feature's size
            feat_size = int(np.sum(feat_bin))

            # Crop out region around that feature's centroid
            y_start = patch_size_px//2 + cnt_y - patch_size_px//2
            y_end = patch_size_px//2 + cnt_y + patch_size_px//2
            x_start = patch_size_px//2 + cnt_x - patch_size_px//2
            x_end = patch_size_px//2 + cnt_x + patch_size_px//2

            # print(f"DEBUG(crop_rois): Crop indices (y_start, y_end, x_start, x_end): ({y_start}, {y_end}, {x_start}, {x_end})")
            # print(f"DEBUG(crop_rois): Desired crop shape: ({PATCH_SIZE_PX}, {PATCH_SIZE_PX})")

            crp_img = padded_img[
                y_start: y_end,
                x_start: x_end
            ]
            crp_seg = padded_seg[
                y_start: y_end,
                x_start: x_end
            ]


            # print(f"DEBUG(crop_rois): Actual crp_img.shape: {crp_img.shape}")
            
            assert crp_img.shape == (patch_size_px, patch_size_px)
            assert crp_seg.shape == (patch_size_px, patch_size_px)

            # Save cropped images
            if cache: 
                dst_stem = f"{slice_i:04d}_{feat_i:02d}_{feat_size:06d}"
                iio.imsave(dst_prnt / f"{dst_stem}_img.png", crp_img)
                iio.imsave(dst_prnt / f"{dst_stem}_seg.png", crp_seg)
            else:
                rois.append({"img": crp_img, "mask": crp_seg, "mar_wt": feat_size})

    # If sampling mode is fixed...
    elif sampling_mode == "fixed":
        patch_size_px = (fov_slices//2) * 2

        # Pad image (with checks)
        padding_size = patch_size_px // 2
        padded_img = np.zeros((
            rsmp_img.shape[0] + padding_size*2,
            rsmp_img.shape[1] + padding_size*2
        ), np.uint8)
        padded_seg = np.zeros((
            rsmp_img.shape[0] + padding_size*2,
            rsmp_img.shape[1] + padding_size*2
        ), np.uint8)
        padded_img[
            padding_size: padding_size + rsmp_img.shape[0],
            padding_size: padding_size + rsmp_img.shape[1]
        ] = rsmp_img
        padded_seg[
            padding_size: padding_size + rsmp_img.shape[0],
            padding_size: padding_size + rsmp_img.shape[1]
        ] = rsmp_seg

        # There is only one feature and it is centered around the centroid
        cnt_y = centroid_info["ax1"]
        cnt_x = centroid_info["ax0"]

        # The feature size is always 1
        feat_size = 1

        # Crop out region around that feature's centroid
        y_start = padding_size + cnt_y - patch_size_px//2
        y_end = y_start + patch_size_px
        x_start = padding_size + cnt_x - patch_size_px//2
        x_end = x_start + patch_size_px
        crp_img = padded_img[
            y_start: y_end,
            x_start: x_end
        ]
        crp_seg = padded_seg[
            y_start: y_end,
            x_start: x_end
        ]

        img_err_msg = f"Image shape: {crp_img.shape}, Expected: {(patch_size_px, patch_size_px)}"
        seg_err_msg = f"Segmentation shape: {crp_seg.shape}, Expected: {(patch_size_px, patch_size_px)}"
        assert crp_img.shape == (patch_size_px, patch_size_px), img_err_msg
        assert crp_seg.shape == (patch_size_px, patch_size_px), seg_err_msg

        # Transpose the result
        crp_img = np.transpose(crp_img)
        crp_seg = np.transpose(crp_seg)

        # Flip over the Y axis to match the orientation of the image
        crp_img = np.flip(crp_img, axis=0)
        crp_seg = np.flip(crp_seg, axis=0)

        # Flip over the X axis to match the orientation of the image
        crp_img = np.flip(crp_img, axis=1)
        crp_seg = np.flip(crp_seg, axis=1)

        # Save cropped images
        if cache:
            dst_stem = f"{slice_i:04d}_00_{feat_size:06d}"
            iio.imsave(dst_prnt / f"{dst_stem}_img.png", crp_img)
            iio.imsave(dst_prnt / f"{dst_stem}_seg.png", crp_seg)
        else:
            rois.append({"img": crp_img, "mask": crp_seg, "arb_wt": feat_size})

    return rois


def extract_slice(
    img_np, seg_np, axis_ind, slice_i, affine, dst_prnt, class_defs,
    sampling_mode, centroid_info, fov_slices, cache
):
    # Slice image and segmentation
    img_slice = variable_axis_slice(img_np, axis_ind, slice_i)
    seg_slice = variable_axis_slice(seg_np, axis_ind, slice_i)

    # Save slice images as is
    if cache:
        dst_tmp = dst_prnt / "tmp"
        dst_tmp.mkdir(exist_ok=True)
        iio.imsave(dst_tmp / f"{slice_i:04d}_{axis_ind}_img.png", img_slice)
        iio.imsave(dst_tmp / f"{slice_i:04d}_{axis_ind}_seg.png", seg_slice*200)

    # Resample slice to isotropic
    axes_not_in_use = [x for x in range(3) if x != axis_ind]
    yx_spacing = list(np.abs(np.sum(affine[:, axes_not_in_use], axis=0)))
    dim1_rsmp_factor = yx_spacing[0]/RSMP_PIXEL_SIZE
    dim2_rsmp_factor = yx_spacing[1]/RSMP_PIXEL_SIZE
    new_dimensions = [
        int(np.round(seg_slice.shape[1]*dim1_rsmp_factor)),
        int(np.round(seg_slice.shape[0]*dim2_rsmp_factor))
    ]
    # print("YX spacing found to be", yx_spacing)
    # print("New dimensions found to be", new_dimensions)
    # print("Old dimensions were", seg_slice.shape)
    rsmp_img = cv2.resize(img_slice, new_dimensions)

    # Update centroid info
    if sampling_mode == "fixed":
        rsmp_cent_info = {"slice_i": centroid_info["slice_i"]}
        if centroid_info is not None:
            rsmp_cent_info["ax0"] = int(np.round(centroid_info["ax0"]*dim1_rsmp_factor))
            rsmp_cent_info["ax1"] = int(np.round(centroid_info["ax1"]*dim2_rsmp_factor))

        print("New centroid info is")
        print(json.dumps(rsmp_cent_info, indent=2))
    else:
        rsmp_cent_info = None

    # Resample corresponding segmentation to same
    rsmp_seg = np.zeros(rsmp_img.shape, np.uint8)
    for class_id in range(1, 4):
        bin_seg = 2*(np.equal(seg_slice, class_id).astype(np.float32))
        rsmp_bin_seg = np.greater(
            cv2.resize(bin_seg, new_dimensions), 0.5
        ).astype(np.uint8)
        rsmp_seg[np.equal(rsmp_bin_seg, 1)] = class_id*(
            rsmp_bin_seg[
                np.equal(rsmp_bin_seg, 1)
            ].astype(np.uint8)
        )
    
    # print(f"DEBUG(extract_slice): rsmp_img.shape AFTER cv2.resize: {rsmp_img.shape}")
    # print(f"DEBUG(extract_slice): rsmp_seg.shape AFTER cv2.resize: {rsmp_seg.shape}")

    # Ensure rsmp_img and rsmp_seg are large enough to contain the patch
    current_height, current_width = rsmp_img.shape
    target_height = max(current_height, PATCH_SIZE_PX + 2 * (PATCH_SIZE_PX // 2)) # Ensure space for patch + padding
    target_width = max(current_width, PATCH_SIZE_PX + 2 * (PATCH_SIZE_PX // 2))

    if current_height < target_height or current_width < target_width:
        # Create new larger arrays and place the current rsmp_img/seg in the center
        padded_rsmp_img = np.zeros((target_height, target_width), dtype=rsmp_img.dtype)
        padded_rsmp_seg = np.zeros((target_height, target_width), dtype=rsmp_seg.dtype)

        # Calculate start positions to center the existing image
        start_y = (target_height - current_height) // 2
        start_x = (target_width - current_width) // 2

        padded_rsmp_img[start_y:start_y + current_height, start_x:start_x + current_width] = rsmp_img
        padded_rsmp_seg[start_y:start_y + current_height, start_x:start_x + current_width] = rsmp_seg

        rsmp_img = padded_rsmp_img
        rsmp_seg = padded_rsmp_seg

    # print(f"DEBUG(extract_slice): rsmp_img.shape AFTER custom padding: {rsmp_img.shape}")
    # print(f"DEBUG(extract_slice): rsmp_img.shape AFTER custom padding: {rsmp_img.shape}")


    # Save slice images as is
    # Still correct here...
    if cache:
        dst_tmp = dst_prnt / "tmp_rsmp"
        dst_tmp.mkdir(exist_ok=True)
        iio.imsave(dst_tmp / f"{slice_i:04d}_{axis_ind}_img.png", rsmp_img)
        iio.imsave(dst_tmp / f"{slice_i:04d}_{axis_ind}_seg.png", rsmp_seg*200)

    # Crop out and save areas around region(s) of interest
    return crop_rois(
        rsmp_img, rsmp_seg, dst_prnt, slice_i, class_defs,
        sampling_mode, rsmp_cent_info, fov_slices, cache=cache
    )


# TODO Jay
# Find the centroid of the segmentation of the primary object
# 1. Get a binary mask of the primary object
# 2. Find its centroid
# 3. Account for the offset_cm_ap and offset_cm_cc values to adjust
#    the focal point
# 4. Modify the loop below to start at the focal point and proceed outwards
#    according to the fov_cm 

#
# Created binary mask for primary object, calculated its centroid,
# adjusted based on offset, and limited slice processing to field of view.


def reorient_to_identity(img):
    reoriented = as_closest_canonical(img)

    new_aff = reoriented.affine
    new_aff[0:3,3] = 0
    shifted = nib.Nifti1Image(
        reoriented.get_fdata(), new_aff
    )

    new_size_1 = int(np.round(reoriented.shape[0]*new_aff[0,0]))
    new_size_2 = int(np.round(reoriented.shape[1]*new_aff[1,1]))
    new_size_3 = int(np.round(reoriented.shape[2]*new_aff[2,2]))

    resampled = nib.Nifti1Image(
        affine_transform(
            shifted.get_fdata(),
            np.linalg.inv(shifted.affine),
            output_shape=(new_size_1, new_size_2, new_size_3),
            order=1  # Use order=1 for linear interpolation
        ),
        np.eye(4)
    )

    return resampled


def reorient_to_identity_deprecated(img):
    """
    Reorients a NIfTI image's data array so that the affine becomes identity
    while preserving the underlying representation.
    
    Parameters:
    -----------
    img : nibabel.Nifti1Image
        Input NIfTI image
        
    Returns:
    --------
    nibabel.Nifti1Image
        New image with identity affine and reoriented data
    """
    # Get the current affine and shape
    old_affine = img.affine
    old_shape = img.shape
    
    # Create a grid of indices for the output space (identity affine)
    x, y, z = np.meshgrid(
        np.arange(old_shape[0]),
        np.arange(old_shape[1]),
        np.arange(old_shape[2]),
        indexing='ij'
    )

    # Stack coordinates and reshape
    coords = np.stack([x, y, z, np.ones_like(x)]).reshape(4, -1)
    
    # Transform coordinates using inverse of old affine
    transformed_coords = np.linalg.inv(old_affine) @ coords
    
    # Reshape coordinates back to 3D grid
    transformed_x = transformed_coords[0].reshape(old_shape)
    transformed_y = transformed_coords[1].reshape(old_shape)
    transformed_z = transformed_coords[2].reshape(old_shape)
    
    # Interpolate the data at the new coordinates
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolator
    x_grid = np.arange(old_shape[0])
    y_grid = np.arange(old_shape[1])
    z_grid = np.arange(old_shape[2])
    interpolator = RegularGridInterpolator(
        (x_grid, y_grid, z_grid),
        img.get_fdata(),
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    
    # Prepare points for interpolation
    points = np.stack([transformed_x, transformed_y, transformed_z], axis=-1)
    
    # Interpolate
    new_data = interpolator(points)
    
    # Create identity affine of appropriate size
    identity = np.eye(4)
    
    # Create new image with identity affine
    new_img = nib.Nifti1Image(new_data, identity)
    
    return new_img


def cache_case_v1(
    img_pth, mask_pth, case_cache_pth, class_defs,
    sampling_mode, offset_cm_ap, offset_cm_cc, offset_cm_lr,
    fov_cm, 
    cache: bool = True
):
    
    # list 
    case_rois = [] if not cache else None

    # Make directory
    if cache:
        case_cache_pth.mkdir(parents=True, exist_ok=True)
        # Set base destination path
        dst_pth = case_cache_pth / "images"

    # Load image and mask
    img_nib = nib.load(str(img_pth))
    mask_nib = nib.load(str(mask_pth))

    # BEFORE RESAMPLING
    print("BEFORE RESAMPLING")
    print(img_nib.affine)
    print(np.sum(np.equal(
        np.asanyarray(mask_nib.dataobj).astype(np.uint8),
        2
    )))

    # Transform the image and segmentation to have a standard affine matrix
    img_nib = reorient_to_identity(img_nib)
    mask_nib = reorient_to_identity(mask_nib)

    # Save mask and image for inspection
    if cache:
        tmp_img_pth = Path(__file__).parent / ".tmp.img.nii.gz"
        tmp_seg_pth = Path(__file__).parent / ".tmp.seg.nii.gz"
        nib.save(img_nib, str(tmp_img_pth))
        nib.save(mask_nib, str(tmp_seg_pth))

    print(img_nib.affine)
    print(img_nib.shape)

    print("AFTER RESAMPLING")
    print(np.sum(np.equal(
        np.asanyarray(mask_nib.dataobj).astype(np.uint8),
        2
    )))

    # Load pixel data
    img_np = np.asanyarray(img_nib.dataobj).astype(np.float32)
    mask_np = np.round(
        np.asanyarray(mask_nib.dataobj)
    ).astype(np.uint8)

    # Normalize image to uint8 range for PNG compression
    img_np = np.clip(
        255*(img_np + 128)/(128 + 256), 0, 255
    ).astype(np.uint8)

    # Create a binary mask for the primary object (likely the tumor)
    primary_mask = np.zeros(mask_np.shape, np.uint8)
    # This assumes class_defs might not have a "primary" key
    if "primary" in class_defs and class_defs["primary"]:
        for class_id in class_defs["primary"]:
            primary_mask[np.equal(mask_np, class_id)] = 1

    # Ensure the primary object is present, with fallback for 'fixed' mode
    if np.sum(primary_mask) == 0:
        if sampling_mode == 'fixed':
            # In 'fixed' mode, if no primary class is found, treat any segmentation
            # as the primary object.
            primary_mask = (mask_np > 0).astype(np.uint8)
        else:
            # If not in 'fixed' mode and no primary object is found, raise an error.
            raise ValueError(f"Primary object not found in mask for case: {str(mask_pth)}")

    # Final check: if the mask is still empty even after the fallback, it's an error.
    if np.sum(primary_mask) == 0:
        raise ValueError(f"Mask is empty, no object found for case: {str(mask_pth)}")

    print(f"Total primary object voxels after all 3D processing: {np.sum(primary_mask)}")

    # If sampling mode is weighted...
    if sampling_mode == "weighted":
        # For each image orientation
        for axis_ind, orientation in zip(
            [0, 1, 2],
            VIEWS
        ):
            
            if cache:
                plane_pth = dst_pth / orientation
                plane_pth.mkdir(exist_ok=True, parents=True)
            else:
                plane_pth = None

            # Get array of booleans indicating whether a given slice contains primary object
            prim_bool = np.zeros(mask_np.shape, np.uint8)
            for class_id in class_defs["primary"]:
                prim_bool[np.equal(mask_np, class_id)] = 1
            contains_prim_array = list(map(lambda x: x == 1, list(np.apply_over_axes(
                np.max, prim_bool, [x for x in range(3) if x != axis_ind]
            ).flatten())))

            # For each slice within the field of view
            for slice_i in range(len(contains_prim_array)):
                if not contains_prim_array[slice_i]:
                    continue
                rois = extract_slice(
                    img_np, mask_np, axis_ind, slice_i,
                    img_nib.affine, plane_pth, class_defs,
                    sampling_mode, None, None, cache=cache
                )
                if not cache:
                    case_rois.extend(rois)

    # If sampling mode is fixed...
    elif sampling_mode == "fixed":
        # Get the centroid of the primary object
        centroid_x, centroid_y, centroid_z = get_centroid_3d(primary_mask)

        # Adjust the centroid based on the offset values
        adjusted_centroid_y = centroid_y + int(offset_cm_ap * 10 / RSMP_PIXEL_SIZE)
        adjusted_centroid_x = centroid_x + int(offset_cm_lr * 10 / RSMP_PIXEL_SIZE)
        adjusted_centroid_z = centroid_z + int(offset_cm_cc * 10 / RSMP_PIXEL_SIZE)

        # Calculate the range of slices to include in the field of view
        n_fov_slices = int(fov_cm * 10 / RSMP_PIXEL_SIZE)

        # Ensure fov_slices is more than zero
        if n_fov_slices <= 0:
            raise ValueError(f"Field of view must be greater than zero for case: {str(mask_pth)}")

        # Centroid components by axis
        adj_cent_components = {
            "axial": {
                "slice_i": adjusted_centroid_z,
                "ax0": adjusted_centroid_x,
                "ax1": adjusted_centroid_y
            },
            "coronal": {
                "slice_i": adjusted_centroid_y,
                "ax0": adjusted_centroid_x,
                "ax1": adjusted_centroid_z
            },
            "sagittal": {
                "slice_i": adjusted_centroid_x,
                "ax0": adjusted_centroid_y,
                "ax1": adjusted_centroid_z
            }
        }

        # Print adjusted centroid information
        print(f"Adjusted centroid for case {str(mask_pth)}:")
        print(json.dumps(adj_cent_components, indent=2))

        # For each image orientation
        for axis_ind, orientation in zip(
            [2, 1, 0],
            VIEWS
        ):
            
            if cache:
                plane_pth = dst_pth / orientation
                plane_pth.mkdir(exist_ok=True, parents=True)
            else:
                plane_pth = None

            # Get centroid components for this orientation
            adj_cent = adj_cent_components[orientation]

            # Get beginning and ending slice indices
            start_slice = max(0, adj_cent["slice_i"] - n_fov_slices)
            end_slice = min(
                primary_mask.shape[axis_ind], adj_cent["slice_i"] + n_fov_slices
            )

            # For each slice within the field of view
            for slice_i in range(start_slice, end_slice):
                rois = extract_slice(
                    img_np, mask_np, axis_ind, slice_i,
                    img_nib.affine, plane_pth, class_defs,
                    sampling_mode, adj_cent, n_fov_slices, cache=cache
                )
                if not cache:
                    case_rois.extend(rois)

    # Save metadata
    if cache:
        meta = {
            "img_pth": str(img_pth.resolve()),
            "mask_pth": str(mask_pth.resolve())
        }
        json.dump(meta, open(case_cache_pth / "meta.json", "w"), indent=2)
    else:
        return case_rois


# TODO Jay
# Pass the offsets and sampling_fov_cm to this function
# and modify the function to extract slices that are within
# the FOV after taking the offset into account.
#
# Addition: Passed offset_cm_ap, offset_cm_cc, and sampling_fov_cm 
# to cache_case_v1. IN LINE 247, I adjusted the range of slices based on the focal point and field of view.

def prep_case_v1(
    img_pth: Path, mask_pth: Path, label: Numeric, case_id: str, item_lsf: dict,
    case_cache_pth: Path, class_defs: dict, sampling_mode: str,
    offset_cm_ap: float, offset_cm_cc: float, offset_cm_lr: float,
    sampling_fov_cm: float
):
    # Determine whether case needs to be re-cached
    recache = False
    if not case_cache_pth.exists():
        recache = True
    elif not (case_cache_pth / "meta.json").exists():
        recache = True
    else:
        meta = json.load(open(case_cache_pth / "meta.json", "r"))
        if meta["img_pth"] != str(img_pth.resolve()):
            recache = True
        elif meta["mask_pth"] != str(mask_pth.resolve()):
            recache = True

    # Cache case if necessary
    if recache:
        cache_case_v1(
            img_pth, mask_pth, case_cache_pth, class_defs,
            sampling_mode,
            offset_cm_ap, offset_cm_cc, offset_cm_lr,
            sampling_fov_cm
        )

    # Get case data
    case_data = get_case_data_v1(case_cache_pth, label, item_lsf)

    return case_data


def create_cum_wts(rel_arb_lst):
    ret = []
    arb_sum = sum(rel_arb_lst)
    running_tot = rel_arb_lst[0]
    ret.append(running_tot/arb_sum)
    for i in range(1, len(rel_arb_lst)):
        running_tot += rel_arb_lst[i]
        ret.append(running_tot/arb_sum)

    return ret


def load_view(view_pth):
    images = []
    arb_wts = []
    for img_pth in view_pth.glob("*_img.png"):
        seg_pth = img_pth.parent / (
            "_".join(img_pth.stem.split("_")[:-1]) + "_seg.png"
        )
        arb_wts.append(int(img_pth.stem.split("_")[2]))
        images.append({
            "img": img_pth,
            "seg": seg_pth
        })

    arb_sum = sum(arb_wts)
    mar_wts = [x/arb_sum for x in arb_wts]

    return {
        "img_pths": images,
        "cum_wts": create_cum_wts(arb_wts),
        "mar_wts": mar_wts,
        "arb_wts": arb_wts
    }


def get_case_data_v1(case_pth: Path, label: Numeric, item_lsf: dict):
    ret = {
        "name": case_pth.name,
        "views": [],
        "label": label,
        "lsf": item_lsf
    }
    for view_pth in (case_pth / "images").glob("*"):
        ret["views"].append(load_view(view_pth))

    return ret


def weighted_sample(cum_wts):
    rnd = np.random.rand()
    for i in range(len(cum_wts)):
        if rnd < cum_wts[i]:
            return i
    return len(cum_wts) - 1

def augment_fn_v2(img, mask):
    """
    Applies a robust set of augmentations using the Albumentations library.
    It correctly handles both the image and the segmentation mask.
    """
    # Define the augmentation pipeline. This is created on each call
    # to ensure randomness.
    transform = A.Compose([
        # --- Geometric Augmentations ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A single, efficient transform for shifting, scaling, and rotating
        A.ShiftScaleRotate(
            shift_limit=0.06,    # Max 6% shift
            scale_limit=0.1,     # Max 10% zoom
            rotate_limit=30,     # Max 30-degree rotation
            p=0.75               # Apply this transform 75% of the time
        ),

        # --- Photometric (Pixel-level) Augmentations ---
        A.RandomBrightnessContrast(p=0.75),
        A.GaussNoise(p=0.5),

        # --- Regularization ---
        A.CoarseDropout(
            max_holes=16,          # Increase the number of holes
            max_height=16,         # Make the max size of each hole small
            max_width=16,
            min_holes=8,           # Ensure at least a few holes are always dropped
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.5
        ),
    ])

    # Apply the transforms to the image and mask
    augmented = transform(image=img, mask=mask)

    img, mask = no_augment_fn(augmented['image'], augmented['mask'])

    return img, mask


def augment_fn(img, mask):
    if len(mask.shape) == 2:
        masks = [mask]
    else:
        masks = [mask[i, :, :] for i in range(mask.shape[0])]

    if np.random.rand() < 0.5:
        img = np.flip(img, axis=0)
        for i in range(len(masks)):
            masks[i] = np.flip(masks[i], axis=0)
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=1)
        for i in range(len(masks)):
            masks[i] = np.flip(masks[i], axis=1)

    rot = np.random.randint(0, 360)
    img = scipy.ndimage.rotate(img, rot, reshape=False)
    for i in range(len(masks)):
        masks[i] = scipy.ndimage.rotate(masks[i], rot, reshape=False)

    h, w = img.shape
    resize_factor = np.random.uniform(0.9, 1.1)
    img = cv2.resize(img, (int(resize_factor*w), int(resize_factor*h)))
    for i in range(len(masks)):
        masks[i] = cv2.resize(masks[i], (int(resize_factor*w), int(resize_factor*h)))

    if np.random.rand() < 0.333:
        rand_img = np.random.rand(img.shape[0], img.shape[1])
        bin_below = np.less(rand_img, 0.05)
        bin_above = np.greater(rand_img, 0.95)
        img[bin_below] = 0
        img[bin_above] = 1

    mask = np.stack(masks, axis=0)

    img, mask = no_augment_fn(img, mask)

    return img, mask


def pad_first_two_dims(img, INNER_PATCH_SIZE_PX):
    """
    Pads the first two dimensions of a numpy array to ensure they are at least
    INNER_PATCH_SIZE_PX, with zero padding and centered content.

    Args:
        img: The input numpy array.
        INNER_PATCH_SIZE_PX: The minimum size for the first two dimensions.

    Returns:
        The padded numpy array.
    """
    img_shape = np.array(img.shape[:2])  # Get the shape of the first two dimensions
    target_shape = np.maximum(img_shape, INNER_PATCH_SIZE_PX)
    padding = target_shape - img_shape
    padding_before = padding // 2
    padding_after = padding - padding_before

    pad_width = tuple((padding_before[i], padding_after[i]) for i in range(2))

    # Add zero padding for the remaining dimensions
    remaining_dims_pad = tuple((0, 0) for _ in range(img.ndim - 2))
    pad_width = pad_width + remaining_dims_pad

    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img


def pad_last_two_dims(img, INNER_PATCH_SIZE_PX):
    """
    Pads the last two dimensions of a numpy array to ensure they are at least
    INNER_PATCH_SIZE_PX, with zero padding and centered content.

    Args:
        img: The input numpy array.
        INNER_PATCH_SIZE_PX: The minimum size for the last two dimensions.

    Returns:
        The padded numpy array.
    """

    img_shape = np.array(img.shape[-2:])  # Get the shape of the last two dimensions
    target_shape = np.maximum(img_shape, INNER_PATCH_SIZE_PX)
    padding = target_shape - img_shape
    padding_before = padding // 2
    padding_after = padding - padding_before

    pad_width_last_two = tuple((padding_before[i], padding_after[i]) for i in range(2))

    # Add zero padding for the preceding dimensions
    preceding_dims_pad = tuple((0, 0) for _ in range(img.ndim - 2))
    pad_width = preceding_dims_pad + pad_width_last_two

    padded_img = np.pad(img, pad_width, mode='constant', constant_values=0)
    return padded_img


def no_augment_fn(img, mask):
    # Pad the image with zeros to ensure that its size is at least INNER_PATCH_SIZE_PX
    padded_img = pad_first_two_dims(img, INNER_PATCH_SIZE_PX)
    padded_mask = pad_last_two_dims(mask, INNER_PATCH_SIZE_PX)

    h, w = padded_img.shape
    crop_h = (h - INNER_PATCH_SIZE_PX)//2
    crop_w = (w - INNER_PATCH_SIZE_PX)//2
    img = padded_img[crop_h: crop_h + INNER_PATCH_SIZE_PX, crop_w: crop_w + INNER_PATCH_SIZE_PX]
    mask = padded_mask[:, crop_h: crop_h + INNER_PATCH_SIZE_PX, crop_w: crop_w + INNER_PATCH_SIZE_PX]
    return img, mask


def construct_lsf_dict(lsf_values, i):
    ret = {}
    for lsf_key in lsf_values:
        ret[lsf_key] = lsf_values[lsf_key][i]

    return ret


# TODO Jay
# Parallelize instance retrieval.
# This could be done using multiprocessing or threading, 
# depending on the best approach for the environment.
# Addition: Parallelization is not yet implemented here.

class SWPDataset_V3():
    
    def __init__(
        self, img_nii_pths: List[Path], mask_nii_pths: List[Path],
        labels: List[Numeric], case_ids: List[str], lsf_values: dict, cache_pth: Path,
        seg_class_definitions={
            "primary": [2],
            "secondary": [1]
        },
        offset_cm_ap=None,
        offset_cm_cc=None,
        offset_cm_lr=None,
        sampling_mode="weighted",
        sampling_fov_cm=None
    ):
        # Ensure data types are correct
        assert isinstance(img_nii_pths, list), "img_nii_pths must be a list"
        assert isinstance(mask_nii_pths, list), "mask_nii_pths must be a list"
        assert isinstance(labels, list), "labels must be a list"
        assert isinstance(case_ids, list), "case_ids must be a list"
        assert isinstance(lsf_values, dict), "lsf_values must be a dict"
        assert isinstance(cache_pth, Path), "cache_pth must be a Path"

        # If weighted sampling, ensure offsets and fov are not set
        if sampling_mode == "weighted":
            assert offset_cm_ap is None, "offset_cm_ap is not compatible with weighted sampling"
            assert offset_cm_cc is None, "offset_cm_cc is not compatible with weighted sampling"
            assert offset_cm_lr is None, "offset_cm_lr is not compatible with weighted sampling"
            assert sampling_fov_cm is None, "sampling_fov_cm is not compatible with weighted sampling"
        else:
            assert sampling_mode == "fixed", "sampling_mode must be 'weighted' or 'fixed'"

        # Ensure lengths are correct
        assert len(img_nii_pths) == len(mask_nii_pths), \
            "img_nii_pths and mask_nii_pths must be the same length"
        assert len(img_nii_pths) == len(labels), \
            "img_nii_pths and labels must be the same length"

        # Ensure all files can be resolved
        for pth in img_nii_pths:
            assert isinstance(pth, Path), \
                "img_nii_pths must be a list of Path objects"
            assert pth.exists(), f"Image file not found: {pth}"
        for pth in mask_nii_pths:
            assert isinstance(pth, Path), \
                "mask_nii_pths must be a list of Path objects"
            assert pth.exists(), f"Mask file not found: {pth}"

        # Ensure all labels are numeric
        for label in labels:
            assert isinstance(label, (int, float, tuple)), \
                "labels must be a list of numeric values"

        # Ensure lsf_values are all lists of numeric values
        for lsf_key in lsf_values:
            assert isinstance(lsf_values[lsf_key], list), \
                "lsf_values must be a dict of lists"
            for lsf_val in lsf_values[lsf_key]:
                assert isinstance(lsf_val, (int, float)), \
                    "lsf_values values must be numeric"
            
            # Also ensure its length is the same as labels
            assert len(lsf_values[lsf_key]) == len(labels), \
                "lsf_values values must be the same length as labels"

        # Store segmentation class definitions
        assert isinstance(seg_class_definitions, dict), \
            "seg_class_definitions must be a dict"
        assert "primary" in seg_class_definitions, \
            "seg_class_definitions must have a 'primary' key"
        assert "secondary" in seg_class_definitions, \
            "seg_class_definitions must have a 'secondary' key"
        assert isinstance(seg_class_definitions["primary"], list), \
            "seg_class_definitions['primary'] must be a list"
        assert isinstance(seg_class_definitions["secondary"], list), \
            "seg_class_definitions['secondary'] must be a list"
        self.seg_class_definitions = seg_class_definitions

        # Create cache_pth
        assert isinstance(cache_pth, Path), \
            "cache_pth must be a Path object"
        self.cache_pth = cache_pth
        self.cache_pth.mkdir(parents=True, exist_ok=True)            

        # Create a composite list with all data
        self.data = []
        queue = [
            (img_pth, mask_pth, label, case_id, construct_lsf_dict(lsf_values, i))
            for img_pth, mask_pth, label, case_id, i in zip(
                img_nii_pths, mask_nii_pths, labels, case_ids, list(range(len(labels)))
            )
        ]
        for item in tqdm(queue):
            img_pth, mask_pth, label, case_id, item_lsf = item
            img_hash = hash_str(str(img_pth.resolve()))
            stem = case_id
            # stem = img_pth.stem
            # if "case" not in stem:
            #     stem = img_pth.parent.stem
            # if ".nii" in stem:
            #     stem = stem.replace(".nii", "")

            # Create a unique hash for the settings
            settings_obj = {
                "img_hash": img_hash,
                "offset_cm_ap": offset_cm_ap,
                "offset_cm_cc": offset_cm_cc,
                "offset_cm_lr": offset_cm_lr,
                "sampling_mode": sampling_mode,
                "seg_class_definitions": seg_class_definitions,
                "rsmp_pixel_size": RSMP_PIXEL_SIZE,
                "inner_patch_size_px": INNER_PATCH_SIZE_PX,
                "patch_size_px": PATCH_SIZE_PX,
                "sampling_fov_cm": sampling_fov_cm
            }
            try:
                settings_hash = hash_str(json.dumps(settings_obj, order_keys=True))
            except Exception:
                settings_hash = hash_str(json.dumps(settings_obj, sort_keys=True))
            hashed_name = f"{stem}_{settings_hash}"
            case_cache_pth = self.cache_pth / hashed_name
            self.data.append(prep_case_v1(
                img_pth, mask_pth, label, case_id, item_lsf, case_cache_pth,
                self.seg_class_definitions, sampling_mode,
                offset_cm_ap, offset_cm_cc, offset_cm_lr, sampling_fov_cm
            ))

    def __len__(self) -> int:
        return len(self.data)

    def get_class_fractions(self):
        # Get label for each case
        labels = [x["label"] for x in self.data]
        
        # Get counts for each label
        counts = {}
        for label in labels:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        
        # Get fractions for each label
        fractions = {}
        for label in counts:
            fractions[label] = counts[label]/len(labels)
        
        # Return result
        return fractions

    def __len__(self) -> int:
        return len(self.data)

    def get_instance(self, balanced=False) -> dict:
        # Ensure not balanced because not yet supported
        if balanced:
            # Sort cases by label
            cases_by_label = {}
            for case_ind in range(len(self.data)):
                label = self.data[case_ind]["label"]
                # check if label is a tuple
                if isinstance(label, tuple):
                    label = label[0]
                if label not in cases_by_label:
                    cases_by_label[label] = []
                cases_by_label[label].append(case_ind)

            # Randomly select a label
            labels = list(cases_by_label.keys())
            label_idx = np.random.randint(len(labels))
            label = labels[label_idx]
            
            # Randomly select a case from that label
            case_ind = np.random.choice(cases_by_label[label])
            case = self.data[case_ind]
        else:
            # Get random case
            case = np.random.choice(self.data)
        
        # Get random view
        view_ind = np.random.randint(len(VIEWS))
        view_name = VIEWS[view_ind]
        view = case["views"][view_ind]

        # Get random image
        img_ind = weighted_sample(view["cum_wts"])
        mar_wt = view["mar_wts"][img_ind]

        # Get image and segmentation paths
        img_pth = view["img_pths"][img_ind]["img"]
        mask_pth = view["img_pths"][img_ind]["seg"]

        # Load image and segmentation
        img = iio.imread(img_pth)
        mask = iio.imread(mask_pth)

        # Resize to 267x267 (INNER_PATCH_SIZE_PX)
        img_rsz = cv2.resize(
            img, (INNER_PATCH_SIZE_PX, INNER_PATCH_SIZE_PX)
        )
        mask_rsz = cv2.resize(
            mask, (INNER_PATCH_SIZE_PX, INNER_PATCH_SIZE_PX),
            interpolation=cv2.INTER_NEAREST
        )

        # Return result
        return {
            "img": img_rsz,
            "mask": mask_rsz,
            "label": case["label"],
            "view": view_name,
            "case_name": case["name"],
            "mar_wt": mar_wt,
            "img_ind": img_ind,
            "lsf": case["lsf"]
        }

    def get_patient_instances(self, ind: int):
        # Get case
        case = self.data[ind]

        # Get all instances
        instances = []
        for view_ind in range(len(VIEWS)):
            view_name = VIEWS[view_ind]
            view = case["views"][view_ind]
            for img_ind in range(len(view["img_pths"])):
                img_pth = view["img_pths"][img_ind]["img"]
                mask_pth = view["img_pths"][img_ind]["seg"]
                img = iio.imread(img_pth)
                mask = iio.imread(mask_pth)
                # Resize to 267x267 (INNER_PATCH_SIZE_PX)
                img_rsz = cv2.resize(
                    img, (INNER_PATCH_SIZE_PX, INNER_PATCH_SIZE_PX)
                )
                mask_rsz = cv2.resize(
                    mask, (INNER_PATCH_SIZE_PX, INNER_PATCH_SIZE_PX),
                    interpolation=cv2.INTER_NEAREST
                )
                instances.append({
                    "img": img_rsz,
                    "mask": mask_rsz,
                    "label": case["label"],
                    "view": view_name,
                    "case_name": case["name"],
                    "img_ind": img_ind,
                    "mar_wt": view["mar_wts"][img_ind],
                    "lsf": case["lsf"]
                })

        # Return result
        return instances

    def get_instances(self, n: int, balanced: bool=False) -> List[dict]:
        # TODO - parallelize
        # Parallelization is not yet supported here.
        return [self.get_instance(balanced=balanced) for _ in range(n)]

    def assemble_batch_np(
        self, instances: List[dict], augment: bool=False, use_seg: bool=True
    ) -> np.ndarray:
        # Get batch size
        batch_size = len(instances)

        # Initialize batch array
        batch_arr = np.zeros(
            (batch_size, 3, INNER_PATCH_SIZE_PX, INNER_PATCH_SIZE_PX),
            dtype=np.float32
        )

        # Populate batch arrays
        for i in range(batch_size):
            instance = instances[i]
            img = instance["img"]
            mask = instance["mask"]

            # Normalize image
            img = img.astype(np.float32)
            img = (img - 128)/128

            # Apply class definitions to mask
            converted_mask = np.zeros((2,) + mask.shape, np.float32)
            for ind in self.seg_class_definitions["primary"]:
                converted_mask[0][np.equal(mask, ind)] = 1
            for ind in self.seg_class_definitions["secondary"]:
                converted_mask[1][np.equal(mask, ind)] = 1

            # Augment
            if augment:
                img, converted_mask = augment_fn(img, converted_mask)
                # img, converted_mask = augment_fn_v2(img, converted_mask)
            else:
                img, converted_mask = no_augment_fn(img, converted_mask)

            # Add to batch
            batch_arr[i, 0] = img
            batch_arr[i, 1:] = img 
            if use_seg:
                batch_arr[i, 1:] = converted_mask

        # Return result
        return batch_arr

    def get_batch_mixup_np(
        self, batch_size: int, balanced: bool=False, augment: bool=False,
        use_seg: bool=True, n_classes: int=2
    ):
        # Sample instances and get labels
        instances = self.get_instances(batch_size, balanced=balanced)
        labels = np.array([x["label"] for x in instances])
        batch_arr = self.assemble_batch_np(instances, augment=augment, use_seg=use_seg)

        # Create a matrix of lsf values
        lsf_data = []
        for i in range(batch_size):
            inst_lsf_vals = []
            for lsf_key in sorted(instances[i]["lsf"]):
                inst_lsf_vals.append(instances[i]["lsf"][lsf_key])
            lsf_data.append(inst_lsf_vals)
        lsf_data = np.array(lsf_data)

        # If no augmentation, just return this as a simple batch
        if not augment:
            if labels.ndim == 2: # Time to event case
                return batch_arr, (labels[:, 1], labels[:, 0]), lsf_data
            else:
                if n_classes == 1:
                    return batch_arr, np.stack([labels], axis=-1), lsf_data
                elif n_classes == 2:
                    ret_labels = np.stack([
                        1.0 - labels,
                        labels
                    ], axis=-1)
                    return batch_arr, ret_labels, lsf_data
                else:
                    ret_labels = np.stack([
                        np.where(labels == i, 1.0, 0.0)
                        for i in range(n_classes)
                    ], axis=-1)
                    return batch_arr, ret_labels, lsf_data

        # Get another batch for mixup purposes
        instances_mixup = self.get_instances(batch_size, balanced=balanced)
        labels_mixup = np.array([x["label"] for x in instances_mixup])
        batch_mixup = self.assemble_batch_np(instances_mixup, augment=augment, use_seg=use_seg)
        lsf_data_mixup = []
        for i in range(batch_size):
            inst_lsf_vals = []
            for lsf_key in sorted(instances_mixup[i]["lsf"]):
                inst_lsf_vals.append(instances_mixup[i]["lsf"][lsf_key])
            lsf_data_mixup.append(inst_lsf_vals)
        lsf_data_mixup = np.array(lsf_data_mixup)

        # Get mixup weights
        mixup_weights = np.random.beta(0.9, 0.9, batch_size).astype(np.float32)
        mw_arr = np.reshape(mixup_weights, (batch_size, 1, 1, 1))

        # Apply mixup
        ret_arr = batch_arr*mw_arr + batch_mixup*(1 - mw_arr)
        ret_lsf_data = lsf_data*mixup_weights.reshape(-1, 1) + lsf_data_mixup*(1 - mixup_weights.reshape(-1, 1))

        if labels.ndim == 2:  # Time to event case
            # 1. Handle DURATIONS (continuous): Interpolate normally.
            durations = labels[:, 1]
            durations_mixup = labels_mixup[:, 1]
            ret_durations = durations * mixup_weights + durations_mixup * (1 - mixup_weights)

            # 2. Handle EVENTS (binary): Interpolate, then ROUND to the nearest binary value.
            events = labels[:, 0]
            events_mixup = labels_mixup[:, 0]
            interpolated_events = events * mixup_weights + events_mixup * (1 - mixup_weights)
            ret_events = np.round(interpolated_events).astype(np.int64)

            # 3. Return a tuple of the two new arrays as the final label
            return ret_arr, (ret_durations, ret_events), ret_lsf_data
        else:
            ret_labels = labels*mixup_weights + labels_mixup*(1 - mixup_weights)
            if n_classes == 1:
                return ret_arr, np.stack([ret_labels], axis=-1), ret_lsf_data
            elif n_classes == 2:
                # Explode labels
                ret_labels = np.stack([
                    1.0 - ret_labels,
                    ret_labels
                ], axis=-1)
                return ret_arr, ret_labels, ret_lsf_data
            else:
                ret_labels = np.stack([
                    np.where(labels == i, 1.0, 0.0)
                    for i in range(n_classes)
                ], axis=-1)
                return ret_arr, ret_labels, ret_lsf_data
