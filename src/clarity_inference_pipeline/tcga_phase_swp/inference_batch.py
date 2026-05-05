import json
from pathlib import Path
from typing import Dict, List

import imageio.v2 as iio
import numpy as np
import torch
from torchvision.models import resnet50

from .v3_preprocess import cache_case_v1, hash_str, load_view, no_augment_fn


def _resolve_torch_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)

BATCH_SIZE = 16
RSMP_PIXEL_SIZE = 1
INNER_PATCH_SIZE_PX = 267
PATCH_SIZE_PX = int(np.sqrt((INNER_PATCH_SIZE_PX**2)*2)) + 1

VIEWS = ["axial", "coronal", "sagittal"]

TCGA_SEG_CLASS_DEFINITIONS = {
    "primary": [1],
    "secondary": [1, 3]
}

KITS_SEG_CLASS_DEFINITIONS = {
    "primary": [2],
    "secondary": [1, 3]
    # vector - find center center of L3 and walk 3cm up
}

PHASE_MAP = ['Noncontrast', 'Nephrographic', 'Arterial', 'Excretory']


def _instance_weight(instance: Dict) -> float:
    """
    Backward-compatible weight extraction for SWP v3 slice dicts.

    Different code paths may emit ``arb_wt`` (expected), ``arb_wts`` (legacy typo),
    or no weight at all. Default to uniform weight 1.0 instead of crashing.
    """
    for key in ("arb_wt", "arb_wts", "weight"):
        value = instance.get(key)
        if value is not None:
            return float(value)
    return 1.0

def load_models(
    model_pths: List[Path], n_classes: int, device: torch.device
) -> List[torch.nn.Module]:
    models = []
    for model_pth in model_pths:
        if not model_pth.exists():
            print(f"Model path {model_pth} does not exist. Skipping.")
            continue
        model = resnet50()
        model.fc = torch.nn.Linear(2048, n_classes)
        model.load_state_dict(torch.load(model_pth, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models

def load_case_from_cache(case_cache_pth: Path) -> List[Dict]:
    """
    Loads a full case from cache by calling `load_view` and processing
    its output into the format expected by the inference pipeline.
    """
    case_slices_buffer = []
    images_dir = case_cache_pth / "images"

    if not images_dir.exists():
        return []

    # Iterate through each view directory (e.g., 'axial', 'coronal')
    for view_dir in images_dir.iterdir():
        if not view_dir.is_dir():
            continue
            
        # 1. Call your updated load_view function
        view_data = load_view(view_dir)

        # 2. Extract the lists of paths and raw weights
        image_path_dicts = view_data["img_pths"]
        raw_weights = view_data["arb_wts"]

        # 3. Load data from paths and assemble the final instance dictionary
        for i, path_dict in enumerate(image_path_dicts):
            # Load the actual image and mask data
            img_np = iio.imread(path_dict["img"])
            seg_np_scaled = iio.imread(path_dict["seg"])

            # Descale the mask to restore original class labels (e.g., 1, 2, 3)
            seg_np = np.round(seg_np_scaled / 200.0).astype(np.uint8)

            # Assemble the final dictionary for the master_buffer
            instance = {
                "img": img_np,
                "mask": seg_np,
                "arb_wt": raw_weights[i] # Use the raw weight under the correct key
            }
            case_slices_buffer.append(instance)
            
    return case_slices_buffer

def assemble_batch_for_inference(
    instances: List[Dict], class_defs: Dict, use_seg: bool
) -> np.ndarray:
    """Assembles a batch of instances into a NumPy tensor for the model."""
    batch_size = len(instances)
    # The input shape should match your model's expectation
    batch_arr = np.zeros(
        (batch_size, 3, INNER_PATCH_SIZE_PX, INNER_PATCH_SIZE_PX),
        dtype=np.float32
    )
    
    for i, instance in enumerate(instances):
        img = instance["img"]
        mask = instance["mask"]

        img = img.astype(np.float32)
        img = (img - 128) / 128 # Normalize

        converted_mask = np.zeros((2, *mask.shape), np.float32)
        for ind in class_defs["primary"]:
            converted_mask[0][np.equal(mask, ind)] = 1
        for ind in class_defs["secondary"]:
            converted_mask[1][np.equal(mask, ind)] = 1
        
        # This function should apply center cropping
        img, converted_mask = no_augment_fn(img, converted_mask)
        
        batch_arr[i, 0] = img
        if use_seg:
            batch_arr[i, 1:] = converted_mask
        else:
            # If not using segmentation, repeat the image channel
            batch_arr[i, 1] = img
            batch_arr[i, 2] = img

    return batch_arr

def run_ensemble_inference(
    master_buffer: Dict[str, List[Dict]],
    models: List[torch.nn.Module],
    device: torch.device,
    n_classes: int,
    model_name: str,
    use_seg: bool = True,
    output_pth: Path | None = None,
    seg_class_definitions: Dict[str, List[int]] | None = None,
) -> Dict[str, Dict]:
    """
    Runs inference for an ensemble of models on a set of preprocessed cases.
    """
    final_predictions = {}
    
    with torch.no_grad():
        for case_id, all_instances in master_buffer.items():
            print(f"  Running inference on {case_id} ({len(all_instances)} slices)...")
            
            if not all_instances:
                print(f"    WARNING: No instances found for case {case_id}. Skipping.")
                final_predictions[case_id] = {
                    "prediction": "Error",
                    "probabilities": [0.0] * n_classes,
                    "error": "No valid slices found during preprocessing."
                }
                continue
                
            case_ensemble_probs = []

            for model_idx, model in enumerate(models):
                tot_pred_probs = np.zeros(n_classes)
                tot_mar_wt = 0
                
                batch_start = 0
                while batch_start < len(all_instances):
                    batch_instances = all_instances[batch_start:batch_start + BATCH_SIZE]
                    batch_weights = [_instance_weight(inst) for inst in batch_instances]
                    
                    batch_np = assemble_batch_for_inference(batch_instances, seg_class_definitions, use_seg)
                    batch_torch = torch.from_numpy(batch_np).to(device)
                    
                    outputs = model(batch_torch)
                    
                    if n_classes == 1:
                        pred_probs_np = outputs.detach().cpu().numpy()
                    else:
                        pred_probs_np = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    
                    for i in range(len(batch_weights)):
                        tot_mar_wt += batch_weights[i]
                        tot_pred_probs += pred_probs_np[i] * batch_weights[i]
                        
                    batch_start += BATCH_SIZE

                if tot_mar_wt > 0:
                    model_pred_probs = tot_pred_probs / tot_mar_wt
                    case_ensemble_probs.append(model_pred_probs)
                else:
                    print(f"    WARNING: Zero total weight for model {model_idx} on case {case_id}. Skipping model.")


            if not case_ensemble_probs:
                print(f"    ERROR: All models failed for case {case_id}.")
                final_predictions[case_id] = {
                     "prediction": "Error",
                    "probabilities": [0.0] * n_classes,
                    "error": "All models had zero weight."
                }
                continue

            # Average probabilities across the model ensemble
            avg_probs = np.mean(case_ensemble_probs, axis=0)
            if n_classes == 1:
                prediction = avg_probs
            else:
                prediction = np.argmax(avg_probs)
            
            final_predictions[case_id] = {
                "prediction": int(prediction),
                "probabilities": avg_probs.tolist(),
                # "prediction_str": PHASE_MAP[int(prediction)]
            }
            print(f"    - Case: {case_id}, Prediction: {prediction}, Probs: {avg_probs.tolist()}")

            # output results to file
            if output_pth:
                output_file = output_pth / case_id / f"{model_name}_prediction.json"
                with open(output_file, 'w') as f:
                    json.dump(final_predictions[case_id], f, indent=4)
                print(f"    - Results saved to {output_file}")

    return final_predictions

def run_inference_on_batch(
    img_pths,
    mask_pths,
    model_dir,
    n_classes,
    sampling_mode,
    use_seg,
    model_name,
    output_pth=None,
    case_ids=None,
    cache_pth=None,
    device: str | torch.device | None = None,
):
    torch_device = _resolve_torch_device(device)
    print("--- Step 1: Preprocessing Cases ---")
    master_buffer: dict[str, list] = {}
    offset_value = None if sampling_mode == "weighted" else 0.0
    sampling_fov_cm = None if sampling_mode == "weighted" else 10.0

    if "tcga" in model_name.lower():
        seg_class_definitions = TCGA_SEG_CLASS_DEFINITIONS
    else:
        seg_class_definitions = KITS_SEG_CLASS_DEFINITIONS

    if not cache_pth:
        cache = False
    else:
        cache = True
        cache_pth = Path(cache_pth)
        if not cache_pth.exists():
            cache_pth.mkdir(parents=True, exist_ok=True)

    for i, (img_pth, mask_pth) in enumerate(zip(img_pths, mask_pths)):
        case_id = case_ids[i] if case_ids else img_pth.parent.name
        print(f"Processing case {i+1}/{len(img_pths)}: {case_id}")

        # Caching logic starts here
        use_caching = cache_pth is not None
        settings_obj = {
            "img_hash": hash_str(str(img_pth.resolve())),
            "offset_cm_ap": offset_value,
            "offset_cm_cc": offset_value,
            "offset_cm_lr": offset_value,
            "sampling_mode": sampling_mode,
            "seg_class_definitions": seg_class_definitions,
            "rsmp_pixel_size": RSMP_PIXEL_SIZE,
            "inner_patch_size_px": INNER_PATCH_SIZE_PX,
            "patch_size_px": PATCH_SIZE_PX,
            "sampling_fov_cm": sampling_fov_cm
        }

        # 2. Create the unique hash from the settings
        settings_hash = hash_str(json.dumps(settings_obj, sort_keys=True))

        # 3. Construct the final cache path to match the other code
        hashed_name = f"{case_id}_{settings_hash}"
        case_cache_pth = cache_pth / hashed_name if use_caching else None
        
        # This logic is inspired by your prep_case_v1 function
        recache_needed = False
        if use_caching:
            meta_path = case_cache_pth / "meta.json"
            if not meta_path.exists():
                recache_needed = True
            else:
                # Optional: Check if file paths have changed to trigger recaching
                meta = json.load(open(meta_path))
                if meta.get("img_pth") != str(img_pth.resolve()):
                    recache_needed = True

        try:
            if use_caching and not recache_needed:
                print(f"   -> Loading {case_id} from cache...")
                # You will create this function in the next step
                case_slices_buffer = load_case_from_cache(case_cache_pth)
            else:
                # If caching is enabled, process and save to disk
                if use_caching:
                    print(f"   -> Caching {case_id} to {case_cache_pth}...")
                    cache_case_v1(
                        img_pth, mask_pth, case_cache_pth, seg_class_definitions,
                        sampling_mode, offset_cm_ap=offset_value, offset_cm_cc=offset_value, offset_cm_lr=offset_value,
                        fov_cm=sampling_fov_cm, cache=True # Enable caching
                    )
                    case_slices_buffer = load_case_from_cache(case_cache_pth)
                else:
                    print("   -> Processing in-memory (caching disabled)...")
                    case_slices_buffer = cache_case_v1(
                        img_pth, mask_pth, None, seg_class_definitions,
                        sampling_mode, offset_cm_ap=offset_value, offset_cm_cc=offset_value, offset_cm_lr=offset_value,
                        fov_cm=sampling_fov_cm, cache=False # Disable caching
                    )
            
            master_buffer[case_id] = case_slices_buffer

        except Exception as e:
            print(f"ERROR processing case {case_id}: {e}")
            master_buffer[case_id] = []

    print("\n--- Step 2: Loading Models ---")
    model_pths = list(Path(model_dir).glob("*.pth"))
    if not model_pths:
        print("No model files found in the specified directory.")
        return {}
    print(f"Found {len(model_pths)} model files in {model_dir}")
    models = load_models(model_pths, n_classes=n_classes, device=torch_device)
    print(f"Loaded {len(models)} models.")

    print("\n--- Step 3: Running Inference ---")
    predictions = run_ensemble_inference(
        master_buffer,
        models,
        torch_device,
        n_classes=n_classes,
        use_seg=use_seg,
        output_pth=output_pth,
        model_name=model_name,
        seg_class_definitions=seg_class_definitions,
    )

    print("Inference completed.")
    return predictions
    