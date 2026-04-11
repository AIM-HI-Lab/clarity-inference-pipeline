import json
import os
from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from segmentation_weighted_planes.data_loader_v5 import SWPDataset_V5, V5CacheConfig
from segmentation_weighted_planes.datasets.nifti_manifest import (
    get_nifti_manifest_dataset_labels,
)
from segmentation_weighted_planes.mil_model import (
    MILNet,
    get_binary_metrics,
    get_continuous_metrics,
    get_n_class_metrics,
    validate_case_mil,
)
from segmentation_weighted_planes.projects import project_registry
from segmentation_weighted_planes.training.training_parameters import TrainingParameters

SHARD_FORMAT = "swp_inference_shard_v1"
SHARD_GLOB = "inference_shard_*.json"


def _load_state_dict_flexible(model: torch.nn.Module, ckpt_pth: Path):
    # PyTorch 2.6+ defaults weights_only=True; full training checkpoints need False.
    _kw = {"map_location": TrainingParameters.DEVICE}
    try:
        state = torch.load(ckpt_pth, **_kw, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_pth, **_kw)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)


def _collect_checkpoints(model_root: Path):
    direct = sorted(model_root.glob("fold_*/checkpoints/best_model.pth"))
    if direct:
        return direct
    nested = sorted(model_root.glob("**/fold_*/checkpoints/best_model.pth"))
    if nested:
        return nested
    # e.g. pnvrn_folds/m1/fold_0.pth (no checkpoints/ subfolder)
    return sorted(model_root.glob("**/fold_*.pth"))


def _collect_checkpoints_flat_dir(checkpoint_dir: Path, recursive: bool = False) -> List[Path]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {checkpoint_dir}")
    if recursive:
        ckpts = sorted(checkpoint_dir.glob("**/*.pth"))
    else:
        ckpts = sorted(checkpoint_dir.glob("*.pth"))
        # Common layout: pnvrn_folds/m1/fold_0.pth — nothing at top level; search down.
        if len(ckpts) == 0:
            ckpts = sorted(checkpoint_dir.glob("**/*.pth"))
    return ckpts


def _build_external_dataset(project_class, training_inputs_json):
    if project_class.dataset == "nifti_manifest":
        dataset_labels = get_nifti_manifest_dataset_labels(
            project_class=project_class,
            image_paths=project_class.image_path_filenames,
            training_inputs_json=training_inputs_json,
        )
    else:
        raise ValueError(f"Unknown dataset {project_class.dataset}")

    img_pairs = dataset_labels["img_nii_pths"]
    labels = dataset_labels["labels"]
    case_ids = dataset_labels["case_ids"]

    cache_parent = getattr(TrainingParameters, "CACHE_ROOT", Path(f"/home/{TrainingParameters.USER}/beegfs"))
    cache_pth = Path(cache_parent) / f"{project_class.dataset}_{project_class.sampling_mode}_cache_v5"
    cache_pth.mkdir(parents=True, exist_ok=True)

    cfg = V5CacheConfig(
        max_patches_per_view=int(getattr(TrainingParameters, "CACHE_MAX_PATCHES_PER_VIEW", 240)),
        stack_slices=max(1, int(getattr(project_class, "slab_depth", 1))),
        patch_quotas=None,
    )

    ds = SWPDataset_V5(
        img_nii_pths=[x[0] for x in img_pairs],
        mask_nii_pths=[x[1] for x in img_pairs],
        labels=labels,
        case_ids=case_ids,
        lsf_values={},
        cache_pth=cache_pth,
        seg_class_definitions=project_class.seg_class_definitions,
        num_workers=int(getattr(TrainingParameters, "DATA_NUM_WORKERS", 8)),
        max_loaded_views=int(getattr(TrainingParameters, "MAX_LOADED_VIEWS", 32)),
        cfg=cfg,
        project_class=project_class,
    )
    return ds


def _metrics_for(project_class, labels, preds, pred_probs):
    if project_class.n_classes == 1:
        return get_continuous_metrics(labels, preds, pred_probs)
    if project_class.n_classes == 2:
        return get_binary_metrics(labels, preds, pred_probs)
    return get_n_class_metrics(labels, preds, pred_probs)


def run_full_validation_mil_with_progress(net, dataset: SWPDataset_V5, project_class, desc: str = "Cases"):
    """
    Same logic as trainer.run_full_validation_mil but with a tqdm progress bar per case.
    """
    n_classes = project_class.n_classes
    class_of_interest = project_class.class_of_interest

    case_labels = []
    case_preds = []
    case_pred_probs = []
    val_json = []

    n = len(dataset)
    it = range(n)
    if n > 0:
        it = tqdm(it, total=n, desc=desc, unit="case", dynamic_ncols=True)

    for case_idx in it:
        pred_probs, pred, label, case = validate_case_mil(net, dataset, case_idx, project_class)

        case_labels.append(label)
        case_preds.append(pred)
        if n_classes == 1:
            case_pred_probs.append(pred_probs[0])
        elif n_classes == 2:
            case_pred_probs.append(pred_probs[class_of_interest])
        else:
            case_pred_probs.append(pred_probs)

        if project_class.output_type == "continuous":
            j_pred = float(pred)
            j_lab = float(label)
        elif project_class.output_type == "string":
            j_pred = str(pred)
            j_lab = str(label)
        else:
            j_pred = int(pred)
            j_lab = int(label)

        val_json.append(
            {
                "case_id": case,
                "label": j_lab,
                "pred": j_pred,
                "pred_probs": [float(x) for x in pred_probs],
            }
        )

    if n_classes == 1:
        metrics = get_continuous_metrics(case_labels, case_preds, case_pred_probs)
    elif n_classes == 2:
        metrics = get_binary_metrics(case_labels, case_preds, case_pred_probs)
    else:
        metrics = get_n_class_metrics(case_labels, case_preds, case_pred_probs)

    return metrics, val_json


def _resolve_checkpoints(
    checkpoint_paths: Optional[List[str]],
    checkpoint_dir: Optional[str],
    checkpoint_dir_recursive: bool,
    model_root: Optional[str],
) -> List[Path]:
    if checkpoint_paths:
        ckpts = []
        for p in checkpoint_paths:
            pp = Path(p).expanduser().resolve()
            if not pp.exists():
                raise FileNotFoundError(f"Checkpoint not found: {pp}")
            if pp.suffix.lower() != ".pth":
                raise ValueError(f"Expected .pth file, got: {pp}")
            ckpts.append(pp)
        return ckpts
    if checkpoint_dir:
        ckpts = _collect_checkpoints_flat_dir(
            Path(checkpoint_dir), recursive=checkpoint_dir_recursive
        )
        if len(ckpts) == 0:
            raise ValueError(
                f"No .pth files found in {Path(checkpoint_dir).resolve()} "
                f"({'recursive' if checkpoint_dir_recursive else 'top-level only'})."
            )
        return ckpts
    if model_root:
        root = Path(model_root)
        ckpts = _collect_checkpoints(root)
        if len(ckpts) == 0:
            raise ValueError(f"No checkpoints found under {root}")
        return ckpts
    raise ValueError(
        "Provide one of: --checkpoint-paths, --checkpoint-dir, or --model-root."
    )


def _build_combined_payload(
    project_class,
    ckpts: List[Path],
    fold_case_tables: List[list],
    fold_metrics: List[dict],
):
    if not fold_case_tables:
        return {
            "metadata": {
                "project_name": project_class.project_name,
                "n_checkpoints": 0,
                "checkpoint_paths": [],
            },
            "ensemble_metrics": {},
            "per_checkpoint_metrics": [],
            "cases": [],
        }

    n_cases = len(fold_case_tables[0])
    cases_out = []

    for case_i in range(n_cases):
        case_id = fold_case_tables[0][case_i]["case_id"]
        label = fold_case_tables[0][case_i]["label"]
        per_model = []
        for f, ckpt in enumerate(ckpts):
            row = fold_case_tables[f][case_i]
            per_model.append(
                {
                    "checkpoint": str(ckpt),
                    "pred": row["pred"],
                    "pred_probs": [float(x) for x in row["pred_probs"]],
                }
            )

        probs_mat = np.array(
            [fold_case_tables[f][case_i]["pred_probs"] for f in range(len(ckpts))]
        )
        mean_probs = probs_mat.mean(axis=0).tolist()

        if project_class.n_classes == 1:
            ensemble_pred = float(mean_probs[0])
            prob_for_metric = ensemble_pred
        elif project_class.n_classes == 2:
            ensemble_pred = int(mean_probs[1] >= 0.5)
            prob_for_metric = float(mean_probs[1])
        else:
            ensemble_pred = int(np.argmax(mean_probs))
            prob_for_metric = mean_probs

        cases_out.append(
            {
                "case_id": case_id,
                "label": label,
                "per_model": per_model,
                "ensemble_pred_probs": [float(x) for x in mean_probs],
                "ensemble_pred": ensemble_pred,
            }
        )

    ens_labels = [c["label"] for c in cases_out]
    ens_preds = [c["ensemble_pred"] for c in cases_out]
    if project_class.n_classes == 1:
        ens_probs = [c["ensemble_pred_probs"][0] for c in cases_out]
    elif project_class.n_classes == 2:
        ens_probs = [c["ensemble_pred_probs"][1] for c in cases_out]
    else:
        ens_probs = [c["ensemble_pred_probs"] for c in cases_out]

    ensemble_metrics = _metrics_for(project_class, ens_labels, ens_preds, ens_probs)

    per_checkpoint_metrics = [
        {"checkpoint": str(ckpt), "metrics": m} for ckpt, m in zip(ckpts, fold_metrics)
    ]

    return {
        "metadata": {
            "project_name": project_class.project_name,
            "n_checkpoints": len(ckpts),
            "checkpoint_paths": [str(p) for p in ckpts],
        },
        "ensemble_metrics": ensemble_metrics,
        "per_checkpoint_metrics": per_checkpoint_metrics,
        "cases": cases_out,
    }


def _effective_array_task_id(cli_value: Optional[int]) -> Optional[int]:
    if cli_value is not None:
        return cli_value
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None or env == "":
        return None
    return int(env)


def _write_shard(
    shard_path: Path,
    project_name: str,
    array_task_id: int,
    n_checkpoints_total: int,
    checkpoint: Path,
    metrics: dict,
    val_rows: list,
):
    payload = {
        "format": SHARD_FORMAT,
        "project_name": project_name,
        "array_task_id": array_task_id,
        "checkpoint_index": array_task_id,
        "n_checkpoints_total": n_checkpoints_total,
        "checkpoint_path": str(checkpoint.resolve()),
        "metrics": metrics,
        "val_rows": val_rows,
    }
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with shard_path.open("w") as f:
        json.dump(payload, f, indent=2)


def _load_and_sort_shards(shards_dir: Path, project_name: str) -> List[dict]:
    paths = sorted(shards_dir.glob(SHARD_GLOB))
    if not paths:
        raise FileNotFoundError(f"No {SHARD_GLOB} under {shards_dir}")
    shards = []
    for p in paths:
        with p.open() as f:
            shards.append(json.load(f))
    for s in shards:
        if s.get("format") != SHARD_FORMAT:
            raise ValueError(f"Unknown shard format in {s}: expected {SHARD_FORMAT!r}")
    shards.sort(key=lambda x: int(x["checkpoint_index"]))
    for s in shards:
        if s.get("project_name") != project_name:
            raise ValueError(
                f"Shard project_name={s.get('project_name')!r} != {project_name!r} (--project-name)"
            )
    return shards


def run_merge_shards(
    project_name: str,
    shards_dir: Path,
    output_json: Path,
    expect_count: Optional[int] = None,
):
    project_class = project_registry.get(project_name)
    if project_class is None:
        raise ValueError(f"Unknown project name {project_name}")

    shards = _load_and_sort_shards(shards_dir, project_name)
    if expect_count is not None and len(shards) != expect_count:
        raise ValueError(
            f"Expected {expect_count} shards, found {len(shards)} in {shards_dir}"
        )

    ckpts = [Path(s["checkpoint_path"]) for s in shards]
    fold_case_tables = [s["val_rows"] for s in shards]
    fold_metrics = [s["metrics"] for s in shards]

    for i, s in enumerate(shards):
        if int(s["checkpoint_index"]) != i:
            raise ValueError(
                f"Shard checkpoint_index mismatch: expected contiguous 0..N-1, got index {s['checkpoint_index']!r} at position {i}"
            )

    combined = _build_combined_payload(project_class, ckpts, fold_case_tables, fold_metrics)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(combined, f, indent=2)

    aux_dir = output_json.parent
    with (aux_dir / "all_checkpoint_metrics.json").open("w") as f:
        json.dump(
            [{"checkpoint": str(c), "metrics": m} for c, m in zip(ckpts, fold_metrics)],
            f,
            indent=2,
        )
    with (aux_dir / "ensemble_metrics.json").open("w") as f:
        json.dump(combined["ensemble_metrics"], f, indent=2)

    print(f"Merged {len(shards)} shards -> {output_json}")


def main():
    parser = ArgumentParser(
        description="External validation: one or more .pth checkpoints; optional Slurm array (one model per task)."
    )
    parser.add_argument("--project-name", type=str, required=True)

    parser.add_argument(
        "--merge-shards-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Merge inference_shard_*.json from parallel array jobs; write --output-json. "
        "Does not run inference (no checkpoint arguments needed beyond project).",
    )
    parser.add_argument(
        "--expect-shard-count",
        type=int,
        default=None,
        help="With --merge-shards-dir, fail unless exactly this many shards are present.",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--checkpoint-paths",
        nargs="+",
        metavar="PATH",
        help="Explicit list of .pth checkpoint files.",
    )
    group.add_argument(
        "--checkpoint-dir",
        type=str,
        metavar="DIR",
        help="Folder: every *.pth (sorted).",
    )
    group.add_argument(
        "--model-root",
        type=str,
        help="Folder with fold_*/checkpoints/best_model.pth.",
    )
    parser.add_argument(
        "--checkpoint-dir-recursive",
        action="store_true",
        help="With --checkpoint-dir, use **/*.pth.",
    )
    parser.add_argument(
        "--training-inputs-json",
        type=str,
        default="{}",
        help="Optional JSON for project-specific filters.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Final combined report path (or merge output).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Per-checkpoint JSON, shards, and defaults for output-json parent.",
    )
    parser.add_argument(
        "--array-task-id",
        type=int,
        default=None,
        help="Slurm array index: run only checkpoints[array_task_id] (one model per node). "
        "If omitted, SLURM_ARRAY_TASK_ID is used when set; otherwise all checkpoints run in one job.",
    )
    args = parser.parse_args()

    if args.merge_shards_dir:
        if not args.output_json:
            raise ValueError("--merge-shards-dir requires --output-json")
        run_merge_shards(
            args.project_name,
            Path(args.merge_shards_dir).expanduser().resolve(),
            Path(args.output_json).expanduser().resolve(),
            expect_count=args.expect_shard_count,
        )
        return

    if not (args.checkpoint_paths or args.checkpoint_dir or args.model_root):
        raise ValueError(
            "Provide --checkpoint-paths, --checkpoint-dir, or --model-root (or use --merge-shards-dir)."
        )

    project_class = project_registry.get(args.project_name)
    if project_class is None:
        raise ValueError(f"Unknown project name {args.project_name}")

    ckpts_full = _resolve_checkpoints(
        args.checkpoint_paths,
        args.checkpoint_dir,
        args.checkpoint_dir_recursive,
        args.model_root,
    )

    array_task_id = _effective_array_task_id(args.array_task_id)
    n_total = len(ckpts_full)

    ckpts = ckpts_full
    if array_task_id is not None:
        if array_task_id < 0 or array_task_id >= n_total:
            raise ValueError(
                f"array_task_id={array_task_id} out of range for {n_total} checkpoint(s). "
                f"Use Slurm --array=0-{n_total - 1}."
            )
        ckpts = [ckpts_full[array_task_id]]

    if array_task_id is not None and args.output_dir is None:
        raise ValueError(
            "Slurm array mode requires a shared --output-dir (same path on every array task) "
            "so shards are written to one directory. Example: --output-dir /beegfs/.../shards_run1"
        )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    elif args.output_json is not None:
        output_dir = Path(args.output_json).expanduser().resolve().parent
    elif args.model_root is not None:
        output_dir = Path(args.model_root) / f"external_validation_{int(time())}"
    elif args.checkpoint_dir is not None:
        output_dir = Path(args.checkpoint_dir).expanduser().resolve() / f"external_validation_{int(time())}"
    else:
        output_dir = Path.cwd() / f"external_validation_{int(time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_json is not None:
        main_json_path = Path(args.output_json).expanduser().resolve()
        main_json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        main_json_path = output_dir / "external_validation_predictions.json"

    training_inputs_json = json.loads(args.training_inputs_json)

    print(f"[inference] Device: {TrainingParameters.DEVICE}")
    print(f"[inference] Building dataset (V5 cache may take a while on first touch)...")
    ext_ds = _build_external_dataset(project_class, training_inputs_json)
    print(f"[inference] Dataset ready: {len(ext_ds)} cases")

    if array_task_id is None:
        print(f"[inference] Running all {n_total} checkpoint(s) in this job.")
    else:
        print(
            f"[inference] Array mode: task {array_task_id} of {n_total} — single checkpoint:\n"
            f"  {ckpts[0]}\n"
            f"  shard file: {output_dir / f'inference_shard_{array_task_id:05d}.json'}"
        )

    fold_results = []
    fold_case_tables = []

    for i, ckpt in enumerate(ckpts):
        # When not array-split, i is 0..N-1; when array-split, only one iter with logical index array_task_id
        logical_idx = array_task_id if array_task_id is not None else i
        label = f"Checkpoint {logical_idx + 1}/{n_total}"
        print(f"\n[inference] {label}: {ckpt}")
        print(f"[inference] Loading weights...")
        net = MILNet(
            n_classes=project_class.n_classes,
            pooling=getattr(project_class, "pooling", "attn"),
            topk=getattr(project_class, "topk", 8),
        ).to(TrainingParameters.DEVICE)
        _load_state_dict_flexible(net, ckpt)
        metrics, val_rows = run_full_validation_mil_with_progress(
            net, ext_ds, project_class, desc=f"{label} | cases"
        )
        fold_results.append({"checkpoint": str(ckpt), "metrics": metrics})
        fold_case_tables.append(val_rows)

        if array_task_id is not None:
            shard_name = f"inference_shard_{array_task_id:05d}.json"
            shard_path = output_dir / shard_name
            _write_shard(
                shard_path,
                args.project_name,
                array_task_id,
                n_total,
                ckpt,
                metrics,
                val_rows,
            )
            print(f"[inference] Wrote shard: {shard_path}")
            print(
                f"[inference] After all array tasks finish, merge with:\n"
                f"  python -m segmentation_weighted_planes.inference \\\n"
                f"    --project-name {args.project_name} \\\n"
                f"    --merge-shards-dir {output_dir} \\\n"
                f"    --expect-shard-count {n_total} \\\n"
                f"    --output-json <path/to/combined.json>"
            )
        else:
            with (output_dir / f"checkpoint_{i}_metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)
            with (output_dir / f"checkpoint_{i}_predictions.json").open("w") as f:
                json.dump(val_rows, f, indent=2)

    if array_task_id is not None:
        return

    fold_metrics = [fr["metrics"] for fr in fold_results]
    combined = _build_combined_payload(
        project_class, ckpts_full, fold_case_tables, fold_metrics
    )

    with main_json_path.open("w") as f:
        json.dump(combined, f, indent=2)

    with (output_dir / "all_checkpoint_metrics.json").open("w") as f:
        json.dump(fold_results, f, indent=2)
    with (output_dir / "ensemble_metrics.json").open("w") as f:
        json.dump(combined["ensemble_metrics"], f, indent=2)

    print(f"\n[inference] Combined report: {main_json_path}")
    print(f"[inference] Aux outputs under: {output_dir}")


if __name__ == "__main__":
    main()
