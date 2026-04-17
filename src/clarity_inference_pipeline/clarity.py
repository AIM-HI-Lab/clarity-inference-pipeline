"""Wrapper around vendored `segmentation_weighted_planes` PNvsRN inference."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from segmentation_weighted_planes.inference import (
    _build_external_dataset,
    _load_state_dict_flexible,
    _resolve_checkpoints,
    build_prediction_only_payload,
    run_prediction_mil_with_progress,
)
from segmentation_weighted_planes.mil_model import MILNet
from segmentation_weighted_planes.projects import project_registry
from segmentation_weighted_planes.training.training_parameters import TrainingParameters

from .config import InferenceConfig


@contextmanager
def _temporary_env(env: dict[str, str]) -> Iterator[None]:
    old_values = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, previous in old_values.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def run_clarity_inference(
    *,
    manifest_path: Path,
    data_root: Path,
    output_json: Path,
    config: InferenceConfig,
) -> Path:
    """Run PNvsRN inference on the SWP-compatible case workspace."""

    project_class = project_registry.get(config.project_name)
    if project_class is None:
        raise ValueError(f"Unknown project name {config.project_name!r}")

    checkpoint_paths = [str(Path(p).expanduser().resolve()) for p in config.checkpoint_paths]
    checkpoint_dir = str(config.checkpoint_dir.expanduser().resolve()) if config.checkpoint_dir else None
    model_root = str(config.model_root.expanduser().resolve()) if config.model_root else None

    env_updates: dict[str, str] = {}
    if config.device:
        env_updates["SWP_DEVICE"] = config.device
    if config.cache_root:
        env_updates["SWP_CACHE_ROOT"] = str(config.cache_root.expanduser().resolve())

    with _temporary_env(env_updates):
        ckpts = _resolve_checkpoints(
            checkpoint_paths or None,
            checkpoint_dir,
            config.checkpoint_dir_recursive,
            model_root,
        )
        _prev = ", ".join(c.name for c in ckpts[:5])
        _more = f" (+{len(ckpts) - 5} more)" if len(ckpts) > 5 else ""
        print(
            f"[clarity-pipeline] SWP inference: {len(ckpts)} checkpoint(s): {_prev}{_more}",
            file=sys.stderr,
            flush=True,
        )
        if (
            config.expected_checkpoint_count is not None
            and len(ckpts) != config.expected_checkpoint_count
        ):
            raise ValueError(
                f"Expected {config.expected_checkpoint_count} checkpoints for CLARITY (SWP) "
                f"inference, found {len(ckpts)}."
            )
        training_inputs_json = {
            "manifest": str(manifest_path.expanduser().resolve()),
            "data_root": str(data_root.expanduser().resolve()),
        }
        dataset = _build_external_dataset(project_class, training_inputs_json)

        fold_case_tables = []
        for ckpt in ckpts:
            net = MILNet(
                n_classes=project_class.n_classes,
                pooling=getattr(project_class, "pooling", "attn"),
                topk=getattr(project_class, "topk", 8),
            ).to(TrainingParameters.DEVICE)
            _load_state_dict_flexible(net, ckpt)
            val_rows = run_prediction_mil_with_progress(
                net,
                dataset,
                project_class,
                desc=f"clarity-pipeline | {ckpt.name}",
            )
            fold_case_tables.append(val_rows)

        combined = build_prediction_only_payload(project_class, ckpts, fold_case_tables)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return output_json
