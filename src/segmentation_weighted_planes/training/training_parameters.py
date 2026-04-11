import json
import os
from pathlib import Path

import torch


class TrainingParameters(object):
    """
    Inference-oriented defaults. Paths are overridable via environment variables
    so Docker / external mounts do not depend on a specific username or host layout.

    SWP_CACHE_ROOT   — root for V5 patch caches (default /tmp/swp_cache)
    SWP_USER         — used only for legacy cache path fragments if needed (default "swp")
    """

    USER = os.environ.get("SWP_USER", os.environ.get("USER", "swp"))

    _cache = os.environ.get("SWP_CACHE_ROOT")
    if _cache:
        CACHE_ROOT = Path(_cache).expanduser().resolve()
    else:
        CACHE_ROOT = Path(f"/tmp/swp_cache_{USER}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    DEVICE = os.environ.get("SWP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Default 4: fewer stuck jobs in Docker / limited /dev/shm; override with SWP_DATA_NUM_WORKERS.
    DATA_NUM_WORKERS = int(os.environ.get("SWP_DATA_NUM_WORKERS", "4"))
    MAX_LOADED_VIEWS = int(os.environ.get("SWP_MAX_LOADED_VIEWS", "32"))
    CACHE_MAX_PATCHES_PER_VIEW = int(os.environ.get("SWP_CACHE_MAX_PATCHES_PER_VIEW", "240"))

    CACHE_STACK_SLICES_FALLBACK = 1
    CACHE_BUILD_ONLY_SHUFFLE = False

    DEFAULT_PATCH_QUOTAS = {
        "core": 0.40,
        "boundary": 0.25,
        "support": 0.20,
        "hard_neg": 0.10,
        "background": 0.05,
    }

    DEFAULT_WEIGHT_BY_PATCH_TYPE = {
        "core": 1.0,
        "boundary": 1.0,
        "support": 1.0,
        "hard_neg": 1.0,
        "background": 1.0,
    }

    MICRO_BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 5
    AMP = True
    ENCODER_CHUNK_SIZE = int(os.environ.get("SWP_ENCODER_CHUNK_SIZE", "192"))

    BASE_MODEL = os.environ.get("SWP_BASE_MODEL", "resnet18")

    INSTANCE_DROP_P = 0
    FEAT_DROPOUT_P = 0.05
    BAG_DROPOUT_P = 0.10
    MIN_KEEP_INSTANCES = 6

    @classmethod
    def to_json(cls):
        out = {}
        for k, v in cls.__dict__.items():
            if k.startswith("__") or callable(v):
                continue
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                pass
        return out
