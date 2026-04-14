"""Optional extra CLI arguments for TotalSegmentator and tumor segmentation from flags or env."""

from __future__ import annotations

import os
import shlex
from typing import Optional


def resolve_totalsegmentator_extra_args(*, cli_extra: Optional[str]) -> tuple[str, ...]:
    """
    TotalSegmentator extra CLI arguments.

    Precedence:

    1. ``--totalseg-extra`` (``cli_extra``)
    2. ``AXIS_TOTALSEG_EXTRA`` environment (shell-style)
    """

    if cli_extra:
        return tuple(shlex.split(cli_extra))
    env = os.environ.get("AXIS_TOTALSEG_EXTRA", "").strip()
    if env:
        return tuple(shlex.split(env))
    return ()


def resolve_tumor_extra_args(*, cli_extra: Optional[str]) -> tuple[str, ...]:
    """
    Extra arguments appended to the tumor segmentation command.

    Precedence:

    1. ``--tumor-extra`` (``cli_extra``)
    2. ``AXIS_TUMOR_EXTRA`` environment (shell-style)
    """

    if cli_extra:
        return tuple(shlex.split(cli_extra))
    env = os.environ.get("AXIS_TUMOR_EXTRA", "").strip()
    if env:
        return tuple(shlex.split(env))
    return ()
