"""Tests for optional TotalSegmentator / tumor extra-arg resolution."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from axis_inference_pipeline.pipeline_profile import (
    resolve_totalsegmentator_extra_args,
    resolve_tumor_extra_args,
)


class TestResolveExtraArgs(unittest.TestCase):
    def test_totalseg_cli_overrides_env(self) -> None:
        with mock.patch.dict(os.environ, {"AXIS_TOTALSEG_EXTRA": "-f"}):
            got = resolve_totalsegmentator_extra_args(cli_extra="-x 1")
        self.assertEqual(got, ("-x", "1"))

    def test_totalseg_env_when_no_cli(self) -> None:
        with mock.patch.dict(os.environ, {"AXIS_TOTALSEG_EXTRA": "-nr 2"}):
            got = resolve_totalsegmentator_extra_args(cli_extra=None)
        self.assertEqual(got, ("-nr", "2"))

    def test_totalseg_empty(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            got = resolve_totalsegmentator_extra_args(cli_extra=None)
        self.assertEqual(got, ())

    def test_tumor_cli_overrides_env(self) -> None:
        with mock.patch.dict(os.environ, {"AXIS_TUMOR_EXTRA": "--disable_tta"}):
            got = resolve_tumor_extra_args(cli_extra="--mode normal")
        self.assertEqual(got, ("--mode", "normal"))

    def test_tumor_env_when_no_cli(self) -> None:
        with mock.patch.dict(os.environ, {"AXIS_TUMOR_EXTRA": "-step_size 0.5"}):
            got = resolve_tumor_extra_args(cli_extra=None)
        self.assertEqual(got, ("-step_size", "0.5"))

    def test_tumor_empty(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            got = resolve_tumor_extra_args(cli_extra=None)
        self.assertEqual(got, ())
