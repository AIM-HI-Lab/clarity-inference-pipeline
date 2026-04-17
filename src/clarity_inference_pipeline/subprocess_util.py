"""Subprocess helpers (streaming logs for long-running external tools)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_subprocess_logged(
    cmd: list[str | Path],
    *,
    env: dict[str, str] | None = None,
    label: str,
) -> subprocess.CompletedProcess[str]:
    """
    Run ``cmd``, merge stderr into stdout, stream each line to stderr with ``[label]`` prefix.

    Returns a :class:`subprocess.CompletedProcess` with combined output in ``stdout``.
    """

    str_cmd = [str(x) for x in cmd]
    proc = subprocess.Popen(
        str_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    chunks: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            chunks.append(line)
            print(f"[{label}] {line}", end="", file=sys.stderr, flush=True)
    rc = proc.wait()
    out = "".join(chunks)
    return subprocess.CompletedProcess(str_cmd, rc, stdout=out, stderr="")
