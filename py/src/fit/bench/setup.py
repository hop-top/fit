"""Environment checks for the external benchmark harness."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path
from typing import Literal

Status = Literal["ok", "missing", "error"]

_VENDOR_DIR = Path(__file__).resolve().parents[5] / "vendor"
_POLYGLOT_DIR = _VENDOR_DIR / "polyglot-benchmark" / "exercises"


def _check_docker() -> Status:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return "ok" if proc.returncode == 0 else "error"
    except FileNotFoundError:
        return "missing"
    except Exception:
        return "error"


def _check_import(module: str) -> Status:
    try:
        importlib.import_module(module)
        return "ok"
    except ImportError:
        return "missing"
    except Exception:
        return "error"


def _check_cli(name: str) -> Status:
    return "ok" if shutil.which(name) else "missing"


def _check_polyglot_exercises() -> Status:
    return "ok" if _POLYGLOT_DIR.is_dir() else "missing"


def check_environment() -> dict[str, Status]:
    """Verify benchmark dependencies are available.

    Returns dict mapping component name to status.
    """
    return {
        "docker": _check_docker(),
        "swebench": _check_import("swebench"),
        "tau2": _check_import("tau2"),
        "aider": _check_cli("aider"),
        "polyglot_exercises": _check_polyglot_exercises(),
    }
