"""Tests for examples.train_advisor — training pipeline example."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Regression: no manual glob in directory branch
# ---------------------------------------------------------------------------


class TestMainDelegatesLoadBatch:
    """``train_advisor.main()`` must delegate directory handling to
    ``ingester.load_batch([traces_path])`` instead of manually
    globbing ``*.jsonl``, ``*.json``, etc. Manual globbing misses
    ``*.ndjson`` files and does not recurse into subdirectories.
    """

    def test_no_manual_glob_in_directory_branch(self) -> None:
        """Directory handling should delegate to load_batch, not
        glob."""
        from examples.train_advisor import main

        source = inspect.getsource(main)

        assert 'traces_path.glob("*.jsonl")' not in source, (
            "train_advisor manually globs *.jsonl "
            "instead of delegating to ingester.load_batch"
        )


# ---------------------------------------------------------------------------
# Regression: paths resolve against cwd
# ---------------------------------------------------------------------------


class TestMainResolvesPathFromCwd:
    """``train_advisor.main`` must resolve relative CLI ``--traces``
    paths against the caller's working directory (``Path.cwd()``),
    not against the script's own directory via ``Path(__file__)``.
    """

    def test_no_path_file_in_trace_resolution(self) -> None:
        """Source of main must not use Path(__file__) for resolving
        the traces path."""
        examples_dir = str(
            Path(__file__).resolve().parent.parent / "examples"
        )
        if examples_dir not in sys.path:
            sys.path.insert(0, examples_dir)

        from train_advisor import main

        src = inspect.getsource(main)
        assert "Path(__file__)" not in src, (
            "Relative trace paths are resolved against "
            "the script directory (Path(__file__)) instead of "
            "the caller's working directory."
        )
