"""Tests for framedex.master_index — sidecar path resolution and grouping."""

import json
import sys
from pathlib import Path

import pytest

from framedex.master_index import main


def _write_sidecar(root: Path, rel: str, path_field: str) -> None:
    """Drop a sidecar at root/<rel>.description.md with the given `path` field."""
    sidecar = root / f"{rel}.description.md"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        f"---\nfile: {Path(rel).name}\npath: {path_field}\n"
        "rating: cull\ncull_reason: test\nduration_seconds: 5.0\n---\n"
        "\n## Description\n\nA clip.\n"
    )


def test_master_index_resolves_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root-relative `path` is resolved against --root, so the clip records
    an absolute path and groups into its top-level folder."""
    rel = "2024-08/drone/c.mov"
    _write_sidecar(tmp_path, rel, rel)
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])

    assert main() == 0

    index = json.loads((tmp_path / "_INDEX.json").read_text())
    assert index["clips"][0]["path"] == str(tmp_path / rel)
    assert index["trip_count"] == 1


def test_master_index_absolute_path_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Older sidecars with an absolute `path` are recorded unchanged."""
    abs_path = "/Volumes/OldDrive/archive/old.mov"
    _write_sidecar(tmp_path, "old.mov", abs_path)
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])

    assert main() == 0

    index = json.loads((tmp_path / "_INDEX.json").read_text())
    assert index["clips"][0]["path"] == abs_path
