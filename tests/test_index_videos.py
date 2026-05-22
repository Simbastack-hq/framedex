"""Tests for framedex.index_videos sidecar writing.

index_videos imports the heavy runtime stack (whisperx, torch, ...) at module
level and exits if it is missing. CI installs only the dev + test dependency
groups, so skip this module there. The sidecar consumers (query, master_index)
are covered by stdlib-light tests that always run.
"""

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytest.importorskip("whisperx", reason="full runtime stack not installed")

from framedex.index_videos import write_sidecar

METADATA = {
    "duration_seconds": 12.3,
    "width": 3840,
    "height": 2160,
    "codec": "hvc1",
    "size_bytes": 245678912,
    "creation_time": "2024-08-14T07:23:11Z",
}


def _frontmatter(sidecar: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", yaml.safe_load(sidecar.read_text().split("---", 2)[1])
    )


def test_sidecar_path_is_relative_to_root(tmp_path: Path) -> None:
    """The `path` field is stored relative to the scan root, never absolute, so
    sidecars survive the archive being moved or remounted. Regression test for
    the absolute-path leak (issue #4)."""
    root = tmp_path / "SSD-2024"
    clipdir = root / "2024-08-construction" / "drone"
    clipdir.mkdir(parents=True)
    video = clipdir / "IMG_4827.mov"
    video.write_bytes(b"fake")

    sidecar = write_sidecar(video, root, METADATA, {}, "", {}, "A drone shot.", {}, [])

    fm = _frontmatter(sidecar)
    assert fm["path"] == "2024-08-construction/drone/IMG_4827.mov"
    assert not Path(fm["path"]).is_absolute()
    # Nothing in the sidecar should leak the absolute root.
    assert str(root) not in sidecar.read_text()


def test_sidecar_path_for_root_level_video(tmp_path: Path) -> None:
    """A video directly at the scan root gets a bare filename as its path."""
    root = tmp_path / "drive"
    root.mkdir()
    video = root / "clip.mp4"
    video.write_bytes(b"x")

    sidecar = write_sidecar(video, root, METADATA, {}, "", {}, "desc", {}, [])

    assert _frontmatter(sidecar)["path"] == "clip.mp4"
