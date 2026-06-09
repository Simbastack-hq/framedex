"""Tests for framedex.index_videos sidecar writing.

The video runtime stack (whisperx/torch) is imported lazily now, so
index_videos imports without it. These tests pin the video sidecar's
frontmatter/body output, which the pipeline.py extraction must keep
byte-for-byte identical.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from framedex import images, index_videos, pipeline
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


def test_sidecar_can_omit_ephemeral_path(tmp_path: Path) -> None:
    """Callers that process a temporary materialized file can omit `path`
    rather than persisting a location that will be deleted after the run."""
    root = tmp_path / "library.photoslibrary"
    root.mkdir()
    video = tmp_path / "download-temp" / "clip.mov"
    video.parent.mkdir()
    video.write_bytes(b"x")

    sidecar = write_sidecar(
        video,
        root,
        METADATA,
        {},
        "",
        {},
        "desc",
        {},
        [],
        sidecar_path_override=tmp_path / "clip.mov.description.md",
        omit_path=True,
    )

    assert "path" not in _frontmatter(sidecar)


class _FrozenDatetime:
    @staticmethod
    def now() -> datetime:
        return datetime(2026, 6, 8, 14, 32, 1)


def test_video_sidecar_body_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Golden-ish: with indexed_at frozen, the video sidecar's full text is
    pinned — section order, transcript heading, and the body bytes. This is the
    contract the pipeline.serialize_sidecar extraction must not break."""
    monkeypatch.setattr(pipeline, "datetime", _FrozenDatetime)
    root = tmp_path / "drive"
    root.mkdir()
    video = root / "IMG.mov"
    video.write_bytes(b"x")
    audio = {
        "language": "es",
        "speaker_count": 2,
        "transcript": "[SPEAKER_00] Hola.",
        "english_translation": "Hello.",
    }
    sidecar = write_sidecar(
        video, root, METADATA, {}, "", audio, "A market scene.", {}, []
    )
    text = sidecar.read_text()
    expected_tail = (
        "# IMG.mov\n\n"
        "## Description\n\n"
        "A market scene.\n\n"
        "## Transcript (es, 2 speakers)\n\n"
        "[SPEAKER_00] Hola.\n\n"
        "## English translation\n\n"
        "Hello.\n"
    )
    assert text.endswith(expected_tail)
    # Section order: Description precedes Transcript precedes translation.
    assert (
        text.index("## Description")
        < text.index("## Transcript")
        < text.index("## English translation")
    )
    assert _frontmatter(sidecar)["indexed_at"] == "2026-06-08T14:32:01"


def test_video_sidecar_omits_translation_when_absent(tmp_path: Path) -> None:
    """No English-translation section when the clip has none (English clip)."""
    root = tmp_path / "d"
    root.mkdir()
    video = root / "c.mov"
    video.write_bytes(b"x")
    audio = {"language": "en", "speaker_count": 1, "transcript": "Hi."}
    sidecar = write_sidecar(video, root, METADATA, {}, "", audio, "desc", {}, [])
    assert "## English translation" not in sidecar.read_text()


def test_image_only_run_never_loads_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core extras-split claim: `fdx --media images` must never call
    setup_whisper (and so never import whisperx/torch)."""
    (tmp_path / "a.jpg").write_bytes(b"x")

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("setup_whisper must not run for an image-only pass")

    monkeypatch.setattr(index_videos, "setup_whisper", _boom)
    monkeypatch.setattr(index_videos, "check_claude_cli", lambda: True)
    monkeypatch.setattr(
        images,
        "process_one_image",
        lambda *a, **k: pipeline.ProcessResult(
            sidecar=tmp_path / "a.jpg.description.md", rating="keep"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fdx", str(tmp_path), "--media", "images", "--no-faces", "--no-geocode"],
    )
    assert index_videos.main() == 0
    assert "whisperx" not in sys.modules
