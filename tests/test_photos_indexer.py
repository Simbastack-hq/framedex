"""Tests for the fdx-photos entry (framedex.photos_indexer).

Covers the Phase-2 feature surface: routing each asset to the video vs still
pipeline by media_type, the whisper gating (an images-only run never loads the
video stack), the extras preflight, and the pure run-shaping helpers.

osxphotos is stubbed at sys.modules (the module imports framedex.photos inside
main()); the heavy pipelines and runner setup are monkeypatched so nothing here
needs whisperx / Pillow / a real Photos library.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

if "osxphotos" not in sys.modules:
    sys.modules["osxphotos"] = types.SimpleNamespace(  # type: ignore[assignment]
        PhotosDB=lambda **kw: None,
    )

from framedex import photos, photos_indexer, pipeline

_SRC_DIR = str(Path(photos_indexer.__file__).resolve().parent.parent)


# --- import safety ---------------------------------------------------------


def test_photos_indexer_imports_without_osxphotos() -> None:
    """The osxphotos import is deferred into main(), so the module (and its pure
    helpers) load under CI's deps-free env. Prove it in a clean interpreter with
    no osxphotos stub — `import framedex.photos_indexer` must succeed and must
    not pull osxphotos into sys.modules at import time."""
    code = (
        "import sys, framedex.photos_indexer;"
        "print('LEAK' if 'osxphotos' in sys.modules else 'CLEAN')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": _SRC_DIR},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CLEAN", proc.stdout


# --- pure helpers ----------------------------------------------------------


def test_is_large_unfiltered_run() -> None:
    assert photos_indexer.is_large_unfiltered_run(1001, has_filter=False) is True
    # any narrowing filter (incl. --max-files) suppresses the warning
    assert photos_indexer.is_large_unfiltered_run(1001, has_filter=True) is False
    # at/below threshold: no warning
    assert photos_indexer.is_large_unfiltered_run(1000, has_filter=False) is False
    assert (
        photos_indexer.is_large_unfiltered_run(5, has_filter=False, threshold=3) is True
    )


def test_missing_media_extras() -> None:
    assert (
        photos_indexer.missing_media_extras(
            True, True, has_whisperx=True, has_pillow=True
        )
        == []
    )
    assert photos_indexer.missing_media_extras(
        True, False, has_whisperx=False, has_pillow=True
    ) == ["video"]
    assert photos_indexer.missing_media_extras(
        False, True, has_whisperx=True, has_pillow=False
    ) == ["images"]
    assert photos_indexer.missing_media_extras(
        True, True, has_whisperx=False, has_pillow=False
    ) == ["video", "images"]
    # an absent extra for media we are NOT indexing is not reported
    assert (
        photos_indexer.missing_media_extras(
            False, False, has_whisperx=False, has_pillow=False
        )
        == []
    )


# --- main() routing + gating -----------------------------------------------


def _asset(media_type: str, path: Path, uuid: str) -> photos.PhotosAsset:
    return photos.PhotosAsset(
        uuid=uuid,
        filename=path.name,
        date=datetime(2024, 1, 1),
        media_type=media_type,  # type: ignore[arg-type]
        path_original=path,
        in_icloud=False,
        raw=object(),
    )


def _patch_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the heavy engine setup so main() exercises only enumeration,
    routing, and reporting."""
    monkeypatch.setattr(photos_indexer, "missing_media_extras", lambda *a, **k: [])
    monkeypatch.setattr("framedex.runner.resolve_vision_model", lambda args: ("m", 0.0))
    monkeypatch.setattr("framedex.runner.announce_cost", lambda *a, **k: None)
    monkeypatch.setattr("framedex.runner.wire_vision_backend", lambda args: None)
    monkeypatch.setattr("framedex.runner.setup_face_db", lambda args: None)


def test_routes_each_asset_by_media_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "L.photoslibrary"
    lib.mkdir()
    out = tmp_path / "mirror"
    vid = tmp_path / "v.mov"
    vid.write_bytes(b"x")
    img = tmp_path / "p.heic"
    img.write_bytes(b"x")
    video_asset = _asset("video", vid, "V-1")
    image_asset = _asset("image", img, "I-1")

    monkeypatch.setattr(
        "framedex.photos.enumerate_assets",
        lambda *a, **k: [image_asset, video_asset],
    )
    _patch_engine(monkeypatch)
    monkeypatch.setattr(
        photos_indexer, "setup_whisper", lambda args: (None, {}, None, [])
    )

    routed: list[tuple[str, Path]] = []

    def fake_video(path: Path, root: Path, opts: Any, ctx: Any, **kw: Any) -> Any:
        routed.append(("video", path))
        return pipeline.ProcessResult(sidecar=tmp_path / "v.sidecar", rating="keep")

    def fake_image(path: Path, root: Path, opts: Any, ctx: Any, **kw: Any) -> Any:
        routed.append(("image", path))
        return pipeline.ProcessResult(sidecar=tmp_path / "i.sidecar", rating="keep")

    monkeypatch.setattr(photos_indexer, "process_one_video", fake_video)
    monkeypatch.setattr("framedex.images.process_one_image", fake_image)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fdx-photos",
            "--library",
            str(lib),
            "--output",
            str(out),
            "--no-faces",
            "--no-geocode",
            "--no-diarize",
        ],
    )

    assert photos_indexer.main() == 0
    assert ("image", img) in routed
    assert ("video", vid) in routed


def test_images_only_run_never_loads_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "L.photoslibrary"
    lib.mkdir()
    out = tmp_path / "mirror"
    img = tmp_path / "p.heic"
    img.write_bytes(b"x")

    monkeypatch.setattr(
        "framedex.photos.enumerate_assets", lambda *a, **k: [_asset("image", img, "I")]
    )
    _patch_engine(monkeypatch)

    def _boom(args: Any) -> Any:
        raise AssertionError("setup_whisper must not run for an images-only pass")

    monkeypatch.setattr(photos_indexer, "setup_whisper", _boom)
    monkeypatch.setattr(
        "framedex.images.process_one_image",
        lambda *a, **k: pipeline.ProcessResult(
            sidecar=tmp_path / "i.sidecar", rating="keep"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fdx-photos",
            "--media",
            "images",
            "--library",
            str(lib),
            "--output",
            str(out),
            "--no-faces",
            "--no-geocode",
        ],
    )
    assert photos_indexer.main() == 0
    assert "whisperx" not in sys.modules


def test_preflight_exits_when_required_extra_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = tmp_path / "L.photoslibrary"
    lib.mkdir()
    out = tmp_path / "mirror"
    img = tmp_path / "p.heic"
    img.write_bytes(b"x")
    monkeypatch.setattr(
        "framedex.photos.enumerate_assets", lambda *a, **k: [_asset("image", img, "I")]
    )
    # Simulate [images] not installed → preflight should sys.exit before setup.
    monkeypatch.setattr(
        photos_indexer, "missing_media_extras", lambda *a, **k: ["images"]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fdx-photos",
            "--media",
            "images",
            "--library",
            str(lib),
            "--output",
            str(out),
            "--no-faces",
            "--no-geocode",
        ],
    )
    with pytest.raises(SystemExit):
        photos_indexer.main()
