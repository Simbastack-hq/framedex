"""Tests for framedex.images — the still-photo pipeline.

exiftool, Pillow, and the vision backends are mocked so the suite runs on CI
(which installs neither the [images] extra nor system exiftool).
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from framedex import images, pipeline


def _frontmatter(sidecar: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", yaml.safe_load(sidecar.read_text().split("---", 2)[1])
    )


# --- discovery -------------------------------------------------------------


def test_find_images_filters(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.CR3").write_bytes(b"x")
    (tmp_path / "c.png").write_bytes(b"x")
    (tmp_path / ".hidden.jpg").write_bytes(b"x")  # hidden → skip
    (tmp_path / "clip.mov").write_bytes(b"x")  # video → not an image
    out = tmp_path / "_output"
    out.mkdir()
    (out / "d.jpg").write_bytes(b"x")  # _-prefixed folder → skip

    found = {p.name for p in images.find_images(tmp_path, [])}
    assert found == {"a.jpg", "b.CR3", "c.png"}


def test_find_images_exclude_pattern(tmp_path: Path) -> None:
    (tmp_path / "keep.jpg").write_bytes(b"x")
    sub = tmp_path / "skipme"
    sub.mkdir()
    (sub / "drop.jpg").write_bytes(b"x")
    found = {p.name for p in images.find_images(tmp_path, ["skipme"])}
    assert found == {"keep.jpg"}


# --- exif datetime ---------------------------------------------------------


def test_normalize_exif_datetime() -> None:
    assert (
        images._normalize_exif_datetime("2024:08:14 07:23:11") == "2024-08-14T07:23:11"
    )
    assert images._normalize_exif_datetime("2024:08:14") == "2024-08-14"
    assert images._normalize_exif_datetime("") == ""


# --- get_image_metadata (mocked exiftool) ----------------------------------


def _fake_exiftool(payload: dict[str, Any], returncode: int = 0) -> Any:
    def run(cmd: list[str], **kw: Any) -> Any:
        return types.SimpleNamespace(
            returncode=returncode, stdout=json.dumps([payload]) if payload else "[]"
        )

    return run


def test_get_image_metadata_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "DSC_4827.RAF"
    img.write_bytes(b"0123456789")
    monkeypatch.setattr(
        "framedex.images.subprocess.run",
        _fake_exiftool(
            {
                "Make": "FUJIFILM",
                "Model": "X-T5",
                "LensModel": "XF16-80mmF4 R OIS WR",
                "FocalLength": "80.0 mm",
                "FNumber": 4.0,
                "ExposureTime": "1/1000",
                "ISO": 400,
                "ImageWidth": 7728,
                "ImageHeight": 5152,
                "Orientation": "Horizontal (normal)",
                "DateTimeOriginal": "2024:08:14 07:23:11",
            }
        ),
    )
    meta = images.get_image_metadata(img)
    assert meta["dimensions"] == "7728x5152"
    assert meta["creation_time"] == "2024-08-14T07:23:11"
    assert meta["size_bytes"] == 10
    cam = meta["camera"]
    assert cam["make"] == "FUJIFILM"
    assert cam["model"] == "X-T5"
    assert cam["lens"] == "XF16-80mmF4 R OIS WR"
    assert cam["aperture"] == 4.0
    assert cam["shutter"] == "1/1000"
    assert cam["iso"] == 400


def test_get_image_metadata_exiftool_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    monkeypatch.setattr(
        "framedex.images.subprocess.run", _fake_exiftool({}, returncode=1)
    )
    meta = images.get_image_metadata(img)
    assert meta["camera"] == {}
    assert meta["dimensions"] == ""
    assert meta["size_bytes"] == 1


# --- frontmatter -----------------------------------------------------------


def _structured() -> dict[str, Any]:
    return {
        "rating": "keep",
        "cull_reason": "",
        "technical": {"focus": "sharp", "exposure": "strong", "composition": "strong"},
        "lighting": "golden_hour",
        "time_of_day": "golden_hour",
        "dominant_colors": ["amber", "ochre"],
        "scene_type": "wildlife",
        "people_count": 0,
        "keywords": ["giraffe", "waterhole"],
    }


def test_build_image_frontmatter_schema(tmp_path: Path) -> None:
    img = tmp_path / "DSC.RAF"
    meta = {
        "size_bytes": 100,
        "creation_time": "2024-08-14T07:23:11",
        "dimensions": "6000x4000",
        "camera": {"make": "FUJIFILM", "model": "X-T5"},
    }
    fm = images.build_image_frontmatter(
        img,
        tmp_path,
        meta,
        {"lat": -1.4, "lon": 35.0},
        "Maasai Mara, Kenya",
        _structured(),
        [],
    )
    assert fm["media_type"] == "image"
    assert fm["path"] == "DSC.RAF"
    assert fm["dimensions"] == "6000x4000"
    assert fm["camera"]["model"] == "X-T5"
    assert fm["location"]["place"] == "Maasai Mara, Kenya"
    assert fm["technical"]["composition"] == "strong"
    assert fm["scene_type"] == "wildlife"
    assert fm["face_count"] == 0
    # video-only fields must NOT appear on a photo sidecar
    for absent in ("duration_seconds", "codec", "speaker_count", "language_detected"):
        assert absent not in fm


def test_build_image_frontmatter_omit_path_and_empties(tmp_path: Path) -> None:
    img = tmp_path / "x.jpg"
    meta = {"size_bytes": 1, "creation_time": "", "dimensions": "", "camera": {}}
    fm = images.build_image_frontmatter(
        img, tmp_path, meta, {}, "", _structured(), [], omit_path=True
    )
    assert "path" not in fm
    assert "camera" not in fm  # empty camera block omitted
    assert "location" not in fm  # no gps → no location


# --- prompt ----------------------------------------------------------------


def test_build_image_vision_prompt() -> None:
    context = {
        "filename": "DSC.RAF",
        "parent_folder": "2024-mara",
        "creation_time": "2024-08-14T07:23:11",
        "camera": {"model": "X-T5", "aperture": 4.0},
        "location": {"place": "Maasai Mara", "lat": -1.4, "lon": 35.0},
    }
    p = images._build_image_vision_prompt(Path("/tmp/preview.jpg"), context, True)
    assert "scene_type" in p
    assert "composition" in p
    assert "## Description" in p
    assert "/tmp/preview.jpg" in p  # include_paths → preview path present
    # photo prompt must not carry video-only concepts
    assert "Transcript" not in p
    assert "Duration" not in p

    p2 = images._build_image_vision_prompt(Path("/tmp/preview.jpg"), context, False)
    assert "/tmp/preview.jpg" not in p2  # api/local send the image inline


# --- render_preview (RAW skip path, no Pillow needed) ----------------------


def test_render_preview_raw_without_embedded_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(images, "_extract_raw_preview", lambda img, out: None)
    assert images.render_preview(tmp_path / "x.cr3", tmp_path) is None


# --- process_one_image -----------------------------------------------------


def _opts() -> pipeline.ProcessOptions:
    return pipeline.ProcessOptions(
        backend="cli",
        vision_model_id="claude-haiku-4-5",
        local_base_url="",
        local_model=None,
        cost_per_call=0.0,
        no_whisper_prompt=True,
    )


def test_process_one_image_no_preview_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "x.cr3"
    img.write_bytes(b"x")
    monkeypatch.setattr(images, "get_image_metadata", lambda p: {"size_bytes": 1})
    monkeypatch.setattr(pipeline, "get_gps", lambda p: {})
    monkeypatch.setattr(images, "render_preview", lambda img, out: None)

    result = images.process_one_image(img, tmp_path, _opts(), pipeline.ProcessContext())
    assert result.skipped_reason == "no_preview"
    assert result.sidecar is None
    assert not pipeline.has_sidecar(img)


def test_process_one_image_writes_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "DSC.jpg"
    img.write_bytes(b"x")
    monkeypatch.setattr(
        images,
        "get_image_metadata",
        lambda p: {
            "size_bytes": 100,
            "creation_time": "2024-08-14T07:23:11",
            "dimensions": "6000x4000",
            "camera": {"model": "X-T5"},
        },
    )
    monkeypatch.setattr(pipeline, "get_gps", lambda p: {})
    monkeypatch.setattr(images, "render_preview", lambda img, out: out / "preview.jpg")
    monkeypatch.setattr("framedex.images.time.sleep", lambda s: None)
    vision = (
        "```yaml\nrating: keep\nscene_type: wildlife\nkeywords: [giraffe]\n```\n\n"
        "## Description\n\nA giraffe at golden hour.\n"
    )
    monkeypatch.setattr(pipeline, "describe_frames_cli", lambda *a, **k: vision)

    result = images.process_one_image(img, tmp_path, _opts(), pipeline.ProcessContext())
    assert result.skipped_reason is None
    assert result.rating == "keep"
    assert result.sidecar is not None and result.sidecar.exists()

    fm = _frontmatter(result.sidecar)
    assert fm["media_type"] == "image"
    assert fm["file"] == "DSC.jpg"
    assert fm["scene_type"] == "wildlife"
    assert fm["keywords"] == ["giraffe"]
    assert "duration_seconds" not in fm
    body = result.sidecar.read_text()
    assert "A giraffe at golden hour." in body


def test_process_one_image_vision_error_writes_no_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vision-backend failure sentinel must NOT persist a sidecar — otherwise
    the photo is marked indexed and silently skipped forever. It should be
    retryable (no sidecar) and reported as a vision_error."""
    img = tmp_path / "DSC.jpg"
    img.write_bytes(b"x")
    monkeypatch.setattr(images, "get_image_metadata", lambda p: {"size_bytes": 1})
    monkeypatch.setattr(pipeline, "get_gps", lambda p: {})
    monkeypatch.setattr(images, "render_preview", lambda img, out: out / "preview.jpg")
    monkeypatch.setattr("framedex.images.time.sleep", lambda s: None)
    monkeypatch.setattr(
        pipeline, "describe_frames_cli", lambda *a, **k: "[CLI timed out]"
    )

    result = images.process_one_image(img, tmp_path, _opts(), pipeline.ProcessContext())
    assert result.skipped_reason == "vision_error"
    assert result.sidecar is None
    assert not pipeline.has_sidecar(img)


def test_image_sidecar_is_queryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a photo sidecar parses back through query.parse_sidecar and
    a video-only --max-duration filter correctly excludes it."""
    from framedex import query

    img = tmp_path / "p.jpg"
    img.write_bytes(b"x")
    monkeypatch.setattr(
        images,
        "get_image_metadata",
        lambda p: {
            "size_bytes": 1,
            "creation_time": "",
            "dimensions": "100x100",
            "camera": {},
        },
    )
    monkeypatch.setattr(pipeline, "get_gps", lambda p: {})
    monkeypatch.setattr(images, "render_preview", lambda img, out: out / "preview.jpg")
    monkeypatch.setattr("framedex.images.time.sleep", lambda s: None)
    monkeypatch.setattr(
        pipeline,
        "describe_frames_cli",
        lambda *a, **k: "```yaml\nrating: keep\n```\n\n## Description\n\nx\n",
    )
    result = images.process_one_image(img, tmp_path, _opts(), pipeline.ProcessContext())
    assert result.sidecar is not None

    rec = query.parse_sidecar(result.sidecar)
    assert rec is not None
    assert rec["media_type"] == "image"
    # A photo has no duration_seconds; the video-only --max-duration must exclude it.
    import argparse

    base: dict[str, Any] = {
        k: None
        for k in (
            "rating",
            "media",
            "lighting",
            "time_of_day",
            "audio_quality",
            "language",
            "focus",
            "stability",
            "exposure",
            "people_count",
            "min_duration",
            "max_duration",
            "place_contains",
            "face_count",
            "person",
            "dominant_color",
        )
    }
    base["keyword"] = []
    base["has_speech"] = False
    assert (
        query.matches(rec, argparse.Namespace(**{**base, "max_duration": 60})) is False
    )
    assert query.matches(rec, argparse.Namespace(**base)) is True
