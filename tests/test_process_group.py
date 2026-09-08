"""Tests for framedex.images.process_group — one vision call per burst /
RAW+JPEG pair, stub sidecars for every other member, written before the
primary's sidecar.

Render, sharpness, exiftool, GPS, and the vision backend are mocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from framedex import grouping, images, pipeline

VISION_OK = (
    "```yaml\n"
    "rating: cull\n"
    "cull_reason: motion blur\n"
    "technical:\n  focus: soft\n  exposure: adequate\n  composition: strong\n"
    "lighting: golden_hour\n"
    "scene_type: wildlife\n"
    "people_count: 0\n"
    "keywords: [cheetah, sprint]\n"
    "```\n\n"
    "## Description\n\n**Scene:** A cheetah mid-sprint.\n"
)


def _frontmatter(sidecar: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", yaml.safe_load(sidecar.read_text().split("---", 2)[1])
    )


def _opts() -> pipeline.ProcessOptions:
    return pipeline.ProcessOptions(
        backend="cli",
        vision_model_id="claude-haiku-4-5",
        local_base_url="",
        local_model=None,
        cost_per_call=0.0,
        no_whisper_prompt=True,
    )


class _Geocoder:
    def __init__(self) -> None:
        self.calls = 0

    def reverse(self, lat: float, lon: float) -> str:
        self.calls += 1
        return "Mara, Kenya"


def _group_mocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scores: dict[str, float],
    vision: str = VISION_OK,
) -> tuple[list[str], list[Path]]:
    """render_preview creates a preview per unit subdir (None for files named
    nopreview*); sharpness is looked up by the unit's primary name; exiftool,
    GPS, and vision are canned. Returns (spy log, temp dirs handed out)."""
    log: list[str] = []
    made: list[Path] = []

    def render(src: Path, out_dir: Path) -> Path | None:
        if src.name.startswith("nopreview"):
            return None
        p = out_dir / "preview.jpg"
        p.write_bytes(b"p")
        log.append(f"render:{src.name}")
        return p

    def fake_mkdtemp(prefix: str = "") -> str:
        d = tmp_path / f"tmp{len(made)}"
        d.mkdir()
        made.append(d)
        return str(d)

    monkeypatch.setattr(images, "render_preview", render)
    monkeypatch.setattr("framedex.images.tempfile.mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(
        "framedex.frame_sampling.laplacian_sharpness",
        lambda p: scores[p.parent.name],
    )
    monkeypatch.setattr(
        images,
        "get_image_metadata",
        lambda p: {
            "size_bytes": 1,
            "creation_time": "2024-08-14T07:23:11",
            "dimensions": "10x10",
            "camera": {"model": "Z8"},
        },
    )
    monkeypatch.setattr(pipeline, "get_gps", lambda p: {"lat": -1.4, "lon": 35.0})
    monkeypatch.setattr("framedex.images.time.sleep", lambda s: None)
    monkeypatch.setattr(pipeline, "describe_frames_cli", lambda *a, **k: vision)
    real = pipeline.serialize_sidecar

    def spy(sidecar: Path, fm: dict[str, Any], title: str, body: Any) -> Path:
        log.append(f"sidecar:{sidecar.name}")
        return real(sidecar, fm, title, body)

    monkeypatch.setattr(pipeline, "serialize_sidecar", spy)
    return log, made


def _files(tmp_path: Path, *names: str) -> list[Path]:
    out = [tmp_path / n for n in names]
    for f in out:
        f.write_bytes(b"x")
    return out


def test_process_group_writes_stubs_then_primary_with_group_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(tmp_path, "1.NEF", "2.NEF", "3.NEF")
    log, made = _group_mocks(
        tmp_path, monkeypatch, {"1.NEF": 1.0, "2.NEF": 9.0, "3.NEF": 5.0}
    )
    grp = grouping.MediaGroup("burst", "b_abc", [grouping.Unit(f) for f in files])
    geocoder = _Geocoder()
    ctx = pipeline.ProcessContext(geocoder=cast(Any, geocoder))

    res = images.process_group(grp, tmp_path, _opts(), ctx)

    assert res.sidecar == pipeline.sidecar_path(tmp_path / "2.NEF")
    assert res.stubs_written == 2
    assert res.rating == "cull"
    # Stubs first, primary last (the primary sidecar is the group's resume marker).
    assert [e for e in log if e.startswith("sidecar:")] == [
        "sidecar:1.NEF.description.md",
        "sidecar:3.NEF.description.md",
        "sidecar:2.NEF.description.md",
    ]
    # Every member rendered exactly once; the pick's preview is reused.
    assert sorted(e for e in log if e.startswith("render:")) == [
        "render:1.NEF",
        "render:2.NEF",
        "render:3.NEF",
    ]
    assert geocoder.calls == 1  # stubs copy the place; no per-stub geocode

    primary = _frontmatter(res.sidecar)
    assert primary["group"] == {
        "kind": "burst",
        "id": "b_abc",
        "primary": True,
        "members": ["1.NEF", "2.NEF", "3.NEF"],
        "sharpness": 9.0,
    }
    stub = _frontmatter(pipeline.sidecar_path(tmp_path / "1.NEF"))
    assert stub["group"] == {
        "kind": "burst",
        "id": "b_abc",
        "primary": False,
        "primary_file": "2.NEF",
        "members": ["1.NEF", "2.NEF", "3.NEF"],
        "sharpness": 1.0,
    }
    for k in images.STUB_COPIED_FIELDS:
        assert stub[k] == primary[k], k
    assert stub["rating"] == "cull" and stub["keywords"] == ["cheetah", "sprint"]
    assert stub["faces"] == [] and stub["face_count"] == 0  # no detection ran here
    assert stub["file"] == "1.NEF" and stub["path"] == "1.NEF"
    assert stub["media_type"] == "image" and stub["camera"] == {"model": "Z8"}
    assert stub["location"] == {"lat": -1.4, "lon": 35.0, "place": "Mara, Kenya"}
    assert (
        "See 2.NEF.description.md (burst primary)."
        in pipeline.sidecar_path(tmp_path / "1.NEF").read_text()
    )
    assert not made[0].exists()  # the group's scoring temp dir is cleaned up


def test_process_group_pair_uses_jpeg_preview_and_raw_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, jpg = _files(tmp_path, "1.NEF", "1.jpg")
    log, _ = _group_mocks(tmp_path, monkeypatch, {"1.NEF": 3.0})
    grp = grouping.MediaGroup("raw_jpeg", "b_p", [grouping.Unit(raw, jpg)])

    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())

    assert res.sidecar == pipeline.sidecar_path(raw)
    assert res.stubs_written == 1
    assert "render:1.jpg" in log and "render:1.NEF" not in log
    stub = _frontmatter(pipeline.sidecar_path(jpg))
    assert stub["group"]["kind"] == "raw_jpeg"
    assert stub["group"]["primary_file"] == "1.NEF"
    assert stub["group"]["sharpness"] == 3.0  # the JPEG was scored as the RAW's preview
    assert "(raw_jpeg primary)" in pipeline.sidecar_path(jpg).read_text()


def test_process_group_no_renderable_member_skips_without_stubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(tmp_path, "nopreview1.NEF", "nopreview2.NEF", "nopreview3.NEF")
    _group_mocks(tmp_path, monkeypatch, {})
    grp = grouping.MediaGroup("burst", "b_n", [grouping.Unit(f) for f in files])

    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())

    assert res.skipped_reason == "no_preview" and res.sidecar is None
    assert not any(pipeline.has_sidecar(f) for f in files)


def test_process_group_unrenderable_member_is_never_the_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(tmp_path, "nopreview1.NEF", "2.NEF")
    _group_mocks(tmp_path, monkeypatch, {"2.NEF": 0.0})
    grp = grouping.MediaGroup("burst", "b_x", [grouping.Unit(f) for f in files])

    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())

    assert res.sidecar == pipeline.sidecar_path(tmp_path / "2.NEF")
    assert _frontmatter(pipeline.sidecar_path(files[0]))["group"]["sharpness"] == -1.0


def test_process_group_vision_error_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(tmp_path, "1.NEF", "2.NEF", "3.NEF")
    _group_mocks(
        tmp_path,
        monkeypatch,
        {"1.NEF": 1.0, "2.NEF": 2.0, "3.NEF": 3.0},
        vision="[CLI timed out]",
    )
    grp = grouping.MediaGroup("burst", "b_v", [grouping.Unit(f) for f in files])

    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())

    assert res.skipped_reason == "vision_error"
    assert res.stubs_written == 0
    assert not any(pipeline.has_sidecar(f) for f in files)
