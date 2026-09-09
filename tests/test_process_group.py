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

_REAL_GET_IMAGE_METADATA = images.get_image_metadata

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
    """render_preview creates a preview per subdir (None for files named
    nopreview*); sharpness is looked up by the rendered source's name;
    exiftool, GPS, and vision are canned. Returns (spy log, temp dirs)."""
    log: list[str] = []
    made: list[Path] = []

    def render(src: Path, out_dir: Path) -> Path | None:
        if src.name.startswith("nopreview"):
            return None
        if src.name.startswith("flaky") and any(e == f"render:{src.name}" for e in log):
            return None  # renders once (scoring), fails the second time
        if src.suffix.upper() == ".NEF":  # a RAW leaves its full-size extract behind
            (out_dir / "raw_preview.jpg").write_bytes(b"full")
        p = out_dir / "preview.jpg"
        p.write_bytes(b"p")
        log.append(f"render:{src.name}")
        rendered[p] = src.name
        return p

    def fake_mkdtemp(prefix: str = "") -> str:
        d = tmp_path / f"tmp{len(made)}"
        d.mkdir()
        made.append(d)
        return str(d)

    monkeypatch.setattr(images, "render_preview", render)
    monkeypatch.setattr("framedex.images.tempfile.mkdtemp", fake_mkdtemp)
    rendered: dict[Path, str] = {}  # preview path -> source file name

    def sharpness(p: Path) -> float:
        return scores[rendered[p]]

    monkeypatch.setattr("framedex.frame_sampling.laplacian_sharpness", sharpness)
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
    # Every member rendered once for scoring (nothing kept), then the primary
    # alone is rendered again for the vision call.
    assert sorted(e for e in log if e.startswith("render:")) == [
        "render:1.NEF",
        "render:2.NEF",
        "render:2.NEF",
        "render:3.NEF",
    ]
    assert geocoder.calls == 3  # each member resolves its own coordinates

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
        "primary_file": "2.NEF",  # the member list lives on the primary only
        "sharpness": 1.0,
    }
    for k in images.STUB_COPIED_FIELDS:
        assert stub[k] == primary[k], k
    assert stub["rating"] == "cull" and stub["keywords"] == ["cheetah", "sprint"]
    assert stub["faces"] == [] and stub["face_count"] == 0  # no detection ran here
    assert stub["file"] == "1.NEF" and stub["path"] == "1.NEF"
    assert stub["media_type"] == "image" and stub["camera"] == {"model": "Z8"}
    assert stub["location"] == {"lat": -1.4, "lon": 35.0, "place": "Mara, Kenya"}
    body = pipeline.sidecar_path(tmp_path / "1.NEF").read_text()
    assert "This file was not assessed individually." in body
    assert "copied from 2.NEF.description.md (the burst primary)" in body
    assert "zero does not mean none are present" in body
    assert not made[0].exists()  # the group's scoring temp dir is cleaned up


def test_process_group_pair_uses_jpeg_preview_and_raw_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, jpg = _files(tmp_path, "1.NEF", "1.jpg")
    log, _ = _group_mocks(tmp_path, monkeypatch, {"1.jpg": 3.0})  # the JPEG is scored
    grp = grouping.MediaGroup("raw_jpeg", "b_p", [grouping.Unit(raw, jpg)])

    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())

    assert res.sidecar == pipeline.sidecar_path(raw)
    assert res.stubs_written == 1
    assert "render:1.jpg" in log and "render:1.NEF" not in log
    stub = _frontmatter(pipeline.sidecar_path(jpg))
    assert stub["group"]["kind"] == "raw_jpeg"
    assert stub["group"]["primary_file"] == "1.NEF"
    assert stub["group"]["sharpness"] == 3.0  # the JPEG was scored as the RAW's preview
    assert (
        "the RAW primary of this RAW+JPEG pair"
        in pipeline.sidecar_path(jpg).read_text()
    )


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


def test_process_group_one_transport_call_for_a_five_frame_burst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(tmp_path, *(f"{i}.NEF" for i in range(1, 6)))
    _group_mocks(tmp_path, monkeypatch, {f.name: float(i) for i, f in enumerate(files)})
    calls: list[int] = []

    def vision(frames: list[Path], prompt: str, model: str) -> str:
        calls.append(len(frames))
        return VISION_OK

    monkeypatch.setattr(pipeline, "describe_frames_cli", vision)
    grp = grouping.MediaGroup("burst", "b_5", [grouping.Unit(f) for f in files])
    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert calls == [1]  # one call, one frame, whatever the burst length
    assert res.stubs_written == 4


def test_process_group_keeps_no_member_previews_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long chain (a timelapse folder) must not pile up every member's
    rendered preview: scoring renders are deleted at once, and only the
    primary's second render exists during the vision call."""
    files = _files(tmp_path, "1.NEF", "2.NEF", "3.NEF", "4.NEF")
    _, made = _group_mocks(
        tmp_path, monkeypatch, {"1.NEF": 1.0, "2.NEF": 9.0, "3.NEF": 5.0, "4.NEF": 2.0}
    )

    def vision(frames: list[Path], prompt: str, model: str) -> str:
        assert [d.name for d in made[0].iterdir()] == ["primary"]
        assert frames == [made[0] / "primary" / "preview.jpg"]
        # The full-size embedded JPEG the RAW yielded is gone by now.
        assert sorted(p.name for p in (made[0] / "primary").iterdir()) == [
            "preview.jpg"
        ]
        return VISION_OK

    monkeypatch.setattr(pipeline, "describe_frames_cli", vision)
    grp = grouping.MediaGroup("burst", "b_l", [grouping.Unit(f) for f in files])
    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert res.sidecar == pipeline.sidecar_path(tmp_path / "2.NEF")


def test_process_group_member_exif_failure_fails_before_the_paid_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real exiftool reader, with the subprocess failing on one member:
    the group fails loudly before any vision call and writes nothing (blank
    EXIF stubs would otherwise overwrite valid metadata on a regroup)."""
    import types

    files = _files(tmp_path, "1.NEF", "2.NEF", "3.NEF")
    _group_mocks(tmp_path, monkeypatch, {"1.NEF": 1.0, "2.NEF": 9.0, "3.NEF": 5.0})
    monkeypatch.setattr(images, "get_image_metadata", _REAL_GET_IMAGE_METADATA)

    def run(cmd: list[str], **kw: Any) -> Any:
        if cmd[-1].endswith("3.NEF"):
            return types.SimpleNamespace(
                returncode=1, stdout="", stderr="Error: bad file"
            )
        return types.SimpleNamespace(
            returncode=0, stdout='[{"Model": "Z8"}]', stderr=""
        )

    monkeypatch.setattr("framedex.images.subprocess.run", run)
    monkeypatch.setattr(
        pipeline, "describe_frames_cli", lambda *a, **k: pytest.fail("paid call made")
    )
    grp = grouping.MediaGroup("burst", "b_m", [grouping.Unit(f) for f in files])
    with pytest.raises(RuntimeError, match=r"exiftool failed on 3\.NEF"):
        images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert not any(pipeline.has_sidecar(f) for f in files)


def test_process_group_marks_old_primary_sidecar_incomplete_before_stubs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force / regrouping a legacy folder: the primary's OLD sidecar exists.
    If the run dies after the stubs and before the new primary sidecar, the
    group must not read as done (every member "has a sidecar") — but the old
    sidecar's content, including keys a human wrote, must survive: it is
    marked incomplete, not deleted."""
    files = _files(tmp_path, "1.NEF", "2.NEF", "3.NEF")
    for f in files:  # legacy per-file sidecars, no group block
        pipeline.sidecar_path(f).write_text(
            "---\nfile: x\nrating: keep\nuser_rating: cull\n---\n\n# x\n\nold body\n"
        )
    _group_mocks(tmp_path, monkeypatch, {"1.NEF": 1.0, "2.NEF": 9.0, "3.NEF": 5.0})
    real = pipeline.serialize_sidecar

    def die_on_primary(
        sidecar: Path, fm: dict[str, Any], title: str, body: Any
    ) -> Path:
        if sidecar.name.startswith("2.NEF"):
            raise RuntimeError("power cut")
        return real(sidecar, fm, title, body)

    monkeypatch.setattr(pipeline, "serialize_sidecar", die_on_primary)
    grp = grouping.MediaGroup(
        "burst", grouping.group_id(files), [grouping.Unit(f) for f in files]
    )
    with pytest.raises(RuntimeError, match="power cut"):
        images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    old = pipeline.sidecar_path(tmp_path / "2.NEF")
    kept = pipeline.read_sidecar_frontmatter(old)
    assert kept is not None
    assert kept["user_rating"] == "cull" and kept["rating"] == "keep"  # nothing lost
    assert kept["group"]["incomplete"] is True and kept["group"]["id"] == grp.id
    assert "old body" in old.read_text()
    assert pipeline.has_sidecar(tmp_path / "1.NEF")  # stubs written
    # …and the next run sees the group as incomplete.
    assert not grouping.group_is_done(grp, pipeline.read_sidecar_frontmatter)

    # A clean re-run replaces the marked sidecar; the marker is gone.
    monkeypatch.setattr(pipeline, "serialize_sidecar", real)
    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert res.sidecar == old
    fresh = pipeline.read_sidecar_frontmatter(old)
    assert fresh is not None and "incomplete" not in fresh["group"]
    assert grouping.group_is_done(grp, pipeline.read_sidecar_frontmatter)


def test_process_group_clears_stale_face_rows_of_demoted_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regrouping a folder indexed per-file earlier: members that become stubs
    must not keep face rows in faces.db (their assessment lives on the
    primary now); the primary's own detections are written as usual."""
    from framedex import face_db

    files = _files(tmp_path, "1.NEF", "2.NEF", "3.NEF")
    _group_mocks(tmp_path, monkeypatch, {"1.NEF": 1.0, "2.NEF": 9.0, "3.NEF": 5.0})
    conn = face_db.open_db(tmp_path / "faces.db")
    face = face_db.DetectedFace(
        cluster_id="tmp_old",
        frame_time_seconds=0.0,
        bbox=[0, 0, 10, 10],
        detection_score=0.9,
        embedding=[0.1] * 512,
    )
    for f in files:  # rows from an earlier per-file index
        face_db.write_faces(conn, f, pipeline.sidecar_path(f), [face])
    assert face_db.db_stats(conn)["faces"] == 3
    monkeypatch.setattr(face_db, "detect_faces_in_frames", lambda fr, t: [face])

    grp = grouping.MediaGroup("burst", "b_f", [grouping.Unit(f) for f in files])
    res = images.process_group(
        grp, tmp_path, _opts(), pipeline.ProcessContext(face_conn=conn)
    )
    assert res.sidecar == pipeline.sidecar_path(tmp_path / "2.NEF")

    def rows(p: Path) -> int:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM faces WHERE video_path = ?", (str(p),)
            ).fetchone()[0]
        )

    assert rows(tmp_path / "1.NEF") == 0 and rows(tmp_path / "3.NEF") == 0
    assert rows(tmp_path / "2.NEF") == 1


def test_process_group_removes_unparsable_old_primary_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _files(tmp_path, "1.NEF", "2.NEF", "3.NEF")
    pipeline.sidecar_path(tmp_path / "2.NEF").write_text("garbage, no fence")
    _group_mocks(tmp_path, monkeypatch, {"1.NEF": 1.0, "2.NEF": 9.0, "3.NEF": 5.0})
    grp = grouping.MediaGroup("burst", "b_g", [grouping.Unit(f) for f in files])
    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert res.sidecar is not None
    fm = pipeline.read_sidecar_frontmatter(res.sidecar)
    assert fm is not None and fm["group"]["primary"] is True


def test_process_group_second_render_failure_is_an_error_not_a_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring proved the primary renders; if the vision-call render then
    fails, that is a real error to retry, never a silent "no preview" skip."""
    files = _files(tmp_path, "flaky1.NEF", "flaky2.NEF", "flaky3.NEF")
    _group_mocks(
        tmp_path, monkeypatch, {"flaky1.NEF": 1.0, "flaky2.NEF": 9.0, "flaky3.NEF": 5.0}
    )
    grp = grouping.MediaGroup("burst", "b_f2", [grouping.Unit(f) for f in files])
    with pytest.raises(RuntimeError, match="rendered for scoring but not"):
        images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert not any(pipeline.has_sidecar(f) for f in files)


def test_group_with_triple_hyphen_member_resumes_as_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the real serializer and reader: a member named
    a---b.NEF must not break the resume check after a completed run."""
    files = _files(tmp_path, "a---b.NEF", "c.NEF", "d.NEF")
    _group_mocks(tmp_path, monkeypatch, {"a---b.NEF": 9.0, "c.NEF": 1.0, "d.NEF": 2.0})
    grp = grouping.MediaGroup(
        "burst", grouping.group_id(files), [grouping.Unit(f) for f in files]
    )
    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert res.sidecar == pipeline.sidecar_path(tmp_path / "a---b.NEF")
    assert grouping.group_is_done(grp, pipeline.read_sidecar_frontmatter)
