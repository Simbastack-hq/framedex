"""Tests for framedex.index_videos sidecar writing.

The video runtime stack (whisperx/torch) is imported lazily now, so
index_videos imports without it. These tests pin the video sidecar's
frontmatter/body output, which the pipeline.py extraction must keep
byte-for-byte identical.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from framedex import face_db, frame_sampling, images, index_videos, pipeline
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


def _opts() -> pipeline.ProcessOptions:
    return pipeline.ProcessOptions(
        backend="cli",
        vision_model_id="claude-haiku-4-5",
        local_base_url="",
        local_model=None,
        cost_per_call=0.0,
        no_whisper_prompt=True,
    )


def _std_video_mocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Mock the whole per-clip pipeline so process_one_video reaches its persist
    section without ffmpeg/whisper/network. Returns the video path."""
    video = tmp_path / "clip.mov"
    video.write_bytes(b"x")
    monkeypatch.setattr(index_videos, "get_metadata", lambda v: dict(METADATA))
    monkeypatch.setattr(index_videos, "get_gps", lambda v: {})
    monkeypatch.setattr(index_videos, "transcribe_audio_whisperx", lambda *a, **k: {})
    monkeypatch.setattr(
        index_videos, "extract_frames", lambda *a, **k: [(tmp_path / "f0.jpg", 0.0)]
    )
    monkeypatch.setattr(index_videos, "load_context_for_clip", lambda *a, **k: "")
    monkeypatch.setattr("framedex.index_videos.time.sleep", lambda s: None)
    monkeypatch.setattr(
        index_videos,
        "describe_frames_cli",
        lambda *a, **k: "```yaml\nrating: keep\n```\n\n## Description\n\nx\n",
    )
    return video


def _face(cluster_id: str = "tmp_vid1") -> Any:
    return face_db.DetectedFace(
        cluster_id=cluster_id,
        frame_time_seconds=0.0,
        bbox=[0, 0, 10, 10],
        detection_score=0.9,
        embedding=[0.1] * 512,
    )


def test_process_one_video_writes_faces_before_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecar is the resume marker — written last, after faces (principle 1)."""
    from framedex import face_db

    video = _std_video_mocks(tmp_path, monkeypatch)
    conn = face_db.open_db(tmp_path / "faces.db")
    monkeypatch.setattr(face_db, "detect_faces_in_frames", lambda f, t: [_face()])

    order: list[str] = []
    real_write_faces = face_db.write_faces
    real_write_sidecar = index_videos.write_sidecar

    def spy_faces(*a: Any, **k: Any) -> Any:
        order.append("faces")
        return real_write_faces(*a, **k)

    def spy_sidecar(*a: Any, **k: Any) -> Any:
        order.append("sidecar")
        return real_write_sidecar(*a, **k)

    monkeypatch.setattr(face_db, "write_faces", spy_faces)
    monkeypatch.setattr(index_videos, "write_sidecar", spy_sidecar)

    index_videos.process_one_video(
        video, tmp_path, _opts(), pipeline.ProcessContext(face_conn=conn)
    )
    assert order == ["faces", "sidecar"]


def test_process_one_video_zero_face_rerun_clears_stale_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run detecting zero faces must clear the clip's prior face rows."""
    from framedex import face_db

    video = _std_video_mocks(tmp_path, monkeypatch)
    conn = face_db.open_db(tmp_path / "faces.db")
    ctx = pipeline.ProcessContext(face_conn=conn)

    monkeypatch.setattr(face_db, "detect_faces_in_frames", lambda f, t: [_face()])
    index_videos.process_one_video(video, tmp_path, _opts(), ctx)
    assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 1

    pipeline.sidecar_path(video).unlink()
    monkeypatch.setattr(face_db, "detect_faces_in_frames", lambda f, t: [])
    index_videos.process_one_video(video, tmp_path, _opts(), ctx)
    assert conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0] == 0


def test_process_one_video_no_yaml_fence_writes_no_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prose with no parseable YAML fence must not persist a defaults-only
    sidecar that permanently skips the clip; it retries as a vision_error."""
    video = _std_video_mocks(tmp_path, monkeypatch)
    monkeypatch.setattr(
        index_videos, "describe_frames_cli", lambda *a, **k: "Just some prose."
    )
    result = index_videos.process_one_video(
        video, tmp_path, _opts(), pipeline.ProcessContext()
    )
    assert result.skipped_reason == "vision_error"
    assert result.sidecar is None
    assert not pipeline.has_sidecar(video)


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
    # Backend wiring (incl. the claude-CLI check) now lives in runner.
    monkeypatch.setattr("framedex.runner.check_claude_cli", lambda: True)
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


# ---------------------------------------------------------------------------
# extract_frames — (path, timestamp) pairs + sampling modes
# ---------------------------------------------------------------------------


class _FakeRun:
    """subprocess.run stand-in: records ffmpeg commands, creates the output
    file (last arg) so extract_frames sees a successful frame write."""

    def __init__(self, fail_outputs: set[str] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.fail_outputs = fail_outputs or set()

    def __call__(self, cmd: list[str], **kwargs: Any) -> Any:
        self.commands.append(cmd)
        out = Path(cmd[-1])
        if out.name not in self.fail_outputs:
            out.write_bytes(b"jpeg")

        class R:
            returncode = 0

        return R()


def test_extract_frames_even_returns_legacy_timestamp_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=12.0, sampling="even"
    )
    assert [ts for _, ts in pairs] == [12.0 * (i + 1) / 6 for i in range(5)]
    assert all(p.exists() for p, _ in pairs)
    # 5 full-res seeks, no thumbnail pool.
    assert len(fake.commands) == 5


def test_extract_frames_failed_write_keeps_pairs_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the timestamp desync bug: a failed middle frame must
    drop its own timestamp, not shift the others (faces.db frame_time)."""
    fake = _FakeRun(fail_outputs={"frame_02.jpg"})
    monkeypatch.setattr(subprocess, "run", fake)
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=12.0, sampling="even"
    )
    even = [12.0 * (i + 1) / 6 for i in range(5)]
    assert [ts for _, ts in pairs] == even[:2] + even[3:]


def test_extract_frames_short_clip_skips_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    called: list[int] = []
    monkeypatch.setattr(
        frame_sampling,
        "choose_frame_timestamps",
        lambda *a, **k: called.append(1),
    )
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=10.0, sampling="diverse"
    )
    assert not called  # under SHORT_CLIP_EVEN_CUTOFF -> no pool, no selection
    assert [ts for _, ts in pairs] == [10.0 * (i + 1) / 6 for i in range(5)]


def test_extract_frames_diverse_uses_chosen_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(
        frame_sampling,
        "choose_frame_timestamps",
        lambda paths, ts, n: [3.0, 17.0, 31.0, 44.0, 58.0],
    )
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=60.0, sampling="diverse"
    )
    assert [ts for _, ts in pairs] == [3.0, 17.0, 31.0, 44.0, 58.0]
    # Pool thumbnails (30 for a 60s clip) + 5 full-res frames.
    assert len(fake.commands) == 30 + 5
    # Thumbnails were cleaned up after selection.
    assert not list(tmp_path.glob("cand_*.jpg"))


def test_extract_frames_diverse_falls_back_when_selection_declines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(
        frame_sampling,
        "choose_frame_timestamps",
        lambda paths, ts, n: None,
    )
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=60.0, sampling="diverse"
    )
    assert [ts for _, ts in pairs] == [60.0 * (i + 1) / 6 for i in range(5)]


def test_short_clip_three_frame_clamp_still_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(subprocess, "run", fake)
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=2.0, sampling="diverse"
    )
    assert len(pairs) == 3


def test_vision_prompt_includes_frame_timestamps() -> None:
    ctx = {
        "filename": "clip.mov",
        "parent_folder": "drone",
        "duration_seconds": 60.0,
        "transcript": "",
    }
    prompt = index_videos._build_vision_prompt(
        [Path("f0.jpg"), Path("f1.jpg")],
        ctx,
        "",
        include_paths=False,
        timestamps=[3.0, 41.0],
    )
    assert "00:00:03" in prompt and "00:00:41" in prompt
    assert "evenly sampled" not in prompt


def test_vision_prompt_without_timestamps_keeps_legacy_wording() -> None:
    ctx = {
        "filename": "clip.mov",
        "parent_folder": "drone",
        "duration_seconds": 60.0,
        "transcript": "",
    }
    prompt = index_videos._build_vision_prompt(
        [Path("f0.jpg")], ctx, "", include_paths=False
    )
    assert "evenly sampled across the clip" in prompt
