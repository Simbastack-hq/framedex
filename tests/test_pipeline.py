"""Tests for framedex.pipeline — the media-agnostic shared core."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from framedex import pipeline


def _frontmatter(sidecar: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", yaml.safe_load(sidecar.read_text().split("---", 2)[1])
    )


# --- atomic_write_text -----------------------------------------------------


def test_atomic_write_text_writes_content_and_cleans_temp(tmp_path: Path) -> None:
    target = tmp_path / "out.md"
    pipeline.atomic_write_text(target, "hello\n")
    assert target.read_text() == "hello\n"
    # no leftover .out.md.<pid>.tmp files on the happy path
    assert list(tmp_path.glob(".out.md.*.tmp")) == []


def test_atomic_write_text_never_leaves_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash/disk-full during os.replace must leave the *target* untouched —
    readers and the resume check never observe a partial file (principle 1)."""
    target = tmp_path / "out.md"
    target.write_text("OLD\n")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        pipeline.atomic_write_text(target, "NEW CONTENT\n")
    # target still holds the old content, not a truncated new write
    assert target.read_text() == "OLD\n"


def test_atomic_write_text_temp_is_hidden_and_pid_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The temp name is dot-prefixed (skipped by discovery) and pid-suffixed
    (two concurrent runs can't collide on one temp path)."""
    target = tmp_path / "clip.MOV.description.md"
    captured: dict[str, str] = {}
    real_replace = os.replace

    def capture(src: Any, dst: Any) -> None:
        captured["tmp"] = Path(src).name
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", capture)
    pipeline.atomic_write_text(target, "x\n")
    assert captured["tmp"].startswith(".clip.MOV.description.md.")
    assert captured["tmp"].endswith(".tmp")
    assert str(os.getpid()) in captured["tmp"]


# --- sidecar paths ---------------------------------------------------------


def test_sidecar_path_appends_full_suffix() -> None:
    assert pipeline.sidecar_path(Path("a/clip.MOV")) == Path(
        "a/clip.MOV.description.md"
    )


def test_sidecar_path_collision_safe_across_extensions() -> None:
    """Two files sharing a stem but differing in extension must NOT collide on a
    single sidecar — the full original extension is part of the name."""
    raw = pipeline.sidecar_path(Path("shoot/DSC_1.RAF"))
    jpg = pipeline.sidecar_path(Path("shoot/DSC_1.JPG"))
    assert raw != jpg
    assert raw.name == "DSC_1.RAF.description.md"
    assert jpg.name == "DSC_1.JPG.description.md"


def test_has_sidecar(tmp_path: Path) -> None:
    media = tmp_path / "x.jpg"
    media.write_bytes(b"x")
    assert pipeline.has_sidecar(media) is False
    pipeline.sidecar_path(media).write_text("---\n---\n")
    assert pipeline.has_sidecar(media) is True


# --- serialize_sidecar -----------------------------------------------------


def test_serialize_sidecar_structure(tmp_path: Path) -> None:
    out = tmp_path / "x.jpg.description.md"
    fm: dict[str, Any] = {"file": "x.jpg", "rating": "keep"}
    pipeline.serialize_sidecar(
        out, fm, "x.jpg", [("Description", "A scene."), ("Notes", "extra")]
    )
    text = out.read_text()
    assert text.startswith("---\n")
    assert "# x.jpg" in text
    assert "## Description\n\nA scene." in text
    assert "## Notes\n\nextra" in text
    parsed = _frontmatter(out)
    assert parsed["file"] == "x.jpg"
    assert parsed["rating"] == "keep"


def test_serialize_sidecar_stamps_indexed_at_last(tmp_path: Path) -> None:
    out = tmp_path / "x.jpg.description.md"
    pipeline.serialize_sidecar(out, {"file": "x.jpg"}, "x.jpg", [("Description", "d")])
    parsed = _frontmatter(out)
    assert "indexed_at" in parsed
    # indexed_at is stamped as the final frontmatter key
    assert list(parsed.keys())[-1] == "indexed_at"


# --- parse_vision_response -------------------------------------------------


def test_parse_vision_response_extracts_yaml_and_prose() -> None:
    raw = (
        "```yaml\nrating: keep\nkeywords: [a, b]\n```\n\n"
        "## Description\n\nA giraffe at a waterhole.\n"
    )
    structured, prose = pipeline.parse_vision_response(raw)
    assert structured["rating"] == "keep"
    assert structured["keywords"] == ["a", "b"]
    assert prose == "A giraffe at a waterhole."


def test_parse_vision_response_error_sentinel_passes_through() -> None:
    # describe_frames_* return "[...]" on failure; parser must not eat it.
    structured, prose = pipeline.parse_vision_response("[CLI timed out]")
    assert structured == {}
    assert prose == "[CLI timed out]"


# --- describe_frames_cli transport lockdown --------------------------------


def test_describe_frames_cli_denies_untrusted_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prompt embeds untrusted transcript/context text. The CLI must run
    with `--permission-mode dontAsk` (deny anything not allowlisted) and a
    read-only `Read` allowlist — never bypassPermissions, which auto-approves
    Bash. Path-scoping Read(<dir>/**) is not honored by the CLI (verified live:
    it denies all reads), so the allowlist is unscoped read-only `Read`."""
    frame = tmp_path / "frame0.jpg"
    frame.write_bytes(b"x")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        import types

        captured["cmd"] = cmd
        return types.SimpleNamespace(
            returncode=0, stdout='{"result": "```yaml\\nrating: keep\\n```"}', stderr=""
        )

    monkeypatch.setattr("framedex.pipeline.subprocess.run", fake_run)
    pipeline.describe_frames_cli([frame], "a prompt", "claude-haiku-4-5")

    cmd = captured["cmd"]
    assert "bypassPermissions" not in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    allowed = cmd[cmd.index("--allowedTools") + 1]
    # read-only: Read granted, no execute/write capability handed out
    assert allowed == "Read"
    for banned in ("Bash", "Write", "Edit"):
        assert banned not in allowed
