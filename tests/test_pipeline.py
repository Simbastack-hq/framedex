"""Tests for framedex.pipeline — the media-agnostic shared core."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from framedex import pipeline


def _frontmatter(sidecar: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", yaml.safe_load(sidecar.read_text().split("---", 2)[1])
    )


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
