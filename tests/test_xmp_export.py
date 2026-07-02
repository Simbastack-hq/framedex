"""Tests for framedex.xmp_export — the fdx-xmp Lightroom sidecar exporter.

Hermetic: pure functions over dicts + tmp trees. No Lightroom, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from framedex import xmp_export

# --- read_sidecar_for_xmp --------------------------------------------------


SIDECAR = """\
---
file: DSC_1.RAF
path: shoot/DSC_1.RAF
media_type: image
rating: keep
scene_type: wildlife
keywords: [lion, golden-hour]
---

# DSC_1.RAF

## Description

**Scene:** Two lions at a kill at dawn.
**Subjects:** Two adult lions feeding.
**Composition:** Low-angle telephoto.
"""


def test_read_sidecar_extracts_frontmatter_and_scene(tmp_path: Path) -> None:
    p = tmp_path / "DSC_1.RAF.description.md"
    p.write_text(SIDECAR)
    result = xmp_export.read_sidecar_for_xmp(p)
    assert result is not None
    fm, scene = result
    assert fm["rating"] == "keep"
    assert fm["keywords"] == ["lion", "golden-hour"]
    assert fm["scene_type"] == "wildlife"
    # only the Scene sentence, not Subjects/Composition
    assert scene == "Two lions at a kill at dawn."


def test_read_sidecar_none_on_non_sidecar(tmp_path: Path) -> None:
    p = tmp_path / "plain.md"
    p.write_text("no frontmatter here\n")
    assert xmp_export.read_sidecar_for_xmp(p) is None


def test_read_sidecar_missing_scene_is_empty(tmp_path: Path) -> None:
    p = tmp_path / "x.RAF.description.md"
    p.write_text("---\nrating: keep\n---\n\n## Description\n\nno bold scene line\n")
    result = xmp_export.read_sidecar_for_xmp(p)
    assert result is not None
    _, scene = result
    assert scene == ""


# --- build_xmp -------------------------------------------------------------


def _fm(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "rating": "keep",
        "scene_type": "wildlife",
        "keywords": ["lion", "golden-hour"],
    }
    base.update(over)
    return base


def test_build_xmp_rating_and_label_by_rating() -> None:
    keep = xmp_export.build_xmp(_fm(rating="keep"), "A scene.")
    review = xmp_export.build_xmp(_fm(rating="review"), "A scene.")
    cull = xmp_export.build_xmp(_fm(rating="cull"), "A scene.")
    assert 'xmp:Rating="3"' in keep
    assert 'xmp:Rating="2"' in review
    assert 'xmp:Rating="1"' in cull
    # only cull gets a color label
    assert 'xmp:Label="Red"' in cull
    assert "xmp:Label" not in keep
    assert "xmp:Label" not in review


def test_build_xmp_subject_bag_includes_keywords_and_scene_type() -> None:
    xmp = xmp_export.build_xmp(_fm(), "A scene.")
    assert "<rdf:li>lion</rdf:li>" in xmp
    assert "<rdf:li>golden-hour</rdf:li>" in xmp
    assert "<rdf:li>wildlife</rdf:li>" in xmp  # scene_type joins the bag


def test_build_xmp_drops_unclear_scene_type_and_empty_keywords() -> None:
    xmp = xmp_export.build_xmp(_fm(keywords=[], scene_type="unclear"), "")
    assert "dc:subject" not in xmp  # no tags at all
    assert "dc:description" not in xmp  # empty scene → no description block


def test_build_xmp_escapes_xml_special_chars() -> None:
    xmp = xmp_export.build_xmp(_fm(keywords=["a&b", "x<y"]), "Cats & dogs <together>.")
    assert "a&amp;b" in xmp
    assert "x&lt;y" in xmp
    assert "Cats &amp; dogs &lt;together&gt;." in xmp
    assert "a&b" not in xmp  # raw ampersand must not leak


def test_build_xmp_carries_creator_tool() -> None:
    xmp = xmp_export.build_xmp(_fm(), "A scene.")
    assert f'xmp:CreatorTool="{xmp_export.CREATOR_TOOL}"' in xmp
    assert xmp_export.CREATOR_TOOL.startswith("framedex ")


def test_build_xmp_is_valid_parseable_xml() -> None:
    import xml.dom.minidom as minidom

    xmp = xmp_export.build_xmp(_fm(), "Two lions & a cub <at dawn>.")
    minidom.parseString(xmp)  # raises if not well-formed


def test_build_xmp_strips_xml_forbidden_control_chars() -> None:
    """Keywords/scene come from an LLM (local models included). A stray C0
    control char would make the XMP not-well-formed and Lightroom would silently
    reject it — so those chars are dropped and the document is always parseable."""
    import xml.dom.minidom as minidom

    xmp = xmp_export.build_xmp(
        _fm(keywords=["lion\x01cub", "gold\x0chour"]), "Dawn\x1blight over the plain."
    )
    minidom.parseString(xmp)  # must be well-formed
    for bad in ("\x01", "\x0c", "\x1b"):
        assert bad not in xmp
    # surrounding text survives, only the control char is removed
    assert "<rdf:li>lioncub</rdf:li>" in xmp
    assert "Dawnlight over the plain." in xmp


# --- naming ----------------------------------------------------------------


def test_original_for_sidecar_strips_suffix() -> None:
    s = Path("/a/DSC_1.RAF.description.md")
    assert xmp_export.original_for_sidecar(s) == Path("/a/DSC_1.RAF")


def test_xmp_target_uses_stem_dot_xmp() -> None:
    assert xmp_export.xmp_target(Path("/a/DSC_1.RAF")) == Path("/a/DSC_1.xmp")
    # extra dots in the stem are preserved
    assert xmp_export.xmp_target(Path("/a/IMG.2024.RAF")) == Path("/a/IMG.2024.xmp")


# --- run() end-to-end ------------------------------------------------------


def _drop(
    root: Path,
    name: str,
    *,
    rating: str = "keep",
    make_original: bool = True,
    **fm: Any,
) -> Path:
    """Write an original media file (optional) + its .description.md sidecar."""
    original = root / name
    if make_original:
        original.write_bytes(b"\x00")
    front = {
        "file": name,
        "media_type": "image",
        "rating": rating,
        "scene_type": "wildlife",
        "keywords": ["lion"],
        **fm,
    }
    lines = "\n".join(f"{k}: {json.dumps(v)}" for k, v in front.items())
    (root / f"{name}.description.md").write_text(
        f"---\n{lines}\n---\n\n## Description\n\n**Scene:** A lion at dawn.\n"
    )
    return original


def test_run_writes_xmp_for_raw_with_rating(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_1.RAF", rating="cull")
    summary = xmp_export.run(tmp_path)
    xmp = tmp_path / "DSC_1.xmp"
    assert xmp.exists()
    text = xmp.read_text()
    assert 'xmp:Rating="1"' in text and 'xmp:Label="Red"' in text
    assert summary.wrote == 1
    # manifest records the file
    manifest = json.loads((tmp_path / "_XMP_MANIFEST.json").read_text())
    assert "DSC_1.xmp" in manifest


def test_run_skips_video_photos_dng_jpeg_and_missing(tmp_path: Path) -> None:
    _drop(tmp_path, "clip.mov", media_type="video")
    _drop(tmp_path, "IMG_9.CR3", photos_uuid="ABCD1234")
    _drop(tmp_path, "flat.dng")
    _drop(tmp_path, "snap.jpg")
    _drop(tmp_path, "gone.NEF", make_original=False)  # sidecar but no original
    summary = xmp_export.run(tmp_path)
    assert list(tmp_path.glob("*.xmp")) == []
    assert summary.wrote == 0
    assert summary.skipped_video == 1
    assert summary.skipped_photos == 1
    assert summary.skipped_nonraw == 2  # .dng and .jpg
    assert summary.skipped_missing_original == 1


def test_run_skips_unknown_rating_as_malformed(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_2.RAF", rating="banana")
    summary = xmp_export.run(tmp_path)
    assert not (tmp_path / "DSC_2.xmp").exists()
    assert summary.skipped_malformed == 1


def test_run_identical_rerun_is_up_to_date(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_3.RAF")
    xmp_export.run(tmp_path)
    before = (tmp_path / "DSC_3.xmp").read_bytes()
    summary = xmp_export.run(tmp_path)
    assert (tmp_path / "DSC_3.xmp").read_bytes() == before  # untouched
    assert summary.up_to_date == 1
    assert summary.wrote == 0


def test_run_regenerates_when_sidecar_changed(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_4.RAF", rating="keep")
    xmp_export.run(tmp_path)
    assert 'xmp:Rating="3"' in (tmp_path / "DSC_4.xmp").read_text()
    # user re-rated in framedex; sidecar now says cull
    _drop(tmp_path, "DSC_4.RAF", rating="cull")
    summary = xmp_export.run(tmp_path)
    assert 'xmp:Rating="1"' in (tmp_path / "DSC_4.xmp").read_text()
    assert summary.wrote == 1


def test_run_never_touches_foreign_xmp(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_5.RAF")
    foreign = tmp_path / "DSC_5.xmp"
    foreign.write_text("<x:xmpmeta>user develop settings</x:xmpmeta>\n")
    summary = xmp_export.run(tmp_path)
    assert foreign.read_text() == "<x:xmpmeta>user develop settings</x:xmpmeta>\n"
    assert summary.conflicts == 1
    assert summary.wrote == 0


def test_run_skips_our_xmp_edited_since_write(tmp_path: Path) -> None:
    """We wrote it, then the user edited it in Lightroom → bytes no longer match
    the manifest → treat as a conflict, never clobber their work."""
    _drop(tmp_path, "DSC_6.RAF")
    xmp_export.run(tmp_path)
    (tmp_path / "DSC_6.xmp").write_text("<x:xmpmeta>edited in LR</x:xmpmeta>\n")
    summary = xmp_export.run(tmp_path)
    assert (
        tmp_path / "DSC_6.xmp"
    ).read_text() == "<x:xmpmeta>edited in LR</x:xmpmeta>\n"
    assert summary.conflicts == 1


def test_run_missing_manifest_skips_existing_xmp(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_7.RAF")
    xmp_export.run(tmp_path)
    (tmp_path / "_XMP_MANIFEST.json").unlink()  # manifest lost
    (tmp_path / "DSC_7.xmp").write_text("hand edit\n")
    summary = xmp_export.run(tmp_path)
    assert (tmp_path / "DSC_7.xmp").read_text() == "hand edit\n"
    assert summary.conflicts == 1


def test_run_same_stem_raw_collision_writes_one(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_8.CR2")
    _drop(tmp_path, "DSC_8.NEF")  # both → DSC_8.xmp
    summary = xmp_export.run(tmp_path)
    assert (tmp_path / "DSC_8.xmp").exists()
    assert summary.wrote == 1
    assert summary.skipped_collision == 1


def test_run_dry_run_writes_nothing(tmp_path: Path) -> None:
    _drop(tmp_path, "DSC_9.RAF")
    summary = xmp_export.run(tmp_path, dry_run=True)
    assert not (tmp_path / "DSC_9.xmp").exists()
    assert not (tmp_path / "_XMP_MANIFEST.json").exists()
    assert summary.wrote == 1  # counted as would-write


def test_main_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys as _sys

    _drop(tmp_path, "DSC_A.RAF")
    monkeypatch.setattr(_sys, "argv", ["fdx-xmp", str(tmp_path)])
    assert xmp_export.main() == 0
    assert (tmp_path / "DSC_A.xmp").exists()
