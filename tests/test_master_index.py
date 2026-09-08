"""Tests for framedex.master_index — sidecar path resolution and grouping."""

import json
import sys
from pathlib import Path

import pytest

from framedex.master_index import main


def _write_sidecar(root: Path, rel: str, path_field: str) -> None:
    """Drop a sidecar at root/<rel>.description.md with the given `path` field."""
    sidecar = root / f"{rel}.description.md"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        f"---\nfile: {Path(rel).name}\npath: {path_field}\n"
        "rating: cull\ncull_reason: test\nduration_seconds: 5.0\n---\n"
        "\n## Description\n\nA clip.\n"
    )


def test_master_index_resolves_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root-relative `path` is resolved against --root, so the clip records
    an absolute path and groups into its top-level folder."""
    rel = "2024-08/drone/c.mov"
    _write_sidecar(tmp_path, rel, rel)
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])

    assert main() == 0

    index = json.loads((tmp_path / "_INDEX.json").read_text())
    assert index["clips"][0]["path"] == str(tmp_path / rel)
    assert index["trip_count"] == 1


def test_master_index_absolute_path_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Older sidecars with an absolute `path` are recorded unchanged."""
    abs_path = "/Volumes/OldDrive/archive/old.mov"
    _write_sidecar(tmp_path, "old.mov", abs_path)
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])

    assert main() == 0

    index = json.loads((tmp_path / "_INDEX.json").read_text())
    assert index["clips"][0]["path"] == abs_path


def test_master_index_skips_non_string_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-string `path` must warn+skip, not crash Path() and abort the whole
    index; valid records are still recorded (issue #14)."""
    (tmp_path / "bad.description.md").write_text(
        "---\nfile: x\npath: 123\nrating: cull\nduration_seconds: 5.0\n---\n"
        "\n## Description\n\nx\n"
    )
    _write_sidecar(tmp_path, "2024/good.mov", "2024/good.mov")
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])

    assert main() == 0
    index = json.loads((tmp_path / "_INDEX.json").read_text())
    paths = [c["path"] for c in index["clips"]]
    assert str(tmp_path / "2024/good.mov") in paths
    assert index["clip_count"] == 1  # bad record excluded
    assert "skipped 1" in capsys.readouterr().err


def test_master_index_skips_photos_asset_with_malformed_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Photos carve-out is path-OMITTED-only. A photos_uuid record with a
    present-but-malformed path must be skipped, not kept (Path(123) would
    otherwise crash the whole index)."""
    (tmp_path / "corrupt.description.md").write_text(
        "---\nfile: x\nphotos_uuid: ABCD\npath: 123\nrating: cull\n"
        "duration_seconds: 5.0\n---\n\n## Description\n\nx\n"
    )
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])

    assert main() == 0  # must not crash on Path(123)
    index = json.loads((tmp_path / "_INDEX.json").read_text())
    assert index["clip_count"] == 0
    assert "skipped 1" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Burst / RAW+JPEG groups: stubs are sidecars but not assessments
# ---------------------------------------------------------------------------


def _write_fm(root: Path, rel: str, fm: dict[str, object]) -> None:
    import yaml

    full = {"file": Path(rel).name, "path": rel, "media_type": "image", **fm}
    sidecar = root / f"{rel}.description.md"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        "---\n"
        + yaml.safe_dump(full, sort_keys=False)
        + "---\n\n## Description\n\nx.\n"
    )


def test_master_index_counts_primaries_only_and_reports_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = {"kind": "burst", "id": "b_1", "members": ["1.NEF", "2.NEF", "3.NEF"]}
    _write_fm(
        tmp_path,
        "2.NEF",
        {
            "rating": "cull",
            "cull_reason": "blur",
            "keywords": ["lion"],
            "face_count": 1,
            "group": {**group, "primary": True, "sharpness": 9.0},
        },
    )
    for n in ("1.NEF", "3.NEF"):
        _write_fm(
            tmp_path,
            n,
            {
                "rating": "cull",
                "cull_reason": "blur",
                "keywords": ["lion"],
                "group": {
                    **group,
                    "primary": False,
                    "primary_file": "2.NEF",
                    "sharpness": 1.0,
                },
            },
        )
    _write_fm(
        tmp_path, "lone.NEF", {"rating": "keep", "keywords": ["lion"], "face_count": 2}
    )
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])

    assert main() == 0

    md = (tmp_path / "_INDEX.md").read_text()
    assert "1 keep, 0 review, 1 cull" in md  # stubs are not counted as ratings
    assert "`lion` (2)" in md  # keyword frequency over primaries only
    assert "**Faces detected:** 3 total" in md
    assert "## Cull pile — 1 clips" in md  # stubs are not listed
    assert f"- `{tmp_path / '2.NEF'}` — blur" in md
    assert "1.NEF` —" not in md and "3.NEF` —" not in md
    assert (
        "- **Grouped:** 3 files in 1 group (bursts / RAW+JPEG pairs). Ratings, "
        "keywords, faces, and the cull list exclude non-primary group members." in md
    )
    idx = json.loads((tmp_path / "_INDEX.json").read_text())
    assert idx["clip_count"] == 4  # every sidecar is still a record
    assert idx["group_count"] == 1 and idx["grouped_file_count"] == 3


def test_master_index_without_groups_has_no_grouped_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fm(tmp_path, "a.NEF", {"rating": "keep"})
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])
    assert main() == 0
    assert "Grouped" not in (tmp_path / "_INDEX.md").read_text()
    idx = json.loads((tmp_path / "_INDEX.json").read_text())
    assert idx["group_count"] == 0 and idx["grouped_file_count"] == 0


def test_master_index_scopes_group_ids_by_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ids hash member names, so two shoots with identical filenames share
    one id; they are still two groups."""
    for folder in ("a", "b"):
        _write_fm(
            tmp_path,
            f"{folder}/1.NEF",
            {
                "rating": "keep",
                "group": {
                    "kind": "burst",
                    "id": "b_same",
                    "primary": True,
                    "members": ["1.NEF", "2.NEF"],
                },
            },
        )
        _write_fm(
            tmp_path,
            f"{folder}/2.NEF",
            {
                "rating": "keep",
                "group": {
                    "kind": "burst",
                    "id": "b_same",
                    "primary": False,
                    "primary_file": "1.NEF",
                },
            },
        )
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])
    assert main() == 0
    idx = json.loads((tmp_path / "_INDEX.json").read_text())
    assert idx["group_count"] == 2 and idx["grouped_file_count"] == 4


def test_master_index_counts_effective_ratings_and_marks_user_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_fm(
        tmp_path,
        "1.NEF",
        {"rating": "cull", "cull_reason": "blur", "user_rating": "keep"},
    )
    _write_fm(
        tmp_path, "2.NEF", {"rating": "keep", "user_rating": "cull", "user_note": "dup"}
    )
    _write_fm(tmp_path, "3.NEF", {"rating": "review", "user_rating": "banana"})
    monkeypatch.setattr(sys, "argv", ["fdx-master", str(tmp_path)])
    assert main() == 0
    md = (tmp_path / "_INDEX.md").read_text()
    assert "1 keep, 1 review, 1 cull" in md  # overrides applied, invalid one ignored
    assert "- **User ratings:** 2 files" in md
    assert (
        f"- `{tmp_path / '2.NEF'}` — dup (user)" in md
    )  # the human's cull, with the note
    assert "1.NEF" not in md.split("## Cull pile")[1].split("## Trips")[0]
