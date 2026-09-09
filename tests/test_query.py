"""Tests for framedex.query — sidecar parsing and the record filter."""

import argparse
import sys
from pathlib import Path

import pytest

from framedex.query import main, matches, parse_sidecar

VALID_SIDECAR = """\
---
rating: keep
lighting: golden_hour
people_count: 3
---

## Description

A wide shot of the savanna.
"""

# --- parse_sidecar ---------------------------------------------------------


def test_parse_sidecar_valid(tmp_path: Path) -> None:
    p = tmp_path / "clip.description.md"
    p.write_text(VALID_SIDECAR)
    fm = parse_sidecar(p)
    assert fm is not None
    assert fm["rating"] == "keep"
    assert fm["people_count"] == 3
    assert fm["_sidecar_path"] == str(p)


def test_parse_sidecar_no_frontmatter(tmp_path: Path) -> None:
    p = tmp_path / "plain.md"
    p.write_text("Just prose, no frontmatter.\n")
    assert parse_sidecar(p) is None


def test_parse_sidecar_truncated(tmp_path: Path) -> None:
    # Opens with --- but never closes the block
    p = tmp_path / "truncated.md"
    p.write_text("---\nrating: keep\n")
    assert parse_sidecar(p) is None


def test_parse_sidecar_malformed_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.md"
    p.write_text("---\nrating: [unclosed\n---\nbody\n")
    assert parse_sidecar(p) is None


def test_parse_sidecar_non_dict_yaml(tmp_path: Path) -> None:
    # Frontmatter parses as a list, not a mapping
    p = tmp_path / "list.md"
    p.write_text("---\n- a\n- b\n---\nbody\n")
    assert parse_sidecar(p) is None


def test_parse_sidecar_missing_file(tmp_path: Path) -> None:
    assert parse_sidecar(tmp_path / "does-not-exist.md") is None


# --- matches ---------------------------------------------------------------


def make_args(**overrides: object) -> argparse.Namespace:
    """Build a query args namespace with every filter disabled by default."""
    defaults: dict[str, object] = {
        "rating": None,
        "media": None,
        "lighting": None,
        "time_of_day": None,
        "audio_quality": None,
        "language": None,
        "focus": None,
        "stability": None,
        "exposure": None,
        "people_count": None,
        "min_duration": None,
        "max_duration": None,
        "place_contains": None,
        "face_count": None,
        "person": None,
        "keyword": None,
        "dominant_color": None,
        "has_speech": False,
        "primary_only": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_matches_no_filters_passes() -> None:
    assert matches({"rating": "keep"}, make_args()) is True


def test_matches_rating_filter() -> None:
    rec = {"rating": "keep"}
    assert matches(rec, make_args(rating="keep")) is True
    assert matches(rec, make_args(rating="cull")) is False


def test_matches_rating_csv_is_or() -> None:
    assert matches({"rating": "review"}, make_args(rating="keep,review")) is True


def test_matches_people_count_plus_suffix() -> None:
    assert matches({"people_count": 5}, make_args(people_count="3+")) is True
    assert matches({"people_count": 2}, make_args(people_count="3+")) is False


def test_matches_people_count_exact() -> None:
    assert matches({"people_count": 4}, make_args(people_count="4")) is True
    assert matches({"people_count": 4}, make_args(people_count="5")) is False


def test_matches_duration_bounds() -> None:
    rec = {"duration_seconds": 10}
    assert matches(rec, make_args(min_duration=5)) is True
    assert matches(rec, make_args(min_duration=20)) is False
    assert matches(rec, make_args(max_duration=20)) is True
    assert matches(rec, make_args(max_duration=8)) is False


def test_matches_face_count_plus_suffix() -> None:
    assert matches({"face_count": 3}, make_args(face_count="2+")) is True
    assert matches({"face_count": 1}, make_args(face_count="2+")) is False


def test_matches_place_contains() -> None:
    rec = {"location": {"place": "Maasai Mara, Kenya"}}
    assert matches(rec, make_args(place_contains="mara")) is True
    assert matches(rec, make_args(place_contains="spain")) is False


def test_matches_technical_field_equality() -> None:
    # focus/stability/exposure all read the nested `technical` dict
    rec = {"technical": {"focus": "sharp"}}
    assert matches(rec, make_args(focus="sharp")) is True
    assert matches(rec, make_args(focus="soft")) is False


def test_matches_keyword_is_and() -> None:
    rec = {"keywords": ["sunset", "giraffe"]}
    assert matches(rec, make_args(keyword=["giraffe"])) is True
    # every requested keyword must be present
    assert matches(rec, make_args(keyword=["giraffe", "elephant"])) is False


def test_matches_dominant_color() -> None:
    rec = {"dominant_colors": ["green", "gold"]}
    assert matches(rec, make_args(dominant_color="green")) is True
    assert matches(rec, make_args(dominant_color="blue")) is False


def test_matches_person_cluster_id() -> None:
    rec = {"faces": [{"cluster_id": "Alex"}]}
    assert matches(rec, make_args(person="alex")) is True
    assert matches(rec, make_args(person="sam")) is False


def test_matches_has_speech() -> None:
    assert matches({"speaker_count": 2}, make_args(has_speech=True)) is True
    assert matches({"speaker_count": 0}, make_args(has_speech=True)) is False


# --- path resolution (issue #4) --------------------------------------------


def _write_sidecar(root: Path, rel: str, path_field: str) -> None:
    """Drop a sidecar at root/<rel>.description.md with the given `path` field."""
    sidecar = root / f"{rel}.description.md"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        f"---\nfile: {Path(rel).name}\npath: {path_field}\nrating: keep\n---\n"
        "\n## Description\n\nA clip.\n"
    )


def test_query_resolves_relative_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sidecar storing a root-relative `path` is resolved to an absolute path
    in query output, so the result stays pipeable (xargs, ffplay, ...)."""
    rel = "2024-08/drone/IMG_1.mov"
    _write_sidecar(tmp_path, rel, rel)
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "keep"])

    assert main() == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / rel)


def test_query_absolute_path_passthrough(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older sidecars with an absolute `path` are printed unchanged."""
    abs_path = "/Volumes/OldDrive/archive/old.mov"
    _write_sidecar(tmp_path, "old.mov", abs_path)
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "keep"])

    assert main() == 0
    assert capsys.readouterr().out.strip() == abs_path


# --- fail loud on malformed `path` (issue #14) -----------------------------


def _write_raw_sidecar(root: Path, name: str, body: str) -> None:
    (root / f"{name}.description.md").write_text(body)


def test_query_skips_non_string_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-string `path` (e.g. a bare int) must not crash the whole query on
    Path(); warn to stderr, skip that record, keep going for the good ones."""
    _write_raw_sidecar(
        tmp_path,
        "bad",
        "---\nfile: x\npath: 123\nrating: keep\n---\n\n## Description\n\nx\n",
    )
    _write_sidecar(tmp_path, "good", "/Volumes/D/good.mov")
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "keep"])

    assert main() == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == "/Volumes/D/good.mov"  # good record still emitted
    assert "bad.description.md" in cap.err
    assert "skipped 1" in cap.err


def test_query_skips_blank_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_raw_sidecar(
        tmp_path,
        "blank",
        '---\nfile: x\npath: "   "\nrating: keep\n---\n\n## Description\n\nx\n',
    )
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "keep"])
    assert main() == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == ""  # nothing usable emitted
    assert "skipped 1" in cap.err


def test_query_skips_missing_path_without_photos_uuid(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A folder-mode sidecar always carries `path`; one missing it (and not a
    Photos-managed asset) is broken — skip rather than print the .md path as if
    it were the media path."""
    _write_raw_sidecar(
        tmp_path, "nopath", "---\nfile: x\nrating: keep\n---\n\n## Description\n\nx\n"
    )
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "keep"])
    assert main() == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == ""
    assert "skipped 1" in cap.err


def test_query_keeps_photos_asset_without_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fdx-photos mirror sidecars for downloaded iCloud assets legitimately omit
    `path` (the original lives in Photos, not on disk) and carry photos_uuid.
    These must NOT be skipped — they fall back to the sidecar path in output."""
    _write_raw_sidecar(
        tmp_path,
        "photo",
        "---\nfile: IMG.jpg\nphotos_uuid: ABCD1234\nrating: keep\n---\n\n## Description\n\nx\n",
    )
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "keep"])
    assert main() == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == str(tmp_path / "photo.description.md")  # fallback
    assert "skipped" not in cap.err


def test_query_skips_photos_asset_with_malformed_present_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Photos carve-out applies only when `path` is truly OMITTED. A
    photos_uuid record with a present-but-malformed `path` (e.g. an int) is
    corrupt — skip it, don't fall back and emit a bogus value."""
    _write_raw_sidecar(
        tmp_path,
        "corrupt",
        "---\nfile: x\nphotos_uuid: ABCD\npath: 123\nrating: keep\n---\n\n## Description\n\nx\n",
    )
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "keep"])
    assert main() == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == ""
    assert "skipped 1" in cap.err


# --- burst / RAW+JPEG group stubs -----------------------------------------


def test_matches_primary_only_filters_group_stubs() -> None:
    """A stub (group.primary: false) carries a copied assessment; by default
    it still matches (finding a burst alternate by keyword is useful), and
    --primary-only hides it. Ungrouped records always pass."""
    stub = {"rating": "keep", "group": {"kind": "burst", "primary": False}}
    primary = {"rating": "keep", "group": {"kind": "burst", "primary": True}}
    assert matches(stub, make_args()) is True
    assert matches(stub, make_args(primary_only=True)) is False
    assert matches(primary, make_args(primary_only=True)) is True
    assert matches({"rating": "keep"}, make_args(primary_only=True)) is True


def test_query_cli_accepts_primary_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "1.NEF.description.md").write_text(
        "---\nfile: 1.NEF\npath: 1.NEF\nrating: keep\n"
        "group: {kind: burst, primary: false, primary_file: 2.NEF}\n---\n"
    )
    (tmp_path / "2.NEF.description.md").write_text(
        "---\nfile: 2.NEF\npath: 2.NEF\nrating: keep\n"
        "group: {kind: burst, primary: true}\n---\n"
    )
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--primary-only"])
    assert main() == 0
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / "2.NEF")]


def test_parse_sidecar_triple_hyphen_in_value_is_not_a_fence(tmp_path: Path) -> None:
    p = tmp_path / "a---b.NEF.description.md"
    p.write_text("---\nfile: a---b.NEF\nrating: keep\n---\n\n## Description\n\nx\n")
    fm = parse_sidecar(p)
    assert fm is not None and fm["file"] == "a---b.NEF"


# --- Filters / run_query (shared by the CLI and fdx-mcp) -------------------


def _sidecar(root: Path, rel: str, fm: dict[str, object], body: str = "") -> None:
    import yaml

    full = {"file": Path(rel).name, "path": rel, "media_type": "image", **fm}
    p = root / f"{rel}.description.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "---\n"
        + yaml.safe_dump(full, sort_keys=False)
        + "---\n\n## Description\n\n"
        + body
    )


def test_run_query_returns_effective_ratings_total_and_paging(tmp_path: Path) -> None:
    from framedex.query import Filters, run_query

    _sidecar(
        tmp_path,
        "a/1.NEF",
        {"rating": "cull", "user_rating": "keep"},
        "**Scene:** A lion.\n",
    )
    _sidecar(tmp_path, "a/2.NEF", {"rating": "keep"})
    _sidecar(tmp_path, "b/3.NEF", {"rating": "keep", "user_rating": "banana"})
    _sidecar(tmp_path, "b/4.NEF", {"rating": "review"})

    res = run_query(tmp_path, Filters(rating="keep"))
    assert [r["path"] for r in res.records] == [
        str(tmp_path / p) for p in ("a/1.NEF", "a/2.NEF", "b/3.NEF")
    ]
    assert res.total == 3 and res.skipped_malformed == 0
    assert res.records[0]["effective_rating"] == "keep"  # the human override wins
    assert res.invalid_user_ratings == 1  # "banana" is reported, not silently used

    page = run_query(tmp_path, Filters(rating="keep", offset=1, limit=1))
    assert [r["path"] for r in page.records] == [str(tmp_path / "a/2.NEF")]
    assert page.total == 3

    sub = run_query(tmp_path, Filters(folder="b"))
    assert sorted(Path(r["path"]).name for r in sub.records) == ["3.NEF", "4.NEF"]


def test_run_query_folder_must_be_inside_root(tmp_path: Path) -> None:
    from framedex.query import Filters, run_query

    with pytest.raises(ValueError, match="folder"):
        run_query(tmp_path, Filters(folder="../elsewhere"))


def test_query_cli_shows_effective_rating_and_user_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _sidecar(
        tmp_path,
        "1.NEF",
        {"rating": "cull", "user_rating": "keep", "keywords": ["lion"]},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["fdx-query", str(tmp_path), "--rating", "keep", "--with-description"],
    )
    assert main() == 0
    out = capsys.readouterr().out
    assert "\tkeep (user)\t" in out
    monkeypatch.setattr(sys, "argv", ["fdx-query", str(tmp_path), "--rating", "cull"])
    assert main() == 0
    assert capsys.readouterr().out.strip() == ""  # the model's cull no longer matches


def test_query_cli_folder_and_offset_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _sidecar(tmp_path, "a/1.NEF", {"rating": "keep"})
    _sidecar(tmp_path, "a/2.NEF", {"rating": "keep"})
    _sidecar(tmp_path, "b/3.NEF", {"rating": "keep"})
    monkeypatch.setattr(
        sys, "argv", ["fdx-query", str(tmp_path), "--folder", "a", "--offset", "1"]
    )
    assert main() == 0
    assert capsys.readouterr().out.splitlines() == [str(tmp_path / "a/2.NEF")]


def test_run_query_counts_unparsable_and_escaping_sidecars_and_tolerates_bad_shapes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from framedex.query import Filters, run_query

    (tmp_path / "bad.NEF.description.md").write_text("---\nrating: [unclosed\n---\n")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.description.md"
    outside.write_text("---\nfile: o.jpg\npath: o.jpg\nrating: keep\n---\n")
    (tmp_path / "o.jpg.description.md").symlink_to(outside)
    _sidecar(
        tmp_path,
        "shape.NEF",
        {
            "rating": "keep",
            "location": "Mara",
            "technical": "sharp",
            "keywords": "lion",
        },
    )
    _sidecar(tmp_path, "ok.NEF", {"rating": "keep", "location": {"place": "Mara"}})
    try:
        res = run_query(tmp_path, Filters(place_contains="mara", focus="sharp"))
        assert [
            Path(r["path"]).name for r in res.records
        ] == []  # bad shapes never match…
        res = run_query(tmp_path, Filters(place_contains="mara"))
        assert [Path(r["path"]).name for r in res.records] == ["ok.NEF"]
        assert (
            res.skipped_malformed == 2
        )  # …and the unparsable + escaping ones are counted
        assert "unparsable" in capsys.readouterr().err
    finally:
        outside.unlink()


def test_run_query_media_path_containment_is_opt_in(tmp_path: Path) -> None:
    """The CLI keeps printing legacy absolute paths; fdx-mcp asks for
    containment, which drops a record whose media path resolves outside."""
    from framedex.query import Filters, run_query

    _sidecar(tmp_path, "in.NEF", {"rating": "keep"})
    (tmp_path / "esc.NEF.description.md").write_text(
        "---\nfile: esc.NEF\npath: ../esc.NEF\nrating: keep\n---\n"
    )
    assert len(run_query(tmp_path, Filters()).records) == 2
    strict = run_query(tmp_path, Filters(), require_media_under_root=True)
    assert [Path(r["path"]).name for r in strict.records] == ["in.NEF"]
    assert strict.skipped_malformed == 1


def test_matches_never_raises_on_garbled_scalar_fields() -> None:
    garbled = {
        "rating": ["keep"],
        "lighting": ["golden_hour"],
        "speaker_count": "two",
        "duration_seconds": "long",
        "face_count": "many",
    }
    assert matches(garbled, make_args(rating="keep")) is False
    assert matches(garbled, make_args(lighting="golden_hour")) is False
    assert matches(garbled, make_args(has_speech=True)) is False
    assert matches(garbled, make_args(min_duration=1.0)) is False
    assert matches(garbled, make_args(face_count="1+")) is False


def test_run_query_survives_a_nul_byte_in_a_path(tmp_path: Path) -> None:
    from framedex.query import Filters, run_query

    (tmp_path / "nul.NEF.description.md").write_text(
        '---\nfile: nul.NEF\npath: "bad\\0name.NEF"\nrating: keep\n---\n'
    )
    _sidecar(tmp_path, "ok.NEF", {"rating": "keep"})
    res = run_query(tmp_path, Filters(), require_media_under_root=True)
    assert [Path(r["path"]).name for r in res.records] == ["ok.NEF"]
    assert res.skipped_malformed == 1


def test_run_query_prefers_the_original_next_to_the_sidecar(tmp_path: Path) -> None:
    """Sidecars store `path` relative to the root they were indexed from. A
    query rooted at a parent (or a subfolder) must still find the file: the
    original next to the sidecar is ground truth in folder mode."""
    from framedex.query import Filters, run_query

    trip = tmp_path / "trip"
    trip.mkdir()
    (trip / "a.NEF").write_bytes(b"x")
    # Indexed with `fdx trip`: path is relative to trip/, not to tmp_path.
    (trip / "a.NEF.description.md").write_text(
        "---\nfile: a.NEF\npath: a.NEF\nrating: keep\n---\n"
    )
    res = run_query(tmp_path, Filters(), require_media_under_root=True)
    assert [r["path"] for r in res.records] == [str(trip / "a.NEF")]
    # A moved-away original falls back to the stored path (still relative to root).
    (trip / "a.NEF").unlink()
    res = run_query(tmp_path, Filters())
    assert [r["path"] for r in res.records] == [str(tmp_path / "a.NEF")]
