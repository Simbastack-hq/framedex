"""Tests for framedex.query — sidecar parsing and the record filter."""

import argparse
from pathlib import Path

from framedex.query import matches, parse_sidecar

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
