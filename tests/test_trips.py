"""Tests for framedex.trips — splitting an archive into trips and drafting a
gallery per trip. Pure logic over Photo records."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from framedex import trips as t
from framedex.share import Photo

T0 = datetime(2025, 11, 7, 10, 0)
MARA = (-1.5, 35.3)
NAIROBI = (-1.29, 36.82)  # ~170 km from the Mara
LISBON = (38.72, -9.14)


def _p(
    rel: str, hours: float | None, at: tuple[float, float] | None = MARA, **kw: str
) -> Photo:
    return Photo(
        rel=rel,
        path=Path("/archive") / rel,
        taken=None if hours is None else T0 + timedelta(hours=hours),
        lat=at[0] if at else None,
        lon=at[1] if at else None,
        place="Siana ward, Narok West, Narok, Kenya" if at == MARA else "",
        rating=kw.get("rating", "keep"),
        focus="sharp",
    )


def test_km_between() -> None:
    assert t.km_between(MARA, MARA) == 0
    assert t.km_between(MARA, NAIROBI) == pytest.approx(170, abs=15)
    assert t.km_between(MARA, LISBON) > 6000


def test_a_long_gap_starts_a_new_trip() -> None:
    photos = [_p("a", 0), _p("b", 20), _p("c", 50), _p("d", 50 + 37)]
    assert [[p.rel for p in trip] for trip in t.split_trips(photos)] == [
        ["a", "b", "c"],
        ["d"],
    ]


def test_a_far_jump_after_half_a_day_starts_a_new_trip() -> None:
    photos = [
        _p("mara", 0),
        _p("nairobi", 3, NAIROBI),  # short drive: same trip
        _p("lisbon", 20, LISBON),  # 17 h later, 6000 km away: new trip
        _p("lisbon-nogps", 21, None),
        _p("lisbon2", 22, LISBON),
    ]
    assert [[p.rel for p in trip] for trip in t.split_trips(photos)] == [
        ["mara", "nairobi"],
        ["lisbon", "lisbon-nogps", "lisbon2"],
    ]


def test_undated_photos_are_left_out() -> None:
    assert t.split_trips([_p("u", None), _p("a", 0)]) == [[_p("a", 0)]]


def test_summarize_names_the_trip_and_drafts_a_gallery() -> None:
    trip = [_p(f"2025-11-mara/IMG_{i}.JPG", i * 2) for i in range(6)]
    trip.append(_p("2025-11-mara/day2/IMG_9.JPG", 30, rating="cull"))
    other = [_p("2025-12-home/IMG_1.JPG", 900)]
    s = t.summarize(trip, trip + other)
    assert (s.start, s.end) == (date(2025, 11, 7), date(2025, 11, 8))
    assert (s.photos, s.keeps, s.draft) == (7, 6, 6)
    assert s.folder == "2025-11-mara" and s.folder_is_trip
    assert s.name == "narok-2025-11"
    assert s.command(Path("/archive")) == (
        f"fdx-share /archive/2025-11-mara --name {s.name}"
    )


def test_a_trip_sharing_its_folder_gets_a_date_range() -> None:
    trip = [_p(f"phone/IMG_{i}.JPG", i) for i in range(5)]
    later = [_p("phone/IMG_99.JPG", 500)]
    s = t.summarize(trip, trip + later)
    assert not s.folder_is_trip
    assert s.command(Path("/archive")).endswith("--from 2025-11-07 --to 2025-11-07")
