"""Tests for framedex.share — fdx-share's selection, ordering, and the
editable selection.txt.

Pure logic over Photo records and sidecar-shaped dicts; no pixels, no
exiftool, no network.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from framedex import share as s

T0 = datetime(2025, 11, 9, 9, 0, 0)
MARA = "Ewangan, Siana ward, Narok West, Narok, Kenya"
TALEK = "Talek, Siana ward, Narok East, Narok, Kenya"


def _p(
    rel: str,
    minutes: float | None = 0.0,
    *,
    place: str = MARA,
    rating: str = "keep",
    focus: str = "sharp",
    exposure: str = "strong",
    composition: str = "strong",
    camera: str = "Apple|iPhone 15 Pro Max",
    scene: str = "",
    group_key: str | None = None,
    group_primary: bool = False,
    user_rated: bool = False,
    aspect: float | None = 1.5,
    subjects: str = "",
    start: datetime = T0,
) -> s.Photo:
    return s.Photo(
        rel=rel,
        path=Path("/shoot") / rel,
        taken=None if minutes is None else start + timedelta(minutes=minutes),
        place=place,
        rating=rating,
        user_rated=user_rated,
        focus=focus,
        exposure=exposure,
        composition=composition,
        scene=scene,
        camera=camera,
        group_key=group_key,
        group_primary=group_primary,
        aspect=aspect,
        subjects=subjects,
    )


def _sig(seed: str, aspect: float = 1.5) -> s.Signature:
    """A stand-in thumbnail: unrelated seeds give unrelated pixels."""
    digest = b"".join(hashlib.sha256(f"{seed}{i}".encode()).digest() for i in range(8))
    return s.Signature(aspect, tuple(digest[: s.SIGNATURE_SIZE**2]))


def _nudged(sig: s.Signature, by: int) -> s.Signature:
    """The same picture, slightly different: every pixel moved by `by`."""
    return s.Signature(sig.aspect, tuple(min(255, max(0, v + by)) for v in sig.pixels))


def _rels(photos: list[s.Photo]) -> list[str]:
    return [p.rel for p in photos]


# --- records -----------------------------------------------------------------


def test_photo_from_record_reads_the_sidecar_fields() -> None:
    rec = {
        "media_type": "image",
        "path": "/shoot/day1/IMG_1.JPG",
        "creation_time": "2025-11-09T09:48:12",
        "location": {"lat": -1.5, "lon": 35.3, "place": MARA},
        "effective_rating": "keep",
        "user_rating": "keep",
        "technical": {"focus": "sharp", "exposure": "adequate", "composition": "weak"},
        "camera": {"make": "Apple", "model": "iPhone", "orientation": "Rotate 90 CW"},
        "dimensions": "4032x3024",
        "group": {"kind": "burst", "id": "b_1234abcd", "primary": True},
        "_scene": "A leopard walks along a dirt track.",
    }
    p = s.photo_from_record(rec, Path("/shoot"))
    assert p is not None
    assert p.rel == "day1/IMG_1.JPG"
    assert p.taken == datetime(2025, 11, 9, 9, 48, 12)
    assert (p.place, p.lat, p.lon) == (MARA, -1.5, 35.3)
    assert p.rating == "keep" and p.user_rated
    assert p.camera == "Apple|iPhone"
    assert p.group_key == "day1/b_1234abcd" and p.group_primary
    assert p.aspect == pytest.approx(3024 / 4032)  # rotated 90: portrait
    assert p.score == 3 + 1 + 0 + s.USER_KEEP_BONUS


def test_photo_from_record_skips_videos_and_files_outside_the_folder() -> None:
    base = {"path": "/shoot/a.JPG", "effective_rating": "keep"}
    assert s.photo_from_record({**base, "media_type": "video"}, Path("/shoot")) is None
    assert s.photo_from_record(base, Path("/shoot")) is None  # absent = video
    assert (
        s.photo_from_record({**base, "media_type": "image"}, Path("/elsewhere")) is None
    )


def test_garbled_sidecar_fields_do_not_crash() -> None:
    rec = {
        "media_type": "image",
        "path": "/shoot/a.JPG",
        "creation_time": "yesterday",
        "location": "somewhere",
        "technical": ["sharp"],
        "camera": None,
        "effective_rating": 3,
        "dimensions": "big",
    }
    p = s.photo_from_record(rec, Path("/shoot"))
    assert p is not None
    assert p.taken is None and p.place == "" and p.rating == "" and p.aspect is None


def test_parse_taken_formats() -> None:
    assert s.parse_taken("2025-11-09T09:48:12+03:00") == datetime(
        2025, 11, 9, 9, 48, 12
    )
    assert s.parse_taken("2025-11-09T09:48:12.5") == datetime(
        2025, 11, 9, 9, 48, 12, 500000
    )
    assert s.parse_taken("2025-11-09") == datetime(2025, 11, 9)
    assert s.parse_taken("") is None
    assert s.parse_taken(None) is None


# --- one per burst -----------------------------------------------------------


def test_sidecar_group_keeps_the_primary() -> None:
    a = _p("a.JPG", 0, group_key="./b_1", group_primary=False, focus="sharp")
    b = _p("b.JPG", 0.01, group_key="./b_1", group_primary=True, focus="soft")
    c = _p("c.JPG", 0.02, group_key="./b_1", group_primary=False)
    kept, dropped = s.collapse_bursts([a, b, c])
    assert _rels(kept) == ["b.JPG"]  # the indexer's sharpest frame, not our score
    assert dropped == {"a.JPG": "b.JPG", "c.JPG": "b.JPG"}


def test_sidecar_group_prefers_a_human_keep() -> None:
    a = _p("a.JPG", 0, group_key="./b_1", group_primary=True)
    b = _p("b.JPG", 0.01, group_key="./b_1", user_rated=True)
    kept, _ = s.collapse_bursts([a, b])
    assert _rels(kept) == ["b.JPG"]


def test_ungrouped_bursts_are_chained_by_timestamp_and_camera() -> None:
    # Five frames a second apart (a burst), one 30 s later (its own moment),
    # and three frames from another camera at the same instants (not chained
    # with the first camera).
    burst = [_p(f"b{i}.JPG", i / 60, focus="soft") for i in range(5)]
    burst[3].focus = "sharp"
    lone = _p("lone.JPG", 0.5)
    other = [_p(f"o{i}.JPG", i / 60, camera="NIKON|Z 8") for i in range(2)]
    kept, dropped = s.collapse_bursts([*burst, lone, *other])
    assert _rels(kept) == ["o0.JPG", "o1.JPG", "b3.JPG", "lone.JPG"]
    assert set(dropped) == {"b0.JPG", "b1.JPG", "b2.JPG", "b4.JPG"}


def test_undated_photos_never_chain() -> None:
    photos = [_p(f"u{i}.JPG", None) for i in range(4)]
    kept, dropped = s.collapse_bursts(photos)
    assert len(kept) == 4 and dropped == {}


# --- near-duplicates ---------------------------------------------------------


def test_same_picture_needs_shape_correlation_and_closeness() -> None:
    a = _sig("leopard")
    assert s.same_picture(a, a)
    assert s.same_picture(a, _nudged(a, 15))  # brighter, same structure
    assert not s.same_picture(a, _nudged(a, 40))  # too different on average
    assert not s.same_picture(a, _sig("lion"))  # unrelated structure
    assert not s.same_picture(a, s.Signature(0.56, a.pixels))  # portrait vs wide
    flat = s.Signature(1.5, (128,) * s.SIGNATURE_SIZE**2)
    assert s.compare(flat, flat) == (0.0, 0.0)  # nothing to correlate
    assert not s.same_picture(flat, flat)
    assert s.compare(a, s.Signature(1.5, (1, 2))) == (0.0, 255.0)


def test_near_duplicates_keep_the_better_frame_within_the_window() -> None:
    a = _p("a.JPG", 0, composition="acceptable")
    b = _p("b.JPG", 1)  # better score, one minute later, same picture
    c = _p("c.JPG", 2)  # different picture
    far = _p("far.JPG", 60 * 5)  # same picture, five hours later
    base = _sig("leopard")
    sigs = {
        "a.JPG": base,
        "b.JPG": _nudged(base, 5),
        "c.JPG": _sig("lion"),
        "far.JPG": base,
    }
    assert s.near_duplicates([a, b, c, far], sigs) == {"a.JPG": "b.JPG"}


def test_near_duplicates_need_a_signature_and_a_date() -> None:
    a, b = _p("a.JPG", 0), _p("b.JPG", None)
    c = _p("c.JPG", 0.5)
    sig = _sig("x")
    assert s.near_duplicates([a, b, c], {"a.JPG": sig, "b.JPG": sig}) == {}


# --- the cap -------------------------------------------------------------------


def test_allocate_everything_fits() -> None:
    assert s.allocate([3, 2], 10) == [3, 2]


def test_allocate_gives_every_section_one_then_shares_the_rest() -> None:
    out = s.allocate([30, 2, 8], 10)
    assert sum(out) == 10
    assert all(n >= 1 for n in out)
    assert out[0] > out[2] > 0
    assert out == [7, 1, 2]


def test_allocate_more_sections_than_slots_favours_large_sections() -> None:
    assert s.allocate([1, 5, 2, 9], 2) == [0, 1, 0, 1]


def test_allocate_never_exceeds_a_section() -> None:
    for sizes in ([1, 1, 50], [2, 3, 4, 100], [5, 5, 5]):
        for cap in range(1, sum(sizes)):
            out = s.allocate(sizes, cap)
            assert sum(out) == cap
            assert all(0 <= o <= n for o, n in zip(out, sizes, strict=True))


def test_pick_spread_prefers_frames_apart_in_time() -> None:
    # Three great frames within one minute, two good ones much later.
    close = [_p(f"c{i}.JPG", i / 4) for i in range(3)]
    later = [_p(f"l{i}.JPG", 30 + i * 10, composition="acceptable") for i in range(2)]
    chosen = s.pick_spread([*close, *later], 3)
    assert _rels(chosen) == ["c0.JPG", "l0.JPG", "l1.JPG"]


def test_pick_spread_fills_the_quota_when_everything_is_close() -> None:
    close = [_p(f"c{i}.JPG", i / 60) for i in range(4)]
    assert len(s.pick_spread(close, 3)) == 3


def test_plan_gallery_end_to_end() -> None:
    photos = [
        _p("cull.JPG", 0, rating="cull"),
        _p("review.JPG", 1, rating="review"),
        *[_p(f"leopard{i}.JPG", 10 + i * 3) for i in range(10)],
        _p("dup.JPG", 41, composition="acceptable"),
        _p("talek.JPG", 300, place=TALEK),
        _p("talek2.JPG", 310, place=TALEK),
    ]
    sigs = {p.rel: _sig(p.rel) for p in photos}
    sigs["dup.JPG"] = _nudged(sigs["leopard9.JPG"], 3)

    plan = s.plan_gallery(photos, max_photos=5, signer=lambda cands: sigs)
    picked = _rels(plan.photos)
    assert len(picked) == 5
    assert "cull.JPG" not in picked and "review.JPG" not in picked
    assert "talek.JPG" in picked or "talek2.JPG" in picked  # every place shows up
    assert [sec.heading for sec in plan.sections] == [
        "Sunday 9 November · Ewangan, Narok West",
        "Sunday 9 November · Talek, Narok East",
    ]
    reasons = dict((p.rel, why) for p, why in plan.not_picked)
    assert reasons["dup.JPG"] == "near-duplicate of leopard9.JPG"
    assert any("cap" in why for why in reasons.values())

    with_review = s.plan_gallery(photos, max_photos=50, include_review=True)
    assert "review.JPG" in _rels(with_review.photos)


# --- ordering ------------------------------------------------------------------


def test_trip_day_starts_at_four_am() -> None:
    assert s.trip_day(datetime(2025, 11, 9, 3, 59)) == date(2025, 11, 8)
    assert s.trip_day(datetime(2025, 11, 9, 4, 0)) == date(2025, 11, 9)


def test_place_area_and_label() -> None:
    assert s.place_area(MARA) == "Narok West"
    assert s.place_label(MARA) == "Ewangan, Narok West"
    assert (
        s.place_label("Odeceixe, Aljezur, Faro, 8670-320, Portugal")
        == "Odeceixe, Aljezur"
    )
    assert s.place_label("Nairobi, Kenya") == "Nairobi"
    assert s.place_label("Talek, Narok, Kenya") == "Talek"
    assert s.place_label("Siana ward, Narok West, Narok, 66601, Kenya") == "Narok West"
    assert s.place_label("") == ""


def test_sections_break_on_day_and_place() -> None:
    photos = [
        _p("sat1.JPG", -24 * 60),
        _p("sat2.JPG", -24 * 60 + 5),
        _p("mara1.JPG", 0),
        _p("mara2.JPG", 5),
        _p("talek1.JPG", 60, place=TALEK),
        _p("talek2.JPG", 65, place=TALEK),
    ]
    secs = s.make_sections(photos)
    assert [(sec.heading, _rels(sec.photos)) for sec in secs] == [
        ("Saturday 8 November · Ewangan, Narok West", ["sat1.JPG", "sat2.JPG"]),
        ("Sunday 9 November · Ewangan, Narok West", ["mara1.JPG", "mara2.JPG"]),
        ("Sunday 9 November · Talek, Narok East", ["talek1.JPG", "talek2.JPG"]),
    ]


def test_headings_carry_the_year_when_a_gallery_crosses_one() -> None:
    photos = [
        _p("a.JPG", 0),
        _p("b.JPG", 1),
        _p("c.JPG", 60 * 24 * 400),
        _p("d.JPG", 60 * 24 * 400 + 1),
    ]
    assert [sec.heading.split(" · ")[0] for sec in s.make_sections(photos)] == [
        "Sunday 9 November 2025",
        "Monday 14 December 2026",
    ]


def test_photos_without_gps_stay_in_the_current_section() -> None:
    photos = [_p("a.JPG", 0), _p("nogps.JPG", 1, place=""), _p("b.JPG", 2)]
    secs = s.make_sections(photos)
    assert len(secs) == 1 and _rels(secs[0].photos) == ["a.JPG", "nogps.JPG", "b.JPG"]


def test_a_single_frame_elsewhere_is_folded_into_its_day() -> None:
    photos = [
        _p("a.JPG", 0),
        _p("b.JPG", 5),
        _p("fuel.JPG", 30, place=TALEK),
        _p("c.JPG", 60),
        _p("d.JPG", 65),
    ]
    secs = s.make_sections(photos)
    assert len(secs) == 1
    assert _rels(secs[0].photos) == ["a.JPG", "b.JPG", "fuel.JPG", "c.JPG", "d.JPG"]


def test_undated_photos_go_last_in_their_own_section() -> None:
    photos = [_p("u.JPG", None, place=""), _p("a.JPG", 0)]
    secs = s.make_sections(photos)
    assert [sec.heading for sec in secs][-1] == "Undated"
    assert _rels(secs[-1].photos) == ["u.JPG"]


# --- captions, labels, names -----------------------------------------------------


def test_caption_keeps_short_sentences_and_cuts_long_ones_at_a_clause() -> None:
    assert (
        s.caption_for("A jackal stands in tall grass.")
        == "A jackal stands in tall grass."
    )
    long = (
        "A luxury safari lodge's pool terrace and grounds captured in late afternoon "
        "light, overlooking savanna landscape with scattered acacia trees and distant "
        "hills under a clear sky."
    )
    assert s.caption_for(long) == (
        "A luxury safari lodge's pool terrace and grounds captured in late afternoon light."
    )
    no_break = "word " * 60
    cut = s.caption_for(no_break)
    assert cut.endswith("…") and len(cut) <= s.CAPTION_MAX_CHARS + 1
    assert s.caption_for("") == ""


def test_caption_prefers_the_subject_over_the_setting() -> None:
    scene = "Midday savanna in Narok County, Kenya; clear bright light."
    subjects = (
        "One adult African elephant walking directly toward camera with trunk "
        "raised and tusks visible. The animal appears calm."
    )
    assert s.caption_for(scene, subjects) == (
        "An adult African elephant walking directly toward camera with trunk "
        "raised and tusks visible."
    )
    assert (
        s.caption_for(scene, "One giraffe (center, alert) and two zebras (grazing).")
        == "A giraffe and two zebras."
    )
    lodge = "A lodge terrace at dusk, overlooking the savanna."
    assert (
        s.caption_for(lodge, "No people. The frame centers on the pool.")
        == "A lodge terrace at dusk, overlooking the savanna."
    )
    assert s.caption_for("Early morning in the savanna (Siana ward, Narok).") == (
        "Early morning in the savanna."
    )
    assert s.caption_for("Dawn (unfinished, cut off", "") == "Dawn."


def test_date_range_label() -> None:
    assert s.date_range_label([date(2025, 11, 9)]) == "9 November 2025"
    assert (
        s.date_range_label([date(2025, 11, 9), date(2025, 11, 8)])
        == "8\u20139 November 2025"
    )
    assert (
        s.date_range_label([date(2025, 10, 30), date(2025, 11, 2)])
        == "30 October \u2013 2 November 2025"
    )
    assert (
        s.date_range_label([date(2025, 12, 30), date(2026, 1, 2)])
        == "30 December 2025 \u2013 2 January 2026"
    )
    assert s.date_range_label([]) == ""


def test_slugs_and_link_tokens() -> None:
    assert s.slugify("Mara — Nov 2025!") == "mara-nov-2025"
    assert s.slugify("ünïcode") == "ncode"
    assert s.slugify("!!!") == "gallery"
    token = s.link_token()
    assert len(token) == s.LINK_TOKEN_LEN and set(token) <= set(s.LINK_TOKEN_ALPHABET)
    assert s.link_token() != token
    slug = s.link_slug("mara-nov-2025")
    assert slug.startswith("mara-nov-2025-") and len(slug) == len("mara-nov-2025-") + 12


def test_title_from_name() -> None:
    assert s.title_from_name("mara-nov-2025") == "Mara Nov 2025"
    assert s.title_from_name("2025_10 Lisbon") == "2025 10 Lisbon"


def test_in_date_range() -> None:
    p = _p("a.JPG", 0)
    assert s.in_date_range(p, None, None)
    assert s.in_date_range(p, date(2025, 11, 9), date(2025, 11, 9))
    assert not s.in_date_range(p, date(2025, 11, 10), None)
    assert not s.in_date_range(_p("u.JPG", None), date(2025, 11, 1), None)


# --- selection.txt ---------------------------------------------------------------


def _gallery() -> s.Gallery:
    return s.Gallery(
        title="Mara, November",
        subtitle="8\u20139 November",
        cover="day2/IMG_2.JPG",
        captions=True,
        sections=[
            s.GallerySection(
                "Saturday 8 November · Talek",
                [
                    s.Item("day1/IMG_1.JPG", ""),
                    s.Item("day1/IMG_5.JPG", "A tent.", hidden=True),
                ],
            ),
            s.GallerySection(
                "Sunday 9 November · Ewangan, Narok West",
                [
                    s.Item("day2/IMG_2.JPG", "A leopard crosses the track."),
                    s.Item("day2/IMG 3 #2.JPG", "Lion #2 | the big one."),
                ],
            ),
        ],
        pool=[
            (
                s.Item("day2/IMG_4.JPG", "Same leopard."),
                "near-duplicate of day2/IMG_2.JPG",
            ),
            (s.Item("day2/IMG_6.JPG", ""), "over the 40-photo cap"),
        ],
    )


def test_selection_round_trip_keeps_hidden_photos_and_the_pool() -> None:
    g = _gallery()
    parsed, warnings = s.parse_selection(s.write_selection(g))
    assert warnings == []
    assert parsed == g  # an editor save (parse -> edit -> write) loses nothing
    assert [i.rel for i in parsed.items] == [
        "day1/IMG_1.JPG",
        "day2/IMG_2.JPG",
        "day2/IMG 3 #2.JPG",
    ]  # the page: hidden photos are not on it
    assert len(parsed.all_items) == 4


def test_selection_edits_drop_reorder_recaption_and_add_back() -> None:
    text = s.write_selection(_gallery())
    edited = (
        text.replace("day1/IMG_1.JPG\n", "# day1/IMG_1.JPG\n")  # dropped by hand
        .replace("# day1/IMG_5.JPG | A tent.  # hidden", "day1/IMG_5.JPG | A tent.")
        .replace("A leopard crosses the track.", "Our first leopard!")
        .replace("title: Mara, November", "title: Mara with the kids")
        .replace("captions: on", "captions: off")
        .replace("# day2/IMG_4.JPG", "day2/IMG_4.JPG")  # added back from the pool
    )
    parsed, _ = s.parse_selection(edited)
    assert parsed.title == "Mara with the kids" and parsed.captions is False
    assert [i.rel for i in parsed.sections[0].items] == ["day1/IMG_5.JPG"]  # restored
    items = parsed.sections[1].items
    assert [i.rel for i in items] == [
        "day2/IMG_2.JPG",
        "day2/IMG 3 #2.JPG",
        "day2/IMG_4.JPG",
    ]
    assert items[0].caption == "Our first leopard!"
    assert items[2].caption == "Same leopard."  # the "  # reason" note is stripped
    assert [i.rel for i, _ in parsed.pool] == ["day2/IMG_6.JPG"]


def test_selection_ignores_comments_that_are_not_photos() -> None:
    text = (
        s.SELECTION_HELP
        + "## Day\na.JPG\n# just a note  # hidden\n# notes.txt  # hidden\n"
    )
    parsed, _ = s.parse_selection(text)
    assert [i.rel for i in parsed.all_items] == ["a.JPG"]


def test_selection_lines_without_sections_or_captions() -> None:
    parsed, warnings = s.parse_selection("a.JPG\nb.JPG |\nc.JPG |   \na.JPG | again\n")
    assert [i.rel for i in parsed.items] == ["a.JPG", "b.JPG", "c.JPG"]
    assert all(i.caption == "" for i in parsed.items)
    assert len(warnings) == 1 and "a.JPG" in warnings[0]
    assert parsed.sections[0].heading == ""


def test_empty_sections_are_dropped() -> None:
    parsed, _ = s.parse_selection("## Empty\n## Full\nx.JPG\n")
    assert [sec.heading for sec in parsed.sections] == ["Full"]


def test_write_selection_file_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "work" / "selection.txt"
    s.write_selection_file(target, _gallery())
    assert s.parse_selection(target.read_text())[0] == _gallery()
    assert [f.name for f in target.parent.iterdir()] == [
        "selection.txt"
    ]  # no temp left


def test_gallery_from_plan_opens_on_the_strongest_frame() -> None:
    strongest = _p("tall.JPG", 0, aspect=0.56, user_rated=True)  # a human keep
    plan = s.Plan(
        [
            s.Section("Saturday", [_p("lodge.JPG", -600, aspect=1.78)]),
            s.Section(
                "Sunday",
                [
                    strongest,
                    _p("wide.JPG", 5, subjects="One lion walking.", scene="Grass."),
                ],
            ),
        ],
        not_picked=[(_p("cut.JPG", 9), "over the 3-photo cap")],
    )
    g = s.gallery_from_plan(plan, title="T")
    assert g.cover == "tall.JPG"  # never an inferior landscape just to have a wide one
    assert g.captions is False
    assert g.items[2].caption == "A lion walking."  # alt text is always there
    assert g.pool == [(s.Item("cut.JPG", ""), "over the 3-photo cap")]
    assert s.pick_cover(s.Plan([])) == ""


def test_pick_with_closer_keeps_the_best_late_frame() -> None:
    early = [_p(f"e{i}.JPG", i * 10) for i in range(6)]
    late = [_p("late-soft.JPG", 70, focus="soft"), _p("late-great.JPG", 75)]
    for p in late:
        p.composition = "acceptable"
    late[1].focus = "sharp"
    chosen = s.pick_with_closer([*early, *late], 3)
    assert len(chosen) == 3 and chosen[-1].rel == "late-great.JPG"
    assert s.pick_with_closer(early, 10) == early  # nothing to cap


def test_auto_title_from_place_and_month() -> None:
    mara = [_p(f"m{i}.JPG", i) for i in range(5)]
    assert s.auto_title(mara, "Folder") == "Narok · November 2025"
    lisbon = "Alfama, Santa Maria Maior, Lisboa, Portugal"
    algarve = "Odeceixe, Aljezur, Faro, Portugal"
    mixed = [
        _p("a", 0, place=lisbon),
        _p("b", 1, place=algarve),
        _p("c", 2, place=lisbon),
    ]
    assert s.trip_place(mixed) == "Portugal"
    no_gps = [_p("x", 0, place=""), _p("y", 1, place="")]
    assert s.auto_title(no_gps, "Moto Spain") == "Moto Spain · November 2025"
    assert s.auto_title(no_gps, "2025 11 Mara") == "Mara · November 2025"
    assert s.auto_title(no_gps, "Mara July") == "Mara · November 2025"
    assert s.auto_title(no_gps, "2025 11") == "November 2025"
    assert s.auto_title([_p("u", None, place="")], "Moto Spain") == "Moto Spain"
    assert (
        s.month_label([date(2025, 10, 30), date(2025, 11, 2)])
        == "October \u2013 November 2025"
    )
    assert s.month_label([date(2025, 12, 30), date(2026, 1, 2)]) == (
        "December 2025 \u2013 January 2026"
    )
    assert (
        s.day_range_label([date(2025, 11, 8), date(2025, 11, 9)]) == "8\u20139 November"
    )
    assert s.day_range_label([date(2025, 11, 9)]) == "Sunday 9 November"
    assert s.day_range_label([date(2025, 10, 30), date(2025, 11, 2)]) == (
        "30 October \u2013 2 November"
    )


def test_resolve_item_refuses_paths_outside_the_folder(tmp_path: Path) -> None:
    shoot = tmp_path / "shoot"
    (shoot / "day1").mkdir(parents=True)
    (shoot / "day1" / "a.JPG").write_bytes(b"x")
    (shoot / "notes.txt").write_text("x")
    (tmp_path / "secret.jpg").write_bytes(b"x")
    assert s.resolve_item(shoot, "day1/a.JPG") == (shoot / "day1" / "a.JPG").resolve()
    assert s.resolve_item(shoot, "../secret.jpg") is None
    assert s.resolve_item(shoot, str(tmp_path / "secret.jpg")) is None
    assert s.resolve_item(shoot, "notes.txt") is None  # not an image
    assert s.resolve_item(shoot, "missing.JPG") is None
