#!/usr/bin/env python3
"""
fdx share — turn an indexed folder of photos into a gallery link.

The sidecars do the curation: keep-rated photos only (a `user_rating` wins
over the model's), one frame per burst, near-duplicates dropped, a cap
spread over every day and place, the strongest frame opening. The pick lives
in a plain-text `selection.txt` that the browser editor (`share_edit`)
rewrites on every change and a person can edit by hand. Originals are only
read; galleries get web copies with GPS stripped.

This module is the pure half (selection, ordering, the selection file) and
the CLI; pixels and HTML live in `share_site`, the editor in `share_edit`,
passcodes in `share_lock`, uploads in `share_publish`.
"""

from __future__ import annotations

import argparse
import re
import secrets
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from framedex import grouping
from framedex.images import IMAGE_EXTENSIONS
from framedex.parsing import is_user_rated

# Default size of a gallery: enough to tell a weekend, few enough that family
# actually scroll to the end.
DEFAULT_MAX_PHOTOS = 40
# Photos taken before this hour belong to the previous day's section (the
# campfire at 00:30 is still Saturday night).
DAY_START_HOUR = 4
# Within a section, prefer frames at least this far apart before taking a
# second frame from the same minute (a 20-minute leopard sighting should not
# fill a section's quota from one pass of the animal).
SPREAD_GAP_SEC = 90.0
# Near-duplicates: two candidates taken within NEAR_DUP_WINDOW_SEC, with the
# same shape (aspect ratios within NEAR_DUP_MAX_RATIO_DIFF), whose 16x16
# grayscale thumbnails correlate at >= NEAR_DUP_MIN_CORR and differ by <=
# NEAR_DUP_MAX_MEAN_DIFF (0-255 levels) on average are the same picture; the
# lower-scored one is dropped. Tuned on phone safari frames: repeats of one
# framing land at 0.6-0.93 / 6-21, a new framing or subject at <= 0.48 or
# >= 27. (A 64-bit difference hash was tried first: on flat savanna and sky,
# repeats and unrelated frames both came out 20-35 bits apart.)
NEAR_DUP_WINDOW_SEC = 600.0
NEAR_DUP_MAX_RATIO_DIFF = 0.1
NEAR_DUP_MIN_CORR = 0.6
NEAR_DUP_MAX_MEAN_DIFF = 24.0
SIGNATURE_SIZE = 16
# Captions: one line from the sidecar's **Subjects:** sentence (what is in the
# frame), or its **Scene:** sentence (the setting) when no subject is named.
CAPTION_MAX_CHARS = 110
# A frame at least this wide (width / height) wins a tie for the opening
# photo, and opens a gallery whose selection.txt names no cover.
COVER_MIN_RATIO = 1.2
# A trip is titled after the region (or country) this share of its geotagged
# photos have in common.
TITLE_PLACE_SHARE = 0.8
# Quality points from the sidecar's technical block, used to rank candidates
# when the cap bites. A human keep outranks any model assessment.
FOCUS_POINTS = {"sharp": 3, "acceptable": 1}
EXPOSURE_POINTS = {"strong": 2, "adequate": 1}
COMPOSITION_POINTS = {"strong": 3, "acceptable": 1}
USER_KEEP_BONUS = 10
# Unguessable link suffix: 12 characters from a 31-symbol alphabet (~59 bits).
LINK_TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
LINK_TOKEN_LEN = 12
# Words that mark a place-name part as an administrative unit, not a place
# anyone would name ("Siana ward"): such a part never leads a heading.
ADMIN_WORDS = {"ward", "county", "district", "municipality", "province", "constituency"}
EN_DASH = "\u2013"
SELECTION_FILE = "selection.txt"
SHARE_DIR = "_share"

_EPOCH = datetime(1970, 1, 1)
_POSTCODE_RE = re.compile(r"\d")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_INLINE_COMMENT = "  #"  # two spaces then '#': the rest of a photo line is a note
_SENTENCE_END_RE = re.compile(r"(?<=[a-z0-9)])\.\s+(?=[A-Z])")
_PAREN_RE = re.compile(r"\s*\([^()]*\)")
_ONE_RE = re.compile(r"^One ([A-Za-z])")
_SUBJECTS_RE = re.compile(r"\*\*Subjects:\*\*\s*(.+)")


# ---------------------------------------------------------------------------
# Photos from sidecar records
# ---------------------------------------------------------------------------


@dataclass
class Photo:
    """One indexed still, reduced to what selection and ordering need."""

    rel: str  # path relative to the share folder, POSIX separators
    path: Path  # the original (only ever read)
    taken: datetime | None = None
    place: str = ""  # reverse-geocoded display name, "" without GPS
    lat: float | None = None
    lon: float | None = None
    rating: str = ""  # effective rating: a valid user_rating wins
    user_rated: bool = False
    focus: str = ""
    exposure: str = ""
    composition: str = ""
    scene: str = ""
    camera: str = ""  # "Make|Model", "" when unknown
    group_key: str | None = None  # "<dir>/<group id>" for burst / RAW+JPEG
    group_primary: bool = False
    subjects: str = ""  # the sidecar's **Subjects:** sentence
    aspect: float | None = None  # upright width / height, from the sidecar

    @property
    def score(self) -> int:
        points = (
            FOCUS_POINTS.get(self.focus, 0)
            + EXPOSURE_POINTS.get(self.exposure, 0)
            + COMPOSITION_POINTS.get(self.composition, 0)
        )
        if self.user_rated and self.rating == "keep":
            points += USER_KEEP_BONUS
        return points

    @property
    def seconds(self) -> float | None:
        """Capture time as seconds on a fixed naive epoch (only differences
        matter; the host's timezone and DST never enter)."""
        return None if self.taken is None else (self.taken - _EPOCH).total_seconds()


def parse_taken(value: Any) -> datetime | None:
    """A sidecar `creation_time` ('2025-11-09T09:48:12', camera-local) as a
    naive datetime; None when absent or unparsable. Any zone suffix is dropped:
    a trip's photos share one, and the day a photo belongs to is local."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"(Z|[+-]\d{2}:?\d{2})$", "", value.strip())
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _str(rec: dict[str, Any], key: str) -> str:
    v = rec.get(key)
    return v.strip() if isinstance(v, str) else ""


def _float(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def upright_aspect(dimensions: Any, orientation: Any) -> float | None:
    """Upright width / height from the sidecar's 'WxH' and EXIF orientation
    text ('Rotate 90 CW' swaps the sides). None when the size is unknown."""
    if not isinstance(dimensions, str) or "x" not in dimensions:
        return None
    w_text, _, h_text = dimensions.partition("x")
    try:
        w, h = int(w_text), int(h_text)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    if "90" in str(orientation or "") or "270" in str(orientation or ""):
        w, h = h, w
    return w / h


def photo_from_record(rec: dict[str, Any], folder: Path) -> Photo | None:
    """Build a Photo from a `query.run_query` record (absolute `path`,
    `effective_rating`, `_scene`, and `_subjects` when the caller read it).
    None for videos, Photos-library records without a file on disk, and
    paths outside `folder`."""
    if (rec.get("media_type") or "video") != "image":
        return None
    p = rec.get("path")
    if not isinstance(p, str) or not p:
        return None
    path = Path(p)
    try:
        rel = path.relative_to(folder).as_posix()
    except ValueError:
        return None
    technical = rec.get("technical")
    tech = technical if isinstance(technical, dict) else {}
    location = rec.get("location")
    loc = location if isinstance(location, dict) else {}
    camera = rec.get("camera")
    cam = camera if isinstance(camera, dict) else {}
    make = str(cam.get("make") or "").strip()
    model = str(cam.get("model") or "").strip()
    group = rec.get("group")
    group_key = None
    group_primary = False
    if isinstance(group, dict) and isinstance(group.get("id"), str):
        group_key = f"{PurePosixPath(rel).parent}/{group['id']}"
        group_primary = group.get("primary") is True
    rating = rec.get("effective_rating")
    return Photo(
        rel=rel,
        path=path,
        taken=parse_taken(rec.get("creation_time")),
        place=str(loc.get("place") or "").strip(),
        lat=_float(loc.get("lat")),
        lon=_float(loc.get("lon")),
        rating=rating if isinstance(rating, str) else "",
        user_rated=is_user_rated(rec),
        focus=str(tech.get("focus") or ""),
        exposure=str(tech.get("exposure") or ""),
        composition=str(tech.get("composition") or ""),
        scene=_str(rec, "_scene"),
        subjects=_str(rec, "_subjects"),
        camera=f"{make}|{model}" if (make or model) else "",
        group_key=group_key,
        group_primary=group_primary,
        aspect=upright_aspect(rec.get("dimensions"), cam.get("orientation")),
    )


# ---------------------------------------------------------------------------
# Selection: rating, one per burst, near-duplicates, the cap
# ---------------------------------------------------------------------------


def _best(photos: list[Photo]) -> Photo:
    """Highest score; ties go to the earliest frame, then the path."""
    return min(photos, key=lambda p: (-p.score, p.seconds or 0.0, p.rel))


def collapse_bursts(photos: list[Photo]) -> tuple[list[Photo], dict[str, str]]:
    """One frame per burst / RAW+JPEG pair. Groups recorded in the sidecars
    (`group` block) keep a human-kept member when there is one, else the
    indexer's primary (the sharpest frame), else the best-scored. Photos
    indexed without grouping are chained with `grouping.detect_bursts` on
    their sidecar timestamps and cameras; the best-scored frame stands for
    the chain. Returns (kept, {dropped rel: kept rel})."""
    dropped: dict[str, str] = {}
    by_group: dict[str, list[Photo]] = {}
    loose: list[Photo] = []
    for p in photos:
        if p.group_key is not None:
            by_group.setdefault(p.group_key, []).append(p)
        else:
            loose.append(p)

    kept: list[Photo] = []
    for members in by_group.values():
        human = [m for m in members if m.user_rated]
        primaries = [m for m in members if m.group_primary]
        rep = _best(human) if human else (primaries[0] if primaries else _best(members))
        kept.append(rep)
        dropped.update({m.rel: rep.rel for m in members if m is not rep})

    by_path = {p.path: p for p in loose}
    meta = {
        p.path: grouping.GroupMeta(timestamp=p.seconds, camera=p.camera) for p in loose
    }
    bursts, rest = grouping.detect_bursts([grouping.Unit(p.path) for p in loose], meta)
    for burst in bursts:
        members = [by_path[u.primary] for u in burst.units]
        rep = _best(members)
        kept.append(rep)
        dropped.update({m.rel: rep.rel for m in members if m is not rep})
    kept += [by_path[u.primary] for u in rest]
    return sorted(kept, key=time_order), dropped


@dataclass(frozen=True)
class Signature:
    """A photo's look at thumbnail scale: upright aspect ratio plus a
    SIGNATURE_SIZE x SIGNATURE_SIZE grayscale thumbnail, row-major 0-255."""

    aspect: float
    pixels: tuple[int, ...]


def compare(a: Signature, b: Signature) -> tuple[float, float]:
    """(correlation, mean absolute difference) of two signatures' pixels.
    Correlation is 0 when either thumbnail is flat (no structure to match)."""
    n = len(a.pixels)
    if n == 0 or n != len(b.pixels):
        return 0.0, 255.0
    mean_a, mean_b = sum(a.pixels) / n, sum(b.pixels) / n
    da = [x - mean_a for x in a.pixels]
    db = [y - mean_b for y in b.pixels]
    var_a, var_b = sum(x * x for x in da), sum(y * y for y in db)
    mad = sum(abs(x - y) for x, y in zip(a.pixels, b.pixels, strict=True)) / n
    if var_a < n or var_b < n:  # std under one level: nothing to correlate
        return 0.0, mad
    corr = sum(x * y for x, y in zip(da, db, strict=True)) / (var_a * var_b) ** 0.5
    return corr, mad


def same_picture(a: Signature, b: Signature) -> bool:
    """Would a viewer call these the same photo? Same shape, and thumbnails
    that correlate strongly and differ little (see NEAR_DUP_* constants)."""
    if abs(a.aspect - b.aspect) > NEAR_DUP_MAX_RATIO_DIFF * max(a.aspect, b.aspect):
        return False
    corr, mad = compare(a, b)
    return corr >= NEAR_DUP_MIN_CORR and mad <= NEAR_DUP_MAX_MEAN_DIFF


def near_duplicates(
    photos: list[Photo],
    signatures: dict[str, Signature],
    *,
    window_sec: float = NEAR_DUP_WINDOW_SEC,
) -> dict[str, str]:
    """{dropped rel: kept rel} for candidates that repeat a better one: both
    dated, taken within `window_sec`, and `same_picture`. Better = higher
    score, then earlier. A photo without a signature or a date is never a
    duplicate (no guessing)."""
    order = sorted(photos, key=lambda p: (-p.score, p.seconds or 0.0, p.rel))
    accepted: list[Photo] = []
    dropped: dict[str, str] = {}
    for p in order:
        sig, t = signatures.get(p.rel), p.seconds
        twin = None
        if sig is not None and t is not None:
            for a in accepted:
                other, ta = signatures.get(a.rel), a.seconds
                if (
                    other is not None
                    and ta is not None
                    and abs(t - ta) <= window_sec
                    and same_picture(sig, other)
                ):
                    twin = a
                    break
        if twin is None:
            accepted.append(p)
        else:
            dropped[p.rel] = twin.rel
    return dropped


def allocate(sizes: list[int], cap: int) -> list[int]:
    """Split `cap` picks across sections of `sizes` candidates: everything
    when it fits; otherwise one per section first (every day/place shows
    up), the rest in proportion to section size by largest remainder. With
    more sections than the cap, the largest sections get one each."""
    total = sum(sizes)
    if cap <= 0 or total == 0:
        return [0] * len(sizes)
    if total <= cap:
        return list(sizes)
    nonempty = [i for i, s in enumerate(sizes) if s > 0]
    out = [0] * len(sizes)
    if len(nonempty) >= cap:
        for i in sorted(nonempty, key=lambda i: (-sizes[i], i))[:cap]:
            out[i] = 1
        return out
    for i in nonempty:
        out[i] = 1
    spare = cap - len(nonempty)
    room = {i: sizes[i] - 1 for i in nonempty}
    pool = sum(room.values())
    exact = {i: spare * room[i] / pool for i in nonempty} if pool else {}
    for i, share in exact.items():
        out[i] += int(share)
    leftover = cap - sum(out)
    by_remainder = sorted(exact, key=lambda i: (-(exact[i] - int(exact[i])), i))
    for i in by_remainder[:leftover]:
        out[i] += 1
    return out


def pick_spread(
    photos: list[Photo], k: int, *, gap_sec: float = SPREAD_GAP_SEC
) -> list[Photo]:
    """The best `k` photos, preferring frames at least `gap_sec` from every
    frame already picked; a second pass fills any shortfall by score alone.
    Returned in time order."""
    if k >= len(photos):
        return sorted(photos, key=time_order)
    ranked = sorted(photos, key=lambda p: (-p.score, p.seconds or 0.0, p.rel))
    chosen: list[Photo] = []
    for p in ranked:
        if len(chosen) == k:
            break
        t = p.seconds
        if t is None or all(
            c.seconds is None or abs(t - c.seconds) >= gap_sec for c in chosen
        ):
            chosen.append(p)
    taken_ids = {id(c) for c in chosen}
    for p in ranked:
        if len(chosen) == k:
            break
        if id(p) not in taken_ids:
            chosen.append(p)
    return sorted(chosen, key=time_order)


def pick_with_closer(photos: list[Photo], k: int) -> list[Photo]:
    """`pick_spread`, but when the cap bites the best frame of the section's
    last quarter (by time) always makes it: the gallery ends on a strong
    frame from the end of the trip, not on whatever the spacing rule left."""
    dated = sorted((p for p in photos if p.taken is not None), key=time_order)
    if k < 2 or k >= len(photos) or not dated:
        return pick_spread(photos, k)
    late = dated[int(len(dated) * 0.75) :] or dated[-1:]
    wide = [p for p in late if (p.aspect or 0.0) >= COVER_MIN_RATIO]
    closer = _best(wide or late)
    rest = pick_spread([p for p in photos if p is not closer], k - 1)
    return sorted([*rest, closer], key=time_order)


@dataclass
class Section:
    heading: str
    photos: list[Photo]


@dataclass
class Plan:
    sections: list[Section]  # the picked photos, in gallery order
    not_picked: list[tuple[Photo, str]] = field(default_factory=list)  # (photo, why)

    @property
    def photos(self) -> list[Photo]:
        return [p for s in self.sections for p in s.photos]


def plan_gallery(
    photos: list[Photo],
    *,
    max_photos: int = DEFAULT_MAX_PHOTOS,
    include_review: bool = False,
    signer: Callable[[list[Photo]], dict[str, Signature]] | None = None,
) -> Plan:
    """The whole pick: rating filter, one per burst, near-duplicates (when a
    `signer` is given: it maps candidates to thumbnail Signatures), sections
    by day and place, then the cap spread across sections."""
    wanted = {"keep", "review"} if include_review else {"keep"}
    rated = [p for p in photos if p.rating in wanted]
    candidates, _burst_alternates = collapse_bursts(rated)
    not_picked: list[tuple[Photo, str]] = []
    if signer is not None and len(candidates) > 1:
        dups = near_duplicates(candidates, signer(candidates))
        by_rel = {p.rel: p for p in candidates}
        not_picked += [
            (by_rel[rel], f"near-duplicate of {kept}") for rel, kept in dups.items()
        ]
        candidates = [p for p in candidates if p.rel not in dups]
    sections = make_sections(candidates)
    quotas = allocate([len(s.photos) for s in sections], max_photos)
    picked_sections: list[Section] = []
    for n, (section, quota) in enumerate(zip(sections, quotas, strict=True)):
        last = n == len(sections) - 1
        chosen = (
            pick_with_closer(section.photos, quota)
            if last
            else pick_spread(section.photos, quota)
        )
        chosen_ids = {id(p) for p in chosen}
        not_picked += [
            (p, f"over the {max_photos}-photo cap")
            for p in section.photos
            if id(p) not in chosen_ids
        ]
        if chosen:
            picked_sections.append(Section(section.heading, chosen))
    not_picked.sort(key=lambda t: time_order(t[0]))
    return Plan(picked_sections, not_picked)


# ---------------------------------------------------------------------------
# Ordering: time, sectioned by day and place
# ---------------------------------------------------------------------------


def time_order(p: Photo) -> tuple[bool, datetime, str]:
    """Dated photos by capture time, undated after them by path."""
    return (p.taken is None, p.taken or datetime.min, p.rel)


def trip_day(taken: datetime) -> date:
    """The day a photo belongs to: before DAY_START_HOUR counts as the
    previous day."""
    return (taken - timedelta(hours=DAY_START_HOUR)).date()


def place_parts(place: str) -> list[str]:
    """A display name's components, fine to coarse, without postcodes."""
    parts = [p.strip() for p in place.split(",")]
    return [p for p in parts if p and not _POSTCODE_RE.search(p)]


def place_area(place: str) -> str:
    """The section key for a place: the third-from-last component (a
    municipality or sub-county in most Nominatim names), so a game drive or a
    walk through town stays one section while a move to the next town starts
    a new one. Short names fall back to their first component."""
    parts = place_parts(place)
    if not parts:
        return ""
    return parts[-3] if len(parts) >= 3 else parts[0]


def place_label(place: str) -> str:
    """Short human label: the most specific component plus the area when they
    differ ("Ewangan, Narok West"). A most-specific part that is only an
    administrative unit ("Siana ward") is left out."""
    parts = place_parts(place)
    if not parts:
        return ""
    area = place_area(place)
    first = parts[0]
    if first == area or set(first.lower().split()) & ADMIN_WORDS:
        return area
    return f"{first}, {area}"


def day_label(d: date, *, year: bool = False) -> str:
    """'9 November' (or '9 November 2025' in a gallery that crosses a year)."""
    return f"{d.day} {d:%B}" + (f" {d.year}" if year else "")


def _most_common(values: Iterable[str]) -> str:
    counts = Counter(v for v in values if v)
    if not counts:
        return ""
    best = max(counts.values())
    return next(v for v in counts if counts[v] == best)  # first seen wins ties


def make_sections(photos: list[Photo]) -> list[Section]:
    """Time-ordered photos broken into runs of (trip day, place area). A
    photo without GPS stays in the current run of its day. A one-photo run is
    folded into its neighbour on the same day (a single frame at the fuel
    stop is not a chapter). Heading: '<day> · <place label>'."""
    ordered = sorted(photos, key=time_order)
    runs: list[tuple[date | None, list[Photo]]] = []
    current: tuple[date | None, str] | None = None
    for p in ordered:
        day = trip_day(p.taken) if p.taken else None
        area = place_area(p.place)
        if current is not None and current[0] == day and (not area or not current[1]):
            # Same day and one side has no place: keep the run, learn its area.
            if area and not current[1]:
                current = (day, area)
            runs[-1][1].append(p)
            continue
        if current is None or current != (day, area):
            runs.append((day, [p]))
            current = (day, area)
        else:
            runs[-1][1].append(p)

    merged: list[tuple[date | None, list[Photo]]] = []
    for i, (day, items) in enumerate(runs):
        if len(items) == 1 and merged and merged[-1][0] == day:
            merged[-1][1].extend(items)
        elif (
            len(items) == 1
            and i + 1 < len(runs)
            and runs[i + 1][0] == day
            and len(runs[i + 1][1]) > 1
        ):
            runs[i + 1][1].insert(0, items[0])
        else:
            merged.append((day, items))

    years = {day.year for day, _ in merged if day}
    sections: list[Section] = []
    for day, items in merged:
        when = day_label(day, year=len(years) > 1) if day else "Undated"
        where = _most_common(place_label(p.place) for p in items)
        heading = f"{when} · {where}" if where else when
        if sections and sections[-1].heading == heading:
            sections[-1].photos.extend(items)
        else:
            sections.append(Section(heading, items))
    return sections


def _names_a_subject(subjects: str) -> bool:
    """False for a Subjects line that only says nothing is there ("No
    people. The frame centers on ...")."""
    head = subjects.strip().lower()
    return bool(head) and not head.startswith(
        ("no ", "none", "unclear", "there are no")
    )


def caption_for(
    scene: str, subjects: str = "", max_chars: int = CAPTION_MAX_CHARS
) -> str:
    """One plain line: the first sentence of Subjects (what is in the frame)
    or, when it names nothing, of Scene (the setting). Parentheticals go,
    a leading "One" becomes "A"/"An", and a long sentence is cut at its first
    clause break that leaves a full thought, else at a word."""
    text = subjects if _names_a_subject(subjects) else scene
    text = " ".join(text.split())
    text = _SENTENCE_END_RE.split(text, maxsplit=1)[0]
    text = _PAREN_RE.sub("", text)
    text = text.split("(", 1)[0]  # an unbalanced "(" never ends a caption
    text = _ONE_RE.sub(
        lambda m: ("An " if m.group(1) in "aeiouAEIOU" else "A ") + m.group(1), text
    )
    text = text.strip().rstrip(".,;:").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text + "."
    for sep in (", ", "; ", " - ", " with ", " under ", " against "):
        i = text.find(sep, 30)
        if 0 < i <= max_chars:
            return text[:i].rstrip(",;") + "."
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(",;") + "…"


def date_range_label(days: list[date]) -> str:
    """'9 November 2025', '8-9 November 2025', '30 October - 2 November 2025',
    '30 December 2025 - 2 January 2026' (with en dashes)."""
    if not days:
        return ""
    a, b = min(days), max(days)
    if a == b:
        return f"{a.day} {a:%B %Y}"
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}{EN_DASH}{b.day} {b:%B %Y}"
    if a.year == b.year:
        return f"{a.day} {a:%B} {EN_DASH} {b.day} {b:%B %Y}"
    return f"{a.day} {a:%B %Y} {EN_DASH} {b.day} {b:%B %Y}"


def region_label(photos: list[Photo]) -> str:
    """The most common last-two place components ('Narok, Kenya')."""
    return _most_common(", ".join(place_parts(p.place)[-2:]) for p in photos)


def trip_place(photos: list[Photo]) -> str:
    """The place a trip is about: the region (second-to-last name part) when
    >= 80% of geotagged photos share it, else the country when they share
    that ('Narok'; 'Portugal' for Lisbon + the Algarve); '' otherwise."""
    geo = [place_parts(p.place) for p in photos if p.place]
    geo = [parts for parts in geo if parts]
    for depth in (2, 1):
        names = [parts[-depth] for parts in geo if len(parts) >= depth]
        common = _most_common(names)
        if common and names.count(common) >= TITLE_PLACE_SHARE * len(geo):
            return common
    return ""


def month_label(days: list[date]) -> str:
    """'November 2025', 'October - November 2025', 'December 2025 - January
    2026' (with en dashes); '' without dates."""
    if not days:
        return ""
    a, b = min(days), max(days)
    if (a.year, a.month) == (b.year, b.month):
        return f"{a:%B %Y}"
    if a.year == b.year:
        return f"{a:%B} {EN_DASH} {b:%B %Y}"
    return f"{a:%B %Y} {EN_DASH} {b:%B %Y}"


def day_range_label(days: list[date]) -> str:
    """The subtitle under a 'Place · Month Year' title: the days without the
    year ('8-9 November', '30 October - 2 November'), the year only when the
    trip crosses one."""
    if not days:
        return ""
    a, b = min(days), max(days)
    if a.year != b.year:
        return date_range_label(days)
    if a == b:
        return f"{a.day} {a:%B}"
    if a.month == b.month:
        return f"{a.day}{EN_DASH}{b.day} {b:%B}"
    return f"{a.day} {a:%B} {EN_DASH} {b.day} {b:%B}"


def auto_title(photos: list[Photo], fallback: str) -> str:
    """'Narok · November 2025' from the photos' places and dates. Without GPS
    the folder name stands in for the place, minus any dates in it
    ('2025-11-mara' -> 'Mara · November 2025'); without dates, just the
    place or the name."""
    place = trip_place(photos) or strip_dates(fallback)
    when = month_label([trip_day(p.taken) for p in photos if p.taken])
    if place and when:
        return f"{place} · {when}"
    return place or when or fallback


_MONTHS = {
    m.lower()
    for i in range(1, 13)
    for m in (date(2000, i, 1).strftime("%B"), date(2000, i, 1).strftime("%b"))
}


def strip_dates(name: str) -> str:
    """A folder name without its date parts: years, month numbers, month
    names ('2025_11 Mara July' -> 'Mara')."""
    words = [
        w
        for w in name.split()
        if not w.isdigit() and w.lower().rstrip(".") not in _MONTHS
    ]
    return " ".join(words)


def title_from_name(name: str) -> str:
    words = [w for w in re.split(r"[-_\s]+", name) if w]
    return " ".join(w if any(c.isupper() for c in w) else w.capitalize() for w in words)


# ---------------------------------------------------------------------------
# selection.txt: the editable plan
# ---------------------------------------------------------------------------


@dataclass
class Item:
    rel: str
    caption: str = ""  # alt text always; shown on the page only with captions: on
    hidden: bool = False  # kept in place (the editor can restore it), not shown


@dataclass
class GallerySection:
    heading: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Gallery:
    title: str
    subtitle: str = ""
    cover: str = ""  # rel of the opening photo; "" = first wide one, "none" = no cover
    captions: bool = False
    sections: list[GallerySection] = field(default_factory=list)
    pool: list[tuple[Item, str]] = field(default_factory=list)  # not picked, why

    @property
    def items(self) -> list[Item]:
        """The photos on the page, in order."""
        return [i for s in self.sections for i in s.items if not i.hidden]

    @property
    def all_items(self) -> list[Item]:
        """Shown and hidden photos, in order."""
        return [i for s in self.sections for i in s.items]


SELECTION_HELP = """\
# fdx share selection. Edit it (or use the browser editor), then re-run
# fdx share on the folder to see it.
#   drop a photo     delete its line, or put "# " in front of it
#   reorder          move lines; a "## " line starts a section
#   add one back     remove the "# " from a line under "Not picked"
#   caption          the text after " | ": alt text, and shown under the photo
#                    when captions: on
#   title, cover     edit the title:/subtitle:/cover: lines (cover: none for
#                    no opening photo)
# Text after two spaces and "#" on a photo line is a note, never shown.
"""
POOL_MARKER = "# ---- Not picked (remove the leading '# ' to add one) ----"
HIDDEN_NOTE = "hidden"


def gallery_from_plan(plan: Plan, *, title: str, captions: bool = False) -> Gallery:
    """The plan as an editable Gallery: the pick in sections, the
    not-picked candidates in its pool, every photo with its one-line
    caption (alt text even when captions are off)."""
    photos = plan.photos
    subtitle = day_range_label([trip_day(p.taken) for p in photos if p.taken])

    def item(p: Photo) -> Item:
        return Item(p.rel, caption_for(p.scene, p.subjects))

    sections = [
        GallerySection(s.heading, [item(p) for p in s.photos]) for s in plan.sections
    ]
    pool = [(item(p), why) for p, why in plan.not_picked]
    return Gallery(title, subtitle, pick_cover(plan), captions, sections, pool)


def pick_cover(plan: Plan) -> str:
    """The opening photo: the strongest frame of the whole pick, whatever its
    shape (it opens uncropped); ties go to a wide frame, then the bigger
    section, then the earlier frame. "" for an empty plan."""
    if not plan.photos:
        return ""
    size = {id(p): len(s.photos) for s in plan.sections for p in s.photos}

    def rank(p: Photo) -> tuple[int, bool, int, float, str]:
        wide = (p.aspect or 0.0) >= COVER_MIN_RATIO
        return (-p.score, not wide, -size[id(p)], p.seconds or 0.0, p.rel)

    return min(plan.photos, key=rank).rel


def _photo_line(item: Item) -> str:
    return f"{item.rel} | {item.caption}" if item.caption else item.rel


def write_selection(gallery: Gallery) -> str:
    lines = [
        SELECTION_HELP,
        f"title: {gallery.title}",
        f"subtitle: {gallery.subtitle}",
        f"cover: {gallery.cover}",
        f"captions: {'on' if gallery.captions else 'off'}",
    ]
    for s in gallery.sections:
        lines += ["", f"## {s.heading}"] if s.heading else [""]
        lines += [
            f"# {_photo_line(i)}{_INLINE_COMMENT} {HIDDEN_NOTE}"
            if i.hidden
            else _photo_line(i)
            for i in s.items
        ]
    if gallery.pool:
        lines += ["", POOL_MARKER]
        lines += [
            f"# {_photo_line(i)}{_INLINE_COMMENT} {why}" for i, why in gallery.pool
        ]
    return "\n".join(lines) + "\n"


def write_selection_file(path: Path, gallery: Gallery) -> None:
    """Write selection.txt atomically (an interrupted save never leaves half
    a file where the whole pick used to be)."""
    from framedex.pipeline import atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, write_selection(gallery))


def _parse_photo(body: str) -> tuple[str, str, str]:
    """'rel | caption  # note' -> (rel, caption, note)."""
    text, _, note = body.partition(_INLINE_COMMENT)
    rel, _, caption = text.strip().partition(" | ")
    return rel.strip().removesuffix(" |").strip(), caption.strip(), note.strip()


def parse_selection(text: str) -> tuple[Gallery, list[str]]:
    """Parse selection.txt back into a Gallery, losing nothing write_selection
    wrote: hidden photos keep their place, the Not picked pool keeps its
    reasons. Returns (gallery, warnings); a repeated photo keeps its first
    position and is reported."""
    gallery = Gallery(title="")
    warnings: list[str] = []
    seen: set[str] = set()
    in_pool = False

    def add(n: int, item: Item) -> bool:
        if item.rel in seen:
            warnings.append(f"line {n}: {item.rel} is listed twice; keeping the first")
            return False
        seen.add(item.rel)
        return True

    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# ----"):
            in_pool = True
            continue
        if line.startswith("## "):
            gallery.sections.append(GallerySection(line[3:].strip()))
            continue
        if line.startswith("#"):
            # A commented photo line is either in the pool (after the marker)
            # or hidden in place (its note says so); any other comment, or a
            # line a person commented out to drop the photo, is skipped.
            rel, caption, note = _parse_photo(line[1:])
            if Path(rel).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            hidden = not in_pool and note == HIDDEN_NOTE and bool(gallery.sections)
            if (in_pool or hidden) and add(n, Item(rel, caption, hidden)):
                if hidden:
                    gallery.sections[-1].items.append(Item(rel, caption, hidden=True))
                else:
                    gallery.pool.append((Item(rel, caption), note))
            continue
        if not gallery.sections and line.startswith(("title:", "subtitle:", "cover:")):
            key, _, value = line.partition(":")
            setattr(gallery, key, value.strip())
            continue
        if not gallery.sections and line.startswith("captions:"):
            gallery.captions = line.partition(":")[2].strip().lower() == "on"
            continue
        rel, caption, _ = _parse_photo(raw)
        if not rel or not add(n, Item(rel, caption)):
            continue
        if not gallery.sections:
            gallery.sections.append(GallerySection(""))
        gallery.sections[-1].items.append(Item(rel, caption))
    gallery.sections = [s for s in gallery.sections if s.items]
    return gallery, warnings


def resolve_item(folder: Path, rel: str) -> Path | None:
    """The original for a selection line, or None when it is missing, not an
    image, or escapes the folder (selection.txt is hand-edited: a line must
    never publish a file from outside the shoot)."""
    if Path(rel).is_absolute():
        return None
    try:
        path = (folder / rel).resolve()
    except (OSError, ValueError):
        return None
    root = folder.resolve()
    if root not in path.parents or not path.is_file():
        return None
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return path


# ---------------------------------------------------------------------------
# Names and links
# ---------------------------------------------------------------------------


def slugify(text: str, max_len: int = 40) -> str:
    ascii_text = text.lower().encode("ascii", "ignore").decode()
    slug = _SLUG_RE.sub("-", ascii_text).strip("-")
    return slug[:max_len].rstrip("-") or "gallery"


def link_token(n: int = LINK_TOKEN_LEN) -> str:
    return "".join(secrets.choice(LINK_TOKEN_ALPHABET) for _ in range(n))


def link_slug(name: str) -> str:
    """'<trip-slug>-<random>': readable, but not guessable or enumerable."""
    return f"{slugify(name)}-{link_token()}"


def in_date_range(p: Photo, start: date | None, end: date | None) -> bool:
    """--from/--to on trip days; an undated photo matches only an open range."""
    if start is None and end is None:
        return True
    if p.taken is None:
        return False
    d = trip_day(p.taken)
    return (start is None or d >= start) and (end is None or d <= end)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_photos(folder: Path) -> list[Photo]:
    from framedex.query import Filters, run_query

    result = run_query(folder, Filters(media="image"))
    out = []
    for rec in result.records:
        rec["_subjects"] = _subjects_sentence(Path(str(rec.get("_sidecar_path"))))
        p = photo_from_record(rec, folder)
        if p is not None and SHARE_DIR not in PurePosixPath(p.rel).parts:
            out.append(p)
    return out


def _subjects_sentence(sidecar: Path) -> str:
    """The `**Subjects:**` line of a sidecar body ("" when absent)."""
    from framedex.pipeline import split_frontmatter

    try:
        parts = split_frontmatter(sidecar.read_text(encoding="utf-8"))
    except OSError:
        return ""
    m = _SUBJECTS_RE.search(parts[1]) if parts else None
    return m.group(1).strip() if m else ""


def unindexed_images(folder: Path) -> list[Path]:
    from framedex.images import find_images
    from framedex.pipeline import has_sidecar

    return [p for p in find_images(folder, []) if not has_sidecar(p)]


def _date_arg(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {text!r}") from e


CLI_HELP = """\
Turn an indexed folder of photos into a gallery link.

  fdx share ~/Pictures/2025-11-mara              pick, render, edit in the browser, Publish
  fdx share ~/Pictures/2025-11-mara --publish    publish the current selection from here
  fdx share ~/Pictures/2025-11-mara --unpublish  take it down

More: docs/share.md
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdx share",
        description=CLI_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder", help="An indexed folder of photos (one shoot or trip)"
    )
    pick = parser.add_argument_group("the pick (first run, or with --reselect)")
    pick.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_PHOTOS,
        dest="max_photos",
        metavar="N",
        help=f"Most photos in the gallery (default {DEFAULT_MAX_PHOTOS}).",
    )
    pick.add_argument(
        "--include-review",
        action="store_true",
        help="Let review-rated photos compete too (default: keep only).",
    )
    pick.add_argument("--title", help="Gallery title (default: place and month).")
    pick.add_argument(
        "--from",
        type=_date_arg,
        dest="date_from",
        help="Only photos from this day (YYYY-MM-DD).",
    )
    pick.add_argument(
        "--to", type=_date_arg, dest="date_to", help="Only photos up to this day."
    )
    pick.add_argument(
        "--captions",
        action="store_true",
        help="Show the one-line captions under photos.",
    )
    pick.add_argument(
        "--reselect",
        action="store_true",
        help="Pick again from the sidecars, replacing selection.txt (the old one "
        "is kept as selection.txt.bak).",
    )
    parser.add_argument(
        "--name",
        help="Gallery name: its work folder under _share/ and the readable part of "
        "the link (default: the folder name).",
    )
    parser.add_argument("--out", help="Work folder (default: FOLDER/_share/NAME).")
    parser.add_argument(
        "--keep-gps",
        action="store_true",
        help="Keep GPS coordinates in the web copies (default: stripped).",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index unindexed photos first (runs fdx --media images on FOLDER).",
    )
    parser.add_argument(
        "--backend", choices=["cli", "api", "local"], help="Vision backend for --index."
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open the browser."
    )
    pub = parser.add_argument_group("publishing without the editor")
    pub.add_argument(
        "--publish",
        action="store_true",
        help="Publish the current selection and print the link.",
    )
    pub.add_argument(
        "--passcode",
        action="store_true",
        help="Protect the gallery with a passcode (asked for, never taken from the command line).",
    )
    pub.add_argument(
        "--no-passcode",
        action="store_true",
        help="Publish a protected gallery open, on purpose.",
    )
    pub.add_argument(
        "--new-link", action="store_true", help="Publish under a fresh random link."
    )
    pub.add_argument("--yes", action="store_true", help="Don't ask before publishing.")
    pub.add_argument(
        "--unpublish",
        action="store_true",
        help="Delete everything this gallery published to the configured target.",
    )
    pub.add_argument(
        "--config",
        help="Publish config (default: $FDX_SHARE_CONFIG or ~/.config/framedex/share.yaml).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Say more.")
    return parser


def _run_indexer(folder: Path, backend: str | None) -> int:
    cmd = [sys.executable, "-m", "framedex.index_videos", "--media", "images"]
    if backend:
        cmd += ["--backend", backend]
    cmd.append(str(folder))
    print("Indexing   " + " ".join(cmd[2:]))
    return subprocess.run(cmd).returncode


def _err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        return _err(f"not a directory: {folder}")
    if args.max_photos < 1:
        return _err("--max must be at least 1")
    if args.passcode and args.no_passcode:
        return _err("--passcode and --no-passcode contradict each other")
    try:
        from framedex import share_site
    except ImportError as e:  # pragma: no cover - needs the [share] extra
        return _err(str(e))

    name = args.name or folder.name
    work = (
        Path(args.out).expanduser().resolve()
        if args.out
        else folder / SHARE_DIR / slugify(name)
    )
    selection = work / SELECTION_FILE

    if args.unpublish:
        from framedex import share_publish

        return share_publish.unpublish_cli(
            work, config=args.config, assume_yes=args.yes
        )

    if (
        args.index
        and unindexed_images(folder)
        and _run_indexer(folder, args.backend) != 0
    ):
        return _err("indexing failed; see above")

    if args.reselect or not selection.exists():
        missing = unindexed_images(folder)
        photos = [
            p
            for p in load_photos(folder)
            if in_date_range(p, args.date_from, args.date_to)
        ]
        if not photos:
            return _err(
                f"no indexed photos under {folder}. Index them first: "
                f"fdx --media images {folder} (or add --index)"
            )
        if missing:
            print(
                f"Note       {len(missing)} photo(s) aren't indexed yet and are left out (--index)"
            )
        plan = plan_gallery(
            photos,
            max_photos=args.max_photos,
            include_review=args.include_review,
            signer=share_site.signatures,
        )
        if not plan.photos:
            wanted = "keep or review" if args.include_review else "keep"
            return _err(
                f"none of the {len(photos)} photos is rated {wanted}; try --include-review"
            )
        gallery = gallery_from_plan(
            plan,
            title=args.title or auto_title(plan.photos, title_from_name(name)),
            captions=args.captions,
        )
        if selection.exists():
            selection.replace(selection.with_suffix(".txt.bak"))
        write_selection_file(selection, gallery)
        print(
            f"Picked     {len(gallery.items)} of {len(photos)} photos  ·  {gallery.title}"
        )
    else:
        pick_flags = {
            "--title": args.title,
            "--from": args.date_from,
            "--to": args.date_to,
            "--max": args.max_photos != DEFAULT_MAX_PHOTOS,
            "--include-review": args.include_review,
            "--captions": args.captions,
        }
        ignored = [flag for flag, given in pick_flags.items() if given]
        if ignored:
            print(
                f"Note       {', '.join(ignored)} only shape a new pick; using the existing "
                f"{selection.name} (add --reselect to pick again)"
            )

    gallery, warnings = parse_selection(selection.read_text(encoding="utf-8"))
    for w in warnings:
        print(f"warning: {selection.name} {w}", file=sys.stderr)
    originals: dict[str, Path] = {}
    for item in gallery.all_items:
        path = resolve_item(folder, item.rel)
        if path is None:
            print(
                f"warning: skipping {item.rel}: not an image file under {folder}",
                file=sys.stderr,
            )
        else:
            originals[item.rel] = path
    for s in gallery.sections:
        s.items = [i for i in s.items if i.rel in originals]
    gallery.sections = [s for s in gallery.sections if s.items]
    if not gallery.items:
        return _err(f"{selection} lists no photos to show")
    inside = [rel for rel, path in originals.items() if work.resolve() in path.parents]
    if inside:
        return _err(
            f"the work folder {work} holds originals ({inside[0]}); pick an --out "
            "outside the photos, or leave the default _share/ folder"
        )

    tty = sys.stdout.isatty()

    def progress(n: int, total: int) -> None:
        if tty:
            print(
                f"\rRendering  {n}/{total}", end="\n" if n == total else "", flush=True
            )
        elif n == total:
            print(f"Rendered   {total} photos", flush=True)

    site = share_site.render_site(
        gallery, originals, work, keep_gps=args.keep_gps, progress=progress
    )
    if args.verbose:
        print(f"Work       {work}  ({site.total_bytes / 1e6:.1f} MB rendered)")

    if args.publish:
        from framedex import share_publish

        return share_publish.publish_cli(
            site,
            gallery,
            work,
            name=name,
            config=args.config,
            want_passcode=args.passcode,
            no_passcode=args.no_passcode,
            new_link=args.new_link,
            assume_yes=args.yes,
        )

    from framedex import share_edit

    return share_edit.serve(
        share_edit.Editor(work, name, gallery, site, args.config),
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    sys.exit(main())
