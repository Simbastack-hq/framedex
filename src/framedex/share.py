#!/usr/bin/env python3
"""
fdx-share — turn an indexed folder of photos into a shareable web gallery.

    fdx-share ~/Pictures/2025-11-mara               # pick, render, open a preview
    $EDITOR ~/Pictures/2025-11-mara/_share/mara/selection.txt
    fdx-share ~/Pictures/2025-11-mara               # re-render from your edits
    fdx-share ~/Pictures/2025-11-mara --publish     # upload, print + copy the URL

The sidecars do the curation: keep-rated photos only (a `user_rating` wins
over the model's), one frame per burst, near-duplicates dropped, capped at
--max with every day/place section represented. The pick is written to a
plain-text `selection.txt` you edit (drop, reorder, re-caption, re-title);
that file, not the sidecars, drives every later render. Originals are only
read: the gallery gets web-size copies (2048 px long edge) with GPS stripped
unless --keep-gps, a noindex page, and nothing else from your archive.

This module is the pure half (selection, ordering, the selection file);
pixels and HTML live in `share_site`, uploads in `share_publish`.
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
# The opening photo is cropped to 21:9; only a photo at least this wide
# (width / height) is picked for it automatically.
COVER_MIN_RATIO = 1.2
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
    for section, quota in zip(sections, quotas, strict=True):
        chosen = pick_spread(section.photos, quota)
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


def day_label(d: date) -> str:
    return f"{d:%A} {d.day} {d:%B}"


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

    sections: list[Section] = []
    for day, items in merged:
        when = day_label(day) if day else "Undated"
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


def title_from_name(name: str) -> str:
    words = [w for w in re.split(r"[-_\s]+", name) if w]
    return " ".join(w if any(c.isupper() for c in w) else w.capitalize() for w in words)


# ---------------------------------------------------------------------------
# selection.txt: the editable plan
# ---------------------------------------------------------------------------


@dataclass
class Item:
    rel: str
    caption: str = ""


@dataclass
class GallerySection:
    heading: str
    items: list[Item] = field(default_factory=list)


@dataclass
class Gallery:
    title: str
    subtitle: str = ""
    cover: str = ""  # rel of the opening photo; "" = first landscape, "none" = no cover
    sections: list[GallerySection] = field(default_factory=list)

    @property
    def items(self) -> list[Item]:
        return [i for s in self.sections for i in s.items]


SELECTION_HELP = """\
# fdx-share selection. Edit it, then re-run fdx-share on the folder to re-render.
#   drop a photo     delete its line, or put "# " in front of it
#   reorder          move lines; a "## " line starts a section
#   add one back     remove the "# " from a line under "Not picked"
#   caption          the text after " | " (leave it empty for none)
#   headings, title  edit any "## " line and the title:/subtitle:/cover: lines
#                    (cover: none drops the opening photo)
# Text after two spaces and "#" on a photo line is a note, never shown.
"""


def gallery_from_plan(
    plan: Plan, *, title: str, captions: bool = True
) -> tuple[Gallery, list[tuple[Item, str]]]:
    """The plan as an editable Gallery plus the not-picked pool as items."""
    photos = plan.photos
    days = [trip_day(p.taken) for p in photos if p.taken]
    subtitle = " · ".join(
        x for x in (date_range_label(days), region_label(photos)) if x
    )

    def item(p: Photo) -> Item:
        return Item(p.rel, caption_for(p.scene, p.subjects) if captions else "")

    sections = [
        GallerySection(s.heading, [item(p) for p in s.photos]) for s in plan.sections
    ]
    pool = [(item(p), why) for p, why in plan.not_picked]
    return Gallery(title, subtitle, pick_cover(plan), sections), pool


def pick_cover(plan: Plan) -> str:
    """The opening photo: the best-assessed wide photo of the biggest
    section (where most of the trip's frames went), trying smaller sections
    when it has none wide enough; "" when no photo is."""
    for section in sorted(plan.sections, key=lambda sec: -len(sec.photos)):
        wide = [p for p in section.photos if (p.aspect or 0.0) >= COVER_MIN_RATIO]
        if wide:
            return _best(wide).rel
    return ""


def _photo_line(item: Item) -> str:
    return f"{item.rel} | {item.caption}" if item.caption else item.rel


def write_selection(
    gallery: Gallery, pool: list[tuple[Item, str]] | None = None
) -> str:
    lines = [
        SELECTION_HELP,
        f"title: {gallery.title}",
        f"subtitle: {gallery.subtitle}",
        f"cover: {gallery.cover}",
    ]
    for s in gallery.sections:
        lines += ["", f"## {s.heading}"] if s.heading else [""]
        lines += [_photo_line(i) for i in s.items]
    if pool:
        lines += ["", "# ---- Not picked (remove the leading '# ' to add one) ----"]
        lines += [f"# {_photo_line(i)}{_INLINE_COMMENT} {why}" for i, why in pool]
    return "\n".join(lines) + "\n"


def parse_selection(text: str) -> tuple[Gallery, list[str]]:
    """Parse selection.txt back into a Gallery. Returns (gallery, warnings);
    a repeated photo keeps its first position and is reported."""
    gallery = Gallery(title="")
    warnings: list[str] = []
    seen: set[str] = set()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or (line.startswith("#") and not line.startswith("## ")):
            continue
        if line.startswith("## "):
            gallery.sections.append(GallerySection(line[3:].strip()))
            continue
        if not gallery.sections and line.startswith(("title:", "subtitle:", "cover:")):
            key, _, value = line.partition(":")
            setattr(gallery, key, value.strip())
            continue
        body = raw.split(_INLINE_COMMENT, 1)[0].strip()
        rel, _, caption = body.partition(" | ")
        rel, caption = rel.strip().removesuffix(" |").strip(), caption.strip()
        if not rel:
            continue
        if rel in seen:
            warnings.append(f"line {n}: {rel} is listed twice; keeping the first")
            continue
        seen.add(rel)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fdx-share",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder", help="An indexed folder of photos (one shoot or trip)"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_PHOTOS,
        dest="max_photos",
        help=f"Most photos in the gallery (default {DEFAULT_MAX_PHOTOS}).",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Let review-rated photos compete too (default: keep only).",
    )
    parser.add_argument(
        "--title", help="Gallery title (default: from the folder name)."
    )
    parser.add_argument(
        "--name",
        help="Gallery name: its work folder under _share/ and the readable part "
        "of the link (default: the folder name).",
    )
    parser.add_argument(
        "--from",
        type=_date_arg,
        dest="date_from",
        help="Only photos from this trip day on (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--to", type=_date_arg, dest="date_to", help="Only photos up to this trip day."
    )
    parser.add_argument(
        "--out",
        help="Work folder for selection.txt and the rendered site "
        "(default: FOLDER/_share/NAME).",
    )
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="Pick again from the sidecars, replacing selection.txt "
        "(your edits there are lost; the old file is kept as selection.txt.bak).",
    )
    parser.add_argument(
        "--no-captions", action="store_true", help="Pick without captions."
    )
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
        "--backend",
        choices=["cli", "api", "local"],
        help="Vision backend for --index (fdx's default when omitted).",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open the preview."
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload the rendered gallery to the configured target and print its URL.",
    )
    parser.add_argument(
        "--config",
        help="Publish config (default: $FDX_SHARE_CONFIG or "
        "~/.config/framedex/share.yaml).",
    )
    parser.add_argument(
        "--new-link",
        action="store_true",
        help="Publish under a fresh random link (the old one keeps working "
        "until you delete it).",
    )
    parser.add_argument(
        "--unpublish",
        action="store_true",
        help="Delete everything published from this gallery (every link) on the "
        "configured target.",
    )
    parser.add_argument("--yes", action="store_true", help="Publish without asking.")
    return parser


def _run_indexer(folder: Path, backend: str | None) -> int:
    cmd = [sys.executable, "-m", "framedex.index_videos", "--media", "images"]
    if backend:
        cmd += ["--backend", backend]
    cmd.append(str(folder))
    print("indexing first: " + " ".join(cmd[2:]))
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 2
    if args.max_photos < 1:
        print("error: --max must be at least 1", file=sys.stderr)
        return 2

    try:
        from framedex import share_site
    except ImportError as e:  # pragma: no cover - needs the [images] extra
        print(f"error: {e}", file=sys.stderr)
        return 2

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
            work, config_path=args.config, assume_yes=args.yes
        )

    if args.index:
        missing = unindexed_images(folder)
        if missing and _run_indexer(folder, args.backend) != 0:
            print("error: indexing failed; see above", file=sys.stderr)
            return 1

    if args.reselect or not selection.exists():
        missing = unindexed_images(folder)
        photos = [
            p
            for p in load_photos(folder)
            if in_date_range(p, args.date_from, args.date_to)
        ]
        if not photos:
            print(
                f"error: no indexed photos under {folder}. Index them first:\n"
                f"  fdx --media images {folder}\n(or re-run with --index)",
                file=sys.stderr,
            )
            return 1
        if missing:
            print(
                f"note: {len(missing)} photo(s) here have no sidecar yet and are left "
                f"out. Index them with `fdx --media images {folder}` or --index."
            )
        print(f"picking from {len(photos)} indexed photos ...")
        plan = plan_gallery(
            photos,
            max_photos=args.max_photos,
            include_review=args.include_review,
            signer=share_site.signatures,
        )
        if not plan.photos:
            wanted = "keep/review" if args.include_review else "keep"
            print(
                f"error: none of the {len(photos)} photos is rated {wanted}. "
                "Try --include-review, or rate some keep (fdx-mcp set_user_rating).",
                file=sys.stderr,
            )
            return 1
        gallery, pool = gallery_from_plan(
            plan,
            title=args.title or title_from_name(name),
            captions=not args.no_captions,
        )
        work.mkdir(parents=True, exist_ok=True)
        if selection.exists():
            selection.replace(selection.with_suffix(".txt.bak"))
        selection.write_text(write_selection(gallery, pool), encoding="utf-8")
        print(
            f"picked {len(gallery.items)} of {len(photos)} "
            f"({len(plan.not_picked)} more listed as not picked) -> {selection}"
        )
    else:
        pick_flags = {
            "--title": args.title,
            "--from": args.date_from,
            "--to": args.date_to,
            "--max": args.max_photos != DEFAULT_MAX_PHOTOS,
            "--include-review": args.include_review,
            "--no-captions": args.no_captions,
        }
        ignored = [flag for flag, given in pick_flags.items() if given]
        if ignored:
            print(
                f"note: {', '.join(ignored)} only shape a new pick; {selection.name} "
                "already exists, so it is used as is (add --reselect to pick again)"
            )

    gallery, warnings = parse_selection(selection.read_text(encoding="utf-8"))
    for w in warnings:
        print(f"warning: {selection.name} {w}", file=sys.stderr)
    resolved: dict[str, Path] = {}
    for item in gallery.items:
        path = resolve_item(folder, item.rel)
        if path is None:
            print(
                f"warning: skipping {item.rel}: not an image file under {folder}",
                file=sys.stderr,
            )
        else:
            resolved[item.rel] = path
    for s in gallery.sections:
        s.items = [i for i in s.items if i.rel in resolved]
    gallery.sections = [s for s in gallery.sections if s.items]
    if not gallery.sections:
        print(f"error: {selection} lists no usable photos", file=sys.stderr)
        return 1

    site = share_site.render_site(gallery, resolved, work, keep_gps=args.keep_gps)
    print(
        f"rendered {len(site.photos)} photos ({site.total_bytes / 1e6:.1f} MB) -> {site.root}"
    )

    if args.publish:
        from framedex import share_publish

        return share_publish.publish_cli(
            site,
            gallery,
            work,
            name=name,
            config_path=args.config,
            new_link=args.new_link,
            assume_yes=args.yes,
        )

    print(f"preview: {site.preview}")
    print(f"edit:    {selection}  (then re-run to re-render)")
    print(f"publish: fdx-share {folder} --publish")
    if not args.no_open:
        import webbrowser

        webbrowser.open(site.preview.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
