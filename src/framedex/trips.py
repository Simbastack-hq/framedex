#!/usr/bin/env python3
"""
fdx trips — find the trips in an indexed photo archive.

    fdx trips /Volumes/SSD-2025              # one line per trip, with a draft
    fdx trips /Volumes/SSD-2025 --json       # the same as JSON

A trip is a run of photos with no gap longer than TRIP_GAP_HOURS between
consecutive captures; a long GPS jump after a shorter gap (home from one
place, out to another) also starts a new one. Each trip comes with a draft
gallery size (the same pick fdx-share would make from the sidecars, minus
the pixel-based near-duplicate check) and the fdx-share command to make it.
Reads sidecars only: no pixels, no network, nothing written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

from framedex.share import (
    DEFAULT_MAX_PHOTOS,
    Photo,
    date_range_label,
    load_photos,
    plan_gallery,
    region_label,
    slugify,
    time_order,
    trip_day,
)

# Consecutive photos further apart than this belong to different trips.
TRIP_GAP_HOURS = 36.0
# ...and so do photos this far apart in space after at least JUMP_GAP_HOURS
# without a photo (a flight home and out again inside a day and a half).
JUMP_KM = 300.0
JUMP_GAP_HOURS = 12.0
# Runs smaller than this are listed as stray photos, not trips.
MIN_TRIP_PHOTOS = 5
EARTH_RADIUS_KM = 6371.0


def km_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def split_trips(photos: list[Photo]) -> list[list[Photo]]:
    """Dated photos in time order, cut where the gap exceeds TRIP_GAP_HOURS
    or where the last and next geotagged photos are JUMP_KM apart after a
    gap of JUMP_GAP_HOURS. Undated photos are left out."""
    dated = sorted((p for p in photos if p.taken is not None), key=time_order)
    trips: list[list[Photo]] = []
    last_fix: tuple[float, float] | None = None
    prev: Photo | None = None
    for p in dated:
        fix = (p.lat, p.lon) if p.lat is not None and p.lon is not None else None
        new_trip = prev is None
        if prev is not None and prev.seconds is not None and p.seconds is not None:
            gap_h = (p.seconds - prev.seconds) / 3600
            jumped = (
                fix is not None
                and last_fix is not None
                and km_between(last_fix, fix) >= JUMP_KM
            )
            new_trip = gap_h > TRIP_GAP_HOURS or (jumped and gap_h >= JUMP_GAP_HOURS)
        if new_trip:
            trips.append([])
        trips[-1].append(p)
        if fix is not None:
            last_fix = fix
        prev = p
    return trips


@dataclass
class Trip:
    start: date
    end: date
    photos: int
    keeps: int
    draft: int  # photos a default fdx-share pick would hang
    region: str
    folder: str  # the deepest folder holding every photo, relative to root
    name: str
    folder_is_trip: bool = False  # the folder holds this trip and nothing else

    def command(self, root: Path) -> str:
        """The fdx-share call for this trip: its folder alone when the folder
        is the trip, else the folder narrowed to the trip's days."""
        target = root / self.folder if self.folder else root
        args = ["fdx-share", str(target), "--name", self.name]
        if not self.folder_is_trip:
            args += ["--from", self.start.isoformat(), "--to", self.end.isoformat()]
        return " ".join(shlex.quote(a) for a in args)


def summarize(trip: list[Photo], all_photos: list[Photo]) -> Trip:
    days = [trip_day(p.taken) for p in trip if p.taken]
    start, end = min(days), max(days)
    dirs = {str(PurePosixPath(p.rel).parent) for p in trip}
    common = os.path.commonpath(sorted(dirs)) if dirs else ""
    folder = "" if common in (".", "") else common
    in_folder = [
        p
        for p in all_photos
        if not folder or p.rel.startswith(folder.rstrip("/") + "/")
    ]
    trip_ids = {id(p) for p in trip}
    region = region_label(trip)
    name_base = region.split(",")[0] if region else "trip"
    plan = plan_gallery(trip, max_photos=DEFAULT_MAX_PHOTOS)
    return Trip(
        start=start,
        end=end,
        photos=len(trip),
        keeps=sum(1 for p in trip if p.rating == "keep"),
        draft=len(plan.photos),
        region=region,
        folder=folder,
        name=f"{slugify(name_base)}-{start:%Y-%m-%d}",
        folder_is_trip=bool(folder) and all(id(p) in trip_ids for p in in_folder),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fdx trips",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", help="An indexed folder tree (a drive, a year)")
    parser.add_argument("--json", action="store_true", help="Print JSON instead.")
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    photos = load_photos(root)
    runs = split_trips(photos)
    trips = [summarize(t, photos) for t in runs if len(t) >= MIN_TRIP_PHOTOS]
    strays = sum(len(t) for t in runs if len(t) < MIN_TRIP_PHOTOS)
    undated = sum(1 for p in photos if p.taken is None)

    if args.json:
        out: list[dict[str, Any]] = [
            {
                "start": t.start.isoformat(),
                "end": t.end.isoformat(),
                "photos": t.photos,
                "keeps": t.keeps,
                "draft_gallery": t.draft,
                "region": t.region,
                "folder": t.folder,
                "name": t.name,
                "command": t.command(root),
            }
            for t in trips
        ]
        print(
            json.dumps(
                {"trips": out, "stray_photos": strays, "undated": undated}, indent=2
            )
        )
        return 0

    if not trips:
        print(f"no trips of {MIN_TRIP_PHOTOS}+ dated photos under {root}")
    for n, t in enumerate(trips, 1):
        days = (t.end - t.start).days + 1
        print(
            f"{n:>3}. {date_range_label([t.start, t.end])} ({days} day{'s' if days != 1 else ''})"
            f"  {t.region or 'no GPS'}\n"
            f"     {t.photos} photos, {t.keeps} keep -> draft gallery of {t.draft}\n"
            f"     {t.command(root)}"
        )
    if strays or undated:
        print(f"\n{strays} stray photo(s) outside any trip, {undated} undated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
