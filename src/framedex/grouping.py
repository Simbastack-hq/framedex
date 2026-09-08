#!/usr/bin/env python3
"""
framedex.grouping — burst + RAW/JPEG grouping pre-pass for still photos.

One moment = one vision call = one primary sidecar. Runs after `find_images`,
before the per-file loop, folder mode only. Everything here is pure logic
over (path, timestamp, camera) tuples so tests never shell out; the single
batched exiftool call is isolated in `read_group_metadata`.

Pairing: same directory + same stem (case-insensitive), one RAW + one camera
JPEG → one Unit (RAW primary, JPEG sibling used as the preview source).
Bursts: within one directory and one camera, frames whose successive
DateTimeOriginal gaps are <= BURST_GAP_SEC chain; a chain of >= BURST_MIN_SIZE
is a burst. Files without a usable date never join a group (indexed
individually — never guess).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from framedex.images import RAW_EXTENSIONS

# Successive frames (same folder, same camera) at most this far apart chain
# into one burst. 2 s covers every burst mode; two deliberate frames rarely
# land inside it.
BURST_GAP_SEC = 2.0
# A chain shorter than this is not a burst; its frames index individually.
BURST_MIN_SIZE = 3
# Only camera-rendered JPEGs pair with a same-stem RAW. PNG/TIFF/HEIC/WebP
# sharing a stem are exports or derivatives, not the same capture.
PAIR_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
# The five tags grouping needs. No `-n`: SubSecTimeOriginal must stay a string
# ("05" is .05 s; as a number it would become 5 → .5 s).
GROUP_EXIF_TAGS = [
    "-DateTimeOriginal",
    "-SubSecDateTimeOriginal",
    "-SubSecTimeOriginal",
    "-Make",
    "-Model",
]

_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class GroupMeta:
    """The per-file EXIF grouping needs. `timestamp` is seconds since a fixed
    naive epoch with subsecond resolution; None when DateTimeOriginal is
    missing or unparsable. `camera` is "Make|Model" ("" when unknown)."""

    timestamp: float | None
    camera: str


@dataclass(frozen=True)
class Unit:
    """A RAW+JPEG pair collapsed to its RAW, or a lone file."""

    primary: Path
    sibling: Path | None = None

    @property
    def preview_source(self) -> Path:
        """The camera JPEG when paired (camera-rendered colour beats the RAW's
        embedded preview and skips the exiftool extraction), else the file."""
        return self.sibling or self.primary

    @property
    def files(self) -> list[Path]:
        return [self.primary] + ([self.sibling] if self.sibling else [])


@dataclass
class MediaGroup:
    kind: str  # "burst" | "raw_jpeg" (a burst of pairs is a burst)
    id: str  # b_<sha1[:8] of the sorted root-relative member paths>
    units: list[Unit]  # timestamp order for a burst; exactly one for raw_jpeg
    primary: Path | None = None  # set by pick_representative
    sharpness: dict[Path, float] = field(default_factory=dict)  # unit.primary -> score

    @property
    def files(self) -> list[Path]:
        return [f for u in self.units for f in u.files]


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def parse_exif_timestamp(
    datetime_original: str | None,
    subsec_datetime_original: str | None,
    subsec_time_original: str | None,
) -> float | None:
    """Resolve a capture instant to seconds. Prefer SubSecDateTimeOriginal
    ('2024:08:14 07:23:11.25+03:00'); else DateTimeOriginal plus
    SubSecTimeOriginal as a fractional suffix; else whole seconds. Any
    timezone suffix is dropped: only gaps between frames of one camera
    matter, and those share a zone. None when nothing parses."""
    raw = (subsec_datetime_original or "").strip()
    frac = ""
    if raw:
        base = _TZ_SUFFIX_RE.sub("", raw)
        if "." in base:
            base, frac = base.split(".", 1)
    else:
        base = _TZ_SUFFIX_RE.sub("", (datetime_original or "").strip())
        digits = (subsec_time_original or "").strip()
        if digits.isdigit():
            frac = digits
    try:
        dt = datetime.strptime(base, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    seconds = (dt - _EPOCH).total_seconds()
    if frac.isdigit():
        seconds += float("0." + frac)
    return seconds


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def pair_raw_jpeg(paths: list[Path]) -> list[Unit]:
    """Collapse RAW+JPEG pairs into Units; everything else becomes a lone Unit.
    A RAW with two JPEG candidates pairs with the first by sorted path (the
    other stays alone); two RAWs sharing a stem are ambiguous and both stay
    alone. Result is sorted by primary path."""
    by_key: dict[tuple[Path, str], list[Path]] = {}
    for p in paths:
        by_key.setdefault((p.parent, p.stem.lower()), []).append(p)
    units: list[Unit] = []
    for members in by_key.values():
        raws = sorted(m for m in members if m.suffix.lower() in RAW_EXTENSIONS)
        jpegs = sorted(m for m in members if m.suffix.lower() in PAIR_JPEG_EXTENSIONS)
        others = [m for m in members if m not in raws and m not in jpegs]
        if len(raws) == 1 and jpegs:
            units.append(Unit(raws[0], jpegs[0]))
            units += [Unit(j) for j in jpegs[1:]]
        else:
            units += [Unit(m) for m in raws + jpegs]
        units += [Unit(m) for m in others]
    return sorted(units, key=lambda u: u.primary)


# ---------------------------------------------------------------------------
# Bursts
# ---------------------------------------------------------------------------


def group_id(files: list[Path], root: Path) -> str:
    """Stable, order-independent id from the sorted root-relative paths."""
    rel = sorted(str(f.relative_to(root)) for f in files)
    return "b_" + hashlib.sha1("\n".join(rel).encode("utf-8")).hexdigest()[:8]


def detect_bursts(
    units: list[Unit], meta: dict[Path, GroupMeta], root: Path
) -> tuple[list[MediaGroup], list[Unit]]:
    """Chain units within one directory + one camera whose successive
    timestamps are <= BURST_GAP_SEC apart; chains of >= BURST_MIN_SIZE become
    burst groups (members in timestamp order). Units without a timestamp
    never chain. Returns (bursts, remaining units)."""
    lanes: dict[tuple[Path, str], list[tuple[float, Unit]]] = {}
    rest: list[Unit] = []
    for u in units:
        m = meta.get(u.primary)
        if m is None or m.timestamp is None:
            rest.append(u)
            continue
        lanes.setdefault((u.primary.parent, m.camera), []).append((m.timestamp, u))

    bursts: list[MediaGroup] = []
    for lane in lanes.values():
        lane.sort(key=lambda t: (t[0], t[1].primary))
        chain: list[Unit] = []
        prev_ts: float | None = None
        for ts, u in lane:
            if prev_ts is not None and ts - prev_ts > BURST_GAP_SEC:
                _flush_chain(chain, bursts, rest, root)
                chain = []
            chain.append(u)
            prev_ts = ts
        _flush_chain(chain, bursts, rest, root)
    return bursts, sorted(rest, key=lambda u: u.primary)


def _flush_chain(
    chain: list[Unit], bursts: list[MediaGroup], rest: list[Unit], root: Path
) -> None:
    if len(chain) >= BURST_MIN_SIZE:
        files = [f for u in chain for f in u.files]
        bursts.append(MediaGroup("burst", group_id(files, root), list(chain)))
    else:
        rest.extend(chain)


def build_groups(
    paths: list[Path], meta: dict[Path, GroupMeta], root: Path
) -> tuple[list[MediaGroup], list[Path]]:
    """Pair, then chain bursts over the pair-collapsed units. Returns
    (groups sorted by first file, ungrouped single files). Every input path
    appears exactly once: in one group's `files` or in the singles."""
    bursts, rest = detect_bursts(pair_raw_jpeg(paths), meta, root)
    pairs = [
        MediaGroup("raw_jpeg", group_id(u.files, root), [u]) for u in rest if u.sibling
    ]
    singles = [u.primary for u in rest if not u.sibling]
    groups = sorted(bursts + pairs, key=lambda grp: grp.files[0])
    return groups, singles


# ---------------------------------------------------------------------------
# Representative
# ---------------------------------------------------------------------------


def pick_representative(group: MediaGroup, score: Callable[[Unit], float]) -> None:
    """Score every unit (the caller renders `unit.preview_source` and returns
    its Laplacian sharpness); the highest becomes `group.primary`, a tie
    going to the earliest unit. Every score is recorded in `group.sharpness`
    so members can carry their own number. Deterministic, local, explainable
    — deliberately not a model call (that would scale vision cost with
    shooting style)."""
    best_i = 0
    best = float("-inf")
    for i, u in enumerate(group.units):
        s = score(u)
        group.sharpness[u.primary] = s
        if s > best:
            best_i, best = i, s
    group.primary = group.units[best_i].primary


# ---------------------------------------------------------------------------
# exiftool (the one impure function)
# ---------------------------------------------------------------------------


def read_group_metadata(paths: list[Path]) -> dict[Path, GroupMeta]:
    """One batched exiftool call for the whole image list (an argfile dodges
    ARG_MAX). Partial output is used as-is (exiftool exits 1 when any file
    fails but still emits JSON for the rest). On a failure to run or parse,
    print a loud warning and return {} — every file then indexes
    individually (RAW+JPEG pairing needs no EXIF and still applies)."""
    if not paths:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False) as fh:
        fh.write("\n".join(str(p) for p in paths) + "\n")
        argfile = Path(fh.name)
    try:
        result = subprocess.run(
            ["exiftool", "-json", *GROUP_EXIF_TAGS, "-@", str(argfile)],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        if not isinstance(data, list):
            raise ValueError("exiftool JSON is not a list")
    except (OSError, ValueError) as e:
        print(
            f"warning: grouping: exiftool batch read failed ({e}) — "
            "no burst detection this run; files index individually",
            file=sys.stderr,
        )
        return {}
    finally:
        argfile.unlink(missing_ok=True)

    out: dict[Path, GroupMeta] = {}
    for entry in data:
        src = entry.get("SourceFile")
        if not src:
            continue
        make = str(entry.get("Make") or "").strip()
        model = str(entry.get("Model") or "").strip()
        camera = f"{make}|{model}" if (make or model) else ""
        ts = parse_exif_timestamp(
            _as_str(entry.get("DateTimeOriginal")),
            _as_str(entry.get("SubSecDateTimeOriginal")),
            _as_str(entry.get("SubSecTimeOriginal")),
        )
        out[Path(src)] = GroupMeta(timestamp=ts, camera=camera)
    return out


def _as_str(v: object) -> str | None:
    return None if v is None else str(v)
