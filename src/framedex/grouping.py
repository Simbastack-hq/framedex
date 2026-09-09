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
is a burst. Files without a usable date or camera identity, and units that are
not independent captures (a RAW's second JPEG, an ambiguous same-stem bucket),
never join a group — they index individually. Never guess.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from framedex.images import RAW_EXTENSIONS
from framedex.pipeline import sidecar_path

# Successive frames (same folder, same camera) at most this far apart chain
# into one burst. 2 s covers every burst mode; two deliberate frames rarely
# land inside it.
BURST_GAP_SEC = 2.0
# A chain shorter than this is not a burst; its frames index individually.
BURST_MIN_SIZE = 3
# Only camera-rendered JPEGs pair with a same-stem RAW. PNG/TIFF/HEIC/WebP
# sharing a stem are exports or derivatives, not the same capture.
PAIR_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
# The five tags grouping needs. Values are read back as strings regardless of
# how exiftool's JSON encoder typed them (see `_as_str`).
GROUP_EXIF_TAGS = [
    "-DateTimeOriginal",
    "-SubSecDateTimeOriginal",
    "-SubSecTimeOriginal",
    "-Make",
    "-Model",
]

# Human-facing names for `MediaGroup.kind` (the YAML keeps the identifier).
KIND_LABELS = {"burst": "burst", "raw_jpeg": "RAW+JPEG"}

_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
_EPOCH = datetime(1970, 1, 1)


class GroupMetadataError(RuntimeError):
    """The batched exiftool read failed outright (binary missing, crashed, or
    returned no parseable JSON). Raised rather than degraded: silently
    indexing every file individually would multiply vision spend on exactly
    the burst-heavy folders grouping exists for. `--no-group` is the explicit
    way to ask for that."""


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
    # False for pairing leftovers (a RAW's second JPEG) and members of an
    # ambiguous bucket (two RAWs sharing a stem): not independent captures,
    # so they never count toward a burst.
    chainable: bool = True

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
    id: str  # b_<sha1[:8] of the sorted member file names>
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
    """Resolve a capture instant to seconds. Sources are tried in priority
    order and the first that parses wins: SubSecDateTimeOriginal
    ('2024:08:14 07:23:11.25+03:00'), then DateTimeOriginal with
    SubSecTimeOriginal appended as the fraction (unless the value already
    carries one). Any timezone suffix is dropped: only gaps between frames of
    one camera matter, and those share a zone. None when nothing parses."""
    digits = (subsec_time_original or "").strip()
    candidates = [
        (subsec_datetime_original, ""),
        (datetime_original, digits if _is_digits(digits) else ""),
    ]
    for text, extra_frac in candidates:
        seconds = _parse_one(text, extra_frac)
        if seconds is not None:
            return seconds
    return None


def _is_digits(s: str) -> bool:
    return bool(s) and s.isascii() and s.isdigit()


def _parse_one(text: str | None, extra_frac: str) -> float | None:
    """One timestamp source → seconds, or None so the next source is tried. A
    present-but-malformed fraction (`.bad`) rejects this source rather than
    silently rounding it to whole seconds."""
    base = _TZ_SUFFIX_RE.sub("", (text or "").strip())
    frac = ""
    if "." in base:
        base, frac = base.split(".", 1)
        if not _is_digits(frac):
            return None
    if not frac:
        frac = extra_frac
    try:
        dt = datetime.strptime(base, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    seconds = (dt - _EPOCH).total_seconds()
    if frac:
        seconds += float("0." + frac)
    return seconds


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def pair_raw_jpeg(paths: list[Path]) -> list[Unit]:
    """Collapse RAW+JPEG pairs into Units; everything else becomes a lone Unit.
    A RAW with two JPEG candidates pairs with the first by sorted path; the
    other stays alone and unchainable (it is not another capture). Two RAWs
    sharing a stem are ambiguous: every file in that bucket stays alone and
    unchainable. Result is sorted by primary path."""
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
            units += [Unit(j, chainable=False) for j in jpegs[1:]]
        elif len(raws) > 1:
            units += [Unit(m, chainable=False) for m in raws + jpegs]
        else:
            units += [Unit(m) for m in raws + jpegs]
        units += [Unit(m) for m in others]
    return sorted(units, key=lambda u: u.primary)


# ---------------------------------------------------------------------------
# Bursts
# ---------------------------------------------------------------------------


def group_id(files: list[Path]) -> str:
    """Stable id from the sorted member file names. Members share one
    directory, so names are unique within a group, and the id is independent
    of the scan root (re-indexing from a different root never changes ids).
    Not unique across folders: `fdx-master` scopes ids by directory."""
    names = sorted(f.name for f in files)
    return "b_" + hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:8]


def detect_bursts(
    units: list[Unit], meta: dict[Path, GroupMeta]
) -> tuple[list[MediaGroup], list[Unit]]:
    """Chain units within one directory + one camera whose successive
    timestamps are <= BURST_GAP_SEC apart; chains of >= BURST_MIN_SIZE become
    burst groups (members in timestamp order). Units without a timestamp,
    without a camera identity, or flagged unchainable never chain. Returns
    (bursts, remaining units)."""
    lanes: dict[tuple[Path, str], list[tuple[float, Unit]]] = {}
    rest: list[Unit] = []
    for u in units:
        m = meta.get(u.primary)
        if not u.chainable or m is None or m.timestamp is None or not m.camera:
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
                _flush_chain(chain, bursts, rest)
                chain = []
            chain.append(u)
            prev_ts = ts
        _flush_chain(chain, bursts, rest)
    return bursts, sorted(rest, key=lambda u: u.primary)


def _flush_chain(chain: list[Unit], bursts: list[MediaGroup], rest: list[Unit]) -> None:
    if len(chain) >= BURST_MIN_SIZE:
        files = [f for u in chain for f in u.files]
        bursts.append(MediaGroup("burst", group_id(files), list(chain)))
    else:
        rest.extend(chain)


def build_groups(
    paths: list[Path], meta: dict[Path, GroupMeta]
) -> tuple[list[MediaGroup], list[Path]]:
    """Pair, then chain bursts over the pair-collapsed units. Returns
    (groups sorted by first file, ungrouped single files). Every input path
    appears exactly once: in one group's `files` or in the singles."""
    bursts, rest = detect_bursts(pair_raw_jpeg(paths), meta)
    pairs = [MediaGroup("raw_jpeg", group_id(u.files), [u]) for u in rest if u.sibling]
    singles = [u.primary for u in rest if not u.sibling]
    groups = sorted(bursts + pairs, key=lambda grp: grp.files[0])
    return groups, singles


# ---------------------------------------------------------------------------
# Representative
# ---------------------------------------------------------------------------


def pick_representative(group: MediaGroup, scores: dict[Path, float]) -> None:
    """Record every unit's sharpness (`scores`, keyed by unit.primary) and make
    the highest `group.primary`; a tie goes to the earliest unit (units are
    in timestamp order). Deterministic, local, explainable — deliberately
    not a model call (that would scale vision cost with shooting style)."""
    group.sharpness = {u.primary: scores[u.primary] for u in group.units}
    best = group.units[0]
    for u in group.units[1:]:
        if scores[u.primary] > scores[best.primary]:
            best = u
    group.primary = best.primary


# ---------------------------------------------------------------------------
# Resume: is a group / single already done?
# ---------------------------------------------------------------------------


def group_is_done(
    group: MediaGroup, read: Callable[[Path], dict[str, Any] | None]
) -> bool:
    """True when every member has a sidecar that belongs to this exact group
    (same id ⇒ same membership) and none is marked `incomplete` (the primary's
    previous sidecar carries that marker while a re-run is in flight), or when
    every member has a sidecar from before grouping existed (no `group` block
    anywhere: those per-file assessments stay valid; `--force` regroups). A
    missing or unparsable sidecar, or a block from a different grouping, means
    the whole group is reprocessed and its stubs rewritten. `read` returns a
    sidecar's frontmatter, or None when it is missing/unparsable."""
    blocks: list[Any] = []
    for f in group.files:
        fm = read(sidecar_path(f))
        if fm is None:
            return False
        blocks.append(fm.get("group"))
    if all(b is None for b in blocks):
        return True
    return all(
        isinstance(b, dict) and b.get("id") == group.id and not b.get("incomplete")
        for b in blocks
    )


def single_is_done(path: Path, read: Callable[[Path], dict[str, Any] | None]) -> bool:
    """A lone file is done when its sidecar exists and is not a leftover of an
    earlier grouping (a stub, or an ex-primary, carries a `group` block and
    must be re-assessed on its own)."""
    fm = read(sidecar_path(path))
    return fm is not None and "group" not in fm


# ---------------------------------------------------------------------------
# exiftool (the one impure function)
# ---------------------------------------------------------------------------


def read_group_metadata(paths: list[Path]) -> dict[Path, GroupMeta]:
    """One batched exiftool call for the whole image list, fed on stdin via
    `-@ -` (one argument per line: no ARG_MAX, no temp file). A path with an
    embedded newline could smuggle an option into that stream (a write flag,
    say), so such paths are refused up front with a warning and left to the
    per-file argv reads. Partial output is used as-is (exiftool exits 1 when
    any file fails but still emits JSON for the rest); files absent from the
    output are reported and treated as undated. Raises GroupMetadataError
    when the batch fails outright."""
    safe = [p for p in paths if "\n" not in str(p) and "\r" not in str(p)]
    if len(safe) != len(paths):
        print(
            f"warning: burst detection skipped {len(paths) - len(safe)} files: "
            "filenames contain newline characters",
            file=sys.stderr,
        )
    if not safe:
        return {}
    try:
        result = subprocess.run(
            # -charset filename=utf8: the stdin stream is UTF-8 on every OS
            # (Windows would otherwise read names in the system code page).
            [
                "exiftool",
                "-json",
                *GROUP_EXIF_TAGS,
                "-charset",
                "filename=utf8",
                "-@",
                "-",
            ],
            input="\n".join(str(p) for p in safe) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode < 0:
            raise ValueError(f"exiftool was killed by signal {-result.returncode}")
        data = json.loads(result.stdout)
        if not isinstance(data, list):
            raise ValueError("exiftool JSON is not a list")
    except FileNotFoundError as e:
        raise GroupMetadataError(
            "error: exiftool executable not found. Install exiftool or add it to "
            "PATH, then retry."
        ) from e
    except (OSError, ValueError) as e:
        raise GroupMetadataError(f"error: exiftool batch read failed ({e})") from e

    out: dict[Path, GroupMeta] = {}
    for entry in data:
        if not isinstance(entry, dict) or not entry.get("SourceFile"):
            continue
        if entry.get("Error"):
            continue  # exiftool could not read this file: counted as missing below
        make = str(entry.get("Make") or "").strip()
        model = str(entry.get("Model") or "").strip()
        camera = f"{make}|{model}" if (make or model) else ""
        ts = parse_exif_timestamp(
            _as_str(entry.get("DateTimeOriginal")),
            _as_str(entry.get("SubSecDateTimeOriginal")),
            _as_str(entry.get("SubSecTimeOriginal")),
        )
        out[Path(str(entry["SourceFile"]))] = GroupMeta(timestamp=ts, camera=camera)
    if not out:
        # Nothing succeeded: that is a broken read, not a folder without EXIF
        # (a file with no tags still comes back as an entry).
        raise GroupMetadataError(
            f"error: exiftool read no metadata from any of {len(safe)} files "
            f"(exit {result.returncode}): {result.stderr.strip()[:200]}"
        )
    missing = sum(1 for p in safe if p not in out)
    if missing:
        print(
            f"warning: exiftool returned no EXIF for {missing} of {len(safe)} files; "
            "excluded from burst detection (RAW+JPEG pairing still applies)",
            file=sys.stderr,
        )
    return out


def _as_str(v: object) -> str | None:
    return None if v is None else str(v)
