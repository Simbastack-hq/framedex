#!/usr/bin/env python3
"""
query.py — Filter the video index by metadata.

Reads all .description.md sidecars under a root, parses their YAML
frontmatter, applies filters, prints matching file paths (one per line).

Pipe into mpv, vlc, ffplay, or just `xargs open`.

Usage:
    fdx-query /Volumes/SSD-2024 --rating keep
    fdx-query /Volumes/SSD-2024 --rating keep --time-of-day golden_hour --stability smooth
    fdx-query /Volumes/SSD-2024 --place-contains California --language es
    fdx-query /Volumes/SSD-2024 --keyword drone --keyword landscape
    fdx-query /Volumes/SSD-2024 --rating cull            # what to delete
    fdx-query /Volumes/SSD-2024 --json                   # full records as JSON

Filters AND together. Multiple --keyword flags AND together (all must match).
Multiple values within a single flag (e.g. --rating keep,review) OR together.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framedex.parsing import (
    effective_rating,
    is_group_stub,
    is_usable_path,
    is_user_rated,
    scene_sentence,
)
from framedex.pipeline import SIDECAR_SUFFIX, split_frontmatter

try:
    import yaml
except ImportError:
    print("Missing PyYAML. Run setup.py.", file=sys.stderr)
    sys.exit(1)


def parse_sidecar(path: Path) -> dict[str, Any] | None:
    """Read sidecar, return parsed frontmatter dict (or None on parse failure)."""
    try:
        text = path.read_text()
    except Exception:
        return None
    parts = split_frontmatter(text)
    if parts is None:
        return None
    try:
        fm = yaml.safe_load(parts[0])
        if isinstance(fm, dict):
            fm["_sidecar_path"] = str(path)
            fm["_scene"] = scene_sentence(parts[1])
            return fm
    except yaml.YAMLError:
        return None
    return None


@dataclass
class Filters:
    """One field per fdx-query filter, with the CLI defaults. The CLI parser
    fills one; fdx-mcp builds one directly from validated arguments, so the
    two can never drift."""

    rating: str | None = None
    media: str | None = None
    lighting: str | None = None
    time_of_day: str | None = None
    audio_quality: str | None = None
    language: str | None = None
    focus: str | None = None
    stability: str | None = None
    exposure: str | None = None
    people_count: str | None = None
    face_count: str | None = None
    person: str | None = None
    keyword: list[str] = field(default_factory=list)
    dominant_color: str | None = None
    place_contains: str | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    has_speech: bool = False
    primary_only: bool = False
    folder: str | None = None  # only sidecars below root/folder
    offset: int = 0  # paging: skip the first N matches
    limit: int | None = None  # paging: at most N matches


@dataclass
class QueryResult:
    records: list[dict[str, Any]]  # the page: matches[offset : offset + limit]
    total: int  # matches before paging
    skipped_malformed: int  # sidecars without a usable path
    invalid_user_ratings: int  # records whose user_rating is not keep/review/cull


def _mapping(rec: dict[str, Any], key: str) -> dict[str, Any]:
    """A frontmatter field that must be a mapping, or {} when it is not
    (a hand-edited or model-garbled value must not crash a query)."""
    v = rec.get(key)
    return v if isinstance(v, dict) else {}


def _strings(rec: dict[str, Any], key: str) -> list[str]:
    """A frontmatter field that must be a list of strings, or [] when not."""
    v = rec.get(key)
    return [x for x in v if isinstance(x, str)] if isinstance(v, list) else []


def matches(rec: dict[str, Any], args: Any) -> bool:
    """Apply all filters (`args` is a Filters or an argparse Namespace with
    the same attribute names). Returns True if record passes all."""
    # Burst / RAW+JPEG members carry a copied assessment; hide them on request.
    if args.primary_only and is_group_stub(rec):
        return False
    # Rating (csv → OR within flag), matched against the effective rating: a
    # valid user_rating (the human's decision) wins over the model's.
    if args.rating:
        wanted = {v.strip() for v in args.rating.split(",")}
        if effective_rating(rec) not in wanted:
            return False
    if args.lighting:
        wanted = {v.strip() for v in args.lighting.split(",")}
        if rec.get("lighting") not in wanted:
            return False
    if args.time_of_day:
        wanted = {v.strip() for v in args.time_of_day.split(",")}
        if rec.get("time_of_day") not in wanted:
            return False
    if args.audio_quality:
        wanted = {v.strip() for v in args.audio_quality.split(",")}
        if rec.get("audio_quality") not in wanted:
            return False
    if args.language:
        wanted = {v.strip() for v in args.language.split(",")}
        if rec.get("language_detected") not in wanted:
            return False
    technical = _mapping(rec, "technical")
    if args.focus and technical.get("focus") != args.focus:
        return False
    if args.stability and technical.get("stability") != args.stability:
        return False
    if args.exposure and technical.get("exposure") != args.exposure:
        return False
    if args.people_count is not None:
        pc = rec.get("people_count")
        # Allow exact match or "+" suffix for >=
        wanted = args.people_count
        if wanted.endswith("+"):
            try:
                threshold = int(wanted[:-1])
                if not isinstance(pc, int) or pc < threshold:
                    return False
            except ValueError:
                return False
        else:
            try:
                if str(pc) != str(int(wanted)):
                    return False
            except ValueError:
                if str(pc) != wanted:
                    return False
    # Duration filters are video-only: a record without `duration_seconds`
    # (i.e. a photo) must NOT match rather than be treated as 0 seconds.
    if args.min_duration is not None:
        dur = rec.get("duration_seconds")
        if dur is None or dur < args.min_duration:
            return False
    if args.max_duration is not None:
        dur = rec.get("duration_seconds")
        if dur is None or dur > args.max_duration:
            return False
    if args.media:
        # Accept the indexer's plural words too (image/images, video/videos).
        alias = {"images": "image", "videos": "video"}
        wanted = {alias.get(v.strip(), v.strip()) for v in args.media.split(",")}
        # Existing video sidecars predate media_type; treat absent as 'video'.
        if (rec.get("media_type") or "video") not in wanted:
            return False
    if args.place_contains:
        place = str(_mapping(rec, "location").get("place") or "").lower()
        if args.place_contains.lower() not in place:
            return False
    if args.face_count is not None:
        wanted = args.face_count
        fc = rec.get("face_count") or 0
        if wanted.endswith("+"):
            try:
                threshold = int(wanted[:-1])
                if fc < threshold:
                    return False
            except ValueError:
                return False
        else:
            try:
                if fc != int(wanted):
                    return False
            except ValueError:
                return False
    if args.person:
        # Search face cluster_ids in this clip for matching name (case-insensitive)
        faces = rec.get("faces")
        names = {
            str(f.get("cluster_id") or "").lower()
            for f in (faces if isinstance(faces, list) else [])
            if isinstance(f, dict)
        }
        # Once fdx-faces relabels, cluster_id will be like "alex" or "sam"
        if args.person.lower() not in names:
            return False
    if args.keyword:
        kws = {k.lower() for k in _strings(rec, "keywords")}
        for required in args.keyword:
            if required.lower() not in kws:
                return False
    if args.dominant_color:
        dcs = {c.lower() for c in _strings(rec, "dominant_colors")}
        if args.dominant_color.lower() not in dcs:
            return False
    if args.has_speech:
        sc = rec.get("speaker_count") or 0
        if sc < 1:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="Drive/folder root to query")

    # Filter flags
    parser.add_argument("--rating", help="keep | review | cull (csv = OR)")
    parser.add_argument("--media", help="media_type: image | video (csv = OR)")
    parser.add_argument("--lighting")
    parser.add_argument("--time-of-day", dest="time_of_day")
    parser.add_argument("--audio-quality", dest="audio_quality")
    parser.add_argument("--language")
    parser.add_argument("--focus", choices=["sharp", "acceptable", "soft"])
    parser.add_argument("--stability", choices=["smooth", "handheld", "jittery"])
    parser.add_argument("--exposure", choices=["strong", "adequate", "poor", "clipped"])
    parser.add_argument(
        "--people-count",
        dest="people_count",
        help="Exact int, or 'N+' for ≥ N (e.g. '3+').",
    )
    parser.add_argument(
        "--face-count", dest="face_count", help="Exact int, or 'N+' for ≥ N."
    )
    parser.add_argument(
        "--person", help="Filter by face cluster name (after fdx-faces labels)."
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Required keyword (repeatable; all must match).",
    )
    parser.add_argument("--dominant-color", dest="dominant_color")
    parser.add_argument(
        "--place-contains",
        dest="place_contains",
        help="Substring match on reverse-geocoded place name.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        dest="min_duration",
        help="Minimum clip duration in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        dest="max_duration",
        help="Maximum clip duration in seconds.",
    )
    parser.add_argument(
        "--has-speech",
        action="store_true",
        dest="has_speech",
        help="Only clips with detected speech (speaker_count ≥ 1).",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        dest="primary_only",
        help="Hide burst / RAW+JPEG members whose assessment is copied from a "
        "group primary (group.primary: false).",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Only sidecars under ROOT/FOLDER (a trip or shoot subfolder). "
        "ROOT stays the scan root the sidecars were written from.",
    )

    # Output flags
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full records as JSON instead of paths.",
    )
    parser.add_argument(
        "--with-description",
        action="store_true",
        help="Show the rating + description preview alongside the path.",
    )
    parser.add_argument(
        "--count", action="store_true", help="Only print the count of matches."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Show at most N results."
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N matches (paging, with --limit).",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    filters = Filters(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(Filters)}
    )
    try:
        result = run_query(root, filters)
    except ValueError as e:
        sys.exit(str(e))
    if result.invalid_user_ratings:
        print(
            f"warning: {result.invalid_user_ratings} sidecar(s) carry an invalid "
            "user_rating (not keep/review/cull); ignored",
            file=sys.stderr,
        )
    matched = result.records

    if args.count:
        print(result.total)
        return 0

    if args.json:
        print(json.dumps(matched, indent=2, default=str))
        return 0

    for rec in matched:
        path = rec.get("path") or rec.get("_sidecar_path", "")
        if args.with_description:
            rating = str(effective_rating(rec) or "?")
            if is_user_rated(rec):
                rating += " (user)"
            # Videos show duration; photos show pixel dimensions in that column.
            if rec.get("duration_seconds") is not None:
                size_col = f"{rec['duration_seconds']:.1f}s"
            else:
                size_col = rec.get("dimensions") or rec.get("media_type") or ""
            place = ((rec.get("location") or {}).get("place") or "")[:40]
            kws = ",".join((rec.get("keywords") or [])[:5])
            line = f"{path}\t{rating}\t{size_col}\t{place}\t{kws}"
            print(line)
        else:
            print(path)
    return 0


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def run_query(
    root: Path, filters: Filters, *, require_media_under_root: bool = False
) -> QueryResult:
    """Scan every sidecar under `root` (or `root/folder`), resolve each
    record's media path against `root`, apply the filters, then page.
    Validation happens during the scan, before paging, so a bad record never
    consumes an offset: a sidecar that resolves outside the root (a symlink),
    one that does not parse, and — with `require_media_under_root` (fdx-mcp)
    — one whose media path resolves outside the root are skipped and counted
    in `skipped_malformed`. The CLI leaves media paths alone so legacy
    absolute paths still print. Warnings go to stderr."""
    root = root.resolve()
    scan_root = root
    if filters.folder:
        scan_root = (root / filters.folder).resolve()
        if not _under(scan_root, root):
            raise ValueError(f"--folder must be inside the root: {filters.folder!r}")
    sidecars = sorted(scan_root.rglob("*" + SIDECAR_SUFFIX))
    matched: list[dict[str, Any]] = []
    n_bad_path = 0
    n_invalid_user = 0
    for s in sidecars:
        real = s.resolve()
        if not (
            _under(real, root) and real.name.endswith(SIDECAR_SUFFIX) and real.is_file()
        ):
            print(f"warning: skipping {s}: resolves outside {root}", file=sys.stderr)
            n_bad_path += 1
            continue
        rec = parse_sidecar(s)
        if rec is None:
            print(f"warning: skipping {s}: unparsable sidecar", file=sys.stderr)
            n_bad_path += 1
            continue
        # Sidecars store `path` relative to the scan root (portable). Resolve it
        # back to an absolute path so the printed output is usable for piping
        # (xargs, ffplay, etc.). Older sidecars with absolute paths pass through.
        p = rec.get("path")
        if is_usable_path(p):
            if not Path(p).is_absolute():
                rec["path"] = str(root / p)
            if require_media_under_root and not _under(
                Path(rec["path"]).resolve(), root
            ):
                print(
                    f"warning: skipping {s}: media path resolves outside {root}",
                    file=sys.stderr,
                )
                n_bad_path += 1
                continue
        elif p is None and rec.get("photos_uuid"):
            # Photos-managed asset: `path` is omitted by design (the original
            # lives in the Photos library, not on disk). Keep it — output falls
            # back to the sidecar path. Only a truly-omitted path qualifies; a
            # present-but-malformed path is still broken and skipped below.
            pass
        else:
            # No usable media path → the record is broken. Warn and skip rather
            # than crash on Path() or print the sidecar path as if it were media.
            print(
                f"warning: skipping {s}: missing or unusable 'path' field",
                file=sys.stderr,
            )
            n_bad_path += 1
            continue
        if rec.get("user_rating") is not None and not is_user_rated(rec):
            n_invalid_user += 1
        rec["effective_rating"] = effective_rating(rec)
        if matches(rec, filters):
            matched.append(rec)

    if n_bad_path:
        print(f"skipped {n_bad_path} sidecar(s) with unusable path", file=sys.stderr)

    total = len(matched)
    start = max(filters.offset, 0)
    end = start + filters.limit if filters.limit else None
    return QueryResult(matched[start:end], total, n_bad_path, n_invalid_user)


if __name__ == "__main__":
    sys.exit(main())
