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
import json
import sys
from pathlib import Path
from typing import Any

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
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1])
        if isinstance(fm, dict):
            fm["_sidecar_path"] = str(path)
            return fm
    except yaml.YAMLError:
        return None
    return None


def matches(rec: dict[str, Any], args: argparse.Namespace) -> bool:
    """Apply all filters. Returns True if record passes all."""
    # Rating (csv → OR within flag)
    if args.rating:
        wanted = {v.strip() for v in args.rating.split(",")}
        if rec.get("rating") not in wanted:
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
    if args.focus and (rec.get("technical") or {}).get("focus") != args.focus:
        return False
    if (
        args.stability
        and (rec.get("technical") or {}).get("stability") != args.stability
    ):
        return False
    if args.exposure and (rec.get("technical") or {}).get("exposure") != args.exposure:
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
    if args.min_duration is not None:
        dur = rec.get("duration_seconds") or 0
        if dur < args.min_duration:
            return False
    if args.max_duration is not None:
        dur = rec.get("duration_seconds") or 0
        if dur > args.max_duration:
            return False
    if args.place_contains:
        place = ((rec.get("location") or {}).get("place") or "").lower()
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
        faces = rec.get("faces") or []
        names = {(f.get("cluster_id") or "").lower() for f in faces}
        # Once fdx-faces relabels, cluster_id will be like "alex" or "sam"
        if args.person.lower() not in names:
            return False
    if args.keyword:
        kws = set(k.lower() for k in (rec.get("keywords") or []))
        for required in args.keyword:
            if required.lower() not in kws:
                return False
    if args.dominant_color:
        dcs = set(c.lower() for c in (rec.get("dominant_colors") or []))
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

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    sidecars = sorted(root.rglob("*.description.md"))
    matched: list[dict[str, Any]] = []
    for s in sidecars:
        rec = parse_sidecar(s)
        if rec is None:
            continue
        # Sidecars store `path` relative to the scan root (portable). Resolve it
        # back to an absolute path so the printed output is usable for piping
        # (xargs, ffplay, etc.). Older sidecars with absolute paths pass through.
        p = rec.get("path")
        if p and not Path(p).is_absolute():
            rec["path"] = str(root / p)
        if matches(rec, args):
            matched.append(rec)

    if args.limit:
        matched = matched[: args.limit]

    if args.count:
        print(len(matched))
        return 0

    if args.json:
        print(json.dumps(matched, indent=2, default=str))
        return 0

    for rec in matched:
        path = rec.get("path") or rec.get("_sidecar_path", "")
        if args.with_description:
            rating = rec.get("rating", "?")
            dur = rec.get("duration_seconds", 0)
            place = ((rec.get("location") or {}).get("place") or "")[:40]
            kws = ",".join((rec.get("keywords") or [])[:5])
            line = f"{path}\t{rating}\t{dur:.1f}s\t{place}\t{kws}"
            print(line)
        else:
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
