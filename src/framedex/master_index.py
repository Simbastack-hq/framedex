#!/usr/bin/env python3
"""
master_index.py — Compile every sidecar + folder summary on a drive into a
machine-readable JSON index and a human-readable markdown overview at the
drive root.

Writes:
    <root>/_INDEX.json   — full array of clip records (downstream tools query)
    <root>/_INDEX.md     — scannable human overview with rating + locations + themes

Idempotent. Safe to re-run after any new indexing.

Usage:
    fdx-master /Volumes/SSD-2024
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Missing PyYAML. Run setup.py.", file=sys.stderr)
    sys.exit(1)


def parse_sidecar(path: Path) -> dict[str, Any] | None:
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
            # Try to pull the description prose for completeness
            body = parts[2].strip()
            import re

            m = re.search(r"##\s*Description\s*\n+(.+?)(?=\n##|\Z)", body, re.S | re.I)
            fm["description"] = m.group(1).strip() if m else ""
            fm["sidecar_path"] = str(path)
            return fm
    except yaml.YAMLError:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Drive/folder root")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    sidecars = sorted(root.rglob("*.description.md"))
    if not sidecars:
        print(f"No sidecars found under {root}. Run index_videos.py first.")
        return 0

    records: list[dict[str, Any]] = [
        r for r in (parse_sidecar(s) for s in sidecars) if r
    ]

    # Group by top-level subfolder
    trips: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        vpath = Path(r.get("path", ""))
        try:
            rel = vpath.relative_to(root)
            trip = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        except ValueError:
            trip = "(unknown)"
        trips.setdefault(trip, []).append(r)

    # ---- JSON ----
    json_out = root / "_INDEX.json"
    json_out.write_text(
        json.dumps(
            {
                "drive_root": str(root),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "clip_count": len(records),
                "trip_count": len(trips),
                "clips": records,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Wrote {json_out.name} ({len(records)} clips)")

    # ---- Markdown ----
    total_dur_min = sum((r.get("duration_seconds") or 0) for r in records) / 60
    rating_counter = Counter(r.get("rating", "review") for r in records)
    languages = Counter(r.get("language_detected") or "none" for r in records)
    keyword_freq: Counter[str] = Counter()
    for r in records:
        for k in r.get("keywords") or []:
            keyword_freq[k] += 1
    place_freq: Counter[str] = Counter()
    for r in records:
        place = (r.get("location") or {}).get("place")
        if place:
            place_freq[place] += 1

    face_total = sum(r.get("face_count") or 0 for r in records)
    named_people: Counter[str] = Counter()
    for r in records:
        for f in r.get("faces") or []:
            cid = f.get("cluster_id", "")
            if cid and not cid.startswith("tmp_"):
                named_people[cid] += 1

    lines: list[str] = [
        f"# Video Knowledge Base — `{root.name}`",
        "",
        f"*Generated {datetime.now().isoformat(timespec='seconds')}*",
        f"*{len(records)} clips, {total_dur_min:.1f} min total, "
        f"across {len(trips)} top-level folders*",
        "",
        "## Drive-level stats",
        "",
        f"- **Ratings:** "
        f"{rating_counter.get('keep', 0)} keep, "
        f"{rating_counter.get('review', 0)} review, "
        f"{rating_counter.get('cull', 0)} cull",
        "- **Languages:** "
        + ", ".join(f"{lang} ({n})" for lang, n in languages.most_common(5)),
    ]
    if place_freq:
        lines.append(
            "- **Top locations:** "
            + "; ".join(f"{p} ({n})" for p, n in place_freq.most_common(5))
        )
    if face_total:
        if named_people:
            lines.append(
                "- **Recognized people:** "
                + ", ".join(f"{name} ({n})" for name, n in named_people.most_common(10))
            )
            lines.append(
                f"- **Faces detected:** {face_total} total ({sum(named_people.values())} named)"
            )
        else:
            lines.append(
                f"- **Faces detected:** {face_total} total — run fdx-faces to label clusters"
            )
    if keyword_freq:
        lines.append(
            "- **Top keywords:** "
            + ", ".join(f"`{k}` ({n})" for k, n in keyword_freq.most_common(15))
        )

    cull_clips = [r for r in records if r.get("rating") == "cull"]
    if cull_clips:
        cull_dur = sum((c.get("duration_seconds") or 0) for c in cull_clips) / 60
        lines.append("")
        lines.append(
            f"## Cull pile — {len(cull_clips)} clips, {cull_dur:.1f} min total"
        )
        lines.append("")
        lines.append("Safe to delete (AI-flagged + spot-check first):")
        lines.append("")
        for c in cull_clips[:30]:
            reason = c.get("cull_reason") or "(no reason given)"
            lines.append(f"- `{c.get('path', '?')}` — {reason}")
        if len(cull_clips) > 30:
            lines.append(f"- … and {len(cull_clips) - 30} more (see `_INDEX.json`)")

    lines += ["", "## Trips / Top-level folders", ""]
    for trip in sorted(trips.keys()):
        clips = trips[trip]
        total = sum((c.get("duration_seconds") or 0) for c in clips)
        ratings = Counter(c.get("rating", "review") for c in clips)
        langs = sorted(
            {
                c.get("language_detected", "")
                for c in clips
                if c.get("language_detected")
            }
        )
        places = sorted(
            {
                (c.get("location") or {}).get("place", "")
                for c in clips
                if (c.get("location") or {}).get("place")
            }
        )
        summary_file = root / trip / "_folder-summary.md"
        link = (
            f"[`{trip}/_folder-summary.md`]({trip}/_folder-summary.md)"
            if summary_file.exists()
            else "_no summary yet — run fdx-summary_"
        )
        lines += [
            f"### {trip}",
            f"- Clips: {len(clips)} ({ratings.get('keep', 0)} keep / "
            f"{ratings.get('review', 0)} review / {ratings.get('cull', 0)} cull)",
            f"- Duration: {total / 60:.1f} min",
        ]
        if langs:
            lines.append(f"- Languages: {', '.join(langs)}")
        if places:
            sample = ", ".join(places[:3]) + ("..." if len(places) > 3 else "")
            lines.append(f"- Locations: {sample}")
        lines.append(f"- Summary: {link}")
        lines.append("")

    md_out = root / "_INDEX.md"
    md_out.write_text("\n".join(lines))
    print(f"Wrote {md_out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
