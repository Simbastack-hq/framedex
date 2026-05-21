#!/usr/bin/env python3
"""
trip_summary.py — Roll up per-clip sidecars into per-folder summaries (recursive).

For each folder under the root containing ≥ --min-clips sidecars, generates a
`_folder-summary.md` with: theme, date range, dominant location, rating
breakdown, visual character (time of day / lighting / colors), best clips,
cull pile, recurring themes, gaps.

Recursive — every nested folder with enough clips gets its own summary, so
you can browse the archive at any depth.

Usage:
    fdx-summary /Volumes/SSD-2024
    fdx-summary /Volumes/SSD-2024 --force
    fdx-summary /Volumes/SSD-2024 --min-clips 10
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import anthropic
    import yaml
except ImportError:
    print("Missing deps (anthropic or PyYAML). Run setup.py.", file=sys.stderr)
    sys.exit(1)


SUMMARY_FILE = "_folder-summary.md"
SUMMARY_MODEL = "claude-sonnet-4-6-20251001"
MAX_SIDECARS_PER_PROMPT = 250
DEFAULT_MIN_CLIPS = 5


def find_sidecars_immediate(folder: Path) -> list[Path]:
    """Sidecars directly in this folder (not recursive)."""
    return sorted([p for p in folder.glob("*.description.md") if p.is_file()])


def find_sidecars_recursive(folder: Path) -> list[Path]:
    """All sidecars under this folder (recursive)."""
    return sorted(folder.rglob("*.description.md"))


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
            fm["_sidecar_path"] = str(path)
            fm["_body"] = parts[2].strip()
            return fm
    except yaml.YAMLError:
        return None
    return None


def resolve_api_key() -> str:
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env.strip()
    p = Path.home() / ".claude" / "credentials" / "anthropic-key.txt"
    if p.exists():
        return p.read_text().strip()
    sys.exit("No Anthropic API key. Set ANTHROPIC_API_KEY for summarization.")


def compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate quick stats from records — used to inject ground-truth facts
    into the summary prompt so Claude doesn't have to count by hand."""
    total_duration = sum(r.get("duration_seconds") or 0 for r in records)
    ratings = Counter(r.get("rating", "review") for r in records)
    lighting = Counter(r.get("lighting", "unclear") for r in records)
    time_of_day = Counter(r.get("time_of_day", "unclear") for r in records)
    languages = Counter(r.get("language_detected") or "none" for r in records)
    audio_q = Counter(r.get("audio_quality", "unclear") for r in records)

    # Locations
    places: Counter[str] = Counter()
    for r in records:
        place = (r.get("location") or {}).get("place")
        if place:
            places[place] += 1

    # Date range
    dates = []
    for r in records:
        ct = r.get("creation_time")
        if ct:
            dates.append(str(ct)[:10])
    date_range = (min(dates), max(dates)) if dates else ("", "")

    # Keywords
    keyword_freq: Counter[str] = Counter()
    for r in records:
        for k in r.get("keywords") or []:
            keyword_freq[k] += 1

    # Color palette
    color_freq: Counter[str] = Counter()
    for r in records:
        for c in r.get("dominant_colors") or []:
            color_freq[c] += 1

    # Stability + focus
    stab: Counter[str] = Counter()
    foc: Counter[str] = Counter()
    for r in records:
        t = r.get("technical") or {}
        stab[t.get("stability", "unclear")] += 1
        foc[t.get("focus", "unclear")] += 1

    # Faces
    face_total = sum(r.get("face_count") or 0 for r in records)
    face_named: Counter[str] = Counter()
    for r in records:
        for f in r.get("faces") or []:
            cid = f.get("cluster_id", "")
            if cid and not cid.startswith("tmp_"):
                face_named[cid] += 1

    return {
        "n_clips": len(records),
        "total_minutes": total_duration / 60,
        "ratings": dict(ratings),
        "lighting": dict(lighting.most_common(5)),
        "time_of_day": dict(time_of_day.most_common(5)),
        "languages": dict(languages),
        "audio_quality": dict(audio_q),
        "places": dict(places.most_common(5)),
        "date_range": date_range,
        "top_keywords": dict(keyword_freq.most_common(15)),
        "top_colors": dict(color_freq.most_common(10)),
        "stability": dict(stab),
        "focus": dict(foc),
        "total_faces_detected": face_total,
        "named_people": dict(face_named.most_common(10)),
    }


def summarize_folder(
    client: anthropic.Anthropic,
    folder: Path,
    sidecars: list[Path],
    records: list[dict[str, Any]],
) -> str:
    """One Claude call: feed it the precomputed stats + a sample of sidecars,
    ask for a structured markdown summary."""
    stats = compute_stats(records)

    # Sample up to N sidecars (we don't need ALL of them — stats already
    # has the aggregates). Pick the highest-rated + variety.
    keep_recs = [r for r in records if r.get("rating") == "keep"]
    review_recs = [r for r in records if r.get("rating") == "review"]
    cull_recs = [r for r in records if r.get("rating") == "cull"]
    sample = (keep_recs[:30] + review_recs[:15] + cull_recs[:8])[
        :MAX_SIDECARS_PER_PROMPT
    ]

    sample_text = "\n\n---\n\n".join(
        f"### {Path(r['_sidecar_path']).name}\n\n{Path(r['_sidecar_path']).read_text()}"
        for r in sample
    )

    prompt = textwrap.dedent(f"""
    You are reading per-clip sidecar files from a folder of video footage.
    Folder name: "{folder.name}"

    PRECOMPUTED STATS (use these — don't recount):
    {yaml.safe_dump(stats, sort_keys=False, allow_unicode=True)}

    Produce a `_folder-summary.md` for this folder using this structure:

    # {folder.name}

    ## What this folder is
    2-4 sentences. Infer the editorial story from clip descriptions + dates +
    locations + recurring themes. Mention date range, dominant location(s),
    main subjects/themes.

    ## Rating breakdown
    Read directly from the precomputed stats. Format as:
    - Keep: N (X%)
    - Review: N (X%)
    - Cull: N (X%)

    ## Visual character
    Brief bullets describing the lighting/time-of-day/color/stability profile.
    Use the precomputed stats. Examples:
    - "Mostly golden hour (18/24 clips), with 4 midday and 2 night."
    - "Dominant palette: amber, ochre, dusty olive — warm savanna tones."
    - "Stability: 20 smooth (drone/gimbal), 4 handheld."

    ## Best moments
    5-10 bullets of the top-rated clips, each as:
    - **[filename]** — short description, why it stands out.
    Pull from the keep-rated sample. Use actual filenames + content from the
    sidecars provided.

    ## Cull pile
    Bullet list of the cull-rated clips with their cull_reason. Tell the user
    these are safe to delete. If 0 culls, write "No clips marked for cull."

    ## Recurring themes / subjects
    1-2 sentences synthesizing the top_keywords stats + content patterns.

    ## People in this folder
    Use the named_people stats. If empty (no named clusters yet), write
    "Faces detected but not yet labeled — run fdx-faces to identify."
    Otherwise list each named person and their clip count.

    ## Gaps
    Honest assessment of what's missing or weak in the coverage. E.g.:
    - No clean B-roll of the finished interior
    - No founder-message clips
    - No multilingual versions

    Keep it factual and concrete. Use real filenames + numbers from the
    stats. No flowery language. No invented details.

    --- SAMPLED SIDECARS BEGIN ---

    {sample_text}

    --- SAMPLED SIDECARS END ---
    """).strip()

    msg = client.messages.create(
        model=SUMMARY_MODEL,
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", "")).strip()
    return ""


def find_summarizable_folders(root: Path, min_clips: int) -> list[Path]:
    """Find every folder under root that contains ≥ min_clips sidecars
    (immediate children only — we summarize each level)."""
    folders: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if p.name.startswith((".", "_")):
            continue
        immediate = find_sidecars_immediate(p)
        if len(immediate) >= min_clips:
            folders.append(p)
    # Also include the root itself if it has direct sidecars
    if len(find_sidecars_immediate(root)) >= min_clips:
        folders.insert(0, root)
    return sorted(set(folders))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="Drive/folder root")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate summaries even if _folder-summary.md exists",
    )
    parser.add_argument(
        "--min-clips",
        type=int,
        default=DEFAULT_MIN_CLIPS,
        help=f"Minimum sidecars in a folder to generate a "
        f"summary (default: {DEFAULT_MIN_CLIPS}).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    client = anthropic.Anthropic(api_key=resolve_api_key())

    folders = find_summarizable_folders(root, args.min_clips)
    print(f"Found {len(folders)} folder(s) with ≥{args.min_clips} sidecars.")

    for folder in folders:
        out = folder / SUMMARY_FILE
        if out.exists() and not args.force:
            print(f"  skip (exists): {out.relative_to(root)}")
            continue
        # For the summary, use ALL sidecars in this folder (recursively)
        # — gives a fuller picture than just immediate children.
        sidecars = find_sidecars_recursive(folder)
        if not sidecars:
            continue
        records = [r for r in (parse_sidecar(s) for s in sidecars) if r]
        print(
            f"  summarizing {folder.relative_to(root) if folder != root else '.'} "
            f"({len(records)} clips)..."
        )
        try:
            summary = summarize_folder(client, folder, sidecars, records)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        out.write_text(summary + "\n")
        print(f"    -> {out.relative_to(root) if folder != root else SUMMARY_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
