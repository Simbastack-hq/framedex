#!/usr/bin/env python3
"""
framedex.xmp_export — `fdx-xmp`, a regenerable Lightroom view of the sidecars.

Reads the plain-text `.description.md` sidecars framedex already wrote and emits
standard `.xmp` sidecars next to the proprietary-RAW originals, so Lightroom
Classic (Metadata -> Read Metadata from Files), Bridge, and exiftool pick up the
rating, keywords, and one-line caption. Plain text stays the source of truth —
delete every `.xmp` and re-run to regenerate them.

Never modifies an original, never touches a foreign or user-edited `.xmp`
(ownership is proven by a content-hash manifest, not by CreatorTool). Pure
stdlib: the XMP payload is a small RDF/XML template, no lxml.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml

import framedex
from framedex import images
from framedex.pipeline import SIDECAR_SUFFIX, atomic_write_text

# Lightroom Classic reads `.xmp` *sidecars* only for proprietary RAW; DNG (like
# JPEG/TIFF/HEIC) embeds XMP in the file and ignores the sidecar, so `.dng` is
# out of the default set.
XMP_SIDECAR_EXTENSIONS = images.RAW_EXTENSIONS - {".dng"}

# The model's `keep` is conservative archive-triage, not a pick — 3 stars leaves
# headroom for the photographer's own 4/5-star selects; only `cull` gets a loud
# color label. Constants, not flags (repo tunables convention).
RATING_MAP = {"keep": 3, "review": 2, "cull": 1}
CULL_LABEL = "Red"

# Provenance stamp only — overwrite safety is manifest-hash-based, not this.
CREATOR_TOOL = f"framedex {framedex.__version__}"

MANIFEST_NAME = "_XMP_MANIFEST.json"

# Uninformative scene_type values that shouldn't pollute the keyword bag.
_SKIP_SCENE_TYPES = {"", "unclear", "other"}

_SCENE_RE = re.compile(r"\*\*Scene:\*\*\s*(.+)")

# Characters XML 1.0 forbids even escaped (C0 controls except tab/newline/CR).
# Keywords/scene come from an LLM, so a stray one would make the .xmp
# not-well-formed and Lightroom would silently reject it — strip them.
_XML_INVALID_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _xml_text(s: str) -> str:
    """Escape a string for XML element text, first dropping the control
    characters XML 1.0 forbids, so the payload is always well-formed."""
    return escape(_XML_INVALID_RE.sub("", s))


# ---------------------------------------------------------------------------
# Sidecar reading (local: also pulls the body's Scene sentence, which
# query.parse_sidecar does not return)
# ---------------------------------------------------------------------------


def read_sidecar_for_xmp(path: Path) -> tuple[dict[str, Any], str] | None:
    """Parse a `.description.md` sidecar into (frontmatter, scene_sentence).

    Returns None if the file isn't a parseable framedex sidecar. `scene` is the
    single `**Scene:**` sentence from the body (not the whole description)."""
    try:
        text = path.read_text()
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict):
        return None
    m = _SCENE_RE.search(parts[2])
    scene = m.group(1).strip() if m else ""
    return fm, scene


# ---------------------------------------------------------------------------
# XMP payload
# ---------------------------------------------------------------------------


def _subject_tags(frontmatter: dict[str, Any]) -> list[str]:
    """keywords + scene_type, deduped and order-preserving, uninformative
    scene_type values dropped."""
    tags: list[str] = []
    for kw in frontmatter.get("keywords") or []:
        if isinstance(kw, str) and kw.strip() and kw.strip() not in tags:
            tags.append(kw.strip())
    scene_type = frontmatter.get("scene_type")
    if (
        isinstance(scene_type, str)
        and scene_type not in _SKIP_SCENE_TYPES
        and scene_type not in tags
    ):
        tags.append(scene_type)
    return tags


def build_xmp(frontmatter: dict[str, Any], scene: str) -> str:
    """Build the `.xmp` sidecar payload (RDF/XML) for one media file. `rating`
    must be a valid key of RATING_MAP (the caller validates + skips otherwise)."""
    stars = RATING_MAP[frontmatter["rating"]]
    label = CULL_LABEL if frontmatter["rating"] == "cull" else ""

    desc_attrs = [
        'rdf:about=""',
        'xmlns:xmp="http://ns.adobe.com/xap/1.0/"',
        'xmlns:dc="http://purl.org/dc/elements/1.1/"',
        f'xmp:Rating="{stars}"',
    ]
    if label:
        desc_attrs.append(f'xmp:Label="{escape(label)}"')
    desc_attrs.append(f'xmp:CreatorTool="{escape(CREATOR_TOOL)}"')

    lines = [
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        "  <rdf:Description " + "\n      ".join(desc_attrs) + ">",
    ]

    tags = _subject_tags(frontmatter)
    if tags:
        bag = "".join(f"<rdf:li>{_xml_text(t)}</rdf:li>" for t in tags)
        lines.append(f"   <dc:subject><rdf:Bag>{bag}</rdf:Bag></dc:subject>")
    if scene:
        lines.append(
            "   <dc:description><rdf:Alt>"
            f'<rdf:li xml:lang="x-default">{_xml_text(scene)}</rdf:li>'
            "</rdf:Alt></dc:description>"
        )

    lines += [
        "  </rdf:Description>",
        " </rdf:RDF>",
        "</x:xmpmeta>",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# File targeting + manifest-based overwrite safety
# ---------------------------------------------------------------------------


def original_for_sidecar(sidecar: Path) -> Path:
    """`DSC_1.RAF.description.md` -> `DSC_1.RAF` (sits next to the sidecar)."""
    return sidecar.with_name(sidecar.name[: -len(SIDECAR_SUFFIX)])


def xmp_target(original: Path) -> Path:
    """`DSC_1.RAF` -> `DSC_1.xmp` — the `<stem>.xmp` name Lightroom/Bridge share."""
    return original.with_suffix(".xmp")


def _payload_hash(payload: str) -> str:
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_manifest(root: Path) -> dict[str, str]:
    try:
        data = json.loads((root / MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class ExportSummary:
    wrote: int = 0
    up_to_date: int = 0
    conflicts: int = 0
    skipped_video: int = 0
    skipped_photos: int = 0
    skipped_nonraw: int = 0
    skipped_missing_original: int = 0
    skipped_malformed: int = 0
    skipped_collision: int = 0


def _decide(target: Path, new_hash: str, key: str, manifest: dict[str, str]) -> str:
    """'write' (safe to create/overwrite), 'up_to_date' (already exactly ours),
    or 'conflict' (foreign or edited-since — never clobber)."""
    if not target.exists():
        return "write"
    try:
        existing_hash = hashlib.sha1(target.read_bytes()).hexdigest()
    except OSError:
        return "conflict"
    if existing_hash == new_hash:
        return "up_to_date"
    # Ours and unmodified since we wrote it → safe to regenerate. Anything else
    # (foreign file, or ours but hand-edited in Lightroom) → conflict.
    if manifest.get(key) == existing_hash:
        return "write"
    return "conflict"


def run(root: Path, *, dry_run: bool = False) -> ExportSummary:
    """Export `.xmp` sidecars for every proprietary-RAW original under `root`
    that has a framedex sidecar. Non-destructive: never touches originals or a
    foreign/edited `.xmp`. Returns an ExportSummary of what happened."""
    summary = ExportSummary()
    manifest = _load_manifest(root)

    # Pre-pass: resolve every sidecar to (target, payload), collecting skips.
    # target -> (original, payload); a target claimed by two RAW originals
    # (same stem, e.g. DSC_1.CR2 + DSC_1.NEF) is a collision — first by sorted
    # original path wins, the rest are skipped, so writes never stream-clobber.
    candidates: dict[Path, tuple[Path, str]] = {}
    for sidecar in sorted(root.rglob("*" + SIDECAR_SUFFIX)):
        parsed = read_sidecar_for_xmp(sidecar)
        if parsed is None:
            summary.skipped_malformed += 1
            continue
        fm, scene = parsed
        if fm.get("media_type") == "video":
            summary.skipped_video += 1
            continue
        if fm.get("photos_uuid"):
            summary.skipped_photos += 1
            continue
        original = original_for_sidecar(sidecar)
        if original.suffix.lower() not in XMP_SIDECAR_EXTENSIONS:
            summary.skipped_nonraw += 1
            continue
        if not original.exists():
            print(f"skip: {sidecar.name} — original {original.name} not found")
            summary.skipped_missing_original += 1
            continue
        if fm.get("rating") not in RATING_MAP:
            print(f"skip: {original.name} — unknown rating {fm.get('rating')!r}")
            summary.skipped_malformed += 1
            continue
        target = xmp_target(original)
        if target in candidates:
            print(
                f"skip: {original.name} — {target.name} already claimed by "
                f"{candidates[target][0].name} (stem collision)"
            )
            summary.skipped_collision += 1
            continue
        candidates[target] = (original, build_xmp(fm, scene))

    # Write pass.
    for target in sorted(candidates):
        _original, payload = candidates[target]
        new_hash = _payload_hash(payload)
        key = str(target.relative_to(root))
        decision = _decide(target, new_hash, key, manifest)
        if decision == "conflict":
            print(f"conflict: {target.name} exists and isn't ours — skipped")
            summary.conflicts += 1
            continue
        if decision == "up_to_date":
            manifest[key] = new_hash
            summary.up_to_date += 1
            continue
        # decision == "write"
        summary.wrote += 1
        if dry_run:
            print(f"would write: {target.name}")
            continue
        atomic_write_text(target, payload)
        manifest[key] = new_hash

    if not dry_run and (summary.wrote or summary.up_to_date):
        atomic_write_text(root / MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fdx-xmp",
        description="Export framedex ratings/keywords to Lightroom via .xmp "
        "sidecars next to proprietary-RAW originals. Regenerable, "
        "non-destructive: never edits an original or a foreign .xmp.",
    )
    parser.add_argument("root", help="Folder to scan for framedex sidecars")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching any file",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        sys.exit(f"Root not found: {root}")

    s = run(root, dry_run=args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"\n{verb} {s.wrote}, up-to-date {s.up_to_date}, conflicts {s.conflicts}; "
        f"skipped (video {s.skipped_video}, photos {s.skipped_photos}, "
        f"non-raw {s.skipped_nonraw}, missing-original "
        f"{s.skipped_missing_original}, malformed {s.skipped_malformed}, "
        f"collision {s.skipped_collision})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
