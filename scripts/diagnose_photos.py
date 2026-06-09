#!/usr/bin/env python3
"""
diagnose_photos.py — Check how many videos in the Photos library are actually
on local disk vs iCloud-only. Useful for understanding what fdx-photos can
process without --download.

Usage:
    .venv/bin/python scripts/diagnose_photos.py
    .venv/bin/python scripts/diagnose_photos.py /path/to/Other.photoslibrary
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

try:
    import osxphotos
except ImportError:
    sys.exit("osxphotos not installed. Run: uv pip install -e '.[photos]'")


def main() -> int:
    if len(sys.argv) > 1:
        library = Path(sys.argv[1]).expanduser()
    else:
        library = Path.home() / "Pictures" / "Photos Library.photoslibrary"

    if not library.exists():
        sys.exit(f"Photos library not found: {library}")

    print(f"Library: {library}\n")

    db = osxphotos.PhotosDB(dbfile=str(library))
    # fdx-photos defaults to --media all, so diagnose all assets (not just movies).
    assets = db.photos(images=True, movies=True)

    on_disk_count = 0
    icloud_count = 0
    edited_count = 0
    n_videos = 0
    n_images = 0
    total_size_mb = 0

    for p in assets:
        if getattr(p, "ismovie", False):
            n_videos += 1
        else:
            n_images += 1
        pth = p.path
        if pth and os.path.exists(pth):
            on_disk_count += 1
            with contextlib.suppress(OSError):
                total_size_mb += os.path.getsize(pth) // (1024 * 1024)
        else:
            icloud_count += 1
        if getattr(p, "hasadjustments", False):
            edited_count += 1

    print(f"Total assets in library: {len(assets)} ({n_videos} videos, {n_images} images)")
    print(f"  on local disk:    {on_disk_count}   ({total_size_mb} MB combined)")
    print(f"  iCloud-only:      {icloud_count}")
    print(f"  with Photos edits: {edited_count}")
    print()

    print("First 10 sample (showing both local and iCloud):")
    for p in assets[:10]:
        pth = p.path
        exists = bool(pth and os.path.exists(pth))
        sz_mb = (os.path.getsize(pth) // (1024 * 1024)) if exists else 0
        tag = "LOCAL" if exists else "iCloud"
        kind = "vid" if getattr(p, "ismovie", False) else "img"
        date = p.date.isoformat()[:10] if p.date else "no-date"
        name = (p.original_filename or "")[:28]
        print(
            f"  [{tag:6s}] {kind}  {p.uuid[:8]}  {name:30s}  {date}  "
            f"{sz_mb:>5}MB  edited={getattr(p, 'hasadjustments', False)}"
        )

    print()
    if on_disk_count == 0:
        print(
            "⚠  All assets are iCloud-only. Two options to process them:\n"
            "   1. Photos → Settings → iCloud → 'Download Originals to this Mac'\n"
            "      (slow first sync but then fdx-photos works without --download)\n"
            "   2. fdx-photos --download (per-asset PhotoKit fetch — needs the\n"
            "      terminal to have Photos permission in System Settings)"
        )
    elif icloud_count == 0:
        print(
            f"✓ All {on_disk_count} assets are local. fdx-photos can process them "
            "directly without --download."
        )
    else:
        print(
            f"✓ {on_disk_count} assets can be indexed immediately without --download.\n"
            f"  Run: fdx-photos --max-files 5     # test on local ones first\n"
            f"  The remaining {icloud_count} need --download (or 'Download Originals')."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
