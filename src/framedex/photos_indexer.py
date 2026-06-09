#!/usr/bin/env python3
"""
fdx-photos: Index videos directly from an Apple Photos Library.

Reads Photos.sqlite via osxphotos (no UI export → no metadata loss) and
runs the standard framedex per-clip pipeline against each movie asset.
Sidecars land in an external mirror tree, never inside the .photoslibrary
bundle.

Photos-side metadata (album, persons, keywords, canonical date) is merged
into each sidecar so the indexed KB preserves what Photos knows on top of
what ffprobe/Whisper/vision extract.

This entry is a thin *source adapter* over the same engine `fdx` uses: it
only decides how assets are enumerated (Photos.sqlite) and where sidecars go
(an external mirror). The per-clip pipeline, backend wiring, face-DB setup,
whisper loading, cost announcement, and result reporting are all shared via
`framedex.runner` + `framedex.index_videos`.

Tip: run `scripts/diagnose_photos.py` first to see how many videos are
already on local disk vs iCloud-only — it tells you whether you need
`--download` or whether you should just turn off Optimize Mac Storage in
Photos preferences instead.

Usage:
    fdx-photos                                  # default library + ~/framedex-photos
    fdx-photos --album "Yosemite 2024"
    fdx-photos --person "Mom" --since 2024-01-01
    fdx-photos --download                       # materialize iCloud-only assets
    fdx-photos --output ~/Documents/photos-kb   # custom mirror tree
    fdx-photos --max-files 5                    # try 5 clips before going wide
    fdx-photos --uuid ABCD1234-... --force      # re-process a single clip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from framedex import face_db, runner
from framedex.index_videos import process_one_video, setup_whisper
from framedex.pipeline import (
    NominatimRateLimiter,
    ProcessContext,
    ProcessOptions,
)

DEFAULT_OUTPUT = Path.home() / "framedex-photos"
DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
VISION_MODEL_DEFAULT = "haiku"
WHISPER_FIXES_DEFAULT = Path.home() / ".framedex" / "whisper_fixes.json"


def _parse_date(s: str) -> datetime:
    """Accept YYYY-MM-DD or full ISO datetime."""
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid date {s!r}: {e}") from e


def main() -> int:
    # osxphotos is imported here (not at module scope) so this module stays
    # importable — and its pure helpers testable — without the [photos] extra.
    try:
        from framedex import photos as photos_mod
    except ImportError as e:
        print(
            "fdx-photos requires the 'osxphotos' extra. "
            f"Install with: uv pip install -e '.[photos]'\n  ({e})",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(
        description="Index videos in an Apple Photos Library (no export).",
    )
    parser.add_argument(
        "--library",
        default=str(photos_mod.DEFAULT_LIBRARY),
        help="Path to .photoslibrary (default: ~/Pictures/Photos Library.photoslibrary)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Mirror tree for sidecars (default: {DEFAULT_OUTPUT}). Sidecars "
        "land at {output}/{YYYY-MM}/{filename}__{uuid8}.description.md. "
        "Never inside the .photoslibrary bundle.",
    )
    parser.add_argument(
        "--album",
        action="append",
        default=[],
        help="Restrict to assets in this album (repeatable, OR-combined)",
    )
    parser.add_argument(
        "--person",
        action="append",
        default=[],
        help="Restrict to assets featuring this Photos-side person name "
        "(repeatable, OR-combined)",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Restrict to assets with this Photos keyword (repeatable, OR)",
    )
    parser.add_argument(
        "--since",
        type=_parse_date,
        default=None,
        help="Only assets dated >= this date (YYYY-MM-DD or ISO datetime)",
    )
    parser.add_argument(
        "--until",
        type=_parse_date,
        default=None,
        help="Only assets dated <= this date",
    )
    parser.add_argument(
        "--uuid",
        action="append",
        default=[],
        help="Process only this asset UUID (repeatable). Useful for re-running "
        "a single problem clip.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Materialize iCloud-only originals via PhotoKit before processing. "
        "Without this, assets with no on-disk original are skipped.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process assets even if a sidecar exists in the mirror tree",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assets that would be processed; no model calls",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Stop after N assets (for test runs)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=30,
        help="Skip clips longer than N minutes (default: 30; 0 = no limit)",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo",
        help="Whisper model: tiny/base/small/medium/large-v3/large-v3-turbo",
    )
    parser.add_argument(
        "--no-diarize", action="store_true", help="Skip speaker diarization"
    )
    parser.add_argument(
        "--no-geocode",
        action="store_true",
        help="Skip Nominatim reverse geocoding (Photos GPS still recorded)",
    )
    parser.add_argument("--no-faces", action="store_true", help="Skip face detection")
    parser.add_argument(
        "--face-db",
        default=str(face_db.DB_PATH_DEFAULT),
        help=f"Path to face DB (default: {face_db.DB_PATH_DEFAULT})",
    )
    parser.add_argument(
        "--backend",
        choices=["cli", "api", "local"],
        default="cli",
        help="Vision backend (cli/api/local). See fdx --help for details.",
    )
    parser.add_argument(
        "--vision-model",
        default=VISION_MODEL_DEFAULT,
        help="Claude vision model when --backend cli or api (default: haiku)",
    )
    parser.add_argument("--local-base-url", default=DEFAULT_LOCAL_BASE_URL)
    parser.add_argument("--local-model", default=None)
    parser.add_argument(
        "--no-whisper-prompt",
        action="store_true",
        help="Disable proper-noun biasing (Photos has no .video-context.md "
        "chain anyway, but kept for parity with fdx)",
    )
    parser.add_argument(
        "--whisper-fixes",
        default=str(WHISPER_FIXES_DEFAULT),
        help=f"Path to JSON regex fixes (default: {WHISPER_FIXES_DEFAULT})",
    )
    args = parser.parse_args()

    library = Path(args.library).expanduser().resolve()
    if not library.exists():
        sys.exit(f"Photos library not found: {library}")
    output_root = Path(args.output).expanduser().resolve()
    # Refuse to write inside the .photoslibrary bundle — Photos will treat
    # foreign files as corruption.
    try:
        output_root.relative_to(library)
        sys.exit(
            f"--output ({output_root}) is inside the Photos library bundle. "
            "Choose a path outside the .photoslibrary."
        )
    except ValueError:
        pass

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Library: {library}")
    print(f"Mirror:  {output_root}\n")

    print("Enumerating assets...")
    all_assets = photos_mod.enumerate_videos(
        library,
        albums=args.album or None,
        persons=args.person or None,
        keywords=args.keyword or None,
        since=args.since,
        until=args.until,
        uuids=args.uuid or None,
    )
    print(f"  found {len(all_assets)} movie asset(s)")

    # Resume: skip assets whose mirror sidecar already exists.
    def existing_sidecar(asset: photos_mod.PhotosAsset) -> Path:
        return photos_mod.mirror_sidecar_path(asset, output_root)

    if args.force:
        todo = list(all_assets)
    else:
        todo = [a for a in all_assets if not existing_sidecar(a).exists()]
    skipped_existing = len(all_assets) - len(todo)
    if skipped_existing:
        print(f"  skipping {skipped_existing} already-indexed (sidecar exists)")

    # iCloud handling
    missing = [a for a in todo if a.in_icloud]
    if missing and not args.download:
        print(
            f"  {len(missing)} asset(s) live only in iCloud and have no local "
            "original. Pass --download to materialize them; for now they will "
            "be skipped."
        )
        todo = [a for a in todo if not a.in_icloud]
    elif missing and args.download:
        print(
            f"  {len(missing)} asset(s) need download from iCloud — will "
            "fetch on demand per clip"
        )

    if args.max_files and len(todo) > args.max_files:
        todo = todo[: args.max_files]
        print(f"  limiting to {len(todo)} for this run")

    if not todo:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        for a in todo:
            sidecar = existing_sidecar(a)
            tag = "iCloud" if a.in_icloud else "local"
            date = a.date.isoformat() if a.date else "no-date"
            print(f"  would process: {a.filename}  [{tag}, {date}, uuid={a.uuid[:8]}]")
            print(f"    -> {sidecar}")
        return 0

    # Shared engine: model resolution, cost announce, backend wiring, face DB,
    # and the whisper stack all come from runner / index_videos so this entry
    # and `fdx` stay in lockstep.
    model_id, cost_per_call = runner.resolve_vision_model(args)
    runner.announce_cost(
        args.backend, model_id, cost_per_call, len(todo), args.local_base_url
    )
    api_client = runner.wire_vision_backend(args)
    face_conn = runner.setup_face_db(args)
    whisper_model, align_models, diarize_pipeline, whisper_fixes = setup_whisper(args)

    geocoder = NominatimRateLimiter() if not args.no_geocode else None

    max_duration_seconds = args.max_duration * 60 if args.max_duration > 0 else None
    opts = ProcessOptions(
        backend=args.backend,
        vision_model_id=model_id,
        local_base_url=args.local_base_url,
        local_model=args.local_model,
        cost_per_call=float(cost_per_call),
        no_whisper_prompt=args.no_whisper_prompt,
        whisper_fixes=whisper_fixes,
        max_duration_seconds=max_duration_seconds,
    )
    ctx = ProcessContext(
        whisper_model=whisper_model,
        align_models=align_models,
        diarize_pipeline=diarize_pipeline,
        geocoder=geocoder,
        api_client=api_client,
        face_conn=face_conn,
    )

    # tmp dir for materialized iCloud assets (only used when --download fetches)
    tmp_root = Path(tempfile.mkdtemp(prefix="fdx-photos-download-"))

    tally = runner.RunTally()
    skipped_missing = 0

    try:
        for i, asset in enumerate(todo, start=1):
            print(f"[{i}/{len(todo)}] {asset.filename}  (uuid={asset.uuid[:8]})")
            try:
                video_path, status = photos_mod.materialize(
                    asset,
                    tmp_root / asset.uuid,
                    allow_download=args.download,
                )
                if status == "downloaded":
                    print(f"  downloaded from iCloud → {video_path}")
                if video_path is None:
                    if status == "missing":
                        print("  skipped (no local original; pass --download to fetch)")
                    elif status.startswith("failed:"):
                        print(f"  skipped — download {status[7:]}")
                    else:
                        print(f"  skipped ({status})")
                    skipped_missing += 1
                    continue

                sidecar_target = photos_mod.mirror_sidecar_path(asset, output_root)
                sidecar_target.parent.mkdir(parents=True, exist_ok=True)

                result = process_one_video(
                    video_path,
                    library,  # used for relative-path computation only
                    opts,
                    ctx,
                    sidecar_path_override=sidecar_target,
                    parent_folder_override=photos_mod.parent_folder_for(asset),
                    metadata_override=photos_mod.to_metadata_override(asset),
                    gps_override=photos_mod.to_gps_override(asset),
                    place_override=None,  # let geocoder run on Photos GPS
                    extra_frontmatter=photos_mod.to_extra_frontmatter(asset),
                    omit_path=status == "downloaded",
                    proper_nouns=[],  # Photos has no .video-context.md chain
                )
                runner.record_result(
                    result,
                    tally,
                    backend=args.backend,
                    max_duration_min=args.max_duration,
                )
            except KeyboardInterrupt:
                print(
                    "\nInterrupted. Re-run to resume — finished sidecars are "
                    "skipped automatically."
                )
                break
            except Exception as e:
                tally.errors += 1
                print(f"  ERROR: {e}")
                traceback.print_exc()
                continue
    finally:
        # Clean up materialized iCloud downloads
        shutil.rmtree(tmp_root, ignore_errors=True)

    summary = f"\nDone. Processed: {tally.processed}, Errors: {tally.errors}"
    if tally.skipped_too_long:
        summary += f", Skipped (too long): {tally.skipped_too_long}"
    if tally.skipped_no_preview:
        summary += f", Skipped (no preview): {tally.skipped_no_preview}"
    if skipped_missing:
        summary += f", Skipped (missing/iCloud): {skipped_missing}"
    if args.backend == "api":
        summary += f", Approx cost: ${tally.actual_cost:.2f}"
    print(summary)
    if face_conn is not None:
        s = face_db.db_stats(face_conn)
        print(
            f"Face DB: {s['faces']} total faces, {s['clusters']} clusters "
            f"({s['named_clusters']} named)."
        )
        face_conn.close()
    return 0 if tally.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
