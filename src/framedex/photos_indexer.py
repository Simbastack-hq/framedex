#!/usr/bin/env python3
"""
fdx-photos: Index media directly from an Apple Photos Library.

Reads Photos.sqlite via osxphotos (no UI export → no metadata loss) and runs
the standard framedex per-asset pipeline against each asset — videos, stills,
or both (`--media all|videos|images`, default `all`). Sidecars land in an
external mirror tree, never inside the .photoslibrary bundle.

Photos-side metadata (album, persons, keywords, canonical date, GPS) is merged
into each sidecar so the indexed KB preserves what Photos knows on top of what
ffprobe/exiftool/Whisper/vision extract.

This entry is a thin *source adapter* over the same engine `fdx` uses: it only
decides how assets are enumerated (Photos.sqlite) and where sidecars go (an
external mirror), then routes each asset to the video or still pipeline by
kind. Backend wiring, face-DB setup, whisper loading, cost announcement, and
result reporting are all shared via `framedex.runner` / `framedex.index_videos`
/ `framedex.images`.

Tip: run `scripts/diagnose_photos.py` first to see how many assets are already
on local disk vs iCloud-only — it tells you whether you need `--download` or
whether you should just turn off Optimize Mac Storage in Photos preferences.

Usage:
    fdx-photos                                  # default library, all media
    fdx-photos --media images                   # stills only (no whisper stack)
    fdx-photos --album "Yosemite 2024"
    fdx-photos --person "Mom" --since 2024-01-01
    fdx-photos --download                       # materialize iCloud-only assets
    fdx-photos --output ~/Documents/photos-kb   # custom mirror tree
    fdx-photos --max-files 5                    # try 5 assets before going wide
    fdx-photos --uuid ABCD1234-... --force      # re-process a single asset
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from framedex import face_db, images, runner
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
# Above this many queued assets, an unfiltered run gets a loud heads-up (it is
# almost certainly the whole library). Runs are incremental + resumable, so we
# warn rather than block.
LARGE_RUN_THRESHOLD = 1000


def _parse_date(s: str) -> datetime:
    """Accept YYYY-MM-DD or full ISO datetime."""
    try:
        return datetime.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid date {s!r}: {e}") from e


def is_large_unfiltered_run(
    n_todo: int, has_filter: bool, threshold: int = LARGE_RUN_THRESHOLD
) -> bool:
    """True when a run is large enough AND unfiltered enough to warrant a loud
    heads-up. Any narrowing filter — including --max-files (a deliberately
    scoped test run) — suppresses the warning."""
    return n_todo > threshold and not has_filter


def missing_media_extras(
    need_video: bool,
    need_image: bool,
    *,
    has_whisperx: bool,
    has_pillow: bool,
) -> list[str]:
    """Return the install extras missing for the requested media kinds.

    Pure (the caller probes availability via importlib) so it is unit-testable.
    `has_pillow` means the full `[images]` extra is present (Pillow *and*
    pillow-heif). With `[photos]` scoped to osxphotos only, a mixed run needs
    `[video]` for clips and `[images]` for stills; this turns a mid-run lazy
    ImportError into one upfront actionable message."""
    missing: list[str] = []
    if need_video and not has_whisperx:
        missing.append("video")
    if need_image and not has_pillow:
        missing.append("images")
    return missing


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
        description="Index media (videos + stills) in an Apple Photos Library "
        "(no export).",
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
        "--media",
        choices=["all", "videos", "images"],
        default="all",
        help="Which media to index: 'all' (default), 'videos' only, or "
        "'images' only. An images-only run skips the whisper/audio stack "
        "entirely (needs [photos,images], not [video]).",
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
    parser.add_argument(
        "--frame-sampling",
        choices=["diverse", "even"],
        default="diverse",
        help="How vision frames are picked. 'diverse' (default) samples "
        "small thumbnails across the clip and keeps the most mutually "
        "different, sharpest moments. 'even' is the legacy evenly-spaced "
        "sampling.",
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
    all_assets = photos_mod.enumerate_assets(
        library,
        media=args.media,
        albums=args.album or None,
        persons=args.person or None,
        keywords=args.keyword or None,
        since=args.since,
        until=args.until,
        uuids=args.uuid or None,
    )
    n_img_all = sum(1 for a in all_assets if a.media_type == "image")
    n_vid_all = len(all_assets) - n_img_all
    print(
        f"  found {len(all_assets)} asset(s) "
        f"({n_vid_all} video(s), {n_img_all} image(s))"
    )

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

    need_video = any(a.media_type == "video" for a in todo)
    need_image = any(a.media_type == "image" for a in todo)

    # Preflight the install extras for the media actually queued so a missing
    # dependency surfaces as one upfront message, not a lazy mid-run failure.
    # ([photos] is osxphotos-only: clips need [video], stills need [images].)
    # [images] ships Pillow + pillow-heif together; require both, since an Apple
    # Photos library is HEIC-heavy and render_preview needs pillow-heif to decode
    # HEIC. Checking PIL alone would false-pass and fail mid-run on the first HEIC.
    has_images_extra = (
        importlib.util.find_spec("PIL") is not None
        and importlib.util.find_spec("pillow_heif") is not None
    )
    missing_extras = missing_media_extras(
        need_video,
        need_image,
        has_whisperx=importlib.util.find_spec("whisperx") is not None,
        has_pillow=has_images_extra,
    )
    if missing_extras:
        extras = ",".join(["photos", *missing_extras])
        sys.exit(
            f"This run needs additional extras for the selected media "
            f"(missing: {', '.join(missing_extras)}).\n"
            f"  Install with: uv pip install -e '.[{extras}]'"
        )

    # Shared engine: model resolution, cost announce, backend wiring, face DB,
    # and (only when a clip is queued) the whisper stack — all from runner /
    # index_videos so this entry and `fdx` stay in lockstep.
    model_id, cost_per_call = runner.resolve_vision_model(args)
    runner.announce_cost(
        args.backend, model_id, cost_per_call, len(todo), args.local_base_url
    )

    has_filter = bool(
        args.album
        or args.person
        or args.keyword
        or args.since
        or args.until
        or args.uuid
        or args.max_files
    )
    if is_large_unfiltered_run(len(todo), has_filter):
        print(
            f"  ! Large run: {len(todo)} assets queued with no filter — this "
            "indexes (most of) your whole library.\n"
            "    It runs incrementally and is resumable: sidecars appear as it "
            "goes, and you can Ctrl-C anytime\n"
            "    and re-run to continue. To scope it, add --album / --person / "
            "--since, or test with --max-files N first.\n"
        )

    api_client = runner.wire_vision_backend(args)
    face_conn = runner.setup_face_db(args)

    whisper_model = None
    align_models: dict[str, Any] = {}
    diarize_pipeline = None
    whisper_fixes: list[tuple[re.Pattern[str], str]] = []
    if need_video:
        whisper_model, align_models, diarize_pipeline, whisper_fixes = setup_whisper(
            args
        )

    geocoder = NominatimRateLimiter() if not args.no_geocode else None

    max_duration_seconds = args.max_duration * 60 if args.max_duration > 0 else None
    opts = ProcessOptions(
        backend=args.backend,
        vision_model_id=model_id,
        local_base_url=args.local_base_url,
        local_model=args.local_model,
        cost_per_call=float(cost_per_call),
        no_whisper_prompt=args.no_whisper_prompt,
        frame_sampling=args.frame_sampling,
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
                asset_path, status = photos_mod.materialize(
                    asset,
                    tmp_root / asset.uuid,
                    allow_download=args.download,
                )
                if status == "downloaded":
                    print(f"  downloaded from iCloud → {asset_path}")
                if asset_path is None:
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

                # Route by asset kind. Both pipelines take the same Photos
                # override surface (sidecar mirror, parent folder, Photos GPS /
                # date, the photos_* frontmatter, and omit_path for temp copies).
                if asset.media_type == "video":
                    result = process_one_video(
                        asset_path,
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
                else:
                    result = images.process_one_image(
                        asset_path,
                        library,
                        opts,
                        ctx,
                        sidecar_path_override=sidecar_target,
                        parent_folder_override=photos_mod.parent_folder_for(asset),
                        metadata_override=photos_mod.to_metadata_override(asset),
                        gps_override=photos_mod.to_gps_override(asset),
                        place_override=None,
                        extra_frontmatter=photos_mod.to_extra_frontmatter(asset),
                        omit_path=status == "downloaded",
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
