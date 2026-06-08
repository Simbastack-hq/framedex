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
import os
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

from framedex import face_db
from framedex.parsing import pick_diar_auth_kwarg

try:
    from framedex import photos as photos_mod
except ImportError as e:
    print(
        "fdx-photos requires the 'osxphotos' extra. "
        f"Install with: uv pip install -e '.[photos]'\n  ({e})",
        file=sys.stderr,
    )
    sys.exit(1)


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

    try:
        import whisperx
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Run: uv pip install -e '.[photos]'", file=sys.stderr)
        return 1

    from framedex.index_videos import (
        load_whisper_fixes,
        process_one_video,
        resolve_hf_token,
    )
    from framedex.pipeline import (
        COST_PER_CALL_USD_CLI,
        COST_PER_CALL_USD_LOCAL,
        VISION_MODELS,
        NominatimRateLimiter,
        ProcessContext,
        ProcessOptions,
        check_claude_cli,
        check_local_endpoint,
        resolve_anthropic_key,
    )

    if args.vision_model not in VISION_MODELS:
        known = ", ".join(sorted(VISION_MODELS))
        sys.exit(f"--vision-model must be one of: {known}")

    model_cfg = VISION_MODELS[args.vision_model]
    if args.backend == "api":
        model_id = str(model_cfg["api"])
        cost_per_call = float(model_cfg["cost_per_call_api"])
    elif args.backend == "cli":
        model_id = str(model_cfg["cli"])
        cost_per_call = COST_PER_CALL_USD_CLI
    else:
        model_id = args.local_model or "(loaded model in LM Studio)"
        cost_per_call = COST_PER_CALL_USD_LOCAL

    est_cost = len(todo) * cost_per_call
    if args.backend == "api":
        print(f"  vision: api / {model_id}")
        print(f"  estimated Anthropic API cost: ~${est_cost:.2f}")
    elif args.backend == "cli":
        print(f"  vision: cli (Max) / {model_id}")
        print("  marginal cost: $0 (Max subscription)")
    else:
        print(f"  vision: local / {model_id} @ {args.local_base_url}")
        print("  marginal cost: $0 (fully local)")
    print()

    # Backend wiring (mirrors fdx main)
    api_client = None
    if args.backend == "api":
        api_key = resolve_anthropic_key()
        if not api_key:
            sys.exit(
                "--backend api requires ANTHROPIC_API_KEY env or "
                "~/.claude/credentials/anthropic-key.txt"
            )
        import anthropic as _anthropic

        api_client = _anthropic.Anthropic(api_key=api_key)
        print("Vision: direct Anthropic API\n")
    elif args.backend == "cli":
        if not check_claude_cli():
            sys.exit(
                "--backend cli requires the `claude` CLI on PATH. "
                "Install Claude Code or pass --backend api."
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "NOTE: ANTHROPIC_API_KEY is set; will be scrubbed from each\n"
                "      `claude` subprocess so calls go via Max subscription.\n"
            )
        else:
            print("Vision: claude CLI -> Max subscription\n")
    else:
        ok, info = check_local_endpoint(args.local_base_url)
        if not ok:
            sys.exit(f"--backend local: cannot reach {args.local_base_url} ({info}).")
        print(f"Vision: local LM Studio at {args.local_base_url} ({info})\n")

    face_conn = None
    if not args.no_faces:
        ok, info = face_db.init_face_app()
        if not ok:
            print(f"Face detection unavailable: {info}")
        else:
            face_conn = face_db.open_db(Path(args.face_db))
            stats = face_db.db_stats(face_conn)
            print(f"Face detection: insightface ({info})")
            print(
                f"Face DB: {args.face_db} (currently {stats['faces']} faces, "
                f"{stats['clusters']} clusters, {stats['named_clusters']} named)\n"
            )

    print(f"Loading Whisper model: {args.whisper_model}")
    whisper_model = whisperx.load_model(
        args.whisper_model, device="cpu", compute_type="int8"
    )
    print("Whisper ready.")

    whisper_fixes: list[tuple[re.Pattern[str], str]] = []
    if not args.no_whisper_prompt:
        whisper_fixes = load_whisper_fixes(Path(args.whisper_fixes).expanduser())
        if whisper_fixes:
            print(f"Whisper canonical fixes: {len(whisper_fixes)} rule(s)")

    diarize_pipeline = None
    if not args.no_diarize:
        hf = resolve_hf_token()
        if hf:
            try:
                from whisperx.diarize import DiarizationPipeline
            except ImportError:
                DiarizationPipeline = getattr(whisperx, "DiarizationPipeline", None)
            if DiarizationPipeline is not None:
                import inspect

                try:
                    params = inspect.signature(DiarizationPipeline.__init__).parameters
                except (ValueError, TypeError):
                    params = {}  # type: ignore[assignment]
                auth_kwarg = pick_diar_auth_kwarg(params)
                try:
                    diarize_pipeline = DiarizationPipeline(
                        **{auth_kwarg: hf}, device="cpu"
                    )
                    print(f"Diarization pipeline ready (auth kwarg: {auth_kwarg}).")
                except Exception as e:
                    print(f"Failed to load diarization pipeline: {e}")
        else:
            print(
                "HF_TOKEN not set — running without diarization. "
                "Pass --no-diarize to silence."
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
        whisper_fixes=whisper_fixes,
        max_duration_seconds=max_duration_seconds,
    )
    ctx = ProcessContext(
        whisper_model=whisper_model,
        diarize_pipeline=diarize_pipeline,
        geocoder=geocoder,
        api_client=api_client,
        face_conn=face_conn,
    )

    # tmp dir for materialized iCloud assets (only used when --download fetches)
    tmp_root = Path(tempfile.mkdtemp(prefix="fdx-photos-download-"))

    processed = 0
    errors = 0
    skipped_too_long = 0
    skipped_missing = 0
    actual_cost = 0.0

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

                if result.skipped_reason == "short":
                    print("  skipped (duration < 0.5s)")
                    continue
                if result.skipped_reason == "too_long":
                    print(
                        f"  skipped (duration > --max-duration {args.max_duration} min)"
                    )
                    skipped_too_long += 1
                    continue

                assert result.sidecar is not None
                actual_cost += result.cost
                processed += 1
                faces_note = (
                    f", {len(result.detected_faces)} faces"
                    if result.detected_faces
                    else ""
                )
                rating_note = f", rated {result.rating}"
                if args.backend == "api":
                    print(
                        f"  -> {result.sidecar.name}  "
                        f"(cost ~${actual_cost:.2f}{rating_note}{faces_note})"
                    )
                else:
                    print(f"  -> {result.sidecar.name}  ({rating_note}{faces_note})")
            except KeyboardInterrupt:
                print(
                    "\nInterrupted. Re-run to resume — finished sidecars are "
                    "skipped automatically."
                )
                break
            except Exception as e:
                errors += 1
                print(f"  ERROR: {e}")
                traceback.print_exc()
                continue
    finally:
        # Clean up materialized iCloud downloads
        shutil.rmtree(tmp_root, ignore_errors=True)

    summary = f"\nDone. Processed: {processed}, Errors: {errors}"
    if skipped_too_long:
        summary += f", Skipped (too long): {skipped_too_long}"
    if skipped_missing:
        summary += f", Skipped (missing/iCloud): {skipped_missing}"
    if args.backend == "api":
        summary += f", Approx cost: ${actual_cost:.2f}"
    print(summary)
    if face_conn is not None:
        s = face_db.db_stats(face_conn)
        print(
            f"Face DB: {s['faces']} total faces, {s['clusters']} clusters "
            f"({s['named_clusters']} named)."
        )
        face_conn.close()
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
