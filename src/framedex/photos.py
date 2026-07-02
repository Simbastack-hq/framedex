"""
photos.py — Apple Photos Library adapter for framedex.

Reads Photos.sqlite via osxphotos to enumerate assets (stills + videos) in the
user's library directly, without going through Photos UI export. This avoids the
metadata loss that "Export edited" / "Export unmodified original" can introduce:
.AAE sidecars, transcoding, GPS/date stripping.

Photos-side metadata (GPS, date, persons, albums, keywords) flows into the
sidecar frontmatter alongside the standard framedex vision/audio passes.

iCloud "Optimize Mac Storage" assets are detected via the on-disk presence
of `PhotoInfo.path` (osxphotos 0.69+, falls back to the older
`path_original` attribute). When the caller passes allow_download=True,
missing originals are materialized via osxphotos's PhotoExporter with
download_missing=True — PhotoKit first, AppleScript-via-PhotoScript as a
fallback when PhotoKit auth isn't granted.

Never writes inside the .photoslibrary bundle — sidecars go to a mirror
tree outside the library. macOS-only (osxphotos depends on PyObjC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _as_aware(dt: datetime) -> datetime:
    """Interpret a naive datetime as local time; pass aware ones through.

    osxphotos returns tz-aware asset dates, but --since/--until parsed from
    'YYYY-MM-DD' are naive. Comparing or sorting the two mixed raises
    TypeError, so we normalize both sides to aware before any comparison."""
    return dt.astimezone() if dt.tzinfo is None else dt


try:
    import osxphotos
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "fdx-photos requires the 'osxphotos' extra. "
        "Install with: uv pip install -e '.[photos]'"
    ) from e


DEFAULT_LIBRARY = Path.home() / "Pictures" / "Photos Library.photoslibrary"
SIDECAR_SUFFIX = ".description.md"


@dataclass
class PhotosAsset:
    """An asset (still or movie) enumerated from the Photos library.

    `raw` holds the underlying osxphotos PhotoInfo for advanced operations
    (export, PhotoKit fetch). All other fields are pre-projected so callers
    that just need to write a sidecar don't have to touch osxphotos types.
    `media_type` lets the indexer route each asset to the video or still
    pipeline without re-touching osxphotos.
    """

    uuid: str
    filename: str  # original filename (e.g. "IMG_4827.MOV")
    date: datetime | None
    media_type: Literal["image", "video"] = "video"
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None
    persons: list[str] = field(default_factory=list)
    albums: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    is_edited: bool = False
    in_icloud: bool = False  # True if original isn't currently on local disk
    path_original: Path | None = None  # absolute path; None when iCloud-only
    raw: Any = None  # osxphotos.PhotoInfo, opaque to callers


def _project(photo: Any) -> PhotosAsset:
    """Map an osxphotos PhotoInfo to our PhotosAsset dataclass."""
    loc = getattr(photo, "location", None)
    if loc and isinstance(loc, tuple) and len(loc) == 2:
        lat, lon = loc[0], loc[1]
    else:
        lat, lon = None, None

    # In osxphotos 0.69+ the canonical attribute for the on-disk original is
    # `.path` (it returns None when the original is still in iCloud).
    # Older 0.5x releases exposed `.path_original`. Probe both so a single
    # binary survives upgrades.
    p_raw = getattr(photo, "path", None) or getattr(photo, "path_original", None)
    p_path = Path(p_raw) if p_raw else None
    on_disk = bool(p_path and p_path.exists())

    # `ismovie` is the osxphotos discriminator. Default to a still when the
    # attribute is missing/falsey (Live Photos and plain images are stills; the
    # paired Live-Photo movie is not a separate asset here).
    media_type: Literal["image", "video"] = (
        "video" if getattr(photo, "ismovie", False) else "image"
    )

    return PhotosAsset(
        uuid=str(photo.uuid),
        filename=str(getattr(photo, "original_filename", None) or photo.filename or ""),
        date=getattr(photo, "date", None),
        media_type=media_type,
        lat=float(lat) if lat is not None else None,
        lon=float(lon) if lon is not None else None,
        altitude_m=None,  # osxphotos doesn't expose altitude on PhotoInfo
        persons=list(getattr(photo, "persons", []) or []),
        albums=list(getattr(photo, "albums", []) or []),
        keywords=list(getattr(photo, "keywords", []) or []),
        is_edited=bool(getattr(photo, "hasadjustments", False)),
        in_icloud=not on_disk,
        path_original=p_path if on_disk else None,
        raw=photo,
    )


def enumerate_assets(
    library: Path = DEFAULT_LIBRARY,
    *,
    media: Literal["all", "images", "videos"] = "all",
    albums: list[str] | None = None,
    persons: list[str] | None = None,
    keywords: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    uuids: list[str] | None = None,
) -> list[PhotosAsset]:
    """Enumerate assets in the Photos library, applying filters.

    `media` selects which kinds to return: 'all' (default), 'images', or
    'videos'. All other filters are AND-combined; empty/None lists mean "no
    filter on this dimension". When a date bound (`since`/`until`) is set,
    undated assets are excluded — they cannot satisfy a date range. Results
    are sorted by date then UUID for deterministic resumability.
    """
    want_images = media in ("all", "images")
    want_movies = media in ("all", "videos")
    db = osxphotos.PhotosDB(dbfile=str(library))
    photos = db.photos(images=want_images, movies=want_movies)
    results: list[PhotosAsset] = []
    album_set = set(albums) if albums else None
    person_set = set(persons) if persons else None
    keyword_set = set(keywords) if keywords else None
    uuid_set = set(uuids) if uuids else None
    for p in photos:
        if uuid_set and str(p.uuid) not in uuid_set:
            continue
        if album_set and not (set(p.albums or []) & album_set):
            continue
        if person_set and not (set(p.persons or []) & person_set):
            continue
        if keyword_set and not (set(p.keywords or []) & keyword_set):
            continue
        # A date filter excludes undated assets: an asset with no date cannot
        # satisfy a "since"/"until" bound.
        if since and (p.date is None or _as_aware(p.date) < _as_aware(since)):
            continue
        if until and (p.date is None or _as_aware(p.date) > _as_aware(until)):
            continue
        results.append(_project(p))
    results.sort(
        key=lambda a: (
            _as_aware(a.date) if a.date else datetime.min.replace(tzinfo=timezone.utc),
            a.uuid,
        )
    )
    return results


def enumerate_videos(
    library: Path = DEFAULT_LIBRARY,
    *,
    albums: list[str] | None = None,
    persons: list[str] | None = None,
    keywords: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    uuids: list[str] | None = None,
) -> list[PhotosAsset]:
    """Backwards-compatible wrapper: enumerate movie assets only."""
    return enumerate_assets(
        library,
        media="videos",
        albums=albums,
        persons=persons,
        keywords=keywords,
        since=since,
        until=until,
        uuids=uuids,
    )


def materialize(
    asset: PhotosAsset,
    tmp_dir: Path,
    *,
    allow_download: bool = True,
) -> tuple[Path | None, str]:
    """Ensure the original file is on disk.

    Returns a (path, status) tuple. `status` is one of:
    - 'on_disk'     — file was already local; path is asset.path_original
    - 'downloaded'  — iCloud asset fetched into tmp_dir
    - 'missing'     — file is iCloud-only and allow_download=False
    - 'failed:<msg>'— download attempted and failed; msg has details

    When status != 'on_disk' and 'downloaded', path is None.

    Uses osxphotos 0.69+ `PhotoExporter` + `ExportOptions` (the modern API
    where `download_missing` and `use_photokit` live on the options object,
    not on PhotoInfo.export directly).
    """
    if asset.path_original and asset.path_original.exists():
        return asset.path_original, "on_disk"
    if not allow_download:
        return None, "missing"
    if asset.raw is None:
        return None, "failed:no PhotoInfo handle"

    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from osxphotos import ExportOptions, PhotoExporter
    except ImportError as e:  # pragma: no cover
        return None, f"failed:osxphotos missing PhotoExporter ({e})"

    # Two transport options for iCloud-download:
    #   use_photokit: PhotoKit API (fast, but unsigned Python binaries often
    #                 get auth_status=denied without ever prompting the user
    #                 → must grant the parent terminal Photos access in TCC).
    #   use_photos_export: AppleScript via PhotoScript (slower, but uses
    #                      Automation permission which is easier to obtain).
    # Try PhotoKit first; if auth fails, fall back to AppleScript path.
    last_err: Exception | None = None
    for transport in ("use_photokit", "use_photos_export"):
        try:
            exporter = PhotoExporter(asset.raw)
            options = ExportOptions(
                download_missing=True,
                edited=False,
                overwrite=True,
                exiftool=False,
                **{transport: True},
            )
            result = exporter.export(str(tmp_dir), options=options)
        except Exception as e:
            last_err = e
            # PhotoKitAuthError is the common failure on unsigned Python —
            # silently advance to the AppleScript path.
            if "PhotoKitAuth" in type(e).__name__:
                continue
            # Other errors: still try the fallback, but remember the cause.
            continue

        exported = list(getattr(result, "exported", []) or [])
        if exported:
            return Path(exported[0]), "downloaded"
        errors = list(getattr(result, "error", []) or [])
        if errors:
            msg = errors[0][1] if isinstance(errors[0], tuple) else str(errors[0])
            last_err = RuntimeError(msg)
            continue
        # No errors and no exports: usually means PhotoKit-side missing.
        # Try the next transport before giving up.

    if last_err:
        return None, f"failed:{type(last_err).__name__}: {last_err}"
    return None, "failed:no transport returned a file"


def mirror_sidecar_path(asset: PhotosAsset, output_root: Path) -> Path:
    """Return the sidecar path under the external mirror tree.

    Layout: {output_root}/{YYYY-MM}/{filename}.description.md, falling back
    to _undated/ when Photos has no date. The filename keeps its original
    extension so .MOV and .description.md sit visibly next to each other in
    a file browser.

    UUID is appended to disambiguate when two assets share a filename
    (common with iPhone-rolling counters like IMG_0001.MOV).
    """
    if asset.date:
        bucket = f"{asset.date.year:04d}-{asset.date.month:02d}"
    else:
        bucket = "_undated"
    stem = Path(asset.filename).stem or asset.uuid
    suffix = Path(asset.filename).suffix
    name = f"{stem}__{asset.uuid[:8]}{suffix}{SIDECAR_SUFFIX}"
    return output_root / bucket / name


def to_metadata_override(asset: PhotosAsset) -> dict[str, Any]:
    """Photos-authoritative metadata to merge over ffprobe's output.

    Currently just the creation_time — Photos's date is the user's ground
    truth (it survives camera-roll edits and timezone normalization) while
    ffprobe's container-level creation_time can be missing or wrong.
    """
    out: dict[str, Any] = {}
    if asset.date:
        out["creation_time"] = asset.date.isoformat()
    return out


def to_gps_override(asset: PhotosAsset) -> dict[str, Any] | None:
    """Photos-side GPS. Returns None if Photos has no location → caller falls
    back to exiftool on the file itself."""
    if asset.lat is None or asset.lon is None:
        return None
    gps: dict[str, Any] = {"lat": asset.lat, "lon": asset.lon}
    if asset.altitude_m is not None:
        gps["altitude_m"] = asset.altitude_m
    return gps


def to_extra_frontmatter(asset: PhotosAsset) -> dict[str, Any]:
    """Photos-specific frontmatter fields to slot into the sidecar.

    Also overrides `file` so the sidecar shows the user-meaningful original
    filename (e.g. "gh010136.MP4") instead of the library's UUID-based
    on-disk name ("B627DB90-...mp4"). The full Photos UUID stays in
    `photos_uuid` for unambiguous lookup.
    """
    extra: dict[str, Any] = {
        "file": asset.filename,
        "original_filename": asset.filename,
        "photos_uuid": asset.uuid,
        "photos_persons": asset.persons,
        "photos_albums": asset.albums,
        "photos_keywords": asset.keywords,
    }
    if asset.is_edited:
        extra["photos_edited"] = True
    return extra


def parent_folder_for(asset: PhotosAsset) -> str:
    """A meaningful 'parent_folder' for the sidecar — first album, or a
    sentinel. Photos has no folder hierarchy so we surface album membership
    here so existing fdx-summary/fdx-query group sensibly."""
    return asset.albums[0] if asset.albums else "(no album)"
