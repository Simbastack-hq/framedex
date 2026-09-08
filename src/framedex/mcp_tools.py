#!/usr/bin/env python3
"""
framedex.mcp_tools — the archive as tools, without the MCP SDK.

Everything `fdx-mcp` exposes lives here as plain functions over a `Roots`
value (the directories the server may see), so it is testable without `mcp`
or Pillow installed; `mcp_server` wraps each function in a thin tool. Caller
mistakes raise ValueError (the server maps them to ToolError); missing
pieces (ffmpeg, an extra) raise RuntimeError with the fix in the message.

Cost model: zero model calls in here. A contact sheet is one bounded image
per successful call (at most SHEET_MAX_IMAGES thumbnails, ~1600 px wide).
The only write is `set_user_rating`: three frontmatter keys in a sidecar,
written atomically with the body preserved byte-for-byte. Originals are only
ever read, to render thumbnails.
"""

from __future__ import annotations

import io
import math
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from framedex import images
from framedex.index_videos import VIDEO_EXTENSIONS
from framedex.parsing import (
    RATING_VALUES,
    effective_rating,
    is_group_stub,
    is_user_rated,
    scene_sentence,
)
from framedex.pipeline import (
    FRAME_MAX_WIDTH,
    SIDECAR_SUFFIX,
    USER_KEYS,
    atomic_write_bytes,
    sidecar_path,
    split_frontmatter,
)
from framedex.query import Filters, run_query

# A contact sheet is one bounded image: at most this many thumbnails, in a
# fixed 4-column grid of 384 px cells (about 1600 px wide).
SHEET_MAX_IMAGES = 20
SHEET_COLUMNS = 4
SHEET_THUMB_PX = 384
SHEET_PAD_PX = 8
SHEET_JPEG_QUALITY = 85
# _INDEX.md is returned whole up to this size, then cut with a note.
OVERVIEW_MAX_BYTES = 65536
QUERY_DEFAULT_LIMIT = 50
QUERY_MAX_LIMIT = 500
# ffprobe / ffmpeg are bounded: a hung decoder must not hang the server.
SUBPROCESS_TIMEOUT_SEC = 60


# ---------------------------------------------------------------------------
# Roots: the only directories the server may read or write under
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Roots:
    roots: tuple[Path, ...]

    @classmethod
    def from_args(cls, paths: list[str]) -> Roots:
        resolved: list[Path] = []
        for p in paths:
            r = Path(p).expanduser().resolve()
            if not r.is_dir():
                raise ValueError(f"not a directory: {p}")
            if r not in resolved:
                resolved.append(r)
        if not resolved:
            raise ValueError("at least one root directory is required")
        return cls(tuple(resolved))

    def resolve(self, candidate: str | Path, *, must_exist: bool = True) -> Path:
        """The canonical path (symlinks and `..` resolved) if it lies under a
        root, else ValueError. Every read and write goes through here, on the
        actual target, not just on tool arguments."""
        p = Path(candidate).expanduser()
        if not p.is_absolute():
            if len(self.roots) != 1:
                raise ValueError(
                    f"{candidate}: relative paths are ambiguous with several roots; "
                    "pass an absolute path"
                )
            p = self.roots[0] / p
        resolved = p.resolve()
        if not any(resolved == r or r in resolved.parents for r in self.roots):
            raise ValueError(
                f"{candidate} is outside the configured roots: "
                + ", ".join(str(r) for r in self.roots)
            )
        if must_exist and not resolved.exists():
            raise ValueError(f"not found: {candidate}")
        return resolved

    def root_of(self, path: Path) -> Path:
        for r in self.roots:
            if path == r or r in path.parents:
                return r
        raise ValueError(f"{path} is outside the configured roots")

    def single(self, root: str | None) -> Path:
        """The root a tool call refers to: the given one (which must be
        configured), or the only one."""
        if root is None:
            if len(self.roots) == 1:
                return self.roots[0]
            raise ValueError(
                "several roots are configured; say which root: "
                + ", ".join(str(r) for r in self.roots)
            )
        r = Path(root).expanduser().resolve()
        if r not in self.roots:
            raise ValueError(
                f"{root} is not one of the configured roots: "
                + ", ".join(str(x) for x in self.roots)
            )
        return r


def sidecar_for(roots: Roots, ref: str) -> tuple[Path, Path | None]:
    """(sidecar, media) for a media path or a sidecar path under the roots.
    The sidecar must exist; the media may not (Photos-managed assets)."""
    if ref.endswith(SIDECAR_SUFFIX):
        sidecar = roots.resolve(ref)
        media = sidecar.with_name(sidecar.name[: -len(SIDECAR_SUFFIX)])
        return sidecar, (media if media.exists() else None)
    media = roots.resolve(ref)
    sidecar = sidecar_path(media)
    if not sidecar.exists():
        raise ValueError(
            f"not indexed: {ref} has no {SIDECAR_SUFFIX} sidecar. "
            f"Run: fdx {roots.root_of(media)}"
        )
    return sidecar, media


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


def query_media(
    roots: Roots,
    *,
    root: str | None = None,
    folder: str | None = None,
    rating: str | None = None,
    media: str | None = None,
    keywords: list[str] | None = None,
    place_contains: str | None = None,
    person: str | None = None,
    time_of_day: str | None = None,
    lighting: str | None = None,
    has_speech: bool = False,
    primary_only: bool = False,
    offset: int = 0,
    limit: int = QUERY_DEFAULT_LIMIT,
) -> dict[str, Any]:
    """The fdx-query filters as a tool. Returns a page of compact records
    (path, sidecar, ratings, place, keywords, scene) plus the total."""
    base = roots.single(root)
    if rating is not None:
        wanted = [v.strip() for v in rating.split(",") if v.strip()]
        if not wanted or any(v not in RATING_VALUES for v in wanted):
            raise ValueError(
                f"rating must be a comma list of {'/'.join(RATING_VALUES)}, got {rating!r}"
            )
    if media is not None and media not in ("image", "video"):
        raise ValueError(f"media must be 'image' or 'video', got {media!r}")
    if not 1 <= limit <= QUERY_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {QUERY_MAX_LIMIT}, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    filters = Filters(
        rating=rating,
        media=media,
        keyword=list(keywords or []),
        place_contains=place_contains,
        person=person,
        time_of_day=time_of_day,
        lighting=lighting,
        has_speech=has_speech,
        primary_only=primary_only,
        folder=folder,
        offset=offset,
        limit=limit,
    )
    result = run_query(base, filters)
    matches: list[dict[str, Any]] = []
    dropped = 0
    for rec in result.records:
        try:
            sidecar = roots.resolve(rec["_sidecar_path"])
            media_path = (
                roots.resolve(rec["path"], must_exist=False)
                if rec.get("path")
                else None
            )
        except ValueError:
            dropped += 1  # a symlinked sidecar or a `path` pointing outside the root
            continue
        matches.append(
            {
                "path": str(media_path) if media_path else None,
                "sidecar": str(sidecar),
                "media_type": rec.get("media_type") or "video",
                "rating": rec.get("rating"),
                "user_rating": rec.get("user_rating") if is_user_rated(rec) else None,
                "effective_rating": effective_rating(rec),
                "place": (rec.get("location") or {}).get("place"),
                "keywords": rec.get("keywords") or [],
                "scene": rec.get("_scene") or "",
                "creation_time": rec.get("creation_time"),
                "group_alternate": is_group_stub(rec),
            }
        )
    return {
        "matches": matches,
        "total": result.total,
        "offset": offset,
        "limit": limit,
        "truncated": offset + len(result.records) < result.total,
        "skipped_malformed": result.skipped_malformed + dropped,
        "invalid_user_ratings": result.invalid_user_ratings,
    }


def read_sidecar(roots: Roots, ref: str) -> str:
    """The full `.description.md` text for a media file (or a sidecar path)."""
    sidecar, _ = sidecar_for(roots, ref)
    return sidecar.read_bytes().decode("utf-8", errors="replace")


def archive_overview(roots: Roots, root: str | None = None) -> str:
    """`_INDEX.md` for a root, marked as the snapshot it is."""
    base = roots.single(root)
    idx = base / "_INDEX.md"
    if not idx.exists():
        return f"No _INDEX.md under {base} yet. Run: fdx-master {base}"
    idx = roots.resolve(idx)
    data = idx.read_bytes()
    text = data[:OVERVIEW_MAX_BYTES].decode("utf-8", errors="replace")
    header = (
        f"[Snapshot: {idx}. It does not update itself; regenerate after new "
        f"indexing or ratings with: fdx-master {base}]\n\n"
    )
    tail = "\n\n[truncated]" if len(data) > OVERVIEW_MAX_BYTES else ""
    return header + text + tail


# ---------------------------------------------------------------------------
# The one write: the photographer's rating
# ---------------------------------------------------------------------------

_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """Serialise read-modify-write per sidecar: the SDK runs sync tools on
    worker threads. (Another process, e.g. `fdx --force` on the same folder,
    is not covered; the docs say not to.)"""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


def set_user_rating(
    roots: Roots, ref: str, rating: str, note: str = ""
) -> dict[str, Any]:
    """Record the human's decision in the sidecar: `user_rating`, `user_note`
    (only when given), `user_rated_at`. `rating=""` removes all three. The
    model's `rating` is never touched; the body is preserved byte-for-byte.
    An unchanged rating+note is a no-op (the timestamp stays)."""
    if rating not in (*RATING_VALUES, ""):
        raise ValueError(
            f"rating must be one of {', '.join(RATING_VALUES)}, or '' to clear; got {rating!r}"
        )
    note = note or ""
    sidecar, media = sidecar_for(roots, ref)
    with _lock_for(sidecar):
        raw = sidecar.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{sidecar.name} is not UTF-8 text") from e
        parts = split_frontmatter(text)
        if parts is None:
            raise ValueError(
                f"{sidecar.name}: no frontmatter fence; not a framedex sidecar"
            )
        try:
            fm = yaml.safe_load(parts[0])
        except yaml.YAMLError as e:
            raise ValueError(
                f"{sidecar.name}: frontmatter is not valid YAML ({e})"
            ) from e
        if not isinstance(fm, dict):
            raise ValueError(f"{sidecar.name}: frontmatter is not a mapping")

        current_rating = fm.get("user_rating") if is_user_rated(fm) else None
        current_note = fm.get("user_note") or ""
        if rating == "":
            changed = any(k in fm for k in USER_KEYS)
            for k in USER_KEYS:
                fm.pop(k, None)
        else:
            changed = not (current_rating == rating and current_note == note)
            if changed:
                for k in USER_KEYS:
                    fm.pop(k, None)
                indexed_at = fm.pop("indexed_at", None)
                fm["user_rating"] = rating
                if note:
                    fm["user_note"] = note
                fm["user_rated_at"] = datetime.now().isoformat(timespec="seconds")
                if indexed_at is not None:
                    fm["indexed_at"] = indexed_at
        if changed:
            fm_text = yaml.safe_dump(
                fm, sort_keys=False, allow_unicode=True, default_flow_style=False
            ).rstrip()
            atomic_write_bytes(
                sidecar,
                b"---\n"
                + fm_text.encode("utf-8")
                + b"\n---\n"
                + parts[1].encode("utf-8"),
            )
    return {
        "sidecar": str(sidecar),
        "path": str(media) if media else None,
        "rating": fm.get("rating"),
        "user_rating": fm.get("user_rating"),
        "user_note": fm.get("user_note"),
        "user_rated_at": fm.get("user_rated_at"),
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# Contact sheet: one bounded image the host's model can look at
# ---------------------------------------------------------------------------


def sheet_layout(n: int) -> tuple[tuple[int, int], list[tuple[int, int, int]]]:
    """Canvas size and (index, x, y) per cell for n thumbnails in the fixed
    grid. Pure geometry, so the bounds are testable without Pillow."""
    if not 1 <= n <= SHEET_MAX_IMAGES:
        raise ValueError(
            f"a contact sheet holds between 1 and {SHEET_MAX_IMAGES} images, got {n}; "
            "narrow the query"
        )
    cols = min(SHEET_COLUMNS, n)
    rows = math.ceil(n / SHEET_COLUMNS)
    step = SHEET_THUMB_PX + SHEET_PAD_PX
    width = cols * step + SHEET_PAD_PX
    height = rows * step + SHEET_PAD_PX
    cells = [
        (
            i + 1,
            SHEET_PAD_PX + (i % SHEET_COLUMNS) * step,
            SHEET_PAD_PX + (i // SHEET_COLUMNS) * step,
        )
        for i in range(n)
    ]
    return (width, height), cells


def parse_timestamp(text: str | None) -> float | None:
    """`MM:SS` or `H:MM:SS` (the sidecar's notable_timestamp) → seconds."""
    if not text:
        return None
    parts = text.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + int(p)
    return seconds


def _run(cmd: list[str], what: str, name: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_SEC
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{cmd[0]} not found on PATH; install ffmpeg to render video thumbnails"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"{what} timed out after {SUBPROCESS_TIMEOUT_SEC}s on {name}"
        ) from e


def video_frame(path: Path, out_dir: Path, *, timestamp: float | None) -> Path:
    """One JPEG frame from a clip: at `timestamp` when it is finite and inside
    the clip, else the midpoint. Loud on every failure."""
    probe = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        "ffprobe",
        path.name,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = math.nan
    if probe.returncode != 0 or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            f"could not read the duration of {path.name}: {probe.stderr.strip()[:200]}"
        )
    if timestamp is not None and math.isfinite(timestamp) and 0 <= timestamp < duration:
        t = timestamp
    else:
        t = duration / 2
    out = out_dir / "frame.jpg"
    result = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(t),
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({FRAME_MAX_WIDTH},iw)':-2",
            "-q:v",
            "3",
            str(out),
        ],
        "ffmpeg",
        path.name,
    )
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"ffmpeg failed on {path.name}: {result.stderr.strip()[:200]}"
        )
    return out


def render_thumb(media: Path, out_dir: Path, notable: str | None) -> Path | None:
    """A preview JPEG for a still (RAW via its embedded preview, HEIC via
    pillow-heif) or one frame of a clip. None only for a RAW without an
    embedded preview; other failures raise."""
    if media.suffix.lower() in VIDEO_EXTENSIONS:
        return video_frame(media, out_dir, timestamp=parse_timestamp(notable))
    return images.render_preview(media, out_dir)


def _sidecar_summary(media: Path) -> tuple[dict[str, Any] | None, str]:
    """(frontmatter, scene) for an indexed file; (None, "") when not indexed."""
    sidecar = sidecar_path(media)
    if not sidecar.exists():
        return None, ""
    parts = split_frontmatter(sidecar.read_bytes().decode("utf-8", errors="replace"))
    if parts is None:
        return None, ""
    try:
        fm = yaml.safe_load(parts[0])
    except yaml.YAMLError:
        return None, ""
    return (fm if isinstance(fm, dict) else None), scene_sentence(parts[1])


def build_contact_sheet(roots: Roots, paths: list[str]) -> tuple[bytes, list[str]]:
    """Render up to SHEET_MAX_IMAGES files into one numbered JPEG grid. Returns
    (jpeg bytes, legend lines). A file that cannot be rendered keeps its
    numbered cell (grey, with the reason) and is listed as such; if nothing
    renders the call fails."""
    (width, height), cells = sheet_layout(len(paths))
    medias = [roots.resolve(p) for p in paths]  # every member validated first
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise RuntimeError(
            "contact sheets need Pillow: uv pip install -e '.[mcp]'"
        ) from e

    canvas = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(canvas)
    legend: list[str] = []
    rendered = 0
    tmp = Path(tempfile.mkdtemp(prefix="fdx-sheet-"))
    try:
        for (idx, x, y), media in zip(cells, medias, strict=True):
            fm, scene = _sidecar_summary(media)
            label = str(effective_rating(fm) or "unrated") if fm else "not indexed"
            if fm and is_user_rated(fm):
                label += " (user)"
            if scene:
                label += f" — {scene}"
            sub = tmp / str(idx)
            sub.mkdir()
            reason: str | None = None
            thumb: Path | None = None
            try:
                thumb = render_thumb(media, sub, (fm or {}).get("notable_timestamp"))
                if thumb is None:
                    reason = "RAW without an embedded preview"
            except (RuntimeError, OSError) as e:
                reason = str(e)
            if thumb is not None:
                with Image.open(thumb) as im:
                    im2 = im.convert("RGB")
                    im2.thumbnail((SHEET_THUMB_PX, SHEET_THUMB_PX))
                    canvas.paste(
                        im2,
                        (
                            x + (SHEET_THUMB_PX - im2.width) // 2,
                            y + (SHEET_THUMB_PX - im2.height) // 2,
                        ),
                    )
                rendered += 1
            else:
                draw.rectangle(
                    [x, y, x + SHEET_THUMB_PX, y + SHEET_THUMB_PX], fill=(200, 200, 200)
                )
                draw.text(
                    (x + 8, y + SHEET_THUMB_PX // 2),
                    f"no preview: {(reason or '')[:40]}",
                    fill=(60, 60, 60),
                )
            draw.rectangle([x, y, x + 30, y + 20], fill=(0, 0, 0))
            draw.text((x + 6, y + 4), str(idx), fill=(255, 255, 255))
            rel = media.relative_to(roots.root_of(media))
            line = f"{idx}. {rel} — {label}"
            if reason:
                line += f" — no preview: {reason}"
            legend.append(line)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if rendered == 0:
        raise ValueError("none of the files could be rendered: " + "; ".join(legend))
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=SHEET_JPEG_QUALITY)
    return buf.getvalue(), legend
