"""
framedex.share_site — the pixels and the page for fdx-share.

Renders each selected original into a web copy (WEB_LONG_EDGE px) and a grid
thumbnail (THUMB_LONG_EDGE px), upright, sRGB-or-original ICC kept, with a
minimal EXIF block (make, model, capture time) and no GPS unless asked. The
originals are opened read-only; nothing is ever written next to them except
inside the gallery's own work folder (`_share/<name>/` by default).

The page is one static index.html (inline CSS + a small inline script for the
lightbox), no framework, no external requests: an unlisted gallery should not
tell a font CDN who is looking at it.
"""

from __future__ import annotations

import hashlib
import html
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from framedex.images import HEIF_EXTENSIONS, RAW_EXTENSIONS, _extract_raw_preview
from framedex.share import COVER_MIN_RATIO

if TYPE_CHECKING:
    from PIL import Image as PILImage

    from framedex.share import Gallery, Item, Photo, Signature

# Long edge of the full-size web copy: sharp on a 4K screen's lightbox,
# ~0.4-0.8 MB as JPEG.
WEB_LONG_EDGE = 2048
WEB_JPEG_QUALITY = 82
# Long edge of the grid image: a full-width phone column at 2-3x, or a grid
# tile on a laptop.
THUMB_LONG_EDGE = 1024
THUMB_JPEG_QUALITY = 78
# Bump to force every web copy to be re-rendered (encoder settings changed).
RENDER_VERSION = 1
FRAMEDEX_URL = "https://github.com/Simbastack-hq/framedex"
# Two consecutive photos at most this wide (width / height) hang side by side.
PAIR_MAX_RATIO = 0.9
SITE_DIR = "site"
_RENDER_NAME_RE = re.compile(r"^[0-9a-f]{16}\.jpg$")

# EXIF tags carried into web copies (everything else, GPS included, is dropped).
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_OFFSET_TIME_ORIGINAL = 0x9011
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825


def _pil() -> Any:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "fdx-share needs Pillow: install the images extra "
            "(uv pip install -e '.[images]')"
        ) from e
    return Image


# ---------------------------------------------------------------------------
# Reading originals
# ---------------------------------------------------------------------------


def render_source(original: Path) -> Path:
    """A RAW with a same-stem camera JPEG next to it renders from the JPEG
    (camera colour, no extraction); everything else from itself."""
    if original.suffix.lower() not in RAW_EXTENSIONS:
        return original
    for sib in sorted(original.parent.glob(original.stem + ".*")):
        if sib.suffix.lower() in (".jpg", ".jpeg") and sib.stem == original.stem:
            return sib
    return original


@contextmanager
def open_upright(original: Path) -> Iterator[PILImage.Image]:
    """Open an original read-only as an upright image (EXIF orientation
    applied). RAW without a JPEG sibling reads its embedded preview; HEIC
    needs pillow-heif. The caller must not keep the image past the block."""
    image_mod = _pil()
    from PIL import ImageOps

    src = render_source(original)
    tmp: Path | None = None
    try:
        if src.suffix.lower() in RAW_EXTENSIONS:
            tmp = Path(tempfile.mkdtemp(prefix="fdx-share-"))
            preview = _extract_raw_preview(src, tmp)
            if preview is None:
                raise ValueError(f"{original.name}: RAW without an embedded preview")
            src = preview
        if src.suffix.lower() in HEIF_EXTENSIONS:
            import pillow_heif

            pillow_heif.register_heif_opener()
        with image_mod.open(src) as opened:
            opened.load()
            upright = ImageOps.exif_transpose(opened) or opened
            upright.info.setdefault("icc_profile", opened.info.get("icc_profile"))
            upright.info["fdx_exif"] = opened.getexif()
            yield upright
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def signatures(photos: list[Photo]) -> dict[str, Signature]:
    """Thumbnail signatures for near-duplicate detection, keyed by rel. JPEGs
    decode at reduced size (draft mode), so a few hundred candidates take
    seconds. A photo that cannot be read gets none (and so is never called a
    duplicate)."""
    from PIL import ImageOps

    from framedex.share import SIGNATURE_SIZE, Signature

    image_mod = _pil()
    size = (SIGNATURE_SIZE, SIGNATURE_SIZE)
    out: dict[str, Signature] = {}
    for p in photos:
        try:
            src = render_source(p.path)
            if src.suffix.lower() in (".jpg", ".jpeg"):
                with image_mod.open(src) as im:
                    im.draft("RGB", (128, 128))
                    upright = ImageOps.exif_transpose(im) or im
                    w, h = upright.size
                    gray = upright.convert("L").resize(size, image_mod.Resampling.BOX)
            else:
                with open_upright(p.path) as im:
                    w, h = im.size
                    gray = im.convert("L").resize(size, image_mod.Resampling.BOX)
            out[p.rel] = Signature(w / h, tuple(gray.tobytes()))
        except Exception as e:  # unreadable file: no signature, never a duplicate
            print(f"  note: no near-duplicate check for {p.rel}: {e}")
    return out


# ---------------------------------------------------------------------------
# Writing web copies
# ---------------------------------------------------------------------------


@dataclass
class Rendered:
    rel: str
    name: str  # "<16 hex>.jpg", the same in photos/ and thumbs/
    width: int
    height: int
    thumb_width: int
    thumb_height: int
    color: str  # average colour, the placeholder while the photo loads
    bytes: int

    @property
    def ratio(self) -> float:
        return self.width / self.height if self.height else 1.0


@dataclass
class Site:
    root: Path  # the publishable folder (index.html, photos/, thumbs/)
    gallery: Gallery
    photos: dict[str, Rendered] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    def files(self) -> list[Path]:
        return sorted(f for f in self.root.rglob("*") if f.is_file())


def render_name(rel: str, original: Path, keep_gps: bool) -> str:
    """Content-addressed output name: changes when the original, the GPS
    choice, or the encoder settings change; reveals nothing about the file."""
    st = original.stat()
    key = f"{RENDER_VERSION}|{rel}|{st.st_mtime_ns}|{st.st_size}|{int(keep_gps)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".jpg"


def web_exif(source_exif: Any, *, keep_gps: bool) -> Any:
    """The EXIF block a web copy carries: make, model, capture time (+ its
    offset), and the GPS block only when `keep_gps`. Orientation is gone (the
    pixels are upright); serial numbers, maker notes, and the rest never
    leave the original."""
    from PIL import Image

    out = Image.Exif()
    for tag in (_TAG_MAKE, _TAG_MODEL):
        if source_exif.get(tag):
            out[tag] = source_exif[tag]
    exif_ifd = source_exif.get_ifd(_IFD_EXIF)
    kept = {
        t: exif_ifd[t]
        for t in (_TAG_DATETIME_ORIGINAL, _TAG_OFFSET_TIME_ORIGINAL)
        if exif_ifd.get(t)
    }
    if kept:
        out[_IFD_EXIF] = kept
    if keep_gps:
        gps = source_exif.get_ifd(_IFD_GPS)
        if gps:
            out[_IFD_GPS] = dict(gps)
    return out


def _fit(w: int, h: int, long_edge: int) -> tuple[int, int]:
    scale = min(1.0, long_edge / max(w, h))
    return max(1, round(w * scale)), max(1, round(h * scale))


def render_one(
    original: Path, out_photo: Path, out_thumb: Path, *, keep_gps: bool
) -> None:
    image_mod = _pil()
    with open_upright(original) as im:
        icc = im.info.get("icc_profile")
        exif = web_exif(im.info["fdx_exif"], keep_gps=keep_gps).tobytes()
        rgb = im if im.mode == "RGB" else im.convert("RGB")
        web = rgb.resize(_fit(*rgb.size, WEB_LONG_EDGE), image_mod.Resampling.LANCZOS)
    opts: dict[str, Any] = {"optimize": True, "progressive": True, "exif": exif}
    if icc:
        opts["icc_profile"] = icc
    web.save(out_photo, "JPEG", quality=WEB_JPEG_QUALITY, **opts)
    thumb = web.resize(_fit(*web.size, THUMB_LONG_EDGE), image_mod.Resampling.LANCZOS)
    thumb.save(out_thumb, "JPEG", quality=THUMB_JPEG_QUALITY, **opts)


def _describe(out_photo: Path, out_thumb: Path, rel: str) -> Rendered:
    image_mod = _pil()
    with image_mod.open(out_photo) as a:
        w, h = a.size
    with image_mod.open(out_thumb) as b:
        tw, th = b.size
        r, g, bl = (
            b.convert("RGB").resize((1, 1), image_mod.Resampling.BOX).getpixel((0, 0))
        )
    return Rendered(
        rel,
        out_photo.name,
        w,
        h,
        tw,
        th,
        f"#{r:02x}{g:02x}{bl:02x}",
        out_photo.stat().st_size + out_thumb.stat().st_size,
    )


def render_site(
    gallery: Gallery,
    originals: dict[str, Path],
    work: Path,
    *,
    keep_gps: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Site:
    """Write work/site/{index.html, photos/, thumbs/}: web copies of every
    photo in the gallery, hidden ones included (the editor can restore them
    without a re-render). Copies are re-used while the original is
    unchanged; copies of photos no longer in the gallery are deleted.
    Publishing never uploads this folder as is: it uploads the artifact
    `share_publish` builds from the shown photos only."""
    root = work / SITE_DIR
    photos_dir, thumbs_dir = root / "photos", root / "thumbs"
    photos_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    site = Site(root, gallery)
    todo = [i for i in gallery.all_items if i.rel in originals]
    for n, item in enumerate(todo, 1):
        original = originals[item.rel]
        name = render_name(item.rel, original, keep_gps)
        out_photo, out_thumb = photos_dir / name, thumbs_dir / name
        if progress is not None:
            progress(n, len(todo))
        if not (out_photo.exists() and out_thumb.exists()):
            tmp_photo = out_photo.with_name("." + name)
            tmp_thumb = out_thumb.with_name("." + name)
            render_one(original, tmp_photo, tmp_thumb, keep_gps=keep_gps)
            tmp_photo.replace(out_photo)
            tmp_thumb.replace(out_thumb)
        site.photos[item.rel] = _describe(out_photo, out_thumb, item.rel)
    live = {r.name for r in site.photos.values()}
    for d in (photos_dir, thumbs_dir):
        for f in d.iterdir():
            stale = _RENDER_NAME_RE.match(f.name) and f.name not in live
            if stale or f.name.startswith("."):
                f.unlink()
    (root / "index.html").write_text(page_html(gallery, site.photos), encoding="utf-8")
    return site


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def asset(name: str) -> str:
    """A file from framedex/share_assets (the page's CSS and scripts)."""
    return (resources.files("framedex") / "share_assets" / name).read_text(
        encoding="utf-8"
    )


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def split_heading(heading: str) -> tuple[str, str]:
    """'Sunday 9 November · Ewangan, Narok West' -> (when, where)."""
    when, _, where = heading.partition(" · ")
    return when.strip(), where.strip()


def hang(items: list[Item], ratios: dict[str, float]) -> list[list[Item]]:
    """Arrange photos in order: two portraits in a row hang side by side as a
    pair (one tall phone frame alone in that much space looks lost);
    everything else hangs on its own."""
    walls: list[list[Item]] = []
    i = 0
    while i < len(items):
        a = items[i]
        b = items[i + 1] if i + 1 < len(items) else None
        if (
            b is not None
            and ratios.get(a.rel, 1.0) <= PAIR_MAX_RATIO
            and ratios.get(b.rel, 1.0) <= PAIR_MAX_RATIO
        ):
            walls.append([a, b])
            i += 2
        else:
            walls.append([a])
            i += 1
    return walls


def pick_cover(
    gallery: Gallery, items: list[Item], photos: dict[str, Rendered]
) -> Item | None:
    """The opening photo: the `cover:` line when it names a shown photo,
    nothing for 'none', else the first clearly wide photo. Never cropped, so
    a portrait can open too when named."""
    if gallery.cover.lower() == "none":
        return None
    named = next((i for i in items if i.rel == gallery.cover), None)
    if named is not None:
        return named
    return next((i for i in items if photos[i.rel].ratio >= COVER_MIN_RATIO), None)


@dataclass
class Sealed:
    """Where a locked gallery's encrypted copies live: rel -> (photo path,
    thumb path), both gallery-relative ('photos/<hash>.bin')."""

    paths: dict[str, tuple[str, str]]


def gallery_markup(
    gallery: Gallery,
    photos: dict[str, Rendered],
    *,
    asset_prefix: str = "",
    sealed: Sealed | None = None,
    edit: bool = False,
) -> str:
    """Title, the opening photo, then each section: the whole visible page.
    With `sealed`, images carry their encrypted paths instead of src (the
    lock script decrypts them); with `edit`, hidden photos stay in place,
    dimmed, and every photo carries its rel for the editor."""
    shown = [i for i in gallery.items if i.rel in photos]
    cover = pick_cover(gallery, shown, photos)

    def figure(item: Item, cls: str) -> str:
        r = photos[item.rel]
        alt = _e(item.caption or "Photo")
        attrs = f' data-rel="{_e(item.rel)}"' if edit else ""
        if sealed is not None:
            photo, thumb = sealed.paths[item.rel]
            img = (
                f'<img data-photo="{_e(photo)}" data-thumb="{_e(thumb)}" '
                f'data-tw="{r.thumb_width}" width="{r.width}" height="{r.height}" alt="{alt}">'
            )
            href = "#"
        else:
            full = _e(f"{asset_prefix}photos/{r.name}")
            small = _e(f"{asset_prefix}thumbs/{r.name}")
            sizes = (
                "(max-width:700px) 100vw, 47vw"
                if "pair" in cls
                else "(max-width:700px) 100vw, min(1440px, 94vw)"
            )
            lazy = "" if "first" in cls else ' loading="lazy"'
            img = (
                f'<img src="{full}" srcset="{small} {r.thumb_width}w, {full} {r.width}w" '
                f'sizes="{sizes}" width="{r.width}" height="{r.height}" alt="{alt}"'
                f'{lazy} decoding="async">'
            )
            href = full
        cap = (
            f"<figcaption>{_e(item.caption)}</figcaption>"
            if gallery.captions and item.caption
            else ""
        )
        return (
            f'<figure class="ph {cls}" style="--ar:{r.ratio:.4f};--c:{r.color}"{attrs}>'
            f'<a href="{href}">{img}</a>{cap}</figure>'
        )

    out = [
        '<header class="top">',
        f"<h1{' contenteditable' if edit else ''}>{_e(gallery.title)}</h1>",
        f"<p>{_e(gallery.subtitle)}</p>" if gallery.subtitle else "",
        "</header>",
    ]
    if cover is not None:
        out.append(f'<div class="hero">{figure(cover, "solo first")}</div>')
    out.append("<main>")
    for s in gallery.sections:
        entries = [i for i in s.items if i.rel in photos and (edit or not i.hidden)]
        if not edit:
            entries = [i for i in entries if i is not cover]
        if not entries:
            continue
        when, where = split_heading(s.heading)
        where_html = f"<span>{' · ' if when else ''}{_e(where)}</span>" if where else ""
        heading = (
            f'<div class="day"><h2>{_e(when)}{where_html}</h2></div>'
            if s.heading
            else ""
        )
        out.append(f"<section>{heading}")
        run: list[Item] = []

        def flush(run: list[Item]) -> None:
            for wall in hang(run, {i.rel: photos[i.rel].ratio for i in run}):
                if len(wall) == 1:
                    out.append(figure(wall[0], "solo"))
                else:
                    total = sum(photos[i.rel].ratio for i in wall)
                    out.append(
                        f'<div class="pair" style="--sum:{total:.4f}">'
                        + "".join(figure(i, "in-pair") for i in wall)
                        + "</div>"
                    )

        for item in entries:
            if item is cover:
                # The editor keeps the cover's place in its section (it moves
                # back there if another photo becomes the cover).
                flush(run)
                run = []
                out.append(
                    f'<span class="slot" data-rel="{_e(item.rel)}" hidden></span>'
                )
            elif item.hidden:
                flush(run)
                run = []
                out.append(figure(item, "solo off"))
            else:
                run.append(item)
        flush(run)
        out.append("</section>")
    out += [
        "</main>",
        f'<footer class="foot">Made with <a href="{FRAMEDEX_URL}">framedex</a></footer>',
    ]
    return "\n".join(o for o in out if o)


_HEAD = (
    '<!doctype html><html lang="en" class="no-js"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    '<meta name="robots" content="noindex,nofollow,noimageindex,noarchive">'
    '<meta name="referrer" content="no-referrer">'
)


def page_html(
    gallery: Gallery,
    photos: dict[str, Rendered],
    *,
    asset_prefix: str = "",
    page_url: str | None = None,
) -> str:
    """The published page, open to anyone with the link. `asset_prefix` is
    prepended to photos/ and thumbs/ (an S3 page object at '<base>/<slug>'
    needs '<slug>/'); `page_url` (absolute, publish only) adds link-preview
    tags so a WhatsApp or iMessage link shows the cover. Every string from
    the selection file is escaped."""
    head = [_HEAD, f"<title>{_e(gallery.title)}</title>"]
    if gallery.subtitle:
        head.append(f'<meta name="description" content="{_e(gallery.subtitle)}">')
    shown = [i for i in gallery.items if i.rel in photos]
    cover = pick_cover(gallery, shown, photos) or (shown[0] if shown else None)
    if page_url and cover is not None:
        image = page_url.rstrip("/") + f"/thumbs/{photos[cover.rel].name}"
        head += [
            '<meta property="og:type" content="website">',
            f'<meta property="og:title" content="{_e(gallery.title)}">',
            f'<meta property="og:description" content="{_e(gallery.subtitle)}">',
            f'<meta property="og:image" content="{_e(image)}">',
            '<meta name="twitter:card" content="summary_large_image">',
        ]
    return "".join(
        [
            *head,
            f"<style>{asset('gallery.css')}</style></head><body>",
            gallery_markup(gallery, photos, asset_prefix=asset_prefix),
            f"<script>{asset('gallery.js')}</script></body></html>\n",
        ]
    )
