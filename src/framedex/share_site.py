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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
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
PREVIEW_FILE = "preview.html"
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
    taken: str = ""  # capture time "HH:MM" from the copy's EXIF, for the wall label

    @property
    def ratio(self) -> float:
        return self.width / self.height if self.height else 1.0


@dataclass
class Site:
    root: Path  # the publishable folder (index.html, photos/, thumbs/)
    preview: Path  # preview.html next to it (local only, never uploaded)
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
        stamp = str(a.getexif().get_ifd(_IFD_EXIF).get(_TAG_DATETIME_ORIGINAL) or "")
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
        clock_time(stamp),
    )


def clock_time(exif_datetime: str) -> str:
    """'2025:11:09 09:48:12' -> '09:48'; '' when it does not parse."""
    _, _, clock = exif_datetime.strip().partition(" ")
    hh, _, rest = clock.partition(":")
    mm = rest[:2]
    if len(hh) == 2 and hh.isdigit() and len(mm) == 2 and mm.isdigit():
        return f"{hh}:{mm}"
    return ""


def render_site(
    gallery: Gallery, originals: dict[str, Path], work: Path, *, keep_gps: bool = False
) -> Site:
    """Write work/site/{index.html, photos/, thumbs/} and work/preview.html.
    Web copies are re-used while the original is unchanged; renders no longer
    in the gallery are deleted, so a dropped photo never reaches a publish."""
    root = work / SITE_DIR
    photos_dir, thumbs_dir = root / "photos", root / "thumbs"
    photos_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    site = Site(root, work / PREVIEW_FILE, gallery)
    for n, item in enumerate(gallery.items, 1):
        original = originals[item.rel]
        name = render_name(item.rel, original, keep_gps)
        out_photo, out_thumb = photos_dir / name, thumbs_dir / name
        if not (out_photo.exists() and out_thumb.exists()):
            print(f"  [{n}/{len(gallery.items)}] {item.rel}")
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
    site.preview.write_text(
        page_html(gallery, site.photos, asset_prefix=f"{SITE_DIR}/", preview_of=work),
        encoding="utf-8",
    )
    return site


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

_CSS = """
:root{--wall:#f1ece3;--ink:#1c1915;--muted:#7a7062;--rule:#d6ccba;--accent:#9c5a22;
--lb:#0c0b09;--shadow:0 1px 2px rgba(40,28,14,.07),0 30px 60px -32px rgba(40,28,14,.5);
--serif:"Iowan Old Style","Palatino Linotype",Palatino,"URW Palladio L",P052,"Book Antiqua",Georgia,serif;
--mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
--space:clamp(96px,18vh,230px);--maxw:min(1320px,90vw);--maxh:84vh;--gap:clamp(32px,5vw,96px);
color-scheme:light dark}
@supports (height:1svh){:root{--maxh:84svh}}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--wall:#121110;--ink:#ebe3d4;
--muted:#9a8f7d;--rule:#2c2822;--accent:#d9965c;--shadow:0 34px 70px -34px rgba(0,0,0,.85)}}
:root[data-theme=dark]{--wall:#121110;--ink:#ebe3d4;--muted:#9a8f7d;--rule:#2c2822;--accent:#d9965c;
--shadow:0 34px 70px -34px rgba(0,0,0,.85)}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--wall);color:var(--ink);font:17px/1.55 var(--serif);
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.035'/%3E%3C/svg%3E")}
a{color:inherit}
.title-wall{min-height:64vh;display:grid;place-content:center;text-align:center;
padding:14vh 24px 10vh;animation:rise 1.2s cubic-bezier(.2,.7,.2,1) both}
.kicker,.meta{margin:0;font:12px/1.7 var(--mono);letter-spacing:.2em;text-transform:uppercase;color:var(--muted)}
.title-wall h1{margin:.35em 0 .4em;font-weight:400;font-size:clamp(3rem,10.5vw,9rem);
line-height:.9;letter-spacing:-.035em;text-wrap:balance}
.cover{margin:0;height:88vh;overflow:hidden;background:var(--c,#1a1712);
animation:fade 1.6s .2s ease both}
.cover img{width:100%;height:100%;object-fit:cover;display:block}
.room{padding-top:var(--space)}
.room-head{text-align:center;padding:0 24px;margin:0 auto var(--space);max-width:60rem}
.room-no{display:flex;align-items:center;justify-content:center;gap:18px;
font:12px var(--mono);letter-spacing:.32em;color:var(--muted)}
.room-no::before,.room-no::after{content:"";width:56px;height:1px;background:var(--rule)}
.room-head h2{margin:.55em 0 .15em;font-weight:400;font-size:clamp(2.1rem,5vw,4rem);
line-height:1.02;letter-spacing:-.02em}
.room-head p{margin:0;font-style:italic;color:var(--muted);font-size:clamp(1.05rem,1.7vw,1.35rem)}
.piece{position:relative;margin:0 auto var(--space);width:min-content;max-width:var(--maxw)}
.piece a{display:block;cursor:zoom-in}
.piece img{display:block;width:min(var(--maxw),calc(var(--maxh) * var(--ar)));height:auto;
aspect-ratio:var(--ar);background:var(--c);box-shadow:var(--shadow)}
.pair{display:flex;align-items:flex-start;justify-content:center;gap:var(--gap);
margin:0 auto var(--space);width:min(var(--maxw),calc(var(--maxh) * var(--sum) + var(--gap)))}
.pair .piece{margin:0;width:auto;max-width:none;min-width:0;flex:var(--ar) 1 0}
.pair .piece img{width:100%}
.label{display:grid;grid-template-columns:auto 1fr;column-gap:16px;margin-top:20px;
max-width:44ch;min-width:24ch;font-size:.97rem;line-height:1.45}
.label .no{grid-row:span 2;font:11.5px/1.9 var(--mono);color:var(--muted);letter-spacing:.08em;white-space:nowrap}
.label .cap{font-style:italic}
.label .t{font:11.5px/1.9 var(--mono);color:var(--muted);letter-spacing:.08em}
.js .piece{opacity:0;transform:translateY(28px);
transition:opacity 1.2s ease,transform 1.2s cubic-bezier(.2,.7,.2,1)}
.js .piece.seen{opacity:1;transform:none}
.piece a:focus-visible{outline:2px solid var(--accent);outline-offset:6px}
.tag{position:absolute;top:8px;left:8px;font:11px var(--mono);background:rgba(0,0,0,.65);
color:#fff;padding:2px 7px;pointer-events:none}
.foot{margin:0 auto;padding:4vh 24px 9vh;text-align:center;font:12px/2 var(--mono);
color:var(--muted);letter-spacing:.16em;text-transform:uppercase}
.foot::before{content:"";display:block;width:56px;height:1px;background:var(--rule);margin:0 auto 26px}
.foot a{text-decoration:none;border-bottom:1px solid var(--rule)}
.foot a:hover{color:var(--accent);border-color:var(--accent)}
.banner{position:sticky;top:0;z-index:5;background:#1c1915;color:#f1ece3;
font:12.5px/1.5 var(--mono);padding:9px 16px}
.banner code{color:#f0b77c;word-break:break-all}
dialog{border:0;padding:0;margin:0;width:100vw;height:100vh;height:100dvh;max-width:none;max-height:none;
background:var(--lb);color:#efe8dc}
dialog::backdrop{background:var(--lb)}
dialog[open]{display:grid;grid-template-rows:minmax(0,1fr) auto}
.stage{position:relative;display:grid;place-items:center;min-height:0;padding:16px}
.stage img{max-width:100%;max-height:100%;object-fit:contain;display:block;animation:fade .35s ease}
.bar{display:flex;gap:20px;justify-content:space-between;align-items:baseline;
padding:12px clamp(16px,3vw,32px) max(14px,env(safe-area-inset-bottom))}
.bar p{margin:0;font-style:italic;font-size:1rem;line-height:1.4;color:#d8cfbf}
.bar span{font:12px var(--mono);color:#8f8676;white-space:nowrap;letter-spacing:.06em}
.nav{position:absolute;top:0;bottom:0;width:22%;border:0;background:none;color:#efe8dc;
font:300 44px/1 var(--serif);cursor:pointer;opacity:0;transition:opacity .2s}
.nav:hover,.nav:focus-visible{opacity:.85}
.prev{left:0;text-align:left;padding-left:22px}.next{right:0;text-align:right;padding-right:22px}
.x{position:absolute;top:10px;right:12px;z-index:2;border:0;background:rgba(0,0,0,.35);color:#efe8dc;
width:42px;height:42px;font:300 28px/1 var(--serif);cursor:pointer;border-radius:50%}
@keyframes rise{from{opacity:0;transform:translateY(16px)}}
@keyframes fade{from{opacity:0}}
@media (max-width:700px){:root{--space:76px;--maxw:100vw}
body{font-size:16px}
.title-wall{min-height:52vh;padding:15vh 16px 8vh}
.cover{height:66vh}
.room-head{padding:0 16px}
.piece{width:100%;max-width:100%}
.piece img{width:100%;box-shadow:none}
.label{padding:0 16px;margin-top:14px;min-width:0}
.pair{flex-direction:column;width:100%;gap:var(--space)}
.pair .piece{flex:none;width:100%}
.nav{opacity:.55;font-size:34px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}
.js .piece{opacity:1;transform:none}}
"""

_JS = """
(function(){
var d=document,dlg=d.getElementById('lb'),img=dlg.querySelector('img'),
cap=dlg.querySelector('p'),cnt=dlg.querySelector('span'),
links=[].slice.call(d.querySelectorAll('.piece a')),i=0,x0=null;
d.documentElement.classList.add('js');
var pieces=[].slice.call(d.querySelectorAll('.piece'));
if('IntersectionObserver' in window){
 var io=new IntersectionObserver(function(es){es.forEach(function(e){
  if(e.isIntersecting){e.target.classList.add('seen');io.unobserve(e.target)}})},
  {rootMargin:'0px 0px -8% 0px'});
 pieces.forEach(function(p){io.observe(p)});
}else pieces.forEach(function(p){p.classList.add('seen')});
function show(n){i=(n+links.length)%links.length;var a=links[i];
 img.src=a.getAttribute('href');img.alt=a.querySelector('img').alt;
 cap.textContent=a.dataset.cap||'';cnt.textContent=(i+1)+' / '+links.length;
 [1,-1].forEach(function(s){var b=links[(i+s+links.length)%links.length];
  if(b)(new Image()).src=b.getAttribute('href')})}
function open(n){show(n);if(!dlg.open){dlg.showModal();history.pushState({lb:1},'')}}
function close(){if(history.state&&history.state.lb)history.back();else dlg.close()}
links.forEach(function(a,n){a.addEventListener('click',function(e){
 if(e.metaKey||e.ctrlKey||e.shiftKey)return;e.preventDefault();open(n)})});
dlg.querySelector('.prev').onclick=function(){show(i-1)};
dlg.querySelector('.next').onclick=function(){show(i+1)};
dlg.querySelector('.x').onclick=close;
dlg.addEventListener('cancel',function(e){e.preventDefault();close()});
addEventListener('popstate',function(){if(dlg.open)dlg.close()});
d.addEventListener('keydown',function(e){if(!dlg.open)return;
 if(e.key==='ArrowLeft')show(i-1);else if(e.key==='ArrowRight')show(i+1)});
dlg.addEventListener('touchstart',function(e){x0=e.touches[0].clientX},{passive:true});
dlg.addEventListener('touchend',function(e){if(x0===null)return;
 var dx=e.changedTouches[0].clientX-x0;x0=null;if(Math.abs(dx)>45)show(i+(dx<0?1:-1))});
})();
"""


def _e(text: str) -> str:
    return html.escape(text, quote=True)


def split_heading(heading: str) -> tuple[str, str]:
    """'Sunday 9 November · Ewangan, Narok West' -> (when, where)."""
    when, _, where = heading.partition(" · ")
    return when.strip(), where.strip()


def roman(n: int) -> str:
    """Room numbers on the wall: 1 -> 'I', 14 -> 'XIV'."""
    out = ""
    for value, numeral in (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ):
        while n >= value:
            out += numeral
            n -= value
    return out


def hang(items: list[Item], ratios: dict[str, float]) -> list[list[Item]]:
    """Arrange a room's photos on the wall, in order: two portraits in a row
    hang side by side as a pair (one tall phone frame alone in that much
    space looks lost); everything else hangs on its own."""
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
    """The opening photo: the `cover:` line when it names a photo in the
    gallery, nothing for 'none', else the first clearly landscape photo (a
    portrait cropped to a wide frame loses its subject, so no cover beats a
    bad one)."""
    if gallery.cover.lower() == "none":
        return None
    named = next((i for i in items if i.rel == gallery.cover), None)
    if named is not None:
        return named
    return next((i for i in items if photos[i.rel].ratio >= COVER_MIN_RATIO), None)


def page_html(
    gallery: Gallery,
    photos: dict[str, Rendered],
    *,
    asset_prefix: str = "",
    preview_of: Path | None = None,
    page_url: str | None = None,
) -> str:
    """The gallery page: a title wall, the opening photo, then one room per
    section with each photo hung large, its wall label (plate number,
    capture time, caption) beneath. `asset_prefix` is prepended to photos/
    and thumbs/ (the published copy at '<base>/<slug>' needs '<slug>/');
    `page_url` (an absolute URL, publish only) adds link-preview tags so a
    WhatsApp or iMessage link shows the cover; `preview_of` adds the local
    edit banner and filename tags. Every string from the selection file is
    escaped."""
    items = [i for i in gallery.items if i.rel in photos]
    cover_item = pick_cover(gallery, items, photos)
    plate = {item.rel: n for n, item in enumerate(items, 1)}

    def src(kind: str, r: Rendered) -> str:
        return _e(f"{asset_prefix}{kind}/{r.name}")

    head = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">',
        '<meta name="robots" content="noindex,nofollow,noimageindex,noarchive">',
        '<meta name="referrer" content="no-referrer">',
        f"<title>{_e(gallery.title)}</title>",
    ]
    if gallery.subtitle:
        head.append(f'<meta name="description" content="{_e(gallery.subtitle)}">')
    if page_url and cover_item is not None:
        base = page_url.rstrip("/") + "/"
        cover_url = base + f"thumbs/{photos[cover_item.rel].name}"
        head += [
            '<meta property="og:type" content="website">',
            f'<meta property="og:title" content="{_e(gallery.title)}">',
            f'<meta property="og:description" content="{_e(gallery.subtitle)}">',
            f'<meta property="og:image" content="{_e(cover_url)}">',
            '<meta name="twitter:card" content="summary_large_image">',
        ]
    head += [f"<style>{_CSS}</style></head><body>"]

    body: list[str] = []
    if preview_of is not None:
        body.append(
            '<div class="banner">Preview, not published. Drop, reorder or re-caption in '
            f"<code>{_e(str(preview_of / 'selection.txt'))}</code>, then re-run fdx-share.</div>"
        )
    n_photos = len(items)
    body += [
        '<header class="title-wall">',
        f'<p class="kicker">{n_photos} photograph{"s" if n_photos != 1 else ""}</p>',
        f"<h1>{_e(gallery.title)}</h1>",
        f'<p class="meta">{_e(gallery.subtitle)}</p>' if gallery.subtitle else "",
        "</header>",
    ]
    if cover_item is not None:
        r = photos[cover_item.rel]
        body.append(
            f'<figure class="cover" style="--c:{r.color}"><img src="{src("photos", r)}" '
            f'alt="" width="{r.width}" height="{r.height}" fetchpriority="high"></figure>'
        )

    def piece(item: Item) -> str:
        r = photos[item.rel]
        no = plate[item.rel]
        alt = item.caption or f"Photograph {no}"
        sizes = (
            "(max-width:700px) 100vw, min(90vw, 1320px)"
            if r.ratio >= 1
            else "(max-width:700px) 100vw, 46vw"
        )
        label = [f'<span class="no">No.&nbsp;{no:02d}</span>']
        if item.caption:
            label.append(f'<span class="cap">{_e(item.caption)}</span>')
        if r.taken:
            label.append(f'<span class="t">{_e(r.taken)}</span>')
        tag = (
            f'<span class="tag">{_e(item.rel)}</span>' if preview_of is not None else ""
        )
        return (
            f'<figure class="piece" style="--ar:{r.ratio:.4f};--c:{r.color}">'
            f'<a href="{src("photos", r)}" data-cap="{_e(item.caption)}">'
            f'<img src="{src("photos", r)}" '
            f'srcset="{src("thumbs", r)} {r.thumb_width}w, {src("photos", r)} {r.width}w" '
            f'sizes="{sizes}" width="{r.width}" height="{r.height}" alt="{_e(alt)}" '
            f'loading="lazy" decoding="async"></a>'
            f'<figcaption class="label">{"".join(label)}</figcaption>{tag}</figure>'
        )

    body.append("<main>")
    room = 0
    for s in gallery.sections:
        shown = [i for i in s.items if i.rel in photos]
        if not shown:
            continue
        room += 1
        when, where = split_heading(s.heading)
        head_bits = [f'<p class="room-no">{roman(room)}</p>']
        if when:
            head_bits.append(f"<h2>{_e(when)}</h2>")
        if where:
            head_bits.append(f"<p>{_e(where)}</p>")
        body.append(
            f'<section class="room"><header class="room-head">{"".join(head_bits)}</header>'
        )
        for wall in hang(shown, {i.rel: photos[i.rel].ratio for i in shown}):
            if len(wall) == 1:
                body.append(piece(wall[0]))
            else:
                ratio_sum = sum(photos[i.rel].ratio for i in wall)
                body.append(
                    f'<div class="pair" style="--sum:{ratio_sum:.4f}">'
                    + "".join(piece(i) for i in wall)
                    + "</div>"
                )
        body.append("</section>")
    body += [
        "</main>",
        f'<footer class="foot">{_e(gallery.title)}<br>'
        f'Made with <a href="{FRAMEDEX_URL}">framedex</a></footer>',
        '<dialog id="lb" aria-label="Photo viewer"><div class="stage"><img alt="">'
        '<button class="nav prev" aria-label="Previous photo">&lsaquo;</button>'
        '<button class="nav next" aria-label="Next photo">&rsaquo;</button>'
        '<button class="x" aria-label="Close">&times;</button></div>'
        '<div class="bar"><p></p><span></span></div></dialog>',
        f"<script>{_JS}</script></body></html>",
    ]
    return "\n".join(b for b in head + body if b) + "\n"
