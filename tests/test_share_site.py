"""Tests for framedex.share_site — the fdx share page and web copies.

The page tests are pure string checks. The render tests need Pillow (the
[images] or [share] extra) and skip without it; they build tiny JPEGs in
tmp_path.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

from framedex import share_site as site
from framedex.share import Gallery, GallerySection, Item


def _rendered(rel: str, w: int = 2048, h: int = 1365) -> site.Rendered:
    name = hashlib.sha1(rel.encode()).hexdigest()[:16] + ".jpg"
    return site.Rendered(rel, name, w, h, w // 2, h // 2, "#806040")


def _gallery(**kw: Any) -> Gallery:
    return Gallery(
        title=kw.get("title", "Narok · November 2025"),
        subtitle=kw.get("subtitle", "8\u20139 November"),
        cover=kw.get("cover", ""),
        captions=kw.get("captions", False),
        sections=[
            GallerySection(
                "Sunday 9 November · Ewangan, Narok West",
                [
                    Item("tall.JPG", "A giraffe."),
                    Item("tall2.JPG", "A zebra.", hidden=kw.get("hide", False)),
                    Item("wide.JPG", kw.get("caption", "A leopard on the track.")),
                ],
            )
        ],
    )


def _photos() -> dict[str, site.Rendered]:
    return {
        "tall.JPG": _rendered("tall.JPG", 1152, 2048),
        "tall2.JPG": _rendered("tall2.JPG", 1152, 2048),
        "wide.JPG": _rendered("wide.JPG"),
    }


def test_page_is_unlisted_quiet_and_credits_framedex() -> None:
    html = site.page_html(_gallery(), _photos())
    assert (
        '<meta name="robots" content="noindex,nofollow,noimageindex,noarchive">' in html
    )
    assert '<meta name="referrer" content="no-referrer">' in html
    assert f'href="{site.FRAMEDEX_URL}">framedex</a>' in html
    assert "<h2>Sunday 9 November<span> · Ewangan, Narok West</span></h2>" in html
    assert "og:image" not in html  # only a published page knows its absolute URL
    assert "<figcaption>" not in html  # captions are alt text unless captions: on
    assert 'alt="A leopard on the track."' in html
    # No third-party requests: every src/href is relative or the credit link.
    assert "https://" not in html.replace(site.FRAMEDEX_URL, "")


def test_the_cover_opens_once_and_portraits_pair() -> None:
    html = site.page_html(_gallery(), _photos())
    wide = _photos()["wide.JPG"].name
    hero = html.index('<div class="hero">')
    assert html.count(f"photos/{wide} 2048w") == 1  # the cover is not repeated below
    assert html.index(f"photos/{wide}") > hero
    pair = re.search(r'<div class="pair" style="--sum:([0-9.]+)">', html)
    assert pair is not None and float(pair.group(1)) == pytest.approx(
        2 * 1152 / 2048, 1e-3
    )


def test_captions_on_shows_them_and_hidden_photos_stay_off_the_page() -> None:
    html = site.page_html(_gallery(captions=True, hide=True), _photos())
    assert "<figcaption>A giraffe.</figcaption>" in html
    assert "A zebra." not in html
    assert '<div class="pair"' not in html  # one portrait left: it hangs alone


def test_page_escapes_everything_from_the_selection_file() -> None:
    evil = '<script>alert("x")</script>'
    html = site.page_html(_gallery(title=evil, caption=evil, subtitle=evil), _photos())
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html


def test_asset_prefix_and_link_preview_for_the_published_page() -> None:
    photos = _photos()
    html = site.page_html(
        _gallery(),
        photos,
        asset_prefix="mara-abc/",
        page_url="https://framedex.art/g/mara-abc",
    )
    wide = photos["wide.JPG"].name
    assert (
        f'srcset="mara-abc/thumbs/{wide} 1024w, mara-abc/photos/{wide} 2048w"' in html
    )
    assert f'href="mara-abc/photos/{wide}"' in html
    assert (
        f'<meta property="og:image" content="https://framedex.art/g/mara-abc/thumbs/{wide}">'
        in html
    )


def test_edit_markup_keeps_hidden_photos_and_the_cover_slot() -> None:
    html = site.gallery_markup(_gallery(hide=True), _photos(), edit=True)
    assert 'class="ph solo off"' in html and 'data-rel="tall2.JPG"' in html
    assert '<span class="slot" data-rel="wide.JPG" hidden></span>' in html
    assert "<h1 contenteditable>" in html


def test_sealed_markup_carries_no_plain_image_paths() -> None:
    photos = _photos()
    sealed = site.Sealed(
        {rel: (f"photos/{rel}.bin", f"thumbs/{rel}.bin") for rel in photos}
    )
    html = site.gallery_markup(_gallery(), photos, sealed=sealed)
    assert ".jpg" not in html
    assert 'data-photo="photos/wide.JPG.bin"' in html and "src=" not in html


def test_cover_rules() -> None:
    photos = _photos()
    items = _gallery().items
    assert site.pick_cover(_gallery(), items, photos) == items[2]  # first wide one
    assert site.pick_cover(_gallery(cover="tall.JPG"), items, photos) == items[0]
    assert site.pick_cover(_gallery(cover="none"), items, photos) is None
    only_tall = {"tall.JPG": photos["tall.JPG"]}
    assert site.pick_cover(_gallery(), items[:1], only_tall) is None


def test_hang_pairs_consecutive_portraits_only() -> None:
    items = [Item(n) for n in ("p1", "p2", "p3", "w1", "p4", "p5", "sq")]
    ratios = {
        "p1": 0.56,
        "p2": 0.75,
        "p3": 0.56,
        "w1": 1.5,
        "p4": 0.56,
        "p5": 0.8,
        "sq": 1.0,
    }
    walls = site.hang(items, ratios)
    assert [[i.rel for i in w] for w in walls] == [
        ["p1", "p2"],
        ["p3"],  # its neighbour is wide
        ["w1"],
        ["p4", "p5"],
        ["sq"],
    ]
    assert site.hang([], {}) == []


def test_split_heading() -> None:
    assert site.split_heading("Sunday 9 November · Talek") == (
        "Sunday 9 November",
        "Talek",
    )
    assert site.split_heading("Undated") == ("Undated", "")


def test_render_source_prefers_a_raw_s_jpeg_sibling(tmp_path: Path) -> None:
    raw = tmp_path / "DSC_1.NEF"
    raw.write_bytes(b"raw")
    assert site.render_source(raw) == raw
    jpg = tmp_path / "DSC_1.JPG"
    jpg.write_bytes(b"jpg")
    assert site.render_source(raw) == jpg
    assert site.render_source(jpg) == jpg


# --- pixels (needs Pillow) -----------------------------------------------------


def _jpeg_with_gps(path: Path, size: tuple[int, int], orientation: int = 1) -> None:
    pil = pytest.importorskip("PIL.Image")
    img = pil.new("RGB", size, (180, 120, 60))
    from PIL import ImageDraw

    w, h = size  # large shapes, so a 9x8 hash sees the same picture at any size
    draw = ImageDraw.Draw(img)
    draw.rectangle((w // 8, h // 5, w // 2, h * 4 // 5), fill=(25, 25, 25))
    draw.ellipse((w * 5 // 8, h // 8, w * 7 // 8, h // 2), fill=(240, 235, 220))
    exif = pil.Exif()
    exif[0x010F] = "NIKON CORPORATION"
    exif[0x0110] = "NIKON Z 8"
    exif[0x0112] = orientation
    exif[0x8769] = {0x9003: "2025:11:09 09:48:12", 0xA431: "6011234"}
    exif[0x8825] = {1: "S", 2: (1.0, 30.0, 41.16), 3: "E", 4: (35.0, 21.0, 41.73)}
    img.save(path, "JPEG", exif=exif.tobytes())


def _exif(path: Path) -> Any:
    pil = pytest.importorskip("PIL.Image")
    with pil.open(path) as im:
        return im.getexif()


def test_render_site_writes_upright_web_copies_without_gps(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    wide, tall = shoot / "wide.JPG", shoot / "tall.JPG"
    _jpeg_with_gps(wide, (3000, 2000))
    _jpeg_with_gps(tall, (3000, 2000), orientation=6)  # stored sideways
    before = {p: p.read_bytes() for p in (wide, tall)}

    work = tmp_path / "work"
    out = site.render_site(_gallery(), {"wide.JPG": wide, "tall.JPG": tall}, work)

    assert {p: p.read_bytes() for p in (wide, tall)} == before  # originals untouched
    w, t = out.photos["wide.JPG"], out.photos["tall.JPG"]
    assert (w.width, w.height) == (2048, 1365)
    assert (t.width, t.height) == (1365, 2048)  # orientation applied
    assert max(w.thumb_width, w.thumb_height) == site.THUMB_LONG_EDGE
    exif = _exif(out.root / "photos" / w.name)
    assert exif.get(0x0110) == "NIKON Z 8"
    assert exif.get_ifd(0x8769).get(0x9003) == "2025:11:09 09:48:12"
    assert not exif.get_ifd(0x8825)  # GPS gone
    assert 0xA431 not in exif.get_ifd(0x8769)  # serial gone
    assert 0x0112 not in exif  # pixels are upright
    for f in (out.root / "thumbs").iterdir():
        assert not _exif(f).get_ifd(0x8825)
    assert (out.root / "index.html").exists()
    with pil.open(out.root / "photos" / t.name) as im:
        assert im.size == (1365, 2048)


def test_keep_gps_opts_in_and_renders_are_reused_then_pruned(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    wide, tall = shoot / "wide.JPG", shoot / "tall.JPG"
    _jpeg_with_gps(wide, (800, 600))
    _jpeg_with_gps(tall, (600, 800))
    originals = {"wide.JPG": wide, "tall.JPG": tall}
    work = tmp_path / "work"

    gps_site = site.render_site(_gallery(), originals, work, keep_gps=True)
    name = gps_site.photos["wide.JPG"].name
    assert _exif(gps_site.root / "photos" / name).get_ifd(0x8825)

    plain = site.render_site(_gallery(), originals, work)
    assert plain.photos["wide.JPG"].name != name  # the GPS choice is in the name
    assert not (plain.root / "photos" / name).exists()  # stale render pruned
    mtime = (plain.root / "photos" / plain.photos["wide.JPG"].name).stat().st_mtime_ns

    hidden = Gallery(
        "T",
        sections=[
            GallerySection("", [Item("wide.JPG"), Item("tall.JPG", hidden=True)])
        ],
    )
    kept_both = site.render_site(hidden, originals, work)
    assert set(kept_both.photos) == {
        "wide.JPG",
        "tall.JPG",
    }  # hidden: rendered, restorable

    stray = plain.root / "photos" / ".private.JPG"  # not ours: never deleted
    stray.write_bytes(b"someone else's file")
    lookalike = (
        plain.root / "photos" / "0123456789abcdef.jpg"
    )  # our pattern, not our file
    lookalike.write_bytes(b"an unrelated original")
    smaller = Gallery("T", sections=[GallerySection("", [Item("wide.JPG")])])
    again = site.render_site(smaller, originals, work)
    kept = again.root / "photos" / again.photos["wide.JPG"].name
    assert kept.stat().st_mtime_ns == mtime  # re-used, not re-encoded
    assert sorted(f.name for f in (again.root / "photos").iterdir()) == sorted(
        [".private.JPG", "0123456789abcdef.jpg", kept.name]
    )  # only renders this work folder recorded were removed


def test_signatures_match_for_the_same_picture(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    from framedex.share import Photo, same_picture

    a, b, sideways = tmp_path / "a.JPG", tmp_path / "b.JPG", tmp_path / "s.JPG"
    _jpeg_with_gps(a, (900, 600))
    _jpeg_with_gps(b, (1800, 1200))
    _jpeg_with_gps(sideways, (1800, 1200), orientation=6)
    sigs = site.signatures(
        [
            Photo("a.JPG", a),
            Photo("b.JPG", b),
            Photo("s.JPG", sideways),
            Photo("c.JPG", tmp_path / "c.JPG"),
        ]
    )
    assert set(sigs) == {"a.JPG", "b.JPG", "s.JPG"}  # unreadable c: none
    assert sigs["a.JPG"].aspect == pytest.approx(1.5)
    assert sigs["s.JPG"].aspect == pytest.approx(2 / 3)  # read upright
    assert same_picture(sigs["a.JPG"], sigs["b.JPG"])
    assert not same_picture(sigs["a.JPG"], sigs["s.JPG"])
