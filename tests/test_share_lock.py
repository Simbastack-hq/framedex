"""Tests for framedex.share_lock — passcode-protected galleries.

Needs `cryptography` (the [share] extra); the artifact tests also need
Pillow. Both skip cleanly on a bare install. The browser side (WebCrypto)
uses the same format: magic + 12-byte nonce + AES-GCM, AAD = gallery path.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from framedex import share_lock as lock
from framedex import share_publish as pub
from framedex.share import Gallery, GallerySection, Item
from framedex.share_site import Rendered, Site

KEY = lock.derive_key("leopard2025", b"\x00" * 16, iterations=1000)


def test_seal_round_trip_and_format() -> None:
    blob = lock.seal(KEY, b"hello", "photos/a.bin")
    assert blob[:4] == b"FDX1" and len(blob) == 4 + 12 + 5 + 16
    assert lock.unseal(KEY, blob, "photos/a.bin") == b"hello"
    assert lock.seal(KEY, b"hello", "photos/a.bin") != blob  # fresh nonce every time


def test_wrong_passcode_wrong_path_and_tampering_all_fail() -> None:
    blob = lock.seal(KEY, b"hello", "photos/a.bin")
    other = lock.derive_key("leopard2026", b"\x00" * 16, iterations=1000)
    with pytest.raises(lock.LockError):
        lock.unseal(other, blob, "photos/a.bin")
    with pytest.raises(lock.LockError):
        lock.unseal(KEY, blob, "photos/b.bin")  # a file moved to another name
    tampered = blob[:-1] + bytes([blob[-1] ^ 1])
    with pytest.raises(lock.LockError):
        lock.unseal(KEY, tampered, "photos/a.bin")
    with pytest.raises(lock.LockError):
        lock.unseal(KEY, b"\xff\xd8\xff not ours", "photos/a.bin")


def test_key_derivation_is_pbkdf2_sha256() -> None:
    expected = hashlib.pbkdf2_hmac("sha256", b"abcd", b"salt" * 4, 1000, dklen=32)
    assert lock.derive_key("abcd", b"salt" * 4, 1000) == expected
    assert lock.KDF_ITERATIONS >= 600_000
    with pytest.raises(lock.LockError, match="at least"):
        lock.derive_key("abc", b"salt" * 4)


def _jpeg(color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> bytes:
    pil = pytest.importorskip("PIL.Image")
    import io

    buf = io.BytesIO()
    pil.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def _site(tmp_path: Path) -> tuple[Site, Gallery]:
    root = tmp_path / "site"
    for kind in ("photos", "thumbs"):
        (root / kind).mkdir(parents=True)
    names = {"leopard.JPG": "a" * 16 + ".jpg", "hidden.JPG": "b" * 16 + ".jpg"}
    photos = {}
    for rel, name in names.items():
        for kind in ("photos", "thumbs"):
            (root / kind / name).write_bytes(_jpeg((200, 150, 90)))
        photos[rel] = Rendered(rel, name, 64, 48, 64, 48, "#c89660", 100)
    gallery = Gallery(
        "Narok · November 2025",
        subtitle="8\u20139 November",
        sections=[
            GallerySection(
                "Sunday 9 November · Talek",
                [
                    Item("leopard.JPG", "A leopard crosses the track."),
                    Item("hidden.JPG", "x", hidden=True),
                ],
            )
        ],
    )
    return Site(root, gallery, photos), gallery


def test_locked_artifact_holds_nothing_readable(tmp_path: Path) -> None:
    site, gallery = _site(tmp_path)
    objs = pub.build_artifact(
        site,
        gallery,
        slug="t-abc",
        page_url="https://fdx.art/g/t-abc",
        passcode="leopard2025",
        page_object=True,
    )
    bins = [o for o in objs if o.path.endswith(".bin")]
    assert len(bins) == 2  # one photo + its thumb; the hidden photo never leaves
    blob = b"".join(o.data for o in objs)
    assert b"\xff\xd8\xff" not in blob  # no JPEG anywhere
    for secret in (
        b"leopard",
        b"Talek",
        b"November \xe2\x80\x93",
        b"8\xe2\x80\x939",
        b".JPG",
        b"aaaaaaaa",
    ):
        assert secret not in blob.replace(b"leopard2025", b""), secret
    page = next(o for o in objs if o.path == "index.html").data.decode()
    assert "<title>Narok · November 2025</title>" in page  # the title is public
    assert 'name="description"' not in page and "og:" not in page  # no preview leaks
    assert (
        '<meta name="robots" content="noindex,nofollow,noimageindex,noarchive">' in page
    )


def test_the_page_unlocks_with_the_passcode_only(tmp_path: Path) -> None:
    site, gallery = _site(tmp_path)
    locked = lock.lock_gallery(gallery, site.photos, site.root, "leopard2025")
    data = json.loads(
        re.search(
            r'<script type="application/json" id="fdx-data">(.*?)</script>',
            lock.locked_html(gallery.title, locked),
        ).group(1)  # type: ignore[union-attr]
    )
    salt = base64.b64decode(data["salt"])
    key = lock.derive_key("leopard2025", salt, data["iter"])
    markup = lock.unseal(
        key, base64.b64decode(data["body"]), lock.MANIFEST_AAD
    ).decode()
    assert "A leopard crosses the track." in markup and "8\u20139 November" in markup
    photo = re.search(r'data-photo="(photos/[0-9a-f]{16}\.bin)"', markup).group(1)  # type: ignore[union-attr]
    assert lock.unseal(key, locked.files[photo], photo)[:3] == b"\xff\xd8\xff"
    with pytest.raises(lock.LockError):
        lock.unseal(
            lock.derive_key("leopard2026", salt, data["iter"]),
            base64.b64decode(data["body"]),
            lock.MANIFEST_AAD,
        )
    again = lock.lock_gallery(gallery, site.photos, site.root, "leopard2025")
    assert set(again.files).isdisjoint(
        locked.files
    )  # fresh names: no stale cache pairing
    assert locked.blur.startswith("data:image/jpeg;base64,")
