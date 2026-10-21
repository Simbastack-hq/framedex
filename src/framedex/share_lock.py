"""
framedex.share_lock — passcode-protected fdx share galleries.

Encryption happens once, at publish time; the host only ever stores
ciphertext, so it needs no server logic, CDN caching is safe, and a gallery
can move between hosts. The browser decrypts with WebCrypto.

    key   = PBKDF2-HMAC-SHA256(passcode, 16-byte random salt, 600,000 rounds)
    file  = b"FDX1" + 12-byte random nonce + AES-256-GCM(key, nonce, data, aad)
    aad   = the file's gallery-relative path as UTF-8 ("photos/<hex>.bin"),
            or "fdx:manifest" for the page body

Every publish draws a new salt, new nonces and new random file names, so a
cached file from an earlier publish can never be paired with a new key.

Before the passcode, a visitor sees the title and a 32 px, already-blurred
copy of the cover (the unlock screen), and can see how many files there are
and their sizes. Everything else (photos, subtitle, section names, captions)
is ciphertext. A weak passcode can be brute-forced offline from the
published files: fine for a family album, not for anything sensitive.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from framedex.share_site import (
    _HEAD,
    Rendered,
    Sealed,
    _e,
    asset,
    gallery_markup,
    pick_cover,
)

if TYPE_CHECKING:
    from framedex.share import Gallery

MAGIC = b"FDX1"
# OWASP's 2023 floor for PBKDF2-HMAC-SHA256; about half a second on a phone.
KDF_ITERATIONS = 600_000
SALT_BYTES = 16
NONCE_BYTES = 12
MANIFEST_AAD = "fdx:manifest"
# The unlock screen's background: the cover shrunk to this width and blurred
# before it is published (the sharp cover never leaves the machine unlocked).
BLUR_WIDTH = 32
BLUR_RADIUS = 2.5
MIN_PASSCODE_LEN = 4


class LockError(ValueError):
    """Wrong passcode, or a file that is not ours / was tampered with."""


def _aesgcm(key: bytes) -> Any:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:  # pragma: no cover - needs the [share] extra
        raise RuntimeError(
            "passcode-protected galleries need the share extra: "
            "uv pip install -e '.[share]'"
        ) from e
    return AESGCM(key)


def derive_key(passcode: str, salt: bytes, iterations: int = KDF_ITERATIONS) -> bytes:
    if len(passcode) < MIN_PASSCODE_LEN:
        raise LockError(f"a passcode needs at least {MIN_PASSCODE_LEN} characters")
    return hashlib.pbkdf2_hmac(
        "sha256", passcode.encode("utf-8"), salt, iterations, dklen=32
    )


def seal(key: bytes, data: bytes, aad: str) -> bytes:
    nonce = os.urandom(NONCE_BYTES)
    sealed: bytes = _aesgcm(key).encrypt(nonce, data, aad.encode("utf-8"))
    return MAGIC + nonce + sealed


def unseal(key: bytes, blob: bytes, aad: str) -> bytes:
    """Decrypt a sealed file; LockError for a wrong key, another file's AAD,
    a truncated or tampered file, or something that is not a sealed file."""
    if blob[: len(MAGIC)] != MAGIC or len(blob) < len(MAGIC) + NONCE_BYTES + 16:
        raise LockError("not an fdx sealed file")
    nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_BYTES]
    try:
        return bytes(
            _aesgcm(key).decrypt(
                nonce, blob[len(MAGIC) + NONCE_BYTES :], aad.encode("utf-8")
            )
        )
    except Exception as e:  # cryptography's InvalidTag
        raise LockError("wrong passcode, or the file was changed") from e


def blurred_cover(jpeg: bytes) -> str:
    """A data: URI of a tiny, already-blurred copy of the cover."""
    from PIL import Image, ImageFilter

    with Image.open(io.BytesIO(jpeg)) as im:
        rgb = im.convert("RGB")
        h = max(1, round(rgb.height * BLUR_WIDTH / rgb.width))
        small = rgb.resize((BLUR_WIDTH, h), Image.Resampling.BOX)
        small = small.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=60)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@dataclass
class Locked:
    """A locked gallery: gallery-relative path -> sealed bytes, plus what the
    page needs to open it."""

    files: dict[str, bytes] = field(default_factory=dict)
    salt: bytes = b""
    body: bytes = b""  # the sealed page markup
    blur: str = ""  # data: URI for the unlock screen


def lock_gallery(
    gallery: Gallery, photos: dict[str, Rendered], site_root: Path, passcode: str
) -> Locked:
    """Seal every shown photo and thumb (fresh random names) and the page
    markup that points at them. Reads the rendered copies under site_root."""
    salt = os.urandom(SALT_BYTES)
    key = derive_key(passcode, salt)
    locked = Locked(salt=salt)
    paths: dict[str, tuple[str, str]] = {}
    shown = [i for i in gallery.items if i.rel in photos]
    for item in shown:
        r = photos[item.rel]
        pair = []
        for kind in ("photos", "thumbs"):
            path = f"{kind}/{secrets.token_hex(8)}.bin"
            locked.files[path] = seal(
                key, (site_root / kind / r.name).read_bytes(), path
            )
            pair.append(path)
        paths[item.rel] = (pair[0], pair[1])
    markup = gallery_markup(gallery, photos, sealed=Sealed(paths))
    locked.body = seal(key, markup.encode("utf-8"), MANIFEST_AAD)
    cover = pick_cover(gallery, shown, photos) or (shown[0] if shown else None)
    if cover is not None:
        locked.blur = blurred_cover(
            (site_root / "thumbs" / photos[cover.rel].name).read_bytes()
        )
    return locked


def locked_html(title: str, locked: Locked, *, asset_prefix: str = "") -> str:
    """The unlock page: the title over the blurred cover and one field.
    Carries no description or link-preview tags (they would leak)."""
    data = {
        "v": 1,
        "salt": base64.b64encode(locked.salt).decode(),
        "iter": KDF_ITERATIONS,
        "body": base64.b64encode(locked.body).decode(),
        "prefix": asset_prefix,
    }
    blob = json.dumps(data).replace("</", "<\\/")
    bg = f' style="background-image:url({locked.blur})"' if locked.blur else ""
    return "".join(
        [
            _HEAD,
            f"<title>{_e(title)}</title>",
            f"<style>{asset('gallery.css')}{asset('lock.css')}</style></head><body>",
            f'<div class="lock"{bg}><form class="gate" autocomplete="off">',
            f"<h1>{_e(title)}</h1><p>This album is private.</p>",
            '<div class="field"><input type="password" name="passcode" '
            'aria-label="Passcode" placeholder="Passcode" autocapitalize="off" '
            'autocorrect="off" spellcheck="false" required>'
            '<button type="submit" aria-label="Open" disabled>&rarr;</button></div>',
            '<p class="msg" role="status" aria-live="polite"></p></form></div>',
            '<div id="g" hidden></div>',
            '<noscript><p class="msg">This album needs JavaScript to open.</p></noscript>',
            f'<script type="application/json" id="fdx-data">{blob}</script>',
            "<script>window.fdx={locked:true}</script>",
            f"<script>{asset('gallery.js')}</script><script>{asset('lock.js')}</script>",
            "</body></html>\n",
        ]
    )
