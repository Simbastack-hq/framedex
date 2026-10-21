"""
framedex.share_edit — the browser editor behind `fdx share <folder>`.

A small HTTP server on 127.0.0.1 shows the gallery as it will look, with
editing on top: click a photo to view it, hide or restore it (on the photo, in
the viewer, or with H), drag to reorder within its day, click the title to
rename, choose the cover, toggle a passcode, Publish.
Every edit is written straight to selection.txt (plain text stays the source
of truth); the server re-renders the page markup and the browser swaps it in
without losing its place.

Boundaries: it listens on loopback only; every path carries a per-run random
token and every POST must repeat it in an `X-FDX-Token` header (a page on
another origin cannot add that header without a CORS preflight, which this
server never answers); the Host header must be exactly this server (no DNS
rebinding); and it serves the rendered web copies and nothing else, never an
original or a sidecar. Passcodes live in memory for the one publish call.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from framedex.share import (
    Gallery,
    GallerySection,
    Item,
    write_selection_file,
)
from framedex.share_site import _HEAD, SITE_DIR, Site, _e, asset, gallery_markup

MAX_BODY = 1_000_000  # bytes; an edit payload is a few KB
MAX_TEXT = 200  # title / heading length
_ASSET_RE = re.compile(r"^site/(photos|thumbs)/[0-9a-f]{16}\.jpg$")


class EditError(ValueError):
    pass


@dataclass
class Editor:
    """What the server edits: one gallery's work folder."""

    work: Path
    name: str
    gallery: Gallery
    site: Site
    config: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def selection(self) -> Path:
        return self.work / "selection.txt"


def apply_edit(gallery: Gallery, payload: Any) -> Gallery:
    """A new Gallery from the editor's JSON: the same photos (every one, in
    its own section, shown or hidden), a new order, title and cover. Anything
    else (a photo it doesn't know, a missing one, a move across sections,
    over-long text) is refused, so a bad request can never drop or add a
    photo. Captions, subtitle and the pool carry over untouched."""
    if not isinstance(payload, dict):
        raise EditError("expected a JSON object")
    title = _text(payload.get("title"), "title")
    cover = payload.get("cover", gallery.cover)
    sections = payload.get("sections")
    if not isinstance(sections, list) or len(sections) != len(gallery.sections):
        raise EditError("sections do not match the gallery")
    new_sections: list[GallerySection] = []
    for old, sec in zip(gallery.sections, sections, strict=True):
        if not isinstance(sec, dict) or not isinstance(sec.get("items"), list):
            raise EditError("each section needs an items list")
        by_rel = {i.rel: i for i in old.items}
        items: list[Item] = []
        for entry in sec["items"]:
            if not isinstance(entry, dict) or entry.get("rel") not in by_rel:
                raise EditError("unknown photo in the edit")
            base = by_rel.pop(entry["rel"])
            items.append(Item(base.rel, base.caption, bool(entry.get("hidden"))))
        if by_rel:
            raise EditError("photos missing from the edit")
        new_sections.append(GallerySection(old.heading, items))
    rels = {i.rel for s in new_sections for i in s.items}
    if not isinstance(cover, str) or (cover not in rels and cover not in ("", "none")):
        raise EditError("the cover must be one of the gallery's photos")
    return Gallery(
        title,
        gallery.subtitle,
        cover,
        gallery.captions,
        new_sections,
        list(gallery.pool),
    )


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise EditError(f"{what} must be text")
    text = " ".join(value.split())
    if not text or len(text) > MAX_TEXT:
        raise EditError(f"{what} must be 1-{MAX_TEXT} characters")
    return text


def editor_html(ed: Editor, token: str, *, protected: bool, configured: bool) -> str:
    g = ed.gallery
    hidden = sum(1 for i in g.all_items if i.hidden)
    data = json.dumps(
        {
            "token": token,
            "cover": g.cover,
            "configured": configured,
        }
    ).replace("</", "<\\/")
    return "".join(
        [
            _HEAD,
            f"<title>Editing · {_e(g.title)}</title>",
            f"<style>{asset('gallery.css')}{asset('edit.css')}</style></head>",
            '<body class="editing">',
            '<div class="bar" role="toolbar" aria-label="Gallery editor">',
            '<p class="hint">Click a photo to view it. Hide what you don&rsquo;t want. '
            "Drag to reorder."
            f'<span class="hid">{f" {hidden} hidden." if hidden else ""}</span></p>',
            '<div class="act">',
            '<label class="sw"><input type="checkbox" class="lockon"'
            f"{' checked' if protected else ''}><span></span>Passcode</label>",
            '<span class="pass" hidden><input type="password" class="code" '
            'placeholder="Passcode" aria-label="Passcode" autocomplete="new-password" '
            'spellcheck="false"><button type="button" class="show">Show</button></span>',
            '<button type="button" class="go">Publish</button>',
            "</div>",
            '<div class="done" hidden><span class="ok">Published</span>',
            '<input class="link" readonly aria-label="Gallery link">',
            '<button type="button" class="copy">Copy</button></div>',
            "</div>",
            '<div class="toast" role="status" aria-live="polite" hidden>'
            '<span class="note"></span><button type="button" class="undo" hidden>Undo</button></div>',
            f'<div id="g">{gallery_markup(g, ed.site.photos, asset_prefix=f"{SITE_DIR}/", edit=True)}</div>',
            f'<script type="application/json" id="fdx-edit">{data}</script>',
            "<script>window.fdx={editing:true}</script>",
            f"<script>{asset('gallery.js')}</script><script>{asset('edit.js')}</script>",
            "</body></html>\n",
        ]
    )


def make_handler(
    ed: Editor, token: str, port_box: list[int]
) -> type[BaseHTTPRequestHandler]:
    from framedex import share_publish as pub

    prefix = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        server_version = "fdx-share"
        sys_version = ""

        def log_message(self, *args: Any) -> None:  # quiet terminal
            pass

        def _allowed(self) -> bool:
            port = port_box[0]
            host = self.headers.get("Host", "")
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self._send(403, b"forbidden", "text/plain")
                return False
            if not self.path.startswith(prefix):
                self._send(404, b"not found", "text/plain")
                return False
            return True

        def _send(
            self,
            code: int,
            body: bytes,
            ctype: str,
            extra: dict[str, str] | None = None,
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if not self._allowed():
                return
            rest = self.path[len(prefix) :].split("?", 1)[0]
            if rest == "":
                try:
                    cfg = pub.load_config_or_explain(ed.config)
                    configured, protected = True, pub.is_protected(ed.work, cfg)
                except pub.PublishError:
                    configured, protected = False, False
                page = editor_html(
                    ed, token, protected=protected, configured=configured
                )
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif _ASSET_RE.match(rest):
                f = ed.work / rest
                if f.is_file():
                    self._send(200, f.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            if not self._allowed():
                return
            if self.headers.get("X-FDX-Token") != token:
                self._send(403, b"forbidden", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._json(400, {"error": "bad request"})
                return
            try:
                payload = json.loads(self.rfile.read(length))
            except ValueError:
                self._json(400, {"error": "bad JSON"})
                return
            route = self.path[len(prefix) :]
            with ed.lock:
                if route == "api/save":
                    self._save(payload)
                elif route == "api/publish":
                    self._publish(payload)
                else:
                    self._json(404, {"error": "not found"})

        def _save(self, payload: Any) -> None:
            try:
                gallery = apply_edit(ed.gallery, payload)
            except EditError as e:
                self._json(400, {"error": str(e)})
                return
            write_selection_file(ed.selection, gallery)
            ed.gallery = ed.site.gallery = gallery
            markup = gallery_markup(
                gallery, ed.site.photos, asset_prefix=f"{SITE_DIR}/", edit=True
            )
            self._json(
                200,
                {
                    "html": markup,
                    "hidden": sum(1 for i in gallery.all_items if i.hidden),
                },
            )

        def _publish(self, payload: Any) -> None:
            if not isinstance(payload, dict):
                self._json(400, {"error": "bad request"})
                return
            passcode = payload.get("passcode") or None
            if passcode is not None and not isinstance(passcode, str):
                self._json(400, {"error": "bad passcode"})
                return
            try:
                cfg = pub.load_config_or_explain(ed.config)
                result = pub.publish(
                    ed.site,
                    ed.gallery,
                    ed.work,
                    name=ed.name,
                    cfg=cfg,
                    passcode=passcode,
                    open_ok=bool(payload.get("open")),
                )
            except (pub.PublishError, ValueError) as e:
                first = str(e).split("\n", 1)[0]
                self._json(200, {"error": first})
                return
            print(
                f"Published  {result.url}"
                + ("  (passcode)" if result.protected else "")
            )
            self._json(200, {"url": result.url, "protected": result.protected})

    return Handler


def serve(ed: Editor, *, open_browser: bool = True) -> int:
    """Run the editor until Ctrl-C."""
    token = secrets.token_urlsafe(18)
    port_box = [0]
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ed, token, port_box))
    port_box[0] = server.server_address[1]
    url = f"http://127.0.0.1:{port_box[0]}/{token}/"
    print(f"Editor     {url}\n           Ctrl-C to stop", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
