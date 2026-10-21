"""Tests for framedex.share_edit — the browser editor's rules and its local
server's boundaries.

`apply_edit` is pure. The server tests run the real handler on 127.0.0.1 in
a thread (loopback only, nothing leaves the machine) to check the security
rules: token in the path, header on POST, exact Host, rendered copies only.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from framedex import share_edit as ed
from framedex.share import Gallery, GallerySection, Item, parse_selection
from framedex.share_site import Rendered, Site


def _gallery() -> Gallery:
    return Gallery(
        "Trip",
        "8\u20139 November",
        "b.JPG",
        True,
        [
            GallerySection(
                "Saturday", [Item("a.JPG", "cap a"), Item("b.JPG", "cap b")]
            ),
            GallerySection("Sunday", [Item("c.JPG", "cap c")]),
        ],
        [(Item("d.JPG", "cap d"), "over the cap")],
    )


def _payload(**kw: Any) -> dict[str, Any]:
    return {
        "title": kw.get("title", "Trip"),
        "cover": kw.get("cover", "b.JPG"),
        "sections": kw.get(
            "sections",
            [
                {"items": [{"rel": "b.JPG"}, {"rel": "a.JPG", "hidden": True}]},
                {"items": [{"rel": "c.JPG"}]},
            ],
        ),
    }


def test_apply_edit_reorders_hides_renames_and_keeps_the_rest() -> None:
    g = ed.apply_edit(
        _gallery(), _payload(title="  Mara   with the kids ", cover="c.JPG")
    )
    assert g.title == "Mara with the kids" and g.cover == "c.JPG"
    assert [(i.rel, i.hidden, i.caption) for i in g.sections[0].items] == [
        ("b.JPG", False, "cap b"),
        ("a.JPG", True, "cap a"),
    ]
    assert g.captions is True and g.subtitle == "8\u20139 November"
    assert g.pool == _gallery().pool and [s.heading for s in g.sections] == [
        "Saturday",
        "Sunday",
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            _payload(sections=[{"items": [{"rel": "a.JPG"}, {"rel": "b.JPG"}]}]),
            "sections",
        ),
        (
            _payload(
                sections=[
                    {"items": [{"rel": "a.JPG"}, {"rel": "c.JPG"}]},
                    {"items": [{"rel": "b.JPG"}]},
                ]
            ),
            "unknown",  # moving across days is refused
        ),
        (
            _payload(
                sections=[{"items": [{"rel": "a.JPG"}]}, {"items": [{"rel": "c.JPG"}]}]
            ),
            "missing",
        ),
        (
            _payload(
                sections=[
                    {
                        "items": [
                            {"rel": "a.JPG"},
                            {"rel": "b.JPG"},
                            {"rel": "/etc/passwd"},
                        ]
                    },
                    {"items": [{"rel": "c.JPG"}]},
                ]
            ),
            "unknown",
        ),
        (_payload(title=""), "title"),
        (_payload(title="x" * 300), "title"),
        (_payload(cover="d.JPG"), "cover"),  # a pool photo is not in the gallery
        ([], "object"),
    ],
)
def test_apply_edit_refuses_anything_that_would_add_or_drop_a_photo(
    payload: Any, message: str
) -> None:
    with pytest.raises(ed.EditError, match=message):
        ed.apply_edit(_gallery(), payload)


# --- the server ----------------------------------------------------------------------


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[ed.Editor, str, int]]:
    work = tmp_path / "work"
    for kind in ("photos", "thumbs"):
        (work / "site" / kind).mkdir(parents=True)
    photos = {}
    for n, rel in enumerate(("a.JPG", "b.JPG", "c.JPG")):
        name = f"{n:016x}.jpg"
        (work / "site" / "photos" / name).write_bytes(b"jpeg " + rel.encode())
        (work / "site" / "thumbs" / name).write_bytes(b"jpeg " + rel.encode())
        photos[rel] = Rendered(rel, name, 800, 600, 400, 300, "#123456")
    (tmp_path / "a.JPG").write_bytes(b"ORIGINAL")
    (tmp_path / "a.JPG.description.md").write_text("sidecar")
    gallery = _gallery()
    editor = ed.Editor(
        work,
        "trip",
        gallery,
        Site(work / "site", gallery, photos),
        str(tmp_path / "none.yaml"),
    )
    token = "tok123tok123tok123tok"
    port: list[int] = [0]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ed.make_handler(editor, token, port))
    port[0] = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield editor, token, port[0]
    srv.shutdown()
    srv.server_close()


def _req(
    port: int,
    method: str,
    path: str,
    *,
    host: str | None = None,
    body: Any = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    hdrs = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    out = (resp.status, resp.read())
    conn.close()
    return out


def test_editor_page_needs_the_token_and_the_right_host(
    server: tuple[ed.Editor, str, int],
) -> None:
    _, token, port = server
    status, page = _req(port, "GET", f"/{token}/")
    assert status == 200 and b'class="bar"' in page and b"Trip" in page
    assert _req(port, "GET", "/")[0] == 404
    assert _req(port, "GET", "/wrong-token/")[0] == 404
    assert (
        _req(port, "GET", f"/{token}/", host="evil.example:80")[0] == 403
    )  # DNS rebinding
    assert _req(port, "GET", f"/{token}/", host=f"localhost:{port}")[0] == 200


def test_editor_serves_rendered_copies_and_nothing_else(
    server: tuple[ed.Editor, str, int],
) -> None:
    _, token, port = server
    status, body = _req(port, "GET", f"/{token}/site/photos/{0:016x}.jpg")
    assert status == 200 and body == b"jpeg a.JPG"
    for path in (
        "a.JPG",
        "../a.JPG",
        "site/../../a.JPG",
        "a.JPG.description.md",
        "site/photos/x.jpg",
        "selection.txt",
    ):
        assert _req(port, "GET", f"/{token}/{path}")[0] == 404, path


def test_saves_need_the_header_and_write_selection_txt(
    server: tuple[ed.Editor, str, int],
) -> None:
    editor, token, port = server
    assert (
        _req(port, "POST", f"/{token}/api/save", body=_payload())[0] == 403
    )  # no header
    assert (
        _req(
            port,
            "POST",
            f"/{token}/api/save",
            body=_payload(),
            headers={"X-FDX-Token": "nope"},
        )[0]
        == 403
    )
    status, body = _req(
        port,
        "POST",
        f"/{token}/api/save",
        body=_payload(title="Renamed"),
        headers={"X-FDX-Token": token},
    )
    assert status == 200
    reply = json.loads(body)
    assert reply["hidden"] == 1 and "Renamed" in reply["html"]
    saved, _ = parse_selection(editor.selection.read_text())
    assert saved.title == "Renamed" and [i.rel for i in saved.items] == [
        "b.JPG",
        "c.JPG",
    ]
    assert saved.pool == _gallery().pool  # the Not picked pool survives an editor save
    status, body = _req(
        port,
        "POST",
        f"/{token}/api/save",
        body={"title": "x"},
        headers={"X-FDX-Token": token},
    )
    assert (
        status == 400 and editor.gallery.title == "Renamed"
    )  # a bad edit changes nothing


def test_publish_without_config_says_so(server: tuple[ed.Editor, str, int]) -> None:
    _, token, port = server
    status, body = _req(
        port,
        "POST",
        f"/{token}/api/publish",
        body={"passcode": None},
        headers={"X-FDX-Token": token},
    )
    assert status == 200 and "no publish config" in json.loads(body)["error"]
