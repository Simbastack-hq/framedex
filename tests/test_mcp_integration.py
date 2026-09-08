"""Integration tests for fdx-mcp that need the [mcp] extra: the real MCP SDK
(in-memory transport) and Pillow (generated pixels). Skipped when the extra
is absent; the `mcp-extra` CI job installs it and runs them for real.
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
from typing import Any

import pytest
import yaml

mcp_pkg = pytest.importorskip("mcp")
PIL = pytest.importorskip("PIL")

from PIL import Image as PILImage  # noqa: E402

from framedex import mcp_server, mcp_tools, pipeline  # noqa: E402


def _photo(
    root: Path, rel: str, fm: dict[str, Any], color: tuple[int, int, int]
) -> Path:
    media = root / rel
    media.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", (640, 480), color).save(media, "JPEG")
    full = {"file": media.name, "path": rel, "media_type": "image", **fm}
    pipeline.sidecar_path(media).write_text(
        "---\n"
        + yaml.safe_dump(full, sort_keys=False)
        + "---\n\n# x\n\n## Description\n\n**Scene:** A test.\n"
    )
    return media


def _call(server: Any, name: str, args: dict[str, Any]) -> Any:
    from mcp.client import Client

    async def go() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, args)

    return asyncio.run(go())


def test_tools_are_registered_and_read_only_mode_drops_the_setter(
    tmp_path: Path,
) -> None:
    roots = mcp_tools.Roots.from_args([str(tmp_path)])
    server = mcp_server.build_server(roots, read_only=False)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "list_roots",
        "query_media",
        "read_sidecar",
        "archive_overview",
        "contact_sheet",
        "set_user_rating",
    }
    ro = mcp_server.build_server(roots, read_only=True)
    assert "set_user_rating" not in {t.name for t in asyncio.run(ro.list_tools())}


def test_round_trip_query_rate_and_survive_errors(tmp_path: Path) -> None:
    a = _photo(tmp_path, "trip/a.jpg", {"rating": "cull"}, (200, 30, 30))
    _photo(tmp_path, "trip/b.jpg", {"rating": "keep"}, (30, 200, 30))
    roots = mcp_tools.Roots.from_args([str(tmp_path)])
    server = mcp_server.build_server(roots, read_only=False)

    res = _call(server, "query_media", {"rating": "keep"})
    assert res.is_error is False
    assert [Path(m["path"]).name for m in res.structured_content["matches"]] == [
        "b.jpg"
    ]

    # A caller mistake is a ToolError with the message, and the server lives on.
    bad = _call(server, "read_sidecar", {"path": str(tmp_path / ".." / "x.jpg")})
    assert (
        bad.is_error is True and "outside the configured roots" in bad.content[0].text
    )

    body_before = pipeline.sidecar_path(a).read_bytes().split(b"\n---\n", 1)[1]
    rated = _call(
        server, "set_user_rating", {"path": str(a), "rating": "keep", "note": "cub"}
    )
    assert rated.is_error is False and rated.structured_content["user_rating"] == "keep"
    assert pipeline.sidecar_path(a).read_bytes().split(b"\n---\n", 1)[1] == body_before
    assert a.read_bytes()[:2] == b"\xff\xd8"  # the original is untouched

    res = _call(server, "query_media", {"rating": "keep"})
    assert sorted(Path(m["path"]).name for m in res.structured_content["matches"]) == [
        "a.jpg",
        "b.jpg",
    ]


def test_contact_sheet_returns_one_image_and_a_legend(tmp_path: Path) -> None:
    a = _photo(
        tmp_path, "a.jpg", {"rating": "keep", "user_rating": "cull"}, (220, 20, 20)
    )
    b = _photo(tmp_path, "b.jpg", {"rating": "review"}, (20, 20, 220))
    roots = mcp_tools.Roots.from_args([str(tmp_path)])
    server = mcp_server.build_server(roots, read_only=True)

    res = _call(server, "contact_sheet", {"paths": [str(a), str(b)]})
    assert res.is_error is False
    kinds = [type(c).__name__ for c in res.content]
    assert kinds == ["ImageContent", "TextContent"], kinds
    img = PILImage.open(io.BytesIO(base64.b64decode(res.content[0].data)))
    (w, h), cells = mcp_tools.sheet_layout(2)
    assert img.size == (w, h)
    # Real pixels: the first cell is red-ish, the second blue-ish.
    _, x1, y1 = cells[0]
    _, x2, y2 = cells[1]
    mid = mcp_tools.SHEET_THUMB_PX // 2
    p1 = img.getpixel((x1 + mid, y1 + mid))
    p2 = img.getpixel((x2 + mid, y2 + mid))
    assert isinstance(p1, tuple) and isinstance(p2, tuple)
    assert p1[0] > 150 and p1[2] < 80  # red-ish
    assert p2[2] > 150 and p2[0] < 80  # blue-ish
    legend = res.content[1].text.splitlines()
    assert legend[0].startswith(f"1. {a} — effective_rating=cull (user) — scene=")
    assert legend[1].startswith(f"2. {b} — effective_rating=review — scene=")

    too_many = _call(
        server, "contact_sheet", {"paths": [str(a)] * (mcp_tools.SHEET_MAX_IMAGES + 1)}
    )
    assert too_many.is_error is True and "between 1 and" in too_many.content[0].text


def test_filesystem_errors_are_explained_and_mixed_sheets_keep_going(
    tmp_path: Path,
) -> None:
    import os

    good = _photo(tmp_path, "good.jpg", {"rating": "keep"}, (10, 200, 10))
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not a jpeg at all")
    pipeline.sidecar_path(broken).write_text(
        "---\nfile: broken.jpg\npath: broken.jpg\nrating: keep\n---\n"
    )
    roots = mcp_tools.Roots.from_args([str(tmp_path)])
    server = mcp_server.build_server(roots, read_only=False)

    # One undecodable member becomes a labelled failure cell; the sheet still renders.
    res = _call(server, "contact_sheet", {"paths": [str(broken), str(good)]})
    assert res.is_error is False, res.content
    legend = res.content[1].text.splitlines()
    assert "preview unavailable" in legend[0] and "good.jpg" in legend[1]

    if os.geteuid() != 0:  # root ignores file modes
        locked = pipeline.sidecar_path(good)
        locked.chmod(0)
        try:
            denied = _call(server, "read_sidecar", {"path": str(good)})
            assert (
                denied.is_error is True
                and "Permission denied" in denied.content[0].text
            )
            again = _call(server, "list_roots", {})  # the server survived
            assert again.is_error is False
        finally:
            locked.chmod(0o644)
