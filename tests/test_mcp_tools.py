"""Tests for framedex.mcp_tools — the archive as tools, without the MCP SDK.

Hermetic: no `mcp`, no Pillow, no ffmpeg (subprocess mocked). The Pillow
compositing and the SDK round-trip live in tests/test_mcp_integration.py and
run in the CI job that installs the [mcp] extra.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from framedex import mcp_tools as mt
from framedex import pipeline


def _sidecar(
    root: Path, rel: str, fm: dict[str, Any], body: str = "**Scene:** A lion.\n"
) -> Path:
    full = {"file": Path(rel).name, "path": rel, "media_type": "image", **fm}
    media = root / rel
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"x")
    p = pipeline.sidecar_path(media)
    p.write_text(
        "---\n"
        + yaml.safe_dump(full, sort_keys=False)
        + "---\n\n# x\n\n## Description\n\n"
        + body
    )
    return p


# --- roots guard -----------------------------------------------------------


def test_roots_from_args_requires_existing_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a directory"):
        mt.Roots.from_args([str(tmp_path / "missing")])
    roots = mt.Roots.from_args([str(tmp_path), str(tmp_path)])
    assert roots.roots == (tmp_path.resolve(),)


def test_roots_resolve_rejects_escapes_and_follows_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"x")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"secret")
    (root / "link.jpg").symlink_to(outside)
    roots = mt.Roots.from_args([str(root)])
    assert roots.resolve(root / "a.jpg") == (root / "a.jpg").resolve()
    assert roots.resolve("a.jpg") == (root / "a.jpg").resolve()  # relative: one root
    for bad in (outside, root / ".." / "outside.jpg", root / "link.jpg"):
        with pytest.raises(ValueError, match="outside the configured roots"):
            roots.resolve(bad)
    with pytest.raises(ValueError, match="not found"):
        roots.resolve(root / "nope.jpg")


def test_roots_relative_path_ambiguous_with_several_roots(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    roots = mt.Roots.from_args([str(a), str(b)])
    with pytest.raises(ValueError, match="absolute"):
        roots.resolve("x.jpg", must_exist=False)
    with pytest.raises(ValueError, match="which root"):
        roots.single(None)
    assert roots.single(str(b)) == b.resolve()
    with pytest.raises(ValueError, match="not one of the configured roots"):
        roots.single(str(tmp_path))


# --- query_media -------------------------------------------------------------


def test_query_media_returns_compact_matches_with_effective_rating(
    tmp_path: Path,
) -> None:
    _sidecar(
        tmp_path,
        "mara/1.NEF",
        {"rating": "cull", "user_rating": "keep", "keywords": ["lion"]},
    )
    _sidecar(tmp_path, "mara/2.NEF", {"rating": "keep", "keywords": ["lion", "dust"]})
    _sidecar(tmp_path, "coast/3.NEF", {"rating": "keep"})
    roots = mt.Roots.from_args([str(tmp_path)])

    out = mt.query_media(roots, rating="keep", keywords=["lion"])
    assert out["total"] == 2 and out["has_more"] is False
    first = out["matches"][0]
    assert first["path"] == str(tmp_path / "mara/1.NEF")
    assert first["sidecar_path"].endswith("1.NEF.description.md")
    assert first["effective_rating"] == "keep" and first["rating"] == "cull"
    assert first["user_rating"] == "keep" and first["scene"] == "A lion."

    page = mt.query_media(roots, folder="mara", offset=0, limit=1)
    assert [Path(m["path"]).name for m in page["matches"]] == ["1.NEF"]
    assert page["total"] == 2 and page["has_more"] is True  # more beyond this page
    last = mt.query_media(roots, folder="mara", offset=1, limit=1)
    assert [Path(m["path"]).name for m in last["matches"]] == ["2.NEF"]
    assert last["has_more"] is False


def test_query_media_validates_arguments(tmp_path: Path) -> None:
    roots = mt.Roots.from_args([str(tmp_path)])
    with pytest.raises(ValueError, match="rating"):
        mt.query_media(roots, rating="banana")
    with pytest.raises(ValueError, match="media"):
        mt.query_media(roots, media="audio")
    with pytest.raises(ValueError, match="limit"):
        mt.query_media(roots, limit=0)
    with pytest.raises(ValueError, match="limit"):
        mt.query_media(roots, limit=mt.QUERY_MAX_LIMIT + 1)
    with pytest.raises(ValueError, match="folder"):
        mt.query_media(roots, folder="../x")


def test_query_media_drops_records_whose_media_escapes_the_root(tmp_path: Path) -> None:
    """A frontmatter `path` (or a symlinked sidecar) that resolves outside the
    root is never handed to the host."""
    root = tmp_path / "root"
    root.mkdir()
    _sidecar(root, "ok.NEF", {"rating": "keep"})
    evil = root / "evil.NEF.description.md"
    evil.write_text("---\nfile: evil.NEF\npath: ../outside.NEF\nrating: keep\n---\n")
    roots = mt.Roots.from_args([str(root)])
    out = mt.query_media(roots)
    assert [Path(m["path"]).name for m in out["matches"]] == ["ok.NEF"]
    assert out["skipped_malformed"] == 1


# --- read_sidecar / archive_overview ----------------------------------------


def test_read_sidecar_accepts_media_or_sidecar_path(tmp_path: Path) -> None:
    p = _sidecar(tmp_path, "a.NEF", {"rating": "keep"})
    roots = mt.Roots.from_args([str(tmp_path)])
    assert mt.read_sidecar(roots, str(tmp_path / "a.NEF")) == p.read_text()
    assert mt.read_sidecar(roots, str(p)) == p.read_text()
    (tmp_path / "b.NEF").write_bytes(b"x")
    with pytest.raises(ValueError, match="not indexed"):
        mt.read_sidecar(roots, str(tmp_path / "b.NEF"))


def test_archive_overview_snapshot_or_instruction(tmp_path: Path) -> None:
    roots = mt.Roots.from_args([str(tmp_path)])
    assert "fdx-master" in mt.archive_overview(roots)
    (tmp_path / "_INDEX.md").write_text(
        "# Media Knowledge Base\n\n*Generated 2026-09-09T01:00:00*\n" + "x" * 100
    )
    text = mt.archive_overview(roots)
    assert text.startswith("[Snapshot") and "fdx-master" in text
    assert "# Media Knowledge Base" in text
    (tmp_path / "_INDEX.md").write_text("y" * (mt.OVERVIEW_MAX_BYTES + 10))
    assert mt.archive_overview(roots).endswith("[truncated]")


# --- set_user_rating ------------------------------------------------------------


def test_set_user_rating_writes_keys_preserves_body_bytes_and_clears(
    tmp_path: Path,
) -> None:
    p = _sidecar(
        tmp_path,
        "a.NEF",
        {"rating": "cull"},
        body="**Scene:** Été.\r\n\r\nline two\r\n",
    )
    original_body = p.read_bytes().split(b"\n---", 1)[1]
    roots = mt.Roots.from_args([str(tmp_path)])

    out = mt.set_user_rating(
        roots, str(tmp_path / "a.NEF"), "keep", note="only frame of the cub"
    )
    assert out["changed"] is True and out["user_rating"] == "keep"
    fm = pipeline.read_sidecar_frontmatter(p)
    assert fm is not None
    assert fm["rating"] == "cull"  # the model's verdict is untouched
    assert fm["user_rating"] == "keep" and fm["user_note"] == "only frame of the cub"
    assert fm["user_rated_at"]
    assert (
        p.read_bytes().split(b"\n---", 1)[1] == original_body
    )  # body byte-for-byte, CRLF kept
    stamp = fm["user_rated_at"]

    again = mt.set_user_rating(
        roots, str(tmp_path / "a.NEF"), "keep", note="only frame of the cub"
    )
    assert again["changed"] is False
    fm2 = pipeline.read_sidecar_frontmatter(p)
    assert fm2 is not None and fm2["user_rated_at"] == stamp  # no-op leaves the stamp

    mt.set_user_rating(roots, str(p), "review")  # note omitted → note removed
    fm3 = pipeline.read_sidecar_frontmatter(p)
    assert fm3 is not None and fm3["user_rating"] == "review" and "user_note" not in fm3

    cleared = mt.set_user_rating(roots, str(p), "")
    assert cleared["changed"] is True and cleared["user_rating"] is None
    fm4 = pipeline.read_sidecar_frontmatter(p)
    assert fm4 is not None and not any(k in fm4 for k in pipeline.USER_KEYS)
    assert mt.set_user_rating(roots, str(p), "")["changed"] is False


def test_set_user_rating_rejects_bad_input_and_writes_nothing(tmp_path: Path) -> None:
    p = _sidecar(tmp_path, "a.NEF", {"rating": "keep"})
    before = p.read_bytes()
    roots = mt.Roots.from_args([str(tmp_path)])
    with pytest.raises(ValueError, match="rating"):
        mt.set_user_rating(roots, str(p), "banana")
    (tmp_path / "b.NEF").write_bytes(b"x")
    with pytest.raises(ValueError, match="not indexed"):
        mt.set_user_rating(roots, str(tmp_path / "b.NEF"), "keep")
    p.write_text("---\nrating: [unclosed\n---\nbody\n")
    with pytest.raises(ValueError, match="frontmatter"):
        mt.set_user_rating(roots, str(p), "keep")
    assert p.read_text() == "---\nrating: [unclosed\n---\nbody\n"
    p.write_bytes(before)
    assert list(tmp_path.glob(".*.tmp")) == []


def test_set_user_rating_concurrent_calls_do_not_lose_updates(tmp_path: Path) -> None:
    p = _sidecar(tmp_path, "a.NEF", {"rating": "keep"})
    roots = mt.Roots.from_args([str(tmp_path)])
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for _ in range(20):
                mt.set_user_rating(roots, str(p), "cull", note=f"w{i}")
                mt.set_user_rating(roots, str(p), "keep", note=f"w{i}")
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    fm = pipeline.read_sidecar_frontmatter(p)
    assert (
        fm is not None
        and fm["user_rating"] in {"keep", "cull"}
        and fm["rating"] == "keep"
    )
    assert list(tmp_path.glob(".*.tmp")) == []


# --- contact sheet: geometry + frame extraction (no Pillow) -------------------


def test_sheet_layout_geometry_and_bounds() -> None:
    (w, h), cells = mt.sheet_layout(5)
    assert len(cells) == 5 and [c[0] for c in cells] == [1, 2, 3, 4, 5]
    assert cells[0][1:] == (mt.SHEET_PAD_PX, mt.SHEET_PAD_PX)
    assert cells[4][1:] == (
        mt.SHEET_PAD_PX,
        mt.SHEET_PAD_PX * 2 + mt.SHEET_THUMB_PX,
    )  # second row
    assert (
        w == mt.SHEET_COLUMNS * (mt.SHEET_THUMB_PX + mt.SHEET_PAD_PX) + mt.SHEET_PAD_PX
    )
    assert h == 2 * (mt.SHEET_THUMB_PX + mt.SHEET_PAD_PX) + mt.SHEET_PAD_PX
    (w1, _), one = mt.sheet_layout(1)
    assert len(one) == 1 and w1 == mt.SHEET_THUMB_PX + 2 * mt.SHEET_PAD_PX
    for n in (0, mt.SHEET_MAX_IMAGES + 1):
        with pytest.raises(ValueError, match="between 1 and"):
            mt.sheet_layout(n)


def test_video_frame_uses_notable_timestamp_or_midpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "c.mov"
    clip.write_bytes(b"x")
    calls: list[list[str]] = []

    def run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        assert kw["timeout"] == mt.SUBPROCESS_TIMEOUT_SEC
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(returncode=0, stdout="12.5\n", stderr="")
        Path(cmd[-1]).write_bytes(b"jpg")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("framedex.mcp_tools.subprocess.run", run)
    out = mt.video_frame(clip, tmp_path, timestamp=None)
    assert out == tmp_path / "frame.jpg"
    ffmpeg = calls[1]
    assert ffmpeg[0] == "ffmpeg" and "-nostdin" in ffmpeg
    assert ffmpeg[ffmpeg.index("-ss") + 1] == "6.25"  # midpoint
    calls.clear()
    mt.video_frame(clip, tmp_path, timestamp=3.0)
    assert calls[1][calls[1].index("-ss") + 1] == "3.0"
    calls.clear()
    mt.video_frame(clip, tmp_path, timestamp=99.0)  # beyond the clip → midpoint
    assert calls[1][calls[1].index("-ss") + 1] == "6.25"


def test_video_frame_failures_are_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "c.mov"
    clip.write_bytes(b"x")

    def missing(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr("framedex.mcp_tools.subprocess.run", missing)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        mt.video_frame(clip, tmp_path, timestamp=None)
    monkeypatch.setattr(
        "framedex.mcp_tools.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="nan\n", stderr=""),
    )
    with pytest.raises(RuntimeError, match="duration"):
        mt.video_frame(clip, tmp_path, timestamp=None)


def test_parse_notable_timestamp() -> None:
    assert mt.parse_timestamp("01:30") == 90.0
    assert mt.parse_timestamp("1:02:03") == 3723.0
    assert mt.parse_timestamp("") is None and mt.parse_timestamp("abc") is None


# --- server entry point without the extra --------------------------------------


def test_server_main_without_mcp_extra_exits_with_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from framedex import mcp_server

    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    monkeypatch.setattr(sys, "argv", ["fdx-mcp", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        mcp_server.main()
    assert "[mcp]" in str(exc.value)


def test_server_main_rejects_missing_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from framedex import mcp_server

    monkeypatch.setattr(sys, "argv", ["fdx-mcp", str(tmp_path / "nope")])
    with pytest.raises(SystemExit) as exc:
        mcp_server.main()
    assert "not a directory" in str(exc.value)
    assert os.path.isdir(tmp_path)


# --- containment at the I/O boundary (Codex code review) ---------------------


def test_read_sidecar_refuses_a_derived_sidecar_symlink_that_escapes(
    tmp_path: Path,
) -> None:
    """`a.jpg.description.md -> /outside/private.md`: the derived sidecar is
    resolved and checked before any byte is read."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"x")
    secret = tmp_path / "private.md"
    secret.write_text("---\nrating: keep\n---\nSECRET\n")
    (root / "a.jpg.description.md").symlink_to(secret)
    roots = mt.Roots.from_args([str(root)])
    with pytest.raises(ValueError, match="outside the configured roots"):
        mt.read_sidecar(roots, str(root / "a.jpg"))
    with pytest.raises(ValueError, match="outside the configured roots"):
        mt.set_user_rating(roots, str(root / "a.jpg"), "keep")
    assert secret.read_text().endswith("SECRET\n")


def test_set_user_rating_never_writes_through_a_symlink_or_to_a_non_sidecar(
    tmp_path: Path,
) -> None:
    """An in-root `alias.jpg.description.md -> notes.md` passes containment but
    is not a sidecar: the setter must refuse, and notes.md must stay intact."""
    notes = tmp_path / "notes.md"
    notes.write_text("---\ntitle: my notes\n---\n\nprivate text\n")
    (tmp_path / "alias.jpg").write_bytes(b"x")
    (tmp_path / "alias.jpg.description.md").symlink_to(notes)
    roots = mt.Roots.from_args([str(tmp_path)])
    with pytest.raises(ValueError, match=r"symlink|not a sidecar"):
        mt.set_user_rating(roots, str(tmp_path / "alias.jpg"), "keep")
    with pytest.raises(ValueError, match=r"symlink|not a sidecar"):
        mt.set_user_rating(roots, str(tmp_path / "alias.jpg.description.md"), "keep")
    assert notes.read_text() == "---\ntitle: my notes\n---\n\nprivate text\n"


def test_query_media_validates_before_paging_and_counts_parse_failures(
    tmp_path: Path,
) -> None:
    """An escaping or unparsable record must not consume an offset or inflate
    the total; it is counted in skipped_malformed regardless of the page."""
    root = tmp_path / "root"
    root.mkdir()
    evil = root / "a-evil.NEF.description.md"
    evil.write_text("---\nfile: a-evil.NEF\npath: ../outside.NEF\nrating: keep\n---\n")
    (root / "b-bad.NEF.description.md").write_text("---\nrating: [unclosed\n---\n")
    _sidecar(root, "c-ok.NEF", {"rating": "keep"})
    _sidecar(root, "d-ok.NEF", {"rating": "keep", "location": "not-a-mapping"})
    roots = mt.Roots.from_args([str(root)])
    page = mt.query_media(roots, limit=1)
    assert [Path(m["path"]).name for m in page["matches"]] == ["c-ok.NEF"]
    assert page["total"] == 2 and page["has_more"] is True
    assert page["skipped_malformed"] == 2
    rest = mt.query_media(roots, offset=1, limit=1)
    assert [Path(m["path"]).name for m in rest["matches"]] == ["d-ok.NEF"]
    assert rest["matches"][0]["place"] is None  # a malformed field, not a crash
    assert rest["has_more"] is False


def test_query_media_skips_a_symlinked_sidecar_that_escapes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.description.md"
    outside.write_text("---\nfile: x.jpg\npath: x.jpg\nrating: keep\n---\n")
    (root / "x.jpg.description.md").symlink_to(outside)
    roots = mt.Roots.from_args([str(root)])
    out = mt.query_media(roots)
    assert out["matches"] == [] and out["skipped_malformed"] == 1


def test_video_frame_forces_the_demuxer_and_refuses_unknown_containers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffmpeg auto-detects playlist formats (ffconcat, m3u8) from file content;
    a `.mov` that is really a playlist could pull in files outside the root.
    Forcing the demuxer by extension closes that door."""
    clip = tmp_path / "c.mov"
    clip.write_bytes(b"ffconcat version 1.0\nfile link.mp4\n")
    calls: list[list[str]] = []

    def run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        assert kw["stdin"] is subprocess.DEVNULL
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(returncode=0, stdout="4.0\n", stderr="")
        Path(cmd[-1]).write_bytes(b"jpg")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("framedex.mcp_tools.subprocess.run", run)
    mt.video_frame(clip, tmp_path, timestamp=None)
    for cmd in calls:
        assert cmd[cmd.index("-f") + 1] == "mov"
        assert cmd.index("-f") < cmd.index(str(clip))  # forced before the input
    weird = tmp_path / "c.xyz"
    weird.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="no fixed demuxer"):
        mt.video_frame(weird, tmp_path, timestamp=None)


def test_set_user_rating_identical_concurrent_requests_serialise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two identical requests at once, with the write slowed so the calls
    overlap at the read→write boundary: exactly one writes (changed=True);
    the other sees the first's result and leaves the timestamp alone.
    Without the per-sidecar lock both read the original and both write."""
    import time

    p = _sidecar(tmp_path, "a.NEF", {"rating": "keep"})
    roots = mt.Roots.from_args([str(tmp_path)])
    from framedex.pipeline import atomic_write_bytes

    real_write = atomic_write_bytes

    def slow_write(path: Path, data: bytes) -> None:
        time.sleep(0.2)
        real_write(path, data)

    monkeypatch.setattr("framedex.mcp_tools.atomic_write_bytes", slow_write)
    barrier = threading.Barrier(2)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            results.append(mt.set_user_rating(roots, str(p), "cull", note="dup"))
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(r["changed"] for r in results) == [False, True]
    assert len({r["user_rated_at"] for r in results}) == 1


def test_set_user_rating_refuses_symlinked_sidecars_and_media_but_reads_them(
    tmp_path: Path,
) -> None:
    """Blocker coverage: a symlink to a *valid* sidecar (final component), a
    symlinked parent directory, and a symlinked media file. Reads follow
    them (inside the roots); writes never do."""
    real = _sidecar(tmp_path, "real/x.jpg", {"rating": "keep"})
    (tmp_path / "alias.jpg.description.md").symlink_to(real)
    (tmp_path / "linkdir").symlink_to(tmp_path / "real", target_is_directory=True)
    (tmp_path / "alias-media.jpg").symlink_to(tmp_path / "real" / "x.jpg")
    roots = mt.Roots.from_args([str(tmp_path)])

    assert mt.read_sidecar(roots, str(tmp_path / "alias.jpg.description.md")) == (
        real.read_text()
    )
    assert (
        mt.read_sidecar(roots, str(tmp_path / "linkdir" / "x.jpg")) == real.read_text()
    )
    before = real.read_bytes()
    for ref in (
        tmp_path / "alias.jpg.description.md",
        tmp_path / "linkdir" / "x.jpg",
        tmp_path / "linkdir" / "x.jpg.description.md",
        tmp_path / "alias-media.jpg",
    ):
        with pytest.raises(ValueError, match="symlink"):
            mt.set_user_rating(roots, str(ref), "cull")
    assert real.read_bytes() == before
    # The real path still works.
    assert mt.set_user_rating(roots, str(tmp_path / "real" / "x.jpg"), "cull")[
        "changed"
    ]


def test_query_media_reports_the_canonical_sidecar_path(tmp_path: Path) -> None:
    real = _sidecar(tmp_path, "real/x.jpg", {"rating": "keep"})
    (tmp_path / "alias.jpg.description.md").symlink_to(real)
    roots = mt.Roots.from_args([str(tmp_path)])
    out = mt.query_media(roots)
    assert {m["sidecar_path"] for m in out["matches"]} == {str(real.resolve())}
    for m in out["matches"]:  # …and that path is accepted by the setter
        assert mt.set_user_rating(roots, m["sidecar_path"], "review")[
            "user_rating"
        ] == ("review")


def test_contact_sheet_refused_sidecar_is_surfaced_not_hidden(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL")
    from PIL import Image as PILImage

    assert pil
    root = tmp_path / "root"
    root.mkdir()
    media = root / "a.jpg"
    PILImage.new("RGB", (64, 64), (200, 20, 20)).save(media, "JPEG")
    secret = tmp_path / "secret.md"
    secret.write_text("---\nrating: keep\n---\n")
    (root / "a.jpg.description.md").symlink_to(secret)
    roots = mt.Roots.from_args([str(root)])
    with pytest.raises(ValueError, match=r"no preview could be rendered.*outside"):
        mt.build_contact_sheet(roots, [str(media)])


def test_parse_timestamp_rejects_non_strings() -> None:
    assert mt.parse_timestamp(90) is None
    assert mt.parse_timestamp(None) is None
    assert mt.parse_timestamp(["01:30"]) is None
