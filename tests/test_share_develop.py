"""Tests for fdx share --edit and its use of fdx-develop's edited copies — a
file convention, not an import: fdx share hands fdx-develop the picked RAWs
(--files-from) and finds each edit by scanning <root>/developed/<dir>/ for a
*.develop.json record (status, source fingerprint, output name, output
SHA-256) that matches the original.

Synthetic files in tmp_path only; the real fdx-develop is never run (a fake
stands in for it on PATH, or its command is mocked).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from framedex import share


def _record(
    raw: Path,
    jpg: Path,
    status: str = "done",
    created_at: str = "2026-10-21T04:00:00+00:00",
) -> dict[str, Any]:
    st = raw.stat()
    return {
        "status": status,
        "created_at": created_at,
        "source": {"name": raw.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns},
        "output": jpg.name,
        "output_sha256": hashlib.sha256(jpg.read_bytes()).hexdigest(),
    }


def _shoot(
    tmp_path: Path, *, naming: str = "stem", status: str = "done"
) -> tuple[Path, Path, Path]:
    folder = tmp_path / "shoot"
    raw = folder / "day1" / "DSC_1.NEF"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw sensor data")
    if naming == "file name":
        # another RAW shares the stem, so fdx-develop names outputs by file name
        (raw.parent / "dsc_1.ARW").write_bytes(b"other raw")
    name = raw.stem if naming == "stem" else raw.name
    out = folder / "developed" / "day1"
    out.mkdir(parents=True)
    jpg = out / f"{name}.jpg"
    jpg.write_bytes(b"edited jpeg bytes")
    (out / f"{name}.develop.json").write_text(json.dumps(_record(raw, jpg, status)))
    return folder, raw, jpg


@pytest.mark.parametrize("naming", ["stem", "file name"])
def test_a_finished_edit_of_this_exact_original_is_used(
    tmp_path: Path, naming: str
) -> None:
    folder, raw, jpg = _shoot(tmp_path, naming=naming)
    assert share.developed_copy(raw, folder) == jpg
    assert share.is_develop_output(jpg)
    assert not share.is_develop_output(raw)


def test_clash_renamed_outputs_are_found_by_their_records(tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    folder.mkdir()
    out = folder / "developed"
    out.mkdir()
    raws = [folder / n for n in ("A.ARW", "A.NEF", "A.NEF.DNG")]
    for (
        raw
    ) in raws:  # fdx-develop's plan_outputs: every clashing name becomes the file name
        raw.write_bytes(f"sensor data of {raw.name}".encode())
        jpg = out / f"{raw.name}.jpg"
        jpg.write_bytes(f"edit of {raw.name}".encode())
        (out / f"{raw.name}.develop.json").write_text(json.dumps(_record(raw, jpg)))
    assert [share.developed_copy(raw, folder) for raw in raws] == [
        out / "A.ARW.jpg",
        out / "A.NEF.jpg",
        out / "A.NEF.DNG.jpg",
    ]


def test_the_newest_record_wins_over_an_edit_left_under_an_old_name(
    tmp_path: Path,
) -> None:
    folder, raw, old_jpg = _shoot(tmp_path)  # DSC_1.jpg, from before a clash
    new_jpg = old_jpg.with_name("DSC_1.NEF.jpg")  # the name fdx-develop uses now
    new_jpg.write_bytes(b"the newer edit")
    record = _record(raw, new_jpg, created_at="2026-10-22T09:00:00+00:00")
    new_jpg.with_name("DSC_1.NEF.develop.json").write_text(json.dumps(record))
    assert share.developed_copy(raw, folder) == new_jpg


def test_a_record_cannot_point_outside_its_folder(tmp_path: Path) -> None:
    folder, raw, jpg = _shoot(tmp_path)
    record = jpg.with_name("DSC_1.develop.json")
    rec = json.loads(record.read_text())
    rec["output"] = "../../day1/DSC_1.NEF"
    record.write_text(json.dumps(rec))
    assert share.developed_copy(raw, folder) is None


@pytest.mark.skipif(os.name == "nt", reason="symlinks")
def test_an_edit_that_resolves_outside_the_shoot_is_never_used(tmp_path: Path) -> None:
    folder, raw, jpg = _shoot(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "private.jpg").write_bytes(jpg.read_bytes())  # same bytes, same SHA
    jpg.unlink()
    jpg.symlink_to(outside / "private.jpg")
    assert share.developed_copy(raw, folder) is None

    folder, raw, jpg = _shoot(tmp_path / "second")
    day = jpg.parent
    day.rename(outside / "day1")
    day.symlink_to(outside / "day1", target_is_directory=True)  # developed/day1 -> out
    assert share.developed_copy(raw, folder) is None

    folder, raw, jpg = _shoot(tmp_path / "third")
    jpg.unlink()
    jpg.symlink_to(jpg.name)  # a loop: Python 3.12's resolve() raises RuntimeError
    assert share.developed_copy(raw, folder) is None


def test_only_raws_have_edits(tmp_path: Path) -> None:
    folder, _, _ = _shoot(tmp_path)
    camera_jpeg = folder / "day1" / "DSC_1.JPG"
    camera_jpeg.write_bytes(b"camera jpeg")
    assert share.developed_copy(camera_jpeg, folder) is None


@pytest.mark.parametrize("status", ["partial", "failed", "pending", "skipped"])
def test_an_unfinished_edit_is_ignored(tmp_path: Path, status: str) -> None:
    folder, raw, _ = _shoot(tmp_path, status=status)
    assert share.developed_copy(raw, folder) is None


def test_an_edit_of_an_older_version_of_the_original_is_ignored(tmp_path: Path) -> None:
    folder, raw, _ = _shoot(tmp_path)
    st = raw.stat()
    os.utime(raw, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # re-saved since
    assert share.developed_copy(raw, folder) is None


def test_an_edited_copy_changed_after_fdx_develop_wrote_it_is_ignored(
    tmp_path: Path,
) -> None:
    folder, raw, jpg = _shoot(tmp_path)
    jpg.write_bytes(b"someone re-exported over it")
    assert share.developed_copy(raw, folder) is None


def test_broken_or_missing_records_fall_back_to_the_original(tmp_path: Path) -> None:
    folder, raw, jpg = _shoot(tmp_path)
    record = jpg.with_name("DSC_1.develop.json")
    record.write_text("{not json")
    assert share.developed_copy(raw, folder) is None
    record.write_text('["a list"]')
    assert share.developed_copy(raw, folder) is None
    record.unlink()
    assert share.developed_copy(raw, folder) is None
    assert share.developed_copy(tmp_path / "elsewhere.NEF", folder) is None


def test_edited_copies_are_not_photos_waiting_to_be_indexed(tmp_path: Path) -> None:
    folder, raw, _ = _shoot(tmp_path)
    plain = folder / "day1" / "IMG_2.JPG"
    plain.write_bytes(b"jpeg")
    # not developed/day1/DSC_1.jpg: its record holds its SHA-256
    assert share.unindexed_images(folder) == [raw, plain]


def test_a_jpeg_put_in_place_of_an_edit_by_hand_is_a_photo(tmp_path: Path) -> None:
    folder, raw, jpg = _shoot(tmp_path)
    first = hashlib.sha256(jpg.read_bytes()).hexdigest()
    jpg.write_bytes(b"the photographer's own export, saved over it")
    assert share.unindexed_images(folder) == [
        raw,
        jpg,
    ]  # the stale record proves nothing
    # an unfinished re-edit leaves the previous output in place, still fdx-develop's
    record = jpg.with_name("DSC_1.develop.json")
    rec = json.loads(record.read_text())
    rec.update(status="pending", output_sha256="", previous_sha256=first)
    record.write_text(json.dumps(rec))
    jpg.write_bytes(b"edited jpeg bytes")  # the bytes `first` was taken from
    assert share.unindexed_images(folder) == [raw]


def test_edit_sends_only_the_shown_raws(tmp_path: Path) -> None:
    folder = tmp_path / "shoot"
    originals = {n: folder / n for n in ("a.NEF", "b.JPG", "c.HEIC", "d.ARW", "e.NEF")}
    gallery = share.Gallery(
        "T",
        sections=[
            share.GallerySection(
                "",
                [
                    share.Item("a.NEF"),
                    share.Item("b.JPG"),  # not a RAW: rendered as it is
                    share.Item("c.HEIC"),
                    share.Item("d.ARW"),
                    share.Item("e.NEF", hidden=True),  # hidden: not sent
                ],
            )
        ],
        pool=[(share.Item("f.NEF"), "over the cap")],  # not picked: not sent
    )
    assert share.edit_picks(gallery, originals) == [folder / "a.NEF", folder / "d.ARW"]


class _Done:
    def __init__(self, code: int, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = code, out, err


def _fake_develop(
    monkeypatch: pytest.MonkeyPatch, *, dry: _Done, real: _Done
) -> list[dict[str, Any]]:
    """fdx-develop 'on PATH': records each call, the --files-from lines it
    received (read while the call runs: the list is a temp file) and the files
    they name when read the way fdx-develop reads them (lines stripped, blank
    and '#' lines skipped, relative to the root)."""
    calls: list[dict[str, Any]] = []

    def run(cmd: list[str], **kw: Any) -> _Done:
        lines = Path(cmd[cmd.index("--files-from") + 1]).read_text().splitlines()
        entries = [ln.strip() for ln in lines]
        named = [Path(cmd[1]) / e for e in entries if e and not e.startswith("#")]
        calls.append({"cmd": cmd, "lines": lines, "files": named})
        return dry if "--dry-run" in cmd else real

    monkeypatch.setattr(
        "framedex.share.shutil.which", lambda name: "/opt/bin/fdx-develop"
    )
    monkeypatch.setattr("framedex.share.subprocess.run", run)
    return calls


def test_fdx_develop_gets_only_the_picks_after_its_estimate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    picks = [tmp_path / "day1" / "a.NEF", tmp_path / "d.ARW"]
    # fdx-develop's dry run: its summary line, then one line per photo to do
    estimate = "9 RAWs, 2 listed, 0 already developed, 2 to do; est. $0.50"
    listed = "  day1/a.NEF -> developed/day1/a.jpg\n  d.ARW -> developed/d.jpg\n"
    calls = _fake_develop(
        monkeypatch, dry=_Done(0, f"{estimate}\n{listed}"), real=_Done(0)
    )
    style = str(tmp_path / "style")
    assert share._run_develop(tmp_path, picks, style, edit_max=60) == 0
    # relative to the shoot, one per line, naming exactly the picks
    assert [c["lines"] for c in calls] == [["./day1/a.NEF", "./d.ARW"]] * 2
    assert [c["files"] for c in calls] == [picks] * 2
    dry, real = (c["cmd"] for c in calls)
    assert dry == [*real, "--dry-run"]
    assert real[:2] == ["/opt/bin/fdx-develop", str(tmp_path)]
    assert real[real.index("--examples") + 1] == style
    # one line (count, cost and the time it'll take), then it runs, unasked
    assert capsys.readouterr().out == f"Editing    {estimate}, ~5 min\n"

    # all edited already: no real run (it would still look for its RAW engine)
    done = "9 RAWs, 2 listed, 2 already developed, 0 to do; est. $0.00"
    calls = _fake_develop(monkeypatch, dry=_Done(0, done + "\n"), real=_Done(1))
    assert share._run_develop(tmp_path, picks, None, edit_max=60) == 0
    assert len(calls) == 1 and capsys.readouterr().out == f"Editing    {done}\n"


def test_odd_file_names_reach_fdx_develop_as_themselves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # unprefixed, '#1.NEF' would be a comment and ' a.NEF' would become 'a.NEF'
    picks = [tmp_path / "#1.NEF", tmp_path / " a.NEF", tmp_path / "day 2" / "b.NEF"]
    calls = _fake_develop(
        monkeypatch, dry=_Done(0, "3 to do; est. $0.75"), real=_Done(0)
    )
    assert share._run_develop(tmp_path, picks, None, edit_max=60) == 0
    assert [c["files"] for c in calls] == [picks] * 2

    for name in ("trailing .NEF ", "line\nbreak.NEF"):  # no line can carry these
        calls = _fake_develop(monkeypatch, dry=_Done(0), real=_Done(0))
        assert share._run_develop(tmp_path, [tmp_path / name], None, edit_max=60) == 1
        assert calls == [] and "rename it or hide it" in capsys.readouterr().err


def test_too_many_picks_are_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_develop(monkeypatch, dry=_Done(0), real=_Done(0))
    picks = [tmp_path / f"{n}.NEF" for n in range(61)]
    assert share._run_develop(tmp_path, picks, None, edit_max=60) == 1
    assert calls == []
    assert "more than --edit-max 60" in capsys.readouterr().err


def test_fdx_develop_failures_are_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    picks = [tmp_path / "a.NEF"]
    _fake_develop(monkeypatch, dry=_Done(2, "", "no RapidRAW engine"), real=_Done(0))
    assert share._run_develop(tmp_path, picks, None, edit_max=60) == 1
    assert "no RapidRAW engine" in capsys.readouterr().err
    calls = _fake_develop(monkeypatch, dry=_Done(0, "1 RAW; est. $0.23"), real=_Done(3))
    assert share._run_develop(tmp_path, picks, None, edit_max=60) == 1
    assert len(calls) == 2 and "exit 3" in capsys.readouterr().err

    def cannot_start(cmd: list[str], **kw: Any) -> _Done:
        raise OSError("bad interpreter")

    monkeypatch.setattr("framedex.share.subprocess.run", cannot_start)
    assert share._run_develop(tmp_path, picks, None, edit_max=60) == 1
    assert capsys.readouterr().err.count("\n") == 1  # no traceback

    def no_space(*a: Any, **kw: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("framedex.share.tempfile.TemporaryDirectory", no_space)
    assert share._run_develop(tmp_path, picks, None, edit_max=60) == 1
    err = capsys.readouterr().err
    assert "No space left" in err and err.count("\n") == 1


def test_edit_without_fdx_develop_says_how_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("framedex.share.shutil.which", lambda name: None)

    def never(*a: Any, **kw: Any) -> Any:
        raise AssertionError("nothing may run")

    monkeypatch.setattr("framedex.share.subprocess.run", never)
    assert share._run_develop(tmp_path, [tmp_path / "a.NEF"], None, edit_max=60) == 1
    err = capsys.readouterr().err
    assert "--edit needs fdx-develop" in err and err.count("\n") == 1  # one clear line
    assert share.main([str(tmp_path), "--examples", "x"]) == 1
    assert share.main([str(tmp_path), "--edit-max", "5"]) == 1
    assert "only apply with --edit" in capsys.readouterr().err


# Stands in for fdx-develop: logs what it was given, prints an estimate on
# --dry-run, else writes an edit plus its record for every listed RAW.
FAKE_DEVELOP = """
import hashlib, json, os, sys
from pathlib import Path

args = sys.argv[1:]
root = Path(args[0])
files = Path(args[args.index("--files-from") + 1]).read_text().splitlines()
with open(os.environ["FAKE_DEVELOP_LOG"], "a") as log:
    log.write(json.dumps({"args": args, "files": files}) + "\\n")
if "--dry-run" in args:
    print(f"5 RAWs, {len(files)} listed, 0 already developed, {len(files)} to do; est. $0.50")
    for rel in files:
        print(f"  {rel} -> developed/{rel}.jpg")
    sys.exit(0)
for rel in files:
    raw = root / rel
    out = root / "developed" / Path(rel).parent
    out.mkdir(parents=True, exist_ok=True)
    jpg = out / (raw.stem + ".jpg")
    jpg.write_bytes(b"edit of " + raw.name.encode())
    st = raw.stat()
    (out / (raw.stem + ".develop.json")).write_text(json.dumps({
        "status": "done",
        "created_at": "2026-10-21T04:00:00+00:00",
        "source": {"name": raw.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns},
        "output": jpg.name,
        "output_sha256": hashlib.sha256(jpg.read_bytes()).hexdigest(),
    }))
"""


class _Rendered(Exception):
    pass


@pytest.mark.skipif(os.name == "nt", reason="the fake fdx-develop is a sh script")
def test_edit_runs_fdx_develop_on_the_picks_and_renders_their_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from framedex import share_site

    folder = tmp_path / "shoot"
    for rel in (
        "day1/a.NEF",
        "day1/b.JPG",
        "day1/c.ARW",
        "day2/f.ARW",
        "d.NEF",
        "e.NEF",
    ):
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(f"sensor data of {rel}".encode())
    gallery = share.Gallery(
        "T",
        sections=[
            share.GallerySection(
                "Day 1",
                [
                    share.Item("day1/a.NEF"),
                    share.Item("day1/b.JPG"),  # a JPEG: rendered as it is
                    share.Item("day1/c.ARW", hidden=True),  # hidden: never edited
                ],
            ),
            share.GallerySection("Day 2", [share.Item("day2/f.ARW")]),
        ],
        pool=[(share.Item("d.NEF"), "near-duplicate of day1/a.NEF")],  # not picked
    )  # e.NEF was never picked at all
    work = folder / share.SHARE_DIR / share.slugify(folder.name)
    work.mkdir(parents=True)
    share.write_selection_file(work / share.SELECTION_FILE, gallery)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "fake_develop.py").write_text(FAKE_DEVELOP)
    exe = bin_dir / "fdx-develop"
    exe.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" "{bin_dir / "fake_develop.py"}" "$@"\n'
    )
    exe.chmod(0o755)
    log = tmp_path / "develop.log"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DEVELOP_LOG", str(log))

    rendered: dict[str, Path] = {}

    def render_site(gallery: Any, sources: dict[str, Path], *a: Any, **kw: Any) -> Any:
        rendered.update(sources)
        raise _Rendered

    monkeypatch.setattr(share_site, "render_site", render_site)

    assert share.main([str(folder), "--edit", "--edit-max", "1", "--no-open"]) == 1
    assert "would send 2 RAWs" in capsys.readouterr().err
    assert not log.exists() and not rendered  # refused before anything ran

    with pytest.raises(_Rendered):
        share.main([str(folder), "--edit", "--no-open"])
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c["files"] for c in calls] == [["./day1/a.NEF", "./day2/f.ARW"]] * 2
    assert ["--dry-run" in c["args"] for c in calls] == [True, False]
    out = capsys.readouterr().out
    est = "5 RAWs, 2 listed, 0 already developed, 2 to do; est. $0.50, ~5 min"
    assert f"Editing    {est}\n" in out
    assert "Edited     2 of 4 photos use their fdx-develop copy" in out
    assert rendered == {
        "day1/a.NEF": folder / "developed/day1/a.jpg",
        "day1/b.JPG": folder / "day1/b.JPG",
        "day1/c.ARW": folder
        / "day1/c.ARW",  # restored later, it renders from its original
        "day2/f.ARW": folder / "developed/day2/f.jpg",
    }


def test_the_web_copy_is_made_from_the_edit(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")
    from framedex import share_site

    folder = tmp_path / "shoot"
    folder.mkdir()
    original = folder / "a.NEF"  # fdx-develop only edits RAWs; Pillow never opens this
    original.write_bytes(b"raw sensor data")
    before = original.read_bytes()
    out = folder / "developed"
    out.mkdir()
    edited = out / "a.jpg"
    pil.new("RGB", (60, 40), (30, 30, 220)).save(edited, "JPEG")  # blue
    (out / "a.develop.json").write_text(json.dumps(_record(original, edited)))

    source = share.developed_copy(original, folder)
    assert source == edited
    gallery = share.Gallery(
        "T", sections=[share.GallerySection("", [share.Item("a.NEF")])]
    )
    site = share_site.render_site(gallery, {"a.NEF": source}, tmp_path / "work")
    r = site.photos["a.NEF"]
    red, _green, blue = (int(r.color[i : i + 2], 16) for i in (1, 3, 5))
    assert blue > red + 100  # the edit's pixels
    assert r.name != share_site.render_name(
        "a.NEF", original, False
    )  # swapping re-renders
    assert original.read_bytes() == before  # the original is untouched
