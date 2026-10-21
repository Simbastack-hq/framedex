"""Tests for fdx share's use of fdx-develop's edited copies — a file
convention, not an import: <root>/developed/<dir>/<name>.jpg plus a
<name>.develop.json record (status, source fingerprint, output SHA-256),
<name> chosen by fdx-develop's own collision rule.

Synthetic files in tmp_path only; fdx-develop itself is never run (its
command is mocked).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from framedex import share


def _record(raw: Path, jpg: Path, status: str = "done") -> dict[str, Any]:
    st = raw.stat()
    return {
        "status": status,
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
    if (
        naming == "file name"
    ):  # another RAW shares the stem: fdx-develop uses file names
        (raw.parent / "dsc_1.ARW").write_bytes(b"other raw")
    name = raw.stem if naming == "stem" else raw.name
    out = folder / "developed" / "day1"
    out.mkdir(parents=True)
    jpg = out / f"{name}.jpg"
    jpg.write_bytes(b"edited jpeg bytes")
    (out / f"{name}.develop.json").write_text(json.dumps(_record(raw, jpg, status)))
    return folder, raw, jpg


def test_develop_names_mirror_fdx_develop() -> None:
    assert share.develop_names(["A.NEF"]) == {"A.NEF": "A"}
    assert share.develop_names(["A.NEF", "a.dng"]) == {
        "A.NEF": "A.NEF",
        "a.dng": "a.dng",
    }
    assert share.develop_names(["A.NEF", "A.DNG", "A.NEF.DNG"]) == {
        "A.NEF": "A.NEF",
        "A.DNG": "A.DNG",
        "A.NEF.DNG": "A.NEF.DNG",  # its stem clashed with A.NEF's file name
    }


@pytest.mark.parametrize("naming", ["stem", "file name"])
def test_a_finished_edit_of_this_exact_original_is_used(
    tmp_path: Path, naming: str
) -> None:
    folder, raw, jpg = _shoot(tmp_path, naming=naming)
    assert share.developed_copy(raw, folder) == jpg
    assert share.is_develop_output(jpg)
    assert not share.is_develop_output(raw)


def test_an_edit_under_a_superseded_name_is_ignored(tmp_path: Path) -> None:
    folder, raw, _ = _shoot(tmp_path)  # DSC_1.jpg, valid on its own
    (raw.parent / "DSC_1.ARW").write_bytes(
        b"a second body's RAW"
    )  # now the stems clash
    assert (
        share.developed_copy(raw, folder) is None
    )  # fdx-develop writes DSC_1.NEF.jpg now


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
    assert share.unindexed_images(folder) == [
        raw,
        plain,
    ]  # not developed/day1/DSC_1.jpg


class _Done:
    def __init__(self, code: int) -> None:
        self.returncode = code


def test_edit_runs_fdx_develop_with_the_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def ok(cmd: list[str]) -> _Done:
        calls.append(cmd)
        return _Done(0)

    monkeypatch.setattr("framedex.share.shutil.which", lambda name: "/bin/fdx-develop")
    monkeypatch.setattr("framedex.share.subprocess.run", ok)
    assert share._run_develop(tmp_path, str(tmp_path / "style")) == 0
    assert calls == [
        ["/bin/fdx-develop", str(tmp_path), "--examples", str(tmp_path / "style")]
    ]

    monkeypatch.setattr("framedex.share.subprocess.run", lambda cmd: _Done(3))
    assert share._run_develop(tmp_path, None) == 1

    def cannot_start(cmd: list[str]) -> _Done:
        raise OSError("bad interpreter")

    monkeypatch.setattr("framedex.share.subprocess.run", cannot_start)
    assert share._run_develop(tmp_path, None) == 1  # one error line, no traceback


def test_edit_without_fdx_develop_says_how_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("framedex.share.shutil.which", lambda name: None)

    def never(*a: Any, **kw: Any) -> Any:
        raise AssertionError("nothing may run")

    monkeypatch.setattr("framedex.share.subprocess.run", never)
    assert share.main([str(tmp_path), "--edit"]) == 1
    err = capsys.readouterr().err
    assert "--edit needs fdx-develop" in err and err.count("\n") == 1  # one clear line
    assert share.main([str(tmp_path), "--examples", "x"]) == 1
    assert "--examples only applies with --edit" in capsys.readouterr().err


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
