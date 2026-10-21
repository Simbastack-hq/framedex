"""Tests for fdx share's use of fdx develop's edited copies — a file
convention, not an import: <root>/developed/<dir>/<stem>.jpg plus a
.develop.json record (status, source fingerprint, output SHA-256).

Synthetic files in tmp_path only; fdx develop itself is never run (its
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


def _shoot(
    tmp_path: Path, *, naming: str = "stem", status: str = "done"
) -> tuple[Path, Path, Path]:
    folder = tmp_path / "shoot"
    raw = folder / "day1" / "DSC_1.NEF"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw sensor data")
    name = raw.stem if naming == "stem" else raw.name
    out = folder / "developed" / "day1"
    out.mkdir(parents=True)
    jpg = out / f"{name}.jpg"
    jpg.write_bytes(b"edited jpeg bytes")
    st = raw.stat()
    record = {
        "status": status,
        "source": {"name": raw.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns},
        "output": jpg.name,
        "output_sha256": hashlib.sha256(jpg.read_bytes()).hexdigest(),
    }
    (out / f"{name}.develop.json").write_text(json.dumps(record))
    return folder, raw, jpg


@pytest.mark.parametrize("naming", ["stem", "file name"])
def test_a_finished_edit_of_this_exact_original_is_used(
    tmp_path: Path, naming: str
) -> None:
    folder, raw, jpg = _shoot(tmp_path, naming=naming)
    assert share.developed_copy(raw, folder) == jpg
    assert share.is_develop_output(jpg)
    assert not share.is_develop_output(raw)


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


def test_edit_runs_fdx_develop_with_the_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    class Done:
        def __init__(self, code: int) -> None:
            self.returncode = code

    monkeypatch.setattr("framedex.share.shutil.which", lambda name: "/bin/fdx-develop")

    def ok(cmd: list[str]) -> Done:
        calls.append(cmd)
        return Done(0)

    monkeypatch.setattr("framedex.share.subprocess.run", ok)
    assert share._run_develop(tmp_path, str(tmp_path / "style")) == 0
    assert calls == [
        ["/bin/fdx-develop", str(tmp_path), "--examples", str(tmp_path / "style")]
    ]

    monkeypatch.setattr("framedex.share.subprocess.run", lambda cmd: Done(3))
    assert share._run_develop(tmp_path, None) == 1


def test_edit_without_fdx_develop_says_how_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("framedex.share.shutil.which", lambda name: None)

    def never(*a: Any, **kw: Any) -> Any:
        raise AssertionError("nothing may run")

    monkeypatch.setattr("framedex.share.subprocess.run", never)
    assert share.main([str(tmp_path), "--edit"]) == 1
    err = capsys.readouterr().err
    assert "--edit needs fdx develop" in err and err.count("\n") == 1  # one clear line
    assert share.main([str(tmp_path), "--examples", "x"]) == 1
    assert "--examples only applies with --edit" in capsys.readouterr().err


def test_the_web_copy_is_made_from_the_edit(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")
    from framedex import share_site

    folder = tmp_path / "shoot"
    folder.mkdir()
    original = folder / "a.JPG"
    pil.new("RGB", (60, 40), (220, 30, 30)).save(original, "JPEG")  # red
    out = folder / "developed"
    out.mkdir()
    edited = out / "a.jpg"
    pil.new("RGB", (60, 40), (30, 30, 220)).save(edited, "JPEG")  # blue
    st = original.stat()
    (out / "a.develop.json").write_text(
        json.dumps(
            {
                "status": "done",
                "source": {
                    "name": "a.JPG",
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                },
                "output_sha256": hashlib.sha256(edited.read_bytes()).hexdigest(),
            }
        )
    )
    before = original.read_bytes()
    gallery = share.Gallery(
        "T", sections=[share.GallerySection("", [share.Item("a.JPG")])]
    )
    source = share.developed_copy(original, folder)
    assert source == edited
    site = share_site.render_site(gallery, {"a.JPG": source}, tmp_path / "work")
    r = site.photos["a.JPG"]
    red, _green, blue = (int(r.color[i : i + 2], 16) for i in (1, 3, 5))
    assert blue > red  # the edit, not the original
    plain = share_site.render_name("a.JPG", original, False)
    assert r.name != plain  # switching to (or away from) the edit re-renders
    assert original.read_bytes() == before  # the original is untouched


def test_edited_copies_are_not_photos_waiting_to_be_indexed(tmp_path: Path) -> None:
    folder, raw, _ = _shoot(tmp_path)
    plain = folder / "day1" / "IMG_2.JPG"
    plain.write_bytes(b"jpeg")
    assert share.unindexed_images(folder) == [raw, plain]  # not developed/DSC_1.jpg
