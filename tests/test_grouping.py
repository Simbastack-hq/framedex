"""Tests for framedex.grouping — burst chaining, RAW/JPEG pairing, the
representative pick, group-aware resume, and the batched exiftool reader.

Pure logic over (path, timestamp, camera) tuples; exiftool is mocked.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from framedex import grouping as g

BASE = "2024:08:14 07:23:11"


def _meta(ts: float | None, camera: str = "NIKON|Z8") -> g.GroupMeta:
    return g.GroupMeta(timestamp=ts, camera=camera)


def _units(d: Path, *names: str) -> list[g.Unit]:
    return [g.Unit(d / n) for n in names]


def _base() -> float:
    t = g.parse_exif_timestamp(BASE, None, None)
    assert t is not None
    return t


# --- timestamps ------------------------------------------------------------


def test_parse_exif_timestamp_prefers_subsec_datetime_and_strips_tz() -> None:
    t = g.parse_exif_timestamp(BASE, f"{BASE}.25+03:00", "99")
    assert t is not None
    assert t - _base() == pytest.approx(0.25, abs=1e-6)
    z = g.parse_exif_timestamp(None, f"{BASE}Z", None)
    assert z is not None
    assert z - _base() == pytest.approx(0.0, abs=1e-6)


def test_parse_exif_timestamp_keeps_leading_zero_of_subsec() -> None:
    t = g.parse_exif_timestamp(BASE, None, "05")
    assert t is not None
    # "05" is five hundredths, not five tenths.
    assert t - _base() == pytest.approx(0.05, abs=1e-6)


def test_parse_exif_timestamp_falls_back_past_a_malformed_subsec_datetime() -> None:
    t = g.parse_exif_timestamp(BASE, "not a date", "25")
    assert t is not None
    assert t - _base() == pytest.approx(0.25, abs=1e-6)


def test_parse_exif_timestamp_accepts_fractional_datetime_original() -> None:
    t = g.parse_exif_timestamp(f"{BASE}.5", None, "99")  # existing fraction wins
    assert t is not None
    assert t - _base() == pytest.approx(0.5, abs=1e-6)


def test_parse_exif_timestamp_malformed_fraction_rejects_that_source() -> None:
    # `.bad` must not silently round to whole seconds: the next source is tried.
    t = g.parse_exif_timestamp(BASE, f"{BASE}.bad", "25")
    assert t is not None
    assert t - _base() == pytest.approx(0.25, abs=1e-6)
    assert g.parse_exif_timestamp(f"{BASE}.bad", None, "25") is None
    # Non-ASCII digits are not a fraction; the whole-second value stands.
    t2 = g.parse_exif_timestamp(BASE, None, "\uff12\uff15")
    assert t2 is not None
    assert t2 - _base() == pytest.approx(0.0, abs=1e-6)


def test_parse_exif_timestamp_none_when_unusable() -> None:
    assert g.parse_exif_timestamp(None, None, None) is None
    assert g.parse_exif_timestamp("0000:00:00 00:00:00", None, None) is None
    assert g.parse_exif_timestamp("garbage", "garbage", "1") is None
    assert g.parse_exif_timestamp("", "", "") is None


# --- pairing ---------------------------------------------------------------


def test_pair_raw_jpeg_matrix(tmp_path: Path) -> None:
    d = tmp_path
    paths = [
        d / "A.RAF",
        d / "a.jpg",  # case-insensitive stem + ext → pair
        d / "B.NEF",  # RAW only
        d / "C.jpg",  # JPEG only
        d / "D.CR3",
        d / "D.jpg",
        d / "D.JPEG",  # RAW + two JPEGs → pair with first by sorted path
        d / "E.NEF",
        d / "E.CR2",
        d / "E.jpg",  # two RAWs share a stem → ambiguous, nobody pairs
        d / "F.ARW",
        d / "F.png",  # PNG never pairs
    ]
    units = {u.primary.name: u for u in g.pair_raw_jpeg(paths)}
    assert units["A.RAF"].sibling == d / "a.jpg" and units["A.RAF"].chainable
    assert units["B.NEF"].sibling is None and units["B.NEF"].chainable
    assert units["C.jpg"].sibling is None and units["C.jpg"].chainable
    assert units["D.CR3"].sibling == d / "D.JPEG"
    # The leftover JPEG is not another capture: never chains into a burst.
    assert units["D.jpg"].sibling is None and not units["D.jpg"].chainable
    for n in ("E.NEF", "E.CR2", "E.jpg"):
        assert units[n].sibling is None and not units[n].chainable
    assert units["F.ARW"].sibling is None and units["F.ARW"].chainable
    assert units["F.png"].sibling is None
    # Every input path appears exactly once across all units.
    assert sorted(f for u in units.values() for f in u.files) == sorted(paths)


def test_pair_raw_jpeg_never_crosses_directories(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    units = g.pair_raw_jpeg([tmp_path / "A.RAF", tmp_path / "sub" / "A.jpg"])
    assert all(u.sibling is None for u in units)


def test_unit_preview_source_is_jpeg_when_paired(tmp_path: Path) -> None:
    assert g.Unit(tmp_path / "1.NEF", tmp_path / "1.jpg").preview_source == (
        tmp_path / "1.jpg"
    )
    assert g.Unit(tmp_path / "1.NEF").preview_source == tmp_path / "1.NEF"


# --- bursts ----------------------------------------------------------------


def test_detect_bursts_chains_at_gap_edge_and_min_size(tmp_path: Path) -> None:
    d = tmp_path
    units = _units(d, "1.NEF", "2.NEF", "3.NEF", "4.NEF", "5.NEF")
    meta = {
        d / "1.NEF": _meta(0.0),
        d / "2.NEF": _meta(2.0),  # gap exactly BURST_GAP_SEC → chains
        d / "3.NEF": _meta(4.0),
        d / "4.NEF": _meta(6.001),  # 2.001 → splits
        d / "5.NEF": _meta(8.0),
    }
    bursts, rest = g.detect_bursts(units, meta)
    assert [[f.name for f in b.files] for b in bursts] == [["1.NEF", "2.NEF", "3.NEF"]]
    # A 2-frame chain is below BURST_MIN_SIZE: not a burst.
    assert {u.primary.name for u in rest} == {"4.NEF", "5.NEF"}
    assert bursts[0].kind == "burst"
    assert bursts[0].primary is None  # not picked yet


def test_detect_bursts_orders_members_by_timestamp(tmp_path: Path) -> None:
    d = tmp_path
    units = _units(d, "c.NEF", "a.NEF", "b.NEF")
    meta = {d / "c.NEF": _meta(2.0), d / "a.NEF": _meta(0.0), d / "b.NEF": _meta(1.0)}
    bursts, _ = g.detect_bursts(units, meta)
    assert [u.primary.name for u in bursts[0].units] == ["a.NEF", "b.NEF", "c.NEF"]


def test_detect_bursts_never_groups_undated_unknown_camera_mixed_or_other_dir(
    tmp_path: Path,
) -> None:
    d = tmp_path
    sub = d / "sub"
    sub.mkdir()
    names = ["1.NEF", "2.NEF", "3.NEF", "x.NEF", "p1.RAF", "p2.RAF", "p3.RAF"]
    names += ["u1.jpg", "u2.jpg", "u3.jpg"]
    units = _units(d, *names) + _units(sub, "4.NEF")
    meta = {
        d / "1.NEF": _meta(0.0),
        d / "2.NEF": _meta(1.0),
        d / "3.NEF": _meta(2.0),
        d / "x.NEF": _meta(None),  # no date: indexed individually
        # Another camera, interleaved in time: its own chain, never merged.
        d / "p1.RAF": _meta(0.5, "FUJI|X-T5"),
        d / "p2.RAF": _meta(1.5, "FUJI|X-T5"),
        d / "p3.RAF": _meta(2.5, "FUJI|X-T5"),
        # Unknown camera: "same unknown" is a guess, so never a burst.
        d / "u1.jpg": _meta(10.0, ""),
        d / "u2.jpg": _meta(10.5, ""),
        d / "u3.jpg": _meta(11.0, ""),
        sub / "4.NEF": _meta(3.0),  # other directory: never joins
    }
    bursts, rest = g.detect_bursts(units, meta)
    members = sorted(sorted(f.name for f in b.files) for b in bursts)
    assert members == [["1.NEF", "2.NEF", "3.NEF"], ["p1.RAF", "p2.RAF", "p3.RAF"]]
    assert {u.primary.name for u in rest} == {
        "x.NEF",
        "4.NEF",
        "u1.jpg",
        "u2.jpg",
        "u3.jpg",
    }


def test_detect_bursts_unit_without_meta_entry_is_undated(tmp_path: Path) -> None:
    d = tmp_path
    bursts, rest = g.detect_bursts(_units(d, "1.NEF", "2.NEF", "3.NEF"), {})
    assert bursts == []
    assert len(rest) == 3


# --- build_groups ----------------------------------------------------------


def test_build_groups_burst_of_pairs_and_lone_pair(tmp_path: Path) -> None:
    d = tmp_path
    paths = [d / f"{n}.{e}" for n in ("1", "2", "3", "9") for e in ("NEF", "jpg")]
    paths.append(d / "s.NEF")
    ts = {"1": 0.0, "2": 1.0, "3": 2.0, "9": 100.0}
    meta = {p: _meta(ts[p.stem]) for p in paths if p.stem in ts}
    meta[d / "s.NEF"] = _meta(None)
    groups, singles = g.build_groups(paths, meta)
    kinds = sorted((grp.kind, sorted(f.name for f in grp.files)) for grp in groups)
    assert kinds == [
        ("burst", ["1.NEF", "1.jpg", "2.NEF", "2.jpg", "3.NEF", "3.jpg"]),
        ("raw_jpeg", ["9.NEF", "9.jpg"]),
    ]
    assert singles == [d / "s.NEF"]
    burst = next(grp for grp in groups if grp.kind == "burst")
    # Pairs collapse to the RAW before chaining; the JPEG is the preview source.
    assert [u.sibling for u in burst.units] == [d / "1.jpg", d / "2.jpg", d / "3.jpg"]


def test_build_groups_pairing_leftovers_are_not_captures(tmp_path: Path) -> None:
    """Two real captures plus a RAW's second JPEG must not make a 3-unit
    burst; nor may an ambiguous same-stem bucket chain."""
    d = tmp_path
    paths = [d / "1.NEF", d / "1.jpg", d / "1.JPEG", d / "2.NEF"]
    meta = {p: _meta(0.0 if p.stem == "1" else 1.0) for p in paths}
    groups, singles = g.build_groups(paths, meta)
    assert [grp.kind for grp in groups] == [
        "raw_jpeg"
    ]  # 1.NEF + 1.JPEG (first by sorted path)
    assert sorted(p.name for p in singles) == ["1.jpg", "2.NEF"]

    amb = [d / "e.NEF", d / "e.CR2", d / "e.jpg", d / "f.NEF"]
    meta = {p: _meta(0.5 if p.stem == "e" else 1.0) for p in amb}
    groups, singles = g.build_groups(amb, meta)
    assert groups == []
    assert sorted(p.name for p in singles) == ["e.CR2", "e.NEF", "e.jpg", "f.NEF"]


def test_build_groups_every_path_exactly_once(tmp_path: Path) -> None:
    d = tmp_path
    paths = [d / f"{i}.NEF" for i in range(6)] + [d / "0.jpg", d / "z.png"]
    meta = {d / f"{i}.NEF": _meta(float(i) * 0.5) for i in range(6)}
    groups, singles = g.build_groups(paths, meta)
    seen = sorted([f for grp in groups for f in grp.files] + singles)
    assert seen == sorted(paths)


def test_group_id_is_order_independent_and_root_independent(tmp_path: Path) -> None:
    a, b = tmp_path / "x" / "1.NEF", tmp_path / "x" / "2.NEF"
    gid = g.group_id([a, b])
    assert gid == g.group_id([b, a])
    assert gid.startswith("b_") and len(gid) == 10
    # Same member names in another folder → the same id (ids hash names, so a
    # changed scan root never changes them; fdx-master scopes by directory).
    assert gid == g.group_id([tmp_path / "y" / "1.NEF", tmp_path / "y" / "2.NEF"])
    assert gid != g.group_id([a, b, tmp_path / "x" / "3.NEF"])


# --- representative --------------------------------------------------------


def test_pick_representative_highest_sharpness_tie_to_earliest(tmp_path: Path) -> None:
    d = tmp_path
    grp = g.MediaGroup(
        kind="burst", id="b_0", units=_units(d, "1.NEF", "2.NEF", "3.NEF")
    )
    g.pick_representative(
        grp, {d / "1.NEF": 10.0, d / "2.NEF": 50.0, d / "3.NEF": 50.0}
    )
    assert grp.primary == d / "2.NEF"
    assert grp.sharpness == {d / "1.NEF": 10.0, d / "2.NEF": 50.0, d / "3.NEF": 50.0}


def test_pick_representative_all_unrenderable_picks_earliest(tmp_path: Path) -> None:
    d = tmp_path
    grp = g.MediaGroup(
        kind="burst", id="b_0", units=_units(d, "1.NEF", "2.NEF", "3.NEF")
    )
    g.pick_representative(grp, dict.fromkeys(grp.files, -1.0))
    assert grp.primary == d / "1.NEF"


# --- resume ----------------------------------------------------------------


def _reader(fms: dict[str, dict[str, Any] | None]) -> Any:
    """Fake sidecar reader keyed by the media file name."""

    def read(sidecar: Path) -> dict[str, Any] | None:
        media = sidecar.name[: -len(".description.md")]
        return fms.get(media)

    return read


def test_group_is_done_matrix(tmp_path: Path) -> None:
    d = tmp_path
    files = [d / "1.NEF", d / "2.NEF", d / "3.NEF"]
    grp = g.MediaGroup("burst", g.group_id(files), [g.Unit(f) for f in files])
    mine = {"id": grp.id, "primary": False}
    # Done: every member carries this group's id.
    assert g.group_is_done(
        grp,
        _reader(
            {
                "1.NEF": {"group": mine},
                "2.NEF": {"group": {"id": grp.id, "primary": True}},
                "3.NEF": {"group": mine},
            }
        ),
    )
    # Done: fully legacy (indexed per-file before grouping existed).
    assert g.group_is_done(grp, _reader({"1.NEF": {}, "2.NEF": {}, "3.NEF": {}}))
    # Redo: a member sidecar is missing (interrupted between stubs and primary).
    assert not g.group_is_done(
        grp, _reader({"1.NEF": {"group": mine}, "3.NEF": {"group": mine}})
    )
    # Redo: a block from a different grouping (membership changed).
    other = {"id": "b_deadbeef", "primary": False}
    assert not g.group_is_done(
        grp,
        _reader(
            {
                "1.NEF": {"group": mine},
                "2.NEF": {"group": other},
                "3.NEF": {"group": mine},
            }
        ),
    )
    # Redo: mixed legacy + grouped (a crash under --force before the primary).
    assert not g.group_is_done(
        grp, _reader({"1.NEF": {"group": mine}, "2.NEF": {}, "3.NEF": {"group": mine}})
    )
    # Redo: unparsable sidecar.
    assert not g.group_is_done(grp, _reader({"1.NEF": None, "2.NEF": {}, "3.NEF": {}}))


def test_single_is_done_rejects_group_leftovers(tmp_path: Path) -> None:
    p = tmp_path / "1.NEF"
    assert g.single_is_done(p, _reader({"1.NEF": {"rating": "keep"}}))
    assert not g.single_is_done(p, _reader({}))  # missing
    # A stub or an ex-primary is a leftover of an earlier grouping: re-assess.
    assert not g.single_is_done(p, _reader({"1.NEF": {"group": {"primary": False}}}))
    assert not g.single_is_done(p, _reader({"1.NEF": {"group": {"primary": True}}}))


# --- exiftool reader -------------------------------------------------------


def _run_ok(payload: list[dict[str, Any]], rc: int = 0) -> Any:
    def run(cmd: list[str], **kw: Any) -> Any:
        run.calls.append((cmd, kw))  # type: ignore[attr-defined]
        return types.SimpleNamespace(
            returncode=rc, stdout=json.dumps(payload), stderr=""
        )

    run.calls = []  # type: ignore[attr-defined]
    return run


def test_read_group_metadata_batches_one_exiftool_call_on_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = tmp_path / "a.NEF", tmp_path / "b.NEF"
    run = _run_ok(
        [
            {
                "SourceFile": str(a),
                "DateTimeOriginal": BASE,
                "SubSecTimeOriginal": 25,  # exiftool may type this as a number
                "Make": "NIKON CORPORATION",
                "Model": "NIKON Z 8",
            },
            {"SourceFile": str(b), "Make": "NIKON CORPORATION", "Model": "NIKON Z 8"},
        ],
        rc=1,  # exiftool exits 1 when any file failed…
    )
    monkeypatch.setattr("framedex.grouping.subprocess.run", run)
    meta = g.read_group_metadata([a, b])
    ((cmd, kw),) = run.calls
    assert cmd[0] == "exiftool" and "-json" in cmd
    assert cmd[-4:] == ["-charset", "filename=utf8", "-@", "-"]
    assert kw["input"] == f"{a}\n{b}\n" and kw["encoding"] == "utf-8"
    assert meta[a].camera == "NIKON CORPORATION|NIKON Z 8"
    ts = meta[a].timestamp
    assert ts is not None
    assert ts - _base() == pytest.approx(0.25, abs=1e-6)
    assert meta[b].timestamp is None  # …but the partial JSON is still used


def test_read_group_metadata_refuses_newline_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A newline in a name would let a filename smuggle an exiftool option
    (even a write) into the stdin argument stream. Such paths never reach
    exiftool; they index individually through the argv-based per-file reads."""
    ok = tmp_path / "ok.NEF"
    evil = tmp_path / "evil\n-overwrite_original.NEF"
    run = _run_ok([{"SourceFile": str(ok)}])
    monkeypatch.setattr("framedex.grouping.subprocess.run", run)
    meta = g.read_group_metadata([ok, evil])
    ((_cmd, kw),) = run.calls
    assert kw["input"] == f"{ok}\n"
    assert "overwrite_original" not in kw["input"]
    assert evil not in meta
    assert "newline" in capsys.readouterr().err


def test_read_group_metadata_warns_about_files_missing_from_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = tmp_path / "a.NEF", tmp_path / "b.NEF"
    monkeypatch.setattr(
        "framedex.grouping.subprocess.run", _run_ok([{"SourceFile": str(a)}])
    )
    meta = g.read_group_metadata([a, b])
    assert set(meta) == {a}
    assert (
        "no EXIF for 1 of 2 files; indexing them individually"
        in capsys.readouterr().err
    )


def test_read_group_metadata_raises_on_batch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed batch must not silently become N individual vision calls."""
    monkeypatch.setattr(
        "framedex.grouping.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="not json", stderr=""
        ),
    )
    with pytest.raises(g.GroupMetadataError, match="exiftool batch read failed"):
        g.read_group_metadata([tmp_path / "a.NEF"])

    def boom(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("exiftool")

    monkeypatch.setattr("framedex.grouping.subprocess.run", boom)
    with pytest.raises(g.GroupMetadataError, match="exiftool executable not found"):
        g.read_group_metadata([tmp_path / "a.NEF"])


def test_read_group_metadata_empty_input_skips_exiftool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "framedex.grouping.subprocess.run", lambda *a, **k: pytest.fail("must not run")
    )
    assert g.read_group_metadata([]) == {}


def test_read_group_metadata_raises_when_nothing_succeeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid JSON that carries no usable entry is a broken read, not a folder
    without EXIF: `[]`, all-`Error` entries, or a signal death must raise."""
    a = tmp_path / "a.NEF"
    for payload in ([], [{"SourceFile": str(a), "Error": "File format error"}]):
        monkeypatch.setattr("framedex.grouping.subprocess.run", _run_ok(payload, rc=1))
        with pytest.raises(g.GroupMetadataError, match="no metadata from any"):
            g.read_group_metadata([a])
    monkeypatch.setattr(
        "framedex.grouping.subprocess.run", _run_ok([{"SourceFile": str(a)}], rc=-9)
    )
    with pytest.raises(g.GroupMetadataError, match="signal 9"):
        g.read_group_metadata([a])


def test_read_group_metadata_error_entries_count_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = tmp_path / "a.NEF", tmp_path / "b.NEF"
    monkeypatch.setattr(
        "framedex.grouping.subprocess.run",
        _run_ok(
            [
                {"SourceFile": str(a)},
                {"SourceFile": str(b), "Error": "Unknown file type"},
            ],
            rc=1,
        ),
    )
    meta = g.read_group_metadata([a, b])
    assert set(meta) == {a}
    assert "no EXIF for 1 of 2 files" in capsys.readouterr().err
