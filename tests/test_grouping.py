"""Tests for framedex.grouping — burst chaining, RAW/JPEG pairing, the
representative pick, and the batched exiftool reader.

Pure logic over (path, timestamp, camera) tuples; exiftool is mocked.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from framedex import grouping as g


def _meta(ts: float | None, camera: str = "NIKON|Z8") -> g.GroupMeta:
    return g.GroupMeta(timestamp=ts, camera=camera)


def _units(d: Path, *names: str) -> list[g.Unit]:
    return [g.Unit(d / n) for n in names]


# --- timestamps ------------------------------------------------------------


def test_parse_exif_timestamp_prefers_subsec_datetime_and_strips_tz() -> None:
    base = g.parse_exif_timestamp("2024:08:14 07:23:11", None, None)
    assert base is not None
    t = g.parse_exif_timestamp(
        "2024:08:14 07:23:11", "2024:08:14 07:23:11.25+03:00", "99"
    )
    assert t == pytest.approx(base + 0.25)
    assert g.parse_exif_timestamp(None, "2024:08:14 07:23:11Z", None) == pytest.approx(
        base
    )


def test_parse_exif_timestamp_keeps_leading_zero_of_subsec() -> None:
    base = g.parse_exif_timestamp("2024:08:14 07:23:11", None, None)
    assert base is not None
    # "05" is five hundredths, not five tenths — the reason the reader has no -n.
    assert g.parse_exif_timestamp("2024:08:14 07:23:11", None, "05") == pytest.approx(
        base + 0.05
    )


def test_parse_exif_timestamp_none_when_unusable() -> None:
    assert g.parse_exif_timestamp(None, None, None) is None
    assert g.parse_exif_timestamp("0000:00:00 00:00:00", None, None) is None
    assert g.parse_exif_timestamp("garbage", None, None) is None
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
    assert units["A.RAF"].sibling == d / "a.jpg"
    assert units["B.NEF"].sibling is None
    assert units["C.jpg"].sibling is None
    assert units["D.CR3"].sibling == d / "D.JPEG"
    assert units["D.jpg"].sibling is None
    assert units["E.NEF"].sibling is None
    assert units["E.CR2"].sibling is None
    assert units["E.jpg"].sibling is None
    assert units["F.ARW"].sibling is None
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
    bursts, rest = g.detect_bursts(units, meta, d)
    assert [[f.name for f in b.files] for b in bursts] == [["1.NEF", "2.NEF", "3.NEF"]]
    # A 2-frame chain is below BURST_MIN_SIZE: not a burst.
    assert {u.primary.name for u in rest} == {"4.NEF", "5.NEF"}
    assert bursts[0].kind == "burst"
    assert bursts[0].primary is None  # not picked yet


def test_detect_bursts_orders_members_by_timestamp(tmp_path: Path) -> None:
    d = tmp_path
    units = _units(d, "c.NEF", "a.NEF", "b.NEF")
    meta = {d / "c.NEF": _meta(2.0), d / "a.NEF": _meta(0.0), d / "b.NEF": _meta(1.0)}
    bursts, _ = g.detect_bursts(units, meta, d)
    assert [u.primary.name for u in bursts[0].units] == ["a.NEF", "b.NEF", "c.NEF"]


def test_detect_bursts_never_groups_missing_dates_mixed_cameras_or_dirs(
    tmp_path: Path,
) -> None:
    d = tmp_path
    sub = d / "sub"
    sub.mkdir()
    units = _units(d, "1.NEF", "2.NEF", "3.NEF", "x.NEF", "p1.RAF", "p2.RAF", "p3.RAF")
    units += _units(sub, "4.NEF")
    meta = {
        d / "1.NEF": _meta(0.0),
        d / "2.NEF": _meta(1.0),
        d / "3.NEF": _meta(2.0),
        d / "x.NEF": _meta(None),  # no date: indexed individually
        # Another camera, interleaved in time: its own chain, never merged.
        d / "p1.RAF": _meta(0.5, "FUJI|X-T5"),
        d / "p2.RAF": _meta(1.5, "FUJI|X-T5"),
        d / "p3.RAF": _meta(2.5, "FUJI|X-T5"),
        sub / "4.NEF": _meta(3.0),  # other directory: never joins
    }
    bursts, rest = g.detect_bursts(units, meta, d)
    members = sorted(sorted(f.name for f in b.files) for b in bursts)
    assert members == [["1.NEF", "2.NEF", "3.NEF"], ["p1.RAF", "p2.RAF", "p3.RAF"]]
    assert {u.primary.name for u in rest} == {"x.NEF", "4.NEF"}


def test_detect_bursts_unit_without_meta_entry_is_undated(tmp_path: Path) -> None:
    d = tmp_path
    bursts, rest = g.detect_bursts(_units(d, "1.NEF", "2.NEF", "3.NEF"), {}, d)
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
    groups, singles = g.build_groups(paths, meta, d)
    kinds = sorted((grp.kind, sorted(f.name for f in grp.files)) for grp in groups)
    assert kinds == [
        ("burst", ["1.NEF", "1.jpg", "2.NEF", "2.jpg", "3.NEF", "3.jpg"]),
        ("raw_jpeg", ["9.NEF", "9.jpg"]),
    ]
    assert singles == [d / "s.NEF"]
    burst = next(grp for grp in groups if grp.kind == "burst")
    # Pairs collapse to the RAW before chaining; the JPEG is the preview source.
    assert [u.sibling for u in burst.units] == [d / "1.jpg", d / "2.jpg", d / "3.jpg"]


def test_build_groups_every_path_exactly_once(tmp_path: Path) -> None:
    d = tmp_path
    paths = [d / f"{i}.NEF" for i in range(6)] + [d / "0.jpg", d / "z.png"]
    meta = {d / f"{i}.NEF": _meta(float(i) * 0.5) for i in range(6)}
    groups, singles = g.build_groups(paths, meta, d)
    seen = sorted([f for grp in groups for f in grp.files] + singles)
    assert seen == sorted(paths)


def test_group_id_is_order_independent_and_root_relative(tmp_path: Path) -> None:
    a, b = tmp_path / "x" / "1.NEF", tmp_path / "x" / "2.NEF"
    gid = g.group_id([a, b], tmp_path)
    assert gid == g.group_id([b, a], tmp_path)
    assert gid.startswith("b_") and len(gid) == 10
    # Same file names in another folder → a different id.
    assert gid != g.group_id(
        [tmp_path / "y" / "1.NEF", tmp_path / "y" / "2.NEF"], tmp_path
    )


# --- representative --------------------------------------------------------


def test_pick_representative_highest_sharpness_tie_to_earliest(tmp_path: Path) -> None:
    d = tmp_path
    grp = g.MediaGroup(
        kind="burst", id="b_0", units=_units(d, "1.NEF", "2.NEF", "3.NEF")
    )
    scores = {"1.NEF": 10.0, "2.NEF": 50.0, "3.NEF": 50.0}
    g.pick_representative(grp, lambda u: scores[u.primary.name])
    assert grp.primary == d / "2.NEF"
    assert grp.sharpness == {d / "1.NEF": 10.0, d / "2.NEF": 50.0, d / "3.NEF": 50.0}


def test_pick_representative_all_unrenderable_picks_earliest(tmp_path: Path) -> None:
    d = tmp_path
    grp = g.MediaGroup(
        kind="burst", id="b_0", units=_units(d, "1.NEF", "2.NEF", "3.NEF")
    )
    g.pick_representative(grp, lambda u: -1.0)
    assert grp.primary == d / "1.NEF"


def test_pick_representative_scores_pair_preview_source(tmp_path: Path) -> None:
    d = tmp_path
    grp = g.MediaGroup(
        kind="raw_jpeg", id="b_1", units=[g.Unit(d / "1.NEF", d / "1.jpg")]
    )
    seen: list[Path] = []

    def score(u: g.Unit) -> float:
        seen.append(u.preview_source)
        return 1.0

    g.pick_representative(grp, score)
    assert seen == [d / "1.jpg"]
    assert grp.primary == d / "1.NEF"


# --- exiftool reader -------------------------------------------------------


def test_read_group_metadata_batches_one_exiftool_call_without_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = tmp_path / "a.NEF", tmp_path / "b.NEF"
    calls: list[list[str]] = []

    def run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        argfile = Path(cmd[cmd.index("-@") + 1])
        assert argfile.read_text().splitlines() == [str(a), str(b)]
        return types.SimpleNamespace(
            returncode=1,  # exiftool exits 1 when any file failed…
            stdout=json.dumps(
                [
                    {
                        "SourceFile": str(a),
                        "DateTimeOriginal": "2024:08:14 07:23:11",
                        "SubSecTimeOriginal": "05",
                        "Make": "NIKON CORPORATION",
                        "Model": "NIKON Z 8",
                    },
                    {
                        "SourceFile": str(b),
                        "Make": "NIKON CORPORATION",
                        "Model": "NIKON Z 8",
                    },
                ]
            ),
            stderr="1 files could not be read",
        )

    monkeypatch.setattr("framedex.grouping.subprocess.run", run)
    meta = g.read_group_metadata([a, b])
    assert len(calls) == 1
    assert "-n" not in calls[0] and "-json" in calls[0] and calls[0][0] == "exiftool"
    assert meta[a].camera == "NIKON CORPORATION|NIKON Z 8"
    assert meta[a].timestamp == pytest.approx(
        g.parse_exif_timestamp("2024:08:14 07:23:11", None, "05")
    )
    assert meta[b].timestamp is None  # …but the partial JSON is still used
    # The argfile is cleaned up.
    assert not Path(calls[0][calls[0].index("-@") + 1]).exists()


def test_read_group_metadata_fails_loud_and_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "framedex.grouping.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="not json", stderr=""
        ),
    )
    assert g.read_group_metadata([tmp_path / "a.NEF"]) == {}
    assert "no burst detection" in capsys.readouterr().err

    def boom(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("exiftool")

    monkeypatch.setattr("framedex.grouping.subprocess.run", boom)
    assert g.read_group_metadata([tmp_path / "a.NEF"]) == {}
    assert "no burst detection" in capsys.readouterr().err


def test_read_group_metadata_empty_input_skips_exiftool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "framedex.grouping.subprocess.run", lambda *a, **k: pytest.fail("must not run")
    )
    assert g.read_group_metadata([]) == {}
