# Phase 3: Burst grouping + RAW/JPEG pairing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One moment = one vision call = one primary sidecar. Burst frames (same camera, same folder, ≤2 s apart, ≥3 frames) and RAW+JPEG pairs are grouped before the per-file loop; only the group's sharpest member (a pair: its RAW) gets the vision call, faces, and a full sidecar; every other member gets a cheap stub sidecar that copies the primary's assessment and points at it.

**Architecture:** A new pure-logic module `grouping.py` (pairing, burst chaining, representative pick, group ids) with one isolated batched `exiftool` reader. `images.process_group` orchestrates a group: renders + scores members locally (Laplacian variance, no model call), then reuses `process_one_image` on the primary with two new hooks (`preview_override`, `before_sidecar`) so stubs are written *before* the primary sidecar (the resume marker stays last). `index_videos.main` gains a group-aware discovery/resume pre-pass; `master_index`, `query`, and `xmp_export` learn to recognise stubs. Folder mode only; `fdx-photos` accepts `--no-group` as a documented no-op.

**Tech Stack:** Python 3.10+, stdlib + cv2 (already a base dep, lazily imported), exiftool (already required). No new dependency. Hermetic tests (mocked subprocess / cv2 / vision).

**Spec:** `docs/superpowers/specs/2026-07-02-trust-xmp-burst-prd.md` §"Phase 3". Handoff with verified code facts: `docs/superpowers/plans/2026-07-02-phase3-burst-grouping-HANDOFF.md`.

---

## Design decisions locked in (deviations from the handoff are marked ⚠)

1. **Resume rule = "every member has a sidecar".** The PRD says "skip a group only when its *primary* has a sidecar". Knowing the primary of a burst requires rendering + scoring every member (expensive on every re-run of a 10k-photo archive). Because stubs are written strictly before the primary, *primary-sidecar-exists ⟺ all-member-sidecars-exist* for anything framedex wrote. The only divergence is a user hand-deleting a stub, which then re-runs the group (one extra vision call, always safe). Also: sidecars that predate grouping (one per file) satisfy the rule, so legacy archives are not re-indexed; `--force` regroups.
2. **⚠ `--no-group` is a discovery-time flag, not a `ProcessOptions` field.** It sits with `--exclude`, `--force`, `--media`, `--max-files`, none of which live in `ProcessOptions` either (that dataclass is per-file processing config). Nothing per-file reads it, so threading it would be a dead field. Both parsers get the flag (repo convention).
3. **`preview_override` + `before_sidecar` on `process_one_image`** instead of splitting it into analyze/persist. Two keyword args, zero behavior change for existing callers (`fdx-photos`). The hook runs after `write_faces` and before `serialize_sidecar`; if it raises, no primary sidecar is written and the whole group is redone next run.
4. **Stubs do NOT copy faces.** `faces: []`, `face_count: 0`, no `faces.db` rows: no detection ran on that frame, and phantom cluster ids would haunt `fdx-faces`. Everything the *model* produced is copied (`STUB_COPIED_FIELDS`). `--person`/`--face-count` therefore match the primary only; alternates are reachable via the group block.
5. **Stub `location.place` is copied from the primary** when the member carries its own GPS (a burst is one place; Nominatim at 1 req/s for 20 identical coordinates is waste). Own lat/lon is kept.
6. **Sharpness helper split as `laplacian_variance(img)` (array in) + `laplacian_sharpness(path)`.** `_signatures` already holds the decoded thumbnail, so calling a path-based helper from it would decode every thumbnail twice. Shared math, no double read.
7. **One render per file.** Members are rendered once for scoring into a per-unit temp subdir; the chosen primary's rendered preview is handed to `process_one_image` via `preview_override` so it is never rendered again.
8. **Exiftool batch reader has no `-n`.** With `-n`, `SubSecTimeOriginal` "05" becomes the number 5 (→ .5 s instead of .05 s). Strings only.
9. **`--max-files N` counts work items** (a group = one item = one vision call), matching the cost model. Dry-run prints `[burst ×12 → 1 call]`.
10. **Group id = `b_` + sha1 of the sorted *root-relative* member paths [:8]** — order-independent, stable across re-runs and across mounts of the same drive, unique across folders with identical filenames.

## Cost model (README)

Vision calls per archive = `#groups + #ungrouped files` ≤ `#files`. Local CPU rises slightly (one preview render + one Laplacian per burst member) and each stub costs two `exiftool` invocations (metadata + GPS), no model call.

## File map

- Create `src/framedex/grouping.py` — `GroupMeta`, `Unit`, `MediaGroup`, `parse_exif_timestamp`, `pair_raw_jpeg`, `detect_bursts`, `build_groups`, `group_id`, `pick_representative`, `read_group_metadata`, constants.
- Modify `src/framedex/frame_sampling.py:129-150` — `laplacian_variance`, `laplacian_sharpness`; `_signatures` uses the former.
- Modify `src/framedex/images.py` — `process_one_image(..., preview_override, before_sidecar)`, `STUB_COPIED_FIELDS`, `build_stub_frontmatter`, `_render_and_score`, `process_group`.
- Modify `src/framedex/pipeline.py:553-565` — `ProcessResult.stubs_written: int = 0`.
- Modify `src/framedex/runner.py:150-212` — `RunTally.groups`, `RunTally.stubs`; `record_result` prints `+N alternates`.
- Modify `src/framedex/index_videos.py:1240-1463` — `--no-group`, grouping pre-pass, group-aware resume, dry-run, loop branch, summary line.
- Modify `src/framedex/photos_indexer.py:252-262` — `--no-group` accepted (documented no-op).
- Modify `src/framedex/master_index.py:120-200` — stats over primaries only, grouped line, JSON counts.
- Modify `src/framedex/query.py` — `--primary-only`.
- Modify `src/framedex/xmp_export.py:103-118` — `burst-pick` / `burst-alternate` tags.
- Create `tests/test_grouping.py`; extend `tests/test_frame_sampling.py`, `tests/test_images.py`, `tests/test_index_videos.py`, `tests/test_photos_indexer.py`, `tests/test_master_index.py`, `tests/test_query.py`, `tests/test_xmp_export.py`.
- Docs: `README.md`, `SKILL.md`, `docs/tuning.md`, `CHANGELOG.md`.

Gates (run exactly as CI): `uv run --no-sync pytest -q`, `uv run --no-sync ruff check src/framedex tests`, `uv run --no-sync ruff format --check src/framedex tests`, `uv run --no-sync mypy src/framedex tests --ignore-missing-imports`.

---

### Task 1: `laplacian_variance` / `laplacian_sharpness` shared helper

**Files:**
- Modify: `src/framedex/frame_sampling.py:129-150`
- Test: `tests/test_frame_sampling.py`

- [ ] **Step 1: Write the failing tests** (CI has no cv2: inject a fake module)

```python
import sys, types

def _fake_cv2(monkeypatch: pytest.MonkeyPatch, imread_result: object = "IMG") -> types.SimpleNamespace:
    class _Lap:
        def __init__(self, v: float) -> None:
            self._v = v
        def var(self) -> float:
            return self._v
    cv2 = types.SimpleNamespace(
        COLOR_BGR2GRAY=6, CV_64F=6,
        imread=lambda p: imread_result,
        cvtColor=lambda img, code: ("gray", img),
        Laplacian=lambda gray, depth: _Lap(42.5),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2

def test_laplacian_variance_returns_laplacian_var_of_grayscale(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_cv2(monkeypatch)
    assert fs.laplacian_variance("IMG") == 42.5

def test_laplacian_sharpness_reads_path_and_fails_loud_on_unreadable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_cv2(monkeypatch)
    assert fs.laplacian_sharpness(tmp_path / "a.jpg") == 42.5
    _fake_cv2(monkeypatch, imread_result=None)
    with pytest.raises(ValueError, match="unreadable"):
        fs.laplacian_sharpness(tmp_path / "missing.jpg")
```

- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_frame_sampling.py -q -k laplacian` → FAIL (`AttributeError: laplacian_variance`).

- [ ] **Step 3: Implement** (in `frame_sampling.py`, above `_signatures`)

```python
def laplacian_variance(img: Any) -> float:
    """Sharpness proxy: variance of the Laplacian over the grayscale image.
    `img` is a decoded BGR array (cv2.imread output). Shared by the video
    frame sampler and the still-photo burst representative pick."""
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def laplacian_sharpness(path: Path) -> float:
    """`laplacian_variance` of the image file at `path`. Raises ValueError on
    an unreadable file rather than scoring it 0 (a silent 0 would quietly
    lose a burst member the pick)."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return laplacian_variance(img)
```

and in `_signatures` replace the two lines
`gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` / `sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))`
with `sharpness.append(laplacian_variance(img))`.

- [ ] **Step 4: Run** the two tests → PASS; full `pytest -q` still green.
- [ ] **Step 5: Commit** `git commit -m "frame_sampling: factor laplacian_variance/laplacian_sharpness out of _signatures"`

---

### Task 2: `grouping.py` — pure grouping logic

**Files:**
- Create: `src/framedex/grouping.py`
- Test: `tests/test_grouping.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for framedex.grouping — pure logic over (path, timestamp, camera)."""
from __future__ import annotations
import json, types
from pathlib import Path
from typing import Any
import pytest
from framedex import grouping as g

def _meta(ts: float | None, camera: str = "NIKON|Z8") -> g.GroupMeta:
    return g.GroupMeta(timestamp=ts, camera=camera)

# --- timestamps ---
def test_parse_exif_timestamp_prefers_subsec_datetime_and_strips_tz() -> None:
    t = g.parse_exif_timestamp("2024:08:14 07:23:11", "2024:08:14 07:23:11.25+03:00", "99")
    assert t == pytest.approx(g.parse_exif_timestamp("2024:08:14 07:23:11", None, None) + 0.25)

def test_parse_exif_timestamp_keeps_leading_zero_of_subsec() -> None:
    base = g.parse_exif_timestamp("2024:08:14 07:23:11", None, None)
    assert g.parse_exif_timestamp("2024:08:14 07:23:11", None, "05") == pytest.approx(base + 0.05)

def test_parse_exif_timestamp_none_when_unusable() -> None:
    assert g.parse_exif_timestamp(None, None, None) is None
    assert g.parse_exif_timestamp("0000:00:00 00:00:00", None, None) is None
    assert g.parse_exif_timestamp("garbage", None, None) is None

# --- pairing ---
def test_pair_raw_jpeg_matrix(tmp_path: Path) -> None:
    d = tmp_path
    paths = [d/"A.RAF", d/"a.jpg", d/"B.NEF", d/"C.jpg", d/"D.CR3", d/"D.jpg", d/"D.JPEG",
             d/"E.NEF", d/"E.CR2", d/"E.jpg", d/"F.ARW", d/"F.png"]
    units = {u.primary.name: u for u in g.pair_raw_jpeg(paths)}
    assert units["A.RAF"].sibling == d/"a.jpg"          # case-insensitive stem + ext
    assert units["B.NEF"].sibling is None                 # RAW only
    assert units["C.jpg"].sibling is None                 # JPEG only stays a lone unit
    assert units["D.CR3"].sibling == d/"D.JPEG"           # first by sorted path; the other JPEG stays alone
    assert "D.jpg" in units and units["D.jpg"].sibling is None
    assert units["E.NEF"].sibling is None and units["E.CR2"].sibling is None  # two RAWs: ambiguous, never guess
    assert "E.jpg" in units
    assert units["F.ARW"].sibling is None and "F.png" in units  # PNG never pairs
    assert sum(len(u.files) for u in units.values()) == len(paths)  # every path appears exactly once

def test_pair_raw_jpeg_never_crosses_directories(tmp_path: Path) -> None:
    (tmp_path/"sub").mkdir()
    units = g.pair_raw_jpeg([tmp_path/"A.RAF", tmp_path/"sub"/"A.jpg"])
    assert all(u.sibling is None for u in units)

# --- bursts ---
def _units(d: Path, *names: str) -> list[g.Unit]:
    return [g.Unit(d/n) for n in names]

def test_detect_bursts_chains_at_gap_edge_and_min_size(tmp_path: Path) -> None:
    d = tmp_path
    units = _units(d, "1.NEF", "2.NEF", "3.NEF", "4.NEF", "5.NEF")
    meta = {d/"1.NEF": _meta(0.0), d/"2.NEF": _meta(2.0), d/"3.NEF": _meta(4.0),   # gaps exactly 2.0 chain
            d/"4.NEF": _meta(6.001),                                             # 2.001 splits
            d/"5.NEF": _meta(8.0)}
    bursts, rest = g.detect_bursts(units, meta, d)
    assert [[f.name for f in b.files] for b in bursts] == [["1.NEF", "2.NEF", "3.NEF"]]
    assert {u.primary.name for u in rest} == {"4.NEF", "5.NEF"}  # a 2-chain is not a burst

def test_detect_bursts_orders_members_by_timestamp(tmp_path: Path) -> None:
    d = tmp_path
    units = _units(d, "c.NEF", "a.NEF", "b.NEF")
    meta = {d/"c.NEF": _meta(2.0), d/"a.NEF": _meta(0.0), d/"b.NEF": _meta(1.0)}
    bursts, _ = g.detect_bursts(units, meta, d)
    assert [u.primary.name for u in bursts[0].units] == ["a.NEF", "b.NEF", "c.NEF"]

def test_detect_bursts_never_groups_missing_dates_or_mixed_cameras_or_dirs(tmp_path: Path) -> None:
    d = tmp_path; sub = d/"sub"; sub.mkdir()
    units = _units(d, "1.NEF", "2.NEF", "3.NEF", "x.NEF", "p1.RAF", "p2.RAF", "p3.RAF") + _units(sub, "4.NEF")
    meta = {d/"1.NEF": _meta(0.0), d/"2.NEF": _meta(1.0), d/"3.NEF": _meta(2.0),
            d/"x.NEF": _meta(None),                                  # no date: indexed individually
            d/"p1.RAF": _meta(0.5, "FUJI|X-T5"), d/"p2.RAF": _meta(1.5, "FUJI|X-T5"), d/"p3.RAF": _meta(2.5, "FUJI|X-T5"),  # interleaved other camera
            sub/"4.NEF": _meta(3.0)}                                 # other directory
    bursts, rest = g.detect_bursts(units, meta, d)
    members = sorted(sorted(f.name for f in b.files) for b in bursts)
    assert members == [["1.NEF", "2.NEF", "3.NEF"], ["p1.RAF", "p2.RAF", "p3.RAF"]]
    assert {u.primary.name for u in rest} == {"x.NEF", "4.NEF"}

def test_detect_bursts_unit_without_meta_entry_is_treated_as_undated(tmp_path: Path) -> None:
    d = tmp_path
    bursts, rest = g.detect_bursts(_units(d, "1.NEF", "2.NEF", "3.NEF"), {}, d)
    assert bursts == [] and len(rest) == 3

# --- build_groups ---
def test_build_groups_burst_of_pairs_and_lone_pair(tmp_path: Path) -> None:
    d = tmp_path
    paths = [d/f"{n}.{e}" for n in ("1", "2", "3", "9") for e in ("NEF", "jpg")] + [d/"s.NEF"]
    ts = {"1": 0.0, "2": 1.0, "3": 2.0, "9": 100.0}
    meta = {p: _meta(ts[p.stem]) for p in paths if p.stem in ts}
    meta[d/"s.NEF"] = _meta(None)
    groups, singles = g.build_groups(paths, meta, d)
    kinds = sorted((grp.kind, sorted(f.name for f in grp.files)) for grp in groups)
    assert kinds == [("burst", ["1.NEF", "1.jpg", "2.NEF", "2.jpg", "3.NEF", "3.jpg"]), ("raw_jpeg", ["9.NEF", "9.jpg"])]
    assert singles == [d/"s.NEF"]
    burst = next(grp for grp in groups if grp.kind == "burst")
    assert [u.sibling for u in burst.units] == [d/"1.jpg", d/"2.jpg", d/"3.jpg"]  # JPEG is the preview source

def test_group_id_is_order_independent_and_root_relative(tmp_path: Path) -> None:
    a, b = tmp_path/"x"/"1.NEF", tmp_path/"x"/"2.NEF"
    assert g.group_id([a, b], tmp_path) == g.group_id([b, a], tmp_path)
    assert g.group_id([a, b], tmp_path).startswith("b_") and len(g.group_id([a, b], tmp_path)) == 10
    assert g.group_id([a, b], tmp_path) != g.group_id([tmp_path/"y"/"1.NEF", tmp_path/"y"/"2.NEF"], tmp_path)

# --- representative ---
def test_pick_representative_highest_sharpness_tie_to_earliest(tmp_path: Path) -> None:
    d = tmp_path
    grp = g.MediaGroup(kind="burst", id="b_0", units=_units(d, "1.NEF", "2.NEF", "3.NEF"))
    scores = {"1.NEF": 10.0, "2.NEF": 50.0, "3.NEF": 50.0}
    g.pick_representative(grp, lambda u: scores[u.primary.name])
    assert grp.primary == d/"2.NEF"
    assert grp.sharpness == {d/"1.NEF": 10.0, d/"2.NEF": 50.0, d/"3.NEF": 50.0}

def test_pick_representative_scores_pair_preview_source(tmp_path: Path) -> None:
    d = tmp_path
    grp = g.MediaGroup(kind="raw_jpeg", id="b_1", units=[g.Unit(d/"1.NEF", d/"1.jpg")])
    seen: list[Path] = []
    g.pick_representative(grp, lambda u: seen.append(u.preview_source) or 1.0)
    assert seen == [d/"1.jpg"] and grp.primary == d/"1.NEF"

# --- exiftool reader ---
def test_read_group_metadata_batches_one_exiftool_call_without_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b = tmp_path/"a.NEF", tmp_path/"b.NEF"
    calls: list[list[str]] = []
    def run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        argfile = Path(cmd[cmd.index("-@") + 1])
        assert argfile.read_text().splitlines() == [str(a), str(b)]
        return types.SimpleNamespace(returncode=1, stdout=json.dumps([
            {"SourceFile": str(a), "DateTimeOriginal": "2024:08:14 07:23:11", "SubSecTimeOriginal": "05", "Make": "NIKON CORPORATION", "Model": "NIKON Z 8"},
            {"SourceFile": str(b), "Make": "NIKON CORPORATION", "Model": "NIKON Z 8"},
        ]), stderr="1 files could not be read")
    monkeypatch.setattr("framedex.grouping.subprocess.run", run)
    meta = g.read_group_metadata([a, b])
    assert len(calls) == 1 and "-n" not in calls[0] and "-json" in calls[0]
    assert meta[a].camera == "NIKON CORPORATION|NIKON Z 8"
    assert meta[a].timestamp == pytest.approx(g.parse_exif_timestamp("2024:08:14 07:23:11", None, "05"))
    assert meta[b].timestamp is None      # rc=1 with valid JSON: partial results still used

def test_read_group_metadata_fails_loud_and_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("framedex.grouping.subprocess.run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="not json", stderr=""))
    assert g.read_group_metadata([tmp_path/"a.NEF"]) == {}
    assert "no burst detection" in capsys.readouterr().err
    def boom(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("exiftool")
    monkeypatch.setattr("framedex.grouping.subprocess.run", boom)
    assert g.read_group_metadata([tmp_path/"a.NEF"]) == {}

def test_read_group_metadata_empty_input_skips_exiftool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("framedex.grouping.subprocess.run", lambda *a, **k: pytest.fail("must not run"))
    assert g.read_group_metadata([]) == {}
```

- [ ] **Step 2: Run** `uv run --no-sync pytest tests/test_grouping.py -q` → FAIL (`ModuleNotFoundError: framedex.grouping`).

- [ ] **Step 3: Implement `src/framedex/grouping.py`**

```python
#!/usr/bin/env python3
"""
framedex.grouping — burst + RAW/JPEG grouping pre-pass for still photos.

One moment = one vision call = one primary sidecar. Runs after `find_images`,
before the per-file loop, folder mode only. Everything here is pure logic
over (path, timestamp, camera) tuples so tests never shell out; the single
batched exiftool call is isolated in `read_group_metadata`.

Pairing: same directory + same stem (case-insensitive), one RAW + one camera
JPEG → one Unit (RAW primary, JPEG sibling used as the preview source).
Bursts: within one directory and one camera, frames whose successive
DateTimeOriginal gaps are <= BURST_GAP_SEC chain; a chain of >= BURST_MIN_SIZE
is a burst. Files without a usable date never join a group (indexed
individually — never guess).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from framedex.images import RAW_EXTENSIONS

# Successive frames (same folder, same camera) at most this far apart chain
# into one burst. 2 s covers every burst mode; two deliberate frames rarely
# land inside it.
BURST_GAP_SEC = 2.0
# A chain shorter than this is not a burst; its frames index individually.
BURST_MIN_SIZE = 3
# Only camera-rendered JPEGs pair with a same-stem RAW. PNG/TIFF/HEIC/WebP
# sharing a stem are exports or derivatives, not the same capture.
PAIR_JPEG_EXTENSIONS = {".jpg", ".jpeg"}
# The five tags grouping needs. No `-n`: SubSecTimeOriginal must stay a string
# ("05" is .05 s; as a number it would become 5 → .5 s).
GROUP_EXIF_TAGS = [
    "-DateTimeOriginal",
    "-SubSecDateTimeOriginal",
    "-SubSecTimeOriginal",
    "-Make",
    "-Model",
]

_TZ_SUFFIX_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")
_EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class GroupMeta:
    """The per-file EXIF grouping needs. `timestamp` is seconds since a fixed
    naive epoch with subsecond resolution; None when DateTimeOriginal is
    missing or unparsable. `camera` is "Make|Model" ("" when unknown)."""

    timestamp: float | None
    camera: str


@dataclass(frozen=True)
class Unit:
    """A RAW+JPEG pair collapsed to its RAW, or a lone file."""

    primary: Path
    sibling: Path | None = None

    @property
    def preview_source(self) -> Path:
        """The camera JPEG when paired (camera-rendered colour beats the RAW's
        embedded preview and skips the exiftool extraction), else the file."""
        return self.sibling or self.primary

    @property
    def files(self) -> list[Path]:
        return [self.primary] + ([self.sibling] if self.sibling else [])


@dataclass
class MediaGroup:
    kind: str  # "burst" | "raw_jpeg" (a burst of pairs is a burst)
    id: str  # b_<sha1[:8] of the sorted root-relative member paths>
    units: list[Unit]  # timestamp order for a burst; exactly one for raw_jpeg
    primary: Path | None = None  # set by pick_representative
    sharpness: dict[Path, float] = field(default_factory=dict)  # unit.primary -> score

    @property
    def files(self) -> list[Path]:
        return [f for u in self.units for f in u.files]


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def parse_exif_timestamp(
    datetime_original: str | None,
    subsec_datetime_original: str | None,
    subsec_time_original: str | None,
) -> float | None:
    """Resolve a capture instant to seconds. Prefer SubSecDateTimeOriginal
    ('2024:08:14 07:23:11.25+03:00'); else DateTimeOriginal plus
    SubSecTimeOriginal as a fractional suffix; else whole seconds. Any
    timezone suffix is dropped: only gaps between frames of one camera
    matter, and those share a zone. None when nothing parses."""
    raw = (subsec_datetime_original or "").strip()
    frac = ""
    if raw:
        base = _TZ_SUFFIX_RE.sub("", raw)
        if "." in base:
            base, frac = base.split(".", 1)
    else:
        base = _TZ_SUFFIX_RE.sub("", (datetime_original or "").strip())
        digits = (subsec_time_original or "").strip()
        if digits.isdigit():
            frac = digits
    try:
        dt = datetime.strptime(base, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    seconds = (dt - _EPOCH).total_seconds()
    if frac.isdigit():
        seconds += float("0." + frac)
    return seconds


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def pair_raw_jpeg(paths: list[Path]) -> list[Unit]:
    """Collapse RAW+JPEG pairs into Units; everything else becomes a lone Unit.
    A RAW with two JPEG candidates pairs with the first by sorted path (the
    other stays alone); two RAWs sharing a stem are ambiguous and both stay
    alone. Result is sorted by primary path."""
    by_key: dict[tuple[Path, str], list[Path]] = {}
    for p in paths:
        by_key.setdefault((p.parent, p.stem.lower()), []).append(p)
    units: list[Unit] = []
    for members in by_key.values():
        raws = sorted(m for m in members if m.suffix.lower() in RAW_EXTENSIONS)
        jpegs = sorted(m for m in members if m.suffix.lower() in PAIR_JPEG_EXTENSIONS)
        others = [m for m in members if m not in raws and m not in jpegs]
        if len(raws) == 1 and jpegs:
            units.append(Unit(raws[0], jpegs[0]))
            units += [Unit(j) for j in jpegs[1:]]
        else:
            units += [Unit(m) for m in raws + jpegs]
        units += [Unit(m) for m in others]
    return sorted(units, key=lambda u: u.primary)


# ---------------------------------------------------------------------------
# Bursts
# ---------------------------------------------------------------------------


def group_id(files: list[Path], root: Path) -> str:
    """Stable, order-independent id from the sorted root-relative paths."""
    rel = sorted(str(f.relative_to(root)) for f in files)
    return "b_" + hashlib.sha1("\n".join(rel).encode("utf-8")).hexdigest()[:8]


def detect_bursts(
    units: list[Unit], meta: dict[Path, GroupMeta], root: Path
) -> tuple[list[MediaGroup], list[Unit]]:
    """Chain units within one directory + one camera whose successive
    timestamps are <= BURST_GAP_SEC apart; chains of >= BURST_MIN_SIZE become
    burst groups (members in timestamp order). Units without a timestamp
    never chain. Returns (bursts, remaining units)."""
    dated: dict[tuple[Path, str], list[tuple[float, Unit]]] = {}
    rest: list[Unit] = []
    for u in units:
        m = meta.get(u.primary)
        if m is None or m.timestamp is None:
            rest.append(u)
            continue
        dated.setdefault((u.primary.parent, m.camera), []).append((m.timestamp, u))

    bursts: list[MediaGroup] = []
    for lane in dated.values():
        lane.sort(key=lambda t: (t[0], t[1].primary))
        chain: list[Unit] = []
        prev_ts: float | None = None

        def flush() -> None:
            if len(chain) >= BURST_MIN_SIZE:
                files = [f for u in chain for f in u.files]
                bursts.append(MediaGroup("burst", group_id(files, root), list(chain)))
            else:
                rest.extend(chain)

        for ts, u in lane:
            if prev_ts is not None and ts - prev_ts > BURST_GAP_SEC:
                flush()
                chain = []
            chain.append(u)
            prev_ts = ts
        flush()
    return bursts, sorted(rest, key=lambda u: u.primary)


def build_groups(
    paths: list[Path], meta: dict[Path, GroupMeta], root: Path
) -> tuple[list[MediaGroup], list[Path]]:
    """Pair, then chain bursts over the pair-collapsed units. Returns
    (groups sorted by first file, ungrouped single files). Every input path
    appears exactly once: in one group's `files` or in the singles."""
    bursts, rest = detect_bursts(pair_raw_jpeg(paths), meta, root)
    pairs = [
        MediaGroup("raw_jpeg", group_id(u.files, root), [u]) for u in rest if u.sibling
    ]
    singles = [u.primary for u in rest if not u.sibling]
    groups = sorted(bursts + pairs, key=lambda grp: grp.files[0])
    return groups, singles


# ---------------------------------------------------------------------------
# Representative
# ---------------------------------------------------------------------------


def pick_representative(group: MediaGroup, score: Callable[[Unit], float]) -> None:
    """Score every unit (the caller renders `unit.preview_source` and returns
    its Laplacian sharpness); the highest becomes `group.primary`, a tie
    going to the earliest unit. Every score is recorded in `group.sharpness`
    so members can carry their own number. Deterministic, local, explainable
    — deliberately not a model call (that would scale vision cost with
    shooting style)."""
    best_i = 0
    best = float("-inf")
    for i, u in enumerate(group.units):
        s = score(u)
        group.sharpness[u.primary] = s
        if s > best:
            best_i, best = i, s
    group.primary = group.units[best_i].primary


# ---------------------------------------------------------------------------
# exiftool (the one impure function)
# ---------------------------------------------------------------------------


def read_group_metadata(paths: list[Path]) -> dict[Path, GroupMeta]:
    """One batched exiftool call for the whole image list (an argfile dodges
    ARG_MAX). Partial output is used as-is (exiftool exits 1 when any file
    fails but still emits JSON for the rest). On a failure to run or parse,
    print a loud warning and return {} — every file then indexes
    individually (RAW+JPEG pairing needs no EXIF and still applies)."""
    if not paths:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False) as fh:
        fh.write("\n".join(str(p) for p in paths) + "\n")
        argfile = Path(fh.name)
    try:
        result = subprocess.run(
            ["exiftool", "-json", *GROUP_EXIF_TAGS, "-@", str(argfile)],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        if not isinstance(data, list):
            raise ValueError("exiftool JSON is not a list")
    except (OSError, ValueError) as e:
        print(
            f"warning: grouping: exiftool batch read failed ({e}) — "
            "no burst detection this run; files index individually",
            file=sys.stderr,
        )
        return {}
    finally:
        argfile.unlink(missing_ok=True)

    out: dict[Path, GroupMeta] = {}
    for entry in data:
        src = entry.get("SourceFile")
        if not src:
            continue
        make = str(entry.get("Make") or "").strip()
        model = str(entry.get("Model") or "").strip()
        camera = f"{make}|{model}" if (make or model) else ""
        ts = parse_exif_timestamp(
            _as_str(entry.get("DateTimeOriginal")),
            _as_str(entry.get("SubSecDateTimeOriginal")),
            _as_str(entry.get("SubSecTimeOriginal")),
        )
        out[Path(src)] = GroupMeta(timestamp=ts, camera=camera)
    return out


def _as_str(v: object) -> str | None:
    return None if v is None else str(v)
```

- [ ] **Step 4: Run** `uv run --no-sync pytest tests/test_grouping.py -q` → PASS. Run ruff + mypy.
- [ ] **Step 5: Commit** `git commit -m "grouping: burst chaining, RAW/JPEG pairing, representative pick, batched exiftool reader"`

---

### Task 3: `process_one_image` hooks — `preview_override` and `before_sidecar`

**Files:**
- Modify: `src/framedex/images.py:436-568`
- Test: `tests/test_images.py`

- [ ] **Step 1: Write the failing tests** (reuse the mock set from `test_process_one_image_writes_sidecar`)

```python
def test_process_one_image_preview_override_skips_render(tmp_path, monkeypatch) -> None:
    img = tmp_path / "a.jpg"; img.write_bytes(b"x")
    pre = tmp_path / "pre.jpg"; pre.write_bytes(b"p")
    monkeypatch.setattr(images, "get_image_metadata", lambda p: {"size_bytes": 1})
    monkeypatch.setattr(pipeline, "get_gps", lambda p: {})
    monkeypatch.setattr(images, "render_preview", lambda *a, **k: pytest.fail("must not render"))
    monkeypatch.setattr("framedex.images.time.sleep", lambda s: None)
    seen: list[Path] = []
    def vision(frames, prompt, model):
        seen.extend(frames); return VISION_OK  # module-level valid response fixture
    monkeypatch.setattr(pipeline, "describe_frames_cli", vision)
    res = images.process_one_image(img, tmp_path, _opts(), pipeline.ProcessContext(), preview_override=pre)
    assert res.sidecar is not None and seen == [pre]

def test_process_one_image_before_sidecar_runs_between_faces_and_sidecar(tmp_path, monkeypatch) -> None:
    # spy order: faces → hook(fm) → sidecar; hook receives the assembled frontmatter
    ...
    order: list[str] = []
    monkeypatch.setattr(face_db, "write_faces", lambda *a, **k: order.append("faces"))
    monkeypatch.setattr(pipeline, "serialize_sidecar", lambda *a, **k: order.append("sidecar"))
    def hook(fm: dict[str, Any]) -> None:
        assert fm["rating"] == "keep"; order.append("hook")
    images.process_one_image(img, tmp_path, _opts(), ctx_with_face_conn, before_sidecar=hook)
    assert order == ["faces", "hook", "sidecar"]

def test_process_one_image_before_sidecar_failure_writes_no_sidecar(tmp_path, monkeypatch) -> None:
    def hook(fm): raise RuntimeError("stub write failed")
    with pytest.raises(RuntimeError):
        images.process_one_image(img, tmp_path, _opts(), pipeline.ProcessContext(), before_sidecar=hook)
    assert not pipeline.sidecar_path(img).exists()
```

- [ ] **Step 2: Run** → FAIL (`unexpected keyword argument`).
- [ ] **Step 3: Implement**: add to the signature

```python
    preview_override: Path | None = None,
    before_sidecar: Callable[[dict[str, Any]], None] | None = None,
```

docstring additions: "`preview_override`: an already-rendered upright ≤1920px JPEG to use as the vision/face input instead of rendering `image` (burst grouping renders each member once for scoring). `before_sidecar`: called with the assembled frontmatter after faces are committed and before the sidecar is written; if it raises, no sidecar is written and the file is redone next run (group stubs are written here so the primary sidecar stays the last write)." In the body: `preview = preview_override or render_preview(image, tmp_dir)`; after the `write_faces` block: `if before_sidecar is not None: before_sidecar(fm)`.

- [ ] **Step 4: Run** tests → PASS.
- [ ] **Step 5: Commit** `git commit -m "images: preview_override + before_sidecar hooks on process_one_image"`

---

### Task 4: `process_group` + stub sidecars

**Files:**
- Modify: `src/framedex/images.py` (append), `src/framedex/pipeline.py:553-565`
- Test: `tests/test_images.py`

- [ ] **Step 1: Write the failing tests**

```python
def _group_mocks(tmp_path, monkeypatch, scores: dict[str, float]) -> list[str]:
    """Render → creates a preview file per unit; sharpness by file name; exif/gps/vision mocked. Returns spy log."""
    log: list[str] = []
    def render(src: Path, out_dir: Path) -> Path | None:
        if src.name.startswith("nopreview"):
            return None
        p = out_dir / "preview.jpg"; p.write_bytes(b"p"); log.append(f"render:{src.name}"); return p
    monkeypatch.setattr(images, "render_preview", render)
    monkeypatch.setattr("framedex.frame_sampling.laplacian_sharpness", lambda p: scores[p.parent.name])
    monkeypatch.setattr(images, "get_image_metadata", lambda p: {"size_bytes": 1, "creation_time": "2024-08-14T07:23:11", "dimensions": "10x10", "camera": {"model": "Z8"}})
    monkeypatch.setattr(pipeline, "get_gps", lambda p: {"lat": -1.4, "lon": 35.0})
    monkeypatch.setattr("framedex.images.time.sleep", lambda s: None)
    monkeypatch.setattr(pipeline, "describe_frames_cli", lambda *a, **k: VISION_OK)
    real = pipeline.serialize_sidecar
    def spy(sidecar, fm, title, body):
        log.append(f"sidecar:{sidecar.name}"); return real(sidecar, fm, title, body)
    monkeypatch.setattr(pipeline, "serialize_sidecar", spy)
    return log

def test_process_group_writes_stubs_then_primary_with_group_blocks(tmp_path, monkeypatch) -> None:
    d = tmp_path; files = [d/"1.NEF", d/"2.NEF", d/"3.NEF"]
    for f in files: f.write_bytes(b"x")
    log = _group_mocks(tmp_path, monkeypatch, {"1.NEF": 1.0, "2.NEF": 9.0, "3.NEF": 5.0})
    grp = grouping.MediaGroup("burst", "b_abc", [grouping.Unit(f) for f in files])
    ctx = pipeline.ProcessContext(geocoder=types.SimpleNamespace(reverse=lambda la, lo: "Mara, Kenya"))
    res = images.process_group(grp, tmp_path, _opts(), ctx)
    assert res.sidecar == pipeline.sidecar_path(d/"2.NEF") and res.stubs_written == 2
    assert [e for e in log if e.startswith("sidecar:")] == ["sidecar:1.NEF.description.md", "sidecar:3.NEF.description.md", "sidecar:2.NEF.description.md"]
    assert log.count("render:2.NEF") == 1                       # primary rendered exactly once
    primary = _frontmatter(res.sidecar)
    assert primary["group"] == {"kind": "burst", "id": "b_abc", "primary": True, "members": ["1.NEF", "2.NEF", "3.NEF"], "sharpness": 9.0}
    stub = _frontmatter(pipeline.sidecar_path(d/"1.NEF"))
    assert stub["group"] == {"kind": "burst", "id": "b_abc", "primary": False, "primary_file": "2.NEF", "members": ["1.NEF", "2.NEF", "3.NEF"], "sharpness": 1.0}
    for k in images.STUB_COPIED_FIELDS: assert stub[k] == primary[k]
    assert stub["faces"] == [] and stub["face_count"] == 0
    assert stub["file"] == "1.NEF" and stub["path"] == "1.NEF" and stub["camera"] == {"model": "Z8"}
    assert stub["location"] == {"lat": -1.4, "lon": 35.0, "place": "Mara, Kenya"}   # place copied, not re-geocoded
    assert "See 2.NEF.description.md (burst primary)." in pipeline.sidecar_path(d/"1.NEF").read_text()

def test_process_group_pair_uses_jpeg_preview_and_raw_primary(tmp_path, monkeypatch) -> None:
    raw, jpg = tmp_path/"1.NEF", tmp_path/"1.jpg"; raw.write_bytes(b"x"); jpg.write_bytes(b"x")
    log = _group_mocks(tmp_path, monkeypatch, {"1.NEF": 3.0})
    grp = grouping.MediaGroup("raw_jpeg", "b_p", [grouping.Unit(raw, jpg)])
    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert res.sidecar == pipeline.sidecar_path(raw) and "render:1.jpg" in log and "render:1.NEF" not in log
    assert _frontmatter(pipeline.sidecar_path(jpg))["group"]["primary_file"] == "1.NEF"

def test_process_group_no_renderable_member_skips_without_stubs(tmp_path, monkeypatch) -> None:
    files = [tmp_path/f"nopreview{i}.NEF" for i in (1, 2, 3)]
    for f in files: f.write_bytes(b"x")
    _group_mocks(tmp_path, monkeypatch, {})
    grp = grouping.MediaGroup("burst", "b_n", [grouping.Unit(f) for f in files])
    res = images.process_group(grp, tmp_path, _opts(), pipeline.ProcessContext())
    assert res.skipped_reason == "no_preview" and not any(pipeline.sidecar_path(f).exists() for f in files)

def test_process_group_vision_error_writes_nothing(tmp_path, monkeypatch) -> None:
    # describe_frames_cli → "[CLI error]" ⇒ no stubs, no primary (hook never reached)
```

- [ ] **Step 2: Run** → FAIL (`no attribute process_group`).
- [ ] **Step 3: Implement**

`pipeline.ProcessResult`: add `stubs_written: int = 0  # group members that got a stub sidecar (no vision call)`.

In `images.py` (imports: `import copy`, `from collections.abc import Callable`, `from framedex import face_db, frame_sampling, grouping, pipeline` — NOTE: `grouping` imports `RAW_EXTENSIONS` from `images`; to avoid a circular import, import `grouping` lazily inside `process_group`, or move `RAW_EXTENSIONS`/`PAIR_JPEG_EXTENSIONS` usage so `grouping` only needs the constant — do this: `grouping.py` keeps `from framedex.images import RAW_EXTENSIONS` and `images.py` uses `from framedex import grouping` **inside** `process_group`/type-annotates with `"grouping.MediaGroup"` under `TYPE_CHECKING`):

```python
# Assessment fields a stub copies from its group primary: everything the
# model produced. Faces are NOT copied — no detection ran on that frame.
STUB_COPIED_FIELDS = (
    "rating", "cull_reason", "technical", "lighting", "time_of_day",
    "dominant_color_palette", "dominant_colors", "scene_type", "people_count",
    "keywords",
)


def _group_block(group: grouping.MediaGroup, member: Path) -> dict[str, Any]:
    assert group.primary is not None
    block: dict[str, Any] = {"kind": group.kind, "id": group.id, "primary": member == group.primary}
    if member != group.primary:
        block["primary_file"] = group.primary.name
    block["members"] = [f.name for f in group.files]
    # A pair's JPEG sibling was scored as the RAW's preview: report that score.
    unit = next(u for u in group.units if member in u.files)
    block["sharpness"] = round(group.sharpness[unit.primary], 1)
    return block


def build_stub_frontmatter(member: Path, root: Path, metadata: dict[str, Any], gps: dict[str, Any], primary_fm: dict[str, Any], group: grouping.MediaGroup) -> dict[str, Any]:
    """Own file/EXIF/GPS fields + the primary's assessment (deep-copied so
    YAML never emits aliases) + a group block. `place` mirrors the primary's
    when this member has GPS (a burst is one place; no per-stub geocode)."""
    structured = {k: copy.deepcopy(primary_fm.get(k)) for k in STUB_COPIED_FIELDS}
    place = (primary_fm.get("location") or {}).get("place", "") if gps.get("lat") is not None else ""
    return build_image_frontmatter(member, root, metadata, gps, place, structured, [], extra_frontmatter={"group": _group_block(group, member)})


def _render_and_score(unit: grouping.Unit, tmp_dir: Path) -> tuple[Path | None, float]:
    """Render the unit's preview into its own subdir (render_preview writes a
    fixed filename) and score it. A RAW with no embedded preview scores -1 so
    it is never the pick while any member renders."""
    sub = tmp_dir / unit.primary.name
    sub.mkdir()
    preview = render_preview(unit.preview_source, sub)
    if preview is None:
        return None, -1.0
    return preview, frame_sampling.laplacian_sharpness(preview)


def process_group(group: grouping.MediaGroup, root: Path, opts: pipeline.ProcessOptions, ctx: pipeline.ProcessContext) -> pipeline.ProcessResult:
    """One vision call for a burst / RAW+JPEG pair. Renders + scores every
    member locally, runs the full pipeline on the sharpest (its rendered
    preview reused), and writes each other member a stub sidecar *before*
    the primary sidecar — the primary is the resume marker for the group."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="fdx-group-"))
    previews: dict[Path, Path] = {}
    try:
        def score(unit: grouping.Unit) -> float:
            preview, s = _render_and_score(unit, tmp_dir)
            if preview is not None:
                previews[unit.primary] = preview
            return s

        grouping.pick_representative(group, score)
        primary = group.primary
        assert primary is not None
        preview = previews.get(primary)
        if preview is None:
            return pipeline.ProcessResult(sidecar=None, skipped_reason="no_preview")
        members = [f for f in group.files if f != primary]
        print(f"  {group.kind}: {len(group.files)} files → 1 vision call; pick {primary.name} (sharpness {group.sharpness[primary]:.1f})")

        def write_stubs(primary_fm: dict[str, Any]) -> None:
            for m in members:
                fm = build_stub_frontmatter(m, root, get_image_metadata(m), pipeline.get_gps(m), primary_fm, group)
                body = f"See {primary.name}{pipeline.SIDECAR_SUFFIX} ({group.kind} primary)."
                pipeline.serialize_sidecar(pipeline.sidecar_path(m), fm, m.name, [("Description", body)])

        result = process_one_image(primary, root, opts, ctx, preview_override=preview, extra_frontmatter={"group": _group_block(group, primary)}, before_sidecar=write_stubs)
        if result.sidecar is not None:
            result.stubs_written = len(members)
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 4: Run** tests → PASS; ruff/mypy clean.
- [ ] **Step 5: Commit** `git commit -m "images: process_group — score members locally, one vision call, stub sidecars before the primary"`

---

### Task 5: `fdx` wiring — `--no-group`, grouping pre-pass, group-aware resume, tally

**Files:**
- Modify: `src/framedex/index_videos.py:1240-1463`, `src/framedex/runner.py:150-212`
- Test: `tests/test_index_videos.py`, `tests/test_runner.py`

- [ ] **Step 1: Write the failing tests** (pattern: `test_image_only_run_never_loads_whisper`)

```python
def _wire_main(tmp_path, monkeypatch, meta, *, calls: list[str]) -> None:
    monkeypatch.setattr("framedex.runner.check_claude_cli", lambda: True)
    monkeypatch.setattr(grouping, "read_group_metadata", lambda paths: meta)
    def one(image, root, opts, ctx, **kw):
        calls.append(f"one:{image.name}"); return pipeline.ProcessResult(sidecar=pipeline.sidecar_path(image), rating="keep")
    def grp(group, root, opts, ctx):
        calls.append("group:" + ",".join(f.name for f in group.files))
        return pipeline.ProcessResult(sidecar=pipeline.sidecar_path(group.files[0]), rating="keep", stubs_written=len(group.files) - 1)
    monkeypatch.setattr(images, "process_one_image", one)
    monkeypatch.setattr(images, "process_group", grp)

def _burst(tmp_path, n=3):  # n NEFs 1s apart + one lone file
    files = [tmp_path / f"{i}.NEF" for i in range(1, n + 1)]
    for f in files: f.write_bytes(b"x")
    (tmp_path / "lone.NEF").write_bytes(b"x")
    meta = {f: grouping.GroupMeta(float(i), "N|Z8") for i, f in enumerate(files)}
    meta[tmp_path / "lone.NEF"] = grouping.GroupMeta(None, "N|Z8")
    return files, meta

def test_main_routes_groups_and_singles(tmp_path, monkeypatch, capsys) -> None:
    files, meta = _burst(tmp_path); calls: list[str] = []
    _wire_main(tmp_path, monkeypatch, meta, calls=calls)
    monkeypatch.setattr(sys, "argv", ["fdx", str(tmp_path), "--media", "images", "--no-faces", "--no-geocode"])
    assert index_videos.main() == 0
    assert calls == ["group:1.NEF,2.NEF,3.NEF", "one:lone.NEF"]
    assert "Grouped: 3 files in 1 group" in capsys.readouterr().out

def test_main_no_group_processes_every_file(tmp_path, monkeypatch) -> None:
    ... argv + ["--no-group"] → calls == ["one:1.NEF", "one:2.NEF", "one:3.NEF", "one:lone.NEF"]; read_group_metadata never called

def test_main_skips_complete_group_and_redoes_incomplete(tmp_path, monkeypatch) -> None:
    files, meta = _burst(tmp_path)
    for f in files: pipeline.sidecar_path(f).write_text("---\nfile: x\n---\n")
    ... run → calls == ["one:lone.NEF"]                      # complete: every member has a sidecar
    pipeline.sidecar_path(files[1]).unlink()                 # primary (or any stub) missing → whole group again
    ... run → calls == ["group:1.NEF,2.NEF,3.NEF"]

def test_main_max_files_counts_a_group_as_one_item(tmp_path, monkeypatch) -> None:
    ... "--max-files", "1" → calls == ["group:1.NEF,2.NEF,3.NEF"]

def test_main_dry_run_lists_groups(tmp_path, monkeypatch, capsys) -> None:
    ... "--dry-run" → "would process [burst ×3 → 1 call]: 1.NEF .. 3.NEF" in out and no calls

# runner
def test_record_result_counts_groups_and_stubs(capsys) -> None:
    tally = runner.RunTally()
    runner.record_result(ProcessResult(sidecar=Path("a.NEF.description.md"), rating="keep", stubs_written=2), tally, backend="cli", max_duration_min=30)
    assert tally.groups == 1 and tally.stubs == 2 and "+2 alternates" in capsys.readouterr().out
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

`runner.RunTally`: `groups: int = 0`, `stubs: int = 0`. In `record_result` after `tally.processed += 1`: `if result.stubs_written: tally.groups += 1; tally.stubs += result.stubs_written`; and the notes: `stubs_note = f", +{result.stubs_written} alternates" if result.stubs_written else ""` appended in both print branches.

`index_videos.main`, parser (next to `--frame-sampling`):

```python
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Index every still individually. By default bursts (same camera, "
        "same folder, ≤2s apart, ≥3 frames) and RAW+JPEG pairs are grouped: "
        "one vision call on the sharpest member, stub sidecars for the rest.",
    )
```

Discovery block replaces the images branch:

```python
    todo: list[tuple[Any, str]] = []  # (Path, "video"|"image") or (MediaGroup, "group")
    ...
    if args.media in ("all", "images"):
        imgs = images.find_images(root, args.exclude)
        n_found += len(imgs)
        print(f"  found {len(imgs)} image files")
        groups: list[grouping.MediaGroup] = []
        singles = imgs
        if not args.no_group:
            groups, singles = grouping.build_groups(imgs, grouping.read_group_metadata(imgs), root)
            if groups:
                n_grouped = sum(len(grp.files) for grp in groups)
                print(f"  grouped {n_grouped} files into {len(groups)} groups (bursts + RAW/JPEG pairs)")
        itodo = singles if args.force else [im for im in singles if not has_sidecar(im)]
        todo += [(im, "image") for im in itodo]
        # A group is done only when every member has a sidecar: stubs are
        # written before the primary, so a crash between them leaves a
        # member without one and the whole group is redone (stubs rewritten).
        gtodo = groups if args.force else [grp for grp in groups if not all(has_sidecar(f) for f in grp.files)]
        todo += [(grp, "group") for grp in gtodo]

    def _sort_key(item: tuple[Any, str]) -> Path:
        return item[0].files[0] if item[1] == "group" else item[0]

    todo.sort(key=_sort_key)
    n_files_todo = sum(len(t[0].files) if t[1] == "group" else 1 for t in todo)
    skipped = n_found - n_files_todo
```

`--max-files` unchanged (slices items). Dry-run:

```python
        for item, kind in todo:
            if kind == "group":
                first, last = item.files[0].name, item.files[-1].name
                print(f"  would process [{item.kind} ×{len(item.files)} → 1 call]: {item.files[0].parent.relative_to(root)}/{first} .. {last}")
            else:
                print(f"  would process [{kind}]: {item.relative_to(root)}")
```

Loop:

```python
    for i, (item, kind) in enumerate(todo, start=1):
        if kind == "group":
            rel = item.files[0].parent.relative_to(root)
            print(f"[{i}/{len(todo)}] {rel}/ ({item.kind}: {len(item.files)} files)")
        else:
            print(f"[{i}/{len(todo)}] {item.relative_to(root)}")
        try:
            if kind == "video":
                result = process_one_video(item, root, opts, ctx)
            elif kind == "group":
                result = images.process_group(item, root, opts, ctx)
            else:
                result = images.process_one_image(item, root, opts, ctx)
```

Summary: `if tally.groups: summary += f", Grouped: {tally.groups + tally.stubs} files in {tally.groups} group{'s' if tally.groups != 1 else ''}"` (test asserts `Grouped: 3 files in 1 group`).

Import: `from framedex import face_db, frame_sampling, grouping, images, pipeline, runner`.

- [ ] **Step 4: Run** full gate → PASS.
- [ ] **Step 5: Commit** `git commit -m "fdx: group-aware discovery/resume, --no-group, group tally"`

---

### Task 6: `fdx-photos` accepts `--no-group` (documented no-op)

**Files:**
- Modify: `src/framedex/photos_indexer.py:252-262`
- Test: `tests/test_photos_indexer.py`

- [ ] **Step 1: Test**: build the parser path used by the existing main tests with `--no-group` in argv and assert the run still completes (follow the file's existing `main()` mocking pattern); assert `"--no-group"` appears in `--help` output.
- [ ] **Step 2: Run** → FAIL (`unrecognized arguments`).
- [ ] **Step 3: Implement** (after `--frame-sampling`):

```python
    parser.add_argument(
        "--no-group",
        action="store_true",
        help="Accepted for parity with fdx; Apple Photos burst grouping is not "
        "implemented yet (Photos exposes native burst info via osxphotos — "
        "planned follow-up), so this flag currently has no effect.",
    )
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `git commit -m "fdx-photos: accept --no-group (no-op until Photos-native burst support)"`

---

### Task 7: `fdx-master` / `fdx-query` — stubs are not assessments

**Files:**
- Modify: `src/framedex/master_index.py:120-200`, `src/framedex/query.py`
- Test: `tests/test_master_index.py`, `tests/test_query.py`

- [ ] **Step 1: Tests**

```python
# master_index: 1 primary (cull) + 2 stubs (copied cull) + 1 single (keep)
def test_master_index_counts_primaries_only_and_reports_groups(tmp_path, monkeypatch) -> None:
    _write_sidecar_fm(tmp_path, "2.NEF", {"rating": "cull", "cull_reason": "blur", "keywords": ["lion"], "group": {"kind": "burst", "id": "b_1", "primary": True, "members": ["1.NEF","2.NEF","3.NEF"], "sharpness": 9.0}})
    for n in ("1.NEF", "3.NEF"):
        _write_sidecar_fm(tmp_path, n, {"rating": "cull", "cull_reason": "blur", "keywords": ["lion"], "group": {"kind": "burst", "id": "b_1", "primary": False, "primary_file": "2.NEF", "members": ["1.NEF","2.NEF","3.NEF"], "sharpness": 1.0}})
    _write_sidecar_fm(tmp_path, "lone.NEF", {"rating": "keep", "keywords": ["lion"]})
    ... main() ...
    md = (tmp_path / "_INDEX.md").read_text()
    assert "1 keep, 0 review, 1 cull" in md                    # stubs not counted
    assert "`lion` (2)" in md                                  # keyword freq over primaries
    assert "## Cull pile — 1 clips" in md                       # stubs not listed
    assert "- **Grouped:** 3 files in 1 group" in md
    idx = json.loads((tmp_path / "_INDEX.json").read_text())
    assert idx["clip_count"] == 4 and idx["group_count"] == 1 and idx["grouped_file_count"] == 3

# query
def test_matches_primary_only_filters_stubs() -> None:
    stub = {"rating": "keep", "group": {"primary": False}}
    assert matches(stub, make_args()) is True
    assert matches(stub, make_args(primary_only=True)) is False
    assert matches({"rating": "keep", "group": {"primary": True}}, make_args(primary_only=True)) is True
    assert matches({"rating": "keep"}, make_args(primary_only=True)) is True   # ungrouped records pass
```

(`make_args` defaults gain `"primary_only": False`.)

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

`parsing.py` (shared): 
```python
def is_group_stub(rec: dict[str, Any]) -> bool:
    """A burst/pair member whose assessment was copied from its group primary
    (`group.primary: false`). Stubs are real sidecars but not assessments."""
    group = rec.get("group")
    return isinstance(group, dict) and group.get("primary") is False
```
`master_index`: `assessed = [r for r in records if not is_group_stub(r)]`; `rating_counter`, `keyword_freq`, `place_freq`, `face_total`, `named_people`, `cull_clips`, per-trip `ratings` computed over assessed; `n_groups = len({r["group"]["id"] for r in records if isinstance(r.get("group"), dict)})`, `n_grouped = sum(1 for r in records if isinstance(r.get("group"), dict))`; JSON adds `"group_count"`, `"grouped_file_count"`; stats line `- **Grouped:** {n_grouped} files in {n_groups} group(s); ratings/keywords/cull pile count group primaries only` when `n_groups`.
`query`: `parser.add_argument("--primary-only", action="store_true", dest="primary_only", help="Hide burst/pair members whose assessment is copied from a group primary.")`; in `matches`: `if args.primary_only and is_group_stub(rec): return False`.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `git commit -m "fdx-master/fdx-query: count group primaries only; --primary-only"`

---

### Task 8: `fdx-xmp` — `burst-pick` / `burst-alternate` keywords

**Files:**
- Modify: `src/framedex/xmp_export.py:103-118`
- Test: `tests/test_xmp_export.py`

- [ ] **Step 1: Test**

```python
def test_subject_tags_mark_burst_pick_and_alternate() -> None:
    assert "burst-pick" in xmp_export._subject_tags(_fm(group={"kind": "burst", "primary": True}))
    assert "burst-alternate" in xmp_export._subject_tags(_fm(group={"kind": "burst", "primary": False}))
    tags = xmp_export._subject_tags(_fm(group={"kind": "raw_jpeg", "primary": True}))
    assert "burst-pick" not in tags and "burst-alternate" not in tags
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** at the end of `_subject_tags`:

```python
    group = frontmatter.get("group")
    if isinstance(group, dict) and group.get("kind") == "burst":
        tag = "burst-pick" if group.get("primary") else "burst-alternate"
        if tag not in tags:
            tags.append(tag)
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `git commit -m "fdx-xmp: tag burst picks and alternates in dc:subject"`

---

### Task 9: Docs + CHANGELOG

**Files:** `README.md` (Still photos section: new "Bursts and RAW+JPEG pairs" paragraph with the cost line; flags table `--no-group`; Known limitations), `SKILL.md` (pipeline bullet, flag, `--primary-only`, stub semantics), `docs/tuning.md` (new `## Burst grouping` section listing `BURST_GAP_SEC`, `BURST_MIN_SIZE`, `PAIR_JPEG_EXTENSIONS`), `CHANGELOG.md` (Unreleased → Added).

Known-limitation text: "Burst detection infers bursts from EXIF timestamp gaps, so two deliberate frames shot within 2 s of each other on the same camera can be merged into one group; `--no-group` opts out. Stub sidecars mirror the primary's assessment (rating, keywords, technical) without the model having seen that frame; faces are not copied."

- [ ] Commit `git commit -m "docs: burst grouping + RAW/JPEG pairing"`

---

## NOT in this phase (from the PRD)
Cross-directory grouping; perceptual-similarity grouping (dedupe, issue #12); auto-culling non-picks; multi-image vision calls; Apple Photos burst integration (osxphotos `burst`/`burst_selected`); batching the per-stub exiftool metadata/GPS reads.

## Definition of done
Full gate green; the wiring tests prove: complete groups skipped, incomplete groups redone whole, stubs before primary, primary rendered once, `--no-group` restores per-file behavior, `--max-files` counts calls; Codex plan + code review triaged in `REVIEW.md`; PR open, CI green.
