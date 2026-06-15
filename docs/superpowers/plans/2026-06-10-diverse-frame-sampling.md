# Diverse Frame Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace evenly-spaced 5-frame extraction with content-diversity selection (spec: `docs/superpowers/specs/2026-06-10-diverse-frame-sampling-design.md`), fixing the frame-timestamp desync bug along the way.

**Architecture:** New `src/framedex/frame_sampling.py` holds pure selection math (plain lists — CI has no cv2/numpy) plus cv2-lazy signature functions, mirroring `face_db.py`'s lazy-import convention. `index_videos.extract_frames` becomes a thin orchestrator returning `(path, timestamp)` pairs; the call site stops re-deriving timestamps. One new flag `--frame-sampling {diverse,even}` threads through `ProcessOptions` into both CLIs.

**Tech Stack:** Python 3.10+, ffmpeg fast-seek extraction, cv2 (HSV histograms, Laplacian) + numpy — both already base deps. pytest, hermetic (no ffmpeg/cv2 in CI tests).

**Constraints recap:** CI (`.github/workflows/ci.yml`) runs `uv sync --only-group dev --only-group test` — tests CANNOT import cv2 or numpy, even transitively at module level. ruff line-length 88, mypy strict.

---

### Task 1: `frame_sampling.py` — constants + timestamp helpers

**Files:**
- Create: `src/framedex/frame_sampling.py`
- Test: `tests/test_frame_sampling.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for framedex.frame_sampling — pure selection math.

CI installs no cv2/numpy, so everything tested here must run on plain
Python lists. The cv2-dependent signature step is exercised indirectly via
monkeypatching in the choose_frame_timestamps tests.
"""

from pathlib import Path

import pytest

from framedex import frame_sampling as fs


def test_even_timestamps_matches_legacy_formula() -> None:
    """Golden: must reproduce the historical evenly-spaced formula exactly,
    because --frame-sampling even pins legacy behavior."""
    assert fs.even_timestamps(12.0, 5) == [
        12.0 * (i + 1) / 6 for i in range(5)
    ]
    assert fs.even_timestamps(2.0, 3) == [0.5, 1.0, 1.5]


def test_candidate_timestamps_clamps_pool_size() -> None:
    # 10s clip: duration/2 = 5 < POOL_MIN -> 20 candidates
    short = fs.candidate_timestamps(10.0)
    assert len(short) == fs.POOL_MIN
    # 1h clip: duration/2 = 1800 > POOL_MAX -> 96 candidates
    long = fs.candidate_timestamps(3600.0)
    assert len(long) == fs.POOL_MAX
    # 120s clip: exactly 60 candidates, centered sampling, strictly increasing
    mid = fs.candidate_timestamps(120.0)
    assert len(mid) == 60
    assert mid[0] == pytest.approx(1.0)  # (0 + 0.5) * 120/60
    assert mid == sorted(mid)
    assert mid[-1] < 120.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'framedex.frame_sampling'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
framedex.frame_sampling — content-aware selection of vision frames.

Picks the N most mutually different, sharpest moments of a clip from a pool
of cheap low-res thumbnails (design: docs/superpowers/specs/
2026-06-10-diverse-frame-sampling-design.md). Selection math is pure Python
over plain lists so its tests stay hermetic (CI installs no cv2/numpy);
cv2 is imported lazily inside the signature step, mirroring face_db.py.
"""

from __future__ import annotations

from pathlib import Path

# One candidate thumbnail every ~2s of footage, clamped: POOL_MAX caps the
# worst-case pool cost (~96 fast-seeks ≈ 26s); POOL_MIN keeps short clips
# meaningfully sampled.
POOL_SECONDS_PER_CANDIDATE = 2.0
POOL_MIN = 20
POOL_MAX = 96
# 160px thumbnails are enough for H-S histograms and blur scoring while
# keeping decode and I/O negligible.
THUMB_WIDTH = 160
# Below this duration the candidate pool costs more than it can return over
# plain even spacing (5 seeks vs 20+); keep legacy behavior.
SHORT_CLIP_EVEN_CUTOFF = 20.0
# Mean-V brightness gates: drop near-black (lens cap, pocket) and blown
# (pointed-at-the-sun) candidates — a blown frame is maximally distant from
# everything and would otherwise always win a diversity slot.
GATE_V_DARK = 20.0
GATE_V_BLOWN = 245.0
# If gating leaves fewer than this multiple of num_frames, the clip is too
# dark/degenerate to select from; fall back to even spacing.
GATE_MIN_FACTOR = 2
# Max pairwise H-S distance below which the clip is visually static and
# "diversity" would just amplify noise; fall back to even spacing.
STATIC_GUARD = 0.05
# A temporal neighbor closer than this to a pick is "the same moment"; the
# sharpest member of the group represents it. (H-S metric, V excluded, so
# auto-exposure breathing does not register as content change.)
NEAR_DUP_D = 0.10


def even_timestamps(duration: float, num_frames: int) -> list[float]:
    """The legacy evenly-spaced formula. Must never change: --frame-sampling
    even and several fallback paths pin this exact output."""
    return [duration * (i + 1) / (num_frames + 1) for i in range(num_frames)]


def candidate_timestamps(duration: float) -> list[float]:
    """Centers of M equal slices of the clip, M clamped to [POOL_MIN, POOL_MAX]."""
    m = max(POOL_MIN, min(POOL_MAX, round(duration / POOL_SECONDS_PER_CANDIDATE)))
    return [(k + 0.5) * duration / m for k in range(m)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/framedex/frame_sampling.py tests/test_frame_sampling.py
git commit -m "feat: frame_sampling module — pool/even timestamp helpers"
```

---

### Task 2: Selection core — medoid seed + greedy farthest-point

**Files:**
- Modify: `src/framedex/frame_sampling.py`
- Test: `tests/test_frame_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_frame_sampling.py`:

```python
def _dist(points: list[float]) -> list[list[float]]:
    """1-D toy metric: distance = abs difference."""
    return [[abs(a - b) for b in points] for a in points]


def test_medoid_resists_outlier() -> None:
    # Points clustered at 0 with one far outlier: medoid is in the cluster.
    d = _dist([0.0, 0.1, 0.2, 9.0])
    assert fs.medoid(d) in (0, 1, 2)


def test_medoid_tie_breaks_to_lowest_index() -> None:
    d = _dist([0.0, 1.0])  # symmetric: both rows sum to 1.0
    assert fs.medoid(d) == 0


def test_select_diverse_spans_clusters() -> None:
    # Three tight clusters; asking for 3 picks one from each.
    points = [0.0, 0.05, 5.0, 5.05, 10.0, 10.05]
    picked = fs.select_diverse(_dist(points), 3)
    assert len(picked) == 3
    clusters = {round(points[i] / 5) for i in picked}
    assert clusters == {0, 1, 2}


def test_select_diverse_returns_sorted_and_deterministic() -> None:
    points = [0.0, 1.0, 2.0, 3.0, 4.0]
    a = fs.select_diverse(_dist(points), 3)
    b = fs.select_diverse(_dist(points), 3)
    assert a == b == sorted(a)


def test_select_diverse_handles_num_ge_pool() -> None:
    d = _dist([0.0, 1.0])
    assert fs.select_diverse(d, 5) == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v -k "medoid or diverse"`
Expected: FAIL with `AttributeError: ... no attribute 'medoid'`

- [ ] **Step 3: Implement**

Append to `src/framedex/frame_sampling.py`:

```python
def medoid(dist: list[list[float]]) -> int:
    """Index with the smallest summed distance to all others — the most
    representative candidate. Ties break to the lowest index (determinism)."""
    sums = [sum(row) for row in dist]
    return min(range(len(sums)), key=lambda i: (sums[i], i))


def select_diverse(dist: list[list[float]], num_frames: int) -> list[int]:
    """Greedy farthest-point selection seeded at the medoid.

    The medoid seed resists outlier-chasing (a garbage frame maximally far
    from everything must beat real content on min-distance to win a slot,
    not just be extreme). Returns chronologically sorted indices.
    """
    n = len(dist)
    if n <= num_frames:
        return list(range(n))
    selected = [medoid(dist)]
    while len(selected) < num_frames:
        best = -1
        best_min = -1.0
        for i in range(n):
            if i in selected:
                continue
            d_min = min(dist[i][j] for j in selected)
            if d_min > best_min:  # strict: ties keep the lowest index
                best, best_min = i, d_min
        selected.append(best)
    return sorted(selected)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/framedex/frame_sampling.py tests/test_frame_sampling.py
git commit -m "feat: medoid-seeded greedy farthest-point selection"
```

---

### Task 3: Gates, static guard, sharpness swap

**Files:**
- Modify: `src/framedex/frame_sampling.py`
- Test: `tests/test_frame_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_frame_sampling.py`:

```python
def test_brightness_gate_drops_dark_and_blown() -> None:
    mean_vs = [10.0, 128.0, 250.0, 60.0]
    assert fs.brightness_gate(mean_vs) == [1, 3]


def test_is_static_thresholds_on_max_distance() -> None:
    assert fs.is_static([[0.0, 0.01], [0.01, 0.0]]) is True
    assert fs.is_static([[0.0, 0.2], [0.2, 0.0]]) is False


def test_sharpness_swap_picks_sharper_near_duplicate() -> None:
    # Candidates 0,1,2 nearly identical (d=0.01); 1 selected but 2 is sharper.
    d = [
        [0.00, 0.01, 0.01, 0.90],
        [0.01, 0.00, 0.01, 0.90],
        [0.01, 0.01, 0.00, 0.90],
        [0.90, 0.90, 0.90, 0.00],
    ]
    sharp = [5.0, 5.0, 50.0, 5.0]
    assert fs.sharpness_swap([1, 3], d, sharp) == [2, 3]


def test_sharpness_swap_refuses_content_change() -> None:
    # Neighbor 2 is sharper but visually different (d=0.5): no swap.
    d = [
        [0.00, 0.01, 0.50],
        [0.01, 0.00, 0.50],
        [0.50, 0.50, 0.00],
    ]
    sharp = [5.0, 5.0, 50.0]
    assert fs.sharpness_swap([1], d, sharp) == [1]


def test_sharpness_swap_never_collides_with_another_pick() -> None:
    # 0 and 1 are near-dups, both selected; 1 sharper but already a pick,
    # so 0 must stay 0 rather than collapse the result to one frame.
    d = [[0.00, 0.01], [0.01, 0.00]]
    sharp = [5.0, 50.0]
    assert fs.sharpness_swap([0, 1], d, sharp) == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v -k "gate or static or swap"`
Expected: FAIL with AttributeError

- [ ] **Step 3: Implement**

Append to `src/framedex/frame_sampling.py`:

```python
def brightness_gate(mean_vs: list[float]) -> list[int]:
    """Indices of candidates whose mean V is within the usable range."""
    return [
        i for i, v in enumerate(mean_vs) if GATE_V_DARK <= v <= GATE_V_BLOWN
    ]


def is_static(dist: list[list[float]]) -> bool:
    """True when no candidate pair differs enough to justify selection."""
    return max(max(row) for row in dist) < STATIC_GUARD


def sharpness_swap(
    selected: list[int], dist: list[list[float]], sharpness: list[float]
) -> list[int]:
    """Replace each pick with its sharpest near-duplicate temporal neighbor
    (±1, distance < NEAR_DUP_D). Never swaps onto another pick and never
    crosses into different content."""
    result = list(selected)
    taken = set(selected)
    for pos, idx in enumerate(result):
        group = [idx]
        for nb in (idx - 1, idx + 1):
            if 0 <= nb < len(sharpness) and nb not in taken:
                if dist[idx][nb] < NEAR_DUP_D:
                    group.append(nb)
        best = max(group, key=lambda i: (sharpness[i], -i))
        if best != idx:
            taken.discard(idx)
            taken.add(best)
            result[pos] = best
    return sorted(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/framedex/frame_sampling.py tests/test_frame_sampling.py
git commit -m "feat: brightness gates, static guard, sharpness swap"
```

---

### Task 4: cv2 signature step + `choose_frame_timestamps` orchestrator

**Files:**
- Modify: `src/framedex/frame_sampling.py`
- Test: `tests/test_frame_sampling.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_frame_sampling.py`. These monkeypatch `_signatures` so
no cv2 import ever happens in CI:

```python
def _fake_signatures(
    dists: list[list[float]], sharps: list[float], vs: list[float]
):
    """Build a _signatures stand-in returning crafted values. The hist return
    slot carries the precomputed distance matrix (signature shape is opaque
    to choose_frame_timestamps — it only hands it to pairwise_distances)."""

    def fake(paths: list[Path]):
        return dists, sharps, vs

    return fake


def test_choose_falls_back_on_static_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    n = 6
    dists = [[0.0 if i == j else 0.01 for j in range(n)] for i in range(n)]
    monkeypatch.setattr(fs, "_signatures", _fake_signatures(dists, [1.0] * n, [100.0] * n))
    monkeypatch.setattr(fs, "pairwise_distances", lambda h: h)
    ts = [float(i) for i in range(n)]
    assert fs.choose_frame_timestamps([Path(f"{i}.jpg") for i in range(n)], ts, 5) is None


def test_choose_falls_back_when_gates_exhaust_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 12 candidates, only 4 survive brightness gates < 2*5.
    n = 12
    vs = [5.0] * 8 + [100.0] * 4
    monkeypatch.setattr(
        fs, "_signatures", _fake_signatures([[0.5] * n for _ in range(n)], [1.0] * n, vs)
    )
    ts = [float(i) for i in range(n)]
    assert fs.choose_frame_timestamps([Path(f"{i}.jpg") for i in range(n)], ts, 5) is None


def test_choose_selects_diverse_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    # 10 candidates at "positions" 0..9, three visual clusters.
    pos = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 9.0, 9.0, 9.0]
    dists = [[abs(a - b) for b in pos] for a in pos]
    monkeypatch.setattr(
        fs, "_signatures", _fake_signatures(dists, [1.0] * 10, [100.0] * 10)
    )
    monkeypatch.setattr(fs, "pairwise_distances", lambda h: h)
    ts = [float(i * 2) for i in range(10)]  # candidate k at 2k seconds
    chosen = fs.choose_frame_timestamps(
        [Path(f"{i}.jpg") for i in range(10)], ts, 3
    )
    assert chosen is not None
    assert len(chosen) == 3
    assert chosen == sorted(chosen)
    # One pick per visual cluster.
    clusters = {0.0: 0, 5.0: 1, 9.0: 2}
    assert {clusters[pos[int(t // 2)]] for t in chosen} == {0, 1, 2}


def test_choose_returns_none_on_signature_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(paths: list[Path]):
        raise RuntimeError("cv2 exploded")

    monkeypatch.setattr(fs, "_signatures", boom)
    assert fs.choose_frame_timestamps([Path("a.jpg")], [1.0], 5) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v -k choose`
Expected: FAIL with AttributeError

- [ ] **Step 3: Implement**

Append to `src/framedex/frame_sampling.py`:

```python
# H-S histogram bins. V (brightness) is deliberately excluded from the
# signature: phone auto-exposure breathing alone otherwise registers as
# "content change" at or above the level of a genuine pan.
_H_BINS = 16
_S_BINS = 16


def _signatures(
    thumb_paths: list[Path],
) -> tuple[list[object], list[float], list[float]]:
    """Per-thumbnail (H-S histogram, sharpness, mean V). cv2/numpy imported
    lazily — base deps at runtime, absent in CI."""
    import cv2

    hists: list[object] = []
    sharpness: list[float] = []
    mean_v: list[float] = []
    for p in thumb_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"unreadable thumbnail: {p}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [_H_BINS, _S_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hists.append(hist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        mean_v.append(float(hsv[:, :, 2].mean()))
    return hists, sharpness, mean_v


def pairwise_distances(hists: list[object]) -> list[list[float]]:
    """Bhattacharyya distance matrix between H-S histograms (0 = identical)."""
    import cv2

    n = len(hists)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(
                cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_BHATTACHARYYA)
            )
            dist[i][j] = dist[j][i] = d
    return dist


def choose_frame_timestamps(
    thumb_paths: list[Path],
    thumb_timestamps: list[float],
    num_frames: int,
) -> list[float] | None:
    """Pick num_frames timestamps maximizing visual diversity, or None to
    signal "fall back to even spacing" (static clip, gated-out pool, or any
    signature failure). Never raises."""
    try:
        hists, sharpness, mean_v = _signatures(thumb_paths)
        kept = brightness_gate(mean_v)
        if len(kept) < GATE_MIN_FACTOR * num_frames:
            return None
        kept_hists = [hists[i] for i in kept]
        dist = pairwise_distances(kept_hists)
        if is_static(dist):
            return None
        kept_sharp = [sharpness[i] for i in kept]
        picks = select_diverse(dist, num_frames)
        picks = sharpness_swap(picks, dist, kept_sharp)
        return sorted(thumb_timestamps[kept[i]] for i in picks)
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_frame_sampling.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/framedex/frame_sampling.py tests/test_frame_sampling.py
git commit -m "feat: cv2 signatures + choose_frame_timestamps orchestrator"
```

---

### Task 5: `extract_frames` refactor — (path, timestamp) pairs + thumbnail pool

**Files:**
- Modify: `src/framedex/index_videos.py:328-359` (extract_frames)
- Modify: `src/framedex/index_videos.py:895-955` (call site in process_one_video)
- Modify: `src/framedex/pipeline.py:489-499` (ProcessOptions)
- Test: `tests/test_index_videos.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index_videos.py`:

```python
class _FakeRun:
    """subprocess.run stand-in: records ffmpeg commands, creates the output
    file (last arg) so extract_frames sees a successful frame write."""

    def __init__(self, fail_outputs: set[str] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.fail_outputs = fail_outputs or set()

    def __call__(self, cmd: list[str], **kwargs: Any) -> Any:
        self.commands.append(cmd)
        out = Path(cmd[-1])
        if out.name not in self.fail_outputs:
            out.write_bytes(b"jpeg")

        class R:
            returncode = 0

        return R()


def test_extract_frames_even_returns_legacy_timestamp_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(index_videos.subprocess, "run", fake)
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=12.0, sampling="even"
    )
    assert [ts for _, ts in pairs] == [12.0 * (i + 1) / 6 for i in range(5)]
    assert all(p.exists() for p, _ in pairs)
    # 5 full-res seeks, no thumbnail pool.
    assert len(fake.commands) == 5


def test_extract_frames_failed_write_keeps_pairs_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the timestamp desync bug: a failed middle frame must
    drop its own timestamp, not shift the others (faces.db frame_time)."""
    fake = _FakeRun(fail_outputs={"frame_02.jpg"})
    monkeypatch.setattr(index_videos.subprocess, "run", fake)
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=12.0, sampling="even"
    )
    even = [12.0 * (i + 1) / 6 for i in range(5)]
    assert [ts for _, ts in pairs] == even[:2] + even[3:]


def test_extract_frames_short_clip_skips_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(index_videos.subprocess, "run", fake)
    called: list[int] = []
    monkeypatch.setattr(
        index_videos.frame_sampling,
        "choose_frame_timestamps",
        lambda *a, **k: called.append(1),
    )
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=10.0, sampling="diverse"
    )
    assert not called  # under SHORT_CLIP_EVEN_CUTOFF -> no pool, no selection
    assert [ts for _, ts in pairs] == [10.0 * (i + 1) / 6 for i in range(5)]


def test_extract_frames_diverse_uses_chosen_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(index_videos.subprocess, "run", fake)
    monkeypatch.setattr(
        index_videos.frame_sampling,
        "choose_frame_timestamps",
        lambda paths, ts, n: [3.0, 17.0, 31.0, 44.0, 58.0],
    )
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=60.0, sampling="diverse"
    )
    assert [ts for _, ts in pairs] == [3.0, 17.0, 31.0, 44.0, 58.0]
    # Pool thumbnails (30 for a 60s clip) + 5 full-res frames.
    assert len(fake.commands) == 30 + 5
    # Thumbnails were cleaned up after selection.
    assert not list(tmp_path.glob("cand_*.jpg"))


def test_extract_frames_diverse_falls_back_when_selection_declines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(index_videos.subprocess, "run", fake)
    monkeypatch.setattr(
        index_videos.frame_sampling,
        "choose_frame_timestamps",
        lambda paths, ts, n: None,
    )
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=60.0, sampling="diverse"
    )
    assert [ts for _, ts in pairs] == [60.0 * (i + 1) / 6 for i in range(5)]


def test_short_clip_three_frame_clamp_still_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeRun()
    monkeypatch.setattr(index_videos.subprocess, "run", fake)
    pairs = index_videos.extract_frames(
        Path("clip.mov"), tmp_path, num_frames=5, duration=2.0, sampling="diverse"
    )
    assert len(pairs) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_index_videos.py -v -k extract_frames`
Expected: FAIL — current `extract_frames` has no `duration`/`sampling` kwargs and probes ffprobe.

- [ ] **Step 3: Implement `extract_frames`**

Replace `index_videos.py:328-359` with (note: `from framedex import frame_sampling` joins the imports at the top of the file; the old `get_metadata` call moves behind a `duration is None` guard for back-compat):

```python
def _extract_one_frame(video: Path, ts: float, out: Path, width_cap: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{ts:.2f}",
        "-i",
        str(video),
        "-vframes",
        "1",
        "-q:v",
        "2",  # higher quality jpeg
        "-vf",
        f"scale='min({width_cap},iw)':-2",
        "-loglevel",
        "error",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True)
    return out.exists() and out.stat().st_size > 0


def extract_frames(
    video: Path,
    out_dir: Path,
    num_frames: int = 5,
    duration: float | None = None,
    sampling: str = "diverse",
) -> list[tuple[Path, float]]:
    """Extract num_frames JPEGs and return (path, timestamp) pairs.

    'diverse' samples a pool of low-res thumbnails across the clip and keeps
    the most mutually different, sharpest moments (frame_sampling module);
    'even' is the legacy evenly-spaced sampling. Static clips, short clips,
    and any selection failure fall back to even spacing, so the function
    always returns up to num_frames frames.
    """
    if duration is None:
        duration = get_metadata(video)["duration_seconds"]
    if duration < 0.5:
        return []
    if duration < 3:
        num_frames = min(num_frames, 3)

    timestamps = frame_sampling.even_timestamps(duration, num_frames)
    if sampling == "diverse" and duration >= frame_sampling.SHORT_CLIP_EVEN_CUTOFF:
        cand_ts = frame_sampling.candidate_timestamps(duration)
        thumbs: list[Path] = []
        kept_ts: list[float] = []
        for k, ts in enumerate(cand_ts):
            out = out_dir / f"cand_{k:03d}.jpg"
            # Quality 5 is plenty for histograms; keeps the pool I/O tiny.
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.2f}",
                "-i",
                str(video),
                "-vframes",
                "1",
                "-q:v",
                "5",
                "-vf",
                f"scale={frame_sampling.THUMB_WIDTH}:-2",
                "-loglevel",
                "error",
                str(out),
            ]
            subprocess.run(cmd, capture_output=True)
            if out.exists() and out.stat().st_size > 0:
                thumbs.append(out)
                kept_ts.append(ts)
        chosen = frame_sampling.choose_frame_timestamps(thumbs, kept_ts, num_frames)
        for t in thumbs:
            t.unlink(missing_ok=True)
        if chosen:
            timestamps = chosen

    frames: list[tuple[Path, float]] = []
    for i, ts in enumerate(timestamps):
        out = out_dir / f"frame_{i:02d}.jpg"
        if _extract_one_frame(video, ts, out, FRAME_MAX_WIDTH):
            frames.append((out, ts))
    return frames
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_index_videos.py tests/test_frame_sampling.py -v`
Expected: all PASS

- [ ] **Step 5: Update ProcessOptions and the call site**

In `pipeline.py`, add to `ProcessOptions` (after `no_whisper_prompt`):

```python
    frame_sampling: str = "diverse"  # 'diverse' | 'even' (legacy)
```

In `index_videos.py` `process_one_video` (lines 900-905), replace:

```python
        frames = extract_frames(video, tmp_frames, num_frames=5)
        duration = metadata["duration_seconds"]
        num = len(frames)
        frame_timestamps = (
            [duration * (i + 1) / (num + 1) for i in range(num)] if num else []
        )
```

with:

```python
        frame_pairs = extract_frames(
            video,
            tmp_frames,
            num_frames=5,
            duration=metadata["duration_seconds"],
            sampling=opts.frame_sampling,
        )
        frames = [p for p, _ in frame_pairs]
        frame_timestamps = [ts for _, ts in frame_pairs]
```

- [ ] **Step 6: Run the full suite**

Run: `uv run --no-sync pytest`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/framedex/index_videos.py src/framedex/pipeline.py tests/test_index_videos.py
git commit -m "feat: extract_frames returns (path,timestamp) pairs; diverse sampling

Fixes the latent desync where process_one_video re-derived timestamps from
the even-spacing formula, so a failed frame write shifted faces.db frame_time."
```

---

### Task 6: Prompt timestamps + CLI flags in both parsers

**Files:**
- Modify: `src/framedex/index_videos.py:573-616` (_build_vision_prompt), call sites at ~921-941
- Modify: `src/framedex/index_videos.py` main() argparse (~line 1202, after --whisper-fixes) and ProcessOptions construction (~1287)
- Modify: `src/framedex/photos_indexer.py` main() argparse (~line 241) and ProcessOptions construction (~405)
- Test: `tests/test_index_videos.py`

- [ ] **Step 1: Write the failing test**

```python
def test_vision_prompt_includes_frame_timestamps() -> None:
    ctx = {
        "filename": "clip.mov",
        "parent_folder": "drone",
        "duration_seconds": 60.0,
        "transcript": "",
    }
    prompt = index_videos._build_vision_prompt(
        [Path("f0.jpg"), Path("f1.jpg")],
        ctx,
        "",
        include_paths=False,
        timestamps=[3.0, 41.0],
    )
    assert "00:00:03" in prompt and "00:00:41" in prompt
    assert "evenly sampled" not in prompt


def test_vision_prompt_without_timestamps_keeps_legacy_wording() -> None:
    ctx = {
        "filename": "clip.mov",
        "parent_folder": "drone",
        "duration_seconds": 60.0,
        "transcript": "",
    }
    prompt = index_videos._build_vision_prompt(
        [Path("f0.jpg")], ctx, "", include_paths=False
    )
    assert "evenly sampled across the clip" in prompt
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest tests/test_index_videos.py -v -k vision_prompt`
Expected: FAIL — unexpected keyword `timestamps`

- [ ] **Step 3: Implement prompt change**

`_build_vision_prompt` gains `timestamps: list[float] | None = None` (last
parameter). Replace the `intro` block (lines 602-607) with:

```python
    sampled_at = ""
    if timestamps:
        sampled_at = ", ".join(_fmt_time(t) for t in timestamps)

    if include_paths:
        intro = (
            f"Read these {len(frames)} JPEG frames in order, then analyze "
            "the video clip."
        )
        if sampled_at:
            intro += f" Frames were sampled at {sampled_at}."
    elif sampled_at:
        intro = (
            f"Analyze this short video clip based on these {len(frames)} "
            f"frames, sampled at {sampled_at}."
        )
    else:
        intro = (
            f"Analyze this short video clip based on these {len(frames)} frames "
            "(evenly sampled across the clip)."
        )
```

At the three call sites in `process_one_video` (api/cli/local branches),
pass `timestamps=frame_timestamps` to `_build_vision_prompt`.

- [ ] **Step 4: Add the flag to BOTH parsers**

Identical block in `index_videos.main()` (after the `--whisper-fixes`
argument, ~line 1210) and `photos_indexer.main()` (after `--local-model`,
~line 240):

```python
    parser.add_argument(
        "--frame-sampling",
        choices=["diverse", "even"],
        default="diverse",
        help="How vision frames are picked. 'diverse' (default) samples "
        "small thumbnails across the clip and keeps the most mutually "
        "different, sharpest moments. 'even' is the legacy evenly-spaced "
        "sampling.",
    )
```

Both `ProcessOptions(...)` constructions gain:

```python
        frame_sampling=args.frame_sampling,
```

- [ ] **Step 5: Run the full suite**

Run: `uv run --no-sync pytest`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/framedex/index_videos.py src/framedex/photos_indexer.py tests/test_index_videos.py
git commit -m "feat: --frame-sampling flag (both CLIs) + real timestamps in vision prompt"
```

---

### Task 7: Docs + CHANGELOG

**Files:**
- Modify: `README.md` (pipeline step 4 ~line 90, Known limitations ~line 253, flag table ~line 192)
- Modify: `SKILL.md` (pipeline step ~line 15, known-limitations/roadmap line that promises ffmpeg `select` — find with `grep -n "scene" SKILL.md`)
- Modify: `docs/tuning.md` (new short section)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: README**

Pipeline step 4 becomes:

```markdown
4. `ffmpeg` → 5 content-diverse JPEG frames (≤1920px): small thumbnails are
   sampled across the clip and the 5 most mutually different, sharpest
   moments are kept (static clips keep plain even spacing; `--frame-sampling
   even` restores the legacy behavior)
```

Known limitations: delete the line `- Frame sampling is evenly-spaced, not scene-detected`.

Flag table: add row `| --frame-sampling diverse\|even | How the 5 vision frames are picked (default: diverse) |`.

- [ ] **Step 2: SKILL.md** — update the pipeline line 15 the same way; replace the roadmap/known-limitation sentence that names `select='gt(scene,0.4)'` with a sentence describing diversity sampling.

- [ ] **Step 3: docs/tuning.md** — append:

```markdown
## Frame sampling

`fdx` picks the 5 vision frames by sampling small thumbnails across the clip
(one per ~2s, capped at 96) and keeping the most mutually different, sharpest
moments — so a clip that pans from a beach to a street gets frames from both,
not five near-duplicates. Brightness is excluded from the difference metric so
auto-exposure drift doesn't count as change. Static clips, clips under 20s,
and night footage that fails the brightness gates keep the legacy evenly-
spaced sampling, as does `--frame-sampling even`. Pool size and thresholds
are documented constants in `src/framedex/frame_sampling.py`.
```

- [ ] **Step 4: CHANGELOG.md** — add an entry under the current unreleased heading following the file's existing format (read it first).

- [ ] **Step 5: Run suite + linters, commit**

```bash
uv run --no-sync pytest && uv run --no-sync ruff check src/framedex tests && uv run --no-sync ruff format --check src/framedex tests && uv run --no-sync mypy src/framedex tests --ignore-missing-imports
git add README.md SKILL.md docs/tuning.md CHANGELOG.md
git commit -m "docs: diverse frame sampling (closes the issue #5 known limitation)"
```

---

### Task 8: Real-clip validation (manual gate before PR)

**Files:**
- Create: `/tmp/fdx-ab/` working dir (not committed)

- [ ] **Step 1:** Pick ~10 clips from `~/Downloads` (mix: `DJI_0113.MOV` drone, several `IMG_*.MOV` phone clips, one long `copy_*.MOV`). Copy to `/tmp/fdx-ab/clips/`.

- [ ] **Step 2:** For each clip, run `extract_frames` twice via a small driver script (import from the repo, call with `sampling="even"` and `sampling="diverse"`), saving frames to `/tmp/fdx-ab/<clip>/<mode>/` and printing chosen timestamps + wall time.

- [ ] **Step 3:** Compare: (a) diverse sets visually more distinct than even sets on clips with change, (b) no garbage frames (blur/black/blown) won slots, (c) static clips fell back to even, (d) per-clip overhead within the spec's 6–26s envelope. Calibrate `STATIC_GUARD` / `NEAR_DUP_D` if real-footage distances demand it; re-run.

- [ ] **Step 4:** Report results to NJ with sample frames before opening the PR. `diverse` stays default only if this passes.

---

## Self-review notes

- Spec coverage: pool/seek strategy (T5), signatures + V exclusion (T4), gates/guards/swap (T3), selection (T2), fallbacks (T4/T5), timestamp-pair refactor + desync fix (T5), prompt (T6), flag in both parsers (T6), docs (T7), validation (T8). Constants all named with comments (T1/T4).
- Types consistent: `extract_frames -> list[tuple[Path, float]]`; `choose_frame_timestamps(paths, ts, num) -> list[float] | None`; dist matrices are `list[list[float]]` throughout.
- No ffmpeg/cv2/numpy in any CI test path (fakes + monkeypatching only).
