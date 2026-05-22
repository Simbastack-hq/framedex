"""Tests for framedex.frames — candidate planning, histogram distance, and the
diverse-frame selector. The pure helpers need neither ffmpeg nor opencv."""

from __future__ import annotations

from pathlib import Path

import pytest

from framedex.frames import (
    MAX_CANDIDATES,
    MIN_FRAMES,
    FrameRef,
    candidate_count,
    compute_histogram,
    extract_scene_frames,
    hist_distance,
    select_diverse_frames,
)


def _onehot(index: int, length: int) -> list[float]:
    """An L1-normalized one-hot histogram — distinct for each `index`."""
    return [1.0 if i == index else 0.0 for i in range(length)]


# --- candidate_count -------------------------------------------------------


def test_candidate_count_short_clip_meets_floor() -> None:
    # A 1s clip would yield 1 candidate at the raw interval — clamped up so the
    # selection floor can still be met.
    assert candidate_count(1.0) == MIN_FRAMES
    assert candidate_count(0.5) == MIN_FRAMES


def test_candidate_count_mid_clip_scales_with_duration() -> None:
    # interval is 2s → a 20s clip wants 10 candidates, a 21s clip wants 11.
    assert candidate_count(20.0) == 10
    assert candidate_count(21.0) == 11


def test_candidate_count_long_clip_capped() -> None:
    assert candidate_count(100_000.0) == MAX_CANDIDATES


# --- hist_distance ---------------------------------------------------------


def test_hist_distance_identical_is_zero() -> None:
    assert hist_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_hist_distance_disjoint_is_one() -> None:
    assert hist_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_hist_distance_partial_overlap_between() -> None:
    d = hist_distance([1.0, 0.0], [0.5, 0.5])
    assert 0.0 < d < 1.0


# --- select_diverse_frames -------------------------------------------------


def test_select_static_clip_returns_only_the_floor() -> None:
    # Eight identical frames: nothing clears the threshold, so the floor
    # top-up supplies exactly MIN_FRAMES, index 0 included.
    static = [_onehot(0, 4) for _ in range(8)]
    result = select_diverse_frames(static, threshold=0.3, floor=3, cap=10)
    assert result == [0, 1, 2]


def test_select_all_distinct_clamps_to_cap() -> None:
    distinct = [_onehot(i, 12) for i in range(12)]
    result = select_diverse_frames(distinct, threshold=0.3, floor=3, cap=10)
    assert result == list(range(10))


def test_select_all_distinct_below_cap_keeps_all() -> None:
    distinct = [_onehot(i, 8) for i in range(8)]
    result = select_diverse_frames(distinct, threshold=0.3, floor=3, cap=10)
    assert result == list(range(8))


def test_select_fewer_candidates_than_floor_returns_all() -> None:
    two = [_onehot(0, 4), _onehot(1, 4)]
    assert select_diverse_frames(two, threshold=0.3, floor=3, cap=10) == [0, 1]


def test_select_exactly_floor_returns_all() -> None:
    three = [_onehot(i, 4) for i in range(3)]
    assert select_diverse_frames(three, threshold=0.3, floor=3, cap=10) == [0, 1, 2]


def test_select_empty_and_single() -> None:
    assert select_diverse_frames([], threshold=0.3, floor=3, cap=10) == []
    assert select_diverse_frames([_onehot(0, 4)], threshold=0.3, floor=3, cap=10) == [0]


def test_select_cap_below_floor_stays_coherent() -> None:
    # cap < floor: the effective floor is clamped to cap, never exceeds it.
    five = [_onehot(i, 5) for i in range(5)]
    assert select_diverse_frames(five, threshold=0.3, floor=3, cap=2) == [0, 1]


def test_select_gradual_drift_spreads_across_the_clip() -> None:
    # Consecutive frames are near-identical, but the clip drifts end to end.
    # The selector skips near-duplicates and keeps a spread that reaches t=last.
    drift = [[1.0 - i / 10, i / 10] for i in range(11)]
    result = select_diverse_frames(drift, threshold=0.3, floor=3, cap=10)
    assert result == [0, 2, 7, 10]


def test_select_floor_topup_picks_farthest_points() -> None:
    # Greedy keeps only index 0 (everything is within threshold of it). The
    # floor top-up must add the *most-distinct* remaining frames — index 4
    # (farthest from 0), then index 2 (farthest from {0, 4}) — not 1 or 3.
    hists = [
        [1.00, 0.00],
        [0.95, 0.05],
        [0.90, 0.10],
        [0.85, 0.15],
        [0.70, 0.30],
    ]
    result = select_diverse_frames(hists, threshold=0.5, floor=3, cap=10)
    assert result == [0, 2, 4]


# --- extract_scene_frames (orchestration, helpers monkeypatched) -----------


def test_extract_scene_frames_too_short_returns_empty(tmp_path: Path) -> None:
    assert extract_scene_frames(tmp_path / "v.mov", tmp_path, 0.2) == []


def test_extract_scene_frames_keeps_path_timestamp_alignment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Six distinct candidates → the selector keeps all; each returned FrameRef
    # must carry its own path *and* its own timestamp (the bug being fixed:
    # timestamps were previously recomputed assuming even spacing).
    cands = [FrameRef(tmp_path / f"c{i}.jpg", float(i) * 3.0) for i in range(6)]
    monkeypatch.setattr(
        "framedex.frames._extract_candidates",
        lambda video, out_dir, duration: cands,
    )
    monkeypatch.setattr(
        "framedex.frames.compute_histogram",
        lambda path: _onehot(int(path.stem[1:]), 6),
    )
    refs = extract_scene_frames(tmp_path / "v.mov", tmp_path, 18.0, max_frames=10)
    assert refs == cands  # all six, in order, paths+timestamps intact
    assert [r.timestamp for r in refs] == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]


def test_extract_scene_frames_falls_back_when_ffmpeg_yields_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "framedex.frames._extract_candidates",
        lambda video, out_dir, duration: [],
    )
    sentinel = [FrameRef(tmp_path / "f.jpg", 1.0)]
    monkeypatch.setattr(
        "framedex.frames._extract_evenly_spaced",
        lambda video, out_dir, duration: sentinel,
    )
    refs = extract_scene_frames(tmp_path / "v.mov", tmp_path, 12.0)
    assert refs == sentinel
    assert "falling back" in capsys.readouterr().err


def test_extract_scene_frames_falls_back_when_too_few_usable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # ffmpeg yields 4 candidates but only 1 produces a usable histogram — below
    # the floor of 3, so it must fall back rather than return a single frame.
    cands = [FrameRef(tmp_path / f"c{i}.jpg", float(i)) for i in range(4)]
    monkeypatch.setattr(
        "framedex.frames._extract_candidates",
        lambda video, out_dir, duration: cands,
    )

    def only_c0(path: Path) -> list[float]:
        if path.stem != "c0":
            raise ValueError("bad frame")
        return _onehot(0, 4)

    monkeypatch.setattr("framedex.frames.compute_histogram", only_c0)
    sentinel = [FrameRef(tmp_path / f"f{i}.jpg", float(i)) for i in range(5)]
    monkeypatch.setattr(
        "framedex.frames._extract_evenly_spaced",
        lambda video, out_dir, duration: sentinel,
    )
    refs = extract_scene_frames(tmp_path / "v.mov", tmp_path, 8.0, max_frames=10)
    assert refs == sentinel
    assert "only 1 usable frame" in capsys.readouterr().err


def test_extract_scene_frames_skips_unreadable_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cands = [FrameRef(tmp_path / f"c{i}.jpg", float(i)) for i in range(4)]
    monkeypatch.setattr(
        "framedex.frames._extract_candidates",
        lambda video, out_dir, duration: cands,
    )

    def fake_hist(path: Path) -> list[float]:
        idx = int(path.stem[1:])
        if idx == 2:
            raise ValueError("bad frame")
        return _onehot(idx, 4)

    monkeypatch.setattr("framedex.frames.compute_histogram", fake_hist)
    refs = extract_scene_frames(tmp_path / "v.mov", tmp_path, 8.0, max_frames=10)
    assert len(refs) == 3  # the unreadable candidate is dropped, not fatal
    assert all(r.timestamp != 2.0 for r in refs)
    assert "skipping unreadable frame" in capsys.readouterr().err


# --- compute_histogram (needs opencv; skipped where unavailable) -----------


def test_compute_histogram_is_normalized(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # solid red (BGR)
    path = tmp_path / "red.jpg"
    cv2.imwrite(str(path), img)

    hist = compute_histogram(path)
    assert len(hist) == 256
    assert sum(hist) == pytest.approx(1.0)
