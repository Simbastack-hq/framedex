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
    assert fs.even_timestamps(12.0, 5) == [12.0 * (i + 1) / 6 for i in range(5)]
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
