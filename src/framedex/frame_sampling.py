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
from typing import Any

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


def brightness_gate(mean_vs: list[float]) -> list[int]:
    """Indices of candidates whose mean V is within the usable range."""
    return [i for i, v in enumerate(mean_vs) if GATE_V_DARK <= v <= GATE_V_BLOWN]


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
        best = idx
        for nb in (idx - 1, idx + 1):
            if (
                0 <= nb < len(sharpness)
                and nb not in taken
                and dist[idx][nb] < NEAR_DUP_D
                and sharpness[nb] > sharpness[best]  # strict: ties keep the pick
            ):
                best = nb
        if best != idx:
            taken.discard(idx)
            taken.add(best)
            result[pos] = best
    return sorted(result)


# H-S histogram bins. V (brightness) is deliberately excluded from the
# signature: phone auto-exposure breathing alone otherwise registers as
# "content change" at or above the level of a genuine pan.
_H_BINS = 16
_S_BINS = 16


def _signatures(
    thumb_paths: list[Path],
) -> tuple[list[Any], list[float], list[float]]:
    """Per-thumbnail (H-S histogram, sharpness, mean V). cv2 imported
    lazily — a base dep at runtime, absent in CI."""
    import cv2

    hists: list[Any] = []
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


def pairwise_distances(hists: list[Any]) -> list[list[float]]:
    """Bhattacharyya distance matrix between H-S histograms (0 = identical)."""
    import cv2

    n = len(hists)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_BHATTACHARYYA))
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
