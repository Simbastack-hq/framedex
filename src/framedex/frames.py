"""Content-aware frame sampling for framedex.

The vision model only ever sees the frames picked here, so frame choice
directly drives description quality. Instead of evenly-spaced sampling (which is
content-blind — it wastes frames on static shots and misses distinct moments on
multi-scene clips), this module extracts dense candidate frames, computes a
color histogram per candidate, and greedily keeps the frames that differ enough
from the last kept one. The result is an *adaptive* count: a static clip yields
the floor, a busy multi-scene clip yields up to the cap.

The pure helpers — ``candidate_count``, ``hist_distance``,
``select_diverse_frames`` — are stdlib-only and unit-tested without ffmpeg or
opencv. ``cv2`` is imported lazily inside ``compute_histogram`` so this module
stays importable in environments without opencv (e.g. the CI test job).
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Frame extraction cap — wider is more informative for the vision model on hard
# clips (low light, motion blur). 1920 is a sweet spot: enough pixels to
# disambiguate small details, still small enough that base64-encoding the
# selected frames keeps the API request reasonable.
FRAME_MAX_WIDTH = 1920

# One candidate frame roughly every CANDIDATE_INTERVAL_SEC seconds, but never
# fewer than MIN_FRAMES (so short clips still meet the floor) and never more
# than MAX_CANDIDATES (cost containment on long clips).
CANDIDATE_INTERVAL_SEC = 2.0
MAX_CANDIDATES = 150

# Selection bounds.
MIN_FRAMES = 3  # floor: every clip with enough candidates gets at least this
DEFAULT_MAX_FRAMES = 10  # cap: --max-frames default
DEFAULT_SCENE_THRESHOLD = 0.30  # Bhattacharyya distance; >= this = a new scene

# Histogram-distance threshold knob and bin layout.
_HIST_H_BINS = 16
_HIST_S_BINS = 16

# Clips shorter than this yield no frames at all (matches prior behavior).
_MIN_DURATION_SEC = 0.5


@dataclass(frozen=True)
class FrameRef:
    """A selected frame: its file path and its timestamp in the source clip."""

    path: Path
    timestamp: float


def candidate_count(duration: float) -> int:
    """Number of candidate frames to extract for a clip of ``duration`` seconds.

    Clamped to ``[MIN_FRAMES, MAX_CANDIDATES]``. The lower bound guarantees even
    very short clips yield enough candidates for the selection floor; the upper
    bound caps decode cost on long clips.
    """
    raw = math.ceil(duration / CANDIDATE_INTERVAL_SEC)
    return max(MIN_FRAMES, min(raw, MAX_CANDIDATES))


def hist_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Bhattacharyya distance between two L1-normalized histograms.

    Returns 0.0 for identical distributions, 1.0 for fully disjoint ones.
    """
    bc = sum(math.sqrt(p * q) for p, q in zip(a, b, strict=True))
    return math.sqrt(max(0.0, 1.0 - bc))


def select_diverse_frames(
    histograms: list[list[float]],
    *,
    threshold: float,
    floor: int,
    cap: int,
) -> list[int]:
    """Pick visually distinct frames from candidate ``histograms``.

    Walks candidates in time order, keeping a frame whenever it differs from the
    last kept frame by at least ``threshold``. Stops at ``cap``. If fewer than
    the effective floor survive, tops up with the most-distinct remaining
    candidates (farthest-point). Returns sorted candidate indices.
    """
    n = len(histograms)
    if n == 0:
        return []
    eff_floor = min(floor, cap, n)
    if n <= eff_floor:
        return list(range(n))

    kept = [0]
    for i in range(1, n):
        if len(kept) >= cap:
            break
        if hist_distance(histograms[i], histograms[kept[-1]]) >= threshold:
            kept.append(i)

    # Floor top-up: add whichever remaining candidate is farthest from every
    # already-kept frame, until the effective floor is met.
    while len(kept) < eff_floor:
        best_idx = -1
        best_dist = -1.0
        for i in range(n):
            if i in kept:
                continue
            d = min(hist_distance(histograms[i], histograms[k]) for k in kept)
            if d > best_dist:
                best_dist = d
                best_idx = i
        if best_idx < 0:
            break
        kept.append(best_idx)

    return sorted(kept)


def compute_histogram(path: Path) -> list[float]:
    """L1-normalized 2D Hue x Saturation histogram of an image.

    ``cv2`` is imported lazily so this module is importable without opencv.
    Raises ``ValueError`` if the image cannot be read or is degenerate.
    """
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cv2 could not read frame: {path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, [_HIST_H_BINS, _HIST_S_BINS], [0, 180, 0, 256]
    )
    flat = [float(v) for v in hist.flatten()]
    total = sum(flat)
    if total <= 0.0:
        raise ValueError(f"empty histogram for frame: {path}")
    return [v / total for v in flat]


def _run_ffmpeg(cmd: list[str], label: str) -> None:
    """Run an ffmpeg command, warning to stderr on a nonzero exit.

    ffmpeg can still emit usable partial output on failure, so the caller
    decides what to do — this just makes the failure loud rather than silent.
    """
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        lines = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = lines[-1] if lines else f"exit code {proc.returncode}"
        print(f"  scene sampling: ffmpeg {label} failed: {tail}", file=sys.stderr)


def _extract_candidates(video: Path, out_dir: Path, duration: float) -> list[FrameRef]:
    """Extract evenly-spaced candidate frames in a single ffmpeg pass."""
    n = candidate_count(duration)
    interval = duration / n
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={1.0 / interval:.6f},scale='min({FRAME_MAX_WIDTH},iw)':-2",
        "-q:v",
        "2",
        "-loglevel",
        "error",
        str(out_dir / "cand_%04d.jpg"),
    ]
    _run_ffmpeg(cmd, f"candidate extraction for {video.name}")
    refs: list[FrameRef] = []
    for i in range(n):
        # ffmpeg's image2 muxer numbers output files from 1.
        out = out_dir / f"cand_{i + 1:04d}.jpg"
        if out.exists() and out.stat().st_size > 0:
            refs.append(FrameRef(path=out, timestamp=i * interval))
    return refs


def _extract_evenly_spaced(
    video: Path, out_dir: Path, duration: float, num_frames: int = 5
) -> list[FrameRef]:
    """Evenly-spaced frame extraction — the fallback when scene sampling fails."""
    if duration < _MIN_DURATION_SEC:
        return []
    if duration < 3:
        num_frames = min(num_frames, 3)
    refs: list[FrameRef] = []
    for i in range(num_frames):
        ts = duration * (i + 1) / (num_frames + 1)
        out = out_dir / f"frame_{i:02d}.jpg"
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
            "2",
            "-vf",
            f"scale='min({FRAME_MAX_WIDTH},iw)':-2",
            "-loglevel",
            "error",
            str(out),
        ]
        _run_ffmpeg(cmd, f"frame extraction for {video.name}")
        if out.exists() and out.stat().st_size > 0:
            refs.append(FrameRef(path=out, timestamp=ts))
    return refs


def extract_scene_frames(
    video: Path,
    out_dir: Path,
    duration: float,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
) -> list[FrameRef]:
    """Pick scene-distinct frames from ``video`` into ``out_dir``.

    Degrades to evenly-spaced sampling (with a warning) when too few usable
    candidate frames survive to meet the floor — whether ffmpeg extracted none,
    or readable ones came up short. A missing ``cv2`` is a broken install, not a
    runtime condition — that ``ImportError`` is left to propagate.
    """
    if duration < _MIN_DURATION_SEC:
        return []

    eff_floor = min(MIN_FRAMES, max_frames)
    candidates = _extract_candidates(video, out_dir, duration)

    histograms: list[list[float]] = []
    usable: list[FrameRef] = []
    for cand in candidates:
        try:
            histograms.append(compute_histogram(cand.path))
            usable.append(cand)
        except ValueError as e:
            print(f"  scene sampling: skipping unreadable frame ({e})", file=sys.stderr)

    if len(usable) < eff_floor:
        print(
            f"  scene sampling: only {len(usable)} usable frame(s) from "
            f"{video.name} (need {eff_floor}); falling back to evenly-spaced "
            f"sampling",
            file=sys.stderr,
        )
        return _extract_evenly_spaced(video, out_dir, duration)

    indices = select_diverse_frames(
        histograms, threshold=threshold, floor=MIN_FRAMES, cap=max_frames
    )
    return [usable[i] for i in indices]
