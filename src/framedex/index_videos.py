#!/usr/bin/env python3
"""
framedex: Build a portable knowledge base of a video archive.

Pipeline per clip:
    ffprobe       → metadata (duration, codec, resolution, creation date)
    exiftool      → GPS lat/lon/altitude
    Nominatim     → reverse-geocoded place name (rate-limited, optional)
    ffmpeg        → 5 representative JPEG frames @ 1920px max
    ffmpeg        → mono 16k WAV
    WhisperX      → Whisper transcribe + word-level alignment + diarization
    WhisperX      → Whisper translate-mode pass (non-English clips only)
    insightface   → face detection + 512-dim ArcFace embeddings on frames
    Vision model  → structured YAML (rating/technical/lighting/keywords/etc.)
                    + prose description (Scene/Subjects/Action/Mood/etc.)
    write         → [filename].description.md sidecar + face row in faces.db

Output: per-clip .description.md sidecars + ~/.framedex/faces.db.
Idempotent + resumable. Run on any folder/drive. Sidecars travel with the data.
Use fdx-summary for folder summaries, fdx-master for drive-level overview,
fdx-query for filtering.

Usage:
    fdx /Volumes/SSD-2024                           # default Max CLI + Haiku
    fdx /Volumes/SSD-2024 --backend local           # local LM Studio (Gemma etc)
    fdx /Volumes/SSD-2024 --vision-model sonnet     # Sonnet for harder clips
    fdx /Volumes/SSD-2024 --no-faces                # skip face detection
    fdx /Volumes/SSD-2024 --max-files 5             # test on first 5
    fdx /Volumes/SSD-2024 --dry-run                 # count only
    fdx /Volumes/SSD-2024 --whisper-model large-v3  # higher Hindi accuracy
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import anthropic

# Defer heavy imports until after we know setup.py has been run.
try:
    import requests
    import whisperx
    import yaml
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Run: uv pip install -e .", file=sys.stderr)
    sys.exit(1)

# anthropic is only required for --backend api; import lazily inside that path.
# insightface is loaded lazily by face_db (heavy module).

from framedex import face_db
from framedex.parsing import (
    coerce_people_count,
    is_permission_denied,
    pick_diar_auth_kwarg,
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".avi",
    ".webm",
    ".hevc",
    ".mts",
    ".m2ts",
}
SIDECAR_SUFFIX = ".description.md"
CONTEXT_FILE = ".video-context.md"

# Default models — overridable via --vision-model. Map common shorthand to the
# full IDs that the API expects, and to the shorthand the CLI accepts.
VISION_MODEL_DEFAULT = "haiku"  # 'haiku' or 'sonnet'
VISION_MODELS: dict[str, dict[str, str | float]] = {
    "haiku": {
        "api": "claude-haiku-4-5-20251001",
        "cli": "claude-haiku-4-5",
        "cost_per_call_api": 0.002,
    },
    "sonnet": {
        "api": "claude-sonnet-4-6-20251001",
        "cli": "claude-sonnet-4-6",
        "cost_per_call_api": 0.008,
    },  # ~4x Haiku for short prompts + 5 frames
}

# Frame extraction cap — wider is more informative for the vision model on
# hard clips (low light, motion blur). 1920 chosen as a sweet spot: enough
# pixels for the model to disambiguate small details, but still small enough
# that base64-encoding 5 frames keeps the API request reasonable.
FRAME_MAX_WIDTH = 1920

COST_PER_CALL_USD_CLI = 0.0  # Max subscription: marginal cost is $0 to user
COST_PER_CALL_USD_LOCAL = 0.0  # Local model: $0, just electricity
USER_AGENT = "framedex/1.0 (personal archive indexer)"
CLI_INTER_CALL_DELAY = 0.4  # seconds, to be polite to Max TPM caps

DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
LOCAL_TIMEOUT_SEC = 180


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_videos(root: Path, exclude_patterns: list[str]) -> list[Path]:
    """Recursively find all videos under root. Skip hidden, sidecars, _output dirs."""
    videos: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        rel = p.relative_to(root)
        # Skip files in _-prefixed folders (used by our own KB outputs)
        if any(part.startswith("_") for part in rel.parts[:-1]):
            continue
        # Apply --exclude patterns
        excluded = False
        for pat in exclude_patterns:
            if pat in str(rel):
                excluded = True
                break
        if excluded:
            continue
        videos.append(p)
    return sorted(videos)


def sidecar_path(video: Path) -> Path:
    return video.with_suffix(video.suffix + SIDECAR_SUFFIX)


def has_sidecar(video: Path) -> bool:
    return sidecar_path(video).exists()


def load_context(root: Path) -> str:
    """Optional .video-context.md at scan root → returned as drive-level context.
    For per-clip context that includes parent folder contexts, use
    load_context_for_clip()."""
    ctx_file = root / CONTEXT_FILE
    if not ctx_file.exists():
        return ""
    return _read_context_file(ctx_file)


def _read_context_file(path: Path) -> str:
    """Read a .video-context.md, strip optional YAML frontmatter."""
    try:
        text = path.read_text().strip()
    except Exception:
        return ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].strip()
    return text


def load_context_for_clip(clip: Path, scan_root: Path) -> str:
    """Walk up from the clip's directory to the scan root, layering any
    .video-context.md files found along the way. Closer-to-clip files come
    last (more specific = higher priority context). Returns concatenated
    context with origin labels so the vision model knows the hierarchy."""
    contexts: list[tuple[Path, str]] = []
    # Start at scan root, walk down toward clip's directory
    try:
        rel = clip.parent.relative_to(scan_root)
    except ValueError:
        # Clip outside scan root somehow — just use the root context
        ctx = load_context(scan_root)
        return ctx

    # Build the chain of directories from scan_root down to the clip's parent
    chain = [scan_root]
    cur = scan_root
    for part in rel.parts:
        cur = cur / part
        chain.append(cur)

    # Read .video-context.md from each level (if present)
    for d in chain:
        ctx_file = d / CONTEXT_FILE
        if ctx_file.exists():
            text = _read_context_file(ctx_file)
            if text:
                # Label with the folder name so the model knows the hierarchy
                label = "(drive root)" if d == scan_root else d.name
                contexts.append((d, f"### Context from `{label}`\n{text}"))

    if not contexts:
        return ""
    return "\n\n".join(text for _, text in contexts)


# ---------------------------------------------------------------------------
# Whisper proper-noun biasing (Layer 1)
# ---------------------------------------------------------------------------
# `.video-context.md` files may contain a line like:
#     **Whisper proper nouns:** Yosemite, El Capitan, Half Dome, Ansel, ...
# The names are unioned along the drive-root → trip → subfolder chain and
# passed to Whisper as `initial_prompt` (prose) + `hotwords` (space-joined)
# so the model spells them correctly when it hears them.

_PROPER_NOUNS_LINE = re.compile(
    r"\*\*\s*Whisper proper nouns\s*:\s*\*\*\s*(.+)",
    re.IGNORECASE,
)


def _extract_proper_nouns(text: str) -> list[str]:
    names: list[str] = []
    for m in _PROPER_NOUNS_LINE.finditer(text):
        # Stop at the first newline so we don't slurp the next bullet
        raw_line = m.group(1).split("\n", 1)[0]
        for raw in raw_line.split(","):
            name = raw.strip().rstrip(".")
            if name and name not in names:
                names.append(name)
    return names


def load_proper_nouns_for_clip(clip: Path, scan_root: Path) -> list[str]:
    """Union proper-noun lists from all .video-context.md files along the chain
    from scan_root down to the clip's parent. Deduped, order preserved."""
    try:
        rel = clip.parent.relative_to(scan_root)
    except ValueError:
        ctx_file = scan_root / CONTEXT_FILE
        if not ctx_file.exists():
            return []
        return _extract_proper_nouns(_read_context_file(ctx_file))
    chain = [scan_root]
    cur = scan_root
    for part in rel.parts:
        cur = cur / part
        chain.append(cur)
    out: list[str] = []
    for d in chain:
        ctx_file = d / CONTEXT_FILE
        if ctx_file.exists():
            for name in _extract_proper_nouns(_read_context_file(ctx_file)):
                if name not in out:
                    out.append(name)
    return out


def _build_whisper_prompt(names: list[str]) -> tuple[str, str]:
    """Return (initial_prompt_prose, hotwords_string). Empty strings if no names."""
    if not names:
        return ("", "")
    name_list = ", ".join(names)
    prose = f"This footage may include the following names and places: {name_list}."
    hotwords = " ".join(names)
    return (prose, hotwords)


# ---------------------------------------------------------------------------
# Whisper canonical-name fixes (Layer 2)
# ---------------------------------------------------------------------------
# A simple regex post-processor that fixes common mishearings even when the
# initial-prompt bias misses. Loaded once at startup from
# ~/.framedex/whisper_fixes.json (override with --whisper-fixes PATH).
# Format:
#   { "fixes": [
#       {"pattern": "\\b(half dome|haff dome)\\b", "replace": "Half Dome"},
#       ...
#   ] }

WHISPER_FIXES_DEFAULT = Path.home() / ".framedex" / "whisper_fixes.json"


def load_whisper_fixes(path: Path) -> list[tuple[re.Pattern[str], str]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  (warn) failed to parse whisper fixes at {path}: {e}")
        return []
    fixes: list[tuple[re.Pattern[str], str]] = []
    for entry in data.get("fixes", []):
        pat = entry.get("pattern")
        rep = entry.get("replace")
        if not pat or rep is None:
            continue
        flags = re.IGNORECASE if str(entry.get("flags", "i")).lower() == "i" else 0
        try:
            fixes.append((re.compile(pat, flags), rep))
        except re.error as e:
            print(f"  (warn) bad regex in whisper fixes ({pat!r}): {e}")
    return fixes


def apply_whisper_fixes(text: str, fixes: list[tuple[re.Pattern[str], str]]) -> str:
    if not text or not fixes:
        return text
    for pat, rep in fixes:
        text = pat.sub(rep, text)
    return text


# ---------------------------------------------------------------------------
# Metadata + GPS + frames
# ---------------------------------------------------------------------------


def get_metadata(video: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "duration_seconds": 0,
            "width": None,
            "height": None,
            "creation_time": "",
            "size_bytes": video.stat().st_size,
            "codec": None,
        }
    data = json.loads(result.stdout or "{}")
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    vs = [s for s in streams if s.get("codec_type") == "video"]
    return {
        "duration_seconds": float(fmt.get("duration", 0)),
        "creation_time": fmt.get("tags", {}).get("creation_time", ""),
        "width": vs[0].get("width") if vs else None,
        "height": vs[0].get("height") if vs else None,
        "codec": vs[0].get("codec_name") if vs else None,
        "size_bytes": video.stat().st_size,
    }


def get_gps(video: Path) -> dict[str, Any]:
    """exiftool → lat/lon/alt. Returns {} if nothing usable."""
    cmd = [
        "exiftool",
        "-json",
        "-n",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-GPSCoordinates",
        "-LocationInformation",
        str(video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    try:
        data = json.loads(result.stdout)[0]
    except (json.JSONDecodeError, IndexError):
        return {}
    out = {}
    lat = data.get("GPSLatitude")
    lon = data.get("GPSLongitude")
    if lat is not None and lon is not None:
        try:
            out["lat"] = float(lat)
            out["lon"] = float(lon)
        except (TypeError, ValueError):
            pass
    alt = data.get("GPSAltitude")
    if alt is not None:
        with contextlib.suppress(TypeError, ValueError):
            out["altitude_m"] = float(alt)
    # Some clips embed GPSCoordinates as "lat, lon, alt"
    if "lat" not in out and data.get("GPSCoordinates"):
        coords = str(data["GPSCoordinates"]).split(",")
        if len(coords) >= 2:
            try:
                out["lat"] = float(coords[0])
                out["lon"] = float(coords[1])
            except ValueError:
                pass
    return out


class NominatimRateLimiter:
    """Polite reverse-geocoder. Per OSM policy: 1 req/sec max + identifying UA."""

    def __init__(self) -> None:
        self.last_call = 0.0
        self.cache: dict[tuple[int, int], str] = {}  # rounded lat/lon → place

    def reverse(self, lat: float, lon: float) -> str:
        # Coarse cache key: rounding to 3 decimals = ~110m precision
        key = (round(lat * 1000), round(lon * 1000))
        if key in self.cache:
            return self.cache[key]

        # Rate limit
        delta = time.time() - self.last_call
        if delta < 1.05:
            time.sleep(1.05 - delta)

        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": str(lat),
                    "lon": str(lon),
                    "format": "json",
                    "zoom": "14",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            self.last_call = time.time()
            if resp.ok:
                data = resp.json()
                place = str(data.get("display_name", ""))
                self.cache[key] = place
                return place
        except Exception:
            pass
        self.cache[key] = ""
        return ""


def extract_frames(video: Path, out_dir: Path, num_frames: int = 5) -> list[Path]:
    meta = get_metadata(video)
    duration = meta["duration_seconds"]
    if duration < 0.5:
        return []
    if duration < 3:
        num_frames = min(num_frames, 3)
    timestamps = [duration * (i + 1) / (num_frames + 1) for i in range(num_frames)]
    frames: list[Path] = []
    for i, ts in enumerate(timestamps):
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
            "2",  # higher quality jpeg
            "-vf",
            f"scale='min({FRAME_MAX_WIDTH},iw)':-2",
            "-loglevel",
            "error",
            str(out),
        ]
        subprocess.run(cmd, capture_output=True)
        if out.exists() and out.stat().st_size > 0:
            frames.append(out)
    return frames


# ---------------------------------------------------------------------------
# WhisperX: transcribe + align + diarize
# ---------------------------------------------------------------------------


def extract_audio(video: Path, out_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        "-loglevel",
        "error",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000


def transcribe_audio_whisperx(
    video: Path,
    whisper_model: Any,
    align_models: dict[str, Any],
    diarize_pipeline: Any | None,
    proper_nouns: list[str] | None = None,
    whisper_fixes: list[tuple[re.Pattern[str], str]] | None = None,
) -> dict[str, Any]:
    """
    Returns dict with keys:
        language, language_probability, transcript, english_translation,
        segments (list with optional 'speaker' field), speaker_count

    If `proper_nouns` is non-empty, biases Whisper toward those spellings
    via `initial_prompt` + `hotwords` for the duration of this call (resets
    after so the bias doesn't leak to the next clip).
    If `whisper_fixes` is non-empty, runs each regex/replace pair over the
    final transcript and english_translation as a Layer-2 safety net.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)
    try:
        if not extract_audio(video, audio_path):
            return _empty_transcript()
        audio = whisperx.load_audio(str(audio_path))

        # Layer 1: bias Whisper toward known proper nouns (from .video-context.md).
        # `whisper_model.options` is a faster_whisper.TranscriptionOptions
        # dataclass — direct field assignment is fine. Restore in finally so
        # the bias doesn't leak between clips.
        prev_prompt = whisper_model.options.initial_prompt
        prev_hotwords = whisper_model.options.hotwords
        if proper_nouns:
            prose, hotwords = _build_whisper_prompt(proper_nouns)
            whisper_model.options.initial_prompt = prose or None
            whisper_model.options.hotwords = hotwords or None
        else:
            whisper_model.options.initial_prompt = None
            whisper_model.options.hotwords = None

        # 1. Transcribe
        result = whisper_model.transcribe(audio, batch_size=8)
        language = result.get("language", "")
        segments = result.get("segments", [])
        if not segments:
            return _empty_transcript(language=language)

        # 2. Align (word-level timestamps) — required for diarization
        try:
            if language not in align_models:
                model_a, meta_a = whisperx.load_align_model(
                    language_code=language, device="cpu"
                )
                align_models[language] = (model_a, meta_a)
            model_a, meta_a = align_models[language]
            aligned = whisperx.align(
                segments,
                model_a,
                meta_a,
                audio,
                device="cpu",
                return_char_alignments=False,
            )
            segments = aligned.get("segments", segments)
        except Exception as e:
            # Alignment failed (e.g. unsupported language for alignment); diarize without
            print(f"    align failed for lang={language}: {e}")

        # 3. Diarize
        speaker_count = 0
        if diarize_pipeline is not None:
            try:
                diarize_segments = diarize_pipeline(audio)
                assigned = whisperx.assign_word_speakers(
                    diarize_segments, {"segments": segments}
                )
                segments = assigned.get("segments", segments)
                speakers = {s.get("speaker") for s in segments if s.get("speaker")}
                speaker_count = len(speakers)
            except Exception as e:
                print(f"    diarization failed: {e}")

        transcript = _segments_to_text(segments)

        # 4. Optional translate pass (only for non-English)
        english_translation = None
        if language and language != "en" and transcript.strip():
            try:
                t_result = whisper_model.transcribe(
                    audio, batch_size=8, task="translate"
                )
                english_translation = (
                    " ".join(
                        s.get("text", "").strip() for s in t_result.get("segments", [])
                    ).strip()
                    or None
                )
            except Exception as e:
                english_translation = f"[translation failed: {e}]"

        # Layer 2: canonical-name regex fixes (cheap safety net for what
        # Layer 1's initial_prompt missed). Applied to both transcript and
        # segments' per-line text so the diarized output stays consistent.
        if whisper_fixes:
            transcript = apply_whisper_fixes(transcript, whisper_fixes)
            if english_translation:
                english_translation = apply_whisper_fixes(
                    english_translation, whisper_fixes
                )
            for s in segments:
                if "text" in s:
                    s["text"] = apply_whisper_fixes(s["text"], whisper_fixes)

        return {
            "language": language,
            "transcript": transcript,
            "segments": segments,
            "english_translation": english_translation,
            "speaker_count": speaker_count,
        }
    finally:
        # Restore whisper options so per-clip bias doesn't leak to subsequent
        # clips (or to other functions sharing this model instance).
        try:
            whisper_model.options.initial_prompt = prev_prompt
            whisper_model.options.hotwords = prev_hotwords
        except Exception:
            pass
        audio_path.unlink(missing_ok=True)


def _empty_transcript(language: str = "") -> dict[str, Any]:
    return {
        "language": language,
        "transcript": "",
        "segments": [],
        "english_translation": None,
        "speaker_count": 0,
    }


def _segments_to_text(segments: list[dict[str, Any]]) -> str:
    """Render segments as 'Speaker N (HH:MM:SS): text' lines."""
    lines = []
    last_speaker = None
    buf: list[str] = []
    buf_start = None
    for seg in segments:
        speaker = seg.get("speaker") or None
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start", 0)
        if speaker != last_speaker:
            if buf:
                lines.append(_format_block(last_speaker, buf_start, " ".join(buf)))
            buf = [text]
            buf_start = start
            last_speaker = speaker
        else:
            buf.append(text)
    if buf:
        lines.append(_format_block(last_speaker, buf_start, " ".join(buf)))
    return "\n".join(lines)


def _format_block(speaker: str | None, start: float | None, text: str) -> str:
    ts = _fmt_time(start) if start is not None else ""
    if speaker:
        return f"[{speaker}] ({ts}) {text}"
    return f"({ts}) {text}" if ts else text


def _fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Vision description
# ---------------------------------------------------------------------------


def _build_vision_prompt(
    frames: list[Path],
    context: dict[str, Any],
    folder_context: str,
    include_paths: bool,
) -> str:
    """Shared prompt builder. include_paths=True for CLI mode (Claude Code reads
    via its Read tool); False for direct API / local where images go as
    content blocks. Asks for a YAML structured block + a prose description."""
    transcript_snippet = (context.get("transcript") or "")[:800]
    translation_snippet = (context.get("english_translation") or "")[:800]
    speech_block = "[no speech detected]"
    if transcript_snippet:
        lang = context.get("language") or "unknown"
        speech_block = f"({lang}):\n{transcript_snippet}"
        if translation_snippet and lang != "en":
            speech_block += f"\n\nEnglish:\n{translation_snippet}"

    location_line = ""
    loc = context.get("location") or {}
    if loc.get("place"):
        location_line = f"Location: {loc['place']} ({loc.get('lat')}, {loc.get('lon')})"
    elif loc.get("lat") is not None:
        location_line = f"GPS: {loc['lat']}, {loc['lon']}"

    folder_block = ""
    if folder_context:
        folder_block = f"\nDrive/folder context (use this to interpret the scene):\n{folder_context}\n"

    intro = (
        f"Read these {len(frames)} JPEG frames in order, then analyze the video clip."
        if include_paths
        else f"Analyze this short video clip based on these {len(frames)} frames "
        "(evenly sampled across the clip)."
    )

    paths_block = ""
    if include_paths:
        paths_lines = "\n".join(str(f) for f in frames)
        paths_block = f"\nFrames (read each one in order):\n{paths_lines}\n"

    long_clip = context.get("duration_seconds", 0) >= 30

    return textwrap.dedent(f"""
    {intro}
    {paths_block}
    File: {context["filename"]}
    Parent folder: {context["parent_folder"]}
    Duration: {context["duration_seconds"]:.1f}s
    Creation date: {context.get("creation_time") or "unknown"}
    {location_line}
    Speech: {speech_block}
    {folder_block}
    Produce TWO blocks in this exact order:

    BLOCK 1 — a YAML code fence with structured assessment fields:

    ```yaml
    rating: keep | review | cull
    cull_reason: ""    # short reason if cull; blank otherwise
    technical:
      focus: sharp | acceptable | soft
      exposure: strong | adequate | poor | clipped
      stability: smooth | handheld | jittery
      motion_blur: clean | some | heavy
    lighting: golden_hour | bright_daylight | overcast | dim_interior | nighttime | mixed | unclear
    time_of_day: predawn | dawn_morning | midday | afternoon | golden_hour | dusk | night | unclear
    dominant_color_palette: "short descriptive phrase, e.g. 'warm savanna: amber, ochre, dusty olive'"
    dominant_colors: [color1, color2, color3]   # 3-5 named colors, lowercase, hyphenated
    audio_quality: clean_speech | ambient | wind_noise | music | silent | unclear
    people_count: 0    # integer 0-99; for crowds, estimate to nearest 5; cap at 99 for very large crowds (always an integer, never a string)
    keywords: [tag1, tag2, tag3, tag4, tag5]    # 5-10 short lowercase tags
    notable_timestamp: ""  # "MM:SS" of peak moment if duration ≥ 30s; blank otherwise{" (clip is long enough — fill this in)" if long_clip else ""}
    ```

    BLOCK 2 — a prose description in this exact markdown structure:

    **Scene:** One sentence describing the setting (where, time of day, lighting).
    **Subjects:** Who or what is in the frames (count + role + activity). Do not
    guess identities of specific people; describe generically (e.g., "site
    supervisor in hi-vis vest" not "John").
    **Action:** What's happening across the clip (movement, change between frames).
    **Mood:** Emotional or atmospheric tone.
    **Shot type:** Wide / medium / close-up / drone aerial / handheld / static / POV.
    **Use cases:** 2-3 short bullets — what kind of content this clip would suit.

    HOW TO BE ACCURATE (read this carefully — it affects both blocks):
    - Describe ONLY what you can clearly see in the frames. If a detail is
      ambiguous (color of a small or moving object, who is holding the camera,
      whether the camera is mounted vs handheld), use "unclear" in the YAML or
      mark with "(unclear)" in prose. Do NOT invent.
    - If the frames are motion-blurred, dark, low-resolution, or otherwise
      hard to read: rating should be "review" or "cull" depending on severity,
      and Scene should say "Frames are X; details unclear."
    - Do not infer fictional details from streetlights, reflections, or
      headlights (e.g., guessing a vehicle's color from a glint).
    - Prefer "handheld"/"POV" over "mounted" unless framing is obviously
      stable in a way consistent with a rig.
    - Rating philosophy: this is a PERSONAL VIDEO ARCHIVE — clips are
      memories of real moments, not artistic photo portfolios. Imperfect
      handheld footage, motion blur, soft focus, camera shake, low light,
      and "vibe-y" rough capture are NOT cull reasons. A motion-blurred
      nighttime motorcycle clip is a memory worth keeping.
    - Cull rating is ONLY for clips that are not real recordings:
        * Lens cap on, pocket recording (no visible subject at all)
        * Black/unviewable/corrupted clips
        * Ground-only or ceiling-only accidental angles
        * Test clips under 2 seconds with no real content
        * Clearly clipped/blown exposure where nothing is visible
      The bar for cull is HIGH. If a clip captured a real moment, even
      badly, default to "keep". When in genuine doubt between cull and
      review, choose "review". When in doubt between review and keep,
      default to "keep".
    - Cull_reason should reflect ONLY these "not a real recording" reasons.
      Do NOT cull for motion blur, soft focus, shaky handheld, low light,
      or imperfect framing.
    - Keep rating is for: any clip that captured a real moment, even
      if technically imperfect. The vast majority of clips should be "keep".
    - Review rating is for: clips where you genuinely can't tell what was
      being recorded — partially obstructed view, very brief, ambiguous
      content that might or might not be intentional. Use sparingly.
    - Better vague-but-correct than confident-but-wrong.

    OUTPUT FORMAT:
    1. The ```yaml fence with structured fields first
    2. Then "## Description" header
    3. Then the prose block

    Do not include preamble, commentary, or any text outside these two blocks.
    """).strip()


# ---------------------------------------------------------------------------
# Parse vision response → (yaml_dict, prose_description)
# ---------------------------------------------------------------------------

YAML_FENCE_RE = re.compile(r"```(?:yaml|yml)?\s*\n(.*?)\n```", re.S | re.I)
DESCRIPTION_RE = re.compile(r"##\s*Description\s*\n+(.+?)(?=\n##|\Z)", re.S | re.I)


def parse_vision_response(raw: str) -> tuple[dict[str, Any], str]:
    """Extract the YAML structured block and the prose Description from the
    model's response. Returns (yaml_dict, prose_description). Tolerant of
    minor formatting variations."""
    if not raw or raw.startswith("["):  # error sentinel from describe_frames_*
        return {}, raw

    # Pull the first ```yaml fence
    structured: dict[str, Any] = {}
    m = YAML_FENCE_RE.search(raw)
    if m:
        yaml_text = m.group(1)
        try:
            parsed = yaml.safe_load(yaml_text)
            if isinstance(parsed, dict):
                structured = parsed
        except yaml.YAMLError:
            structured = {}

    # Pull the prose Description section
    prose = ""
    m2 = DESCRIPTION_RE.search(raw)
    if m2:
        prose = m2.group(1).strip()
    else:
        # Fall back: anything outside the yaml fence
        if YAML_FENCE_RE.search(raw):
            prose = YAML_FENCE_RE.sub("", raw).strip()
        else:
            prose = raw.strip()
    return structured, prose


def describe_frames_api(
    client: anthropic.Anthropic,
    frames: list[Path],
    context: dict[str, Any],
    folder_context: str,
    model_id: str,
) -> str:
    """Direct Anthropic API call. Sends images as base64 content blocks."""
    if not frames:
        return "[no frames extracted]"

    content: list[dict[str, Any]] = []
    for f in frames:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(f.read_bytes()).decode(),
                },
            }
        )
    prompt = _build_vision_prompt(frames, context, folder_context, include_paths=False)
    content.append({"type": "text", "text": prompt})

    msg = client.messages.create(
        model=model_id,
        max_tokens=800,
        messages=[{"role": "user", "content": cast(Any, content)}],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", "")).strip()
    return "[no description returned]"


def describe_frames_local(
    frames: list[Path],
    context: dict[str, Any],
    folder_context: str,
    base_url: str,
    model_name: str | None,
    timeout_sec: int = LOCAL_TIMEOUT_SEC,
) -> str:
    """LM Studio (or any OpenAI-compatible local server) vision call.

    Uses OpenAI's image_url content block format, which differs from
    Anthropic's image/base64 format. Same prompt content, different envelope.
    """
    if not frames:
        return "[no frames extracted]"

    # OpenAI vision content blocks: image_url with data URI for inline images.
    content: list[dict[str, Any]] = []
    for f in frames:
        b64 = base64.b64encode(f.read_bytes()).decode()
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )
    prompt = _build_vision_prompt(frames, context, folder_context, include_paths=False)
    content.append({"type": "text", "text": prompt})

    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 800,
        "temperature": 0.3,  # lower temperature → less confabulation
    }
    if model_name:
        payload["model"] = model_name
    else:
        # LM Studio accepts "loaded-model" or the actual loaded id; let it pick.
        payload["model"] = "loaded-model"

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout_sec,
        )
    except requests.exceptions.ConnectionError:
        return (
            f"[local backend unreachable at {base_url}. "
            "Is LM Studio running with a vision model loaded?]"
        )
    except requests.exceptions.Timeout:
        return f"[local backend timed out after {timeout_sec}s]"
    except Exception as e:
        return f"[local backend error: {e}]"

    if not resp.ok:
        return f"[local backend HTTP {resp.status_code}: {resp.text[:300]}]"

    try:
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return f"[local backend returned no choices: {json.dumps(data)[:300]}]"
        msg = choices[0].get("message") or {}
        text = msg.get("content")
        if isinstance(text, list):
            # Some servers return content as list of blocks
            for block in text:
                if isinstance(block, dict):
                    t = block.get("text") or block.get("content")
                    if isinstance(t, str) and t.strip():
                        return t.strip()
            return json.dumps(text)[:1000]
        if isinstance(text, str) and text.strip():
            return text.strip()
        return f"[local backend returned empty content: {json.dumps(data)[:300]}]"
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        return f"[local backend response parse error: {e}, body={resp.text[:300]}]"


def check_local_endpoint(base_url: str) -> tuple[bool, str]:
    """Quick reachability probe. Returns (ok, message_or_loaded_model_name)."""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=5)
        if r.ok:
            data = r.json()
            models = data.get("data") or []
            if not models:
                return False, "no models loaded in LM Studio"
            ids = [m.get("id", "?") for m in models]
            return True, f"loaded: {', '.join(ids)}"
        return False, f"HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"connection refused at {base_url} (LM Studio not running?)"
    except Exception as e:
        return False, str(e)


def describe_frames_cli(
    frames: list[Path],
    context: dict[str, Any],
    folder_context: str,
    model_id: str,
    timeout_sec: int = 180,
) -> str:
    """Shell out to `claude -p` (Claude Code CLI) using Max subscription auth.

    Critically, we strip ANTHROPIC_API_KEY from the subprocess env. If it were
    present, the CLI would bill against API quota instead of Max OAuth.
    """
    if not frames:
        return "[no frames extracted]"

    prompt = _build_vision_prompt(frames, context, folder_context, include_paths=True)

    env = os.environ.copy()
    # Force Max OAuth path — the API key would otherwise take precedence.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_API_KEY", None)

    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model_id,
        "--output-format",
        "json",
        # Headless mode: auto-approve tool uses (Claude Code needs Read to open
        # the frame JPEGs we created in /tmp). Without this, the CLI returns
        # "I need permission to read the image frames" as the response text.
        "--permission-mode",
        "bypassPermissions",
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return f"[CLI timed out after {timeout_sec}s]"

    if result.returncode != 0:
        return (
            f"[CLI error: rc={result.returncode}, stderr={result.stderr.strip()[:300]}]"
        )

    stdout = result.stdout.strip()
    if not stdout:
        return f"[CLI returned empty output, stderr={result.stderr.strip()[:300]}]"

    # JSON output mode wraps the response. Be tolerant of shape variation
    # across CLI versions.
    text_out: str | None = None
    try:
        data = json.loads(stdout)
        for key in ("result", "response", "content", "text", "output"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                text_out = v.strip()
                break
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        t = item.get("text") or item.get("content")
                        if isinstance(t, str) and t.strip():
                            text_out = t.strip()
                            break
                if text_out:
                    break
        if text_out is None:
            text_out = json.dumps(data)[:1000]
    except json.JSONDecodeError:
        # CLI may have returned raw text despite --output-format json
        text_out = stdout

    # Guard against permission-denied responses being silently accepted.
    # The CLI exits 0 with text like "I need permission to read..." when a
    # tool use is blocked. Treat these as errors so we don't write useless
    # sidecars.
    if is_permission_denied(text_out):
        return f"[CLI permission-denied response: {text_out[:300]}]"

    return text_out


# ---------------------------------------------------------------------------
# Sidecar write
# ---------------------------------------------------------------------------


def write_sidecar(
    video: Path,
    root: Path,
    metadata: dict[str, Any],
    gps: dict[str, Any],
    place: str,
    audio: dict[str, Any],
    description: str,
    structured: dict[str, Any],
    faces: list[face_db.DetectedFace],
) -> Path:
    sidecar = sidecar_path(video)
    parent = video.parent.name
    transcript = audio.get("transcript") or "[no speech detected]"

    # Build the frontmatter as a single Python dict, then serialize via PyYAML.
    # This gives us robust YAML quoting/escaping for free.
    # Store the path relative to the scan root so sidecars stay portable: the
    # archive can be moved or remounted at a different mountpoint without every
    # sidecar going stale. Downstream tools (query, master_index) resolve it
    # back against their own --root.
    fm: dict[str, Any] = {
        "file": video.name,
        "path": str(video.relative_to(root)),
        "parent_folder": parent,
        "duration_seconds": round(metadata["duration_seconds"], 1),
        "resolution": f"{metadata.get('width')}x{metadata.get('height')}",
        "codec": metadata.get("codec") or "",
        "size_bytes": metadata["size_bytes"],
        "creation_time": metadata.get("creation_time") or "",
    }
    if gps.get("lat") is not None:
        loc: dict[str, Any] = {
            "lat": gps["lat"],
            "lon": gps["lon"],
        }
        if gps.get("altitude_m") is not None:
            loc["altitude_m"] = gps["altitude_m"]
        if place:
            loc["place"] = place
        fm["location"] = loc

    fm["language_detected"] = audio.get("language") or ""
    fm["speaker_count"] = audio.get("speaker_count", 0)

    # Merge structured fields from vision (with safe defaults)
    fm["rating"] = structured.get("rating") or "review"
    fm["cull_reason"] = structured.get("cull_reason") or ""
    fm["technical"] = structured.get("technical") or {
        "focus": "unclear",
        "exposure": "unclear",
        "stability": "unclear",
        "motion_blur": "unclear",
    }
    fm["lighting"] = structured.get("lighting") or "unclear"
    fm["time_of_day"] = structured.get("time_of_day") or "unclear"
    fm["dominant_color_palette"] = structured.get("dominant_color_palette") or ""
    fm["dominant_colors"] = structured.get("dominant_colors") or []
    fm["audio_quality"] = structured.get("audio_quality") or "unclear"
    fm["people_count"] = coerce_people_count(
        structured.get("people_count", 0), len(faces)
    )
    fm["keywords"] = structured.get("keywords") or []
    fm["notable_timestamp"] = structured.get("notable_timestamp") or ""

    # Face detection results (from face_db, separate from people_count which is
    # the vision model's estimate of how many people are in frame)
    if faces:
        fm["faces"] = [f.to_sidecar_dict() for f in faces]
        fm["face_count"] = len(faces)
    else:
        fm["faces"] = []
        fm["face_count"] = 0

    fm["indexed_at"] = datetime.now().isoformat(timespec="seconds")

    frontmatter_text = yaml.safe_dump(
        fm, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).rstrip()

    body_parts = [
        "---",
        frontmatter_text,
        "---",
        "",
        f"# {video.name}",
        "",
        "## Description",
        "",
        description,
        "",
        f"## Transcript ({audio.get('language') or 'none'}, {audio.get('speaker_count', 0)} speakers)",
        "",
        transcript,
    ]
    if audio.get("english_translation"):
        body_parts += [
            "",
            "## English translation",
            "",
            audio["english_translation"],
        ]
    sidecar.write_text("\n".join(body_parts) + "\n")
    return sidecar


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def resolve_anthropic_key() -> str | None:
    """Return API key if available; None if not. Caller decides if it's required."""
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env.strip()
    key_path = Path.home() / ".claude" / "credentials" / "anthropic-key.txt"
    if key_path.exists():
        return key_path.read_text().strip()
    return None


def check_claude_cli() -> bool:
    """Verify `claude` CLI is on PATH and responds to --version."""
    import shutil

    if not shutil.which("claude"):
        return False
    try:
        r = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=10
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def resolve_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Index videos in a folder tree.")
    parser.add_argument("root", help="Root folder to scan recursively")
    parser.add_argument(
        "--force", action="store_true", help="Re-process clips even if a sidecar exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be processed; no API/model calls",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Stop after N files (for testing)"
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=30,
        help="Skip clips longer than N minutes (default: 30; 0 = no limit). "
        "Useful for mixed folders that contain movies/long recordings "
        "you don't want to transcribe.",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3-turbo",
        help="Whisper model: tiny/base/small/medium/large-v3/large-v3-turbo",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization (faster, no HF_TOKEN needed)",
    )
    parser.add_argument(
        "--no-geocode",
        action="store_true",
        help="Skip Nominatim reverse geocoding (GPS still recorded)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path substring to exclude (repeatable)",
    )
    parser.add_argument(
        "--backend",
        choices=["cli", "api", "local"],
        default="cli",
        help="Vision backend. 'cli' shells out to `claude -p` "
        "(uses Max subscription, $0 marginal cost). "
        "'api' uses anthropic SDK with ANTHROPIC_API_KEY "
        "(faster, ~$0.002/clip via Haiku). "
        "'local' uses LM Studio (or any OpenAI-compatible "
        "endpoint) at --local-base-url ($0, fully offline). "
        "Default: cli.",
    )
    parser.add_argument(
        "--vision-model",
        choices=list(VISION_MODELS.keys()),
        default=VISION_MODEL_DEFAULT,
        help="Claude vision model when --backend cli or api. "
        "Ignored for --backend local. 'haiku' (default) "
        "or 'sonnet' (~4x cost via api, better accuracy "
        "on motion/low-light/ambiguous clips).",
    )
    parser.add_argument(
        "--local-base-url",
        default=DEFAULT_LOCAL_BASE_URL,
        help=f"OpenAI-compatible base URL for --backend local "
        f"(default: {DEFAULT_LOCAL_BASE_URL}). LM Studio "
        "exposes /v1 by default.",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Model name to send to --backend local. If unset, "
        "uses whatever model LM Studio has loaded. Pass "
        "the exact loaded model id (e.g. "
        "'google/gemma-4-26b') if your server has multiple.",
    )
    parser.add_argument(
        "--no-faces",
        action="store_true",
        help="Skip face detection step (faster; no embeddings "
        "captured for later fdx-faces clustering).",
    )
    parser.add_argument(
        "--face-db",
        default=str(face_db.DB_PATH_DEFAULT),
        help=f"Path to face DB SQLite file (default: {face_db.DB_PATH_DEFAULT}).",
    )
    parser.add_argument(
        "--no-whisper-prompt",
        action="store_true",
        help="Disable per-clip Whisper proper-noun biasing. "
        "By default the indexer reads `**Whisper proper "
        "nouns:**` lines from .video-context.md files and "
        "passes the union as initial_prompt + hotwords to "
        "Whisper so names get spelled right.",
    )
    parser.add_argument(
        "--whisper-fixes",
        default=str(WHISPER_FIXES_DEFAULT),
        help=f"Path to JSON file with Layer-2 canonical-name "
        f"regex fixes applied post-Whisper (default: "
        f"{WHISPER_FIXES_DEFAULT}). Empty / missing file = "
        f"no fixes. Schema: "
        f'{{"fixes": [{{"pattern": "...", "replace": "..."}}]}}.',
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        sys.exit(f"Root path is not a directory: {root}")

    print(f"Scanning {root}")
    all_videos = find_videos(root, args.exclude)
    print(f"  found {len(all_videos)} video files")

    todo = all_videos if args.force else [v for v in all_videos if not has_sidecar(v)]
    skipped = len(all_videos) - len(todo)
    if skipped:
        print(f"  skipping {skipped} already-indexed")
    if args.max_files:
        todo = todo[: args.max_files]
        print(f"  limiting to {len(todo)} for this run")

    if not todo:
        print("Nothing to do.")
        return 0

    model_cfg = VISION_MODELS[args.vision_model]
    if args.backend == "api":
        model_id = str(model_cfg["api"])
        cost_per_call = float(model_cfg["cost_per_call_api"])
    elif args.backend == "cli":
        model_id = str(model_cfg["cli"])
        cost_per_call = COST_PER_CALL_USD_CLI
    else:  # local
        model_id = args.local_model or "(loaded model in LM Studio)"
        cost_per_call = COST_PER_CALL_USD_LOCAL

    est_cost = len(todo) * cost_per_call
    if args.backend == "api":
        print(f"  vision: api / {model_id}")
        print(f"  estimated Anthropic API cost: ~${est_cost:.2f}")
    elif args.backend == "cli":
        print(f"  vision: cli (Max) / {model_id}")
        print("  marginal cost: $0 (Max subscription)")
    else:
        print(f"  vision: local / {model_id} @ {args.local_base_url}")
        print("  marginal cost: $0 (fully local)")
    print()

    if args.dry_run:
        for v in todo:
            print(f"  would process: {v.relative_to(root)}")
        return 0

    # Drive-level context is loaded per-clip via load_context_for_clip() which
    # walks up the tree and layers contexts. Just inform the user if any
    # .video-context.md files exist anywhere under root.
    ctx_files = list(root.rglob(CONTEXT_FILE))
    if ctx_files:
        print(
            f"  found {len(ctx_files)} {CONTEXT_FILE} file(s) — will layer "
            "per-clip context from drive root → trip → subfolder\n"
        )

    # Backend wiring
    api_client = None
    if args.backend == "api":
        api_key = resolve_anthropic_key()
        if not api_key:
            sys.exit(
                "--backend api requires ANTHROPIC_API_KEY env or "
                "~/.claude/credentials/anthropic-key.txt"
            )
        import anthropic as _anthropic

        api_client = _anthropic.Anthropic(api_key=api_key)
        print("Vision: direct Anthropic API\n")
    elif args.backend == "cli":
        if not check_claude_cli():
            sys.exit(
                "--backend cli requires the `claude` CLI on PATH. "
                "Install Claude Code or pass --backend api with ANTHROPIC_API_KEY."
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "NOTE: ANTHROPIC_API_KEY is set in your environment. The script will\n"
                "      explicitly remove it from each `claude` subprocess so calls go\n"
                "      against your Max subscription instead of API billing.\n"
            )
        else:
            print("Vision: claude CLI -> Max subscription\n")
    else:  # local
        ok, info = check_local_endpoint(args.local_base_url)
        if not ok:
            sys.exit(
                f"--backend local: cannot reach {args.local_base_url} ({info}). "
                "Start LM Studio and load a vision-capable model first."
            )
        print(f"Vision: local LM Studio at {args.local_base_url} ({info})\n")

    # Face detection setup
    face_conn = None
    if not args.no_faces:
        ok, info = face_db.init_face_app()
        if not ok:
            print(f"Face detection unavailable: {info}")
            print("Continuing without face detection. Pass --no-faces to silence.")
        else:
            face_conn = face_db.open_db(Path(args.face_db))
            stats = face_db.db_stats(face_conn)
            print(f"Face detection: insightface ({info})")
            print(
                f"Face DB: {args.face_db} (currently {stats['faces']} faces, "
                f"{stats['clusters']} clusters, {stats['named_clusters']} named)\n"
            )

    print(f"Loading Whisper model: {args.whisper_model}")
    whisper_model = whisperx.load_model(
        args.whisper_model, device="cpu", compute_type="int8"
    )
    print("Whisper ready.")

    # Load Layer-2 canonical-name fixes once at startup
    whisper_fixes: list[tuple[re.Pattern[str], str]] = []
    if not args.no_whisper_prompt:
        whisper_fixes = load_whisper_fixes(Path(args.whisper_fixes).expanduser())
        if whisper_fixes:
            print(
                f"Whisper canonical fixes: loaded {len(whisper_fixes)} rule(s) "
                f"from {args.whisper_fixes}"
            )
        else:
            print(f"Whisper canonical fixes: none (no rules at {args.whisper_fixes})")

    align_models: dict[str, Any] = {}

    diarize_pipeline = None
    if not args.no_diarize:
        hf = resolve_hf_token()
        if not hf:
            print(
                "HF_TOKEN not set — running without diarization. Pass --no-diarize to silence this notice."
            )
        else:
            # WhisperX 3.3.4+ moved DiarizationPipeline to the diarize submodule.
            # WhisperX 3.8+ renamed the auth kwarg from use_auth_token → token
            # (inherited from pyannote-audio 3.x). Introspect to pick the right
            # one so this survives future shuffles too.
            DiarizationPipeline = None
            try:
                from whisperx.diarize import DiarizationPipeline as _DP

                DiarizationPipeline = _DP
            except ImportError:
                DiarizationPipeline = getattr(whisperx, "DiarizationPipeline", None)
            if DiarizationPipeline is None:
                print(
                    "DiarizationPipeline not found in this whisperx version. "
                    "Continuing without diarization. Update whisperx or run with --no-diarize to silence."
                )
            else:
                import inspect

                try:
                    params = inspect.signature(DiarizationPipeline.__init__).parameters
                except (ValueError, TypeError):
                    params = {}  # type: ignore[assignment]
                auth_kwarg = pick_diar_auth_kwarg(params)
                try:
                    diarize_pipeline = DiarizationPipeline(
                        **{auth_kwarg: hf}, device="cpu"
                    )
                    print(f"Diarization pipeline ready (auth kwarg: {auth_kwarg}).")
                except TypeError:
                    # Last-resort: if the kwarg we picked is wrong, try the other one.
                    other = "use_auth_token" if auth_kwarg == "token" else "token"
                    try:
                        diarize_pipeline = DiarizationPipeline(
                            **{other: hf}, device="cpu"
                        )
                        print(
                            f"Diarization pipeline ready (auth kwarg: {other}, fallback)."
                        )
                    except Exception as e2:
                        print(f"Failed to load diarization pipeline: {e2}")
                        print(
                            "Continuing without diarization. (Did you accept terms on the pyannote model pages?)"
                        )
                except Exception as e:
                    print(f"Failed to load diarization pipeline: {e}")
                    print(
                        "Continuing without diarization. (Did you accept terms on the pyannote model pages?)"
                    )
    print()

    geocoder = NominatimRateLimiter() if not args.no_geocode else None

    processed = 0
    errors = 0
    skipped_too_long = 0
    actual_cost = 0.0
    max_duration_seconds = args.max_duration * 60 if args.max_duration > 0 else None
    for i, video in enumerate(todo, start=1):
        rel = video.relative_to(root)
        print(f"[{i}/{len(todo)}] {rel}")
        try:
            metadata = get_metadata(video)
            if metadata["duration_seconds"] < 0.5:
                print("  skipped (duration < 0.5s)")
                continue
            if (
                max_duration_seconds
                and metadata["duration_seconds"] > max_duration_seconds
            ):
                mins = metadata["duration_seconds"] / 60
                print(
                    f"  skipped (duration {mins:.1f} min > --max-duration {args.max_duration} min)"
                )
                skipped_too_long += 1
                continue

            gps = get_gps(video)
            place = ""
            if gps.get("lat") is not None and geocoder is not None:
                place = geocoder.reverse(gps["lat"], gps["lon"])
                if place:
                    print(f"  location: {place}")

            # Gather proper nouns from the .video-context.md chain for this clip
            # (drive root → trip → subfolder, unioned + deduped). Skipped if
            # --no-whisper-prompt was passed.
            clip_proper_nouns: list[str] = []
            if not args.no_whisper_prompt:
                clip_proper_nouns = load_proper_nouns_for_clip(video, root)

            audio = transcribe_audio_whisperx(
                video,
                whisper_model,
                align_models,
                diarize_pipeline,
                proper_nouns=clip_proper_nouns,
                whisper_fixes=whisper_fixes,
            )
            if audio.get("speaker_count"):
                print(
                    f"  transcribed ({audio['language']}, {audio['speaker_count']} speakers)"
                )
            elif audio.get("language"):
                print(f"  transcribed ({audio['language']})")

            # Frames need to outlive the temp dir when we're using the CLI
            # backend (claude subprocess reads them by path). Use a per-clip
            # persistent tmpdir we clean up after the vision call returns.
            tmp_frames = Path(tempfile.mkdtemp(prefix="fdx-frames-"))
            try:
                frames = extract_frames(video, tmp_frames, num_frames=5)
                # Compute frame timestamps to pass to face detection
                duration = metadata["duration_seconds"]
                num = len(frames)
                frame_timestamps = (
                    [duration * (i + 1) / (num + 1) for i in range(num)] if num else []
                )

                context = {
                    "filename": video.name,
                    "parent_folder": video.parent.name,
                    "duration_seconds": metadata["duration_seconds"],
                    "creation_time": metadata.get("creation_time", ""),
                    "language": audio.get("language"),
                    "transcript": audio.get("transcript", ""),
                    "english_translation": audio.get("english_translation"),
                    "location": {**gps, "place": place} if gps else {"place": place},
                }

                # Load layered context (drive root → trip → shoot)
                clip_context = load_context_for_clip(video, root)

                # Vision call returns raw text (yaml + prose); parse it
                if args.backend == "api":
                    assert api_client is not None
                    raw = describe_frames_api(
                        api_client, frames, context, str(clip_context), model_id
                    )
                elif args.backend == "cli":
                    raw = describe_frames_cli(
                        frames, context, str(clip_context), model_id
                    )
                    time.sleep(CLI_INTER_CALL_DELAY)  # be polite to Max TPM
                else:  # local
                    raw = describe_frames_local(
                        frames,
                        context,
                        str(clip_context),
                        args.local_base_url,
                        args.local_model,
                    )

                structured, description = parse_vision_response(raw)

                # Face detection on the same frames (free reuse)
                detected_faces: list[face_db.DetectedFace] = []
                if face_conn is not None and frames:
                    try:
                        detected_faces = face_db.detect_faces_in_frames(
                            frames, frame_timestamps
                        )
                    except Exception as e:
                        print(f"  face detection failed: {e}")
            finally:
                # Clean up frames
                for f in tmp_frames.glob("*"):
                    f.unlink(missing_ok=True)
                tmp_frames.rmdir()

            sidecar = write_sidecar(
                video,
                root,
                metadata,
                gps,
                place,
                audio,
                description,
                structured,
                detected_faces,
            )
            if face_conn is not None and detected_faces:
                face_db.write_faces(face_conn, video, sidecar, detected_faces)
            actual_cost += float(cost_per_call)
            processed += 1
            faces_note = f", {len(detected_faces)} faces" if detected_faces else ""
            rating_note = f", rated {structured.get('rating', '?')}"
            if args.backend == "api":
                print(
                    f"  -> {sidecar.name}  (cost ~${actual_cost:.2f}{rating_note}{faces_note})"
                )
            else:
                print(f"  -> {sidecar.name}  ({rating_note}{faces_note})")
        except KeyboardInterrupt:
            print(
                "\nInterrupted. Re-run to resume — finished sidecars are skipped automatically."
            )
            break
        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    summary = f"\nDone. Processed: {processed}, Errors: {errors}"
    if skipped_too_long:
        summary += f", Skipped (too long): {skipped_too_long}"
    if args.backend == "api":
        summary += f", Approx cost: ${actual_cost:.2f}"
    print(summary)
    if face_conn is not None:
        s = face_db.db_stats(face_conn)
        print(
            f"Face DB: {s['faces']} total faces, {s['clusters']} clusters "
            f"({s['named_clusters']} named). Next: run fdx-faces to label "
            "clusters (once it's built)."
        )
        face_conn.close()
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
