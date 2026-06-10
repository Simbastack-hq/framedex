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
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

from framedex import face_db, frame_sampling, images, pipeline, runner
from framedex.parsing import (
    coerce_people_count,
    pick_diar_auth_kwarg,
)
from framedex.pipeline import (
    CLI_INTER_CALL_DELAY,
    DEFAULT_LOCAL_BASE_URL,
    FRAME_MAX_WIDTH,
    VISION_MODEL_DEFAULT,
    VISION_MODELS,
    NominatimRateLimiter,
    ProcessContext,
    ProcessOptions,
    ProcessResult,
    describe_frames_api,
    describe_frames_cli,
    describe_frames_local,
    get_gps,
    has_sidecar,
    parse_vision_response,
    sidecar_path,
)

# Heavy ML deps (whisperx / torch) are imported lazily inside the transcribe +
# model-loading paths so an image-only `fdx --media images` run never needs the
# video stack installed. anthropic is imported lazily in the --backend api path.
# insightface is loaded lazily by face_db (heavy module).

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
CONTEXT_FILE = ".video-context.md"


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


def _extract_candidate_pool(
    video: Path, out_dir: Path, cand_ts: list[float]
) -> tuple[list[Path], list[float]]:
    """Low-res thumbnail per candidate timestamp via fast-seek. Quality 5 is
    plenty for histograms; keeps the pool I/O tiny."""
    thumbs: list[Path] = []
    kept_ts: list[float] = []
    for k, ts in enumerate(cand_ts):
        out = out_dir / f"cand_{k:03d}.jpg"
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
    return thumbs, kept_ts


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
        thumbs, kept_ts = _extract_candidate_pool(video, out_dir, cand_ts)
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
    import whisperx

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
    timestamps: list[float] | None = None,
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
    *,
    sidecar_path_override: Path | None = None,
    parent_folder_override: str | None = None,
    extra_frontmatter: dict[str, Any] | None = None,
    omit_path: bool = False,
) -> Path:
    sidecar = sidecar_path_override or sidecar_path(video)
    parent = (
        parent_folder_override
        if parent_folder_override is not None
        else video.parent.name
    )
    transcript = audio.get("transcript") or "[no speech detected]"

    # Build the frontmatter as a single Python dict, then serialize via PyYAML.
    # This gives us robust YAML quoting/escaping for free.
    # Store the path relative to the scan root so sidecars stay portable: the
    # archive can be moved or remounted at a different mountpoint without every
    # sidecar going stale. Downstream tools (query, master_index) resolve it
    # back against their own --root.
    fm: dict[str, Any] = {
        "file": video.name,
        "parent_folder": parent,
        "duration_seconds": round(metadata["duration_seconds"], 1),
        "resolution": f"{metadata.get('width')}x{metadata.get('height')}",
        "codec": metadata.get("codec") or "",
        "size_bytes": metadata["size_bytes"],
        "creation_time": metadata.get("creation_time") or "",
    }
    if not omit_path:
        try:
            path_field = str(video.relative_to(root))
        except ValueError:
            # video lives outside root (e.g. inside an Apple Photos library bundle
            # while sidecars mirror to a separate output tree). Fall back to the
            # absolute path so downstream tools can still locate the original.
            path_field = str(video)
        fm["path"] = path_field
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

    if extra_frontmatter:
        for k, v in extra_frontmatter.items():
            fm[k] = v

    body_sections: list[tuple[str, str]] = [
        ("Description", description),
        (
            f"Transcript ({audio.get('language') or 'none'}, "
            f"{audio.get('speaker_count', 0)} speakers)",
            transcript,
        ),
    ]
    if audio.get("english_translation"):
        body_sections.append(("English translation", audio["english_translation"]))

    return pipeline.serialize_sidecar(sidecar, fm, video.name, body_sections)


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------


def resolve_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


# ---------------------------------------------------------------------------
# Per-clip pipeline (callable from fdx and fdx-photos)
# ---------------------------------------------------------------------------


def process_one_video(
    video: Path,
    root: Path,
    opts: ProcessOptions,
    ctx: ProcessContext,
    *,
    sidecar_path_override: Path | None = None,
    parent_folder_override: str | None = None,
    metadata_override: dict[str, Any] | None = None,
    gps_override: dict[str, Any] | None = None,
    place_override: str | None = None,
    extra_frontmatter: dict[str, Any] | None = None,
    omit_path: bool = False,
    proper_nouns: list[str] | None = None,
) -> ProcessResult:
    """Run the full per-clip pipeline for one video and emit a sidecar.

    Overrides let callers (e.g. fdx-photos) inject metadata from a richer
    source (Photos.sqlite) instead of probing the file again, and redirect
    sidecar output to a mirror tree outside the original's directory.
    """
    metadata = get_metadata(video)
    if metadata_override:
        # Caller (e.g. fdx-photos) has authoritative fields from a richer source.
        # Shallow-merge over ffprobe results so fields like creation_time can be
        # corrected without losing duration/codec/resolution.
        metadata.update(metadata_override)
    if metadata["duration_seconds"] < 0.5:
        return ProcessResult(sidecar=None, skipped_reason="short")
    if (
        opts.max_duration_seconds
        and metadata["duration_seconds"] > opts.max_duration_seconds
    ):
        return ProcessResult(sidecar=None, skipped_reason="too_long")

    gps = gps_override if gps_override is not None else get_gps(video)
    if place_override is not None:
        place = place_override
    elif gps.get("lat") is not None and ctx.geocoder is not None:
        place = ctx.geocoder.reverse(gps["lat"], gps["lon"])
        if place:
            print(f"  location: {place}")
    else:
        place = ""

    clip_proper_nouns: list[str] = proper_nouns if proper_nouns is not None else []
    if proper_nouns is None and not opts.no_whisper_prompt:
        clip_proper_nouns = load_proper_nouns_for_clip(video, root)

    audio = transcribe_audio_whisperx(
        video,
        ctx.whisper_model,
        ctx.align_models,
        ctx.diarize_pipeline,
        proper_nouns=clip_proper_nouns,
        whisper_fixes=opts.whisper_fixes,
    )
    if audio.get("speaker_count"):
        print(f"  transcribed ({audio['language']}, {audio['speaker_count']} speakers)")
    elif audio.get("language"):
        print(f"  transcribed ({audio['language']})")

    tmp_frames = Path(tempfile.mkdtemp(prefix="fdx-frames-"))
    detected_faces: list[face_db.DetectedFace] = []
    structured: dict[str, Any] = {}
    description: str = ""
    try:
        frame_pairs = extract_frames(
            video,
            tmp_frames,
            num_frames=5,
            duration=metadata["duration_seconds"],
            sampling=opts.frame_sampling,
        )
        frames = [p for p, _ in frame_pairs]
        frame_timestamps = [ts for _, ts in frame_pairs]

        context = {
            "filename": video.name,
            "parent_folder": parent_folder_override
            if parent_folder_override is not None
            else video.parent.name,
            "duration_seconds": metadata["duration_seconds"],
            "creation_time": metadata.get("creation_time", ""),
            "language": audio.get("language"),
            "transcript": audio.get("transcript", ""),
            "english_translation": audio.get("english_translation"),
            "location": {**gps, "place": place} if gps else {"place": place},
        }
        clip_context = load_context_for_clip(video, root)

        if opts.backend == "api":
            assert ctx.api_client is not None
            prompt = _build_vision_prompt(
                frames,
                context,
                str(clip_context),
                include_paths=False,
                timestamps=frame_timestamps,
            )
            raw = describe_frames_api(
                ctx.api_client, frames, prompt, opts.vision_model_id
            )
        elif opts.backend == "cli":
            prompt = _build_vision_prompt(
                frames,
                context,
                str(clip_context),
                include_paths=True,
                timestamps=frame_timestamps,
            )
            raw = describe_frames_cli(frames, prompt, opts.vision_model_id)
            time.sleep(CLI_INTER_CALL_DELAY)
        else:  # local
            prompt = _build_vision_prompt(
                frames,
                context,
                str(clip_context),
                include_paths=False,
                timestamps=frame_timestamps,
            )
            raw = describe_frames_local(
                frames, prompt, opts.local_base_url, opts.local_model
            )

        structured, description = parse_vision_response(raw)

        if ctx.face_conn is not None and frames:
            try:
                detected_faces = face_db.detect_faces_in_frames(
                    frames, frame_timestamps
                )
            except Exception as e:
                print(f"  face detection failed: {e}")
    finally:
        for f in tmp_frames.glob("*"):
            f.unlink(missing_ok=True)
        tmp_frames.rmdir()

    # The vision backends return a "[...]" sentinel on failure (timeout, HTTP
    # error, permission-denied). Don't persist a sidecar for those — that would
    # mark the clip permanently "indexed" and silently skip it on every re-run.
    # Returning vision_error keeps the clip in the todo list next time.
    if not structured and description.startswith("["):
        print(f"  vision call failed: {description[:200]}")
        return ProcessResult(sidecar=None, skipped_reason="vision_error")

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
        sidecar_path_override=sidecar_path_override,
        parent_folder_override=parent_folder_override,
        extra_frontmatter=extra_frontmatter,
        omit_path=omit_path,
    )
    if ctx.face_conn is not None and detected_faces:
        face_db.write_faces(ctx.face_conn, video, sidecar, detected_faces)

    return ProcessResult(
        sidecar=sidecar,
        detected_faces=detected_faces,
        cost=opts.cost_per_call,
        skipped_reason=None,
        rating=str(structured.get("rating", "?")),
        structured=structured,
    )


def setup_whisper(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], Any, list[tuple[re.Pattern[str], str]]]:
    """Load the whisper model, Layer-2 canonical fixes, and (optionally)
    the diarization pipeline. Only called when the run actually has videos
    to process, so an image-only run never imports whisperx/torch."""
    try:
        import whisperx
    except ImportError as e:
        sys.exit(
            "Video indexing needs the 'video' extra (whisperx/torch). Install it "
            f"with: uv pip install -e '.[video]'  (or '.[all]')  ({e})"
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
    return whisper_model, align_models, diarize_pipeline, whisper_fixes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index videos and photos in a folder tree."
    )
    parser.add_argument("root", help="Root folder to scan recursively")
    parser.add_argument(
        "--media",
        choices=["all", "videos", "images"],
        default="all",
        help="Which media to index: 'all' (default), 'videos' only, or "
        "'images' only. An image-only run skips the whisper/audio stack "
        "entirely (no torch/whisperx needed).",
    )
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
    parser.add_argument(
        "--frame-sampling",
        choices=["diverse", "even"],
        default="diverse",
        help="How vision frames are picked. 'diverse' (default) samples "
        "small thumbnails across the clip and keeps the most mutually "
        "different, sharpest moments. 'even' is the legacy evenly-spaced "
        "sampling.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        sys.exit(f"Root path is not a directory: {root}")

    print(f"Scanning {root}")

    # Enumerate each requested media type, drop already-indexed (unless --force),
    # and tag each survivor with its kind so the loop can route it. Both kinds
    # share the suffix-append sidecar scheme, so resume works identically.
    todo: list[tuple[Path, str]] = []
    n_found = 0
    if args.media in ("all", "videos"):
        vids = find_videos(root, args.exclude)
        n_found += len(vids)
        print(f"  found {len(vids)} video files")
        vtodo = vids if args.force else [v for v in vids if not has_sidecar(v)]
        todo += [(v, "video") for v in vtodo]
    if args.media in ("all", "images"):
        imgs = images.find_images(root, args.exclude)
        n_found += len(imgs)
        print(f"  found {len(imgs)} image files")
        itodo = imgs if args.force else [im for im in imgs if not has_sidecar(im)]
        todo += [(im, "image") for im in itodo]

    todo.sort(key=lambda t: t[0])
    skipped = n_found - len(todo)
    if skipped:
        print(f"  skipping {skipped} already-indexed")
    if args.max_files and len(todo) > args.max_files:
        todo = todo[: args.max_files]
        print(f"  limiting to {len(todo)} for this run")

    if not todo:
        print("Nothing to do.")
        return 0

    need_video = any(kind == "video" for _, kind in todo)

    model_id, cost_per_call = runner.resolve_vision_model(args)
    runner.announce_cost(
        args.backend, model_id, cost_per_call, len(todo), args.local_base_url
    )

    if args.dry_run:
        for path, kind in todo:
            print(f"  would process [{kind}]: {path.relative_to(root)}")
        return 0

    # Drive-level context is loaded per-clip via load_context_for_clip() which
    # walks up the tree and layers contexts. Just inform the user if any
    # .video-context.md files exist anywhere under root. (Video-only feature.)
    ctx_files = list(root.rglob(CONTEXT_FILE)) if need_video else []
    if ctx_files:
        print(
            f"  found {len(ctx_files)} {CONTEXT_FILE} file(s) — will layer "
            "per-clip context from drive root → trip → subfolder\n"
        )

    # Backend wiring + face detection setup (shared with fdx-photos via runner).
    api_client = runner.wire_vision_backend(args)
    face_conn = runner.setup_face_db(args)

    whisper_model = None
    align_models: dict[str, Any] = {}
    diarize_pipeline = None
    whisper_fixes: list[tuple[re.Pattern[str], str]] = []
    if need_video:
        whisper_model, align_models, diarize_pipeline, whisper_fixes = setup_whisper(
            args
        )

    geocoder = NominatimRateLimiter() if not args.no_geocode else None

    max_duration_seconds = args.max_duration * 60 if args.max_duration > 0 else None
    opts = ProcessOptions(
        backend=args.backend,
        vision_model_id=model_id,
        local_base_url=args.local_base_url,
        local_model=args.local_model,
        cost_per_call=float(cost_per_call),
        no_whisper_prompt=args.no_whisper_prompt,
        frame_sampling=args.frame_sampling,
        whisper_fixes=whisper_fixes,
        max_duration_seconds=max_duration_seconds,
    )
    ctx = ProcessContext(
        whisper_model=whisper_model,
        align_models=align_models,
        diarize_pipeline=diarize_pipeline,
        geocoder=geocoder,
        api_client=api_client,
        face_conn=face_conn,
    )

    tally = runner.RunTally()
    for i, (path, kind) in enumerate(todo, start=1):
        rel = path.relative_to(root)
        print(f"[{i}/{len(todo)}] {rel}")
        try:
            if kind == "video":
                result = process_one_video(path, root, opts, ctx)
            else:
                result = images.process_one_image(path, root, opts, ctx)
            runner.record_result(
                result, tally, backend=args.backend, max_duration_min=args.max_duration
            )
        except KeyboardInterrupt:
            print(
                "\nInterrupted. Re-run to resume — finished sidecars are skipped automatically."
            )
            break
        except Exception as e:
            tally.errors += 1
            print(f"  ERROR: {e}")
            traceback.print_exc()
            continue

    summary = f"\nDone. Processed: {tally.processed}, Errors: {tally.errors}"
    if tally.skipped_too_long:
        summary += f", Skipped (too long): {tally.skipped_too_long}"
    if tally.skipped_no_preview:
        summary += f", Skipped (no preview): {tally.skipped_no_preview}"
    if args.backend == "api":
        summary += f", Approx cost: ${tally.actual_cost:.2f}"
    print(summary)
    if face_conn is not None:
        s = face_db.db_stats(face_conn)
        print(
            f"Face DB: {s['faces']} total faces, {s['clusters']} clusters "
            f"({s['named_clusters']} named). Next: run fdx-faces to label "
            "clusters (once it's built)."
        )
        face_conn.close()
    return 0 if tally.errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
