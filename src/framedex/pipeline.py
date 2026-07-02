#!/usr/bin/env python3
"""
framedex.pipeline — media-agnostic shared core.

Everything here is used by BOTH the video pipeline (index_videos) and the
still-photo pipeline (images): GPS + reverse-geocoding, the three vision
backends (cli / api / local) that take a prebuilt prompt and a set of image
frames, the vision-response parser, key/CLI resolution, the per-run dataclasses,
and the YAML sidecar serializer.

Media-specific work (ffprobe/whisper for video; EXIF/preview for stills) stays
in the respective pipeline module, which builds its own prompt + frontmatter and
calls in here for the shared transport and serialization.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests
import yaml

from framedex import face_db
from framedex.parsing import is_permission_denied

if TYPE_CHECKING:
    import anthropic

SIDECAR_SUFFIX = ".description.md"

# Default vision models — overridable via --vision-model. Maps shorthand to the
# full IDs the API expects and the shorthand the CLI accepts.
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

# Frame / preview width cap — wide enough for the model to disambiguate small
# details, small enough that base64-encoding stays reasonable.
FRAME_MAX_WIDTH = 1920

COST_PER_CALL_USD_CLI = 0.0  # Max subscription: marginal cost is $0 to user
COST_PER_CALL_USD_LOCAL = 0.0  # Local model: $0, just electricity
USER_AGENT = "framedex/1.0 (personal archive indexer)"
CLI_INTER_CALL_DELAY = 0.4  # seconds, to be polite to Max TPM caps

DEFAULT_LOCAL_BASE_URL = "http://localhost:1234/v1"
LOCAL_TIMEOUT_SEC = 180


# ---------------------------------------------------------------------------
# Sidecar paths (shared: a sidecar sits next to its media file, suffix-appended)
# ---------------------------------------------------------------------------


def sidecar_path(media: Path) -> Path:
    """`clip.MOV` -> `clip.MOV.description.md`. The full original extension is
    preserved so two files sharing a stem but differing in extension (e.g.
    `DSC_1.RAF` and `DSC_1.JPG`) never collide on one sidecar."""
    return media.with_suffix(media.suffix + SIDECAR_SUFFIX)


def has_sidecar(media: Path) -> bool:
    return sidecar_path(media).exists()


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: a same-directory temp file, then
    `os.replace`. Readers and the resume check (`has_sidecar`) never observe a
    partial file, so a Ctrl-C / crash / disk-full mid-write can't leave a
    truncated sidecar or index that marks a file permanently indexed.

    The temp is same-directory (`os.replace` is only atomic within one
    filesystem), dot-prefixed (both discovery walkers skip hidden files, so a
    stale temp is invisible to discovery and `has_sidecar`), and pid-suffixed
    (two runs racing on the same root can't collide on one temp path). A stale
    `.<name>.<pid>.tmp` from a killed run is inert and safe to delete."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    # Explicit utf-8 (not the platform default) so the bytes on disk are
    # locale-independent — fdx-xmp hashes the payload it writes and compares it
    # to the file's bytes on re-run, which only holds if the encoding is fixed.
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# GPS + reverse geocoding
# ---------------------------------------------------------------------------


def get_gps(media: Path) -> dict[str, Any]:
    """exiftool -> lat/lon/alt. Returns {} if nothing usable. Works for any
    file exiftool understands (video container or still)."""
    cmd = [
        "exiftool",
        "-json",
        "-n",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-GPSCoordinates",
        "-LocationInformation",
        str(media),
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
    # Some files embed GPSCoordinates as "lat, lon, alt"
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


# ---------------------------------------------------------------------------
# Vision backends — each takes a prebuilt prompt + a list of image frames.
# The prompt is media-specific (built by the video or image pipeline); the
# transport (base64 encoding, API/CLI/local envelope, response unwrapping) is
# shared.
# ---------------------------------------------------------------------------


def describe_frames_api(
    client: anthropic.Anthropic,
    frames: list[Path],
    prompt: str,
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
    prompt: str,
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
    prompt: str,
    model_id: str,
    timeout_sec: int = 180,
) -> str:
    """Shell out to `claude -p` (Claude Code CLI) using Max subscription auth.

    Critically, we strip ANTHROPIC_API_KEY from the subprocess env. If it were
    present, the CLI would bill against API quota instead of Max OAuth. The
    prompt must reference the frame paths (the CLI reads them via its Read tool).
    """
    if not frames:
        return "[no frames extracted]"

    env = os.environ.copy()
    # Force Max OAuth path — the API key would otherwise take precedence.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDE_API_KEY", None)

    # The prompt embeds untrusted text (up to 800 chars of Whisper transcript
    # from arbitrary footage + user-editable .video-context.md). The vision task
    # needs exactly one capability: reading the frame JPEGs. Grant Read only.
    # `dontAsk` denies anything not on the allowlist without prompting, so an
    # injected "use Bash to run rm -rf" becomes a denied tool call surfaced as a
    # retryable vision_error — not an auto-approved Bash (the old
    # bypassPermissions was arbitrary code execution).
    #
    # Verified live against the installed CLI (see the Phase 1 plan smoke test):
    #   dontAsk + "Read"            -> frame Read allowed, Bash denied  ✓
    #   dontAsk + "Read(<dir>/**)"  -> ALL reads denied (path-scoping for Read is
    #                                  not honored by --allowedTools in this CLI)
    # so we grant unscoped read-only `Read`. Residual: the agent could read other
    # local files, but it cannot execute anything and its only output is the
    # local sidecar text — a far smaller surface than the prior Bash access.
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model_id,
        "--output-format",
        "json",
        "--permission-mode",
        "dontAsk",
        "--allowedTools",
        "Read",
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


# ---------------------------------------------------------------------------
# Key / CLI resolution
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


# ---------------------------------------------------------------------------
# Per-run dataclasses (shared by both pipelines)
# ---------------------------------------------------------------------------


@dataclass
class ProcessOptions:
    """Per-run config — built once, shared across every file."""

    backend: str  # 'cli' | 'api' | 'local'
    vision_model_id: str
    local_base_url: str
    local_model: str | None
    cost_per_call: float
    no_whisper_prompt: bool
    frame_sampling: str = "diverse"  # 'diverse' | 'even' (legacy)
    whisper_fixes: list[tuple[re.Pattern[str], str]] = field(default_factory=list)
    max_duration_seconds: int | None = None


@dataclass
class ProcessContext:
    """Heavy objects loaded once at startup (models, network clients, DB conns).

    Video-only handles (whisper/align/diarize) default to None so an
    image-only run can build a context without loading the video stack.
    """

    whisper_model: Any = None
    align_models: dict[str, Any] = field(default_factory=dict)
    diarize_pipeline: Any = None
    geocoder: NominatimRateLimiter | None = None
    api_client: Any = None  # anthropic.Anthropic when backend == 'api'
    face_conn: sqlite3.Connection | None = None


@dataclass
class ProcessResult:
    """Outcome of processing one file."""

    sidecar: Path | None
    detected_faces: list[face_db.DetectedFace] = field(default_factory=list)
    cost: float = 0.0
    # 'short' | 'too_long' | 'no_preview' | 'vision_error' | None
    skipped_reason: str | None = None
    rating: str = "?"
    structured: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sidecar serialization (shared: frontmatter dict + prose body → file)
# ---------------------------------------------------------------------------


def serialize_sidecar(
    sidecar: Path,
    frontmatter: dict[str, Any],
    title: str,
    body_sections: list[tuple[str, str]],
) -> Path:
    """Write a `.description.md` sidecar: YAML frontmatter fence, an H1 title,
    then each (heading, content) as a `## heading` section. `indexed_at` is
    stamped as the final frontmatter key. Each pipeline assembles its own
    frontmatter dict and body sections; this only serializes."""
    frontmatter["indexed_at"] = datetime.now().isoformat(timespec="seconds")
    fm_text = yaml.safe_dump(
        frontmatter, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).rstrip()

    parts = ["---", fm_text, "---", "", f"# {title}"]
    for heading, content in body_sections:
        parts += ["", f"## {heading}", "", content]
    atomic_write_text(sidecar, "\n".join(parts) + "\n")
    return sidecar
