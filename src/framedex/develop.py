#!/usr/bin/env python3
"""
framedex.develop — `fdx-develop`: a vision model edits your RAWs.

The model is the editor, an external RAW engine does the pixels. Per RAW:

    engine     → neutral render (cropped with the user's Lightroom crop, if any)
    engine     → calibration sheet: the same photo with a few single-slider
                 probes, so the model learns this engine's slider magnitudes
    model      → looks (photo + context + optional before/after examples of
                 the photographer's own edits), returns slider values as JSON
    engine     → renders them; model looks again and revises (≤ --rounds)
    write      → <out>/<name>.jpg + <out>/<name>.develop.json
                 (+ <raw>.rrdata so RapidRAW re-opens the edit)

Opt-in and experimental. Originals are never modified, moved or renamed;
nothing is overwritten that framedex didn't write. Rounds are stateless (each
call carries everything it needs), so every backend works: api, local, cli.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import requests

import framedex
from framedex import develop_engine as eng
from framedex.images import RAW_EXTENSIONS, get_image_metadata
from framedex.pipeline import (
    DEFAULT_LOCAL_BASE_URL,
    SIDECAR_SUFFIX,
    atomic_write_bytes,
    atomic_write_text,
    describe_frames_cli,
    read_sidecar_frontmatter,
    resolve_anthropic_key,
    split_frontmatter,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_ROUNDS = 3
MAX_ROUNDS = 4
# The model may accept its round-1 render from round 2 on. Forcing a second
# edit made it drift back toward its own "neutral" taste (blind-test finding).
MIN_EDIT_ROUNDS = 1
PREVIEW_LONG = 1600  # px, long edge of what the model sees
EXAMPLE_LONG = 1024  # px, long edge of each before/after example
OUTPUT_LONG = 2048  # px, default long edge of the deliverable
OUTPUT_QUALITY = 90
MAX_EXAMPLES = 3
MAX_OUTPUT_TOKENS = 16000  # one JSON edit + adaptive thinking; bounded per call
API_EFFORT = "high"
LOCAL_TIMEOUT_SEC = 300
OUT_DIRNAME = "developed"
RECORD_SUFFIX = ".develop.json"
# $/MTok (input, output, cache write, cache read). Unknown models report cost
# as unknown rather than a guess.
PRICES_USD_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0, 5.0, 0.20)}
EST_USD_PER_PHOTO = (
    0.25  # measured $0.19-0.28 (avg $0.23) on Opus 5.5, effort high, 3 rounds
)
# One-slider probes rendered once per photo so the model sees what the engine's
# numbers do to *this* image before it picks any.
CALIBRATION_PROBES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("neutral", {}),
    ("exposure +1.0", {"basic": {"exposure": 1.0}}),
    ("shadows +60", {"basic": {"shadows": 60}}),
    ("highlights -60", {"basic": {"highlights": -60}}),
    ("temperature +30", {"white_balance": {"temperature": 30}}),
    ("vibrance +40", {"presence": {"vibrance": 40}}),
)

SYSTEM_PROMPT = """You are a world-class photo editor finishing RAW files for a working photographer. Each image will be a hero shot in a hosted gallery. Edit it the way a top professional would: a clear intent for the image, controlled highlights, intentional contrast, color that serves the subject and the light. When the photographer's own before/after examples are provided, they define the target look: match their brightness, warmth, saturation and contrast choices, not a generic "natural" look.

You work through a real RAW engine by choosing slider values; you never see sliders, only renders. The framing is fixed: you see the image already cropped to its final composition and control tone and color only. This engine's neutral render is darker and flatter than a camera JPEG, so most photos need decisive moves. Small numbers do little: use the calibration sheet to see what +1 exposure or +60 shadows actually does to this photo.

Controls and ranges (0 = neutral unless noted). Values are ABSOLUTE settings, not deltas: every round returns the complete set, and an omitted control resets to neutral.
- tone_mapper: "basic" (standard) or "agx" (filmic: softer highlight roll-off, gentler saturation in bright areas; usually needs more contrast).
- basic.exposure -5..5 (stops), contrast, highlights, shadows, whites, blacks: -100..100.
- white_balance.temperature -100..100 (+ warmer), tint -100..100 (+ magenta), relative to as-shot.
- presence: clarity, dehaze, texture, vibrance, saturation: -100..100.
- detail: sharpness 0..100, luma_noise_reduction 0..100.
- hsl: per band (reds, oranges, yellows, greens, aquas, blues, purples, magentas) hue/saturation/luminance -100..100. Skin lives mostly in oranges.
- color_grading: shadows/midtones/highlights each {hue 0..360, saturation 0..100, luminance -100..100}; blending 0..100 (50 default); balance -100..100.
- tone_curve: luma curve, 2-6 points {x, y} in 0..255 including x=0 and x=255.
- vignette: amount -100..100 (negative darkens edges), midpoint 0..100, feather 0..100.
- masks: 0-3 local adjustments; coordinates are fractions of the frame you see (x 0 left..1 right, y 0 top..1 bottom).
  - "radial": ellipse at (center_x, center_y), radius_x as a fraction of frame width, radius_y of frame height, feather 0..1. Affects the inside unless invert=true.
  - "graduated": a soft gradient from an edge ("top", "bottom", "left", "right") to a midline at `position` (fraction of the frame from that edge), fading over `softness` (fraction of the frame) on each side of the midline.
  - mask adjustments: exposure, contrast, highlights, shadows, whites, blacks, saturation, temperature, tint, clarity, dehaze.

Reviewing a render: judge it against the target look, not against your own taste. With the photographer's examples, the target is their look: compare brightness, warmth, saturation and contrast with their AFTER images, and don't pull warmth or saturation back below theirs unless something clips or skin turns unnatural. Without examples, the target is what a top magazine would print. Name concrete shortcomings and fix only those. If the render already hits the target, set done=true: an unnecessary change is a regression. When done=true your returned values are ignored.

Reply with exactly one JSON object, no prose before or after, with these keys (this is the neutral template; replace values as you choose):
"""


# ---------------------------------------------------------------------------
# Backends: one stateless call = system + labelled images + text -> JSON text
# ---------------------------------------------------------------------------


@dataclass
class Img:
    label: str
    jpeg: bytes
    cache_after: bool = False  # api: put a prompt-cache breakpoint after this image


@dataclass
class Reply:
    text: str
    cost_usd: float | None  # None = unknown (cli, unpriced model)
    usage: dict[str, int] = field(default_factory=dict)


class Backend(Protocol):
    name: str
    model: str

    def complete(self, system: str, images: list[Img], text: str) -> Reply: ...


def _api_cost(model: str, usage: dict[str, int]) -> float | None:
    p = PRICES_USD_PER_MTOK.get(model)
    if p is None:
        return None
    return (
        usage.get("input_tokens", 0) * p[0]
        + usage.get("output_tokens", 0) * p[1]
        + usage.get("cache_creation_input_tokens", 0) * p[2]
        + usage.get("cache_read_input_tokens", 0) * p[3]
    ) / 1e6


class ApiBackend:
    """Anthropic Messages API: adaptive thinking, bounded output."""

    name = "api"

    def __init__(self, model: str, api_key: str) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, timeout=600, max_retries=2)

    def complete(self, system: str, images: list[Img], text: str) -> Reply:
        content: list[dict[str, Any]] = []
        for im in images:
            content.append({"type": "text", "text": im.label})
            block: dict[str, Any] = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(im.jpeg).decode(),
                },
            }
            if im.cache_after:
                block["cache_control"] = {"type": "ephemeral"}
            content.append(block)
        content.append({"type": "text", "text": text})
        with self._client.messages.stream(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=cast(Any, [{"role": "user", "content": content}]),
            thinking=cast(Any, {"type": "adaptive"}),
            output_config=cast(Any, {"effort": API_EFFORT}),
        ) as stream:
            msg = stream.get_final_message()
        u = msg.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": u.cache_read_input_tokens or 0,
        }
        out = "".join(
            str(getattr(b, "text", "")) for b in msg.content if b.type == "text"
        )
        if msg.stop_reason not in ("end_turn", "stop_sequence"):
            out = ""  # truncated or refused: treat as no edit this round
        return Reply(out, _api_cost(self.model, usage), usage)


class LocalBackend:
    """Any OpenAI-compatible server (LM Studio, llama.cpp, vLLM). $0."""

    name = "local"

    def __init__(self, base_url: str, model: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model or "loaded-model"

    def complete(self, system: str, images: list[Img], text: str) -> Reply:
        content: list[dict[str, Any]] = []
        for im in images:
            content.append({"type": "text", "text": im.label})
            b64 = base64.b64encode(im.jpeg).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": text})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.3,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=LOCAL_TIMEOUT_SEC,
            )
            r.raise_for_status()
            msg = (r.json().get("choices") or [{}])[0].get("message") or {}
            out = msg.get("content")
        except (requests.RequestException, ValueError) as e:
            raise RuntimeError(f"local backend error: {e}") from e
        return Reply(out if isinstance(out, str) else "", 0.0)


class CliBackend:
    """Claude Code CLI (`claude -p`, subscription auth). Cost not reported."""

    name = "cli"

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, system: str, images: list[Img], text: str) -> Reply:
        with tempfile.TemporaryDirectory(prefix="fdx-develop-") as td:
            paths: list[Path] = []
            lines = []
            for i, im in enumerate(images, 1):
                p = Path(td) / f"image{i}.jpg"
                p.write_bytes(im.jpeg)
                paths.append(p)
                lines.append(f"{i}. {im.label}: {p}")
            prompt = (
                f"{system}\n\nRead these image files in order before answering:\n"
                + "\n".join(lines)
                + f"\n\n{text}"
            )
            out = describe_frames_cli(paths, prompt, self.model, timeout_sec=600)
        if out.startswith("[CLI"):
            raise RuntimeError(out)
        return Reply(out, None)


# ---------------------------------------------------------------------------
# Rendering (the engine + Pillow), injectable for tests
# ---------------------------------------------------------------------------


@dataclass
class Rendered:
    preview: bytes  # cropped, PREVIEW_LONG, what the model sees
    stats: str
    full_size: tuple[int, int]  # engine output, displayed orientation, pre-crop
    deliverable: bytes  # cropped, --size long edge, final JPEG


class Renderer(Protocol):
    def render(
        self,
        raw: Path,
        adjustments: dict[str, Any],
        tag: str,
        crop: eng.Crop,
        orientation: int,
    ) -> Rendered: ...

    def calibration_sheet(
        self, raw: Path, full_size: tuple[int, int], crop: eng.Crop, orientation: int
    ) -> bytes: ...

    def example_after(self, jpeg: Path) -> bytes: ...


class EngineRenderer:
    """RapidRAW for pixels, Pillow for crop/resize/stats."""

    def __init__(self, engine: Path, work_dir: Path, size: int) -> None:
        self.engine = engine
        self.work_dir = work_dir
        self.size = size

    def _render_cropped(
        self,
        raw: Path,
        adjustments: dict[str, Any],
        tag: str,
        crop: eng.Crop,
        orientation: int,
    ) -> tuple[PILImage, tuple[int, int]]:
        from PIL import Image

        path = eng.render(self.engine, raw, adjustments, self.work_dir, tag)
        try:
            with Image.open(path) as im:
                full = im.convert("RGB")
        finally:
            path.unlink(missing_ok=True)
        return eng.apply_crop(full, crop, orientation), full.size

    def render(
        self,
        raw: Path,
        adjustments: dict[str, Any],
        tag: str,
        crop: eng.Crop,
        orientation: int,
    ) -> Rendered:
        img, full_size = self._render_cropped(raw, adjustments, tag, crop, orientation)
        prev = eng.downscale(img, PREVIEW_LONG)
        return Rendered(
            preview=eng.jpeg_bytes(prev),
            stats=eng.render_stats(prev),
            full_size=full_size,
            deliverable=eng.jpeg_bytes(eng.downscale(img, self.size), OUTPUT_QUALITY),
        )

    def calibration_sheet(
        self, raw: Path, full_size: tuple[int, int], crop: eng.Crop, orientation: int
    ) -> bytes:
        from PIL import Image, ImageDraw, ImageFont

        tiles: list[tuple[str, PILImage]] = []
        for i, (label, probe) in enumerate(CALIBRATION_PROBES):
            edit = eng.normalize_edit(probe)
            adj = eng.to_rapidraw(edit, full_size, crop, orientation)
            img, _ = self._render_cropped(raw, adj, f"probe{i}", crop, orientation)
            tiles.append((label, eng.downscale(img, 640)))
        tw = max(t.size[0] for _, t in tiles)
        th = max(t.size[1] for _, t in tiles)
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        band = 44
        sheet = Image.new("RGB", (cols * tw, rows * (th + band)), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=32)
        for k, (label, t) in enumerate(tiles):
            x, y = (k % cols) * tw, (k // cols) * (th + band)
            sheet.paste(t, (x, y))
            draw.text((x + 8, y + th + 4), label, fill="black", font=font)
        return eng.jpeg_bytes(eng.downscale(sheet, PREVIEW_LONG))

    def example_after(self, jpeg: Path) -> bytes:
        from PIL import Image, ImageOps

        with Image.open(jpeg) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")
        return eng.jpeg_bytes(eng.downscale(img, EXAMPLE_LONG))


# ---------------------------------------------------------------------------
# Inputs: discovery, crop, context, examples
# ---------------------------------------------------------------------------


def discover(root: Path, out_dir: Path) -> list[Path]:
    """RAWs under root (recursive), skipping hidden dirs and the output dir."""
    found: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in RAW_EXTENSIONS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if out_dir in p.parents:
            continue
        found.append(p)
    return found


def read_crop(raw: Path) -> eng.Crop:
    """Lightroom crop from `<name>.xmp` / `<file>.xmp` next to the RAW."""
    for xmp in (raw.with_suffix(".xmp"), raw.with_name(raw.name + ".xmp")):
        try:
            crop = eng.parse_lr_crop(xmp.read_text(errors="replace"))
        except OSError:
            continue
        if crop is not None:
            return crop
    return eng.IDENTITY_CROP


def embedded_crop(jpeg: Path) -> eng.Crop:
    """Lightroom crop from the XMP packet embedded in an exported JPEG."""
    try:
        data = jpeg.read_bytes()
    except OSError:
        return eng.IDENTITY_CROP
    return eng.parse_lr_crop(data.decode("latin-1")) or eng.IDENTITY_CROP


def photo_context(raw: Path) -> tuple[str, int]:
    """(context text for the prompt, EXIF orientation). Uses a framedex
    `.description.md` sidecar when one exists."""
    lines: list[str] = []
    orientation = 1
    try:
        meta = get_image_metadata(raw)
    except RuntimeError:
        meta = {}
    cam = meta.get("camera") or {}
    orientation = eng.orientation_code(cam.get("orientation"))
    bits = [str(cam.get(k)) for k in ("model", "lens", "focal_length") if cam.get(k)]
    if cam.get("aperture"):
        bits.append(f"f/{cam['aperture']}")
    if cam.get("shutter"):
        bits.append(f"{cam['shutter']}s")
    if cam.get("iso"):
        bits.append(f"ISO {cam['iso']}")
    if bits:
        lines.append("Camera: " + ", ".join(bits))
    if meta.get("creation_time"):
        lines.append(f"Captured (camera clock): {meta['creation_time']}")
    sidecar = raw.with_name(raw.name + SIDECAR_SUFFIX)
    fm = read_sidecar_frontmatter(sidecar) if sidecar.is_file() else None
    if fm:
        loc = fm.get("location") or {}
        if isinstance(loc, dict) and loc.get("place"):
            lines.append(f"Place: {loc['place']}")
        for k in ("scene_type", "time_of_day", "lighting"):
            if fm.get(k) and fm.get(k) != "unclear":
                lines.append(f"{k.replace('_', ' ').capitalize()}: {fm[k]}")
        split = split_frontmatter(sidecar.read_text(errors="replace"))
        if split:
            body = " ".join(split[1].split())
            if body:
                lines.append(f"Scene (from the archive index): {body[:700]}")
    return "\n".join(lines), orientation


@dataclass
class Example:
    name: str
    before: bytes
    after: bytes


def load_examples(folder: Path, renderer: Renderer) -> list[Example]:
    """Up to MAX_EXAMPLES (RAW, finished-JPEG) pairs sharing a stem. The RAW is
    rendered neutral and cropped with the JPEG's embedded Lightroom crop."""
    out: list[Example] = []
    for raw in sorted(folder.iterdir()):
        if len(out) >= MAX_EXAMPLES:
            break
        if raw.suffix.lower() not in RAW_EXTENSIONS:
            continue
        jpeg = next(
            (
                raw.with_suffix(s)
                for s in (".jpg", ".jpeg", ".JPG", ".JPEG")
                if raw.with_suffix(s).is_file()
            ),
            None,
        )
        if jpeg is None:
            continue
        _, orientation = photo_context(raw)
        neutral = eng.to_rapidraw(eng.default_edit(), (1, 1))
        before = renderer.render(
            raw, neutral, f"example-{raw.stem}", embedded_crop(jpeg), orientation
        )
        out.append(Example(raw.stem, before.preview, renderer.example_after(jpeg)))
    return out


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def system_prompt() -> str:
    return (
        SYSTEM_PROMPT
        + json.dumps(eng.default_edit())
        + "\nEach entry in masks looks like:\n"
        + json.dumps(eng.default_mask())
    )


def _example_images(examples: Sequence[Example]) -> list[Img]:
    imgs: list[Img] = []
    for i, ex in enumerate(examples, 1):
        imgs.append(
            Img(f"Photographer's example {i}, BEFORE (neutral render):", ex.before)
        )
        imgs.append(
            Img(
                f"Photographer's example {i}, AFTER (their finished edit):",
                ex.after,
                cache_after=i == len(examples),
            )
        )
    return imgs


@dataclass
class RoundLog:
    round: int
    edit: dict[str, Any]
    stats: str = ""
    cost_usd: float | None = None
    usage: dict[str, int] = field(default_factory=dict)
    error: str = ""
    accepted_previous: bool = False


def round_text(
    context: str,
    rnd: int,
    rounds: int,
    has_examples: bool,
    neutral_stats: str,
    prev: RoundLog | None,
) -> str:
    parts = []
    if context:
        parts.append(f"Photo context:\n{context}")
    if has_examples:
        parts.append(
            "The before/after pairs above are this photographer's own edits from "
            "similar work: match their look (brightness, warmth, saturation, "
            "contrast), not their crop. Don't convert to black and white unless "
            "this photo clearly calls for it."
        )
    parts.append(f"Neutral render measurements: {neutral_stats}")
    if prev is None:
        parts.append(
            f"Round 1 of {rounds}: describe what you see, decide the intent, and "
            "return your first complete edit."
        )
    else:
        parts.append(
            f"Your round {prev.round} values were:\n{json.dumps(prev.edit)}\n"
            f"The last image is the render of those values. Its measurements: "
            f"{prev.stats}"
        )
        final_ok = rnd > MIN_EDIT_ROUNDS
        target = (
            "the photographer's examples (their brightness, warmth, saturation "
            "and contrast)"
            if has_examples
            else "your intent"
        )
        parts.append(
            f"Round {rnd} of {rounds}: compare that render with {target}. List "
            "its concrete shortcomings in changes_this_round and return the "
            "complete revised values."
            + (
                " If it already hits the target, set done=true instead of changing it."
                if final_ok
                else " (done=true is not allowed yet.)"
            )
        )
    return "\n\n".join(parts)


@dataclass
class PhotoResult:
    raw: Path
    status: str  # done | partial | failed | skipped
    output: Path | None = None
    cost_usd: float = 0.0  # known spend
    cost_complete: bool = True  # False: some call's cost is unknown
    message: str = ""


def read_files_from(listing: Path, root: Path, raws: Sequence[Path]) -> list[Path]:
    """The RAWs named in `listing` (one path per line, relative to root or
    absolute inside it; blank lines and # comments ignored), in discovery
    order. Raises ValueError naming every bad line: outside root, missing, or
    not a RAW that discovery would develop.

    Matching uses the listed path itself, not a symlink's target: directories
    are resolved (so `..` can't escape and /var vs /private/var both work), the
    file name is taken as written, and a different-case name only matches when
    it is the same file (case-insensitive volumes)."""

    def key(p: Path) -> Path:
        return p.parent.resolve() / p.name

    by_key = {key(r): r for r in raws}
    by_folded: dict[tuple[Path, str], list[Path]] = {}
    for r in raws:
        by_folded.setdefault((r.parent.resolve(), r.name.casefold()), []).append(r)
    chosen: set[Path] = set()
    errors: list[str] = []
    for n, line in enumerate(listing.read_text().splitlines(), 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        path = Path(entry).expanduser()
        cand = key(path if path.is_absolute() else root / path)
        match = by_key.get(cand)
        if match is None and cand.exists():
            match = next(
                (
                    r
                    for r in by_folded.get((cand.parent, cand.name.casefold()), [])
                    if os.path.samefile(r, cand)
                ),
                None,
            )
        if not cand.parent.is_relative_to(root):
            errors.append(f"line {n}: {entry}: outside {root}")
        elif not cand.exists():
            errors.append(f"line {n}: {entry}: no such file")
        elif cand.suffix.lower() not in RAW_EXTENSIONS or match is None:
            errors.append(f"line {n}: {entry}: not a RAW fdx-develop would process")
        else:
            chosen.add(match)
    if errors:
        raise ValueError(f"--files-from {listing}:\n  " + "\n  ".join(errors))
    return [r for r in raws if r in chosen]


def plan_outputs(
    raws: Sequence[Path], root: Path, out_dir: Path
) -> dict[Path, tuple[Path, Path]]:
    """(jpg, record) for every RAW: <out>/<subdir>/<stem>.jpg, switching to the
    full file name (<file name>.jpg) for every RAW whose name would clash with
    another output in the same folder. Full file names are unique per folder,
    so repeating until nothing clashes always terminates."""
    names = {r: r.stem for r in raws}
    while True:
        seen: dict[tuple[Path, str], list[Path]] = {}
        for r in raws:
            seen.setdefault((r.parent, names[r].lower()), []).append(r)
        clashes = [r for group in seen.values() if len(group) > 1 for r in group]
        if not clashes:
            break
        for r in clashes:
            names[r] = r.name
    out: dict[Path, tuple[Path, Path]] = {}
    for r in raws:
        base = out_dir / r.relative_to(root).parent / names[r]
        out[r] = (
            base.with_name(base.name + ".jpg"),
            base.with_name(base.name + RECORD_SUFFIX),
        )
    return out


def _fingerprint(raw: Path) -> dict[str, Any]:
    st = raw.stat()
    return {"name": raw.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_record(path: Path) -> dict[str, Any] | None:
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def is_complete(raw: Path, jpg: Path, record: Path) -> bool:
    rec = _load_record(record)
    return bool(
        rec
        and rec.get("status") == "done"
        and jpg.is_file()
        and rec.get("source") == _fingerprint(raw)
        and rec.get("output_sha256") == _sha256(jpg)
    )


def _owned_digests(rec: dict[str, Any]) -> set[str]:
    return {d for d in (rec.get("output_sha256"), rec.get("previous_sha256")) if d}


def owned_digest(raw: Path, jpg: Path, record: Path) -> str:
    """Digest of the JPEG at `jpg` if framedex's record for this RAW proves it
    wrote exactly those bytes (this run's output, or the previous output a
    pending/failed attempt left in place); '' otherwise."""
    if not jpg.is_file():
        return ""
    rec = _load_record(record)
    if rec is None or (rec.get("source") or {}).get("name") != raw.name:
        return ""
    digest = _sha256(jpg)
    return digest if digest in _owned_digests(rec) else ""


def owns_output(raw: Path, jpg: Path, record: Path) -> bool:
    """May framedex (re)write `jpg` for `raw`? Only if nothing is there or the
    record proves framedex wrote exactly what is there. A user's own export, or
    a file the user replaced, is never touched."""
    return not jpg.exists() or bool(owned_digest(raw, jpg, record))


def write_rrdata(raw: Path, adjustments: dict[str, Any]) -> bool:
    """Publish `<raw>.rrdata` atomically and only if nothing (file or symlink)
    is there: write a temp file, then hard-link it into place, which fails
    instead of replacing anything."""
    path = eng.rrdata_path(raw)
    if os.path.lexists(path):
        return False
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(eng.rrdata_payload(adjustments), indent=1))
        os.link(tmp, path)
    except OSError:
        return False  # appeared meanwhile, read-only media, no hard links: never fatal
    finally:
        tmp.unlink(missing_ok=True)
    return True


def rrdata_adjustments(
    adjustments: dict[str, Any],
    full_size: tuple[int, int],
    crop: eng.Crop,
    orientation: int,
) -> dict[str, Any]:
    """The engine adjustments plus the crop in RapidRAW's own terms, so the
    re-opened edit has the same framing. Angled crops are not carried (the
    engine's rotation semantics differ); the straightening is lost on re-open."""
    out = dict(adjustments)
    if not crop.is_identity and abs(crop.angle) < 1e-6:
        x0, y0, x1, y1 = eng.crop_box_display(full_size, crop, orientation)
        out["crop"] = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
    return out


@dataclass
class _Run:
    """Per-photo bookkeeping shared by the loop and the record writer."""

    raw: Path
    record: Path
    backend: Backend
    engine_info: dict[str, str]
    crop: eng.Crop
    history: list[RoundLog] = field(default_factory=list)
    cost: float = 0.0
    cost_complete: bool = True
    previous_sha256: str = ""  # the owned JPEG already at the output path, if any

    def write(
        self,
        status: str,
        message: str = "",
        *,
        final_round: int | None = None,
        adjustments: dict[str, Any] | None = None,
        output: Path | None = None,
        output_sha256: str = "",
        rrdata_written: bool = False,
    ) -> None:
        rec = {
            "framedex": framedex.__version__,
            "status": status,
            "message": message,
            "source": _fingerprint(self.raw),
            "output": output.name if output else "",
            "output_sha256": output_sha256,
            "previous_sha256": self.previous_sha256,
            "engine": self.engine_info,
            "backend": self.backend.name,
            "model": self.backend.model,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "crop": None if self.crop.is_identity else vars(self.crop),
            "final_round": final_round,
            "rrdata_written": rrdata_written,
            "cost_usd": round(self.cost, 4),
            "cost_complete": self.cost_complete,
            "rounds": [vars(h) for h in self.history],
            "adjustments": adjustments,
        }
        self.record.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.record, json.dumps(rec, indent=1, default=str))

    def result(
        self, status: str, output: Path | None = None, message: str = ""
    ) -> PhotoResult:
        return PhotoResult(
            self.raw, status, output, self.cost, self.cost_complete, message
        )


def develop_photo(
    raw: Path,
    *,
    jpg: Path,
    record: Path,
    backend: Backend,
    renderer: Renderer,
    rounds: int,
    examples: Sequence[Example],
    calibration: bool,
    write_engine_sidecar: bool,
    engine_info: dict[str, str],
    log: Callable[[str], None] = print,
) -> PhotoResult:
    context, orientation = photo_context(raw)
    crop = read_crop(raw)
    run = _Run(raw, record, backend, engine_info, crop)
    # Keep proof of ownership of an earlier output across a failed or
    # interrupted redo, so it stays replaceable instead of turning "foreign".
    run.previous_sha256 = owned_digest(raw, jpg, record)
    fixed: list[Img] = _example_images(examples)
    try:
        neutral = renderer.render(
            raw, eng.to_rapidraw(eng.default_edit(), (1, 1)), "r0", crop, orientation
        )
        full_size = neutral.full_size
        if calibration:
            fixed.append(
                Img(
                    "Calibration sheet: this photo with single-slider probes in this engine (labels under each tile):",
                    renderer.calibration_sheet(raw, full_size, crop, orientation),
                )
            )
    except RuntimeError as e:  # engine can't read this RAW: fail the photo, not the run
        run.write("failed", f"render: {e}")
        return run.result("failed", message=f"render: {e}")
    fixed.append(
        Img("The photo, neutral render (cropped to the final frame):", neutral.preview)
    )
    system = system_prompt()
    best: Rendered | None = None
    best_round: RoundLog | None = None
    for rnd in range(1, rounds + 1):
        imgs = list(fixed)
        if best is not None and best_round is not None:
            imgs.append(
                Img(f"Render of your round {best_round.round} values:", best.preview)
            )
        text = round_text(
            context, rnd, rounds, bool(examples), neutral.stats, best_round
        )
        try:
            reply = backend.complete(system, imgs, text)
        except Exception as e:  # backend down: keep the best render so far
            # The call may have been billed before it failed (e.g. a stream
            # timeout), so its cost is unknown rather than zero.
            run.cost_complete = False
            run.history.append(RoundLog(rnd, {}, error=f"backend: {e}"))
            break
        if reply.cost_usd is None:
            run.cost_complete = False
        else:
            run.cost += reply.cost_usd
        try:
            edit = eng.normalize_edit(eng.extract_json_object(reply.text))
        except ValueError as e:
            run.history.append(
                RoundLog(
                    rnd,
                    {},
                    cost_usd=reply.cost_usd,
                    usage=reply.usage,
                    error=f"bad reply: {e}",
                )
            )
            break
        entry = RoundLog(rnd, edit, cost_usd=reply.cost_usd, usage=reply.usage)
        if edit["done"] and best is not None and rnd > MIN_EDIT_ROUNDS:
            entry.accepted_previous = True
            run.history.append(entry)
            log(
                f"  round {rnd}: accepted round {best_round.round if best_round else 0}"
            )
            break
        adj = eng.to_rapidraw(edit, full_size, crop, orientation)
        try:
            rendered = renderer.render(raw, adj, f"r{rnd}", crop, orientation)
        except RuntimeError as e:
            entry.error = f"render: {e}"
            run.history.append(entry)
            break
        entry.stats = rendered.stats
        run.history.append(entry)
        best, best_round = rendered, entry
        log(f"  round {rnd}: {edit.get('intent', '')[:90]}")
    errors = [h.error for h in run.history if h.error]
    if best is None or best_round is None:
        msg = errors[0] if errors else "no edit produced"
        run.write("failed", msg)
        return run.result("failed", message=msg)
    adjustments = eng.to_rapidraw(best_round.edit, full_size, crop, orientation)
    # Claim the path before publishing, so an interrupt between the two leaves
    # a JPEG framedex can prove it owns (status pending), not an orphan.
    new_sha256 = hashlib.sha256(best.deliverable).hexdigest()
    run.write(
        "pending", final_round=best_round.round, output=jpg, output_sha256=new_sha256
    )
    jpg.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(jpg, best.deliverable)
    rr = write_engine_sidecar and write_rrdata(
        raw, rrdata_adjustments(adjustments, full_size, crop, orientation)
    )
    # A round that failed after a good render still ships that render, but the
    # photo stays retryable ("partial") rather than silently "done".
    status = "partial" if errors else "done"
    run.previous_sha256 = ""
    run.write(
        status,
        errors[0] if errors else "",
        final_round=best_round.round,
        adjustments=adjustments,
        output=jpg,
        output_sha256=new_sha256,
        rrdata_written=rr,
    )
    return run.result(status, jpg, errors[0] if errors else "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "local":
        return LocalBackend(args.local_base_url, args.local_model)
    if args.backend == "cli":
        return CliBackend(args.model)
    key = resolve_anthropic_key()
    if not key:
        sys.exit("No Anthropic API key (set ANTHROPIC_API_KEY) for --backend api.")
    return ApiBackend(args.model, key)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fdx-develop",
        description="Edit RAWs with a vision model driving an external RAW engine "
        "(RapidRAW). Writes edited copies + plain-text sidecars; never touches "
        "originals. Experimental and opt-in.",
    )
    p.add_argument("root", help="Folder of RAW files (searched recursively)")
    p.add_argument("--out", help=f"Output folder (default: <root>/{OUT_DIRNAME})")
    p.add_argument(
        "--engine",
        help="RapidRAW binary or .app (default: $FDX_RAPIDRAW, PATH, /Applications)",
    )
    p.add_argument("--backend", choices=("api", "local", "cli"), default="api")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model for api/cli (default {DEFAULT_MODEL})",
    )
    p.add_argument("--local-base-url", default=DEFAULT_LOCAL_BASE_URL)
    p.add_argument("--local-model", default=None)
    p.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Max look-and-adjust rounds, 1-{MAX_ROUNDS}",
    )
    p.add_argument(
        "--examples",
        help="Folder of up to 3 RAW + finished-JPEG pairs (same stem) that show your style",
    )
    p.add_argument(
        "--size",
        type=int,
        default=OUTPUT_LONG,
        help="Long edge of the edited copies in px",
    )
    p.add_argument(
        "--files-from",
        help="Develop only the RAWs listed in this file (one path per line, relative "
        "to root or absolute inside it; blank lines and # comments ignored)",
    )
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument(
        "--force", action="store_true", help="Redo photos framedex already developed"
    )
    p.add_argument(
        "--no-rrdata",
        action="store_true",
        help="Don't write a RapidRAW .rrdata sidecar next to the RAW",
    )
    p.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip the slider calibration sheet",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would run and the cost estimate",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")
    if not 1 <= args.rounds <= MAX_ROUNDS:
        sys.exit(f"--rounds must be 1-{MAX_ROUNDS}")
    out_dir = Path(args.out).expanduser().resolve() if args.out else root / OUT_DIRNAME
    raws = discover(root, out_dir)
    # Names are planned over the whole folder, so a photo's output name is the
    # same whether or not --files-from selects a subset.
    outputs = plan_outputs(raws, root, out_dir)
    selected = raws
    if args.files_from is not None:
        if not args.files_from.strip():
            sys.exit("--files-from needs a file name")
        try:
            selected = read_files_from(Path(args.files_from).expanduser(), root, raws)
        except (OSError, ValueError) as e:
            sys.exit(str(e))
    plan = [(r, *outputs[r]) for r in selected]
    todo = [(r, j, c) for r, j, c in plan if args.force or not is_complete(r, j, c)]
    if args.max_files is not None:
        todo = todo[: args.max_files]
    est = len(todo) * EST_USD_PER_PHOTO if args.backend == "api" else 0.0
    print(
        f"{len(raws)} RAWs"
        + (f", {len(plan)} listed" if args.files_from is not None else "")
        + f", {len(plan) - len(todo)} already developed, {len(todo)} to do; est. ${est:.2f}"
    )
    if args.dry_run:
        for r, j, _ in todo:
            print(f"  {r.relative_to(root)} -> {j}")
        return 0
    engine = eng.find_engine(args.engine)
    if engine is None:
        sys.exit(
            "RapidRAW not found. Install it (github.com/CyberTimon/RapidRAW) and pass --engine or set $FDX_RAPIDRAW."
        )
    backend = make_backend(args)
    engine_info = {"name": eng.ENGINE_NAME, "version": eng.engine_version(engine)}
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    unknown_cost = False
    counts = {"done": 0, "partial": 0, "failed": 0, "skipped": 0}
    with tempfile.TemporaryDirectory(prefix=".fdx-develop-", dir=out_dir) as td:
        renderer = EngineRenderer(engine, Path(td), args.size)
        try:
            examples = (
                load_examples(Path(args.examples).expanduser(), renderer)
                if args.examples
                else []
            )
        except (OSError, RuntimeError) as e:
            sys.exit(f"Could not load --examples: {e}")
        if args.examples:
            print(f"style examples: {len(examples)} pair(s)")
        for raw, jpg, rec in todo:
            if not owns_output(raw, jpg, rec):
                print(f"skip {raw.name}: {jpg} exists and framedex didn't write it")
                counts["skipped"] += 1
                continue
            print(f"{raw.relative_to(root)}")
            t0 = time.time()
            try:
                res = develop_photo(
                    raw,
                    jpg=jpg,
                    record=rec,
                    backend=backend,
                    renderer=renderer,
                    rounds=args.rounds,
                    examples=examples,
                    calibration=not args.no_calibration,
                    write_engine_sidecar=not args.no_rrdata,
                    engine_info=engine_info,
                )
            except Exception as e:  # one bad photo never stops the batch
                res = PhotoResult(raw, "failed", cost_complete=False, message=repr(e))
            counts[res.status] = counts.get(res.status, 0) + 1
            total += res.cost_usd
            unknown_cost = unknown_cost or not res.cost_complete
            print(f"  {res.status} in {time.time() - t0:.0f}s {res.message}".rstrip())
    cost = f"${total:.2f}" + (" + unknown" if unknown_cost else "")
    print(
        f"\ndone {counts['done']}, partial {counts['partial']}, failed "
        f"{counts['failed']}, skipped {counts['skipped']}; cost {cost}; output {out_dir}"
    )
    return 1 if counts["failed"] or counts["partial"] else 0


if __name__ == "__main__":
    sys.exit(main())
