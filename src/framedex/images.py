#!/usr/bin/env python3
"""
framedex.images — still-photo pipeline.

The photo twin of the video pipeline in index_videos: same plain-text
`.description.md` sidecars, same shared transport/serializer/faces from
`pipeline`, but the audio/motion half is replaced by an EXIF/camera half.

Per still:
    exiftool      → EXIF (camera, lens, exposure, dimensions, orientation, date)
    exiftool      → GPS  → Nominatim reverse-geocoded place   (shared)
    render        → one upright 1920px JPEG preview            (RAW via embedded
                    preview; HEIC via pillow-heif; EXIF orientation normalized)
    insightface   → face detection + 512-dim embeddings on the preview (shared)
    Vision model  → structured YAML + prose (photo-tuned prompt)
    write         → [filename].description.md sidecar + face row in faces.db

No frame sampling, no audio, no whisper — so an image-only `fdx --media images`
run never needs the video stack installed.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

from framedex import face_db, pipeline
from framedex.parsing import coerce_people_count
from framedex.pipeline import (
    CLI_INTER_CALL_DELAY,
    FRAME_MAX_WIDTH,
)

# RAW formats whose pixels a vision model can't read directly — we pull the
# full-res JPEG preview every modern RAW embeds instead of decoding the sensor.
RAW_EXTENSIONS = {
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".rw2",
    ".orf",
    ".dng",
}
# Stills Pillow (with pillow-heif) reads directly.
RENDERABLE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".webp",
}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | RENDERABLE_EXTENSIONS
HEIF_EXTENSIONS = {".heic", ".heif"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_images(root: Path, exclude_patterns: list[str]) -> list[Path]:
    """Recursively find still photos under root. Mirrors find_videos: skips
    hidden files, sidecars, and our own `_`-prefixed output folders."""
    images: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = p.relative_to(root)
        if any(part.startswith("_") for part in rel.parts[:-1]):
            continue
        excluded = False
        for pat in exclude_patterns:
            if pat in str(rel):
                excluded = True
                break
        if excluded:
            continue
        images.append(p)
    return sorted(images)


# ---------------------------------------------------------------------------
# EXIF metadata
# ---------------------------------------------------------------------------


def _normalize_exif_datetime(raw: str) -> str:
    """exiftool dates look like '2024:08:14 07:23:11'. Normalize the date half
    to ISO ('2024-08-14T07:23:11') so it sorts/parses like video creation_time."""
    raw = raw.strip()
    if not raw:
        return ""
    parts = raw.split(" ", 1)
    date = parts[0].replace(":", "-")
    if len(parts) == 2 and parts[1]:
        return f"{date}T{parts[1]}"
    return date


def get_image_metadata(image: Path) -> dict[str, Any]:
    """exiftool → dimensions + camera block (make/model/lens/exposure) +
    creation_time + size_bytes. Readable human values (e.g. shutter '1/1000'),
    no `-n`, so the sidecar shows what a photographer expects."""
    cmd = [
        "exiftool",
        "-json",
        "-Make",
        "-Model",
        "-LensModel",
        "-LensID",
        "-FocalLength",
        "-FNumber",
        "-ExposureTime",
        "-ISO",
        "-ImageWidth",
        "-ImageHeight",
        "-Orientation",
        "-DateTimeOriginal",
        "-CreateDate",
        str(image),
    ]
    meta: dict[str, Any] = {
        "size_bytes": image.stat().st_size,
        "creation_time": "",
        "dimensions": "",
        "camera": {},
    }
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return meta
    try:
        data = json.loads(result.stdout)[0]
    except (ValueError, IndexError):
        return meta

    width = data.get("ImageWidth")
    height = data.get("ImageHeight")
    if width and height:
        meta["dimensions"] = f"{width}x{height}"

    ct = data.get("DateTimeOriginal") or data.get("CreateDate") or ""
    meta["creation_time"] = _normalize_exif_datetime(str(ct))

    camera: dict[str, Any] = {}
    if data.get("Make"):
        camera["make"] = str(data["Make"]).strip()
    if data.get("Model"):
        camera["model"] = str(data["Model"]).strip()
    lens = data.get("LensModel") or data.get("LensID")
    if lens:
        camera["lens"] = str(lens).strip()
    if data.get("FocalLength"):
        camera["focal_length"] = str(data["FocalLength"]).strip()
    if data.get("FNumber") is not None:
        camera["aperture"] = data["FNumber"]
    if data.get("ExposureTime") is not None:
        camera["shutter"] = str(data["ExposureTime"]).strip()
    if data.get("ISO") is not None:
        camera["iso"] = data["ISO"]
    if data.get("Orientation"):
        camera["orientation"] = str(data["Orientation"]).strip()
    meta["camera"] = camera
    return meta


# ---------------------------------------------------------------------------
# Preview rendering (the image IS the frame — no ffmpeg sampling)
# ---------------------------------------------------------------------------


def _extract_raw_preview(raw_image: Path, out_dir: Path) -> Path | None:
    """Pull the embedded JPEG preview from a RAW file via exiftool. Tries the
    full-size JpgFromRaw first, then the smaller PreviewImage. Returns the
    extracted JPEG path, or None if the RAW carries no usable preview."""
    for tag in ("-JpgFromRaw", "-PreviewImage"):
        out = out_dir / "raw_preview.jpg"
        result = subprocess.run(
            ["exiftool", "-b", tag, str(raw_image)],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            out.write_bytes(result.stdout)
            if out.stat().st_size > 0:
                return out
    return None


def render_preview(image: Path, out_dir: Path) -> Path | None:
    """Produce one upright JPEG (≤ FRAME_MAX_WIDTH wide) for the vision call and
    face detection. RAW → embedded preview; everything else → Pillow. EXIF
    orientation is applied so the model and face detector see upright pixels.
    Returns None when a RAW has no embedded preview to read."""
    ext = image.suffix.lower()

    # Resolve the source pixels first. A RAW with no embedded preview is a clean
    # skip and never needs Pillow loaded.
    if ext in RAW_EXTENSIONS:
        src = _extract_raw_preview(image, out_dir)
        if src is None:
            return None
    else:
        src = image

    try:
        from PIL import Image, ImageOps
    except ImportError as e:  # pragma: no cover - exercised via the [images] extra
        raise RuntimeError(
            "still-photo indexing requires the 'images' extra. "
            f"Install with: uv pip install -e '.[images]'  ({e})"
        ) from e

    if ext in HEIF_EXTENSIONS:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise RuntimeError(
                "HEIC/HEIF indexing requires pillow-heif. "
                f"Install with: uv pip install -e '.[images]'  ({e})"
            ) from e

    # A decode/save failure here is a real error (corrupt or unreadable file),
    # not a clean skip — let it propagate so the run loop reports it loudly and
    # the file is retried, rather than masquerading as "no preview".
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)  # normalize rotation
        if im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        if w > FRAME_MAX_WIDTH:
            new_h = round(h * FRAME_MAX_WIDTH / w)
            im = im.resize((FRAME_MAX_WIDTH, new_h), Image.LANCZOS)
        out = out_dir / "preview.jpg"
        im.save(out, "JPEG", quality=90)
    return out


# ---------------------------------------------------------------------------
# Vision prompt (still-photo tuned)
# ---------------------------------------------------------------------------


def _build_image_vision_prompt(
    preview: Path,
    context: dict[str, Any],
    include_paths: bool,
) -> str:
    """Photo analogue of the video prompt. Same YAML/prose contract minus the
    audio/motion fields, plus composition + scene_type. include_paths=True for
    CLI mode (Claude Code reads the preview via its Read tool)."""
    location_line = ""
    loc = context.get("location") or {}
    if loc.get("place"):
        location_line = f"Location: {loc['place']} ({loc.get('lat')}, {loc.get('lon')})"
    elif loc.get("lat") is not None:
        location_line = f"GPS: {loc['lat']}, {loc['lon']}"

    camera = context.get("camera") or {}
    camera_line = ""
    if camera:
        bits = []
        if camera.get("model"):
            bits.append(str(camera["model"]))
        if camera.get("lens"):
            bits.append(str(camera["lens"]))
        if camera.get("focal_length"):
            bits.append(str(camera["focal_length"]))
        if camera.get("aperture") is not None:
            bits.append(f"f/{camera['aperture']}")
        if camera.get("shutter"):
            bits.append(f"{camera['shutter']}s")
        if camera.get("iso") is not None:
            bits.append(f"ISO {camera['iso']}")
        if bits:
            camera_line = "Camera: " + ", ".join(bits)

    intro = (
        "Read this JPEG, then analyze the photograph."
        if include_paths
        else "Analyze this photograph."
    )
    paths_block = f"\nImage (read it):\n{preview}\n" if include_paths else ""

    return textwrap.dedent(f"""
    {intro}
    {paths_block}
    File: {context["filename"]}
    Parent folder: {context["parent_folder"]}
    Creation date: {context.get("creation_time") or "unknown"}
    {camera_line}
    {location_line}

    Produce TWO blocks in this exact order:

    BLOCK 1 — a YAML code fence with structured assessment fields:

    ```yaml
    rating: keep | review | cull
    cull_reason: ""    # short reason if cull; blank otherwise
    technical:
      focus: sharp | acceptable | soft
      exposure: strong | adequate | poor | clipped
      composition: strong | acceptable | weak
    lighting: golden_hour | bright_daylight | overcast | dim_interior | nighttime | mixed | unclear
    time_of_day: predawn | dawn_morning | midday | afternoon | golden_hour | dusk | night | unclear
    dominant_color_palette: "short descriptive phrase, e.g. 'warm savanna: amber, ochre, dusty olive'"
    dominant_colors: [color1, color2, color3]   # 3-5 named colors, lowercase, hyphenated
    scene_type: wildlife | landscape | portrait | street | architecture | macro | event | food | documentary | abstract | other
    people_count: 0    # integer 0-99; estimate crowds to nearest 5; cap at 99 (always an integer)
    keywords: [tag1, tag2, tag3, tag4, tag5]    # 5-10 short lowercase tags — subjects, setting, action
    ```

    BLOCK 2 — a prose description in this exact markdown structure:

    **Scene:** One sentence describing the setting (where, time of day, light).
    **Subjects:** Who or what is in frame (count + role + what they're doing). Do
    not guess identities of specific people; describe generically.
    **Composition:** Framing, depth, focal point, notable technique.
    **Mood:** Emotional or atmospheric tone.
    **Use cases:** 2-3 short bullets — what this photo would suit.

    HOW TO BE ACCURATE (affects both blocks):
    - Describe ONLY what you can clearly see. If a detail is ambiguous, use
      "unclear" in the YAML or "(unclear)" in prose. Do NOT invent.
    - Rating is for a PERSONAL PHOTO ARCHIVE. Default to "keep" for any frame
      that captured a real moment. Use "cull" for genuine technical failures
      that make the frame unusable: badly missed focus on the main subject,
      heavy unintended blur, blown/clipped exposure with no detail, eyes fully
      closed in an otherwise-portrait, lens-cap/pocket/accidental frames. Use
      "review" when you genuinely can't tell. Soft-but-intentional, grain, and
      moody low light are NOT cull reasons.
    - cull_reason should name the specific technical failure; blank otherwise.

    OUTPUT FORMAT:
    1. The ```yaml fence with structured fields first
    2. Then "## Description" header
    3. Then the prose block

    Do not include preamble, commentary, or any text outside these two blocks.
    """).strip()


# ---------------------------------------------------------------------------
# Frontmatter assembly
# ---------------------------------------------------------------------------


def build_image_frontmatter(
    image: Path,
    root: Path,
    metadata: dict[str, Any],
    gps: dict[str, Any],
    place: str,
    structured: dict[str, Any],
    faces: list[face_db.DetectedFace],
    *,
    parent_folder_override: str | None = None,
    extra_frontmatter: dict[str, Any] | None = None,
    omit_path: bool = False,
) -> dict[str, Any]:
    """Assemble the photo sidecar frontmatter dict (video schema minus
    audio/motion, plus a camera block + media_type discriminator)."""
    parent = (
        parent_folder_override
        if parent_folder_override is not None
        else image.parent.name
    )
    fm: dict[str, Any] = {
        "file": image.name,
        "parent_folder": parent,
        "media_type": "image",
        "size_bytes": metadata["size_bytes"],
        "creation_time": metadata.get("creation_time") or "",
        "dimensions": metadata.get("dimensions") or "",
    }
    if not omit_path:
        try:
            fm["path"] = str(image.relative_to(root))
        except ValueError:
            fm["path"] = str(image)
    if metadata.get("camera"):
        fm["camera"] = metadata["camera"]
    if gps.get("lat") is not None:
        loc: dict[str, Any] = {"lat": gps["lat"], "lon": gps["lon"]}
        if gps.get("altitude_m") is not None:
            loc["altitude_m"] = gps["altitude_m"]
        if place:
            loc["place"] = place
        fm["location"] = loc

    fm["rating"] = structured.get("rating") or "review"
    fm["cull_reason"] = structured.get("cull_reason") or ""
    fm["technical"] = structured.get("technical") or {
        "focus": "unclear",
        "exposure": "unclear",
        "composition": "unclear",
    }
    fm["lighting"] = structured.get("lighting") or "unclear"
    fm["time_of_day"] = structured.get("time_of_day") or "unclear"
    fm["dominant_color_palette"] = structured.get("dominant_color_palette") or ""
    fm["dominant_colors"] = structured.get("dominant_colors") or []
    fm["scene_type"] = structured.get("scene_type") or "unclear"
    fm["people_count"] = coerce_people_count(
        structured.get("people_count", 0), len(faces)
    )
    fm["keywords"] = structured.get("keywords") or []

    if faces:
        fm["faces"] = [f.to_sidecar_dict() for f in faces]
        fm["face_count"] = len(faces)
    else:
        fm["faces"] = []
        fm["face_count"] = 0

    if extra_frontmatter:
        for k, v in extra_frontmatter.items():
            fm[k] = v
    return fm


# ---------------------------------------------------------------------------
# Per-photo pipeline
# ---------------------------------------------------------------------------


def process_one_image(
    image: Path,
    root: Path,
    opts: pipeline.ProcessOptions,
    ctx: pipeline.ProcessContext,
    *,
    sidecar_path_override: Path | None = None,
    parent_folder_override: str | None = None,
    metadata_override: dict[str, Any] | None = None,
    gps_override: dict[str, Any] | None = None,
    place_override: str | None = None,
    extra_frontmatter: dict[str, Any] | None = None,
    omit_path: bool = False,
) -> pipeline.ProcessResult:
    """Run the full per-photo pipeline for one still and emit a sidecar.

    Mirrors process_one_video's override surface so fdx-photos can reuse it for
    Apple Photos stills. `metadata_override` shallow-merges over exiftool's
    output (e.g. Photos' canonical creation_time). Returns
    skipped_reason='no_preview' for a RAW with no embedded preview to read."""
    metadata = get_image_metadata(image)
    if metadata_override:
        # Caller (e.g. fdx-photos) has authoritative fields from a richer source.
        # Shallow-merge over exiftool so a field like creation_time can be
        # corrected without losing dimensions/camera/size.
        metadata.update(metadata_override)

    gps = gps_override if gps_override is not None else pipeline.get_gps(image)
    if place_override is not None:
        place = place_override
    elif gps.get("lat") is not None and ctx.geocoder is not None:
        place = ctx.geocoder.reverse(gps["lat"], gps["lon"])
        if place:
            print(f"  location: {place}")
    else:
        place = ""

    tmp_dir = Path(tempfile.mkdtemp(prefix="fdx-image-"))
    detected_faces: list[face_db.DetectedFace] = []
    face_detection_ran = False
    structured: dict[str, Any] = {}
    description: str = ""
    try:
        preview = render_preview(image, tmp_dir)
        if preview is None:
            return pipeline.ProcessResult(sidecar=None, skipped_reason="no_preview")

        context = {
            "filename": image.name,
            "parent_folder": parent_folder_override
            if parent_folder_override is not None
            else image.parent.name,
            "creation_time": metadata.get("creation_time", ""),
            "camera": metadata.get("camera", {}),
            "location": {**gps, "place": place} if gps else {"place": place},
        }

        if opts.backend == "api":
            assert ctx.api_client is not None
            prompt = _build_image_vision_prompt(preview, context, include_paths=False)
            raw = pipeline.describe_frames_api(
                ctx.api_client, [preview], prompt, opts.vision_model_id
            )
        elif opts.backend == "cli":
            prompt = _build_image_vision_prompt(preview, context, include_paths=True)
            raw = pipeline.describe_frames_cli([preview], prompt, opts.vision_model_id)
            time.sleep(CLI_INTER_CALL_DELAY)
        else:  # local
            prompt = _build_image_vision_prompt(preview, context, include_paths=False)
            raw = pipeline.describe_frames_local(
                [preview], prompt, opts.local_base_url, opts.local_model
            )

        structured, description = pipeline.parse_vision_response(raw)

        if ctx.face_conn is not None:
            try:
                detected_faces = face_db.detect_faces_in_frames([preview], [0.0])
                face_detection_ran = True
            except Exception as e:
                print(f"  face detection failed: {e}")
    finally:
        for f in tmp_dir.glob("*"):
            f.unlink(missing_ok=True)
        tmp_dir.rmdir()

    # No structured YAML parsed → the result is unusable. Skip and retry next
    # run rather than persist a defaults-only sidecar (rating:review, all
    # "unclear") that would mark the photo indexed forever. Covers both a
    # transport "[...]" sentinel and a response that carried no parseable fence.
    if not structured:
        if description.startswith("["):
            print(f"  vision call failed: {description[:200]}")
        else:
            print("  vision response had no parsable YAML block — will retry next run")
        return pipeline.ProcessResult(sidecar=None, skipped_reason="vision_error")

    fm = build_image_frontmatter(
        image,
        root,
        metadata,
        gps,
        place,
        structured,
        detected_faces,
        parent_folder_override=parent_folder_override,
        extra_frontmatter=extra_frontmatter,
        omit_path=omit_path,
    )
    sidecar = sidecar_path_override or pipeline.sidecar_path(image)
    # Faces first, sidecar last: the sidecar is the resume marker, so it must be
    # the last thing written for a file — a crash in the gap re-runs the file
    # cleanly (write_faces deletes prior rows first). Write only when detection
    # actually ran: a *successful* zero-face detection still writes (clearing
    # stale rows via the DELETE), but a detection *failure* must not — otherwise
    # a transient error would wipe previously-committed faces for this file.
    if ctx.face_conn is not None and face_detection_ran:
        face_db.write_faces(ctx.face_conn, image, sidecar, detected_faces)
    pipeline.serialize_sidecar(sidecar, fm, image.name, [("Description", description)])

    return pipeline.ProcessResult(
        sidecar=sidecar,
        detected_faces=detected_faces,
        cost=opts.cost_per_call,
        skipped_reason=None,
        rating=str(structured.get("rating", "?")),
        structured=structured,
    )
