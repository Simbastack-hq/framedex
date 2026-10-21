"""
framedex.develop_engine — the RAW-engine adapter behind `fdx-develop`.

The vision model speaks a small engine-neutral "edit" vocabulary (exposure,
contrast, white balance, HSL, color grading, a tone curve, vignette, up to three
radial/graduated masks). This module maps that onto RapidRAW's adjustments
JSON, clamps every value to the engine's slider ranges, converts mask geometry
from fractions of the cropped frame to engine pixels, and runs the user's
installed RapidRAW binary headless as a separate program (never vendored or
linked, so framedex stays MIT).

The pure core (mapping, parsing, crop geometry, command lines) is stdlib-only.
Pixel work (crop, resize, render statistics) imports Pillow lazily.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

ENGINE_NAME = "rapidraw"
ENGINE_ENV = "FDX_RAPIDRAW"
MAC_APP_BINARY = Path("/Applications/RapidRAW.app/Contents/MacOS/RapidRAW")
RENDER_TIMEOUT_SEC = (
    600  # a 45 MP RAW takes ~4 s on Apple silicon; CPU-only boxes are far slower
)
RENDER_QUALITY = (
    95  # the engine's JPEG is an intermediate; framedex re-encodes the deliverable
)
MAX_MASKS = 3

HSL_BANDS = (
    "reds",
    "oranges",
    "yellows",
    "greens",
    "aquas",
    "blues",
    "purples",
    "magentas",
)
BASIC_KEYS = ("exposure", "contrast", "highlights", "shadows", "whites", "blacks")
PRESENCE_KEYS = ("clarity", "dehaze", "texture", "vibrance", "saturation")
MASK_ADJ_KEYS = (
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "saturation",
    "temperature",
    "tint",
    "clarity",
    "dehaze",
)
GRADE_ZONES = ("shadows", "midtones", "highlights")

# EXIF orientation as exiftool prints it (no -n) -> numeric tag value.
ORIENTATION_BY_NAME = {
    "horizontal (normal)": 1,
    "rotate 180": 3,
    "rotate 90 cw": 6,
    "rotate 270 cw": 8,
}


def _band() -> dict[str, float]:
    return {"hue": 0, "saturation": 0, "luminance": 0}


def default_edit() -> dict[str, Any]:
    """The neutral edit: every control at its engine default."""
    return {
        "what_i_see": "",
        "intent": "",
        "changes_this_round": "",
        "done": False,
        "tone_mapper": "basic",
        "basic": dict.fromkeys(BASIC_KEYS, 0),
        "white_balance": {"temperature": 0, "tint": 0},
        "presence": dict.fromkeys(PRESENCE_KEYS, 0),
        "detail": {"sharpness": 0, "luma_noise_reduction": 0},
        "hsl": {b: _band() for b in HSL_BANDS},
        "color_grading": {
            **{z: _band() for z in GRADE_ZONES},
            "blending": 50,
            "balance": 0,
        },
        "tone_curve": [{"x": 0, "y": 0}, {"x": 255, "y": 255}],
        "vignette": {"amount": 0, "midpoint": 50, "feather": 50},
        "masks": [],
    }


def default_mask() -> dict[str, Any]:
    return {
        "name": "mask",
        "kind": "graduated",
        "invert": False,
        "radial": {
            "center_x": 0.5,
            "center_y": 0.5,
            "radius_x": 0.3,
            "radius_y": 0.3,
            "feather": 0.5,
        },
        "graduated": {"edge": "top", "position": 0.35, "softness": 0.1},
        "adjustments": dict.fromkeys(MASK_ADJ_KEYS, 0),
    }


def _merge(base: Any, new: Any) -> Any:
    """Deep-merge `new` onto `base`, keeping base's shape: dict keys the base
    doesn't know are dropped, and a type mismatch keeps the base value."""
    if isinstance(base, dict):
        if not isinstance(new, dict):
            return base
        return {k: _merge(v, new[k]) if k in new else v for k, v in base.items()}
    if isinstance(base, list):
        return new if isinstance(new, list) else base
    if isinstance(base, bool):
        return new if isinstance(new, bool) else base
    if isinstance(base, (int, float)):
        return (
            new if isinstance(new, (int, float)) and not isinstance(new, bool) else base
        )
    if isinstance(base, str):
        return new if isinstance(new, str) else base
    return new


def normalize_edit(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge a model's (possibly partial or sloppy) edit onto the neutral one.
    Missing keys mean neutral, never a crash."""
    edit: dict[str, Any] = _merge(default_edit(), raw)
    masks = raw.get("masks") if isinstance(raw.get("masks"), list) else []
    edit["masks"] = [
        _merge(default_mask(), m) for m in (masks or []) if isinstance(m, dict)
    ][:MAX_MASKS]
    curve = raw.get("tone_curve")
    edit["tone_curve"] = curve if isinstance(curve, list) else edit["tone_curve"]
    return edit


def extract_json_object(text: str) -> dict[str, Any]:
    """The first-`{` to last-`}` span of a model reply, parsed. Tolerates prose
    or a code fence around the object; raises ValueError if there is none."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object in reply")
    obj = json.loads(text[a : b + 1])
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def _num(v: Any, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return max(lo, min(hi, f))


def sanitize_curve(points: Any) -> list[dict[str, float]]:
    """Clamp to 0..255, dedupe x, sort, force the 0 and 255 endpoints, cap at
    16 points (the engine's limit)."""
    pts: dict[float, float] = {}
    for p in points if isinstance(points, list) else []:
        if isinstance(p, dict) and "x" in p and "y" in p:
            pts[_num(p["x"], 0, 255)] = _num(p["y"], 0, 255)
    pts.setdefault(0.0, 0.0)
    pts.setdefault(255.0, 255.0)
    ordered = sorted(pts.items())
    if len(ordered) > 16:
        ordered = [*ordered[:15], ordered[-1]]
    return [{"x": x, "y": y} for x, y in ordered]


# ---------------------------------------------------------------------------
# Crop (Lightroom's crs:Crop*, honored, never invented)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Crop:
    """Lightroom crop: (left, top) is the crop's top-left corner and
    (right, bottom) its bottom-right corner, normalized to the un-oriented
    sensor frame; the rectangle is rotated by `angle` degrees."""

    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0
    angle: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.left <= 0
            and self.top <= 0
            and self.right >= 1
            and self.bottom >= 1
            and abs(self.angle) < 1e-6
        )


IDENTITY_CROP = Crop()

_CRS_ATTR = r'crs:{name}\s*=\s*"([^"]*)"'
_CRS_ELEM = r"<crs:{name}>([^<]*)</crs:{name}>"


def _crs_value(xmp: str, name: str) -> str | None:
    for pat in (_CRS_ATTR, _CRS_ELEM):
        m = re.search(pat.format(name=name), xmp)
        if m:
            return m.group(1).strip()
    return None


def parse_lr_crop(xmp: str) -> Crop | None:
    """Lightroom/ACR crop from XMP text (sidecar or embedded packet), or None
    when there is no active crop."""
    has = _crs_value(xmp, "HasCrop")
    if has is None or has.lower() != "true":
        return None
    vals: dict[str, float] = {}
    for name, key, default in (
        ("CropLeft", "left", 0.0),
        ("CropTop", "top", 0.0),
        ("CropRight", "right", 1.0),
        ("CropBottom", "bottom", 1.0),
        ("CropAngle", "angle", 0.0),
    ):
        raw = _crs_value(xmp, name)
        try:
            vals[key] = float(raw) if raw is not None else default
        except ValueError:
            vals[key] = default
    crop = Crop(**vals)
    if not (0 <= crop.left < crop.right <= 1 and 0 <= crop.top < crop.bottom <= 1):
        return None
    return None if crop.is_identity else crop


def orientation_code(value: Any) -> int:
    """exiftool's Orientation (string or number) -> 1, 3, 6 or 8."""
    if isinstance(value, (int, float)) and int(value) in (1, 3, 6, 8):
        return int(value)
    return ORIENTATION_BY_NAME.get(str(value or "").strip().lower(), 1)


def crop_box_display(
    full_size: tuple[int, int], crop: Crop, orientation: int
) -> tuple[float, float, float, float]:
    """Axis-aligned crop box (x0, y0, x1, y1) in displayed full-frame pixels,
    ignoring the crop angle: RapidRAW's own `crop` field for unrotated crops."""
    wd, hd = full_size
    w, h = (hd, wd) if orientation in (6, 8) else (wd, hd)
    corners = [(crop.left * w, crop.top * h), (crop.right * w, crop.bottom * h)]
    disp: list[tuple[float, float]] = []
    for x, y in corners:
        if orientation == 3:
            disp.append((w - x, h - y))
        elif orientation == 6:  # displayed = sensor rotated 90° clockwise
            disp.append((h - y, x))
        elif orientation == 8:  # displayed = sensor rotated 90° counter-clockwise
            disp.append((y, w - x))
        else:
            disp.append((x, y))
    xs, ys = [p[0] for p in disp], [p[1] for p in disp]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Edit -> RapidRAW adjustments JSON
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameMap:
    """Maps fractions of the displayed, cropped frame (what the model sees) to
    displayed full-frame engine pixels, through the inverse of the crop
    (including its straightening angle) and the EXIF orientation."""

    full_size: tuple[int, int]
    crop: Crop = IDENTITY_CROP
    orientation: int = 1

    def _sensor(self) -> tuple[float, float, float, float, float, float, float]:
        wd, hd = self.full_size
        w, h = (hd, wd) if self.orientation in (6, 8) else (wd, hd)
        c = self.crop
        left, top, right, bottom = c.left * w, c.top * h, c.right * w, c.bottom * h
        th = math.radians(c.angle)
        dx, dy = right - left, bottom - top
        cw = dx * math.cos(th) + dy * math.sin(th)
        ch = -dx * math.sin(th) + dy * math.cos(th)
        return w, h, (left + right) / 2, (top + bottom) / 2, cw, ch, th

    @property
    def frame_size(self) -> tuple[float, float]:
        """(width, height) of the displayed cropped frame in pixels."""
        _, _, _, _, cw, ch, _ = self._sensor()
        return (ch, cw) if self.orientation in (6, 8) else (cw, ch)

    def __call__(self, u: float, v: float) -> tuple[float, float]:
        w, h, cx, cy, cw, ch, th = self._sensor()
        # displayed-frame fractions -> sensor-frame fractions
        if self.orientation == 3:
            us, vs = 1 - u, 1 - v
        elif self.orientation == 6:  # displayed = sensor rotated 90° clockwise
            us, vs = v, 1 - u
        elif self.orientation == 8:  # displayed = sensor rotated 90° counter-clockwise
            us, vs = 1 - v, u
        else:
            us, vs = u, v
        # undo the straightening rotation about the crop centre
        ox, oy = (us - 0.5) * cw, (vs - 0.5) * ch
        sx = cx + ox * math.cos(th) - oy * math.sin(th)
        sy = cy + ox * math.sin(th) + oy * math.cos(th)
        # sensor pixels -> displayed full-frame pixels
        if self.orientation == 3:
            return w - sx, h - sy
        if self.orientation == 6:
            return h - sy, sx
        if self.orientation == 8:
            return sy, w - sx
        return sx, sy


def _mask_to_engine(i: int, mask: dict[str, Any], fmap: FrameMap) -> dict[str, Any]:
    fw, fh = fmap.frame_size
    params: dict[str, Any]
    if mask.get("kind") == "radial":
        r = mask["radial"]
        u = _num(r["center_x"], -0.5, 1.5, 0.5)
        v = _num(r["center_y"], -0.5, 1.5, 0.5)
        rx = max(1.0 / max(fw, 1.0), _num(r["radius_x"], 0, 3, 0.3))
        ry = max(1.0 / max(fh, 1.0), _num(r["radius_y"], 0, 3, 0.3))
        cx, cy = fmap(u, v)
        ax, ay = fmap(u + rx, v)  # end of the ellipse's own x-axis
        bx, by = fmap(u, v + ry)
        params = {
            "centerX": cx,
            "centerY": cy,
            "radiusX": max(1.0, math.hypot(ax - cx, ay - cy)),
            "radiusY": max(1.0, math.hypot(bx - cx, by - cy)),
            # the engine rotates the ellipse's x-axis to (cos r, sin r) on screen
            "rotation": math.degrees(math.atan2(ay - cy, ax - cx)),
            "feather": _num(r["feather"], 0, 1, 0.5),
        }
        kind = "radial"
    else:
        g = mask["graduated"]
        pos = _num(g["position"], 0, 1, 0.35)
        soft = _num(g["softness"], 0.01, 1, 0.1)
        edge = g.get("edge", "top")
        # The engine's linear mask is fully on at -range from the start->end
        # line, on the left-hand side of that direction (screen coordinates).
        # Orientation and straightening are rotations, so that side survives
        # the mapping.
        if edge == "bottom":
            (su, sv), (eu, ev), rng = (1.0, 1 - pos), (0.0, 1 - pos), soft * fh
        elif edge == "left":
            (su, sv), (eu, ev), rng = (pos, 1.0), (pos, 0.0), soft * fw
        elif edge == "right":
            (su, sv), (eu, ev), rng = (1 - pos, 0), (1 - pos, 1), soft * fw
        else:
            (su, sv), (eu, ev), rng = (0, pos), (1, pos), soft * fh
        start, end = fmap(su, sv), fmap(eu, ev)
        params = {
            "startX": start[0],
            "startY": start[1],
            "endX": end[0],
            "endY": end[1],
            "range": max(1.0, rng),
        }
        kind = "linear"
    adj = {
        k: _num(mask["adjustments"].get(k), -5, 5)
        if k == "exposure"
        else _num(mask["adjustments"].get(k), -100, 100)
        for k in MASK_ADJ_KEYS
    }
    return {
        "id": f"fdx-mask-{i}",
        "name": str(mask.get("name", "mask"))[:40],
        "visible": True,
        "invert": bool(mask.get("invert", False)),
        "opacity": 100,
        "adjustments": adj,
        "subMasks": [
            {
                "id": f"fdx-sub-{i}",
                "type": kind,
                "visible": True,
                "invert": False,
                "opacity": 100,
                "mode": "additive",
                "parameters": params,
            }
        ],
    }


def to_rapidraw(
    edit: dict[str, Any],
    full_size: tuple[int, int],
    crop: Crop = IDENTITY_CROP,
    orientation: int = 1,
) -> dict[str, Any]:
    """A normalized edit -> RapidRAW adjustments JSON (as passed to
    `rapidraw export --adjustments` and stored in `.rrdata`)."""
    b, wb, pr, d = (
        edit["basic"],
        edit["white_balance"],
        edit["presence"],
        edit["detail"],
    )
    cg = edit["color_grading"]
    ident = [{"x": 0, "y": 0}, {"x": 255, "y": 255}]
    adj: dict[str, Any] = {
        "toneMapper": "agx" if edit.get("tone_mapper") == "agx" else "basic",
        "exposure": _num(b["exposure"], -5, 5),
        **{k: _num(b[k], -100, 100) for k in BASIC_KEYS if k != "exposure"},
        "temperature": _num(wb["temperature"], -100, 100),
        "tint": _num(wb["tint"], -100, 100),
        "clarity": _num(pr["clarity"], -100, 100),
        "dehaze": _num(pr["dehaze"], -100, 100),
        "structure": _num(
            pr["texture"], -100, 100
        ),  # RapidRAW calls texture "structure"
        "vibrance": _num(pr["vibrance"], -100, 100),
        "saturation": _num(pr["saturation"], -100, 100),
        "sharpness": _num(d["sharpness"], 0, 100),
        "lumaNoiseReduction": _num(d["luma_noise_reduction"], 0, 100),
        "hsl": {
            band: {
                k: _num(edit["hsl"][band][k], -100, 100)
                for k in ("hue", "saturation", "luminance")
            }
            for band in HSL_BANDS
        },
        "colorGrading": {
            **{
                z: {
                    "hue": _num(cg[z]["hue"], 0, 360),
                    "saturation": _num(cg[z]["saturation"], 0, 100),
                    "luminance": _num(cg[z]["luminance"], -100, 100),
                }
                for z in GRADE_ZONES
            },
            "global": {"hue": 0, "saturation": 0, "luminance": 0},
            "blending": _num(cg["blending"], 0, 100, 50),
            "balance": _num(cg["balance"], -100, 100),
        },
        "curveMode": "point",
        "curves": {
            "luma": sanitize_curve(edit["tone_curve"]),
            "red": ident,
            "green": ident,
            "blue": ident,
        },
        "vignetteAmount": _num(edit["vignette"]["amount"], -100, 100),
        "vignetteMidpoint": _num(edit["vignette"]["midpoint"], 0, 100, 50),
        "vignetteFeather": _num(edit["vignette"]["feather"], 0, 100, 50),
    }
    fmap = FrameMap(full_size, crop, orientation)
    adj["masks"] = [
        _mask_to_engine(i, m, fmap) for i, m in enumerate(edit["masks"][:MAX_MASKS])
    ]
    return adj


# ---------------------------------------------------------------------------
# The engine binary
# ---------------------------------------------------------------------------


def find_engine(explicit: str | None = None) -> Path | None:
    """--engine, then $FDX_RAPIDRAW, then `rapidraw` on PATH, then the macOS app."""
    for cand in (explicit, os.environ.get(ENGINE_ENV)):
        if cand:
            p = Path(cand).expanduser()
            if p.suffix == ".app":
                p = p / "Contents" / "MacOS" / "RapidRAW"
            return p if p.is_file() else None
    on_path = shutil.which("rapidraw") or shutil.which("RapidRAW")
    if on_path:
        return Path(on_path)
    return MAC_APP_BINARY if MAC_APP_BINARY.is_file() else None


def engine_version(engine: Path) -> str:
    """CFBundleShortVersionString for a macOS .app install, else 'unknown'."""
    plist = engine.parent.parent / "Info.plist"
    try:
        m = re.search(
            r"<key>CFBundleShortVersionString</key>\s*<string>([^<]+)</string>",
            plist.read_text(errors="replace"),
        )
    except OSError:
        return "unknown"
    return m.group(1) if m else "unknown"


def engine_command(
    engine: Path, raw: Path, out: Path, adjustments_path: Path
) -> list[str]:
    return [
        str(engine),
        "export",
        str(raw),
        "--output",
        str(out),
        "--format",
        "jpeg",
        "--quality",
        str(RENDER_QUALITY),
        "--adjustments",
        str(adjustments_path),
    ]


def render(
    engine: Path,
    raw: Path,
    adjustments: dict[str, Any],
    work_dir: Path,
    tag: str,
) -> Path:
    """Run the engine headless; return the rendered full-frame JPEG path (in
    work_dir). The adjustments override any sidecar the engine would read.
    Raises RuntimeError with the engine's own output on failure."""
    work_dir.mkdir(parents=True, exist_ok=True)
    adj_path = work_dir / f"{tag}.adjustments.json"
    adj_path.write_text(json.dumps(adjustments))
    out = work_dir / f"{tag}.render.jpg"
    out.unlink(missing_ok=True)
    try:
        p = subprocess.run(
            engine_command(engine, raw, out, adj_path),
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT_SEC,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"{ENGINE_NAME} timed out after {RENDER_TIMEOUT_SEC}s on {raw.name}"
        ) from e
    if p.returncode != 0 or not out.is_file():
        tail = (p.stderr or p.stdout or "").strip()[-400:]
        raise RuntimeError(
            f"{ENGINE_NAME} failed on {raw.name} (exit {p.returncode}): {tail}"
        )
    return out


def rrdata_path(raw: Path) -> Path:
    """RapidRAW's own sidecar: `<file name>.rrdata` next to the image."""
    return raw.with_name(raw.name + ".rrdata")


def rrdata_payload(adjustments: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "rating": 0, "adjustments": adjustments, "tags": None}


# ---------------------------------------------------------------------------
# Pixels (Pillow, imported lazily)
# ---------------------------------------------------------------------------


def apply_crop(img: PILImage, crop: Crop, orientation: int = 1) -> PILImage:
    """Apply a Lightroom crop to a displayed-orientation render."""
    from PIL import Image

    if crop.is_identity:
        return img
    unorient = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_90}
    reorient = {3: Image.Transpose.ROTATE_180, 6: Image.Transpose.ROTATE_270}
    unorient[8] = Image.Transpose.ROTATE_270
    reorient[8] = Image.Transpose.ROTATE_90
    if orientation in unorient:
        img = img.transpose(unorient[orientation])
    w, h = img.size
    left, top = crop.left * w, crop.top * h
    right, bottom = crop.right * w, crop.bottom * h
    th = math.radians(crop.angle)
    dx, dy = right - left, bottom - top
    cw = dx * math.cos(th) + dy * math.sin(th)
    ch = -dx * math.sin(th) + dy * math.cos(th)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    if abs(th) > 1e-6:
        img = img.rotate(crop.angle, resample=Image.Resampling.BICUBIC, center=(cx, cy))
    img = img.crop(
        (
            round(cx - cw / 2),
            round(cy - ch / 2),
            round(cx + cw / 2),
            round(cy + ch / 2),
        )
    )
    if orientation in reorient:
        img = img.transpose(reorient[orientation])
    return img


def downscale(img: PILImage, long_edge: int) -> PILImage:
    from PIL import Image

    out = img.copy()
    out.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return out


def jpeg_bytes(img: PILImage, quality: int = 88) -> bytes:
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def render_stats(img: PILImage) -> str:
    """Measured summary the model can't eyeball from a downscaled preview."""
    from PIL import ImageChops, ImageStat

    rgb = img.convert("RGB")
    luma = rgb.convert("L")
    n = luma.size[0] * luma.size[1]
    hist = luma.histogram()
    mean = sum(i * c for i, c in enumerate(hist)) / n
    acc, median = 0, 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= n / 2:
            median = i
            break
    r, g, b = rgb.split()
    brightest = ImageChops.lighter(ImageChops.lighter(r, g), b).histogram()
    clipped = sum(brightest[250:]) / n * 100
    black = sum(hist[:5]) / n * 100
    means = ImageStat.Stat(rgb).mean
    bins = [round(sum(hist[i : i + 16]) / n * 100, 1) for i in range(0, 256, 16)]
    return (
        f"mean luma {mean:.0f}/255, median {median}; "
        f"{clipped:.2f}% of pixels have a channel >=250 (clipping risk); "
        f"{black:.2f}% near-black (<=4); "
        f"mean R/G/B {means[0]:.0f}/{means[1]:.0f}/{means[2]:.0f}; "
        f"luma histogram, 16 bins dark->bright, % of pixels: {bins}"
    )
