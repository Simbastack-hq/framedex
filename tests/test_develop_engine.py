"""Hermetic tests for framedex.develop_engine: no engine binary, no network,
no real photos. Pixel tests need Pillow and skip without it."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest

from framedex import develop_engine as eng

# ---------------------------------------------------------------------------
# Edit normalization + parsing
# ---------------------------------------------------------------------------


def test_normalize_fills_missing_keys_with_neutral() -> None:
    e = eng.normalize_edit({"basic": {"exposure": 0.7}})
    assert e["basic"]["exposure"] == 0.7
    assert e["basic"]["contrast"] == 0
    assert e["hsl"]["oranges"] == {"hue": 0, "saturation": 0, "luminance": 0}
    assert e["tone_curve"] == [{"x": 0, "y": 0}, {"x": 255, "y": 255}]
    assert e["masks"] == []


def test_normalize_keeps_base_type_on_mismatch_and_drops_unknown_keys() -> None:
    e = eng.normalize_edit(
        {"basic": {"exposure": "lots", "shadows": True}, "bogus": 1, "done": "yes"}
    )
    assert e["basic"]["exposure"] == 0
    assert e["basic"]["shadows"] == 0
    assert "bogus" not in e
    assert e["done"] is False


def test_normalize_caps_masks_and_drops_non_dicts() -> None:
    masks: list[Any] = [{"kind": "radial"}] * 5 + ["nope"]
    e = eng.normalize_edit({"masks": masks})
    assert len(e["masks"]) == eng.MAX_MASKS
    assert e["masks"][0]["kind"] == "radial"
    assert set(e["masks"][0]["adjustments"]) == set(eng.MASK_ADJ_KEYS)


@pytest.mark.parametrize(
    "text",
    [
        '{"done": true}',
        'Sure! Here you go:\n```json\n{"done": true}\n```\nthanks',
        'prose {"done": true, "basic": {"exposure": 1}} trailing',
    ],
)
def test_extract_json_object_tolerates_wrapping(text: str) -> None:
    assert eng.extract_json_object(text)["done"] is True


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2]", "{not json}"])
def test_extract_json_object_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError):
        eng.extract_json_object(text)


def test_sanitize_curve_forces_endpoints_clamps_and_sorts() -> None:
    pts = eng.sanitize_curve([{"x": 300, "y": -5}, {"x": 128, "y": 140}, {"x": 64}])
    assert pts == [
        {"x": 0.0, "y": 0.0},
        {"x": 128.0, "y": 140.0},
        {"x": 255.0, "y": 0.0},
    ]


def test_sanitize_curve_caps_points_keeping_the_white_point() -> None:
    pts = eng.sanitize_curve([{"x": i * 10, "y": i * 10} for i in range(26)])
    assert len(pts) == 16
    assert pts[-1]["x"] == 255.0


def test_sanitize_curve_garbage_is_identity() -> None:
    assert eng.sanitize_curve("nope") == [
        {"x": 0.0, "y": 0.0},
        {"x": 255.0, "y": 255.0},
    ]


# ---------------------------------------------------------------------------
# Edit -> RapidRAW adjustments
# ---------------------------------------------------------------------------


def _edit(**over: Any) -> dict[str, Any]:
    return eng.normalize_edit(over)


def test_to_rapidraw_clamps_and_renames() -> None:
    e = _edit(
        basic={"exposure": 9, "contrast": -500, "highlights": float("nan")},
        presence={"texture": 30},
        color_grading={"shadows": {"hue": 720, "saturation": 150}},
        tone_mapper="agx",
    )
    adj = eng.to_rapidraw(e, (1000, 500))
    assert adj["exposure"] == 5
    assert adj["contrast"] == -100
    assert adj["highlights"] == 0
    assert adj["structure"] == 30  # RapidRAW's name for texture
    assert adj["colorGrading"]["shadows"]["hue"] == 360
    assert adj["colorGrading"]["shadows"]["saturation"] == 100
    assert adj["toneMapper"] == "agx"
    assert adj["curveMode"] == "point"
    assert adj["masks"] == []


def test_to_rapidraw_unknown_tone_mapper_falls_back_to_basic() -> None:
    assert (
        eng.to_rapidraw(_edit(tone_mapper="filmic"), (10, 10))["toneMapper"] == "basic"
    )


def _mask(kind: str, **geo: Any) -> dict[str, Any]:
    m: dict[str, Any] = {"kind": kind, "adjustments": {"exposure": -9, "clarity": 500}}
    m[kind] = geo
    return m


def test_radial_mask_maps_frame_fractions_to_full_frame_pixels() -> None:
    e = _edit(
        masks=[
            _mask(
                "radial",
                center_x=0.5,
                center_y=0.25,
                radius_x=0.1,
                radius_y=0.2,
                feather=0.4,
            )
        ]
    )
    crop = eng.Crop(left=0.1, top=0.2, right=0.6, bottom=0.8)
    m = eng.to_rapidraw(e, (1000, 500), crop)["masks"][0]
    p = m["subMasks"][0]["parameters"]
    assert m["subMasks"][0]["type"] == "radial"
    # crop box is x 100..600, y 100..400 (frame 500 x 300)
    assert p["centerX"] == pytest.approx(350)
    assert p["centerY"] == pytest.approx(175)
    assert p["radiusX"] == pytest.approx(50)
    assert p["radiusY"] == pytest.approx(60)
    assert p["feather"] == pytest.approx(0.4)
    assert m["adjustments"]["exposure"] == -5
    assert m["adjustments"]["clarity"] == 100


@pytest.mark.parametrize(
    ("edge", "start", "end"),
    [
        ("top", (0, 30), (1000, 30)),
        ("bottom", (1000, 70), (0, 70)),
        ("left", (300, 100), (300, 0)),
        ("right", (700, 0), (700, 100)),
    ],
)
def test_graduated_mask_direction_puts_the_edge_on_the_masked_side(
    edge: str, start: tuple[float, float], end: tuple[float, float]
) -> None:
    e = _edit(masks=[_mask("graduated", edge=edge, position=0.3, softness=0.1)])
    p = eng.to_rapidraw(e, (1000, 100))["masks"][0]["subMasks"][0]["parameters"]
    assert (p["startX"], p["startY"]) == pytest.approx(start)
    assert (p["endX"], p["endY"]) == pytest.approx(end)
    # The engine masks the left-hand side of start->end: check the edge is there.
    dx, dy = p["endX"] - p["startX"], p["endY"] - p["startY"]
    edge_point = {
        "top": (500, 0),
        "bottom": (500, 100),
        "left": (0, 50),
        "right": (1000, 50),
    }[edge]
    px, py = edge_point[0] - p["startX"], edge_point[1] - p["startY"]
    perp = (-dy, dx)
    assert px * perp[0] + py * perp[1] < 0


def test_crop_box_display_handles_rotated_raws() -> None:
    crop = eng.Crop(left=0.25, top=0.0, right=0.75, bottom=0.5)
    # sensor 400x200, displayed rotated 90° CW -> 200x400
    assert eng.crop_box_display((200, 400), crop, 6) == pytest.approx(
        (100, 100, 200, 300)
    )
    # rotated 90° CCW
    assert eng.crop_box_display((200, 400), crop, 8) == pytest.approx(
        (0, 100, 100, 300)
    )
    assert eng.crop_box_display((400, 200), crop, 3) == pytest.approx(
        (100, 100, 300, 200)
    )
    assert eng.crop_box_display((400, 200), crop, 1) == pytest.approx(
        (100, 0, 300, 100)
    )


# ---------------------------------------------------------------------------
# Lightroom crop + orientation
# ---------------------------------------------------------------------------

ATTR_XMP = (
    '<rdf:Description crs:HasCrop="True" crs:CropLeft="0.1" crs:CropTop="0.2" '
    'crs:CropRight="0.9" crs:CropBottom="0.95" crs:CropAngle="-1.5"/>'
)
ELEM_XMP = (
    "<crs:HasCrop>True</crs:HasCrop><crs:CropLeft>0.1</crs:CropLeft>"
    "<crs:CropRight>0.8</crs:CropRight>"
)


def test_parse_lr_crop_attribute_form() -> None:
    assert eng.parse_lr_crop(ATTR_XMP) == eng.Crop(0.1, 0.2, 0.9, 0.95, -1.5)


def test_parse_lr_crop_element_form_defaults_missing_edges() -> None:
    assert eng.parse_lr_crop(ELEM_XMP) == eng.Crop(0.1, 0.0, 0.8, 1.0, 0.0)


@pytest.mark.parametrize(
    "xmp",
    [
        "",
        'crs:HasCrop="False" crs:CropLeft="0.1"',
        'crs:HasCrop="True"',  # identity
        'crs:HasCrop="True" crs:CropLeft="0.9" crs:CropRight="0.1"',  # inverted
        'crs:HasCrop="True" crs:CropLeft="x"',  # unparsable -> identity
    ],
)
def test_parse_lr_crop_none_when_no_usable_crop(xmp: str) -> None:
    assert eng.parse_lr_crop(xmp) is None


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("Rotate 270 CW", 8),
        ("rotate 90 cw", 6),
        ("Horizontal (normal)", 1),
        (3, 3),
        (None, 1),
        ("Mirror horizontal", 1),
    ],
)
def test_orientation_code(value: Any, code: int) -> None:
    assert eng.orientation_code(value) == code


# ---------------------------------------------------------------------------
# Engine binary
# ---------------------------------------------------------------------------


def test_find_engine_explicit_app_bundle(tmp_path: Path) -> None:
    binary = tmp_path / "RapidRAW.app" / "Contents" / "MacOS" / "RapidRAW"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    assert eng.find_engine(str(tmp_path / "RapidRAW.app")) == binary
    assert eng.find_engine(str(tmp_path / "missing")) is None


def test_find_engine_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = tmp_path / "rapidraw"
    binary.write_text("")
    monkeypatch.setenv(eng.ENGINE_ENV, str(binary))
    assert eng.find_engine() == binary


def test_engine_version_reads_app_plist(tmp_path: Path) -> None:
    binary = tmp_path / "RapidRAW.app" / "Contents" / "MacOS" / "RapidRAW"
    binary.parent.mkdir(parents=True)
    (binary.parent.parent / "Info.plist").write_text(
        "<key>CFBundleShortVersionString</key>\n<string>1.6.6</string>"
    )
    assert eng.engine_version(binary) == "1.6.6"
    assert eng.engine_version(tmp_path / "rapidraw") == "unknown"


def test_render_runs_headless_export_with_adjustments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen["cmd"] = cmd
        seen["adj"] = json.loads(Path(cmd[cmd.index("--adjustments") + 1]).read_text())
        Path(cmd[cmd.index("--output") + 1]).write_bytes(b"jpeg")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = eng.render(Path("/bin/rr"), Path("/x/a.NEF"), {"exposure": 1}, tmp_path, "r1")
    assert out.read_bytes() == b"jpeg"
    assert seen["cmd"][:3] == ["/bin/rr", "export", "/x/a.NEF"]
    assert seen["adj"] == {"exposure": 1}


def test_render_failure_raises_with_engine_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 3, "", "unsupported camera")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="unsupported camera"):
        eng.render(Path("/bin/rr"), Path("/x/a.NEF"), {}, tmp_path, "r1")


def test_rrdata_naming_and_payload() -> None:
    assert eng.rrdata_path(Path("/p/_DSC1.NEF")) == Path("/p/_DSC1.NEF.rrdata")
    assert eng.rrdata_payload({"exposure": 1})["adjustments"] == {"exposure": 1}


# ---------------------------------------------------------------------------
# Pixels (Pillow)
# ---------------------------------------------------------------------------


def _marked(size: tuple[int, int], box: tuple[int, int, int, int]) -> Any:
    image = pytest.importorskip("PIL.Image")
    img = image.new("RGB", size, (0, 0, 0))
    img.paste((255, 0, 0), box)
    return img


def test_apply_crop_extracts_the_rectangle() -> None:
    img = _marked((400, 200), (100, 50, 300, 150))
    out = eng.apply_crop(img, eng.Crop(0.25, 0.25, 0.75, 0.75))
    assert out.size == (200, 100)
    assert out.getpixel((100, 50)) == (255, 0, 0)


def test_apply_crop_rotated_raw_round_trip() -> None:
    # Sensor 400x200 with the crop rect marked; display = rotated 90° CCW (EXIF 8).
    image = pytest.importorskip("PIL.Image")
    sensor = _marked((400, 200), (100, 0, 300, 100))
    display = sensor.transpose(image.Transpose.ROTATE_90)
    out = eng.apply_crop(display, eng.Crop(0.25, 0.0, 0.75, 0.5), orientation=8)
    assert out.size == (100, 200)
    colors = {out.getpixel((x, y)) for x in (5, 50, 95) for y in (5, 100, 195)}
    assert colors == {(255, 0, 0)}


def test_apply_crop_angled_size_matches_lightroom_geometry() -> None:
    img = _marked((1000, 800), (0, 0, 1, 1))
    th = math.radians(2.0)
    w, h = 600, 400
    # Lightroom stores the rotated rect's TL and BR corners.
    dx, dy = w * math.cos(th) - h * math.sin(th), w * math.sin(th) + h * math.cos(th)
    left, top = 200, 150
    crop = eng.Crop(left / 1000, top / 800, (left + dx) / 1000, (top + dy) / 800, 2.0)
    assert eng.apply_crop(img, crop).size == (w, h)


def test_identity_crop_is_a_no_op() -> None:
    img = _marked((40, 20), (0, 0, 1, 1))
    assert eng.apply_crop(img, eng.IDENTITY_CROP) is img


def test_render_stats_reports_clipping() -> None:
    image = pytest.importorskip("PIL.Image")
    img = image.new("RGB", (10, 10), (255, 255, 255))
    s = eng.render_stats(img)
    assert "100.00% of pixels have a channel >=250" in s
    assert "mean luma 255/255" in s


def test_frame_map_inverts_an_angled_crop() -> None:
    # Codex's case: 1000x800, crop (0.1, 0.2, 0.8, 0.9) rotated 10°: the
    # frame point (0.8, 0.2) sits at about (705, 354) in the source.
    fmap = eng.FrameMap((1000, 800), eng.Crop(0.1, 0.2, 0.8, 0.9, 10.0))
    assert fmap(0.8, 0.2) == pytest.approx((704.8, 353.96), abs=0.1)


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_frame_map_matches_crop_box_when_unrotated(orientation: int) -> None:
    full = (200, 400) if orientation in (6, 8) else (400, 200)
    crop = eng.Crop(0.25, 0.1, 0.75, 0.5)
    fmap = eng.FrameMap(full, crop, orientation)
    x0, y0, x1, y1 = eng.crop_box_display(full, crop, orientation)
    corners = {fmap(0, 0), fmap(1, 1), fmap(0, 1), fmap(1, 0)}
    xs = sorted(round(c[0], 6) for c in corners)
    ys = sorted(round(c[1], 6) for c in corners)
    assert (xs[0], ys[0], xs[-1], ys[-1]) == pytest.approx((x0, y0, x1, y1))
    assert fmap(0, 0) == pytest.approx((x0, y0))  # top-left stays top-left


def test_radial_mask_follows_crop_rotation() -> None:
    e = eng.normalize_edit(
        {
            "masks": [
                {
                    "kind": "radial",
                    "radial": {
                        "center_x": 0.5,
                        "center_y": 0.5,
                        "radius_x": 0.2,
                        "radius_y": 0.1,
                    },
                }
            ]
        }
    )
    crop = eng.Crop(0.1, 0.2, 0.8, 0.9, 10.0)
    p = eng.to_rapidraw(e, (1000, 800), crop)["masks"][0]["subMasks"][0]["parameters"]
    assert p["rotation"] == pytest.approx(10.0)
    fw, fh = eng.FrameMap((1000, 800), crop).frame_size
    assert p["radiusX"] == pytest.approx(0.2 * fw)
    assert p["radiusY"] == pytest.approx(0.1 * fh)


@pytest.mark.parametrize("orientation", [1, 6, 8])
def test_frame_map_agrees_with_apply_crop_pixels(orientation: int) -> None:
    image = pytest.importorskip("PIL.Image")
    sensor = (600, 400)
    full = (400, 600) if orientation in (6, 8) else sensor
    crop = eng.Crop(0.15, 0.1, 0.8, 0.85, 3.0)
    fmap = eng.FrameMap(full, crop, orientation)
    x, y = fmap(0.3, 0.6)
    img = image.new("RGB", full, (0, 0, 0))
    img.paste((255, 255, 255), (round(x) - 2, round(y) - 2, round(x) + 3, round(y) + 3))
    out = eng.apply_crop(img, crop, orientation)
    w, h = out.size
    px = out.getpixel((round(0.3 * w), round(0.6 * h)))
    assert px[0] > 128, (orientation, out.size)
