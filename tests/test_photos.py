"""Tests for the Photos library adapter.

These exercise the pure projection / path-derivation helpers without
requiring osxphotos or a real Photos library — those layers are mocked or
stubbed. We stub osxphotos at sys.modules level before importing the
module since the module raises ImportError at top level otherwise.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest

# Stub osxphotos before importing framedex.photos. The stub doesn't need to
# do anything — none of the tested code paths reach osxphotos itself.
if "osxphotos" not in sys.modules:
    sys.modules["osxphotos"] = types.SimpleNamespace(  # type: ignore[assignment]
        PhotosDB=lambda **kw: None,
    )

from framedex import photos


def _make_asset(**overrides: Any) -> photos.PhotosAsset:
    base: dict[str, Any] = dict(
        uuid="ABCD1234-EF56-7890-ABCD-1234567890AB",
        filename="IMG_4827.MOV",
        date=datetime(2024, 8, 14, 7, 23, 11),
        lat=37.7456,
        lon=-119.5936,
        altitude_m=1842.5,
        persons=["Mom", "Dad"],
        albums=["Yosemite 2024"],
        keywords=["sunset", "drone"],
        is_edited=False,
        in_icloud=False,
        path_original=Path(
            "/Users/test/Pictures/Photos.photoslibrary/originals/A/file.MOV"
        ),
        raw=None,
    )
    base.update(overrides)
    return photos.PhotosAsset(**base)


class TestMirrorSidecarPath:
    def test_dated_asset_buckets_by_year_month(self, tmp_path: Path) -> None:
        asset = _make_asset(date=datetime(2024, 8, 14))
        out = photos.mirror_sidecar_path(asset, tmp_path)
        assert out.parent == tmp_path / "2024-08"
        assert out.name.endswith(".description.md")
        assert "IMG_4827" in out.name
        # UUID prefix is appended for disambiguation
        assert "ABCD1234"[:8] in out.name

    def test_undated_asset_goes_to_undated_bucket(self, tmp_path: Path) -> None:
        asset = _make_asset(date=None)
        out = photos.mirror_sidecar_path(asset, tmp_path)
        assert out.parent == tmp_path / "_undated"

    def test_two_assets_same_filename_get_distinct_paths(self, tmp_path: Path) -> None:
        # Two iPhone photos with the same IMG_0001 counter-rolled filename
        # must produce distinct sidecar paths — the UUID8 suffix is what
        # guarantees this.
        a = _make_asset(uuid="AAAAAAAA-1111-2222-3333-444444444444")
        b = _make_asset(uuid="BBBBBBBB-1111-2222-3333-444444444444")
        assert photos.mirror_sidecar_path(a, tmp_path) != photos.mirror_sidecar_path(
            b, tmp_path
        )

    def test_preserves_video_extension(self, tmp_path: Path) -> None:
        asset = _make_asset(filename="clip.mp4")
        out = photos.mirror_sidecar_path(asset, tmp_path)
        # .mp4 should appear before the sidecar suffix so file browsers
        # still associate it with a video
        assert ".mp4.description.md" in out.name

    def test_zero_byte_filename_falls_back_to_uuid_stem(self, tmp_path: Path) -> None:
        asset = _make_asset(filename="")
        out = photos.mirror_sidecar_path(asset, tmp_path)
        # When filename is empty, the uuid is used as the stem so we don't
        # produce a sidecar literally named ".description.md"
        assert out.name.startswith(asset.uuid)


class TestGpsOverride:
    def test_with_full_gps(self) -> None:
        asset = _make_asset()
        gps = photos.to_gps_override(asset)
        assert gps == {"lat": 37.7456, "lon": -119.5936, "altitude_m": 1842.5}

    def test_without_gps_returns_none(self) -> None:
        asset = _make_asset(lat=None, lon=None)
        assert photos.to_gps_override(asset) is None

    def test_partial_gps_lat_only_returns_none(self) -> None:
        # Defensive: a half-set GPS is not usable — better to fall through
        # to exiftool than emit a partial location block in the sidecar
        asset = _make_asset(lat=37.0, lon=None)
        assert photos.to_gps_override(asset) is None

    def test_no_altitude(self) -> None:
        asset = _make_asset(altitude_m=None)
        gps = photos.to_gps_override(asset)
        assert gps == {"lat": 37.7456, "lon": -119.5936}


class TestMetadataOverride:
    def test_with_date(self) -> None:
        asset = _make_asset(date=datetime(2024, 8, 14, 7, 23, 11))
        meta = photos.to_metadata_override(asset)
        assert meta == {"creation_time": "2024-08-14T07:23:11"}

    def test_no_date_returns_empty(self) -> None:
        asset = _make_asset(date=None)
        assert photos.to_metadata_override(asset) == {}


class TestExtraFrontmatter:
    def test_includes_uuid_persons_albums_keywords(self) -> None:
        asset = _make_asset()
        extra = photos.to_extra_frontmatter(asset)
        assert extra["photos_uuid"] == asset.uuid
        assert extra["photos_persons"] == ["Mom", "Dad"]
        assert extra["photos_albums"] == ["Yosemite 2024"]
        assert extra["photos_keywords"] == ["sunset", "drone"]

    def test_overrides_file_with_original_filename(self) -> None:
        # The on-disk filename inside .photoslibrary is UUID-based; users want
        # to see the original camera filename in the sidecar.
        asset = _make_asset(filename="IMG_4827.MOV")
        extra = photos.to_extra_frontmatter(asset)
        assert extra["file"] == "IMG_4827.MOV"
        assert extra["original_filename"] == "IMG_4827.MOV"

    def test_edited_flag_only_set_when_true(self) -> None:
        extra = photos.to_extra_frontmatter(_make_asset(is_edited=False))
        assert "photos_edited" not in extra
        extra = photos.to_extra_frontmatter(_make_asset(is_edited=True))
        assert extra["photos_edited"] is True


class TestParentFolder:
    def test_uses_first_album(self) -> None:
        asset = _make_asset(albums=["Yosemite 2024", "Summer 2024"])
        assert photos.parent_folder_for(asset) == "Yosemite 2024"

    def test_no_album_sentinel(self) -> None:
        asset = _make_asset(albums=[])
        assert photos.parent_folder_for(asset) == "(no album)"


class TestProjection:
    """Light tests against the _project helper using a duck-typed PhotoInfo."""

    def _make_photoinfo(self, **overrides: Any) -> types.SimpleNamespace:
        """Mock that quacks like osxphotos.PhotoInfo."""
        defaults = {
            "uuid": "ABCD1234-EF56-7890-ABCD-1234567890AB",
            "filename": "IMG_4827.MOV",
            "original_filename": "IMG_4827.MOV",
            "date": datetime(2024, 8, 14),
            "location": (37.7456, -119.5936),
            "persons": ["Mom"],
            "albums": ["Yosemite"],
            "keywords": ["sunset"],
            "hasadjustments": False,
            "path": None,
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_missing_path_marks_in_icloud(self, tmp_path: Path) -> None:
        # osxphotos 0.69+ uses `.path` (returns None for iCloud-only assets).
        photo = self._make_photoinfo(path=None)
        asset = photos._project(photo)
        assert asset.in_icloud is True
        assert asset.path_original is None

    def test_existing_path_marks_on_disk(self, tmp_path: Path) -> None:
        f = tmp_path / "x.MOV"
        f.write_bytes(b"fake")
        photo = self._make_photoinfo(path=str(f))
        asset = photos._project(photo)
        assert asset.in_icloud is False
        assert asset.path_original == f

    def test_path_that_does_not_exist_marks_in_icloud(self, tmp_path: Path) -> None:
        # osxphotos can return a path string even when the file's been evicted
        # to iCloud and isn't actually on disk. Treat that as in_icloud.
        photo = self._make_photoinfo(path=str(tmp_path / "ghost.MOV"))
        asset = photos._project(photo)
        assert asset.in_icloud is True

    def test_legacy_path_original_attribute_still_works(self, tmp_path: Path) -> None:
        # Backwards compat: older osxphotos (<0.69) only had `.path_original`.
        # Our projection should fall back to it when `.path` is missing.
        f = tmp_path / "y.MOV"
        f.write_bytes(b"fake")
        photo = types.SimpleNamespace(
            uuid="LEGACY00-0000-0000-0000-000000000000",
            filename="legacy.MOV",
            original_filename="legacy.MOV",
            date=None,
            location=None,
            persons=[],
            albums=[],
            keywords=[],
            hasadjustments=False,
            path_original=str(f),
            # Note: no `path` attribute at all
        )
        asset = photos._project(photo)
        assert asset.in_icloud is False
        assert asset.path_original == f

    def test_no_location_yields_none_coords(self) -> None:
        photo = self._make_photoinfo(location=None)
        asset = photos._project(photo)
        assert asset.lat is None and asset.lon is None

    def test_location_tuple_of_nones(self) -> None:
        # Some versions return (None, None) when GPS is absent rather than None
        photo = self._make_photoinfo(location=(None, None))
        asset = photos._project(photo)
        assert asset.lat is None and asset.lon is None

    def test_ismovie_true_marks_video(self) -> None:
        asset = photos._project(self._make_photoinfo(ismovie=True))
        assert asset.media_type == "video"

    def test_ismovie_false_marks_image(self) -> None:
        asset = photos._project(self._make_photoinfo(ismovie=False))
        assert asset.media_type == "image"

    def test_missing_ismovie_attr_defaults_to_image(self) -> None:
        # A PhotoInfo with no `ismovie` attribute (defensive) is treated as a
        # still — Live Photos and plain images are stills.
        photo = self._make_photoinfo()  # builder omits ismovie
        assert not hasattr(photo, "ismovie")
        asset = photos._project(photo)
        assert asset.media_type == "image"


class _FakeDB:
    """Captures the db.photos(images=, movies=) call and returns canned photos."""

    last_kwargs: ClassVar[dict[str, Any]] = {}
    canned: ClassVar[list[Any]] = []

    def __init__(self, **kw: Any) -> None:
        pass

    def photos(self, *, images: bool, movies: bool) -> list[Any]:
        _FakeDB.last_kwargs = {"images": images, "movies": movies}
        return list(_FakeDB.canned)


def _photoinfo(**overrides: Any) -> types.SimpleNamespace:
    defaults: dict[str, Any] = {
        "uuid": "U-0000",
        "filename": "x.jpg",
        "original_filename": "x.jpg",
        "date": datetime(2024, 6, 1),
        "location": None,
        "persons": [],
        "albums": [],
        "keywords": [],
        "hasadjustments": False,
        "path": None,
        "ismovie": False,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class TestEnumerateAssets:
    def test_media_images_sets_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeDB.canned = []
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        photos.enumerate_assets(Path("/lib"), media="images")
        assert _FakeDB.last_kwargs == {"images": True, "movies": False}

    def test_media_videos_sets_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeDB.canned = []
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        photos.enumerate_assets(Path("/lib"), media="videos")
        assert _FakeDB.last_kwargs == {"images": False, "movies": True}

    def test_media_all_sets_both_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeDB.canned = []
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        photos.enumerate_assets(Path("/lib"), media="all")
        assert _FakeDB.last_kwargs == {"images": True, "movies": True}

    def test_enumerate_videos_wrapper_is_movies_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakeDB.canned = []
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        photos.enumerate_videos(Path("/lib"))
        assert _FakeDB.last_kwargs == {"images": False, "movies": True}

    def test_date_bound_excludes_undated_assets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dated = _photoinfo(uuid="D", date=datetime(2024, 6, 1))
        undated = _photoinfo(uuid="U", date=None)
        _FakeDB.canned = [dated, undated]
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        out = photos.enumerate_assets(Path("/lib"), since=datetime(2024, 1, 1))
        uuids = {a.uuid for a in out}
        assert uuids == {"D"}  # undated asset cannot satisfy a date bound

    def test_tz_aware_asset_with_naive_since_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """osxphotos returns tz-aware dates in practice; a naive --since bound
        (from _parse_date on 'YYYY-MM-DD') must not raise on comparison."""
        aware = _photoinfo(uuid="A", date=datetime(2024, 6, 1, tzinfo=timezone.utc))
        _FakeDB.canned = [aware]
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        out = photos.enumerate_assets(Path("/lib"), since=datetime(2024, 1, 1))
        assert {a.uuid for a in out} == {"A"}

    def test_naive_asset_with_tz_aware_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        naive = _photoinfo(uuid="N", date=datetime(2024, 6, 1))
        _FakeDB.canned = [naive]
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        out = photos.enumerate_assets(
            Path("/lib"), until=datetime(2024, 12, 1, tzinfo=timezone.utc)
        )
        assert {a.uuid for a in out} == {"N"}

    def test_mixed_aware_and_undated_assets_sort_without_crashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sort key must not mix naive datetime.min with aware dates —
        one undated asset alongside aware ones would otherwise TypeError."""
        aware = _photoinfo(uuid="A", date=datetime(2024, 6, 1, tzinfo=timezone.utc))
        undated = _photoinfo(uuid="U", date=None)
        _FakeDB.canned = [aware, undated]
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        out = photos.enumerate_assets(Path("/lib"))  # no bound → undated kept
        assert {a.uuid for a in out} == {"A", "U"}

    def test_no_date_bound_keeps_undated_assets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _FakeDB.canned = [_photoinfo(uuid="U", date=None)]
        monkeypatch.setattr(
            photos, "osxphotos", types.SimpleNamespace(PhotosDB=_FakeDB)
        )
        out = photos.enumerate_assets(Path("/lib"))
        assert {a.uuid for a in out} == {"U"}


class TestMaterialize:
    """The materialize() contract changed in osxphotos 0.69+: the function
    now returns a (path, status) tuple so callers can distinguish "no local
    original" from "download attempted but failed"."""

    def test_on_disk_short_circuits_with_no_download_attempt(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "x.MOV"
        f.write_bytes(b"x")
        asset = _make_asset(path_original=f, in_icloud=False, raw=object())
        path, status = photos.materialize(asset, tmp_path / "scratch")
        assert path == f
        assert status == "on_disk"

    def test_missing_without_download_returns_missing(self, tmp_path: Path) -> None:
        # iCloud-only asset and the caller doesn't want to pay the download
        # cost — caller should see status='missing' and a clear None path.
        asset = _make_asset(path_original=None, in_icloud=True, raw=object())
        path, status = photos.materialize(
            asset, tmp_path / "scratch", allow_download=False
        )
        assert path is None
        assert status == "missing"

    def test_missing_with_download_but_no_raw_handle(self, tmp_path: Path) -> None:
        # Defensive: caller asks to download but the asset has no PhotoInfo
        # handle (e.g. it was reconstructed from a sidecar). Should report
        # 'failed:' rather than silently returning None like the old API.
        asset = _make_asset(path_original=None, in_icloud=True, raw=None)
        path, status = photos.materialize(
            asset, tmp_path / "scratch", allow_download=True
        )
        assert path is None
        assert status.startswith("failed:")


def test_sidecar_suffix_constant_matches_pipeline() -> None:
    """Photos sidecars must use the same suffix as the shared pipeline so the
    existing fdx-query / fdx-master / fdx-summary tools can pick them up."""
    from framedex import pipeline

    assert pipeline.SIDECAR_SUFFIX == photos.SIDECAR_SUFFIX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
