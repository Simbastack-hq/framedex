"""Tests for framedex.share_publish — config, SigV4 signing, the upload plan,
and the publish/unpublish flow against the local `dir` target.

The signer is checked against AWS's published S3 examples; the S3 uploader
runs against a fake session. Nothing touches the network or the clipboard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from framedex import share_publish as pub
from framedex.share import Gallery, GallerySection, Item
from framedex.share_site import Rendered, Site

AK = "AKIAIOSFODNN7EXAMPLE"
SK = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
CREDS = {"FDX_SHARE_ACCESS_KEY_ID": "ak", "FDX_SHARE_SECRET_ACCESS_KEY": "sk"}


# --- SigV4 ---------------------------------------------------------------------


def test_sigv4_matches_the_aws_s3_get_example() -> None:
    headers = pub.sigv4_headers(
        "GET",
        "https://examplebucket.s3.amazonaws.com/test.txt",
        region="us-east-1",
        access_key=AK,
        secret_key=SK,
        payload_sha256=pub.EMPTY_SHA256,
        amz_date="20130524T000000Z",
        extra_headers={"Range": "bytes=0-9"},
    )
    assert headers["Authorization"] == (
        f"AWS4-HMAC-SHA256 Credential={AK}/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )
    assert "host" not in headers  # requests sets Host itself


def test_sigv4_matches_the_aws_s3_put_example() -> None:
    body = b"Welcome to Amazon S3."
    headers = pub.sigv4_headers(
        "PUT",
        "https://examplebucket.s3.amazonaws.com" + pub.canonical_uri("/test$file.text"),
        region="us-east-1",
        access_key=AK,
        secret_key=SK,
        payload_sha256=hashlib.sha256(body).hexdigest(),
        amz_date="20130524T000000Z",
        extra_headers={
            "Date": "Fri, 24 May 2013 00:00:00 GMT",
            "x-amz-storage-class": "REDUCED_REDUNDANCY",
        },
    )
    assert headers["Authorization"].endswith(
        "SignedHeaders=date;host;x-amz-content-sha256;x-amz-date;x-amz-storage-class, "
        "Signature=98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"
    )


def test_canonical_uri_encodes_once() -> None:
    assert (
        pub.canonical_uri("/b/g/trip-x/photos/a b+c.jpg")
        == "/b/g/trip-x/photos/a%20b%2Bc.jpg"
    )
    assert pub.canonical_uri("/b/ü~") == "/b/%C3%BC~"


# --- config --------------------------------------------------------------------


def test_r2_config_builds_the_endpoint_from_the_account() -> None:
    cfg = pub.parse_config(
        {
            "target": "r2",
            "account_id": "abc123",
            "bucket": "g",
            "base_url": "https://p.ex/",
        },
        CREDS,
    )
    assert cfg.endpoint == "https://abc123.r2.cloudflarestorage.com"
    assert (cfg.region, cfg.base_url, cfg.prefix) == ("auto", "https://p.ex", "")
    assert (cfg.access_key, cfg.secret_key) == ("ak", "sk")
    assert "sk" not in repr(cfg)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"bucket": "g", "account_id": "a"}, "base_url"),
        ({"base_url": "https://p.ex", "account_id": "a"}, "bucket"),
        ({"base_url": "https://p.ex", "bucket": "g"}, "account_id"),
        (
            {
                "base_url": "https://p.ex",
                "bucket": "g",
                "account_id": "YOUR_CLOUDFLARE_ACCOUNT_ID",
            },
            "account_id",
        ),
        ({"target": "ftp", "base_url": "https://p.ex"}, "unknown target"),
        ({"target": "dir", "base_url": "https://p.ex"}, "path"),
        (
            {
                "target": "s3",
                "base_url": "https://p.ex",
                "bucket": "g",
                "endpoint": "http://x",
            },
            "https",
        ),
        ("just a string", "mapping"),
    ],
)
def test_bad_configs_say_what_is_missing(raw: Any, message: str) -> None:
    with pytest.raises(pub.ConfigError, match=message):
        pub.parse_config(raw, CREDS)


def test_plain_http_only_for_a_local_endpoint() -> None:
    base = {"target": "s3", "bucket": "g", "base_url": "http://127.0.0.1:9000"}
    local = pub.parse_config({**base, "endpoint": "http://127.0.0.1:9000"}, CREDS)
    assert local.endpoint == "http://127.0.0.1:9000"
    assert pub.parse_config({**base, "endpoint": "http://localhost:9000"}, CREDS)
    with pytest.raises(pub.ConfigError, match="localhost"):
        pub.parse_config({**base, "endpoint": "http://minio.lan:9000"}, CREDS)


def test_the_template_is_rejected_until_filled_in() -> None:
    import yaml

    with pytest.raises(pub.ConfigError):
        pub.parse_config(yaml.safe_load(pub.CONFIG_TEMPLATE), CREDS)


def test_credentials_from_env_or_file(tmp_path: Path) -> None:
    base = {
        "target": "s3",
        "endpoint": "https://s3.x",
        "bucket": "g",
        "base_url": "https://p",
    }
    aws = {"AWS_ACCESS_KEY_ID": "a2", "AWS_SECRET_ACCESS_KEY": "s2"}
    assert pub.parse_config(base, aws).access_key == "a2"
    assert pub.parse_config(base, {**aws, **CREDS}).access_key == "ak"  # ours win
    with pytest.raises(pub.ConfigError, match="no credentials"):
        pub.parse_config(base, {})
    creds = tmp_path / "c.yaml"
    creds.write_text("access_key_id: fa\nsecret_access_key: fs\n")
    cfg = pub.parse_config({**base, "credentials_file": str(creds)}, {})
    assert (cfg.access_key, cfg.secret_key, cfg.region) == ("fa", "fs", "us-east-1")
    creds.write_text("access_key_id: only\n")
    with pytest.raises(pub.ConfigError, match="secret_access_key"):
        pub.parse_config({**base, "credentials_file": str(creds)}, {})


def test_config_path_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pub.CONFIG_ENV, raising=False)
    assert pub.config_path(None) == pub.DEFAULT_CONFIG.expanduser()
    monkeypatch.setenv(pub.CONFIG_ENV, "/etc/share.yaml")
    assert pub.config_path(None) == Path("/etc/share.yaml")
    assert pub.config_path("/x.yaml") == Path("/x.yaml")


# --- links and the upload plan -----------------------------------------------------


def _r2() -> pub.PublishConfig:
    return pub.PublishConfig(
        "r2", "https://p.ex", "g", "bucket", "https://e.r2", "auto"
    )


def test_links() -> None:
    assert pub.link_url(_r2(), "mara-abc") == "https://p.ex/g/mara-abc"
    d = pub.PublishConfig("dir", "https://p.ex", "", path=Path("/srv"))
    assert pub.link_url(d, "mara-abc") == "https://p.ex/mara-abc/"


def _site(tmp_path: Path, names: list[str]) -> tuple[Site, Gallery]:
    root = tmp_path / "work" / "site"
    gallery = Gallery(
        "Trip", sections=[GallerySection("Day", [Item(n) for n in names])]
    )
    s = Site(root, tmp_path / "work" / "preview.html", gallery)
    for kind in ("photos", "thumbs"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    for n in names:
        file = hashlib.sha1(n.encode()).hexdigest()[:16] + ".jpg"
        for kind in ("photos", "thumbs"):
            (root / kind / file).write_bytes(f"{kind}:{n}".encode())
        s.photos[n] = Rendered(n, file, 2048, 1365, 1024, 683, "#000000", 10)
    return s, gallery


def test_build_objects_for_a_bucket(tmp_path: Path) -> None:
    site, gallery = _site(tmp_path, ["a.JPG"])
    objs = {o.key: o for o in pub.build_objects(site, gallery, _r2(), "trip-abc")}
    file = site.photos["a.JPG"].name
    assert set(objs) == {
        "g/trip-abc",
        "g/trip-abc/index.html",
        f"g/trip-abc/photos/{file}",
        f"g/trip-abc/thumbs/{file}",
    }
    page = objs["g/trip-abc"].data.decode()
    assert f'href="trip-abc/photos/{file}"' in page  # resolves from <base>/g/
    index = objs["g/trip-abc/index.html"].data.decode()
    assert f'href="photos/{file}"' in index
    assert f"https://p.ex/g/trip-abc/thumbs/{file}" in page  # og:image
    assert objs["g/trip-abc"].content_type == pub.HTML_TYPE
    assert objs[f"g/trip-abc/photos/{file}"].cache_control == pub.CACHE_IMMUTABLE


def test_plan_sync_uploads_changes_and_deletes_drops() -> None:
    a = pub.Obj("k/a", b"a", "t", "c")
    b = pub.Obj("k/b", b"b2", "t", "c")
    uploaded = {
        "k/a": a.sha256,
        "k/b": hashlib.sha256(b"b1").hexdigest(),
        "k/gone": "x",
    }
    puts, deletes = pub.plan_sync([a, b], uploaded)
    assert [o.key for o in puts] == ["k/b"]
    assert deletes == ["k/gone"]


# --- the S3 uploader against a fake session ------------------------------------------


class _Resp:
    def __init__(self, status: int) -> None:
        self.status_code = status
        self.text = f"status {status}"


class _Session:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kw: Any) -> _Resp:
        self.calls.append({"method": method, "url": url, **kw})
        return _Resp(self.statuses.pop(0))


def _uploader(
    statuses: list[int], monkeypatch: pytest.MonkeyPatch
) -> tuple[pub.S3Uploader, _Session]:
    monkeypatch.setattr("framedex.share_publish.time.sleep", lambda s: None)
    cfg = pub.PublishConfig(
        "r2",
        "https://p.ex",
        "g",
        "bucket",
        "https://acct.r2.cloudflarestorage.com",
        access_key="ak",
        secret_key="sk",
    )
    up = pub.S3Uploader(cfg)
    session = _Session(statuses)
    up.session = session  # type: ignore[assignment]
    return up, session


def test_s3_put_is_path_style_and_signed(monkeypatch: pytest.MonkeyPatch) -> None:
    up, session = _uploader([200], monkeypatch)
    up.put("g/trip x/photos/a.jpg", b"jpeg", "image/jpeg", pub.CACHE_IMMUTABLE)
    call = session.calls[0]
    assert (
        call["url"]
        == "https://acct.r2.cloudflarestorage.com/bucket/g/trip%20x/photos/a.jpg"
    )
    h = call["headers"]
    assert h["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=ak/")
    assert "/auto/s3/aws4_request" in h["Authorization"]
    assert h["x-amz-content-sha256"] == hashlib.sha256(b"jpeg").hexdigest()
    assert (h["Content-Type"], h["Cache-Control"]) == (
        "image/jpeg",
        pub.CACHE_IMMUTABLE,
    )


def test_s3_retries_server_errors_but_not_auth_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    up, session = _uploader([503, 200], monkeypatch)
    up.put("k", b"x", "t", "c")
    assert len(session.calls) == 2
    up, session = _uploader([403, 200], monkeypatch)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        up.put("k", b"x", "t", "c")
    assert len(session.calls) == 1


def test_s3_delete_treats_missing_as_done(monkeypatch: pytest.MonkeyPatch) -> None:
    up, session = _uploader([404], monkeypatch)
    up.delete("k")
    assert session.calls[0]["method"] == "DELETE"
    assert session.calls[0]["headers"]["x-amz-content-sha256"] == pub.EMPTY_SHA256


# --- publish / unpublish end to end on the dir target ----------------------------------


def _dir_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "share.yaml"
    cfg.write_text(
        f"target: dir\npath: {tmp_path / 'www'}\nbase_url: https://p.ex\nprefix: g\n"
    )
    return cfg


def test_publish_then_edit_then_unpublish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied: list[str] = []

    def fake_copy(text: str) -> bool:
        copied.append(text)
        return True

    monkeypatch.setattr(pub, "copy_to_clipboard", fake_copy)
    cfg = str(_dir_config(tmp_path))
    work = tmp_path / "work"
    site, gallery = _site(tmp_path, ["a.JPG", "b.JPG"])

    kw: dict[str, Any] = {
        "name": "Mara Trip",
        "config_path": cfg,
        "new_link": False,
        "assume_yes": True,
    }
    assert pub.publish_cli(site, gallery, work, **kw) == 0
    state = json.loads((work / pub.STATE_FILE).read_text())
    link = state["links"][0]
    assert link["slug"].startswith("mara-trip-") and copied == [link["url"]]
    www = tmp_path / "www" / "g" / link["slug"]
    assert (www / "index.html").exists()
    assert len(list((www / "photos").iterdir())) == 2

    # Drop b: the same link, b's files deleted, nothing else re-sent.
    b_file = site.photos.pop("b.JPG").name
    for kind in ("photos", "thumbs"):
        (site.root / kind / b_file).unlink()
    gallery.sections[0].items.pop()
    assert pub.publish_cli(site, gallery, work, **kw) == 0
    again = json.loads((work / pub.STATE_FILE).read_text())["links"]
    assert len(again) == 1 and again[0]["slug"] == link["slug"]
    assert [f.name for f in (www / "photos").iterdir()] == [site.photos["a.JPG"].name]

    # A fresh link is a second link; unpublish removes both.
    assert pub.publish_cli(site, gallery, work, **{**kw, "new_link": True}) == 0
    links = json.loads((work / pub.STATE_FILE).read_text())["links"]
    assert len(links) == 2 and links[0]["slug"] != links[1]["slug"]
    assert pub.unpublish_cli(work, config_path=cfg, assume_yes=True) == 0
    assert not any(f.is_file() for f in (tmp_path / "www").rglob("*"))
    assert all(
        not lk["live"]
        for lk in json.loads((work / pub.STATE_FILE).read_text())["links"]
    )


def test_publish_without_config_prints_the_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site, gallery = _site(tmp_path, ["a.JPG"])
    rc = pub.publish_cli(
        site,
        gallery,
        tmp_path / "work",
        name="t",
        config_path=str(tmp_path / "missing.yaml"),
        new_link=False,
        assume_yes=True,
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "no publish config" in err and "target: r2" in err


def test_publish_refuses_without_a_terminal_or_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("framedex.share_publish.sys.stdin.isatty", lambda: False)
    site, gallery = _site(tmp_path, ["a.JPG"])
    rc = pub.publish_cli(
        site,
        gallery,
        tmp_path / "work",
        name="t",
        config_path=str(_dir_config(tmp_path)),
        new_link=False,
        assume_yes=False,
    )
    assert rc == 1
    assert not (tmp_path / "www").exists()


def test_dir_uploader_refuses_keys_that_escape(tmp_path: Path) -> None:
    up = pub.DirUploader(tmp_path / "www")
    with pytest.raises(ValueError):
        up.put("../outside.html", b"x", "t", "c")
