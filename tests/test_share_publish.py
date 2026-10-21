"""Tests for framedex.share_publish — config, SigV4 signing, the upload plan,
and the publish/unpublish flow against the local `dir` target.

The signer is checked against AWS's published S3 examples; the S3 uploader
runs against a fake session. Nothing touches the network or the clipboard.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
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


def test_ssh_target_from_a_url() -> None:
    cfg = pub.parse_config(
        {
            "target": "ssh://deploy@lab:/srv/fdx-galleries/",
            "base_url": "https://fdx.art/",
            "prefix": "g",
        },
        CREDS,
    )
    assert (cfg.target, cfg.host, str(cfg.path)) == (
        "ssh",
        "deploy@lab",
        "/srv/fdx-galleries",
    )
    assert (cfg.base_url, cfg.prefix) == ("https://fdx.art", "g")
    assert pub.parse_ssh_target("ssh://lab/srv/g") == ("lab", PurePosixPath("/srv/g"))


@pytest.mark.parametrize(
    "url",
    [
        "ssh://deploy@lab:srv/relative",
        "ssh://deploy@lab:/srv/../etc",
        "ssh://deploy@lab:/",
        "ssh://de ploy@lab:/srv/g",
        "ssh://lab:/srv/g;rm -rf /",
        "ssh://$(id)@lab:/srv/g",
    ],
)
def test_ssh_targets_that_could_escape_or_inject_are_refused(url: str) -> None:
    with pytest.raises(pub.ConfigError):
        pub.parse_ssh_target(url)


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
        ({"target": "r2", "bucket": "g", "account_id": "a"}, "base_url"),
        ({"target": "r2", "base_url": "https://p.ex", "account_id": "a"}, "bucket"),
        ({"target": "r2", "base_url": "https://p.ex", "bucket": "g"}, "account_id"),
        (
            {
                "target": "r2",
                "base_url": "https://p.ex",
                "bucket": "g",
                "account_id": "YOUR_ID",
            },
            "placeholder",
        ),
        ({"base_url": "https://p.ex"}, "unknown target"),
        ({"target": "ftp", "base_url": "https://p.ex"}, "unknown target"),
        ({"target": "dir", "base_url": "https://p.ex"}, "path"),
        (
            {
                "target": "dir",
                "base_url": "https://p.ex",
                "path": "/x",
                "prefix": "../up",
            },
            "prefix",
        ),
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

    with pytest.raises(pub.ConfigError, match="placeholder"):
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


# --- the artifact ------------------------------------------------------------------


def _r2() -> pub.PublishConfig:
    return pub.PublishConfig(
        "r2", "https://fdx.art", "g", "bucket", "https://e.r2", "auto"
    )


def test_links() -> None:
    assert pub.link_url(_r2(), "mara-abc") == "https://fdx.art/g/mara-abc"
    d = pub.PublishConfig("dir", "https://fdx.art", "", path=Path("/srv"))
    assert pub.link_url(d, "mara-abc") == "https://fdx.art/mara-abc/"
    ssh = pub.PublishConfig(
        "ssh", "https://fdx.art", "g", path=Path("/srv"), host="lab"
    )
    assert pub.link_url(ssh, "mara-abc") == "https://fdx.art/g/mara-abc/"


def _site(
    tmp_path: Path, names: list[str], hidden: tuple[str, ...] = ()
) -> tuple[Site, Gallery]:
    root = tmp_path / "work" / "site"
    items = [Item(n, f"caption of {n}", hidden=n in hidden) for n in names]
    gallery = Gallery("Trip", sections=[GallerySection("Day · Place", items)])
    s = Site(root, gallery)
    for kind in ("photos", "thumbs"):
        (root / kind).mkdir(parents=True, exist_ok=True)
    for n in names:
        file = hashlib.sha1(n.encode()).hexdigest()[:16] + ".jpg"
        for kind in ("photos", "thumbs"):
            (root / kind / file).write_bytes(f"{kind}:{n}".encode())
        s.photos[n] = Rendered(n, file, 2048, 1365, 1024, 683, "#000000")
    return s, gallery


def test_artifact_is_shown_photos_then_pages(tmp_path: Path) -> None:
    site, gallery = _site(tmp_path, ["a.JPG", "b.JPG"], hidden=("b.JPG",))
    objs = pub.build_artifact(
        site,
        gallery,
        slug="trip-abc",
        page_url="https://fdx.art/g/trip-abc",
        page_object=True,
    )
    a = site.photos["a.JPG"].name
    assert [o.path for o in objs] == [f"photos/{a}", f"thumbs/{a}", "index.html", ""]
    assert site.photos["b.JPG"].name not in "".join(
        o.path for o in objs
    )  # hidden: never sent
    page = objs[-1].data.decode()
    assert f'href="trip-abc/photos/{a}"' in page  # resolves from <base>/g/
    assert f'href="photos/{a}"' in objs[2].data.decode()
    assert f"https://fdx.art/g/trip-abc/thumbs/{a}" in page  # og:image
    assert objs[0].cache_control == pub.CACHE_IMMUTABLE
    assert (
        objs[2].content_type == pub.HTML_TYPE
        and objs[2].cache_control == pub.CACHE_PAGE
    )
    folder = pub.build_artifact(site, gallery, slug="trip-abc", page_url="https://x")
    assert "" not in [o.path for o in folder]  # a web server resolves index.html


def test_plan_sync_uploads_changes_and_deletes_drops() -> None:
    a = pub.Obj("a", b"a", "t", "c")
    b = pub.Obj("b", b"b2", "t", "c")
    uploaded = {
        "k/a": a.sha256,
        "k/b": hashlib.sha256(b"b1").hexdigest(),
        "k/gone": "x",
    }
    puts, deletes = pub.plan_sync([a, b], ["k/a", "k/b"], uploaded)
    assert [k for k, _ in puts] == ["k/b"]
    assert deletes == ["k/gone"]


# --- the ssh target (rsync), with subprocess mocked ----------------------------------


def test_rsync_goes_into_one_gallery_folder_only(tmp_path: Path) -> None:
    t = pub.SshTarget("deploy@lab", PurePosixPath("/srv/fdx"))
    first, second = t.commands(tmp_path, "g/trip-abc")
    dest = "deploy@lab:/srv/fdx/g/trip-abc/"
    assert first[-1] == dest and second[-1] == dest
    assert "--exclude=/index.html" in first and "--delete" not in first  # assets first
    assert "--delete" in second and not any(a.startswith("--exclude") for a in second)
    assert "--checksum" in second  # a same-size page from the same second still lands
    guard = next(a for a in first if a.startswith("--rsync-path="))
    assert guard.startswith("--rsync-path=mkdir -p /srv/fdx/g/trip-abc && ")
    assert (
        '"$(cd /srv/fdx/g/trip-abc && pwd -P)" = "$(cd /srv/fdx && pwd -P)/g/trip-abc"'
        in guard
    )
    assert (
        guard.endswith(" && rsync") and guard in second
    )  # both passes refuse a symlinked folder
    assert "ssh -o BatchMode=yes" in first
    for bad in ("../x", "g/../../etc", "g;rm", ""):
        with pytest.raises(pub.PublishError):
            t.dest(bad)


def test_rsync_runs_both_passes_and_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        calls.append(cmd)
        staged = Path(cmd[-2])
        assert (staged / "index.html").read_text() == "page"
        assert (staged / "photos" / "x.jpg").read_bytes() == b"jpg"
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("framedex.share_publish.subprocess.run", fake_run)
    objs = [
        pub.Obj("photos/x.jpg", b"jpg", "image/jpeg", "c"),
        pub.Obj("index.html", b"page", "t", "c"),
    ]
    pub.SshTarget("lab", PurePosixPath("/srv/fdx")).sync(objs, "g/trip")
    assert len(calls) == 2

    def failing(cmd: list[str], **kw: Any) -> Any:
        return subprocess.CompletedProcess(cmd, 23, "", "permission denied")

    monkeypatch.setattr("framedex.share_publish.subprocess.run", failing)
    with pytest.raises(pub.PublishError, match="permission denied"):
        pub.SshTarget("lab", PurePosixPath("/srv/fdx")).sync(objs, "g/trip")


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
        f"target: dir\npath: {tmp_path / 'www'}\nbase_url: https://fdx.art\nprefix: g\n"
    )
    return cfg


def _dir_cfg(tmp_path: Path) -> pub.PublishConfig:
    return pub.load_config(_dir_config(tmp_path))


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
        "config": cfg,
        "want_passcode": False,
        "no_passcode": False,
        "new_link": False,
        "assume_yes": True,
    }
    assert pub.publish_cli(site, gallery, work, **kw) == 0
    link = json.loads((work / pub.STATE_FILE).read_text())["links"][0]
    assert link["slug"].startswith("mara-trip-") and copied == [link["url"]]
    assert (
        link["url"] == f"https://fdx.art/g/{link['slug']}/"
        and link["protected"] is False
    )
    www = tmp_path / "www" / "g" / link["slug"]
    assert (www / "index.html").exists()
    assert len(list((www / "photos").iterdir())) == 2

    # Hide b in the editor: same link, b's files deleted, nothing else re-sent.
    gallery.sections[0].items[1].hidden = True
    assert pub.publish_cli(site, gallery, work, **kw) == 0
    again = json.loads((work / pub.STATE_FILE).read_text())["links"]
    assert len(again) == 1 and again[0]["slug"] == link["slug"]
    assert [f.name for f in (www / "photos").iterdir()] == [site.photos["a.JPG"].name]

    # A fresh link is a second link; unpublish removes both.
    assert pub.publish_cli(site, gallery, work, **{**kw, "new_link": True}) == 0
    links = json.loads((work / pub.STATE_FILE).read_text())["links"]
    assert len(links) == 2 and links[0]["slug"] != links[1]["slug"]
    assert pub.unpublish_cli(work, config=cfg, assume_yes=True) == 0
    assert not any(f.is_file() for f in (tmp_path / "www").rglob("*"))
    assert all(
        not lk["live"]
        for lk in json.loads((work / pub.STATE_FILE).read_text())["links"]
    )


def test_a_protected_gallery_never_goes_out_open_by_accident(tmp_path: Path) -> None:
    site, gallery = _site(tmp_path, ["a.JPG"])
    work, cfg = tmp_path / "work", _dir_cfg(tmp_path)
    work.mkdir(exist_ok=True)
    pub.write_state(
        work,
        {
            "links": [
                {
                    "target": cfg.target_id,
                    "slug": "t-abc",
                    "uploaded": {},
                    "live": True,
                    "protected": True,
                }
            ]
        },
    )
    with pytest.raises(pub.PublishError, match="protected"):
        pub.publish(site, gallery, work, name="t", cfg=cfg, passcode=None)
    assert not (tmp_path / "www").exists()
    done = pub.publish(
        site, gallery, work, name="t", cfg=cfg, passcode=None, open_ok=True
    )
    assert done.url.endswith("/g/t-abc/") and not done.protected  # opened on purpose
    assert not pub.is_protected(work, cfg)


def test_confirm_sees_the_summary_and_can_cancel(tmp_path: Path) -> None:
    site, gallery = _site(tmp_path, ["a.JPG"])
    seen: list[str] = []

    def no(summary: str) -> bool:
        seen.append(summary)
        return False

    with pytest.raises(pub.PublishError, match="not published"):
        pub.publish(
            site,
            gallery,
            tmp_path / "work",
            name="t",
            cfg=_dir_cfg(tmp_path),
            passcode=None,
            confirm=no,
        )
    assert "1 photos" in seen[0] and "https://fdx.art/g/t-" in seen[0]
    assert not (tmp_path / "www").exists()
    assert not (tmp_path / "work" / pub.STATE_FILE).exists()  # nothing recorded


def test_publish_without_config_prints_the_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    site, gallery = _site(tmp_path, ["a.JPG"])
    rc = pub.publish_cli(
        site,
        gallery,
        tmp_path / "work",
        name="t",
        config=str(tmp_path / "missing.yaml"),
        want_passcode=False,
        no_passcode=False,
        new_link=False,
        assume_yes=True,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "no publish config" in err and "target: ssh://" in err


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
        config=str(_dir_config(tmp_path)),
        want_passcode=False,
        no_passcode=False,
        new_link=False,
        assume_yes=False,
    )
    assert rc == 1
    assert not (tmp_path / "www").exists()


def test_dir_uploader_refuses_keys_that_escape(tmp_path: Path) -> None:
    up = pub.DirUploader(tmp_path / "www")
    with pytest.raises(ValueError):
        up.put("../outside.html", b"x", "t", "c")


def test_a_symlinked_gallery_folder_is_refused_before_rsync_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "srv"
    (root / "g").mkdir(parents=True)
    (root / "g" / "other").mkdir()
    (root / "g" / "trip").symlink_to(root / "g" / "other")

    def boom(*a: Any, **kw: Any) -> Any:
        raise AssertionError("rsync must not run")

    monkeypatch.setattr("framedex.share_publish.subprocess.run", boom)
    with pytest.raises(pub.PublishError, match="refusing"):
        pub.SshTarget(None, PurePosixPath(str(root))).sync([], "g/trip")


def test_an_unreadable_publish_record_stops_everything(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    assert pub.read_state(work) == {"links": []}  # none yet: a fresh start
    (work / pub.STATE_FILE).write_text("{not json")
    with pytest.raises(pub.PublishError, match="unreadable"):
        pub.read_state(work)
    (work / pub.STATE_FILE).write_text('{"links": "nope"}')
    with pytest.raises(pub.PublishError, match="not a publish record"):
        pub.read_state(work)


def test_protection_is_recorded_before_anything_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cryptography")
    pytest.importorskip("PIL.Image")
    import io

    from PIL import Image

    site, gallery = _site(tmp_path, ["a.JPG"])
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, "JPEG")
    for kind in ("photos", "thumbs"):
        (site.root / kind / site.photos["a.JPG"].name).write_bytes(buf.getvalue())
    work, cfg = tmp_path / "work", _dir_cfg(tmp_path)

    def broken(
        self: Any, key: str, data: bytes, content_type: str, cache_control: str
    ) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pub.DirUploader, "put", broken)
    with pytest.raises(pub.PublishError, match="disk full"):
        pub.publish(site, gallery, work, name="t", cfg=cfg, passcode="leopard2025")
    assert pub.is_protected(work, cfg)  # a retry without the passcode is refused
    with pytest.raises(pub.PublishError, match="protected"):
        pub.publish(site, gallery, work, name="t", cfg=cfg, passcode=None)
