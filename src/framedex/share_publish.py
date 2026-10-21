"""
framedex.share_publish — upload a rendered fdx-share gallery.

Targets (set in ~/.config/framedex/share.yaml, see docs/share.md):

    r2   Cloudflare R2 over its S3 API (the default, and the first supported)
    s3   any S3-compatible endpoint (AWS S3, Backblaze B2, MinIO, ...)
    dir  copy into a local folder (a static-site repo, a synced folder, a test)

S3/R2 requests are signed with AWS Signature V4 by the ~80 lines below on
`requests` (already a framedex dependency) instead of pulling boto3.

Each gallery gets one link, `<base_url>/<prefix>/<trip-slug>-<random>`. The
page is stored twice: as the extensionless object at that key (R2 and S3 do
not resolve index.html) and as `<key>/index.html` for hosts that do. The
photos sit under `<key>/photos/` and `<key>/thumbs/`. What was uploaded is
recorded in `<work>/publish.json`, so a re-publish after an edit uploads only
what changed, deletes what was dropped, and keeps the same link.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import yaml

from framedex.share import Gallery, link_slug
from framedex.share_site import Site, page_html

CONFIG_ENV = "FDX_SHARE_CONFIG"
DEFAULT_CONFIG = Path("~/.config/framedex/share.yaml")
STATE_FILE = "publish.json"
ACCESS_KEY_ENVS = ("FDX_SHARE_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID")
SECRET_KEY_ENVS = ("FDX_SHARE_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY")
HTML_TYPE = "text/html; charset=utf-8"
# Web copies are content-addressed: cache forever. The page revalidates so an
# edit shows up on the next load.
CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
CACHE_PAGE = "no-cache"
UPLOAD_TIMEOUT_SEC = 120
UPLOAD_ATTEMPTS = 3
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

CONFIG_TEMPLATE = """\
# fdx-share publish target. Credentials never go in this file: set
# FDX_SHARE_ACCESS_KEY_ID / FDX_SHARE_SECRET_ACCESS_KEY (or AWS_*), or point
# credentials_file at a chmod-600 YAML with access_key_id / secret_access_key.
target: r2                       # r2 | s3 | dir
account_id: YOUR_CLOUDFLARE_ACCOUNT_ID   # r2: endpoint is https://<id>.r2.cloudflarestorage.com
bucket: galleries
prefix: g                        # key prefix inside the bucket ("" for none)
base_url: https://photos.example.com    # public URL of the bucket root
# endpoint: https://s3.eu-central-1.amazonaws.com   # s3 targets: the S3 API endpoint
# region: auto                   # r2: auto; s3: the bucket's region
# credentials_file: ~/.config/framedex/share-credentials.yaml
# path: ~/sites/photos/public    # dir target: copy here instead of uploading
"""


class ConfigError(ValueError):
    pass


@dataclass
class PublishConfig:
    target: str
    base_url: str
    prefix: str = ""
    bucket: str = ""
    endpoint: str = ""
    region: str = "auto"
    path: Path | None = None
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)

    @property
    def target_id(self) -> str:
        """Identity of where links live; a change starts a fresh link."""
        if self.target == "dir":
            return f"dir:{self.path}"
        return f"{self.target}:{self.endpoint}/{self.bucket}"


def config_path(explicit: str | None) -> Path:
    raw = explicit or os.environ.get(CONFIG_ENV) or str(DEFAULT_CONFIG)
    return Path(raw).expanduser()


def _read_credentials(cfg: dict[str, Any], env: dict[str, str]) -> tuple[str, str]:
    access = next((env[k] for k in ACCESS_KEY_ENVS if env.get(k)), "")
    secret = next((env[k] for k in SECRET_KEY_ENVS if env.get(k)), "")
    if access and secret:
        return access, secret
    cred_file = cfg.get("credentials_file")
    if cred_file:
        p = Path(str(cred_file)).expanduser()
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            raise ConfigError(f"cannot read credentials_file {p}: {e}") from e
        if isinstance(data, dict):
            access = str(data.get("access_key_id") or "")
            secret = str(data.get("secret_access_key") or "")
            if access and secret:
                return access, secret
        raise ConfigError(f"{p} needs access_key_id and secret_access_key")
    raise ConfigError(
        "no credentials: set FDX_SHARE_ACCESS_KEY_ID and FDX_SHARE_SECRET_ACCESS_KEY "
        "(an R2 API token's S3 keys), or credentials_file in the config"
    )


def _is_local_http(url: str) -> bool:
    """http://localhost / 127.0.0.1 / [::1]: a MinIO or test server on this
    machine, where keys and photos never cross a network."""
    parts = urlsplit(url)
    return parts.scheme == "http" and parts.hostname in (
        "localhost",
        "127.0.0.1",
        "::1",
    )


def parse_config(cfg: Any, env: dict[str, str] | None = None) -> PublishConfig:
    """Validate a loaded config mapping. Pure apart from reading a
    credentials_file when the environment has no keys."""
    env = dict(os.environ) if env is None else env
    if not isinstance(cfg, dict):
        raise ConfigError("the config must be a YAML mapping")
    target = str(cfg.get("target") or "r2")
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url.startswith(("https://", "http://")):
        raise ConfigError(
            "base_url must be the public http(s) URL your bucket is served from"
        )
    prefix = str(cfg.get("prefix") or "").strip("/")
    if target == "dir":
        if not cfg.get("path"):
            raise ConfigError("a dir target needs path:")
        return PublishConfig(
            "dir", base_url, prefix, path=Path(str(cfg["path"])).expanduser()
        )
    if target not in ("r2", "s3"):
        raise ConfigError(f"unknown target {target!r} (r2, s3, or dir)")
    bucket = str(cfg.get("bucket") or "")
    if not bucket:
        raise ConfigError("bucket: is required")
    endpoint = str(cfg.get("endpoint") or "").rstrip("/")
    if not endpoint:
        account = str(cfg.get("account_id") or "")
        if target != "r2" or not account or account.startswith("YOUR_"):
            raise ConfigError(
                "set account_id (r2) or endpoint (the S3 API URL) in the config"
            )
        endpoint = f"https://{account}.r2.cloudflarestorage.com"
    if not endpoint.startswith("https://") and not _is_local_http(endpoint):
        raise ConfigError("endpoint must be https:// (plain http only for localhost)")
    region = str(cfg.get("region") or ("auto" if target == "r2" else "us-east-1"))
    access, secret = _read_credentials(cfg, env)
    return PublishConfig(
        target,
        base_url,
        prefix,
        bucket,
        endpoint,
        region,
        access_key=access,
        secret_key=secret,
    )


def load_config(path: Path) -> PublishConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"no publish config at {path}") from e
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    return parse_config(raw)


# ---------------------------------------------------------------------------
# AWS Signature Version 4 (S3 flavour: single-chunk payload, path-style URLs)
# ---------------------------------------------------------------------------


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def canonical_uri(path: str) -> str:
    """S3 canonical URI: every byte but unreserved characters and '/'
    percent-encoded, once."""
    return quote(path or "/", safe="/~")


def sigv4_headers(
    method: str,
    url: str,
    *,
    region: str,
    access_key: str,
    secret_key: str,
    payload_sha256: str,
    amz_date: str,
    extra_headers: dict[str, str] | None = None,
    service: str = "s3",
) -> dict[str, str]:
    """Headers that sign one request: x-amz-date, x-amz-content-sha256,
    Authorization (plus `extra_headers`, which are signed too). `url` is the
    already-encoded request URL without a query string; `amz_date` is
    'YYYYMMDDTHHMMSSZ'."""
    parts = urlsplit(url)
    headers = {
        "host": parts.netloc,
        "x-amz-content-sha256": payload_sha256,
        "x-amz-date": amz_date,
    }
    for k, v in (extra_headers or {}).items():
        headers[k.lower()] = " ".join(v.strip().split())
    names = sorted(headers)
    canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in names)
    signed = ";".join(names)
    canonical_request = "\n".join(
        [
            method,
            parts.path or "/",
            parts.query,
            canonical_headers,
            signed,
            payload_sha256,
        ]
    )
    scope = f"{amz_date[:8]}/{region}/{service}/aws4_request"
    to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    key = _hmac(("AWS4" + secret_key).encode("utf-8"), amz_date[:8])
    for part in (region, service, "aws4_request"):
        key = _hmac(key, part)
    signature = hmac.new(key, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    out = {k: v for k, v in headers.items() if k != "host"}
    out["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed}, Signature={signature}"
    )
    return out


# ---------------------------------------------------------------------------
# Uploaders
# ---------------------------------------------------------------------------


class Uploader(Protocol):
    def put(
        self, key: str, data: bytes, content_type: str, cache_control: str
    ) -> None: ...

    def delete(self, key: str) -> None: ...


class S3Uploader:
    def __init__(self, cfg: PublishConfig) -> None:
        import requests

        self.cfg = cfg
        self.session = requests.Session()

    def _url(self, key: str) -> str:
        return f"{self.cfg.endpoint}{canonical_uri(f'/{self.cfg.bucket}/{key}')}"

    def _send(
        self, method: str, key: str, data: bytes, headers: dict[str, str]
    ) -> None:
        import requests

        url = self._url(key)
        last = ""
        for attempt in range(UPLOAD_ATTEMPTS):
            amz_date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            signed = sigv4_headers(
                method,
                url,
                region=self.cfg.region,
                access_key=self.cfg.access_key,
                secret_key=self.cfg.secret_key,
                payload_sha256=hashlib.sha256(data).hexdigest()
                if data
                else EMPTY_SHA256,
                amz_date=amz_date,
            )
            try:
                resp = self.session.request(
                    method,
                    url,
                    data=data or None,
                    headers={**headers, **signed},
                    timeout=UPLOAD_TIMEOUT_SEC,
                )
            except requests.RequestException as e:
                last = str(e)
            else:
                if resp.status_code < 300 or (
                    method == "DELETE" and resp.status_code == 404
                ):
                    return
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code < 500 and resp.status_code != 429:
                    break  # auth/config errors do not get better on retry
            time.sleep(2**attempt)
        raise RuntimeError(f"{method} {key} failed: {last}")

    def put(self, key: str, data: bytes, content_type: str, cache_control: str) -> None:
        self._send(
            "PUT",
            key,
            data,
            {"Content-Type": content_type, "Cache-Control": cache_control},
        )

    def delete(self, key: str) -> None:
        self._send("DELETE", key, b"", {})


class DirUploader:
    """`dir` target: keys become files under `root` (written atomically)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"key escapes the target folder: {key}")
        return p

    def put(self, key: str, data: bytes, content_type: str, cache_control: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


def make_uploader(cfg: PublishConfig) -> Uploader:
    if cfg.target == "dir":
        assert cfg.path is not None
        return DirUploader(cfg.path)
    return S3Uploader(cfg)


# ---------------------------------------------------------------------------
# What to upload
# ---------------------------------------------------------------------------


@dataclass
class Obj:
    key: str
    data: bytes
    content_type: str
    cache_control: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def link_key(prefix: str, slug: str) -> str:
    return f"{prefix}/{slug}" if prefix else slug


def link_url(cfg: PublishConfig, slug: str) -> str:
    """The URL to share. A dir target's static host resolves index.html, so
    its link is the folder; S3/R2 serve the extensionless page object."""
    url = f"{cfg.base_url}/{link_key(cfg.prefix, slug)}"
    return url + "/" if cfg.target == "dir" else url


def build_objects(
    site: Site, gallery: Gallery, cfg: PublishConfig, slug: str
) -> list[Obj]:
    key = link_key(cfg.prefix, slug)
    page_url = f"{cfg.base_url}/{key}"
    objs = [
        Obj(
            f"{key}/index.html",
            page_html(gallery, site.photos, page_url=page_url).encode("utf-8"),
            HTML_TYPE,
            CACHE_PAGE,
        )
    ]
    if cfg.target != "dir":
        # The page at '<base>/<key>' resolves relative URLs against '<base>/',
        # so its assets are referenced as '<slug>/photos/...'.
        page = page_html(
            gallery,
            site.photos,
            asset_prefix=f"{PurePosixPath(key).name}/",
            page_url=page_url,
        )
        objs.append(Obj(key, page.encode("utf-8"), HTML_TYPE, CACHE_PAGE))
    for kind in ("photos", "thumbs"):
        for f in sorted((site.root / kind).glob("*.jpg")):
            if not f.name.startswith("."):
                objs.append(
                    Obj(
                        f"{key}/{kind}/{f.name}",
                        f.read_bytes(),
                        "image/jpeg",
                        CACHE_IMMUTABLE,
                    )
                )
    return objs


def plan_sync(
    objects: list[Obj], uploaded: dict[str, str]
) -> tuple[list[Obj], list[str]]:
    """(objects to upload: new or changed, keys to delete: uploaded before
    but no longer part of the gallery)."""
    wanted = {o.key for o in objects}
    puts = [o for o in objects if uploaded.get(o.key) != o.sha256]
    deletes = sorted(k for k in uploaded if k not in wanted)
    return puts, deletes


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def read_state(work: Path) -> dict[str, Any]:
    try:
        data = json.loads((work / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"links": []}
    if not isinstance(data, dict) or not isinstance(data.get("links"), list):
        return {"links": []}
    return data


def write_state(work: Path, state: dict[str, Any]) -> None:
    tmp = work / f".{STATE_FILE}.tmp"
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(work / STATE_FILE)


def current_link(state: dict[str, Any], target_id: str) -> dict[str, Any] | None:
    for link in reversed(state["links"]):
        if (
            isinstance(link, dict)
            and link.get("target") == target_id
            and link.get("live", True)
        ):
            return link
    return None


def copy_to_clipboard(text: str) -> bool:
    for cmd in (
        ["pbcopy"],
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["clip"],
    ):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text, text=True, check=True, timeout=5)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(
            "refusing to publish without a terminal to confirm; pass --yes",
            file=sys.stderr,
        )
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _config_or_help(config: str | None) -> PublishConfig | None:
    path = config_path(config)
    try:
        return load_config(path)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        if not path.exists():
            print(
                f"\nCreate {path} (see docs/share.md). A starting point:\n\n{CONFIG_TEMPLATE}",
                file=sys.stderr,
            )
        return None


def publish_cli(
    site: Site,
    gallery: Gallery,
    work: Path,
    *,
    name: str,
    config_path: str | None,
    new_link: bool,
    assume_yes: bool,
) -> int:
    cfg = _config_or_help(config_path)
    if cfg is None:
        return 2
    state = read_state(work)
    link = None if new_link else current_link(state, cfg.target_id)
    if link is None:
        link = {
            "target": cfg.target_id,
            "slug": link_slug(name),
            "uploaded": {},
            "live": True,
        }
        state["links"].append(link)
    slug = str(link["slug"])
    url = link_url(cfg, slug)
    link["url"] = url
    uploaded: dict[str, str] = link.setdefault("uploaded", {})

    objects = build_objects(site, gallery, cfg, slug)
    puts, deletes = plan_sync(objects, uploaded)
    size = sum(len(o.data) for o in puts)
    where = (
        str(cfg.path) if cfg.target == "dir" else f"{cfg.target} bucket {cfg.bucket}"
    )
    print(
        f"\n{len(site.photos)} photos, {len(objects)} files -> {where}\n"
        f"  upload {len(puts)} new/changed ({size / 1e6:.1f} MB), delete {len(deletes)}\n"
        f"  link: {url}\n"
        "  anyone with the link can view it; it is not listed or indexed."
    )
    if not puts and not deletes:
        print("already up to date.")
    elif not _confirm("publish?", assume_yes):
        print("not published.")
        return 1
    else:
        uploader = make_uploader(cfg)
        try:
            for n, o in enumerate(puts, 1):
                print(f"  [{n}/{len(puts)}] {o.key}")
                uploader.put(o.key, o.data, o.content_type, o.cache_control)
                uploaded[o.key] = o.sha256
            for k in deletes:
                uploader.delete(k)
                uploaded.pop(k, None)
        except (RuntimeError, OSError, ValueError) as e:
            print(
                f"error: {e}\n(what did upload is recorded; re-run to finish)",
                file=sys.stderr,
            )
            return 1
        finally:
            write_state(work, state)
    write_state(work, state)
    copied = copy_to_clipboard(url)
    print(f"\n{url}" + ("\n(copied to the clipboard)" if copied else ""))
    return 0


def unpublish_cli(work: Path, *, config_path: str | None, assume_yes: bool) -> int:
    """Delete every object of every link recorded for the configured target."""
    cfg = _config_or_help(config_path)
    if cfg is None:
        return 2
    state = read_state(work)
    links = [
        lk
        for lk in state["links"]
        if isinstance(lk, dict)
        and lk.get("target") == cfg.target_id
        and lk.get("uploaded")
    ]
    keys = sorted({k for lk in links for k in lk["uploaded"]})
    if not keys:
        print("nothing published from here for this target.")
        return 0
    for lk in links:
        print(f"  {lk.get('url')}")
    if not _confirm(f"delete {len(keys)} files and break these links?", assume_yes):
        return 1
    uploader = make_uploader(cfg)
    try:
        for lk in links:
            for k in sorted(lk["uploaded"]):
                uploader.delete(k)
                del lk["uploaded"][k]
            lk["live"] = False
    finally:
        write_state(work, state)
    print("unpublished.")
    return 0
