"""
framedex.share_publish — publish a rendered fdx share gallery.

Targets (~/.config/framedex/share.yaml, see docs/share.md):

    ssh://user@host:/srv/fdx-galleries   rsync over SSH into a folder a web
                                         server (Caddy) serves as static files
    r2 / s3                              Cloudflare R2 or any S3-compatible
                                         bucket, signed with SigV4 below on
                                         `requests` (no boto3)
    dir                                  a local folder (a static-site repo,
                                         a synced folder, tests)

Every publish uploads one artifact, built here from the gallery's shown
photos only (`build_artifact`): web copies, or with a passcode their sealed
versions, then the page last. The link is `<base_url>/<prefix>/<name>-<random>`,
recorded with what was uploaded in `<work>/publish.json`, so a re-publish
keeps the link, sends only what changed and deletes what was dropped; a
gallery published with a passcode stays protected until published open on
purpose.
"""

from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
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
# Photos never change under a name (rendered and sealed copies are named by
# content or at random): cache forever. Pages revalidate so edits show.
CACHE_IMMUTABLE = "public, max-age=31536000, immutable"
CACHE_PAGE = "no-cache"
UPLOAD_TIMEOUT_SEC = 120
UPLOAD_ATTEMPTS = 3
RSYNC_TIMEOUT_SEC = 600
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
# Remote paths, hosts and prefixes go into an rsync/ssh command line: only
# these characters, so nothing in a config can turn into shell syntax.
_SAFE_PATH = re.compile(r"^/[A-Za-z0-9._/-]+$")
_SAFE_HOST = re.compile(r"^[A-Za-z0-9._@-]+$")
_SAFE_PREFIX = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")

CONFIG_TEMPLATE = """\
# fdx share publish target (see docs/share.md). base_url is where the
# target is served from; fdx.art is the planned domain, not live yet.
target: ssh://deploy@YOUR-SERVER:/srv/fdx-galleries
base_url: https://fdx.art
prefix: g                     # galleries live under <base_url>/g/ ("" for none)

# Cloudflare R2 instead (credentials from FDX_SHARE_ACCESS_KEY_ID and
# FDX_SHARE_SECRET_ACCESS_KEY, or credentials_file: a chmod-600 YAML with
# access_key_id / secret_access_key; never in this file):
# target: r2
# account_id: YOUR_CLOUDFLARE_ACCOUNT_ID
# bucket: galleries
"""


class ConfigError(ValueError):
    pass


class PublishError(RuntimeError):
    pass


@dataclass
class PublishConfig:
    target: str  # "ssh" | "r2" | "s3" | "dir"
    base_url: str
    prefix: str = ""
    bucket: str = ""
    endpoint: str = ""
    region: str = "auto"
    path: Path | None = None  # dir: local folder; ssh: remote root (POSIX)
    host: str = ""  # ssh: user@host (or an ~/.ssh/config alias)
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)

    @property
    def target_id(self) -> str:
        """Identity of where links live; a change starts a fresh link."""
        if self.target == "dir":
            return f"dir:{self.path}"
        if self.target == "ssh":
            return f"ssh:{self.host}:{self.path}"
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


def parse_ssh_target(url: str) -> tuple[str, PurePosixPath]:
    """'ssh://deploy@host:/srv/fdx-galleries' (or 'ssh://host/srv/x') ->
    ('deploy@host', /srv/fdx-galleries). Ports and keys belong in
    ~/.ssh/config. Refuses anything but plain hosts and absolute paths."""
    rest = url[len("ssh://") :]
    if ":/" in rest:
        host, path = rest.split(":", 1)
    else:
        host, slash, path = rest.partition("/")
        path = slash + path
    path = path.rstrip("/")
    if not _SAFE_HOST.match(host):
        raise ConfigError(
            f"ssh target host must be user@host or an ssh alias: {host!r}"
        )
    if not _SAFE_PATH.match(path) or ".." in path.split("/") or path == "/":
        raise ConfigError(
            f"ssh target needs an absolute folder like /srv/fdx-galleries (got {path!r})"
        )
    return host, PurePosixPath(path)


def parse_config(cfg: Any, env: dict[str, str] | None = None) -> PublishConfig:
    """Validate a loaded config mapping. Pure apart from reading a
    credentials_file when the environment has no keys."""
    env = dict(os.environ) if env is None else env
    if not isinstance(cfg, dict):
        raise ConfigError("the config must be a YAML mapping")
    target = str(cfg.get("target") or "")
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url.startswith(("https://", "http://")):
        raise ConfigError(
            "base_url must be the public http(s) URL the gallery is served from"
        )
    prefix = str(cfg.get("prefix") or "").strip("/")
    if prefix and (not _SAFE_PREFIX.match(prefix) or ".." in prefix.split("/")):
        raise ConfigError(f"prefix may use letters, digits, . _ - and /: {prefix!r}")
    if "YOUR" in target or "YOUR" in str(cfg.get("account_id") or ""):
        raise ConfigError(
            "replace the YOUR-... placeholder in the config with your own value"
        )
    if target.startswith("ssh://"):
        host, path = parse_ssh_target(target)
        return PublishConfig("ssh", base_url, prefix, path=Path(str(path)), host=host)
    if target == "dir":
        if not cfg.get("path"):
            raise ConfigError("a dir target needs path:")
        return PublishConfig(
            "dir", base_url, prefix, path=Path(str(cfg["path"])).expanduser()
        )
    if target not in ("r2", "s3"):
        raise ConfigError(
            f"unknown target {target!r}: use ssh://user@host:/folder, r2, s3, or dir"
        )
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


class SshTarget:
    """rsync into exactly one gallery folder, <root>/<prefix>/<slug>/, and
    nowhere else: the destination is a validated absolute path built from
    config + our own slug, `--delete` runs against that folder alone (no
    include/exclude filters decide what may be deleted), and the folder is
    created with `mkdir -p` of that same path. Two passes so visitors never
    see a page whose photos are not there yet: everything but index.html
    (no deletes), then index.html and the deletions. With `host=None` the
    destination is a local folder (tests, a mounted share)."""

    def __init__(self, host: str | None, root: PurePosixPath) -> None:
        self.host = host
        self.root = root

    def dest(self, gallery_key: str) -> str:
        if not _SAFE_PREFIX.match(gallery_key) or ".." in gallery_key.split("/"):
            raise PublishError(f"refusing an unsafe gallery path: {gallery_key!r}")
        return str(self.root / gallery_key)

    def commands(self, stage: Path, gallery_key: str) -> list[list[str]]:
        """The two rsync calls. Over SSH the remote side first creates the
        gallery folder and refuses to start unless that folder, with every
        symlink resolved, is exactly <root>/<key> (a symlink planted on the
        server can never point `--delete` somewhere else). The paths in it
        are validated above, so nothing here is shell syntax from config.
        The final pass compares checksums, so a page re-published within
        the same second at the same size is still replaced."""
        dest = self.dest(gallery_key)
        remote = []
        if self.host:
            guard = (
                f"mkdir -p {dest} && "
                f'[ "$(cd {dest} && pwd -P)" = "$(cd {self.root} && pwd -P)/{gallery_key}" ] && '
                "rsync"
            )
            remote = ["-e", "ssh -o BatchMode=yes", f"--rsync-path={guard}"]
        target = f"{self.host}:{dest}/" if self.host else f"{dest}/"
        src = f"{stage}/"
        return [
            ["rsync", "-rt", *remote, "--exclude=/index.html", src, target],
            ["rsync", "-rt", "--checksum", "--delete", *remote, src, target],
        ]

    def _run(self, stage: Path, gallery_key: str) -> None:
        if not self.host:
            local = Path(self.dest(gallery_key))
            local.mkdir(parents=True, exist_ok=True)
            expected = Path(str(self.root)).resolve() / gallery_key
            if local.resolve() != expected:
                raise PublishError(
                    f"{local} is not {expected} once resolved; refusing to sync"
                )
        for cmd in self.commands(stage, gallery_key):
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=RSYNC_TIMEOUT_SEC
                )
            except FileNotFoundError as e:
                raise PublishError("rsync is not installed") from e
            except subprocess.TimeoutExpired as e:
                raise PublishError("rsync timed out") from e
            if result.returncode != 0:
                raise PublishError(
                    f"rsync failed (exit {result.returncode}): {result.stderr.strip()[-400:]}"
                )

    def sync(self, objects: list[Obj], gallery_key: str) -> None:
        with tempfile.TemporaryDirectory(prefix="fdx-share-") as tmp:
            stage = Path(tmp)
            for o in objects:
                f = stage / o.path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(o.data)
            self._run(stage, gallery_key)

    def remove(self, gallery_key: str) -> None:
        """Empty the gallery folder (the empty folder stays; it serves 404)."""
        with tempfile.TemporaryDirectory(prefix="fdx-share-") as tmp:
            for cmd in self.commands(Path(tmp), gallery_key)[1:]:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=RSYNC_TIMEOUT_SEC
                )
                if result.returncode != 0:
                    raise PublishError(f"rsync failed: {result.stderr.strip()[-400:]}")


def make_uploader(cfg: PublishConfig) -> Uploader:
    if cfg.target == "dir":
        assert cfg.path is not None
        return DirUploader(cfg.path)
    return S3Uploader(cfg)


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


@dataclass
class Obj:
    path: str  # relative to the gallery folder; "" = the page object at the gallery key (S3/R2)
    data: bytes
    content_type: str
    cache_control: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def link_key(prefix: str, slug: str) -> str:
    return f"{prefix}/{slug}" if prefix else slug


def link_url(cfg: PublishConfig, slug: str) -> str:
    """The URL to share. A web server (ssh, dir) resolves index.html, so the
    link is the folder; S3/R2 serve the extensionless page object."""
    url = f"{cfg.base_url}/{link_key(cfg.prefix, slug)}"
    return url if cfg.target in ("r2", "s3") else url + "/"


def build_artifact(
    site: Site,
    gallery: Gallery,
    *,
    slug: str,
    page_url: str,
    passcode: str | None = None,
    page_object: bool = False,
) -> list[Obj]:
    """Everything one publish uploads, assets first and pages last: web
    copies of the shown photos only (hidden ones never leave the machine),
    or with a passcode their sealed copies under fresh random names and the
    unlock page. `page_object` adds the extensionless page S3/R2 serve at
    the gallery key itself; it resolves assets from '<slug>/'."""
    shown = [i for i in gallery.items if i.rel in site.photos]
    objs: list[Obj] = []
    if passcode:
        from framedex.share_lock import lock_gallery, locked_html

        locked = lock_gallery(gallery, site.photos, site.root, passcode)
        objs += [
            Obj(path, data, "application/octet-stream", CACHE_IMMUTABLE)
            for path, data in locked.files.items()
        ]
        pages = {"index.html": locked_html(gallery.title, locked)}
        if page_object:
            pages[""] = locked_html(gallery.title, locked, asset_prefix=f"{slug}/")
    else:
        for item in shown:
            name = site.photos[item.rel].name
            for kind in ("photos", "thumbs"):
                data = (site.root / kind / name).read_bytes()
                objs.append(Obj(f"{kind}/{name}", data, "image/jpeg", CACHE_IMMUTABLE))
        pages = {"index.html": page_html(gallery, site.photos, page_url=page_url)}
        if page_object:
            pages[""] = page_html(
                gallery, site.photos, asset_prefix=f"{slug}/", page_url=page_url
            )
    objs += [
        Obj(p, html.encode("utf-8"), HTML_TYPE, CACHE_PAGE) for p, html in pages.items()
    ]
    return objs


def plan_sync(
    objects: list[Obj], keys: list[str], uploaded: dict[str, str]
) -> tuple[list[tuple[str, Obj]], list[str]]:
    """(key, object) pairs to upload (new or changed, in artifact order:
    assets before pages) and keys to delete afterwards (uploaded before, no
    longer in the artifact)."""
    wanted = set(keys)
    puts = [
        (k, o)
        for k, o in zip(keys, objects, strict=True)
        if uploaded.get(k) != o.sha256
    ]
    deletes = sorted(k for k in uploaded if k not in wanted)
    return puts, deletes


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def read_state(work: Path) -> dict[str, Any]:
    """publish.json, or a fresh state when there is none yet. Anything else
    unreadable is an error: guessing would forget a passcode-protected link
    (and publish it open) or forget files still online."""
    path = work / STATE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"links": []}
    except (OSError, ValueError) as e:
        raise PublishError(f"{path} is unreadable ({e}); fix or move it first") from e
    if not isinstance(data, dict) or not isinstance(data.get("links"), list):
        raise PublishError(f"{path} is not a publish record; fix or move it first")
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


def is_protected(work: Path, cfg: PublishConfig) -> bool:
    link = current_link(read_state(work), cfg.target_id)
    return bool(link and link.get("protected"))


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


# ---------------------------------------------------------------------------
# Publish / unpublish (shared by the CLI and the editor)
# ---------------------------------------------------------------------------


@dataclass
class Published:
    url: str
    protected: bool


def publish(
    site: Site,
    gallery: Gallery,
    work: Path,
    *,
    name: str,
    cfg: PublishConfig,
    passcode: str | None,
    open_ok: bool = False,
    new_link: bool = False,
    confirm: Callable[[str], bool] | None = None,
) -> Published:
    """Publish the gallery to its link (created on first publish). A link
    published with a passcode refuses to go out open unless `open_ok` (the
    explicit --no-passcode / toggle-off), so protection never lapses
    because a passcode was not typed again. `confirm` sees a one-paragraph
    summary and may cancel (PublishError)."""
    state = read_state(work)
    link = None if new_link else current_link(state, cfg.target_id)
    if link and link.get("protected") and not passcode and not open_ok:
        raise PublishError(
            "this gallery is protected with a passcode: give it again to publish, "
            "or publish it open on purpose (--no-passcode)"
        )
    fresh = link is None
    if link is None:
        link = {
            "target": cfg.target_id,
            "slug": link_slug(name),
            "uploaded": {},
            "live": True,
        }
    slug = str(link["slug"])
    url = link_url(cfg, slug)
    key = link_key(cfg.prefix, slug)
    objects = build_artifact(
        site,
        gallery,
        slug=slug,
        page_url=f"{cfg.base_url}/{key}",
        passcode=passcode,
        page_object=cfg.target in ("r2", "s3"),
    )
    keys = [f"{key}/{o.path}" if o.path else key for o in objects]
    uploaded: dict[str, str] = dict(link.get("uploaded") or {})
    puts, deletes = plan_sync(objects, keys, uploaded)
    if cfg.target == "ssh":
        puts, deletes = list(zip(keys, objects, strict=True)), []
    size = sum(len(o.data) for _, o in puts)
    if confirm is not None:
        where = cfg.target if cfg.target != "ssh" else f"{cfg.host}:{cfg.path}"
        summary = (
            f"{len(gallery.items)} photos{', passcode-protected' if passcode else ''} -> {where}\n"
            f"  send {len(puts)} files ({size / 1e6:.1f} MB), delete {len(deletes)}\n"
            f"  link: {url}\n"
            "  anyone with the link"
            + (" and the passcode" if passcode else "")
            + " can view it; it is not listed or indexed."
        )
        if not confirm(summary):
            raise PublishError("not published")
    if fresh:
        state["links"].append(link)
    link["url"] = url
    if passcode:
        # Before any upload: if this publish dies halfway, the link already
        # counts as protected and the next publish asks for the passcode.
        link["protected"] = True
    try:
        if cfg.target == "ssh":
            assert cfg.path is not None
            SshTarget(cfg.host, PurePosixPath(str(cfg.path))).sync(objects, key)
            uploaded = {k: o.sha256 for k, o in zip(keys, objects, strict=True)}
        else:
            uploader = make_uploader(cfg)
            for k, o in puts:
                uploader.put(k, o.data, o.content_type, o.cache_control)
                uploaded[k] = o.sha256
            for k in deletes:
                uploader.delete(k)
                uploaded.pop(k, None)
        link["protected"] = bool(passcode)
    except (RuntimeError, OSError, ValueError) as e:
        raise PublishError(
            f"{e} (what did upload is recorded; publish again to finish)"
        ) from e
    finally:
        link["uploaded"] = uploaded
        write_state(work, state)
    return Published(url, bool(passcode))


def unpublish(work: Path, cfg: PublishConfig) -> list[str]:
    """Delete every file of every link published from this gallery to the
    configured target; returns the links that stopped working."""
    state = read_state(work)
    links = [
        lk
        for lk in state["links"]
        if isinstance(lk, dict)
        and lk.get("target") == cfg.target_id
        and lk.get("live", True)
    ]
    gone: list[str] = []
    try:
        for lk in links:
            key = link_key(cfg.prefix, str(lk["slug"]))
            if cfg.target == "ssh":
                assert cfg.path is not None
                SshTarget(cfg.host, PurePosixPath(str(cfg.path))).remove(key)
                lk["uploaded"] = {}
            else:
                uploader = make_uploader(cfg)
                for k in sorted(lk.get("uploaded") or {}):
                    uploader.delete(k)
                    del lk["uploaded"][k]
            lk["live"] = False
            gone.append(str(lk.get("url")))
    finally:
        write_state(work, state)
    return gone


def _confirm_tty(summary: str) -> bool:
    print(summary)
    if not sys.stdin.isatty():
        print(
            "refusing to publish without a terminal to confirm; pass --yes",
            file=sys.stderr,
        )
        return False
    return input("publish? [y/N] ").strip().lower() in ("y", "yes")


def load_config_or_explain(config: str | None) -> PublishConfig:
    path = config_path(config)
    try:
        return load_config(path)
    except ConfigError as e:
        hint = (
            f"\nCreate {path} (see docs/share.md). A starting point:\n\n{CONFIG_TEMPLATE}"
            if not path.exists()
            else ""
        )
        raise PublishError(f"{e}{hint}") from e


def ask_passcode() -> str:
    """Prompt twice without echo; never from argv (it would sit in shell
    history and `ps`)."""
    from framedex.share_lock import MIN_PASSCODE_LEN

    first = getpass.getpass("passcode: ")
    if len(first) < MIN_PASSCODE_LEN:
        raise PublishError(f"a passcode needs at least {MIN_PASSCODE_LEN} characters")
    if getpass.getpass("again: ") != first:
        raise PublishError("the passcodes did not match")
    return first


def publish_cli(
    site: Site,
    gallery: Gallery,
    work: Path,
    *,
    name: str,
    config: str | None,
    want_passcode: bool,
    no_passcode: bool,
    new_link: bool,
    assume_yes: bool,
) -> int:
    try:
        cfg = load_config_or_explain(config)
        passcode = ask_passcode() if want_passcode else None
        result = publish(
            site,
            gallery,
            work,
            name=name,
            cfg=cfg,
            passcode=passcode,
            open_ok=no_passcode,
            new_link=new_link,
            confirm=None if assume_yes else _confirm_tty,
        )
    except PublishError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    copied = copy_to_clipboard(result.url)
    print(f"Published  {result.url}" + ("  (copied)" if copied else ""))
    return 0


def unpublish_cli(work: Path, *, config: str | None, assume_yes: bool) -> int:
    try:
        cfg = load_config_or_explain(config)
    except PublishError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    state = read_state(work)
    live = [
        lk
        for lk in state["links"]
        if isinstance(lk, dict)
        and lk.get("target") == cfg.target_id
        and lk.get("live", True)
    ]
    if not live:
        print("nothing published from here to this target.")
        return 0
    summary = "will stop working:\n" + "\n".join(f"  {lk.get('url')}" for lk in live)
    if not assume_yes and not _confirm_tty(summary):
        return 1
    try:
        unpublish(work, cfg)
    except (PublishError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("Unpublished.")
    return 0
