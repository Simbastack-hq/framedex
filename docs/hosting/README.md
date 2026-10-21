# Hosting fdx share galleries on your own server

`fdx share` publishes static files over rsync into one folder on a server;
Caddy serves that folder. The server needs no framedex, no database and no
application code: a gallery is a directory of files, and a protected one is
a directory of ciphertext.

Nothing here runs automatically. These are the steps to do once, by hand,
on the box you choose.

**Status:** `framedex.art` is registered, but its DNS isn't pointed at any
server yet. Which server hosts it, and its A/AAAA records, are the owner's
call; nothing in this repo deploys anything or changes any server or DNS.
Galleries live under `/g/` (`https://framedex.art/g/<name>-<random>/`); the
bare root is left for a future landing page.

## 1. The folder and the account

```bash
sudo adduser --disabled-password --gecos "" deploy        # or reuse an account
sudo mkdir -p /srv/fdx-galleries
sudo chown deploy: /srv/fdx-galleries
sudo apt install rsync                                    # if it isn't there
```

On a box that runs databases, install with `NEEDRESTART_MODE=l sudo -E apt
install rsync` so needrestart does not restart services behind your back.

Add your laptop's public key to `~deploy/.ssh/authorized_keys` and check:

```bash
ssh deploy@your-server true
```

## 2. Caddy

Add [Caddyfile](Caddyfile) to the server's Caddyfile (change the domain and
the root if yours differ), then:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy
```

The block serves galleries under `/g/` only and answers everything else,
the bare root included, with a 404 until a landing page goes there. Under
`/g/` it serves static files only: pages revalidate (`no-cache`), photos
and their encrypted twins are cached for a year (`immutable`; their names
change whenever their content does), every response carries
`X-Robots-Tag: noindex, nofollow, noimageindex, noarchive` and
`Referrer-Policy: no-referrer`, there is no directory listing, dotfiles
(rsync's in-flight temp files) are never served, and errors are a bare
status code. Caddy gets the TLS certificate on its own
once DNS points at the box.

## 3. DNS

An `A` (and `AAAA`) record for the domain pointing at the server. For
`framedex.art` this hasn't been done yet (see Status above).

## 4. Your laptop

`~/.config/framedex/share.yaml`:

```yaml
target: ssh://deploy@your-server:/srv/fdx-galleries
base_url: https://framedex.art
prefix: g
```

Then `fdx share <folder>` and Publish. Check a published gallery:

```bash
curl -sI https://framedex.art/g/<name>-<random>/ | grep -iE 'x-robots|cache-control'
```

## What a publish can touch

Each publish syncs into exactly `<root>/<prefix>/<name>-<random>/`: the path
is built from the config (letters, digits, `.`, `_`, `-`, `/` only; no
`..`) and framedex's own random name, rsync's `--delete` runs against that
folder alone, and there are no include/exclude filters deciding what may be
deleted. Other galleries and anything else under the root are never
touched; this was tested with real rsync (3.5 and macOS's openrsync) against
sibling folders. `fdx share <folder> --unpublish` empties the gallery's
folder.
