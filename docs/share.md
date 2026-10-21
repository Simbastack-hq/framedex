# fdx share: a gallery link from an indexed shoot

`fdx share` turns a folder framedex has indexed into one quiet web page you
can send to family and friends. framedex already knows which frames are
keepers, which ones are a burst, and when and where each was taken; `fdx
share` uses that to pick and order the photos, opens the gallery in your
browser so you can adjust it, and publishes it to your own server or bucket
at an unlisted link.

```bash
uv pip install -e '.[share]'          # Pillow, pillow-heif, cryptography
fdx --media images ~/Pictures/2025-11-mara
fdx share ~/Pictures/2025-11-mara     # pick, render, open the editor
```

```
Picked     30 of 41 photos  ·  Narok · November 2025
Rendering  30/30
Editor     http://127.0.0.1:60428/s0jOVoWuvxtR6OUn3ufhwbI7/
           Ctrl-C to stop
Published  https://framedex.art/g/narok-2025-11-k3j9x2q7m4pt/
```

## The editor

The browser opens on the gallery as visitors will see it, with a thin
toolbar on top:

- **Click a photo** to view it full screen, in the same viewer visitors get
  (arrows to move on, Esc to close).
- **Hide** what you don't want: the Hide button on a photo (it shows when you
  point at it or tab to it), the Hide button in the viewer, or the H key. A
  hidden photo stays in place, dimmed; click it to look at it again, or
  **Restore** to bring it back (Undo works too). Hidden photos never leave
  your machine.
- **Drag** a photo to reorder it within its day (or focus it and press
  Alt+arrow).
- **Click the title** to rename the gallery.
- **Make cover** on any photo chooses the opening frame.
- **Passcode** protects the gallery (below).
- **Publish** uploads it and shows the link with a Copy button. Publish
  again after more edits: the link stays the same.

Every edit is saved to `selection.txt` straight away, so closing the browser
or pressing Ctrl-C loses nothing; run `fdx share` on the folder again to pick
up where you left off. The editor listens on 127.0.0.1 only, behind a random
per-run token, and serves the rendered web copies and nothing else.

## What it picks

From every photo sidecar under the folder:

1. **Keep-rated only.** The effective rating: a `user_rating` you set (via
   fdx-mcp) wins over the model's. `--include-review` lets review-rated
   photos compete too.
2. **One frame per burst.** Bursts and RAW+JPEG pairs keep the frame you
   rated keep, else the indexer's primary (the sharpest). Folders indexed
   with `--no-group` are chained on the fly by the indexer's own rule.
3. **No near-duplicates.** Two frames of the same shape taken within 10
   minutes whose 16x16 thumbnails correlate at 0.6 or more and differ by 24
   levels or less on average are the same picture; the better-assessed one
   stays.
4. **A cap (`--max`, default 40)** spread across days and places: each gets
   at least one photo, the rest go where you shot the most. Within a day the
   best-assessed frames win, preferring frames at least 90 s apart, and the
   last day always keeps its best late frame, so the gallery ends well.
   Ranking uses the sidecar's focus, exposure and composition; a photo you
   rated keep yourself outranks any model assessment.

The **cover** is the strongest frame, whatever its shape; it opens the
gallery once, uncropped. `--from` / `--to` (YYYY-MM-DD) limit the pick to
some days of a longer folder.

## How it reads

By capture time, in sections by **day** (04:00 to 04:00, so the campfire
after midnight stays with its evening) and **place** (the reverse-geocoded
area in the sidecar: a game drive or a walk around town stays one section).
The title is made from the place most photos share and the month
(`Narok · November 2025`, `Portugal · August 2024`); the line under it gives
the days. Wide frames hang large, two portraits in a row hang as a pair, and
on a phone every photo runs edge to edge. Each photo's alt text is a short
line from its sidecar (`An adult leopard crossing the road.`); `--captions`
also shows those lines under the photos.

## Passcodes

Turn on **Passcode** in the editor (or `fdx share <folder> --publish
--passcode`, which asks for it; it is never taken from the command line) and
the gallery is encrypted before it leaves your machine:

- a key from the passcode with PBKDF2-HMAC-SHA256 (600,000 rounds, a random
  salt per publish);
- every photo and the page itself sealed with AES-256-GCM, each file bound
  to its own path;
- visitors see the title over a heavily blurred cover and one field; the
  browser decrypts with WebCrypto, so the server only ever holds ciphertext
  and needs no logic of its own.

What stays visible without the passcode: the title, a 32 px blurred copy of
the cover, the link's readable part (`narok-2025-11-...`), and how many
files there are and how big. **A weak passcode can be brute-forced offline
from the published files.** That is fine for a family album and not for
anything sensitive. A server you don't control could also swap the page's
script for one that records the passcode. Turning protection on for a
gallery that was published open deletes the open copies from your server,
but cannot recall copies someone already saved.

Once a gallery is protected it stays protected: publishing again needs the
passcode, or switching the toggle off (`--no-passcode`) on purpose.

## Publishing

The config lives at `~/.config/framedex/share.yaml` (or `$FDX_SHARE_CONFIG`,
or `--config`). `framedex.art` is registered for galleries (under `/g/`;
the bare root is left for a landing page), but its DNS isn't pointed at a
server yet; use whatever host you serve from.

**Your own server (rsync over SSH, served by Caddy)**, the default:

```yaml
target: ssh://deploy@your-server:/srv/fdx-galleries
base_url: https://framedex.art
prefix: g
```

Each publish syncs into exactly one folder, `/srv/fdx-galleries/g/<name>-<random>/`:
photos first, the page last, then whatever you dropped is deleted, in that
folder only. Ports and keys go in `~/.ssh/config`. The server side (a
Caddy site block that serves static files with `noindex` headers and long
cache lifetimes) is in [hosting/](hosting/README.md).

**Cloudflare R2 or any S3-compatible bucket:**

```yaml
target: r2                       # or s3 with endpoint: and region:
account_id: 0123456789abcdef0123456789abcdef
bucket: galleries
prefix: g
base_url: https://framedex.art        # where the bucket is served
```

Keys come from `FDX_SHARE_ACCESS_KEY_ID` / `FDX_SHARE_SECRET_ACCESS_KEY` (or
`AWS_*`), or `credentials_file:` pointing at a chmod-600 YAML with
`access_key_id` / `secret_access_key`; never from this file. Requests are
signed with AWS Signature V4 by framedex itself. R2 does not resolve
`/<key>/index.html`, so the page is also stored as the object `g/<name>-<random>`
and that is the link you get.

**A local folder:** `target: dir` with `path:` copies the gallery there (a
static-site repo, a synced folder).

Without the editor:

```bash
fdx share <folder> --publish            # publish the current selection (asks first)
fdx share <folder> --publish --passcode # ... protected
fdx share <folder> --publish --new-link # under a fresh link
fdx share <folder> --unpublish          # delete it from the target (all its links)
```

## What leaves your machine

Only the published gallery:

- **Web copies, never originals:** 2048 px on the long edge plus a 1024 px
  copy, upright, colour profile kept. Originals are only ever read.
- **No GPS:** the copies keep make, model and capture time, nothing else
  from EXIF (no GPS, serial numbers, maker notes or XMP). `--keep-gps`
  keeps the GPS block. Section headings show place *names*; edit them in
  `selection.txt` if a name says too much (a rhino sighting, a home).
- **Unlisted:** the link ends in 12 random characters (about 59 bits). The
  page carries `noindex, nofollow, noimageindex, noarchive` and
  `referrer: no-referrer`, makes no third-party requests (no web fonts, no
  analytics), and links only to framedex in its footer.

## selection.txt

The editor writes it; you can too. Under `FOLDER/_share/<name>/`
(`_`-prefixed folders are skipped by the indexer):

```
title: Narok · November 2025
subtitle: 8–9 November
cover: IMG_0320.JPG
captions: off

## 9 November · Narok West
IMG_0293.JPG | An impala, tan and dark-marked.
# IMG_0302.JPG | An adult African elephant with a young calf.  # hidden

# ---- Not picked (remove the leading '# ' to add one) ----
# IMG_0520.JPG | An adult male lion.  # near-duplicate of IMG_0518.JPG
```

Delete a line to drop a photo, move lines to reorder, edit the text after
` | ` for its alt text and caption, edit `## ` headings and the header lines
(`cover: none` for no opening photo). `--reselect` picks again from the
sidecars (the old file is kept as `selection.txt.bak`).

## fdx trips

```bash
fdx trips /Volumes/SSD-2025
```

lists the trips in an archive (split where there is more than a day and a
half without a photo, or a long GPS jump), each with a draft gallery size
and the `fdx share` command that makes it.

## Known limitations (v0)

- Photos only; videos in the folder are ignored.
- One gallery per work folder; `--name` gives a folder a second one.
- The editor reorders within a day; moving a photo to another day is a
  `selection.txt` edit.
- No expiry and no per-visitor access: the link (and the passcode, if set)
  is the key. `--unpublish` takes a gallery down.
- Hosting is your own server or bucket. framedex does not run a shared host.
