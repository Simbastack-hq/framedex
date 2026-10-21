# fdx-share: a gallery link from an indexed shoot

`fdx-share` turns a folder of photos that framedex has indexed into one
static web page you can send to family and friends. The sidecars do the
curation; you get a local preview and a plain-text list to adjust; one
more command uploads it and puts the link on your clipboard.

```bash
uv pip install -e '.[images]'                 # Pillow + pillow-heif
fdx --media images ~/Pictures/2025-11-mara    # index first (or pass --index below)
fdx-share ~/Pictures/2025-11-mara             # pick, render, open the preview
fdx-share ~/Pictures/2025-11-mara --publish   # upload, print + copy the link
```

`exiftool` must be on PATH for RAW files without a camera JPEG next to them
(their embedded preview is used).

## What it picks

From every photo sidecar under the folder:

1. **Keep-rated only.** The effective rating: a `user_rating` you set (via
   fdx-mcp) wins over the model's. `--include-review` lets review-rated photos
   compete too.
2. **One frame per burst.** Bursts and RAW+JPEG pairs recorded by the indexer
   keep the frame you rated keep, else the indexer's primary (the sharpest).
   Folders indexed with `--no-group` are chained on the fly with the same
   rule as the indexer (same folder and camera, frames ≤ 2 s apart).
3. **No near-duplicates.** Two candidates taken within 10 minutes whose
   64-bit difference hashes differ by ≤ 10 bits are the same picture; the one
   with the better sidecar assessment stays.
4. **A cap (`--max`, default 40)** spread across the sections: every
   day/place gets at least one photo, the rest go in proportion to how much
   you shot there. Within a section the best-assessed frames win, preferring
   frames at least 90 s apart. Ranking uses the sidecar's technical block
   (focus, exposure, composition), and a photo you rated keep yourself
   outranks any model assessment.

`--from` / `--to` (YYYY-MM-DD) limit the pick to some days of a longer folder.

## How it is ordered

By capture time, in sections by **day** and **place**. A day runs 04:00 to
04:00, so the campfire after midnight stays with its evening. The place is the
reverse-geocoded name in the sidecar: a section changes when the area changes
(the third-from-last part of the Nominatim name, a town or sub-county in most
countries), so a game drive or a walk around town stays one section. Photos
without GPS stay in the section of their day; a single frame elsewhere is
folded into its day rather than becoming its own section. Headings read
`Sunday 9 November · Ewangan, Narok West`.

Each photo gets a one-line caption from the sidecar's `**Scene:**` sentence,
cut at the first clause if it runs long. `--no-captions` leaves them off.

## Preview and edit

The first run writes, under `FOLDER/_share/<name>/` (`--out` to put it
elsewhere; `_`-prefixed folders are skipped by the indexer):

```
selection.txt     the pick, in gallery order: edit this
site/             the publishable page: index.html, photos/, thumbs/
preview.html      the page with an edit banner and file names on each photo
publish.json      after --publish: the link and what was uploaded
```

`selection.txt` looks like this:

```
title: Mara Nov 2025
subtitle: 4–9 November 2025 · Narok, Kenya
cover: IMG_0175.JPG

## Sunday 9 November · Ewangan, Narok West
IMG_0293.JPG | A black-backed jackal stands in tall grass at dawn.
IMG_0739.JPG | A leopard walks along a dirt track.

# ---- Not picked (remove the leading '# ' to add one) ----
# IMG_0740.JPG | A leopard walks along a dirt track.  # near-duplicate of IMG_0739.JPG
```

Delete a line (or put `# ` in front) to drop a photo, move lines to reorder,
edit the text after ` | ` to re-caption, edit `## ` lines, `title:`,
`subtitle:` and `cover:` (`cover: none` for no opening photo). Text after two
spaces and `#` is a note and never shown. Then re-run `fdx-share FOLDER`: it
re-renders from the file, re-using web copies that have not changed.
`--reselect` picks again from the sidecars (the old file is kept as
`selection.txt.bak`).

## What leaves your machine

Only the files in `site/`:

- **Web copies, never originals.** 2048 px on the long edge (JPEG q82) plus
  a 1024 px grid copy, upright, colour profile kept. Originals are opened
  read-only and never modified, moved or uploaded.
- **GPS stripped by default.** The copies carry make, model and capture time
  and nothing else from EXIF: no GPS, no serial numbers, no maker notes, no
  XMP. `--keep-gps` keeps the GPS block. Section headings show place
  *names*; edit them in `selection.txt` if a name says too much (a rhino
  or leopard location, a home).
- **Unlisted.** The link ends in a 12-character random token (~59 bits);
  the page carries `noindex, nofollow, noimageindex, noarchive` and
  `referrer: no-referrer`, makes no third-party requests (no web fonts, no
  analytics), and links only to framedex in the footer. Anyone who has the
  link can open it; there is no password.

## Publish

`--publish` re-renders, shows what will be uploaded and the link, and asks
before sending anything (`--yes` skips the question; without a terminal it
refuses). The link is `<base_url>/<prefix>/<name>-<random>`, recorded in
`publish.json`: re-publishing after an edit keeps the same link, uploads only
new or changed files, and deletes the files of photos you dropped.
`--new-link` publishes under a fresh link; `--unpublish` deletes everything
this gallery published to the configured target and breaks its links.

The config lives at `~/.config/framedex/share.yaml` (or `$FDX_SHARE_CONFIG`,
or `--config`). Credentials never go in it.

### Cloudflare R2

1. Create a bucket (Cloudflare dashboard, R2). Keep it private to listing:
   the default.
2. Make it publicly readable through a custom domain on your zone (R2 →
   bucket → Settings → Custom Domains) or the `r2.dev` subdomain (rate
   limited; fine for testing). That URL is `base_url`.
3. Create an R2 API token with Object Read & Write on that bucket; it shows
   an Access Key ID and a Secret Access Key.
4. Write the config and export the keys:

```yaml
target: r2
account_id: 0123456789abcdef0123456789abcdef   # dashboard, R2 overview
bucket: galleries
prefix: g
base_url: https://photos.example.com
```

```bash
export FDX_SHARE_ACCESS_KEY_ID=...        # or AWS_ACCESS_KEY_ID
export FDX_SHARE_SECRET_ACCESS_KEY=...    # or credentials_file: in the config
```

R2 does not turn `/<key>/` into `/<key>/index.html`, so the page is also
stored as the extensionless object `<prefix>/<slug>` and that is the link
you get; `<prefix>/<slug>/index.html` is uploaded too for hosts that do
resolve it. Photos are uploaded with `Cache-Control: immutable` (their names
change when their content does); the page with `no-cache`.

### Any S3-compatible store

`target: s3` with `endpoint:` (the S3 API URL, path-style) and `region:`
instead of `account_id` works the same way (AWS S3, Backblaze B2, MinIO).
Requests are signed with AWS Signature V4 by framedex itself (no boto3).

### A local folder

`target: dir` with `path:` copies the site into `path/<prefix>/<slug>/`
(a static-site repo, a synced folder) and prints `<base_url>/<prefix>/<slug>/`.

## Known limitations (v0)

- Photos only; video clips in the folder are ignored.
- One gallery per work folder; `--name` gives a folder a second one.
- Captions come from the indexing model's scene sentence; they are plain
  but not checked by a person until you read `selection.txt`.
- There is no password or expiry: the link is the only key. `--unpublish`
  is the way to take a gallery down.
- Hosting is your own bucket. framedex does not run a shared host.
