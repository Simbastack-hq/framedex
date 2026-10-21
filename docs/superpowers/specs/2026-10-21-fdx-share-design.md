# fdx share: design (v0)

Date: 2026-10-21. Status: in progress on branch `fdx-share-v0`.

## Problem

A photographer comes back from a weekend with 900 frames and wants to send
family a link to the best 40. Gallery products (Pixieset, SmugMug) solve
hosting; the hard part is curation, and framedex's sidecars already hold most
of it: keep/review/cull, burst groups, capture time, reverse-geocoded place,
a one-line scene description. `fdx share` turns that into one command:
pick, order, preview and adjust in the browser, publish, get a link.

Not a hosted product. v0 hosts the owner's own galleries on the owner's own
server or bucket. Later the same renderer makes same-night branded safari
galleries for lodge guests.

## Shape

```
fdx share <folder>        pick -> render -> local editor in the browser -> Publish -> link
fdx share <folder> --publish [--yes]     headless publish of the current selection
fdx trips <root>          list trips in an archive with a draft per trip
```

`fdx-share` / `fdx-trips` are the entry points; `fdx share` / `fdx trips`
dispatch to them from `fdx` when the first argument is `share` / `trips` and
no directory of that name exists (so `fdx share` never shadows a folder).

State is plain text in `<folder>/_share/<name>/` (`_`-prefixed: the indexer
skips it): `selection.txt` (the pick, editable by hand or by the browser
editor), `site/` (the rendered gallery), `publish.json` (link + what was
uploaded). Originals are only ever read.

## Selection (pure, tested)

1. Effective rating `keep` (a `user_rating` wins); `--include-review` widens.
2. One frame per burst / RAW+JPEG group: a human-kept member, else the
   indexer's primary, else best score. Ungrouped folders are chained with
   `grouping.detect_bursts` on sidecar time + camera.
3. Near-duplicates: same shape (aspect within 10%), taken within 10 min,
   16x16 grayscale thumbnails correlate >= 0.6 and differ <= 24 levels on
   average. (A 64-bit dHash was tried first: on flat savanna/sky, repeats and
   unrelated frames both came out 20-35 bits apart. The correlation rule
   separated 13/13 hand-labelled real pairs.)
4. Cap (`--max`, default 40) spread over sections by largest remainder, at
   least one per section; within a section best score first, preferring
   frames >= 90 s apart.
5. Story pass (new): (a) the opener is the strongest wide frame and is shown
   once, as the hero, out of sequence; (b) no three consecutive picks share a
   subject keyword (first non-generic keyword): the middle one is swapped for
   the nearest unpicked frame of a different subject in the same section, or
   dropped; (c) the closer: if one of the last three picks of the final
   section outscores the last by >= 2, it moves to the end.

Score = sidecar technical block (focus, exposure, composition) + 10 for a
human keep.

## Ordering

Capture time, sectioned by trip day (04:00-04:00) and place area (the
third-from-last Nominatim component; admin units like "Siana ward" never lead
a label). GPS-less photos stay in their day's section; a one-photo run folds
into its day. Title defaults to `<place> · <Month YYYY>` where place is the
most specific of region/country shared by >= 80% of geotagged picks
("Narok · November 2025", "Portugal · August 2024"), else the folder name.

## The page

The photos are the product; the UI nearly disappears.
- System font stack (-apple-system / SF / system-ui), no web fonts, no
  external requests of any kind.
- White / near-black following the system theme (judge warm-neutral on
  screen).
- Big quiet title, one small line (dates), then the hero (never cropped),
  then sections with small quiet headings.
- Rhythm from aspect ratio: wide frames hang full width (<= 94vw, <= 88vh
  tall), consecutive portraits pair side by side, a lone portrait hangs
  centred at <= 88vh. 390 px: edge-to-edge single column.
- Placeholder in each photo's average colour, lazy loading, soft fade-in.
- Captions off on the page by default (alt text from the sidecar caption);
  `--captions` shows them, small and quiet.
- Lightbox: full-bleed, no chrome; arrows, swipe, Esc, tap to close.
- `noindex,nofollow,noimageindex,noarchive`, `referrer: no-referrer`, tiny
  "Made with framedex" footer. No counters, no share buttons.

## Passcode (`--passcode`, or the toggle in the editor)

Static encryption at build time; the host only ever holds ciphertext, so it
needs no server logic, CDN caching is safe, and a gallery can move hosts.
- Key: PBKDF2-HMAC-SHA256, 600,000 iterations, 16-byte random salt per
  publish -> AES-256-GCM key.
- Every photo and thumb: `FDX1` magic + 12-byte random nonce + ciphertext
  (GCM tag included), AAD = the file's path in the gallery, saved as
  `<name>.bin`. The gallery manifest (subtitle, sections, sizes, colours,
  alt text) is encrypted the same way and inlined in index.html.
- Plaintext before unlock, on purpose: the title and a 24 px heavily blurred
  cover (for the iCloud-invite-style unlock screen). Nothing else: no
  captions, places, dates, file names, JPEG bytes.
- Browser: WebCrypto PBKDF2 -> AES-GCM; images decrypted lazily into blob:
  URLs as they near the viewport; wrong passcode = GCM tag failure = a calm
  "That's not it" with the field kept.
- Python: `cryptography` (AESGCM), in a new `[share]` extra with Pillow;
  imported only when a passcode is used.
- Honest limit, in the docs: a weak passcode can be brute-forced offline
  from the published files. Fine for family albums, not for anything
  sensitive. The passcode is never written to disk.

## The editor (local, one screen)

`fdx share <folder>` starts a server on 127.0.0.1 (random port) and opens
the gallery in edit mode: tap a photo to hide/restore it, drag (or the
small arrows) to reorder, click the title to rename, a passcode toggle, one
Publish button that ends on a success sheet with the link and Copy.
- Edits POST to the server, which rewrites `selection.txt` (hidden photos
  move to the "Not picked" pool) and returns the re-rendered page body; the
  layout logic stays in Python.
- Security: binds 127.0.0.1 only; every request must carry the per-run
  random token (query on the page, `X-FDX-Token` header on POSTs, so a
  cross-origin page cannot forge a POST without a preflight we refuse);
  `Host` must be the loopback address (DNS rebinding).
- Ctrl-C stops it. Terminal output: a few lines (picked, rendered, preview
  URL, published link); `-v` for detail.

## Publish targets

Config `~/.config/framedex/share.yaml` (or `$FDX_SHARE_CONFIG`/`--config`);
`base_url` e.g. `https://fdx.art` (planned domain, not live).
- `ssh` (NJ's pick: the Hetzner box): `target: ssh://user@host:/srv/fdx-galleries`.
  The gallery is staged locally as `<prefix>/<slug>/...` and sent with
  `rsync -rt --delete` filtered to that one gallery's directory, so a
  publish can only ever change its own folder. Caddy serves it as static
  files (snippet + deploy note in docs). Link: `<base_url>/<prefix>/<slug>/`.
- `r2` / `s3`: SigV4 on `requests` (already written, kept as the second
  target; verified against AWS's published signature examples and a local
  fake S3 that re-derives the signature from the wire).
- `dir`: copy into a local folder (tests, a static-site repo).
Link = `<name>-<12 random chars from a 31-symbol alphabet>` (~59 bits), kept
across re-publishes; `--new-link` rotates; `--unpublish` deletes.

## Testing

Pure logic (selection, story pass, ordering, title, selection.txt, hang,
crypto round trip + wrong passcode + "nothing readable in the output", rsync
command construction, SigV4 vectors, config validation, editor request
validation) in the existing style: no network, no ffmpeg, subprocess mocked.
End to end by hand on copies of real folders (Mara safari, Good/, Portugal,
moto-spain): editor in the browser pane at 1440 and 390, light and dark,
with and without passcode; rsync to a local directory; S3 to a local fake.

## 99% vs 1%

Real: one folder per trip, 20-400 indexed stills, mostly phone portraits or
camera landscapes, shared once with a few people, maybe re-published after
a tweak. Deferred: video in galleries, multiple galleries per folder beyond
`--name`, passcode rotation without re-upload, per-photo captions in the
editor, expiry, analytics, custom themes, a hosted multi-tenant service.

## Open questions for NJ

- Domain: fdx.art (being bought) vs a subdomain; and lab vs prod Hetzner box.
- Unlisted-only by default, passcode opt-in: confirm that default.
