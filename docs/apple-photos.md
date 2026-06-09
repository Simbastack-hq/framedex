# Apple Photos library (macOS)

[← Back to the framedex README](../README.md)

`fdx-photos` indexes media, videos **and** stills, that live inside an Apple Photos
library directly: no export step, no metadata loss. It reads `Photos.sqlite` via
[osxphotos](https://github.com/RhetTbull/osxphotos), uses the original bytes Photos
manages, routes each asset to the same video or still pipeline `fdx` uses, and writes
sidecars to an external mirror tree (never inside the `.photoslibrary` bundle).
`--media images|videos|all` scopes a run (default `all`). macOS only.

**The common case is dead simple.** If your Photos library is local on disk (you don't
use iCloud Photos at all, or you use it but keep originals on this Mac with no Optimize
Storage), `fdx-photos` is just:

```bash
uv pip install -e '.[all]'       # one-time: osxphotos + video + image readers
fdx-photos                       # indexes the whole library (videos + stills)
```

`[photos]` is the Apple Photos *source* adapter (osxphotos) only; per-media processing
is composable, so install just what you index:

```bash
uv pip install -e '.[photos,images]'   # stills only, no torch/whisper
fdx-photos --media images

uv pip install -e '.[photos,video]'    # clips only
fdx-photos --media videos
```

`fdx-photos` preflights the queued media and prints an actionable install hint if an
extra is missing, so a wrong combo fails fast with the exact command to run. No iCloud
round-trip for any of this; the iCloud-Optimized variant is a separate edge case at the
bottom of this page.

macOS privacy note: the terminal app that runs `fdx-photos` may need **Full Disk
Access** to read `Photos.sqlite` inside the Photos library bundle. TCC often does not
show a permission prompt for this; without it, `osxphotos` can fail with `Operation not
permitted`. Grant access in **System Settings → Privacy & Security → Full Disk Access**
for Terminal, iTerm, VS Code, or whichever parent app launches the command.

## Why this instead of `osxphotos export` + `fdx`?

You *could* export your videos out of Photos and run regular `fdx` on the exported
directory. Reasons not to:

- The Photos UI's "Export edited" / "Export unmodified original" can transcode, strip
  container metadata, write `.AAE` sidecars, and split Live Photos into `.MOV` +
  `.HEIC` pairs.
- Even a clean `osxphotos export` leaves behind the Photos-side metadata that doesn't
  live in the file container: album membership, named-person labels from Photos' face
  recognition, user-added keywords, the canonical creation date Photos may have
  corrected.
- An export doubles the disk space: you keep one copy in `.photoslibrary` and another
  in the export directory.

`fdx-photos` reads the unedited original bytes (video or still) and threads Photos-side
metadata (albums, persons, keywords, canonical date, GPS) straight into the sidecar
frontmatter alongside the standard vision/audio/EXIF passes. If Photos has edits on the
asset, the sidecar records `photos_edited: true`; the indexed pixels still come from the
original. Stills additionally get the full camera/EXIF block (`fdx --media images`
schema) on top of the Photos-side fields.

## Setup

```bash
uv pip install -e '.[all]'             # everything
# or scope it: [photos,images] for stills, [photos,video] for clips
```

## Usage

```bash
# Default: ~/Pictures/Photos Library.photoslibrary → ~/framedex-photos/, all media
fdx-photos

# Scope by media type (an images-only run never loads the whisper stack)
fdx-photos --media images
fdx-photos --media videos

# Filter by album / person / date (all repeatable, OR-combined within a flag)
fdx-photos --album "Yosemite 2024" --max-files 5
fdx-photos --person "Mom" --since 2024-01-01 --until 2024-12-31
fdx-photos --keyword sunset --keyword drone

# Custom output tree (still must be outside the .photoslibrary)
fdx-photos --output ~/Documents/photos-kb

# Re-process a single problem asset by UUID
fdx-photos --uuid ABCD1234-EF56-7890-ABCD-1234567890AB --force

# Materialize iCloud-only originals on demand (only needed if Optimize Storage is on)
fdx-photos --download
```

A bare `fdx-photos` on a large library is a big job (stills usually outnumber videos
10-100x). It runs incrementally and is fully resumable (sidecars appear as it goes, and
you can Ctrl-C and re-run to continue), so above ~1000 unfiltered assets it prints a
loud heads-up rather than blocking. Narrow with `--album`/`--person`/`--since` or test
with `--max-files N` first.

## Sidecar layout

Sidecars mirror the library by date:

```text
~/framedex-photos/
├── 2024-08/
│   ├── IMG_4827__a1b2c3d4.MOV.description.md     # video
│   └── IMG_4830__b2c3d4e5.HEIC.description.md    # still
├── 2024-09/
│   └── IMG_4912__c9d0e1f2.MOV.description.md
└── _undated/
    └── clip__99887766.mov.description.md
```

The `__{uuid8}` suffix disambiguates iPhone counter-rolled filenames (two `IMG_0001.MOV`
from different camera sessions). The Photos UUID is also stored in the frontmatter as
`photos_uuid:` for lookup.

The mirror tree is a normal directory, so `fdx-query`, `fdx-summary`, and `fdx-master`
all work against it:

```bash
fdx-query ~/framedex-photos --rating keep --person Mom
fdx-summary ~/framedex-photos
fdx-master  ~/framedex-photos
```

## Frontmatter additions

Photos-side fields layered into each sidecar:

```yaml
file: IMG_4827.MOV                # original camera filename, not the library's UUID
original_filename: IMG_4827.MOV
photos_uuid: ABCD1234-EF56-7890-ABCD-1234567890AB
photos_persons:
  - Mom
  - Dad
photos_albums:
  - Yosemite 2024
photos_keywords:
  - sunset
  - drone
photos_edited: true     # only present when Photos has edits on this clip
```

`file:` is overridden to the user-meaningful camera filename (the on-disk name inside
`.photoslibrary` is a UUID like `B627DB90-...mp4`). `photos_uuid:` is the stable Photos
lookup key. `path:` is present only when the original is a durable local file; assets
materialized through `--download` are processed from a temporary copy, so their sidecars
omit `path` instead of persisting a dead temp location.

## Diagnose library state

Before launching the full indexer, check how many videos are already on local disk vs
iCloud-only:

```bash
.venv/bin/python scripts/diagnose_photos.py
```

Output tells you total / on-disk / iCloud-only / edited counts, plus a sample of the
first 10 and a recommendation for which flag (or Photos setting) you actually need.

## Notes for iCloud users (edge case)

Skip this section unless `diagnose_photos.py` reports a non-zero `iCloud-only` count.

If "Optimize Mac Storage" is on, some originals live only in iCloud and aren't on local
disk. Two paths to handle them:

- **Simplest, turn off Optimize Mac Storage**: Photos → Settings → iCloud → "Download
  Originals to this Mac". Photos downloads everything in the background; once done,
  `fdx-photos` runs with no special flags.
- **Per-clip `--download`**: materializes missing originals via PhotoKit, with an
  AppleScript-via-`PhotoScript` fallback when PhotoKit auth isn't granted. Slow and
  network-heavy. The materialized copy lives in a tempdir for the duration of one clip
  and is deleted after the sidecar is written, so downloaded-asset sidecars omit `path`
  and rely on `photos_uuid`.

PhotoKit requires the calling terminal app to have "Photos" access in **System Settings
→ Privacy & Security → Photos**. Reading `Photos.sqlite` can separately require **Full
Disk Access**. Unsigned Python (the standard `.venv` install) often can't trigger the OS
permission prompt, so the terminal may not appear in that panel until you explicitly add
it. Disabling Optimize Mac Storage sidesteps the PhotoKit download path, but Full Disk
Access may still be needed for the database read.

## Spot-checking metadata preservation

If you want to confirm GPS and dates survive, compare what we put in the sidecar against
the bits we processed:

```bash
exiftool ~/Pictures/Photos\ Library.photoslibrary/originals/A/IMG_4827.MOV | grep -i gps
exiftool ~/Pictures/Photos\ Library.photoslibrary/originals/A/IMG_4827.MOV | grep -i date
```

Those values should match the `location:` and `creation_time:` blocks in the
corresponding `~/framedex-photos/.../IMG_4827__*.description.md`. When Photos disagrees
with the file's container metadata (common on edited clips), `fdx-photos` trusts the
Photos database: that's the value the Photos UI shows, which is what a human would
expect.

---

Part of [framedex](../README.md), an open-source project from
**[SimbaStack](https://simbastack.com/)**.
