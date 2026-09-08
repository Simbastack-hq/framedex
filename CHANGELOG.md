# Changelog

Notable changes to framedex. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). The project is pre-1.0, so the
public surface (CLI flags, sidecar schema) can still shift between minor versions.

## [Unreleased]

### Added

- **`fdx-mcp`: the archive as MCP tools.** A new optional command (`[mcp]`
  extra: the official MCP SDK + Pillow) serves indexed roots to any MCP host
  over stdio (Claude Code, Claude Desktop, LM Studio with a local model).
  Tools: `list_roots`, `query_media` (a subset of `fdx-query`'s metadata filters plus `folder`
  and `offset`/`limit` paging, effective ratings, the scene sentence),
  `read_sidecar`, `archive_overview` (`_INDEX.md`, labelled as a snapshot),
  `contact_sheet` (1-20 files rendered into one numbered 4-column grid with a
  legend; a clip contributes one frame at its `notable_timestamp` or
  midpoint), and `set_user_rating`. Every path, including paths derived from
  sidecars, is resolved and must lie under a configured root; `--read-only`
  removes the setter. fdx-mcp makes no model call; each successful
  contact-sheet call is one bounded image, and host inference is where cost
  and data flow live. See `docs/mcp.md`.
- **`user_rating`: the person's decision next to the model's.**
  `set_user_rating` writes `user_rating` (keep/review/cull), `user_note`, and
  `user_rated_at` into the sidecar frontmatter atomically, body preserved
  byte-for-byte; the model's `rating` is never modified. A valid user rating
  wins everywhere ratings are read: `fdx-query --rating` and
  `--with-description` (`keep (user)`), JSON records gain `effective_rating`,
  `fdx-master` (counts, a "User ratings" line, `(user)` in the cull list),
  `fdx-xmp` (stars, plus the keyword `user-rated`). Re-indexing carries the
  keys over. An invalid value is reported once and ignored.
- **`fdx-query --folder SUB` and `--offset N`** (paging with `--limit`;
  `--count` now reports matches before paging). The CLI and the MCP tool
  share one `Filters`/`run_query` implementation.
- **Burst grouping + RAW/JPEG pairing: one assessment per group.** Before
  the per-file loop, `fdx` groups a folder's stills: a RAW and its same-stem
  camera JPEG become one pair (RAW primary, JPEG as preview source), and at
  least 3 frames in one folder with matching camera make/model and gaps of at
  most 2s become a burst (a burst of pairs is one burst). Each group gets a
  single vision call on its primary (the member with the highest local
  Laplacian sharpness score; no model choice); every alternate gets a stub
  sidecar that copies the primary's assessment fields (not faces), carries its
  own EXIF/GPS, says so in its body, and points at the primary via a `group:`
  block. Vision calls per run = groups + ungrouped files, never more than
  before. Resume is group-aware: stubs are written before the primary, the
  primary's old sidecar is removed first, and an incomplete or changed group is
  reprocessed together (repeating that group's call). `--no-group` (both `fdx`
  and `fdx-photos`; a documented no-op in the latter until Photos-native burst
  support) disables grouping for a run; `--force --no-group` re-indexes a
  grouped folder individually. A failed batched EXIF read stops the run with a
  clear message instead of silently indexing every file. `fdx-master` counts
  ratings, keywords, faces, and the cull list over primaries only and reports
  `Grouped: N files in M groups`; `fdx-query --primary-only` hides stubs;
  `fdx-xmp` tags burst members `burst-primary` / `burst-alternate`. Thresholds
  are constants documented in `docs/tuning.md`. No new dependency.
- **`fdx-xmp` — get framedex ratings into Lightroom.** A new standalone command
  that projects the rating, keywords, and one-line scene caption from
  `.description.md` sidecars into standard `.xmp` sidecars next to proprietary-RAW
  originals, so Lightroom Classic / Bridge pick them up via *Read Metadata from
  Files*. `keep`/`review`/`cull` → 3★/2★/1★, `cull` also gets a Red label;
  keywords + scene_type → `dc:subject`; the Scene sentence → `dc:description`.
  It's a **regenerable view** (delete every `.xmp` and re-run) and strictly
  non-destructive: it never edits an original and never overwrites a `.xmp` it
  didn't write — a foreign or hand-edited sidecar is reported as a conflict and
  skipped (ownership tracked by a content hash in `_XMP_MANIFEST.json`).
  `--dry-run` previews. Pure stdlib, no new dependency. Scope in v1 is
  proprietary RAW → Lightroom (`.dng` and JPEG/HEIC/TIFF embed metadata in-file,
  which framedex never touches).


- **Content-diverse frame sampling** (issue #5). The 5 vision frames are now the
  most mutually different, sharpest moments of a clip (H-S histogram
  farthest-point selection over a fast-seek thumbnail pool) instead of
  evenly-spaced grabs, so a clip that pans across scenes no longer sends five
  near-duplicates. Exactly 5 frames as before — vision cost is unchanged — and
  static/short clips keep the old even spacing. `--frame-sampling even`
  restores legacy behavior. No new dependencies.

- **Still-photo indexing.** `fdx` now indexes photos (RAW / JPEG / HEIC) alongside
  video in a single pass. `--media images|videos|all` scopes a run. Photos get the
  same `.description.md` sidecars — an EXIF `camera:` block (make, lens, focal
  length, aperture, shutter, ISO), GPS + reverse-geocoded place, face detection,
  and an AI scene description with keywords and a keep/review/cull rating — and are
  queryable with `fdx-query` and rolled up by `fdx-master`. RAW is read from the
  embedded JPEG preview (no libraw); EXIF orientation is normalized.
- `fdx-query --media image|video` filter (accepts the plural `images`/`videos`
  too); `fdx-master` reports media-neutral counts.

### Fixed

- **exiftool failures are errors, not blanks.** `get_image_metadata` and
  `get_gps` now raise when exiftool exits non-zero or returns no JSON, instead
  of silently producing a sidecar with an empty camera block / no location. A
  per-file run reports the file as an error and retries it next time; a
  burst/pair group fails before its vision call.
- **Atomic writes use exclusive, unique temp files.** `atomic_write_text`
  (every sidecar, index, and XMP write) now creates its same-directory temp via
  `mkstemp`: two writers can't collide on one temp name, and a symlink planted
  at a predictable temp name can no longer redirect a write into another file.
  Stale temps are `.<name>.<random>.tmp` (still hidden, still safe to delete).
- **Sidecar parsers are fence-aware.** `fdx-query`, `fdx-master`, `fdx-xmp`,
  `fdx-summary`, and the resume check now end the frontmatter at a line that
  is exactly `---`, so a filename or value containing `---` no longer
  truncates the YAML.
- **Trust hardening — the resumability/idempotency promise now holds under
  Ctrl-C, re-runs, and hostile input.**
  - Sidecars and the `_INDEX.*` / folder-summary files are written atomically
    (temp file + `os.replace`), so an interrupt or disk-full mid-write can no
    longer leave a truncated, permanently-"indexed" file.
  - Faces are committed to `faces.db` *before* the sidecar (the sidecar is the
    resume marker), and face rows are now rewritten even when a re-run detects
    zero faces — so a crash between the two writes, or a changed detection
    result, no longer strands stale rows.
  - `faces.db` is now idempotent: re-runs dedupe on `video_path` **or**
    `sidecar_path` (fixing duplicate rows for `fdx-photos --download` assets,
    whose temp path changes each run), `member_count` is recomputed instead of
    incremented (no more inflation on `--force`), and unnamed zero-member
    clusters are reaped while user-named ones are kept.
  - `fdx-photos --since/--until` no longer crashes with a timezone
    `TypeError` when the library has tz-aware dates or any undated asset.
  - The `cli` vision backend no longer runs `claude -p` with
    `bypassPermissions`. It uses `--permission-mode dontAsk` plus a read-only
    `Read` allowlist (Bash/writes denied), so untrusted transcript /
    `.video-context.md` text embedded in the prompt can't drive tool use.
  - A vision response with no parseable YAML block is retried on the next run
    (loud stderr line) instead of being silently written as a defaults-only
    sidecar (`rating: review`, all `unclear`) that skipped the file forever.
  - `fdx-query` and `fdx-master` now warn and skip a sidecar whose `path` field
    is missing, blank, or non-string instead of crashing or emitting the
    `.description.md` path as if it were the media path; Photos-managed assets
    (which omit `path` by design) are unaffected. Closes #14.
- Frame timestamps recorded for face detection (`faces.db` `frame_time`) now
  come from the actual extraction instead of being re-derived, fixing a silent
  desync when a frame write failed mid-clip.

### Changed

- **Install is now extras-scoped** so a photo-only setup never pulls torch:
  - `framedex[images]` — Pillow + pillow-heif (still photos)
  - `framedex[video]` — whisperx/torch (video indexing)
  - `framedex[photos]` — osxphotos + the video stack (Apple Photos)
  - `framedex[all]` — everything
  - **Breaking:** `pip install -e .` no longer installs the video stack. Existing
    video workflows need `'.[video]'` (or `'.[all]'`). A video run without it now
    prints a clear "install the video extra" message instead of a stack trace.
- Internal: media-agnostic logic extracted into `framedex/pipeline.py`;
  `whisperx`/`torch` are imported lazily. Video sidecar output is byte-identical.
- **Docs: README split into focused guides.** The README now covers the core
  workflow; deeper and edge-case topics moved to `docs/` and are linked near the top:
  `docs/apple-photos.md` (the Apple Photos library + iCloud "Optimize Storage" edge
  case), `docs/tuning.md` (folder context, proper-noun biasing, languages,
  diarization setup), and `docs/troubleshooting.md`. An ASCII flow diagram at
  the top of the README shows the pipeline at a glance. No behavior change.

### Fixed

- Vision-backend failures (timeout, HTTP error, permission-denied) no longer write
  a junk sidecar that would permanently skip the file — the item is reported as an
  error and retried on the next run.
- `fdx-query` duration filters (`--min/max-duration`) no longer match photos, which
  have no duration.

## [0.1.0]

### Added

- Initial release: the `fdx` video indexer, `fdx-query` / `fdx-master` /
  `fdx-summary`, a shared face DB, and `fdx-photos` (index videos directly from an
  Apple Photos library).
