# Changelog

Notable changes to framedex. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). The project is pre-1.0, so the
public surface (CLI flags, sidecar schema) can still shift between minor versions.

## [Unreleased]

### Added

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
