# Changelog

Notable changes to framedex. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/). The project is pre-1.0, so the
public surface (CLI flags, sidecar schema) can still shift between minor versions.

## [Unreleased]

### Added

- **Still-photo indexing.** `fdx` now indexes photos (RAW / JPEG / HEIC) alongside
  video in a single pass. `--media images|videos|all` scopes a run. Photos get the
  same `.description.md` sidecars — an EXIF `camera:` block (make, lens, focal
  length, aperture, shutter, ISO), GPS + reverse-geocoded place, face detection,
  and an AI scene description with keywords and a keep/review/cull rating — and are
  queryable with `fdx-query` and rolled up by `fdx-master`. RAW is read from the
  embedded JPEG preview (no libraw); EXIF orientation is normalized.
- `fdx-query --media image|video` filter (accepts the plural `images`/`videos`
  too); `fdx-master` reports media-neutral counts.

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
