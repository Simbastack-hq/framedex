# Design: Still-photo indexing in `fdx`

- **Date:** 2026-06-08
- **Status:** Approved (design), pending implementation plan
- **Author:** NJ (with Claude)
- **Repo:** Simbastack-hq/framedex

## Summary

Extend framedex to index **still photographs** the same way it already indexes
video: one exhaustive vision pass per file, GPS + reverse-geocoded place, face
detection, written to a non-destructive plain-text `.description.md` sidecar that
the existing `fdx-query` / `fdx-master` / `fdx-summary` tools already read.

The photo pipeline is the video pipeline with the **audio/motion half removed**
and a **camera/EXIF half added**. Nothing about the search model changes:
keywords + reverse-geocoded place + prose scene description remain the
queryable substrate, so "that sunset photo in Mara" and "giraffes drinking
water" resolve exactly as the equivalent video queries do today.

This is the planned direction the project already advertises ("video — and
eventually photo — archive").

## Goals

- Point `fdx` at a folder tree of stills (RAW + JPEG/HEIC) on an SSD and get a
  searchable index, with the same defaults-and-`--max-files`-first ergonomics as
  video.
- A photographer's drive of **mixed photos + clips** becomes **one queryable
  corpus** under one command.
- Keep the door open for adoption: "I just want to index my photos" must not
  require installing the ~2 GB whisperx/torch video stack.

## Non-goals (YAGNI for v1)

- Semantic / embedding vector search. The plain-text keyword + place + prose
  model is what video uses and what NJ asked to mirror. Revisit only if it
  proves insufficient at real library size.
- Full RAW sensor decoding / color-managed rendering. We use the embedded
  preview (see below).
- Indexing stills inside an **Apple Photos library** (`fdx-photos --images`).
  Natural fast-follow once the folder path exists; out of scope here.
- A `fdx-faces` clustering tool. Unchanged from today — embeddings are captured,
  labeling is a separate future tool.

## User-facing design

### One command, mixed media

`fdx <path>` walks the tree and indexes **whatever media it finds** — videos and
stills both — routing each file to the right pipeline by extension. No new
top-level command; nothing new for users to learn. `fdx-photos` (Apple Photos
*video* indexer) is untouched.

```bash
fdx /Volumes/SSD-2024 --max-files 5      # test first — works across photos + clips
fdx /Volumes/SSD-2024                     # full drive, both media types
fdx /Volumes/SSD-2024 --media images      # stills only
fdx /Volumes/SSD-2024 --media videos      # clips only (today's behaviour)
```

- New flag `--media images|videos|all` (default `all`).
- Existing flags (`--max-files`, `--force`, `--backend`, `--vision-model`,
  `--local-model`, `--no-faces`, `--no-geocode`) apply unchanged.
- Supported still extensions: `.jpg/.jpeg .png .tif/.tiff .heic .webp` and RAW
  `.cr2/.cr3 .nef .arw .raf .rw2 .orf .dng`. (Exact set finalized in the plan.)

### RAW = embedded preview

Modern RAW formats embed a full- or near-full-resolution JPEG preview. We extract
that preview (via `exiftool -b -JpgFromRaw`/`-PreviewImage`), downscale to 1920px
for the vision call, and run face detection on it. No `rawpy`/libraw dependency.
If a RAW has no usable embedded preview, the file is **skipped and reported** at
the end of the run (same contract as iCloud-only assets in `fdx-photos`).

### Install extras (adoption)

`pyproject.toml` gains media-scoped extras so the install matches the use case:

| Extra | Pulls | For |
|---|---|---|
| `framedex[images]` | `Pillow`, `pillow-heif` | photographers — light, no ML video stack |
| `framedex[video]` | `whisperx`, `torch`, … (today's runtime deps) | video indexing |
| `framedex[photos]` | `osxphotos` (existing) | Apple Photos library |
| `framedex[all]` | everything | |

`exiftool` and `insightface` are shared by both media types and stay in the base
install. Running `fdx` against a video without `[video]` installed produces a
clear, actionable error (not a stack trace) — and **only when an actual video
file is reached**, thanks to lazy imports (below).

## Architecture

### Module layout

```
src/framedex/
  pipeline.py      # NEW — shared core: GPS+geocode, vision backends (cli/api/local),
                   #       parse_vision_response, face_db glue, write_sidecar,
                   #       ProcessOptions / ProcessContext / ProcessResult, Nominatim
  index_videos.py  # KEEP NAME (it is the `fdx` entry point) — video-only steps
                   #       (frame extraction, audio, whisperx) + the `fdx` CLI/dispatcher
  images.py        # NEW — still-only steps: render_preview(), EXIF→metadata,
                   #       process_one_image()
  photos.py / photos_indexer.py   # unchanged (fdx-photos)
  query.py / master_index.py / trip_summary.py   # unchanged behaviour; see Search
```

- The genuinely-shared helpers — already standalone functions today — move from
  `index_videos.py` into `pipeline.py`. `index_videos.py` and `images.py` both
  import from `pipeline.py`. This avoids `images.py` importing from a module
  named "index_videos", and gives contributors an obvious home for shared logic
  as the project grows.
- **No rename** of `index_videos.py` (it backs the `fdx = framedex.index_videos:main`
  entry point — renaming churns wiring for zero user benefit). **No rewrite** of
  the video internals; they move modules at most, behaviour byte-identical.
- The `fdx` `main()` becomes a thin dispatcher: enumerate files, filter by
  `--media`, route each to `process_one_video` or `process_one_image`, share one
  `ProcessContext` (geocoder, face DB, vision client) across both.

### Lazy video imports

`whisperx`/`torch` move from module-level imports in `index_videos.py` into the
video code path (function-level), so importing the `fdx` entry point — and an
`fdx --media images` run — never touches them. This is the high-value, low-risk
core of "invest now": it enables the light `[images]` install and fixes the
brittleness that required the `importorskip` guard in `test_photos.py`.

### Photo pipeline (`process_one_image`)

| # | Step | Reuse |
|---|---|---|
| 1 | `exiftool` → EXIF: camera/lens/focal/aperture/shutter/ISO/dimensions/orientation/creation_time | new projection, exiftool already a dep |
| 2 | `exiftool` → GPS lat/lon/altitude | shared (`pipeline.get_gps`) |
| 3 | Nominatim → reverse-geocoded place | shared (`NominatimRateLimiter`) |
| 4 | `render_preview()` → 1920px JPEG (Pillow; RAW via embedded preview; HEIC via pillow-heif) | new |
| 5 | `insightface` → faces + 512-dim embeddings on the preview | shared (`face_db`) |
| 6 | vision model → structured YAML + prose (photo-tuned prompt) | shared backends; new prompt |
| 7 | write `.description.md` sidecar + face rows into `~/.framedex/faces.db` | shared writer, see note |

No audio, no transcript, no frame sampling. Lighter and faster than video.

**Sidecar writer refactor (required, in scope):** today's `write_sidecar`
hardcodes video-only frontmatter (`speaker_count`, `language_detected`,
`duration_seconds`, `technical.motion_blur`/`stability`). Split it so each
pipeline **assembles its own ordered frontmatter dict** and a shared
`pipeline.write_sidecar(frontmatter, description, body_sections, faces, ...)`
just serializes YAML + writes the prose body + inserts face rows. The video
path's emitted bytes stay identical (regression-tested); the image path passes
its own dict. This is the one non-mechanical change to existing code.

## Sidecar schema (stills)

The video frontmatter minus the audio/motion fields, plus a `camera:` block.
Same `.description.md` suffix, same `## Description` prose body.

```yaml
file: DSC_4827.RAF
path: 2024-mara/DSC_4827.RAF          # relative to scan root, like video
parent_folder: 2024-mara
media_type: image                     # NEW discriminator: image | video
size_bytes: 52428800
creation_time: 2024-08-14T07:23:11
dimensions: 7728x5152
megapixels: 40.2
camera:
  make: Fujifilm
  model: X-T5
  lens: "XF16-80mmF4 R OIS WR"
  focal_length_mm: 80
  aperture: 4.0
  shutter: "1/1000"
  iso: 400
  orientation: horizontal
location:
  lat: -1.4061
  lon: 35.0117
  altitude_m: 1580
  place: "Maasai Mara National Reserve, Narok County, Kenya"
rating: keep                          # keep | review | cull
cull_reason: ""
technical:
  focus: sharp                        # sharp | acceptable | soft
  exposure: strong                    # strong | adequate | poor | clipped
  composition: strong                 # strong | acceptable | weak  (replaces stability/motion_blur)
lighting: golden_hour
time_of_day: golden_hour
dominant_color_palette: "warm dusk: amber, ochre, dusty olive"
dominant_colors: [amber, ochre, olive]
scene_type: wildlife                  # wildlife | landscape | portrait | street | architecture | ...
people_count: 0
keywords: [giraffe, waterhole, drinking, savanna, golden-hour, wide-shot]
faces:
  - cluster_id: tmp_a3f78c
    bbox: [120, 80, 180, 240]
    detection_quality: high
face_count: 0
indexed_at: 2026-06-08T14:32:01
```

**Dropped vs video:** `duration_seconds`, `resolution`→`dimensions`, `codec`,
`language_detected`, `speaker_count`, `audio_quality`, `notable_timestamp`,
`technical.stability`, `technical.motion_blur`, and the `## Transcript` /
`## English translation` body sections.

**Added:** `media_type`, `camera:` block, `dimensions`, `megapixels`,
`scene_type`, `technical.composition`.

## Search / query

Unchanged model — this is the whole point. Photo sidecars are the same shape, so:

- `fdx-query` already filters on `rating`, `place-contains`, `keyword`,
  `time-of-day`, `lighting`, `person`, `face-count`, etc. — all present on photo
  sidecars. **Touch needed:** video-only filters (`--min/max-duration`,
  `--has-speech`, `--language`) must simply not-match photo records rather than
  error; and the result printer must tolerate missing video fields. Optionally
  add `--media`, `--camera-contains`, `--lens-contains`, `--scene-type` filters.
- `fdx-master` / `fdx-summary` roll photo sidecars into the same `_INDEX.*` /
  `_folder-summary.md`. **Touch needed:** don't assume `duration_seconds`.
- Natural-language retrieval ("giraffes drinking water") works as today: an LLM
  reads `_INDEX.md` / greps sidecars; keywords + place + prose carry the meaning.

## Compatibility

- `fdx` on a video-only tree behaves exactly as today (modulo lazy imports, which
  are transparent when `[video]` is installed).
- Existing video sidecars are untouched and still valid.
- `fdx-photos` is untouched.

## Testing

- Unit: EXIF projection (camera block), `render_preview` for JPEG/HEIC and a RAW
  fixture (embedded-preview extraction), schema/frontmatter projection, the
  `--media` router, "RAW with no preview → skipped+reported".
- CI-safety: photo tests must run under the dev+test-only install (no whisperx) —
  the lazy-import work makes this natural; stub Pillow/pillow-heif/exiftool where
  needed, mirroring how `test_photos.py` stubs osxphotos.
- Strict mypy + ruff clean (project standard).
- Manual: `fdx <folder> --media images --max-files 1` end-to-end on a real RAW;
  confirm `fdx-query --place-contains Mara --keyword giraffe` returns it.

## Open questions / future

- Whether `scene_type` should be a fixed enum or free-form (lean: short open set,
  validated softly like `lighting`).
- Fast-follow: `fdx-photos --images` (Apple Photos stills, reusing this pipeline +
  Photos-side albums/persons/keywords).
- Fast-follow: optional embeddings layer **if** keyword+place+prose underperforms.
