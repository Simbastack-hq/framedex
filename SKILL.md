---
name: framedex
description: "Build a portable knowledge base of your video (and eventually photo) archive across multiple SSDs or directly from an Apple Photos library. For each clip: GPS + reverse-geocoded place, speaker-diarized multi-lingual transcript with English translation, face detection + embeddings for later named-person queries, Claude/Gemma structured assessment (keep/review/cull rating + technical quality + lighting + time of day + dominant colors + audio quality + people count + keywords + notable timestamps), and prose scene description. Writes plain-text sidecars next to originals (or to a mirror tree for Photos libraries — never inside the .photoslibrary bundle) + persistent face DB. Photos-library mode also threads through albums, named persons, and keywords from the Photos database. Non-destructive, idempotent, resumable. Use whenever you want to: index videos, tag footage, organize a drive, build the video knowledge base, transcribe audio, describe clips, rate clips, find clips by location/lighting/person/keyword, generate folder summaries, identify duplicates or cull pile, index the Apple Photos library directly. Trigger phrases: 'index this drive', 'tag my videos', 'index my Photos library', 'index Apple Photos videos', 'what's on this SSD', 'rate these clips', 'find me clips of X', 'what should I cull', 'build the video knowledge base'."
---

# framedex — Video Archive Knowledge Base

Cross-project, cross-drive. An entire video archive turned into a portable plain-text knowledge base + queryable face DB.

## Per-clip pipeline

1. `ffprobe` — metadata
2. `exiftool` — GPS lat/lon/altitude (iPhone, DJI, drone all supported)
3. Nominatim — reverse-geocoded place name (rate-limited 1/sec, free, no key)
4. `ffmpeg` — 5 representative JPEG frames @ 1920px max
5. `ffmpeg` — audio extraction → WhisperX transcribe + diarization + alignment
6. WhisperX translate-mode — English translation for non-English clips
7. `insightface` (RetinaFace + ArcFace) — face detection + 512-dim embeddings on the same frames
8. Vision model (Claude Haiku/Sonnet via Max CLI / API, OR local Gemma via LM Studio) → structured YAML + prose description in one call
9. Write `[filename].description.md` sidecar + insert face rows into `~/.framedex/faces.db`

## Output schema

Each sidecar's YAML frontmatter:

```yaml
file: IMG_4827.mov
path: /Volumes/SSD-2024/...
parent_folder: drone
duration_seconds: 12.3
resolution: 3840x2160
codec: hvc1
size_bytes: 245678912
creation_time: 2024-08-14T07:23:11Z
location:
  lat: 37.7456
  lon: -119.5936
  altitude_m: 1842.5
  place: "Yosemite Valley, Mariposa County, USA"
language_detected: es
speaker_count: 2
rating: keep                  # keep | review | cull
cull_reason: ""
technical:
  focus: sharp                # sharp | acceptable | soft
  exposure: strong            # strong | adequate | poor | clipped
  stability: smooth           # smooth | handheld | jittery
  motion_blur: clean          # clean | some | heavy
lighting: golden_hour
time_of_day: golden_hour
dominant_color_palette: "warm dusk: amber, ochre, dusty olive"
dominant_colors: [amber, ochre, olive, sky-blue]
audio_quality: clean_speech
people_count: 3               # vision model's estimate
keywords: [drone, landscape, construction, golden-hour, wide-shot, speech, workers]
notable_timestamp: ""         # MM:SS of peak moment if clip ≥ 30s
faces:                        # from insightface, separate from people_count
  - cluster_id: tmp_a3f78c    # temporary until fdx-faces labels it 'alex' / 'sam' / etc
    frame_time: 1.2
    bbox: [120, 80, 180, 240]
    detection_quality: high
face_count: 2
indexed_at: 2026-05-17T14:32:01
```

When indexed via `fdx-photos`, the sidecar also carries Photos-side fields:

```yaml
original_filename: IMG_4827.MOV     # user-meaningful camera filename
photos_uuid: ABCD1234-EF56-7890-ABCD-1234567890AB
photos_persons: [Mom, Dad]          # from Photos' face recognition labels
photos_albums: [Yosemite 2024]      # Photos album membership
photos_keywords: [sunset]           # tags added in the Photos UI
live_photo: true                    # only when true
photos_edited: true                 # only when Photos has edits on the clip
```

Body follows: `## Description` (Scene/Subjects/Action/Mood/Shot type/Use cases prose), `## Transcript` (with speaker labels if diarized), `## English translation` (if applicable).

## Three vision backends

| Backend | Quality | Speed | Cost | Privacy |
|---|---|---|---|---|
| `cli` (default) | Claude Haiku/Sonnet via Max | ~10-30s per clip | $0 (Max subscription) | Cloud (frames sent to Anthropic) |
| `api` | Claude Haiku/Sonnet via API | ~2-3s per clip | ~$0.002 (Haiku) / ~$0.008 (Sonnet) per clip | Cloud (frames sent to Anthropic) |
| `local` | Local model via LM Studio (Gemma 4, Qwen2-VL, etc.) | ~3-90s depending on model | $0 | Fully local |

`--vision-model haiku|sonnet` picks Claude model for `cli`/`api`. `--local-model NAME` picks LM Studio model. The script auto-strips `ANTHROPIC_API_KEY` from `claude -p` subprocess env so CLI mode hits Max OAuth even if API key is set globally.

## Face detection

Always on by default. `~/.framedex/faces.db` is the single shared face database across all drives. Per-clip embeddings stored as 512 float32 vectors + bbox + detection score. Temporary cluster IDs (`tmp_<hash>`) get replaced with real names by the (not-yet-built) `fdx-faces` clustering tool — that tool will be a follow-up that doesn't require re-running the indexing pass, because all embeddings are captured here.

Skip with `--no-faces` if you don't want face data.

## Companion scripts / aliases

| Alias | Script | Purpose |
|---|---|---|
| `fdx` | `framedex.index_videos` | Main indexer for folder trees (drives, project dirs) |
| `fdx-photos` | `framedex.photos_indexer` | Index videos directly from an Apple Photos library (macOS, requires `[photos]` extra) |
| `fdx-summary` | `framedex.trip_summary` | Recursive folder summaries (`_folder-summary.md` in each ≥5-clip folder) |
| `fdx-master` | `framedex.master_index` | Drive-level `_INDEX.md` + `_INDEX.json` |
| `fdx-query` | `framedex.query` | Filter sidecars by metadata (rating, lighting, person, keyword, etc.) |

## Set up once

```bash
cd ~/.claude/skills/framedex

# Install Python deps (editable — changes take effect immediately)
uv pip install -e .

# For the Apple Photos integration (macOS only) — adds osxphotos:
uv pip install -e '.[photos]'

# Verify system binaries + pre-download models
python3 scripts/setup.py

# HF token for pyannote diarization (one-time)
# Accept terms on https://huggingface.co/pyannote/speaker-diarization-3.1
# and https://huggingface.co/pyannote/segmentation-3.0 first
export HF_TOKEN=hf_...

# Only if using --backend api:
# export ANTHROPIC_API_KEY=sk-ant-...

# Commands are now on PATH after editable install:
#   fdx, fdx-photos, fdx-summary, fdx-master, fdx-query
```

## Common run patterns

```bash
# Test 5 clips first — always
fdx /Volumes/SSD-2024 --max-files 5

# Full drive on default (Max CLI + Haiku)
fdx /Volumes/SSD-2024

# Higher accuracy via Max — slower, $0
fdx /Volumes/SSD-2024 --vision-model sonnet

# Local Gemma — fully offline
fdx /Volumes/SSD-2024 --backend local

# Skip movies (default cuts at 30 min)
fdx /Volumes/SSD-2024 --max-duration 30

# Re-process everything with new model
fdx /Volumes/SSD-2024 --force --vision-model sonnet

# After indexing: per-folder summaries
fdx-summary /Volumes/SSD-2024

# Drive overview
fdx-master /Volumes/SSD-2024

# Query examples
fdx-query /Volumes/SSD-2024 --rating keep --time-of-day golden_hour
fdx-query /Volumes/SSD-2024 --rating cull              # cull pile
fdx-query /Volumes/SSD-2024 --place-contains California --language es
fdx-query /Volumes/SSD-2024 --keyword drone --keyword landscape
fdx-query /Volumes/SSD-2024 --stability smooth --people-count 0
fdx-query /Volumes/SSD-2024 --rating keep --json | jq '.[] | .path'
```

### Apple Photos library

```bash
# First, check library state (how many local vs iCloud-only)
.venv/bin/python scripts/diagnose_photos.py

# Smoke-test on 5 already-downloaded clips
fdx-photos --max-files 5

# Full library (default path: ~/Pictures/Photos Library.photoslibrary)
# Sidecars land at ~/framedex-photos/{YYYY-MM}/...
fdx-photos

# Filter by Photos-side metadata (repeatable, OR-combined within a flag)
fdx-photos --album "Yosemite 2024" --since 2024-01-01
fdx-photos --person "Mom" --keyword sunset

# Force per-clip iCloud download via PhotoKit (slow; needs Photos permission
# for the terminal in System Settings → Privacy & Security)
fdx-photos --download

# Custom mirror output (must be OUTSIDE the .photoslibrary bundle)
fdx-photos --output ~/Documents/photos-kb

# Re-process a single problem clip by UUID
fdx-photos --uuid ABCD1234-EF56-7890-ABCD-1234567890AB --force

# After indexing, fdx-query / fdx-summary / fdx-master work on the mirror
fdx-query ~/framedex-photos --rating keep --person "Mom"
fdx-master ~/framedex-photos
```

## Optional folder context

Drop `.video-context.md` at the root of any scan target with a paragraph describing what's on that drive ("construction site, 2023-2026", "family travel, 2024", etc). The vision prompt prepends it for context-aware descriptions.

## Privacy

| Component | Local or cloud? |
|---|---|
| ffmpeg / exiftool / Whisper / pyannote / insightface | Local |
| Nominatim reverse geocoding | Sends lat/lon (not video). Skip with `--no-geocode`. |
| Vision (`--backend cli`/`api`) | Frames sent to Anthropic. By default not used for training. |
| Vision (`--backend local`) | Fully local, fully offline. |
| Face DB (`~/.framedex/faces.db`) | Local only, never uploaded. Back up the file manually if you care. |

## Multiple SSDs

Run on each drive separately. Sidecars travel with the data; the face DB is centralized at `~/.framedex/faces.db` so cross-drive person queries work.

## Known limitations (v1)

- Frame sampling is evenly-spaced, not scene-detected (future: ffmpeg `select=gt(scene,0.4)`)
- pyannote diarization degrades on heavy ambient noise (wind, music, crowd)
- WhisperX runs on CPU on Apple Silicon (CTranslate2 doesn't have M-series GPU acceleration yet; 64GB CPU is still plenty)
- `fdx-faces` (clustering + labeling tool) not built yet — face embeddings are captured but cluster IDs are temporary hashes until that tool ships
- RAW image format support not yet (videos only; photos are coming)
- `fdx-photos --download` relies on PhotoKit, which requires the parent terminal to have "Photos" access in macOS Privacy & Security. Unsigned Python (the default `.venv` install) often can't trigger the OS prompt, so the terminal may never appear in that panel. The supported workaround is to turn off "Optimize Mac Storage" in Photos → Settings → iCloud, which makes all originals local and removes the need for `--download` entirely.

## File layout

```
~/.claude/skills/framedex/
├── SKILL.md                       # this file
├── README.md
├── pyproject.toml                 # deps, ruff/mypy config, entry points
├── .pre-commit-config.yaml        # pre-commit hooks
├── .github/workflows/ci.yml       # CI (ruff + mypy)
├── scripts/
│   ├── setup.py                   # system binaries + model pre-download
│   └── diagnose_photos.py         # one-shot Photos-library state report
└── src/framedex/
    ├── __init__.py                # package init, version from pyproject.toml
    ├── index_videos.py            # main worker (fdx) — also exports
    │                              # ProcessOptions/ProcessContext/process_one_video
    │                              # so fdx-photos can reuse the per-clip pipeline
    ├── photos.py                  # Apple Photos library adapter (osxphotos)
    ├── photos_indexer.py          # Photos library indexer (fdx-photos)
    ├── face_db.py                 # face detection + SQLite face DB module
    ├── trip_summary.py            # recursive folder summaries (fdx-summary)
    ├── master_index.py            # drive-level KB (fdx-master)
    └── query.py                   # filter sidecars (fdx-query)
```
