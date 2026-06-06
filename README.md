# framedex

**A queryable knowledge base for your video archive.**

Turn a scattered video archive — across multiple SSDs and years — into a portable, plain-text knowledge base. Each clip gets a `.description.md` sidecar with GPS location + place name, a speaker-diarized multilingual transcript, an English translation (if needed), face detection, and an AI vision scene description with a keep/review/cull rating.

Sidecars live next to the videos. Originals are never modified. Local-first, non-destructive, resumable.

framedex is a [Claude Code](https://docs.claude.com/en/docs/claude-code) skill. It installs the `fdx` command-line tool.

## Install

```bash
# Clone into your Claude Code skills directory
git clone git@github.com:Simbastack-hq/framedex.git ~/.claude/skills/framedex
cd ~/.claude/skills/framedex

# Install Python deps (editable — changes take effect immediately)
uv pip install -e .

# Verify system binaries + pre-download models
python3 scripts/setup.py
```

## Quick start

```bash
# 1. Get a Hugging Face token + accept pyannote terms (one-time, for diarization)
#    https://huggingface.co/pyannote/speaker-diarization-3.1   (click Agree)
#    https://huggingface.co/pyannote/segmentation-3.0          (click Agree)
#    https://huggingface.co/settings/tokens                    (create read token)
export HF_TOKEN=hf_yourTokenHere

# 2. (Optional) Set an Anthropic API key — only needed for --backend api
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Commands are on PATH after editable install. Use fdx, fdx-summary, fdx-master, fdx-query.

# 4. Test on 5 clips before unleashing on a full drive
fdx /Volumes/SSD-2024 --max-files 5

# 5. Inspect the sidecars. If happy, run the full drive.
fdx /Volumes/SSD-2024

# 6. After indexing, generate folder summaries + a master index
fdx-summary /Volumes/SSD-2024
fdx-master  /Volumes/SSD-2024
```

## Per-clip pipeline

1. `ffprobe` → metadata (duration, codec, resolution, creation date)
2. `exiftool` → GPS lat/lon/altitude
3. Nominatim → reverse-geocoded place name (rate-limited 1/sec, polite UA)
4. `ffmpeg` → 5 evenly-spaced JPEG frames (≤1920px wide)
5. `ffmpeg` → mono 16k WAV
6. WhisperX → Whisper transcribe + word-level alignment + pyannote diarization
7. WhisperX translate mode → English translation (non-English only)
8. `insightface` → face detection + 512-dim embeddings on the same frames
9. Vision model → single-call structured description (Scene/Subjects/Action/Mood/Shot type/Use cases) + keep/review/cull rating
10. Write `[filename].description.md` next to the video

## What sidecars look like

```markdown
---
file: IMG_4827.mov
path: 2024-08-construction/drone/IMG_4827.mov
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
rating: keep
indexed_at: 2026-05-17T14:32:01
---

# IMG_4827.mov

## Description

**Scene:** Wide drone aerial of a construction site at golden hour...
**Subjects:** Three workers in high-vis vests near a partially-built structure...
**Action:** Drone slowly orbits; workers carry materials between two structures.
**Mood:** Industrious, expansive, hopeful.
**Shot type:** Drone aerial, slow orbit.
**Use cases:**
- Construction milestone post
- "From the ground up" origin-story reel
- B-roll behind a voiceover

## Transcript (es, 2 speakers)

[SPEAKER_00] (00:00:01) Pon esta viga aquí primero.
[SPEAKER_01] (00:00:04) Sí, vale.
[SPEAKER_00] (00:00:07) Cuidado con el ángulo.

## English translation

Place this beam here first. Yes, OK. Careful with the angle.
```

## Optional folder context

Drop `.video-context.md` at the root of any scan target to give the vision model better priors:

```
/Volumes/SSD-2024/.video-context.md
---
This drive contains construction-site footage, 2023-2026. Many clips
are drone aerials, crew training, and site walkthroughs. Languages mix
English and Spanish.
---
```

Without it, descriptions are generic.

### Proper-noun biasing

A `.video-context.md` can also carry a line of names Whisper should spell correctly:

```
**Whisper proper nouns:** Yosemite, El Capitan, Half Dome, ...
```

These get passed to Whisper as `initial_prompt` + `hotwords` so place names and people names in speech don't come back garbled. A second regex pass (`~/.framedex/whisper_fixes.json`) catches anything the prompt bias misses.

## Multiple SSDs

Run on each drive separately:

```bash
fdx /Volumes/SSD-2023
fdx /Volumes/SSD-2024
fdx /Volumes/SSD-2025
```

Each drive ends up self-contained with its own sidecars + `_INDEX.json`. Knowledge travels with the data. The face DB at `~/.framedex/faces.db` is centralized so cross-drive person queries work.

## Apple Photos Library (macOS)

`fdx-photos` indexes videos that live inside an Apple Photos library directly — no export step, no metadata loss. It reads `Photos.sqlite` via [osxphotos](https://github.com/RhetTbull/osxphotos), uses the original video bytes Photos manages, and writes sidecars to an external mirror tree (never inside the `.photoslibrary` bundle). macOS only.

**The common case is dead simple.** If your Photos library is local on disk — you don't use iCloud Photos at all, or you use it but keep originals on this Mac (no Optimize Storage) — `fdx-photos` is just:

```bash
uv pip install -e '.[photos]'   # one-time, adds osxphotos
fdx-photos                       # indexes the whole library
```

No flags, no iCloud round-trip. This is the path for everyone who keeps their photos on their own machine — no iCloud+ subscription required, no monthly storage fee, no PhotoKit download flow. The iCloud-Optimized variant is a separate edge case at the bottom of this section.

macOS privacy note: the terminal app that runs `fdx-photos` may need **Full Disk Access** to read `Photos.sqlite` inside the Photos library bundle. TCC often does not show a permission prompt for this; without it, `osxphotos` can fail with `Operation not permitted`. Grant access in **System Settings → Privacy & Security → Full Disk Access** for Terminal, iTerm, VS Code, or whichever parent app launches the command.

### Why this instead of `osxphotos export` + `fdx`?

You *could* export your videos out of Photos and run regular `fdx` on the exported directory. Reasons not to:

- The Photos UI's "Export edited" / "Export unmodified original" can transcode, strip container metadata, write `.AAE` sidecars, and split Live Photos into `.MOV` + `.HEIC` pairs.
- Even a clean `osxphotos export` leaves behind the Photos-side metadata that doesn't live in the file container: album membership, named-person labels from Photos' face recognition, user-added keywords, the canonical creation date Photos may have corrected.
- An export doubles the disk space — you keep one copy in `.photoslibrary` and another in the export directory.

`fdx-photos` reads the unedited original video bytes and threads Photos-side metadata (albums, persons, keywords, canonical date) straight into the sidecar frontmatter alongside the standard vision/audio passes. If Photos has edits on the clip, the sidecar records `photos_edited: true`; the indexed pixels/audio still come from the original asset.

### Setup

```bash
uv pip install -e '.[photos]'   # adds osxphotos
```

### Usage

```bash
# Default: ~/Pictures/Photos Library.photoslibrary → ~/framedex-photos/
fdx-photos

# Filter by album / person / date — all repeatable, OR-combined within a flag
fdx-photos --album "Yosemite 2024" --max-files 5
fdx-photos --person "Mom" --since 2024-01-01 --until 2024-12-31
fdx-photos --keyword sunset --keyword drone

# Custom output tree (still must be outside the .photoslibrary)
fdx-photos --output ~/Documents/photos-kb

# Re-process a single problem clip by UUID
fdx-photos --uuid ABCD1234-EF56-7890-ABCD-1234567890AB --force

# Materialize iCloud-only originals on demand (only needed if Optimize Storage is on)
fdx-photos --download
```

### Sidecar layout

Sidecars mirror the library by date:

```text
~/framedex-photos/
├── 2024-08/
│   ├── IMG_4827__a1b2c3d4.MOV.description.md
│   └── IMG_4831__e5f6a7b8.MOV.description.md
├── 2024-09/
│   └── IMG_4912__c9d0e1f2.MOV.description.md
└── _undated/
    └── clip__99887766.mov.description.md
```

The `__{uuid8}` suffix disambiguates iPhone counter-rolled filenames (two `IMG_0001.MOV` from different camera sessions). The Photos UUID is also stored in the frontmatter as `photos_uuid:` for lookup.

The mirror tree is a normal directory, so `fdx-query`, `fdx-summary`, and `fdx-master` all work against it:

```bash
fdx-query ~/framedex-photos --rating keep --person Mom
fdx-summary ~/framedex-photos
fdx-master  ~/framedex-photos
```

### Frontmatter additions

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

`file:` is overridden to the user-meaningful camera filename (the on-disk name inside `.photoslibrary` is a UUID like `B627DB90-...mp4`). `photos_uuid:` is the stable Photos lookup key. `path:` is present only when the original is a durable local file; assets materialized through `--download` are processed from a temporary copy, so their sidecars omit `path` instead of persisting a dead temp location.

### Diagnose library state

Before launching the full indexer, check how many videos are already on local disk vs iCloud-only:

```bash
.venv/bin/python scripts/diagnose_photos.py
```

Output tells you total / on-disk / iCloud-only / edited counts, plus a sample of the first 10 and a recommendation for which flag (or Photos setting) you actually need.

### Notes for iCloud users (edge case)

Skip this section unless `diagnose_photos.py` reports a non-zero `iCloud-only` count.

If "Optimize Mac Storage" is on, some originals live only in iCloud and aren't on local disk. Two paths to handle them:

- **Simplest — turn off Optimize Mac Storage**: Photos → Settings → iCloud → "Download Originals to this Mac". Photos downloads everything in the background; once done, `fdx-photos` runs with no special flags.
- **Per-clip `--download`**: materializes missing originals via PhotoKit, with an AppleScript-via-`PhotoScript` fallback when PhotoKit auth isn't granted. Slow and network-heavy. The materialized copy lives in a tempdir for the duration of one clip and is deleted after the sidecar is written, so downloaded-asset sidecars omit `path` and rely on `photos_uuid`.

PhotoKit requires the calling terminal app to have "Photos" access in **System Settings → Privacy & Security → Photos**. Reading `Photos.sqlite` can separately require **Full Disk Access**. Unsigned Python (the standard `.venv` install) often can't trigger the OS permission prompt, so the terminal may not appear in that panel until you explicitly add it. Disabling Optimize Mac Storage sidesteps the PhotoKit download path, but Full Disk Access may still be needed for the database read.

### Spot-checking metadata preservation

If you want to confirm GPS and dates survive, compare what we put in the sidecar against the bits we processed:

```bash
exiftool ~/Pictures/Photos\ Library.photoslibrary/originals/A/IMG_4827.MOV | grep -i gps
exiftool ~/Pictures/Photos\ Library.photoslibrary/originals/A/IMG_4827.MOV | grep -i date
```

Those values should match the `location:` and `creation_time:` blocks in the corresponding `~/framedex-photos/.../IMG_4827__*.description.md`. When Photos disagrees with the file's container metadata (common on edited clips), `fdx-photos` trusts the Photos database — that's the value the Photos UI shows, which is what a human would expect.

## Common flags

| Flag | Purpose |
|---|---|
| `--dry-run` | Show what would be processed; no API/model calls |
| `--max-files N` | Stop after N clips (testing) |
| `--force` | Re-process clips even if a sidecar exists |
| `--whisper-model large-v3` | Higher quality, slower (default is large-v3-turbo) |
| `--no-diarize` | Skip speaker diarization (faster; no HF_TOKEN needed) |
| `--no-faces` | Skip face detection + embeddings |
| `--no-geocode` | Skip Nominatim reverse geocoding (GPS still recorded) |
| `--max-duration MINUTES` | Skip clips longer than N minutes (default: 30; 0 = no limit) |
| `--exclude PATTERN` | Skip paths matching substring (repeatable) |
| `--backend cli\|api\|local` | Vision backend (see below) |
| `--vision-model haiku\|sonnet` | Claude model for `cli`/`api`. Default `haiku` |
| `--local-base-url URL` | Override LM Studio endpoint (default `http://localhost:1234/v1`) |
| `--local-model NAME` | Specify which loaded model to use when LM Studio has multiple |
| `--no-whisper-prompt` | Disable proper-noun biasing |
| `--whisper-fixes PATH` | Override the canonical-name regex fixes file |

## Vision backends

| Backend | What it uses | Speed | Cost | Privacy |
|---|---|---|---|---|
| `cli` (default) | `claude -p` via a Claude Max subscription | ~10-30s/clip | $0 marginal | Frames sent to Anthropic |
| `api` | Anthropic SDK with an API key | ~2-3s/clip | ~$0.002/clip (Haiku) | Frames sent to Anthropic |
| `local` | LM Studio (or any OpenAI-compatible server) | ~3-90s/clip | $0 | Fully local, fully offline |

For huge archives, `api` is fastest. For routine indexing on a Max plan, `cli` is free. For full privacy, `local` keeps everything on-device.

## Privacy

| Component | Local or cloud? |
|---|---|
| ffmpeg, exiftool, Whisper, pyannote, insightface | Local |
| Nominatim reverse geocode | Cloud — sends lat/lon only, never video. Skip with `--no-geocode` |
| Vision (`--backend cli`/`api`) | Cloud — sends 5 JPEG frames + a transcript snippet per clip |
| Vision (`--backend local`) | Fully local |
| Face DB (`~/.framedex/faces.db`) | Local only, never uploaded |

## Languages

Whisper supports 99 languages with auto-detection. For non-English clips the script automatically runs a second translate-mode pass and stores the English version alongside the original transcript. For best quality on important non-English footage:

```bash
fdx /Volumes/SSD-2024 --whisper-model large-v3 --force
```

## Speaker diarization

WhisperX uses `pyannote/speaker-diarization-3.1` under the hood. First-time setup requires:

1. A Hugging Face account + read token (`HF_TOKEN` env var)
2. Clicking "Agree" on both pyannote model pages (linked in Quick start)

If `HF_TOKEN` is missing, the script logs a notice and continues without diarization. Transcripts still work; they just won't have speaker labels.

## Resumable + idempotent

Already-indexed clips are skipped on re-runs (a sidecar existing = done). Ctrl-C any time; a restart picks up where it stopped. `--force` regenerates everything.

## Troubleshooting

**"Missing dependency: whisperx"** — Run `setup.py`.

**"Failed to load diarization pipeline"** — You didn't accept the pyannote model terms on Hugging Face. Visit the two model pages, click Agree, then re-run.

**Whisper model download stalls** — `setup.py --skip-model-download`, then `index_videos.py` downloads on first use. Make sure you have disk space (~3GB for large-v3, ~1.5GB for turbo).

**"No GPS data in this file"** — Many clips don't have GPS metadata. The script handles this silently — the frontmatter just omits the location block.

**Apple Silicon GPU not used** — CTranslate2 (via WhisperX) currently runs on CPU on M-series Macs. For archive indexing, CPU is plenty fast (10-30× realtime).

## Companion tools

| Command | Script | Purpose |
|---|---|---|
| `fdx` | `index_videos.py` | Main indexer |
| `fdx-photos` | `photos_indexer.py` | Index videos directly from an Apple Photos library (no export) |
| `fdx-summary` | `trip_summary.py` | Recursive per-folder summaries |
| `fdx-master` | `master_index.py` | Drive-level `_INDEX.md` + `_INDEX.json` |
| `fdx-query` | `query.py` | Filter sidecars by rating, lighting, person, keyword, location, language |

```bash
fdx-query /Volumes/SSD-2024 --rating keep --time-of-day golden_hour
fdx-query /Volumes/SSD-2024 --rating cull                  # the cull pile
fdx-query /Volumes/SSD-2024 --keyword drone --keyword landscape
fdx-query /Volumes/SSD-2024 --place-contains California --language es
```

## Known limitations

- Frame sampling is evenly-spaced, not scene-detected
- pyannote diarization degrades on heavy ambient noise (wind, music, crowd)
- WhisperX runs on CPU on Apple Silicon
- Face cluster IDs are temporary hashes until the `fdx-faces` labeling tool ships — embeddings are captured now, so no re-indexing will be needed
- RAW photo support not yet (videos only)

## License

MIT — see [LICENSE](LICENSE).
