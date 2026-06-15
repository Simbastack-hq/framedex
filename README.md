# framedex

**A queryable knowledge base for your video and photo archive.**

```text
 folders / SSDs of clips + photos        Apple Photos library (macOS)
                |                                     |
               fdx                               fdx-photos
                |                                     |
                +-----------------+-------------------+
                                  |
                                  v
                  per-file pipeline (local, resumable)
         metadata, GPS -> place, faces, AI description + rating
          transcript + speakers + English translation (video)
                                  |
                                  v
             a plain-text .description.md sidecar per file
        (originals never modified; Photos writes a mirror tree)
                                  |
                                  v
          fdx-summary / fdx-master -> _INDEX.md + _INDEX.json
                                  |
                                  v
     fdx-query --person Mom --place-contains Yosemite --rating keep
              ...or just ask Claude to read the index
```

Turn a scattered media archive, spread across multiple SSDs and years, into a portable, plain-text knowledge base. Each video clip gets a `.description.md` sidecar with GPS location + place name, a speaker-diarized multilingual transcript, an English translation (if needed), face detection, and an AI vision scene description with a keep/review/cull rating. Each still photo gets the same treatment minus the audio, plus a camera/lens/exposure block read from EXIF.

Sidecars live next to the originals. Originals are never modified. Local-first, non-destructive, resumable.

framedex is a [Claude Code](https://docs.claude.com/en/docs/claude-code) skill. It installs the `fdx` command-line tool.

## Guides

The README covers the core workflow. Deeper or edge-case topics live in `docs/`:

- **[Apple Photos library](docs/apple-photos.md)**: index a `.photoslibrary` directly with `fdx-photos`, including the iCloud "Optimize Storage" edge case.
- **[Tuning and advanced config](docs/tuning.md)**: folder-context priors, proper-noun biasing, languages, speaker-diarization setup.
- **[Troubleshooting](docs/troubleshooting.md)**: common errors and their fixes.

## Install

```bash
# Clone into your Claude Code skills directory
git clone git@github.com:Simbastack-hq/framedex.git ~/.claude/skills/framedex
cd ~/.claude/skills/framedex

# Pick what you index. The heavy video stack (whisperx/torch) and the still-photo
# readers (Pillow) are optional extras, so a photo-only setup never pulls torch:
uv pip install -e '.[all]'        # video + photos + Apple Photos (everything)
# uv pip install -e '.[video]'    # video only (folders of clips)
# uv pip install -e '.[images]'   # still photos only (RAW / JPEG / HEIC)

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

# 2. (Optional) Set an Anthropic API key, only needed for --backend api
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
4. `ffmpeg` → 5 content-diverse JPEG frames (≤1920px wide): small thumbnails
   are sampled across the clip and the 5 most mutually different, sharpest
   moments are kept (static or short clips keep plain even spacing;
   `--frame-sampling even` restores the legacy behavior)
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

For folder-context priors and proper-noun biasing that sharpen these descriptions, see
[Tuning and advanced config](docs/tuning.md).

## Multiple SSDs

Run on each drive separately:

```bash
fdx /Volumes/SSD-2023
fdx /Volumes/SSD-2024
fdx /Volumes/SSD-2025
```

Each drive ends up self-contained with its own sidecars + `_INDEX.json`. Knowledge travels with the data. The face DB at `~/.framedex/faces.db` is centralized so cross-drive person queries work.

## Still photos (RAW / JPEG / HEIC)

`fdx` indexes photos the same way it indexes video: same `.description.md` sidecars, queryable with the same `fdx-query` and rolled up by `fdx-master`. Point it at a folder of stills (a Lightroom/Capture One export, an SSD of RAWs) and each photo gets EXIF (camera, lens, aperture, shutter, ISO), GPS + reverse-geocoded place, face detection, and a scene description with keywords and a keep/review/cull rating. (`fdx-summary`'s prose is still video-tuned; photo-aware summaries are a fast-follow.)

```bash
uv pip install -e '.[images]'                          # one-time: Pillow + pillow-heif, no torch

fdx /Volumes/SSD-photos --media images --max-files 5   # test on 5 first
fdx /Volumes/SSD-photos                                 # mixed photos + clips, one corpus
fdx /Volumes/SSD-photos --media images                 # stills only
```

- **One command, mixed media.** A drive with both photos and clips becomes a single queryable corpus; `fdx` routes each file by extension. `--media images|videos|all` scopes a run.
- **RAW** is read from the full-res JPEG preview every modern RAW embeds (no libraw needed): `.cr2 .cr3 .nef .arw .raf .rw2 .orf .dng`, plus `.jpg .jpeg .png .tif .tiff .heic .webp`.
- **Search is identical to video:** `fdx-query /Volumes/SSD-photos --media images --place-contains Mara --keyword giraffe`, or just ask Claude to read `_INDEX.md` for "that sunset photo in Mara".

Photo sidecars add a `camera:` block, `dimensions`, `scene_type`, and `media_type: image`, and drop the video-only audio/duration fields.

## Apple Photos library (macOS)

`fdx-photos` indexes videos **and** stills straight from an Apple Photos library: no export, no metadata loss. The common case is one command:

```bash
uv pip install -e '.[all]'       # one-time: osxphotos + video + image readers
fdx-photos                       # indexes the whole library (videos + stills)
```

Album/person/date filters, the sidecar mirror layout, Photos-side frontmatter, and the iCloud "Optimize Storage" edge case are all in the full guide: **[docs/apple-photos.md](docs/apple-photos.md)**.

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
| `--frame-sampling diverse\|even` | How the 5 vision frames are picked (default: diverse; `even` = legacy evenly-spaced) |
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
| Nominatim reverse geocode | Cloud: sends lat/lon only, never video. Skip with `--no-geocode` |
| Vision (`--backend cli`/`api`) | Cloud: sends 5 JPEG frames + a transcript snippet per clip |
| Vision (`--backend local`) | Fully local |
| Face DB (`~/.framedex/faces.db`) | Local only, never uploaded |

## Resumable + idempotent

Already-indexed clips are skipped on re-runs (a sidecar existing = done). Ctrl-C any time; a restart picks up where it stopped. `--force` regenerates everything.

## Companion tools

| Command | Script | Purpose |
|---|---|---|
| `fdx` | `index_videos.py` | Main indexer |
| `fdx-photos` | `photos_indexer.py` | Index media (videos + stills) directly from an Apple Photos library (no export); `--media images\|videos\|all`. See [docs/apple-photos.md](docs/apple-photos.md) |
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

- pyannote diarization degrades on heavy ambient noise (wind, music, crowd)
- WhisperX runs on CPU on Apple Silicon
- Face cluster IDs are temporary hashes until the `fdx-faces` labeling tool ships; embeddings are captured now, so no re-indexing will be needed

## Built by SimbaStack

framedex is an open-source project from **[SimbaStack](https://simbastack.com/)**, an AI consulting and development studio. We help businesses figure out where AI actually fits in their operations, then build and ship it. Working systems in production, not strategy decks.

If you want something like this built for your company (agents, automation, AI that removes a real bottleneck), get in touch: **[nj@simbastack.com](mailto:nj@simbastack.com)**.

## License

MIT. See [LICENSE](LICENSE).
