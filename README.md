# framedex

**A queryable knowledge base for your video archive.**

Turn a scattered video archive — across multiple SSDs and years — into a portable, plain-text knowledge base. Each clip gets a `.description.md` sidecar with GPS location + place name, a speaker-diarized multilingual transcript, an English translation (if needed), face detection, and an AI vision scene description with a keep/review/cull rating.

Sidecars live next to the videos. Originals are never modified. Local-first, non-destructive, resumable.

framedex is a [Claude Code](https://docs.claude.com/en/docs/claude-code) skill. It installs the `fdx` command-line tool.

## Install

```bash
# Clone into your Claude Code skills directory
git clone git@github.com:Simbastack-hq/framedex.git ~/.claude/skills/framedex

# Install deps + pre-download the Whisper + face-detection models
python3 ~/.claude/skills/framedex/scripts/setup.py
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

# 3. Add aliases to ~/.zshrc
alias fdx="python3 $HOME/.claude/skills/framedex/scripts/index_videos.py"
alias fdx-summary="python3 $HOME/.claude/skills/framedex/scripts/trip_summary.py"
alias fdx-master="python3 $HOME/.claude/skills/framedex/scripts/master_index.py"
alias fdx-query="python3 $HOME/.claude/skills/framedex/scripts/query.py"

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
path: /Volumes/SSD-2024/2024-08-construction/drone/IMG_4827.mov
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
