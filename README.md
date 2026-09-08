# framedex

**A queryable knowledge base for your video and photo archive.**

```text
   folders / SSDs of clips + photos          an Apple Photos library
                |  fdx                               |  fdx-photos
                +----------------+------------------+
                                 |
                                 v
                 per-file pipeline (local, resumable)
       metadata, GPS -> place, faces, AI scene + keep/review/cull
         transcript + speakers + English translation (video)
                                 |
                                 v
            a plain-text .description.md sidecar per file
              (originals never modified; Photos -> mirror tree)
                                 |
        +------------------------+------------------------+
        |                        |                        |
        v                        v                        v
   ask Claude /             fdx-master               fdx-xmp
   fdx-query                _INDEX.md + .json         ratings + keywords
   by person, place,        whole-drive rollup        -> Lightroom / Bridge
   rating, keyword                                    (.xmp sidecars)
```

Turn a scattered media archive, spread across multiple SSDs and years, into a portable, plain-text knowledge base. Each video clip gets a `.description.md` sidecar with GPS location + place name, a speaker-diarized multilingual transcript, an English translation (if needed), face detection, and an AI vision scene description with a keep/review/cull rating. Each still photo gets the same treatment minus the audio, plus a camera/lens/exposure block read from EXIF.

Sidecars live next to the originals. Originals are never modified. Local-first, non-destructive, resumable.

framedex is a [Claude Code](https://docs.claude.com/en/docs/claude-code) skill. It installs the `fdx` command-line tool.

## Guides

The README covers the core workflow. Deeper or edge-case topics live in `docs/`:

- **[Apple Photos library](docs/apple-photos.md)**: index a `.photoslibrary` directly with `fdx-photos`, including the iCloud "Optimize Storage" edge case.
- **[Tuning and advanced config](docs/tuning.md)**: folder-context priors, proper-noun biasing, languages, speaker-diarization setup.
- **[Troubleshooting](docs/troubleshooting.md)**: common errors and their fixes.
- **[fdx-mcp](docs/mcp.md)**: serve an indexed archive to Claude Desktop, LM Studio, or any MCP host; the tool contract, host setup, what leaves the machine.

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

### Bursts and RAW+JPEG pairs

Before the per-file loop, `fdx` groups a folder's stills so that a burst or a RAW+JPEG pair shares one assessment per group:

- **RAW+JPEG pair:** a RAW and a same-stem camera JPEG (`.jpg`/`.jpeg` only) in the same folder. The RAW is the primary (the edit target); the JPEG is used as the preview, so no embedded-preview extraction is needed.
- **Burst:** at least 3 frames in one folder with matching camera make/model (EXIF) and gaps of at most 2 s between consecutive frames (`DateTimeOriginal`). Separate shots taken that quickly therefore join one burst. Pairs collapse to their RAW before chaining, so a burst of pairs is one burst.

Each group gets one vision call, on the member with the highest sharpness score (variance of the Laplacian over the rendered preview, computed locally; no model is asked to choose). That member is the group's **primary**; every other member is an **alternate** and gets a **stub sidecar**: its own file/EXIF/GPS/place fields, a copy of the primary's assessment (`rating`, `cull_reason`, `technical`, `lighting`, `scene_type`, `keywords`, ...), a `group:` block that names the primary, and a body that says so. Faces are not copied (no detection ran on that frame; an empty list means "not checked"), and any face rows an earlier per-file index left for that frame are cleared so `faces.db` mirrors the sidecars. The primary's block lists the members; a stub's block points at the primary:

```yaml
group:                     # on the primary
  kind: burst              # burst | raw_jpeg
  id: b_3f9a1c2e           # hash of the member file names
  primary: true
  members: [DSC_0141.NEF, DSC_0142.NEF, DSC_0143.NEF]
  sharpness: 214.7         # this file's Laplacian score

group:                     # on a stub
  kind: burst
  id: b_3f9a1c2e
  primary: false
  primary_file: DSC_0142.NEF
  sharpness: 88.2
```

Vision calls for a run = groups + ungrouped files, never more than the file count. Incomplete or changed groups are reprocessed together, which repeats that group's call. Stubs cost two `exiftool` reads each and no model call. Files without a usable `DateTimeOriginal` or camera make/model never join a burst; they index individually. Grouping reads EXIF for the whole folder in one batched `exiftool` call; if that call fails, indexing stops and says so rather than silently paying for every file (`--no-group` indexes files individually). `fdx-master` counts ratings, keywords, faces, and the cull list over primaries only. `fdx-query --primary-only` hides stubs (by default an alternate still matches, e.g. by keyword). `fdx-xmp` tags burst members `burst-primary` / `burst-alternate` so Lightroom can filter the alternates. `--no-group` disables grouping for a run and skips existing sidecars like any run; to re-index a grouped folder individually run `fdx --force --no-group`. Thresholds are constants documented in [docs/tuning.md](docs/tuning.md).

## Getting ratings into Lightroom (`fdx-xmp`)

The `.description.md` sidecars are the source of truth, but Lightroom can't read them. `fdx-xmp` projects the rating, keywords, and one-line caption into standard `.xmp` sidecars next to your RAW files, so an editor picks them up:

```bash
fdx-xmp /Volumes/SSD-2024            # writes DSC_1234.xmp next to DSC_1234.RAF
fdx-xmp /Volumes/SSD-2024 --dry-run  # preview: print what it would write
```

Then in Lightroom Classic: select the photos → **Metadata → Read Metadata from Files**. `keep`/`review`/`cull` land as **3★ / 2★ / 1★**, `cull` also gets a **Red** label, keywords fill the keyword list, and the scene sentence becomes the caption. Filter by `1★` or Red to sweep the cull pile; the 3★ ceiling leaves room for your own 4/5★ picks.

It's a **regenerable view**: delete every `.xmp` and re-run to rebuild them. It never edits an original, and never touches a `.xmp` it didn't write; a hand-edited or foreign sidecar is reported as a conflict and skipped (ownership is tracked by a content hash in `_XMP_MANIFEST.json`, not a filename). Don't run it while Lightroom is writing metadata to the same files.

Scope in v1: **proprietary RAW → Lightroom Classic**. Lightroom reads `.xmp` *sidecars* only for proprietary RAW; for JPEG/HEIC/TIFF/DNG it reads metadata embedded in the file, which framedex never modifies, so those shooters get no Lightroom integration here. (`.dng` is excluded for the same reason.)

## Talk to your archive from an MCP host (`fdx-mcp`)

`fdx-mcp` serves an indexed archive to any MCP client over stdio: Claude Code, Claude Desktop, LM Studio with a local model, or your own agent. framedex makes no model call itself; the host's model does the reasoning and framedex hands it facts from the sidecars and, on request, one picture.

```bash
uv pip install -e '.[mcp]'        # MCP SDK + Pillow (add [images] for HEIC thumbnails)
fdx-mcp /Volumes/SSD-2024         # one or more indexed roots; --read-only drops the one write tool
```

Six tools: `list_roots`; `query_media` (the `fdx-query` filters, plus `folder` and paging); `read_sidecar`; `archive_overview` (`_INDEX.md` as the snapshot it is); `contact_sheet` (1-20 files rendered into one numbered JPEG grid with a legend, so the model can compare candidates in a single look); and `set_user_rating`, which records the person's `user_rating` (keep/review/cull) and a note in the sidecar next to the model's `rating`, never over it. A user rating wins in `fdx-query`, `fdx-master`, and `fdx-xmp` (keyword `user-rated`; a user `keep` is still 3★), and survives re-indexing.

One bounded image per successful `contact_sheet` call (at most 20 thumbnails, about 1600 px wide), zero model calls inside framedex. What leaves your machine is the host's decision: it may send its model the thumbnails, paths, GPS, names, notes, and transcripts the tools return; with LM Studio and a local model, nothing does. Host setup and the full tool contract: [docs/mcp.md](docs/mcp.md).

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
| `--max-files N` | Stop after N work items; each single file or group counts once (testing) |
| `--force` | Re-process clips even if a sidecar exists |
| `--whisper-model large-v3` | Higher quality, slower (default is large-v3-turbo) |
| `--no-diarize` | Skip speaker diarization (faster; no HF_TOKEN needed) |
| `--no-faces` | Skip face detection + embeddings |
| `--no-geocode` | Skip Nominatim reverse geocoding (GPS still recorded) |
| `--max-duration MINUTES` | Skip clips longer than N minutes (default: 30; 0 = no limit) |
| `--frame-sampling diverse\|even` | How the 5 vision frames are picked (default: diverse; `even` = legacy evenly-spaced) |
| `--no-group` | Disable grouping for this run (default: bursts and RAW+JPEG pairs share one assessment). Existing sidecars are skipped; `--force --no-group` re-indexes a grouped folder individually |
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

The `cli` backend runs `claude -p` locked down: `--permission-mode dontAsk` plus a read-only `--allowedTools Read` allowlist. Untrusted text embedded in the prompt (a transcript snippet, your `.video-context.md`) therefore can't drive tool use: Bash, writes, and edits are denied, so a would-be injection surfaces as a per-file vision error and retries instead of running a command. (The read scope is the whole filesystem, not just the frames: the CLI doesn't honor path-scoped `Read(<dir>/**)` allowlists, and read-only-no-execute is the security boundary that matters here.) Needs a `claude` CLI new enough to support `--permission-mode dontAsk` (check `claude --help`); older builds error loudly rather than degrade silently.

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

Writes are atomic: every sidecar and index goes to a temp file and is renamed into place, so an interrupt mid-write never leaves a truncated, half-indexed file. Faces are committed before the sidecar (the sidecar is the "done" marker), so a crash in between just re-runs that file cleanly. A stale `.<name>.<random>.tmp` file left by a killed run is inert (hidden from discovery) and safe to delete.

A burst or RAW+JPEG pair is done only when every member's sidecar belongs to that exact group (same `group.id`), or when every member's sidecar predates grouping. Stubs are written before the primary's sidecar, and the primary's previous sidecar (if any, e.g. under `--force`) is first marked `group.incomplete: true` (its content is kept), so an interrupt inside a group always leaves the group visibly unfinished. Incomplete or changed groups are reprocessed together next run; this repeats that group's vision call. A file whose folder was regrouped, so that its old stub no longer matches, is redone too.

## Companion tools

| Command | Script | Purpose |
|---|---|---|
| `fdx` | `index_videos.py` | Main indexer |
| `fdx-photos` | `photos_indexer.py` | Index media (videos + stills) directly from an Apple Photos library (no export); `--media images\|videos\|all`. See [docs/apple-photos.md](docs/apple-photos.md) |
| `fdx-summary` | `trip_summary.py` | Recursive per-folder summaries |
| `fdx-master` | `master_index.py` | Drive-level `_INDEX.md` + `_INDEX.json` |
| `fdx-query` | `query.py` | Filter sidecars by rating, lighting, person, keyword, location, language; `--folder`, `--offset`/`--limit` |
| `fdx-xmp` | `xmp_export.py` | Export ratings/keywords/caption to `.xmp` sidecars for Lightroom (RAW) |
| `fdx-mcp` | `mcp_server.py` | Serve the archive as MCP tools over stdio (query, read, overview, contact sheet, user rating); `[mcp]` extra |

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
- Bursts are inferred from EXIF: at least 3 frames in one folder with matching camera make/model and gaps of at most 2 s between consecutive frames. Separate shots taken that quickly join one burst (`--no-group` opts out). Two bodies of the same model in one folder look like one camera. Stub sidecars mirror the primary's assessment without the model having seen that frame; `fdx-photos` does not group yet (Apple Photos exposes native burst info; planned)

## Built by SimbaStack

framedex is an open-source project from **[SimbaStack](https://simbastack.com/)**, an AI consulting and development studio. We help businesses figure out where AI actually fits in their operations, then build and ship it. Working systems in production, not strategy decks.

If you want something like this built for your company (agents, automation, AI that removes a real bottleneck), get in touch: **[nj@simbastack.com](mailto:nj@simbastack.com)**.

## License

MIT. See [LICENSE](LICENSE).
