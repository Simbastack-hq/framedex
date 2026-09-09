# Tuning and advanced config

[← Back to the framedex README](../README.md)

Optional knobs that improve description and transcript quality. None are required to get
useful sidecars; reach for them when defaults aren't good enough.

## Optional folder context

Drop `.video-context.md` at the root of any scan target to give the vision model better
priors:

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

These get passed to Whisper as `initial_prompt` + `hotwords` so place names and people
names in speech don't come back garbled. A second regex pass
(`~/.framedex/whisper_fixes.json`) catches anything the prompt bias misses.

## Languages

Whisper supports 99 languages with auto-detection. For non-English clips the script
automatically runs a second translate-mode pass and stores the English version alongside
the original transcript. For best quality on important non-English footage:

```bash
fdx /Volumes/SSD-2024 --whisper-model large-v3 --force
```

## Speaker diarization

WhisperX uses `pyannote/speaker-diarization-3.1` under the hood. First-time setup
requires:

1. A Hugging Face account + read token (`HF_TOKEN` env var):
   <https://huggingface.co/settings/tokens>
2. Clicking "Agree" on both pyannote model pages:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

If `HF_TOKEN` is missing, the script logs a notice and continues without diarization.
Transcripts still work; they just won't have speaker labels.

---

Part of [framedex](../README.md), an open-source project from
**[SimbaStack](https://simbastack.com/)**.

## Frame sampling

`fdx` picks the 5 vision frames by sampling small thumbnails across the clip
(one per ~2s, capped at 96) and keeping the most mutually different, sharpest
moments — so a clip that pans from a beach to a street gets frames from both,
not five near-duplicates. Brightness is excluded from the difference metric so
auto-exposure drift doesn't count as change. Static clips, clips under 20s,
and night footage that fails the brightness gates keep the legacy evenly-
spaced sampling, as does `--frame-sampling even`. Pool size and thresholds
are documented constants in `src/framedex/frame_sampling.py`.

## Burst grouping

`fdx` groups a folder's stills before the per-file loop so one moment costs one
vision call (see the README section "Bursts and RAW+JPEG pairs"). The
thresholds are constants in `src/framedex/grouping.py`, not flags:

- `BURST_GAP_SEC = 2.0`: consecutive frames in the same folder with matching
  camera make/model at most this far apart chain into one burst. Timestamps come from
  `SubSecDateTimeOriginal` when present, else `DateTimeOriginal` plus
  `SubSecTimeOriginal`; whole-second precision still works (a 10 fps burst
  simply has gap 0). Files with no usable date never join a group.
- `BURST_MIN_SIZE = 3`: a shorter chain is not a burst; its frames index
  individually.
- `PAIR_JPEG_EXTENSIONS = {".jpg", ".jpeg"}`: only camera JPEGs pair with a
  same-stem RAW. PNG/TIFF/HEIC/WebP sharing a stem are treated as exports, not
  the same capture. Two RAWs sharing a stem never pair (ambiguous).

The primary is the member with the highest Laplacian variance over its
rendered preview (the pair's JPEG when there is one); ties go to the earliest
frame. `--no-group` disables grouping for a run. Grouping runs one batched
`exiftool` call over the whole image list; if that call fails, indexing stops.
Use `--no-group` to retry with individual indexing. Files that come back
without EXIF from a successful batch never join a burst (RAW+JPEG pairing,
which needs no EXIF, still applies).
