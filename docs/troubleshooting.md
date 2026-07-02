# Troubleshooting

[← Back to the framedex README](../README.md)

**"Video indexing needs the 'video' extra"** (or `ModuleNotFoundError: whisperx`). Video
indexing lives in an optional extra now: `uv pip install -e '.[video]'` (or `'.[all]'`),
then re-run.

**"Failed to load diarization pipeline"**. You didn't accept the pyannote model terms on
Hugging Face. Visit the two model pages, click Agree, then re-run. See
[Tuning and advanced config](tuning.md#speaker-diarization) for the links.

**Whisper model download stalls**. Run `setup.py --skip-model-download`, then
`index_videos.py` downloads on first use. Make sure you have disk space (~3GB for
large-v3, ~1.5GB for turbo).

**"No GPS data in this file"**. Many clips don't have GPS metadata. The script handles
this silently; the frontmatter just omits the location block.

**Apple Silicon GPU not used**. CTranslate2 (via WhisperX) currently runs on CPU on
M-series Macs. For archive indexing, CPU is plenty fast (10-30× realtime).

**A file keeps getting re-indexed every run** (`vision response had no parsable YAML
block — will retry next run`). The vision model returned text without a parseable
`​```yaml` block, so framedex refuses to write a defaults-only sidecar and retries
instead. This is usually a local model (`--backend local`) that doesn't follow the
output format — switch model or backend. A sidecar is only written once a real
structured result comes back, so the file is never silently marked done with garbage.

**`skipped N sidecar(s) with unusable path`** from `fdx-query` / `fdx-master`. One or
more sidecars have a missing, blank, or non-string `path:` field and were skipped so
they can't emit a `.description.md` path where a media path is expected. Apple Photos
mirror sidecars (which carry `photos_uuid` and omit `path` by design) are not affected.

**Stale `.<name>.<pid>.tmp` files next to my media**. A run that was killed mid-write
can leave a hidden temp file. It's inert — skipped by discovery, ignored on resume —
and safe to delete.

---

Part of [framedex](../README.md), an open-source project from
**[SimbaStack](https://simbastack.com/)**.
