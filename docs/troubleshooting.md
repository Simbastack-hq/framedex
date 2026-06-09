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

---

Part of [framedex](../README.md), an open-source project from
**[SimbaStack](https://simbastack.com/)**.
