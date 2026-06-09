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
