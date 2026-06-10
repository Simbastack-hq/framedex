# framedex — context for agents

## North star

framedex turns a photographer's or videographer's scattered media archive
(SSDs of clips and RAWs, an Apple Photos library) into a **plain-text knowledge
base they can search, reason about, and trust for decades** — find the moment,
judge what to keep, understand what a drive holds, without ever opening a
timeline scrubber.

The bet: the durable artifact is **portable plain text next to the originals**,
not an app, not a database, not a subscription. Tools come and go; sidecars
survive.

## Principles (in priority order)

1. **Originals are sacred.** Never modify, move, or rename media files.
   Sidecars and indexes only. Non-destructive, idempotent, resumable — a
   Ctrl-C must never corrupt anything.
2. **LLM-agnostic, local-preferred.** Every AI step must work with a local
   model (`--backend local`, LM Studio / any OpenAI-compatible server). Cloud
   backends (Claude CLI/API) are optimizations, never requirements. Don't add
   features that only work with one provider; don't let prompts depend on one
   model's quirks.
3. **Plain text outlives software.** Output is human-readable Markdown/YAML/
   JSON a user can grep, sync, and read in 2040. No opaque stores (the face DB
   is the one pragmatic exception — embeddings aren't text; everything written
   there is re-derivable except the embeddings themselves).
4. **Predictable cost.** Vision spend per file is fixed and documented
   (5 frames/clip, 1 call/file). Changes must not silently scale cost with
   archive size or clip length.
5. **Earn every dependency.** Prefer ~100 lines on cv2/numpy/ffmpeg (already
   required) over a new package. Heavy stacks stay in optional extras
   ([video]/[images]/[photos]) — a photo-only user never pulls torch.
6. **Useful to professionals, not just hobbyists.** Ratings respect that this
   is a working archive of real moments: the bar for "cull" is high, the
   default is "keep", and queries (person, place, lighting, keyword, rating)
   serve a culling/selecting workflow.

## Working conventions

- Source in `src/framedex/`; shared media-agnostic core in `pipeline.py`;
  media-specific work stays in its pipeline module. cv2 image math lives in
  dedicated modules (`face_db.py`, `frame_sampling.py`), not in orchestrators.
- Any new CLI flag goes in BOTH parsers (`index_videos.main` and
  `photos_indexer.main`) and threads through `ProcessOptions`.
- Tests are hermetic: never shell out to ffmpeg or hit the network; pure
  functions over numpy arrays / mocked subprocess.
- Tunables ship as named module constants with a one-line comment, not flags,
  unless a real user decision exists (`--frame-sampling`, `--backend`).
- Docs that must stay in sync with behavior: README.md, SKILL.md, docs/*.md.
  Known-limitation lists are honest and updated in the same PR that changes
  behavior.
- Design specs live in `docs/superpowers/specs/` and are committed before
  implementation.
